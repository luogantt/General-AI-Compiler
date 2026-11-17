#!/usr/bin/env python3
"""Isolated NPU Q/K norm+RoPE fusion probe; no checkpoint or model timing claim.

The owner supplies the built bridge/OPP environment. Every output used for
verification is copied to CPU immediately; verification and cold calls are
excluded from the separately synchronized timing batches. No fallback is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback


def make_formula(torch, torch_npu, *, head_stride, epsilon, rotation=None, native=False):
    """Original FP32 islands, including the BF16 norm-output round trip."""
    heads = 8 if head_stride == 512 else 2

    def formula(x, weight, cos, sin):
        with torch.autocast("npu", enabled=False):
            raw = x.view(1, 32, heads, head_stride)[..., :256].float()
            gamma = 1.0 + weight.float()
            if native:
                normalized = torch_npu.npu_rms_norm(raw, gamma, epsilon=epsilon)[0]
            else:
                normalized = raw * torch.rsqrt(raw.pow(2).mean(-1, keepdim=True) + epsilon)
                normalized = normalized * gamma
            # This cast must precede the FP32 RoPE math. Keeping it all FP32
            # until the final store is a different model, not a safe fusion.
            value = normalized.to(torch.bfloat16).transpose(1, 2).float()
            c, s = cos[:, :, :32, :].float(), sin[:, :, :32, :].float()
            if native:
                c = torch.cat((c, torch.ones_like(c).repeat(1, 1, 1, 3)), -1)
                s = torch.cat((s, torch.zeros_like(s).repeat(1, 1, 1, 3)), -1)
                out = torch_npu.npu_rotary_mul(value, c, s, "half", rotation)
            else:
                partial, tail = value[..., :64], value[..., 64:]
                rotated = torch.cat((-partial[..., 32:], partial[..., :32]), -1)
                out = torch.cat((partial * c + rotated * s, tail), -1)
            return out.to(torch.bfloat16).contiguous()
    return formula


def compare(torch, expected, actual):
    delta = actual - expected
    return {"finite": bool(torch.isfinite(actual).all()), "exact": bool(torch.equal(expected, actual)),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-30)),
            "rmse": float(delta.square().mean().sqrt())}


def assess_variant(checks, relative_limit, graph_compilations, inputs_unchanged):
    """One unchanged numerical gate, with explicit per-case failure details."""
    failures = []
    for check in checks:
        for basis in ("vs_original", "vs_native"):
            error = check[basis]
            reasons = []
            if not error["finite"] or not all(math.isfinite(error[key]) for key in ("max_abs", "relative_l2", "rmse")):
                reasons.append("nonfinite")
            if error["max_abs"] > check["max_abs_limit"]:
                reasons.append("max_abs")
            if error["relative_l2"] > relative_limit:
                reasons.append("relative_l2")
            if reasons:
                failures.append({"case": check["case"], "reference": basis, "reasons": reasons,
                                 "metrics": error, "max_abs_limit": check["max_abs_limit"],
                                 "relative_l2_limit": relative_limit})
    if not checks:
        failures.append({"reason": "missing_checks"})
    if graph_compilations is not None and graph_compilations["dynamo_backend_compilations"] != 1:
        failures.append({"reason": "expected_one_graph_compilation", **graph_compilations})
    if not inputs_unchanged:
        failures.append({"reason": "input_mutation"})
    return not failures, failures


def summarize_validation(rows, require_ge):
    """Separate required Ascend C acceptance from observational GE controls."""
    required = ("ascendc_eager", "ascendc_ge") if require_ge else ("ascendc_eager",)
    candidate_failures, reference_failures, control_deviations = [], [], []
    if not rows:
        candidate_failures.append({"reason": "no_cases"})
    for row in rows:
        identity = {"heads": row["heads"], "rope_length": row["rope_length"]}
        for name in required:
            result = row["variants"].get(name)
            if result is None or not result["passed"]:
                candidate_failures.append({**identity, "variant": name,
                                           "failures": result.get("failures", []) if result else [{"reason": "missing_variant"}]})
        for name in ("original_eager", "native_eager"):
            result = row["variants"].get(name)
            if result is None or not result["passed"]:
                reference_failures.append({**identity, "variant": name,
                                           "failures": result.get("failures", []) if result else [{"reason": "missing_reference"}]})
        for name in ("original_ge", "native_ge"):
            result = row["variants"].get(name)
            if result is not None and not result["passed"]:
                control_deviations.append({**identity, "variant": name, "failures": result.get("failures", [])})
    return {"status": "reference_validation_failed" if reference_failures else
                      "candidate_validation_failed" if candidate_failures else "success",
            "candidate_validation": {"passed": not candidate_failures and not reference_failures,
                                     "required_variants": list(required), "failures": candidate_failures,
                                     "references": ["original_eager", "native_eager"], "thresholds_unchanged": True},
            "reference_validation": {"passed": not reference_failures, "failures": reference_failures},
            "control_observations": {"status": "deviations_observed" if control_deviations else "within_thresholds",
                                     "required_for_candidate_acceptance": False, "deviations": control_deviations}}


def provenance():
    root = Path(__file__).resolve().parent
    files = [Path(__file__), root / "g05_ae_qknorm_rope.py"]
    source = root / "ascendc_qknorm_rope"
    files += [source / name for name in ("torch_bridge.cpp", "build_opp.py", "build_bridge.py", "GacQkNormRope.json",
                                        "op_host/gac_qk_norm_rope.cpp", "op_kernel/gac_qk_norm_rope.cpp",
                                        "op_kernel/gac_qk_norm_rope_tiling.h")]
    return {"ASCEND_CACHE_PATH": os.environ.get("ASCEND_CACHE_PATH"),
            "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
            "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH"),
            "GAC_QKNORM_ROPE_LIBRARY": os.environ.get("GAC_QKNORM_ROPE_LIBRARY"),
            "source_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             if path.is_file() else {"path": str(path), "missing": True} for path in files],
            "cache_contract": "fresh process and a new ASCEND_CACHE_PATH per built revision; no automatic cache deletion"}


def run(args, report):
    import torch
    import torch_npu
    from g05_ae_qknorm_rope import make_qknorm_rope, register_qknorm_rope_op
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch._dynamo.config.suppress_errors = False
    op, registration = register_qknorm_rope_op()
    fused = make_qknorm_rope(torch, op)
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    def ge_backend():
        config = CompilerConfig()
        # Explicit weight-value changes are part of this operator-level probe.
        # Whole-model frozen weights are a separate, immutable-weight contract.
        config.experimental_config.frozen_parameter = False
        return torchair.get_npu_backend(compiler_config=config)

    def snapshot(function, values):
        with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
            output = function(*values)
        torch.npu.synchronize()
        if output.dtype != torch.bfloat16 or not output.is_contiguous():
            raise RuntimeError("Output must be contiguous BF16")
        expected_shape = (1, 8 if values[0].shape[-1] == 4096 else 2, 32, 256)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"Output shape changed: {tuple(output.shape)} != {expected_shape}")
        if any(output.data_ptr() == value.data_ptr() for value in values):
            raise RuntimeError("Output unexpectedly aliases an input")
        return output.detach().float().cpu().clone()

    def timings(function, values):
        with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
            for _ in range(args.warmup):
                function(*values)
            torch.npu.synchronize()
            groups = []
            for _ in range(args.timing_groups):
                started = time.perf_counter()
                for _ in range(args.iterations):
                    function(*values)
                torch.npu.synchronize()
                groups.append((time.perf_counter() - started) * 1000 / args.iterations)
        ordered = sorted(groups)
        return {"batch_mean_ms_per_call": groups, "median_batch_mean_ms": ordered[len(ordered) // 2],
                "min_batch_mean_ms": ordered[0], "max_batch_mean_ms": ordered[-1]}

    report.update(scope="isolated AE Q/K norm+RoPE64+layout; not whole-model latency or task accuracy",
                  timing_scope="synchronized batches / calls; includes Python dispatch; verification/cold calls excluded",
                  torch=torch.__version__, torch_npu=torch_npu.__version__, device=args.device,
                  registration=registration, fullgraph=True, dynamic=False, fallback_allowed=False,
                  frozen_parameter=False,
                  parameter_scope="operator-level dynamic weight inputs; whole-model frozen=True is validated separately",
                  thresholds={"max_abs_floor": args.max_abs_error, "relative_l2": args.max_relative_l2,
                              "per_case_max_abs": "max(floor, max(abs(original_reference))/128), a conservative BF16-spacing allowance"}, cases=[])
    with torch.inference_mode():
        rotation = torch.zeros(256, 256, dtype=torch.float32, device=args.device)
        indices = torch.arange(32, device=args.device)
        rotation[indices, indices + 32], rotation[indices + 32, indices] = 1, -1
        for head_stride in (512, 256):
            heads, channels = (8, 4096) if head_stride == 512 else (2, 512)
            for rope_length in args.rope_lengths:
                report["stage"] = f"heads={heads}/Srope={rope_length}"
                x = torch.randn(1, 32, channels, device=args.device, dtype=torch.float32).to(torch.bfloat16)
                # eval() does not clear this flag on real model Parameters.
                weight = torch.nn.Parameter(torch.linspace(-.9, .8, 256, device=args.device, dtype=torch.float32),
                                            requires_grad=True)
                angle = torch.randn(1, 1, rope_length, 32, device=args.device).repeat(1, 1, 1, 2)
                cos, sin = angle.cos(), angle.sin()
                alternate_x = (-1.7 * x.float() + .13).to(torch.bfloat16)
                variants = {
                    "A": (x, weight, cos, sin),
                    "X": (alternate_x, weight, cos, sin),
                    "W": (x, (weight * -.7 + .11).contiguous(), cos, sin),
                    "W_zero": (x, torch.zeros_like(weight), cos, sin),
                    "W_small": (x, torch.full_like(weight, 1e-8), cos, sin),
                    "W_large": (x, torch.full_like(weight, 1000), cos, sin),
                    "C": (x, weight, (cos * -.8 + .07).contiguous(), sin),
                    "S": (x, weight, cos, (sin * -.6 - .09).contiguous()),
                    "zero": (torch.zeros_like(x), weight, cos, sin),
                    "small": (torch.full_like(x, 1e-8), weight, cos, sin),
                    "large": (torch.full_like(x, 1000), weight, cos, sin),
                }
                if head_stride == 512:
                    gate_only = x.clone()
                    gate_only.view(1, 32, heads, head_stride)[..., 256:] = 64
                    variants["gate_only"] = (gate_only, weight, cos, sin)
                if rope_length > 32:
                    c_tail, s_tail = cos.clone(), sin.clone()
                    c_tail[:, :, 32:, :], s_tail[:, :, 32:, :] = 7, -9
                    variants["rope_suffix_only"] = (x, weight, c_tail, s_tail)
                # Keep input types/grad metadata identical across W cases so a
                # weight-value change must exercise the same captured graph.
                variants = {name: (values[0], values[1] if isinstance(values[1], torch.nn.Parameter)
                                   else torch.nn.Parameter(values[1], requires_grad=True), values[2], values[3])
                            for name, values in variants.items()}
                original = make_formula(torch, torch_npu, head_stride=head_stride, epsilon=args.epsilon)
                native = make_formula(torch, torch_npu, head_stride=head_stride, epsilon=args.epsilon,
                                      rotation=rotation, native=True)

                def ascendc(x, weight, cos, sin):
                    return fused(x, weight, cos, sin, head_stride, args.epsilon)

                expected = {name: snapshot(original, values) for name, values in variants.items()}
                native_expected = {name: snapshot(native, values) for name, values in variants.items()}
                if any(not bool(torch.isfinite(value).all()) for value in (*expected.values(), *native_expected.values())):
                    raise RuntimeError("Original/native eager references contain nonfinite values")
                input_copies = {name: [value.detach().float().cpu().clone() for value in values]
                                for name, values in variants.items()}
                for name in ("X", "W", "C", "S"):
                    if torch.equal(expected["A"], expected[name]):
                        raise RuntimeError(f"Reference {name} does not distinguish changed input")
                for name in ("gate_only", "rope_suffix_only"):
                    if name in expected and not torch.equal(expected["A"], expected[name]):
                        raise RuntimeError(f"Reference unexpectedly consumes {name}")
                row = {"heads": heads, "head_stride": head_stride, "rope_length": rope_length,
                       "input_shape": list(x.shape), "input_stride": list(x.stride()),
                       "output_shape": [1, heads, 32, 256], "variants": {}, "reference_inputs_distinguishable": True,
                       "inference_parameter_requires_grad": weight.requires_grad}
                report["cases"].append(row)
                candidates = [("original_eager", original, None), ("native_eager", native, None), ("ascendc_eager", ascendc, None)]
                if args.backend == "torchair":
                    def compile_counted(fn):
                        counts = {"dynamo_backend_compilations": 0}
                        backend = ge_backend()
                        def counted(graph, example_inputs):
                            counts["dynamo_backend_compilations"] += 1
                            return backend(graph, example_inputs)
                        return torch.compile(fn, backend=counted, fullgraph=True, dynamic=False), counts
                    for name, fn in (("original", original), ("native", native), ("ascendc", ascendc)):
                        compiled, counts = compile_counted(fn)
                        candidates.append((name + "_ge", compiled, counts))
                for name, function, counts in candidates:
                    report["stage"] = f"heads={heads}/Srope={rope_length}/{name}"
                    torch.npu.synchronize()
                    started = time.perf_counter()
                    snapshot(function, variants["A"])
                    cold_ms = (time.perf_counter() - started) * 1000
                    checks = []
                    # Reuse the same allocations across repeated changes, then
                    # return to A. Clones on CPU prevent graph-buffer aliasing.
                    for replay in range(2):
                        for label, values in variants.items():
                            actual = snapshot(function, values)
                            original_error = compare(torch, expected[label], actual)
                            native_error = compare(torch, native_expected[label], actual)
                            tail = {"vs_original": compare(torch, expected[label][..., 64:], actual[..., 64:]),
                                    "vs_native": compare(torch, native_expected[label][..., 64:], actual[..., 64:]),
                                    "different_values_vs_original": int((actual[..., 64:] != expected[label][..., 64:]).sum()),
                                    "different_values_vs_native": int((actual[..., 64:] != native_expected[label][..., 64:]).sum())}
                            # BF16 spacing scales with magnitude. Large-weight
                            # stress allows at most this conservative one-ULP
                            # absolute bound, while retaining the global L2 gate.
                            abs_limit = max(args.max_abs_error, float(expected[label].abs().max()) / 128)
                            checks.append({"case": f"{label}_{replay + 1}", "vs_original": original_error,
                                           "vs_native": native_error, "tail_64_256": tail,
                                           "max_abs_limit": abs_limit})
                        actual = snapshot(function, variants["A"])
                        checks.append({"case": f"return_A_{replay + 1}",
                                       "vs_original": compare(torch, expected["A"], actual),
                                       "vs_native": compare(torch, native_expected["A"], actual),
                                       "max_abs_limit": max(args.max_abs_error, float(expected["A"].abs().max()) / 128)})
                    inputs_unchanged = all(torch.equal(value.detach().float().cpu(), input_copies[label][index])
                                           for label, values in variants.items() for index, value in enumerate(values))
                    valid, failures = assess_variant(checks, args.max_relative_l2, counts, inputs_unchanged)
                    result = {"passed": valid, "checks": checks, "first_call_with_copy_ms": cold_ms,
                              "role": "required_candidate" if name.startswith("ascendc_") else
                                      "eager_reference" if name.endswith("_eager") else "observational_control",
                              "failures": failures,
                              "inputs_unchanged": inputs_unchanged,
                              "graph_compilations": dict(counts) if counts is not None else None,
                              "timing": timings(function, variants["A"])}
                    row["variants"][name] = result
                    print(json.dumps({"stage": report["stage"], "passed": valid, "timing": result["timing"]}), flush=True)
    report.update(summarize_validation(report["cases"], args.backend == "torchair"), stage="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("eager", "torchair"), default="torchair")
    parser.add_argument("--rope-lengths", type=lambda value: tuple(map(int, value.split(","))), default=(32, 159))
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timing-groups", type=int, default=5)
    parser.add_argument("--max-abs-error", type=float, default=.03125)
    parser.add_argument("--max-relative-l2", type=float, default=.001)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.timing_groups) < 1 or not args.rope_lengths or min(args.rope_lengths) < 32:
        parser.error("Require positive repeat counts and RoPE sequence lengths >=32")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    report = {"status": "failure", "stage": "imports", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "provenance": provenance()}
    try:
        run(args, report)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "stage": report["stage"]}), flush=True)
    if report["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
