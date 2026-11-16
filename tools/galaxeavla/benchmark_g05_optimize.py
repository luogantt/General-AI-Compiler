#!/usr/bin/env python3
"""Explicit R1 Lite execution/precision experiments, separate from the release benchmark.

FM-only is a reduced execution profile of g05-base (three cameras, 27D actions),
not an exact reproduction of the two-camera/20D LIBERO checkpoint benchmark.
TorchAir/npugraph compilation targets Action Expert.forward; VLM.forward can
be compiled independently. A backend failure is fatal; this tool never
substitutes eager and reports it as compiled.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import benchmark_g05_full as base
from g05_npu_graphs import (
    GDN_HELPERS,
    install_ae_backend,
    install_gdn_backends,
    install_module_backend,
    validate_graph_options,
)


def profile_overrides(profile: str, weights: str) -> list[str]:
    return [
        "model.use_torch_compile=false",
        f"model.model_weights_to_bf16={str(weights == 'bf16').lower()}",
        f"model.model_arch.predict_cot={str(profile == 'full').lower()}",
        f"model.model_arch.discrete_action={str(profile == 'full').lower()}",
        "model.model_arch.continuous_action=true",
        "model.model_arch.return_continuous_action=true",
    ]


def validate_profile(config: Any, profile: str, weights: str) -> dict[str, Any]:
    expected = {
        "model.use_torch_compile": False,
        "model.model_weights_to_bf16": weights == "bf16",
        "model.model_arch.predict_cot": profile == "full",
        "model.model_arch.discrete_action": profile == "full",
        "model.model_arch.continuous_action": True,
        "model.model_arch.return_continuous_action": True,
        "model.model_arch.fm.num_inference_steps": 10,
        "model.model_arch.fm.use_correlated_noise": False,
        "model.model_arch.action_dim": 27,
        "model.model_arch.cond_steps": 1,
        "model.model_arch.max_text_tokens": 160,
        "model.model_arch.input_preprocessor.pred_eov": True,
        "model.model_arch.ar.do_sample": False,
        "model.model_arch.ar.max_new_tokens": 300,
        "model.model_arch.attn_implementation": "eager",
        "model.processor.num_output_cameras": 3,
        "data.action_size": 32,
        "data.obs_size": 1,
    }
    observed = {key: base._select(config, key) for key in expected}
    mismatches = {key: {"expected": wanted, "actual": observed[key]}
                  for key, wanted in expected.items() if observed[key] != wanted}
    if mismatches:
        raise RuntimeError("Optimization profile mismatch: " + json.dumps(mismatches))
    if profile == "full":
        observed.update(base.validate_shipped_default_contract(config))
    return observed


def validate_outputs(core: Mapping[str, Any], served: Mapping[str, Any], profile: str) -> dict:
    import numpy as np
    required = ["action", "ar_action"] if profile == "full" else ["action"]
    expected = {f"core.{name}": [1, 32, 27] for name in required}
    expected.update({f"served.{name}": [1, 32, width] for name, width in base.STATE_DIMS.items()})
    checked = {}
    for prefix, values in (("core", core), ("served", served)):
        for name, value in values.items():
            if not hasattr(value, "shape"):
                continue
            if hasattr(value, "isfinite"):
                finite = bool(value.isfinite().all().item())
            else:
                finite = bool(np.isfinite(np.asarray(value)).all())
            path = f"{prefix}.{name}"
            if not finite:
                raise RuntimeError(f"Nonfinite output: {path}")
            checked[path] = {"shape": list(value.shape), "all_finite": True}
    for path, shape in expected.items():
        if path not in checked or checked[path]["shape"] != shape:
            raise RuntimeError(f"Missing/wrong-shaped output: {path}; expected {shape}")
    return checked


def compare_outputs(reference: Mapping[str, Any], actual: Mapping[str, Any], *, atol: float, rtol: float) -> dict:
    import numpy as np

    def array(value):
        return value.float().numpy() if hasattr(value, "float") else np.asarray(value, dtype=np.float64)

    if "action" not in reference or set(reference) != set(actual):
        raise RuntimeError("Verification requires matching nonempty action output keys")
    results = {}
    for key, ref_value in reference.items():
        ref, got = array(ref_value), array(actual[key])
        if ref.shape != got.shape:
            raise RuntimeError(f"Verification output shape changed: {key}")
        difference = np.asarray(got, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
        results[key] = {
            "shape": list(ref.shape), "all_finite": finite,
            "max_abs_error": float(np.max(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "relative_l2_error": float(np.linalg.norm(difference.ravel()) / max(float(np.linalg.norm(ref.ravel())), 1e-12)),
            "reference_max_abs": float(np.max(np.abs(ref))),
            "passed": finite and bool(np.allclose(got, ref, rtol=rtol, atol=atol)),
        }
    return {"passed": all(row["passed"] for row in results.values()), "atol": atol, "rtol": rtol,
            "scope": "Same-weight, same-seed eager vs selected AE/VLM/GDN backends; not BF16-vs-FP32 or task accuracy", "outputs": results}


def parameter_dtypes(policy: Any) -> dict:
    roots = [policy]
    codec = getattr(getattr(policy, "action_tokenizer", None), "action_tokenizer", None)
    if codec is not None and hasattr(codec, "parameters"):
        roots.append(codec)
    seen, result = set(), {}
    for root in roots:
        for parameter in root.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            key = f"{parameter.dtype}@{parameter.device}"
            row = result.setdefault(key, {"numel": 0, "bytes": 0})
            row["numel"] += parameter.numel()
            row["bytes"] += parameter.numel() * parameter.element_size()
    return result


def seed_all(torch: Any, np: Any, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(seed)


def snapshot_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep results independent of buffers reused by later graph executions."""
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value.copy()
        for key, value in outputs.items()
    }


def build_verification_cases(raw: Mapping[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    import numpy as np

    alternate_seed = (seed + 1) % (2 ** 32)
    alternate = base.build_synthetic_raw_observation(alternate_seed, raw["task"])
    # The base builder varies images only. Also vary proprioception while keeping
    # its schema and shapes, and keep task text fixed to preserve prompt length.
    alternate["state"] = {
        key: (-value + np.float32(0.03125)).astype(np.float32)
        for key, value in alternate["state"].items()
    }
    cases = {}
    for name, observation, generation_seed in (("a", raw, seed), ("b", alternate, alternate_seed)):
        cases[name] = {
            "raw_obs": observation, "seed": generation_seed,
            "metadata": {
                "generation_seed": generation_seed,
                "observation_sha256": base.observation_sha256(observation),
                "task": observation["task"],
                "shapes": {f"{group}.{key}": list(value.shape)
                           for group in ("images", "state") for key, value in observation[group].items()},
            },
        }
    if cases["a"]["metadata"]["shapes"] != cases["b"]["metadata"]["shapes"]:
        raise RuntimeError("Verification A/B raw input shapes must match")
    if cases["a"]["metadata"]["observation_sha256"] == cases["b"]["metadata"]["observation_sha256"]:
        raise RuntimeError("Verification requires distinct A/B observations")
    return cases


def collect_eager_references(cases, execute) -> dict[str, dict[str, Any]]:
    references = {}
    for name, case in cases.items():
        print(f"[progress] Eager verification reference {name.upper()} (seed {case['seed']})", flush=True)
        timing, outputs = execute(case)
        references[name] = {"timing_ms": timing, "outputs": snapshot_outputs(outputs)}
    return references


def verify_request_replay(cases, references, cold_result, execute, *, atol: float, rtol: float) -> dict:
    """Check first use and repeated A/B inputs without adding benchmark samples.

    B's first call can specialize a second graph (e.g. full-profile CoT changes
    the KV length), so both A and B are repeated after their first selected call.
    This validates the returned values; it does not count backend graph replays.
    """
    cold_timing, cold_outputs = cold_result
    selected = [("cold_a", "a", cold_timing, snapshot_outputs(cold_outputs))]
    for label, name in (("first_b", "b"), ("replay_a", "a"), ("replay_b", "b")):
        print(f"[progress] Verification {label} (seed {cases[name]['seed']})", flush=True)
        timing, outputs = execute(cases[name])
        selected.append((label, name, timing, snapshot_outputs(outputs)))

    # Compare after all calls: every retained output must remain independent of
    # later graph writes, not merely look correct immediately after capture.
    rows = []
    for label, name, timing, outputs in selected:
        comparison = compare_outputs(references[name]["outputs"], outputs, atol=atol, rtol=rtol)
        rows.append({"case": label, "input_case": name, "timing_ms": timing, **comparison})
    contrast = compare_outputs(references["a"]["outputs"], references["b"]["outputs"], atol=atol, rtol=rtol)
    distinguishable = not contrast["outputs"]["action"]["passed"]
    aggregate = {}
    for key in rows[0]["outputs"]:
        values = [row["outputs"][key] for row in rows]
        aggregate[key] = {
            "shape": values[0]["shape"],
            "max_abs_error": max(value["max_abs_error"] for value in values),
            "rmse": max(value["rmse"] for value in values),
            "relative_l2_error": max(value["relative_l2_error"] for value in values),
            "all_finite": all(value["all_finite"] for value in values),
            "passed": all(value["passed"] for value in values),
        }
    return {
        "passed": distinguishable and all(row["passed"] for row in rows),
        "atol": atol, "rtol": rtol,
        "scope": "Same-weight eager vs selected backend, changed observations/seeds, A/B/A/B requests; no task-accuracy claim",
        "shape_coverage": "Same raw shapes/task text; full-profile generated CoT may specialize different KV lengths. Arbitrary shapes are not covered.",
        "replay_coverage": "A and B each run again after first selected use; backend replay counts are not instrumented",
        "output_snapshot": "Independent CPU copies, compared after all selected requests",
        "reference_inputs_distinguishable": distinguishable,
        "reference_contrast": contrast,
        "inputs": {name: case["metadata"] for name, case in cases.items()},
        "eager_reference_requests": {name: reference["timing_ms"] for name, reference in references.items()},
        "cases": rows, "outputs": aggregate,
    }


def run_one(*, raw_obs, processor, build_obs_dict, inferencer, runtime, profile):
    runtime.synchronize()
    start = time.monotonic()
    observation = build_obs_dict(raw_obs, processor)
    built = time.monotonic()
    results, timings = inferencer.infer_with_timing([observation])
    runtime.synchronize()
    end = time.monotonic()
    sample = {key: float(value) for key, value in timings.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)}
    sample.update(raw_build_obs_ms=(built - start) * 1000, request_total_ms=(end - start) * 1000)
    sample.update(inferencer.last_transfer_timing)
    sample["cpu_preprocess_total_ms"] = sample["raw_build_obs_ms"] + sample.get("preprocess_ms", 0)
    calls = inferencer.last_model_metadata["ar_generation_calls"]
    expected = ["cot", "action"] if profile == "full" else []
    if [call["stage"] for call in calls] != expected:
        raise RuntimeError(f"AR calls do not match {profile} profile: {calls}")
    sample["ar_generation_calls"] = calls
    if calls:
        sample["cot_raw_generated_tokens"] = calls[0]["generated_tokens"]
        sample["action_ar_raw_generated_tokens"] = calls[1]["generated_tokens"]
        sample["ar_decoded_action_tokens"] = sum(inferencer.last_model_metadata.get("ar_token_lengths", []))
    sample["output_validation"] = validate_outputs(inferencer.last_core_actions, results[0], profile)
    return sample, snapshot_outputs(inferencer.last_core_actions)


def run(args) -> dict:
    print("[progress] Validate source/checkpoint", flush=True)
    source = args.source_dir.expanduser().resolve()
    revision = base.source_revision(source)
    if revision != base.EXPECTED_SOURCE_REVISION and not args.allow_source_revision:
        raise RuntimeError("Unexpected source revision")
    layout = base.resolve_asset_layout(args.checkpoint, args.asset_root)
    checkpoint_hash = base.validate_asset_layout(layout, verify_checkpoint_hash=not args.skip_checkpoint_sha256)
    sys.path[:0] = [str(source / "src"), str(source)]
    if args.device.startswith("npu"):
        importlib.import_module("torch_npu")
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.utils.checkpoint.ckpt_utils import find_run_dir, load_config_from_run_dir
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.eval.eval_utils import filter_embodiment
    from scripts.serve_policy import setup, build_obs_dict

    runtime = base.DeviceRuntime(torch, args.device)
    seed_all(torch, np, args.seed)
    raw = base.build_synthetic_raw_observation(args.seed, args.task)
    overrides = base._override_paths(layout) + profile_overrides(args.profile, args.weights) + list(args.override)
    with base.working_directory(source):
        started = time.monotonic()
        cfg, compatibility = base.load_r1lite_checkpoint_config(
            source_dir=source, run_dir=find_run_dir(str(layout["checkpoint"])), checkpoint=layout["checkpoint"],
            overrides=overrides, omega_conf=OmegaConf, load_config_from_run_dir=load_config_from_run_dir)
        if "embodiment_datasets" in cfg.data:
            filter_embodiment(cfg, base.DEFAULT_EMBODIMENT)
        contract = validate_profile(cfg, args.profile, args.weights)
        print(f"[progress] Load {args.profile} / {args.weights}", flush=True)
        policy, processor = setup(cfg, device=args.device)
        runtime.synchronize()
        load_ms = (time.monotonic() - started) * 1000
        print(f"[progress] Loaded in {load_ms:.3f} ms", flush=True)
        Inferencer = base.make_instrumented_inferencer(PolicyInferencer, torch, dict_apply, runtime)
        inferencer = Inferencer(policy, processor, device=args.device)
        arguments = dict(raw_obs=raw, processor=processor, build_obs_dict=build_obs_dict,
                         inferencer=inferencer, runtime=runtime, profile=args.profile)
        verification = None
        eager_reference = None
        verification_cases = None
        references = None
        def execute_case(case):
            seed_all(torch, np, case["seed"])
            return run_one(**{**arguments, "raw_obs": case["raw_obs"]})
        if args.verify:
            verification_cases = build_verification_cases(raw, args.seed)
            references = collect_eager_references(verification_cases, execute_case)
            eager_reference = references["a"]["timing_ms"]
        original, compilation = install_ae_backend(torch, policy, args.ae_backend, options=args.ae_options)
        original_vlm = policy.model.vlm.forward
        vlm_compilation = None
        gdn_installation = None
        try:
            original_vlm, vlm_compilation = install_module_backend(
                torch, policy.model.vlm, args.vlm_backend, "policy.model.vlm.forward only", options=args.vlm_options)
            gdn_installation = install_gdn_backends(
                torch, args.gdn_backend, helpers=args.gdn_helpers, options=args.gdn_options)
            runtime.reset_peak_memory()
            seed_all(torch, np, args.seed)
            print("[progress] First selected-backend request (includes compile if enabled)", flush=True)
            cold, cold_outputs = run_one(**arguments)
            cold_memory = runtime.memory()
            if args.verify:
                verification = verify_request_replay(
                    verification_cases, references, (cold, cold_outputs), execute_case, atol=args.atol, rtol=args.rtol)
                print("[verification] " + json.dumps(verification), flush=True)
                if not verification["passed"]:
                    failed = [row["case"] for row in verification["cases"] if not row["passed"]]
                    raise RuntimeError(f"Cross-request eager/backend verification failed: {failed}; distinct references={verification['reference_inputs_distinguishable']}")
                del references
            seed_all(torch, np, args.seed)
            print(f"[progress] First request {cold['request_total_ms']:.3f} ms", flush=True)
            warmups, samples = [], []
            for phase, count, destination in (("warmup", args.warmup, warmups), ("measure", args.iterations, samples)):
                if phase == "measure":
                    runtime.reset_peak_memory()
                for index in range(count):
                    sample, outputs = run_one(**arguments)
                    destination.append(sample)
                    print(f"[progress] {phase} {index+1}/{count}: request {sample['request_total_ms']:.3f} ms; model {sample['model_predict_action_ms']:.3f} ms", flush=True)
            memory = runtime.memory()
            dtypes = parameter_dtypes(policy)
        finally:
            if gdn_installation is not None:
                gdn_installation.restore()
            policy.model.action_expert.forward = original
            policy.model.vlm.forward = original_vlm

    return {
        "schema_version": 1, "benchmark": "g05-r1lite-explicit-optimization-profile",
        "result_scope": f"synthetic R1 Lite {args.profile} execution experiment",
        "comparison_scope": "Same g05-base weights/input across Ascend experiments; not exact LIBERO 3090 checkpoint/input parity",
        "accuracy_scope": "finite/shape checks and optional same-weight eager/backend comparison; no robot task accuracy",
        "profile": args.profile, "weight_conversion": args.weights, "parameter_dtypes": dtypes,
        "autocast": {"enabled": True, "dtype": "bfloat16"},
        "fp32_parameters_policy": "Official serve_policy.setup/apply_fp32_params after optional BF16 weight conversion",
        "compilation": compilation, "vlm_compilation": vlm_compilation, "verification": verification,
        "gdn_compilation": gdn_installation.metadata,
        "eager_reference_request": eager_reference,
        "source": {"path": str(source), "revision": revision, "tree_state": base.source_tree_state(source)},
        "checkpoint": {"path": str(layout["checkpoint"]), "sha256": checkpoint_hash,
                       "hash_verification_skipped": args.skip_checkpoint_sha256},
        "input": {"kind": "deterministic synthetic R1 Lite", "sha256": base.observation_sha256(raw),
                  "seed": args.seed, "task": args.task, "cameras": 3, "model_camera_shape": [3,256,256],
                  "raw_state_dim": 14, "action_dim": 27, "horizon": 32},
        "configuration_contract": contract, "configuration_compatibility": compatibility,
        "overrides": overrides,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "device": runtime.info(), "cpu_threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}},
        "timing_boundaries": {
            "request_total_ms": "Raw observation build + CPU preprocess + H2D + policy + D2H + postprocess; excludes validation",
            "model_predict_action_ms": "Prepared device batch -> policy.predict_action -> normalized action tensors; includes policy Python preparation",
            "forward_inference_total_ms": "Prepared samples -> prefill -> enabled CoT/action heads -> normalized action tensors; upstream synchronized boundary",
            "cold": "First selected-backend request; includes compilation if enabled. If --verify, eager A/B references ran before it.",
            "verification": "A/B/A/B selected requests are checked before warmup/measurement; excluded from measurement samples and may populate graph caches",
        },
        "load_ms": load_ms, "cold_request": {"timing_ms": cold, "memory": cold_memory},
        "warmup": {"iterations": args.warmup, "samples": warmups},
        "measurement": {"iterations": args.iterations, "samples": samples, "summary": base.summarize_samples(samples) if samples else {}, "memory": memory},
    }


def parse_graph_options(value: str) -> dict[str, bool]:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("NPU graph options must be a JSON object")
        return validate_graph_options(parsed)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_gdn_helpers(value: str) -> tuple[str, ...]:
    helpers = tuple(part.strip() for part in value.split(","))
    if not helpers or len(set(helpers)) != len(helpers) or set(helpers) - set(GDN_HELPERS):
        raise argparse.ArgumentTypeError("Use distinct comma-separated GDN helpers: " + ",".join(GDN_HELPERS))
    return helpers


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(warmup=3, iterations=10)
    parser.add_argument("--profile", choices=("full", "fm-only"), default="full")
    parser.add_argument("--weights", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--ae-backend", choices=("eager", "torchair", "npugraph_ex"), default="eager")
    parser.add_argument("--vlm-backend", choices=("eager", "npugraph_ex"), default="eager",
                        help="Optional independent compilation of VLM.forward; default keeps it eager")
    parser.add_argument("--gdn-backend", choices=("eager", "npugraph_ex"), default="eager",
                        help="Compile selected NPU GatedDeltaNet tensor helpers without compiling the whole VLM")
    parser.add_argument("--gdn-helpers", type=parse_gdn_helpers, default=("chunk", "recurrent"),
                        help="Comma-separated helper names: chunk,recurrent,causal-conv")
    for scope in ("ae", "vlm", "gdn"):
        parser.add_argument(f"--{scope}-options", type=parse_graph_options, default=None,
                            help="JSON object of npugraph_ex boolean options; e.g. frozen_parameter,static_kernel_compile,super_kernel_optimize")
    parser.add_argument("--verify", action="store_true", help="Check changed-observation/seed A/B/A/B requests against same-weight eager references before timing")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke_only:
        args.warmup, args.iterations = 0, 0
    if args.warmup < 0 or (args.iterations < 1 and not args.smoke_only):
        parser.error("warmup must be nonnegative and iterations positive")
    if args.action_steps != 16:
        parser.error("R1 Lite service chunk is 16 steps (model horizon remains 32)")
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        parser.error("Verification tolerances must be finite and nonnegative")
    for scope in ("ae", "vlm", "gdn"):
        options = getattr(args, f"{scope}_options")
        backend = getattr(args, f"{scope}_backend")
        if options and backend not in {"npugraph_ex", "torchair"}:
            parser.error(f"--{scope}-options requires a graph backend")
        if options and backend == "torchair" and set(options) - {"frozen_parameter", "softplus_decomposition", "single_stream"}:
            parser.error("Unsupported TorchAir option")
        if options and options.get("softplus_decomposition") and not args.verify:
            parser.error("Softplus decomposition requires --verify")
    if args.vlm_backend != "eager" and args.gdn_backend != "eager":
        parser.error("Measure VLM and GDN helper compilation in separate experiments")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    print(json.dumps(report["measurement"]["summary"].get("model_predict_action_ms", report["cold_request"]["timing_ms"]["model_predict_action_ms"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
