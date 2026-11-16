#!/usr/bin/env python3
"""Standalone experiment: FP32 sequential GDN row recurrence in one Triton kernel.

No production patch, model loading, or remote execution. CPU mode uses NumPy only.
NPU mode requires the existing TorchNPU/Triton-Ascend image; do not replace packages.

  python probe_gdn_row_inverse_triton.py --mode cpu --output cpu.json
  python probe_gdn_row_inverse_triton.py --mode npu --batch 80 --output npu.json
  python probe_gdn_row_inverse_triton.py --mode npu --batch 80 \
      --benchmark-torch-graph --output npu-graph.json

L: finite contiguous FP32 [N,32,32], strictly lower triangular.
For i=1..31: row=L[i,:i]; L[i,:i]=row+sum(row[:,None]*L[:i,:i], axis=0).
Finally add I. The existing row dependency order is retained, with no doubling,
tl.dot, TF32/HF32, or lower-precision storage. FP fusion is explicitly disabled.
tl.sum may use a different reduction tree (and padded zeros) than torch.sum on
an i-by-i view. Bitwise agreement is measured, never assumed.

The upstream recurrence is in gated_deltanet.py:105-109 at source baseline
89f2322b4ad016e192437adc1a2c253b05bab246. Ascend launch/reduction references:
https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/05-layer-norm.py
https://gitee.com/ascend/triton-ascend/blob/master/ascend/backend/compiler.py
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np


def validate_array(value):
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[1:] != (32, 32):
        raise ValueError("Expected FP32 [N,32,32]")
    if value.shape[0] < 1 or not value.flags.c_contiguous:
        raise ValueError("Expected positive N and contiguous storage")
    if not np.isfinite(value).all() or np.count_nonzero(np.triu(value)):
        raise ValueError("Expected finite, strictly lower-triangular L")


def numpy_original(value):
    """CPU FP32 formula oracle, not a claim to reproduce Torch reduction bits."""
    validate_array(value)
    out = value.copy()
    for i in range(1, 32):
        row = out[:, i, :i].copy()
        sub = out[:, :i, :i].copy()
        out[:, i, :i] = row + (row[..., None] * sub).sum(axis=-2, dtype=np.float32)
    return out + np.eye(32, dtype=np.float32)


def numpy_masked_model(value):
    """Independent CPU model of the fixed-width masked Triton data flow."""
    validate_array(value)
    out = value.copy()
    rows = np.arange(32)[:, None]
    cols = np.arange(32)[None, :]
    for i in range(1, 32):
        row = np.where(rows[None] == i, out, np.float32(0)).sum(axis=1, dtype=np.float32)
        sub = np.where(((rows < i) & (cols < i))[None], out, np.float32(0))
        update = row + (row[:, :, None] * sub).sum(axis=1, dtype=np.float32)
        out = np.where(((rows == i) & (cols < i))[None], update[:, None, :], out)
    return out + np.eye(32, dtype=np.float32)


def metrics(reference, actual, atol, rtol):
    ref = np.asarray(reference)
    out = np.asarray(actual)
    delta = out.astype(np.float64) - ref.astype(np.float64)
    finite = bool(np.isfinite(ref).all() and np.isfinite(out).all())
    return {
        "shape": list(out.shape), "all_finite": finite,
        "max_abs_error": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(ref.astype(np.float64).ravel()), 1e-30)),
        "value_exact": bool(np.array_equal(ref, out)),
        "bitwise_exact": bool(ref.dtype == out.dtype and ref.tobytes() == out.tobytes()),
        "passed_tolerance": bool(finite and np.allclose(out, ref, atol=atol, rtol=rtol)),
        "atol": atol, "rtol": rtol,
    }


def residual(value, result):
    identity = np.eye(32, dtype=np.float64)
    error = (identity - value.astype(np.float64)) @ result.astype(np.float64) - identity
    return float(np.max(np.abs(error)))


def cases(batch, seed):
    rng = np.random.default_rng(seed)
    shape = (batch, 32, 32)
    yield "zero", np.zeros(shape, dtype=np.float32)
    yield "negative_lower_ones", np.ascontiguousarray(-np.tril(np.ones(shape, dtype=np.float32), -1))
    for scale in (0.03125, 0.125, 0.5):
        value = (rng.standard_normal(shape) * scale).astype(np.float32)
        yield f"random_scale_{scale}", np.ascontiguousarray(np.tril(value, -1))
    for perturbation in (0.0, 1e-4, 1e-2):
        # GDN-like near-collinear normalized keys; negative Gram coefficients.
        base = rng.standard_normal((batch, 1, 128))
        keys = base + perturbation * rng.standard_normal((batch, 32, 128))
        keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
        gram = keys @ keys.transpose(0, 2, 1)
        g = -0.001 * np.arange(1, 33, dtype=np.float64)
        decay = np.exp(g[:, None] - g[None, :])
        value = np.tril(-(gram * decay), -1).astype(np.float32)
        yield f"near_collinear_{perturbation}", np.ascontiguousarray(value)


def make_triton_kernel():
    # Lazy import permits --mode cpu and AST checks on machines without Torch.
    global triton, tl
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")

    @triton.jit
    def row_inverse_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        rows = tl.arange(0, BLOCK)
        cols = tl.arange(0, BLOCK)
        offsets = program * BLOCK * BLOCK + rows[:, None] * BLOCK + cols[None, :]
        matrix = tl.load(input_ptr + offsets).to(tl.float32)
        # Explicit ordered updates. Each iteration sees all earlier updated rows.
        for i in tl.static_range(1, BLOCK):
            row = tl.sum(tl.where(rows[:, None] == i, matrix, 0.0), axis=0)
            sub = tl.where((rows[:, None] < i) & (cols[None, :] < i), matrix, 0.0)
            products = row[:, None] * sub
            updated_row = row + tl.sum(products, axis=0)
            matrix = tl.where(
                (rows[:, None] == i) & (cols[None, :] < i),
                updated_row[None, :], matrix,
            )
        matrix = matrix + tl.where(rows[:, None] == cols[None, :], 1.0, 0.0)
        tl.store(output_ptr + offsets, matrix)

    return row_inverse_kernel


def torch_original(torch, value):
    # Mirrors the upstream slice lengths, clones and row update order exactly.
    out = value.clone()
    for i in range(1, 32):
        row = out[..., i, :i].clone()
        sub = out[..., :i, :i].clone()
        out[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return out + torch.eye(32, dtype=out.dtype, device=out.device)


def distribution(samples):
    return {"count": len(samples), "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples}


def json_safe(value):
    """Preserve a failing stress report even when candidate arithmetic overflows."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def benchmark(torch, fn, warmup, repeats):
    torch.npu.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"first_call_ms": cold_ms, **distribution(samples)}, out


def run_cpu(args, result):
    rows = []
    for name, value in cases(args.batch, args.seed):
        original = numpy_original(value)
        candidate = numpy_masked_model(value)
        row = {"name": name, "masked_model_vs_original": metrics(original, candidate, args.atol, args.rtol),
               "original_fp64_residual_max": residual(value, original),
               "masked_fp64_residual_max": residual(value, candidate)}
        if name == "negative_lower_ones":
            exact_inverse = np.eye(32, dtype=np.float32) - np.eye(32, k=-1, dtype=np.float32)
            row["known_exact_inverse_matches"] = bool(np.array_equal(candidate, np.broadcast_to(exact_inverse, candidate.shape)))
            if not row["known_exact_inverse_matches"]:
                raise AssertionError("Negative-lower-ones inverse failed")
        rows.append(row)
    result["cpu_validation"] = rows
    result["cpu_scope"] = "NumPy only; validates indexing/formula, not Triton compilation or Torch/NPU equivalence"
    return all(row["masked_model_vs_original"]["passed_tolerance"] for row in rows)


def run_npu(args, result):
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "config"):
        torch.npu.config.allow_internal_format = False
    kernel = make_triton_kernel()
    result["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                             "triton": triton.__version__, "device": args.device,
                             "name": torch.npu.get_device_name(args.device)}

    def launch(value, out):
        if value.dtype != torch.float32 or value.shape[1:] != (32, 32) or not value.is_contiguous():
            raise ValueError("Triton requires contiguous FP32 [N,32,32]")
        kernel[(value.shape[0],)](value, out, BLOCK=32, enable_fp_fusion=False)
        return out

    validation = []
    with torch.inference_mode(), torch.autocast("npu", enabled=False):
        for name, value in cases(args.batch, args.seed):
            print(f"[progress] Validate {name}, N={args.batch}", flush=True)
            validate_array(value)
            cpu_input = torch.from_numpy(value.copy())
            cpu_reference = torch_original(torch, cpu_input).numpy()
            npu_input = cpu_input.to(args.device)
            output = torch.empty_like(npu_input)
            start = time.perf_counter()
            launch(npu_input, output)
            torch.npu.synchronize()
            launch_ms = (time.perf_counter() - start) * 1000
            actual = output.cpu().numpy().copy()
            npu_reference = torch_original(torch, npu_input).cpu().numpy()
            validation.append({"name": name, "first_launch_ms": launch_ms,
                "triton_vs_npu_original": metrics(npu_reference, actual, args.atol, args.rtol),
                "triton_vs_cpu_torch_original": metrics(cpu_reference, actual, args.atol, args.rtol),
                "npu_vs_cpu_torch_original": metrics(cpu_reference, npu_reference, args.atol, args.rtol),
                "triton_fp64_residual_max": residual(value, actual),
                "input_unmodified": bool(np.array_equal(value, npu_input.cpu().numpy()))})
        result["npu_validation"] = validation
        # Only the GDN-like random case is timed; not an entire GDN helper/model.
        value = dict(cases(args.batch, args.seed))["random_scale_0.125"]
        tensor = torch.from_numpy(value.copy()).to(args.device)
        output = torch.empty_like(tensor)
        identity = torch.eye(32, dtype=torch.float32, device=args.device).expand(args.batch, -1, -1)
        variants = {
            "triton_single_kernel_preallocated_output": lambda: launch(tensor, output),
            "torch_original_31_rows_eager": lambda: torch_original(torch, tensor),
            "torch_matrix_triangular_solve_eager": lambda: torch.linalg.solve_triangular(
                identity - tensor, identity, upper=False, unitriangular=True),
        }
        if args.benchmark_torch_graph:
            compiled = torch.compile(lambda value: torch_original(torch, value),
                                     backend="npugraph_ex", fullgraph=True, dynamic=False)
            variants["torch_original_31_rows_npugraph_ex"] = lambda: compiled(tensor)
        measurements = {}
        original = torch_original(torch, tensor).cpu().numpy()
        for name, fn in variants.items():
            print(f"[progress] Benchmark {name}", flush=True)
            timing, measured_output = benchmark(torch, fn, args.warmup, args.repeats)
            measurements[name] = {**timing, "output_vs_npu_original": metrics(
                original, measured_output.cpu().numpy(), args.atol, args.rtol)}
        result["benchmark"] = measurements
    return all(row["triton_vs_npu_original"]["passed_tolerance"] and row["input_unmodified"]
               for row in validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch", type=int, default=80, help="N matrices; 16 heads x 5 chunks = 80")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--benchmark-torch-graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error("Invalid size, repetition count or tolerance")
    result = {"schema_version": 1, "experiment": "gdn-row-inverse-triton-sequential-fp32",
        "mode": args.mode, "shape": [args.batch, 32, 32], "seed": args.seed,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision": {"input_storage_accumulation_output": "float32", "fp_fusion": False,
                      "dot_or_cube_operations": False, "tf32_hf32": False,
                      "row_update_order": "1..31 in order; no doubling",
                      "within_row_reduction": "tl.sum fixed-width masked reduction; tree may differ from torch.sum",
                      "bitwise_equivalence_claimed": False},
        "timing_scope": "inverse recurrence+I only, one input per call; excludes GDN projections/decay/state and full model",
        "timing_details": "synchronized wall clock; Triton output preallocated; Torch eager includes clone/temporary allocation; CPU refs and H2D excluded",
        "warmup": args.warmup, "repeats": args.repeats,
        "gate_scope": "diagnostic tolerances only; not model or closed-loop acceptance"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        passed = run_cpu(args, result)
        if args.mode == "npu":
            passed = run_npu(args, result) and passed
        result["status"] = "passed" if passed else "numerical_mismatch"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result["nonfinite_metric_encoding"] = "Nonfinite scalar metrics are null; all_finite/pass fields remain false"
    args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
