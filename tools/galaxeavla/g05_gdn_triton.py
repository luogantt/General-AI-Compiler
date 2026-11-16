#!/usr/bin/env python3
"""Opt-in FP32 GDN row inverse, wrapped as an opaque Torch custom operator.

Imports do not load Torch, Triton or NPU. Install only in an inference experiment
that independently verifies the whole model; no automatic backend fallback.
Only the strict-lower 32x32 inverse changes. Other chunk math is reused from
g05_gdn_triangular with its original precision policy and QK epsilon (1e-6).

The 31 sequential row dependencies match the original GDN recurrence, while
Triton's padded tl.sum tree can differ from Torch's reduction tree. No claim of
bitwise equivalence. This is not the numerically unstable doubling formulation.

Owner-run opaque-op/ACLGraph microprobe (no model download/loading):
  python g05_gdn_triton.py --mode probe --device npu:0 --batch 80 \
      --backend torchair_aclgraph --output triton-custom-op-aclgraph.json

The registered operator is process-persistent; restore() removes only the
upstream chunk dispatch. Captured graph output buffers may be reused by the
backend, so validation snapshots each result before the next request.
"""

import argparse
import functools
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
import traceback


GDN_MODULE = "g05.models.g05.qwen35.gated_deltanet"
OP_NAME = "gac_g05::row_inverse32_fp32"
_MARKER = "_g05_gdn_triton_installation"
_OTHER_MARKERS = ("_g05_gdn_triangular_installation", "_g05_npu_graph_installation", _MARKER)
_ROW_OP = None
_KERNEL = None


def _make_kernel():
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


def _check_tensor(torch, value):
    """Metadata-only checks, also valid under FakeTensor; no item()/data reads."""
    if value.device.type != "npu":
        raise ValueError("GDN Triton inverse supports NPU only; no fallback")
    if value.dtype != torch.float32:
        raise TypeError("GDN Triton inverse requires FP32 storage")
    if value.ndim != 3 or value.shape[1:] != (32, 32) or value.shape[0] < 1:
        raise ValueError("GDN Triton inverse requires [N,32,32], N > 0")
    if not value.is_contiguous():
        raise ValueError("GDN Triton inverse requires contiguous input")
    if value.requires_grad:
        raise ValueError("GDN Triton inverse supports inference only")


def register_row_inverse_op():
    """Register once; fake and real implementations have identical metadata."""
    global _ROW_OP, _KERNEL
    if _ROW_OP is not None:
        return _ROW_OP
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    kernel = _make_kernel()

    def implementation(value):
        _check_tensor(torch, value)
        # No persistent output buffer and no alias of input. Graph replay may
        # reuse allocator buffers according to the graph backend's own contract.
        output = torch.empty_like(value)
        kernel[(value.shape[0],)](value, output, BLOCK=32, enable_fp_fusion=False)
        return output

    # Assign actual classes rather than unresolved lazy-import annotations.
    implementation.__annotations__ = {"value": torch.Tensor, "return": torch.Tensor}
    custom_op = torch.library.custom_op(OP_NAME, mutates_args=(), device_types="npu")(implementation)

    @custom_op.register_fake
    def fake(value):
        _check_tensor(torch, value)
        return torch.empty_like(value)

    _KERNEL, _ROW_OP = kernel, custom_op
    return custom_op


def install_triton_chunk(*, verify=False):
    """Install before graph compilation; require explicit whole-model checking."""
    if verify is not True:
        raise ValueError("GDN Triton requires verify=True and independent full-model verification")
    module = importlib.import_module(GDN_MODULE)
    original = module.torch_chunk_gated_delta_rule
    if any(getattr(original, marker, False) for marker in _OTHER_MARKERS):
        raise RuntimeError("Install GDN Triton only once, before any algorithm or graph replacement")
    shared = importlib.import_module("g05_gdn_triangular")
    inverse_op = register_row_inverse_op()

    def inverse_nd(value):
        shape = value.shape
        if shape[-2:] != (32, 32):
            raise ValueError("GDN Triton requires chunk_size=32")
        return inverse_op(value.reshape(-1, 32, 32).contiguous()).reshape(shape)

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        query = args[0] if args else kwargs.get("query")
        if getattr(getattr(query, "device", None), "type", None) != "npu":
            raise ValueError("GDN Triton chunk requires NPU inputs; no fallback")
        chunk_size = args[5] if len(args) > 5 else kwargs.get("chunk_size", 64)
        if chunk_size != 32:
            raise ValueError("GDN Triton chunk requires chunk_size=32")
        if "inverse_impl" in kwargs:
            raise ValueError("GDN Triton owns the inverse_impl callback")
        return shared.torch_chunk_gated_delta_rule_solve(*args, **kwargs, inverse_impl=inverse_nd)

    setattr(dispatch, _MARKER, True)
    # The existing triangular installer also rejects this marker, preventing
    # it from silently nesting a second algorithm over this one.
    setattr(dispatch, "_g05_gdn_triangular_installation", True)
    metadata = {
        "name": "gdn-chunk-fp32-triton-sequential-row-inverse", "experimental": True,
        "default_enabled": False, "scope": f"{GDN_MODULE}.torch_chunk_gated_delta_rule; NPU chunk_size=32 only",
        "operator": OP_NAME, "operator_schema": "(Tensor value) -> Tensor",
        "mutates_args": [], "output_aliases_input": False, "persistent_output_cache": False,
        "row_update_order": "1..31 sequential, then +I", "row_update_order_preserved": True,
        "within_row_reduction": "FP32 tl.sum; masked length-32 reduction tree may differ from torch.sum",
        "accumulation_order_changed": True, "bitwise_equivalence_claimed": False,
        "input_and_output_dtype": "torch.float32", "accumulation_dtype": "torch.float32",
        "enable_fp_fusion": False, "dot_tf32_hf32_used": False,
        "inverse_epsilon_added": False, "qk_l2norm_epsilon": 1e-6,
        "other_precision_policy_changed": False, "chunk_size_changed": False,
        "fallback_allowed": False, "verification_requested": True,
        "requires_independent_npu_and_full_model_validation": True,
        "runtime_validation": "not performed by installer",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_chunk_sha256": hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
    }
    module.torch_chunk_gated_delta_rule = dispatch
    return shared.TriangularInstallation(metadata, module, original, dispatch)


def _original(torch, value):
    output = value.clone()
    for i in range(1, 32):
        row = output[..., i, :i].clone()
        sub = output[..., :i, :i].clone()
        output[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return output + torch.eye(32, dtype=output.dtype, device=output.device)


def _stats(torch, reference, actual, atol, rtol):
    reference, actual = reference.double(), actual.double()
    delta = actual - reference
    return {"finite": bool(torch.isfinite(actual).all()),
            "max_abs_error": delta.abs().max().item(),
            "relative_l2": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
            "exact": bool(torch.equal(reference, actual)),
            "passed": bool(torch.isfinite(actual).all() and torch.allclose(actual, reference, atol=atol, rtol=rtol))}


def _probe(args, result):
    """One opaque operator only; A/B/A/B changes values without changing shape."""
    import statistics
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    from g05_npu_graphs import _compile_callable
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    op = register_row_inverse_op()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    a = torch.tril(torch.randn(args.batch, 32, 32, generator=generator, dtype=torch.float32) * 0.125, -1).to(args.device)
    b = -torch.tril(torch.ones(args.batch, 32, 32, dtype=torch.float32), -1).to(args.device)
    values = {"A": a, "B": b}
    with torch.inference_mode(), torch.autocast("npu", enabled=False):
        references = {name: _original(torch, value).cpu().clone() for name, value in values.items()}
        snapshots = {name: value.cpu().clone() for name, value in values.items()}
        # Complete JIT compilation and test eager ownership before graph capture.
        first, second = op(a), op(a)
        torch.npu.synchronize()
        if first.data_ptr() == second.data_ptr() or first.data_ptr() == a.data_ptr():
            raise RuntimeError("Custom op eager outputs alias another live output or input")
        first_snapshot = first.cpu().clone()
        second.zero_()
        if not torch.equal(first.cpu(), first_snapshot):
            raise RuntimeError("Mutating second eager output changed first output")
        result["eager_output_ownership"] = "fresh outputs; no input/other-live-output alias"
        selected, metadata = _compile_callable(torch, lambda value: op(value), args.backend,
                                               "opaque row_inverse32 custom op", {})
        result["compilation"] = metadata
        checks, pointers = [], []
        for label in ("A", "B", "A", "B"):
            torch.npu.synchronize()
            start = time.perf_counter()
            output = selected(values[label])
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if output.data_ptr() == values[label].data_ptr():
                raise RuntimeError("Captured custom-op output aliases input")
            pointers.append(output.data_ptr())
            snapshot = output.cpu().clone()
            checks.append({"case": label, "ms_including_first_compile": elapsed,
                           **_stats(torch, references[label], snapshot, args.atol, args.rtol)})
        result["replay_checks"] = checks
        result["graph_return_buffer_reuse_observed"] = len(set(pointers)) != len(pointers)
        result["graph_output_lifetime"] = "each replay result copied before the next request; backend may reuse graph buffers"
        result["reference_contrast"] = _stats(torch, references["A"], references["B"], args.atol, args.rtol)
        result["inputs_unchanged"] = all(torch.equal(values[name].cpu(), snapshots[name]) for name in values)
        for _ in range(args.warmup):
            selected(a)
        torch.npu.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            selected(a)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        result["timing"] = {"p50_ms": statistics.median(samples), "samples_ms": samples}
    result["status"] = "passed" if result["inputs_unchanged"] and all(x["passed"] for x in checks) else "numerical_mismatch"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe",), required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch", type=int, default=80)
    parser.add_argument("--backend", choices=("eager", "torchair_aclgraph"), default="torchair_aclgraph")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument("--rtol", type=float, default=5e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("npu") or args.batch < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Requires NPU, positive batch/repeats and nonnegative warmup")
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0:
        parser.error("Requires finite nonnegative tolerances")
    result = {"schema_version": 1, "experiment": "gdn-triton-opaque-custom-op-replay",
              "shape": [args.batch, 32, 32], "atol": args.atol, "rtol": args.rtol,
              "scope": "inverse+I only, not full GDN/model; synchronized wall clock after warmup",
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        _probe(args, result)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
