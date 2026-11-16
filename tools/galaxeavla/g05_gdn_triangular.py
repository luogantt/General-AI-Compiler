#!/usr/bin/env python3
"""Experimental GDN prefill triangular solve; never installed by default.

The chunk implementation below is adapted from GalaxeaVLA's Qwen3.5 fallback,
ported from Hugging Face transformers by the Qwen/Hugging Face teams.
Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
Licensed under Apache-2.0; see upstream THIRD_PARTY_NOTICES.md.
Modified by General-AI-Compiler: replace the row recurrence with an FP32 solve.

For the strictly lower triangular L in the original recurrence, its final
matrix is (I-L)^-1. Solving this unit-lower system preserves that equation but
changes floating-point accumulation order. It requires independent output,
recurrent-state, and complete-model validation on the actual NPU.

CPU validation (does not load the NPU backend):
  python g05_gdn_triangular.py --source-dir /path/to/GalaxeaVLA --mode validate

NPU microbenchmark, to be run by the owner of the device:
  python g05_gdn_triangular.py --source-dir /path/to/GalaxeaVLA \
    --mode benchmark --device npu:0 --autocast-bf16 --backend npugraph_ex \
    --seq-len 159 --heads 16 --head-dim 128 --value-dim 128 --output result.json

Install the triangular replacement before optional GDN graph helpers and restore
in reverse order. Do not install over an already compiled helper. This module
does not edit upstream files, install packages, or change the default backend.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Callable


GDN_MODULE = "g05.models.g05.qwen35.gated_deltanet"
_MARKER = "_g05_gdn_triangular_installation"


def torch_chunk_gated_delta_rule_solve(
    query, key, value, g, beta, chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False,
    *, inverse_impl=None,
):
    """Original chunk math with FP32 solve, or an explicit inverse callback.

    ``inverse_impl`` receives FP32 [..., chunk_size, chunk_size] strict-lower L
    and must return the same-shaped (I-L)^-1. None keeps the solve path intact.
    """
    import torch
    import torch.nn.functional as F

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    # Original: 31 dependent row updates for chunk_size=32, then add I.
    # L is strictly lower triangular, so the diagonal of I-L is exactly one.
    # Disable the caller's BF16 autocast ONLY here; all other original casts
    # and matmuls retain the caller's precision policy.
    with torch.autocast(device_type=attn.device.type, enabled=False):
        if inverse_impl is None:
            identity = torch.eye(chunk_size, dtype=torch.float32, device=attn.device)
            attn = torch.linalg.solve_triangular(
                identity - attn.float(), identity.expand_as(attn),
                upper=False, left=True, unitriangular=True,
            )
        else:
            attn = inverse_impl(attn.float())

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            device=value.device, dtype=value.dtype,
        )
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1,
    )
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1],
    )[:, :, :sequence_length]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


@dataclass
class TriangularInstallation:
    metadata: dict[str, Any]
    module: Any
    original: Callable
    installed: Callable
    restored: bool = False

    def restore(self) -> None:
        if not self.restored:
            if self.module.torch_chunk_gated_delta_rule is self.installed:
                self.module.torch_chunk_gated_delta_rule = self.original
            self.restored = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def install_triangular_chunk() -> TriangularInstallation:
    """Opt in for NPU inputs only; returns a reversible process-local handle."""
    module = importlib.import_module(GDN_MODULE)
    original = module.torch_chunk_gated_delta_rule
    if getattr(original, _MARKER, False) or getattr(original, "_g05_npu_graph_installation", False):
        raise RuntimeError("Install the triangular experiment before GDN graph compilation, only once")

    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        query = args[0] if args else kwargs.get("query")
        if getattr(getattr(query, "device", None), "type", None) == "npu":
            return torch_chunk_gated_delta_rule_solve(*args, **kwargs)
        return original(*args, **kwargs)

    setattr(dispatch, _MARKER, True)
    module.torch_chunk_gated_delta_rule = dispatch
    return TriangularInstallation(
        metadata={
            "name": "gdn-chunk-fp32-triangular-solve", "experimental": True,
            "scope": f"{GDN_MODULE}.torch_chunk_gated_delta_rule; NPU inputs only",
            "accumulation_order_changed": True, "solve_dtype": "torch.float32",
            "solve_autocast_enabled": False, "other_precision_policy_changed": False,
            "chunk_size_changed": False, "fallback_allowed": False,
            "requires_independent_npu_and_full_model_validation": True,
        },
        module=module, original=original, installed=dispatch,
    )


def _load_reference(source_dir: Path, torch: Any):
    """Read only the two original tensor functions, avoiding model dependencies."""
    path = source_dir / "src/g05/models/g05/qwen35/gated_deltanet.py"
    payload = path.read_bytes()
    tree = ast.parse(payload.decode("utf-8"), filename=str(path))
    names = {"l2norm", "torch_chunk_gated_delta_rule"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError("Reference source does not contain the expected GDN helpers")
    reference_module = ModuleType("_g05_gdn_original_reference")
    reference_module.__file__ = str(path)
    namespace = reference_module.__dict__
    namespace.update(torch=torch, F=torch.nn.functional)
    sys.modules[reference_module.__name__] = reference_module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["torch_chunk_gated_delta_rule"], {
        "path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _inputs(torch, shape, dtype, profile, has_state, seed):
    batch, seq, heads, key_dim, value_dim = shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def random(*size):
        return torch.randn(*size, generator=generator, dtype=torch.float32)
    q, k = random(batch, seq, heads, key_dim), random(batch, seq, heads, key_dim)
    value = random(batch, seq, heads, value_dim)
    if profile == "aligned-slow-decay":
        # Nearly collinear keys and beta close to one cause cancellation in the
        # triangular inverse. L2 normalization and negative g keep inputs bounded.
        direction = random(batch, 1, heads, key_dim)
        k = direction + 1e-3 * k
        q = direction + 1e-3 * q
        beta = torch.full((batch, seq, heads), 0.999, dtype=torch.float32)
        g = torch.full((batch, seq, heads), -1e-5, dtype=torch.float32)
    else:
        beta = random(batch, seq, heads).sigmoid()
        g = -random(batch, seq, heads).abs() * 0.3
    state = random(batch, heads, key_dim, value_dim) if has_state else None
    return (q.to(dtype), k.to(dtype), value.to(dtype)), {
        "g": g, "beta": beta.to(dtype), "initial_state": state,
        "output_final_state": True, "use_qk_l2norm_in_kernel": True,
    }


def _errors(torch, expected, actual):
    ref, got = expected.detach().float().cpu(), actual.detach().float().cpu()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(got).all())
    if not finite:
        raise RuntimeError("Nonfinite triangular experiment output/state")
    difference = got.double() - ref.double()
    return {
        "dtype": str(actual.dtype), "shape": list(actual.shape), "finite": finite,
        "max_abs_error": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2_error": (difference.norm() / ref.double().norm().clamp_min(1e-30)).item(),
        "exact": bool(torch.equal(expected, actual)),
    }


def validate_cpu(torch, reference):
    shapes = [(1, 1, 2, 8, 8), (1, 7, 2, 8, 12), (2, 31, 2, 8, 8),
              (1, 32, 2, 16, 16), (1, 33, 2, 16, 16),
              (1, 159, 16, 128, 128), (1, 234, 16, 128, 128)]
    precision = [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]
    rows = []
    for shape in shapes:
        for dtype, autocast in precision:
            for profile in ("random", "aligned-slow-decay"):
                for has_state in (False, True):
                    inputs, kwargs = _inputs(torch, shape, dtype, profile, has_state, seed=42)
                    kwargs["chunk_size"] = 32
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                        expected = reference(*inputs, **kwargs)
                        actual = torch_chunk_gated_delta_rule_solve(*inputs, **kwargs)
                    if actual[1].dtype != torch.float32:
                        raise RuntimeError("Experiment changed final recurrent-state dtype")
                    rows.append({
                        "shape": list(shape), "input_dtype": str(dtype), "autocast_bf16": autocast,
                        "profile": profile, "has_initial_state": has_state,
                        "output": _errors(torch, expected[0], actual[0]),
                        "state": _errors(torch, expected[1], actual[1]),
                    })
    return {"cases": len(rows), "all_finite": True, "rows": rows,
            "max_output_abs_error": max(row["output"]["max_abs_error"] for row in rows),
            "max_state_abs_error": max(row["state"]["max_abs_error"] for row in rows),
            "max_output_relative_l2": max(row["output"]["relative_l2_error"] for row in rows),
            "max_state_relative_l2": max(row["state"]["relative_l2_error"] for row in rows),
            "acceptance": "Diagnostics only; no NPU or full-model accuracy acceptance inferred"}


def benchmark(torch, reference, args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    shape = (args.batch, args.seq_len, args.heads, args.head_dim, args.value_dim)
    inputs, kwargs = _inputs(torch, shape, dtype, args.profile, args.initial_state, args.seed)
    inputs = tuple(value.to(args.device) for value in inputs)
    kwargs = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in kwargs.items()}
    kwargs["chunk_size"] = args.chunk_size
    synchronize = torch.npu.synchronize if args.device.startswith("npu") else lambda: None
    variants = {"original_row_recurrence": reference, "fp32_triangular_solve": torch_chunk_gated_delta_rule_solve}
    if args.backend != "eager":
        torch._dynamo.config.suppress_errors = False
        variants = {name: torch.compile(fn, backend=args.backend, fullgraph=True, dynamic=False,
                                       options={"clone_input": True, "clone_output": True})
                    for name, fn in variants.items()}
    timing, outputs = {}, {}
    with torch.no_grad(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16,
                                        enabled=args.autocast_bf16):
        for name, fn in variants.items():
            synchronize()
            started = time.perf_counter()
            result = fn(*inputs, **kwargs)
            synchronize()
            cold_ms = (time.perf_counter() - started) * 1000
            for _ in range(args.warmup):
                result = fn(*inputs, **kwargs)
            synchronize()
            samples = []
            for _ in range(args.repeats):
                synchronize()
                started = time.perf_counter()
                result = fn(*inputs, **kwargs)
                synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            outputs[name] = tuple(value.detach().clone() for value in result)
            timing[name] = {"cold_ms": cold_ms, "samples_ms": samples,
                            "p50_ms": statistics.median(samples),
                            "p95_ms": sorted(samples)[max(0, math.ceil(len(samples) * .95) - 1)]}
    expected, actual = outputs.values()
    return {"scope": "GDN chunk helper only; excludes projections, convolution, vision and model",
            "shape": list(shape), "chunk_size": args.chunk_size, "backend": args.backend,
            "input_dtype": args.dtype, "autocast_bf16": args.autocast_bf16,
            "profile": args.profile, "has_initial_state": args.initial_state,
            "warmup": args.warmup, "repeats": args.repeats, "timing": timing,
            "output": _errors(torch, expected[0], actual[0]),
            "state": _errors(torch, expected[1], actual[1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("validate", "benchmark"), default="validate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("eager", "npugraph_ex"), default="eager")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--autocast-bf16", action="store_true")
    parser.add_argument("--profile", choices=("random", "aligned-slow-decay"), default="random")
    parser.add_argument("--initial-state", action="store_true")
    for name, default in (("batch", 1), ("seq-len", 159), ("heads", 16),
                          ("head-dim", 128), ("value-dim", 128), ("chunk-size", 32),
                          ("warmup", 3), ("repeats", 10), ("seed", 42), ("cpu-threads", 1)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "validate" and (args.device != "cpu" or args.backend != "eager"):
        parser.error("validate mode is a CPU-only numerical scan; use benchmark for NPU")
    if args.device != "cpu" and not args.device.startswith("npu"):
        parser.error("Supported experimental devices: cpu, npu:0")
    if args.backend != "eager" and not args.device.startswith("npu"):
        parser.error("npugraph_ex requires NPU")
    if args.warmup < 0 or args.repeats < 1 or min(args.batch, args.seq_len, args.heads,
                                                args.head_dim, args.value_dim, args.chunk_size, args.cpu_threads) < 1:
        parser.error("Shapes, repeats and CPU threads must be positive; warmup must be nonnegative")
    if args.device == "cpu":
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # LAPACK/OpenBLAS may ignore torch.set_num_threads; bound its worker pool
    # before importing torch so a CPU precision scan cannot consume the host.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(args.cpu_threads)
    import torch
    if args.device.startswith("npu"):
        importlib.import_module("torch_npu")
    torch.set_num_threads(args.cpu_threads)
    reference, source = _load_reference(args.source_dir.resolve(), torch)
    result = {"schema_version": 1, "experiment": "gdn-chunk-fp32-triangular-solve",
              "mode": args.mode, "device": args.device, "torch_version": torch.__version__,
              "cpu_threads": args.cpu_threads,
              "reference_source": source, "accumulation_order_changed": True,
              "results": validate_cpu(torch, reference) if args.mode == "validate" else benchmark(torch, reference, args)}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        summary = {key: value for key, value in result["results"].items() if key != "rows"}
        print(json.dumps({"output": str(args.output), "results": summary}, ensure_ascii=False, allow_nan=False))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
