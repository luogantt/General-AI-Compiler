#!/usr/bin/env python3
"""Small NPU RMSNorm/GE probe, without checkpoint loading or upstream imports.

Tests the production factories at actual AE widths/shapes under BF16 autocast.
Reference math preserves the original FP32 islands. Micro timings include Python
dispatch and are not a prediction of complete model latency. Root owns execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"


def run(args, report):
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.manual_seed(17)
    torch.npu.set_device(args.device)
    torch._dynamo.config.suppress_errors = False
    spec = importlib.util.spec_from_file_location("g05_ae_fused_norm", args.helper)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    if args.backend == "torchair":
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        def backend():
            config = CompilerConfig()
            config.experimental_config.frozen_parameter = True
            return torchair.get_npu_backend(compiler_config=config)
    else:
        def backend():
            return "eager"

    class NormProbe(torch.nn.Module):
        def __init__(self, kind, width, heads, dtype):
            super().__init__()
            self.kind, self.eps, self.dim = kind, 1e-6, width
            if kind == "adaptive":
                gamma = torch.ones(width, dtype=torch.float32, device=args.device)
                self.optimized = helper.make_adaptive_norm(torch, torch_npu, self, gamma)
            else:
                self.weight = torch.nn.Parameter(
                    torch.linspace(-.99, .8, width).to(device=args.device, dtype=dtype), requires_grad=False)
                self.optimized = helper.make_qk_norm(torch, torch_npu, self, heads=heads)

        def forward(self, x):
            return self.optimized(x)

        def reference(self, x):
            with torch.autocast("npu", enabled=False):
                if self.kind == "adaptive":
                    variance = torch.square(x.float()).mean(dim=-1, keepdim=True)
                    return x * torch.rsqrt(variance + self.eps)
                value = x.float()
                value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                return (value * (1.0 + self.weight.float())).to(x.dtype)

    def snapshot(function, x):
        with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
            value = function(x)
        torch.npu.synchronize()
        return value.detach().cpu().float().clone(), str(value.dtype)

    def compare(expected, observed):
        delta = observed - expected
        return {"exact": bool(torch.equal(expected, observed)),
                "finite": bool(torch.isfinite(observed).all()),
                "max_abs": float(delta.abs().max()),
                "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-30))}

    def time_calls(function, x):
        with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
            for _ in range(args.warmup):
                function(x)
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                function(x)
            torch.npu.synchronize()
        return (time.perf_counter() - started) * 1000 / args.iterations

    report.update(torch=torch.__version__, torch_npu=torch_npu.__version__, device=args.device,
                  backend=args.backend, fullgraph=True, dynamic=False, fallback_allowed=False,
                  scope="small isolated normalization shapes; no full-model accuracy/speed claim",
                  timing_scope="synchronized batch wall time divided by calls; includes Python dispatch",
                  cases=[])
    passed = True
    shapes = (("adaptive", (1, 32, 1024), None),
              ("q", (1, 32, 8, 256), 8), ("k", (1, 32, 2, 256), 2))
    for kind, shape, heads in shapes:
        for dtype in (torch.bfloat16, torch.float32):
            report["stage"] = f"{kind}/{dtype}"
            module = NormProbe(kind, shape[-1], heads, dtype).eval()
            # Q norm sees a first-half view of interleaved Q+gate projections.
            storage_shape = (*shape[:-1], shape[-1] * 2) if kind == "q" else shape
            tensors = []
            for factor in (1.0, -1.7):
                storage = (torch.randn(storage_shape) * factor).to(device=args.device, dtype=dtype)
                tensors.append(storage[..., :shape[-1]] if kind == "q" else storage)
            checks = []
            expected_dtype = "torch.float32" if kind == "adaptive" else str(dtype)
            for label, x in (("A", tensors[0]), ("B", tensors[1]),
                             ("zeros", torch.zeros_like(tensors[0])),
                             ("small", torch.full_like(tensors[0], 1e-8)),
                             ("large", torch.full_like(tensors[0], 1000))):
                expected, _ = snapshot(module.reference, x)
                actual, actual_dtype = snapshot(module, x)
                metric = compare(expected, actual)
                metric.update(path="npu_eager", input=label, output_dtype=actual_dtype)
                checks.append(metric)
                passed &= actual_dtype == expected_dtype
            compiled = torch.compile(module.forward, backend=backend(), fullgraph=True, dynamic=False)
            torch.npu.synchronize()
            cold_started = time.perf_counter()
            snapshot(compiled, tensors[0])
            cold_ms = (time.perf_counter() - cold_started) * 1000
            for label, x in (("A", tensors[0]), ("B", tensors[1]), ("A", tensors[0]), ("B", tensors[1])):
                expected, _ = snapshot(module.reference, x)
                actual, actual_dtype = snapshot(compiled, x)
                metric = compare(expected, actual)
                metric.update(path=args.backend, input=label, output_dtype=actual_dtype)
                checks.append(metric)
                passed &= actual_dtype == expected_dtype
            fp32_output = kind == "adaptive" or dtype == torch.float32
            abs_limit, relative_limit = (1e-5, 1e-6) if fp32_output else (.03125, .001)
            passed &= all(item["finite"] and item["max_abs"] <= abs_limit and item["relative_l2"] <= relative_limit for item in checks)
            row = {"kind": kind, "shape": shape, "input_dtype": str(dtype),
                   "input_stride": tensors[0].stride(), "output_dtype_expected": expected_dtype,
                   "thresholds": {"max_abs": abs_limit, "relative_l2": relative_limit},
                   "checks": checks, "first_compile_and_call_ms": cold_ms,
                   "mean_wall_ms_per_call": {"original": time_calls(module.reference, tensors[0]),
                                             "npu_eager": time_calls(module, tensors[0]),
                                             args.backend + "_compiled": time_calls(compiled, tensors[0])}}
            report["cases"].append(row)
            print(json.dumps({"progress": report["stage"], "timings": row["mean_wall_ms_per_call"]}), flush=True)
    report["status"] = "success" if passed else "numerical_validation_failed"
    report["stage"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("eager", "torchair"), default="torchair")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    report = {"status": "failure", "stage": "imports"}
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
