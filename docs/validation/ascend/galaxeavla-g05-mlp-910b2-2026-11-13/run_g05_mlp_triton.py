from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a_offsets = offsets_m[:, None] * K + k_start + offsets_k[None, :]
        b_offsets = (k_start + offsets_k[:, None]) * N + offsets_n[None, :]
        a = tl.load(
            a_ptr + a_offsets,
            mask=(offsets_m[:, None] < M) & (k_start + offsets_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + b_offsets,
            mask=(k_start + offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
    c_offsets = offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(
        c_ptr + c_offsets,
        accumulator,
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )


@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, output_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(output_ptr + offsets, gate * sigmoid * up, mask=mask)


def launch_matmul(a: torch.Tensor, b: torch.Tensor, output: torch.Tensor) -> None:
    m, k = a.shape
    k_right, n = b.shape
    if k_right != k:
        raise ValueError("matmul K mismatch")
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
    matmul_kernel[grid](
        a,
        b,
        output,
        M=m,
        N=n,
        K=k,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
    )


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.detach().cpu().float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_rel": float(relative.max()),
        "allclose_rtol_1e-4_atol_1e-4": bool(
            torch.allclose(actual.detach().cpu().float(), expected.float(), rtol=1e-4, atol=1e-4)
        ),
    }


def benchmark(callable_, *, warmup: int = 50, measure: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(measure):
        started = time.perf_counter_ns()
        callable_()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(samples)
    return {
        "warmup": warmup,
        "measure": measure,
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "mean_ms": statistics.fmean(samples),
        "max_ms": max(samples),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(os.environ.get("A2_VALIDATION_ROOT", "/root/work/a2-triton-validation"))
    cache = Path(os.environ.get("TRITON_CACHE_DIR", root / "cache"))
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261113)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.reset_peak_memory_stats(device)
    target = triton.runtime.driver.active.get_current_target()

    tokens, hidden, intermediate = 32, 1024, 4096
    x_cpu = torch.randn((tokens, hidden), dtype=torch.float32) * 0.2
    gate_weight_cpu = torch.randn((hidden, intermediate), dtype=torch.float32) / hidden**0.5
    up_weight_cpu = torch.randn((hidden, intermediate), dtype=torch.float32) / hidden**0.5
    down_weight_cpu = torch.randn((intermediate, hidden), dtype=torch.float32) / intermediate**0.5
    x = x_cpu.to(device)
    gate_weight = gate_weight_cpu.to(device)
    up_weight = up_weight_cpu.to(device)
    down_weight = down_weight_cpu.to(device)
    gate = torch.empty((tokens, intermediate), dtype=torch.float32, device=device)
    up = torch.empty_like(gate)
    gated_up = torch.empty_like(gate)
    output = torch.empty((tokens, hidden), dtype=torch.float32, device=device)

    def triton_pipeline() -> None:
        launch_matmul(x, gate_weight, gate)
        launch_matmul(x, up_weight, up)
        swiglu_kernel[(triton.cdiv(gate.numel(), 256),)](
            gate, up, gated_up, n=gate.numel(), BLOCK=256
        )
        launch_matmul(gated_up, down_weight, output)

    triton_pipeline()
    torch.npu.synchronize()

    accuracy_cases: list[dict[str, float | bool | int]] = []
    global_max_abs = 0.0
    global_max_rel = 0.0
    absolute_sum = 0.0
    element_count = 0
    passed_cases = 0
    expected = torch.empty((tokens, hidden), dtype=torch.float32)
    for case_index in range(100):
        generator = torch.Generator().manual_seed(20261113 + case_index)
        case_input = torch.randn(
            (tokens, hidden), generator=generator, dtype=torch.float32
        ) * 0.2
        x.copy_(case_input.to(device))
        triton_pipeline()
        torch.npu.synchronize()
        gate_expected = case_input @ gate_weight_cpu
        up_expected = case_input @ up_weight_cpu
        expected = (
            gate_expected * torch.sigmoid(gate_expected) * up_expected
        ) @ down_weight_cpu
        actual_cpu = output.detach().cpu().float()
        difference = (actual_cpu - expected).abs()
        relative = difference / expected.abs().clamp_min(1e-6)
        passed = bool(torch.allclose(actual_cpu, expected, rtol=1e-4, atol=1e-4))
        passed_cases += int(passed)
        case_max_abs = float(difference.max())
        case_max_rel = float(relative.max())
        global_max_abs = max(global_max_abs, case_max_abs)
        global_max_rel = max(global_max_rel, case_max_rel)
        absolute_sum += float(difference.sum())
        element_count += difference.numel()
        accuracy_cases.append(
            {
                "case": case_index,
                "seed": 20261113 + case_index,
                "passed": passed,
                "max_abs": case_max_abs,
                "mean_abs": float(difference.mean()),
                "max_rel": case_max_rel,
            }
        )
    accuracy = {
        "cases": len(accuracy_cases),
        "passed_cases": passed_cases,
        "rtol": 1e-4,
        "atol": 1e-4,
        "global_max_abs": global_max_abs,
        "global_mean_abs": absolute_sum / element_count,
        "global_max_rel": global_max_rel,
        "all_cases_passed": passed_cases == len(accuracy_cases),
        "per_case": accuracy_cases,
    }
    triton_latency = benchmark(triton_pipeline)

    eager_output: torch.Tensor | None = None

    def eager_pipeline() -> None:
        nonlocal eager_output
        eager_gate = x @ gate_weight
        eager_up = x @ up_weight
        eager_output = (eager_gate * torch.sigmoid(eager_gate) * eager_up) @ down_weight

    eager_pipeline()
    torch.npu.synchronize()
    eager_accuracy = errors(eager_output, expected) if eager_output is not None else {}
    eager_latency = benchmark(eager_pipeline)

    npubin_artifacts = []
    for path in sorted(cache.rglob("*.npubin")):
        npubin_artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    report = {
        "schema_version": 1,
        "kind": "galaxeavla-g05-action-expert-layer0-mlp-triton-ascend",
        "status": "passed" if accuracy["all_cases_passed"] else "failed",
        "synthetic_weights": True,
        "checkpoint_loaded": False,
        "full_model": False,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "triton_python": getattr(triton, "__version__", "unknown"),
        "triton_ascend_package": importlib.metadata.version("triton_ascend"),
        "cann_version": "9.0.0",
        "gac_source_commit": subprocess.check_output(
            [
                "git",
                "-C",
                "/root/work/General-AI-Compiler",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "device": torch.npu.get_device_name(0),
        "target": {
            "backend": str(getattr(target, "backend", "")),
            "arch": str(getattr(target, "arch", "")),
            "warp_size": int(getattr(target, "warp_size", 0)),
        },
        "contract": {
            "input_shape": [1, tokens, hidden],
            "intermediate_size": intermediate,
            "output_shape": [1, tokens, hidden],
            "dtype": "float32",
            "semantic_ops": ["MatMul", "Sigmoid", "Mul", "MatMul", "Mul", "MatMul"],
            "triton_dispatches": ["matmul", "matmul", "fused-swiglu-pointwise", "matmul"],
        },
        "implementation_note": (
            "correctness baseline with conservative 16x16x16 tl.dot tiles; "
            "not an optimized Triton-Ascend matmul performance claim"
        ),
        "triton_accuracy": accuracy,
        "torch_npu_eager_accuracy": eager_accuracy,
        "triton_latency": triton_latency,
        "torch_npu_eager_latency": eager_latency,
        "peak_allocated_bytes": int(torch.npu.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.npu.max_memory_reserved(device)),
        "npubin_artifacts": npubin_artifacts,
    }
    path = root / "g05_mlp_triton.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
