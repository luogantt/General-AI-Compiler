from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    value = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, value, mask=mask)


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
        a_offsets = offsets_m[:, None] * K + (k_start + offsets_k[None, :])
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


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    difference = (actual_cpu - expected_cpu).abs()
    relative = difference / expected_cpu.abs().clamp_min(1e-6)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_rel": float(relative.max()),
    }


def benchmark(callable_, *, warmup: int = 20, measure: int = 100) -> dict[str, float]:
    for _ in range(warmup):
        callable_()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(measure):
        start = time.perf_counter_ns()
        callable_()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
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
    output_root = Path(os.environ.get("A2_VALIDATION_ROOT", "/root/work/a2-triton-validation"))
    cache_root = Path(os.environ.get("TRITON_CACHE_DIR", output_root / "cache"))
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261113)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    target = triton.runtime.driver.active.get_current_target()

    n = 65_537
    x_cpu = torch.randn(n, dtype=torch.float32)
    y_cpu = torch.randn(n, dtype=torch.float32)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    add_output = torch.empty_like(x)
    add_call = lambda: vector_add_kernel[(triton.cdiv(n, 256),)](
        x, y, add_output, n=n, BLOCK=256
    )
    add_call()
    torch.npu.synchronize()
    add_result = {
        "shape": list(add_output.shape),
        "errors": errors(add_output, x_cpu + y_cpu),
        "latency": benchmark(add_call),
    }

    sigmoid_input_cpu = torch.linspace(-8.0, 8.0, n, dtype=torch.float32)
    sigmoid_input = sigmoid_input_cpu.to(device)
    sigmoid_output = torch.empty_like(sigmoid_input)
    sigmoid_call = lambda: sigmoid_kernel[(triton.cdiv(n, 256),)](
        sigmoid_input, sigmoid_output, n=n, BLOCK=256
    )
    sigmoid_call()
    torch.npu.synchronize()
    sigmoid_result = {
        "shape": list(sigmoid_output.shape),
        "errors": errors(sigmoid_output, torch.sigmoid(sigmoid_input_cpu)),
        "latency": benchmark(sigmoid_call),
    }

    m, k, n_columns = 64, 64, 64
    a_cpu = torch.randn((m, k), dtype=torch.float32) * 0.1
    b_cpu = torch.randn((k, n_columns), dtype=torch.float32) * 0.1
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    c = torch.empty((m, n_columns), dtype=torch.float32, device=device)
    grid = (triton.cdiv(m, 16), triton.cdiv(n_columns, 16))
    matmul_call = lambda: matmul_kernel[grid](
        a,
        b,
        c,
        M=m,
        N=n_columns,
        K=k,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
    )
    matmul_call()
    torch.npu.synchronize()
    matmul_result = {
        "shape": list(c.shape),
        "errors": errors(c, a_cpu @ b_cpu),
        "latency": benchmark(matmul_call),
    }

    artifacts = []
    for path in sorted(cache_root.rglob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    report = {
        "schema_version": 1,
        "kind": "triton-ascend-physical-smoke",
        "status": "passed",
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "triton_python": getattr(triton, "__version__", "unknown"),
        "target": {
            "backend": str(getattr(target, "backend", "")),
            "arch": str(getattr(target, "arch", "")),
            "warp_size": int(getattr(target, "warp_size", 0)),
        },
        "device_name": torch.npu.get_device_name(0),
        "kernels": {
            "vector_add": add_result,
            "sigmoid": sigmoid_result,
            "matmul": matmul_result,
        },
        "cache_artifacts": artifacts,
    }
    report_path = output_root / "triton_smoke.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
