"""Benchmark the standalone MLIR SM86 LayerNorm PTX reference.

Lower ``examples/mlir_sm86_layernorm_fusion.mlir`` to PTX before invoking this
tool.  The benchmark loads that PTX with CuPy, validates all three MLIR kernel
entries, and uses CUDA Events for device-resident timing.  It is independent of
the GAC product runtime; it does not exercise ``MlirCudaSession``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np


ROWS = 2500
WIDTH = 256
EPSILON = np.float32(1.0e-5)
SEED = 20261010
RTOL = 2.0e-4
ATOL = 2.0e-4


ADD_KERNELS = r"""
extern "C" __global__ void add_bias(
    const float* input, const float* bias, float* output, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[i] + bias[i & 255];
}

extern "C" __global__ void add_residual(
    const float* input, const float* residual, float* output, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = input[i] + residual[i];
}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_argument(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 digest")
    return normalized


def git_revision_argument(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{40}", normalized) is None:
        raise argparse.ArgumentTypeError("expected a 40-character Git revision")
    return normalized


def extract_embedded_ptx(binary_mlir: Path, destination: Path) -> None:
    """Extract the first textual GPU object emitted with cubin-format=isa."""

    source = binary_mlir.read_text(encoding="utf-8")
    match = re.search(
        r'assembly = "((?:\\[0-9A-Fa-f]{2}|\\.|[^"\\])*)"', source, re.S
    )
    if match is None:
        raise ValueError(f"embedded textual GPU assembly not found in {binary_mlir}")

    def decode_escape(found: re.Match[str]) -> str:
        token = found.group(1)
        if len(token) == 2 and all(char in "0123456789abcdefABCDEF" for char in token):
            return chr(int(token, 16))
        return {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}.get(
            token, token
        )

    ptx = re.sub(r"\\([0-9A-Fa-f]{2}|.)", decode_escape, match.group(1))
    if ".target sm_86" not in ptx:
        raise ValueError(f"embedded assembly in {binary_mlir} is not SM86 PTX")
    destination.write_text(ptx, encoding="utf-8", newline="\n")


def distribution(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_us": float(statistics.median(values)),
        "p10_us": float(np.percentile(array, 10)),
        "p90_us": float(np.percentile(array, 90)),
        "min_us": float(array.min()),
        "samples_us": values,
    }


def benchmark(launch, warmup: int, iterations: int, repeats: int) -> dict:
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        launch()
    stream.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)) * 1000.0 / iterations)
    return distribution(samples)


def correctness(actual: cp.ndarray, expected: cp.ndarray) -> dict[str, float | bool]:
    difference = cp.abs(actual - expected)
    maximum = float(difference.max().get())
    scale = cp.maximum(cp.abs(expected), cp.float32(1.0e-3))
    relative = float((difference / scale).max().get())
    return {
        "allclose": bool(cp.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "max_abs_error": maximum,
        "max_rel_error_with_1e-3_floor": relative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and time the standalone SM86 MLIR LayerNorm PTX"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ptx",
        type=Path,
        help="PTX extracted from examples/mlir_sm86_layernorm_fusion.mlir",
    )
    source.add_argument(
        "--binary-mlir",
        type=Path,
        help="mlir-opt output containing an embedded textual SM86 PTX object",
    )
    parser.add_argument(
        "--mlir-source",
        type=Path,
        required=True,
        help="tracked MLIR source used to produce the PTX",
    )
    parser.add_argument("--mlir-opt-version", required=True)
    parser.add_argument("--mlir-opt-sha256", type=sha256_argument, required=True)
    parser.add_argument(
        "--llvm-source-revision", type=git_revision_argument, required=True
    )
    parser.add_argument(
        "--lowering-options",
        required=True,
        help="exact mlir-opt lowering argument used to produce binary MLIR",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    mlir_source_path = args.mlir_source.resolve()
    if not mlir_source_path.is_file():
        raise FileNotFoundError(f"MLIR source not found: {mlir_source_path}")

    rng = np.random.default_rng(SEED)
    rng_name = type(rng.bit_generator).__name__
    projected = cp.asarray(rng.normal(0.0, 0.2, (ROWS, WIDTH)).astype(np.float32))
    residual = cp.asarray(rng.normal(0.0, 0.1, (ROWS, WIDTH)).astype(np.float32))
    bias = cp.asarray(rng.normal(0.0, 0.01, (WIDTH,)).astype(np.float32))
    gamma = cp.asarray(rng.normal(1.0, 0.02, (WIDTH,)).astype(np.float32))
    beta = cp.asarray(rng.normal(0.0, 0.01, (WIDTH,)).astype(np.float32))
    combined = projected + bias + residual

    mean = combined.mean(axis=1, keepdims=True)
    variance = ((combined - mean) ** 2).mean(axis=1, keepdims=True)
    expected = (combined - mean) * cp.float32(1.0) / cp.sqrt(variance + EPSILON)
    expected = expected * gamma + beta

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.binary_mlir is not None:
        binary_mlir_path = args.binary_mlir.resolve()
        temporary_directory = tempfile.TemporaryDirectory(prefix="gac-mlir-layernorm-")
        ptx_path = Path(temporary_directory.name) / "layernorm-fusion-sm86.ptx"
        extract_embedded_ptx(binary_mlir_path, ptx_path)
        ptx_source = binary_mlir_path.name
    else:
        binary_mlir_path = None
        ptx_path = args.ptx.resolve()
        ptx_source = ptx_path.name

    benchmark_driver_path = Path(__file__).resolve()
    source_sha256 = {
        "examples/mlir_sm86_layernorm_fusion.mlir": sha256_file(mlir_source_path),
        "tools/benchmark_mlir_layernorm_fusion.py": sha256_file(
            benchmark_driver_path
        ),
    }
    artifact_sha256 = {
        "binary_mlir": (
            sha256_file(binary_mlir_path) if binary_mlir_path is not None else None
        ),
        "ptx": sha256_file(ptx_path),
    }

    module = cp.RawModule(path=str(ptx_path))
    serial_layer_norm = module.get_function("layer_norm_serial")
    block_layer_norm = module.get_function("layer_norm_block")
    fused_layer_norm = module.get_function("residual_bias_layer_norm_block")
    add_module = cp.RawModule(code=ADD_KERNELS, options=("--std=c++11",))
    add_bias = add_module.get_function("add_bias")
    add_residual = add_module.get_function("add_residual")

    serial_output = cp.empty_like(combined)
    block_output = cp.empty_like(combined)
    unfused_output = cp.empty_like(combined)
    fused_output = cp.empty_like(combined)
    temporary0 = cp.empty_like(combined)
    temporary1 = cp.empty_like(combined)
    count = np.int32(ROWS * WIDTH)
    flat_grid = ((ROWS * WIDTH + 255) // 256,)

    def launch_serial() -> None:
        serial_layer_norm((10,), (256,), (combined, gamma, beta, serial_output))

    def launch_block() -> None:
        block_layer_norm((ROWS,), (WIDTH,), (combined, gamma, beta, block_output))

    def launch_unfused() -> None:
        add_bias(flat_grid, (256,), (projected, bias, temporary0, count))
        add_residual(flat_grid, (256,), (temporary0, residual, temporary1, count))
        block_layer_norm((ROWS,), (WIDTH,), (temporary1, gamma, beta, unfused_output))

    def launch_fused() -> None:
        fused_layer_norm(
            (ROWS,),
            (WIDTH,),
            (projected, residual, bias, gamma, beta, fused_output),
        )

    # Trigger PTX/NVRTC JIT before validation and timing.
    launch_serial()
    launch_block()
    launch_unfused()
    launch_fused()
    cp.cuda.get_current_stream().synchronize()

    checks = {
        "serial_mlir": correctness(serial_output, expected),
        "block_reduce_mlir": correctness(block_output, expected),
        "unfused_bias_residual_block_layernorm": correctness(
            unfused_output, expected
        ),
        "fused_residual_bias_layernorm_mlir": correctness(fused_output, expected),
    }
    if not all(bool(item["allclose"]) for item in checks.values()):
        raise AssertionError(f"LayerNorm correctness failure: {checks}")

    timings = {
        "serial_mlir_layernorm": benchmark(
            launch_serial, args.warmup, args.iterations, args.repeats
        ),
        "block_reduce_mlir_layernorm": benchmark(
            launch_block, args.warmup, args.iterations, args.repeats
        ),
        "unfused_bias_add_residual_add_block_layernorm": benchmark(
            launch_unfused, args.warmup, args.iterations, args.repeats
        ),
        "fused_residual_bias_layernorm_mlir": benchmark(
            launch_fused, args.warmup, args.iterations, args.repeats
        ),
    }

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    serial_us = float(timings["serial_mlir_layernorm"]["median_us"])
    block_us = float(timings["block_reduce_mlir_layernorm"]["median_us"])
    unfused_us = float(
        timings["unfused_bias_add_residual_add_block_layernorm"]["median_us"]
    )
    fused_us = float(timings["fused_residual_bias_layernorm_mlir"]["median_us"])
    report = {
        "schema_version": 2,
        "kind": "mlir-layernorm-fusion-benchmark",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "standalone MLIR GPU dialect lowered through NVVM to PTX",
        "ptx_source": ptx_source,
        "source_sha256": source_sha256,
        "artifact_sha256": artifact_sha256,
        "toolchain": {
            "name": "mlir-opt",
            "version": args.mlir_opt_version,
            "executable_sha256": args.mlir_opt_sha256,
            "llvm_source_revision": args.llvm_source_revision,
            "lowering_options": args.lowering_options,
            "lowering_options_sha256": sha256_text(args.lowering_options),
        },
        "shape": [ROWS, WIDTH],
        "dtype": "float32",
        "epsilon": float(EPSILON),
        "seed": SEED,
        "rng": {"library": "numpy", "api": "default_rng", "bit_generator": rng_name},
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "device": name,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "cupy": cp.__version__,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
        },
        "warmup": args.warmup,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "timing_protocol": {
            "clock": "CUDA Events",
            "device_resident": True,
            "jit_primed": True,
        },
        "correctness": checks,
        "timings": timings,
        "speedups": {
            "block_reduce_over_serial_layernorm": serial_us / block_us,
            "fused_over_unfused_epilogue": unfused_us / fused_us,
            "fused_over_serial_layernorm_only": serial_us / fused_us,
        },
        "boundary": (
            "Standalone MLIR-to-NVVM/PTX microbenchmark; not MlirCudaSession. "
            "The device-resident fused timing includes bias add, residual add and "
            "gamma/beta LayerNorm; it excludes GEMM, host transfers and runtime setup."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    main()
