#!/usr/bin/env python3
"""Execute the MLIR-generated async tiled MatMul PTX on an SM86 GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

from gac.targets.cuda.async_tiled_matmul import (
    ASYNC_MATMUL_BLOCK,
    ASYNC_MATMUL_GRID,
    ASYNC_MATMUL_K,
    ASYNC_MATMUL_M,
    ASYNC_MATMUL_N,
    ASYNC_MATMUL_SYMBOL,
    validate_async_tiled_matmul_ptx,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load the MLIR-generated 16x16x64 async-copy MatMul PTX, compare "
            "all outputs with NumPy, and measure CUDA-event launch latency."
        )
    )
    parser.add_argument("--ptx", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=_positive_int, default=30)
    parser.add_argument("--launches-per-sample", type=_positive_int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser


def _compute_capability(device: object) -> tuple[int, int]:
    value = getattr(device, "compute_capability")
    text = value.decode("ascii") if isinstance(value, bytes) else str(value)
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 2:
        raise RuntimeError(f"unrecognized CUDA compute capability {value!r}")
    return int(digits[:-1]), int(digits[-1])


def _device_name(cp: object, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    value = properties.get("name", properties.get(b"name", "unknown"))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_json(path: Path, report: dict[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.replace(target)


def _paths_alias(first: Path, second: Path) -> bool:
    first_path = first.expanduser()
    second_path = second.expanduser()
    if first_path.resolve(strict=False) == second_path.resolve(strict=False):
        return True
    try:
        return first_path.exists() and second_path.exists() and os.path.samefile(
            first_path, second_path
        )
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.device < 0:
        raise ValueError("device must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.output_json is not None and _paths_alias(args.ptx, args.output_json):
        raise ValueError("--output-json must not overwrite or alias --ptx")

    ptx_path = args.ptx.expanduser().resolve()
    ptx = ptx_path.read_text(encoding="utf-8")
    evidence = validate_async_tiled_matmul_ptx(ptx)
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is required to execute the async MatMul probe") from exc

    rng = np.random.default_rng(args.seed)
    # Small integer values make the expected FP32 sum exact, so a schedule or
    # stage-index bug cannot hide behind normal floating-point tolerance.
    host_a = rng.integers(-2, 3, size=(ASYNC_MATMUL_M, ASYNC_MATMUL_K)).astype(np.float32)
    host_b = rng.integers(-2, 3, size=(ASYNC_MATMUL_K, ASYNC_MATMUL_N)).astype(np.float32)
    reference = host_a @ host_b

    device = cp.cuda.Device(args.device)
    with device:
        capability = _compute_capability(device)
        if capability != (8, 6):
            raise RuntimeError(
                "the PTX candidate targets SM86 exactly; selected device has "
                f"compute capability {capability[0]}.{capability[1]}"
            )
        module = cp.RawModule(path=str(ptx_path))
        kernel = module.get_function(ASYNC_MATMUL_SYMBOL)
        a = cp.asarray(host_a, order="C")
        b = cp.asarray(host_b, order="C")
        c = cp.empty((ASYNC_MATMUL_M, ASYNC_MATMUL_N), dtype=cp.float32)
        launch_args = (a, b, c)

        kernel(ASYNC_MATMUL_GRID, ASYNC_MATMUL_BLOCK, launch_args)
        device.synchronize()
        actual = cp.asnumpy(c)
        max_abs_error = float(np.max(np.abs(actual - reference)))
        correct = bool(np.array_equal(actual, reference))
        if not correct:
            raise RuntimeError(
                "async tiled MatMul result mismatch; maximum absolute error is "
                f"{max_abs_error:g}"
            )

        for _ in range(args.warmup):
            kernel(ASYNC_MATMUL_GRID, ASYNC_MATMUL_BLOCK, launch_args)
        device.synchronize()
        latency_us: list[float] = []
        for _ in range(args.samples):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(args.launches_per_sample):
                kernel(ASYNC_MATMUL_GRID, ASYNC_MATMUL_BLOCK, launch_args)
            end.record()
            end.synchronize()
            elapsed_ms = float(cp.cuda.get_elapsed_time(start, end))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                raise RuntimeError("CUDA Event returned a non-positive latency")
            latency_us.append(elapsed_ms * 1000.0 / args.launches_per_sample)

        report: dict[str, object] = {
            "kind": "gac-functional-async-tiled-matmul-sm86-run-v1",
            "ptx": {
                "path": str(ptx_path),
                "sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
                "evidence": asdict(evidence),
            },
            "kernel": {
                "symbol": ASYNC_MATMUL_SYMBOL,
                "shape": {"m": ASYNC_MATMUL_M, "n": ASYNC_MATMUL_N, "k": ASYNC_MATMUL_K},
                "grid": list(ASYNC_MATMUL_GRID),
                "block": list(ASYNC_MATMUL_BLOCK),
                "abi": "A_f32_row_major, B_f32_row_major, C_f32_row_major",
            },
            "device": {
                "id": args.device,
                "name": _device_name(cp, args.device),
                "compute_capability": list(capability),
            },
            "correctness": {
                "seed": args.seed,
                "exact": correct,
                "max_abs_error": max_abs_error,
                "checked_output_elements": ASYNC_MATMUL_M * ASYNC_MATMUL_N,
            },
            "timing": {
                "warmup_launches": args.warmup,
                "samples": args.samples,
                "launches_per_sample": args.launches_per_sample,
                "latency_us": latency_us,
                "median_us": statistics.median(latency_us),
                "minimum_us": min(latency_us),
                "maximum_us": max(latency_us),
            },
            "production_ffn_integration": False,
        }

    if args.output_json is not None:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
