#!/usr/bin/env python3
"""Run a fixed-shape real-input TF32 or FP16 MMA PTX probe on SM86."""

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

from gac.mlir_tiled_matmul import (
    FUNCTIONAL_MMA_BLOCK,
    FUNCTIONAL_MMA_FP16_K,
    FUNCTIONAL_MMA_FP16_K_STEP,
    FUNCTIONAL_MMA_FP16_SYMBOL,
    FUNCTIONAL_MMA_GRID,
    FUNCTIONAL_MMA_K,
    FUNCTIONAL_MMA_K_STEP,
    FUNCTIONAL_MMA_M,
    FUNCTIONAL_MMA_N,
    FUNCTIONAL_MMA_TF32_SYMBOL,
    validate_functional_mma_ptx,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load MLIR-generated SM86 PTX, run a real-input TF32 or FP16 MMA "
            "kernel, and compare all 128 outputs with a float32 NumPy oracle."
        )
    )
    parser.add_argument("--ptx", required=True, type=Path)
    parser.add_argument("--mode", choices=("tf32", "fp16"), default="tf32")
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
    evidence = validate_functional_mma_ptx(ptx, args.mode)

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required to execute the functional MMA probe"
        ) from exc

    rng = np.random.default_rng(args.seed)
    if args.mode == "tf32":
        symbol = FUNCTIONAL_MMA_TF32_SYMBOL
        k = FUNCTIONAL_MMA_K
        k_step = FUNCTIONAL_MMA_K_STEP
        input_dtype = np.float32
        abi = "A_f32_row_major, B_f32_row_major, C_f32_row_major"
    else:
        symbol = FUNCTIONAL_MMA_FP16_SYMBOL
        k = FUNCTIONAL_MMA_FP16_K
        k_step = FUNCTIONAL_MMA_FP16_K_STEP
        input_dtype = np.float16
        abi = "A_f16_row_major, B_f16_row_major, C_f32_row_major"
    # Small integer-valued operands are exactly representable by TF32 and f16.
    # A float32 oracle therefore isolates lane mapping and K accumulation from
    # expected reduced-precision input rounding.
    host_a = rng.integers(-3, 4, size=(FUNCTIONAL_MMA_M, k)).astype(input_dtype)
    host_b = rng.integers(-3, 4, size=(k, FUNCTIONAL_MMA_N)).astype(input_dtype)
    reference = host_a.astype(np.float32) @ host_b.astype(np.float32)

    device = cp.cuda.Device(args.device)
    with device:
        capability = _compute_capability(device)
        if capability != (8, 6):
            raise RuntimeError(
                "the PTX probe targets SM86 exactly; selected device has compute "
                f"capability {capability[0]}.{capability[1]}"
            )
        module = cp.RawModule(path=str(ptx_path))
        kernel = module.get_function(symbol)
        a = cp.asarray(host_a, order="C")
        b = cp.asarray(host_b, order="C")
        c = cp.empty((FUNCTIONAL_MMA_M, FUNCTIONAL_MMA_N), dtype=cp.float32)
        launch_args = (a, b, c)

        kernel(FUNCTIONAL_MMA_GRID, FUNCTIONAL_MMA_BLOCK, launch_args)
        device.synchronize()
        actual = cp.asnumpy(c)
        max_abs_error = float(np.max(np.abs(actual - reference)))
        correct = bool(np.allclose(actual, reference, rtol=0.0, atol=1.0e-4))
        if not correct:
            raise RuntimeError(
                "functional MMA result mismatch; maximum absolute error is "
                f"{max_abs_error:g}"
            )

        for _ in range(args.warmup):
            kernel(FUNCTIONAL_MMA_GRID, FUNCTIONAL_MMA_BLOCK, launch_args)
        device.synchronize()

        latency_us: list[float] = []
        for _ in range(args.samples):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()
            for _ in range(args.launches_per_sample):
                kernel(FUNCTIONAL_MMA_GRID, FUNCTIONAL_MMA_BLOCK, launch_args)
            end.record()
            end.synchronize()
            elapsed_ms = float(cp.cuda.get_elapsed_time(start, end))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                raise RuntimeError("CUDA Event returned a non-positive latency")
            latency_us.append(elapsed_ms * 1000.0 / args.launches_per_sample)

        report: dict[str, object] = {
            "kind": f"gac-functional-{args.mode}-mma-sm86-run-v1",
            "ptx": {
                "path": str(ptx_path),
                "sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
                "evidence": asdict(evidence),
            },
            "kernel": {
                "symbol": symbol,
                "mode": args.mode,
                "shape": {
                    "m": FUNCTIONAL_MMA_M,
                    "n": FUNCTIONAL_MMA_N,
                    "k": k,
                },
                "k_step": k_step,
                "k_slices": k // k_step,
                "grid": list(FUNCTIONAL_MMA_GRID),
                "block": list(FUNCTIONAL_MMA_BLOCK),
                "abi": abi,
                "accumulator_dtype": "float32",
            },
            "device": {
                "id": args.device,
                "name": _device_name(cp, args.device),
                "compute_capability": list(capability),
            },
            "correctness": {
                "seed": args.seed,
                "allclose": correct,
                "rtol": 0.0,
                "atol": 1.0e-4,
                "max_abs_error": max_abs_error,
                "checked_output_elements": FUNCTIONAL_MMA_M * FUNCTIONAL_MMA_N,
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
