#!/usr/bin/env python3
"""Compile the real-input NVGPU async double-buffer MatMul to SM86 PTX."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gac.targets.cuda.async_tiled_matmul import lower_async_tiled_matmul_to_ptx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lower a fixed 16x16x64 real-input MatMul with two-stage NVGPU "
            "device_async_copy tiling and record fail-closed PTX evidence."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError(f"output directory must be empty: {output}")
    else:
        output.mkdir(parents=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _prepare_output(args.output_dir)
    program = lower_async_tiled_matmul_to_ptx(
        mlir_opt=args.mlir_opt,
        use_wsl=args.wsl,
        wsl_distribution=args.wsl_distribution,
        timeout=args.timeout,
    )

    source_name = "async-tiled-matmul.gpu.mlir"
    binary_name = "async-tiled-matmul.binary.mlir"
    ptx_name = "async-tiled-matmul.ptx"
    (output / source_name).write_text(program.gpu_source, encoding="utf-8", newline="\n")
    (output / binary_name).write_text(
        program.artifact.binary_mlir, encoding="utf-8", newline="\n"
    )
    (output / ptx_name).write_text(program.ptx, encoding="utf-8", newline="\n")

    report: dict[str, object] = {
        "kind": "gac-functional-async-tiled-matmul-sm86-probe-v1",
        "scope": "real-input-fixed-shape-executable-candidate",
        "production_ffn_integration": False,
        "kernel": {
            "symbol": program.symbol,
            "shape": {"m": program.m, "n": program.n, "k": program.k},
            "tile_k": program.tile_k,
            "stages": program.stages,
            "grid": list(program.grid),
            "block": list(program.block),
            "abi": "three-bare-global-pointers-row-major-f32",
        },
        "evidence": asdict(program.evidence),
        "has_ping_pong_schedule": program.evidence.has_ping_pong_schedule,
        "computes_real_matmul": program.evidence.computes_real_matmul,
        "files": {
            "gpu_mlir": source_name,
            "binary_mlir": binary_name,
            "ptx": ptx_name,
        },
        "sha256": {
            "gpu_mlir": _sha256(program.gpu_source),
            "binary_mlir": _sha256(program.artifact.binary_mlir),
            "ptx": _sha256(program.ptx),
        },
        "toolchain": {
            "version": program.artifact.mlir_opt_version,
            "llvm_source_revision": program.artifact.mlir_opt_source_revision,
            "chip": program.artifact.target_chip,
            "features": program.artifact.target_features,
            "ptx_version": program.artifact.ptx_version,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output_dir": str(output), "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
