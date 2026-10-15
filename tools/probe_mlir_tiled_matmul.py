#!/usr/bin/env python3
"""Lower BEVFormer tiled MatMul candidates with a real LLVM/MLIR toolchain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gac.targets.cuda.tiled_matmul import (
    FunctionalMmaPtxProbe,
    bevformer_fc1_tiled_config,
    bevformer_fc2_tiled_config,
    lower_tiled_matmul_to_ptx,
    probe_nvgpu_async_double_buffer_to_ptx,
    probe_nvgpu_fp16_matmul_to_ptx,
    probe_nvgpu_mma_to_ptx,
    probe_nvgpu_tf32_matmul_to_ptx,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile explicit shared-memory BEVFormer FFN MatMuls to SM86 PTX "
            "and record instruction evidence."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mlir-opt", required=True)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--which", choices=("fc1", "fc2", "both"), default="both")
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tile-n", type=int, default=16)
    parser.add_argument("--tile-k", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--probe-mma",
        action="store_true",
        help="also prove that FP16 and TF32 NVGPU MMA smoke modules reach PTX mma.sync",
    )
    parser.add_argument(
        "--probe-functional-mma",
        action="store_true",
        help=(
            "also lower executable real-input TF32 and FP16 warp MatMuls and "
            "require global load/store plus typed PTX mma.sync evidence"
        ),
    )
    parser.add_argument(
        "--probe-async-double-buffer",
        "--probe-double-buffer",
        dest="probe_async_double_buffer",
        action="store_true",
        help=(
            "also lower the two-stage NVGPU async-copy overlap probe and require "
            "PTX cp.async commit/wait evidence"
        ),
    )
    return parser


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare_output(directory: Path) -> Path:
    directory = directory.expanduser().resolve()
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"output path is not a directory: {directory}")
        if any(directory.iterdir()):
            raise ValueError(f"output directory must be empty: {directory}")
    else:
        directory.mkdir(parents=True)
    return directory


def _write_functional_probe(
    output: Path, probe: FunctionalMmaPtxProbe
) -> dict[str, object]:
    """Write one validated functional MMA probe and return its report record."""

    mode = probe.mode
    source_name = f"nvgpu-{mode}-functional-matmul.gpu.mlir"
    binary_name = f"nvgpu-{mode}-functional-matmul.binary.mlir"
    ptx_name = f"nvgpu-{mode}-functional-matmul.ptx"
    (output / source_name).write_text(
        probe.gpu_source, encoding="utf-8", newline="\n"
    )
    (output / binary_name).write_text(
        probe.artifact.binary_mlir, encoding="utf-8", newline="\n"
    )
    (output / ptx_name).write_text(
        probe.artifact.ptx, encoding="utf-8", newline="\n"
    )
    return {
        "scope": "real-input-fixed-shape-executable-capability-probe",
        "production_ffn_integration": False,
        "symbol": probe.symbol,
        "mode": mode,
        "shape": {"m": probe.m, "n": probe.n, "k": probe.k},
        "k_step": probe.k_step,
        "k_slices": probe.k // probe.k_step,
        "abi": {
            "parameters": [
                {
                    "name": "a",
                    "shape": [probe.m, probe.k],
                    "dtype": probe.input_dtype,
                },
                {
                    "name": "b",
                    "shape": [probe.k, probe.n],
                    "dtype": probe.input_dtype,
                },
                {
                    "name": "c",
                    "shape": [probe.m, probe.n],
                    "dtype": probe.output_dtype,
                },
            ],
            "layout": "row-major-contiguous",
            "calling_convention": "three-bare-global-pointers",
            "accumulator_dtype": "float32",
        },
        "launch": {"grid": list(probe.grid), "block": list(probe.block)},
        "evidence": asdict(probe.evidence),
        "uses_real_global_inputs": probe.evidence.uses_real_global_inputs,
        "writes_global_output": probe.evidence.writes_global_output,
        "files": {
            "gpu_mlir": source_name,
            "binary_mlir": binary_name,
            "ptx": ptx_name,
        },
        "sha256": {
            "gpu_mlir": _sha256(probe.gpu_source),
            "binary_mlir": _sha256(probe.artifact.binary_mlir),
            "ptx": _sha256(probe.artifact.ptx),
        },
        "toolchain": {
            "version": probe.artifact.mlir_opt_version,
            "llvm_source_revision": probe.artifact.mlir_opt_source_revision,
            "chip": probe.artifact.target_chip,
            "features": probe.artifact.target_features,
            "ptx_version": probe.artifact.ptx_version,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _prepare_output(args.output_dir)
    lower_options = {
        "mlir_opt": args.mlir_opt,
        "use_wsl": args.wsl,
        "wsl_distribution": args.wsl_distribution,
        "timeout": args.timeout,
    }

    factories = []
    if args.which in ("fc1", "both"):
        factories.append(("fc1", bevformer_fc1_tiled_config))
    if args.which in ("fc2", "both"):
        factories.append(("fc2", bevformer_fc2_tiled_config))

    report: dict[str, object] = {
        "kind": "gac-bevformer-tiled-matmul-sm86-probe-v1",
        "tiled_matmuls": {},
        "mma_capability_probes": {},
        "functional_mma_capability_probe": None,
        "functional_mma_capability_probes": {},
        "async_double_buffer_capability_probe": None,
    }
    tiled_results = report["tiled_matmuls"]
    assert isinstance(tiled_results, dict)
    for name, factory in factories:
        config = factory(
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
        )
        program = lower_tiled_matmul_to_ptx(config, **lower_options)
        source_name = f"{name}.gpu.mlir"
        binary_name = f"{name}.binary.mlir"
        ptx_name = f"{name}.ptx"
        (output / source_name).write_text(program.gpu_source, encoding="utf-8", newline="\n")
        (output / binary_name).write_text(
            program.artifact.binary_mlir, encoding="utf-8", newline="\n"
        )
        (output / ptx_name).write_text(program.ptx, encoding="utf-8", newline="\n")
        tiled_results[name] = {
            "config": asdict(program.config),
            "grid": list(program.grid),
            "block": list(program.block),
            "shared_memory_bytes": program.config.shared_memory_bytes,
            "evidence": asdict(program.evidence),
            "uses_tensor_cores": program.evidence.uses_tensor_cores,
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

    if args.probe_mma:
        mma_results = report["mma_capability_probes"]
        assert isinstance(mma_results, dict)
        for mode in ("fp16", "tf32"):
            probe = probe_nvgpu_mma_to_ptx(mode, **lower_options)
            source_name = f"nvgpu-{mode}-smoke.gpu.mlir"
            ptx_name = f"nvgpu-{mode}-smoke.ptx"
            (output / source_name).write_text(
                probe.gpu_source, encoding="utf-8", newline="\n"
            )
            (output / ptx_name).write_text(
                probe.artifact.ptx, encoding="utf-8", newline="\n"
            )
            mma_results[mode] = {
                "scope": "compiler-capability-smoke-only",
                "evidence": asdict(probe.evidence),
                "uses_tensor_cores": probe.evidence.uses_tensor_cores,
                "files": {"gpu_mlir": source_name, "ptx": ptx_name},
                "ptx_sha256": _sha256(probe.artifact.ptx),
            }

    if args.probe_functional_mma:
        functional_results = report["functional_mma_capability_probes"]
        assert isinstance(functional_results, dict)
        for probe in (
            probe_nvgpu_tf32_matmul_to_ptx(**lower_options),
            probe_nvgpu_fp16_matmul_to_ptx(**lower_options),
        ):
            functional_results[probe.mode] = _write_functional_probe(output, probe)
        # Keep the original singular TF32 record for report-reader compatibility.
        report["functional_mma_capability_probe"] = functional_results["tf32"]

    if args.probe_async_double_buffer:
        probe = probe_nvgpu_async_double_buffer_to_ptx(**lower_options)
        source_name = "nvgpu-async-double-buffer-smoke.gpu.mlir"
        binary_name = "nvgpu-async-double-buffer-smoke.binary.mlir"
        ptx_name = "nvgpu-async-double-buffer-smoke.ptx"
        (output / source_name).write_text(
            probe.gpu_source, encoding="utf-8", newline="\n"
        )
        (output / binary_name).write_text(
            probe.artifact.binary_mlir, encoding="utf-8", newline="\n"
        )
        (output / ptx_name).write_text(
            probe.artifact.ptx, encoding="utf-8", newline="\n"
        )
        report["async_double_buffer_capability_probe"] = {
            "scope": "compiler-capability-and-overlap-schedule-smoke-only",
            "production_ffn_integration": False,
            "stage_count": probe.stage_count,
            "stage_elements": probe.stage_elements,
            "shared_memory_bytes": probe.shared_memory_bytes,
            "evidence": asdict(probe.evidence),
            "has_two_stage_async_pipeline": (
                probe.evidence.has_two_stage_async_pipeline
            ),
            "files": {
                "gpu_mlir": source_name,
                "binary_mlir": binary_name,
                "ptx": ptx_name,
            },
            "sha256": {
                "gpu_mlir": _sha256(probe.gpu_source),
                "binary_mlir": _sha256(probe.artifact.binary_mlir),
                "ptx": _sha256(probe.artifact.ptx),
            },
            "toolchain": {
                "version": probe.artifact.mlir_opt_version,
                "llvm_source_revision": probe.artifact.mlir_opt_source_revision,
                "chip": probe.artifact.target_chip,
                "features": probe.artifact.target_features,
                "ptx_version": probe.artifact.ptx_version,
            },
        }

    manifest = output / "report.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output_dir": str(output), "report": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
