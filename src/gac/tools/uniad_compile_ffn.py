#!/usr/bin/env python3
"""Compile a UniAD FFN bundle to CUDA/HXP and MLIR deployment artifacts.

Run this command in the separate General AI Compiler Python 3.11 / CUDA 12
environment. Compilation itself is CPU-only. The offline PTX backends require
an upstream ``mlir-opt`` with the NVPTX target; their emitted artifacts do not
require NVRTC when executed on an SM86 GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from gac.targets.cuda.compiler import compile_cuda
from gac.targets.cuda.cublas_ptx import TARGET as MLIR_CUBLAS_PTX_TARGET, compile_mlir_cublas_ptx
from gac.targets.cuda.mlir import compile_mlir_cuda
from gac.models.api import get_model_adapter
from gac.artifacts.cuda_cublas_ptx import (
    ARTIFACT_FILES as MLIR_CUBLAS_PTX_FILES,
    MANIFEST_FILE as MLIR_CUBLAS_PTX_MANIFEST,
    write_mlir_cublas_ptx_artifact,
)
from gac.targets.cuda.tiled_ffn import DEFAULT_TILE, TARGET as MLIR_TILED_FFN_TARGET, compile_mlir_tiled_ffn
from gac.artifacts.cuda_tiled_ffn import (
    ARTIFACT_FILES as MLIR_TILED_FFN_FILES,
    MANIFEST_FILE as MLIR_TILED_FFN_MANIFEST,
    write_mlir_tiled_ffn_artifact,
)
from gac.toolchains.mlir import DEFAULT_TIMEOUT_SECONDS
from gac.artifacts.hxp import write_package
from gac.models.uniad import load_bundle


CUDA_HXP_TARGET = "cuda-hxp"
MLIR_CUDA_TARGET = "mlir-cuda-sm86"
LEGACY_BOTH = "both"
ALL_TARGETS = "all"
INCOMPLETE_MARKER = ".gac-incomplete"
BACKEND_CHOICES = (
    CUDA_HXP_TARGET,
    MLIR_CUDA_TARGET,
    LEGACY_BOTH,
    MLIR_CUBLAS_PTX_TARGET,
    MLIR_TILED_FFN_TARGET,
    ALL_TARGETS,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return timeout


def _tile(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "tile must be three comma-separated positive integers: M,N,K"
        ) from exc
    if len(result) != 3 or any(dimension <= 0 for dimension in result):
        raise argparse.ArgumentTypeError(
            "tile must be three comma-separated positive integers: M,N,K"
        )
    return result[0], result[1], result[2]


def _selected(requested: str, target: str) -> bool:
    if requested == ALL_TARGETS:
        return True
    if requested == LEGACY_BOTH:
        return target in {CUDA_HXP_TARGET, MLIR_CUDA_TARGET}
    return requested == target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=LEGACY_BOTH,
        help=(
            "'both' preserves the CUDA/HXP + legacy MLIR call-plan pair; "
            "'all' additionally emits both offline MLIR/PTX artifacts"
        ),
    )
    parser.add_argument(
        "--mlir-opt",
        help="mlir-opt executable, or its Linux path when --wsl is used",
    )
    parser.add_argument(
        "--wsl",
        action="store_true",
        help="run the Linux mlir-opt selected by --mlir-opt through WSL",
    )
    parser.add_argument(
        "--wsl-distribution",
        help="optional WSL distribution name, for example Ubuntu-24.04",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "timeout for each MLIR-to-PTX tool invocation in seconds "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--tile",
        type=_tile,
        default=DEFAULT_TILE,
        metavar="M,N,K",
        help=(
            "executable MatMul tile for mlir-tiled-ffn-sm86 "
            f"(default: {','.join(str(value) for value in DEFAULT_TILE)})"
        ),
    )
    return parser


def _validate_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    has_offline_ptx = any(
        _selected(args.backend, target)
        for target in (MLIR_CUBLAS_PTX_TARGET, MLIR_TILED_FFN_TARGET)
    )
    has_tiled = _selected(args.backend, MLIR_TILED_FFN_TARGET)
    if args.wsl_distribution and not args.wsl:
        parser.error("--wsl-distribution requires --wsl")
    if args.wsl and not args.mlir_opt:
        parser.error("--wsl requires --mlir-opt with a Linux executable path")
    if not has_offline_ptx and (
        args.mlir_opt
        or args.wsl
        or args.wsl_distribution
        or args.timeout != DEFAULT_TIMEOUT_SECONDS
    ):
        parser.error(
            "MLIR toolchain options require mlir-cublas-ptx-sm86, "
            "mlir-tiled-ffn-sm86, or all"
        )
    if not has_tiled and args.tile != DEFAULT_TILE:
        parser.error("--tile requires mlir-tiled-ffn-sm86 or all")
    if args.output.exists() or args.output.is_symlink():
        parser.error(
            "output must not already exist; use a fresh directory so artifacts "
            "from different compiler runs cannot be mixed"
        )


def _validate_offline_profile(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    profile: Any,
) -> None:
    """Keep UniAD-only PTX commands from silently selecting the BEV profile."""

    has_offline_ptx = any(
        _selected(args.backend, target)
        for target in (MLIR_CUBLAS_PTX_TARGET, MLIR_TILED_FFN_TARGET)
    )
    if not has_offline_ptx:
        return
    actual = (
        profile.batch,
        profile.bev_h,
        profile.bev_w,
        profile.embed_dim,
        profile.hidden_dim,
        profile.dtype,
        profile.layer_norm_epsilon,
    )
    expected = (1, 200, 200, 256, 512, "float32", 1.0e-5)
    if actual != expected:
        parser.error(
            "offline MLIR/PTX backends require the production UniAD profile "
            "batch=1, bev=200x200, embed=256, hidden=512, float32, epsilon=1e-5"
        )


def _write_mlir_artifact(output: Path, program: Any) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "abi.mlir": program.source.encode("utf-8"),
        "tensor.mlir": program.tensor_source.encode("utf-8"),
        "kernels.cu": program.kernel_source.encode("utf-8"),
        "calls.json": _canonical_json(
            {"calls": [call.to_dict() for call in program.calls]}
        ).encode("utf-8"),
    }
    records: dict[str, Any] = {}
    for name, payload in files.items():
        path = output / name
        path.write_bytes(payload)
        records[name] = {"sha256": _sha256_bytes(payload), "size": len(payload)}

    manifest = {
        "schema_version": 1,
        "format": "uniad-mlir-cuda-call-plan",
        "boundary": "one UniAD stage2 BEV encoder FFN; not end-to-end UniAD",
        "compiler_backend": MLIR_CUDA_TARGET,
        "executed_custom_artifact": "runtime-NVRTC-compiled-CUDA",
        "ptx_generated_offline": False,
        "nvrtc_required": True,
        "description": program.describe(),
        "files": records,
    }
    rendered = _canonical_json(manifest).encode("utf-8")
    (output / "manifest.json").write_bytes(rendered)
    return {
        "directory": str(output.resolve()),
        "manifest_sha256": _sha256_bytes(rendered),
        "integrity_sha256": program.integrity_sha256,
        "call_count": len(program.calls),
        "calls": [call.to_dict() for call in program.calls],
        "executed_custom_artifact": "runtime-NVRTC-compiled-CUDA",
        "ptx_generated_offline": False,
        "nvrtc_required": True,
    }


def _official_ptx_report(
    *,
    directory: Path,
    manifest_name: str,
    manifest: dict[str, Any],
    artifact_files: dict[str, str],
    compile_ms: float,
) -> dict[str, Any]:
    """Summarize one official artifact without hiding its executable PTX origin."""

    ptx_files = {
        key: dict(manifest["files"][key])
        for key in artifact_files
        if key == "ptx" or key.endswith("_ptx")
    }
    manifest_path = directory / manifest_name
    integrity = manifest.get("integrity", {})
    manifest_sha256 = (
        _sha256_file(manifest_path) if manifest_path.is_file() else None
    )
    return {
        "directory": str(directory.resolve()),
        "compile_ms": compile_ms,
        "artifact": manifest["artifact"],
        "artifact_schema_version": manifest["artifact_schema_version"],
        "compiler_backend": manifest["compiler_backend"],
        "deployment_boundary": manifest["deployment_boundary"],
        "executable": manifest["executable"],
        "executed_custom_artifact": manifest.get(
            "executed_custom_artifact", "textual-ptx"
        ),
        "ptx_generated_offline": True,
        "ptx_dispatch_count": manifest["ptx_dispatch_count"],
        "ptx_files": ptx_files,
        "nvrtc_required": False,
        "matmul_lowering": manifest["matmul_lowering"],
        "fusion_lowering": manifest["fusion_lowering"],
        "dispatch_count": manifest["dispatch_count"],
        "integrity_sha256": integrity.get("integrity_sha256"),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "payload_sha256": manifest["manifest_payload_sha256"],
        },
        "toolchain": manifest.get("toolchain"),
        "toolchains": manifest.get("toolchains"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_options(parser, args)

    loaded = load_bundle(args.bundle)
    profile = loaded.contract
    _validate_offline_profile(parser, args, profile)
    bundle_sha256 = _sha256_file(args.bundle)
    if profile.hidden_dim != profile.embed_dim * 2:
        raise RuntimeError("current GAC FFN lowering requires hidden_dim == 2 * embed_dim")

    graph = get_model_adapter("uniad-ffn").build(loaded)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(
            "output was created concurrently; choose a fresh output directory"
        )
    incomplete_marker = args.output / INCOMPLETE_MARKER
    incomplete_marker.write_text(
        "Compilation is incomplete until compile-report.json is published.\n",
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "one UniAD stage2 BEV encoder FFN",
        "boundary": "compiler subgraph artifact; not end-to-end UniAD inference",
        "profile": profile.to_dict(),
        "source_bundle": {
            "path": str(args.bundle.resolve()),
            "sha256": bundle_sha256,
            "format": loaded.manifest["format"],
            "schema_version": loaded.manifest["schema_version"],
        },
        "requested_backend": args.backend,
        "selected_backends": [
            target
            for target in (
                CUDA_HXP_TARGET,
                MLIR_CUDA_TARGET,
                MLIR_CUBLAS_PTX_TARGET,
                MLIR_TILED_FFN_TARGET,
            )
            if _selected(args.backend, target)
        ],
        "offline_mlir_toolchain": {
            "mlir_opt": args.mlir_opt,
            "use_wsl": args.wsl,
            "wsl_distribution": args.wsl_distribution,
            "timeout_seconds": args.timeout,
            "tiled_matmul_tile": list(args.tile),
        },
        "backends": {},
    }

    if _selected(args.backend, CUDA_HXP_TARGET):
        started = time.perf_counter()
        cuda_program = compile_cuda(graph)
        compile_ms = (time.perf_counter() - started) * 1000.0
        hxp_path = args.output / (
            "uniad-stage2-encoder-layer-%d-sm86.hxp" % profile.layer
        )
        write_package(
            hxp_path,
            cuda_program,
            metadata={
                "model": "UniAD 2.0",
                "component": "stage2 BEV encoder FFN",
                "layer": profile.layer,
                "bundle_sha256": bundle_sha256,
            },
        )
        report["backends"]["cuda_hxp"] = {
            "path": str(hxp_path.resolve()),
            "sha256": _sha256_file(hxp_path),
            "compile_ms": compile_ms,
            "dispatch_count": len(cuda_program.launches),
            "matmul_lowering": "runtime-select cupy-cublas; generated CUDA fallback is explicit",
            "executed_custom_artifact": "runtime-NVRTC-compiled-CUDA",
            "ptx_generated_offline": False,
            "nvrtc_required": True,
        }

    if _selected(args.backend, MLIR_CUDA_TARGET):
        started = time.perf_counter()
        mlir_program = compile_mlir_cuda(graph)
        compile_ms = (time.perf_counter() - started) * 1000.0
        artifact = _write_mlir_artifact(args.output / "mlir-cuda-sm86", mlir_program)
        artifact["compile_ms"] = compile_ms
        report["backends"]["mlir_cuda_sm86"] = artifact

    toolchain = {
        "mlir_opt": args.mlir_opt,
        "use_wsl": args.wsl,
        "wsl_distribution": args.wsl_distribution,
        "timeout": args.timeout,
    }
    cublas_ptx_program = None
    cublas_ptx_compile_ms: float | None = None
    if _selected(args.backend, MLIR_CUBLAS_PTX_TARGET):
        started = time.perf_counter()
        cublas_ptx_program = compile_mlir_cublas_ptx(graph, **toolchain)
        cublas_ptx_compile_ms = (time.perf_counter() - started) * 1000.0
        artifact_directory = args.output / MLIR_CUBLAS_PTX_TARGET
        manifest = write_mlir_cublas_ptx_artifact(
            artifact_directory, cublas_ptx_program
        )
        backend_report = _official_ptx_report(
            directory=artifact_directory,
            manifest_name=MLIR_CUBLAS_PTX_MANIFEST,
            manifest=manifest,
            artifact_files=MLIR_CUBLAS_PTX_FILES,
            compile_ms=cublas_ptx_compile_ms,
        )
        backend_report["compile_ms_scope"] = "complete standalone artifact"
        report["backends"]["mlir_cublas_ptx_sm86"] = backend_report

    if _selected(args.backend, MLIR_TILED_FFN_TARGET):
        reused_base_program = cublas_ptx_program is not None
        started = time.perf_counter()
        tiled_program = compile_mlir_tiled_ffn(
            graph,
            tile_m=args.tile[0],
            tile_n=args.tile[1],
            tile_k=args.tile[2],
            base_program=cublas_ptx_program,
            **toolchain,
        )
        incremental_compile_ms = (time.perf_counter() - started) * 1000.0
        dependency_compile_ms = (
            cublas_ptx_compile_ms if reused_base_program else None
        )
        compile_ms = incremental_compile_ms + (dependency_compile_ms or 0.0)
        artifact_directory = args.output / MLIR_TILED_FFN_TARGET
        manifest = write_mlir_tiled_ffn_artifact(
            artifact_directory, tiled_program
        )
        backend_report = _official_ptx_report(
            directory=artifact_directory,
            manifest_name=MLIR_TILED_FFN_MANIFEST,
            manifest=manifest,
            artifact_files=MLIR_TILED_FFN_FILES,
            compile_ms=compile_ms,
        )
        backend_report.update(
            {
                "compile_ms_scope": (
                    "complete standalone artifact including fusion dependency"
                ),
                "invocation_compile_ms": incremental_compile_ms,
                "incremental_compile_ms": (
                    incremental_compile_ms if reused_base_program else None
                ),
                "dependency_compile_ms": dependency_compile_ms,
                "base_program_reused": reused_base_program,
            }
        )
        report["backends"]["mlir_tiled_ffn_sm86"] = backend_report

    report_path = args.output / "compile-report.json"
    report_path.write_text(_canonical_json(report), encoding="utf-8")
    incomplete_marker.unlink()
    print(report_path.resolve())


if __name__ == "__main__":
    main()
