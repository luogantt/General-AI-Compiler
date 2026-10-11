#!/usr/bin/env python3
"""Compare executable MLIR/cuBLAS, NVRTC, and optional tiled PTX FFNs.

The default measured unit remains one production-shape FP32 BEVFormer FFN.
``--uniad-bundle`` selects one production-shape UniAD stage-2 BEV encoder FFN
from a verified compiler bundle instead.  Neither mode measures end-to-end
model inference.  The primary pair uses two CuPy/cuBLAS MatMuls and differs
only in whether its two fusion kernels come from official MLIR/NVVM textual
PTX or NVRTC CUDA.  An optional three-dispatch candidate executes two real
shared-memory tiled MLIR PTX MatMuls plus the final MLIR PTX fusion.
Device-resident latency is measured with CUDA Events.  Public run latency
separately includes input validation, H2D, compute, D2H, and sync.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np

from benchmark_mlir_cuda_ffn import (
    ATOL,
    RTOL,
    _correctness,
    _distribution,
    _git_state,
    _make_graph,
    _positive,
    _public_run_timing,
    _resident_input_provenance,
    _resident_sample,
    _upload_resident_query,
    _warmup_resident_once,
)
from gac.backends.mlir_cublas_ptx import compile_mlir_cublas_ptx
from gac.backends.mlir_cuda import compile_mlir_cuda
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.mlir_cublas_ptx_artifact import (
    ARTIFACT_FILES,
    MANIFEST_FILE,
    load_mlir_cublas_ptx_artifact,
    write_mlir_cublas_ptx_artifact,
)
from gac.mlir_cublas_ptx_benchmark import (
    FfnSessionAutotuneRunner,
    executable_ffn_candidates,
    skipped_tiled_candidate_report,
)
from gac.mlir_cublas_ptx_runtime import MlirCublasPtxSession
from gac.mlir_ptx_autotune import AutotuneConfig, autotune_mlir_ptx_ffn
from gac.mlir_runtime import MlirCudaSession
from gac.mlir_tiled_ffn import compile_mlir_tiled_ffn
from gac.mlir_tiled_ffn_artifact import (
    ARTIFACT_FILES as TILED_ARTIFACT_FILES,
    MANIFEST_FILE as TILED_MANIFEST_FILE,
    load_mlir_tiled_ffn_artifact,
    write_mlir_tiled_ffn_artifact,
)
from gac.mlir_tiled_ffn_runtime import MlirTiledFfnSession
from gac.runtime import reference_run
from gac.uniad import ArtifactError, load_bundle


PTX_PATH = "mlir_cublas_ptx"
NVRTC_PATH = "mlir_cuda_nvrtc"
TILED_PATH = "mlir_tiled_ffn"
UNIAD_DEFAULT_SEED = 20261011


@dataclass(frozen=True, slots=True)
class _Workload:
    graph: Any
    query: np.ndarray
    expected: np.ndarray | None
    family: str
    metadata: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return parsed


def _prepare_workload(uniad_bundle: Path | None, seed: int) -> _Workload:
    """Build the unchanged BEV workload or one verified UniAD bundle workload."""

    if uniad_bundle is None:
        graph, query = _make_graph()
        return _Workload(
            graph=graph,
            query=query,
            expected=None,
            family="bevformer",
            metadata={
                "model": "BEVFormer-tiny",
                "component": "BEV encoder FFN",
                "profile": "bevformer-tiny",
                "query_shape": list(query.shape),
                "output_shape": list(query.shape),
                "hidden_channels": 512,
                "dtype": query.dtype.name,
                "query_source": "built-in deterministic synthetic workload",
                "reference_source": "GAC NumPy reference",
            },
        )

    bundle_path = uniad_bundle.expanduser().resolve()
    loaded = load_bundle(bundle_path)
    profile = loaded.contract
    actual_profile = (
        profile.batch,
        profile.bev_h,
        profile.bev_w,
        profile.embed_dim,
        profile.hidden_dim,
        profile.dtype,
        profile.layer_norm_epsilon,
    )
    if actual_profile != (1, 200, 200, 256, 512, "float32", 1.0e-5):
        raise ValueError(
            "--uniad-bundle must use the production UniAD FP32 profile "
            "batch=1, bev=200x200, embed=256, hidden=512, epsilon=1e-5"
        )
    config = BEVFormerFFNConfig(
        batch=profile.batch,
        bev_h=profile.bev_h,
        bev_w=profile.bev_w,
        embed_dim=profile.embed_dim,
        post_layer_norm=True,
        layer_norm_epsilon=profile.layer_norm_epsilon,
        ffn_dropout=0.1,
    )
    graph = build_bevformer_ffn_graph(
        loaded.weights,
        config,
        name=f"uniad_stage2_encoder_layer_{profile.layer}_ffn",
    )
    if loaded.sample is None:
        generator = np.random.default_rng(seed)
        query = generator.normal(0.0, 0.1, profile.input_shape).astype(np.float32)
        expected = None
        query_source = "deterministic seeded synthetic query"
        reference_source = "GAC NumPy reference"
        effective_seed: int | None = seed
    else:
        query = np.ascontiguousarray(loaded.sample["query"], dtype=np.float32)
        expected = np.ascontiguousarray(loaded.sample["output"], dtype=np.float32)
        query_source = "bundle-provided query sample"
        reference_source = "bundle-provided expected output"
        effective_seed = None
    return _Workload(
        graph=graph,
        query=np.ascontiguousarray(query),
        expected=expected,
        family="uniad",
        metadata={
            "model": "UniAD 2.0",
            "component": "stage2 BEV encoder FFN",
            "layer": profile.layer,
            "profile": profile.to_dict(),
            "query_shape": list(profile.input_shape),
            "output_shape": list(profile.output_shape),
            "hidden_channels": profile.hidden_dim,
            "dtype": "float32",
            "query_source": query_source,
            "reference_source": reference_source,
            "seed": effective_seed,
            "source_bundle": {
                "path": str(bundle_path),
                "sha256": _sha256_file(bundle_path),
                "format": loaded.manifest["format"],
                "schema_version": loaded.manifest["schema_version"],
                "contains_expected_sample": loaded.sample is not None,
            },
        },
    )


def _report_identity(workload: _Workload, *, compile_only: bool) -> dict[str, str]:
    if workload.family == "uniad":
        layer = workload.metadata["layer"]
        return {
            "kind": (
                "uniad-stage2-encoder-ffn-mlir-cublas-ptx-offline-compilation"
                if compile_only
                else "uniad-stage2-encoder-ffn-mlir-cublas-ptx-vs-mlir-cuda-nvrtc"
            ),
            "scope": (
                f"one FP32 UniAD 2.0 stage2 BEV encoder layer {layer} FFN "
                "[1,40000,256] -> 512 -> 256"
            ),
            "boundary": (
                "production-shape stage2 BEV encoder FFN fragment; "
                "not full UniAD inference"
            ),
        }
    return {
        "kind": (
            "bevformer-ffn-mlir-cublas-ptx-offline-compilation"
            if compile_only
            else "bevformer-ffn-mlir-cublas-ptx-vs-mlir-cuda-nvrtc"
        ),
        "scope": "one FP32 BEVFormer FFN [1,2500,256] -> 512 -> 256",
        "boundary": "production-shape FFN fragment; not full BEVFormer inference",
    }


def _workload_sha256(
    graph: Any, query: np.ndarray, expected: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            graph.to_dict(include_constant_data=False),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(graph.constants):
        value = np.ascontiguousarray(graph.constants[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.tobytes(order="C"))
    for name, value in (("query", query), ("expected.output", expected)):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _path_is_within(path: Path, directory: Path) -> bool:
    candidate = path.expanduser().resolve(strict=False)
    parent = directory.expanduser().resolve(strict=False)
    return candidate == parent or parent in candidate.parents


def _path_aliases_artifact(path: Path, directory: Path) -> bool:
    if _path_is_within(path, directory):
        return True
    candidate = path.expanduser()
    for name in (
        *ARTIFACT_FILES.values(),
        MANIFEST_FILE,
        *TILED_ARTIFACT_FILES.values(),
        TILED_MANIFEST_FILE,
    ):
        member = directory.expanduser() / name
        try:
            if candidate.exists() and member.exists() and candidate.samefile(member):
                return True
        except OSError:
            pass
    return False


def _paths_alias(first: Path, second: Path) -> bool:
    if first.expanduser().resolve(strict=False) == second.expanduser().resolve(
        strict=False
    ):
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def _order(repeat: int, initial: str) -> tuple[str, str]:
    if initial == "ptx-first":
        first = (PTX_PATH, NVRTC_PATH)
    elif initial == "nvrtc-first":
        first = (NVRTC_PATH, PTX_PATH)
    else:
        raise ValueError(f"unknown benchmark order {initial!r}")
    return first if repeat % 2 == 0 else (first[1], first[0])


def _paired_timing(
    sessions: dict[str, Any],
    submits: dict[str, Callable[[], None]],
    query: np.ndarray,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    initial: str,
) -> dict[str, Any]:
    for name in (PTX_PATH, NVRTC_PATH):
        _upload_resident_query(sessions[name], query)
    for index in range(warmup):
        for name in _order(index, initial):
            _warmup_resident_once(sessions[name], submits[name])

    gpu = {PTX_PATH: [], NVRTC_PATH: []}
    wall = {PTX_PATH: [], NVRTC_PATH: []}
    samples: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = _order(repeat, initial)
        sample: dict[str, Any] = {"repeat": repeat + 1, "order": list(order)}
        for name in order:
            gpu_us, wall_us = _resident_sample(
                sessions[name], submits[name], iterations=iterations
            )
            gpu[name].append(gpu_us)
            wall[name].append(wall_us)
            sample[name] = {
                "gpu_us": gpu_us,
                "synchronized_wall_us": wall_us,
            }
        samples.append(sample)
    return {
        "gpu_us": {name: _distribution(gpu[name]) for name in gpu},
        "synchronized_wall_us": {
            name: _distribution(wall[name]) for name in wall
        },
        "repeat_samples": samples,
    }


def _single_path_timing(
    session: Any,
    submit: Callable[[], None],
    query: np.ndarray,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure one optional real session with the same resident protocol."""

    _upload_resident_query(session, query)
    for _ in range(warmup):
        _warmup_resident_once(session, submit)
    gpu: list[float] = []
    wall: list[float] = []
    for _ in range(repeats):
        gpu_us, wall_us = _resident_sample(
            session, submit, iterations=iterations
        )
        gpu.append(gpu_us)
        wall.append(wall_us)
    return _distribution(gpu), _distribution(wall)


def _tile(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tile must be M,N,K positive integers") from exc
    if len(result) != 3 or any(dimension <= 0 for dimension in result):
        raise argparse.ArgumentTypeError("tile must be M,N,K positive integers")
    return result


def _comparison(paired: dict[str, Any]) -> dict[str, Any]:
    speedups: list[float] = []
    differences: list[float] = []
    by_order = {"ptx-first": [], "nvrtc-first": []}
    for sample in paired["repeat_samples"]:
        ptx_us = float(sample[PTX_PATH]["gpu_us"])
        nvrtc_us = float(sample[NVRTC_PATH]["gpu_us"])
        if ptx_us <= 0.0 or nvrtc_us <= 0.0:
            raise ValueError("CUDA Event timings must be positive")
        speedup = nvrtc_us / ptx_us
        key = "ptx-first" if sample["order"][0] == PTX_PATH else "nvrtc-first"
        speedups.append(speedup)
        differences.append(ptx_us - nvrtc_us)
        by_order[key].append(speedup)
    if not all(by_order.values()):
        raise ValueError("paired comparison requires both measurement orders")
    order_distributions = {
        name: _distribution(values) for name, values in by_order.items()
    }
    balanced = math.sqrt(
        order_distributions["ptx-first"]["median"]
        * order_distributions["nvrtc-first"]["median"]
    )
    return {
        "method": "paired-interleaved-order-balanced",
        "primary_estimator": (
            "geometric mean of the PTX-first and NVRTC-first median per-repeat "
            "NVRTC/PTX CUDA Event latency ratios"
        ),
        "mlir_cublas_ptx_speedup_over_mlir_cuda_nvrtc": balanced,
        "mlir_cublas_ptx_minus_mlir_cuda_nvrtc_us": _distribution(differences),
        "speedup_samples": _distribution(speedups),
        "speedup_by_order": order_distributions,
        "ratio_of_backend_medians": (
            paired["gpu_us"][NVRTC_PATH]["median"]
            / paired["gpu_us"][PTX_PATH]["median"]
        ),
    }


def _path_report(
    program: Any,
    session: Any,
    first: Any,
    correctness: dict[str, Any],
    resident_gpu: dict[str, Any],
    resident_wall: dict[str, Any],
    public_gpu: dict[str, Any],
    public_wall: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lowering": program.describe(),
        "session_setup_ms": float(session.setup_ms),
        "runtime": {
            "compiler_backend": session.info.get("compiler_backend"),
            "matmul_backend": session.info.get("matmul_backend"),
            "fusion_backend": session.info.get("fusion_backend"),
            "implementations": [
                entry.get("implementation") for entry in session.kernel_info
            ],
        },
        "correctness": correctness,
        "first_public_run": {
            "gpu_stream_us": float(first.gpu_stream_ms) * 1000.0,
            "wall_us": float(first.wall_ms) * 1000.0,
        },
        "device_resident_pipeline_gpu_us": resident_gpu,
        "device_resident_synchronized_wall_us": resident_wall,
        "public_run_h2d_compute_gpu_us": public_gpu,
        "public_run_h2d_compute_d2h_wall_us": public_wall,
        "ffn_calls_per_second": 1.0e6 / resident_gpu["median"],
    }


def _source_manifest(root: Path) -> dict[str, str]:
    paths = sorted((root / "src" / "gac").rglob("*.py"))
    paths.extend(
        root / "tools" / name
        for name in (
            "benchmark_mlir_cuda_ffn.py",
            "benchmark_mlir_cublas_ptx_ffn.py",
        )
    )
    return {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def _render(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uniad-bundle",
        type=Path,
        help=(
            "benchmark one verified UniAD stage2 encoder FFN bundle instead "
            "of the default BEVFormer workload"
        ),
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=UNIAD_DEFAULT_SEED,
        help=(
            "synthetic UniAD query seed when --uniad-bundle has no Torch golden "
            f"(default: {UNIAD_DEFAULT_SEED})"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=_positive, default=20)
    parser.add_argument("--iterations", type=_positive, default=50)
    parser.add_argument("--repeats", type=_positive, default=20)
    parser.add_argument("--public-samples", type=_positive, default=30)
    parser.add_argument(
        "--benchmark-order",
        choices=("ptx-first", "nvrtc-first"),
        default="ptx-first",
    )
    parser.add_argument("--mlir-opt")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--emit-artifact-dir", type=Path)
    parser.add_argument(
        "--tiled-artifact-dir",
        type=Path,
        help="load one real three-dispatch tiled FFN candidate",
    )
    parser.add_argument(
        "--emit-tiled-artifact-dir",
        type=Path,
        help="compile and atomically emit one tiled FFN candidate",
    )
    parser.add_argument(
        "--tiled-tile",
        type=_tile,
        default=(16, 16, 16),
        metavar="M,N,K",
        help="real MatMul tile used when compiling --emit-tiled-artifact-dir",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile and atomically emit the artifact without accessing CUDA",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help=(
            "tune the two library-backed full-FFN sessions and, when supplied, "
            "the real tiled full-FFN session"
        ),
    )
    parser.add_argument("--autotune-cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.repeats % 2:
        parser.error("--repeats must be even for order-balanced comparison")
    if (
        args.uniad_bundle is not None
        and not args.uniad_bundle.expanduser().is_file()
    ):
        parser.error(f"--uniad-bundle does not exist: {args.uniad_bundle}")
    if args.artifact_dir and args.emit_artifact_dir:
        parser.error("--artifact-dir and --emit-artifact-dir are mutually exclusive")
    if args.tiled_artifact_dir and args.emit_tiled_artifact_dir:
        parser.error(
            "--tiled-artifact-dir and --emit-tiled-artifact-dir are mutually exclusive"
        )
    if args.compile_only and (args.artifact_dir or not args.emit_artifact_dir):
        parser.error("--compile-only requires --emit-artifact-dir and no --artifact-dir")
    if args.compile_only and (args.autotune or args.autotune_cache):
        parser.error("--compile-only cannot run autotuning")
    if args.compile_only and args.tiled_artifact_dir:
        parser.error("--compile-only cannot load --tiled-artifact-dir")
    if args.autotune_cache and not args.autotune:
        parser.error("--autotune-cache requires --autotune")
    if args.wsl_distribution and not args.wsl:
        parser.error("--wsl-distribution requires --wsl")
    if args.artifact_dir and (args.mlir_opt or args.wsl or args.wsl_distribution):
        parser.error("toolchain options cannot be used with --artifact-dir")
    artifact_directories = (
        args.artifact_dir,
        args.emit_artifact_dir,
        args.tiled_artifact_dir,
        args.emit_tiled_artifact_dir,
    )
    present_artifact_directories = [
        path for path in artifact_directories if path is not None
    ]
    resolved_artifacts = [
        path.expanduser().resolve(strict=False)
        for path in present_artifact_directories
    ]
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        parser.error("base and tiled artifact directories must be different")
    for artifact_directory in artifact_directories:
        if artifact_directory is None:
            continue
        if args.output and _path_aliases_artifact(args.output, artifact_directory):
            parser.error("--output must be outside the artifact directory")
        if args.autotune_cache and _path_aliases_artifact(
            args.autotune_cache, artifact_directory
        ):
            parser.error("--autotune-cache must be outside the artifact directory")
    if args.output and args.autotune_cache and _paths_alias(
        args.output, args.autotune_cache
    ):
        parser.error("--output and --autotune-cache must name different files")
    if args.uniad_bundle and args.output and _paths_alias(
        args.uniad_bundle, args.output
    ):
        parser.error("--output must not replace --uniad-bundle")
    if args.uniad_bundle and args.autotune_cache and _paths_alias(
        args.uniad_bundle, args.autotune_cache
    ):
        parser.error("--autotune-cache must not replace --uniad-bundle")

    root = Path(__file__).resolve().parents[1]
    try:
        workload = _prepare_workload(args.uniad_bundle, args.seed)
    except (ArtifactError, OSError, ValueError) as exc:
        parser.error(f"cannot prepare benchmark workload: {exc}")
    graph, query = workload.graph, workload.query
    preparation_started = time.perf_counter()
    if args.artifact_dir:
        ptx_program = load_mlir_cublas_ptx_artifact(args.artifact_dir, graph)
        preparation = "verified-precompiled-mlir-cublas-ptx-artifact"
    else:
        ptx_program = compile_mlir_cublas_ptx(
            graph,
            mlir_opt=args.mlir_opt,
            use_wsl=args.wsl,
            wsl_distribution=args.wsl_distribution,
            timeout=args.timeout,
        )
        preparation = "official-mlir-offline-compilation"
    preparation_ms = (time.perf_counter() - preparation_started) * 1000.0
    artifact_manifest = None
    if args.emit_artifact_dir:
        artifact_manifest = write_mlir_cublas_ptx_artifact(
            args.emit_artifact_dir, ptx_program
        )
    tiled_program = None
    tiled_preparation_ms = None
    tiled_preparation = None
    tiled_artifact_manifest = None
    if args.tiled_artifact_dir:
        tiled_started = time.perf_counter()
        tiled_program = load_mlir_tiled_ffn_artifact(
            args.tiled_artifact_dir, graph
        )
        tiled_preparation_ms = (time.perf_counter() - tiled_started) * 1000.0
        tiled_preparation = "verified-precompiled-mlir-tiled-ffn-artifact"
    elif args.emit_tiled_artifact_dir:
        tiled_started = time.perf_counter()
        tiled_program = compile_mlir_tiled_ffn(
            graph,
            tile_m=args.tiled_tile[0],
            tile_n=args.tiled_tile[1],
            tile_k=args.tiled_tile[2],
            base_program=ptx_program,
            mlir_opt=args.mlir_opt,
            use_wsl=args.wsl,
            wsl_distribution=args.wsl_distribution,
            timeout=args.timeout,
        )
        tiled_preparation_ms = (time.perf_counter() - tiled_started) * 1000.0
        tiled_preparation = "official-mlir-offline-tiled-compilation"
        tiled_artifact_manifest = write_mlir_tiled_ffn_artifact(
            args.emit_tiled_artifact_dir, tiled_program
        )
    if args.compile_only:
        identity = _report_identity(workload, compile_only=True)
        _render(
            {
                "schema_version": 1,
                **identity,
                "workload": workload.metadata,
                "artifact_directory": str(args.emit_artifact_dir.resolve()),
                "compile_ms": preparation_ms,
                "lowering": ptx_program.describe(),
                "artifact_manifest": artifact_manifest,
                "tiled_lowering": (
                    tiled_program.describe() if tiled_program is not None else None
                ),
                "tiled_compile_ms": tiled_preparation_ms,
                "tiled_artifact_directory": (
                    str(args.emit_tiled_artifact_dir.resolve())
                    if args.emit_tiled_artifact_dir
                    else None
                ),
                "tiled_artifact_manifest": tiled_artifact_manifest,
            },
            args.output,
        )
        return

    if workload.expected is None:
        reference_started = time.perf_counter()
        expected = reference_run(graph, {"query": query})["output"]
        reference_ms: float | None = (
            time.perf_counter() - reference_started
        ) * 1000.0
    else:
        expected = workload.expected
        reference_ms = None
    baseline_started = time.perf_counter()
    nvrtc_program = compile_mlir_cuda(graph)
    baseline_compile_ms = (time.perf_counter() - baseline_started) * 1000.0

    setup_order = list(_order(0, args.benchmark_order))
    with ExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in setup_order:
            if name == PTX_PATH:
                sessions[name] = stack.enter_context(
                    MlirCublasPtxSession(ptx_program, device_id=args.device)
                )
            else:
                sessions[name] = stack.enter_context(
                    MlirCudaSession(nvrtc_program, device_id=args.device)
                )

        if tiled_program is not None:
            sessions[TILED_PATH] = stack.enter_context(
                MlirTiledFfnSession(tiled_program, device_id=args.device)
            )

        first = {
            name: session.run({"query": query})
            for name, session in sessions.items()
        }
        correctness = {
            name: _correctness(result.outputs["output"], expected)
            for name, result in first.items()
        }
        submits = {
            PTX_PATH: sessions[PTX_PATH]._execute_calls,
            NVRTC_PATH: sessions[NVRTC_PATH]._execute_calls,
        }
        paired = _paired_timing(
            sessions,
            submits,
            query,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            initial=args.benchmark_order,
        )
        public = {
            name: _public_run_timing(
                sessions[name], query, samples=args.public_samples
            )
            for name in setup_order
        }
        tiled_resident = None
        tiled_public = None
        if tiled_program is not None:
            tiled_resident = _single_path_timing(
                sessions[TILED_PATH],
                sessions[TILED_PATH]._execute_calls,
                query,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            tiled_public = _public_run_timing(
                sessions[TILED_PATH], query, samples=args.public_samples
            )
        ptx_report = _path_report(
            ptx_program,
            sessions[PTX_PATH],
            first[PTX_PATH],
            correctness[PTX_PATH],
            paired["gpu_us"][PTX_PATH],
            paired["synchronized_wall_us"][PTX_PATH],
            public[PTX_PATH][0],
            public[PTX_PATH][1],
        )
        nvrtc_report = _path_report(
            nvrtc_program,
            sessions[NVRTC_PATH],
            first[NVRTC_PATH],
            correctness[NVRTC_PATH],
            paired["gpu_us"][NVRTC_PATH],
            paired["synchronized_wall_us"][NVRTC_PATH],
            public[NVRTC_PATH][0],
            public[NVRTC_PATH][1],
        )
        device = dict(sessions[PTX_PATH].info)

        candidates = executable_ffn_candidates(tiled_program)
        autotune_report: dict[str, Any] = {
            "executed": False,
            "executable_candidates": [
                candidate.to_dict() for candidate in candidates
            ],
            "skipped_candidates": skipped_tiled_candidate_report(),
        }
        if args.autotune:
            tuning_sessions = {
                ptx_program.target: sessions[PTX_PATH],
                nvrtc_program.target: sessions[NVRTC_PATH],
            }
            if tiled_program is not None:
                tuning_sessions[tiled_program.target] = sessions[TILED_PATH]
            runner = FfnSessionAutotuneRunner(query, tuning_sessions)
            source_integrities = {
                ptx_program.target: ptx_program.integrity_sha256,
                nvrtc_program.target: nvrtc_program.integrity_sha256,
            }
            if tiled_program is not None:
                source_integrities[tiled_program.target] = (
                    tiled_program.integrity_sha256
                )
            source_digest = _sha256_bytes(
                json.dumps(
                    source_integrities,
                    sort_keys=True,
                ).encode("utf-8")
            )
            tuned = autotune_mlir_ptx_ffn(
                candidates,
                runner,
                {"output": expected},
                graph_sha256=_workload_sha256(graph, query, expected),
                source_sha256=source_digest,
                config=AutotuneConfig(
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    atol=ATOL,
                    rtol=RTOL,
                ),
                cache_path=args.autotune_cache,
            )
            autotune_report.update(
                {
                    "executed": not tuned.cache_hit,
                    "cache_hit": tuned.cache_hit,
                    "selection": tuned.to_cache_dict(),
                }
            )

        tiled_report = None
        if tiled_program is not None:
            assert tiled_resident is not None and tiled_public is not None
            tiled_report = _path_report(
                tiled_program,
                sessions[TILED_PATH],
                first[TILED_PATH],
                correctness[TILED_PATH],
                tiled_resident[0],
                tiled_resident[1],
                tiled_public[0],
                tiled_public[1],
            )

    ptx_report["compile_or_load_ms"] = preparation_ms
    ptx_report["preparation"] = preparation
    if args.artifact_dir:
        ptx_report["artifact_directory"] = str(args.artifact_dir.resolve())
    elif args.emit_artifact_dir:
        ptx_report["artifact_directory"] = str(args.emit_artifact_dir.resolve())
    nvrtc_report["compile_ms"] = baseline_compile_ms
    if tiled_report is not None:
        tiled_report["compile_or_load_ms"] = tiled_preparation_ms
        tiled_report["preparation"] = tiled_preparation
        tiled_report["artifact_directory"] = str(
            (
                args.tiled_artifact_dir
                if args.tiled_artifact_dir is not None
                else args.emit_tiled_artifact_dir
            ).resolve()
        )
    identity = _report_identity(workload, compile_only=False)
    report = {
        "schema_version": 1,
        **identity,
        "workload": workload.metadata,
        "device": device,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": _git_state(root),
        "source_sha256": _source_manifest(root),
        "protocol": {
            "warmup": args.warmup,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "public_samples": args.public_samples,
            "initial_order": args.benchmark_order,
            "order_reversed_after_each_repeat": True,
            "each_repeat_serialized_by_end_event": True,
            "sessions_co_resident": True,
            "resident_inputs_and_weights": True,
            "resident_query": _resident_input_provenance(query),
            "public_run_boundary": "validation + H2D + compute + D2H + sync",
            "numpy_reference_ms": reference_ms,
            "reference_source": workload.metadata["reference_source"],
            "rtol": RTOL,
            "atol": ATOL,
        },
        "mlir_cublas_ptx": ptx_report,
        "mlir_cuda_nvrtc": nvrtc_report,
        "mlir_tiled_ffn": tiled_report,
        "comparison": _comparison(paired),
        "repeat_samples": paired["repeat_samples"],
        "autotune": autotune_report,
    }
    _render(report, args.output)


if __name__ == "__main__":
    main()
