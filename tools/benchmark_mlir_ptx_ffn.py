#!/usr/bin/env python3
"""Benchmark the all-PTX BEVFormer FFN against the cuBLAS hybrid path.

The measured unit is one production-shape FP32 FFN, not end-to-end
BEVFormer.  ``mlir-ptx-sm86`` executes three kernels emitted as GPU Dialect
and lowered by upstream MLIR through NVVM to PTX.  ``mlir-cuda-sm86`` is the
established comparison path with two cuBLAS MatMuls and two fused CUDA kernels.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
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
    _resident_sample,
    _resident_input_provenance,
    _upload_resident_query,
    _warmup_resident_once,
)
from gac.backends.mlir_cuda import compile_mlir_cuda
from gac.backends.mlir_ptx import compile_mlir_ptx
from gac.mlir_ptx_artifact import (
    ARTIFACT_FILES,
    load_mlir_ptx_artifact,
    write_mlir_ptx_artifact,
)
from gac.mlir_ptx_runtime import MlirPtxSession
from gac.mlir_runtime import MlirCudaSession
from gac.runtime import reference_run


PTX_PATH = "mlir_ptx"
HYBRID_PATH = "mlir_cublas_hybrid"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_is_within(path: Path, directory: Path) -> bool:
    candidate = path.expanduser().resolve(strict=False)
    parent = directory.expanduser().resolve(strict=False)
    return candidate == parent or parent in candidate.parents


def _path_aliases_artifact(path: Path, directory: Path) -> bool:
    if _path_is_within(path, directory):
        return True
    members = (*ARTIFACT_FILES.values(), "manifest.json")
    candidate = path.expanduser()
    for name in members:
        member = directory.expanduser() / name
        try:
            if candidate.exists() and member.exists() and candidate.samefile(member):
                return True
        except OSError:
            pass
    return False


def _order(repeat: int, initial: str) -> tuple[str, str]:
    first = (
        (PTX_PATH, HYBRID_PATH)
        if initial == "ptx-first"
        else (HYBRID_PATH, PTX_PATH)
    )
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
    for name in (PTX_PATH, HYBRID_PATH):
        _upload_resident_query(sessions[name], query)
    for index in range(warmup):
        for name in _order(index, initial):
            _warmup_resident_once(sessions[name], submits[name])

    gpu = {PTX_PATH: [], HYBRID_PATH: []}
    wall = {PTX_PATH: [], HYBRID_PATH: []}
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
            sample[name] = {"gpu_us": gpu_us, "synchronized_wall_us": wall_us}
        samples.append(sample)
    return {
        "gpu_us": {name: _distribution(gpu[name]) for name in gpu},
        "synchronized_wall_us": {
            name: _distribution(wall[name]) for name in wall
        },
        "repeat_samples": samples,
    }


def _comparison(paired: dict[str, Any]) -> dict[str, Any]:
    ratios = []
    differences = []
    ratios_by_order = {"ptx-first": [], "hybrid-first": []}
    for sample in paired["repeat_samples"]:
        ptx_us = float(sample[PTX_PATH]["gpu_us"])
        hybrid_us = float(sample[HYBRID_PATH]["gpu_us"])
        ratio = ptx_us / hybrid_us
        key = "ptx-first" if sample["order"][0] == PTX_PATH else "hybrid-first"
        ratios.append(ratio)
        differences.append(ptx_us - hybrid_us)
        ratios_by_order[key].append(ratio)
    by_order = {
        name: _distribution(values) for name, values in ratios_by_order.items()
    }
    balanced = statistics.geometric_mean(
        (by_order["ptx-first"]["median"], by_order["hybrid-first"]["median"])
    )
    return {
        "method": "paired-interleaved-order-balanced",
        "all_ptx_slowdown_vs_cublas_hybrid": balanced,
        "cublas_hybrid_speedup_over_all_ptx": balanced,
        "all_ptx_throughput_relative_to_cublas_hybrid": 1.0 / balanced,
        "all_ptx_minus_cublas_hybrid_us": _distribution(differences),
        "slowdown_samples": _distribution(ratios),
        "slowdown_by_order": by_order,
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
        (root / "tools" / name)
        for name in ("benchmark_mlir_cuda_ffn.py", "benchmark_mlir_ptx_ffn.py")
    )
    return {
        path.relative_to(root).as_posix(): _sha256_bytes(path.read_bytes())
        for path in paths
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=_positive, default=20)
    parser.add_argument("--iterations", type=_positive, default=50)
    parser.add_argument("--repeats", type=_positive, default=20)
    parser.add_argument("--public-samples", type=_positive, default=30)
    parser.add_argument(
        "--benchmark-order",
        choices=("ptx-first", "hybrid-first"),
        default="ptx-first",
    )
    parser.add_argument("--mlir-opt")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--emit-artifact-dir", type=Path)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile and emit the offline artifact without accessing CUDA",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats % 2:
        parser.error("--repeats must be even for order-balanced comparison")
    if args.artifact_dir and args.emit_artifact_dir:
        parser.error("--artifact-dir and --emit-artifact-dir are mutually exclusive")
    if args.compile_only and (args.artifact_dir or not args.emit_artifact_dir):
        parser.error("--compile-only requires --emit-artifact-dir and no --artifact-dir")
    if args.wsl_distribution and not args.wsl:
        parser.error("--wsl-distribution requires --wsl")
    if args.artifact_dir and (args.mlir_opt or args.wsl or args.wsl_distribution):
        parser.error("toolchain options cannot be used with --artifact-dir")
    if args.output and args.emit_artifact_dir and _path_aliases_artifact(
        args.output, args.emit_artifact_dir
    ):
        parser.error("--output must be outside --emit-artifact-dir")
    if args.output and args.artifact_dir and _path_aliases_artifact(
        args.output, args.artifact_dir
    ):
        parser.error("--output must be outside --artifact-dir")

    root = Path(__file__).resolve().parents[1]
    graph, query = _make_graph()

    compile_started = time.perf_counter()
    if args.artifact_dir:
        ptx_program = load_mlir_ptx_artifact(args.artifact_dir, graph)
        preparation = "verified-precompiled-artifact"
    else:
        ptx_program = compile_mlir_ptx(
            graph,
            mlir_opt=args.mlir_opt,
            use_wsl=args.wsl,
            wsl_distribution=args.wsl_distribution,
            timeout=args.timeout,
        )
        preparation = "official-mlir-offline-compilation"
    ptx_compile_ms = (time.perf_counter() - compile_started) * 1000.0
    if args.emit_artifact_dir:
        write_mlir_ptx_artifact(args.emit_artifact_dir, ptx_program)
    if args.compile_only:
        report = {
            "schema_version": 1,
            "kind": "bevformer-ffn-mlir-ptx-offline-compilation",
            "scope": "one FP32 BEVFormer FFN [1,2500,256] -> 512 -> 256",
            "artifact_directory": str(args.emit_artifact_dir.resolve()),
            "compile_ms": ptx_compile_ms,
            "lowering": ptx_program.describe(),
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return

    reference_started = time.perf_counter()
    expected = reference_run(graph, {"query": query})["output"]
    reference_ms = (time.perf_counter() - reference_started) * 1000.0

    compile_started = time.perf_counter()
    hybrid_program = compile_mlir_cuda(graph)
    hybrid_compile_ms = (time.perf_counter() - compile_started) * 1000.0

    setup_order = list(_order(0, args.benchmark_order))
    with ExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in setup_order:
            if name == PTX_PATH:
                sessions[name] = stack.enter_context(
                    MlirPtxSession(ptx_program, device_id=args.device)
                )
            else:
                sessions[name] = stack.enter_context(
                    MlirCudaSession(hybrid_program, device_id=args.device)
                )
        first = {
            name: sessions[name].run({"query": query}) for name in setup_order
        }
        correctness = {
            name: _correctness(first[name].outputs["output"], expected)
            for name in setup_order
        }
        submits = {
            PTX_PATH: sessions[PTX_PATH]._execute_launches,
            HYBRID_PATH: sessions[HYBRID_PATH]._execute_calls,
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
        hybrid_report = _path_report(
            hybrid_program,
            sessions[HYBRID_PATH],
            first[HYBRID_PATH],
            correctness[HYBRID_PATH],
            paired["gpu_us"][HYBRID_PATH],
            paired["synchronized_wall_us"][HYBRID_PATH],
            public[HYBRID_PATH][0],
            public[HYBRID_PATH][1],
        )
        device = dict(sessions[PTX_PATH].info)

    ptx_report["compile_or_load_ms"] = ptx_compile_ms
    ptx_report["preparation"] = preparation
    hybrid_report["compile_ms"] = hybrid_compile_ms
    report = {
        "schema_version": 1,
        "kind": "bevformer-ffn-mlir-all-ptx-vs-cublas-hybrid",
        "scope": "one FP32 BEVFormer FFN [1,2500,256] -> 512 -> 256",
        "boundary": "production-shape FFN fragment; not full BEVFormer inference",
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
            "sessions_co_resident": True,
            "resident_inputs_and_weights": True,
            "resident_query": _resident_input_provenance(query),
            "numpy_reference_ms": reference_ms,
            "rtol": RTOL,
            "atol": ATOL,
        },
        "mlir_all_ptx": ptx_report,
        "mlir_cublas_hybrid": hybrid_report,
        "comparison": _comparison(paired),
        "repeat_samples": paired["repeat_samples"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
