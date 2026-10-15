#!/usr/bin/env python3
"""Benchmark the executable MLIR/cuBLAS BEVFormer FFN slice on a real SM86 GPU.

The primary comparison is paired and interleaved with one CUDA Event pair
around each device-resident pipeline block.  Public ``run`` latency is reported
separately because it also validates inputs, uploads the query, downloads the
output, and synchronizes.  This is one production-shape FFN fragment; it is not
an end-to-end BEVFormer benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from gac.targets.cuda.compiler import compile_cuda
from gac.targets.cuda.mlir import compile_mlir_cuda
from gac.models.bevformer.ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.execution.mlir_cuda import MlirCudaSession
from gac.execution.cuda import CudaSession, reference_run


SEED = 20261010
RTOL = 1.0e-4
ATOL = 1.0e-5
MLIR_PATH = "mlir_cuda"
CUDA_BASELINE_PATH = "cuda_codegen_cublas"


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(statistics.median(values)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
        "samples": values,
    }


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _source_manifest(root: Path) -> dict[str, str]:
    """Hash the complete Python package and the benchmark driver under test."""

    package_root = root / "src" / "gac"
    paths = [
        path.relative_to(root)
        for path in sorted(package_root.rglob("*.py"))
    ]
    paths.append(Path("tools/benchmark_mlir_cuda_ffn.py"))
    return {
        path.as_posix(): hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
    }


def _make_graph() -> tuple[Any, np.ndarray]:
    config = BEVFormerFFNConfig()
    rng = np.random.default_rng(SEED)
    weights: dict[str, np.ndarray] = {}
    for name, shape in ffn_weight_shapes(config).items():
        if name == "norm_weight":
            value = rng.normal(1.0, 0.02, shape)
        elif name.endswith("weight"):
            value = rng.normal(0.0, 0.02, shape)
        else:
            value = rng.normal(0.0, 0.01, shape)
        weights[name] = value.astype(np.float32)
    query = rng.normal(
        0.0,
        0.1,
        (config.batch, config.queries, config.embed_dim),
    ).astype(np.float32)
    return build_bevformer_ffn_graph(weights, config), query


def _correctness(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = difference / np.maximum(np.abs(expected.astype(np.float64)), 1.0e-3)
    allclose = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
    report = {
        "allclose": allclose,
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_rel_error_with_1e-3_floor": float(relative.max(initial=0.0)),
    }
    if not allclose:
        raise AssertionError(
            "GPU output differs from the NumPy reference: "
            f"max_abs={report['max_abs_error']}, "
            f"max_rel={report['max_rel_error_with_1e-3_floor']}"
        )
    return report


def _resident_input_provenance(query: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(query)
    return {
        "name": "query",
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
        "nbytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _repeat_order(repeat: int, initial_order: str) -> tuple[str, str]:
    if initial_order == "mlir-first":
        first = (MLIR_PATH, CUDA_BASELINE_PATH)
    elif initial_order == "cuda-first":
        first = (CUDA_BASELINE_PATH, MLIR_PATH)
    else:
        raise ValueError(f"unknown initial order {initial_order!r}")
    return first if repeat % 2 == 0 else (first[1], first[0])


def _resident_sample(
    session: Any,
    submit: Callable[[], None],
    *,
    iterations: int,
) -> tuple[float, float]:
    cp = session._cp
    with session._device, session._stream:
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        start.record(session._stream)
        for _ in range(iterations):
            submit()
        end.record(session._stream)
        end.synchronize()
        gpu_us = float(cp.cuda.get_elapsed_time(start, end)) * 1000.0 / iterations
        wall_us = (time.perf_counter_ns() - wall_start) / 1000.0 / iterations
    return gpu_us, wall_us


def _upload_resident_query(session: Any, query: np.ndarray) -> None:
    with session._device, session._stream:
        session._inputs["query"].set(query, stream=session._stream)
        session._stream.synchronize()


def _warmup_resident_once(
    session: Any,
    submit: Callable[[], None],
) -> None:
    with session._device, session._stream:
        submit()
        session._stream.synchronize()


def _resident_timing(
    session: Any,
    query: np.ndarray,
    submit: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu_us: list[float] = []
    synchronized_wall_us: list[float] = []
    _upload_resident_query(session, query)
    with session._device, session._stream:
        for _ in range(warmup):
            submit()
        session._stream.synchronize()

    for _ in range(repeats):
        gpu_sample, wall_sample = _resident_sample(
            session,
            submit,
            iterations=iterations,
        )
        gpu_us.append(gpu_sample)
        synchronized_wall_us.append(wall_sample)
    return _distribution(gpu_us), _distribution(synchronized_wall_us)


def _paired_resident_timing(
    sessions: dict[str, Any],
    submits: dict[str, Callable[[], None]],
    query: np.ndarray,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    initial_order: str,
) -> dict[str, Any]:
    for name in (MLIR_PATH, CUDA_BASELINE_PATH):
        _upload_resident_query(sessions[name], query)

    # Warm both paths in the same alternating, serialized schedule as the timed
    # repeats.  Synchronizing every warmup invocation prevents the two streams
    # from overlapping and accidentally measuring concurrent execution.
    for warmup_repeat in range(warmup):
        for name in _repeat_order(warmup_repeat, initial_order):
            _warmup_resident_once(sessions[name], submits[name])

    gpu_samples = {MLIR_PATH: [], CUDA_BASELINE_PATH: []}
    wall_samples = {MLIR_PATH: [], CUDA_BASELINE_PATH: []}
    repeat_samples: list[dict[str, Any]] = []
    for repeat in range(repeats):
        order = _repeat_order(repeat, initial_order)
        sample: dict[str, Any] = {
            "repeat": repeat + 1,
            "order": list(order),
        }
        for name in order:
            gpu_us, wall_us = _resident_sample(
                sessions[name],
                submits[name],
                iterations=iterations,
            )
            gpu_samples[name].append(gpu_us)
            wall_samples[name].append(wall_us)
            sample[name] = {
                "gpu_us": gpu_us,
                "synchronized_wall_us": wall_us,
            }
        repeat_samples.append(sample)

    return {
        "gpu_us": {
            name: _distribution(gpu_samples[name])
            for name in (MLIR_PATH, CUDA_BASELINE_PATH)
        },
        "synchronized_wall_us": {
            name: _distribution(wall_samples[name])
            for name in (MLIR_PATH, CUDA_BASELINE_PATH)
        },
        "repeat_samples": repeat_samples,
    }


def _paired_comparison(paired: dict[str, Any]) -> dict[str, Any]:
    speedups: list[float] = []
    differences: list[float] = []
    speedups_by_order: dict[str, list[float]] = {
        "mlir-first": [],
        "cuda-first": [],
    }
    differences_by_order: dict[str, list[float]] = {
        "mlir-first": [],
        "cuda-first": [],
    }
    for sample in paired["repeat_samples"]:
        mlir_us = float(sample[MLIR_PATH]["gpu_us"])
        cuda_us = float(sample[CUDA_BASELINE_PATH]["gpu_us"])
        if mlir_us <= 0.0 or cuda_us <= 0.0:
            raise ValueError("CUDA Event timings must be positive")
        speedup = cuda_us / mlir_us
        difference = mlir_us - cuda_us
        order = "mlir-first" if sample["order"][0] == MLIR_PATH else "cuda-first"
        speedups.append(speedup)
        differences.append(difference)
        speedups_by_order[order].append(speedup)
        differences_by_order[order].append(difference)

    if not all(speedups_by_order.values()):
        raise ValueError("paired comparison requires both measurement orders")
    speedup_by_order = {
        name: _distribution(values) for name, values in speedups_by_order.items()
    }
    difference_by_order = {
        name: _distribution(values)
        for name, values in differences_by_order.items()
    }
    order_balanced_speedup = math.sqrt(
        speedup_by_order["mlir-first"]["median"]
        * speedup_by_order["cuda-first"]["median"]
    )
    order_balanced_difference = statistics.mean(
        (
            difference_by_order["mlir-first"]["median"],
            difference_by_order["cuda-first"]["median"],
        )
    )
    mlir_median = paired["gpu_us"][MLIR_PATH]["median"]
    cuda_median = paired["gpu_us"][CUDA_BASELINE_PATH]["median"]
    return {
        "method": "paired-interleaved",
        "primary_estimator": (
            "geometric mean of the MLIR-first and CUDA-first median per-repeat "
            "CUDA/MLIR Event-time ratios"
        ),
        "mlir_speedup_over_cuda_codegen_cublas": order_balanced_speedup,
        "mlir_minus_cuda_codegen_cublas_us": order_balanced_difference,
        "ratio_of_backend_medians": cuda_median / mlir_median,
        "paired_repeat_speedup": _distribution(speedups),
        "paired_repeat_mlir_minus_cuda_us": _distribution(differences),
        "paired_repeat_speedup_by_order": speedup_by_order,
        "paired_repeat_mlir_minus_cuda_us_by_order": difference_by_order,
        "repeat_samples": paired["repeat_samples"],
    }


def _public_run_timing(
    session: Any,
    query: np.ndarray,
    *,
    samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(min(5, samples)):
        session.run({"query": query})
    gpu_us: list[float] = []
    wall_us: list[float] = []
    for _ in range(samples):
        result = session.run({"query": query})
        # The end Event is recorded after input upload and execution.  Output
        # download and its wait are part of wall_us instead.
        gpu_us.append(float(result.gpu_stream_ms) * 1000.0)
        wall_us.append(float(result.wall_ms) * 1000.0)
    return _distribution(gpu_us), _distribution(wall_us)


def _mlir_report(
    program: Any,
    description: dict[str, Any],
    compile_ms: float,
    session: Any,
    first: Any,
    correctness: dict[str, Any],
    resident_gpu: dict[str, Any],
    resident_wall: dict[str, Any],
    public_gpu: dict[str, Any],
    public_wall: dict[str, Any],
) -> dict[str, Any]:
    return {
        "compiler_backend": "mlir-cuda-sm86",
        "compile_ms": compile_ms,
        "session_setup_ms": float(session.setup_ms),
        "call_count": len(program.calls),
        "calls": [call.to_dict() for call in program.calls],
        "artifact_integrity": description.get("integrity"),
        "precision_policy": description["precision"],
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


def _cuda_submitter(session: Any) -> Callable[[], None]:
    values = dict(session._constants)
    values.update(session._inputs)
    values.update(session._outputs)
    values.update(session._arena_views)

    def submit() -> None:
        for launch in session.program.launches:
            session._execute_launch(launch, values)

    return submit


def _cuda_report(
    program: Any,
    compile_ms: float,
    session: Any,
    first: Any,
    correctness: dict[str, Any],
    resident_gpu: dict[str, Any],
    resident_wall: dict[str, Any],
    public_gpu: dict[str, Any],
    public_wall: dict[str, Any],
) -> dict[str, Any]:
    return {
        "compiler_backend": "cuda-sm86-codegen",
        "artifact_boundary": (
            "compiled in memory from the same graph; HXP package I/O is outside "
            "the timed execution boundary"
        ),
        "generated_cuda_source_sha256": hashlib.sha256(
            program.source.encode("utf-8")
        ).hexdigest(),
        "matmul_backend": dict(session.info["matmul_backend"]),
        "compile_ms": compile_ms,
        "session_setup_ms": float(session.setup_ms),
        "dispatch_count": len(program.launches),
        "implementations": [entry["implementation"] for entry in session.kernel_info],
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


def _benchmark_mlir(
    graph: Any,
    query: np.ndarray,
    expected: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compile_started = time.perf_counter()
    program = compile_mlir_cuda(graph)
    description = program.describe()
    compile_ms = (time.perf_counter() - compile_started) * 1000.0
    with MlirCudaSession(program, device_id=args.device) as session:
        first = session.run({"query": query})
        correctness = _correctness(first.outputs["output"], expected)
        resident_gpu, resident_wall = _resident_timing(
            session,
            query,
            session._execute_calls,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        public_gpu, public_wall = _public_run_timing(
            session,
            query,
            samples=args.public_samples,
        )
        result = _mlir_report(
            program,
            description,
            compile_ms,
            session,
            first,
            correctness,
            resident_gpu,
            resident_wall,
            public_gpu,
            public_wall,
        )
        device = dict(session.info)
    return result, device


def _benchmark_paired(
    graph: Any,
    query: np.ndarray,
    expected: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    compile_started = time.perf_counter()
    mlir_program = compile_mlir_cuda(graph)
    mlir_description = mlir_program.describe()
    mlir_compile_ms = (time.perf_counter() - compile_started) * 1000.0
    compile_started = time.perf_counter()
    cuda_program = compile_cuda(graph)
    cuda_compile_ms = (time.perf_counter() - compile_started) * 1000.0

    setup_order = list(_repeat_order(0, args.benchmark_order))
    with ExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in setup_order:
            if name == MLIR_PATH:
                sessions[name] = stack.enter_context(
                    MlirCudaSession(mlir_program, device_id=args.device)
                )
            else:
                sessions[name] = stack.enter_context(
                    CudaSession(
                        cuda_program,
                        device_id=args.device,
                        matmul_backend="auto",
                    )
                )

        cuda_matmul = sessions[CUDA_BASELINE_PATH].info["matmul_backend"]
        if (
            cuda_matmul["selected"] != "cupy-cublas"
            or cuda_matmul["generated_launch_count"] != 0
        ):
            raise RuntimeError(
                "paired CUDA baseline requires every MatMul to use cupy-cublas, "
                f"got {cuda_matmul}"
            )

        first: dict[str, Any] = {}
        correctness: dict[str, Any] = {}
        for name in setup_order:
            first[name] = sessions[name].run({"query": query})
            correctness[name] = _correctness(
                first[name].outputs["output"],
                expected,
            )

        submits = {
            MLIR_PATH: sessions[MLIR_PATH]._execute_calls,
            CUDA_BASELINE_PATH: _cuda_submitter(sessions[CUDA_BASELINE_PATH]),
        }
        paired = _paired_resident_timing(
            sessions,
            submits,
            query,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            initial_order=args.benchmark_order,
        )

        # Public run samples intentionally remain outside the paired resident
        # protocol because they include validation, transfers, and a D2H wait.
        public: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for name in setup_order:
            public[name] = _public_run_timing(
                sessions[name],
                query,
                samples=args.public_samples,
            )

        mlir = _mlir_report(
            mlir_program,
            mlir_description,
            mlir_compile_ms,
            sessions[MLIR_PATH],
            first[MLIR_PATH],
            correctness[MLIR_PATH],
            paired["gpu_us"][MLIR_PATH],
            paired["synchronized_wall_us"][MLIR_PATH],
            public[MLIR_PATH][0],
            public[MLIR_PATH][1],
        )
        baseline = _cuda_report(
            cuda_program,
            cuda_compile_ms,
            sessions[CUDA_BASELINE_PATH],
            first[CUDA_BASELINE_PATH],
            correctness[CUDA_BASELINE_PATH],
            paired["gpu_us"][CUDA_BASELINE_PATH],
            paired["synchronized_wall_us"][CUDA_BASELINE_PATH],
            public[CUDA_BASELINE_PATH][0],
            public[CUDA_BASELINE_PATH][1],
        )
        device = dict(sessions[MLIR_PATH].info)

    protocol = {
        "comparison_mode": "paired-interleaved",
        "compile_order": [MLIR_PATH, CUDA_BASELINE_PATH],
        "session_setup_order": setup_order,
        "first_public_run_order": setup_order,
        "public_run_measurement_order": setup_order,
        "sessions_co_resident_during_resident_timing": True,
        "resident_repeat_order": (
            "initial order from --benchmark-order, reversed after every repeat"
        ),
        "resident_repeat_serialization": (
            "each path's end CUDA Event is synchronized before the other path starts"
        ),
    }
    return mlir, baseline, device, _paired_comparison(paired), protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=_positive, default=50)
    parser.add_argument("--iterations", type=_positive, default=100)
    parser.add_argument("--repeats", type=_positive, default=20)
    parser.add_argument("--public-samples", type=_positive, default=50)
    parser.add_argument(
        "--benchmark-order",
        choices=("mlir-first", "cuda-first"),
        default="mlir-first",
        help="first path in the paired schedule; every repeat reverses it",
    )
    parser.add_argument("--skip-cuda-baseline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.skip_cuda_baseline and args.repeats % 2 != 0:
        parser.error(
            "paired comparison requires an even --repeats value so both paths "
            "occupy each measurement position equally"
        )

    root = Path(__file__).resolve().parents[1]
    graph, query = _make_graph()
    reference_started = time.perf_counter()
    expected = reference_run(graph, {"query": query})["output"]
    reference_ms = (time.perf_counter() - reference_started) * 1000.0

    baseline: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    if args.skip_cuda_baseline:
        mlir, device = _benchmark_mlir(graph, query, expected, args)
        paired_protocol = {
            "comparison_mode": "mlir-only",
            "sessions_co_resident_during_resident_timing": False,
        }
    else:
        mlir, baseline, device, comparison, paired_protocol = _benchmark_paired(
            graph,
            query,
            expected,
            args,
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "kind": "mlir-cuda-cublas-fused-ffn-benchmark",
        "scope": (
            "one BEVFormer FFN [1,2500,256] -> 512 -> 256 with ReLU, "
            "query residual and LayerNorm"
        ),
        "boundary": "production-shape FFN fragment; not full BEVFormer inference",
        "precision": (
            "float32 storage; CuPy/cuBLAS compute policy is runtime-default and "
            "TF32 is not forced or disabled by this compiler"
        ),
        "seed": SEED,
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "git": _git_state(root),
        "source_sha256": _source_manifest(root),
        "device": device,
        "protocol": {
            "warmup": args.warmup,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "public_run_samples": args.public_samples,
            "benchmark_order": args.benchmark_order,
            "resident_gpu_clock": (
                "one CUDA Event pair around each backend's whole pipeline repeat"
            ),
            "resident_inputs_and_weights": True,
            "resident_query": _resident_input_provenance(query),
            "resident_query_binding": (
                "the same host query is copied once into each session's distinct "
                "resident device input buffer before warmup"
            ),
            "public_run_boundary": (
                "validation + H2D query + compute + D2H output + sync"
            ),
            "public_run_timing": "separate from paired resident timing",
            "numpy_reference_ms": reference_ms,
            **paired_protocol,
        },
        "mlir_cuda": mlir,
    }
    if baseline is not None:
        report["cuda_codegen_cublas_baseline"] = baseline
        assert comparison is not None
        report["comparison"] = comparison

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
