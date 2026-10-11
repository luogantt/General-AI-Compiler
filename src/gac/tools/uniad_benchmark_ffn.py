#!/usr/bin/env python3
"""Benchmark the UniAD FFN CUDA/HXP baseline against the optimized MLIR path.

The primary measurement keeps inputs, weights and workspaces on the GPU and
alternates backend order for every repeat.  Public run latency, which includes
input upload and output download, is reported separately.  Results cover one
stage-2 BEV encoder FFN and must not be presented as full UniAD latency.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable

import numpy as np

from gac.uniad import load_bundle


MLIR = "mlir_cuda_cublas_fused"
CUDA = "cuda_hxp_cublas_baseline"
RTOL = 1.0e-4
ATOL = 1.0e-5
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


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


def _source_checkout_root() -> Path | None:
    """Return the GAC checkout root when this module runs from ``src/gac``."""

    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "gac" / "__init__.py"
    ).is_file():
        return candidate
    return None


def _source_manifest(checkout: Path | None) -> dict[str, str]:
    """Hash all measured Python sources when GAC runs from a checkout."""

    if checkout is None:
        return {}
    paths: list[tuple[str, Path]] = []
    package_root = checkout / "src" / "gac"
    for path in sorted(package_root.rglob("*.py")):
        paths.append((path.relative_to(checkout).as_posix(), path))
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths
    }


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    """Return checkout HEAD and tracked-file dirty state when Git can verify them."""

    try:
        head_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    head = head_result.stdout.strip().lower()
    if head_result.returncode != 0 or _GIT_REVISION.fullmatch(head) is None:
        return None, None
    try:
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return head, None
    dirty = None if status_result.returncode != 0 else bool(status_result.stdout.strip())
    return head, dirty


def _installed_gac_distribution() -> dict[str, Any]:
    """Read non-secret installed-package provenance from distribution metadata."""

    result: dict[str, Any] = {
        "distribution_name": "gac-heterogeneous",
        "distribution_version": None,
        "pep610_vcs_commit": None,
    }
    try:
        distribution = importlib_metadata.distribution("gac-heterogeneous")
    except importlib_metadata.PackageNotFoundError:
        return result
    result["distribution_version"] = distribution.version
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return result
    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if isinstance(commit, str):
        normalized = commit.strip().lower()
        if _GIT_REVISION.fullmatch(normalized) is not None:
            result["pep610_vcs_commit"] = normalized
    return result


def _compiler_source_provenance() -> dict[str, Any]:
    """Describe the code measured by this benchmark without a pinned revision."""

    checkout = _source_checkout_root()
    manifest = _source_manifest(checkout)
    if checkout is not None:
        checkout = checkout.resolve()
        actual_revision, tracked_files_dirty = _git_state(checkout)
        gac_source: dict[str, Any] = {
            "mode": "checkout",
            "checkout": str(checkout),
            "revision_detection": "git-head" if actual_revision else "unavailable",
            "actual_revision": actual_revision,
            "tracked_files_dirty": tracked_files_dirty,
            "python_source_hashes_collected": bool(manifest),
            "python_source_hash_count": len(manifest),
        }
    else:
        distribution = dict(_installed_gac_distribution())
        actual_revision = distribution.pop("pep610_vcs_commit")
        gac_source = {
            "mode": "installed-package",
            "module_file": str(Path(__file__).resolve()),
            **distribution,
            "revision_detection": "pep-610" if actual_revision else "unavailable",
            "actual_revision": actual_revision,
            "tracked_files_dirty": None,
            "python_source_hashes_collected": False,
            "python_source_hash_count": 0,
        }
    return {
        "gac_source": gac_source,
        "source_sha256": manifest,
    }


def _correctness(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = difference / np.maximum(np.abs(expected.astype(np.float64)), 1.0e-3)
    result = {
        "allclose": bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_rel_error_with_1e-3_floor": float(relative.max(initial=0.0)),
    }
    if not result["allclose"]:
        raise AssertionError(
            "compiled output failed parity: max_abs=%s max_rel=%s"
            % (result["max_abs_error"], result["max_rel_error_with_1e-3_floor"])
        )
    return result


def _order(repeat: int, initial: str) -> tuple[str, str]:
    first = (MLIR, CUDA) if initial == "mlir-first" else (CUDA, MLIR)
    return first if repeat % 2 == 0 else (first[1], first[0])


def _upload(session: Any, query: np.ndarray) -> None:
    with session._device, session._stream:
        session._inputs["query"].set(query, stream=session._stream)
        session._stream.synchronize()


def _sample(
    session: Any, submit: Callable[[], None], iterations: int
) -> tuple[float, float]:
    cp = session._cp
    with session._device, session._stream:
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_started = time.perf_counter_ns()
        start.record(session._stream)
        for _ in range(iterations):
            submit()
        end.record(session._stream)
        end.synchronize()
        gpu_us = float(cp.cuda.get_elapsed_time(start, end)) * 1000.0 / iterations
        wall_us = (time.perf_counter_ns() - wall_started) / 1000.0 / iterations
    return gpu_us, wall_us


def _cuda_submitter(session: Any) -> Callable[[], None]:
    values = dict(session._constants)
    values.update(session._inputs)
    values.update(session._outputs)
    values.update(session._arena_views)

    def submit() -> None:
        for launch in session.program.launches:
            session._execute_launch(launch, values)

    return submit


def _paired_timing(
    sessions: dict[str, Any],
    submits: dict[str, Callable[[], None]],
    query: np.ndarray,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    initial_order: str,
) -> dict[str, Any]:
    for name in (MLIR, CUDA):
        _upload(sessions[name], query)
    for index in range(warmup):
        for name in _order(index, initial_order):
            with sessions[name]._device, sessions[name]._stream:
                submits[name]()
                sessions[name]._stream.synchronize()

    gpu = {MLIR: [], CUDA: []}
    wall = {MLIR: [], CUDA: []}
    samples: list[dict[str, Any]] = []
    for index in range(repeats):
        current_order = _order(index, initial_order)
        record: dict[str, Any] = {
            "repeat": index + 1,
            "order": list(current_order),
        }
        for name in current_order:
            gpu_us, wall_us = _sample(sessions[name], submits[name], iterations)
            gpu[name].append(gpu_us)
            wall[name].append(wall_us)
            record[name] = {"gpu_us": gpu_us, "synchronized_wall_us": wall_us}
        samples.append(record)
    return {
        "gpu_us": {name: _distribution(gpu[name]) for name in (MLIR, CUDA)},
        "synchronized_wall_us": {
            name: _distribution(wall[name]) for name in (MLIR, CUDA)
        },
        "repeat_samples": samples,
    }


def _comparison(timing: dict[str, Any]) -> dict[str, Any]:
    by_position = {"mlir-first": [], "cuda-first": []}
    differences = {"mlir-first": [], "cuda-first": []}
    for sample in timing["repeat_samples"]:
        mlir_us = float(sample[MLIR]["gpu_us"])
        cuda_us = float(sample[CUDA]["gpu_us"])
        position = "mlir-first" if sample["order"][0] == MLIR else "cuda-first"
        by_position[position].append(cuda_us / mlir_us)
        differences[position].append(mlir_us - cuda_us)
    speedup_distributions = {
        name: _distribution(values) for name, values in by_position.items()
    }
    difference_distributions = {
        name: _distribution(values) for name, values in differences.items()
    }
    balanced = math.sqrt(
        speedup_distributions["mlir-first"]["median"]
        * speedup_distributions["cuda-first"]["median"]
    )
    return {
        "method": "paired-interleaved-order-balanced",
        "mlir_speedup_over_cuda_hxp_cublas": balanced,
        "mlir_minus_cuda_gpu_us": statistics.mean(
            [
                difference_distributions["mlir-first"]["median"],
                difference_distributions["cuda-first"]["median"],
            ]
        ),
        "speedup_by_measurement_position": speedup_distributions,
        "difference_by_measurement_position_us": difference_distributions,
    }


def _public_timing(session: Any, query: np.ndarray, count: int) -> dict[str, Any]:
    session.run({"query": query})
    gpu: list[float] = []
    wall: list[float] = []
    for _ in range(count):
        result = session.run({"query": query})
        gpu.append(float(result.gpu_stream_ms) * 1000.0)
        wall.append(float(result.wall_ms) * 1000.0)
    return {"gpu_us": _distribution(gpu), "wall_us": _distribution(wall)}


def _query_and_reference(loaded: Any, reference_run: Any, graph: Any, seed: int):
    if loaded.sample is None:
        generator = np.random.default_rng(seed)
        query = generator.normal(
            0.0, 0.1, loaded.contract.input_shape
        ).astype(np.float32)
        query = np.ascontiguousarray(query)
    else:
        query = loaded.sample["query"]
    if loaded.sample is None:
        expected = reference_run(graph, {"query": query})["output"]
        source = "GAC NumPy reference"
    else:
        expected = loaded.sample["output"]
        source = "exported UniAD Torch eager golden"
    return query, expected, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=_positive, default=10)
    parser.add_argument("--iterations", type=_positive, default=20)
    parser.add_argument("--repeats", type=_positive, default=10)
    parser.add_argument("--public-samples", type=_positive, default=5)
    parser.add_argument(
        "--initial-order",
        choices=("mlir-first", "cuda-first"),
        default="mlir-first",
    )
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats % 2:
        parser.error("--repeats must be even for order-balanced comparison")

    from gac.backends.cuda import compile_cuda
    from gac.backends.mlir_cuda import compile_mlir_cuda
    from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
    from gac.mlir_runtime import MlirCudaSession
    from gac.package import load_package, rebuild_cuda_program, write_package
    from gac.runtime import CudaSession, reference_run

    loaded = load_bundle(args.bundle)
    profile = loaded.contract
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
        name="uniad_stage2_encoder_layer_%d_ffn" % profile.layer,
    )
    query, expected, reference_source = _query_and_reference(
        loaded, reference_run, graph, args.seed
    )

    compile_started = time.perf_counter()
    mlir_program = compile_mlir_cuda(graph)
    mlir_compile_ms = (time.perf_counter() - compile_started) * 1000.0
    compile_started = time.perf_counter()
    cuda_program = compile_cuda(graph)
    cuda_compile_ms = (time.perf_counter() - compile_started) * 1000.0
    hxp_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="uniad-ffn-hxp-") as directory:
        hxp_path = Path(directory) / "uniad-stage2-encoder-ffn.hxp"
        write_package(
            hxp_path,
            cuda_program,
            metadata={
                "model": "UniAD 2.0",
                "scope": "one stage2 BEV encoder FFN",
                "source_bundle_sha256": hashlib.sha256(
                    args.bundle.read_bytes()
                ).hexdigest(),
            },
        )
        hxp_sha256 = hashlib.sha256(hxp_path.read_bytes()).hexdigest()
        cuda_program = rebuild_cuda_program(load_package(hxp_path))
    hxp_roundtrip_ms = (time.perf_counter() - hxp_started) * 1000.0

    setup_order = _order(0, args.initial_order)
    with ExitStack() as stack:
        sessions: dict[str, Any] = {}
        for name in setup_order:
            if name == MLIR:
                sessions[name] = stack.enter_context(
                    MlirCudaSession(mlir_program, device_id=args.device)
                )
            else:
                sessions[name] = stack.enter_context(
                    CudaSession(
                        cuda_program,
                        device_id=args.device,
                        matmul_backend="cupy",
                    )
                )

        selected = sessions[CUDA].info["matmul_backend"]
        if selected["selected"] != "cupy-cublas" or selected["generated_launch_count"]:
            raise RuntimeError(f"CUDA baseline did not select cuBLAS: {selected}")

        first_results: dict[str, Any] = {}
        parity: dict[str, Any] = {}
        for name in setup_order:
            first_results[name] = sessions[name].run({"query": query})
            parity[name] = _correctness(
                first_results[name].outputs["output"], expected
            )

        submits = {
            MLIR: sessions[MLIR]._execute_calls,
            CUDA: _cuda_submitter(sessions[CUDA]),
        }
        timing = _paired_timing(
            sessions,
            submits,
            query,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            initial_order=args.initial_order,
        )
        public = {
            name: _public_timing(sessions[name], query, args.public_samples)
            for name in setup_order
        }
        device = dict(sessions[MLIR].info)
        setup_ms = {name: float(sessions[name].setup_ms) for name in sessions}

    report = {
        "schema_version": 1,
        "kind": "uniad-ffn-mlir-cuda-paired-benchmark",
        "scope": "one UniAD 2.0 stage2 BEV encoder FFN",
        "boundary": "[1,40000,256] -> 512 -> 256 fragment; not full UniAD inference",
        "profile": profile.to_dict(),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "query": {
            "shape": list(query.shape),
            "dtype": query.dtype.name,
            "sha256": hashlib.sha256(query.tobytes(order="C")).hexdigest(),
            "reference_source": reference_source,
        },
        "device": device,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "compiler_source": _compiler_source_provenance(),
        "protocol": {
            "resident_inputs_weights_workspaces": True,
            "warmup": args.warmup,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "initial_order": args.initial_order,
            "order_reversed_each_repeat": True,
            "serialization": "end CUDA Event synchronized before other backend",
            "public_run_boundary": "validation + H2D + compute + D2H + sync",
        },
        "mlir_cuda": {
            "compile_ms": mlir_compile_ms,
            "setup_ms": setup_ms[MLIR],
            "dispatch_count": len(mlir_program.calls),
            "calls": [call.to_dict() for call in mlir_program.calls],
            "fusion": ["bias+relu", "bias+residual+layernorm"],
            "mlir_source_role": "verifiable ABI manifest",
            "runtime_dispatch_source": "validated call records",
            "mlir_source_executed": False,
            "correctness": parity[MLIR],
            "first_public_run_wall_us": first_results[MLIR].wall_ms * 1000.0,
            "resident": {
                "gpu_us": timing["gpu_us"][MLIR],
                "synchronized_wall_us": timing["synchronized_wall_us"][MLIR],
            },
            "public": public[MLIR],
        },
        "cuda_hxp_cublas_baseline": {
            "compile_ms": cuda_compile_ms,
            "hxp_verified_roundtrip_ms": hxp_roundtrip_ms,
            "hxp_sha256": hxp_sha256,
            "program_source": "verified HXP load + deterministic rebuild",
            "setup_ms": setup_ms[CUDA],
            "dispatch_count": len(cuda_program.launches),
            "matmul_backend": selected,
            "correctness": parity[CUDA],
            "first_public_run_wall_us": first_results[CUDA].wall_ms * 1000.0,
            "resident": {
                "gpu_us": timing["gpu_us"][CUDA],
                "synchronized_wall_us": timing["synchronized_wall_us"][CUDA],
            },
            "public": public[CUDA],
        },
        "comparison": _comparison(timing),
        "repeat_samples": timing["repeat_samples"],
    }
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
