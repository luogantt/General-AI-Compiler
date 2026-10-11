from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from threading import Lock
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.mlir_cublas_ptx import TARGET as CUBLAS_PTX_TARGET
from gac.backends.mlir_cuda import TARGET as MLIR_CUDA_TARGET
from gac.errors import BackendError, ValidationError
from gac.mlir_cublas_ptx_benchmark import (
    FfnSessionAutotuneRunner,
    UnsupportedFfnCandidate,
    executable_ffn_candidates,
    skipped_tiled_candidate_report,
    unwired_tiled_candidates,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import benchmark_mlir_cublas_ptx_ffn as benchmark_driver  # noqa: E402


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _Stream(_Context):
    def __init__(self):
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class _Input:
    def __init__(self):
        self.uploads = []

    def set(self, value, *, stream):
        self.uploads.append((np.array(value, copy=True), stream))


class _Event:
    def __init__(self):
        self.records = []
        self.synchronizations = 0

    def record(self, stream):
        self.records.append(stream)

    def synchronize(self):
        self.synchronizations += 1


class _Cupy:
    def __init__(self, elapsed_ms=4.0):
        self.events = []
        self.elapsed_ms = elapsed_ms

        def event():
            value = _Event()
            self.events.append(value)
            return value

        self.cuda = SimpleNamespace(
            Event=event,
            get_elapsed_time=lambda _start, _end: self.elapsed_ms,
        )


class _Session:
    def __init__(self, target, output, *, device_id=0, query_shape=(1, 2500, 256)):
        self.query_shape = tuple(query_shape)
        self.program = SimpleNamespace(
            target=target,
            graph=SimpleNamespace(
                inputs={
                    "query": SimpleNamespace(
                        shape=self.query_shape,
                        dtype="float32",
                        layout="contiguous",
                    )
                }
            ),
        )
        self._cp = _Cupy()
        self._device = _Context()
        self._stream = _Stream()
        self._inputs = {"query": _Input()}
        self._lock = Lock()
        self._closed = False
        self._failed = False
        self.info = {
            "device_id": device_id,
            "name": "fake RTX 3070",
            "compute_capability": [8, 6],
            "driver_version": 13000,
        }
        self.output = np.asarray(output, dtype=np.float32)
        self.submit_count = 0
        self.run_count = 0

    def _execute_calls(self):
        self.submit_count += 1

    def run(self, inputs):
        self.run_count += 1
        assert inputs["query"].shape == self.query_shape
        return SimpleNamespace(outputs={"output": self.output.copy()})


def _sessions(query_shape=(1, 2500, 256)):
    output = np.arange(8, dtype=np.float32).reshape(2, 4)
    return {
        CUBLAS_PTX_TARGET: _Session(
            CUBLAS_PTX_TARGET, output, query_shape=query_shape
        ),
        MLIR_CUDA_TARGET: _Session(
            MLIR_CUDA_TARGET, output, query_shape=query_shape
        ),
    }


class FfnSessionAutotuneRunnerTests(unittest.TestCase):
    def test_candidates_execute_real_mapped_sessions(self):
        sessions = _sessions()
        runner = FfnSessionAutotuneRunner(
            np.zeros((1, 2500, 256), dtype=np.float32), sessions
        )
        candidates = executable_ffn_candidates()
        self.assertEqual(
            {candidate.backend_strategy for candidate in candidates},
            {CUBLAS_PTX_TARGET, MLIR_CUDA_TARGET},
        )
        for candidate in candidates:
            outputs = runner.run(candidate)
            self.assertEqual(outputs["output"].shape, (2, 4))
            samples = runner.benchmark(
                candidate, warmup=2, iterations=4, repeats=3
            )
            self.assertEqual(samples, (1.0, 1.0, 1.0))
            session = sessions[candidate.backend_strategy]
            self.assertEqual(session.run_count, 1)
            self.assertEqual(session.submit_count, 14)
            self.assertEqual(len(session._cp.events), 6)
            self.assertGreaterEqual(len(session._inputs["query"].uploads), 2)

    def test_tile_and_tensor_core_metadata_are_explicitly_skipped(self):
        sessions = _sessions()
        runner = FfnSessionAutotuneRunner(
            np.zeros((1, 2500, 256), dtype=np.float32), sessions
        )
        candidates = unwired_tiled_candidates()
        for candidate in candidates:
            self.assertIn("no executable", runner.unsupported_reason(candidate))
            with self.assertRaises(UnsupportedFfnCandidate):
                runner.run(candidate)
        report = skipped_tiled_candidate_report()
        self.assertEqual([entry["status"] for entry in report], ["skipped", "skipped"])
        self.assertTrue(
            all("not executed or timed" in entry["reason"] for entry in report)
        )
        self.assertEqual(sum(session.submit_count for session in sessions.values()), 0)

    def test_same_device_and_exact_query_are_required(self):
        sessions = _sessions()
        sessions[MLIR_CUDA_TARGET].info["device_id"] = 1
        with self.assertRaisesRegex(ValidationError, "same CUDA device"):
            FfnSessionAutotuneRunner(
                np.zeros((1, 2500, 256), dtype=np.float32), sessions
            )
        with self.assertRaisesRegex(ValidationError, r"\[1,2500,256\]"):
            FfnSessionAutotuneRunner(
                np.zeros((2, 256), dtype=np.float32), _sessions()
            )

        sessions = _sessions()
        sessions[MLIR_CUDA_TARGET] = _Session(
            MLIR_CUDA_TARGET,
            np.zeros((2, 4), dtype=np.float32),
            query_shape=(1, 40000, 256),
        )
        with self.assertRaisesRegex(ValidationError, "same FFN query profile"):
            FfnSessionAutotuneRunner(
                np.zeros((1, 2500, 256), dtype=np.float32), sessions
            )

    def test_uniad_production_query_profile_is_supported(self):
        shape = (1, 40000, 256)
        sessions = _sessions(shape)
        runner = FfnSessionAutotuneRunner(
            np.zeros(shape, dtype=np.float32), sessions
        )
        for candidate in executable_ffn_candidates():
            self.assertEqual(runner.run(candidate)["output"].shape, (2, 4))

    def test_initial_upload_obeys_session_lock_and_lifecycle(self):
        sessions = _sessions()
        locked = sessions[CUBLAS_PTX_TARGET]
        self.assertTrue(locked._lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(BackendError, "concurrent autotune"):
                FfnSessionAutotuneRunner(
                    np.zeros((1, 2500, 256), dtype=np.float32), sessions
                )
        finally:
            locked._lock.release()

        sessions = _sessions()
        sessions[CUBLAS_PTX_TARGET]._closed = True
        with self.assertRaisesRegex(BackendError, "closed or failed"):
            FfnSessionAutotuneRunner(
                np.zeros((1, 2500, 256), dtype=np.float32), sessions
            )
        self.assertTrue(
            sessions[CUBLAS_PTX_TARGET]._lock.acquire(blocking=False)
        )
        sessions[CUBLAS_PTX_TARGET]._lock.release()

    def test_candidate_fields_cannot_be_metadata_only_overrides(self):
        sessions = _sessions()
        runner = FfnSessionAutotuneRunner(
            np.zeros((1, 2500, 256), dtype=np.float32), sessions
        )
        base = executable_ffn_candidates()[0]
        from dataclasses import replace

        tiled = replace(base, matmul_tile=(64, 64, 32))
        self.assertIn("not wired", runner.unsupported_reason(tiled))
        with self.assertRaisesRegex(UnsupportedFfnCandidate, "not wired"):
            runner.benchmark(tiled, warmup=0, iterations=1, repeats=1)
        self.assertEqual(sessions[CUBLAS_PTX_TARGET].submit_count, 0)

    def test_benchmark_obeys_session_lock_and_lifecycle(self):
        sessions = _sessions()
        runner = FfnSessionAutotuneRunner(
            np.zeros((1, 2500, 256), dtype=np.float32), sessions
        )
        candidate = executable_ffn_candidates()[0]
        session = sessions[candidate.backend_strategy]
        self.assertTrue(session._lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(BackendError, "concurrent benchmark"):
                runner.benchmark(candidate, warmup=0, iterations=1, repeats=1)
        finally:
            session._lock.release()
        session._closed = True
        with self.assertRaisesRegex(BackendError, "closed or failed"):
            runner.benchmark(candidate, warmup=0, iterations=1, repeats=1)
        self.assertTrue(session._lock.acquire(blocking=False))
        session._lock.release()


class MlirCublasPtxBenchmarkCliTests(unittest.TestCase):
    @staticmethod
    def _run(*arguments: str):
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        source = str(root / "src")
        environment["PYTHONPATH"] = (
            source
            if not environment.get("PYTHONPATH")
            else source + os.pathsep + environment["PYTHONPATH"]
        )
        return subprocess.run(
            [
                sys.executable,
                str(root / "tools" / "benchmark_mlir_cublas_ptx_ffn.py"),
                *arguments,
            ],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_output_cannot_be_hardlink_alias_of_autotune_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "autotune.json"
            cache.write_text("{}\n", encoding="utf-8")
            output = root / "report.json"
            try:
                output.hardlink_to(cache)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")
            completed = self._run(
                "--autotune",
                "--autotune-cache",
                str(cache),
                "--output",
                str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("must name different files", completed.stderr)
            self.assertEqual(cache.read_text(encoding="utf-8"), "{}\n")

    def test_report_cannot_be_hardlink_alias_of_artifact_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            member = artifact / "fusions.ptx"
            member.write_text("device code\n", encoding="utf-8")
            output = root / "report.json"
            try:
                output.hardlink_to(member)
            except OSError as exc:
                self.skipTest(f"hard links are unavailable: {exc}")
            completed = self._run(
                "--artifact-dir",
                str(artifact),
                "--output",
                str(output),
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("outside the artifact directory", completed.stderr)
            self.assertEqual(member.read_text(encoding="utf-8"), "device code\n")

    def test_cli_exposes_uniad_bundle_and_seed(self):
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--uniad-bundle", completed.stdout)
        self.assertIn("--seed", completed.stdout)

    def test_missing_uniad_bundle_is_rejected_before_compilation(self):
        completed = self._run("--uniad-bundle", "missing.uniadffn")
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--uniad-bundle does not exist", completed.stderr)

    def test_autotune_cache_cannot_alias_uniad_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "layer0.uniadffn"
            bundle.write_bytes(b"bundle-placeholder")
            completed = self._run(
                "--uniad-bundle",
                str(bundle),
                "--autotune",
                "--autotune-cache",
                str(bundle),
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("must not replace --uniad-bundle", completed.stderr)

    def test_default_workload_keeps_the_existing_bev_graph_and_query(self):
        expected_graph, expected_query = benchmark_driver._make_graph()
        workload = benchmark_driver._prepare_workload(None, seed=17)
        self.assertEqual(workload.family, "bevformer")
        self.assertEqual(
            workload.graph.to_dict(include_constant_data=False),
            expected_graph.to_dict(include_constant_data=False),
        )
        for name in expected_graph.constants:
            np.testing.assert_array_equal(
                workload.graph.constants[name], expected_graph.constants[name]
            )
        np.testing.assert_array_equal(workload.query, expected_query)
        self.assertIsNone(workload.expected)
        identity = benchmark_driver._report_identity(workload, compile_only=False)
        self.assertEqual(
            identity["kind"],
            "bevformer-ffn-mlir-cublas-ptx-vs-mlir-cuda-nvrtc",
        )
        self.assertEqual(
            identity["scope"],
            "one FP32 BEVFormer FFN [1,2500,256] -> 512 -> 256",
        )

    def test_uniad_workload_uses_bundle_expected_and_distinct_report_contract(self):
        class _Profile:
            batch = 1
            bev_h = 200
            bev_w = 200
            embed_dim = 256
            hidden_dim = 512
            layer_norm_epsilon = 1.0e-5
            layer = 3
            dtype = "float32"
            input_shape = (1, 40000, 256)
            output_shape = input_shape

            @staticmethod
            def to_dict():
                return {"layer": 3, "bev_h": 200, "bev_w": 200}

        query = np.zeros((1, 2, 4), dtype=np.float32)
        output = np.ones((1, 2, 4), dtype=np.float32)
        loaded = SimpleNamespace(
            contract=_Profile(),
            weights={},
            sample={"query": query, "output": output},
            manifest={"format": "uniad-compiler-ffn-bundle", "schema_version": 1},
        )
        graph = SimpleNamespace(name="uniad_stage2_encoder_layer_3_ffn")
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "layer3.uniadffn"
            bundle.write_bytes(b"verified-test-bundle")
            with (
                patch.object(benchmark_driver, "load_bundle", return_value=loaded),
                patch.object(
                    benchmark_driver,
                    "build_bevformer_ffn_graph",
                    return_value=graph,
                ) as build_graph,
            ):
                workload = benchmark_driver._prepare_workload(bundle, seed=29)

        self.assertEqual(workload.family, "uniad")
        self.assertIs(workload.graph, graph)
        np.testing.assert_array_equal(workload.query, query)
        np.testing.assert_array_equal(workload.expected, output)
        self.assertEqual(
            workload.metadata["reference_source"],
            "bundle-provided expected output",
        )
        self.assertIsNone(workload.metadata["seed"])
        self.assertTrue(
            workload.metadata["source_bundle"]["contains_expected_sample"]
        )
        config = build_graph.call_args.args[1]
        self.assertEqual((config.bev_h, config.bev_w), (200, 200))
        identity = benchmark_driver._report_identity(workload, compile_only=False)
        self.assertTrue(identity["kind"].startswith("uniad-stage2-encoder-ffn"))
        self.assertIn("layer 3", identity["scope"])
        self.assertIn("not full UniAD inference", identity["boundary"])

    def test_uniad_workload_without_golden_uses_seeded_numpy_reference(self):
        profile = SimpleNamespace(
            batch=1,
            bev_h=200,
            bev_w=200,
            embed_dim=256,
            hidden_dim=512,
            layer_norm_epsilon=1.0e-5,
            layer=1,
            dtype="float32",
            input_shape=(1, 40000, 256),
            output_shape=(1, 40000, 256),
            to_dict=lambda: {"layer": 1},
        )
        loaded = SimpleNamespace(
            contract=profile,
            weights={},
            sample=None,
            manifest={"format": "uniad-compiler-ffn-bundle", "schema_version": 1},
        )
        generated = np.full((1, 2, 4), 0.25, dtype=np.float32)
        generator = SimpleNamespace(normal=lambda *_args, **_kwargs: generated)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "layer1.uniadffn"
            bundle.write_bytes(b"verified-test-bundle")
            with (
                patch.object(benchmark_driver, "load_bundle", return_value=loaded),
                patch.object(
                    benchmark_driver,
                    "build_bevformer_ffn_graph",
                    return_value=SimpleNamespace(name="layer1"),
                ),
                patch.object(
                    benchmark_driver.np.random,
                    "default_rng",
                    return_value=generator,
                ) as default_rng,
            ):
                workload = benchmark_driver._prepare_workload(bundle, seed=73)

        default_rng.assert_called_once_with(73)
        np.testing.assert_array_equal(workload.query, generated)
        self.assertIsNone(workload.expected)
        self.assertEqual(workload.metadata["seed"], 73)
        self.assertEqual(
            workload.metadata["reference_source"], "GAC NumPy reference"
        )
        self.assertFalse(
            workload.metadata["source_bundle"]["contains_expected_sample"]
        )

    def test_uniad_workload_rejects_nonproduction_grid_or_epsilon(self):
        bundle = Path("profile-check.uniadffn")
        for profile in (
            SimpleNamespace(
                batch=1,
                bev_h=100,
                bev_w=400,
                embed_dim=256,
                hidden_dim=512,
                layer_norm_epsilon=1.0e-5,
                dtype="float32",
            ),
            SimpleNamespace(
                batch=1,
                bev_h=200,
                bev_w=200,
                embed_dim=256,
                hidden_dim=512,
                layer_norm_epsilon=1.0e-3,
                dtype="float32",
            ),
        ):
            loaded = SimpleNamespace(contract=profile)
            with (
                self.subTest(profile=vars(profile)),
                patch.object(benchmark_driver, "load_bundle", return_value=loaded),
                patch.object(Path, "resolve", return_value=bundle),
                self.assertRaisesRegex(ValueError, "production UniAD FP32 profile"),
            ):
                benchmark_driver._prepare_workload(bundle, seed=1)


if __name__ == "__main__":
    unittest.main()
