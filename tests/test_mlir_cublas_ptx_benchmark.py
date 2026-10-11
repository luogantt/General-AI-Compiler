from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from threading import Lock
import unittest

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
    def __init__(self, target, output, *, device_id=0):
        self.program = SimpleNamespace(target=target)
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
        assert inputs["query"].shape == (1, 2500, 256)
        return SimpleNamespace(outputs={"output": self.output.copy()})


def _sessions():
    output = np.arange(8, dtype=np.float32).reshape(2, 4)
    return {
        CUBLAS_PTX_TARGET: _Session(CUBLAS_PTX_TARGET, output),
        MLIR_CUDA_TARGET: _Session(MLIR_CUDA_TARGET, output),
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


if __name__ == "__main__":
    unittest.main()
