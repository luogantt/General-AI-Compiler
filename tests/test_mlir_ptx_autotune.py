from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gac.errors import ValidationError
from gac.mlir_ptx_autotune import (
    AutotuneCacheError,
    AutotuneCandidate,
    AutotuneConfig,
    AutotuneError,
    autotune_mlir_ptx_ffn,
    load_autotune_cache,
)


GRAPH_SHA256 = "a" * 64
SOURCE_SHA256 = "b" * 64
REFERENCE = {"output": np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)}


def _candidate(
    backend: str,
    *,
    tile: tuple[int, int, int] = (64, 64, 32),
    block: tuple[int, int, int] = (256, 1, 1),
    precision: str = "fp32",
    fusion: str = "none",
) -> AutotuneCandidate:
    return AutotuneCandidate(
        backend_strategy=backend,
        matmul_tile=tile,
        thread_block=block,
        precision=precision,
        fusion_strategy=fusion,
    )


class _FakeRunner:
    def __init__(self) -> None:
        self.identity = "fake-cuda-context-1"
        self.device = {
            "name": "NVIDIA GeForce RTX 3070",
            "compute_capability": [8, 6],
            "driver_version": 13000,
            "device_uuid": "GPU-test-3070",
        }
        self.outputs: dict[str, object] = {}
        self.timings: dict[str, object] = {}
        self.run_errors: set[str] = set()
        self.benchmark_errors: set[str] = set()
        self.change_session_after_run: set[str] = set()
        self.run_calls: list[str] = []
        self.benchmark_calls: list[tuple[str, int, int, int]] = []

    def session_identity(self) -> str:
        return self.identity

    def describe_device(self):
        return dict(self.device)

    def run(self, candidate: AutotuneCandidate):
        name = candidate.backend_strategy
        self.run_calls.append(name)
        if name in self.run_errors:
            raise RuntimeError("injected launch failure")
        value = self.outputs.get(name, REFERENCE)
        if name in self.change_session_after_run:
            self.identity = "different-cuda-context"
        return value

    def benchmark(
        self,
        candidate: AutotuneCandidate,
        *,
        warmup: int,
        iterations: int,
        repeats: int,
    ):
        name = candidate.backend_strategy
        self.benchmark_calls.append((name, warmup, iterations, repeats))
        if name in self.benchmark_errors:
            raise RuntimeError("injected event timing failure")
        return self.timings.get(name, [1.0] * repeats)


class MlirPtxAutotuneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AutotuneConfig(
            warmup=2, iterations=7, repeats=3, atol=1e-5, rtol=1e-5
        )

    def _tune(self, candidates, runner, **kwargs):
        return autotune_mlir_ptx_ffn(
            candidates,
            runner,
            REFERENCE,
            graph_sha256=GRAPH_SHA256,
            source_sha256=SOURCE_SHA256,
            config=self.config,
            **kwargs,
        )

    def test_candidate_describes_complete_codegen_policy_and_hash_is_stable(self):
        candidate = _candidate(
            "mlir-nvgpu-mma",
            tile=(128, 64, 32),
            block=(128, 2, 1),
            precision="fp16",
            fusion="full-ffn",
        )
        self.assertEqual(
            candidate.to_dict(),
            {
                "backend_strategy": "mlir-nvgpu-mma",
                "matmul_tile": [128, 64, 32],
                "thread_block": [128, 2, 1],
                "precision": "fp16",
                "fusion_strategy": "full-ffn",
            },
        )
        self.assertEqual(
            candidate.sha256,
            AutotuneCandidate.from_dict(candidate.to_dict()).sha256,
        )
        self.assertEqual(len(candidate.sha256), 64)

        invalid = (
            {"backend_strategy": "../../command"},
            {"matmul_tile": (0, 32, 32)},
            {"thread_block": (1024, 2, 1)},
            {"precision": "int8"},
            {"fusion_strategy": "arbitrary"},
        )
        defaults = {
            "backend_strategy": "cublaslt",
            "matmul_tile": (64, 64, 32),
            "thread_block": (256, 1, 1),
            "precision": "fp32",
            "fusion_strategy": "none",
        }
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValidationError):
                AutotuneCandidate(**(defaults | override))

    def test_correctness_gate_precedes_timing_and_median_selects_winner(self):
        median_winner = _candidate("mlir-ptx-tiled")
        low_mean_loser = _candidate("cublaslt")
        wrong = _candidate("mlir-nvgpu-mma")
        runner = _FakeRunner()
        runner.outputs[wrong.backend_strategy] = {
            "output": REFERENCE["output"] + np.float32(1.0)
        }
        runner.timings[median_winner.backend_strategy] = [1.0, 1.1, 100.0]
        runner.timings[low_mean_loser.backend_strategy] = [0.1, 1.2, 1.2]

        result = self._tune([wrong, low_mean_loser, median_winner], runner)

        self.assertEqual(result.selected_candidate, median_winner)
        self.assertEqual(result.median_latency_ms, 1.1)
        self.assertEqual(
            set(runner.run_calls),
            {
                item.backend_strategy
                for item in (wrong, low_mean_loser, median_winner)
            },
        )
        benchmarked = {call[0] for call in runner.benchmark_calls}
        self.assertNotIn(wrong.backend_strategy, benchmarked)
        self.assertEqual(
            {call[1:] for call in runner.benchmark_calls}, {(2, 7, 3)}
        )
        rejected = next(
            item for item in result.evaluations if item.candidate == wrong
        )
        self.assertEqual(rejected.rejection_reason, "incorrect-output")
        self.assertFalse(rejected.eligible)

    def test_invalid_outputs_and_nonfinite_timings_can_never_win(self):
        missing_output = _candidate("missing-output")
        nan_output = _candidate("nan-output")
        nan_timing = _candidate("nan-timing")
        valid = _candidate("valid")
        runner = _FakeRunner()
        runner.outputs[missing_output.backend_strategy] = {"other": np.ones((1,))}
        runner.outputs[nan_output.backend_strategy] = {
            "output": np.full((2, 2), np.nan, dtype=np.float32)
        }
        runner.timings[nan_timing.backend_strategy] = [0.01, float("nan"), 0.01]
        runner.timings[valid.backend_strategy] = [2.1, 2.0, 1.9]

        result = self._tune(
            [missing_output, nan_output, nan_timing, valid], runner
        )
        self.assertEqual(result.selected_candidate, valid)
        reasons = {
            item.candidate.backend_strategy: item.rejection_reason
            for item in result.evaluations
        }
        self.assertEqual(reasons["missing-output"], "invalid-output")
        self.assertEqual(reasons["nan-output"], "invalid-output")
        self.assertEqual(reasons["nan-timing"], "invalid-timing")
        self.assertNotIn("missing-output", {item[0] for item in runner.benchmark_calls})
        self.assertNotIn("nan-output", {item[0] for item in runner.benchmark_calls})

    def test_all_invalid_timings_fail_without_selecting_or_caching(self):
        candidate = _candidate("invalid-zero-timing")
        runner = _FakeRunner()
        runner.timings[candidate.backend_strategy] = [1.0, 0.0, 1.0]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "tuning.json"
            with self.assertRaisesRegex(AutotuneError, "no autotune candidate"):
                self._tune([candidate], runner, cache_path=cache)
            self.assertFalse(cache.exists())

    def test_session_or_device_change_aborts_tuning(self):
        candidate = _candidate("session-switch")
        runner = _FakeRunner()
        runner.change_session_after_run.add(candidate.backend_strategy)
        with self.assertRaisesRegex(AutotuneError, "changed CUDA session"):
            self._tune([candidate], runner)

    def test_cache_is_atomic_round_trips_and_skips_runner_work(self):
        first = _candidate("cublaslt", fusion="bias-relu")
        second = _candidate("mlir-nvgpu-mma", precision="tf32")
        runner = _FakeRunner()
        runner.timings[first.backend_strategy] = [0.22, 0.20, 0.21]
        runner.timings[second.backend_strategy] = [0.35, 0.34, 0.36]

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "nested" / "ffn-autotune.json"
            initial = self._tune([second, first], runner, cache_path=cache)
            self.assertTrue(cache.is_file())
            self.assertFalse(initial.cache_hit)
            self.assertEqual(initial.selected_candidate, first)
            self.assertEqual(
                list(cache.parent.glob(f".{cache.name}.*.tmp")), []
            )

            runner.run_calls.clear()
            runner.benchmark_calls.clear()
            cached = self._tune([first, second], runner, cache_path=cache)
            self.assertTrue(cached.cache_hit)
            self.assertEqual(cached.selected_candidate, first)
            self.assertEqual(cached.device_sha256, initial.device_sha256)
            self.assertEqual(
                cached.candidate_space_sha256,
                initial.candidate_space_sha256,
            )
            self.assertEqual(runner.run_calls, [])
            self.assertEqual(runner.benchmark_calls, [])

            document = json.loads(cache.read_text(encoding="utf-8"))
            self.assertEqual(document["graph_sha256"], GRAPH_SHA256)
            self.assertEqual(document["source_sha256"], SOURCE_SHA256)
            self.assertEqual(document["device"]["device_uuid"], "GPU-test-3070")
            self.assertEqual(len(document["payload_sha256"]), 64)

    def test_cache_rejects_graph_source_device_candidate_and_config_mismatch(self):
        candidate = _candidate("cublaslt")
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            self._tune([candidate], runner, cache_path=cache)
            common = {
                "path": cache,
                "graph_sha256": GRAPH_SHA256,
                "source_sha256": SOURCE_SHA256,
                "device": runner.device,
                "candidates": [candidate],
                "config": self.config,
            }
            mismatches = (
                ("graph", {"graph_sha256": "c" * 64}),
                ("source", {"source_sha256": "d" * 64}),
                ("device", {"device": runner.device | {"device_uuid": "other"}}),
                ("candidate", {"candidates": [_candidate("different")]}),
                ("configuration", {"config": AutotuneConfig(repeats=5)}),
            )
            for label, override in mismatches:
                with self.subTest(label=label), self.assertRaisesRegex(
                    AutotuneCacheError, "does not match"
                ):
                    load_autotune_cache(**(common | override))

    def test_cache_rejects_payload_tampering_duplicate_keys_and_nonfinite_json(self):
        candidate = _candidate("cublaslt")
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            self._tune([candidate], runner, cache_path=cache)
            document = json.loads(cache.read_text(encoding="utf-8"))
            document["selected_candidate"]["backend_strategy"] = "tampered"
            cache.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheError, "payload digest"):
                self._tune([candidate], runner, cache_path=cache)

            cache.write_text('{"kind":"one","kind":"two"}', encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheError, "repeats JSON key"):
                self._tune([candidate], runner, cache_path=cache)

            cache.write_text('{"latency":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(AutotuneCacheError, "invalid number"):
                self._tune([candidate], runner, cache_path=cache)

    def test_runner_exceptions_are_rejected_but_a_valid_candidate_survives(self):
        launch_error = _candidate("launch-error")
        timing_error = _candidate("timing-error")
        valid = _candidate("valid")
        runner = _FakeRunner()
        runner.run_errors.add(launch_error.backend_strategy)
        runner.benchmark_errors.add(timing_error.backend_strategy)
        runner.timings[valid.backend_strategy] = [0.7, 0.8, 0.9]
        result = self._tune([launch_error, timing_error, valid], runner)
        self.assertEqual(result.selected_candidate, valid)
        reasons = {
            item.candidate.backend_strategy: item.rejection_reason
            for item in result.evaluations
        }
        self.assertEqual(reasons["launch-error"], "execution-error")
        self.assertEqual(reasons["timing-error"], "benchmark-error")


if __name__ == "__main__":
    unittest.main()
