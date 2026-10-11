"""Unit tests for the dependency-light UniAD end-to-end benchmark helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


from gac.tools import uniad_benchmark_e2e as benchmark


class IntegerArgumentTests(unittest.TestCase):
    def test_integer_types_return_integers(self):
        self.assertEqual(benchmark._positive_int("7"), 7)
        self.assertEqual(benchmark._nonnegative_int("0"), 0)
        self.assertIsInstance(benchmark._positive_int("3"), int)

    def test_invalid_integer_values_are_rejected(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark._positive_int(value)
        for value in ("-1", "1.5", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark._nonnegative_int(value)

    def test_parser_keeps_warmup_measure_and_workers_numeric(self):
        args = benchmark.build_parser().parse_args(
            [
                "--uniad-root",
                "uniad",
                "config.py",
                "model.pth",
                "--output",
                "report.json",
                "--warmup",
                "5",
                "--measure",
                "20",
                "--workers",
                "0",
            ]
        )
        self.assertEqual((args.warmup, args.measure, args.workers), (5, 20, 0))
        self.assertTrue(
            all(
                isinstance(value, int)
                for value in (args.warmup, args.measure, args.workers)
            )
        )

    def test_parser_requires_explicit_uniad_root(self):
        with self.assertRaises(SystemExit):
            benchmark.build_parser().parse_args(
                ["config.py", "model.pth", "--output", "report.json"]
            )


class ExternalUniADRootTests(unittest.TestCase):
    def test_validation_accepts_external_checkout_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "UniAD-2.0"
            plugin = root / "projects" / "mmdet3d_plugin" / "__init__.py"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("", encoding="utf-8")
            config = root / "projects" / "configs" / "stage2.py"
            config.parent.mkdir(parents=True)
            config.write_text("", encoding="utf-8")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            output = Path(directory) / "report.json"
            args = argparse.Namespace(
                uniad_root=root,
                config=config,
                checkpoint=checkpoint,
                output=output,
            )
            benchmark._validate_files(argparse.ArgumentParser(), args)

        self.assertTrue(args.uniad_root.is_absolute())
        self.assertTrue(args.config.is_absolute())
        self.assertTrue(args.checkpoint.is_absolute())
        self.assertTrue(args.output.is_absolute())

    def test_validation_rejects_root_without_uniad_plugin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.py"
            checkpoint = root / "model.pth"
            config.write_text("", encoding="utf-8")
            checkpoint.write_bytes(b"x")
            args = argparse.Namespace(
                uniad_root=root,
                config=config,
                checkpoint=checkpoint,
                output=root / "report.json",
            )
            with self.assertRaises(SystemExit):
                benchmark._validate_files(argparse.ArgumentParser(), args)


class ConfigOptionTests(unittest.TestCase):
    def test_cfg_options_support_paths_scalars_containers_and_empty_root(self):
        parsed = benchmark._parse_cfg_options(
            [
                "data.test.ann_file=/data/mini.pkl",
                "data.test.data_root=",
                "data.workers_per_gpu=0",
                "flag=true",
                "shape=(1, 6)",
            ]
        )
        self.assertEqual(parsed["data.test.ann_file"], "/data/mini.pkl")
        self.assertEqual(parsed["data.test.data_root"], "")
        self.assertEqual(parsed["data.workers_per_gpu"], 0)
        self.assertIs(parsed["flag"], True)
        self.assertEqual(parsed["shape"], (1, 6))

    def test_malformed_and_duplicate_cfg_options_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark._parse_cfg_options(["missing-separator"])
        with self.assertRaises(ValueError):
            benchmark._parse_cfg_options(["x=1", "x=2"])


class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_loader_requires_strict_state_dict_match(self):
        calls = []
        model = object()

        def loader(*args, **kwargs):
            calls.append((args, kwargs))
            return {"meta": {"epoch": 6}}

        result = benchmark._load_checkpoint_strict(
            loader, model, Path("model.pth")
        )

        self.assertEqual(result, {"meta": {"epoch": 6}})
        self.assertEqual(calls[0][0], (model, "model.pth"))
        self.assertEqual(
            calls[0][1], {"map_location": "cpu", "strict": True}
        )


class TrackQueryTests(unittest.TestCase):
    def test_active_count_matches_uniad_motion_filter(self):
        result = benchmark._track_query_counts(
            object_indexes=[-1, 7, 8, -2, 9],
            scores=[0.99, 0.34, 0.35, 0.99, 0.80],
            filter_score_threshold=0.35,
        )
        self.assertEqual(result["track_query_count"], 5)
        self.assertEqual(result["active_track_query_count"], 2)

    def test_query_vector_size_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark._track_query_counts([1, 2], [0.9], 0.35)


class StatisticsTests(unittest.TestCase):
    def test_distribution_reports_requested_latency_metrics(self):
        result = benchmark._distribution([1.0, 2.0, 3.0, 4.0], include_fps=True)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["p90"], 3.7)
        self.assertAlmostEqual(result["p99"], 3.97)
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 4.0)
        self.assertEqual(result["fps_from_mean_latency"], 400.0)

    def test_empty_frame_group_is_explicit(self):
        result = benchmark._distribution([], include_fps=True)
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["median"])
        self.assertIsNone(result["fps_from_mean_latency"])

    def test_timing_groups_separate_first_and_steady_frames(self):
        samples = [
            {"frame_kind": "first-frame", "model_call_wall_ms": 4.0},
            {"frame_kind": "steady-state", "model_call_wall_ms": 2.0},
            {"frame_kind": "steady-state", "model_call_wall_ms": 3.0},
        ]
        grouped = benchmark._timing_groups(samples, "model_call_wall_ms")
        self.assertEqual(grouped["all"]["count"], 3)
        self.assertEqual(grouped["first_frame"]["median"], 4.0)
        self.assertEqual(grouped["steady_state"]["median"], 2.5)

    def test_observed_end_to_end_group_reports_fps(self):
        samples = [
            {
                "frame_kind": "steady-state",
                "observed_next_plus_model_wall_ms": 2500.0,
            }
        ]
        grouped = benchmark._timing_groups(
            samples, "observed_next_plus_model_wall_ms"
        )
        self.assertAlmostEqual(grouped["all"]["fps_from_mean_latency"], 0.4)


class ProvenanceAndOutputTests(unittest.TestCase):
    def test_sha256_file_streaming_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"checkpoint-bytes")
            self.assertEqual(
                benchmark._sha256_file(path, block_size=3),
                hashlib.sha256(b"checkpoint-bytes").hexdigest(),
            )

    def test_atomic_json_replaces_destination_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("old", encoding="utf-8")
            benchmark._atomic_write_json(path, {"value": 3, "items": [1, 2]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["value"], 3)
            self.assertEqual(list(Path(directory).glob(".report.json-*.tmp")), [])

    def test_scene_token_prefers_dataset_sequence(self):
        class Dataset:
            data_infos = [{"scene_token": "scene-a"}, {"scene_token": "scene-b"}]

        self.assertEqual(benchmark._scene_token(Dataset(), 1, {}), "scene-b")

    def test_repository_git_state_is_explicit_outside_a_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = benchmark._repository_git_state(Path(directory))
        self.assertIsNone(result["head"])
        self.assertIsNone(result["tracked_files_dirty"])
        self.assertTrue(Path(result["root"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
