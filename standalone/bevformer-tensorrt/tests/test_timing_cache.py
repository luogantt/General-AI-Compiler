"""CPU checks of strict timing-cache loading, serialization and build audit."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bevformer_trt import engine


class FakeConfig:
    def __init__(self, *, accept=True, serialized=b"updated TensorRT timing cache"):
        self.accept = accept
        self.cache = SimpleNamespace(serialize=lambda: serialized)
        self.created = []
        self.attached = []

    def create_timing_cache(self, payload):
        self.created.append(payload)
        return self.cache

    def set_timing_cache(self, cache, *, ignore_mismatch):
        self.attached.append((cache, ignore_mismatch))
        return self.accept

    def get_timing_cache(self):
        return self.cache

    def set_memory_pool_limit(self, *_args):
        pass

    def clear_flag(self, *_args):
        pass


class TimingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"model fixture")
        self.input = self.root / "input.cache"
        self.input.write_bytes(b"original TensorRT timing cache")
        self.output = self.root / "output.cache"

    def args(self, **changes):
        args = dict(onnx=self.model, engine=self.root / "model.engine", report=self.root / "build.json",
                    precision="fp32", plan_only=False, calibration_dir=None, calibration_cache=None,
                    plugins_fp32=False, workspace_mib=1024, max_aux_streams=0,
                    timing_cache_input=self.input, timing_cache_output=self.output)
        args.update(changes)
        return SimpleNamespace(**args)

    def fake_runtime(self, config):
        network = SimpleNamespace(num_inputs=0, num_outputs=0, num_layers=0,
                                  get_input=lambda _: None, get_output=lambda _: None)
        builder = SimpleNamespace(create_network=lambda _: network, create_builder_config=lambda: config,
                                  build_serialized_network=lambda *_: b"serialized engine")
        parser = SimpleNamespace(parse_from_file=lambda _: True, num_errors=0)
        inspector = SimpleNamespace(get_engine_information=lambda _: "{}")
        built = SimpleNamespace(create_engine_inspector=lambda: inspector)
        runtime = SimpleNamespace(deserialize_cuda_engine=lambda _: built)
        trt = SimpleNamespace(Builder=lambda _: builder, OnnxParser=lambda *_: parser, Runtime=lambda _: runtime,
                              MemoryPoolType=SimpleNamespace(WORKSPACE=0), BuilderFlag=SimpleNamespace(TF32=0),
                              ProfilingVerbosity=SimpleNamespace(DETAILED=0), LayerInformationFormat=SimpleNamespace(JSON=0))
        return trt, None, {}, None

    def test_build_loads_strictly_and_saves_exact_cache_bytes_and_audit(self):
        config = FakeConfig()
        with patch.object(engine, "initialize_builder", return_value=self.fake_runtime(config)):
            report, code = engine.build(self.args())
        self.assertEqual(code, 0)
        self.assertEqual(config.created, [self.input.read_bytes()])
        self.assertEqual(config.attached, [(config.cache, False)])
        self.assertEqual(self.output.read_bytes(), b"updated TensorRT timing cache")
        audit = report["timing_cache"]
        self.assertEqual(audit["input"]["sha256"], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(audit["output"]["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(audit["input"]["path"], str(self.input.resolve()))
        self.assertEqual(audit["output"]["path"], str(self.output.resolve()))
        self.assertEqual(audit["output"]["bytes"], self.output.stat().st_size)
        self.assertEqual(audit["status"], "attached_and_built")
        self.assertFalse(audit["ignore_mismatch"])
        self.assertEqual(self.input.read_bytes(), b"original TensorRT timing cache")

    def test_output_only_creates_empty_cache_and_legacy_args_keep_default_behavior(self):
        config = FakeConfig()
        with patch.object(engine, "initialize_builder", return_value=self.fake_runtime(config)):
            report, _ = engine.build(self.args(timing_cache_input=None))
        self.assertEqual(config.created, [b""])
        self.assertEqual(report["timing_cache"]["initialization"], "empty_cache")
        self.assertIsNone(report["timing_cache"]["input"])
        args = self.args(engine=self.root / "legacy.engine", report=self.root / "legacy.json")
        del args.timing_cache_input, args.timing_cache_output
        default_config = FakeConfig()
        with patch.object(engine, "initialize_builder", return_value=self.fake_runtime(default_config)):
            default_report, _ = engine.build(args)
        self.assertEqual(default_config.created, [])
        self.assertEqual(default_config.attached, [])
        self.assertNotIn("timing_cache", default_report)
        self.assertEqual(report["precision_policy"], default_report["precision_policy"])
        self.assertEqual(report["policy_fingerprint"], default_report["policy_fingerprint"])

    def test_invalid_paths_rejected_before_gpu_initialization(self):
        existing = self.root / "existing.cache"
        existing.write_bytes(b"do not overwrite")
        empty = self.root / "empty.cache"
        empty.write_bytes(b"")
        cases = [dict(timing_cache_output=existing), dict(timing_cache_input=self.root / "missing.cache"),
                 dict(timing_cache_input=empty), dict(timing_cache_output=self.input),
                 dict(timing_cache_input=self.model), dict(timing_cache_output=self.root / "model.engine"),
                 dict(timing_cache_output=self.root / "build.layers.json"), dict(plan_only=True),
                 dict(timing_cache_output=self.model / "child.cache")]
        for changes in cases:
            with self.subTest(changes=changes), patch.object(engine, "initialize_builder") as initialize:
                with self.assertRaises((ValueError, FileExistsError, FileNotFoundError)):
                    engine.build(self.args(**changes))
                initialize.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"do not overwrite")
        self.assertFalse(self.output.exists())

    def test_incompatible_cache_is_rejected_without_engine_or_cache_output(self):
        config = FakeConfig(accept=False)
        with patch.object(engine, "initialize_builder", return_value=self.fake_runtime(config)):
            with self.assertRaisesRegex(RuntimeError, "mismatch is not ignored"):
                engine.build(self.args())
        self.assertEqual(config.attached, [(config.cache, False)])
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "model.engine").exists())

    def test_missing_serialized_cache_fails_before_saving_engine(self):
        config = FakeConfig(serialized=None)
        with patch.object(engine, "initialize_builder", return_value=self.fake_runtime(config)):
            with self.assertRaisesRegex(RuntimeError, "serialize"):
                engine.build(self.args())
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "model.engine").exists())


if __name__ == "__main__":
    unittest.main()
