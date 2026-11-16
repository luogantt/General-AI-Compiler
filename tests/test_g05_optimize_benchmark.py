from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

DIRECTORY = Path(__file__).resolve().parents[1] / "tools" / "galaxeavla"
sys.path.insert(0, str(DIRECTORY))
SPEC = importlib.util.spec_from_file_location("benchmark_g05_optimize", DIRECTORY / "benchmark_g05_optimize.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OptimizationBenchmarkTests(unittest.TestCase):
    @staticmethod
    def configuration(profile="full", weights="bf16"):
        return {
            "model": {
                "use_torch_compile": False,
                "model_weights_to_bf16": weights == "bf16",
                "processor": {"num_output_cameras": 3},
                "model_arch": {
                    "predict_cot": profile == "full", "discrete_action": profile == "full",
                    "continuous_action": True, "return_continuous_action": True,
                    "max_text_tokens": 160, "input_preprocessor": {"pred_eov": True},
                    "action_dim": 27, "cond_steps": 1, "attn_implementation": "eager",
                    "fm": {"num_inference_steps": 10, "use_correlated_noise": False},
                    "ar": {"do_sample": False, "max_new_tokens": 300},
                },
            },
            "data": {"action_size": 32, "obs_size": 1},
        }

    def test_full_contract_accepts_bf16_and_fp32_without_disabling_heads(self):
        for weights in ("bf16", "fp32"):
            with self.subTest(weights=weights):
                observed = MODULE.validate_profile(self.configuration(weights=weights), "full", weights)
                self.assertEqual(observed["model.model_weights_to_bf16"], weights == "bf16")
                self.assertTrue(observed["model.model_arch.predict_cot"])
                self.assertTrue(observed["model.model_arch.discrete_action"])
        reduced = self.configuration(profile="fm-only")
        with self.assertRaisesRegex(RuntimeError, "predict_cot"):
            MODULE.validate_profile(reduced, "full", "bf16")

    def test_fm_only_retains_camera_shape_horizon_and_ten_fm_steps(self):
        config = self.configuration(profile="fm-only")
        MODULE.validate_profile(config, "fm-only", "bf16")
        config["model"]["model_arch"]["fm"]["num_inference_steps"] = 5
        with self.assertRaisesRegex(RuntimeError, "num_inference_steps"):
            MODULE.validate_profile(config, "fm-only", "bf16")
        config["model"]["model_arch"]["fm"]["num_inference_steps"] = 10
        config["model"]["processor"]["num_output_cameras"] = 2
        with self.assertRaisesRegex(RuntimeError, "num_output_cameras"):
            MODULE.validate_profile(config, "fm-only", "bf16")

    def test_explicit_profiles_and_bf16_flag(self):
        reduced = MODULE.profile_overrides("fm-only", "bf16")
        self.assertIn("model.model_arch.predict_cot=false", reduced)
        self.assertIn("model.model_arch.discrete_action=false", reduced)
        self.assertIn("model.model_weights_to_bf16=true", reduced)
        full = MODULE.profile_overrides("full", "fp32")
        self.assertIn("model.model_arch.predict_cot=true", full)
        self.assertIn("model.model_arch.discrete_action=true", full)
        self.assertIn("model.model_weights_to_bf16=false", full)

    def test_shape_and_finiteness_depend_on_profile(self):
        core = {"action": np.zeros((1, 32, 27), dtype=np.float32)}
        served = {key: np.zeros((1, 32, width), dtype=np.float32) for key, width in MODULE.base.STATE_DIMS.items()}
        self.assertEqual(len(MODULE.validate_outputs(core, served, "fm-only")), 5)
        with self.assertRaisesRegex(RuntimeError, "ar_action"):
            MODULE.validate_outputs(core, served, "full")
        core["ar_action"] = core["action"].copy()
        self.assertEqual(len(MODULE.validate_outputs(core, served, "full")), 6)
        core["action"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            MODULE.validate_outputs(core, served, "full")

    def test_wrong_shape_fails_even_when_finite(self):
        core = {"action": np.zeros((1, 16, 27))}
        served = {key: np.zeros((1, 32, width)) for key, width in MODULE.base.STATE_DIMS.items()}
        with self.assertRaisesRegex(RuntimeError, "wrong-shaped"):
            MODULE.validate_outputs(core, served, "fm-only")

    def test_output_comparison_reports_error_and_threshold(self):
        reference = {"action": np.array([1.0, 2.0])}
        same = MODULE.compare_outputs(reference, reference, atol=0, rtol=0)
        self.assertTrue(same["passed"])
        self.assertEqual(same["outputs"]["action"]["max_abs_error"], 0)
        mismatch = MODULE.compare_outputs(reference, {"action": np.array([1.0, 2.1])}, atol=.001, rtol=.001)
        self.assertFalse(mismatch["passed"])
        self.assertAlmostEqual(mismatch["outputs"]["action"]["max_abs_error"], .1)
        for ref, actual in (({}, {}), (reference, {**reference, "ar_action": reference["action"]})):
            with self.assertRaisesRegex(RuntimeError, "matching nonempty"):
                MODULE.compare_outputs(ref, actual, atol=.001, rtol=.001)

    def test_request_timing_excludes_validation_and_requires_profile_ar_calls(self):
        events = []
        # These are CPU snapshots: cloning happens after the request timer ends.
        class Snapshot(np.ndarray):
            def copy(self):
                events.append("clone")
                return super().copy()

        core = {"action": np.zeros((1, 32, 27)).view(Snapshot)}
        served = {key: np.zeros((1, 32, width)) for key, width in MODULE.base.STATE_DIMS.items()}
        inferencer = SimpleNamespace(
            infer_with_timing=lambda obs: ([served], {"preprocess_ms": 12, "forward_inference_total_ms": 60}),
            last_transfer_timing={"h2d_ms": 2, "model_predict_action_ms": 70, "d2h_ms": 1},
            last_model_metadata={"ar_generation_calls": []}, last_core_actions=core,
        )
        clock_values = iter((1., 1.01, 1.1))
        def clock():
            events.append("clock")
            return next(clock_values)
        original_validation = MODULE.validate_outputs
        def validate(*args):
            events.append("validation")
            return original_validation(*args)
        with patch.object(MODULE.time, "monotonic", side_effect=clock), patch.object(MODULE, "validate_outputs", side_effect=validate):
            sample, output = MODULE.run_one(
                raw_obs={}, processor=None, build_obs_dict=lambda obs, processor: obs,
                inferencer=inferencer, runtime=SimpleNamespace(synchronize=lambda: events.append("sync")), profile="fm-only")
        self.assertEqual(events, ["sync", "clock", "clock", "sync", "clock", "validation", "clone"])
        self.assertAlmostEqual(sample["request_total_ms"], 100.)
        self.assertAlmostEqual(sample["cpu_preprocess_total_ms"], 22.)
        self.assertEqual(sample["model_predict_action_ms"], 70)
        self.assertIsNot(output["action"], core["action"])
        inferencer.last_model_metadata["ar_generation_calls"] = [{"stage": "cot", "generated_tokens": 1}]
        with self.assertRaisesRegex(RuntimeError, "AR calls"):
            MODULE.run_one(raw_obs={}, processor=None, build_obs_dict=lambda obs, processor: obs,
                           inferencer=inferencer, runtime=SimpleNamespace(synchronize=lambda: None), profile="fm-only")

    def test_compile_targets_forward_only_and_disallows_fallback(self):
        calls = []
        original = lambda value: value + 1
        module = SimpleNamespace(forward=original, embed=object(), decode=object())
        compiled = lambda value: value + 1
        def compile_method(fn, **kwargs):
            calls.append((fn, kwargs))
            return compiled
        torch = SimpleNamespace(compile=compile_method, _dynamo=SimpleNamespace(config=SimpleNamespace(suppress_errors=True)))
        options = {"frozen_parameter": True}
        previous, metadata = MODULE.install_module_backend(torch, module, "npugraph_ex", "AE.forward", options=options)
        self.assertIs(previous, original)
        self.assertIs(module.forward, compiled)
        self.assertIsNotNone(module.embed)
        self.assertFalse(torch._dynamo.config.suppress_errors)
        self.assertEqual(calls[0][1], {"backend": "npugraph_ex", "fullgraph": True, "dynamic": False, "options": options})
        self.assertFalse(metadata["fallback_allowed"])

    def test_cli_graph_options_reject_unknown_keys_nonbooleans_and_bad_helpers(self):
        self.assertEqual(MODULE.parse_graph_options('{"frozen_parameter":true}'), {"frozen_parameter": True})
        for invalid in ('[]', 'null', '{"unknown":true}', '{"clone_output":1}'):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                MODULE.parse_graph_options(invalid)
        self.assertEqual(MODULE.parse_gdn_helpers("chunk, recurrent"), ("chunk", "recurrent"))
        for invalid in ("", "chunk,chunk", "not-a-helper"):
            with self.subTest(invalid=invalid), self.assertRaises(argparse.ArgumentTypeError):
                MODULE.parse_gdn_helpers(invalid)

    def test_parser_defaults_keep_vlm_eager(self):
        args = MODULE.build_parser().parse_args(["--source-dir", "source", "--checkpoint", "weights.pt", "--output", "report.json"])
        self.assertEqual((args.profile, args.weights, args.ae_backend, args.vlm_backend), ("full", "bf16", "eager", "eager"))
        self.assertEqual((args.gdn_backend, args.gdn_helpers, args.gdn_options), ("eager", ("chunk", "recurrent"), None))
        self.assertEqual((args.warmup, args.iterations), (3, 10))

    def test_verification_inputs_change_images_state_and_seed_but_keep_task_and_shapes(self):
        def make_raw(seed, task):
            return {
                "images": {key: np.full((3, 2, 2), seed, dtype=np.uint8) for key in MODULE.base.CAMERA_KEYS},
                "state": {key: np.zeros(width, dtype=np.float32) for key, width in MODULE.base.STATE_DIMS.items()},
                "task": task, "frequency": 15.0, "embodiment_type": MODULE.base.DEFAULT_EMBODIMENT,
            }
        with patch.object(MODULE.base, "build_synthetic_raw_observation", side_effect=make_raw):
            cases = MODULE.build_verification_cases(make_raw(10, "fixed task"), 10)
        self.assertEqual([case["seed"] for case in cases.values()], [10, 11])
        self.assertEqual(cases["a"]["metadata"]["shapes"], cases["b"]["metadata"]["shapes"])
        self.assertEqual(cases["a"]["raw_obs"]["task"], cases["b"]["raw_obs"]["task"])
        self.assertNotEqual(cases["a"]["metadata"]["observation_sha256"], cases["b"]["metadata"]["observation_sha256"])
        for group in ("images", "state"):
            for key in cases["a"]["raw_obs"][group]:
                self.assertFalse(np.array_equal(cases["a"]["raw_obs"][group][key], cases["b"]["raw_obs"][group][key]))

    def test_replay_verification_copies_reused_buffers_and_checks_both_inputs_again(self):
        cases = {name: {"seed": seed, "metadata": {"generation_seed": seed}, "raw_obs": {}}
                 for name, seed in (("a", 7), ("b", 8))}
        buffer = np.zeros((1, 2))
        calls = []
        def execute(case):
            calls.append(case["seed"])
            buffer[:] = case["seed"]
            return {"request_total_ms": 1.0}, {"action": buffer}
        with contextlib.redirect_stdout(io.StringIO()):
            references = MODULE.collect_eager_references(cases, execute)
            self.assertTrue(np.all(references["a"]["outputs"]["action"] == 7))
            self.assertTrue(np.all(references["b"]["outputs"]["action"] == 8))
            cold = execute(cases["a"])
            report = MODULE.verify_request_replay(cases, references, cold, execute, atol=0, rtol=0)
        self.assertEqual(calls, [7, 8, 7, 8, 7, 8])
        self.assertTrue(report["passed"])
        self.assertTrue(report["reference_inputs_distinguishable"])
        self.assertEqual([case["case"] for case in report["cases"]], ["cold_a", "first_b", "replay_a", "replay_b"])
        self.assertTrue(all(case["outputs"]["action"]["max_abs_error"] == 0 for case in report["cases"]))

    def test_replay_verification_rejects_capture_correct_but_stale_replayed_output(self):
        cases = {name: {"seed": seed, "metadata": {}, "raw_obs": {}}
                 for name, seed in (("a", 7), ("b", 8))}
        references = {name: {"timing_ms": {}, "outputs": {"action": np.array([case["seed"]], dtype=float)}}
                      for name, case in cases.items()}
        buffer = np.array([7.])
        calls = []
        def bad_execute(case):
            calls.append(case["seed"])
            # Cold A and first B are correct; subsequent requests replay B's
            # stale buffer irrespective of the changed A input.
            buffer[:] = 8.
            return {}, {"action": buffer}
        with contextlib.redirect_stdout(io.StringIO()):
            report = MODULE.verify_request_replay(cases, references, ({}, {"action": buffer}), bad_execute, atol=0, rtol=0)
        self.assertEqual(calls, [8, 7, 8])
        self.assertFalse(report["passed"])
        self.assertEqual([case["passed"] for case in report["cases"]], [True, True, False, True])
        self.assertEqual(report["outputs"]["action"]["max_abs_error"], 1.)

    def test_replay_verification_rejects_indistinguishable_references(self):
        cases = {name: {"seed": seed, "metadata": {}, "raw_obs": {}} for name, seed in (("a", 7), ("b", 8))}
        output = {"action": np.array([1.])}
        references = {name: {"timing_ms": {}, "outputs": output} for name in cases}
        with contextlib.redirect_stdout(io.StringIO()):
            report = MODULE.verify_request_replay(cases, references, ({}, output), lambda case: ({}, output), atol=0, rtol=0)
        self.assertFalse(report["passed"])
        self.assertFalse(report["reference_inputs_distinguishable"])
        self.assertTrue(all(case["passed"] for case in report["cases"]))


if __name__ == "__main__":
    unittest.main()
