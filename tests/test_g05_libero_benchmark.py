from __future__ import annotations

import copy
import contextlib
import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "galaxeavla"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("benchmark_g05_libero", TOOLS / "benchmark_g05_libero.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config():
    return {"model": {"use_torch_compile": False, "model_arch": {
        "action_dim": 20, "proprio_dim": 20, "predict_cot": False,
        "discrete_action": False, "continuous_action": True, "return_continuous_action": True,
        "attn_implementation": "eager", "fm": {"num_inference_steps": 10,
            "time_convention": "pi_convention", "use_correlated_noise": False}},
        "processor": {"num_output_cameras": 2, "num_input_cameras": 2,
            "camera_size_config": {"exterior": [256, 256], "wrist_right": [256, 256]}}},
        "data": {"action_size": 32, "obs_size": 1}}


class LiberoBenchmarkTests(unittest.TestCase):
    def test_rejects_r1lite_or_reduced_sampling_workloads(self):
        MODULE.validate_contract(config())
        for section, key, changed in (("model_arch", "action_dim", 27),
                                      ("model_arch", "predict_cot", True),
                                      ("model_arch", "discrete_action", True),
                                      ("processor", "num_output_cameras", 3)):
            value = config()
            value["model"][section][key] = changed
            with self.assertRaisesRegex(ValueError, "LIBERO workload mismatch"):
                MODULE.validate_contract(value)
        value = config()
        value["model"]["model_arch"]["fm"]["num_inference_steps"] = 5
        with self.assertRaisesRegex(ValueError, "num_inference_steps"):
            MODULE.validate_contract(value)

    def test_error_statistics_do_not_dilute_over_thirteen_padding_dimensions(self):
        reference = np.ones(MODULE.ACTION_SHAPE, dtype=np.float32)
        candidate = reference.copy()
        pad = np.ones((1, 20), dtype=bool)
        pad[:, MODULE.VALID_DIMS] = False
        reference[..., pad[0]] = 0
        candidate[..., pad[0]] = 9
        candidate[..., MODULE.VALID_DIMS] += 2
        result = MODULE.action_comparison(reference, candidate, pad)
        self.assertEqual(result["sample_count"], 32 * 7)
        self.assertEqual(result["mae"], 2)
        self.assertEqual(result["relative_l2"], 2)
        self.assertEqual(result["padded_dimensions_max_abs_candidate"], 9)
        candidate[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            MODULE.action_comparison(reference, candidate, pad)

    def test_valid_dimensions_must_match_libero_embodiment(self):
        value = np.ones(MODULE.ACTION_SHAPE)
        with self.assertRaisesRegex(ValueError, "valid dimensions"):
            MODULE.action_comparison(value, value, np.zeros((1, 20), dtype=bool))
        with self.assertRaisesRegex(ValueError, "actions"):
            MODULE.action_comparison(value[..., :7], value[..., :7], np.zeros((1, 7), dtype=bool))

    def test_shared_noise_is_saved_exactly_and_rejects_wrong_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, first = MODULE.prepare_noise(None, root / "first.npy", 5)
            restored, second = MODULE.prepare_noise(root / "first.npy", root / "second.npy", 99)
            np.testing.assert_array_equal(original, restored)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["tensor_sha256"], hashlib.sha256(original.tobytes()).hexdigest())
            np.save(root / "bad.npy", original.astype(np.float64))
            with self.assertRaisesRegex(ValueError, "finite float32"):
                MODULE.prepare_noise(root / "bad.npy", root / "unused.npy", 1)

    def test_explicit_noise_clones_before_in_place_euler_and_restores_helper(self):
        class Tensor:
            device, dtype, shape = "npu:0", "float32", MODULE.ACTION_SHAPE

            def __init__(self, values):
                self.values = values

            def clone(self):
                return Tensor(self.values.copy())

        class Helper:
            def _sample_noise(self, *_args):
                return "original"

        helper = Helper()
        tensor = Tensor(np.ones(MODULE.ACTION_SHAPE))
        with MODULE.explicit_noise(helper, tensor) as calls:
            first = helper._sample_noise(tensor, tensor.dtype, ["libero"])
            first.values[:] = 99
            second = helper._sample_noise(tensor, tensor.dtype, ["libero"])
            np.testing.assert_array_equal(second.values, 1)
            calls["tensor"] = Tensor(-tensor.values)
            third = helper._sample_noise(tensor, tensor.dtype, ["libero"])
            np.testing.assert_array_equal(third.values, -1)
            self.assertEqual(calls["count"], 3)
        self.assertNotIn("_sample_noise", vars(helper))
        self.assertEqual(helper._sample_noise(), "original")
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with MODULE.explicit_noise(helper, tensor):
                raise RuntimeError("test failure")
        self.assertNotIn("_sample_noise", vars(helper))

    def test_same_seed_is_insufficient_for_reference_matching(self):
        identity = {"noise_tensor_sha256": "shared", "weights": "bf16", "contract": config()}
        MODULE.check_reference_identity(copy.deepcopy(identity), identity)
        other = dict(identity, noise_tensor_sha256="different-platform-rng")
        with self.assertRaisesRegex(ValueError, "noise_tensor_sha256"):
            MODULE.check_reference_identity(other, identity)

    def test_sidecar_content_is_verified_not_just_file_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"model")
            sidecar = root / "sidecar.json"
            sidecar.write_bytes(b"123")
            git_hash = hashlib.sha1(b"blob 3\x00123").hexdigest()
            with patch.object(MODULE, "ASSET_LFS", {}), patch.object(MODULE, "CHECKPOINT_SIZE", 5), \
                 patch.object(MODULE, "ASSET_BLOBS", {"sidecar.json": (3, git_hash)}):
                MODULE.validate_assets(checkpoint, root, hashlib.sha256(b"model").hexdigest())
                sidecar.write_bytes(b"321")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    MODULE.validate_assets(checkpoint, root, hashlib.sha256(b"model").hexdigest())

    def test_case_rejects_wrong_episode_or_checkpoint(self):
        case = {"dataset": {"episode": 0, "frame": 0, "global_index": 0,
                            "task": MODULE.TASK, "fps": 20., "raw_state": [0.] * 8},
                "checkpoint": {"bytes": MODULE.CHECKPOINT_SIZE}}
        MODULE.validate_case(case)
        case["dataset"]["frame"] = 1
        with self.assertRaisesRegex(ValueError, "episode 0/frame 0"):
            MODULE.validate_case(case)

    def test_experimental_algorithm_requires_live_numerical_verification(self):
        args = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output", "--gdn-algorithm", "triangular"]
        for suffix in ([], ["--verify", "--device", "cpu"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                MODULE.parse_args(args + suffix)
        parsed = MODULE.parse_args(args + ["--verify", "--device", "npu:0"])
        self.assertEqual(parsed.gdn_algorithm, "triangular")

    def test_ge_options_do_not_silently_accept_acl_only_settings(self):
        args = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output", "--ae-backend", "torchair"]
        parsed = MODULE.parse_args(args + ["--ae-options", '{"frozen_parameter":true}'])
        self.assertEqual(parsed.ae_options, {"frozen_parameter": True})
        parsed = MODULE.parse_args(args + ["--verify", "--ae-options", '{"eye_decomposition":true}'])
        self.assertEqual(parsed.ae_options, {"eye_decomposition": True})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(args + ["--ae-options", '{"super_kernel_optimize":true}'])

    def test_ae_fusion_requires_npu_and_verification(self):
        base = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output"]
        parsed = MODULE.parse_args(base)
        self.assertEqual((parsed.ae_norm, parsed.ae_rope), ("eager", "eager"))
        self.assertFalse(parsed.functional_mrope)
        for flag, mode in (("--ae-norm", "npu"), ("--ae-rope", "npu"), ("--ae-rope", "matrix")):
            for suffix in ([], ["--verify", "--device", "cpu"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    MODULE.parse_args(base + [flag, mode] + suffix)
            parsed = MODULE.parse_args(base + [flag, mode, "--verify", "--device", "npu:0"])
            self.assertEqual(getattr(parsed, flag[2:].replace("-", "_")), mode)
        for suffix in ([], ["--verify", "--device", "cpu"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                MODULE.parse_args(base + ["--functional-mrope"] + suffix)
        self.assertTrue(MODULE.parse_args(base + ["--functional-mrope", "--verify", "--device", "npu:0"]).functional_mrope)

    def test_mixed_backend_uses_one_package_to_avoid_operator_registration_collision(self):
        args = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output", "--ae-backend", "torchair"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(args + ["--vlm-backend", "npugraph_ex"])
        parsed = MODULE.parse_args(args + ["--vlm-backend", "torchair_aclgraph",
            "--vlm-options", '{"frozen_parameter":true}'])
        self.assertEqual(parsed.vlm_backend, "torchair_aclgraph")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(args + ["--vlm-backend", "torchair_aclgraph",
                "--vlm-options", '{"single_stream":true}'])

    def test_workload_is_opt_in_and_keeps_existing_optional_diagnostics(self):
        args = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output"]
        self.assertFalse(MODULE.parse_args(args).count_workload)
        parsed = MODULE.parse_args(args + ["--count-workload", "--profile-modules"])
        self.assertTrue(parsed.count_workload)
        self.assertTrue(parsed.profile_modules)

    def test_context_optimization_requires_verification_and_preserves_real_input(self):
        args = ["--source-dir", "/source", "--asset-root", "/assets", "--checkpoint", "/model.pt",
                "--input-dir", "/input", "--output-dir", "/output", "--skip-unused-action-codec"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(args)
        self.assertTrue(MODULE.parse_args(args + ["--verify"]).skip_unused_action_codec)
        raw = {"state": {"right_ee_pose": np.arange(6, dtype=np.float32)}, "task": MODULE.TASK,
               "images": {"image": np.zeros((3, 2, 2), dtype=np.uint8)}}
        cases = MODULE.context_verification_observations(raw)
        np.testing.assert_array_equal(raw["state"]["right_ee_pose"], np.arange(6))
        self.assertEqual(raw["task"], MODULE.TASK)
        np.testing.assert_allclose(cases["state_C"]["state"]["right_ee_pose"], np.arange(6) + .01)
        self.assertNotEqual(cases["task_D"]["task"], raw["task"])
        cases["state_C"]["images"]["image"][:] = 1
        self.assertFalse(raw["images"]["image"].any())

    def test_context_comparison_rejects_changed_tokens_or_semantic_mask(self):
        reference = {"input_ids": np.array([[3, 2, 1]]), "attention_mask": np.array([[4, 2, 1]])}
        self.assertTrue(all(MODULE.compare_encoded_context(reference, copy.deepcopy(reference)).values()))
        for key in reference:
            changed = copy.deepcopy(reference)
            changed[key][0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, "encoded tokens or semantic mask"):
                MODULE.compare_encoded_context(reference, changed)

    def test_counted_request_is_after_backend_restoration_and_outside_measured_loops(self):
        tree = ast.parse((TOOLS / "benchmark_g05_libero.py").read_text(encoding="utf-8"))
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]
        counted = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "collect_counted_workload"]
        self.assertEqual(len(counted), 1)
        count_line = counted[0].lineno
        restores = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "restore"]
        self.assertTrue(restores)
        self.assertGreater(count_line, max(node.lineno for node in restores))
        for node in ast.walk(run):
            if isinstance(node, ast.For):
                self.assertNotIn(counted[0], list(ast.walk(node)))
        noise_context = next(node for node in ast.walk(run) if isinstance(node, ast.With)
                             and any(isinstance(item.context_expr, ast.Call)
                                     and isinstance(item.context_expr.func, ast.Name)
                                     and item.context_expr.func.id == "explicit_noise" for item in node.items))
        self.assertIn(counted[0], list(ast.walk(noise_context)))
        expected = next(node for node in ast.walk(run) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "expected_calls" for target in node.targets))
        self.assertIn("int(args.count_workload)", ast.unparse(expected.value))

    def test_workload_counter_lifecycle_discards_timing_and_consumes_noise_once(self):
        events = []
        policy = SimpleNamespace(instrumented=False)
        tensor = SimpleNamespace(shape=MODULE.ACTION_SHAPE, device="npu:0", dtype="float32", clone=lambda: "shared-noise-copy")

        class Helper:
            def _sample_noise(self, *_args):
                raise AssertionError("Native RNG must not run in the counted request")

        class Action:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return np.zeros(MODULE.ACTION_SHAPE, dtype=np.float32)

        class Counter:
            def __init__(self, _torch):
                self.roots = [{"name": "policy", "type": "fake"}]
                self.parameters = {"total_numel": 5, "total_bytes": 20}
                self.prefill = {"prefix_tokens": 159}
                self.model_calls = [{"component": "action_expert", "stage": "fm"}]

            def attach(self, current_policy):
                self.policy = current_policy
                current_policy.instrumented = True
                events.append("attach")

            def close(self):
                self.policy.instrumented = False
                events.append("close")

            def summary(self):
                self_summary = {"counted_flops": 10, "stages": {"fm": {"counted_flops": 10}},
                                "operators": [{"operator": "Linear", "counted_flops": 10}]}
                events.append("summary")
                return self_summary

        helper = Helper()
        module = SimpleNamespace(WorkloadCounter=Counter, __file__=TOOLS / "count_g05_workload.py")
        pad = np.ones((1, 20), dtype=bool)
        pad[:, MODULE.VALID_DIMS] = False

        def inference():
            self.assertTrue(policy.instrumented)
            self.assertEqual(helper._sample_noise(tensor, tensor.dtype, ["libero"]), "shared-noise-copy")
            events.append("inference")
            return None, Action(), {"total_ms": 123456789.0}

        with MODULE.explicit_noise(helper, tensor) as noise_calls:
            report = MODULE.collect_counted_workload(None, policy, inference, pad, counter_module=module)
            self.assertEqual(noise_calls["count"], 1)
        self.assertEqual(events, ["attach", "inference", "close", "summary"])
        self.assertFalse(policy.instrumented)
        self.assertNotIn("_sample_noise", vars(helper))
        self.assertEqual(report["summary"]["counted_flops"], 10)
        self.assertEqual(report["operators"][0]["operator"], "Linear")
        self.assertEqual(report["prefill"]["prefix_tokens"], 159)
        self.assertEqual(report["diagnostic_requests"], 1)
        self.assertEqual(report["execution"]["gdn_algorithm"], "original")
        self.assertEqual(report["instrumentation_sha256"], MODULE.base.sha256_file(module.__file__))
        self.assertNotIn("123456789", str(report))
        self.assertIn("not total model FLOPs", report["interpretation"])
        self.assertTrue(report["not_counted"])

    def test_workload_counter_closes_on_attach_or_inference_failure(self):
        for failure_point in ("attach", "inference"):
            events = []
            policy = SimpleNamespace(instrumented=False)

            class Counter:
                def __init__(self, _torch): pass
                def attach(self, current_policy):
                    self.policy = current_policy
                    current_policy.instrumented = True
                    events.append("attach")
                    if failure_point == "attach":
                        raise RuntimeError("failed attach")
                def close(self):
                    self.policy.instrumented = False
                    events.append("close")

            def fail_inference():
                events.append("inference")
                raise RuntimeError("failed inference")

            module = SimpleNamespace(WorkloadCounter=Counter, __file__=TOOLS / "count_g05_workload.py")
            with self.assertRaisesRegex(RuntimeError, "failed " + failure_point):
                MODULE.collect_counted_workload(None, policy, fail_inference, None, counter_module=module)
            self.assertFalse(policy.instrumented)
            self.assertEqual(events[-1], "close")


if __name__ == "__main__":
    unittest.main()
