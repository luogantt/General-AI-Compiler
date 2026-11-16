from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "galaxeavla" / "benchmark_g05_full.py"
SPEC = importlib.util.spec_from_file_location("benchmark_g05_full", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FullG05BenchmarkTests(unittest.TestCase):
    def test_percentile_and_summary_are_deterministic(self):
        self.assertAlmostEqual(MODULE.percentile([4, 1, 3, 2], 50), 2.5)
        self.assertAlmostEqual(MODULE.percentile([1, 2, 3], 95), 2.9)
        result = MODULE.summarize([1, 2, 3])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["p50"], 2.0)
        self.assertAlmostEqual(result["p95"], 2.9)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)

    def test_summarize_samples_ignores_non_numeric_fields(self):
        result = MODULE.summarize_samples(
            [
                {"request_total_ms": 10.0, "label": "a"},
                {"request_total_ms": 20.0, "label": "b"},
            ]
        )
        self.assertEqual(set(result), {"request_total_ms"})
        self.assertEqual(result["request_total_ms"]["p50"], 15.0)

    def test_released_default_contract_accepts_only_full_dual_head(self):
        config = {
            "model": {
                "use_torch_compile": False,
                "processor": {"num_output_cameras": 3},
                "model_arch": {
                    "continuous_action": True,
                    "discrete_action": True,
                    "return_continuous_action": True,
                    "predict_cot": True,
                    "max_text_tokens": 160,
                    "input_preprocessor": {"pred_eov": True},
                    "attn_implementation": "eager",
                    "action_dim": 27,
                    "cond_steps": 1,
                    "fm": {"num_inference_steps": 10, "use_correlated_noise": False},
                    "ar": {"do_sample": False, "max_new_tokens": 300},
                }
            },
            "data": {"action_size": 32, "obs_size": 1},
        }
        observed = MODULE.validate_shipped_default_contract(config)
        self.assertIs(observed["model.model_arch.discrete_action"], True)

        config["model"]["model_arch"]["discrete_action"] = False
        with self.assertRaisesRegex(RuntimeError, "released-default full-model"):
            MODULE.validate_shipped_default_contract(config)

        config["model"]["model_arch"]["discrete_action"] = True
        config["model"]["model_arch"]["predict_cot"] = False
        with self.assertRaisesRegex(RuntimeError, "released-default full-model"):
            MODULE.validate_shipped_default_contract(config)

        config["model"]["model_arch"]["predict_cot"] = True
        config["model"]["model_arch"]["fm"]["use_correlated_noise"] = True
        with self.assertRaisesRegex(RuntimeError, "use_correlated_noise"):
            MODULE.validate_shipped_default_contract(config)

    def test_ar_observation_preserves_calls_outputs_and_restores_method(self):
        import numpy as np

        class Model:
            def inference_ar(self, generated_ids, **kwargs):
                self.kwargs = kwargs
                return {"generated_ids": generated_ids}

        model = Model()
        original = model.inference_ar
        first = np.zeros((1, 18), dtype=np.int64)
        terminal = np.zeros((1, 1), dtype=np.int64)
        with MODULE.record_ar_generation(model) as calls:
            self.assertIs(model.inference_ar(first, max_new_tokens=300)["generated_ids"], first)
            self.assertEqual(model.kwargs, {"max_new_tokens": 300})
            self.assertIs(model.inference_ar(terminal)["generated_ids"], terminal)
        self.assertEqual([call["stage"] for call in calls], ["cot", "action"])
        self.assertEqual([call["generated_tokens"] for call in calls], [18, 1])
        self.assertEqual(model.inference_ar, original)
        self.assertNotIn("inference_ar", vars(model))
        with self.assertRaisesRegex(ValueError, "failed"):
            with MODULE.record_ar_generation(model):
                raise ValueError("failed")
        self.assertEqual(model.inference_ar, original)

    def test_output_validation_checks_both_heads_and_served_parts(self):
        import numpy as np

        core = {"action": np.zeros((1, 32, 27)), "ar_action": np.zeros((1, 32, 27))}
        served = {"left_arm": np.zeros((16, 6)), "_cot_text": "text"}
        checked = MODULE.validate_action_outputs(core, served)
        self.assertEqual(set(checked), {"core.action", "core.ar_action", "served.left_arm"})
        core["ar_action"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "core.ar_action"):
            MODULE.validate_action_outputs(core, served)
        core["ar_action"][0, 0, 0] = 0
        served["left_arm"][0, 0] = np.inf
        with self.assertRaisesRegex(RuntimeError, "served.left_arm"):
            MODULE.validate_action_outputs(core, served)
        with self.assertRaisesRegex(RuntimeError, "Missing.*ar_action"):
            MODULE.validate_action_outputs({"action": core["action"]}, served)

    def test_run_one_distinguishes_parsed_tokens_from_raw_action_ar_tokens(self):
        import numpy as np

        served = {"left_arm": np.zeros((16, 6))}
        inferencer = SimpleNamespace(
            infer_with_timing=lambda obs: ([served], {"ar_decode_ms": 3.0}),
            last_transfer_timing={"h2d_ms": 1.0},
            last_core_actions={"action": np.zeros((1, 32, 27)), "ar_action": np.zeros((1, 32, 27))},
            last_model_metadata={
                "ar_token_lengths": [18],
                "ar_generation_calls": [
                    {"stage": "cot", "generated_tokens": 22},
                    {"stage": "action", "generated_tokens": 1},
                ],
            },
        )
        sample, result = MODULE._run_one(
            raw_obs={}, processor=None, build_obs_dict=lambda obs, processor: obs,
            inferencer=inferencer, runtime=SimpleNamespace(synchronize=lambda: None),
        )
        self.assertIs(result, served)
        self.assertEqual(sample["ar_decoded_action_tokens"], 18)
        self.assertEqual(sample["action_ar_raw_generated_tokens"], 1)
        self.assertEqual(sample["cot_raw_generated_tokens"], 22)
        self.assertNotIn("ar_decoded_action_tokens_per_second", sample)
        self.assertIn("core.ar_action", sample["output_validation"])

    def test_deployment_config_paths_are_public_r1lite_fragments(self):
        source = Path("/source")
        paths = MODULE.deployment_config_paths(source)
        self.assertEqual(paths["data"], source / "configs" / "data" / "r1lite.yaml")
        self.assertEqual(paths["task"], source / "configs" / "task" / "r1lite.yaml")

    def test_checkpoint_config_compatibility_keeps_model_and_reanchors_run_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            run_dir = root / "assets" / "g05-base"
            checkpoint = run_dir / "checkpoints" / "model_state_dict.pt"
            data_path = source / "configs" / "data" / "r1lite.yaml"
            task_path = source / "configs" / "task" / "r1lite.yaml"
            checkpoint_config = run_dir / ".hydra" / "config.yaml"
            for path in (data_path, task_path, checkpoint_config):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name, encoding="utf-8")

            raw = SimpleNamespace(
                data="legacy-data",
                model=SimpleNamespace(processor="checkpoint-processor"),
                tokenizer="checkpoint-tokenizer",
            )
            task = SimpleNamespace(model=SimpleNamespace(processor="r1lite-processor"), tokenizer="r1lite-tokenizer")
            public_data = SimpleNamespace(name="public-r1lite-data")

            class FakeOmegaConf:
                @staticmethod
                def load(path):
                    path = Path(path)
                    if path == checkpoint_config:
                        return raw
                    if path == task_path:
                        return task
                    if path == data_path:
                        return public_data
                    raise AssertionError(f"unexpected load: {path}")

                @staticmethod
                def set_struct(_config, _enabled):
                    return None

                @staticmethod
                def merge(first, second):
                    return (first, second)

                @staticmethod
                def save(_config, path, resolve=False):
                    self.assertFalse(resolve)
                    Path(path).write_text("shadow", encoding="utf-8")

            loaded_cfg = SimpleNamespace(run_dir="shadow", output_dir="shadow-output")

            def fake_loader(shadow_run_dir, checkpoint_path, overrides):
                self.assertTrue((Path(shadow_run_dir) / ".hydra" / "config.yaml").is_file())
                self.assertEqual(checkpoint_path, str(checkpoint))
                self.assertEqual(overrides, ["eval_embodiment=galaxea_r1lite"])
                self.assertIs(raw.data, public_data)
                self.assertEqual(
                    raw.model.processor,
                    ("checkpoint-processor", "r1lite-processor"),
                )
                return loaded_cfg

            cfg, metadata = MODULE.load_r1lite_checkpoint_config(
                source_dir=source,
                run_dir=run_dir,
                checkpoint=checkpoint,
                overrides=["eval_embodiment=galaxea_r1lite"],
                omega_conf=FakeOmegaConf,
                load_config_from_run_dir=fake_loader,
            )
            self.assertEqual(cfg.run_dir, str(run_dir))
            self.assertEqual(cfg.output_dir, str(run_dir / "eval_model_state_dict"))
            self.assertEqual(
                metadata["mode"],
                "checkpoint-model-plus-public-r1lite-deployment-config",
            )

    def test_synthetic_r1lite_observation_has_exact_schema(self):
        import numpy as np

        first = MODULE.build_synthetic_raw_observation(7)
        second = MODULE.build_synthetic_raw_observation(7)
        self.assertEqual(tuple(first["images"]), MODULE.CAMERA_KEYS)
        self.assertTrue(
            all(value.shape == (3, 720, 1280) for value in first["images"].values())
        )
        self.assertTrue(all(value.dtype == np.uint8 for value in first["images"].values()))
        self.assertEqual(
            {key: value.shape[0] for key, value in first["state"].items()},
            MODULE.STATE_DIMS,
        )
        self.assertEqual(sum(value.size for value in first["state"].values()), 14)
        self.assertEqual(first["embodiment_type"], "galaxea_r1lite")
        self.assertEqual(MODULE.observation_sha256(first), MODULE.observation_sha256(second))

    def test_asset_layout_matches_official_snapshot_shape(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "g05-base" / "checkpoints" / "model_state_dict.pt"
            layout = MODULE.resolve_asset_layout(checkpoint)
            resolved_root = root.resolve()
            self.assertEqual(layout["asset_root"], resolved_root)
            self.assertEqual(
                layout["hydra_config"],
                resolved_root / "g05-base" / ".hydra" / "config.yaml",
            )
            self.assertEqual(
                layout["action_tokenizer"],
                resolved_root / "action_tokenizer.pt",
            )

    def test_cli_defaults_pin_full_benchmark_contract(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = MODULE.build_parser()
            args = parser.parse_args(
                [
                    "--source-dir",
                    str(root / "source"),
                    "--checkpoint",
                    str(root / "model.pt"),
                    "--output",
                    str(root / "result.json"),
                ]
            )
            self.assertEqual(args.device, "npu:0")
            self.assertEqual(args.warmup, 5)
            self.assertEqual(args.iterations, 30)
            self.assertEqual(args.action_steps, 16)
            self.assertFalse(args.smoke_only)

    def test_smoke_only_writes_report_with_empty_measurement_summary(self):
        import contextlib
        import io
        import json
        import tempfile

        report = {
            "measurement": {"summary": {}},
            "cold_request": {"timing_ms": {"request_total_ms": 12.5}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            with patch.object(MODULE, "run_benchmark", return_value=report) as run:
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    code = MODULE.main([
                        "--source-dir", directory, "--checkpoint", str(Path(directory) / "model.pt"),
                        "--output", str(output), "--smoke-only", "--iterations", "0",
                    ])
            self.assertEqual(code, 0)
            self.assertTrue(run.call_args.args[0].smoke_only)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertIn("Smoke request: 12.500 ms", printed.getvalue())
            self.assertNotIn("p50", printed.getvalue())

    def test_source_pins_official_revision_and_checkpoint_hash(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("89f2322b4ad016e192437adc1a2c253b05bab246", text)
        self.assertIn(
            "072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327",
            text,
        )
        self.assertIn("synthetic-input full-model throughput", text)
        self.assertIn("continuous_flow_matching", text)
        self.assertIn("chain_of_thought_autoregressive", text)
        self.assertIn("discrete_autoregressive", text)
        self.assertIn("tracked_diff_sha256", text)
        self.assertIn("configuration_compatibility", text)
        self.assertNotIn('"chain_of_thought_max_tokens"', text)


if __name__ == "__main__":
    unittest.main()
