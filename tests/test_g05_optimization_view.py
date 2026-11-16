from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/galaxeavla/build_optimization_view.py"
SPEC = importlib.util.spec_from_file_location("build_optimization_view", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PATHS = [MODULE.EVIDENCE / name for name in (
    "reference-3090/bf16resident_steps10_compileae.json", "libero-bf16-eager/report.json",
    "libero-bf16-torchair-ae/report.json", "libero-bf16-npugraph-frozen-ae-vlm/report.json")]
GE_PATH = MODULE.EVIDENCE / "libero-bf16-ge-frozen-compatible/report.json"
CONTEXT_PATH = MODULE.EVIDENCE / "libero-bf16-graph-triangular-context/report.json"
MIXED_PATH = MODULE.EVIDENCE / "libero-bf16-ge-ae-acl-vlm-triangular-context/report.json"


def synthetic_workload(report):
    # Deliberately tiny test arithmetic; never written into production evidence.
    return {
        "identity": copy.deepcopy(report["comparison_identity"]), "source_revision": report["source_revision"],
        "diagnostic_requests": 1, "not_counted": ["QA functional operations omitted"],
        "summary": {"counted_flops": 30, "stages": {"prefill": {"counted_flops": 10}, "fm": {"counted_flops": 20}}},
        "operators": [{"stage": "prefill", "counted_flops": 10}, {"stage": "fm", "counted_flops": 20}],
        "parameters": {"total_numel": 100},
    }


class OptimizationViewTests(unittest.TestCase):
    def test_four_real_reports_choose_fastest_ascend_and_label_actual_frozen_options(self):
        data = MODULE.make_data(PATHS)
        self.assertEqual(data["defaultId"], "extra-1")
        best = data["configs"][-1]
        self.assertEqual(best["label"], "910B2 · NPUGraph AE + VLM · frozen")
        self.assertAlmostEqual(best["timing"]["total"]["p50"], 297.8925099596381)
        self.assertTrue(all(config["workload"] is None for config in data["configs"]))
        self.assertEqual(best["verification"]["maxAbs"], 0)

    def test_frozen_label_does_not_claim_unfrozen_vlm_is_frozen(self):
        label = MODULE.backend_label("npugraph_ex", "npugraph_ex", {"frozen_parameter": True}, {})
        self.assertEqual(label, "NPUGraph AE + VLM · AE frozen")
        self.assertEqual(MODULE.backend_label("eager", "eager", {}, {}), "eager")

    def test_single_stream_label_names_only_the_enabled_scope(self):
        label = MODULE.backend_label("torchair", "torchair_aclgraph",
                                     {"frozen_parameter": True, "single_stream": True}, {"frozen_parameter": True})
        self.assertEqual(label, "TorchAir AE + TorchAir ACL Graph VLM · frozen · AE single_stream")
        self.assertNotIn("single_stream", MODULE.backend_label("torchair", "eager", {"single_stream": False}, {}))

    def test_norm_and_rope_fields_distinguish_both_fp32_modes(self):
        original = json.loads(MIXED_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for mode, suffix in (("sliced", "切片"), ("matrix", "矩阵")):
                report = copy.deepcopy(original)
                report["ae_norm"] = {"enabled": True, "operator": "torch_npu.npu_rms_norm"}
                report["ae_rope"] = {"enabled": True, "mode": mode}
                report["compilation"]["options"]["single_stream"] = True
                path.write_text(json.dumps(report), encoding="utf-8")
                normalized, _ = MODULE.normalize(path, "candidate")
                self.assertTrue(normalized["aeNorm"])
                self.assertEqual(normalized["aeRope"], mode)
                self.assertEqual(normalized["singleStreamScopes"], ["AE"])
                self.assertIn("NPU RMSNorm (FP32)", normalized["label"])
                self.assertIn("NPU RoPE " + suffix + " (FP32)", normalized["label"])

    def test_enabled_helpers_cannot_be_used_as_an_eager_baseline_even_when_first(self):
        for flag in ("ae_norm", "ae_rope", "functional_mrope"):
            report = json.loads(MIXED_PATH.read_text())
            report["compilation"] = {"name": "eager", "options": {}}
            report["vlm_compilation"] = {"name": "eager", "options": {}}
            report["gdn_algorithm"] = {"name": "original"}
            for key in MODULE.EXPERIMENT_FLAGS:
                report[key] = {"enabled": key == flag}
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                data = MODULE.make_data([PATHS[0], path, PATHS[2], PATHS[1]])
                self.assertTrue(all(row["speedupBaselineId"] == "extra-1" for row in data["configs"]))
                with self.assertRaisesRegex(ValueError, "unmodified eager baseline"):
                    MODULE.make_data([PATHS[0], path, PATHS[2]])

    def test_functional_mrope_field_and_label_require_explicit_enable(self):
        report = json.loads(MIXED_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            for enabled in (False, True):
                report["functional_mrope"] = {"enabled": enabled, "scope": "AE and VLM MRoPE position construction"}
                path.write_text(json.dumps(report), encoding="utf-8")
                normalized, _ = MODULE.normalize(path, "candidate")
                self.assertEqual(normalized["functionalMrope"], enabled)
                self.assertEqual(normalized["functionalMropeScope"], report["functional_mrope"]["scope"])
                self.assertEqual("函数式 MRoPE" in normalized["label"], enabled)

    def test_mixed_ge_acl_graph_report_is_named_distinctly_and_selected_without_old_flops(self):
        data = MODULE.make_data([*PATHS, GE_PATH, CONTEXT_PATH, MIXED_PATH])
        self.assertEqual(data["defaultId"], "extra-4")
        best = data["configs"][-1]
        self.assertEqual(best["label"], "910B2 · TorchAir AE + TorchAir ACL Graph VLM · frozen · GDN 三角解 · 移除无用编码")
        self.assertAlmostEqual(best["timing"]["total"]["p50"], 225.58002988807857)
        self.assertIsNone(best["workload"])
        self.assertEqual(len(best["verification"]["cases"]), 8)
        self.assertTrue(best["verification"]["passed"])

    def test_workload_requires_complete_exact_identity_including_noise_ids_and_sidecars(self):
        report = json.loads(PATHS[-1].read_text())
        workload = synthetic_workload(report)
        normalized = MODULE.validate_workload(workload, report["comparison_identity"])
        self.assertEqual(normalized["stageFlops"], {"prefill": 10, "action": 20})
        for field in MODULE.IDENTITY_FIELDS:
            changed = copy.deepcopy(workload)
            changed["identity"][field] = "wrong"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                MODULE.validate_workload(changed, report["comparison_identity"])
        missing = copy.deepcopy(workload)
        del missing["identity"]["input_files_sha256"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            MODULE.validate_workload(missing, report["comparison_identity"])

    def test_workload_total_and_stages_must_match_operator_records(self):
        report = json.loads(PATHS[-1].read_text())
        workload = synthetic_workload(report)
        workload["operators"][0]["counted_flops"] = 11
        with self.assertRaisesRegex(ValueError, "reconcile"):
            MODULE.validate_workload(workload, report["comparison_identity"])

    def test_embedded_workload_shares_only_with_exact_identity_never_historical_reference(self):
        report = json.loads(PATHS[-1].read_text())
        report["workload"] = synthetic_workload(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            data = MODULE.make_data([*PATHS[:3], path])
        self.assertIsNone(data["configs"][0]["workload"])
        for config in data["configs"][1:]:
            self.assertEqual(config["workload"]["countedFlops"], 30)
            self.assertTrue(config["workload"]["identityMatched"])
        self.assertTrue(data["configs"][-1]["workload"]["sameReport"])
        self.assertFalse(data["configs"][1]["workload"]["sameReport"])

    def test_mismatched_explicit_workload_is_rejected_instead_of_displayed(self):
        report = json.loads(PATHS[-1].read_text())
        workload = synthetic_workload(report)
        workload["identity"]["input_files_sha256"]["image.png"] = "unrelated-image"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(workload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matches no report identity"):
                MODULE.make_data(PATHS, [path])

    def test_context_execution_blocks_reuse_of_codec_workload_despite_same_input_identity(self):
        data = MODULE.make_data([*PATHS, GE_PATH, CONTEXT_PATH])
        self.assertEqual(data["defaultId"], "extra-3")
        for config in data["configs"][1:5]:
            self.assertEqual(config["workload"]["countedFlops"], 1064344764416)
        context = data["configs"][-1]
        self.assertIsNone(context["workload"])
        self.assertIn("GDN 三角解 · 移除无用编码", context["label"])
        self.assertEqual(len(context["verification"]["cases"]), 8)
        self.assertTrue(context["verification"]["passed"])
        self.assertAlmostEqual(context["timing"]["total"]["p50"], 264.46218974888325)
        # A count collected with the same execution choice is permitted. The
        # deliberately tiny fixture arithmetic never enters production evidence.
        report = json.loads(CONTEXT_PATH.read_text())
        report["workload"] = synthetic_workload(report)
        report["workload"]["execution"] = {"context_optimization": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            data = MODULE.make_data([*PATHS, GE_PATH, path])
        self.assertEqual(data["configs"][-1]["workload"]["countedFlops"], 30)
        self.assertEqual(data["configs"][1]["workload"]["countedFlops"], 1064344764416)

    def test_context_optimization_requires_complete_state_task_checks_and_matching_tokens(self):
        original = json.loads(CONTEXT_PATH.read_text())
        for problem in ("missing_task_replay", "token_mismatch"):
            report = copy.deepcopy(original)
            cases = report["context_verification"]["cases"]
            if problem == "missing_task_replay":
                del cases["task_D_2"]
                message = "state/task replay"
            else:
                cases["state_C_1"]["encoded_context"]["input_ids"] = False
                message = "mismatched encoded"
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.normalize(path, "context")

    def test_native_noise_uses_its_own_eager_baseline(self):
        root = MODULE.EVIDENCE
        native_eager = root / "libero-bf16-eager-native/report.json"
        native_graph = root / "libero-bf16-ge-acl-native/report.json"
        data = MODULE.make_data([*PATHS, native_eager, native_graph])
        candidate = data["configs"][-1]
        baseline = json.loads(native_eager.read_text())["summary_ms"]["total_ms"]["p50"]
        self.assertEqual(candidate["speedupBaselineNoise"], "native")
        self.assertTrue(candidate["speedupSameNoise"])
        self.assertEqual(candidate["speedupBaselineMs"], baseline)
        self.assertAlmostEqual(candidate["speedup"], baseline / candidate["timing"]["total"]["p50"])
        self.assertEqual(data["configs"][1]["speedupBaselineNoise"], "shared")

    def test_static_vision_allows_extra_cases_but_requires_each_camera_replay(self):
        report = json.loads(CONTEXT_PATH.read_text())
        report["static_vision"] = {"enabled": True}
        cases = report["context_verification"]["cases"]
        for name in ("camera_E_1", "camera_E_2", "wrist_F_1", "wrist_F_2"):
            cases[name] = copy.deepcopy(cases["state_C_1"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            normalized, _ = MODULE.normalize(path, "vision")
            self.assertTrue(normalized["staticVision"])
            self.assertIn("固定几何 Vision graph", normalized["label"])
            self.assertEqual(len(normalized["verification"]["cases"]), 12)
            del cases["wrist_F_2"]
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent camera/wrist"):
                MODULE.normalize(path, "vision")

    def test_triton_fused_result_wins_and_slower_exact_time_experiment_is_retained(self):
        additions = [MODULE.EVIDENCE / name / "report.json" for name in (
            "libero-bf16-eager-native", "libero-bf16-ge-acl-native",
            "libero-bf16-ge-acl-native-time", "libero-bf16-native-triton-fused")]
        data = MODULE.make_data([*PATHS, GE_PATH, CONTEXT_PATH, MIXED_PATH, *additions])
        self.assertEqual(len(data["configs"]), 11)
        self.assertEqual(data["defaultId"], "extra-8")
        best, slower = data["configs"][-1], data["configs"][-2]
        self.assertAlmostEqual(best["timing"]["total"]["p50"], 200.69559989497066)
        self.assertIn("GDN Triton", best["label"])
        self.assertIn("NPU 融合注意力", best["label"])
        self.assertIn("时间条件预计算 (exact)", slower["label"])
        self.assertGreater(slower["timing"]["total"]["p50"], data["configs"][-3]["timing"]["total"]["p50"])
        self.assertTrue(best["speedupSameNoise"])
        self.assertEqual(best["speedupBaselineNoise"], "native")
        self.assertIsNone(best["workload"])

    def test_real_static_vision_request_lookup_result_has_all_twelve_cases(self):
        additions = [MODULE.EVIDENCE / name / "report.json" for name in (
            "libero-bf16-eager-native", "libero-bf16-ge-acl-native",
            "libero-bf16-ge-acl-native-time", "libero-bf16-native-triton-fused",
            "libero-bf16-native-triton-fused-time-vision")]
        data = MODULE.make_data([*PATHS, GE_PATH, CONTEXT_PATH, MIXED_PATH, *additions])
        self.assertEqual(len(data["configs"]), 12)
        self.assertEqual(data["defaultId"], "extra-9")
        best = data["configs"][-1]
        self.assertAlmostEqual(best["timing"]["total"]["p50"], 172.49503498896956)
        self.assertEqual(len(best["verification"]["cases"]), 12)
        self.assertTrue(best["verification"]["passed"])
        self.assertTrue(best["staticVision"])
        self.assertIn("时间条件预计算 (request)", best["label"])
        self.assertEqual(best["speedupBaselineNoise"], "native")
        self.assertIsNone(best["workload"])

    def test_final_fifteen_reports_select_p50_with_matching_native_baseline(self):
        additions = [MODULE.EVIDENCE / name / "report.json" for name in (
            "libero-bf16-eager-native", "libero-bf16-ge-acl-native",
            "libero-bf16-ge-acl-native-time", "libero-bf16-native-triton-fused",
            "libero-bf16-native-triton-fused-time-vision",
            "libero-bf16-native-triton-time-vision-single",
            "libero-bf16-native-vision-rope-matrix",
            "libero-bf16-native-vision-rope-norm-mrope")]
        data = MODULE.make_data([*PATHS, GE_PATH, CONTEXT_PATH, MIXED_PATH, *additions])
        self.assertEqual(len(data["configs"]), 15)
        self.assertEqual(data["defaultId"], "extra-12")
        best = data["configs"][-1]
        original = json.loads(additions[-1].read_text())
        self.assertEqual(best["timing"]["total"]["p50"], original["summary_ms"]["total_ms"]["p50"])
        self.assertNotEqual(best["timing"]["total"]["p50"], original["summary_ms"]["total_ms"]["mean"])
        self.assertAlmostEqual(best["timing"]["total"]["p50"], 155.574289849028)
        self.assertEqual((best["count"], best["warmup"]), (20, 5))
        self.assertEqual(best["speedupBaselineNoise"], "native")
        self.assertAlmostEqual(best["speedupBaselineMs"], 1011.5459698718041)
        self.assertTrue(best["speedupSameNoise"])
        self.assertTrue(best["aeNorm"] and best["functionalMrope"])
        self.assertEqual(best["aeRope"], "matrix")
        for label in ("NPU RMSNorm (FP32)", "NPU RoPE 矩阵 (FP32)", "函数式 MRoPE"):
            self.assertIn(label, best["label"])
        self.assertEqual(len(best["verification"]["cases"]), 12)
        self.assertTrue(best["verification"]["passed"])
        self.assertIsNone(best["workload"])
        single = data["configs"][-3]
        self.assertEqual(single["singleStreamScopes"], ["AE"])
        self.assertAlmostEqual(single["timing"]["total"]["p50"], 173.09103021398187)


if __name__ == "__main__":
    unittest.main()
