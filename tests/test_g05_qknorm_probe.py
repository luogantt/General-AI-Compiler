"""Pure report-gating checks; no Torch/NPU imports or benchmark execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/probe_ae_qknorm_rope.py"
SPEC = importlib.util.spec_from_file_location("g05_qknorm_probe_status", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeGatingTests(unittest.TestCase):
    def row(self):
        return {"heads": 8, "rope_length": 32, "variants": {
            name: {"passed": True, "failures": []} for name in
            ("original_eager", "native_eager", "ascendc_eager", "original_ge", "native_ge", "ascendc_ge")}}

    def test_ge_controls_cannot_override_candidate_or_reference_status(self):
        row = self.row()
        for name in ("original_ge", "native_ge"):
            row["variants"][name] = {"passed": False, "failures": [{"case": "W_large_1", "reasons": ["relative_l2"]}]}
        summary = PROBE.summarize_validation([row], True)
        self.assertEqual(summary["status"], "success")
        self.assertTrue(summary["candidate_validation"]["passed"])
        self.assertEqual(summary["control_observations"]["status"], "deviations_observed")
        self.assertEqual(len(summary["control_observations"]["deviations"]), 2)
        row["variants"]["ascendc_ge"]["passed"] = False
        self.assertEqual(PROBE.summarize_validation([row], True)["status"], "candidate_validation_failed")
        row["variants"]["native_eager"]["passed"] = False
        self.assertEqual(PROBE.summarize_validation([row], True)["status"], "reference_validation_failed")

    def test_missing_required_ge_or_empty_cases_cannot_pass(self):
        row = self.row()
        del row["variants"]["ascendc_ge"]
        self.assertEqual(PROBE.summarize_validation([row], True)["status"], "candidate_validation_failed")
        self.assertEqual(PROBE.summarize_validation([row], False)["status"], "success")
        self.assertEqual(PROBE.summarize_validation([], True)["status"], "candidate_validation_failed")

    def test_candidate_thresholds_and_failure_details_are_unchanged(self):
        metric = {"finite": True, "exact": True, "max_abs": 0., "relative_l2": 0., "rmse": 0.}
        check = {"case": "W_1", "vs_original": dict(metric), "vs_native": dict(metric), "max_abs_limit": .03125}
        graph = {"dynamo_backend_compilations": 1}
        self.assertTrue(PROBE.assess_variant([check], .001, graph, True)[0])
        bad = copy.deepcopy(check)
        bad["vs_original"]["relative_l2"] = .00185
        passed, failures = PROBE.assess_variant([bad], .001, graph, True)
        self.assertFalse(passed)
        self.assertEqual(failures[0]["reference"], "vs_original")
        self.assertEqual(failures[0]["relative_l2_limit"], .001)
        self.assertEqual(failures[0]["reasons"], ["relative_l2"])
        bad["vs_original"]["relative_l2"] = 0.
        bad["vs_native"]["max_abs"] = .03126
        self.assertFalse(PROBE.assess_variant([bad], .001, graph, True)[0])
        self.assertFalse(PROBE.assess_variant([check], .001, {"dynamo_backend_compilations": 2}, True)[0])
        self.assertFalse(PROBE.assess_variant([check], .001, graph, False)[0])

    def test_nonfinite_reference_metrics_and_absent_checks_are_rejected(self):
        metric = {"finite": False, "exact": False, "max_abs": float("nan"), "relative_l2": float("nan"), "rmse": float("nan")}
        check = {"case": "A_1", "vs_original": metric, "vs_native": metric, "max_abs_limit": .03125}
        self.assertFalse(PROBE.assess_variant([check], .001, None, True)[0])
        self.assertFalse(PROBE.assess_variant([], .001, None, True)[0])


if __name__ == "__main__":
    unittest.main()
