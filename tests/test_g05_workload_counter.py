from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "galaxeavla" / "count_g05_workload.py"
SPEC = importlib.util.spec_from_file_location("count_g05_workload", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.pop(0)


class TensorShape:
    dtype = "torch.bfloat16"

    def __init__(self, *dimensions):
        self.shape = dimensions


class WorkloadCounterTests(unittest.TestCase):
    def test_linear_counts_batch_and_sequence_once(self):
        self.assertEqual(MODULE.linear_flops([2, 32, 1024], 1024, 4096), 536870912)
        with self.assertRaises(ValueError):
            MODULE.linear_flops([2, 32, 1023], 1024, 4096)

    def test_grouped_conv_uses_per_group_input_width_from_weight(self):
        # 8 input channels and groups=2 -> 4 input channels in each kernel.
        self.assertEqual(MODULE.convolution_flops([2, 16, 8, 8], [16, 4, 3, 3]), 147456)

    def test_adaln_conditions_batch_without_horizon_multiplier(self):
        nn = SimpleNamespace(**{name: type(name, (), {}) for name in ("Linear", "Conv1d", "Conv2d", "Conv3d")})
        counter = MODULE.WorkloadCounter(SimpleNamespace(nn=nn))
        adaln = type("AdaptiveRMSNorm", (), {})()
        adaln.dense = SimpleNamespace(in_features=1024, out_features=3072)
        x, cond = TensorShape(2, 32, 1024), TensorShape(2, 1024)
        with counter.phase("fm"):
            counter.arithmetic_hook("policy.model.action_expert.norm")(adaln, (x, cond), {}, (x, x))
        record = counter.summary()["operators"][0]
        self.assertEqual(record["counted_flops"], 2 * 2 * 1024 * 3072)
        self.assertEqual(record["output_shape"], [2, 3072])
        self.assertEqual(record["output_dtype"], "torch.float32")

    def test_phase_totals_are_disjoint_and_repeated_calls_accumulate(self):
        counter = MODULE.WorkloadCounter(None)
        x, y = TensorShape(1, 2), TensorShape(1, 3)
        for stage in ("cot", "cot", "action_ar"):
            with counter.phase(stage):
                counter.add("policy.model.vlm.output_proj", "Linear", x, y, 12)
        report = counter.summary()
        self.assertEqual(report["counted_flops"], 36)
        self.assertEqual(report["stages"]["cot"]["counted_flops"], 24)
        self.assertEqual(report["stages"]["action_ar"]["counted_flops"], 12)
        self.assertEqual(report["stages"]["cot"]["components"]["vlm"]["calls"], 2)
        self.assertEqual(len(report["operators"]), 2)

    def test_wrappers_preserve_result_arguments_and_inherited_method(self):
        counter = MODULE.WorkloadCounter(None)
        observed = []

        class Target:
            def method(self, value, *, key):
                observed.append((counter.stage, value, key))
                return value

        target = Target()
        value = object()
        counter.wrap(target, "method", "prefill")
        self.assertIs(target.method(value, key=7), value)
        self.assertEqual(observed, [("prefill", value, 7)])
        self.assertEqual(counter.stage, "unscoped")
        counter.close()
        self.assertNotIn("method", vars(target))

    def test_nested_ar_retains_cot_stage_and_restores_on_exception(self):
        counter = MODULE.WorkloadCounter(None)

        def failing():
            self.assertEqual(counter.stage, "cot")
            raise ValueError("expected")

        obj = SimpleNamespace(method=failing)
        counter.wrap(obj, "method", "action_ar")
        with counter.phase("cot"):
            with self.assertRaises(ValueError):
                obj.method()
            self.assertEqual(counter.stage, "cot")
        counter.close()
        self.assertIs(obj.method, failing)
        self.assertEqual(counter.stage, "unscoped")


if __name__ == "__main__":
    unittest.main()
