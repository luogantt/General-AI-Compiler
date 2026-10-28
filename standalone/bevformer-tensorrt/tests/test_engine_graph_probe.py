"""CPU contracts for full-engine graph probe; no CUDA/TensorRT imports required."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("engine_graph_probe", ROOT / "scripts" / "benchmark_engine_graph.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class GraphProbeTests(unittest.TestCase):
    def test_exact_rejects_signed_zero_and_nonfinite(self):
        reference = {"output": np.array([0.0, 2.0], dtype=np.float16)}
        self.assertTrue(PROBE.compare_outputs(reference, reference, np)["passed"])
        signed = {"output": np.array([-0.0, 2.0], dtype=np.float16)}
        self.assertFalse(PROBE.compare_outputs(reference, signed, np)["passed"])
        for value in (np.nan, np.inf):
            bad = {"output": np.array([value], dtype=np.float16)}
            self.assertFalse(PROBE.compare_outputs(bad, bad, np)["passed"])

    def test_exact_rejects_dtype_shape_and_missing_outputs(self):
        reference = {"output": np.array([1, 2], dtype=np.float16)}
        for array in (np.array([1, 2], dtype=np.float32), np.array([[1, 2]], dtype=np.float16)):
            self.assertFalse(PROBE.compare_outputs(reference, {"output": array}, np)["passed"])
        self.assertFalse(PROBE.compare_outputs({}, {}, np)["passed"])
        with self.assertRaises(ValueError):
            PROBE.compare_outputs(reference, {}, np)

    def test_only_capture_errors_are_not_supported(self):
        def check(*errors):
            return PROBE.capture_is_unsupported([{"error": message} for message in errors])
        self.assertTrue(check("CUDA error: operation not permitted when stream is capturing"))
        self.assertFalse(check("TensorRT execute_async_v3 returned failure", "stream capture is invalidated"))
        self.assertTrue(check("CUDA error: operation not permitted when stream is capturing\n"
                              "CUDA kernel errors might be asynchronously reported. Compile with device-side assertions."))
        self.assertFalse(check("TensorRT execute_async_v3 returned failure"))
        self.assertFalse(check("stream capture invalidated", "CUDA out of memory"))
        self.assertFalse(check("CUDA illegal memory access during stream capture"))
        self.assertFalse(check("Outputs differ from ordinary execution"))


if __name__ == "__main__":
    unittest.main()
