"""CPU unit tests for the result checker, not CUDA operator validation."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_cuda_tests import compare_outputs


class CudaOutputChecks(unittest.TestCase):
    def compare(self, actual, expected):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            actual.tofile(directory / "output_0.bin")
            np.savez(directory / "reference.npz", Y=expected)
            plan = {"outputs": ["Y"], "tensors": [{"name": "Y", "shape": list(actual.shape),
                    "dtype": str(actual.dtype), "bytes": actual.nbytes}]}
            return compare_outputs(plan, directory, directory / "reference.npz", 2e-5, 2e-5)["Y"]

    def test_finite_float_tolerance(self):
        result = self.compare(np.array([1.000001], dtype="float32"), np.array([1.0], dtype="float32"))
        self.assertTrue(result["passed"])
        self.assertIn("sha256", result["artifact"])

    def test_large_integer_requires_exact_equality(self):
        result = self.compare(np.array([2**60 + 1], dtype="int64"), np.array([2**60], dtype="int64"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatched_elements"], 1)
        self.assertEqual(result["max_abs_error"], 1)

    def test_nan_is_not_a_pass(self):
        self.assertFalse(self.compare(np.array([np.nan], dtype="float32"), np.array([np.nan], dtype="float32"))["passed"])

    def test_precision_dtype_boundary(self):
        self.assertFalse(self.compare(np.array([1], dtype="float32"), np.array([1], dtype="float16"))["passed"])

    def test_shape_is_not_broadcast(self):
        self.assertFalse(self.compare(np.array([[1, 1]], dtype="float32"), np.array([1, 1], dtype="float32"))["passed"])

    def test_numerical_error_fails(self):
        self.assertFalse(self.compare(np.array([2], dtype="float32"), np.array([1], dtype="float32"))["passed"])

    def test_wrong_reference_names_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory / "reference.npz", wrong=np.ones(1, dtype="float32"))
            with self.assertRaisesRegex(ValueError, "Reference names"):
                compare_outputs({"outputs": ["Y"], "tensors": []}, directory,
                                directory / "reference.npz", 2e-5, 2e-5)

    def test_truncated_output_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.savez(directory / "reference.npz", Y=np.ones(1, dtype="float32"))
            (directory / "output_0.bin").write_bytes(b"\x00")
            with self.assertRaisesRegex(ValueError, "byte count"):
                compare_outputs({"outputs": ["Y"], "tensors": [{"name": "Y", "bytes": 4}]},
                                directory, directory / "reference.npz", 2e-5, 2e-5)


if __name__ == "__main__":
    unittest.main()
