"""CPU regressions for observed NVIDIA FP16 arithmetic and benchmark acceptance."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "msda_fp16_hfma" / "capture.npz"
INPUT_KEYS = ("value", "shapes", "starts", "locations", "weights")


class CPUArray:
    """The oracle only needs these two data-access methods, never a GPU tensor."""

    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def load_benchmark():
    spec = importlib.util.spec_from_file_location("msda_benchmark_cpu_regression", ROOT / "scripts" / "benchmark_msda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MSDAOracleTests(unittest.TestCase):
    def test_half_fma_oracle_matches_captured_nvidia_bits_without_gpu_imports(self):
        imported_before = set(sys.modules)
        benchmark = load_benchmark()
        with np.load(FIXTURE, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {*INPUT_KEYS, "expected"})
            inputs = {key: CPUArray(archive[key]) for key in INPUT_KEYS}
            expected = archive["expected"]
        actual = benchmark.oracle(inputs)
        self.assertEqual(actual.dtype, np.dtype("float16"))
        self.assertEqual(actual.shape, (2, 7, 2, 5))
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(np.isfinite(actual).all())
        # Comparing uint16 includes signed-zero bits and rejects even one ULP.
        np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
        new_imports = set(sys.modules) - imported_before
        self.assertFalse(any(name.split(".")[0] in ("torch", "tensorrt") for name in new_imports))

    def test_identical_infinities_never_pass_acceptance(self):
        benchmark = load_benchmark()
        identical = np.array([np.inf, -np.inf], dtype=np.float16)
        comparison = {"finite": bool(np.isfinite(identical).all()),
                      "allclose": bool(np.allclose(identical, identical)),
                      "exact": bool(np.array_equal(identical, identical))}
        self.assertTrue(comparison["allclose"] and comparison["exact"])
        self.assertFalse(benchmark.accepted(comparison))
        self.assertFalse(benchmark.accepted(comparison, exact=True))
        self.assertTrue(benchmark.accepted({"finite": True, "allclose": True, "exact": True}, exact=True))
        self.assertFalse(benchmark.accepted({"finite": True, "allclose": True, "exact": False}, exact=True))


if __name__ == "__main__":
    unittest.main()
