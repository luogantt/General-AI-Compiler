from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from gac.errors import ValidationError
from gac.golden import (
    GoldenBundle,
    TensorTolerance,
    compare_bundle,
    compare_tensor,
    read_golden,
    write_golden,
)


class GoldenTests(unittest.TestCase):
    def test_metrics_and_thresholds(self):
        expected = np.asarray([1.0, 2.0, 3.0], np.float32)
        close = expected + np.asarray([1e-6, -1e-6, 0], np.float32)
        report = compare_tensor(close, expected)
        self.assertTrue(report.passed)
        self.assertGreater(report.cosine_similarity, 0.99999)
        failed = compare_tensor(
            expected + 1,
            expected,
            TensorTolerance(atol=0, rtol=0, cosine_min=0.99999),
        )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.close_fraction, 0)

    def test_roundtrip_is_deterministic_and_checksums_are_enforced(self):
        bundle = GoldenBundle(
            {
                "backbone/c5": np.arange(12, dtype=np.float32).reshape(3, 4),
                "encoder/layer_0": np.ones((2, 3), np.float16),
            },
            {"checkpoint_sha256": "abc", "frame": 0},
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.gacgolden"
            second = Path(directory) / "second.gacgolden"
            write_golden(first, bundle)
            write_golden(second, bundle)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            restored = read_golden(first)
            self.assertEqual(restored.metadata, bundle.metadata)
            for name in bundle.tensors:
                np.testing.assert_array_equal(restored.tensors[name], bundle.tensors[name])
            report = compare_bundle(restored.tensors, restored)
            self.assertTrue(all(item.passed for item in report.values()))

            corrupt = Path(directory) / "corrupt.gacgolden"
            with ZipFile(first, "r") as source, ZipFile(corrupt, "w", ZIP_DEFLATED) as target:
                for name in source.namelist():
                    payload = source.read(name)
                    if name.endswith("c5.npy"):
                        payload = payload[:-1] + bytes([payload[-1] ^ 1])
                    target.writestr(name, payload)
            with self.assertRaisesRegex(ValidationError, "checksum"):
                read_golden(corrupt)

    def test_rejects_bad_names_and_stage_sets(self):
        with self.assertRaises(ValidationError):
            GoldenBundle({"../escape": np.zeros(1)}, {})
        bundle = GoldenBundle({"a": np.zeros(1)}, {})
        with self.assertRaises(ValidationError):
            compare_bundle({"b": np.zeros(1)}, bundle)


if __name__ == "__main__":
    unittest.main()
