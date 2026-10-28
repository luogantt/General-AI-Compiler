#!/usr/bin/env python3
"""CPU-only contract tests. Synthetic arrays/fake CUDA do not validate model PTQ.

Run: python -m unittest discover -s tests -p test_quantization.py
Requires NumPy; does not import TensorRT, Torch, or a CUDA driver.
"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from bevformer_trt.quantization import CalibrationDataset, create_calibrator


class FakeCalibratorBase:
    pass


class FakeDriver:
    def __init__(self):
        self.allocations = []
        self.copies = []
        self.releases = []
        self.fail_copy = False

    def malloc(self, nbytes):
        self.allocations.append(nbytes)
        return 4096

    def copy_host_to_device(self, address, array):
        if self.fail_copy:
            raise RuntimeError("injected CUDA copy failure")
        self.copies.append((address, array.copy()))

    def free(self, address):
        self.releases.append(address)


class CalibrationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.shape = (1, 6, 3, 2, 3)
        self.images = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        np.save(self.root / "a.npy", self.images)
        np.savez(self.root / "b.npz", images=self.images + 1)
        self.manifest = {
            "schema_version": 1, "source_kind": "synthetic_interface_test",
            "dataset": "Generated arrays for CPU interface tests only",
            "preprocessing": "None; no model calibration or accuracy claim",
            "files": ["a.npy", "b.npz"],
        }
        self.save_manifest()
        self.trt = SimpleNamespace(__version__="10.13.3.9", IInt8EntropyCalibrator2=FakeCalibratorBase)
        self.cache = self.root / "calibration.cache"
        self.payload = b"TRT-10133-test-cache\nscale: 1234\x00"

    def save_manifest(self):
        (self.root / "calibration.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def make(self, driver=None, trt=None, **changes):
        options = dict(directory=self.root, model_sha256="a" * 64, policy_fingerprint="b" * 64,
                       input_shape=self.shape, cache_path=self.cache, driver=driver or FakeDriver(),
                       allow_synthetic_for_testing=True)
        options.update(changes)
        calibrator, metadata = create_calibrator(trt or self.trt, **options)
        self.addCleanup(calibrator.close)
        return calibrator, metadata

    def complete_and_cache(self):
        calibrator, metadata = self.make()
        self.assertIsNone(calibrator.read_calibration_cache())
        self.assertEqual(calibrator.get_batch(["images"]), [4096])
        self.assertEqual(calibrator.get_batch(["images"]), [4096])
        self.assertIsNone(calibrator.get_batch(["images"]))
        calibrator.assert_complete()
        calibrator.write_calibration_cache(self.payload)
        self.assertTrue(metadata["cache_written"])
        return calibrator, metadata

    def test_requires_manifest_and_explicit_real_data_declaration(self):
        with self.assertRaisesRegex(ValueError, "real_preprocessed"):
            self.make(allow_synthetic_for_testing=False)
        self.manifest["source_kind"] = "random"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "real_preprocessed"):
            self.make()
        (self.root / "calibration.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.make()

    def test_requires_preprocessing_description_and_explicit_unique_files(self):
        for changes in ({"preprocessing": ""}, {"dataset": ""}, {"files": []},
                        {"files": ["a.npy", "a.npy"]}, {"files": ["../outside.npy"]}):
            with self.subTest(changes=changes):
                original = dict(self.manifest)
                self.manifest.update(changes)
                self.save_manifest()
                with self.assertRaises(ValueError):
                    self.make()
                self.manifest = original

    def test_rejects_dtype_shape_and_nonfinite_before_gpu_allocation(self):
        candidates = [self.images.astype(np.float16), np.zeros((1, 3), dtype=np.float32),
                      np.full(self.shape, np.nan, dtype=np.float32),
                      np.full(self.shape, np.inf, dtype=np.float32)]
        for index, images in enumerate(candidates):
            with self.subTest(candidate=index):
                np.save(self.root / "a.npy", images)
                driver = FakeDriver()
                with self.assertRaises(ValueError):
                    self.make(driver=driver)
                self.assertEqual(driver.allocations, [])

    def test_npz_key_must_match_input(self):
        np.savez(self.root / "b.npz", wrong=self.images)
        with self.assertRaisesRegex(ValueError, "expected only key"):
            self.make()

    def test_production_shape_contract_is_not_weakened_by_toy_tests(self):
        self.manifest["source_kind"] = "real_preprocessed"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "Base calibration requires"):
            CalibrationDataset.load(self.root, input_shape=self.shape)

    def test_batches_reuse_one_buffer_and_close_once(self):
        driver = FakeDriver()
        calibrator, metadata = self.make(driver=driver)
        self.assertEqual(calibrator.get_batch_size(), 1)
        self.assertEqual(driver.allocations, [])
        for _ in range(2):
            self.assertEqual(calibrator.get_batch(["images"]), [4096])
        self.assertIsNone(calibrator.get_batch(["images"]))
        self.assertEqual(driver.allocations, [self.images.nbytes])
        np.testing.assert_array_equal(driver.copies[0][1], self.images)
        np.testing.assert_array_equal(driver.copies[1][1], self.images + 1)
        self.assertEqual(metadata["consumed_batches"], 2)
        self.assertEqual(metadata["source_kind"], "synthetic_interface_test")
        self.assertFalse(metadata["cache_hit"])
        calibrator.assert_complete()
        calibrator.close()
        calibrator.close()
        self.assertEqual(driver.releases, [4096])

    def test_callback_errors_remain_visible_after_exception(self):
        for copy_failure in (False, True):
            with self.subTest(copy_failure=copy_failure):
                driver = FakeDriver()
                driver.fail_copy = copy_failure
                calibrator, metadata = self.make(driver=driver)
                with self.assertRaises((ValueError, RuntimeError)):
                    calibrator.get_batch(["images"] if copy_failure else ["wrong"])
                self.assertIsNotNone(metadata["error"])
                self.assertEqual(metadata["consumed_batches"], 0)
                with self.assertRaises(RuntimeError):
                    calibrator.raise_if_error()
                with self.assertRaises(RuntimeError):
                    calibrator.assert_complete()

    def test_partial_dataset_cannot_be_saved_as_complete_cache(self):
        calibrator, metadata = self.make()
        calibrator.get_batch(["images"])
        with self.assertRaisesRegex(RuntimeError, "did not consume"):
            calibrator.assert_complete()
        with self.assertRaisesRegex(RuntimeError, "did not consume"):
            calibrator.write_calibration_cache(self.payload)
        self.assertIsNotNone(metadata["error"])
        self.assertFalse(self.cache.exists())

    def test_valid_cache_hit_requires_no_device_allocation(self):
        self.complete_and_cache()
        driver = FakeDriver()
        calibrator, metadata = self.make(driver=driver)
        self.assertEqual(calibrator.read_calibration_cache(), self.payload)
        calibrator.assert_complete()
        self.assertTrue(metadata["cache_hit"])
        self.assertEqual(metadata["consumed_batches"], 0)
        self.assertEqual(driver.allocations, [])

    def test_cache_rejects_model_policy_version_and_dataset_changes(self):
        self.complete_and_cache()
        for changes in ({"model_sha256": "c" * 64}, {"policy_fingerprint": "d" * 64},
                        {"trt": SimpleNamespace(__version__="10.13.2", IInt8EntropyCalibrator2=FakeCalibratorBase)}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "does not match"):
                self.make(**changes)
        np.save(self.root / "a.npy", self.images + 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.make()

    def test_cache_rejects_payload_corruption(self):
        self.complete_and_cache()
        self.cache.write_bytes(self.cache.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.make()

    def test_input_changed_after_preflight_is_rejected(self):
        calibrator, metadata = self.make()
        np.save(self.root / "a.npy", self.images + 3)
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            calibrator.get_batch(["images"])
        self.assertIsNotNone(metadata["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
