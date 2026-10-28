"""CPU checks for the independent oracle and deterministic GPU test fixtures."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'check_dcn_kernel.py'
SPEC = importlib.util.spec_from_file_location('_check_dcn_kernel_test', PATH)
CHECK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK
SPEC.loader.exec_module(CHECK)


class DcnKernelOracleTests(unittest.TestCase):
    def test_fractional_sampling_padding_and_mask(self):
        case = CHECK.Case('hand', 1, 1, 2, 2, kernel_h=1, kernel_w=1, pad_h=0, pad_w=0)
        image = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        offset = np.zeros((1, 2, 2, 2), dtype=np.float32)
        offset[0, :, 0, 0] = [0.5, 0.5]
        offset[0, :, 0, 1] = [-0.5, -1.5]
        offset[0, :, 1, 0] = [1, 0]
        mask = np.ones((1, 1, 2, 2), dtype=np.float32)
        mask[0, 0, 0, 0] = 0.25
        actual = CHECK.fp32_oracle(case, {'input': image, 'offset': offset, 'mask': mask})
        np.testing.assert_array_equal(actual, np.array([[[[0.625, 0.25], [0, 4]]]]))

    def test_batch_group_and_column_layout(self):
        case = CHECK.Case('layout', 2, 6, 1, 1, kernel_h=1, kernel_w=1, pad_h=0, pad_w=0, deform_groups=2)
        image = np.arange(1, 13, dtype=np.float32).reshape(2, 6, 1, 1)
        offset = np.zeros((2, 4, 1, 1), dtype=np.float32)
        mask = np.array([[0.5, 2], [3, -1]], dtype=np.float32).reshape(2, 2, 1, 1)
        actual = CHECK.fp32_oracle(case, {'input': image, 'offset': offset, 'mask': mask})
        expected = np.array([[0.5, 21], [1, 24], [1.5, 27], [8, -10], [10, -11], [12, -12]]).reshape(6, 2, 1, 1)
        np.testing.assert_array_equal(actual, expected)

    def test_fixture_shapes_dtype_and_reproducibility(self):
        for case in CHECK.cases():
            for dtype in (np.float32, np.float16):
                with self.subTest(case=case.name, dtype=dtype):
                    a = CHECK.make_arrays(case, dtype, 127)
                    b = CHECK.make_arrays(case, dtype, 127)
                    self.assertEqual(len(case.arguments), 15)
                    self.assertGreater(case.out_h * case.out_w, 0)
                    k = case.kernel_h * case.kernel_w
                    self.assertEqual(a['offset'].shape, (case.batch, 2 * case.deform_groups * k, case.out_h, case.out_w))
                    for name in a:
                        self.assertEqual(a[name].dtype, dtype)
                        self.assertTrue(a[name].flags.c_contiguous)
                        self.assertEqual(CHECK.array_digest(a[name]), CHECK.array_digest(b[name]))
                    self.assertTrue(np.isfinite(a['input']).all())
                    self.assertTrue(np.isfinite(a['mask']).all())

    def test_fp32_oracle_rejects_half_and_masks_nonfinite_coordinates(self):
        case = CHECK.Case('nan', 1, 1, 1, 1, kernel_h=1, kernel_w=1, pad_h=0, pad_w=0)
        arrays = {'input': np.ones((1, 1, 1, 1), np.float32),
                  'offset': np.full((1, 2, 1, 1), np.nan, np.float32),
                  'mask': np.ones((1, 1, 1, 1), np.float32)}
        np.testing.assert_array_equal(CHECK.fp32_oracle(case, arrays), np.zeros((1, 1, 1, 1)))
        with self.assertRaises(ValueError):
            CHECK.fp32_oracle(case, {n: a.astype(np.float16) for n, a in arrays.items()})

    def test_failure_is_archived_and_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            args = SimpleNamespace(library=folder / 'missing.so', report=folder / 'report.json',
                                   seed=0, device=0, blocks=[128])
            with self.assertRaises(FileNotFoundError):
                CHECK.run(args)
            archive = args.report.read_bytes()
            report = json.loads(archive)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error']['type'], 'FileNotFoundError')
            with self.assertRaises(FileExistsError):
                CHECK.run(args)
            self.assertEqual(args.report.read_bytes(), archive)


if __name__ == '__main__':
    unittest.main()
