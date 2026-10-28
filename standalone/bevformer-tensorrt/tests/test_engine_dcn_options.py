"""Reject incompatible DCN build policies before loading CUDA/TensorRT."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import onnx
from onnx import helper, TensorProto

from bevformer_trt import engine


class GPUInitializationBoundary(RuntimeError):
    pass


class EngineDCNOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.onnx'
        value = helper.make_tensor_value_info('images', TensorProto.FLOAT, [1])
        result = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1])
        graph = helper.make_graph([helper.make_node('Identity', ['images'], ['output'])],
                                  'input-policy-fixture', [value], [result])
        onnx.save(helper.make_model(graph), self.source)
        self.ranges = self.root / 'ranges.json'
        self.ranges.write_text('{}', encoding='utf-8')
        self.config = self.root / 'dcn.json'
        self.config.write_text(json.dumps({'node_names': ['explicit-node'], 'block_threads': 128,
                                          'specialize': 4}), encoding='utf-8')

    def args(self, **changes):
        values = dict(onnx=self.source, engine=self.root / 'model.engine', report=self.root / 'build.json',
                      precision='fp16', plan_only=False, dcn_config=self.config, plugins_fp32=False,
                      calibrated_ranges=None, calibration_dir=None, calibration_cache=None,
                      workspace_mib=1024, max_aux_streams=0)
        values.update(changes)
        return SimpleNamespace(**values)

    def assert_rejected_before_cuda(self, pattern, **changes):
        with patch.object(engine, 'initialize_builder') as initialize:
            with self.assertRaisesRegex(ValueError, pattern):
                engine.build(self.args(**changes))
            initialize.assert_not_called()
        self.assertFalse((self.root / 'model.engine').exists())
        self.assertFalse((self.root / 'build.json').exists())

    def test_fp32_and_fp32_plugin_override_rejected(self):
        self.assert_rejected_before_cuda('FP16 plugin edges', precision='fp32')
        self.assert_rejected_before_cuda('FP16 plugin edges', plugins_fp32=True)
        self.assert_rejected_before_cuda('FP16 plugin edges', precision='mixed-int8',
                                         calibrated_ranges=self.ranges, plugins_fp32=True)

    def test_mixed_requires_frozen_ranges_and_rejects_recalibration(self):
        self.assert_rejected_before_cuda('frozen', precision='mixed-int8')
        for path_option in ('calibration_dir', 'calibration_cache'):
            for ranges in (None, self.ranges):
                with self.subTest(option=path_option, ranges=ranges):
                    self.assert_rejected_before_cuda('recalibration is a separate experiment',
                        precision='mixed-int8', calibrated_ranges=ranges,
                        **{path_option: self.root / 'new-calibration'})

    def test_allowed_policies_reach_gpu_boundary_without_altering_sources(self):
        snapshots = {path: path.read_bytes() for path in (self.source, self.ranges, self.config)}
        cases = ({}, {'precision': 'mixed-int8', 'calibrated_ranges': self.ranges},
                 {'dcn_config': None, 'precision': 'fp32'})
        for changes in cases:
            with self.subTest(changes=changes), patch.object(engine, 'initialize_builder',
                    side_effect=GPUInitializationBoundary('CUDA boundary reached')) as initialize:
                with self.assertRaises(GPUInitializationBoundary):
                    engine.build(self.args(**changes))
                initialize.assert_called_once()
        for path, payload in snapshots.items():
            self.assertEqual(path.read_bytes(), payload)
        self.assertFalse((self.root / 'model.engine').exists())
        self.assertFalse((self.root / 'build.json').exists())

    def test_cli_passes_explicit_config_and_defaults_to_disabled(self):
        fake_report = {'status': 'test_only', 'scope': 'CPU parser test', 'policy_fingerprint': 'test'}
        required = ['build', '--onnx', str(self.source), '--engine', str(self.root / 'model.engine'),
                    '--report', str(self.root / 'build.json'), '--precision', 'fp16']
        for extra, expected in (([], None), (['--dcn-config', str(self.config)], self.config)):
            with contextlib.redirect_stdout(io.StringIO()), patch.object(engine, 'build',
                    return_value=(fake_report, 0)) as build:
                self.assertEqual(engine.main([*required, *extra]), 0)
            self.assertEqual(build.call_args.args[0].dcn_config, expected)


if __name__ == '__main__':
    unittest.main()
