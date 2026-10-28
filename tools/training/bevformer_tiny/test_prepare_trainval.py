"""Small CPU-only checks for the preparation guards (no dataset or GPU needed)."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location(
    'prepare_trainval', str(Path(__file__).with_name('prepare_trainval.py')))
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class PreparationTests(unittest.TestCase):
    def test_can_cache_preserves_upstream_dictionary_mutation(self):
        pose = {'utime': 1, 'pos': [1, 2, 3], 'orientation': [1, 0, 0, 0],
                'accel': [0, 0, 0], 'rotation_rate': [0, 0, 0], 'vel': [0, 0, 0]}
        api = mock.Mock()
        api.get_messages.return_value = [pose]
        cache = prep.CheckedCanBus()
        cache.select(api, 'scene-0001')
        first = cache.get_messages('scene-0001', 'pose')
        first[0].pop('utime')
        first[0].pop('pos')
        cache.select(api, 'scene-0001')
        self.assertEqual(cache.get_messages('scene-0001', 'pose'), [pose])
        api.get_messages.assert_called_once()

    def test_missing_can_does_not_silently_fall_back(self):
        api = mock.Mock()
        api.get_messages.side_effect = FileNotFoundError('missing pose file')
        with self.assertRaises(FileNotFoundError):
            prep.CheckedCanBus().select(api, 'scene-0001')

    def test_download_report_requires_full_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report = {'status': 'passed', 'error_count': 0, 'version': prep.VERSION,
                      'root': str(root), 'splits': {'checked': True},
                      'sensor_files': {'references_checked': 2631083, 'valid_files': 2631083},
                      'can_bus': {'valid_pose_scenes': 835}}
            path = root / 'validation-report.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertEqual(prep.check_download_report(root)['valid_can_scenes'], 835)
            report['sensor_files']['valid_files'] -= 1
            path.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                prep.check_download_report(root)

    def test_payload_rejects_wrong_dataset_or_zero_can(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / 'sensor.bin'
            image.write_bytes(b'not-empty')
            sample = {'token': 's1', 'prev': '', 'next': '', 'scene_token': 'scene1',
                      'timestamp': 100}
            info = dict(sample, frame_idx=0, can_bus=np.ones(18),
                        cams={name: {'data_path': str(image), 'cam_intrinsic': np.eye(3),
                                     'sensor2lidar_rotation': np.eye(3),
                                     'sensor2lidar_translation': np.zeros(3)}
                              for name in prep.CAMERAS},
                        lidar_path=str(image), sweeps=[], gt_boxes=np.array([[0, 0, 0, 1, 1, 1, 0]]),
                        gt_names=['car'], valid_flag=[True], gt_velocity=np.array([[np.nan, np.nan]]),
                        num_lidar_pts=[1], num_radar_pts=[0], ego2global_translation=[0, 0, 0],
                        ego2global_rotation=[1, 0, 0, 0], lidar2ego_translation=[0, 0, 0],
                        lidar2ego_rotation=[1, 0, 0, 0])
            payload = {'metadata': {'version': prep.VERSION}, 'infos': [info]}
            arguments = ('train', root, {'scene1': 'scene-0001'}, {'s1': sample},
                         {'s1'}, {'s1': 0}, np)
            with mock.patch.dict(prep.COUNTS, train=(1, 1)):
                report = prep.validate_infos(payload, *arguments)
                self.assertEqual(report['unknown_velocity_boxes'], 1)
                info['can_bus'] = np.zeros(18)
                with self.assertRaisesRegex(RuntimeError, 'Unexpected zero CAN'):
                    prep.validate_infos(payload, *arguments)
                info['can_bus'] = np.ones(18)
                payload['metadata']['version'] = 'v1.0-mini'
                with self.assertRaisesRegex(RuntimeError, 'Incorrect dataset version'):
                    prep.validate_infos(payload, *arguments)

    def test_smoke_requires_linked_samples(self):
        def info(index):
            return {'token': str(index), 'prev': str(index - 1) if index else '',
                    'next': str(index + 1), 'timestamp': index, 'scene_token': 'scene1',
                    'gt_boxes': np.array([[0, 0, 0, 1, 1, 1, 0]]),
                    'gt_names': ['car'], 'valid_flag': [True]}
        payload = {'metadata': {'version': prep.VERSION}, 'infos': [info(i) for i in range(5)]}
        self.assertEqual(len(prep.smoke_subset(payload, 4, True, np)['infos']), 4)
        broken = copy.deepcopy(payload)
        broken['infos'][2]['prev'] = 'broken'
        with self.assertRaisesRegex(RuntimeError, 'No valid contiguous'):
            prep.smoke_subset(broken, 4, True, np)


if __name__ == '__main__':
    unittest.main()
