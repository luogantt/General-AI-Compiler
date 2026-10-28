#!/usr/bin/env python3
"""Build and validate BEVFormer temporal infos for complete nuScenes trainval.

Run with the BEVFormer training environment. The original dataset is read-only;
generated PKLs and the success report are published under --output-dir only.
"""

import argparse
import collections
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from datetime import datetime, timezone

VERSION = 'v1.0-trainval'
COUNTS = {'train': (700, 28130), 'val': (150, 6019)}
CAMERAS = {'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
CLASSES = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'}
CAN_EXCEPTIONS = {'scene-%04d' % n for n in
                  (161, 162, 163, 164, 165, 166, 167, 168,
                   170, 171, 172, 173, 174, 175, 176)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def check_download_report(root):
    path = root / 'validation-report.json'
    report = read_json(path)
    checks = (report.get('status') == 'passed', report.get('error_count') == 0,
              report.get('version') == VERSION,
              Path(report.get('root', '')).resolve() == root,
              report.get('splits', {}).get('checked') is True,
              report.get('sensor_files', {}).get('references_checked') == 2631083,
              report.get('sensor_files', {}).get('valid_files') == 2631083,
              report.get('can_bus', {}).get('valid_pose_scenes') == 835)
    if not all(checks):
        raise RuntimeError('A complete, passed download validation is required: ' + str(path))
    return {'path': str(path), 'sha256': sha256(path),
            'sensor_references_checked': 2631083, 'valid_can_scenes': 835}


def official_samples(root):
    """Only load small scene/sample tables, not a second full NuScenes SDK."""
    from nuscenes.utils.splits import train, val
    scenes = {s['token']: s['name'] for s in read_json(root / VERSION / 'scene.json')}
    if set(scenes.values()) != set(train) | set(val) or len(scenes) != 850:
        raise RuntimeError('Dataset scenes differ from official trainval splits.')
    samples = read_json(root / VERSION / 'sample.json')
    by_token = {s['token']: s for s in samples}
    if len(samples) != 34149 or len(by_token) != len(samples):
        raise RuntimeError('Expected exactly 34,149 unique official samples.')
    expected = {}
    for split, scene_names in [('train', set(train)), ('val', set(val))]:
        expected[split] = {s['token'] for s in samples if scenes[s['scene_token']] in scene_names}
        if len(expected[split]) != COUNTS[split][1]:
            raise RuntimeError('Unexpected official sample count: ' + split)
    frame_indices = {}
    grouped = collections.defaultdict(list)
    for sample in samples:
        grouped[sample['scene_token']].append(sample)
    for scene_samples in grouped.values():
        for index, sample in enumerate(sorted(scene_samples, key=lambda s: s['timestamp'])):
            frame_indices[sample['token']] = index
    return scenes, by_token, expected, frame_indices


class CheckedCanBus:
    """Preload each scene once; return fresh dicts because upstream pops keys.

    Only one scene is cached. Nested numeric lists are never modified by the
    converter. Missing or malformed non-blacklisted scenes must raise outside
    its broad exception handler instead of silently becoming zero vectors.
    """

    def __init__(self):
        self.scene = None
        self.poses = None

    def select(self, api, scene):
        if self.scene != scene:
            poses = api.get_messages(scene, 'pose')
            if not poses:
                raise RuntimeError('Empty CAN pose list: ' + scene)
            required = {'utime', 'pos', 'orientation', 'accel', 'rotation_rate', 'vel'}
            if any(set(pose) != required for pose in poses):
                raise RuntimeError('Unexpected CAN pose fields: ' + scene)
            self.scene, self.poses = scene, poses

    def get_messages(self, scene, message):
        if scene != self.scene or message != 'pose':
            raise RuntimeError('Unexpected CAN cache request.')
        return [dict(pose) for pose in self.poses]


def convert(repo, root, output, np):
    source = repo / 'tools/data_converter/nuscenes_converter.py'
    spec = importlib.util.spec_from_file_location('trainval_converter', str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._get_can_bus_info
    cache = CheckedCanBus()

    def checked_can_info(nusc, api, sample):
        scene = nusc.get('scene', sample['scene_token'])['name']
        if scene in CAN_EXCEPTIONS:
            return np.zeros(18)
        cache.select(api, scene)  # Exceptions propagate before upstream's try/except.
        result = original(nusc, cache, sample)
        if result.shape != (18,) or not np.isfinite(result).all() or not np.any(result):
            raise RuntimeError('Invalid or unexpected zero CAN data: ' + scene)
        return result

    module._get_can_bus_info = checked_can_info
    module.create_nuscenes_infos(str(root), str(output), str(root), 'nuscenes',
                                 version=VERSION, max_sweeps=10)
    del module, cache
    gc.collect()
    return {'source': str(source), 'sha256': sha256(source),
            'can_policy': 'Official 15 scene exceptions only; one-scene copy-on-read cache.',
            'upstream_semantics': 'Preserved, including legacy next-pose motion fields.'}


def finite_array(value, shape, label, np):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise RuntimeError('Invalid shape or nonfinite values: ' + label)
    return array


def validate_infos(payload, split, root, scenes, samples, expected, frame_indices, np):
    if payload.get('metadata', {}).get('version') != VERSION:
        raise RuntimeError('Incorrect dataset version: ' + split)
    infos = payload['infos']
    actual = {info['token'] for info in infos}
    if len(infos) != len(actual) or actual != expected:
        raise RuntimeError('%s sample tokens differ: %d missing, %d extra or duplicates' %
                           (split, len(expected - actual), len(actual - expected)))
    required = {'token', 'prev', 'next', 'scene_token', 'frame_idx', 'can_bus', 'timestamp',
                'cams', 'lidar_path', 'sweeps', 'gt_boxes', 'gt_names', 'valid_flag',
                'gt_velocity', 'num_lidar_pts', 'num_radar_pts',
                'ego2global_translation', 'ego2global_rotation',
                'lidar2ego_translation', 'lidar2ego_rotation'}
    zero_scenes, zero_samples, unknown_velocity = set(), 0, 0
    for info in infos:
        if required - info.keys():
            raise RuntimeError('Missing temporal/annotation fields: ' + str(required - info.keys()))
        sample = samples[info['token']]
        for field in ('prev', 'next', 'scene_token', 'timestamp'):
            if info[field] != sample[field]:
                raise RuntimeError('Metadata mismatch for %s: %s' % (info['token'], field))
        if info['frame_idx'] != frame_indices[info['token']]:
            raise RuntimeError('Incorrect temporal frame index: ' + info['token'])
        can = finite_array(info['can_bus'], (18,), 'CAN', np)
        scene = scenes[info['scene_token']]
        if not np.any(can):
            if scene not in CAN_EXCEPTIONS:
                raise RuntimeError('Unexpected zero CAN: ' + scene)
            zero_scenes.add(scene)
            zero_samples += 1
        elif scene in CAN_EXCEPTIONS:
            raise RuntimeError('Unexpected nonzero CAN in official unavailable scene: ' + scene)
        if set(info['cams']) != CAMERAS:
            raise RuntimeError('Missing or extra camera: ' + info['token'])
        paths = [info['lidar_path']]
        for camera in info['cams'].values():
            paths.append(camera['data_path'])
            finite_array(camera['cam_intrinsic'], (3, 3), 'camera intrinsic', np)
            finite_array(camera['sensor2lidar_rotation'], (3, 3), 'camera rotation', np)
            finite_array(camera['sensor2lidar_translation'], (3,), 'camera translation', np)
        if len(info['sweeps']) > 10:
            raise RuntimeError('More than 10 LiDAR sweeps in info.')
        paths.extend(sweep['data_path'] for sweep in info['sweeps'])
        for value in paths:
            path = Path(value).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                raise RuntimeError('Info references a different dataset: ' + str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError('Missing/empty sensor file: ' + str(path))
        count = len(info['gt_names'])
        boxes = finite_array(info['gt_boxes'], (count, 7), 'boxes', np)
        if count and not (boxes[:, 3:6] > 0).all():
            raise RuntimeError('Nonpositive ground-truth box dimension.')
        for field in ('valid_flag', 'num_lidar_pts', 'num_radar_pts'):
            finite_array(info[field], (count,), field, np)
        velocity = np.asarray(info['gt_velocity'])
        if velocity.shape != (count, 2) or np.isinf(velocity).any():
            raise RuntimeError('Invalid annotation velocity.')
        # nuScenes deliberately uses NaN when an object's velocity is unknown;
        # the dataset loader handles these. Do not reject valid annotations.
        unknown_velocity += int(np.isnan(velocity).any(axis=1).sum())
    scene_count = len({info['scene_token'] for info in infos})
    if (scene_count, len(infos)) != COUNTS[split]:
        raise RuntimeError('Incomplete split: ' + split)
    return {'samples': len(infos), 'scenes': scene_count, 'zero_can_samples': zero_samples,
            'zero_can_scenes': sorted(zero_scenes), 'unknown_velocity_boxes': unknown_velocity}


def has_target(info, np):
    boxes = np.asarray(info['gt_boxes'])
    if not len(boxes):
        return False
    known = np.asarray([str(name) in CLASSES for name in info['gt_names']])
    in_range = ((boxes[:, 0] > -51.2) & (boxes[:, 0] < 51.2)
                & (boxes[:, 1] > -51.2) & (boxes[:, 1] < 51.2))
    return bool(np.any(np.asarray(info['valid_flag'], dtype=bool) & known & in_range))


def smoke_subset(payload, count, need_targets, np):
    grouped = collections.OrderedDict()
    for info in sorted(payload['infos'], key=lambda item: item['timestamp']):
        grouped.setdefault(info['scene_token'], []).append(info)
    for infos in grouped.values():
        for start in range(len(infos) - count + 1):
            window = infos[start:start + count]
            contiguous = all(a['next'] == b['token'] and b['prev'] == a['token']
                             for a, b in zip(window, window[1:]))
            if contiguous and (not need_targets or all(has_target(info, np) for info in window)):
                return {'infos': window, 'metadata': dict(payload['metadata'])}
    raise RuntimeError('No valid contiguous smoke window of %d samples.' % count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-infos', action='store_true')
    args = parser.parse_args()
    repo, root, output = (path.expanduser().resolve() for path in
                          (args.repo_root, args.data_root, args.output_dir))
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the read-only raw dataset.')
    evidence = check_download_report(root)
    os.chdir(str(repo))
    sys.path.insert(0, str(repo))
    import numpy as np
    scenes, samples, expected, frame_indices = official_samples(root)
    output.mkdir(parents=True, exist_ok=True)
    names = {split: 'nuscenes_infos_temporal_%s.pkl' % split for split in ('train', 'val')}
    if not args.reuse_infos and any((output / name).exists() for name in names.values()):
        raise FileExistsError('Infos already exist; use --reuse-infos to validate them.')
    # No success report can remain visible while its outputs are being replaced.
    report_path = output / 'preparation_report.json'
    if report_path.exists():
        report_path.replace(output / 'preparation_report.previous.json')
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=str(output)) as temporary:
        staging = Path(temporary)
        conversion = {'reused': True}
        if not args.reuse_infos:
            print('Converting full trainval with the original BEVFormer converter...', flush=True)
            conversion = convert(repo, root, staging, np)
        report = {'passed': False, 'version': VERSION, 'repo_root': str(repo),
                  'data_root': str(root), 'canbus_root': str(root), 'output_dir': str(output),
                  'download_validation': evidence, 'converter': conversion, 'artifacts': {}}
        all_zero_scenes = set()
        for split in ('train', 'val'):
            path = (output if args.reuse_infos else staging) / names[split]
            with path.open('rb') as handle:
                payload = pickle.load(handle)
            report[split] = validate_infos(payload, split, root, scenes, samples,
                                          expected[split], frame_indices, np)
            all_zero_scenes.update(report[split]['zero_can_scenes'])
            count = 4 if split == 'train' else 3
            subset = smoke_subset(payload, count, split == 'train', np)
            smoke_name = 'nuscenes_infos_temporal_smoke_%s.pkl' % split
            with (staging / smoke_name).open('wb') as handle:
                pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report['smoke_' + split] = {'samples': count,
                'scene_token': subset['infos'][0]['scene_token'],
                'sample_tokens': [info['token'] for info in subset['infos']],
                'path': str(output / smoke_name)}
            report['artifacts'][names[split]] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            del payload, subset
            gc.collect()
        if all_zero_scenes != CAN_EXCEPTIONS:
            raise RuntimeError('Zero-CAN scenes differ from the exact 15 official exceptions.')
        for path in staging.glob('*.pkl'):
            report['artifacts'][path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
            path.replace(output / path.name)
        report.update(passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
                      official_missing_can_scenes=sorted(all_zero_scenes))
        temporary_report = output / 'preparation_report.json.tmp'
        temporary_report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        temporary_report.replace(report_path)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
