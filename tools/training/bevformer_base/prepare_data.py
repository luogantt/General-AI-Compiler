#!/usr/bin/env python3
"""Prepare camera-training nuScenes data from read-only AutoDL public archives.

Outputs: RUN/data/{mini,trainval}, RUN/infos, and a mode-specific report.
This profile preserves every official sample, but deliberately does not extract
radar or sweeps. It is not a full all-sensor archive-integrity certification.
"""

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import shutil
import sys
import tarfile
import tempfile
import zipfile

CAMERAS = {'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'}
CHANNELS = CAMERAS | {'LIDAR_TOP'}
TABLES = ('attribute', 'calibrated_sensor', 'category', 'ego_pose', 'instance',
          'log', 'map', 'sample', 'sample_annotation', 'sample_data', 'scene',
          'sensor', 'visibility')
CLASSES = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'}
CAN_EXCEPTIONS = {'scene-%04d' % n for n in
                  (161, 162, 163, 164, 165, 166, 167, 168,
                   170, 171, 172, 173, 174, 175, 176)}
RESERVE_BYTES = 30 * 1024 ** 3


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_json(path):
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_records(path):
    """Stream the large sample_data table without a second full SDK instance."""
    decoder = json.JSONDecoder()
    with path.open(encoding='utf-8') as handle:
        buffer, position, eof = '', 0, False

        def character():
            nonlocal buffer, position, eof
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof:
                    return ''
                block = handle.read(1024 * 1024)
                buffer, position, eof = block, 0, not block

        if character() != '[':
            raise ValueError('Expected a JSON array: ' + str(path))
        position += 1
        if character() == ']':
            return
        while True:
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof or len(buffer) - position > 16 * 1024 * 1024:
                        raise ValueError('Invalid or oversized JSON record: ' + str(path))
                    block = handle.read(1024 * 1024)
                    buffer, position, eof = buffer[position:] + block, 0, not block
            yield value
            separator = character()
            if separator == ']':
                position += 1
                if character():
                    raise ValueError('Trailing content in JSON array.')
                return
            if separator != ',':
                raise ValueError('Malformed JSON array: ' + str(path))
            position += 1
            character()


def safe_relative(name):
    if not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive/data name: ' + repr(name))
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe archive/data name: ' + name)
    return path


def regular_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError('Missing or empty required file: ' + str(path))


def check_space(root, incoming=0):
    free = shutil.disk_usage(str(root)).free
    if free < incoming + RESERVE_BYTES:
        raise RuntimeError('Insufficient disk space; preserve 30 GiB for infos/checkpoints/logs.')
    return free


def copy_file(source, target, size, root):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.parent.resolve() != target.parent:
        raise ValueError('Refusing extraction through a symlink: ' + str(target))
    if target.is_file() and target.stat().st_size == size:
        return
    check_space(root, size)
    temporary = target.with_name(target.name + '.extracting')
    if temporary.is_symlink():
        raise ValueError('Unsafe extraction temporary path: ' + str(temporary))
    written = 0
    with temporary.open('wb') as handle:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            written += len(block)
            if written > size:
                raise ValueError('Archive member exceeds its declared size.')
            handle.write(block)
    if written != size:
        raise ValueError('Truncated archive member: ' + str(target))
    temporary.replace(target)


def extract_tar(archive, data, state_dir, version):
    info = archive.stat()
    identity = {'source': str(archive), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}
    marker = state_dir / (archive.name + '.json')
    if marker.is_file() and load_json(marker).get('source_identity') == identity:
        return load_json(marker)
    result = {'source_identity': identity, 'selected_files': 0, 'selected_bytes': 0}
    print('Extracting selected data: ' + str(archive), flush=True)
    with tarfile.open(str(archive), mode='r|*') as bundle:
        for member in bundle:
            path = safe_relative(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError('Links/special archive members are unsupported: ' + member.name)
            parts = path.parts
            selected = bool(parts) and (parts[0] in (version, 'maps') or
                (len(parts) >= 3 and parts[0] == 'samples' and parts[1] in CHANNELS))
            if selected:
                with bundle.extractfile(member) as source:
                    copy_file(source, data.joinpath(*parts), member.size, data)
                result['selected_files'] += 1
                result['selected_bytes'] += member.size
                if result['selected_files'] % 10000 == 0:
                    print('  selected files: %d' % result['selected_files'], flush=True)
            # Python 3.8's streaming mode otherwise retains every TarInfo object.
            bundle.members.clear()
    save_json(marker, result)
    return result


def extract_can(archive, data, scene_names):
    needed = scene_names - CAN_EXCEPTIONS
    wanted = {name + '_pose.json' for name in needed}
    found = set()
    with zipfile.ZipFile(str(archive)) as bundle:
        for member in bundle.infolist():
            path = safe_relative(member.filename)
            if member.is_dir() or path.name not in wanted:
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('CAN archive contains a symlink.')
            if path.name in found:
                raise ValueError('Duplicate CAN pose entry: ' + path.name)
            with bundle.open(member) as source:
                copy_file(source, data / 'can_bus' / path.name, member.file_size, data)
            found.add(path.name)
    if found != wanted:
        raise RuntimeError('CAN archive is missing required scene pose files.')


def metadata(data, mode):
    from nuscenes.utils import splits
    version = 'v1.0-' + mode
    for name in TABLES:
        regular_file(data / version / (name + '.json'))
    scenes = {item['token']: item['name'] for item in load_json(data / version / 'scene.json')}
    samples = {item['token']: item for item in load_json(data / version / 'sample.json')}
    counts = ((8, 323), (2, 81)) if mode == 'mini' else ((700, 28130), (150, 6019))
    official = (splits.mini_train, splits.mini_val) if mode == 'mini' else (splits.train, splits.val)
    if set(scenes.values()) != set(official[0]) | set(official[1]):
        raise RuntimeError('Scene names do not match the official dataset split.')
    expected = {}
    for split, names, count in zip(('train', 'val'), official, counts):
        expected[split] = {token for token, sample in samples.items()
                           if scenes[sample['scene_token']] in set(names)}
        if len(names) != count[0] or len(expected[split]) != count[1]:
            raise RuntimeError('Incorrect official split size: ' + split)
    if len(samples) != sum(count[1] for count in counts):
        raise RuntimeError('Unexpected sample count.')
    return scenes, samples, expected


def validate_raw(data, mode, scenes, samples):
    version = 'v1.0-' + mode
    sensors = {row['token']: row['channel'] for row in load_json(data / version / 'sensor.json')}
    calibrations = {row['token']: row['sensor_token']
                    for row in load_json(data / version / 'calibrated_sensor.json')}
    observed = collections.defaultdict(set)
    references, checked = 0, 0
    for row in json_records(data / version / 'sample_data.json'):
        references += 1
        channel = sensors[calibrations[row['calibrated_sensor_token']]]
        if not row['is_key_frame'] or channel not in CHANNELS:
            continue
        token = row['sample_token']
        if token not in samples or channel in observed[token]:
            raise RuntimeError('Invalid/duplicate keyframe sensor reference: ' + token)
        relative = safe_relative(row['filename'])
        if len(relative.parts) < 3 or relative.parts[:2] != ('samples', channel):
            raise RuntimeError('Unexpected keyframe sensor path: ' + row['filename'])
        regular_file(data.joinpath(*relative.parts))
        observed[token].add(channel)
        checked += 1
    if set(observed) != set(samples) or any(channels != CHANNELS for channels in observed.values()):
        raise RuntimeError('Not every sample has six cameras and one LiDAR keyframe.')
    map_rows = load_json(data / version / 'map.json')
    for row in map_rows:
        regular_file(data.joinpath(*safe_relative(row['filename']).parts))
    import numpy as np
    required_can = set(scenes.values()) - CAN_EXCEPTIONS
    for scene in sorted(required_can):
        poses = load_json(data / 'can_bus' / (scene + '_pose.json'))
        if not poses:
            raise RuntimeError('Empty CAN poses: ' + scene)
        previous = None
        for pose in poses:
            timestamp = pose['utime']
            if previous is not None and timestamp < previous:
                raise RuntimeError('Non-monotonic CAN timestamp: ' + scene)
            previous = timestamp
            for field, count in [('pos', 3), ('orientation', 4), ('accel', 3),
                                 ('rotation_rate', 3), ('vel', 3)]:
                vector = np.asarray(pose[field])
                if vector.shape != (count,) or not np.isfinite(vector).all():
                    raise RuntimeError('Malformed CAN field: %s %s' % (scene, field))
    return {'profile': 'camera_training', 'status': 'passed', 'root': str(data),
            'metadata_sensor_references_scanned': references,
            'keyframe_sensor_files_checked': checked,
            'camera_keyframe_files_checked': len(samples) * 6,
            'lidar_keyframe_files_checked': len(samples), 'maps_checked': len(map_rows),
            'can_scenes_checked': len(required_can),
            'official_missing_can_scenes': sorted(set(scenes.values()) & CAN_EXCEPTIONS),
            'not_materialized': ['radar files', 'camera sweeps', 'LiDAR sweeps'],
            'scope_note': 'All official train/val samples; not an all-sensor archive verification.'}


def convert(repo, data, destination, version):
    import numpy as np
    path = repo / 'tools/data_converter/nuscenes_converter.py'
    spec = importlib.util.spec_from_file_location('base_nuscenes_converter', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._get_can_bus_info

    class CanCache:
        scene, poses = None, None

        def select(self, api, name):
            if self.scene != name:
                poses = api.get_messages(name, 'pose')
                if not poses:
                    raise RuntimeError('Missing real CAN poses: ' + name)
                self.scene, self.poses = name, poses

        def get_messages(self, name, message):
            if name != self.scene or message != 'pose':
                raise RuntimeError('Unexpected CAN request.')
            return [dict(pose) for pose in self.poses]

    cache = CanCache()

    def checked_can(nusc, api, sample):
        name = nusc.get('scene', sample['scene_token'])['name']
        if name in CAN_EXCEPTIONS:
            return np.zeros(18)
        cache.select(api, name)  # Missing scenes must fail outside upstream's catch-all.
        value = original(nusc, cache, sample)
        if value.shape != (18,) or not np.isfinite(value).all() or not np.any(value):
            raise RuntimeError('Invalid CAN vector: ' + name)
        return value

    module._get_can_bus_info = checked_can
    module.create_nuscenes_infos(str(data), str(destination), str(data), 'nuscenes',
                                 version=version, max_sweeps=10)
    gc.collect()
    return {'source': str(path), 'sha256': sha256(path), 'max_sweeps_metadata': 10,
            'can_semantics': 'Original converter, including legacy next-pose motion fields; copy-on-read cache.'}


def validate_infos(payload, version, data, scenes, samples, expected):
    import numpy as np
    infos = payload['infos']
    if payload.get('metadata', {}).get('version') != version:
        raise RuntimeError('PKL version does not match the data.')
    if len(infos) != len(expected) or {item['token'] for item in infos} != expected:
        raise RuntimeError('PKL samples do not match the complete official split.')
    frame_indices = {}
    grouped = collections.defaultdict(list)
    for sample in samples.values():
        grouped[sample['scene_token']].append(sample)
    for group in grouped.values():
        for index, sample in enumerate(sorted(group, key=lambda item: item['timestamp'])):
            frame_indices[sample['token']] = index
    zero_can, sweeps = 0, 0
    for item in infos:
        sample = samples[item['token']]
        for key in ('prev', 'next', 'scene_token', 'timestamp'):
            if item[key] != sample[key]:
                raise RuntimeError('Temporal metadata mismatch: ' + item['token'])
        if item['frame_idx'] != frame_indices[item['token']]:
            raise RuntimeError('Incorrect frame index.')
        can = np.asarray(item['can_bus'])
        if can.shape != (18,) or not np.isfinite(can).all():
            raise RuntimeError('Invalid CAN shape/values.')
        zero = not np.any(can)
        if zero != (scenes[item['scene_token']] in CAN_EXCEPTIONS):
            raise RuntimeError('Unexpected CAN fallback policy.')
        zero_can += int(zero)
        if set(item['cams']) != CAMERAS:
            raise RuntimeError('PKL must have exactly six cameras.')
        files = [item['lidar_path']]
        for camera in item['cams'].values():
            files.append(camera['data_path'])
            for key, shape in [('cam_intrinsic', (3, 3)), ('sensor2lidar_rotation', (3, 3)),
                               ('sensor2lidar_translation', (3,))]:
                value = np.asarray(camera[key])
                if value.shape != shape or not np.isfinite(value).all():
                    raise RuntimeError('Invalid camera calibration: ' + key)
        for filename in files:
            path = Path(filename).resolve()
            path.relative_to(data)
            regular_file(path)
        if len(item['sweeps']) > 10:
            raise RuntimeError('Unexpected sweep metadata count.')
        sweeps += len(item['sweeps'])  # Deliberately metadata-only in camera_training.
        count = len(item['gt_names'])
        boxes = np.asarray(item['gt_boxes'])
        if boxes.shape != (count, 7) or not np.isfinite(boxes).all():
            raise RuntimeError('Invalid ground-truth boxes.')
        if count and not (boxes[:, 3:6] > 0).all():
            raise RuntimeError('Nonpositive ground-truth dimensions.')
        for key in ('valid_flag', 'num_lidar_pts', 'num_radar_pts'):
            if np.asarray(item[key]).shape != (count,):
                raise RuntimeError('Invalid annotation length: ' + key)
        velocity = np.asarray(item['gt_velocity'])
        if velocity.shape != (count, 2) or np.isinf(velocity).any():
            raise RuntimeError('Invalid ground-truth velocities.')
    return {'samples': len(infos), 'scenes': len({item['scene_token'] for item in infos}),
            'zero_can_samples': zero_can, 'unused_sweep_metadata_references': sweeps}


def smoke_subset(payload, count, targets):
    import numpy as np

    def has_target(item):
        boxes = np.asarray(item['gt_boxes'])
        if not len(boxes):
            return False
        valid = np.asarray(item['valid_flag'], dtype=bool)
        known = np.asarray([str(name) in CLASSES for name in item['gt_names']])
        inside = ((boxes[:, 0] > -51.2) & (boxes[:, 0] < 51.2) &
                  (boxes[:, 1] > -51.2) & (boxes[:, 1] < 51.2))
        return bool(np.any(valid & known & inside))

    grouped = collections.OrderedDict()
    for item in sorted(payload['infos'], key=lambda item: item['timestamp']):
        grouped.setdefault(item['scene_token'], []).append(item)
    for group in grouped.values():
        for start in range(len(group) - count + 1):
            window = group[start:start + count]
            linked = all(a['next'] == b['token'] and b['prev'] == a['token']
                         for a, b in zip(window, window[1:]))
            if linked and (not targets or all(has_target(item) for item in window)):
                return {'metadata': dict(payload['metadata']), 'infos': window}
    raise RuntimeError('No contiguous real smoke sequence with valid targets.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('mini', 'trainval'), required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=1,
                        help='Concurrent trainval blob archives; mini/metadata stay serial.')
    args = parser.parse_args()
    public, output, repo = (p.expanduser().resolve() for p in
                            (args.public_root, args.output_root, args.repo_root))
    if output == public or public in output.parents:
        raise ValueError('Output must be outside the read-only public dataset.')
    version = 'v1.0-' + args.mode
    data, infos = output / 'data' / args.mode, output / 'infos'
    data.mkdir(parents=True, exist_ok=True)
    infos.mkdir(parents=True, exist_ok=True)
    state_path = output / ('data-preparation-%s-status.json' % args.mode)
    state = {'state': 'running', 'mode': args.mode, 'profile': 'camera_training',
             'data_root': str(data), 'started_at': datetime.now(timezone.utc).isoformat(),
             'disk_free_at_start': check_space(data), 'completed_archives': [],
             'extraction_workers': args.workers, 'reserve_bytes': RESERVE_BYTES}
    save_json(state_path, state)
    if args.mode == 'mini':
        archives = [public / 'Fulldatasetv1.0/Mini/v1.0-mini.tgz']
    else:
        folder = public / 'Fulldatasetv1.0/Trainval'
        names = ['v1.0-trainval_meta.tgz'] + ['v1.0-trainval%02d_blobs.tgz' % i for i in range(1, 11)]
        archives = [folder / name for name in names]
    can_archive = public / 'CANbusexpansion/can_bus.zip'
    try:
        for archive in archives + [can_archive]:
            regular_file(archive)
        state['source_archive_bytes'] = sum(archive.stat().st_size for archive in archives)
        state['space_policy'] = ('Check every selected member before writing, retaining 30 GiB. '
                                 'Compressed archive size is not an extracted-size estimate.')
        report_name = 'preparation_%s_report.json' % args.mode
        for folder in (output, infos):
            previous = folder / report_name
            if previous.exists():
                previous.replace(previous.with_name(previous.stem + '.previous.json'))

        def record_archive(archive):
            state.update(phase='extract', current_archive=str(archive))
            save_json(state_path, state)
            details = extract_tar(archive, data, output / 'state' / args.mode, version)
            state['completed_archives'].append(details)
            save_json(state_path, state)

        # Metadata first; independent blob archives can then use separate CPU cores.
        record_archive(archives[0])
        if args.workers > 1 and len(archives) > 1:
            state['active_archives'] = [str(path) for path in archives[1:]]
            save_json(state_path, state)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(extract_tar, archive, data, output / 'state' / args.mode,
                                       version): archive for archive in archives[1:]}
                for future in as_completed(futures):
                    state['completed_archives'].append(future.result())
                    state['active_archives'].remove(str(futures[future]))
                    save_json(state_path, state)
        else:
            for archive in archives[1:]:
                record_archive(archive)
        os.chdir(str(repo))
        sys.path.insert(0, str(repo))
        scenes, samples, expected = metadata(data, args.mode)
        extract_can(can_archive, data, set(scenes.values()))
        state['phase'] = 'validate_camera_training'
        save_json(state_path, state)
        raw_report = validate_raw(data, args.mode, scenes, samples)
        save_json(data / 'camera-training-validation.json', raw_report)
        state['phase'] = 'temporal_infos'
        save_json(state_path, state)
        report = {'passed': False, 'mode': args.mode, 'version': version,
                  'profile': 'camera_training', 'data_root': str(data), 'infos_root': str(infos),
                  'raw_validation': raw_report, 'artifacts': {}}
        with tempfile.TemporaryDirectory(prefix='.%s-' % args.mode, dir=str(infos)) as temporary:
            staging = Path(temporary)
            report['converter'] = convert(repo, data, staging, version)
            for split in ('train', 'val'):
                source = staging / ('nuscenes_infos_temporal_%s.pkl' % split)
                with source.open('rb') as handle:
                    payload = pickle.load(handle)
                report[split] = validate_infos(payload, version, data, scenes, samples, expected[split])
                name = ('nuscenes_infos_temporal_mini_%s.pkl' if args.mode == 'mini'
                        else 'nuscenes_infos_temporal_%s.pkl') % split
                if args.mode == 'mini':
                    subset = smoke_subset(payload, 8 if split == 'train' else 3, split == 'train')
                    smoke_name = 'nuscenes_infos_temporal_smoke_%s.pkl' % split
                    with (staging / smoke_name).open('wb') as handle:
                        pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    report['smoke_' + split] = {
                        'path': str(infos / smoke_name), 'stored_frames': len(subset['infos']),
                        'context_frames': 4 if split == 'train' else 0,
                        'training_samples': 4 if split == 'train' else 0,
                        'sample_tokens': [item['token'] for item in subset['infos']],
                        'dataset_contract': 'Train: BaseSmokeTrainDataset exposes indices 4..7 of 8 stored infos.'
                                            if split == 'train' else 'Three sequential validation frames.'}
                    del subset
                if source.name != name:
                    source.rename(staging / name)
                del payload
                gc.collect()
            for path in staging.glob('*.pkl'):
                report['artifacts'][path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
                path.replace(infos / path.name)
        report.update(passed=True, completed_at=datetime.now(timezone.utc).isoformat())
        name = 'preparation_%s_report.json' % args.mode
        save_json(output / name, report)
        save_json(infos / name, report)
        state.update(state='completed', phase='completed', report=str(output / name),
                     completed_at=datetime.now(timezone.utc).isoformat())
        save_json(state_path, state)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
        save_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
