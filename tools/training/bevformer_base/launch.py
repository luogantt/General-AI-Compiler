"""Launch the full-size BEVFormer-Base FP16 experiment with explicit gates.

The mini smoke tests memory and real gradients. Only the independent trainval
job computes full nuScenes metrics. Use the task's validated env.sh first.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from process_support import ProcessScope, StopRequested

STAGES = {
    'smoke': ('base_smoke.py', 'smoke', 'base_smoke_check.json', 1),
    'ddp-smoke': ('base_ddp_smoke.py', 'ddp-smoke', 'base_ddp_smoke_check.json', 3),
    'train': ('base_trainval.py', 'trainval', None, 3),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, report):
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def require_passed(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('passed') is not True:
        raise RuntimeError('Required check has not passed: ' + str(path))
    return report


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config_sources(path, files):
    path = path.resolve()
    if path in files:
        return
    files.add(path)
    for statement in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '_base_'
                for target in statement.targets):
            parents = ast.literal_eval(statement.value)
            for parent in ([parents] if isinstance(parents, str) else parents):
                config_sources(path.parent / parent, files)


def support_sources(path, task, files):
    path = path.resolve()
    if path in files:
        return
    files.add(path)
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                   else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for module in modules:
            local = task / (module.split('.')[0] + '.py')
            if local.is_file():
                support_sources(local, task, files)


def runtime_fingerprint(task, repo, weights, stage='smoke'):
    """DDP-only files do not invalidate the independent single-GPU check."""
    files = set()
    config_sources(task / 'configs/base_smoke.py', files)
    support_sources(task / 'base_training_support.py', task, files)
    if stage == 'ddp-smoke':
        config_sources(task / 'configs/base_ddp_smoke.py', files)
        support_sources(task / 'ddp_preflight.py', task, files)
    files.add(repo / 'tools/fp16/train.py')
    files.update((repo / 'projects/mmdet3d_plugin').rglob('*.py'))
    files.add(weights.resolve())
    for split in ('train', 'val'):
        files.add(task / 'infos' / ('nuscenes_infos_temporal_smoke_%s.pkl' % split))
    packages = {}
    for package in ('torch', 'torchvision', 'mmcv-full', 'mmdet', 'mmdet3d', 'nuscenes-devkit'):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {'files': {str(path.resolve()): file_hash(path) for path in sorted(files)},
            'python': sys.version, 'executable': str(Path(sys.executable).resolve()),
            'packages': packages}


def require_smoke_evidence(task, repo, weights, stage):
    _, directory, report_name, _ = STAGES[stage]
    work = task / 'work_dirs' / directory
    require_passed(work / report_name)
    evidence = require_passed(work / 'runtime-evidence.json')
    if (evidence.get('stage') != stage or
            evidence.get('smoke_report_sha256') != file_hash(work / report_name) or
            evidence.get('runtime') != runtime_fingerprint(task, repo, weights, stage)):
        raise RuntimeError('Runtime/config/weights changed since %s passed; rerun that preflight.' % stage)
    return evidence


def require_training_evidence(task, repo):
    from mmcv import Config
    weights = Path(Config.fromfile(str(task / 'configs/base_trainval.py')).load_from).resolve()
    for stage in ('smoke', 'ddp-smoke'):
        require_smoke_evidence(task, repo, weights, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--port', type=int, default=28529)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    task, repo = args.task_root.resolve(), args.repo_root.resolve()
    config_name, directory, report_name, default_gpus = STAGES[args.stage]
    gpus = args.gpus if args.gpus is not None else default_gpus
    if gpus < 1 or (args.stage == 'smoke' and gpus != 1):
        raise ValueError('Smoke currently verifies a single GPU; train supports DDP.')
    if args.resume_from and args.stage != 'train':
        raise ValueError('The smoke test must use a fresh optimizer.')
    if args.resume_from and not args.resume_from.is_file():
        raise FileNotFoundError(args.resume_from)
    require_passed(task / ('preparation_mini_report.json' if args.stage != 'train'
                           else 'preparation_trainval_report.json'))
    require_passed(task / 'weights/weight-check.json')
    if args.stage in ('train', 'ddp-smoke'):
        require_passed(task / 'work_dirs/smoke/base_smoke_check.json')
    if args.stage == 'train':
        require_passed(task / 'work_dirs/ddp-smoke/base_ddp_smoke_check.json')
    config = task / 'configs' / config_name
    if not config.is_file():
        raise FileNotFoundError(config)
    logs = task / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.detach:
        arguments = [value for value in sys.argv[1:] if value != '--detach']
        with (logs / (args.stage + '.log')).open('ab', buffering=0) as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())] + arguments,
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'supervisor_pid': child.pid, 'stage': args.stage,
                          'status_file': str(task / (args.stage + '-status.json'))}))
        return 0
    import fcntl
    with (task / '.training.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        from mmcv import Config
        # The supervisor must not create a CUDA context: on 12 GiB cards it
        # would reserve memory for the entire child training lifetime.
        inventory = subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.total',
            '--format=csv,noheader,nounits'], text=True)
        devices = []
        for line in inventory.strip().splitlines():
            name, memory = line.rsplit(',', 1)
            devices.append({'name': name.strip(), 'total_memory_mib': int(memory.strip())})
        if gpus > len(devices):
            raise RuntimeError('Requested GPU count exceeds visible GPUs.')
        cfg = Config.fromfile(str(config))
        expected_world = cfg.get('base_world_size', 1 if args.stage == 'smoke' else None)
        if expected_world != gpus:
            raise RuntimeError('Config/launcher world size mismatch: %r != %r' % (expected_world, gpus))
        work = Path(cfg.work_dir).resolve()
        if work != (task / 'work_dirs' / directory).resolve():
            raise RuntimeError('work_dir must match the task/stage directory.')
        work.mkdir(parents=True, exist_ok=True)
        weights = Path(cfg.load_from).resolve()
        if args.stage in ('train', 'ddp-smoke'):
            require_smoke_evidence(task, repo, weights, 'smoke')
        if args.stage == 'train':
            require_smoke_evidence(task, repo, weights, 'ddp-smoke')
        if args.stage == 'train' and not args.resume_from and any(work.glob('*.log.json')):
            raise RuntimeError('Existing train run; select a checkpoint with --resume-from.')
        runtime = None
        if args.stage != 'train':
            save(work / report_name, {'passed': False, 'status': 'starting'})
            save(work / 'runtime-evidence.json', {'passed': False, 'status': 'starting'})
            runtime = runtime_fingerprint(task, repo, weights, args.stage)
        state = dict(state='running', stage=args.stage, supervisor_pid=os.getpid(),
                     started_at=now(), config=str(config), world_size=gpus,
                     config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                     task_root=str(task), repo_root=str(repo), work_dir=str(work),
                     model='BEVFormer-Base R101 DCNv2, BEV 200x200, 6 encoder layers',
                     resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                     torch_version=torch.__version__, cuda_version=torch.version.cuda,
                     devices=devices[:gpus])
        status = task / (args.stage + '-status.json')
        save(status, state)
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(task) + os.pathsep + str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
        environment['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpus))
        command = [sys.executable, '-m', 'torch.distributed.launch', '--nproc_per_node=' + str(gpus),
                   '--master_port=' + str(args.port), 'tools/fp16/train.py', str(config),
                   '--launcher', 'pytorch', '--work-dir', str(work), '--seed', '0']
        if args.resume_from:
            command += ['--resume-from', str(args.resume_from.resolve())]
        with ProcessScope() as processes:
            try:
                child = processes.spawn(command, cwd=str(repo), env=environment)
                state.update(training_launcher_pid=child.pid, training_process_group=child.pid)
                save(status, state)
                code = processes.wait(child)
                # Clean up any orphaned workers before accepting success/releasing the lock.
                processes.close()
                state.update(exit_code=code, state='completed' if code == 0 else 'failed')
                if code == 0 and args.stage != 'train':
                    require_passed(work / report_name)
                    if runtime != runtime_fingerprint(task, repo, weights, args.stage):
                        raise RuntimeError('Runtime/config/weights changed during preflight; evidence rejected.')
                    processes.check_stop()
                    save(work / 'runtime-evidence.json', dict(
                        passed=True, stage=args.stage, created_at=now(), runtime=runtime,
                        smoke_report_sha256=file_hash(work / report_name),
                        config=str(config), world_size=gpus))
            except StopRequested as exc:
                code = 128 + exc.signum
                state.update(state='stopped', signal=exc.signum, exit_code=code)
            except BaseException as exc:
                state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
                raise
            finally:
                processes.close()
                state['finished_at'] = now()
                save(status, state)
        return code


if __name__ == '__main__':
    sys.exit(main())
