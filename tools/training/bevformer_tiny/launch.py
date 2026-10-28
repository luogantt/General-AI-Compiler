"""Launch an isolated single-GPU BEVFormer job with durable status and logs.

Source the validated training env.sh before invoking this script. A train job
requires both complete trainval preparation and a passed real-data smoke test.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'train'], required=True)
    parser.add_argument('--port', type=int, default=28519)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    task, repo = args.task_root.resolve(), args.repo_root.resolve()
    log_dir = task / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    work = task / 'work_dirs' / ('smoke' if args.stage == 'smoke' else 'trainval')
    work.mkdir(parents=True, exist_ok=True)
    config = task / 'configs' / ('tiny_smoke.py' if args.stage == 'smoke' else 'tiny_trainval.py')
    preparation = json.loads((task / 'infos/preparation_report.json').read_text())
    if not preparation.get('passed'):
        raise RuntimeError('Full trainval preparation must pass first')
    if args.stage == 'train':
        smoke = json.loads((task / 'work_dirs/smoke/smoke_training_check.json').read_text())
        if not smoke.get('passed'):
            raise RuntimeError('Real full-trainval smoke test must pass first')
    if args.resume_from and not args.resume_from.is_file():
        raise FileNotFoundError(args.resume_from)
    if args.stage == 'smoke' and args.resume_from:
        raise ValueError('Smoke must start with a fresh optimizer')
    # Parent exits after dispatch; the child owns the lock for its entire lifetime.
    if args.detach:
        child_args = [x for x in sys.argv[1:] if x != '--detach']
        with (log_dir / (args.stage + '.log')).open('ab', buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + child_args,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        print(json.dumps({'supervisor_pid': process.pid, 'stage': args.stage,
                          'status': str(task / (args.stage + '-status.json'))}))
        return 0
    import fcntl
    with (task / '.training.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'smoke':
            write_json(work / 'smoke_training_check.json', {'passed': False, 'status': 'starting'})
        elif not args.resume_from and any(work.glob('*.log.json')):
            raise RuntimeError('Existing train run: supply --resume-from or use a new task root')
        state_path = task / (args.stage + '-status.json')
        state = dict(state='running', stage=args.stage, supervisor_pid=os.getpid(),
                     started_at=now(), task_root=str(task), repo_root=str(repo),
                     config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                     preparation_sha256=hashlib.sha256((task / 'infos/preparation_report.json').read_bytes()).hexdigest(),
                     dataset='nuScenes v1.0-trainval', work_dir=str(work),
                     log=str(log_dir / (args.stage + '.log')),
                     resume_from=str(args.resume_from.resolve()) if args.resume_from else None)
        write_json(state_path, state)
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(task) + os.pathsep + str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
        environment.setdefault('CUDA_VISIBLE_DEVICES', '0')
        command = [sys.executable, '-m', 'torch.distributed.launch', '--nproc_per_node=1',
                   '--master_port=' + str(args.port), 'tools/fp16/train.py', str(config),
                   '--launcher', 'pytorch', '--work-dir', str(work), '--seed', '0']
        if args.resume_from:
            command.extend(['--resume-from', str(args.resume_from.resolve())])
        try:
            process = subprocess.Popen(command, cwd=str(repo), env=environment)
            state['training_launcher_pid'] = process.pid
            write_json(state_path, state)
            state['exit_code'] = process.wait()
            state['state'] = 'completed' if state['exit_code'] == 0 else 'failed'
        except BaseException as exc:
            state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            state['finished_at'] = now()
            write_json(state_path, state)
        return state['exit_code']


if __name__ == '__main__':
    sys.exit(main())
