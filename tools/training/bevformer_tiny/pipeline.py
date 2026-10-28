"""Prepare full trainval, verify real training, then run the 24-epoch experiment.

Source the validated env.sh first. All stages are separate processes so the
converter's large NuScenes tables are released before CUDA training begins.
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


def save(path, payload):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    task, repo, data = (path.resolve() for path in (args.task_root, args.repo_root, args.data_root))
    logs = task / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != '--detach']
        with (logs / 'pipeline.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps({'pipeline_pid': process.pid, 'status': str(task / 'pipeline-status.json')}))
        return 0
    import fcntl
    with (task / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((task / 'work_dirs/trainval').glob('*.log.json')):
            raise RuntimeError('Existing full training: resume explicitly through launch.py, not the pipeline')
        scripts = ['prepare_trainval.py', 'make_configs.py', 'training_support.py', 'launch.py', 'pipeline.py']
        state = dict(state='running', phase='starting', pipeline_pid=os.getpid(), started_at=now(),
                     task_root=str(task), repo_root=str(repo), data_root=str(data),
                     scripts_sha256={name: hashlib.sha256((task / name).read_bytes()).hexdigest() for name in scripts},
                     completed_stages=[])
        status = task / 'pipeline-status.json'
        save(status, state)
        preparation = [sys.executable, '-u', str(task / 'prepare_trainval.py'),
                       '--repo-root', str(repo), '--data-root', str(data), '--output-dir', str(task / 'infos')]
        if all((task / 'infos' / ('nuscenes_infos_temporal_%s.pkl' % split)).is_file() for split in ('train', 'val')):
            preparation.append('--reuse-infos')
        configs = [sys.executable, str(task / 'make_configs.py'), '--repo-root', str(repo),
                   '--data-root', str(data), '--task-root', str(task)]
        launch = [sys.executable, '-u', str(task / 'launch.py'), '--task-root', str(task), '--repo-root', str(repo)]
        stages = [('prepare', preparation), ('configs', configs),
                  ('smoke', launch + ['--stage', 'smoke']), ('train', launch + ['--stage', 'train'])]
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(task) + os.pathsep + str(repo) + os.pathsep + environment.get('PYTHONPATH', '')
        try:
            for name, command in stages:
                state.update(phase=name, stage_started_at=now(), updated_at=now(), log=str(logs / (name + '.log')))
                save(status, state)
                with (logs / (name + '.log')).open('ab', buffering=0) as log:
                    child = subprocess.Popen(command, cwd=str(repo), env=environment,
                                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    state['child_pid'] = child.pid
                    save(status, state)
                    code = child.wait()
                if code != 0:
                    raise RuntimeError('%s stage exited with code %d; see %s' % (name, code, state['log']))
                state['completed_stages'].append(name)
                save(status, state)
            state.update(state='completed', phase='completed', finished_at=now(), updated_at=now())
            save(status, state)
        except BaseException as exc:
            state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc),
                         finished_at=now(), updated_at=now())
            save(status, state)
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
