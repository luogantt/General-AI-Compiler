"""Wait for an existing full-data preparation job, then launch Base training.

Start only after the independent real-data memory/gradient smoke test passes.
The preparation PID is verified against the expected task and command; failures
stop the pipeline instead of treating missing data as a reason to train on mini.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from launch import require_passed, require_training_evidence, save
from process_support import ProcessScope, StopRequested


def completed_preparation(task, prep):
    if prep['state'] == 'failed':
        raise RuntimeError('Full dataset preparation failed: ' + prep.get('error', 'unknown'))
    if prep['state'] != 'completed':
        return False
    report = require_passed(task / 'preparation_trainval_report.json')
    if (report.get('version') != 'v1.0-trainval' or
            report.get('profile') != 'camera_training' or
            Path(report['data_root']).resolve() != (task / 'data/trainval').resolve() or
            report.get('train', {}).get('samples') != 28130 or
            report.get('val', {}).get('samples') != 6019):
        raise RuntimeError('Full dataset report does not match this experiment.')
    return True


def poll_preparation(task, pid, read_command=None):
    """Recheck completion after /proc disappears between the two reads."""
    status = task / 'data-preparation-trainval-status.json'
    prep = json.loads(status.read_text(encoding='utf-8'))
    if completed_preparation(task, prep):
        return prep, True
    if read_command is None:
        read_command = lambda value: Path('/proc/%d/cmdline' % value).read_bytes()
    try:
        command = read_command(pid).split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        command = []
    if not command or command == [b'']:
        # Successful preparation publishes report then completed status before
        # exiting. The earlier "running" snapshot is no longer authoritative.
        prep = json.loads(status.read_text(encoding='utf-8'))
        if completed_preparation(task, prep):
            return prep, True
        raise RuntimeError('Preparation process exited without a completed, passed report.')
    if (str(task / 'prepare_data.py').encode() not in command
            or str(task).encode() not in command or b'trainval' not in command):
        raise RuntimeError('Preparation process identity no longer matches this task.')
    return prep, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--preparation-pid', type=int, required=True)
    parser.add_argument('--gpus', type=int, default=3)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    task, repo = args.task_root.resolve(), args.repo_root.resolve()
    require_training_evidence(task, repo)
    logs = task / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve())]
        command += [value for value in sys.argv[1:] if value != '--detach']
        with (logs / 'pipeline.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pipeline_pid': child.pid, 'status': str(task / 'pipeline-status.json')}))
        return 0
    import fcntl
    with (task / '.pipeline.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = task / 'pipeline-status.json'
        state = dict(state='running', phase='await_full_data', pipeline_pid=os.getpid(),
                     preparation_pid=args.preparation_pid, world_size=args.gpus,
                     task_root=str(task), started_at=datetime.now(timezone.utc).isoformat())
        save(status, state)
        with ProcessScope(grace_seconds=30) as processes:
            try:
                while True:
                    processes.check_stop()
                    prep, complete = poll_preparation(task, args.preparation_pid)
                    if complete:
                        break
                    state.update(data_phase=prep.get('phase'), updated_at=datetime.now(timezone.utc).isoformat())
                    save(status, state)
                    processes.sleep(30)
                # Evidence is checked again by launch.py after the wait. It may
                # have been invalidated by a new configuration or smoke attempt.
                state.update(phase='train', updated_at=datetime.now(timezone.utc).isoformat())
                save(status, state)
                command = [sys.executable, '-u', str(task / 'launch.py'), '--task-root', str(task),
                           '--repo-root', str(repo), '--stage', 'train', '--gpus', str(args.gpus)]
                with (logs / 'train.log').open('ab', buffering=0) as stream:
                    child = processes.spawn(command, stdin=subprocess.DEVNULL, stdout=stream,
                                            stderr=subprocess.STDOUT, cwd=str(repo))
                    state['training_supervisor_pid'] = child.pid
                    save(status, state)
                    code = processes.wait(child)
                processes.close()
                if code:
                    raise RuntimeError('Training failed with exit code %d; see logs/train.log' % code)
                state.update(state='completed', phase='completed')
            except StopRequested as exc:
                state.update(state='stopped', signal=exc.signum, exit_code=128 + exc.signum)
                return 128 + exc.signum
            except BaseException as exc:
                state.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
                raise
            finally:
                processes.close()
                state['finished_at'] = datetime.now(timezone.utc).isoformat()
                save(status, state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
