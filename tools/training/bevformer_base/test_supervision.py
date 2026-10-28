"""CPU-only Linux regression tests for signals, gates, and preparation races."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from continue_train import poll_preparation
from launch import file_hash, require_smoke_evidence, runtime_fingerprint, save
from process_support import ProcessScope, StopRequested


WORKER = r'''
import fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
from process_support import ProcessScope, StopRequested
role, directory, kind = sys.argv[1:]
root = Path(directory)
(root / (role + '.pid')).write_text(str(os.getpid()))
if role == 'leaf':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    (root / 'leaf-ready').write_text('ready')
    while True: time.sleep(1)
elif role == 'child':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    subprocess.Popen([sys.executable, __file__, 'leaf', directory, kind], start_new_session=True)
    while not (root / 'leaf-ready').exists(): time.sleep(.01)
    if kind == 'orphan': sys.exit(0)
    while True: time.sleep(1)
elif role == 'nested':
    result = 0
    with ProcessScope(grace_seconds=.2, poll_seconds=.02) as scope:
        try:
            child = scope.spawn([sys.executable, __file__, 'child', directory, kind])
            result = scope.wait(child)
        except StopRequested as exc:
            result = 128 + exc.signum
        finally:
            scope.close()
            (root / 'nested-cleaned').write_text(str(result))
    sys.exit(result)
else:
    result = 0
    with (root / 'training.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ProcessScope(grace_seconds=.8, poll_seconds=.02) as scope:
            try:
                child = scope.spawn([sys.executable, __file__,
                                     'nested' if kind == 'nested' else 'child', directory, kind])
                while not (root / 'leaf-ready').exists(): scope.sleep(.01)
                (root / 'ready').write_text('ready')
                result = scope.wait(child)
            except StopRequested as exc:
                (root / 'received-signal').write_text(str(exc.signum))
                result = 128 + exc.signum
            finally:
                scope.close()
                (root / 'cleanup-finished').write_text('done')
    sys.exit(result)
'''


def wait_file(path, process=None, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError('Test supervisor exited early with %s' % process.returncode)
        time.sleep(.01)
    raise AssertionError('Timed out waiting for ' + str(path))


class ProcessTests(unittest.TestCase):
    def run_tree(self, signum, nested=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker.py'
            worker.write_text(WORKER)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent))
            process = subprocess.Popen([sys.executable, str(worker), 'supervisor', directory,
                                        'nested' if nested else 'normal'], env=env,
                                       start_new_session=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                wait_file(root / 'ready', process)
                os.kill(process.pid, signum)  # Signal only the owner, not its children.
                wait_file(root / 'received-signal', process)
                if not nested:
                    with (root / 'training.lock').open('a+') as lock:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 128 + signum, output)
                self.assertTrue((root / 'cleanup-finished').exists())
                if nested:
                    self.assertTrue((root / 'nested-cleaned').exists(), output)
                for path in root.glob('*.pid'):
                    pid = int(path.read_text())
                    self.assertFalse(Path('/proc/%d' % pid).exists(), 'Leaked ' + path.name)
                with (root / 'training.lock').open('a+') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                if process.poll() is None:
                    os.kill(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)

    def test_sigterm_cleans_child_grandchild_and_holds_lock(self):
        self.run_tree(signal.SIGTERM)

    def test_sigint_cleans_child_grandchild_and_holds_lock(self):
        self.run_tree(signal.SIGINT)

    def test_nested_supervisor_runs_its_cleanup(self):
        self.run_tree(signal.SIGTERM, nested=True)

    def test_leader_success_does_not_leak_an_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker.py'
            worker.write_text(WORKER)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent))
            with ProcessScope(grace_seconds=.15, poll_seconds=.01) as scope:
                child = scope.spawn([sys.executable, str(worker), 'child', directory, 'orphan'], env=env)
                self.assertEqual(scope.wait(child), 0)
            self.assertFalse(Path('/proc/%s' % (root / 'leaf.pid').read_text()).exists())

    def test_signal_between_popen_and_registration_is_safe(self):
        actual = subprocess.Popen
        created = []

        def interrupt_after_spawn(*args, **kwargs):
            child = actual(*args, **kwargs)
            created.append(child.pid)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        with self.assertRaises(StopRequested):
            with ProcessScope(grace_seconds=.15, poll_seconds=.01) as scope:
                with mock.patch('process_support.subprocess.Popen', side_effect=interrupt_after_spawn):
                    scope.spawn([sys.executable, '-c', 'import time; time.sleep(60)'])
        self.assertEqual(len(created), 1)
        self.assertFalse(Path('/proc/%d' % created[0]).exists())


class GateTests(unittest.TestCase):
    def test_prepare_completion_race_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            state = task / 'data-preparation-trainval-status.json'
            save(state, {'state': 'running'})

            def exited_after_completing(pid):
                save(task / 'preparation_trainval_report.json', {
                    'passed': True, 'version': 'v1.0-trainval', 'profile': 'camera_training',
                    'data_root': str(task / 'data/trainval'),
                    'train': {'samples': 28130}, 'val': {'samples': 6019}})
                save(state, {'state': 'completed'})
                raise FileNotFoundError('process exited between the reads')

            prep, done = poll_preparation(task, 123, read_command=exited_after_completing)
            self.assertTrue(done)
            self.assertEqual(prep['state'], 'completed')

    def test_prepare_crash_without_completed_report_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            save(task / 'data-preparation-trainval-status.json', {'state': 'running'})
            with self.assertRaisesRegex(RuntimeError, 'without a completed'):
                poll_preparation(task, 123, read_command=lambda pid: b'')

    def make_stack(self, root):
        task, repo = root / 'task', root / 'repo'
        files = {
            task / 'configs/base_smoke.py': "_base_=['./base_trainval.py']\n",
            task / 'configs/base_trainval.py': "_base_=['../../repo/base.py']\n",
            repo / 'base.py': 'model = dict()\n',
            task / 'base_training_support.py': 'from checkpoint_layers import Layer\n',
            task / 'checkpoint_layers.py': 'class Layer: pass\n',
            repo / 'tools/fp16/train.py': '# train\n',
            repo / 'projects/mmdet3d_plugin/model.py': '# model\n',
            task / 'infos/nuscenes_infos_temporal_smoke_train.pkl': 'train',
            task / 'infos/nuscenes_infos_temporal_smoke_val.pkl': 'val',
            task / 'weights.pth': 'weights',
        }
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return task, repo, task / 'weights.pth'

    def test_ddp_files_do_not_invalidate_single_but_model_and_weights_do(self):
        with tempfile.TemporaryDirectory() as directory:
            task, repo, weights = self.make_stack(Path(directory))
            single = runtime_fingerprint(task, repo, weights)
            (task / 'configs/base_ddp_smoke.py').write_text("_base_=['./base_smoke.py']\n")
            (task / 'ddp_preflight.py').write_text('from base_training_support import Layer\n')
            self.assertEqual(single, runtime_fingerprint(task, repo, weights))
            ddp = runtime_fingerprint(task, repo, weights, 'ddp-smoke')
            (task / 'ddp_preflight.py').write_text('# changed DDP implementation\n')
            self.assertEqual(single, runtime_fingerprint(task, repo, weights))
            self.assertNotEqual(ddp, runtime_fingerprint(task, repo, weights, 'ddp-smoke'))
            (task / 'checkpoint_layers.py').write_text('class ChangedLayer: pass\n')
            self.assertNotEqual(single, runtime_fingerprint(task, repo, weights))
            baseline = runtime_fingerprint(task, repo, weights)
            weights.write_text('different weights')
            self.assertNotEqual(baseline, runtime_fingerprint(task, repo, weights))

    def test_success_report_is_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            task, repo, weights = self.make_stack(Path(directory))
            work = task / 'work_dirs/smoke'
            work.mkdir(parents=True)
            report = work / 'base_smoke_check.json'
            save(report, {'passed': True})
            save(work / 'runtime-evidence.json', dict(
                passed=True, stage='smoke', smoke_report_sha256=file_hash(report),
                runtime=runtime_fingerprint(task, repo, weights)))
            require_smoke_evidence(task, repo, weights, 'smoke')
            save(report, {'passed': True, 'changed': True})
            with self.assertRaisesRegex(RuntimeError, 'changed since'):
                require_smoke_evidence(task, repo, weights, 'smoke')


if __name__ == '__main__':
    unittest.main()
