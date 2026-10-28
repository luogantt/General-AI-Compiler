"""Linux process ownership for training supervisors (no CUDA imports).

Each launched child starts a new session. Supervisors are subreapers, so a
launcher exiting early cannot leave its workers outside the cleanup tree.
SIGINT/SIGTERM are recorded, then acted on at safe points after Popen returns.
"""

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class StopRequested(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__('Received ' + signal.Signals(signum).name)


def process_table():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(directory.name)] = {
                'state': fields[0], 'parent': int(fields[1]),
                'group': int(fields[2]), 'start': int(fields[19])}
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def descendants(table, parent):
    found, pending = {}, [parent]
    children = {}
    for pid, record in table.items():
        children.setdefault(record['parent'], []).append(pid)
    while pending:
        for pid in children.get(pending.pop(), []):
            if pid not in found:
                found[pid] = table[pid]
                pending.append(pid)
    return found


class ProcessScope:
    """Keep caller-held locks until all owned descendants have exited.

    After a bounded grace period, remaining processes receive SIGKILL. If a
    process is stuck in uninterruptible kernel I/O, keep the lock until it
    actually exits; claiming completion and releasing it would be unsafe.
    """

    def __init__(self, grace_seconds=20.0, poll_seconds=0.1):
        self.grace_seconds = grace_seconds
        self.poll_seconds = poll_seconds
        self.signum = None
        self.children = []
        self.closed = False

    def __enter__(self):
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Training process supervision requires Linux /proc.')
        self.libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_GET_CHILD_SUBREAPER failed')
        self.previous_subreaper = previous.value
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
        self.handlers = {number: signal.getsignal(number)
                         for number in (signal.SIGINT, signal.SIGTERM)}
        for number in self.handlers:
            signal.signal(number, self._record_signal)
        return self

    def _record_signal(self, signum, frame):
        # Do not throw between fork/exec and registering the returned child.
        if self.signum is None:
            self.signum = signum

    def check_stop(self):
        if self.signum is not None:
            raise StopRequested(self.signum)

    def spawn(self, command, **kwargs):
        self.check_stop()
        if self.closed:
            raise RuntimeError('Cannot launch a child after its process scope was closed.')
        if 'start_new_session' in kwargs:
            raise ValueError('ProcessScope owns the child session policy.')
        child = subprocess.Popen(command, start_new_session=True, **kwargs)
        self.children.append(child)
        self.check_stop()
        return child

    def wait(self, child):
        while True:
            self.check_stop()
            code = child.poll()
            if code is not None:
                return code
            time.sleep(self.poll_seconds)

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check_stop()
            time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))
        self.check_stop()

    def _owned(self):
        for child in self.children:
            child.poll()
        owned = descendants(process_table(), os.getpid())
        direct = {child.pid for child in self.children}
        for pid, record in owned.items():
            if record['state'] == 'Z' and record['parent'] == os.getpid() and pid not in direct:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
        return {pid: record for pid, record in descendants(process_table(), os.getpid()).items()
                if record['state'] not in ('Z', 'X')}

    @staticmethod
    def _signal_process(pid, record, signum):
        try:
            fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) != record['start'] or int(fields[2]) != record['group']:
                return  # Never signal a recycled PID/process group from an old snapshot.
            # Groups include DataLoader workers; scanning descendants also finds
            # nested supervisors whose children started their own sessions.
            if record['group'] != os.getpgrp():
                os.killpg(record['group'], signum)
            else:
                os.kill(pid, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass

    def close(self):
        if self.closed:
            return
        deadline = time.monotonic() + self.grace_seconds
        signaled = set()
        while True:
            owned = self._owned()
            if not owned:
                self._owned()  # Reap descendants adopted when the leader exited.
                self.closed = True
                return
            signum = signal.SIGKILL if time.monotonic() >= deadline else (self.signum or signal.SIGTERM)
            for pid, record in owned.items():
                identity = (pid, record['start'], signum)
                if identity not in signaled:
                    self._signal_process(pid, record, signum)
                    signaled.add(identity)
            time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        finally:
            for number, handler in self.handlers.items():
                signal.signal(number, handler)
            self.libc.prctl(36, self.previous_subreaper, 0, 0, 0)
