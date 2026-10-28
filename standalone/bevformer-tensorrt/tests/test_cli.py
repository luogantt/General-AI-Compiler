"""The installed entry point must not need the original checkout or a GPU."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from bevformer_trt import __version__


class StandaloneCLITests(unittest.TestCase):
    def invoke(self, *arguments):
        # -S removes all site packages, including torch, tensorrt and gac.
        # Only this independent package is placed on the import path. Run from
        # an unrelated directory so accidental source-tree imports cannot work.
        source = Path(__file__).resolve().parents[1] / "src"
        environment = dict(os.environ, PYTHONPATH=str(source))
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(
                [sys.executable, "-S", "-m", "bevformer_trt", *arguments],
                cwd=directory, env=environment, capture_output=True, text=True,
                check=False,
            )

    def test_help_and_version_work_without_any_third_party_packages(self):
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare-calibration", result.stdout)
        self.assertIn("verify-plugins", result.stdout)
        result = self.invoke("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), __version__)

    def test_gpu_command_help_does_not_initialize_or_import_cuda(self):
        for command in ("check", "build", "run", "profile"):
            with self.subTest(command=command):
                result = self.invoke(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--device", result.stdout)

    def test_unknown_command_fails_without_importing_runtime(self):
        result = self.invoke("not-a-command")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command", result.stderr)


if __name__ == "__main__":
    unittest.main()
