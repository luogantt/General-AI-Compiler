from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from gac.mlir_toolchain import (
    MAX_STDERR_CHARS,
    MlirOptNotFoundError,
    MlirToolchainError,
    MlirVerificationError,
    discover_mlir_opt,
    probe_mlir_opt_source_revision,
    probe_mlir_opt_version,
    verify_mlir,
)


MINIMAL_MLIR = """module {
  func.func @identity(%arg0: i32) -> i32 {
    return %arg0 : i32
  }
}
"""


def make_executable(directory: Path, name: str | None = None) -> Path:
    filename = name or ("mlir-opt.exe" if os.name == "nt" else "mlir-opt")
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("mock executable\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class MlirDiscoveryTests(unittest.TestCase):
    def test_explicit_path_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = make_executable(root / "explicit")
            configured = make_executable(root / "configured")
            result = discover_mlir_opt(
                explicit,
                env={"GAC_MLIR_OPT": str(configured), "PATH": ""},
            )
            self.assertEqual(result, explicit.resolve())

    def test_invalid_explicit_path_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = make_executable(root / "configured")
            with mock.patch("gac.mlir_toolchain.shutil.which") as which:
                with self.assertRaisesRegex(MlirOptNotFoundError, "explicit"):
                    discover_mlir_opt(
                        root / "missing" / "mlir-opt",
                        env={"GAC_MLIR_OPT": str(configured), "PATH": str(root)},
                    )
            which.assert_not_called()

    def test_invalid_environment_override_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = make_executable(root / "fallback")
            with mock.patch("gac.mlir_toolchain.shutil.which", return_value=str(fallback)) as which:
                with self.assertRaisesRegex(MlirOptNotFoundError, "GAC_MLIR_OPT"):
                    discover_mlir_opt(
                        env={
                            "GAC_MLIR_OPT": str(root / "missing" / "mlir-opt"),
                            "PATH": str(root),
                        }
                    )
            which.assert_not_called()

    def test_path_precedes_llvm_root_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from_path = make_executable(root / "path")
            make_executable(root / "llvm" / "build" / "bin")
            with mock.patch("gac.mlir_toolchain.shutil.which", return_value=str(from_path)):
                result = discover_mlir_opt(
                    env={"PATH": str(from_path.parent), "LLVM_ROOT": str(root / "llvm")}
                )
            self.assertEqual(result, from_path.resolve())

    def test_llvm_root_checks_build_then_build_mlir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "llvm"
            build = make_executable(root / "build" / "bin")
            build_mlir = make_executable(root / "build-mlir" / "bin")
            with mock.patch("gac.mlir_toolchain.shutil.which", return_value=None):
                self.assertEqual(
                    discover_mlir_opt(env={"PATH": "", "LLVM_ROOT": str(root)}),
                    build.resolve(),
                )
                build.unlink()
                self.assertEqual(
                    discover_mlir_opt(env={"PATH": ""}, llvm_root=root),
                    build_mlir.resolve(),
                )

    def test_missing_tool_has_actionable_error(self):
        with mock.patch("gac.mlir_toolchain.shutil.which", return_value=None):
            with self.assertRaisesRegex(MlirOptNotFoundError, "GAC_MLIR_OPT"):
                discover_mlir_opt(env={"PATH": ""})


class MlirVersionProbeTests(unittest.TestCase):
    def test_local_build_marker_accepts_only_complete_git_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            executable = make_executable(build / "bin")
            marker = build / ".gac-build-revision"

            for revision in ("A" * 40, "b" * 64):
                with self.subTest(length=len(revision)):
                    marker.write_text(f"{revision}\n", encoding="utf-8")
                    self.assertEqual(
                        probe_mlir_opt_source_revision(executable), revision.lower()
                    )

    def test_missing_and_invalid_local_build_markers_return_none(self):
        invalid_values = (
            "",
            "a" * 39,
            "a" * 41,
            "g" * 40,
            f"{'a' * 40}\n{'b' * 40}",
        )
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            executable = make_executable(build / "bin")
            marker = build / ".gac-build-revision"

            self.assertIsNone(probe_mlir_opt_source_revision(executable))
            for value in invalid_values:
                with self.subTest(value=value):
                    marker.write_text(value, encoding="utf-8")
                    self.assertIsNone(
                        probe_mlir_opt_source_revision(executable)
                    )

    def test_local_probe_uses_argv_and_normalizes_version_banner(self):
        completed = subprocess.CompletedProcess(
            ["mock-mlir-opt", "--version"],
            0,
            stdout="\r\nLLVM version 24.0.0git  \r\n  Optimized build.\r\n",
            stderr="",
        )
        with mock.patch(
            "gac.mlir_toolchain.subprocess.run", return_value=completed
        ) as run:
            version = probe_mlir_opt_version(
                command=("mock-mlir-opt",), timeout=6
            )

        self.assertEqual(version, "LLVM version 24.0.0git\n  Optimized build.")
        run.assert_called_once()
        argv = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        self.assertEqual(argv, ["mock-mlir-opt", "--version"])
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["timeout"], 6.0)

    def test_probe_failure_and_empty_banner_are_actionable(self):
        failed = subprocess.CompletedProcess(
            ["mock-mlir-opt", "--version"],
            3,
            stdout="",
            stderr="loader error",
        )
        with mock.patch("gac.mlir_toolchain.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(
                MlirToolchainError, "--version exited with status 3"
            ) as raised:
                probe_mlir_opt_version(command=("mock-mlir-opt",))
        self.assertIn("loader error", str(raised.exception))

        empty = subprocess.CompletedProcess(
            ["mock-mlir-opt", "--version"], 0, stdout=" \r\n", stderr=""
        )
        with mock.patch("gac.mlir_toolchain.subprocess.run", return_value=empty):
            with self.assertRaisesRegex(MlirToolchainError, "no version text"):
                probe_mlir_opt_version(command=("mock-mlir-opt",))

    @unittest.skipUnless(os.name == "nt", "WSL command wrapping is Windows-specific")
    def test_wsl_probe_reuses_shell_free_command_wrapper(self):
        linux_tool = "/home/thomas/gac-mlir-build-lf/bin/mlir-opt"
        completed = subprocess.CompletedProcess(
            [], 0, stdout="LLVM version 24.0.0git\n", stderr=""
        )
        with mock.patch(
            "gac.mlir_toolchain.subprocess.run", return_value=completed
        ) as run:
            version = probe_mlir_opt_version(
                command=(linux_tool,),
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
            )

        self.assertEqual(version, "LLVM version 24.0.0git")
        self.assertEqual(
            run.call_args.args[0],
            [
                "wsl.exe",
                "-d",
                "Ubuntu-24.04",
                "--",
                linux_tool,
                "--version",
            ],
        )
        self.assertIs(run.call_args.kwargs["shell"], False)

    @unittest.skipUnless(os.name == "nt", "WSL command wrapping is Windows-specific")
    def test_wsl_build_marker_uses_stable_build_link_with_argv_only_cat(self):
        revision = "765392a6b4498f1dc57233f03a31b87462fdef01"
        linux_tool = "/home/thomas/.cache/gac-mlir/current-build/bin/mlir-opt"
        completed = subprocess.CompletedProcess(
            [], 0, stdout=f"{revision}\n", stderr=""
        )
        with mock.patch(
            "gac.mlir_toolchain.subprocess.run", return_value=completed
        ) as run:
            result = probe_mlir_opt_source_revision(
                linux_tool,
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
                timeout=4,
            )

        self.assertEqual(result, revision)
        self.assertEqual(
            run.call_args.args[0],
            [
                "wsl.exe",
                "-d",
                "Ubuntu-24.04",
                "--",
                "cat",
                "/home/thomas/.cache/gac-mlir/current-build/.gac-build-revision",
            ],
        )
        kwargs = run.call_args.kwargs
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["timeout"], 4.0)

    @unittest.skipUnless(os.name == "nt", "WSL command wrapping is Windows-specific")
    def test_wsl_missing_or_invalid_build_marker_returns_none(self):
        linux_tool = "/opt/mlir/build/bin/mlir-opt"
        outcomes = (
            subprocess.CompletedProcess([], 1, stdout="", stderr="not found"),
            subprocess.CompletedProcess([], 0, stdout="not-a-revision\n", stderr=""),
        )
        for completed in outcomes:
            with self.subTest(returncode=completed.returncode, stdout=completed.stdout):
                with mock.patch(
                    "gac.mlir_toolchain.subprocess.run", return_value=completed
                ):
                    self.assertIsNone(
                        probe_mlir_opt_source_revision(
                            linux_tool,
                            use_wsl=True,
                            wsl_distribution="Ubuntu-24.04",
                        )
                    )


class MlirVerificationTests(unittest.TestCase):
    def test_runs_with_argv_verifies_each_pass_and_cleans_temporary_files(self):
        observed: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            observed["argv"] = argv
            observed["kwargs"] = kwargs
            output = Path(argv[argv.index("-o") + 1])
            input_path = Path(argv[argv.index("-o") - 1])
            observed["input"] = input_path
            observed["output"] = output
            self.assertEqual(input_path.read_text(encoding="utf-8"), MINIMAL_MLIR)
            output.write_text("module {\n}\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with mock.patch("gac.mlir_toolchain.subprocess.run", side_effect=fake_run):
            result = verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",), timeout=7)

        self.assertEqual(result, "module {\n}\n")
        argv = observed["argv"]
        self.assertEqual(argv[0], "mock-mlir-opt")
        self.assertIn("--verify-each", argv)
        self.assertIn("--canonicalize", argv)
        self.assertIn("--cse", argv)
        kwargs = observed["kwargs"]
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["timeout"], 7.0)
        self.assertFalse(observed["input"].exists())
        self.assertFalse(observed["output"].exists())

    def test_real_child_process_receives_files_and_writes_verified_module(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "mock_mlir_opt.py"
            script.write_text(
                """import pathlib
import sys

args = sys.argv[1:]
required = {"--verify-each", "--canonicalize", "--cse"}
if not required.issubset(args):
    raise SystemExit(11)
output_index = args.index("-o")
input_path = pathlib.Path(args[output_index - 1])
output_path = pathlib.Path(args[output_index + 1])
source = input_path.read_text(encoding="utf-8")
output_path.write_text(source.replace("identity", "verified"), encoding="utf-8")
""",
                encoding="utf-8",
            )
            result = verify_mlir(
                MINIMAL_MLIR,
                command=(sys.executable, str(script)),
            )
        self.assertIn("@verified", result)

    def test_nonzero_status_raises_with_bounded_stderr(self):
        stderr = "start:" + "x" * (MAX_STDERR_CHARS * 3) + ":end"
        completed = subprocess.CompletedProcess(
            ["mock-mlir-opt"], 9, stdout="", stderr=stderr
        )
        with mock.patch("gac.mlir_toolchain.subprocess.run", return_value=completed):
            with self.assertRaises(MlirVerificationError) as raised:
                verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",))
        message = str(raised.exception)
        self.assertIn("status 9", message)
        self.assertIn("stderr truncated", message)
        self.assertLess(len(message), MAX_STDERR_CHARS + 256)

    def test_invalid_utf8_stderr_is_replaced_instead_of_masking_tool_failure(self):
        script = (
            "import os; "
            "os.write(2, bytes([98, 97, 100, 58, 255])); "
            "raise SystemExit(7)"
        )
        with self.assertRaises(MlirVerificationError) as raised:
            verify_mlir(MINIMAL_MLIR, command=(sys.executable, "-c", script))

        message = str(raised.exception)
        self.assertIn("status 7", message)
        self.assertIn("bad:\ufffd", message)

    def test_timeout_is_translated_and_stderr_is_bounded(self):
        failure = subprocess.TimeoutExpired(
            ["mock-mlir-opt"], 0.01, stderr="diagnostic" * 4_000
        )
        with mock.patch("gac.mlir_toolchain.subprocess.run", side_effect=failure):
            with self.assertRaises(MlirVerificationError) as raised:
                verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",), timeout=0.01)
        self.assertIn("timed out", str(raised.exception))
        self.assertLess(len(str(raised.exception)), MAX_STDERR_CHARS + 256)

    def test_missing_and_empty_output_are_rejected(self):
        success = subprocess.CompletedProcess(["mock-mlir-opt"], 0, stdout="", stderr="note")
        with mock.patch("gac.mlir_toolchain.subprocess.run", return_value=success):
            with self.assertRaisesRegex(MlirVerificationError, "readable output"):
                verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",))

        def write_empty(argv, **kwargs):
            Path(argv[argv.index("-o") + 1]).write_text(" \n", encoding="utf-8")
            return success

        with mock.patch("gac.mlir_toolchain.subprocess.run", side_effect=write_empty):
            with self.assertRaisesRegex(MlirVerificationError, "empty output"):
                verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",))

    def test_launch_error_is_translated(self):
        with mock.patch(
            "gac.mlir_toolchain.subprocess.run", side_effect=OSError("cannot execute")
        ):
            with self.assertRaisesRegex(MlirToolchainError, "failed to launch"):
                verify_mlir(MINIMAL_MLIR, command=("mock-mlir-opt",))

    def test_argument_validation(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            verify_mlir(
                MINIMAL_MLIR,
                mlir_opt="tool",
                command=("other-tool",),
            )
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=value):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    verify_mlir(MINIMAL_MLIR, command=("tool",), timeout=value)
        with self.assertRaisesRegex(ValueError, "command"):
            verify_mlir(MINIMAL_MLIR, command=())

    @unittest.skipUnless(os.name == "nt", "WSL command wrapping is Windows-specific")
    def test_explicit_wsl_command_is_wrapped_without_a_shell(self):
        observed: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            observed["argv"] = argv
            observed["kwargs"] = kwargs
            wsl_output = argv[argv.index("-o") + 1]
            self.assertRegex(wsl_output, r"^/mnt/[a-z]/")
            drive, *parts = wsl_output.removeprefix("/mnt/").split("/")
            output = Path(f"{drive.upper()}:\\", *parts)
            output.write_text("module {\n}\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with mock.patch("gac.mlir_toolchain.subprocess.run", side_effect=fake_run):
            verify_mlir(
                MINIMAL_MLIR,
                command=("/home/thomas/gac-mlir-build-lf/bin/mlir-opt",),
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
            )

        argv = observed["argv"]
        self.assertEqual(
            argv[:5],
            [
                "wsl.exe",
                "-d",
                "Ubuntu-24.04",
                "--",
                "/home/thomas/gac-mlir-build-lf/bin/mlir-opt",
            ],
        )
        self.assertIs(observed["kwargs"]["shell"], False)


class MlirRealIntegrationTests(unittest.TestCase):
    def test_installed_mlir_opt_when_available_or_required(self):
        required = os.environ.get("GAC_REQUIRE_MLIR") == "1"
        try:
            executable = discover_mlir_opt()
        except MlirOptNotFoundError as exc:
            if required:
                self.fail(f"GAC_REQUIRE_MLIR=1 but mlir-opt is unavailable: {exc}")
            self.skipTest(str(exc))

        try:
            verified = verify_mlir(MINIMAL_MLIR, mlir_opt=executable)
        except MlirToolchainError:
            if required:
                raise
            self.fail(f"discovered mlir-opt is unusable: {executable}")
        self.assertIn("module", verified)
        self.assertIn("func.func @identity", verified)


if __name__ == "__main__":
    unittest.main()
