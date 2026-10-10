"""CLI coverage for the explicit MLIR compilation and verification path."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gac.cli import main
from gac.mlir_toolchain import MlirToolchainError, MlirVerificationError


ROOT = Path(__file__).resolve().parents[1]
MLIR_OPT_VERSION = "LLVM version 24.0.0git\n  Optimized build."
LLVM_SOURCE_REVISION = "765392a6b4498f1dc57233f03a31b87462fdef01"


class _FakeProgram:
    source = "module {\n}\n"
    graph = SimpleNamespace(name="fake_graph")
    pass_reports = (
        {"name": "shape-inference", "changed": False, "details": {}},
    )

    def describe(self):
        return {
            "schema_version": 1,
            "compiler_backend": "mlir-linalg",
            "target_intent": "cuda-sm86",
            "planned_codegen_backend": "mlir-nvvm",
            "lowering_stage": "tensor",
            "executable": False,
            "artifact_kind": "textual-mlir-module",
            "name": "fake_graph",
            "dialects": ["builtin", "func", "tensor", "linalg", "arith"],
            "pass_reports": list(self.pass_reports),
        }


def _invoke(arguments: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = main(arguments)
    return result, stdout.getvalue(), stderr.getvalue()


class MlirCliTests(unittest.TestCase):
    def test_verified_local_compile_writes_normalized_artifact_and_report(self):
        normalized = "module {\n  // canonicalized\n}\n"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.mlir"
            report_path = Path(directory) / "report.json"
            selected_tool = Path(directory) / "llvm" / "bin" / "mlir-opt.exe"
            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()) as compile_mock,
                patch(
                    "gac.cli.discover_mlir_opt", return_value=selected_tool
                ) as discover_mock,
                patch(
                    "gac.cli.probe_mlir_opt_version",
                    return_value=MLIR_OPT_VERSION,
                ) as version_mock,
                patch(
                    "gac.cli.probe_mlir_opt_source_revision",
                    return_value=LLVM_SOURCE_REVISION,
                ) as revision_mock,
                patch("gac.cli.verify_mlir", return_value=normalized) as verify_mock,
            ):
                code, stdout, stderr = _invoke(
                    [
                        "mlir-compile",
                        "model.json",
                        "-o",
                        str(output),
                        "--no-fuse",
                        "--mlir-opt",
                        "C:/llvm/bin/mlir-opt.exe",
                        "--timeout",
                        "12.5",
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(code, 0, stderr)
            self.assertEqual(stderr, "")
            compile_mock.assert_called_once_with(unittest.mock.ANY, fuse=False)
            discover_mock.assert_called_once_with("C:/llvm/bin/mlir-opt.exe")
            version_mock.assert_called_once_with(
                command=(str(selected_tool),),
                timeout=12.5,
            )
            revision_mock.assert_called_once_with(
                str(selected_tool),
                timeout=12.5,
            )
            verify_mock.assert_called_once_with(
                _FakeProgram.source,
                command=(str(selected_tool),),
                timeout=12.5,
            )
            self.assertEqual(output.read_text(encoding="utf-8"), normalized)
            report = json.loads(stdout)
            self.assertEqual(report, json.loads(report_path.read_text(encoding="utf-8")))
            self.assertEqual(report["target_intent"], "cuda-sm86")
            self.assertEqual(report["compiler_backend"], "mlir-linalg")
            self.assertEqual(report["planned_codegen_backend"], "mlir-nvvm")
            self.assertEqual(report["lowering_stage"], "tensor")
            self.assertTrue(report["verified"])
            self.assertEqual(report["verification"], "mlir-opt")
            self.assertIn("linalg", report["dialects"])
            self.assertEqual(
                report["passes"]["mlir_opt"],
                ["verify-each", "canonicalize", "cse"],
            )
            self.assertEqual(report["output_file"], str(output.resolve()))
            self.assertEqual(report["report_file"], str(report_path.resolve()))
            self.assertEqual(
                report["verification_tool"],
                {
                    "name": "mlir-opt",
                    "executable": str(selected_tool),
                    "command_prefix": [str(selected_tool)],
                    "arguments": ["--verify-each", "--canonicalize", "--cse"],
                    "timeout_seconds": 12.5,
                    "version": MLIR_OPT_VERSION,
                    "source_revision": LLVM_SOURCE_REVISION,
                },
            )
            self.assertEqual(
                report["source_sha256"],
                hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            )

    def test_wsl_compile_passes_an_argv_tuple_without_shell_text(self):
        linux_tool = "/home/thomas/gac-mlir-build-lf/bin/mlir-opt"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.mlir"
            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()),
                patch(
                    "gac.cli.probe_mlir_opt_version",
                    return_value=MLIR_OPT_VERSION,
                ) as version_mock,
                patch(
                    "gac.cli.probe_mlir_opt_source_revision",
                    return_value=LLVM_SOURCE_REVISION,
                ) as revision_mock,
                patch("gac.cli.verify_mlir", return_value=_FakeProgram.source) as verify_mock,
            ):
                code, stdout, stderr = _invoke(
                    [
                        "mlir-compile",
                        "model.json",
                        "-o",
                        str(output),
                        "--wsl",
                        "--wsl-distribution",
                        "Ubuntu-24.04",
                        "--mlir-opt",
                        linux_tool,
                        "--timeout",
                        "17",
                    ]
                )

            self.assertEqual(code, 0, stderr)
            report = json.loads(stdout)
            self.assertEqual(report["verification"], "mlir-opt-via-wsl")
            self.assertEqual(
                report["verification_tool"]["command_prefix"],
                [
                    "wsl.exe",
                    "-d",
                    "Ubuntu-24.04",
                    "--",
                    linux_tool,
                ],
            )
            verify_mock.assert_called_once_with(
                _FakeProgram.source,
                command=(linux_tool,),
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
                timeout=17.0,
            )
            version_mock.assert_called_once_with(
                command=(linux_tool,),
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
                timeout=17.0,
            )
            revision_mock.assert_called_once_with(
                linux_tool,
                use_wsl=True,
                wsl_distribution="Ubuntu-24.04",
                timeout=17.0,
            )
            self.assertEqual(
                report["verification_tool"]["version"], MLIR_OPT_VERSION
            )
            self.assertEqual(
                report["verification_tool"]["source_revision"],
                LLVM_SOURCE_REVISION,
            )

    def test_invalid_build_marker_is_reported_as_null_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.mlir"
            selected_tool = root / "toolchain" / "bin" / "mlir-opt.exe"
            selected_tool.parent.mkdir(parents=True)
            (selected_tool.parent.parent / ".gac-build-revision").write_text(
                "main\n", encoding="utf-8"
            )
            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()),
                patch("gac.cli.discover_mlir_opt", return_value=selected_tool),
                patch(
                    "gac.cli.probe_mlir_opt_version",
                    return_value=MLIR_OPT_VERSION,
                ),
                patch("gac.cli.verify_mlir", return_value=_FakeProgram.source),
            ):
                code, stdout, stderr = _invoke(
                    ["mlir-compile", "model.json", "-o", str(output)]
                )

            self.assertEqual(code, 0, stderr)
            report = json.loads(stdout)
            self.assertIsNone(report["verification_tool"]["source_revision"])
            self.assertTrue(output.exists())

    def test_verification_failure_does_not_leave_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected.mlir"
            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()),
                patch(
                    "gac.cli.discover_mlir_opt",
                    return_value=Path(directory) / "mlir-opt.exe",
                ),
                patch(
                    "gac.cli.probe_mlir_opt_version",
                    return_value=MLIR_OPT_VERSION,
                ),
                patch(
                    "gac.cli.verify_mlir",
                    side_effect=MlirVerificationError("invalid linalg.generic"),
                ),
            ):
                code, stdout, stderr = _invoke(
                    ["mlir-compile", "model.json", "-o", str(output)]
                )

            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertTrue(stderr.startswith("gac: "), stderr)
            self.assertIn("invalid linalg.generic", stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_version_probe_failure_does_not_publish_or_verify_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.mlir"
            output.write_text("old artifact\n", encoding="utf-8")
            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()),
                patch(
                    "gac.cli.discover_mlir_opt",
                    return_value=Path(directory) / "mlir-opt.exe",
                ),
                patch(
                    "gac.cli.probe_mlir_opt_version",
                    side_effect=MlirToolchainError(
                        "mlir-opt --version produced no version text"
                    ),
                ),
                patch("gac.cli.verify_mlir") as verify_mock,
            ):
                code, stdout, stderr = _invoke(
                    ["mlir-compile", "model.json", "-o", str(output)]
                )

            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("--version produced no version text", stderr)
            verify_mock.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "old artifact\n")
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_emit_only_is_explicitly_unverified(self):
        model = ROOT / "examples" / "bevformer_tiny_ffn.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bevformer_tiny_ffn.mlir"
            with patch("gac.cli.verify_mlir") as verify_mock:
                code, stdout, stderr = _invoke(
                    [
                        "mlir-compile",
                        str(model),
                        "-o",
                        str(output),
                        "--emit-only",
                    ]
                )

            self.assertEqual(code, 0, stderr)
            verify_mock.assert_not_called()
            report = json.loads(stdout)
            self.assertFalse(report["verified"])
            self.assertEqual(report["verification"], "skipped-by-emit-only")
            self.assertIsNone(report["verification_tool"])
            self.assertEqual(report["passes"]["mlir_opt"], [])
            source = output.read_text(encoding="utf-8")
            self.assertIn("module {", source)
            self.assertIn("linalg.matmul", source)
            self.assertEqual(
                report["source_sha256"],
                hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )

    def test_wsl_verification_requires_linux_mlir_opt_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.mlir"
            code, stdout, stderr = _invoke(
                ["mlir-compile", "model.json", "-o", str(output), "--wsl"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("--wsl requires --mlir-opt", stderr)
            self.assertFalse(output.exists())

    def test_distribution_requires_wsl_and_output_requires_mlir_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_distribution = Path(directory) / "model.mlir"
            code, _, stderr = _invoke(
                [
                    "mlir-compile",
                    "model.json",
                    "-o",
                    str(invalid_distribution),
                    "--wsl-distribution",
                    "Ubuntu-24.04",
                    "--emit-only",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("--wsl-distribution requires --wsl", stderr)
            self.assertFalse(invalid_distribution.exists())

            invalid_suffix = Path(directory) / "model.txt"
            code, _, stderr = _invoke(
                [
                    "mlir-compile",
                    "model.json",
                    "-o",
                    str(invalid_suffix),
                    "--emit-only",
                ]
            )
            self.assertEqual(code, 1)
            self.assertIn("output must end in .mlir", stderr)
            self.assertFalse(invalid_suffix.exists())

    def test_output_and_report_reject_relative_absolute_alias_before_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "same.mlir"
            relative_alias = os.path.relpath(destination, Path.cwd())
            with (
                patch("gac.cli.load_graph") as load_mock,
                patch("gac.cli.compile_mlir") as compile_mock,
            ):
                code, stdout, stderr = _invoke(
                    [
                        "mlir-compile",
                        "model.json",
                        "-o",
                        relative_alias,
                        "--report",
                        str(destination.resolve()),
                        "--emit-only",
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("--output and --report must name different files", stderr)
            load_mock.assert_not_called()
            compile_mock.assert_not_called()
            self.assertFalse(destination.exists())

    def test_report_replacement_failure_preserves_old_report_and_artifact(self):
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model.mlir"
            report_path = root / "report.json"
            report_path.write_text("old report\n", encoding="utf-8")

            def fail_report_replace(source, destination):
                if Path(destination).resolve() == report_path.resolve():
                    raise OSError("simulated report publication failure")
                return real_replace(source, destination)

            with (
                patch("gac.cli.load_graph", return_value=object()),
                patch("gac.cli.compile_mlir", return_value=_FakeProgram()),
                patch("gac.cli.os.replace", side_effect=fail_report_replace),
            ):
                code, stdout, stderr = _invoke(
                    [
                        "mlir-compile",
                        "model.json",
                        "-o",
                        str(output),
                        "--report",
                        str(report_path),
                        "--emit-only",
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("simulated report publication failure", stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), _FakeProgram.source)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "old report\n")
            self.assertFalse(any(path.suffix == ".tmp" for path in root.iterdir()))


if __name__ == "__main__":
    unittest.main()
