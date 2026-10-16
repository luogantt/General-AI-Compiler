"""The compatibility CLI must never silently substitute Python Graph compilation."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gac.tools import build_vortex_graph


class NativeOnnxCliTests(unittest.TestCase):
    def test_onnx_launches_native_compiler_once_with_layer_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            with mock.patch.object(build_vortex_graph.subprocess, "run") as run, \
                    mock.patch.object(build_vortex_graph, "load_model", side_effect=AssertionError("Python Graph used")), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = build_vortex_graph.main([
                    "--model", "model.onnx", "--output", str(output),
                    "--native-compiler", "native-gac-compile", "--no-optimize"])
            self.assertEqual(result, 0)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[0], "native-gac-compile")
            self.assertIn("--save-stages", command)
            self.assertIn("--emit=llvm", command)
            self.assertIn("--no-optimize", command)
            self.assertTrue(run.call_args.kwargs["check"])

    def test_missing_native_compiler_is_an_error_without_graph_fallback(self):
        with mock.patch.dict(build_vortex_graph.os.environ, {"GAC_NATIVE_COMPILER": ""}), \
                mock.patch.object(build_vortex_graph.shutil, "which", return_value=None), \
                mock.patch.object(build_vortex_graph, "load_model", side_effect=AssertionError("Python Graph used")), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            result = build_vortex_graph.main(["--model", "model.onnx", "--output", "unused"])
        self.assertEqual(result, 1)
        self.assertIn("C++ MLIR compiler", errors.getvalue())

    def test_native_compilation_failure_is_not_retried_with_python(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(build_vortex_graph.subprocess, "run",
                                   side_effect=subprocess.CalledProcessError(1, "gac-compile")) as run, \
                    mock.patch.object(build_vortex_graph, "load_model", side_effect=AssertionError("Python Graph used")), \
                    contextlib.redirect_stderr(io.StringIO()):
                result = build_vortex_graph.main([
                    "--model", "model.onnx", "--output", directory,
                    "--native-compiler", "native-gac-compile"])
            self.assertEqual(result, 1)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
