"""The ONNX inspection artifact must remain a faithful, reloadable frontend view."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gac.core.ir import Graph
from gac.validation.reference import reference_run


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "inspect_onnx_ir.py"
_SPEC = importlib.util.spec_from_file_location("gac_test_onnx_ir_inspector", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
inspector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inspector)


@unittest.skipUnless(importlib.util.find_spec("onnx"), "optional ONNX dependency is not installed")
class OnnxIrInspectorTests(unittest.TestCase):
    @staticmethod
    def _gemm_model():
        from onnx import TensorProto, helper, numpy_helper

        weight = np.arange(6, dtype=np.float32).reshape(3, 2) / 10
        bias = np.asarray([1, 2, 3], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "file", "bias"], ["y"], transB=1, name="projection")],
            "inspection_gemm",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
            [numpy_helper.from_array(weight, name="file"), numpy_helper.from_array(bias, name="bias")],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    def test_external_weights_roundtrip_preserves_gemm_and_source_distinction(self):
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx.save(self._gemm_model(), root / "model.onnx")
            report = inspector.inspect_onnx_ir(root / "model.onnx", root / "view")
            self.assertTrue(report["success"], report)
            self.assertTrue(report["strict_import"]["success"], report)
            self.assertEqual(report["source_operator_counts"], {"ai.onnx::Gemm": 1})
            self.assertEqual(report["canonical_operator_counts"], {"Add": 1, "MatMul": 1})
            self.assertEqual(report["source_node_count"], 1)
            self.assertEqual(report["canonical_node_count"], 2)
            self.assertFalse(report["mlir"]["generated"])
            payload = json.loads((root / "view" / "canonical.graph.json").read_text(encoding="utf-8"))
            self.assertNotIn("data", payload["constants"]["file"])
            with np.load(root / "view" / "constants.npz", allow_pickle=False) as arrays:
                graph = Graph.from_dict(payload, external_constants=arrays)
            x = np.asarray([[2.0, 3.0]], dtype=np.float32)
            expected = x @ graph.constants["file"].T + graph.constants["bias"]
            np.testing.assert_allclose(reference_run(graph, {"x": x})["y"], expected)
            self.assertIn("NOT MLIR", (root / "view" / "graph.ir.txt").read_text(encoding="utf-8"))
            self.assertIn("ai.onnx::Gemm", (root / "view" / "source.onnx.txt").read_text(encoding="utf-8"))

    def test_unknown_operator_is_diagnostic_not_a_successful_strict_lowering(self):
        import onnx
        from onnx import helper

        model = self._gemm_model()
        del model.graph.node[:]
        model.graph.node.append(helper.make_node("CustomAttention", ["x"], ["y"], domain="example", name="attention"))
        model.opset_import.append(helper.make_opsetid("example", 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx.save(model, root / "custom.onnx")
            report = inspector.inspect_onnx_ir(root / "custom.onnx", root / "view")
            self.assertTrue(report["success"], report)
            self.assertFalse(report["strict_import"]["success"])
            self.assertEqual(report["canonical_operator_counts"], {"example::CustomAttention": 1})
            self.assertEqual(report["unsupported_nodes"], [{"node": "attention", "op": "example::CustomAttention"}])
            self.assertIn("faithful GAC lowering", report["strict_import"]["error"])

    def test_fatal_import_retains_source_counts_and_clears_stale_graph(self):
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._gemm_model()
            onnx.save(model, root / "model.onnx")
            self.assertTrue(inspector.inspect_onnx_ir(root / "model.onnx", root / "view")["success"])
            model.graph.node[0].input[0] = "missing_input"
            onnx.save(model, root / "broken.onnx")
            with redirect_stdout(io.StringIO()):
                status = inspector.main(["--model", str(root / "broken.onnx"), "--output", str(root / "view")])
            self.assertEqual(status, 1)
            report = json.loads((root / "view" / "import-report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["success"])
            self.assertFalse(report["diagnostic_import"]["success"])
            self.assertEqual(report["source_node_count"], 1)
            self.assertNotIn("canonical_node_count", report)
            self.assertEqual(report["error"]["phase"], "diagnostic-import")
            self.assertFalse((root / "view" / "canonical.graph.json").exists())
            self.assertFalse((root / "view" / "constants.npz").exists())

    def test_empty_tensor_and_script_like_names_are_safe_to_inspect(self):
        import onnx
        from onnx import helper, numpy_helper

        model = self._gemm_model()
        malicious_name = "projection</script><script>alert(1)</script>"
        model.graph.node[0].name = malicious_name
        model.graph.node.append(helper.make_node(
            "Constant", [], ["empty"], name="empty_constant",
            value=numpy_helper.from_array(np.empty((0, 3), dtype=np.float32)),
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx.save(model, root / "model.onnx")
            report = inspector.inspect_onnx_ir(root / "model.onnx", root / "view")
            self.assertTrue(report["success"], report)
            constants = json.loads((root / "view" / "constants.index.json").read_text(encoding="utf-8"))
            self.assertEqual(constants["empty"]["nbytes"], 0)
            self.assertEqual(constants["empty"]["preview"], [])
            html = (root / "view" / "index.html").read_text(encoding="utf-8")
            self.assertNotIn(malicious_name, html)
            self.assertIn("\\u003c/script", html)

    def test_export_scope_is_accepted_only_for_the_matching_onnx_hash(self):
        import onnx

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            onnx.save(self._gemm_model(), model)
            details = {
                "scope": "cold raw-head path; no box decoding",
                "metadata": "fixed synthetic calibration; not dataset accuracy",
                "input": {"x": [1, 2]},
                "onnx": {"sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
            }
            export_report = root / "export-report.json"
            export_report.write_text(json.dumps(details), encoding="utf-8")
            report = inspector.inspect_onnx_ir(model, root / "view", export_report=export_report)
            self.assertTrue(report["success"], report)
            self.assertTrue(report["export_report"]["model_sha256_verified"])
            self.assertEqual(report["export_report"]["details"]["scope"], details["scope"])
            self.assertIn(details["scope"], (root / "view" / "index.html").read_text(encoding="utf-8"))
            details["onnx"]["sha256"] = "0" * 64
            export_report.write_text(json.dumps(details), encoding="utf-8")
            report = inspector.inspect_onnx_ir(model, root / "view", export_report=export_report)
            self.assertFalse(report["success"])
            self.assertEqual(report["error"]["phase"], "validate-export-report")
            self.assertNotIn("export_report", report)
            self.assertFalse((root / "view" / "canonical.graph.json").exists())


if __name__ == "__main__":
    unittest.main()
