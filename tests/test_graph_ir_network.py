"""Network viewer data must retain graph edges and real tensor coordinates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "render_graph_ir.py"
_SPEC = importlib.util.spec_from_file_location("graph_ir_network_tool", _TOOL)
renderer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer)


class GraphIrNetworkTests(unittest.TestCase):
    def _write_graph(self, root: Path, *, hostile: bool = False) -> Graph:
        weight_name = "weight</script><script>alert(1)</script>" if hostile else "weight"
        graph = Graph(
            name="matrix graph", inputs={"x": TensorSpec((2, 3), "float32")},
            outputs=("y",),
            nodes=[Node("mm", "MatMul", ("x", weight_name), ("mid",)),
                   Node("bias", "Add", ("mid", "b"), ("y",))],
            constants={weight_name: np.arange(12, dtype=np.float32).reshape(3, 4),
                       "b": np.asarray([1, 2, 3, 4], dtype=np.float32)},
            value_specs={"mid": TensorSpec((2, 4), "float32"),
                         "y": TensorSpec((2, 4), "float32")},
        )
        (root / "canonical.graph.json").write_text(graph.to_json(include_constant_data=False), encoding="utf-8")
        np.savez(root / "constants.npz", **graph.constants)
        (root / "graph.ir.txt").write_text("named graph IR", encoding="utf-8")
        return graph

    def test_reload_preserves_actual_edges_specs_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = self._write_graph(root)
            payload = renderer.build_payload(root, root / "graph-network.html")
            self.assertEqual(payload["graph"], graph.to_dict(include_constant_data=False))
            self.assertEqual(payload["graph"]["nodes"][0]["outputs"], ["mid"])
            self.assertEqual(payload["graph"]["nodes"][1]["inputs"], ["mid", "b"])
            self.assertEqual(payload["constants"]["weight"]["matrix"]["values"], graph.constants["weight"].tolist())
            self.assertEqual(payload["constants"]["weight"]["finite_count"], 12)
            self.assertEqual(payload["constants"]["weight"]["min"], 0)
            self.assertEqual(payload["constants"]["weight"]["max"], 11)
            self.assertFalse(payload["provenance"]["activation_values_available"])
            self.assertEqual(payload["provenance"]["artifact_links"]["graph.ir.txt"], "graph.ir.txt")

    def test_high_rank_planes_and_samples_use_original_coordinates(self):
        data = np.arange(4 * 3 * 29 * 31, dtype=np.int64).reshape(4, 3, 29, 31)
        summary = renderer._constant_summary(data)
        self.assertEqual(summary["plane_count"], 12)
        self.assertEqual(len(summary["planes"]), 8)
        self.assertEqual(summary["planes"][0]["leading_indices"], [0, 0])
        self.assertEqual(summary["planes"][-1]["leading_indices"], [3, 2])
        for plane in summary["planes"]:
            matrix = plane["matrix"]
            prefix = tuple(plane["leading_indices"])
            rows, cols = matrix["row_indices"], matrix["col_indices"]
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(cols), 16)
            self.assertEqual((rows[0], rows[-1], cols[0], cols[-1]), (0, 28, 0, 30))
            np.testing.assert_array_equal(matrix["values"], data[prefix][np.ix_(rows, cols)])
            self.assertEqual(matrix["axes"], [2, 3])
            self.assertTrue(matrix["sampled"])
        flat = summary["flattened_matrix"]
        self.assertEqual(flat["axes"], [0, "flatten(1..3)"])
        self.assertNotIn("leading_indices", flat)
        self.assertEqual((flat["rows"], flat["cols"]), (4, 3 * 29 * 31))
        np.testing.assert_array_equal(
            flat["values"],
            data.reshape(4, 3 * 29 * 31)[np.ix_(flat["row_indices"], flat["col_indices"])],
        )

    def test_scalar_vector_and_empty_tensors_have_no_fabricated_values(self):
        scalar = renderer._constant_summary(np.array(7, dtype=np.int64))
        self.assertEqual(scalar["matrix"]["values"], [[7]])
        self.assertEqual(scalar["matrix"]["axes"], [None, None])
        vector = renderer._constant_summary(np.array([3.0, 1.0]))
        self.assertEqual(vector["matrix"]["values"], [[3.0, 1.0]])
        self.assertEqual(vector["matrix"]["axes"], [None, 0])
        for shape in ((0,), (0, 3), (2, 0), (0, 2, 3)):
            summary = renderer._constant_summary(np.empty(shape))
            self.assertEqual(summary["numel"], 0)
            self.assertEqual(summary["finite_count"], 0)
            self.assertIsNone(summary["min"])
            self.assertIsNone(summary["max"])
            self.assertEqual(sum(len(row) for row in summary["matrix"]["values"]), 0)
        no_plane = renderer._constant_summary(np.empty((0, 2, 3)))
        self.assertEqual(no_plane["planes"], [])
        self.assertEqual(no_plane["flattened_matrix"]["rows"], 0)
        self.assertEqual(no_plane["flattened_matrix"]["cols"], 6)
        self.assertEqual(no_plane["flattened_matrix"]["values"], [])

    def test_nonfinite_numbers_remain_visible_with_finite_whole_array_stats(self):
        summary = renderer._constant_summary(np.array([np.nan, np.inf, -np.inf, -2.0, 8.0]))
        self.assertEqual(summary["finite_count"], 2)
        self.assertEqual((summary["min"], summary["max"]), (-2.0, 8.0))
        self.assertEqual(summary["matrix"]["values"], [["nan", "inf", "-inf", -2.0, 8.0]])
        json.dumps(summary, allow_nan=False)

    def test_int64_values_above_browser_number_precision_remain_exact(self):
        maximum = int(np.iinfo(np.int64).max)
        summary = renderer._constant_summary(np.array([maximum, -maximum], dtype=np.int64))
        self.assertEqual(summary["matrix"]["values"], [[str(maximum), str(-maximum)]])
        self.assertEqual(summary["max"], str(maximum))
        self.assertEqual(summary["finite_count"], 2)

    def test_external_arrays_must_match_graph_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_graph(root)
            np.savez(root / "constants.npz", weight=np.zeros((99,), dtype=np.float32), b=np.zeros(4, dtype=np.float32))
            with self.assertRaisesRegex(ValidationError, "does not match"):
                renderer.build_payload(root, root / "graph-network.html")

    def test_constant_aliases_follow_only_identity_chains_and_preserve_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = Graph(
                name="identity aliases", inputs={"x": TensorSpec((2,), "float32")},
                outputs=("weight_alias_2", "y", "not_an_alias"),
                nodes=[Node("copy_weight", "Identity", ("weight",), ("weight_alias",)),
                       Node("copy_again", "Identity", ("weight_alias",), ("weight_alias_2",)),
                       Node("activation_copy", "Identity", ("x",), ("y",)),
                       Node("constant_math", "ReLU", ("weight",), ("computed",)),
                       Node("computed_copy", "Identity", ("computed",), ("not_an_alias",))],
                constants={"weight": np.array([1.0, -2.0], dtype=np.float32)},
            )
            original = graph.to_dict(include_constant_data=False)
            (root / "canonical.graph.json").write_text(json.dumps(original), encoding="utf-8")
            np.savez(root / "constants.npz", **graph.constants)
            payload = renderer.build_payload(root, root / "view.html")
            self.assertEqual(payload["constant_aliases"], {
                "weight_alias": {"source": "weight", "via": ["copy_weight"]},
                "weight_alias_2": {"source": "weight", "via": ["copy_weight", "copy_again"]},
            })
            self.assertEqual(payload["graph"], original)
            self.assertEqual(set(payload["constants"]), {"weight"})

    def test_html_embedding_escapes_source_names_and_uses_relative_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_graph(root, hostile=True)
            template = root / "template.html"
            template.write_text('<script id="data" type="application/json">__GRAPH_IR_DATA__</script><script>__GRAPH_IR_LAYOUT__</script>', encoding="utf-8")
            layout = root / "layout.js"
            layout.write_text('window.layoutReady = true;', encoding="utf-8")
            output = root / "nested folder" / "network.html"
            result = renderer.render_graph_ir(root, output, template_path=template, layout_path=layout)
            html = output.read_text(encoding="utf-8")
            self.assertNotIn("weight</script>", html)
            self.assertIn(r"weight\u003c/script\u003e", html)
            self.assertIn('window.layoutReady = true;', html)
            embedded = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
            payload = json.loads(embedded)
            self.assertIn("weight</script><script>alert(1)</script>", payload["constants"])
            self.assertEqual(payload["provenance"]["artifact_links"]["graph.ir.txt"], "../graph.ir.txt")
            self.assertEqual(result["node_count"], 2)


if __name__ == "__main__":
    unittest.main()
