from __future__ import annotations

import unittest

from gac.capability import analyze_support
from gac.errors import BackendError
from gac.ir import Graph, Node, TensorSpec


class CapabilityTests(unittest.TestCase):
    def graph(self):
        return Graph(
            "mixed",
            {"x": TensorSpec((2, 2), "float32")},
            ("z",),
            [
                Node("a", "Identity", ("x",), ("a",)),
                Node("b", "CameraProject", ("a",), ("b",)),
                Node("c", "ReLU", ("b",), ("c",)),
                Node("d", "Unsupported", ("c",), ("d",)),
                Node("e", "Identity", ("d",), ("z",)),
            ],
        )

    def test_report_and_segments_are_deterministic(self):
        report = analyze_support(
            self.graph(),
            target="test",
            supported_ops={"Identity", "CameraProject", "ReLU"},
            check=lambda node, _: "camera calibration is dynamic" if node.op == "CameraProject" else None,
        )
        self.assertFalse(report.fully_supported)
        self.assertEqual([node.node_id for node in report.unsupported], ["b", "d"])
        self.assertEqual(
            [(segment.supported, segment.node_ids) for segment in report.segments],
            [(True, ("a",)), (False, ("b",)), (True, ("c",)), (False, ("d",)), (True, ("e",))],
        )
        serialized = report.to_dict()
        self.assertEqual(serialized["op_counts"]["Identity"], 2)
        self.assertEqual(serialized["unsupported_op_counts"], {"CameraProject": 1, "Unsupported": 1})
        with self.assertRaisesRegex(BackendError, "camera calibration is dynamic"):
            report.require_supported()

    def test_empty_graph_is_supported(self):
        graph = Graph("empty", {"x": TensorSpec((1,), "float32")}, ("x",), [])
        report = analyze_support(graph, target="test", supported_ops=())
        self.assertTrue(report.fully_supported)
        self.assertEqual(report.segments, ())
        report.require_supported()


if __name__ == "__main__":
    unittest.main()
