from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.compose import GraphComposer
from gac.errors import ValidationError
from gac.ir import Graph, Node, TensorSpec
from gac.runtime import reference_run


def component() -> Graph:
    return Graph(
        "bias_relu",
        {
            "x": TensorSpec((2, 3), "float32"),
            "external_bias": TensorSpec((3,), "float32"),
        },
        ("output",),
        [
            Node("add_external", "Add", ("x", "external_bias"), ("sum0",)),
            Node("add_constant", "Add", ("sum0", "inner_bias"), ("sum1",)),
            Node("relu", "ReLU", ("sum1",), ("output",)),
        ],
        constants={"inner_bias": np.asarray([1.0, -1.0, 0.5], np.float32)},
    )


class GraphComposerTests(unittest.TestCase):
    def test_two_component_instances_are_namespaced_compilable_and_equivalent(self):
        composer = GraphComposer(
            "two_blocks",
            {
                "x": TensorSpec((2, 3), "float32"),
                "bias": TensorSpec((3,), "float32"),
            },
            metadata={"kind": "composition-test"},
        )
        first = composer.inline(
            component(),
            prefix="block0",
            bindings={"x": "x", "external_bias": "bias"},
            output_bindings={"output": "hidden"},
        )
        self.assertEqual(first, {"output": "hidden"})
        second = composer.inline(
            component(),
            prefix="block1",
            bindings={"x": "hidden", "external_bias": "bias"},
            output_bindings={"output": "y"},
        )
        graph = composer.build((second["output"],))
        self.assertEqual(graph.outputs, ("y",))
        self.assertEqual(
            set(graph.constants), {"block0.inner_bias", "block1.inner_bias"}
        )
        self.assertEqual(graph.nodes[0].id, "block0.add_external")
        self.assertEqual(graph.nodes[-1].id, "block1.relu")
        x = np.asarray([[-3, 1, 2], [4, -5, 6]], np.float32)
        bias = np.asarray([0.25, 0.5, -0.75], np.float32)
        once = np.maximum(x + bias + np.asarray([1, -1, 0.5], np.float32), 0)
        expected = np.maximum(
            once + bias + np.asarray([1, -1, 0.5], np.float32), 0
        )
        np.testing.assert_array_equal(
            reference_run(graph, {"x": x, "bias": bias})["y"], expected
        )
        program = compile_cuda(graph)
        self.assertEqual(len(program.launches), 4)
        self.assertEqual(sum(node.op == "AddReLU" for node in program.graph.nodes), 2)

    def test_rejects_missing_unknown_duplicate_and_colliding_bindings(self):
        composer = GraphComposer(
            "bad",
            {
                "x": TensorSpec((2, 3), "float32"),
                "bias": TensorSpec((3,), "float32"),
            },
        )
        with self.assertRaises(ValidationError):
            composer.inline(
                component(), prefix="x", bindings={"x": "x"}
            )
        with self.assertRaises(ValidationError):
            composer.inline(
                component(),
                prefix="x",
                bindings={"x": "unknown", "external_bias": "bias"},
            )
        composer.inline(
            component(),
            prefix="good",
            bindings={"x": "x", "external_bias": "bias"},
        )
        before_values = composer.known_values
        before_node_count = len(composer.nodes)
        before_constants = set(composer.constants)
        with self.assertRaises(ValidationError):
            composer.inline(
                component(),
                prefix="late_failure",
                bindings={"x": "x", "external_bias": "bias"},
                output_bindings={"output": "good.output"},
            )
        self.assertEqual(composer.known_values, before_values)
        self.assertEqual(len(composer.nodes), before_node_count)
        self.assertEqual(set(composer.constants), before_constants)
        with self.assertRaises(ValidationError):
            composer.inline(
                component(),
                prefix="good",
                bindings={"x": "x", "external_bias": "bias"},
            )
        with self.assertRaises(ValidationError):
            GraphComposer("empty", {}).build(("missing",))


if __name__ == "__main__":
    unittest.main()
