from __future__ import annotations

import math
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.errors import ShapeInferenceError
from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass
from gac.runtime import reference_run


class TensorTransformTests(unittest.TestCase):
    def test_reshape_infers_dimension_and_preserves_linear_order(self):
        graph = Graph(
            "reshape",
            {"x": TensorSpec((2, 3, 4), "float32")},
            ("y",),
            [Node("reshape", "Reshape", ("x",), ("y",), {"shape": [3, -1, 2]})],
        )
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["y"].shape, (3, 4, 2))
        self.assertIn("out[i] = p0[i]", program.source)
        value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, {"x": value})["y"],
            value.reshape(3, 4, 2),
        )

    def test_transpose_explicit_and_default_permutations(self):
        value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        cases = [
            ({"perm": [1, 0, 2]}, (3, 2, 4), value.transpose(1, 0, 2)),
            ({}, (4, 3, 2), value.transpose()),
        ]
        for attrs, shape, expected in cases:
            with self.subTest(attrs=attrs):
                graph = Graph(
                    "transpose",
                    {"x": TensorSpec(value.shape, "float32")},
                    ("y",),
                    [Node("transpose", "Transpose", ("x",), ("y",), attrs)],
                )
                program = compile_cuda(graph)
                self.assertEqual(program.graph.value_specs["y"].shape, shape)
                self.assertIn("out[i] = p0[", program.source)
                np.testing.assert_array_equal(
                    reference_run(program.reference_graph, {"x": value})["y"],
                    expected,
                )

    def test_concat_materializes_negative_axis(self):
        graph = Graph(
            "concat",
            {
                "a": TensorSpec((2, 3, 2), "float32"),
                "b": TensorSpec((2, 3, 1), "float32"),
                "c": TensorSpec((2, 3, 4), "float32"),
            },
            ("y",),
            [Node("concat", "Concat", ("a", "b", "c"), ("y",), {"axis": -1})],
        )
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["y"].shape, (2, 3, 7))
        self.assertIn("axis_index", program.source)
        inputs = {
            "a": np.full((2, 3, 2), 1, dtype=np.float32),
            "b": np.full((2, 3, 1), 2, dtype=np.float32),
            "c": np.full((2, 3, 4), 3, dtype=np.float32),
        }
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, inputs)["y"],
            np.concatenate(tuple(inputs.values()), axis=-1),
        )

    def test_transform_chain_reference(self):
        graph = Graph(
            "qkv_layout",
            {
                "q": TensorSpec((2, 5, 3, 4), "float32"),
                "k": TensorSpec((2, 7, 3, 4), "float32"),
            },
            ("joined",),
            [
                Node("q_transpose", "Transpose", ("q",), ("qt",), {"perm": [0, 2, 1, 3]}),
                Node("k_transpose", "Transpose", ("k",), ("kt",), {"perm": [0, 2, 1, 3]}),
                Node("concat", "Concat", ("qt", "kt"), ("cat",), {"axis": 2}),
                Node("flatten", "Reshape", ("cat",), ("joined",), {"shape": [6, -1, 4]}),
            ],
        )
        rng = np.random.default_rng(9)
        q = rng.standard_normal(graph.inputs["q"].shape, dtype=np.float32)
        k = rng.standard_normal(graph.inputs["k"].shape, dtype=np.float32)
        actual = reference_run(compile_cuda(graph).reference_graph, {"q": q, "k": k})["joined"]
        expected = np.concatenate((q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3)), axis=2).reshape(6, 12, 4)
        np.testing.assert_array_equal(actual, expected)

    def test_expand_slice_and_reductions(self):
        expand = Graph(
            "expand",
            {"x": TensorSpec((2, 1, 3), "float32")},
            ("y",),
            [Node("expand", "Expand", ("x",), ("y",), {"shape": [2, 4, 3]})],
        )
        value = np.arange(6, dtype=np.float32).reshape(2, 1, 3)
        program = compile_cuda(expand)
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, {"x": value})["y"],
            np.broadcast_to(value, (2, 4, 3)),
        )
        self.assertIn("out[i] = p0[", program.source)

        sliced = Graph(
            "slice",
            {"x": TensorSpec((3, 4, 5), "float32")},
            ("y",),
            [Node(
                "slice", "Slice", ("x",), ("y",),
                {"starts": [2, 4], "ends": [-4, -6], "axes": [0, 2], "steps": [-1, -2]},
            )],
        )
        source = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        program = compile_cuda(sliced)
        self.assertEqual(program.graph.value_specs["y"].shape, (3, 4, 3))
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, {"x": source})["y"],
            source[2:-4:-1, :, 4:-6:-2],
        )

        for op in ("ReduceSum", "ReduceMean"):
            for keepdims in (True, False):
                with self.subTest(op=op, keepdims=keepdims):
                    graph = Graph(
                        "reduce",
                        {"x": TensorSpec((2, 3, 4), "float32")},
                        ("y",),
                        [Node("reduce", op, ("x",), ("y",), {"axes": [1, -1], "keepdims": keepdims})],
                    )
                    program = compile_cuda(graph)
                    reducer = np.sum if op == "ReduceSum" else np.mean
                    expected = reducer(source[:2, :3, :4], axis=(1, 2), keepdims=keepdims)
                    actual = reference_run(program.reference_graph, {"x": source[:2, :3, :4].copy()})["y"]
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
                    self.assertIn("for (int r", program.source)

    def test_grid_sample_identity_and_zero_padding(self):
        graph = Graph(
            "grid",
            {
                "value": TensorSpec((1, 2, 2, 3), "float32"),
                "grid": TensorSpec((1, 2, 3, 2), "float32"),
            },
            ("y",),
            [Node("grid", "GridSample2D", ("value", "grid"), ("y",), {"align_corners": False})],
        )
        value = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
        grid = np.empty((1, 2, 3, 2), dtype=np.float32)
        for y in range(2):
            for x in range(3):
                grid[0, y, x] = ((2 * x + 1) / 3 - 1, (2 * y + 1) / 2 - 1)
        program = compile_cuda(graph)
        np.testing.assert_allclose(
            reference_run(program.reference_graph, {"value": value, "grid": grid})["y"],
            value,
            rtol=0,
            atol=1e-6,
        )
        self.assertIn("gac_grid_load", program.source)

        outside = grid.copy()
        outside[0, 0, 0] = (-3, -3)
        result = reference_run(program.reference_graph, {"value": value, "grid": outside})["y"]
        np.testing.assert_array_equal(result[0, :, 0, 0], 0)

    def test_trigonometry_and_nearest_grid_sample(self):
        angles = np.asarray(
            [-np.pi, -np.pi / 2, 0.0, np.pi / 3, np.pi], dtype=np.float32
        )
        for op, function, source_token in (
            ("Sin", np.sin, "sinf"),
            ("Cos", np.cos, "cosf"),
        ):
            with self.subTest(op=op):
                graph = Graph(
                    op.lower(),
                    {"x": TensorSpec(angles.shape, "float32")},
                    ("y",),
                    [Node("trig", op, ("x",), ("y",))],
                )
                program = compile_cuda(graph)
                self.assertEqual(program.graph.value_specs["y"], graph.inputs["x"])
                self.assertIn(source_token, program.source)
                np.testing.assert_allclose(
                    reference_run(program.reference_graph, {"x": angles})["y"],
                    function(angles),
                    rtol=1e-6,
                    atol=1e-7,
                )

        # GridSample follows PyTorch's nearbyint rule for nearest mode.  The
        # half-pixel cases below therefore exercise round-to-nearest-even.
        value = np.asarray([[[[10.0, 20.0, 30.0, 40.0]]]], dtype=np.float32)
        source_x = np.asarray([0.5, 1.5, 2.5, -2.0], dtype=np.float32)
        grid_x = (source_x + np.float32(0.5)) * np.float32(0.5) - np.float32(1.0)
        grid = np.stack((grid_x, np.zeros_like(grid_x)), axis=-1).reshape(1, 1, 4, 2)
        graph = Graph(
            "nearest_grid",
            {
                "value": TensorSpec(value.shape, "float32"),
                "grid": TensorSpec(grid.shape, "float32"),
            },
            ("y",),
            [Node(
                "grid", "GridSample2D", ("value", "grid"), ("y",),
                {"mode": "nearest", "padding_mode": "zeros", "align_corners": False},
            )],
        )
        program = compile_cuda(graph)
        self.assertIn("nearbyintf", program.source)
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, {"value": value, "grid": grid})["y"],
            np.asarray([[[[10.0, 30.0, 30.0, 0.0]]]], dtype=np.float32),
        )

    def test_where_int32_condition_broadcasts_float32_choices(self):
        graph = Graph(
            "where",
            {
                "condition": TensorSpec((1, 2, 1), "int32"),
                "when_true": TensorSpec((1, 2, 3), "float32"),
                "when_false": TensorSpec((3,), "float32"),
            },
            ("y",),
            [Node(
                "select", "Where",
                ("condition", "when_true", "when_false"), ("y",),
            )],
        )
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["y"].shape, (1, 2, 3))
        self.assertIn("const int* p0", program.source)
        self.assertIn("!= 0 ?", program.source)
        condition = np.asarray([[[0], [-3]]], dtype=np.int32)
        when_true = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        when_false = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
        np.testing.assert_array_equal(
            reference_run(
                program.reference_graph,
                {
                    "condition": condition,
                    "when_true": when_true,
                    "when_false": when_false,
                },
            )["y"],
            np.where(condition != 0, when_true, when_false),
        )

        # The streaming BEVFormer ABI supplies one batch flag as shape [1].
        # It must broadcast over either rank-3 BEV state or rank-4 references.
        for output_shape in ((1, 6, 4), (1, 6, 1, 2)):
            with self.subTest(output_shape=output_shape):
                rank_one = Graph(
                    "where_rank_one_condition",
                    {
                        "condition": TensorSpec((1,), "int32"),
                        "when_true": TensorSpec(output_shape, "float32"),
                        "when_false": TensorSpec(output_shape, "float32"),
                    },
                    ("y",),
                    [Node(
                        "select", "Where",
                        ("condition", "when_true", "when_false"), ("y",),
                    )],
                )
                rank_one_program = compile_cuda(rank_one)
                self.assertIn("p0[0] != 0", rank_one_program.source)
                choice = np.arange(math.prod(output_shape), dtype=np.float32).reshape(output_shape)
                zeros = np.zeros(output_shape, dtype=np.float32)
                actual = reference_run(
                    rank_one_program.reference_graph,
                    {
                        "condition": np.asarray([1], dtype=np.int32),
                        "when_true": choice,
                        "when_false": zeros,
                    },
                )["y"]
                np.testing.assert_array_equal(actual, choice)

        invalid = [
            Graph(
                "where_float_condition",
                {
                    "c": TensorSpec((1,), "float32"),
                    "x": TensorSpec((2,), "float32"),
                    "z": TensorSpec((2,), "float32"),
                },
                ("y",),
                [Node("n", "Where", ("c", "x", "z"), ("y",))],
            ),
            Graph(
                "where_half_choice",
                {
                    "c": TensorSpec((1,), "int32"),
                    "x": TensorSpec((2,), "float16"),
                    "z": TensorSpec((2,), "float32"),
                },
                ("y",),
                [Node("n", "Where", ("c", "x", "z"), ("y",))],
            ),
            Graph(
                "where_bad_broadcast",
                {
                    "c": TensorSpec((2,), "int32"),
                    "x": TensorSpec((3,), "float32"),
                    "z": TensorSpec((3,), "float32"),
                },
                ("y",),
                [Node("n", "Where", ("c", "x", "z"), ("y",))],
            ),
        ]
        for bad_graph in invalid:
            with self.subTest(graph=bad_graph.name), self.assertRaises(
                ShapeInferenceError
            ):
                ShapeInferencePass().run(bad_graph)

    def test_invalid_transform_contracts_are_rejected(self):
        cases = [
            Graph("bad_reshape", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "Reshape", ("x",), ("y",), {"shape": [4, 2]})]),
            Graph("two_inferred", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "Reshape", ("x",), ("y",), {"shape": [-1, -1]})]),
            Graph("bad_perm", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "Transpose", ("x",), ("y",), {"perm": [0, 0]})]),
            Graph("bad_concat", {"a": TensorSpec((2, 3), "float32"), "b": TensorSpec((2, 4), "float32")},
                  ("y",), [Node("n", "Concat", ("a", "b"), ("y",), {"axis": 0})]),
            Graph("bad_expand", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "Expand", ("x",), ("y",), {"shape": [2, 4]})]),
            Graph("bad_slice", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "Slice", ("x",), ("y",), {"starts": [0], "ends": [2], "steps": [0]})]),
            Graph("bad_reduce", {"x": TensorSpec((2, 3), "float32")}, ("y",),
                  [Node("n", "ReduceSum", ("x",), ("y",), {"axes": [1, -1]})]),
            Graph("bad_grid", {"x": TensorSpec((1, 1, 2, 2), "float32"), "g": TensorSpec((1, 2, 2, 2), "float32")},
                  ("y",), [Node("n", "GridSample2D", ("x", "g"), ("y",), {"mode": "bicubic"})]),
        ]
        for graph in cases:
            with self.subTest(graph=graph.name), self.assertRaises(ShapeInferenceError):
                ShapeInferencePass().run(graph)


if __name__ == "__main__":
    unittest.main()
