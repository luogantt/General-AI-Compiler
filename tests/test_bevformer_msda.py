"""Contract tests for BEVFormer's multi-scale deformable attention op.

The graph helpers live in :mod:`gac.bevformer` and the public operation API in
:mod:`gac.ops`.  These tests intentionally keep a small, independent NumPy
implementation here so a backend cannot validate itself with its own indexing
code.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer import build_msda_graph, make_msda_inputs
from gac.errors import BackendError, ShapeInferenceError
from gac.ir import Graph, Node, TensorSpec
from gac.ops import evaluate_op, infer_node


def _node(spatial_shapes) -> Node:
    return Node(
        "msda",
        "MSDeformableAttention",
        ("value", "sampling_locations", "attention_weights"),
        ("output",),
        {"spatial_shapes": spatial_shapes},
    )


def _specs(
    *,
    value=(2, 20, 8, 32),
    locations=(2, 11, 8, 2, 4, 2),
    weights=(2, 11, 8, 2, 4),
    dtype="float32",
) -> dict[str, TensorSpec]:
    return {
        "value": TensorSpec(value, dtype),
        "sampling_locations": TensorSpec(locations, dtype),
        "attention_weights": TensorSpec(weights, dtype),
    }


def _independent_msda(
    value: np.ndarray,
    sampling_locations: np.ndarray,
    attention_weights: np.ndarray,
    spatial_shapes,
) -> np.ndarray:
    """Slow definition using align_corners=False and zero padding.

    Coordinates are normalized to [0, 1].  For a level of size H x W, its
    continuous pixel coordinate is ``x = loc_x * W - 0.5`` and likewise for
    y.  Each head owns D adjacent channels in ``value``.
    """

    batch, _, heads, channels = value.shape
    queries = sampling_locations.shape[1]
    levels = sampling_locations.shape[3]
    points = sampling_locations.shape[4]
    result = np.zeros((batch, queries, heads, channels), dtype=np.float64)

    level_start = 0
    for level in range(levels):
        height, width = spatial_shapes[level]
        level_size = height * width
        level_value = value[:, level_start : level_start + level_size]
        level_value = level_value.reshape(batch, height, width, heads, channels)
        level_start += level_size

        for batch_index in range(batch):
            for query in range(queries):
                for head in range(heads):
                    for point in range(points):
                        location = sampling_locations[
                            batch_index, query, head, level, point
                        ]
                        x = float(location[0]) * width - 0.5
                        y = float(location[1]) * height - 0.5
                        x0, y0 = math.floor(x), math.floor(y)
                        x1, y1 = x0 + 1, y0 + 1
                        wx1, wy1 = x - x0, y - y0
                        wx0, wy0 = 1.0 - wx1, 1.0 - wy1
                        sampled = np.zeros(channels, dtype=np.float64)
                        for pixel_x, pixel_y, coefficient in (
                            (x0, y0, wx0 * wy0),
                            (x1, y0, wx1 * wy0),
                            (x0, y1, wx0 * wy1),
                            (x1, y1, wx1 * wy1),
                        ):
                            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                                sampled += coefficient * level_value[
                                    batch_index, pixel_y, pixel_x, head
                                ]
                        result[batch_index, query, head] += (
                            float(attention_weights[
                                batch_index, query, head, level, point
                            ])
                            * sampled
                        )

    return result.reshape(batch, queries, heads * channels)


class MSDeformableAttentionShapeTests(unittest.TestCase):
    def test_shape_inference(self):
        inferred = infer_node(
            _node([[4, 4], [2, 2]]),
            _specs(),
        )
        self.assertEqual(
            inferred,
            {"output": TensorSpec((2, 11, 256), "float32")},
        )

    def test_shape_inference_preserves_value_dtype(self):
        inferred = infer_node(
            _node([[4, 4], [2, 2]]),
            _specs(dtype="float16"),
        )
        self.assertEqual(inferred["output"].dtype, "float16")

    def test_shape_inference_rejects_malformed_contract(self):
        cases = {
            "value rank": (_specs(value=(2, 20, 256)), [[4, 4], [2, 2]]),
            "locations rank": (
                _specs(locations=(2, 11, 8, 8, 2)),
                [[4, 4], [2, 2]],
            ),
            "weights rank": (
                _specs(weights=(2, 11, 8, 8)),
                [[4, 4], [2, 2]],
            ),
            "coordinate size": (
                _specs(locations=(2, 11, 8, 2, 4, 3)),
                [[4, 4], [2, 2]],
            ),
            "batch mismatch": (
                _specs(locations=(1, 11, 8, 2, 4, 2)),
                [[4, 4], [2, 2]],
            ),
            "query mismatch": (
                _specs(weights=(2, 12, 8, 2, 4)),
                [[4, 4], [2, 2]],
            ),
            "head mismatch": (
                _specs(locations=(2, 11, 4, 2, 4, 2)),
                [[4, 4], [2, 2]],
            ),
            "level mismatch": (
                _specs(locations=(2, 11, 8, 1, 4, 2)),
                [[4, 4], [2, 2]],
            ),
            "point mismatch": (
                _specs(weights=(2, 11, 8, 2, 3)),
                [[4, 4], [2, 2]],
            ),
            "flattened value size": (_specs(value=(2, 19, 8, 32)), [[4, 4], [2, 2]]),
            "malformed spatial shape": (_specs(), [[4], [2, 2]]),
            "non-positive spatial shape": (_specs(), [[4, 4], [0, 2]]),
            "non-floating value": (_specs(dtype="int32"), [[4, 4], [2, 2]]),
        }
        for name, (specs, spatial_shapes) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ShapeInferenceError):
                    infer_node(_node(spatial_shapes), specs)


class MSDeformableAttentionReferenceTests(unittest.TestCase):
    def test_one_by_one_center_is_exact(self):
        value = np.asarray([[[[7.25, -3.5]]]], dtype=np.float32)
        locations = np.asarray([[[[[[0.5, 0.5]]]]]], dtype=np.float32)
        weights = np.ones((1, 1, 1, 1, 1), dtype=np.float32)

        (output,) = evaluate_op(
            "MSDeformableAttention",
            (value, locations, weights),
            {"spatial_shapes": [[1, 1]]},
        )

        np.testing.assert_array_equal(output, np.asarray([[[7.25, -3.5]]], np.float32))

    def test_two_by_two_bilinear_and_zero_padding(self):
        # Flattened row-major image: [[1, 2], [3, 4]].
        value = np.asarray([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]], np.float32)
        locations = np.asarray(
            [
                [0.25, 0.25],  # exact center of the top-left pixel
                [0.50, 0.50],  # center of all four pixels
                [0.00, 0.00],  # one quarter of the top-left pixel
                [1.50, 1.50],  # entirely outside the image
            ],
            np.float32,
        ).reshape(1, 4, 1, 1, 1, 2)
        weights = np.ones((1, 4, 1, 1, 1), np.float32)

        (output,) = evaluate_op(
            "MSDeformableAttention",
            (value, locations, weights),
            {"spatial_shapes": [[2, 2]]},
        )

        np.testing.assert_allclose(
            output[..., 0],
            np.asarray([[1.0, 2.5, 0.25, 0.0]], np.float32),
            rtol=0,
            atol=1e-7,
        )

    def test_multilevel_matches_independent_definition(self):
        # Level 0 is 1x2 and level 1 is 2x1.  H=2 and D=2 exercise the
        # flattened output order [head0 channels, head1 channels].
        value = np.arange(1, 1 + 1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2)
        locations = np.asarray(
            [[
                [
                    [
                        [[0.25, 0.5], [0.75, 0.5]],
                        [[0.5, 0.25], [0.5, 0.75]],
                    ],
                    [
                        [[0.0, 0.5], [1.0, 0.5]],
                        [[0.5, 0.0], [0.5, 1.0]],
                    ],
                ],
                [
                    [
                        [[0.5, 0.5], [1.25, 0.5]],
                        [[0.5, 0.5], [0.5, 1.25]],
                    ],
                    [
                        [[0.25, 0.5], [0.75, 0.5]],
                        [[0.5, 0.25], [0.5, 0.75]],
                    ],
                ],
            ]],
            dtype=np.float32,
        )
        weights = np.asarray(
            [[
                [
                    [[0.10, 0.20], [0.30, 0.40]],
                    [[0.40, 0.30], [0.20, 0.10]],
                ],
                [
                    [[0.25, 0.25], [0.25, 0.25]],
                    [[0.05, 0.15], [0.35, 0.45]],
                ],
            ]],
            dtype=np.float32,
        )
        expected = _independent_msda(value, locations, weights, [[1, 2], [2, 1]])

        (output,) = evaluate_op(
            "MSDeformableAttention",
            (value, locations, weights),
            {"spatial_shapes": [[1, 2], [2, 1]]},
        )

        self.assertEqual(output.shape, (1, 2, 4))
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


class BEVFormerTinyProfileTests(unittest.TestCase):
    def test_tiny_spatial_attention_graph_and_inputs(self):
        # BEVFormer-tiny: six cameras, C5 at 15x25 for a padded 480x800
        # image, 2,500 BEV queries, eight heads, head dimension 32, and
        # eight sampling points at one feature level.
        graph = build_msda_graph()
        graph.validate()
        self.assertEqual(
            graph.inputs,
            {
                "value": TensorSpec((6, 375, 8, 32), "float32"),
                "sampling_locations": TensorSpec((6, 2500, 8, 1, 8, 2), "float32"),
                "attention_weights": TensorSpec((6, 2500, 8, 1, 8), "float32"),
            },
        )
        self.assertEqual(graph.outputs, ("output",))
        self.assertEqual(graph.value_specs["output"], TensorSpec((6, 2500, 256), "float32"))
        self.assertEqual(graph.nodes[0].attrs["spatial_shapes"], [[15, 25]])

        inputs = make_msda_inputs(graph, seed=20261010)
        repeated = make_msda_inputs(graph, seed=20261010)

        self.assertEqual(set(inputs), set(graph.inputs))
        for name, value in inputs.items():
            self.assertEqual(value.shape, graph.inputs[name].shape)
            self.assertEqual(value.dtype, np.dtype(graph.inputs[name].dtype))
            self.assertTrue(value.flags.c_contiguous)
            self.assertTrue(np.all(np.isfinite(value)))
            np.testing.assert_array_equal(value, repeated[name])
        np.testing.assert_allclose(
            inputs["attention_weights"].sum(axis=(3, 4)),
            np.ones((6, 2500, 8), np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_tiny_graph_lowers_to_one_sm86_kernel(self):
        graph = build_msda_graph()
        first = compile_cuda(graph)
        second = compile_cuda(graph)

        self.assertEqual(first.source, second.source)
        self.assertEqual(len(first.launches), 1)
        self.assertEqual(first.launches[0].node_id, "msda")
        self.assertEqual(first.launches[0].grid, (15000, 1, 1))
        self.assertEqual(first.launches[0].block, (256, 1, 1))
        self.assertIn("gac_msda_load", first.source)
        self.assertEqual(first.describe()["tensor_bytes_upper_bound"], 29_184_000)

    def test_sm86_msda_requires_reference_fp32_boundary(self):
        graph = Graph(
            "msda_fp16",
            {
                "value": TensorSpec((1, 1, 1, 1), "float16"),
                "sampling_locations": TensorSpec((1, 1, 1, 1, 1, 2), "float16"),
                "attention_weights": TensorSpec((1, 1, 1, 1, 1), "float16"),
            },
            ("output",),
            [
                Node(
                    "msda",
                    "MSDeformableAttention",
                    ("value", "sampling_locations", "attention_weights"),
                    ("output",),
                    {"spatial_shapes": [[1, 1]]},
                )
            ],
        )
        with self.assertRaises(BackendError):
            compile_cuda(graph)


if __name__ == "__main__":
    unittest.main()
