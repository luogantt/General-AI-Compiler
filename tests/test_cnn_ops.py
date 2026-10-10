from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.errors import ShapeInferenceError
from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass
from gac.runtime import reference_run


class CnnOpTests(unittest.TestCase):
    def test_conv_batchnorm_constants_are_folded(self):
        rng = np.random.default_rng(12)
        constants = {
            "weight": rng.standard_normal((3, 2, 3, 3), dtype=np.float32),
            "conv_bias": rng.standard_normal((3,), dtype=np.float32),
            "scale": rng.uniform(0.5, 1.5, size=(3,)).astype(np.float32),
            "bn_bias": rng.standard_normal((3,), dtype=np.float32),
            "mean": rng.standard_normal((3,), dtype=np.float32),
            "variance": rng.uniform(0.5, 2.0, size=(3,)).astype(np.float32),
        }
        graph = Graph(
            "fold",
            {"x": TensorSpec((1, 2, 5, 5), "float32")},
            ("y",),
            [
                Node("conv", "Conv2D", ("x", "weight", "conv_bias"), ("conv_out",), {"pads": [1, 1, 1, 1]}),
                Node("bn", "BatchNorm", ("conv_out", "scale", "bn_bias", "mean", "variance"), ("y",), {"epsilon": 1e-4}),
            ],
            constants=constants,
        )
        before = graph.to_json()
        program = compile_cuda(graph)
        self.assertEqual(graph.to_json(), before)
        self.assertEqual(len(program.graph.nodes), 1)
        self.assertEqual(program.graph.nodes[0].op, "Conv2D")
        self.assertEqual(program.graph.nodes[0].source_nodes, ("conv", "bn"))
        self.assertEqual(len(program.launches), 1)
        self.assertNotIn("weight", program.graph.constants)
        self.assertNotIn("variance", program.graph.constants)
        x = rng.standard_normal((1, 2, 5, 5), dtype=np.float32)
        expected = reference_run(program.reference_graph, {"x": x})["y"]
        actual = reference_run(program.graph, {"x": x})["y"]
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_conv2d_known_result_and_codegen(self):
        graph = Graph(
            "conv",
            {
                "x": TensorSpec((1, 1, 3, 3), "float32"),
                "weight": TensorSpec((1, 1, 2, 2), "float32"),
                "bias": TensorSpec((1,), "float32"),
            },
            ("y",),
            [Node("conv", "Conv2D", ("x", "weight", "bias"), ("y",))],
        )
        x = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
        weight = np.asarray([[[[1, 0], [0, -1]]]], dtype=np.float32)
        bias = np.asarray([0.5], dtype=np.float32)
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["y"].shape, (1, 1, 2, 2))
        self.assertIn("weight_index", program.source)
        np.testing.assert_array_equal(
            reference_run(program.reference_graph, {"x": x, "weight": weight, "bias": bias})["y"],
            np.full((1, 1, 2, 2), -3.5, dtype=np.float32),
        )

    def test_grouped_dilated_conv(self):
        graph = Graph(
            "grouped",
            {
                "x": TensorSpec((1, 2, 3, 3), "float32"),
                "weight": TensorSpec((2, 1, 2, 2), "float32"),
            },
            ("y",),
            [Node(
                "conv", "Conv2D", ("x", "weight"), ("y",),
                {"groups": 2, "dilations": [2, 2], "pads": [1, 1, 1, 1]},
            )],
        )
        x = np.arange(18, dtype=np.float32).reshape(1, 2, 3, 3)
        weight = np.asarray(
            [
                [[[1, 2], [3, 4]]],
                [[[-1, 0.5], [2, -0.25]]],
            ],
            dtype=np.float32,
        )
        actual = reference_run(
            compile_cuda(graph).reference_graph,
            {"x": x, "weight": weight},
        )["y"]

        expected = np.zeros((1, 2, 3, 3), dtype=np.float32)
        for channel in range(2):
            for out_y in range(3):
                for out_x in range(3):
                    for kernel_y in range(2):
                        for kernel_x in range(2):
                            source_y = out_y - 1 + kernel_y * 2
                            source_x = out_x - 1 + kernel_x * 2
                            if 0 <= source_y < 3 and 0 <= source_x < 3:
                                expected[0, channel, out_y, out_x] += (
                                    x[0, channel, source_y, source_x]
                                    * weight[channel, 0, kernel_y, kernel_x]
                                )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_batchnorm_and_maxpool_chain(self):
        graph = Graph(
            "norm_pool",
            {
                "x": TensorSpec((1, 2, 3, 3), "float32"),
                "scale": TensorSpec((2,), "float32"),
                "bias": TensorSpec((2,), "float32"),
                "mean": TensorSpec((2,), "float32"),
                "variance": TensorSpec((2,), "float32"),
            },
            ("y",),
            [
                Node("bn", "BatchNorm", ("x", "scale", "bias", "mean", "variance"), ("normalized",), {"epsilon": 1e-4}),
                Node("relu", "ReLU", ("normalized",), ("activated",)),
                Node("pool", "MaxPool2D", ("activated",), ("y",), {"kernel_shape": [2, 2], "strides": [1, 1]}),
            ],
        )
        x = np.linspace(-3, 5, 18, dtype=np.float32).reshape(1, 2, 3, 3)
        scale = np.asarray([2.0, 0.5], dtype=np.float32)
        bias = np.asarray([-1.0, 2.0], dtype=np.float32)
        mean = np.asarray([0.25, 1.5], dtype=np.float32)
        variance = np.asarray([1.25, 0.75], dtype=np.float32)
        program = compile_cuda(graph)
        actual = reference_run(
            program.reference_graph,
            {"x": x, "scale": scale, "bias": bias, "mean": mean, "variance": variance},
        )["y"]
        normalized = (x - mean.reshape(1, 2, 1, 1)) / np.sqrt(
            variance.reshape(1, 2, 1, 1) + 1e-4
        ) * scale.reshape(1, 2, 1, 1) + bias.reshape(1, 2, 1, 1)
        activated = np.maximum(normalized, 0)
        expected = np.empty((1, 2, 2, 2), dtype=np.float32)
        for y in range(2):
            for x_coordinate in range(2):
                expected[:, :, y, x_coordinate] = np.max(
                    activated[:, :, y:y + 2, x_coordinate:x_coordinate + 2],
                    axis=(2, 3),
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertIn("rsqrtf", program.source)
        self.assertIn("maximum = fmaxf", program.source)

    def test_invalid_cnn_contracts(self):
        cases = [
            Graph("channels", {"x": TensorSpec((1, 3, 4, 4), "float32"), "w": TensorSpec((4, 2, 3, 3), "float32")},
                  ("y",), [Node("n", "Conv2D", ("x", "w"), ("y",), {"groups": 1})]),
            Graph("groups", {"x": TensorSpec((1, 3, 4, 4), "float32"), "w": TensorSpec((4, 1, 3, 3), "float32")},
                  ("y",), [Node("n", "Conv2D", ("x", "w"), ("y",), {"groups": 2})]),
            Graph("bias", {"x": TensorSpec((1, 1, 4, 4), "float32"), "w": TensorSpec((2, 1, 3, 3), "float32"), "b": TensorSpec((1,), "float32")},
                  ("y",), [Node("n", "Conv2D", ("x", "w", "b"), ("y",))]),
            Graph("bn", {name: TensorSpec(shape, "float32") for name, shape in {
                "x": (1, 2, 3, 3), "s": (2,), "b": (2,), "m": (2,), "v": (3,)
            }.items()}, ("y",), [Node("n", "BatchNorm", ("x", "s", "b", "m", "v"), ("y",))]),
            Graph("pool", {"x": TensorSpec((1, 1, 2, 2), "float32")}, ("y",),
                  [Node("n", "MaxPool2D", ("x",), ("y",), {"kernel_shape": [3, 3]})]),
        ]
        for graph in cases:
            with self.subTest(graph=graph.name), self.assertRaises(ShapeInferenceError):
                ShapeInferencePass().run(graph)


if __name__ == "__main__":
    unittest.main()
