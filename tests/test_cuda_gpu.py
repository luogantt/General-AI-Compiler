"""Physical-device tests. GAC_REQUIRE_CUDA=1 makes missing CUDA a failure."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from gac.bevformer import build_msda_graph, make_msda_inputs
from gac.bevformer_motion import BevFormerMotionConfig, build_bevformer_motion_graph
from gac.backends.cuda import compile_cuda
from gac.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.frontends.json_frontend import load_graph
from gac.ir import Graph, Node, TensorSpec
from gac.runtime import CudaSession, cuda_device_info, reference_run
from gac.transformer import MultiHeadAttentionConfig, build_multihead_attention_graph
from test_cuda_compiler import single_op


def gelu(x):
    return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))).astype(np.float32)


class PhysicalCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            info = cuda_device_info()
            if info["supported_target"] != "cuda-sm86":
                raise DeviceUnavailableError(f"SM 8.6 required, got {info}")
        except DeviceUnavailableError as exc:
            if os.environ.get("GAC_REQUIRE_CUDA") == "1":
                raise
            raise unittest.SkipTest(str(exc)) from exc
        cls.rng = np.random.default_rng(1729)

    def compare(self, graph, inputs, expected, *, fuse=True):
        output_name = graph.outputs[0]
        with CudaSession(compile_cuda(graph, fuse=fuse)) as session:
            for kernel in session.kernel_info:
                self.assertEqual(kernel["attributes"]["binary_version"], 86)
            result = session.run(inputs, trace=True)
            np.testing.assert_allclose(result.outputs[output_name], expected, rtol=1e-4, atol=1e-5)
            for event in result.profiler.events:
                if event.kind == "dispatch":
                    self.assertGreater(event.metadata["gpu_elapsed_ms"], 0)
                    self.assertTrue(event.source_nodes)
            return result.outputs[output_name]

    def test_bevformer_tiny_msda_slice(self):
        graph = build_msda_graph()
        inputs = make_msda_inputs(graph, seed=3070)
        expected = reference_run(graph, inputs)["output"]
        result = self.compare(graph, inputs, expected)
        self.assertEqual(result.shape, (6, 2500, 256))
        self.assertTrue(np.isfinite(result).all())

    def test_odd_matmul(self):
        for m, k, n in [(1, 1, 1), (17, 19, 23), (31, 65, 33)]:
            with self.subTest(shape=(m, k, n)):
                a = self.rng.normal(size=(m, k)).astype(np.float32)
                b = self.rng.normal(size=(k, n)).astype(np.float32)
                self.compare(single_op("MatMul", [a.shape, b.shape]), {"x0": a, "x1": b}, a @ b)

    def test_batched_matmul_broadcast(self):
        for left_shape, right_shape in [
            ((2, 3, 4), (4, 5)),
            ((2, 1, 3, 4), (1, 7, 4, 5)),
        ]:
            with self.subTest(left_shape=left_shape, right_shape=right_shape):
                left = self.rng.normal(size=left_shape).astype(np.float32)
                right = self.rng.normal(size=right_shape).astype(np.float32)
                self.compare(
                    single_op("MatMul", [left_shape, right_shape]),
                    {"x0": left, "x1": right},
                    np.matmul(left, right),
                )

    def test_fused_matmul_bias_gelu(self):
        a = self.rng.normal(size=(7, 31)).astype(np.float32)
        b = self.rng.normal(size=(31, 33)).astype(np.float32)
        bias = self.rng.normal(size=(33,)).astype(np.float32)
        self.compare(single_op("MatMulBiasGELU", [a.shape, b.shape, bias.shape]), {"x0": a, "x1": b, "x2": bias}, gelu(a @ b + bias))

    def test_broadcast_add(self):
        for shapes in [[(2, 1, 3), (1, 4, 1)], [(), (17, 19)], [(1,), (1,)]]:
            with self.subTest(shapes=shapes):
                a, b = [self.rng.normal(size=shape).astype(np.float32) for shape in shapes]
                self.compare(single_op("Add", shapes), {"x0": a, "x1": b}, a + b)

    def test_fused_broadcast_add_relu(self):
        graph = Graph(
            "broadcast_add_relu",
            {
                "left": TensorSpec((2, 1, 3), "float32"),
                "right": TensorSpec((1, 4, 1), "float32"),
            },
            ("y",),
            [
                Node("add", "Add", ("left", "right"), ("summed",)),
                Node("relu", "ReLU", ("summed",), ("y",)),
            ],
        )
        left = self.rng.normal(size=(2, 1, 3)).astype(np.float32)
        right = self.rng.normal(size=(1, 4, 1)).astype(np.float32)
        program = compile_cuda(graph)
        self.assertEqual(len(program.launches), 1)
        self.assertEqual(program.graph.nodes[0].op, "AddReLU")
        self.compare(graph, {"left": left, "right": right}, np.maximum(left + right, 0))

    def test_transformer_elementwise(self):
        x = self.rng.normal(size=(2, 3, 4)).astype(np.float32)
        bias = self.rng.normal(size=(4,)).astype(np.float32)
        scale = self.rng.uniform(0.25, 2.0, size=(1, 3, 1)).astype(np.float32)
        denominator = np.asarray(2.0, dtype=np.float32)
        cases = [
            ("Sub", x, bias, x - bias),
            ("Mul", x, scale, x * scale),
            ("Div", x, denominator, x / denominator),
        ]
        for op, left, right, expected in cases:
            with self.subTest(op=op):
                self.compare(
                    single_op(op, [left.shape, right.shape]),
                    {"x0": left, "x1": right},
                    expected,
                )
        expected = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        self.compare(single_op("Sigmoid", [x.shape]), {"x0": x}, expected)

    def test_static_tensor_transforms_and_reduction(self):
        x = self.rng.normal(size=(2, 3, 4)).astype(np.float32)
        transpose = Graph(
            "transpose", {"x": TensorSpec(x.shape, "float32")}, ("y",),
            [Node("transpose", "Transpose", ("x",), ("y",), {"perm": [1, 0, 2]})],
        )
        self.compare(transpose, {"x": x}, x.transpose(1, 0, 2))

        reduce = Graph(
            "reduce", {"x": TensorSpec(x.shape, "float32")}, ("y",),
            [Node("reduce", "ReduceMean", ("x",), ("y",), {"axes": [1, 2], "keepdims": False})],
        )
        self.compare(reduce, {"x": x}, x.mean(axis=(1, 2)))

    def test_internal_reshape_executes_as_an_arena_view(self):
        x = self.rng.normal(size=(2, 3)).astype(np.float32)
        graph = Graph(
            "reshape_view",
            {"x": TensorSpec(x.shape, "float32")},
            ("y",),
            [
                Node("materialize", "Identity", ("x",), ("materialized",)),
                Node(
                    "reshape",
                    "Reshape",
                    ("materialized",),
                    ("view",),
                    {"shape": [3, 2]},
                ),
                Node("consume", "Identity", ("view",), ("y",)),
            ],
        )
        program = compile_cuda(graph)
        self.assertEqual([launch.node_id for launch in program.launches], ["materialize", "consume"])
        with CudaSession(program) as session:
            source = session.memory_plan.allocation("materialized")
            view = session.memory_plan.allocation("view")
            self.assertEqual(view.view_of, "materialized")
            self.assertEqual(view.offset, source.offset)
            self.assertEqual(session.memory_plan.arena_size, source.size_bytes)
            actual = session.run({"x": x}, trace=True).outputs["y"]
        np.testing.assert_array_equal(actual, x.reshape(3, 2))

    def test_contiguous_slice_executes_as_an_offset_arena_view(self):
        x = self.rng.normal(size=(2, 3, 4)).astype(np.float32)
        graph = Graph(
            "slice_view",
            {"x": TensorSpec(x.shape, "float32")},
            ("y",),
            [
                Node("materialize", "Identity", ("x",), ("materialized",)),
                Node(
                    "slice",
                    "Slice",
                    ("materialized",),
                    ("view",),
                    {"starts": [1], "ends": [2], "axes": [0], "steps": [1]},
                ),
                Node("consume", "Identity", ("view",), ("y",)),
            ],
        )
        program = compile_cuda(graph)
        self.assertEqual([launch.node_id for launch in program.launches], ["materialize", "consume"])
        with CudaSession(program) as session:
            source = session.memory_plan.allocation("materialized")
            view = session.memory_plan.allocation("view")
            self.assertEqual(view.view_of, "materialized")
            self.assertEqual(view.offset, source.offset + 48)
            actual = session.run({"x": x}, trace=True).outputs["y"]
        np.testing.assert_array_equal(actual, x[1:2])

    def test_grid_sample_identity(self):
        value = self.rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        grid = np.empty((1, 3, 4, 2), dtype=np.float32)
        for y in range(3):
            for x in range(4):
                grid[0, y, x] = ((2 * x + 1) / 4 - 1, (2 * y + 1) / 3 - 1)
        graph = Graph(
            "grid",
            {"value": TensorSpec(value.shape, "float32"), "grid": TensorSpec(grid.shape, "float32")},
            ("y",),
            [Node("grid", "GridSample2D", ("value", "grid"), ("y",), {"align_corners": False})],
        )
        self.compare(graph, {"value": value, "grid": grid}, value)

    def test_streaming_motion_and_int32_state_selection(self):
        motion = build_bevformer_motion_graph(
            BevFormerMotionConfig(
                bev_h=3,
                bev_w=4,
                embed_dim=2,
                grid_length_y=2.0,
                grid_length_x=1.5,
                rotate_center=(2.0, 1.5),
            )
        )
        motion_inputs = {
            "prev_bev": np.arange(24, dtype=np.float32).reshape(1, 12, 2),
            "can_bus_delta": np.asarray(
                [[0.25, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 17.0]],
                dtype=np.float32,
            ),
        }
        expected_motion = reference_run(motion, motion_inputs)
        with CudaSession(compile_cuda(motion)) as session:
            actual_motion = session.run(motion_inputs).outputs
        for name in motion.outputs:
            np.testing.assert_allclose(
                actual_motion[name], expected_motion[name], rtol=1e-5, atol=1e-6
            )

        select = Graph(
            "stream_state_select",
            {
                "has_prev": TensorSpec((1,), "int32"),
                "history": TensorSpec((1, 2, 3), "float32"),
                "initial": TensorSpec((1, 2, 3), "float32"),
            },
            ("selected",),
            [
                Node(
                    "select",
                    "Where",
                    ("has_prev", "history", "initial"),
                    ("selected",),
                )
            ],
        )
        history = self.rng.normal(size=(1, 2, 3)).astype(np.float32)
        initial = self.rng.normal(size=(1, 2, 3)).astype(np.float32)
        with CudaSession(compile_cuda(select)) as session:
            for flag, expected in ((0, initial), (1, history)):
                result = session.run(
                    {
                        "has_prev": np.asarray([flag], dtype=np.int32),
                        "history": history,
                        "initial": initial,
                    }
                )
                np.testing.assert_array_equal(result.outputs["selected"], expected)

    def test_cnn_baseline_ops(self):
        x = self.rng.normal(size=(1, 2, 5, 5)).astype(np.float32)
        weight = self.rng.normal(size=(4, 1, 3, 3)).astype(np.float32)
        graph = Graph(
            "grouped_conv",
            {"x": TensorSpec(x.shape, "float32"), "weight": TensorSpec(weight.shape, "float32")},
            ("y",),
            [Node("conv", "Conv2D", ("x", "weight"), ("y",), {"groups": 2, "pads": [1, 1, 1, 1]})],
        )
        expected = reference_run(graph, {"x": x, "weight": weight})["y"]
        self.compare(graph, {"x": x, "weight": weight}, expected)

        pooled_graph = Graph(
            "pool", {"x": TensorSpec(x.shape, "float32")}, ("y",),
            [Node("pool", "MaxPool2D", ("x",), ("y",), {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]})],
        )
        expected = reference_run(pooled_graph, {"x": x})["y"]
        self.compare(pooled_graph, {"x": x}, expected)

    def test_lowered_multihead_attention(self):
        config = MultiHeadAttentionConfig(1, 5, 7, embed_dim=32, heads=4)
        weights = {}
        for prefix in ("q", "k", "v", "out"):
            weights[f"{prefix}_weight"] = (
                self.rng.normal(size=(32, 32)).astype(np.float32) / np.sqrt(32)
            ).astype(np.float32)
            weights[f"{prefix}_bias"] = self.rng.normal(size=(32,)).astype(np.float32)
        graph = build_multihead_attention_graph(config, weights)
        inputs = {
            "query": self.rng.normal(size=(1, 5, 32)).astype(np.float32),
            "key": self.rng.normal(size=(1, 7, 32)).astype(np.float32),
            "value": self.rng.normal(size=(1, 7, 32)).astype(np.float32),
        }
        expected = reference_run(graph, inputs)[graph.outputs[0]]
        self.compare(graph, inputs, expected)

    def test_unary(self):
        x = self.rng.normal(size=(7, 35)).astype(np.float32)
        for op, expected in [("Identity", x), ("ReLU", np.maximum(x, 0)), ("GELU", gelu(x))]:
            with self.subTest(op=op):
                self.compare(single_op(op, [x.shape]), {"x0": x}, expected)

    def test_explicit_float16_cast_roundtrip(self):
        graph = Graph(
            "cast_roundtrip",
            {"x": TensorSpec((7, 35), "float32")},
            ("y",),
            [
                Node("to_half", "Cast", ("x",), ("half",), {"to": "float16"}),
                Node("to_float", "Cast", ("half",), ("y",), {"to": "float32"}),
            ],
        )
        x = self.rng.normal(size=(7, 35)).astype(np.float32)
        self.compare(graph, {"x": x}, x.astype(np.float16).astype(np.float32))

    def test_layernorm_reduction_boundaries(self):
        for width in [1, 7, 255, 256, 257, 1025]:
            with self.subTest(width=width):
                x = self.rng.normal(size=(2, 3, width)).astype(np.float32)
                scale = self.rng.normal(size=(width,)).astype(np.float32)
                bias = self.rng.normal(size=(width,)).astype(np.float32)
                mean = np.mean(x, axis=-1, keepdims=True)
                expected = (x - mean) / np.sqrt(np.mean((x - mean)**2, axis=-1, keepdims=True) + 1e-5)
                expected = expected * scale + bias
                self.compare(single_op("LayerNorm", [x.shape, scale.shape, bias.shape]), {"x0": x, "x1": scale, "x2": bias}, expected)

    def test_softmax_stability(self):
        for width in [1, 13, 257]:
            with self.subTest(width=width):
                x = self.rng.uniform(-1000, 1000, size=(5, width)).astype(np.float32)
                e = np.exp(x - np.max(x, axis=-1, keepdims=True))
                result = self.compare(single_op("Softmax", [x.shape]), {"x0": x}, e / e.sum(axis=-1, keepdims=True))
                np.testing.assert_allclose(result.sum(axis=-1), 1, atol=1e-6)

    def test_empty_dispatch_graph(self):
        graph = Graph("passthrough", {"x": TensorSpec((3,), "float32")}, ("x",), [])
        x = np.arange(3, dtype=np.float32)
        with CudaSession(compile_cuda(graph)) as session:
            np.testing.assert_array_equal(session.run({"x": x}).outputs["x"], x)

    def test_constant_only_graph(self):
        graph = Graph("constant", {}, ("y",), [Node("a", "Add", ("a", "b"), ("y",))],
                      constants={"a": np.ones((3,), np.float32), "b": np.full((3,), 2, np.float32)})
        self.compare(graph, {}, np.full((3,), 3, np.float32))

    def test_mlp_repeated_runs_and_input_guards(self):
        path = Path(__file__).resolve().parents[1] / "examples/heterogeneous_mlp.json"
        graph = load_graph(path)
        w, bias = graph.constants["W"].copy(), graph.constants["bias"].copy()
        for fuse in [True, False]:
            with self.subTest(fuse=fuse), CudaSession(compile_cuda(graph, fuse=fuse)) as session:
                for seed in [1, 7, 1]:
                    x = np.random.default_rng(seed).normal(size=(64, 128)).astype(np.float32)
                    hidden = gelu(x @ w + bias)
                    mean = hidden.mean(axis=-1, keepdims=True)
                    expected = (hidden - mean) / np.sqrt(((hidden - mean)**2).mean(axis=-1, keepdims=True) + 1e-5)
                    result = session.run({"input": x})
                    np.testing.assert_allclose(result.outputs["output"], expected, rtol=1e-4, atol=1e-5)
                    result.outputs["output"].fill(0)
                with self.assertRaises(ValidationError):
                    session.run({"input": x.astype(np.float64)})
                np.testing.assert_allclose(session.run({"input": x}).outputs["output"], expected, rtol=1e-4, atol=1e-5)
            with self.assertRaises(BackendError):
                session.run({"input": x})
        np.testing.assert_array_equal(graph.constants["W"], w)
        np.testing.assert_array_equal(graph.constants["bias"], bias)


if __name__ == "__main__":
    unittest.main()
