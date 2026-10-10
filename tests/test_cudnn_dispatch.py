"""CPU-only checks for the optional cuDNN Conv2D dispatch policy."""

from __future__ import annotations

import unittest

from gac.backends.cuda import compile_cuda
from gac.errors import ValidationError
from gac.ir import Graph, Node, TensorSpec
from gac.runtime import (
    CudaSession,
    _supports_cublas_matmul,
    _supports_cudnn_convolution,
)


def _conv_graph(*, pads=(1, 1, 1, 1)) -> Graph:
    return Graph(
        "conv_dispatch",
        {
            "x": TensorSpec((1, 4, 7, 7), "float32"),
            "weight": TensorSpec((8, 2, 3, 3), "float32"),
            "bias": TensorSpec((8,), "float32"),
        },
        ("y",),
        [
            Node(
                "conv",
                "Conv2D",
                ("x", "weight", "bias"),
                ("y",),
                {
                    "pads": list(pads),
                    "strides": [2, 2],
                    "dilations": [1, 1],
                    "groups": 2,
                },
            )
        ],
    )


class CudnnDispatchTests(unittest.TestCase):
    def test_compatibility_requires_symmetric_padding(self):
        symmetric = compile_cuda(_conv_graph()).graph
        asymmetric = compile_cuda(_conv_graph(pads=(1, 0, 2, 1))).graph

        self.assertTrue(
            _supports_cudnn_convolution(symmetric.nodes[0], symmetric)
        )
        self.assertEqual(
            compile_cuda(_conv_graph()).launches[0].implementation_candidates,
            ("cudnn", "generated-cuda"),
        )
        self.assertFalse(
            _supports_cudnn_convolution(asymmetric.nodes[0], asymmetric)
        )
        self.assertEqual(
            compile_cuda(
                _conv_graph(pads=(1, 0, 2, 1))
            ).launches[0].implementation_candidates,
            ("generated-cuda",),
        )

    def test_cudnn_dispatch_preserves_conv_attributes_and_buffers(self):
        program = compile_cuda(_conv_graph())
        launch = program.launches[0]
        calls = []

        class FakeCudnn:
            @staticmethod
            def convolution_forward(*args, **kwargs):
                calls.append((args, kwargs))

        values = {name: object() for name in (*launch.inputs, launch.output)}
        session = CudaSession.__new__(CudaSession)
        session._cudnn_launch_ids = {launch.node_id}
        session._nodes_by_id = {node.id: node for node in program.graph.nodes}
        session._cudnn = FakeCudnn()
        session._cudnn_tensor_core = "never"

        session._execute_launch(launch, values)

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertIs(args[0], values["x"])
        self.assertIs(args[1], values["weight"])
        self.assertIs(args[2], values["bias"])
        self.assertIs(args[3], values["y"])
        self.assertEqual(args[4:], ((1, 1), (2, 2), (1, 1), 2))
        self.assertEqual(
            kwargs, {"auto_tune": False, "tensor_core": "never"}
        )

    def test_invalid_backend_policy_is_rejected_before_device_access(self):
        with self.assertRaisesRegex(ValidationError, "conv_backend"):
            CudaSession(compile_cuda(_conv_graph()), conv_backend="invalid")

    def test_cupy_matmul_dispatch_writes_the_planned_output(self):
        graph = Graph(
            "matmul_dispatch",
            {
                "left": TensorSpec((2, 3, 4), "float32"),
                "right": TensorSpec((4, 5), "float32"),
            },
            ("output",),
            [Node("matmul", "MatMul", ("left", "right"), ("output",))],
        )
        program = compile_cuda(graph)
        launch = program.launches[0]
        self.assertTrue(
            _supports_cublas_matmul(program.graph.nodes[0], program.graph)
        )
        self.assertEqual(
            launch.implementation_candidates,
            ("cupy-cublas", "generated-cuda"),
        )
        calls = []

        class FakeCupy:
            @staticmethod
            def matmul(left, right, *, out):
                calls.append((left, right, out))

        values = {name: object() for name in (*launch.inputs, launch.output)}
        session = CudaSession.__new__(CudaSession)
        session._cudnn_launch_ids = set()
        session._cublas_launch_ids = {launch.node_id}
        session._cp = FakeCupy()

        session._execute_launch(launch, values)

        self.assertEqual(
            calls, [(values["left"], values["right"], values["output"])]
        )

    def test_invalid_matmul_policy_is_rejected_before_device_access(self):
        with self.assertRaisesRegex(ValidationError, "matmul_backend"):
            CudaSession(compile_cuda(_conv_graph()), matmul_backend="invalid")


if __name__ == "__main__":
    unittest.main()
