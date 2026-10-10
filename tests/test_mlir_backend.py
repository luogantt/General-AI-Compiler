"""Contract and textual lowering tests for the standard-dialect MLIR path."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

from gac.backends.mlir import compile_mlir, export_mlir
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError
from gac.ir import Graph, Node, TensorSpec
from gac.mlir_toolchain import MlirOptNotFoundError, discover_mlir_opt


def _ffn_graph() -> Graph:
    config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
    rng = np.random.default_rng(1729)
    weights = {
        name: rng.standard_normal(shape, dtype=np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }
    return build_bevformer_ffn_graph(weights, config, name="bevformer-ffn/mlir")


def _single(op: str, shapes, attrs=None, *, dtype="float32", layout="contiguous"):
    inputs = {
        f"x{index}": TensorSpec(tuple(shape), dtype, layout)
        for index, shape in enumerate(shapes)
    }
    return Graph(
        f"test-{op}",
        inputs,
        ("y",),
        [Node("operation", op, tuple(inputs), ("y",), attrs or {})],
    )


class MlirBackendTests(unittest.TestCase):
    def test_bevformer_ffn_is_deterministic_standard_mlir(self):
        graph = _ffn_graph()
        before = graph.to_json()
        first = compile_mlir(graph)
        second = compile_mlir(graph)

        self.assertEqual(graph.to_json(), before)
        self.assertEqual(first.source, second.source)
        self.assertEqual(export_mlir(graph), first.source)
        self.assertEqual(
            [node.op for node in first.graph.nodes],
            ["MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm"],
        )
        self.assertIn("func.func @bevformer_ffn_mlir", first.source)
        self.assertEqual(first.source.count("linalg.matmul"), 2)
        self.assertEqual(first.source.count("linalg.reduce"), 2)
        self.assertIn("tensor.collapse_shape", first.source)
        self.assertIn("tensor.expand_shape", first.source)
        self.assertIn("math.rsqrt", first.source)
        self.assertIn("arith.maximumf", first.source)
        self.assertNotIn("gac.", first.source)
        self.assertNotIn("allow-unregistered", first.source)

        description = first.describe()
        self.assertEqual(description["compiler_backend"], "mlir-linalg")
        self.assertEqual(description["target_intent"], "cuda-sm86")
        self.assertEqual(description["planned_codegen_backend"], "mlir-nvvm")
        self.assertEqual(description["lowering_stage"], "tensor")
        self.assertFalse(description["executable"])

    def test_rank_three_matmul_collapses_and_restores_batch_shape(self):
        source = compile_mlir(_single("MatMul", [(2, 3, 4), (4, 5)])).source
        self.assertIn(
            "tensor.collapse_shape %arg0 [[0, 1], [2]] : "
            "tensor<2x3x4xf32> into tensor<6x4xf32>",
            source,
        )
        self.assertIn("linalg.matmul", source)
        self.assertIn(
            "tensor.expand_shape", source
        )
        self.assertIn("output_shape [2, 3, 5]", source)

    def test_numpy_broadcast_uses_projected_affine_indexing_maps(self):
        graph = _single("Add", [(2, 3, 4), (4,)])
        source = compile_mlir(graph).source
        self.assertIn(
            "affine_map<(d0, d1, d2) -> (d0, d1, d2)>", source
        )
        self.assertIn("affine_map<(d0, d1, d2) -> (d2)>", source)

        singleton = _single("Add", [(2, 3, 4), (1, 3, 1)])
        singleton_source = compile_mlir(singleton).source
        self.assertIn(
            "affine_map<(d0, d1, d2) -> (0, d1, 0)>", singleton_source
        )

    def test_layer_norm_uses_prefix_projection_not_numpy_tail_alignment(self):
        graph = _single(
            "LayerNorm",
            [(2, 3, 4), (4,), (4,)],
            {"axis": -1, "epsilon": 1e-5},
        )
        source = compile_mlir(graph).source
        self.assertEqual(source.count("linalg.reduce"), 2)
        self.assertIn("outs(", source)
        self.assertIn("tensor<2x3xf32>) dimensions = [2]", source)
        self.assertIn("affine_map<(d0, d1, d2) -> (d0, d1)>", source)
        self.assertIn("affine_map<(d0, d1, d2) -> (d2)>", source)

        parameter_free = _single(
            "LayerNorm", [(2, 3, 4)], {"axis": -1, "epsilon": 1e-5}
        )
        parameter_free_source = compile_mlir(parameter_free).source
        self.assertEqual(parameter_free_source.count("linalg.reduce"), 2)
        self.assertNotIn("%scaled =", parameter_free_source)
        self.assertNotIn("%shifted =", parameter_free_source)

        rank_one = _single("LayerNorm", [(4,)], {"axis": -1})
        rank_one_source = compile_mlir(rank_one).source
        self.assertIn("outs(", rank_one_source)
        self.assertIn("tensor<f32>) dimensions = [0]", rank_one_source)
        self.assertIn("affine_map<(d0) -> ()>", rank_one_source)

    def test_constants_are_inline_deterministic_and_splats_are_compact(self):
        graph = Graph(
            "constant_identity",
            {},
            ("y",),
            [Node("identity", "Identity", ("constant",), ("y",))],
            constants={"constant": np.ones((2, 3), np.float32)},
        )
        source = compile_mlir(graph).source
        self.assertIn(
            "arith.constant dense<1.0> : tensor<2x3xf32>", source
        )
        self.assertEqual(source, compile_mlir(graph).source)

        values = np.asarray([[0.25, -1.5], [2.0, 3.125]], np.float32)
        graph = Graph(
            "constant_identity",
            {},
            ("y",),
            [Node("identity", "Identity", ("constant",), ("y",))],
            constants={"constant": values},
        )
        source = compile_mlir(graph).source
        self.assertIn("dense<[[0.25, -1.5], [2.0, 3.125]]>", source)

        signed_zeros = np.asarray([0.0, -0.0], np.float32)
        graph = Graph(
            "signed_zero_identity",
            {},
            ("y",),
            [Node("identity", "Identity", ("constant",), ("y",))],
            constants={"constant": signed_zeros},
        )
        source = compile_mlir(graph).source
        self.assertIn("dense<[0.0, -0.0]>", source)

    def test_fusion_can_be_disabled(self):
        graph = Graph(
            "add_relu",
            {
                "x": TensorSpec((2, 4), "float32"),
                "bias": TensorSpec((4,), "float32"),
            },
            ("y",),
            [
                Node("add", "Add", ("x", "bias"), ("sum",)),
                Node("relu", "ReLU", ("sum",), ("y",)),
            ],
        )
        fused = compile_mlir(graph)
        unfused = compile_mlir(graph, fuse=False)
        self.assertEqual([node.op for node in fused.graph.nodes], ["AddReLU"])
        self.assertEqual([node.op for node in unfused.graph.nodes], ["Add", "ReLU"])
        self.assertEqual(fused.source.count("linalg.generic"), 1)
        self.assertEqual(unfused.source.count("linalg.generic"), 2)

    def test_fused_matmul_bias_gelu_decomposes_to_standard_ops(self):
        graph = _single("MatMulBiasGELU", [(2, 4), (4, 5), (5,)])
        source = compile_mlir(graph).source
        self.assertEqual(source.count("linalg.matmul"), 1)
        self.assertEqual(source.count("linalg.generic"), 2)
        self.assertIn("arith.addf", source)
        self.assertIn("math.tanh", source)
        self.assertNotIn('"gac.', source)

    def test_rejects_unsupported_semantics_with_backend_errors(self):
        layer_norm_axis = _single(
            "LayerNorm", [(2, 4), (4,), (4,)], {"axis": 0}
        )
        cases = [
            _single("MatMul", [(None, 4), (4, 5)]),
            _single("MatMul", [(2, 4), (4, 5)], dtype="float16"),
            _single("MatMul", [(2, 4), (1, 4, 5)]),
            _single("GELU", [(2, 4)], {"approximation": "erf"}),
            _single("Softmax", [(2, 4)], {"axis": -1}),
            _single("Identity", [(2, 4)], layout="NCHW"),
            layer_norm_axis,
        ]
        for graph in cases:
            with self.subTest(graph=graph.name, inputs=graph.inputs):
                with self.assertRaises(BackendError):
                    compile_mlir(graph)

        malformed_attrs = _single("ReLU", [(2, 4)], {"surprise": True})
        with self.assertRaisesRegex(
            BackendError, r"node 'operation' \(ReLU\): unsupported attributes"
        ):
            compile_mlir(malformed_attrs)

    def test_upstream_mlir_opt_accepts_full_bevformer_ffn(self):
        required = os.environ.get("GAC_REQUIRE_MLIR") == "1"
        if not os.environ.get("GAC_MLIR_OPT") and not required:
            self.skipTest(
                "set GAC_MLIR_OPT to run the upstream parser/verifier integration"
            )
        try:
            executable = str(discover_mlir_opt())
        except MlirOptNotFoundError as exc:
            if required:
                self.fail(f"GAC_REQUIRE_MLIR=1 but mlir-opt is unavailable: {exc}")
            self.skipTest(str(exc))
        result = subprocess.run(
            [executable, "--verify-each"],
            input=compile_mlir(_ffn_graph()).source,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
