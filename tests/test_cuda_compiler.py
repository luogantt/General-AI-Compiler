from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.errors import BackendError, ValidationError
from gac.frontends.json_frontend import load_graph
from gac.ir import Graph, Node, TensorSpec
from gac.runtime import bind_inputs, reference_run


ROOT = Path(__file__).resolve().parents[1]


def single_op(op, shapes, attrs=None, dtype="float32"):
    names = tuple(f"x{i}" for i in range(len(shapes)))
    return Graph(op, {name: TensorSpec(tuple(shape), dtype) for name, shape in zip(names, shapes)},
                 ("y",), [Node("operation", op, names, ("y",), attrs or {})])


class CompilerTests(unittest.TestCase):
    def test_compile_mlp_preserves_original_and_provenance(self):
        original = load_graph(ROOT / "examples/heterogeneous_mlp.json")
        before = original.to_json()
        program = compile_cuda(original)
        self.assertEqual(original.to_json(), before)
        self.assertEqual(len(program.launches), 2)
        self.assertEqual(program.launches[0].source_nodes, ("matmul", "add_bias", "gelu"))
        self.assertEqual(program.launches[0].grid, (16, 4, 1))
        self.assertEqual(len(compile_cuda(original, fuse=False).launches), 4)
        self.assertEqual(program.source, compile_cuda(original).source)

    def test_offline_compile_does_not_import_cupy(self):
        code = "from gac.backends.cuda import compile_cuda; from gac.frontends.json_frontend import load_graph; import sys; compile_cuda(load_graph('examples/heterogeneous_mlp.json')); assert 'cupy' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_unsupported_semantics(self):
        cases = [
            single_op("MatMul", [(None, 4), (4, 5)]),
            single_op("MatMul", [(0, 4), (4, 5)]),
            single_op("MatMul", [(2, 4), (4, 5)], dtype="float16"),
            single_op("GELU", [(3, 4)], {"approximation": "erf"}),
            single_op("GELU", [(3, 4)], {"unexpected": True}),
            single_op("LayerNorm", [(3, 4)], {"axis": 0}),
            single_op("LayerNorm", [(3, 4)], {"epsilon": -1}),
            single_op("LayerNorm", [(3, 4)], {"epsilon": float("nan")}),
            single_op("LayerNorm", [(3, 4), (1,)]),
            single_op("Softmax", [(3, 4)], {"axis": True}),
        ]
        for graph in cases:
            with self.subTest(graph=graph.name, inputs=graph.inputs, attrs=graph.nodes[0].attrs):
                with self.assertRaises(BackendError):
                    compile_cuda(graph)

    def test_dense_nchw_layout_is_accepted_but_unknown_layout_is_rejected(self):
        nchw = Graph(
            "nchw_identity",
            {"x": TensorSpec((1, 2, 3, 4), "float32", "NCHW")},
            ("y",),
            [Node("identity", "Identity", ("x",), ("y",))],
        )
        self.assertEqual(compile_cuda(nchw).graph.value_specs["x"].layout, "NCHW")
        unknown = Graph(
            "blocked_layout",
            {"x": TensorSpec((1, 2, 3, 4), "float32", "NCHWc8")},
            ("y",),
            [Node("identity", "Identity", ("x",), ("y",))],
        )
        with self.assertRaisesRegex(BackendError, "logical layout"):
            compile_cuda(unknown)

    def test_batched_matmul_broadcast_codegen_and_reference(self):
        cases = [
            ((2, 3, 4), (4, 5), (2, 3, 5)),
            ((2, 1, 3, 4), (1, 7, 4, 5), (2, 7, 3, 5)),
        ]
        rng = np.random.default_rng(41)
        for left_shape, right_shape, output_shape in cases:
            with self.subTest(left_shape=left_shape, right_shape=right_shape):
                graph = single_op("MatMul", [left_shape, right_shape])
                program = compile_cuda(graph)
                self.assertEqual(program.graph.value_specs["y"].shape, output_shape)
                self.assertIn("int batch_index", program.source)
                left = rng.standard_normal(left_shape, dtype=np.float32)
                right = rng.standard_normal(right_shape, dtype=np.float32)
                actual = reference_run(program.reference_graph, {"x0": left, "x1": right})["y"]
                np.testing.assert_allclose(actual, np.matmul(left, right), rtol=1e-6, atol=1e-6)

    def test_transformer_elementwise_codegen_and_reference(self):
        graph = Graph(
            "elementwise_chain",
            {
                "x": TensorSpec((2, 3, 4), "float32"),
                "bias": TensorSpec((4,), "float32"),
                "scale": TensorSpec((1, 3, 1), "float32"),
                "denominator": TensorSpec((), "float32"),
            },
            ("y",),
            [
                Node("sub", "Sub", ("x", "bias"), ("sub",)),
                Node("mul", "Mul", ("sub", "scale"), ("mul",)),
                Node("div", "Div", ("mul", "denominator"), ("div",)),
                Node("sigmoid", "Sigmoid", ("div",), ("sigmoid",)),
                Node("log", "Log", ("sigmoid",), ("y",)),
            ],
        )
        program = compile_cuda(graph)
        self.assertEqual(
            [launch.node_id for launch in program.launches],
            ["sub", "mul", "div", "sigmoid", "log"],
        )
        self.assertIn("expf", program.source)
        self.assertIn("logf", program.source)
        x = np.linspace(-3, 3, 24, dtype=np.float32).reshape(2, 3, 4)
        bias = np.asarray([0.5, -0.25, 1.0, -2.0], dtype=np.float32)
        scale = np.asarray([1.0, 2.0, 0.5], dtype=np.float32).reshape(1, 3, 1)
        denominator = np.asarray(2.0, dtype=np.float32)
        actual = reference_run(
            program.reference_graph,
            {"x": x, "bias": bias, "scale": scale, "denominator": denominator},
        )["y"]
        expected_input = ((x - bias) * scale) / denominator
        expected = np.log(1.0 / (1.0 + np.exp(-expected_input)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_add_relu_fuses_one_broadcast_kernel_and_preserves_reference(self):
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
        before = graph.to_json()
        program = compile_cuda(graph)

        self.assertEqual(graph.to_json(), before)
        self.assertEqual(len(program.launches), 1)
        self.assertEqual(program.launches[0].node_id, "fused_add_relu")
        self.assertEqual(program.launches[0].source_nodes, ("add", "relu"))
        self.assertEqual(program.graph.nodes[0].op, "AddReLU")
        self.assertNotIn("summed", program.graph.value_specs)
        self.assertIn("float value = p0[", program.source)
        self.assertIn("value < 0.0f ? 0.0f : value", program.source)
        self.assertEqual(program.source, compile_cuda(graph).source)

        left = np.linspace(-6, 5, 6, dtype=np.float32).reshape(2, 1, 3)
        right = np.asarray([-2, -0.5, 1, 3], dtype=np.float32).reshape(1, 4, 1)
        actual = reference_run(
            program.reference_graph, {"left": left, "right": right}
        )["y"]
        np.testing.assert_array_equal(actual, np.maximum(left + right, 0))

        unfused = compile_cuda(graph, fuse=False)
        self.assertEqual([launch.node_id for launch in unfused.launches], ["add", "relu"])
        self.assertEqual([node.op for node in unfused.graph.nodes], ["Add", "ReLU"])

    def test_add_relu_fusion_keeps_observable_or_shared_sum(self):
        observable = Graph(
            "observable_sum",
            {"left": TensorSpec((4,), "float32"), "right": TensorSpec((4,), "float32")},
            ("summed", "y"),
            [
                Node("add", "Add", ("left", "right"), ("summed",)),
                Node("relu", "ReLU", ("summed",), ("y",)),
            ],
        )
        self.assertEqual(
            [node.op for node in compile_cuda(observable).graph.nodes],
            ["Add", "ReLU"],
        )

        shared = Graph(
            "shared_sum",
            {"left": TensorSpec((4,), "float32"), "right": TensorSpec((4,), "float32")},
            ("y", "copy"),
            [
                Node("add", "Add", ("left", "right"), ("summed",)),
                Node("relu", "ReLU", ("summed",), ("y",)),
                Node("copy", "Identity", ("summed",), ("copy",)),
            ],
        )
        self.assertEqual(
            [node.op for node in compile_cuda(shared).graph.nodes],
            ["Add", "ReLU", "Identity"],
        )

    def test_unsupported_fusion_keeps_unfused_graph(self):
        graph = Graph("matrix_bias", {"x": TensorSpec((3, 4), "float32"), "w": TensorSpec((4, 5), "float32"), "b": TensorSpec((3, 5), "float32")},
                      ("y",), [Node("m", "MatMul", ("x", "w"), ("m",)), Node("a", "Add", ("m", "b"), ("a",)), Node("g", "GELU", ("a",), ("y",))])
        program = compile_cuda(graph)
        self.assertEqual(len(program.launches), 3)
        self.assertEqual(program.pass_reports[-1]["name"], "cuda-fusion-rejected")

    def test_input_contract(self):
        graph = single_op("Identity", [(3, 4)])
        good = np.ones((3, 4), np.float32)
        cases = [{}, {"x0": good.astype(np.float64)}, {"x0": good.T},
                 {"x0": np.ones((3, 8), np.float32)[:, ::2]}, {"x0": np.full((3, 4), np.nan, np.float32)}]
        for inputs in cases:
            with self.assertRaises(ValidationError):
                bind_inputs(graph, inputs)

    def test_explicit_mixed_precision_cast_codegen_and_reference(self):
        graph = Graph(
            "cast_roundtrip",
            {"x": TensorSpec((3, 5), "float32")},
            ("y",),
            [
                Node("to_half", "Cast", ("x",), ("half",), {"to": "float16"}),
                Node("to_float", "Cast", ("half",), ("y",), {"to": "float32"}),
            ],
        )
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["half"].dtype, "float16")
        self.assertEqual(program.graph.value_specs["y"].dtype, "float32")
        self.assertIn("__float2half_rn", program.source)
        self.assertIn("__half2float", program.source)
        x = np.linspace(-2, 2, 15, dtype=np.float32).reshape(3, 5)
        expected = x.astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(reference_run(program.reference_graph, {"x": x})["y"], expected)

    def test_cli_artifacts_and_cpu_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.cu"
            result = subprocess.run([sys.executable, "-m", "gac.cli", "compile", "examples/heterogeneous_mlp.json", "-o", str(output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.with_suffix(".plan.json").read_text())
            self.assertFalse(plan["device_code_compiled"])
            self.assertEqual(plan["compute_capability"], [8, 6])
            result = subprocess.run([sys.executable, "-m", "gac.cli", "run", "examples/heterogeneous_mlp.json", "--backend", "numpy-cpu", "--verify", "--warmup", "0", "--repeat", "1"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["verification"], "passed")
            self.assertFalse(report["device"]["physical_cuda_execution"])

            package = Path(directory) / "model.hxp"
            result = subprocess.run(
                [sys.executable, "-m", "gac.cli", "compile", "examples/heterogeneous_mlp.json", "-o", str(package)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["artifact"], "hxp")
            self.assertTrue(package.exists())

            result = subprocess.run(
                [sys.executable, "-m", "gac.cli", "inspect", str(package)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            inspected = json.loads(result.stdout)
            self.assertTrue(inspected["codegen_verified"])
            self.assertEqual(inspected["target"], "cuda-sm86")
            self.assertEqual(inspected["node_count"], 2)

            result = subprocess.run(
                [sys.executable, "-m", "gac.cli", "run", str(package), "--backend", "numpy-cpu", "--verify", "--warmup", "0", "--repeat", "1"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            packaged_run = json.loads(result.stdout)
            self.assertEqual(packaged_run["verification"], "passed")
            self.assertEqual(packaged_run["program_preparation"], "verified-package-load-and-codegen-check")


if __name__ == "__main__":
    unittest.main()
