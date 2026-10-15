from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.vortex_scalar_ir import ScalarBuilder, Value, TYPES
from gac.backends.vortex_scalar_ops import emit_scalar_op
from gac.errors import BackendError
from gac.ir import Graph, Node, TensorSpec
from gac.ops import SUPPORTED_OPS, evaluate_op, infer_node


class NumericalBuilder(ScalarBuilder):
    """Execute the emitted scalar algorithm, checking every buffer access.

    The production builder supplies indexing helpers unchanged; numerical
    primitive operations replace MLIR emission. NumPy tensor operations remain
    an independent oracle for the complete lowering, including axis indexing.
    """

    def __init__(self, arrays):
        super().__init__()
        self.arrays = arrays
        self.values = {}

    def number(self, value):
        return self.values[value.name]

    def const(self, value, type="i32"):
        type = TYPES.get(type, type)
        result = self.fresh(type)
        dtype = {"i1": "bool", "i8": "uint8", "i32": "int32", "i64": "int64", "f16": "float16", "f32": "float32"}[type]
        self.values[result.name] = np.asarray(value).astype(dtype)[()]
        return result

    def binary(self, op, left, right):
        right = self.number(right) if isinstance(right, Value) else right
        left_value = self.number(left)
        operations = {"add": np.add, "sub": np.subtract, "mul": np.multiply,
                      "div": np.divide, "rem": np.fmod, "and": np.bitwise_and, "or": np.bitwise_or}
        return self.const(operations[op](left_value, right), left.type)

    def cmp(self, predicate, left, right):
        right = self.number(right) if isinstance(right, Value) else right
        operations = {"lt": np.less, "le": np.less_equal, "gt": np.greater,
                      "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}
        return self.const(operations[predicate](self.number(left), right), "i1")

    def select(self, condition, left, right):
        return left if self.number(condition) else right

    def cast(self, value, type):
        return self.const(self.number(value), type)

    def math(self, name, value):
        return self.const(getattr(np, name)(self.number(value)), value.type)

    def load(self, ptr, index, dtype="float32"):
        offset = int(self.number(index))
        assert 0 <= offset < self.arrays[ptr].size, (ptr, offset, self.arrays[ptr].size)
        return self.const(self.arrays[ptr].flat[offset], "i1" if dtype == "bool" else dtype)

    def store(self, value, ptr, index, dtype="float32"):
        offset = int(self.number(index))
        assert 0 <= offset < self.arrays[ptr].size
        self.arrays[ptr].flat[offset] = self.number(value)

    def loop(self, count, initial, body):
        count = int(self.number(count)) if isinstance(count, Value) else count
        result = tuple(initial)
        for index in range(count):
            result = tuple(body(self.const(index), result))
        return result

    def if_value(self, condition, then, otherwise):
        return then() if self.number(condition) else otherwise()


class VortexScalarOpsTests(unittest.TestCase):
    def check(self, op, arrays, attrs=None, *, rtol=2e-5, atol=2e-6):
        specs = tuple(TensorSpec(array.shape, array.dtype) for array in arrays)
        node = Node("test", op, tuple(f"x{i}" for i in range(len(arrays))), ("y",), attrs or {})
        output_spec = infer_node(node, dict(zip(node.inputs, specs, strict=True)))["y"]
        result = np.empty(output_spec.shape, dtype=output_spec.dtype)
        memory = dict(zip(node.inputs, arrays, strict=True)) | {"y": result}
        builder = NumericalBuilder(memory)
        for index in range(result.size):
            self.assertTrue(emit_scalar_op(builder, node, node.inputs, specs, "y", output_spec, builder.const(index)))
        expected = evaluate_op(op, arrays, attrs)[0]
        if result.dtype.kind in "biu":
            np.testing.assert_array_equal(result, expected)
        else:
            np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        # Exercise real typed emission too: missing primitive/type support must
        # fail here, rather than remaining hidden by the numerical interpreter.
        emitted = ScalarBuilder()
        self.assertTrue(emit_scalar_op(emitted, node, node.inputs, specs, "%out", output_spec, emitted.const(0)))
        self.assertTrue(any("llvm.store" in line for line in emitted.lines))
        return result

    def test_broadcast_arithmetic_comparisons_and_condition(self):
        a = np.asarray([[1, -3, 4], [-2, 0, 8]], dtype=np.float32)
        b = np.asarray([2, 3, 4], dtype=np.float32)
        for op in ("Add", "AddReLU", "Sub", "Mul", "Div", "Greater", "Less"):
            with self.subTest(op=op):
                self.check(op, (a, b))
        self.check("Where", (np.asarray([[0], [-2]], np.int32), a, b))
        self.check("Add", (np.asarray([16777217], np.int32), np.asarray([1], np.int32)))

    def test_unary_functions_and_f16_storage(self):
        a = np.asarray([-100, -4, -0.5, 0, 0.3, 5, 100], np.float32)
        for op in ("ReLU", "GELU", "Sigmoid", "Sin", "Cos", "Identity"):
            with self.subTest(op=op):
                self.check(op, (a,))
        self.check("Log", (np.asarray([0.01, 0.3, 1, 10], np.float32),))
        self.check("Clamp", (a,), {"min": -1.0, "max": 0.7})
        self.check("Clamp", (a,), {"max": 0.7})
        self.check("Cast", (a,), {"to": "float16"})
        half = np.asarray([-1, 0, 0.25, 2], np.float16)
        self.check("GELU", (half,), rtol=5e-3, atol=5e-4)
        self.check("Cast", (half,), {"to": "float32"})

    def test_matmul_vectors_broadcast_batches_and_bias(self):
        rng = np.random.default_rng(2)
        shapes = [((3,), (3,)), ((3,), (3, 4)), ((2, 3), (3,)),
                  ((2, 1, 3, 4), (1, 5, 4, 2)), ((4,), (2, 4, 3)),
                  ((2, 3, 4), (4,)), ((2, 3), (3, 4))]
        for a_shape, b_shape in shapes:
            with self.subTest(left=a_shape, right=b_shape):
                arrays = (rng.normal(size=a_shape).astype(np.float32), rng.normal(size=b_shape).astype(np.float32))
                self.check("MatMul", arrays)
        self.check("MatMulBiasGELU", (
            rng.normal(size=(2, 3)).astype(np.float32),
            rng.normal(size=(3, 4)).astype(np.float32),
            rng.normal(size=(5, 1, 4)).astype(np.float32),
        ))
        self.check("MatMul", (np.arange(6, dtype=np.float16).reshape(2, 3),
                              np.arange(12, dtype=np.float16).reshape(3, 4)))
        self.check("MatMul", (np.asarray([[16777217, 3]], np.int32),
                              np.asarray([[1], [2]], np.int32)))

    def test_f16_reductions_accumulate_in_f32(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(2, 4, 8)).astype(np.float16)
        self.check("LayerNorm", (a,), {"axis": 1}, rtol=5e-3, atol=2e-3)
        self.check("Softmax", (a,), {"axis": 1}, rtol=3e-3, atol=2e-4)
        self.check("ReduceSum", (a,), {"axes": [0, 2], "keepdims": False}, rtol=2e-3, atol=2e-3)
        self.check("MatMul", (a.reshape(8, 8), rng.normal(size=(8, 3)).astype(np.float16)), rtol=2e-3, atol=2e-3)

    def test_tensor_layout_and_concat_read_bounds(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self.check("Reshape", (a,), {"shape": [6, 4]})
        self.check("Transpose", (a,), {"perm": [2, 0, 1]})
        self.check("Transpose", (a,))
        self.check("Expand", (a[:, :1, :],), {"shape": [2, 3, 4]})
        self.check("Slice", (a,), {"starts": [1, 3], "ends": [-3, -5], "axes": [0, 2], "steps": [-1, -2]})
        self.check("Concat", (a, a[:, :1, :], a), {"axis": 1})
        self.check("Concat", (a, a[:, :, :2]), {"axis": -1})

    def test_reductions_arbitrary_axes_and_scalar_output(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 7
        for axes in ([], [0], [1], [-1], [0, 2], [0, 1, 2]):
            for keepdims in (False, True):
                for op in ("ReduceMean", "ReduceSum"):
                    with self.subTest(op=op, axes=axes, keepdims=keepdims):
                        self.check(op, (a,), {"axes": axes, "keepdims": keepdims})

    def test_layernorm_and_softmax_arbitrary_axes(self):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(2, 3, 4)).astype(np.float32)
        for axis in (0, 1, 2, -1):
            with self.subTest(axis=axis):
                self.check("LayerNorm", (a,), {"axis": axis})
                self.check("Softmax", (a * 1000,), {"axis": axis}, rtol=1e-4)
        self.check("LayerNorm", (a, np.ones((3, 1), np.float32) * 2, np.ones((4,), np.float32)), {"axis": 1})

    def test_only_spatial_and_custom_ops_are_delegated(self):
        delegated = {"Conv2D", "BatchNorm", "MaxPool2D", "GridSample2D", "MSDeformableAttention"}
        for op in delegated:
            node = Node("test", op, ("x",), ("y",), {})
            self.assertFalse(emit_scalar_op(ScalarBuilder(), node, ("%x",), (TensorSpec((1,), "float32"),),
                                            "%out", TensorSpec((1,), "float32"), Value("%tid", "i32")))
        self.assertEqual(SUPPORTED_OPS - delegated, {
            "MatMul", "MatMulBiasGELU", "Add", "AddReLU", "Sub", "Mul", "Div",
            "Greater", "Less", "ReLU", "GELU", "LayerNorm", "Softmax", "Identity",
            "Cast", "Sigmoid", "Log", "Sin", "Cos", "Where", "Reshape", "Transpose",
            "Concat", "Expand", "Slice", "ReduceSum", "ReduceMean", "Clamp",
        })


@unittest.skipUnless(os.environ.get("GAC_VORTEX_LLVM_BIN") and os.name == "posix",
                     "set GAC_VORTEX_LLVM_BIN to enable native MLIR execution checks")
class NativeVortexScalarOpsTests(VortexScalarOpsTests):
    """Compile and run every oracle case through the real MLIR/LLVM pipeline."""

    def check(self, op, arrays, attrs=None, *, rtol=2e-5, atol=2e-6):
        from gac.backends.vortex_graph import compile_vortex_graph
        from gac.backends.vortex_graph_validation import NativeVortexGraphSession

        # The public Runtime ABI requires contiguous buffers; sliced NumPy test
        # inputs are materialized without changing their logical values.
        arrays = tuple(np.array(array, copy=True, order="C") for array in arrays)
        names = tuple(f"x{i}" for i in range(len(arrays)))
        graph = Graph("scalar-op-native", dict(zip(names, (
            TensorSpec(array.shape, array.dtype) for array in arrays
        ), strict=True)), ("y",), [Node("test", op, names, ("y",), attrs or {})])
        program = compile_vortex_graph(graph, optimize=False)
        with NativeVortexGraphSession(program, llvm_bin=os.environ["GAC_VORTEX_LLVM_BIN"]) as session:
            result = session.run(dict(zip(names, arrays, strict=True)))["y"]
        expected = evaluate_op(op, arrays, attrs)[0]
        if result.dtype.kind in "biu":
            np.testing.assert_array_equal(result, expected)
        else:
            np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        return result

    def test_native_preserves_scalar_constant_outputs(self):
        from gac.backends.vortex_graph import compile_vortex_graph
        from gac.backends.vortex_graph_validation import NativeVortexGraphSession

        graph = Graph("scalar-output", {"x": TensorSpec((), "float32")}, ("y", "c"),
                      [Node("identity", "Identity", ("x",), ("y",))],
                      constants={"c": np.array(3.25, dtype=np.float32)})
        program = compile_vortex_graph(graph)
        with NativeVortexGraphSession(program, llvm_bin=os.environ["GAC_VORTEX_LLVM_BIN"]) as session:
            result = session.run({"x": np.array(1.25, dtype=np.float32)})
        self.assertEqual(result["y"].shape, ())
        self.assertEqual(result["c"].shape, ())
        self.assertEqual(result["y"].item(), 1.25)
        self.assertEqual(result["c"].item(), 3.25)


class VortexGraphInputValidationTests(unittest.TestCase):
    def test_overflowing_shape_is_rejected_before_inference_or_cloning(self):
        from gac.backends.vortex_graph import compile_vortex_graph

        graph = Graph("overflow", {"x": TensorSpec((2**32, 2**32), "bool")}, ("y",),
                      [Node("identity", "Identity", ("x",), ("y",))])
        with patch("gac.backends.vortex_graph.ShapeInferencePass.run") as infer:
            with self.assertRaisesRegex(BackendError, "RV32"):
                compile_vortex_graph(graph)
            infer.assert_not_called()

    def test_boolean_arithmetic_fails_before_constant_folding(self):
        from gac.backends.vortex_graph import compile_vortex_graph

        arrays = {"x": np.array([True, False]), "z": np.array([True, False])}
        graph = Graph("boolean-add", {}, ("y",), [Node("add", "Add", ("x", "z"), ("y",))],
                      constants=arrays)
        with self.assertRaisesRegex(BackendError, "bool arithmetic"):
            compile_vortex_graph(graph)
        identity = Graph("boolean-copy", {"x": TensorSpec((2,), "bool")}, ("y",),
                         [Node("identity", "Identity", ("x",), ("y",))])
        self.assertEqual(len(compile_vortex_graph(identity).dispatches), 1)

    def test_scalar_artifact_tensors_keep_rank_and_raw_bytes(self):
        from gac.backends.vortex_graph import compile_vortex_graph, write_vortex_graph_artifact

        graph = Graph("scalar-artifact", {"x": TensorSpec((), "float32")}, ("y", "c"),
                      [Node("identity", "Identity", ("x",), ("y",))],
                      constants={"c": np.array(3.25, dtype=np.float32)})
        with tempfile.TemporaryDirectory() as directory:
            report = write_vortex_graph_artifact(directory, compile_vortex_graph(graph),
                                                 inputs={"x": np.array(1.25, dtype=np.float32)})
            for role, value in (("inputs", 1.25), ("constants", 3.25)):
                record = report["tensor_files"][role][0]
                self.assertEqual(record["shape"], [])
                data = np.fromfile(Path(directory) / record["file"], dtype="<f4").reshape(record["shape"])
                self.assertEqual(data.shape, ())
                self.assertEqual(data.item(), value)


if __name__ == "__main__":
    unittest.main()
