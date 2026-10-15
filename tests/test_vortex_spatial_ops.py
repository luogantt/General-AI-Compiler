"""Numerical and memory-safety checks for Vortex scalar spatial lowering.

The interpreter executes the emitter's control flow and rejects every invalid
memory access. This tests indexing/edge behavior against the independent graph
oracle without requiring a Vortex board or MLIR binary on the test host.
"""

import math
import os
import unittest

import numpy as np

from gac.backends.vortex_scalar_ir import ScalarBuilder, Value
from gac.backends.vortex_spatial_ops import emit_spatial_op
from gac.backends.vortex_graph import compile_vortex_graph
from gac.backends.vortex_graph_validation import NativeVortexGraphSession
from gac.ir import Graph, Node, TensorSpec
from gac.ops import evaluate_op, infer_node


class ScalarInterpreter:
    def __init__(self, buffers):
        self.buffers = {name: value.reshape(-1) for name, value in buffers.items()}

    def const(self, value, type="i32"):
        return np.float32(value) if type == "f32" else int(value)

    def binary(self, op, a, c):
        if op == "and":
            return bool(a) and bool(c)
        if op == "or":
            return bool(a) or bool(c)
        operation = {
            "add": lambda: a + c, "sub": lambda: a - c,
            "mul": lambda: a * c, "div": lambda: a / c if isinstance(a, np.floating) else int(a / c),
            "rem": lambda: a - int(a / c) * c,
        }[op]
        value = operation()
        return np.float32(value) if isinstance(a, np.floating) else value

    def add(self, a, c): return self.binary("add", a, c)
    def sub(self, a, c): return self.binary("sub", a, c)
    def mul(self, a, c): return self.binary("mul", a, c)
    def div(self, a, c): return self.binary("div", a, c)
    def rem(self, a, c): return self.binary("rem", a, c)
    def cmp(self, op, a, c):
        return {"ge": a >= c, "gt": a > c, "le": a <= c, "lt": a < c, "eq": a == c, "ne": a != c}[op]
    def select(self, cond, a, c): return a if cond else c
    def cast(self, value, type): return np.float32(value) if type == "f32" else int(value)
    def math(self, name, value): return np.float32({"floor": np.floor, "sqrt": np.sqrt}[name](value))
    def if_value(self, cond, then, otherwise): return then() if cond else otherwise()

    def f32(self, ptr, index, spec):
        assert 0 <= index < len(self.buffers[ptr]), (ptr, index)
        return np.float32(self.buffers[ptr][index])

    def store(self, value, ptr, index, dtype):
        assert 0 <= index < len(self.buffers[ptr]), (ptr, index)
        self.buffers[ptr][index] = value

    def coords(self, index, shape):
        result = []
        for axis in range(len(shape)):
            result.append((int(index) // math.prod(shape[axis + 1:])) % shape[axis])
        return tuple(result)

    def linear(self, coords, shape):
        return sum(coord * math.prod(shape[axis + 1:]) for axis, coord in enumerate(coords))

    def loop(self, count, initial, callback):
        values = initial
        for index in range(count):
            values = callback(index, values)
        return values


class VortexSpatialOpsTests(unittest.TestCase):
    def assert_kernel(self, op, arrays, attrs=None, *, tolerance=2e-5):
        ptrs = tuple(f"p{i}" for i in range(len(arrays)))
        node = Node("test", op, ptrs, ("out",), attrs or {})
        specs = tuple(TensorSpec(array.shape, array.dtype) for array in arrays)
        out_spec = infer_node(node, dict(zip(ptrs, specs)))["out"]
        output = np.empty(out_spec.shape, dtype=out_spec.dtype)
        interpreter = ScalarInterpreter(dict(zip(ptrs, arrays)) | {"out": output})
        for tid in range(output.size):
            self.assertTrue(emit_spatial_op(interpreter, node, ptrs, specs, "out", out_spec, tid))
        expected, = evaluate_op(op, arrays, attrs)
        np.testing.assert_allclose(output, expected, atol=tolerance, rtol=tolerance)
        builder = ScalarBuilder()
        self.assertTrue(emit_spatial_op(builder, node, ptrs, specs, "%out", out_spec, Value("%tid", "i32")))
        self.assertIn("llvm.store", "\n".join(builder.lines))
        return output

    def test_grouped_dilated_conv_with_asymmetric_padding(self):
        rng = np.random.default_rng(11)
        for dtype in (np.float32, np.float16):
            arrays = [rng.normal(size=s).astype(dtype) for s in ((2, 4, 5, 6), (6, 2, 2, 3), (6,))]
            self.assert_kernel("Conv2D", arrays, {"groups": 2, "strides": [2, 1], "dilations": [2, 1], "pads": [1, 2, 2, 0]}, tolerance=2e-3 if dtype == np.float16 else 2e-5)
        self.assert_kernel("Conv2D", [np.ones((1, 2, 2, 2), np.float32), np.ones((2, 1, 1, 1), np.float32)], {"groups": 2})

    def test_batchnorm_and_pool(self):
        rng = np.random.default_rng(12)
        self.assert_kernel("BatchNorm", [rng.normal(size=(2, 3, 2, 3)).astype(np.float32), np.array([1, 2, -1], np.float32), np.array([2, -1, 0], np.float32), np.array([1, 1, 2], np.float32), np.array([0.1, 2, 3], np.float32)])
        self.assert_kernel("MaxPool2D", [rng.normal(size=(2, 3, 4, 5)).astype(np.float32)], {"kernel_shape": [2, 3], "strides": [1, 2], "dilations": [2, 1], "pads": [1, 2, 0, 1]})

    def test_grid_boundaries_and_nearest_even_ties(self):
        value = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
        # align_corners=False maps these to -1, -.5, .5, 1.5, 2.5, 3.5, 4.
        coordinates = np.array([-1.25, -1, -.5, 0, .5, 1, 1.25, -1e20, 1e20], np.float32)
        grid = np.stack(np.meshgrid(coordinates, coordinates), axis=-1)[None]
        for align in (False, True):
            for mode in ("nearest", "bilinear"):
                self.assert_kernel("GridSample2D", [value, grid], {"align_corners": align, "mode": mode})
        one = self.assert_kernel("GridSample2D", [np.ones((1, 1, 1, 1), np.float32), np.zeros((1, 1, 1, 2), np.float32)])
        self.assertEqual(one.item(), 1)

    def test_msda_multiple_levels_heads_and_padding(self):
        rng = np.random.default_rng(13)
        value = rng.normal(size=(2, 8, 2, 3)).astype(np.float32)
        locations = rng.uniform(-.4, 1.4, size=(2, 3, 2, 2, 3, 2)).astype(np.float32)
        weights = rng.uniform(size=(2, 3, 2, 2, 3)).astype(np.float32)
        self.assert_kernel("MSDeformableAttention", [value, locations, weights], {"spatial_shapes": [[2, 3], [1, 2]]})


@unittest.skipUnless(os.environ.get("GAC_VORTEX_LLVM_BIN"), "set GAC_VORTEX_LLVM_BIN for native MLIR numerical validation")
class VortexSpatialNativeTests(unittest.TestCase):
    """Lower real MLIR through LLVM, then execute the generated host code.

    This validates compiler transformations and kernel numerics; Vortex SIMT,
    CSR access and device memory transfers require separate simulator tests.
    """

    def assert_native(self, op, arrays, attrs=None, *, tolerance=2e-5):
        names = tuple(f"x{i}" for i in range(len(arrays)))
        graph = Graph(
            name=f"native_{op}",
            inputs={name: TensorSpec(array.shape, array.dtype) for name, array in zip(names, arrays)},
            outputs=("out",),
            nodes=[Node("spatial", op, names, ("out",), attrs or {})],
        )
        program = compile_vortex_graph(graph, optimize=False)
        with NativeVortexGraphSession(program, llvm_bin=os.environ["GAC_VORTEX_LLVM_BIN"]) as session:
            actual = session.run(dict(zip(names, arrays)))["out"]
        expected, = evaluate_op(op, arrays, attrs)
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        self.assertEqual(actual.dtype, expected.dtype)
        return actual

    def test_conv_grouped_dilated_fp32_and_fp16_storage(self):
        rng = np.random.default_rng(111)
        for dtype in (np.float32, np.float16):
            with self.subTest(dtype=dtype):
                arrays = [rng.normal(size=s).astype(dtype) for s in ((2, 4, 5, 6), (6, 2, 2, 3), (6,))]
                self.assert_native("Conv2D", arrays, {"groups": 2, "strides": [2, 1], "dilations": [2, 1], "pads": [1, 2, 2, 0]}, tolerance=2e-3 if dtype == np.float16 else 2e-5)
        self.assert_native("Conv2D", [np.ones((1, 2, 2, 2), np.float32), np.array([2, 3], np.float32).reshape(2, 1, 1, 1)], {"groups": 2})

    def test_batchnorm_channel_indexing_and_epsilon(self):
        rng = np.random.default_rng(112)
        parameters = [np.array([1, 2, -1], np.float32), np.array([2, -1, 0], np.float32), np.array([1, 1, 2], np.float32), np.array([0, 2, 3], np.float32)]
        for shape in ((2, 3), (2, 3, 2, 3)):
            with self.subTest(shape=shape):
                self.assert_native("BatchNorm", [rng.normal(size=shape).astype(np.float32), *parameters], {"epsilon": 1e-3})

    def test_maxpool_padding_dilation_negative_values(self):
        value = -np.arange(1, 121, dtype=np.float32).reshape(2, 3, 4, 5)
        self.assert_native("MaxPool2D", [value], {"kernel_shape": [2, 3], "strides": [1, 2], "dilations": [2, 1], "pads": [1, 2, 0, 1]})

    def test_grid_sample_bilinear_and_nearest_even(self):
        value = np.arange(1, 33, dtype=np.float32).reshape(1, 2, 4, 4)
        coordinates = np.array([-1.25, -1, -.5, 0, .5, 1, 1.25, -1e20, 1e20], np.float32)
        grid = np.stack(np.meshgrid(coordinates, coordinates), axis=-1)[None]
        for align in (False, True):
            for mode in ("bilinear", "nearest"):
                with self.subTest(align_corners=align, mode=mode):
                    self.assert_native("GridSample2D", [value, grid], {"align_corners": align, "mode": mode})
        self.assert_native("GridSample2D", [np.ones((1, 1, 1, 1), np.float32), np.zeros((1, 1, 1, 2), np.float32)], {"align_corners": True})

    def test_msda_multibatch_multilevel_multihead(self):
        rng = np.random.default_rng(113)
        value = rng.normal(size=(2, 8, 2, 3)).astype(np.float32)
        locations = rng.uniform(-.4, 1.4, size=(2, 3, 2, 2, 3, 2)).astype(np.float32)
        weights = rng.uniform(size=(2, 3, 2, 2, 3)).astype(np.float32)
        self.assert_native("MSDeformableAttention", [value, locations, weights], {"spatial_shapes": [[2, 3], [1, 2]]})


if __name__ == "__main__":
    unittest.main()
