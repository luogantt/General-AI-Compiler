#!/usr/bin/env python3
"""Fixtures for the C++ MLIR constant evaluator; Python is test-only.

The first fixture catches silently rounding f64 constants to f32. The second
models BEV camera visibility selection: a constant geometry program creates
indices, while Gather still reads a genuinely dynamic activation tensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as n
from onnx.reference import ReferenceEvaluator


def generate(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    cases = []

    def save(name, nodes, constants, inputs, output_types, expected_shapes):
        graph = h.make_graph(
            nodes, name,
            [h.make_tensor_value_info(key, h.np_dtype_to_tensor_dtype(value.dtype),
                                      list(value.shape)) for key, value in inputs.items()],
            [h.make_tensor_value_info(key, output_types[key], shape)
             for key, shape in expected_shapes.items()],
            [n.from_array(value, key) for key, value in constants.items()],
        )
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9)
        onnx.checker.check_model(model)
        path = directory / name
        path.mkdir(exist_ok=True)
        onnx.save(model, path / "model.onnx")
        np.savez(path / "inputs.npz", **inputs)
        expected = ReferenceEvaluator(model).run(None, inputs)
        np.savez(path / "expected.npz", **dict(zip(expected_shapes, expected)))
        cases.append({"name": name, "valid": True, "expect_fusion": False,
                      "inputs": list(inputs), "outputs": list(expected_shapes),
                      "unsupported": False})

    # 2**24 + 1 cannot be represented in FP32. Subtracting before an explicit
    # cast must produce 1, not 0. Batched 2x2 MatMul also evaluates in FP64.
    save("constant_fp64_geometry", [
        h.make_node("Sub", ["A", "B"], ["delta"]),
        h.make_node("MatMul", ["delta", "W"], ["projected"]),
        h.make_node("Div", ["projected", "scale"], ["normalized"]),
        h.make_node("Clip", ["normalized", "min", "max"], ["clipped"]),
        h.make_node("Cast", ["clipped"], ["Y"], to=T.FLOAT),
    ], {
        "A": np.array([[[2**24 + 1., 3.], [5., 7.]]], np.float64),
        "B": np.array([[[2**24, 1.], [2., 3.]]], np.float64),
        "W": np.array([[1., -2.], [0.5, 3.]], np.float64),
        "scale": np.array(2., np.float64),
        "min": np.array(-1., np.float64), "max": np.array(8., np.float64),
    }, {}, {"Y": T.FLOAT}, {"Y": [1, 2, 2]})

    save("constant_visibility_indices", [
        h.make_node("MatMul", ["points", "projection"], ["projected"]),
        h.make_node("Greater", ["projected", "threshold"], ["positive"]),
        h.make_node("Cast", ["positive"], ["mask"], to=T.INT64),
        h.make_node("ReduceSum", ["mask", "axes"], ["visible"], keepdims=0),
        h.make_node("NonZero", ["visible"], ["index_matrix"]),
        h.make_node("Transpose", ["index_matrix"], ["transposed"], perm=[1, 0]),
        h.make_node("Squeeze", ["transposed", "axes"], ["indices"]),
        h.make_node("Gather", ["X", "indices"], ["Y"], axis=0),
    ], {
        "points": np.array([[1., 1.], [-2., -1.], [3., -4.], [-2., 3.]], np.float64),
        "projection": np.array([[1., 0.], [0., 1.]], np.float64),
        "threshold": np.array(0., np.float64), "axes": np.array([1], np.int64),
    }, {"X": np.arange(12, dtype=np.float32).reshape(4, 3)},
        {"Y": T.FLOAT}, {"Y": [3, 3]})

    save("constant_scatter_tile_split", [
        h.make_node("ConstantOfShape", ["shape"], ["zeros"],
                    value=n.from_array(np.array([0.], np.float64))),
        h.make_node("ScatterND", ["zeros", "indices", "updates"], ["scattered"]),
        h.make_node("Split", ["scattered", "splits"], ["left", "right"], axis=0),
        h.make_node("Tile", ["right", "repeats"], ["tiled"]),
        h.make_node("Expand", ["left", "expanded_shape"], ["expanded"]),
        h.make_node("Concat", ["expanded", "tiled"], ["joined"], axis=0),
        h.make_node("Less", ["joined", "limit"], ["condition"]),
        h.make_node("Where", ["condition", "joined", "fallback"], ["selected"]),
        h.make_node("Cast", ["selected"], ["Y"], to=T.FLOAT),
    ], {
        "shape": np.array([3, 2], np.int64),
        "indices": np.array([[0], [-1]], np.int64),
        "updates": np.array([[1.25, 2.5], [3.75, 5.]], np.float64),
        "splits": np.array([1, 2], np.int64),
        "repeats": np.array([2, 1], np.int64),
        "expanded_shape": np.array([2, 2], np.int64),
        "limit": np.array(4., np.float64), "fallback": np.array(-1., np.float64),
    }, {}, {"Y": T.FLOAT}, {"Y": [6, 2]})

    # A layout-only FP64 chain verifies negative Slice steps and bit-preserving
    # Gather/Transpose/Concat independently from floating arithmetic.
    save("constant_fp64_layout", [
        h.make_node("Transpose", ["A"], ["transposed"], perm=[1, 0]),
        h.make_node("Slice", ["transposed", "starts", "ends", "axes", "steps"], ["sliced"]),
        h.make_node("Gather", ["sliced", "indices"], ["gathered"], axis=0),
        h.make_node("Concat", ["gathered", "gathered"], ["joined"], axis=1),
        h.make_node("Cast", ["joined"], ["Y"], to=T.FLOAT),
    ], {
        "A": np.array([[1., -0., 3.], [4., 5., 6.]], np.float64),
        "starts": np.array([2], np.int64), "ends": np.array([-4], np.int64),
        "axes": np.array([0], np.int64), "steps": np.array([-1], np.int64),
        "indices": np.array([0, -1], np.int64),
    }, {}, {"Y": T.FLOAT}, {"Y": [2, 4]})

    save("constant_fp64_bool_cast", [
        h.make_node("Cast", ["A"], ["boolean"], to=T.BOOL),
        h.make_node("Cast", ["boolean"], ["Y"], to=T.FLOAT),
    ], {"A": np.array([-0., 0., np.nan, np.inf, -np.inf, 1.e-100, -3.], np.float64)},
        {}, {"Y": T.FLOAT}, {"Y": [7]})

    save("constant_empty_reduce", [
        h.make_node("ReduceSum", ["A", "axes"], ["sum"], keepdims=0),
        h.make_node("Cast", ["sum"], ["Y"], to=T.FLOAT),
    ], {"A": np.zeros((0, 2), np.float64), "axes": np.array([0], np.int64)},
        {}, {"Y": T.FLOAT}, {"Y": [2]})
    report = {"scope": "native constant geometry folding", "cases": cases}
    (directory / "cases.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
