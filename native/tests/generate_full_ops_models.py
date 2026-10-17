#!/usr/bin/env python3
"""ONNX fixtures for native multi-result import and tensor legalization.

This file only generates test inputs and independent ONNX numerical references.
All import, shape inference, graph rewrites and lowering under test are C++ MLIR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cases = []

    def info(name, dtype, shape):
        return helper.make_tensor_value_info(name, dtype, shape)

    def save(name, nodes, inputs, outputs, constants=None, *, opset=17,
             expect_rejection=False, stage="llvm", diagnostic=None):
        """outputs maps public name to (ONNX element type, shape)."""
        graph = helper.make_graph(
            nodes, name,
            [info(key, helper.np_dtype_to_tensor_dtype(value.dtype),
                  list(value.shape)) for key, value in inputs.items()],
            [info(key, dtype, shape) for key, (dtype, shape) in outputs.items()],
            [numpy_helper.from_array(value, name=key)
             for key, value in (constants or {}).items()],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                                  ir_version=9, producer_name="gac-native-full-ops-tests")
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        if not expect_rejection:
            onnx.checker.check_model(model)
            results = ReferenceEvaluator(model).run(None, inputs)
            np.savez(folder / "expected.npz", **dict(zip(outputs, results)))
        cases.append({"name": name,
                      "valid": not expect_rejection or stage == "llvm",
                      "reject_device": expect_rejection and stage == "llvm",
                      "expect_rejection": expect_rejection, "stage": stage,
                      "inputs": list(inputs), "outputs": list(outputs),
                      "expected_diagnostic": diagnostic, "expect_fusion": False,
                      "unsupported": False})
        if name.startswith("split_") and not expect_rejection:
            cases[-1]["expect_optimized_absent"] = ['"gac.split"(']
        if name == "nonzero_runtime_dynamic":
            cases[-1]["expect_optimized_contains"] = ['"gac.nonzero"(', "tensor<2x?xi64>"]

    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    save("tile_runtime", [helper.make_node("Tile", ["X", "repeats"], ["Y"])],
         {"X": x[:, :3].copy()}, {"Y": (f32, [4, 9])},
         {"repeats": np.array([2, 3], dtype=np.int64)})
    save("split_explicit_negative_axis", [helper.make_node(
        "Split", ["X", "sizes"], ["left", "middle", "right"], axis=-1)],
         {"X": x}, {"left": (f32, [2, 1]), "middle": (f32, [2, 3]),
                    "right": (f32, [2, 2])},
         {"sizes": np.array([1, 3, 2], dtype=np.int64)})
    save("split_equal_opset17", [helper.make_node(
        "Split", ["X"], ["first", "second", "third"], axis=1)],
         {"X": x}, {key: (f32, [2, 2]) for key in ("first", "second", "third")})
    save("split_uneven_opset18", [helper.make_node(
        "Split", ["X"], ["first", "second", "third"], axis=0, num_outputs=3)],
         {"X": np.arange(8, dtype=np.float32)},
         {"first": (f32, [3]), "second": (f32, [3]), "third": (f32, [2])},
         opset=18)
    save("split_constant_multiresult", [helper.make_node(
        "Split", ["X", "sizes"], ["first", "second"], axis=0)], {},
         {"first": (i64, [1, 3]), "second": (i64, [2, 3])},
         {"X": np.arange(9, dtype=np.int64).reshape(3, 3),
          "sizes": np.array([1, 2], dtype=np.int64)})
    # Runtime sizes cannot be legalized just because output value_info happens
    # to contain the shape observed for this sample input.
    save("split_runtime_sizes", [helper.make_node(
        "Split", ["X", "sizes"], ["first", "second"], axis=1)],
         {"X": x, "sizes": np.array([2, 4], dtype=np.int64)},
         {"first": (f32, [2, 2]), "second": (f32, [2, 4])},
         expect_rejection=True, diagnostic="single-output GAC")
    save("split_invalid_sum", [helper.make_node(
        "Split", ["X", "sizes"], ["first", "second"], axis=1)],
         {"X": x}, {"first": (f32, [2, 2]), "second": (f32, [2, 3])},
         {"sizes": np.array([2, 3], dtype=np.int64)},
         expect_rejection=True, stage="optimized", diagnostic="sum")
    save("scatter_slice_negative_indices", [helper.make_node(
        "ScatterND", ["X", "indices", "updates"], ["Y"])],
         {"X": np.arange(12, dtype=np.float32).reshape(3, 2, 2),
          "updates": np.array([[[91, 92], [93, 94]], [[71, 72], [73, 74]]],
                              dtype=np.float32)}, {"Y": (f32, [3, 2, 2])},
         {"indices": np.array([[-1], [0]], dtype=np.int64)})
    save("scatter_element_negative_indices", [helper.make_node(
        "ScatterND", ["X", "indices", "updates"], ["Y"])],
         {"X": x, "updates": np.array([50, 60], dtype=np.float32)},
         {"Y": (f32, [2, 6])},
         {"indices": np.array([[0, -1], [-1, 0]], dtype=np.int64)})
    save("scatter_duplicate_indices", [helper.make_node(
        "ScatterND", ["X", "indices", "updates"], ["Y"])],
         {"X": x, "updates": np.ones((2, 6), dtype=np.float32)},
         {"Y": (f32, [2, 6])},
         {"indices": np.array([[0], [0]], dtype=np.int64)},
         expect_rejection=True, diagnostic="duplicate")
    save("scatter_invalid_update_shape", [helper.make_node(
        "ScatterND", ["X", "indices", "updates"], ["Y"])],
         {"X": x, "updates": np.ones((1, 5), dtype=np.float32)},
         {"Y": (f32, [2, 6])}, {"indices": np.array([[0]], dtype=np.int64)},
         expect_rejection=True, stage="optimized", diagnostic="updates")
    save("scatter_unsigned_indices", [helper.make_node(
        "ScatterND", ["X", "indices", "updates"], ["Y"])],
         {"X": x, "updates": np.ones((1, 6), dtype=np.float32)},
         {"Y": (f32, [2, 6])}, {"indices": np.array([[0]], dtype=np.uint64)},
         expect_rejection=True, stage="optimized", diagnostic="i64 elements")
    save("nonzero_constant", [helper.make_node("NonZero", ["X"], ["Y"])],
         {}, {"Y": (i64, [2, 3])},
         {"X": np.array([[0., -2., 0.], [3., 0., 4.]], dtype=np.float32)})
    save("nonzero_runtime_dynamic", [helper.make_node("NonZero", ["X"], ["Y"])],
         {"X": np.array([[0., -2., 0.], [3., 0., 4.]], dtype=np.float32)},
         {"Y": (i64, [2, None])}, expect_rejection=True, diagnostic="positive static")
    save("tile_invalid_repeat_rank", [helper.make_node("Tile", ["X", "repeats"], ["Y"])],
         {"X": x}, {"Y": (f32, [2, 6])},
         {"repeats": np.array([2], dtype=np.int64)},
         expect_rejection=True, stage="optimized", diagnostic="rank")

    manifest = {"schema_version": 1, "cases": cases}
    (output / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Generated {len(manifest['cases'])} native tensor-operation fixtures in {args.output}")
