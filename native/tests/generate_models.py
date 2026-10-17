#!/usr/bin/env python3
"""Generate small ONNX inputs and independent numerical references.

Python is used only for test fixtures/reference evaluation. The compiler under
test reads each .onnx directly in C++ and runs native MLIR passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def tensor(name: str, shape: list[int | str | None], dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(431)
    cases = []

    def save(name, nodes, inputs, outputs, constants=None, *, valid=True,
             custom_domain=None, expect_fusion=False, output_types=None,
             input_shapes=None):
        constants = constants or {}
        graph = helper.make_graph(
            nodes, name, [tensor(key, (input_shapes or {}).get(key, list(value.shape)),
                                helper.np_dtype_to_tensor_dtype(value.dtype))
                          for key, value in inputs.items()],
            [tensor(key, shape, (output_types or {}).get(key, TensorProto.FLOAT))
             for key, shape in outputs.items()],
            [numpy_helper.from_array(value, name=key) for key, value in constants.items()],
        )
        opsets = [helper.make_opsetid("", 18)]
        if custom_domain:
            opsets.append(helper.make_opsetid(custom_domain, 1))
        model = helper.make_model(graph, opset_imports=opsets, ir_version=9,
                                  producer_name="gac-native-regression-fixtures")
        case_dir = output / name
        case_dir.mkdir(exist_ok=True)
        onnx.save(model, case_dir / "model.onnx")
        np.savez(case_dir / "inputs.npz", **inputs)
        if valid:
            onnx.checker.check_model(model)
            # ONNX 1.23's reference GridSample uses the opset-20 mode names.
            # The official converter normalizes only the reference graph; the
            # compiler fixture remains opset 18 with its original semantics.
            reference_model = (onnx.version_converter.convert_version(model, 20)
                               if any(node.op_type == "GridSample" for node in model.graph.node)
                               else model)
            result = ReferenceEvaluator(reference_model).run(None, inputs)
            np.savez(case_dir / "expected.npz", **dict(zip(outputs, result)))
        cases.append({"name": name, "valid": valid, "expect_fusion": expect_fusion,
                      "inputs": list(inputs), "outputs": list(outputs),
                      "unsupported": custom_domain is not None})

    x = rng.normal(size=(2, 3)).astype(np.float32)
    weights = rng.normal(size=(3, 4)).astype(np.float32)
    bias = rng.normal(size=(4,)).astype(np.float32)
    matmul = helper.make_node("MatMul", ["X", "W"], ["M"], name="matmul")
    add_bias = helper.make_node("Add", ["M", "B"], ["Z"], name="bias")
    relu = helper.make_node("Relu", ["Z"], ["Y"], name="activation")
    save("matmul_bias_relu", [matmul, add_bias, relu], {"X": x}, {"Y": [2, 4]},
         {"W": weights, "B": bias}, expect_fusion=True)
    # M is externally observable; fusion must preserve this additional result.
    save("multi_consumer", [matmul, add_bias, relu], {"X": x},
         {"Y": [2, 4], "M": [2, 4]}, {"W": weights, "B": bias})

    save("conv_bias_relu", [
        helper.make_node("Conv", ["X", "W", "B"], ["Z"], name="conv",
                         pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["Z"], ["Y"], name="activation"),
    ], {"X": rng.normal(size=(1, 2, 4, 5)).astype(np.float32)},
         {"Y": [1, 3, 4, 5]},
         {"W": rng.normal(size=(3, 2, 3, 3)).astype(np.float32),
          "B": rng.normal(size=(3,)).astype(np.float32)}, expect_fusion=True)

    save("residual_add_relu", [
        helper.make_node("Add", ["X", "R"], ["Z"], name="residual"),
        helper.make_node("Relu", ["Z"], ["Y"], name="activation"),
    ], {"X": rng.normal(size=(2, 4)).astype(np.float32),
        "R": rng.normal(size=(2, 4)).astype(np.float32)}, {"Y": [2, 4]},
         expect_fusion=True)

    save("softmax", [helper.make_node("Softmax", ["X"], ["Y"], name="softmax", axis=-1)],
         {"X": np.array([[1000, 1001, 999, 1002], [-1000, -1000, -999, -1001]],
                        dtype=np.float32)}, {"Y": [2, 4]})
    # No intermediate value_info is provided. Native shape inference must fold
    # the SSA shape program before it can determine Reshape's output dimensions.
    save("shape_chain", [
        helper.make_node("Shape", ["X"], ["shape"], name="shape_of_input"),
        helper.make_node("Gather", ["shape", "col_index"], ["cols"], name="get_cols", axis=0),
        helper.make_node("Gather", ["shape", "row_index"], ["rows"], name="get_rows", axis=0),
        helper.make_node("Unsqueeze", ["cols", "axis_zero"], ["col_vector"], name="cols_to_vector"),
        helper.make_node("Unsqueeze", ["rows", "axis_zero"], ["row_vector"], name="rows_to_vector"),
        helper.make_node("Concat", ["col_vector", "row_vector"], ["target_shape"], name="shape_concat", axis=0),
        helper.make_node("Cast", ["target_shape"], ["target_i32"], name="shape_cast_i32", to=TensorProto.INT32),
        helper.make_node("Cast", ["target_i32"], ["target_i64"], name="shape_cast_i64", to=TensorProto.INT64),
        helper.make_node("Reshape", ["X", "target_i64"], ["Y"], name="reshape_from_shape_program"),
    ], {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}, {"Y": [3, 2]},
         {"col_index": np.array(1, dtype=np.int64),
          "row_index": np.array(0, dtype=np.int64),
          "axis_zero": np.array([0], dtype=np.int64)})
    cases[-1]["expect_shape_fold"] = True
    save("invalid_broadcast", [helper.make_node("Add", ["X", "B"], ["Y"], name="bad_add")],
         {"X": np.ones((2, 3), dtype=np.float32)}, {"Y": [2, 3]},
         {"B": np.ones((4,), dtype=np.float32)}, valid=False)
    save("unsupported", [helper.make_node("Mystery", ["X"], ["Y"], name="unknown",
                                          domain="org.gac.test")],
         {"X": np.ones((2, 3), dtype=np.float32)}, {"Y": [2, 3]},
         valid=False, custom_domain="org.gac.test")

    save("matmul_rank_and_broadcast", [
        helper.make_node("MatMul", ["X", "W"], ["batched"], name="broadcast_matmul"),
        helper.make_node("MatMul", ["U", "V"], ["dot"], name="vector_dot"),
    ], {"X": rng.normal(size=(2, 1, 3, 4)).astype(np.float32),
        "U": np.array([1, -2, 3, 4], dtype=np.float32)},
         {"batched": [2, 5, 3, 2], "dot": []},
         {"W": rng.normal(size=(1, 5, 4, 2)).astype(np.float32),
          "V": np.array([2, 5, -1, 2], dtype=np.float32)})

    save("grouped_dilated_conv", [
        helper.make_node("Conv", ["X", "W", "B"], ["Y"], name="grouped_conv",
                         group=2, strides=[2, 2], dilations=[2, 1], pads=[1, 0, 0, 1]),
    ], {"X": rng.normal(size=(1, 4, 5, 6)).astype(np.float32)}, {"Y": [1, 6, 2, 3]},
         {"W": rng.normal(size=(6, 2, 2, 3)).astype(np.float32),
          "B": rng.normal(size=(6,)).astype(np.float32)})

    save("layer_norm_axes_and_bias", [
        helper.make_node("LayerNormalization", ["X", "S", "B"], ["Y"],
                         name="norm", axis=1, epsilon=1e-5, stash_type=1),
    ], {"X": rng.normal(size=(2, 3, 4)).astype(np.float32)}, {"Y": [2, 3, 4]},
         {"S": rng.normal(size=(3, 4)).astype(np.float32),
          "B": rng.normal(size=(4,)).astype(np.float32)})

    save("reductions_axes", [
        helper.make_node("ReduceSum", ["X", "sum_axes"], ["sum"], name="sum_noncontiguous", keepdims=0),
        helper.make_node("ReduceMean", ["X", "mean_axes"], ["mean"], name="mean_negative_axis", keepdims=1),
    ], {"X": rng.normal(size=(2, 3, 4)).astype(np.float32)},
         {"sum": [3], "mean": [2, 3, 1]},
         {"sum_axes": np.array([0, 2], dtype=np.int64),
          "mean_axes": np.array([-1], dtype=np.int64)})

    save("layout_concat_negative_slice", [
        helper.make_node("Transpose", ["X"], ["T"], name="permute", perm=[1, 0, 2]),
        helper.make_node("Concat", ["T", "T"], ["C"], name="concat_middle_axis", axis=1),
        helper.make_node("Slice", ["C", "starts", "ends", "axes", "steps"], ["Y"], name="reverse_slice"),
    ], {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}, {"Y": [3, 3, 2]},
         {"starts": np.array([3, -1], dtype=np.int64),
          "ends": np.array([-5, 0], dtype=np.int64),
          "axes": np.array([2, 1], dtype=np.int64),
          "steps": np.array([-2, -1], dtype=np.int64)})

    grid = np.array([[[[-1, -1], [-0.5, 0], [0, 0], [1, 1], [1.5, 0]],
                      [[-1.2, 0.5], [0.25, -1.2], [0.5, 0.5], [2, 2], [-2, -2]]]], dtype=np.float32)
    save("grid_sample_bilinear_boundaries", [
        helper.make_node("GridSample", ["X", "G"], ["unaligned"], name="grid_half_pixel",
                         mode="bilinear", padding_mode="zeros", align_corners=0),
        helper.make_node("GridSample", ["X", "G"], ["aligned"], name="grid_corners",
                         mode="bilinear", padding_mode="zeros", align_corners=1),
    ], {"X": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4), "G": grid},
         {"unaligned": [1, 2, 2, 5], "aligned": [1, 2, 2, 5]})

    # If MatMul/Add/ReLU are fused without intermediate half rounding, this
    # changes from exactly 0 to about 0.0003. Check both IR and machine results.
    save("fp16_rounding_no_fusion", [
        helper.make_node("MatMul", ["X", "W"], ["M"], name="half_matmul"),
        helper.make_node("Add", ["M", "B"], ["A"], name="half_bias"),
        helper.make_node("Relu", ["A"], ["Y"], name="half_relu"),
    ], {"X": np.array([[1, 1]], dtype=np.float16)}, {"Y": [1, 1]},
         {"W": np.array([[1], [0.0003]], dtype=np.float16),
          "B": np.array([-1], dtype=np.float16)}, output_types={"Y": TensorProto.FLOAT16})
    cases[-1]["expect_no_fusion"] = True

    save("cast_bool_values", [
        helper.make_node("Cast", ["X"], ["Y"], name="nonzero_to_bool", to=TensorProto.BOOL),
    ], {"X": np.array([0, -0.0, -2, 1.0, np.nan, np.inf, -np.inf, 1e-40], dtype=np.float32)},
         {"Y": [8]}, output_types={"Y": TensorProto.BOOL})

    save("identity_input_output_alias", [
        helper.make_node("Identity", ["X"], ["Y"], name="identity"),
    ], {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}, {"X": [2, 3], "Y": [2, 3]})

    save("constant_only_output", [], {}, {"Y": [2, 2]},
         {"Y": np.array([[1, -2], [3, 4]], dtype=np.float32)})

    save("runtime_shape_rejected", [
        helper.make_node("Reshape", ["X", "S"], ["Y"], name="runtime_shape"),
    ], {"X": np.arange(6, dtype=np.float32).reshape(2, 3),
        "S": np.array([3, 2], dtype=np.int64)}, {"Y": [3, 2]})
    cases[-1]["reject_device"] = True

    save("constant_shape_zero_division", [
        helper.make_node("Div", ["six", "zero"], ["S"], name="invalid_shape_division"),
        helper.make_node("Reshape", ["X", "S"], ["Y"], name="invalid_shape"),
    ], {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}, {"Y": [2, 3]},
         {"six": np.array([6], dtype=np.int64), "zero": np.array([0], dtype=np.int64)}, valid=False)

    # Distinct ONNX display names must not prevent upstream MLIR CSE. The two
    # Adds are identical SSA computations; the final public output stays "Y".
    save("duplicate_add_cse", [
        helper.make_node("Add", ["X", "B"], ["left"], name="left_branch_add"),
        helper.make_node("Add", ["X", "B"], ["right"], name="right_branch_add"),
        helper.make_node("Mul", ["left", "right"], ["Y"], name="multiply_branches"),
    ], {"X": np.array([[1, -2, 3], [4, 0, -1]], dtype=np.float32)}, {"Y": [2, 3]},
         {"B": np.array([0.5, 1, -2], dtype=np.float32)})
    cases[-1]["expect_cse"] = {"add": [2, 1]}

    # After internal display names are removed, unnamed constants use value_N
    # fallbacks. Those names must not steal a later public ONNX output name.
    save("public_output_fallback_name_collision", [
        helper.make_node("Add", ["X", "B"], ["value_2"], name="public_add"),
        helper.make_node("Mul", ["X", "C"], ["value_3"], name="public_mul"),
    ], {"X": np.arange(6, dtype=np.float32).reshape(2, 3)},
         {"value_2": [2, 3], "value_3": [2, 3]},
         {"B": np.array([1, -1, 0.5], dtype=np.float32),
          "C": np.array([2, -3, 4], dtype=np.float32)})

    # ONNX -1 is an inference marker, not MLIR's dynamic-dimension sentinel.
    # Test flattening and zero-copy axes on either side of the inferred axis.
    save("reshape_inferred_and_zero_copy", [
        helper.make_node("Reshape", ["X", "flatten_shape"], ["flat"], name="flatten_inferred"),
        helper.make_node("Reshape", ["X", "leading_zero_shape"], ["leading"], name="copy_leading_axis"),
        helper.make_node("Reshape", ["X", "trailing_zero_shape"], ["trailing"], name="copy_trailing_axis"),
    ], {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)},
         {"flat": [24], "leading": [2, 12], "trailing": [8, 3]},
         {"flatten_shape": np.array([-1], dtype=np.int64),
          "leading_zero_shape": np.array([0, -1], dtype=np.int64),
          "trailing_zero_shape": np.array([-1, 0], dtype=np.int64)})

    # An unknown copied axis and its inferred partner must remain valid '?'
    # tensor extents; the static-only device target must still reject them.
    save("reshape_dynamic_zero_copy", [
        helper.make_node("Reshape", ["X", "S"], ["Y"], name="dynamic_zero_copy"),
    ], {"X": np.arange(24, dtype=np.float32).reshape(2, 3, 4)},
         {"Y": ["batch", None]}, {"S": np.array([0, -1], dtype=np.int64)},
         input_shapes={"X": ["batch", 3, 4]})
    cases[-1]["reject_device"] = True
    cases[-1]["expect_optimized_contains"] = ["tensor<?x?xf32>"]

    # Zero-element tensors are valid graph IR even though the current device
    # ABI requires positive extents. A positive known divisor infers zero, and
    # allowzero=1 preserves literal zero instead of copying an input dimension.
    save("reshape_zero_extent_semantics", [
        helper.make_node("Reshape", ["X", "inferred_shape"], ["inferred"], name="infer_zero_extent"),
        helper.make_node("Reshape", ["X", "literal_shape"], ["literal"], name="literal_zero", allowzero=1),
    ], {"X": np.empty((0, 3), dtype=np.float32)},
         {"inferred": [0, 3], "literal": [3, 0]},
         {"inferred_shape": np.array([-1, 3], dtype=np.int64),
          "literal_shape": np.array([3, 0], dtype=np.int64)})
    cases[-1]["reject_device"] = True
    cases[-1]["expect_optimized_contains"] = ["tensor<0x3xf32>", "tensor<3x0xf32>"]

    invalid_reshapes = [
        ("reshape_multiple_inferred", [-1, -1], 0, (2, 3, 4), "multiple inferred dimensions"),
        ("reshape_nonintegral_inference", [5, -1], 0, (2, 3, 4), "unique integral dimension"),
        ("reshape_literal_zero_and_inferred", [0, -1], 1, (0, 3), "cannot combine zero and -1"),
        ("reshape_copied_zero_ambiguous", [0, -1], 0, (0, 3), "unique dimension from a zero product"),
        ("reshape_invalid_negative", [-2], 0, (2, 3, 4), "invalid Reshape dimension"),
        ("reshape_extent_overflow", [np.iinfo(np.int64).max, 2, -1], 0, (1,), "dimension product overflows i64"),
    ]
    for name, requested, allowzero, input_shape, diagnostic in invalid_reshapes:
        save(name, [helper.make_node("Reshape", ["X", "S"], ["Y"], name=name, allowzero=allowzero)],
             {"X": np.zeros(input_shape, dtype=np.float32)}, {"Y": list(input_shape)},
             {"S": np.array(requested, dtype=np.int64)}, valid=False)
        cases[-1]["expected_diagnostic"] = diagnostic
    manifest = {"seed": 431, "reference": "onnx.reference.ReferenceEvaluator",
                "onnx_version": onnx.__version__, "cases": cases}
    (output / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(f"Generated {len(result['cases'])} ONNX cases in {args.output}")
