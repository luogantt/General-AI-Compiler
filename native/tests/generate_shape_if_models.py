#!/usr/bin/env python3
"""Fixtures for the native, shape-guarded ONNX If legalization.

These models exercise the original ONNX semantics with ReferenceEvaluator. The
compiler must prove the graph pattern in C++, preserve a registered MLIR op,
and select its branch only after MLIR shape inference. No Python graph rewrite
is used by this fixture generator or by the compilation path.
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

    def save(name, shape, axis=-1, *, constant=False, nonzero=False,
             reverse_equal=False, scalar_indices=False, reshape=False,
             mutation=None, opset=17):
        x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        if constant:
            x = x.astype(np.int64) % 2 == 1
        dtype = helper.np_dtype_to_tensor_dtype(x.dtype)
        normalized_axis = axis % len(shape)
        squeezed_shape = list(shape)
        squeezed_shape.pop(normalized_axis)
        actual_shape = squeezed_shape if shape[normalized_axis] == 1 else list(shape)
        constants = []
        inputs = {} if constant else {"X": x}
        nodes = []
        source = "X"
        if constant:
            constants.append(numpy_helper.from_array(x, "X"))
        if reshape:
            constants.append(numpy_helper.from_array(np.array(shape, np.int64), "shape"))
            nodes.append(helper.make_node("Reshape", ["X", "shape"], ["view"]))
            source = "view"
        condition_source = source
        if mutation == "different_capture":
            inputs["Other"] = x + 10
            condition_source = "Other"
        index_value = axis if mutation != "axis_mismatch" else (axis + 1) % len(shape)
        index = np.array(index_value if scalar_indices else [index_value], np.int64)
        one = np.array(1 if scalar_indices else [1], np.int64)
        if mutation == "not_one":
            one[...] = 0
        constants.extend([numpy_helper.from_array(index, "index"),
                          numpy_helper.from_array(one, "one")])
        shape_attrs = {"start": 0} if mutation == "shape_slice" else {}
        nodes.append(helper.make_node("Shape", [condition_source], ["dims"], **shape_attrs))
        nodes.append(helper.make_node("Gather", ["dims", "index"], ["extent"],
                                      axis=1 if mutation == "gather_axis" else 0))
        equal_inputs = ["one", "extent"] if reverse_equal else ["extent", "one"]
        nodes.append(helper.make_node("Equal", equal_inputs, ["cond"]))
        axes = numpy_helper.from_array(np.array([axis], np.int64), "axis_value")
        then_nodes = [helper.make_node("Constant", [], ["local_axes"], value=axes),
                      helper.make_node("Squeeze", [source, "local_axes"], ["then_out"])]
        else_nodes = [helper.make_node("Identity", [source], ["else_out"])]
        if mutation == "branch_domain":
            then_nodes[1].domain = "foreign"
        if mutation == "branch_attribute":
            else_nodes[0].attribute.append(helper.make_attribute("unexpected", 1))
        then_output = "then_out"
        if mutation == "extra_branch_op":
            then_nodes.append(helper.make_node("Neg", [then_output], ["neg_out"]))
            then_output = "neg_out"
        if mutation == "wrong_branch_output":
            then_output = "local_axes"
        then_branch = helper.make_graph(then_nodes, "unit_extent", [], [
            helper.make_tensor_value_info(then_output, dtype, squeezed_shape)])
        else_branch = helper.make_graph(else_nodes, "other_extent", [], [
            helper.make_tensor_value_info("else_out", dtype, list(shape))])
        if_node = helper.make_node("If", ["cond"], ["Y"],
                                   then_branch=then_branch, else_branch=else_branch)
        if mutation == "extra_if_attribute":
            if_node.attribute.append(helper.make_attribute("unexpected", 1))
        if mutation == "runtime_condition":
            inputs["runtime_cond"] = np.array(True, np.bool_)
            if_node.input[0] = "runtime_cond"
        nodes.append(if_node)
        outputs = [helper.make_tensor_value_info("Y", dtype, actual_shape)]
        if nonzero:
            nodes.append(helper.make_node("NonZero", ["Y"], ["NZ"]))
            outputs = [helper.make_tensor_value_info("NZ", TensorProto.INT64,
                       [len(actual_shape), int(np.count_nonzero(x))])]
        input_infos = [helper.make_tensor_value_info(key,
                       helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
                       for key, value in inputs.items()]
        if mutation == "unresolved_extent":
            input_infos[0].type.tensor_type.shape.dim[normalized_axis].ClearField("dim_value")
            input_infos[0].type.tensor_type.shape.dim[normalized_axis].dim_param = "dynamic_axis"
        graph = helper.make_graph(nodes, name, input_infos, outputs, constants)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)],
                                  ir_version=9, producer_name="gac-shape-if-tests")
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        valid = mutation is None and opset in range(13, 19)
        if valid:
            onnx.checker.check_model(model)
            values = ReferenceEvaluator(model).run(None, inputs)
            np.savez(folder / "expected.npz", **dict(zip([o.name for o in outputs], values)))
        diagnostic = None
        if mutation == "unresolved_extent":
            diagnostic = "requires a static input shape"
        elif mutation is not None:
            diagnostic = "shape-guarded Squeeze/Identity pattern"
        elif not valid:
            diagnostic = "opset"
        cases.append({"name": name, "valid": valid, "unsupported": False,
                      "expect_fusion": False, "inputs": list(inputs),
                      "outputs": [o.name for o in outputs],
                      "expected_diagnostic": diagnostic,
                      "expect_optimized_absent": ['"gac.squeeze_if_unit_dim"(']})
        if constant:
            cases[-1]["expect_optimized_absent"].append('"gac.nonzero"(')

    save("shape_if_squeeze_negative_axis", [2, 3, 1])
    save("shape_if_identity_dead_invalid_squeeze", [2, 3])
    save("shape_if_squeeze_middle_reverse_equal", [2, 1, 3], axis=1, reverse_equal=True)
    save("shape_if_rank_one_to_scalar", [1], scalar_indices=True)
    save("shape_if_scalar_condition_identity", [4], axis=0, scalar_indices=True)
    save("shape_if_after_runtime_reshape", [2, 3, 1], reshape=True)
    save("shape_if_constant_nonzero_geometry", [2, 3, 1], constant=True, nonzero=True)
    for mutation in ("different_capture", "axis_mismatch", "gather_axis", "not_one",
                     "shape_slice", "branch_domain", "branch_attribute",
                     "extra_branch_op", "wrong_branch_output", "extra_if_attribute",
                     "runtime_condition", "unresolved_extent"):
        save("shape_if_reject_" + mutation, [2, 3, 1], mutation=mutation)
    save("shape_if_reject_opset12", [2, 3, 1], opset=12)
    save("shape_if_reject_opset19", [2, 3, 1], opset=19)
    result = {"cases": cases}
    (output / "cases.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = generate(args.output)
    print(f"Generated {len(result['cases'])} ONNX shape-guarded If cases: {args.output}")
