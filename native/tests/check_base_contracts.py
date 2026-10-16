#!/usr/bin/env python3
"""Exercise native custom import contracts and bounded constant views.

Python constructs test inputs and checks compiler output only. All model
transformations under test are performed by the C++ MLIR compiler.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contracts-only", action="store_true",
                        help="Skip the larger constant-view test while exporting a full model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []

    def model(node, inputs, constants, shape):
        return h.make_model(h.make_graph([node], "contract",
            [h.make_tensor_value_info(key, dtype, dims) for key, dtype, dims in inputs],
            [h.make_tensor_value_info("Y", T.FLOAT, shape)],
            [nh.from_array(value, key) for key, value in constants.items()]),
            opset_imports=[h.make_opsetid("", 17), h.make_opsetid(node.domain, 1)],
            ir_version=9)

    def check(name, graph, diagnostic=None):
        path = args.output / f"{name}.onnx"
        onnx.save(graph, path)
        result = subprocess.run([str(args.compiler), str(path), "--emit=optimized",
                                 "-o", str(path.with_suffix(".mlir"))],
                                capture_output=True, text=True, timeout=60)
        if diagnostic is None:
            assert result.returncode == 0, result.stderr
        else:
            assert result.returncode != 0 and re.search(diagnostic, result.stderr), result.stderr
        report.append({"name": name, "passed": True, "expected_rejection": diagnostic is not None})

    dcn = model(h.make_node("MMCVModulatedDeformConv2d", ["X", "offset", "mask", "W"], ["Y"],
                           domain="mmcv", stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                           groups=1, deform_groups=1),
                [("X", T.FLOAT, [1, 2, 3, 4]), ("offset", T.FLOAT, [1, 2, 3, 4]),
                 ("mask", T.FLOAT, [1, 1, 3, 4])],
                {"W": np.ones((3, 2, 1, 1), np.float32)}, [1, 3, 3, 4])
    check("dcn_contract", dcn)
    bad = copy.deepcopy(dcn)
    bad.opset_import[1].version = 2
    check("dcn_wrong_custom_version", bad, "domain opset 1")
    bad = copy.deepcopy(dcn)
    bad.graph.node[0].attribute.append(h.make_attribute("strides", [1, 1]))
    check("dcn_unknown_attribute", bad, "unrecognized custom operator attribute")
    bad = copy.deepcopy(dcn)
    bad.graph.node[0].input.insert(2, "")
    check("dcn_input_hole", bad, "cannot contain holes")
    bad = copy.deepcopy(dcn)
    bad.graph.input[0].type.tensor_type.elem_type = T.FLOAT16
    check("dcn_fp16_rejected", bad, "requires FP32")
    bad = copy.deepcopy(dcn)
    bad.graph.node[0].attribute.extend([h.make_attribute("unknown_layout", "NHWC")])
    check("dcn_layout_not_guessed", bad, "unrecognized custom operator attribute")

    msda = model(h.make_node("MSDeformAttn", ["value", "shapes", "locations", "weights"], ["Y"],
                            domain="gac", version=1, mode="bilinear", padding_mode="zeros", align_corners=0),
                 [("value", T.FLOAT, [1, 8, 2, 3]), ("locations", T.FLOAT, [1, 3, 2, 2, 2, 2]),
                  ("weights", T.FLOAT, [1, 3, 2, 2, 2])],
                 {"shapes": np.array([[2, 3], [1, 2]], np.int64)}, [1, 3, 6])
    check("msda_contract", msda)
    bad = copy.deepcopy(msda)
    for attr in bad.graph.node[0].attribute:
        if attr.name == "align_corners":
            attr.i = 1
    check("msda_align_corners_rejected", bad, "align_corners=0")
    bad = copy.deepcopy(msda)
    bad.graph.input[2].type.tensor_type.shape.dim[4].dim_value = 3
    check("msda_weights_shape_rejected", bad, "weight/sampling dimensions")
    deferred = copy.deepcopy(msda)
    # Import must accept a legal shape-producing graph, and inference must fold
    # it before the device's constant-only spatial_shapes contract is checked.
    deferred.graph.initializer[0].name = "source_shapes"
    deferred.graph.node.insert(0, h.make_node("Identity", ["source_shapes"], ["shapes"]))
    check("msda_deferred_constant_shapes", deferred)
    if args.contracts_only:
        (args.output / "contract-results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS {len(report)} native contract checks")
        return

    # Just beyond the old 4M-element bound: Tile and Cast are consumed through
    # indexed views; the f64 MatMul result is bounded and folded exactly.
    repetitions = 262145
    matrix = np.array([[[2**24 + 1., -(2**24), 0., 0.],
                        [1., 2., 3., 4.], [4., 3., 2., 1.], [0., 1., 0., 0.]]], np.float64)
    nodes = [h.make_node("Tile", ["matrix", "repeats"], ["tiled"]),
             h.make_node("Cast", ["tiled"], ["casted"], to=T.DOUBLE),
             h.make_node("MatMul", ["casted", "vector"], ["projected"]),
             h.make_node("Cast", ["projected"], ["Y"], to=T.FLOAT)]
    graph = h.make_model(h.make_graph(nodes, "large_geometry_view", [],
        [h.make_tensor_value_info("Y", T.FLOAT, [repetitions, 4, 1])],
        [nh.from_array(matrix, "matrix"), nh.from_array(np.array([repetitions, 1, 1], np.int64), "repeats"),
         nh.from_array(np.ones((4, 1), np.float64), "vector")]),
         opset_imports=[h.make_opsetid("", 17)], ir_version=9)
    folder = args.output / "large_geometry_view"
    folder.mkdir(exist_ok=True)
    path = folder / "model.onnx"
    onnx.save(graph, path)
    command = [str(args.compiler), str(path), "--target=native", "--emit=device",
               "--constant-fold-profile=bevformer-base", "--manifest", str(folder / "plan.json"),
               "--constants-dir", str(folder / "constants"), "-o", str(folder / "device.mlir")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stderr
    plan = json.loads((folder / "plan.json").read_text())
    assert plan["dispatches"] == [], "The bounded geometry must fold, not run on the target"
    output = next(t for t in plan["tensors"] if t["name"] == plan["outputs"][0])
    binary = folder / "constants" / Path(output["file"]).name
    actual = np.fromfile(binary, dtype=np.float32).reshape(repetitions, 4)
    expected = np.broadcast_to(np.array([1., 10., 10., 1.], np.float32), actual.shape)
    np.testing.assert_array_equal(actual, expected)
    report.append({"name": "large_geometry_view", "passed": True, "dispatches": 0,
                   "logical_tile_elements": repetitions * 16, "output_elements": actual.size,
                   "fp64_cancellation_result": float(actual[0, 0])})
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {len(report)} native contract/view checks")


if __name__ == "__main__":
    main()
