#!/usr/bin/env python3
"""Check fusion legality and emitted code against an independent ONNX reference.

The same fixtures run with/without graph optimization through both Linalg and
legacy lowering. Python only generates inputs, invokes tools and checks results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import onnxruntime as ort

from run_full_model import execute_host, load_plan, read_inputs, sha256
from run_matmul_bias_tests import check
from run_tests import run


def generate(directory: Path) -> list[dict]:
    rng = np.random.default_rng(20261021)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    cases = []

    def random(shape):
        return rng.uniform(-0.5, 0.5, shape).astype(np.float32)

    def save(name, x, weights, bias, *, fuse=True, relu=True, left=False,
             constant_bias=False, public=False, fork=False, old_bias=None,
             pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), groups=1,
             exact=False):
        output_shape = [x.shape[0], weights.shape[0]] + [
            (x.shape[axis + 2] + pads[axis] + pads[axis + 2]
             - dilations[axis] * (weights.shape[axis + 2] - 1) - 1)
            // strides[axis] + 1 for axis in range(2)]
        final_shape = list(np.broadcast_shapes(tuple(output_shape), bias.shape))
        dtype = TensorProto.FLOAT16 if x.dtype == np.float16 else TensorProto.FLOAT
        conv_inputs = ["X", "W"] + (["old_bias"] if old_bias is not None else [])
        nodes = [helper.make_node("Conv", conv_inputs, ["C"], name="conv",
                                  pads=list(pads), strides=list(strides),
                                  dilations=list(dilations), group=groups),
                 helper.make_node("Add", ["B", "C"] if left else ["C", "B"],
                                  ["A" if relu else "Y"], name="channel_bias")]
        if relu:
            nodes.append(helper.make_node("Relu", ["A"], ["Y"], name="activation"))
        outputs = {"Y": final_shape}
        if public:
            outputs["C"] = output_shape
        if fork:
            nodes.append(helper.make_node("Mul", ["C", "C"], ["other"]))
            outputs["other"] = output_shape
        constants = {"W": weights}
        inputs = {"X": x}
        (constants if constant_bias else inputs)["B"] = bias
        if old_bias is not None:
            constants["old_bias"] = old_bias
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info(key, dtype, list(value.shape))
             for key, value in inputs.items()],
            [helper.make_tensor_value_info(key, dtype, shape) for key, shape in outputs.items()],
            [numpy_helper.from_array(value, key) for key, value in constants.items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                                  ir_version=9, producer_name="gac-fusion-contract-tests")
        onnx.checker.check_model(model)
        folder = directory / name
        folder.mkdir(parents=True, exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        if dtype == TensorProto.FLOAT16:
            values = ReferenceEvaluator(model).run(list(outputs), inputs)
            reference = "ONNX ReferenceEvaluator, explicit FP16 boundaries"
        else:
            session = ort.InferenceSession(str(folder / "model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            values = session.run(list(outputs), inputs)
            reference = "ONNX Runtime CPU, graph optimizations disabled"
        np.savez(folder / "expected.npz", **dict(zip(outputs, values)))
        cases.append({"name": name, "fuse_channel_bias": fuse, "relu": relu,
                      "exact": exact, "reference": reference})

    x, weights = random((1, 2, 3, 3)), random((3, 2, 1, 1))
    save("channel_3d_runtime", x, weights, random((3, 1, 1)))
    save("channel_4d_constant", x, weights, random((1, 3, 1, 1)), constant_bias=True)
    save("channel_bias_without_activation", x, weights, random((3, 1, 1)), relu=False)
    save("public_conv_retained", x, weights, random((3, 1, 1)), public=True, fuse=False)
    save("multiple_consumers_retained", x, weights, random((3, 1, 1)), fork=True, fuse=False)
    save("left_bias_order_retained", x, weights, random((3, 1, 1)), left=True, fuse=False)
    # Cout == W makes [Cout] a valid broadcast, but it still addresses width.
    save("width_bias_is_not_channel_bias", x, weights, random((3,)), fuse=False)
    save("batch_specific_bias_retained", random((2, 2, 3, 3)), weights,
         random((2, 3, 1, 1)), fuse=False)
    save("expanded_batch_domain_retained", x, weights, random((2, 3, 1, 1)), fuse=False)
    save("group_padding_stride_dilation", random((2, 4, 5, 7)), random((6, 2, 3, 3)),
         random((1, 6, 1, 1)), groups=2, pads=(1, 0, 2, 1), strides=(2, 1),
         dilations=(1, 2))
    half = np.asarray([[[[1.0234375]]]], dtype=np.float16)
    rounded = np.float16(float(half.item()) ** 2)
    save("fp16_rounding_boundary", half, half.copy(),
         np.asarray([[[-rounded]]], dtype=np.float16), fuse=False, exact=True)
    save("existing_bias_not_reassociated", np.asarray([[[[1e10]]]], np.float32),
         np.ones((1, 1, 1, 1), np.float32), np.ones((1, 1, 1), np.float32),
         old_bias=np.asarray([-1e10], np.float32), fuse=False, exact=True)
    special = np.asarray([[[[np.nan, np.inf, -np.inf, -2, -0.0, 0.0]]]], np.float32)
    save("nonfinite_activation_semantics", special, np.ones((1, 1, 1, 1), np.float32),
         np.zeros((1, 1, 1), np.float32), exact=True)
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


def execute_case(args, folder, case, mode, optimize):
    destination = folder / (mode + ("-fused" if optimize else "-unfused"))
    destination.mkdir(exist_ok=True)
    manifest = destination / "plan.json"
    constants = destination / "constants"
    llvm_mlir, llvm_ir = destination / "llvm.mlir", destination / "kernel.ll"
    library = destination / "kernels.so"
    command = [args.compiler, folder / "model.onnx", "--target=native",
               f"--regular-lowering={mode}", "--emit=llvm", "-o", llvm_mlir,
               "--manifest", manifest, "--constants-dir", constants]
    if not optimize:
        command.append("--no-optimize")
    run(command)
    plan = load_plan(manifest)  # Also rejects overlapping live arena allocations.
    run([args.llvm_bin / "mlir-translate", "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir])
    run([args.llvm_bin / "opt", "-passes=verify", "-disable-output", llvm_ir])
    run([args.llvm_bin / "clang", "-shared", "-fPIC", "-O2", llvm_ir, "-lm", "-o", library])
    invocation = SimpleNamespace(manifest=manifest, constants_dir=constants, library=library,
        inputs=folder / "inputs.npz", output_dir=destination, resume=False, dump_intermediates=None,
        intermediate_reference=None, progress_every=1000, checkpoint_every=0, stop_after=0,
        rtol=2e-5, atol=2e-5)
    execution = {}
    actual = execute_host(invocation, plan, read_inputs(invocation.inputs, plan), execution)
    with np.load(folder / "expected.npz", allow_pickle=False) as expected:
        assert set(actual) == set(expected.files)
        comparisons = {name: check(value, expected[name], case["exact"])
                       for name, value in actual.items()}
    return {"dispatches": len(plan["dispatches"]), "arena_bytes": plan["arena_bytes"],
            "outputs": comparisons, "artifact_identity": execution["artifact_identity"]}, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    cases = generate(args.fixtures)
    report = {"scope": "Graph fusion contracts and generated CPU kernels; no GPU latency claims",
              "compiler_sha256_before": sha256(args.compiler), "cases": []}
    for case in cases:
        folder = args.fixtures / case["name"]
        entry = {**case, "passed": False, "runs": {}}
        try:
            optimized = folder / "optimized.mlir"
            run([args.compiler, folder / "model.onnx", "--target=native", "--emit=optimized",
                 "-o", optimized])
            text = optimized.read_text()
            has_add = '"gac.add"(' in text or '"gac.add_relu"(' in text
            assert has_add != case["fuse_channel_bias"], text
            if case["fuse_channel_bias"]:
                name = "conv2d_relu" if case["relu"] else "conv2d"
                assert text.count(f'"gac.{name}"(') == 1, text
            for mode in ("legacy", "linalg"):
                reference = None
                for optimize in (False, True):
                    key = mode + ("_fused" if optimize else "_unfused")
                    entry["runs"][key], actual = execute_case(args, folder, case, mode, optimize)
                    if reference is None:
                        reference = {name: array.copy() for name, array in actual.items()}
                    else:
                        for name, array in actual.items():
                            check(array, reference[name], exact=True)
                if case["fuse_channel_bias"]:
                    if mode == "legacy":
                        assert entry["runs"][mode + "_fused"]["dispatches"] < entry["runs"][mode + "_unfused"]["dispatches"]
            entry["passed"] = True
        except (AssertionError, OSError, ValueError) as error:
            entry["error"] = str(error)
        report["cases"].append(entry)
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {case['name']}", flush=True)
        if not entry["passed"]:
            print(entry["error"], flush=True)
        (args.fixtures / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["compiler_sha256_after"] = sha256(args.compiler)
    report["passed"] = report["compiler_sha256_before"] == report["compiler_sha256_after"] and all(
        case["passed"] for case in report["cases"])
    (args.fixtures / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{sum(case['passed'] for case in report['cases'])}/{len(cases)} fusion contract cases passed")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
