#!/usr/bin/env python3
"""Validate the real GAC -> Linalg -> SCF -> LLVM path and mixed dispatch DAGs.

Python creates small ONNX/reference fixtures only. Every tested output is
computed by the C++ compiler's generated native machine code, via run_tests.
CUDA/Vortex hardware execution and performance are not asserted by this suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from run_tests import execute, run
from run_full_model import load_plan


REGULAR_OPS = (
    "matmul", "gemm", "add", "sub", "mul", "div", "relu", "add_relu",
    "matmul_bias", "matmul_bias_relu", "conv2d", "conv2d_relu", "reduce_sum", "reduce_mean",
    "layer_norm",
)
STRUCTURED_OP = re.compile(r'^\s*(?:%[^=\n]+\s*=\s*)?"?(?:linalg|tensor)\.', re.M)


def generate(fixtures: Path) -> list[dict]:
    fixtures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7813)
    cases = []

    def array(shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    def save(name, nodes, inputs, output_shapes, constants=None, *, expected=None,
             output_dtype=np.float32, minimum_linalg=1, irregular=False,
             expected_linalg=None, expected_arena_tensors=None,
             arena_bytes_limit=None):
        def info(key, value):
            return helper.make_tensor_value_info(
                key, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))

        graph = helper.make_graph(
            nodes, name, [info(key, value) for key, value in inputs.items()],
            [helper.make_tensor_value_info(key,
                helper.np_dtype_to_tensor_dtype(np.dtype(output_dtype)), shape)
             for key, shape in output_shapes.items()],
            [numpy_helper.from_array(value, name=key)
             for key, value in (constants or {}).items()],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                                  ir_version=9, producer_name="gac-linalg-regression")
        onnx.checker.check_model(model)
        if expected is None:
            # ReferenceEvaluator's newer GridSample implementation uses the
            # opset-20 names; the compiler still receives the original opset 18.
            reference = onnx.version_converter.convert_version(model, 20) if irregular else model
            values = ReferenceEvaluator(reference).run(None, inputs)
            expected = dict(zip(output_shapes, values))
        folder = fixtures / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        np.savez(folder / "expected.npz", **expected)
        cases.append({"name": name, "minimum_linalg": minimum_linalg,
                      "irregular": irregular, "integer": np.issubdtype(output_dtype, np.integer),
                      "expected_linalg": expected_linalg,
                      "expected_arena_tensors": expected_arena_tensors,
                      "arena_bytes_limit": arena_bytes_limit})

    node = helper.make_node
    save("simple_add", [node("Add", ["X", "Y"], ["Z"])],
         {"X": array((2, 3)), "Y": array((2, 3))}, {"Z": [2, 3]})
    save("matmul_broadcast_batches", [node("MatMul", ["A", "B"], ["Y"])],
         {"A": array((2, 1, 3, 5)), "B": array((1, 4, 5, 2))}, {"Y": [2, 4, 3, 2]})
    save("matmul_vector_dot", [node("MatMul", ["A", "B"], ["Y"])],
         {"A": array((7,)), "B": array((7,))}, {"Y": []})
    save("matmul_left_vector", [node("MatMul", ["A", "B"], ["Y"])],
         {"A": array((5,)), "B": array((2, 5, 3))}, {"Y": [2, 3]})
    # The initializer, contraction and epilogue must share one dispatch. Only
    # the final 3x4 result needs device arena storage (rounded to 64-byte ABI
    # alignment), rather than three full-size intermediate tensors.
    save("matmul_bias_relu_fused_dispatch", [node("MatMul", ["A", "B"], ["M"]),
         node("Add", ["M", "Bias"], ["P"]), node("Relu", ["P"], ["Y"])],
         {"A": array((3, 5)), "B": array((5, 4)), "Bias": array((4,))},
         {"Y": [3, 4]}, expected_linalg=1, expected_arena_tensors=1,
         arena_bytes_limit=64)
    # A public intermediate and two consumers exercise the new Linalg results'
    # arena lifetimes: later dispatches must not overwrite M or Y prematurely.
    save("matmul_fanout_public_intermediate", [node("MatMul", ["A", "B"], ["M"]),
         node("Add", ["M", "Bias"], ["Y"]), node("Relu", ["M"], ["Z"])],
         {"A": array((2, 3)), "B": array((3, 4)), "Bias": array((4,))},
         {"M": [2, 4], "Y": [2, 4], "Z": [2, 4]}, minimum_linalg=3,
         expected_linalg=3, expected_arena_tensors=3)
    # This producer is NOT a graph output: two consumers alone must prevent
    # its erasure. Otherwise fusion would either recompute it or leave one
    # consumer reading storage with no producing dispatch.
    save("pointwise_fanout_keeps_producer", [node("Add", ["X", "B"], ["M"]),
         node("Mul", ["M", "S"], ["Y"]), node("Sub", ["M", "C"], ["Z"])],
         {"X": array((2, 3)), "B": array((3,)), "S": array((3,)), "C": array((3,))},
         {"Y": [2, 3], "Z": [2, 3]}, minimum_linalg=3,
         expected_linalg=3, expected_arena_tensors=3)
    save("gemm_transposed_alpha_beta", [node("Gemm", ["A", "B", "C"], ["Y"],
         transA=1, transB=1, alpha=0.25, beta=-0.5)],
         {"A": array((5, 3)), "B": array((4, 5)), "C": array((4,))}, {"Y": [3, 4]})
    # FP16 accumulation would lose the central +1. FP32 accumulation followed
    # by one final FP16 cast must produce exactly 1 in each output element.
    a = np.ones((2, 3), np.float16)
    b = np.array([[2048, -2048], [1, 1], [-2048, 2048]], np.float16)
    save("matmul_fp16_fp32_accumulation", [node("MatMul", ["A", "B"], ["Y"])],
         {"A": a, "B": b}, {"Y": [2, 2]}, output_dtype=np.float16,
         expected={"Y": (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)})
    # 2049 rounds to FP16 2048 at the MatMul result boundary. Fusing away that
    # conversion would make the subsequent Add produce +/-1 instead of zero.
    # Dispatch fusion must keep the explicit truncation/extension in its body.
    half_a = np.ones((2, 2), np.float16)
    half_b = np.array([[2048, -2048], [1, -1]], np.float16)
    half_bias = np.array([-2048, 2048], np.float16)
    half_product = (half_a.astype(np.float32) @ half_b.astype(np.float32)).astype(np.float16)
    half_expected = (half_product.astype(np.float32) + half_bias.astype(np.float32)).astype(np.float16)
    save("matmul_fp16_fusion_keeps_rounding", [node("MatMul", ["A", "B"], ["M"]),
         node("Add", ["M", "Bias"], ["Y"])],
         {"A": half_a, "B": half_b, "Bias": half_bias}, {"Y": [2, 2]},
         expected={"Y": half_expected}, output_dtype=np.float16,
         expected_linalg=1, expected_arena_tensors=1, arena_bytes_limit=64)
    save("add_broadcast", [node("Add", ["X", "B"], ["Y"])],
         {"X": array((2, 3, 4)), "B": array((1, 3, 1))}, {"Y": [2, 3, 4]})
    save("add_int64_exact", [node("Add", ["X", "B"], ["Y"])],
         {"X": np.arange(6, dtype=np.int64).reshape(2, 3) + 2**42,
          "B": np.array([3, -5, 11], dtype=np.int64)}, {"Y": [2, 3]}, output_dtype=np.int64)
    # Equal-shape pointwise fusion must not silently turn into broadcast
    # recomputation: this source result is smaller than its consumer result.
    save("broadcast_consumer_keeps_producer", [node("Add", ["X", "B"], ["M"]),
         node("Add", ["M", "C"], ["Y"])],
         {"X": array((1, 3)), "B": array((3,)), "C": array((2, 3))},
         {"Y": [2, 3]}, minimum_linalg=2, expected_linalg=2,
         expected_arena_tensors=2)
    # Even matching input/output shapes do not authorize arbitrary fusion
    # into a reduction. Its extent-one reduction axis is still a reduction;
    # only its no-input initializer may be fused by the current policy.
    save("reduction_consumer_keeps_producer", [node("Add", ["X", "B"], ["M"]),
         node("ReduceSum", ["M", "axes"], ["Y"], keepdims=1)],
         {"X": array((2, 1)), "B": array((2, 1))}, {"Y": [2, 1]},
         {"axes": np.array([1], np.int64)}, minimum_linalg=2,
         expected_linalg=2, expected_arena_tensors=2)

    x = array((2, 3, 4))
    save("reduce_sum_nonlast_axes", [node("ReduceSum", ["X", "axes"], ["Y"], keepdims=0)],
         {"X": x}, {"Y": [3]}, {"axes": np.array([0, 2], np.int64)})
    save("reduce_sum_negative_keepdims", [node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)],
         {"X": x}, {"Y": [2, 1, 4]}, {"axes": np.array([-2], np.int64)})
    save("reduce_mean_all_rank_zero", [node("ReduceMean", ["X"], ["Y"], keepdims=0)],
         {"X": x}, {"Y": []})
    save("reduce_mean_nonlast_axes", [node("ReduceMean", ["X", "axes"], ["Y"], keepdims=0)],
         {"X": x}, {"Y": [3]}, {"axes": np.array([0, -1], np.int64)})
    save("reduce_sum_empty_axes_noop", [node("ReduceSum", ["X", "axes"], ["Y"],
         keepdims=0, noop_with_empty_axes=1)], {"X": x}, {"Y": [2, 3, 4]},
         {"axes": np.empty((0,), np.int64)}, minimum_linalg=0)

    for axis, shape, suffix in [(-1, (4,), "last_axis"), (1, (3, 4), "suffix_axes")]:
        save("layer_norm_" + suffix, [node("LayerNormalization", ["X", "S", "B"], ["Y"],
             axis=axis, epsilon=0.03125)], {"X": x}, {"Y": [2, 3, 4]},
             {"S": array(shape), "B": array(shape)})
    # Nontrivial group, asymmetric pad, stride and dilation in the same case.
    save("conv_grouped_padded_strided_dilated", [node("Conv", ["X", "W", "B"], ["C"],
         group=2, pads=[1, 0, 2, 1], strides=[2, 1], dilations=[2, 1]),
         node("Relu", ["C"], ["Y"])], {"X": array((1, 4, 6, 7))}, {"Y": [1, 6, 4, 6]},
         {"W": array((6, 2, 2, 3)), "B": array((6,))})

    grid = np.array([[[[-1.2, -0.8], [0.1, -0.3], [1.1, 0.0]],
                       [[-0.4, 0.6], [0.2, 0.8], [0.9, 1.2]]]], np.float32)
    save("mixed_add_gridsample_relu", [node("Add", ["X", "B"], ["A"]),
         node("GridSample", ["A", "G"], ["S"], mode="bilinear",
              padding_mode="zeros", align_corners=0), node("Relu", ["S"], ["Y"])],
         {"X": array((1, 2, 3, 4)), "G": grid}, {"Y": [1, 2, 2, 3]},
         {"B": array((1, 2, 1, 1))}, minimum_linalg=2, irregular=True)
    (fixtures / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    return cases


def check_case(args, case):
    folder = args.fixtures / case["name"]
    stages, constants = folder / "stages", folder / "constants"
    llvm_mlir, llvm_ir = folder / "llvm.mlir", folder / "kernel.ll"
    manifest, library = folder / "plan.json", folder / "kernels.so"
    run([args.compiler, folder / "model.onnx", "--target=native", "--emit=llvm",
         "--save-stages", stages, "--constants-dir", constants,
         "--manifest", manifest, "-o", llvm_mlir], timeout=args.timeout)
    structured = (stages / "02-linalg.mlir").read_text()
    device = (stages / "03-device.mlir").read_text()
    lowered = llvm_mlir.read_text()
    if case["minimum_linalg"]:
        assert "linalg.generic" in structured or "linalg.matmul" in structured, structured
    covered = "|".join(REGULAR_OPS)
    assert not re.search(r'"gac\.(?:' + covered + r')"\(', structured), structured
    assert not STRUCTURED_OP.search(device), device
    assert not STRUCTURED_OP.search(lowered), lowered
    assert not re.search(r'^\s*(?:%[^=\n]+\s*=\s*)?(?:scf|memref|affine|arith)\.', lowered, re.M)
    # Reuse the runtime's independent producer/liveness/arena-overlap checks.
    plan = load_plan(manifest)
    assert plan["regular_lowering"] == "linalg", plan
    assert plan["linalg_dispatches"] >= case["minimum_linalg"], plan
    if case["expected_linalg"] is not None:
        assert plan["linalg_dispatches"] == case["expected_linalg"], plan
    arena_tensors = [tensor for tensor in plan["tensors"] if tensor["storage"] == "arena"]
    if case["expected_arena_tensors"] is not None:
        assert len(arena_tensors) == case["expected_arena_tensors"], arena_tensors
    if case["arena_bytes_limit"] is not None:
        assert plan["arena_bytes"] <= case["arena_bytes_limit"], plan
    if case["irregular"]:
        assert '"gac.grid_sample"(' in structured, structured
        kinds = [item["op"] for item in plan["dispatches"]]
        grid_index = kinds.index("gac.grid_sample")
        assert any(kind.startswith("linalg.") for kind in kinds[:grid_index]), kinds
        assert any(kind.startswith("linalg.") for kind in kinds[grid_index + 1:]), kinds
    run([args.mlir_translate, "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir], timeout=args.timeout)
    run([args.clang, "-shared", "-fPIC", "-O2", llvm_ir, "-lm", "-o", library], timeout=args.timeout)
    result = execute(manifest, constants, library, folder)
    if case["integer"]:
        assert all(error == 0 for error in result["max_absolute_errors"].values()), result
    return {"name": case["name"], "passed": True, "linalg_dispatches": plan["linalg_dispatches"],
            "arena_bytes": plan["arena_bytes"], "arena_tensor_count": len(arena_tensors),
            "buffer_lifetimes_validated": True, **result}


def check_authoritative_ir(args):
    folder = args.fixtures / "simple_add"
    original, mutated = folder / "editable.mlir", folder / "mutated.mlir"
    run([args.compiler, folder / "model.onnx", "--emit=linalg", "-o", original])
    text = original.read_text()
    assert text.count("arith.addf") == 1, text
    mutated.write_text(text.replace("arith.addf", "arith.subf"))
    output = folder / "mutated-cuda-device.mlir"
    run([args.optimizer, mutated, "--gac-lower-to-cuda", "-o", output])
    device = output.read_text()
    assert "arith.subf" in device and "arith.addf" not in device, device
    assert not STRUCTURED_OP.search(device), device
    return {"name": "linalg_payload_is_authoritative", "passed": True,
            "check": "Changed Linalg Add payload to Sub; CUDA device IR contains Sub, not reconstructed Add"}


def check_schedule_boundary(args):
    model = args.fixtures / "simple_add" / "model.onnx"
    for schedule in ("--cuda-matmul-schedule=shared-tiled",
                     "--cuda-reduction-schedule=cooperative",
                     "--vortex-matmul-schedule=lmem-tiled"):
        result = run([args.compiler, model, "--emit=device", schedule], succeeds=False)
        assert "legacy" in result.stderr and "Linalg" in result.stderr, result.stderr
    # Compatibility must be intentional and visible in the generated manifest.
    folder = args.fixtures / "legacy-explicit"
    folder.mkdir(exist_ok=True)
    plan = folder / "plan.json"
    run([args.compiler, model, "--target=native", "--emit=llvm", "--regular-lowering=legacy",
         "--manifest", plan, "--constants-dir", folder / "constants", "-o", folder / "llvm.mlir"])
    metadata = json.loads(plan.read_text())
    assert metadata["regular_lowering"] == "legacy" and metadata["linalg_dispatches"] == 0, metadata
    return {"name": "legacy_schedule_requires_explicit_opt_in", "passed": True}


def check_ir_boundaries(args):
    folder = args.fixtures / "rejected-mlir"
    folder.mkdir(exist_ok=True)
    # These inputs are well-formed MLIR, but exceed the device ABI's contract.
    # Each must fail with a diagnostic rather than alias an unrelated buffer,
    # discard an encoding, or leave references to the subsequently erased graph.
    empty = """module {
  func.func @main() -> tensor<2xf32> {
    %empty = tensor.empty() : tensor<2xf32>
    return %empty : tensor<2xf32>
  }
}
"""
    generic = """module {
  func.func @main(%input: TYPE) -> TYPE {
    %empty = tensor.empty() : TYPE
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%input : TYPE) outs(%empty : TYPE) {
      ^bb0(%in: f32, %out: f32):
        PAYLOAD
    } -> TYPE
    return %result : TYPE
  }
}
"""
    encoded = generic.replace("TYPE", 'tensor<2xf32, "non-row-major">').replace(
        "PAYLOAD", "linalg.yield %in : f32")
    captured = generic.replace("TYPE", "tensor<2xf32>").replace("PAYLOAD", """%zero = arith.constant 0 : index
        %outside = tensor.extract %input[%zero] : tensor<2xf32>
        linalg.yield %outside : f32""")
    specifications = [
        ("reject_empty_graph_output", empty, "no initialized device tensor"),
        ("reject_encoded_linalg_tensor", encoded, "unencoded row-major tensors"),
        ("reject_linalg_external_capture", captured, "must not capture external SSA values"),
    ]
    results = []
    for name, source, diagnostic in specifications:
        path = folder / (name + ".mlir")
        path.write_text(source)
        result = run([args.optimizer, path, "--gac-lower-to-cuda", "-o", folder / (name + ".out")],
                     succeeds=False)
        assert diagnostic in result.stderr, result.stderr
        results.append({"name": name, "passed": True, "diagnostic": diagnostic})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("compiler", "optimizer", "mlir-translate", "clang", "fixtures"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    for name in ("compiler", "optimizer", "mlir_translate", "clang", "fixtures"):
        setattr(args, name, getattr(args, name).resolve())
    cases = generate(args.fixtures)
    unknown = set(args.selected or []) - {case["name"] for case in cases}
    if unknown:
        parser.error(f"Unknown cases: {sorted(unknown)}")
    if args.selected:
        cases = [case for case in cases if case["name"] in args.selected]
    digest = lambda: hashlib.sha256(args.compiler.read_bytes()).hexdigest()
    report = {"scope": "real Linalg lowering + native CPU numerical tests + CUDA device IR mutation",
              "cuda_or_vortex_hardware_execution": False, "performance_measured": False,
              "compiler_sha256": digest(), "cases": []}
    path = args.fixtures / ("selected-results.json" if args.selected else "results.json")
    try:
        for case in cases:
            result = check_case(args, case)
            report["cases"].append(result)
            print(f"PASS {case['name']}", flush=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        for check in (check_authoritative_ir, check_schedule_boundary):
            result = check(args)
            report["cases"].append(result)
            print(f"PASS {result['name']}", flush=True)
        for result in check_ir_boundaries(args):
            report["cases"].append(result)
            print(f"PASS {result['name']}", flush=True)
        assert digest() == report["compiler_sha256"], "Compiler changed during validation"
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["failure"] = str(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['cases'])} Linalg regression checks passed; {path}")


if __name__ == "__main__":
    main()
