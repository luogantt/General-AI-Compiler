#!/usr/bin/env python3
"""Exercise ONNX -> C++ Linalg -> Triton/native LLVM -> one vxbin -> SimX.

NumPy is only an independent oracle. No model computation runs in Python on
the execution path. This is a correctness/ABI suite, not a hardware benchmark.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def run(command, log, *, env=None, failure=None):
    result = subprocess.run([str(x) for x in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env)
    log.write_text(result.stdout)
    if failure is not None:
        assert result.returncode != 0 and failure in result.stdout, result.stdout
    elif result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}\n{result.stdout[-6000:]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).resolve().parents[1] / "tools/compile-triton-vortex.py"
    driver_spec = importlib.util.spec_from_file_location(
        "gac_triton_vortex_driver_validation", driver)
    assert driver_spec and driver_spec.loader
    driver_module = importlib.util.module_from_spec(driver_spec)
    driver_spec.loader.exec_module(driver_module)
    rng = np.random.default_rng(20261022)
    cases = []

    def reject_in_driver(plan, source_dir, *, block=128, warps=1):
        driver_args = SimpleNamespace(block=block, num_warps=warps, lanes=4)
        try:
            list(driver_module.triton_dispatches(plan, source_dir, driver_args))
        except ValueError:
            return
        raise AssertionError("malformed Triton manifest was accepted by build driver")

    def random(shape):
        return rng.uniform(-1, 1, shape).astype(np.float32)

    def batch_distinct(shape):
        value = random(shape)
        if len(shape) > 2:
            flat = value.reshape((-1, *shape[-2:]))
            for batch in range(flat.shape[0]):
                flat[batch] += np.float32(batch * 0.25)
        return value

    def add(name, nodes, inputs, outputs, constants=None, *, triton=True,
            native=False, block=128, warps=1):
        dtype = next(iter(inputs.values())).dtype
        proto = helper.np_dtype_to_tensor_dtype(dtype)
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info(k, helper.np_dtype_to_tensor_dtype(v.dtype), v.shape)
             for k, v in inputs.items()],
            [helper.make_tensor_value_info(k, proto, shape) for k, shape in outputs.items()],
            [numpy_helper.from_array(v, k) for k, v in (constants or {}).items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                                  ir_version=9)
        onnx.checker.check_model(model)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = ReferenceEvaluator(model).run(None, inputs)
        cases.append((name, model, inputs, expected, triton, native, block, warps))

    for n in (1, 17, 1003):
        add(f"add_{n}", [helper.make_node("Add", ["A", "B"], ["Y"])],
            {"A": random((n,)), "B": random((n,))}, {"Y": (n,)})
    for dtype in (np.float32, np.int32):
        shape = (2, 3, 19)
        values = {k: (random(shape) * 7).astype(dtype) for k in ("A", "B", "C")}
        add(f"fused_{np.dtype(dtype).name}", [
            helper.make_node("Add", ["A", "B"], ["sum"]),
            helper.make_node("Mul", ["sum", "C"], ["product"]),
            helper.make_node("Relu", ["product"], ["Y"])], values,
            {"Y": shape}, block=64, warps=2)
    add("duplicate_input", [helper.make_node("Add", ["A", "A"], ["Y"])],
        {"A": random((33,))}, {"Y": (33,)})
    add("nonfinite_relu", [helper.make_node("Relu", ["A"], ["Y"])],
        {"A": np.array([np.nan, np.inf, -np.inf, -0., 0., -3., 4.], np.float32)},
        {"Y": (7,)})
    add("view_alias", [helper.make_node("Reshape", ["A", "shape"], ["view"]),
                       helper.make_node("Add", ["view", "B"], ["Y"])],
        {"A": random((3, 7)), "B": random((21,))}, {"Y": (21,)},
        {"shape": np.array([21], np.int64)})
    add("broadcast_native", [helper.make_node("Add", ["A", "B"], ["Y"])],
        {"A": random((3, 7)), "B": random((7,))}, {"Y": (3, 7)},
        triton=False, native=True)
    add("mixed_matmul", [helper.make_node("Add", ["A", "B"], ["sum"]),
        helper.make_node("MatMul", ["sum", "W"], ["M"]),
        helper.make_node("Mul", ["M", "R"], ["scaled"]),
        helper.make_node("Relu", ["scaled"], ["Y"])],
        {"A": random((5, 7)), "B": random((5, 7)), "R": random((5, 9))},
        {"M": (5, 9), "Y": (5, 9)}, {"W": random((7, 9))}, native=True)
    add("native_fused_matmul", [helper.make_node("MatMul", ["A", "W"], ["M"]),
        helper.make_node("Relu", ["M"], ["Y"])], {"A": random((5, 16))},
        {"Y": (5, 9)}, {"W": random((16, 9))}, triton=False, native=True)
    add("triton_matmul_aligned", [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((16, 16)), "B": random((16, 16))}, {"Y": (16, 16)},
        triton=True, native=False)
    add("triton_matmul_tail", [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((17, 32)), "B": random((32, 19))}, {"Y": (17, 19)},
        triton=True, native=False)
    add("triton_matmul_batched_matched",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": batch_distinct((2, 17, 32)), "B": random((2, 32, 19))},
        {"Y": (2, 17, 19)}, triton=True, native=False)
    add("triton_matmul_batched_shared_rhs",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((2, 17, 32)), "B": random((32, 19))},
        {"Y": (2, 17, 19)}, triton=True, native=False)
    add("triton_matmul_rank4_matched",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((2, 3, 5, 16)), "B": random((2, 3, 16, 7))},
        {"Y": (2, 3, 5, 7)}, triton=True, native=False)
    add("triton_matmul_rank4_shared_rhs_k48_constant",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": batch_distinct((2, 2, 5, 48))}, {"Y": (2, 2, 5, 7)},
        {"B": random((48, 7))}, triton=True, native=False)
    add("triton_matmul_unit_prefix",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((1, 2, 5, 16)), "B": random((1, 2, 16, 7))},
        {"Y": (1, 2, 5, 7)}, triton=True, native=False)
    add("native_matmul_unaligned_k",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((7, 15)), "B": random((15, 11))}, {"Y": (7, 11)},
        triton=False, native=True)
    add("native_matmul_vector_lhs",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((16,)), "B": random((16, 7))}, {"Y": (7,)},
        triton=False, native=True)
    add("native_matmul_vector_rhs",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((5, 16)), "B": random((16,))}, {"Y": (5,)},
        triton=False, native=True)
    add("native_matmul_singleton_broadcast",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((2, 17, 32)), "B": random((1, 32, 19))},
        {"Y": (2, 17, 19)}, triton=False, native=True)
    add("native_matmul_shared_lhs",
        [helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"A": random((17, 32)), "B": random((2, 32, 19))},
        {"Y": (2, 17, 19)}, triton=False, native=True)
    add("native_matmul_alias_inputs",
        [helper.make_node("MatMul", ["A", "A"], ["Y"])],
        {"A": random((2, 16, 16))}, {"Y": (2, 16, 16)},
        triton=False, native=True)
    add("native_matmul_input_reshape", [
            helper.make_node("Reshape", ["ARaw", "AShape"], ["A"]),
            helper.make_node("MatMul", ["A", "B"], ["Y"])],
        {"ARaw": random((34, 32)), "B": random((2, 32, 19))},
        {"Y": (2, 17, 19)},
        {"AShape": np.array([2, 17, 32], np.int64)},
        triton=False, native=True)
    add("triton_reduce_sum_scalar",
        [helper.make_node("ReduceSum", ["A", "axes"], ["Y"], keepdims=0)],
        {"A": random((37,))}, {"Y": ()},
        {"axes": np.array([0], np.int64)}, triton=True, native=False)
    add("triton_reduce_sum_singleton",
        [helper.make_node("ReduceSum", ["A", "axes"], ["Y"], keepdims=1)],
        {"A": random((19,))}, {"Y": (1,)},
        {"axes": np.array([0], np.int64)}, triton=True, native=False)
    add("native_reduce_sum_rank2",
        [helper.make_node("ReduceSum", ["A", "axes"], ["Y"], keepdims=0)],
        {"A": random((3, 17))}, {"Y": (3,)},
        {"axes": np.array([1], np.int64)}, triton=False, native=True)
    add("native_reduce_sum_too_wide",
        [helper.make_node("ReduceSum", ["A", "axes"], ["Y"], keepdims=0)],
        {"A": random((1025,))}, {"Y": ()},
        {"axes": np.array([0], np.int64)}, triton=False, native=True)
    add("triton_softmax_width_1",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.array([3.5], np.float32)}, {"Y": (1,)},
        triton=True, native=False)
    add("triton_softmax_16", [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": random((16,))}, {"Y": (16,)}, triton=True, native=False)
    add("triton_softmax_tail_19",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": random((19,))}, {"Y": (19,)}, triton=True, native=False)
    add("triton_softmax_extreme_finite",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.array([10000., 9999., 1., 0., -1., -9999., -10000.],
                       np.float32)}, {"Y": (7,)}, triton=True, native=False)
    add("triton_softmax_nan",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.array([1., np.nan, -2., 3., 0.], np.float32)},
        {"Y": (5,)}, triton=True, native=False)
    add("triton_softmax_positive_inf",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.array([1., np.inf, -2., 3., 0.], np.float32)},
        {"Y": (5,)}, triton=True, native=False)
    add("triton_softmax_all_negative_inf",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.full((5,), -np.inf, np.float32)},
        {"Y": (5,)}, triton=True, native=False)
    add("triton_softmax_partial_negative_inf",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": np.array([-np.inf, 1., 2., -np.inf, -3.], np.float32)},
        {"Y": (5,)}, triton=True, native=False)
    add("triton_softmax_rank2_tail_19",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=-1)],
        {"A": random((3, 19))}, {"Y": (3, 19)},
        triton=True, native=False)
    add("triton_softmax_rank3_tail_19",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=2)],
        {"A": random((2, 3, 19))}, {"Y": (2, 3, 19)},
        triton=True, native=False)
    add("triton_softmax_rank3_bev_width_256",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=-1)],
        {"A": random((1, 2, 256))}, {"Y": (1, 2, 256)},
        triton=True, native=False)
    add("triton_softmax_rank3_width_1",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=-1)],
        {"A": random((2, 3, 1))}, {"Y": (2, 3, 1)},
        triton=True, native=False)
    add("triton_softmax_rank2_width_1024",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=-1)],
        {"A": random((1, 1024))}, {"Y": (1, 1024)},
        triton=True, native=False)
    add("native_softmax_rank2_axis0",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": random((3, 19))}, {"Y": (3, 19)},
        triton=False, native=True)
    add("native_softmax_too_wide",
        [helper.make_node("Softmax", ["A"], ["Y"], axis=0)],
        {"A": random((1025,))}, {"Y": (1025,)},
        triton=False, native=True)
    add("native_softmax_input_reshape", [
            helper.make_node("Reshape", ["XRaw", "Shape"], ["A"]),
            helper.make_node("Softmax", ["A"], ["Y"], axis=-1)],
        {"XRaw": random((57,))}, {"Y": (3, 19)},
        {"Shape": np.array([3, 19], np.int64)},
        triton=False, native=True)
    # Rejecting one Softmax must not disable Triton selection for an
    # independent pointwise sibling in the same graph.
    add("mixed_softmax_axis0_sibling_add", [
            helper.make_node("Softmax", ["X"], ["Y"], axis=0),
            helper.make_node("Add", ["A", "B"], ["Z"])],
        {"X": random((2, 19)), "A": random((2, 19)),
         "B": random((2, 19))},
        {"Y": (2, 19), "Z": (2, 19)},
        triton=True, native=True)

    def layer_norm(name, shape, *, axis=-1, epsilon=1e-5,
                   bias=True, shared_affine=False, triton=True, native=False):
        width_shape = shape[axis if axis >= 0 else len(shape) + axis:]
        scale_name = "Affine" if shared_affine else "Scale"
        constants = {scale_name: random(width_shape)}
        node_inputs = ["X", scale_name]
        if bias:
            bias_name = scale_name if shared_affine else "Bias"
            if not shared_affine:
                constants[bias_name] = random(width_shape)
            node_inputs.append(bias_name)
        add(name, [helper.make_node(
                "LayerNormalization", node_inputs, ["Y"], axis=axis,
                epsilon=epsilon, stash_type=1)],
            {"X": random(shape)}, {"Y": shape}, constants,
            triton=triton, native=native)

    # Width 256 is the BEVFormer hidden size. Width 19 proves that inactive
    # power-of-two tail lanes do not contaminate the variance reduction.
    layer_norm("triton_layer_norm_bev_width_256", (2, 256),
               epsilon=0.03125, triton=True, native=False)
    layer_norm("triton_layer_norm_tail_19", (2, 3, 19),
               epsilon=0.0009765625, triton=True, native=False)
    # Width one exercises the constant-zero scale/bias indexing map.  1e-5 is
    # intentionally not binary-exact, so the manifest/runtime path must retain
    # its original IEEE-754 float32 bit pattern.
    layer_norm("triton_layer_norm_width_1_epsilon_1e5", (2, 1),
               epsilon=1e-5, triton=True, native=False)
    # The upper accepted width is compiled and contract-checked below.  SimX
    # execution is skipped because it is boundary coverage rather than a
    # useful correctness/performance workload.
    layer_norm("triton_layer_norm_width_1024", (1, 1024),
               epsilon=1e-5, triton=True, native=False)
    layer_norm("native_layer_norm_rank1", (19,), triton=False, native=True)
    layer_norm("native_layer_norm_axis0", (2, 19), axis=0,
               triton=False, native=True)
    layer_norm("native_layer_norm_too_wide", (1, 1025),
               triton=False, native=True)
    layer_norm("native_layer_norm_missing_bias", (2, 19), bias=False,
               triton=False, native=True)
    layer_norm("native_layer_norm_shared_scale_bias", (2, 19),
               shared_affine=True, triton=False, native=True)
    affine = random((19,))
    add("native_layer_norm_physical_view_alias", [
            helper.make_node("Reshape", ["Affine", "ScaleShape"], ["ScaleView"]),
            helper.make_node("Reshape", ["Affine", "BiasShape"], ["BiasView"]),
            helper.make_node("LayerNormalization",
                             ["X", "ScaleView", "BiasView"], ["Y"],
                             axis=-1, epsilon=1e-5, stash_type=1)],
        {"X": random((2, 19))}, {"Y": (2, 19)},
        {"Affine": affine,
         "ScaleShape": np.array([19], np.int64),
         "BiasShape": np.array([19], np.int64)},
        triton=False, native=True)
    add("native_layer_norm_input_reshape", [
            helper.make_node("Reshape", ["XRaw", "XShape"], ["X"]),
            helper.make_node("LayerNormalization", ["X", "Scale", "Bias"],
                             ["Y"], axis=-1, epsilon=1e-5, stash_type=1)],
        {"XRaw": random((38,))}, {"Y": (2, 19)},
        {"XShape": np.array([2, 19], np.int64),
         "Scale": random((19,)), "Bias": random((19,))},
        triton=False, native=True)
    add("native_layer_norm_affine_reshape", [
            helper.make_node("Reshape", ["ScaleRaw", "AffineShape"],
                             ["Scale"]),
            helper.make_node("Reshape", ["BiasRaw", "AffineShape"],
                             ["Bias"]),
            helper.make_node("LayerNormalization", ["X", "Scale", "Bias"],
                             ["Y"], axis=-1, epsilon=1e-5, stash_type=1)],
        {"X": random((2, 19)), "ScaleRaw": random((1, 19)),
         "BiasRaw": random((1, 19))},
        {"Y": (2, 19)}, {"AffineShape": np.array([19], np.int64)},
        triton=False, native=True)
    layer_norm("native_layer_norm_zero_epsilon", (2, 19), epsilon=0.0,
               triton=False, native=True)
    layer_norm("native_layer_norm_negative_epsilon", (2, 19), epsilon=-0.125,
               triton=False, native=True)
    layer_norm("native_layer_norm_nan_epsilon", (2, 19), epsilon=float("nan"),
               triton=False, native=True)
    layer_norm("native_layer_norm_inf_epsilon", (2, 19), epsilon=float("inf"),
               triton=False, native=True)
    # A rejected LayerNorm must not disable Triton for an independent sibling
    # in the same function.  Axis 0 forces the complete LayerNorm statistics
    # chain to remain native while the same-shape Add remains pointwise TTIR.
    sibling_shape = (2, 19)
    add("mixed_layer_norm_axis0_sibling_add", [
            helper.make_node("LayerNormalization", ["X", "Scale", "Bias"],
                             ["Y"], axis=0, epsilon=1e-5,
                             stash_type=1),
            helper.make_node("Add", ["A", "B"], ["Z"])],
        {"X": random(sibling_shape), "A": random(sibling_shape),
         "B": random(sibling_shape)},
        {"Y": sibling_shape, "Z": sibling_shape},
        {"Scale": random(sibling_shape), "Bias": random(sibling_shape)},
        triton=True, native=True)

    rows = []
    for name, model, inputs, expected, want_triton, want_native, block, warps in cases:
        folder = args.output / name
        folder.mkdir()
        onnx.save(model, folder / "model.onnx")
        artifact = folder / "artifact"
        run([sys.executable, driver, folder / "model.onnx", "--compiler", args.compiler,
             "--output", artifact, "--block", block, "--num-warps", warps, "--lanes", 4],
            folder / "build.log")
        plan = json.loads((artifact / "plan.json").read_text())
        selected = [d for d in plan["dispatches"] if "triton" in d]
        native = [d for d in plan["dispatches"] if "triton" not in d]
        assert bool(selected) == want_triton, plan
        assert bool(native) == want_native, plan
        for d in native:
            assert d.get("triton_exclusion"), d
        if name.startswith("native_layer_norm_"):
            # A rejected aggregate must remain the complete native
            # statistics/final chain; no internal node may be half-fused into
            # a Triton dispatch or silently removed.
            assert not selected and len(native) == 3, plan
            assert plan["linalg_fused_producers"] == 3, plan
        if name == "mixed_layer_norm_axis0_sibling_add":
            assert len(selected) == 1 and \
                selected[0]["triton"]["kind"] == "pointwise", plan
            assert len(native) == 3, plan
            assert plan["linalg_fused_producers"] == 3, plan
        if name.startswith("native_softmax_"):
            # Softmax is one aggregate Linalg op.  A rejected candidate must
            # remain exactly one complete native dispatch; a preceding dense
            # Reshape view may be elided but must never change the ABI shape.
            assert not selected and len(native) == 1, plan
        if name == "mixed_softmax_axis0_sibling_add":
            assert len(selected) == 1 and \
                selected[0]["triton"]["kind"] == "pointwise", plan
            assert len(native) == 1 and native[0]["op"] == "linalg.softmax", plan
        if name.startswith("fused_"):
            assert len(selected) == 1 and plan["linalg_fused_producers"] >= 1, plan
        if name == "duplicate_input":
            assert len(selected[0]["inputs"]) == 1, plan
        if name == "mixed_matmul":
            # The first temporary is dead before the final Triton dispatch;
            # both backends must honor the common arena reuse plan.
            offsets = [t["offset"] for t in plan["tensors"] if t["storage"] == "arena"]
            assert len(set(offsets)) < len(offsets), plan
        if name == "native_fused_matmul":
            assert plan["linalg_fused_producers"] >= 1, plan
        if name.startswith("native_matmul_"):
            assert not selected and len(native) == 1, plan
        if name.startswith("triton_matmul_"):
            assert len(selected) == 1 and not native, plan
            config = selected[0]["triton"]
            a_shape = list(config["a_shape"])
            b_shape = list(config["b_shape"])
            c_shape = list(expected[0].shape)
            m, k = a_shape[-2:]
            n = b_shape[-1]
            block_k = 1 << (k - 1).bit_length()
            batches = int(np.prod(c_shape[:-2])) if len(c_shape) > 2 else 1
            batch_mode = ("rank2" if len(c_shape) == 2 else
                          "shared-rhs" if len(b_shape) == 2 else
                          "matched-prefix")
            assert config == {
                "kind": "matmul", "ttir_file": selected[0]["kernel"] + ".ttir",
                "m": m, "n": n, "k": k, "block_m": 16, "block_n": 16,
                "block_k": block_k, "num_warps": warps, "lanes": 4,
                "block_size": warps * 4,
                "batches": batches, "batch_mode": batch_mode,
                "a_shape": a_shape, "b_shape": b_shape, "c_shape": c_shape,
                "grid_size": (batches * ((m + 15) // 16) *
                              ((n + 15) // 16)),
                "abi": "vortex-u64-pointers"}, config
            if name == "triton_matmul_rank4_shared_rhs_k48_constant":
                # ONNX initializers are materialized as compiler-owned constant
                # tensors, whose physical name is intentionally not tied to the
                # source graph spelling.  Prove the ABI slot is the expected
                # immutable [K,N] buffer instead of hard-coding that name.
                assert len(selected[0]["inputs"]) == 2 and \
                    selected[0]["inputs"][0] == "A", selected[0]
                rhs_name = selected[0]["inputs"][1]
                rhs = next(t for t in plan["tensors"] if t["name"] == rhs_name)
                assert rhs["storage"] == "constant" and \
                    rhs["shape"] == [48, 7] and rhs["dtype"] == "float32", rhs
            else:
                assert selected[0]["inputs"] == ["A", "B"], selected[0]
            ttir = (artifact / "triton" / config["ttir_file"]).read_text()
            assert ttir.count('"tt.dot"') == 1, ttir
            assert ttir.count('"tt.get_program_id"') == 1, ttir
            if name == "triton_matmul_rank4_shared_rhs_k48_constant":
                # Triton requires a power-of-two make_range.  Prove both A and
                # B use the one logical K<48 predicate before the 64-wide dot,
                # with exact zero in every invalid reduction lane.
                assert config["k"] == 48 and config["block_k"] == 64, config
                assert '"tt.make_range"() {end = 64 : i32' in ttir, ttir
                assert ttir.count('"arith.andi"') == 3, ttir
                assert ttir.count("tensor<64xi1>") >= 2, ttir
                assert "dense<0.000000e+00> : tensor<16x64xf32>" in ttir, ttir
                assert "dense<0.000000e+00> : tensor<64x16xf32>" in ttir, ttir
                assert ttir.count("predicate = 2 : i64") == 3, ttir
            elif k in (16, 32):
                # Exact-power-of-two K has only the output row/column mask; it
                # must not accidentally inherit the padded-K reduction mask.
                assert ttir.count('"arith.andi"') == 1, ttir
                assert ttir.count("predicate = 2 : i64") == 2, ttir
        if name.startswith("triton_reduce_sum_"):
            assert len(selected) == 1 and not native, plan
            config = selected[0]["triton"]
            width = inputs["A"].shape[0]
            tile = max(16, 1 << (width - 1).bit_length())
            assert config == {
                "kind": "reduction",
                "ttir_file": selected[0]["kernel"] + ".ttir",
                "width": width, "tile": tile, "num_warps": warps,
                "lanes": 4, "block_size": warps * 4, "grid_size": 1,
                "abi": "vortex-u64-pointers"}, config
            assert selected[0]["inputs"] == ["A"], selected[0]
            ttir = (artifact / "triton" / config["ttir_file"]).read_text()
            assert ttir.count('"tt.reduce"') == 1, ttir
            assert ttir.count('"tt.reduce.return"') == 1, ttir
        if name.startswith("triton_softmax_"):
            assert len(selected) == 1 and not native, plan
            config = selected[0]["triton"]
            shape = inputs["A"].shape
            rows_count = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            width = shape[-1]
            tile = max(16, 1 << (width - 1).bit_length())
            assert config == {
                "kind": "softmax",
                "ttir_file": selected[0]["kernel"] + ".ttir",
                "rows": rows_count, "width": width, "n": width,
                "tile": tile,
                "num_warps": warps, "lanes": 4,
                "block_size": warps * 4, "grid_size": rows_count,
                "abi": "vortex-u64-pointers-i32-n"}, config
            assert selected[0]["inputs"] == ["A"], selected[0]
            ttir = (artifact / "triton" / config["ttir_file"]).read_text()
            assert ttir.count('"tt.reduce"') == 2, ttir
            assert ttir.count('"tt.reduce.return"') == 2, ttir
            assert ttir.count('"tt.get_program_id"') == 1, ttir
            assert ttir.count("arith.muli") == 1, ttir
            assert ttir.count("arith.addi") == 1, ttir
            assert "arith.maxnumf" in ttir and "arith.addf" in ttir, ttir
            assert "math.exp" in ttir and "0xFF800000" in ttir, ttir
            assert "i32" in ttir and '"tt.store"' in ttir, ttir
            vxmlir = (artifact / "triton" /
                      f"{selected[0]['kernel']}.vxmlir").read_text()
            assert vxmlir.count('"llvm.riscv.vx.split.i32"') == 1, vxmlir
            assert vxmlir.count('"llvm.riscv.vx.join.i32"') == 1, vxmlir
        if name.startswith("triton_layer_norm_"):
            assert len(selected) == 1 and not native, plan
            config = selected[0]["triton"]
            shape = inputs["X"].shape
            rows_count = int(np.prod(shape[:-1]))
            width = shape[-1]
            tile = max(16, 1 << (width - 1).bit_length())
            node = model.graph.node[0]
            epsilon = next(attribute.f for attribute in node.attribute
                           if attribute.name == "epsilon")
            epsilon_bits = struct.unpack("<I", struct.pack("<f", epsilon))[0]
            assert config == {
                "kind": "layer_norm",
                "ttir_file": selected[0]["kernel"] + ".ttir",
                "rows": rows_count, "width": width, "n": width,
                "tile": tile, "epsilon_f32_bits": epsilon_bits,
                "num_warps": warps, "lanes": 4,
                "block_size": warps * 4, "grid_size": rows_count,
                "abi": "vortex-u64-pointers-i32-n-f32-epsilon"}, config
            assert len(selected[0]["inputs"]) == 3, selected[0]
            assert selected[0]["inputs"][0] == "X", selected[0]
            # The complete LayerNorm aggregate owns exactly five internal
            # Linalg operations.  Selecting it must remove that whole set
            # atomically, without swallowing an unrelated producer.
            assert plan["linalg_fused_producers"] == 5, plan
            assert len(plan["tensors"]) == 4, plan
            assert {tensor["name"] for tensor in plan["tensors"]} == {
                *selected[0]["inputs"], "Y"}, plan
            ttir = (artifact / "triton" / config["ttir_file"]).read_text()
            assert ttir.count('"tt.reduce"') == 2, ttir
            assert ttir.count('"tt.reduce.return"') == 2, ttir
            assert ttir.count("arith.addf") >= 3, ttir
            assert "arith.select" in ttir and "arith.mulf" in ttir, ttir
            assert "arith.sitofp" in ttir and "math.sqrt" in ttir, ttir
            assert "i32, f32" in ttir and '"tt.store"' in ttir, ttir
        if name in ("native_softmax_too_wide", "native_layer_norm_too_wide",
                    "triton_softmax_rank2_width_1024",
                    "triton_layer_norm_width_1024"):
            # Width 1024 proves the accepted compile boundary; width 1025
            # proves complete native fallback.  The smaller cases provide the
            # real SimX numerical coverage without turning a contract suite
            # into a long-running scalar boundary workload.
            rows.append({
                "case": name, "status": "passed",
                "validation": "compile-and-selection-only",
                "triton_dispatches": len(selected),
                "native_dispatches": len(native), "block": block,
                "warps": warps, "max_abs_errors": [],
                "runtime_skipped_reason":
                    "wide boundary is compile/selection coverage only",
                "vxbin_sha256": hashlib.sha256(
                    (artifact / "kernel.vxbin").read_bytes()).hexdigest(),
            })
            print(json.dumps(rows[-1]), flush=True)
            continue
        if name in ("native_layer_norm_nan_epsilon",
                    "native_layer_norm_inf_epsilon"):
            rows.append({
                "case": name, "status": "passed",
                "validation": "compile-and-selection-only",
                "triton_dispatches": len(selected),
                "native_dispatches": len(native), "block": block,
                "warps": warps, "max_abs_errors": [],
                "runtime_skipped_reason":
                    "nonfinite epsilon is a Triton-selection rejection case",
                "vxbin_sha256": hashlib.sha256(
                    (artifact / "kernel.vxbin").read_bytes()).hexdigest(),
            })
            print(json.dumps(rows[-1]), flush=True)
            continue
        # Runtime expects inputs and constants in the artifact directory.
        input_hashes = {}
        for i, tensor in enumerate(plan["tensors"]):
            if tensor["storage"] == "input":
                input_file = artifact / f"input_{i}.bin"
                inputs[tensor["name"]].tofile(input_file)
                input_hashes[input_file] = hashlib.sha256(
                    input_file.read_bytes()).hexdigest()
        run([args.runtime, artifact / "plan.json", artifact / "kernel.vxbin", artifact,
             folder / "output", "--trace-dispatches"], folder / "run.log")
        for input_file, digest in input_hashes.items():
            assert hashlib.sha256(input_file.read_bytes()).hexdigest() == digest
        report = json.loads((folder / "output/runtime-report.json").read_text())
        assert report["execution"] == "vortex-device" and not report["host_operator_fallback"]
        if name.startswith("triton_matmul_"):
            launches = [launch for launch in report["launches"]
                        if launch["backend"] == "triton-vortex"]
            assert len(launches) == 1 and launches[0]["kind"] == "matmul", report
            assert launches[0]["batches"] == selected[0]["triton"]["batches"], report
            assert launches[0]["batch_mode"] == \
                selected[0]["triton"]["batch_mode"], report
            assert launches[0]["a_shape"] == selected[0]["triton"]["a_shape"], report
            assert launches[0]["b_shape"] == selected[0]["triton"]["b_shape"], report
            assert launches[0]["c_shape"] == selected[0]["triton"]["c_shape"], report
        if name.startswith("triton_reduce_sum_"):
            launches = [launch for launch in report["launches"]
                        if launch["backend"] == "triton-vortex"]
            width = inputs["A"].shape[0]
            tile = max(16, 1 << (width - 1).bit_length())
            assert len(launches) == 1 and launches[0]["kind"] == "reduction", report
            assert launches[0]["width"] == width and launches[0]["tile"] == tile, report
        if name.startswith("triton_softmax_"):
            launches = [launch for launch in report["launches"]
                        if launch["backend"] == "triton-vortex"]
            shape = inputs["A"].shape
            rows_count = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            width = shape[-1]
            tile = max(16, 1 << (width - 1).bit_length())
            assert len(launches) == 1 and launches[0]["kind"] == "softmax", report
            assert launches[0]["rows"] == rows_count, report
            assert launches[0]["width"] == width and launches[0]["n"] == width, report
            assert launches[0]["tile"] == tile, report
            assert launches[0]["argument_abi"] == "vortex-u64-pointers-i32-n", report
        if name.startswith("triton_layer_norm_"):
            launches = [launch for launch in report["launches"]
                        if launch["backend"] == "triton-vortex"]
            shape = inputs["X"].shape
            rows_count = int(np.prod(shape[:-1]))
            width = shape[-1]
            tile = max(16, 1 << (width - 1).bit_length())
            assert len(launches) == 1 and launches[0]["kind"] == "layer_norm", report
            assert launches[0]["rows"] == rows_count, report
            assert launches[0]["width"] == width and launches[0]["n"] == width, report
            assert launches[0]["tile"] == tile, report
            assert launches[0]["epsilon_f32_bits"] == selected[0]["triton"]["epsilon_f32_bits"], report
            assert launches[0]["argument_abi"] == \
                "vortex-u64-pointers-i32-n-f32-epsilon", report
        errors = []
        actual_outputs = []
        for i, reference in enumerate(expected):
            output_file = folder / f"output/output_{i}.bin"
            assert output_file.stat().st_size == reference.nbytes
            actual = np.fromfile(output_file, dtype=reference.dtype).reshape(reference.shape)
            actual_outputs.append(actual)
            if np.issubdtype(reference.dtype, np.integer):
                np.testing.assert_array_equal(actual, reference)
            else:
                np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5, equal_nan=True)
            finite = np.isfinite(reference) & np.isfinite(actual)
            errors.append(float(np.max(np.abs(actual[finite] - reference[finite]))) if finite.any() else 0)
        if name in ("triton_softmax_nan", "triton_softmax_positive_inf",
                    "triton_softmax_all_negative_inf"):
            assert np.isnan(actual_outputs[0]).all(), actual_outputs[0]
        if name.startswith("triton_softmax_") and \
                np.isfinite(expected[0]).all():
            np.testing.assert_allclose(
                actual_outputs[0].sum(axis=-1),
                np.ones(actual_outputs[0].shape[:-1] or (), np.float32),
                atol=2e-5, rtol=2e-5)
        if name == "triton_softmax_partial_negative_inf":
            actual = actual_outputs[0]
            assert actual[0] == 0 and actual[3] == 0, actual
            assert np.isfinite(actual).all(), actual
            np.testing.assert_allclose(actual.sum(), 1., atol=2e-5, rtol=2e-5)
        if name == "triton_softmax_extreme_finite":
            actual = actual_outputs[0]
            assert np.isfinite(actual).all() and int(np.argmax(actual)) == 0, actual
            np.testing.assert_allclose(actual.sum(), 1., atol=2e-5, rtol=2e-5)
        rows.append({"case": name, "status": "passed", "triton_dispatches": len(selected),
                     "native_dispatches": len(native), "block": block, "warps": warps,
                     "max_abs_errors": errors,
                     "vxbin_sha256": hashlib.sha256((artifact / "kernel.vxbin").read_bytes()).hexdigest()})
        print(json.dumps(rows[-1]), flush=True)

    # Malformed launch metadata must fail before opening a device. The bad
    # driver name makes accidental device access distinguishable from rejection.
    source = args.output / "add_17/artifact"
    plan = json.loads((source / "plan.json").read_text())
    mutations = {
        "grid": lambda p: p["dispatches"][0]["triton"].update(grid_size=999),
        "block": lambda p: p["dispatches"][0]["triton"].update(block_elements=96),
        "abi": lambda p: p["dispatches"][0]["triton"].update(abi="direct-pointers"),
        "missing_metadata": lambda p: p["dispatches"][0].pop("triton"),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(plan)
        mutate(bad)
        reject_in_driver(bad, source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        run([args.runtime, path, source / "kernel.vxbin", source, args.output / "bad-output"],
            args.output / f"invalid_{name}.log", env=env, failure="Triton")
        rows.append({"case": f"reject_{name}", "status": "passed"})

    # Mutate a real multi-batch manifest so driver and runtime independently
    # prove every batch/shape/grid/alias field before device access.
    dot_source = args.output / "triton_matmul_batched_matched/artifact"
    dot_plan = json.loads((dot_source / "plan.json").read_text())
    dot_selector_contract = args.compiler.with_name(
        "gac-triton-batched-matmul-selector-contract")
    assert dot_selector_contract.is_file(), dot_selector_contract
    run([dot_selector_contract],
        args.output / "batched_matmul_selector_contract.log")
    rows.append({"case": "batched_matmul_selector_boundaries",
                 "status": "passed"})
    dot_mutations = {
        "dot_grid": lambda p: p["dispatches"][0]["triton"].update(grid_size=1),
        "dot_k": lambda p: p["dispatches"][0]["triton"].update(k=16),
        "dot_tile": lambda p: p["dispatches"][0]["triton"].update(block_m=8),
        "dot_kind": lambda p: p["dispatches"][0]["triton"].update(kind="unknown"),
        "dot_batches": lambda p: p["dispatches"][0]["triton"].update(batches=1),
        "dot_batch_overflow":
            lambda p: p["dispatches"][0]["triton"].update(batches=0x7fffffff),
        "dot_batch_mode": lambda p: p["dispatches"][0]["triton"].update(
            batch_mode="shared-rhs"),
        "dot_partial_batch_contract":
            lambda p: p["dispatches"][0]["triton"].pop("a_shape"),
        "dot_a_shape": lambda p: p["dispatches"][0]["triton"].update(
            a_shape=[34, 32]),
        "dot_b_shape": lambda p: p["dispatches"][0]["triton"].update(
            b_shape=[2, 16, 38]),
        "dot_c_shape": lambda p: p["dispatches"][0]["triton"].update(
            c_shape=[1, 34, 19]),
        "dot_physical_a_shape": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "A").update(
                shape=[34, 32]),
        "dot_duplicate_inputs": lambda p: p["dispatches"][0]["inputs"].__setitem__(
            1, p["dispatches"][0]["inputs"][0]),
        "dot_output_alias": lambda p: p["dispatches"][0].update(
            output=p["dispatches"][0]["inputs"][0]),
    }
    for name, mutate in dot_mutations.items():
        bad = copy.deepcopy(dot_plan)
        mutate(bad)
        reject_in_driver(bad, dot_source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        failure = "dispatch must define" if name == "dot_output_alias" else "Triton"
        run([args.runtime, path, dot_source / "kernel.vxbin", dot_source,
             args.output / "bad-output"], args.output / f"invalid_{name}.log",
            env=env, failure=failure)
        rows.append({"case": f"reject_{name}", "status": "passed"})

    # K=48 is represented by a 64-wide physical dot.  Bind the manifest to
    # that exact padded contract rather than accepting logical or oversized
    # block_k values, and independently reject a K/shape disagreement.
    k48_source = args.output / \
        "triton_matmul_rank4_shared_rhs_k48_constant/artifact"
    k48_plan = json.loads((k48_source / "plan.json").read_text())
    k48_mutations = {
        "dot_k48_block_k_logical":
            lambda p: p["dispatches"][0]["triton"].update(block_k=48),
        "dot_k48_block_k_oversized":
            lambda p: p["dispatches"][0]["triton"].update(block_k=128),
        "dot_k48_shape_mismatch":
            lambda p: p["dispatches"][0]["triton"].update(k=32, block_k=32),
    }
    for name, mutate in k48_mutations.items():
        bad = copy.deepcopy(k48_plan)
        mutate(bad)
        reject_in_driver(bad, k48_source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        run([args.runtime, path, k48_source / "kernel.vxbin", k48_source,
             args.output / "bad-output"], args.output / f"invalid_{name}.log",
            env=env, failure="Triton")
        rows.append({"case": f"reject_{name}", "status": "passed"})

    # Backward compatibility is deliberately narrow: an old rank-two manifest
    # may omit all five batch fields, but a partial contract or a batched tensor
    # without them is rejected above.
    legacy_source = args.output / "triton_matmul_tail/artifact"
    legacy_plan = json.loads((legacy_source / "plan.json").read_text())
    for key in ("batches", "batch_mode", "a_shape", "b_shape", "c_shape"):
        legacy_plan["dispatches"][0]["triton"].pop(key)
    list(driver_module.triton_dispatches(
        legacy_plan, legacy_source / "triton",
        SimpleNamespace(block=128, num_warps=1, lanes=4)))
    legacy_path = args.output / "legacy_rank2_matmul_manifest.json"
    legacy_path.write_text(json.dumps(legacy_plan))
    run([args.runtime, legacy_path, legacy_source / "kernel.vxbin",
         legacy_source, args.output / "legacy-rank2-output"],
        args.output / "legacy_rank2_matmul_manifest.log")
    rows.append({"case": "legacy_rank2_matmul_manifest", "status": "passed"})

    reduction_source = args.output / "triton_reduce_sum_scalar/artifact"
    reduction_plan = json.loads((reduction_source / "plan.json").read_text())
    reduction_mutations = {
        "reduction_grid": lambda p: p["dispatches"][0]["triton"].update(grid_size=2),
        "reduction_width": lambda p: p["dispatches"][0]["triton"].update(width=16),
        "reduction_tile": lambda p: p["dispatches"][0]["triton"].update(tile=32),
    }
    for name, mutate in reduction_mutations.items():
        bad = copy.deepcopy(reduction_plan)
        mutate(bad)
        reject_in_driver(bad, reduction_source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        run([args.runtime, path, reduction_source / "kernel.vxbin",
             reduction_source, args.output / "bad-output"],
            args.output / f"invalid_{name}.log", env=env, failure="Triton")
        rows.append({"case": f"reject_{name}", "status": "passed"})

    softmax_source = args.output / "triton_softmax_rank3_tail_19/artifact"
    softmax_plan = json.loads((softmax_source / "plan.json").read_text())
    softmax_selector_contract = args.compiler.with_name(
        "gac-triton-softmax-selector-contract")
    assert softmax_selector_contract.is_file(), softmax_selector_contract
    run([softmax_selector_contract],
        args.output / "softmax_selector_contract.log")
    rows.append({"case": "softmax_selector_shape_boundaries",
                 "status": "passed"})
    softmax_mutations = {
        "softmax_grid": lambda p: p["dispatches"][0]["triton"].update(grid_size=5),
        "softmax_rows": lambda p: p["dispatches"][0]["triton"].update(rows=5),
        "softmax_rows_missing": lambda p: p["dispatches"][0]["triton"].pop("rows"),
        "softmax_elements_overflow":
            lambda p: p["dispatches"][0]["triton"].update(rows=0x7fffffff),
        "softmax_width": lambda p: p["dispatches"][0]["triton"].update(width=17),
        "softmax_n": lambda p: p["dispatches"][0]["triton"].update(n=18),
        "softmax_tile": lambda p: p["dispatches"][0]["triton"].update(tile=64),
        "softmax_abi": lambda p: p["dispatches"][0]["triton"].update(
            abi="vortex-u64-pointers"),
        "softmax_input_shape": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "A").update(
                shape=[2, 2, 19], bytes=2 * 2 * 19 * 4),
        "softmax_output_shape": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "Y").update(
                shape=[3, 2, 19]),
        "softmax_output_bytes": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "Y").update(
                bytes=452),
        "softmax_output_alias": lambda p: p["dispatches"][0].update(
            output=p["dispatches"][0]["inputs"][0]),
    }
    for name, mutate in softmax_mutations.items():
        bad = copy.deepcopy(softmax_plan)
        mutate(bad)
        reject_in_driver(bad, softmax_source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        failure = ("tensor shape/dtype/byte count disagree"
                   if name == "softmax_output_bytes" else
                   "dispatch must define" if name == "softmax_output_alias"
                   else "missing integer: rows"
                   if name == "softmax_rows_missing"
                   else "Triton")
        run([args.runtime, path, softmax_source / "kernel.vxbin",
             softmax_source, args.output / "bad-output"],
            args.output / f"invalid_{name}.log", env=env, failure=failure)
        rows.append({"case": f"reject_{name}", "status": "passed"})

    layer_norm_source = args.output / "triton_layer_norm_tail_19/artifact"
    layer_norm_plan = json.loads(
        (layer_norm_source / "plan.json").read_text())
    selector_contract = args.compiler.with_name(
        "gac-triton-layernorm-selector-contract")
    assert selector_contract.is_file(), selector_contract
    selector_artifacts = args.output / "layer_norm_selector_contract_artifacts"
    selector_artifacts.mkdir()
    run([selector_contract, layer_norm_source / "02-linalg.mlir",
         selector_artifacts],
        args.output / "layer_norm_selector_contract.log")
    rows.append({"case": "layer_norm_selector_internal_fanout",
                 "status": "passed"})
    layer_norm_inputs = layer_norm_plan["dispatches"][0]["inputs"]
    layer_norm_mutations = {
        "layer_norm_grid": lambda p: p["dispatches"][0]["triton"].update(
            grid_size=7),
        "layer_norm_rows": lambda p: p["dispatches"][0]["triton"].update(
            rows=5),
        "layer_norm_elements_overflow":
            lambda p: p["dispatches"][0]["triton"].update(
                rows=0x7fffffff),
        "layer_norm_width": lambda p: p["dispatches"][0]["triton"].update(
            width=17),
        "layer_norm_n": lambda p: p["dispatches"][0]["triton"].update(n=18),
        "layer_norm_tile": lambda p: p["dispatches"][0]["triton"].update(
            tile=64),
        "layer_norm_epsilon_zero": lambda p: p["dispatches"][0]["triton"].update(
            epsilon_f32_bits=0),
        "layer_norm_epsilon_inf": lambda p: p["dispatches"][0]["triton"].update(
            epsilon_f32_bits=0x7f800000),
        "layer_norm_epsilon_overflow": lambda p: p["dispatches"][0]["triton"].update(
            epsilon_f32_bits=0x100000000),
        "layer_norm_abi": lambda p: p["dispatches"][0]["triton"].update(
            abi="vortex-u64-pointers"),
        "layer_norm_duplicate_affine_inputs":
            lambda p: p["dispatches"][0]["inputs"].__setitem__(
                2, p["dispatches"][0]["inputs"][1]),
        "layer_norm_output_alias":
            lambda p: p["dispatches"][0].update(
                output=p["dispatches"][0]["inputs"][0]),
        "layer_norm_x_shape": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "X").update(
                shape=[3, 38]),
        "layer_norm_scale_shape": lambda p: next(
            tensor for tensor in p["tensors"]
            if tensor["name"] == layer_norm_inputs[1]).update(
                shape=[1, 19]),
        "layer_norm_output_shape": lambda p: next(
            tensor for tensor in p["tensors"] if tensor["name"] == "Y").update(
                shape=[2, 57]),
    }
    for name, mutate in layer_norm_mutations.items():
        bad = copy.deepcopy(layer_norm_plan)
        mutate(bad)
        reject_in_driver(bad, layer_norm_source / "triton")
        path = args.output / f"invalid_{name}.json"
        path.write_text(json.dumps(bad))
        env = dict(os.environ, VORTEX_DRIVER="must-not-open-invalid-plan")
        runtime_failure = ("dispatch must define" if
                           name == "layer_norm_output_alias" else "Triton")
        run([args.runtime, path, layer_norm_source / "kernel.vxbin",
             layer_norm_source, args.output / "bad-output"],
            args.output / f"invalid_{name}.log", env=env,
            failure=runtime_failure)
        rows.append({"case": f"reject_{name}", "status": "passed"})
    report = {"scope": "ONNX/C++ MLIR/Triton/VOLT/Vortex SimX integration; not full BEVFormer or silicon performance",
              "status": "passed", "checks": len(rows), "results": rows}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "checks": len(rows)}))


if __name__ == "__main__":
    main()
