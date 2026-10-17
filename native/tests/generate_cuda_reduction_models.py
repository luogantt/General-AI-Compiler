#!/usr/bin/env python3
"""Independent ONNX fixtures for native CUDA row-reduction templates.

Python is used only to generate inputs and NumPy reference results. Compiling
and running an operator still uses the C++ importer, MLIR and device PTX path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    cases = []

    def save(name, nodes, x, expected, constants=None,
             *, optimized=True, description="NumPy FP32 two-pass reference"):
        constants = constants or {}
        dtype = helper.np_dtype_to_tensor_dtype(x.dtype)
        graph = helper.make_graph(
            nodes, name,
            [helper.make_tensor_value_info("X", dtype, list(x.shape))],
            [helper.make_tensor_value_info("Y", dtype, list(expected.shape))],
            [numpy_helper.from_array(value, key)
             for key, value in constants.items()],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)],
                                  ir_version=9, producer_name="gac-cuda-reduction-tests")
        onnx.checker.check_model(model)
        directory = output / name
        directory.mkdir(exist_ok=True)
        onnx.save(model, directory / "model.onnx")
        np.savez(directory / "inputs.npz", X=x)
        np.savez(directory / "expected.npz", Y=expected)
        cases.append({"name": name, "valid": True, "inputs": ["X"],
                      "outputs": ["Y"], "expect_cuda_reduction": optimized,
                      "reference_description": description})

    def softmax(name, x, axis=-1, *, reshape=None):
        value = x.reshape(reshape) if reshape else x
        full = value.astype(np.float32)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            exponent = np.exp(full - np.max(full, axis=axis, keepdims=True))
            result = exponent / np.sum(exponent, axis=axis, keepdims=True)
        nodes, constants, source = [], {}, "X"
        if reshape:
            constants["shape"] = np.array(reshape, dtype=np.int64)
            nodes.append(helper.make_node("Reshape", [source, "shape"], ["view"]))
            source = "view"
        nodes.append(helper.make_node("Softmax", [source], ["Y"], axis=axis))
        save(name, nodes, x, result.astype(x.dtype), constants,
             optimized=x.dtype == np.float32,
             description="NumPy stable max-subtracted Softmax on the stated axis")

    def layer_norm(name, x, axis=-1, *, scale=None, bias=None, epsilon=1e-5,
                   reshape=None):
        value = x.reshape(reshape) if reshape else x
        axis = axis if axis >= 0 else value.ndim + axis
        axes = tuple(range(axis, value.ndim))
        full = value.astype(np.float32)
        scale = (np.ones(value.shape[axis:], dtype=x.dtype)
                 if scale is None else np.asarray(scale, dtype=x.dtype))
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            mean = np.mean(full, axis=axes, keepdims=True, dtype=np.float32)
            delta = full - mean
            variance = np.mean(delta * delta, axis=axes, keepdims=True, dtype=np.float32)
            result = delta / np.sqrt(variance + np.float32(epsilon))
            result = result * scale.astype(np.float32)
            if bias is not None:
                result = result + np.asarray(bias, dtype=np.float32)
        nodes, constants, source = [], {"scale": scale}, "X"
        if reshape:
            constants["shape"] = np.array(reshape, dtype=np.int64)
            nodes.append(helper.make_node("Reshape", [source, "shape"], ["view"]))
            source = "view"
        operands = [source, "scale"]
        if bias is not None:
            constants["bias"] = np.asarray(bias, dtype=x.dtype)
            operands.append("bias")
        nodes.append(helper.make_node("LayerNormalization", operands, ["Y"],
                                       axis=axis, epsilon=epsilon, stash_type=1))
        save(name, nodes, x, result.astype(x.dtype), constants,
             optimized=x.dtype == np.float32)

    # Exercise power-of-two boundaries, lane loops, inactive lanes and tails.
    for width in (1, 17, 33, 127, 257, 900, 2049):
        softmax(f"softmax_width_{width}",
                rng.normal(size=(3, width)).astype(np.float32))
    for width in (1, 17, 33, 127, 257, 768, 4097):
        layer_norm(f"layer_norm_width_{width}",
                   rng.normal(size=(3, width)).astype(np.float32),
                   scale=rng.uniform(0.5, 1.5, width).astype(np.float32),
                   bias=rng.normal(size=(width,)).astype(np.float32))
    softmax("softmax_axis_zero", rng.normal(size=(17, 3, 4)).astype(np.float32), 0)
    softmax("softmax_axis_middle", rng.normal(size=(3, 33, 5)).astype(np.float32), 1)
    softmax("softmax_high_magnitude", np.array(
        [[1000, 999, 1001, -1000], [-1000, -1001, -999, -1002]], dtype=np.float32))
    softmax("softmax_shared_view", rng.normal(size=(3, 4, 5)).astype(np.float32),
            0, reshape=(5, 12))
    layer_norm("layer_norm_suffix_axes", rng.normal(size=(3, 5, 7)).astype(np.float32),
               1, scale=rng.normal(size=(5, 7)).astype(np.float32),
               bias=rng.normal(size=(7,)).astype(np.float32))
    layer_norm("layer_norm_scalar_scale_bias", rng.normal(size=(5, 33)).astype(np.float32),
               scale=np.float32(1.75), bias=np.float32(-0.25))
    layer_norm("layer_norm_no_bias", rng.normal(size=(3, 257)).astype(np.float32))
    layer_norm("layer_norm_shared_view", rng.normal(size=(3, 4, 5)).astype(np.float32),
               axis=1, reshape=(5, 12), scale=np.float32(0.75), bias=np.float32(0.25))
    layer_norm("layer_norm_constant_rows", np.full((3, 257), 1000, dtype=np.float32))
    # Cancellation-sensitive case requires centered variance, not E[x²]-E[x]².
    layer_norm("layer_norm_centered_variance",
               np.float32(100) + rng.uniform(-0.25, 0.25, (3, 257)).astype(np.float32))
    special = rng.normal(size=(5, 33)).astype(np.float32)
    special[0, 5], special[1, 7], special[2, 2] = np.nan, np.inf, -np.inf
    special[3, :] = -np.inf
    special[4, 3], special[4, 7] = np.inf, -np.inf
    softmax("softmax_nonfinite", special)
    cases[-1]["expected_nonfinite"] = True
    layer_norm("layer_norm_nonfinite", special)
    cases[-1]["expected_nonfinite"] = True
    softmax("softmax_fp16_fallback", rng.normal(size=(3, 17)).astype(np.float16))
    layer_norm("layer_norm_fp16_fallback", rng.normal(size=(3, 17)).astype(np.float16))
    # Existing native legalization rejects half LayerNorm rather than silently
    # changing ONNX's stash/output rounding contract. Keep that boundary tested.
    cases[-1].update(reject_device=True,
                     expected_error="native LayerNorm currently requires FP32")
    # Representative widths/row counts, still small enough for quick GPU tests.
    softmax("softmax_bev_attention", rng.normal(size=(4096, 32)).astype(np.float32))
    layer_norm("layer_norm_bev_embedding", rng.normal(size=(4096, 256)).astype(np.float32))
    report = {"cases": cases, "seed": 20261017,
              "reference_versions": {"numpy": np.__version__, "onnx": onnx.__version__}}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = generate(args.output_dir)
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output_dir)}))
