#!/usr/bin/env python3
"""Independent Conv/DCNv2 fixtures for CUDA output-channel register tiling.

PyTorch/torchvision only generate test references. The production compiler and
generated device kernels remain C++ MLIR -> LLVM -> PTX.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
import torch.nn.functional as functional
import torchvision


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    torch.set_num_threads(1)
    cases = []

    def random(shape):
        return (rng.standard_normal(shape) * 0.125).astype(np.float32)

    def save(name, nodes, inputs, constants, expected, groups, description):
        expected = np.ascontiguousarray(expected)
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info(key, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
             for key, value in inputs.items()],
            [helper.make_tensor_value_info("Y", helper.np_dtype_to_tensor_dtype(expected.dtype), list(expected.shape))],
            [numpy_helper.from_array(value, key) for key, value in constants.items()])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                  helper.make_opsetid("mmcv", 1)], ir_version=9,
                                  producer_name="gac-cuda-convolution-tests")
        onnx.checker.check_model(model)
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        np.savez(folder / "expected.npz", Y=expected)
        per_group = expected.shape[1] // groups
        cases.append({"name": name, "valid": True, "inputs": list(inputs), "outputs": ["Y"],
                      "reference_description": description,
                      "optimized_channel_tiles": [tile for tile in (2, 4, 8)
                          if expected.dtype == np.float32 and per_group % tile == 0]})

    def convolution(name, *, cin=4, cout=8, groups=1, batch=2, height=7, width=9,
                    kernel=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1),
                    relu=False, bias=True, half=False):
        x = random((batch, cin, height, width))
        w = random((cout, cin // groups, *kernel))
        b = random((cout,)) if bias else None
        if half:
            # Dyadic values keep the rounding-boundary fallback check exact.
            x = (np.round(x * 4) / 4).astype(np.float16)
            w = (np.round(w * 4) / 4).astype(np.float16)
            if bias:
                b = (np.round(b * 4) / 4).astype(np.float16)
        with torch.inference_mode():
            y = functional.conv2d(torch.from_numpy(x).float(), torch.from_numpy(w).float(),
                torch.from_numpy(b).float() if bias else None, stride=stride,
                padding=padding, dilation=dilation, groups=groups)
            if relu:
                y = torch.relu(y)
        constants = {"W": w, **({"bias": b} if bias else {})}
        nodes = [helper.make_node("Conv", ["X", "W"] + (["bias"] if bias else []),
                    ["conv" if relu else "Y"], strides=list(stride),
                    pads=list(padding) * 2, dilations=list(dilation), group=groups)]
        if relu:
            nodes.append(helper.make_node("Relu", ["conv"], ["Y"]))
        save(name, nodes, {"X": x}, constants, y.numpy().astype(x.dtype), groups,
             "torch.nn.functional.conv2d CPU FP32 + optional ReLU/output cast")

    convolution("conv_bias_relu_channels", relu=True)
    convolution("conv_grouped_strided_dilated", cin=8, cout=16, groups=2,
                kernel=(2, 3), stride=(2, 1), dilation=(2, 1), padding=(1, 0))
    convolution("conv_depthwise_multiplier", cin=4, cout=16, groups=4, bias=False)
    convolution("conv_odd_group_channels_fallback", cin=4, cout=6, groups=2)
    convolution("conv_fp16_fallback", half=True)
    convolution("conv_1x1_channels", kernel=(1, 1), padding=(0, 0), bias=False)
    convolution("conv_representative_channels", cin=32, cout=64, batch=1, height=32, width=32)

    # A padding step must be skipped, not implemented as 0 * Inf/NaN.
    weights = np.full((8, 1, 3, 3), np.inf, dtype=np.float32)
    weights[:, :, 0, 0] = np.nan
    weights[:, 0, 1, 1] = np.arange(1, 9, dtype=np.float32)
    x = np.array([[[[2]]]], dtype=np.float32)
    y = (2 * np.arange(1, 9, dtype=np.float32)).reshape(1, 8, 1, 1)
    save("conv_padding_nonfinite_weights", [helper.make_node("Conv", ["X", "W"], ["Y"], pads=[1, 1, 1, 1])],
         {"X": x}, {"W": weights}, y, 1,
         "Analytic scalar-compatibility reference: only center weight is in bounds")

    def deform(name, *, cin=4, cout=8, groups=1, deform_groups=1, batch=2,
               height=6, width=7, bias=True, zero=False):
        kernel, stride, padding, dilation = (3, 2), (1, 1), (1, 1), (1, 1)
        oh, ow = height, width + 1
        inputs = {"X": random((batch, cin, height, width)),
                  "offset": rng.uniform(-1.25, 1.25, (batch, 12 * deform_groups, oh, ow)).astype(np.float32),
                  "mask": rng.uniform(0, 1, (batch, 6 * deform_groups, oh, ow)).astype(np.float32)}
        inputs["mask"][..., 0, 0] = 0
        if zero:
            inputs["offset"].fill(0)
            inputs["mask"].fill(1)
        constants = {"W": random((cout, cin // groups, *kernel))}
        if bias:
            constants["bias"] = random((cout,))
        with torch.inference_mode():
            y = torchvision.ops.deform_conv2d(torch.from_numpy(inputs["X"]),
                torch.from_numpy(inputs["offset"]), torch.from_numpy(constants["W"]),
                torch.from_numpy(constants["bias"]) if bias else None,
                stride=stride, padding=padding, dilation=dilation,
                mask=torch.from_numpy(inputs["mask"]))
        node = helper.make_node("MMCVModulatedDeformConv2d",
            ["X", "offset", "mask", "W"] + (["bias"] if bias else []), ["Y"],
            domain="mmcv", stride=list(stride), padding=list(padding), dilation=list(dilation),
            groups=groups, deform_groups=deform_groups)
        save(name, [node], inputs, constants, y.numpy(), groups,
             "torchvision.ops.deform_conv2d compiled CPU operator")

    deform("dcn_shared_samples")
    deform("dcn_groups_and_deform_groups", cin=8, cout=16, groups=2, deform_groups=4)
    deform("dcn_grouped_no_bias", cin=4, cout=16, groups=4, deform_groups=2, bias=False)
    deform("dcn_zero_offsets", zero=True)
    deform("dcn_odd_group_channels_fallback", cin=4, cout=6, groups=2, deform_groups=2)
    deform("dcn_representative_channels", cin=32, cout=64, batch=1, height=16, width=16)
    report = {"cases": cases, "seed": 20261017,
              "reference_versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                     "numpy": np.__version__, "onnx": onnx.__version__}}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.output_dir)
    print(json.dumps({"cases": len(result["cases"]), "output": str(args.output_dir)}))
