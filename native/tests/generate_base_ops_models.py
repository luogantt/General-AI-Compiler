#!/usr/bin/env python3
"""Generate independent DCNv2, Resize and MSDA references for base adaptation.

Run with the existing Windows bev-export-env (PyTorch + compiled torchvision).
These are test fixtures, not a compiler fallback. DCNv2 reference executes
torchvision's compiled CPU operator; MSDA uses PyTorch grid_sample; Resize uses
PyTorch interpolate. The compiler under test remains the native C++ MLIR path.
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


def msda_reference(value, spatial_shapes, locations, weights):
    """Independent small-shape reference matching the published MMCV algorithm.

    This intentionally materializes sampled tensors for tiny test fixtures.
    Production compiler kernels must avoid that expansion for full base shapes.
    Locations are normalized [x, y]; grid_sample receives 2 * locations - 1.
    """
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = locations.shape
    sampled = []
    start = 0
    for level, (height, width) in enumerate(spatial_shapes.tolist()):
        feature = value[:, start:start + height * width]
        feature = feature.permute(0, 2, 3, 1).reshape(batch * heads, channels, height, width)
        grid = locations[:, :, :, level].permute(0, 2, 1, 3, 4)
        grid = grid.reshape(batch * heads, queries, points, 2) * 2 - 1
        sampled.append(functional.grid_sample(feature, grid, mode="bilinear",
                       padding_mode="zeros", align_corners=False))
        start += height * width
    attention = weights.permute(0, 2, 1, 3, 4).reshape(batch * heads, 1, queries, levels * points)
    result = (torch.stack(sampled, dim=-2).flatten(-2) * attention).sum(-1)
    return result.reshape(batch, heads * channels, queries).transpose(1, 2).contiguous()


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    cases = []
    torch.set_num_threads(1)

    def random(shape, scale=0.5):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def save(name, node, inputs, constants, expected=None, *, shape=None, reference=None,
             reject_regex=None, details=None, simx_representative=False):
        inputs = {key: np.ascontiguousarray(value) for key, value in inputs.items()}
        constants = {key: np.ascontiguousarray(value) for key, value in constants.items()}
        output_shape = list(expected.shape) if expected is not None else shape
        graph = helper.make_graph([node], name,
            [helper.make_tensor_value_info(key, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape))
             for key, value in inputs.items()],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, output_shape)],
            [numpy_helper.from_array(value, name=key) for key, value in constants.items()])
        domains = [helper.make_opsetid("", 17)]
        if node.domain:
            domains.append(helper.make_opsetid(node.domain, 1))
        model = helper.make_model(graph, opset_imports=domains, ir_version=9,
                                  producer_name="gac-native-base-ops-reference-tests")
        onnx.checker.check_model(model)
        folder = output / name
        folder.mkdir(exist_ok=True)
        onnx.save(model, folder / "model.onnx")
        np.savez(folder / "inputs.npz", **inputs)
        if expected is not None:
            assert np.isfinite(expected).all(), name
            np.savez(folder / "expected.npz", Y=np.ascontiguousarray(expected))
        cases.append({"name": name, "kind": node.op_type, "valid": expected is not None,
                      "expected_rejection_regex": reject_regex,
                      "reference": reference, "details": details or {},
                      "simx_representative": simx_representative})

    def dcn(name, *, batch=1, cin=2, cout=3, height=5, width=6, kernel=(3, 3),
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, deform_groups=1,
            bias=True, zero_offsets=False, fractional_edges=False, simx=False):
        kh, kw = kernel
        oh = (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        inputs = {"X": random((batch, cin, height, width)),
                  "offset": rng.uniform(-0.8, 0.8, (batch, 2 * deform_groups * kh * kw, oh, ow)).astype(np.float32),
                  "mask": rng.uniform(0.05, 1.0, (batch, deform_groups * kh * kw, oh, ow)).astype(np.float32)}
        constants = {"W": random((cout, cin // groups, kh, kw))}
        if bias:
            constants["bias"] = random((cout,))
        # Offset channels follow dy,dx pairs, independently for every kernel
        # point/deform group. Partial zero masks exercise actual modulation.
        inputs["mask"][..., 0, 0] = 0
        if zero_offsets:
            inputs["offset"].fill(0)
            inputs["mask"].fill(1)
        if fractional_edges:
            assert deform_groups == 1 and kernel == (1, 1) and oh * ow == 6
            inputs["offset"][0, 0] = np.array([[-0.25, -1.0, 0.5], [0.5, 1.0, -0.3]], dtype=np.float32)
            inputs["offset"][0, 1] = np.array([[-0.4, 0.25, 0.5], [-1.0, 0.5, 0.0]], dtype=np.float32)
            inputs["mask"].fill(0.75)
        attrs = dict(stride=list(stride), padding=list(padding), dilation=list(dilation),
                     groups=groups, deform_groups=deform_groups)
        operands = ["X", "offset", "mask", "W"] + (["bias"] if bias else [])
        node = helper.make_node("MMCVModulatedDeformConv2d", operands, ["Y"], name=name,
                                domain="mmcv", **attrs)
        with torch.inference_mode():
            actual = torchvision.ops.deform_conv2d(torch.from_numpy(inputs["X"]),
                torch.from_numpy(inputs["offset"]), torch.from_numpy(constants["W"]),
                torch.from_numpy(constants["bias"]) if bias else None,
                stride=stride, padding=padding, dilation=dilation,
                mask=torch.from_numpy(inputs["mask"]))
            if zero_offsets:
                ordinary = functional.conv2d(torch.from_numpy(inputs["X"]),
                    torch.from_numpy(constants["W"]), torch.from_numpy(constants["bias"]) if bias else None,
                    stride=stride, padding=padding, dilation=dilation, groups=groups)
                torch.testing.assert_close(actual, ordinary, rtol=2e-5, atol=2e-5)
        save(name, node, inputs, constants, actual.numpy(),
             reference="torchvision.ops.deform_conv2d compiled CPU operator",
             details={**attrs, "offset_channel_order": "interleaved dy,dx", "zero_offset_conv_equivalence": zero_offsets},
             simx_representative=simx)
        return node, inputs, constants, list(actual.shape)

    basic = dcn("dcn_mask_bias", simx=True)
    dcn("dcn_batch_grouped", batch=2, cin=4, cout=6, height=6, width=7,
        kernel=(2, 3), stride=(2, 1), padding=(1, 0), dilation=(2, 1), groups=2, deform_groups=2)
    dcn("dcn_four_deform_groups", cin=4, cout=4, height=7, width=8,
        kernel=(3, 2), stride=(1, 2), padding=(2, 1), dilation=(2, 1), deform_groups=4, bias=False)
    dcn("dcn_zero_offset_conv_equivalence", cin=4, cout=4, groups=2, deform_groups=1,
        zero_offsets=True, bias=False)
    dcn("dcn_fractional_zero_padding", cin=1, cout=1, height=2, width=3,
        kernel=(1, 1), padding=(0, 0), fractional_edges=True)
    node, inputs, constants, shape = basic
    bad_offset = {**inputs, "offset": inputs["offset"][:, :-1].copy()}
    save("dcn_reject_offset_channels", node, bad_offset, constants, shape=shape,
         reject_regex="offset|channels|shape")
    bad_group = helper.make_node("MMCVModulatedDeformConv2d", list(node.input), ["Y"],
        name="dcn_reject_groups", domain="mmcv", stride=[1, 1], padding=[1, 1],
        dilation=[1, 1], groups=3, deform_groups=1)
    save("dcn_reject_groups", bad_group, inputs, constants, shape=shape,
         reject_regex="groups|channels|divisib")

    def resize(name, image, *, scales=None, sizes=None, simx=False):
        constants = {}
        if scales is not None:
            constants["scales"] = np.asarray(scales, dtype=np.float32)
            operands = ["X", "", "scales"]
            with torch.inference_mode():
                expected = functional.interpolate(torch.from_numpy(image),
                    scale_factor=tuple(float(x) for x in constants["scales"][2:]),
                    mode="nearest", recompute_scale_factor=False).numpy()
        else:
            constants["sizes"] = np.asarray(sizes, dtype=np.int64)
            operands = ["X", "", "", "sizes"]
            with torch.inference_mode():
                expected = functional.interpolate(torch.from_numpy(image), size=tuple(sizes[2:]), mode="nearest").numpy()
        attrs = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
        node = helper.make_node("Resize", operands, ["Y"], name=name, **attrs)
        save(name, node, {"X": image}, constants, expected,
             reference="torch.nn.functional.interpolate nearest CPU", details=attrs,
             simx_representative=simx)
        return node, {"X": image}, constants, list(expected.shape)

    resized = resize("resize_fpn_2x", random((1, 2, 3, 4)), scales=[1, 1, 2, 2], simx=True)
    resize("resize_sizes_batch", random((2, 3, 3, 5)), sizes=[2, 3, 5, 8])
    resize("resize_noninteger_scales", random((1, 2, 3, 4)), scales=[1, 1, 1.6, 1.7])
    node, inputs, constants, shape = resized
    # Real exporters sometimes encode optional operands as zero-element
    # initializers rather than the empty-string omission. These placeholders
    # have no runtime data and must not become zero-byte device allocations.
    empty_roi = {**constants, "roi": np.empty((0,), dtype=np.float32)}
    empty_roi_node = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"],
        name="resize_empty_roi", mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    with torch.inference_mode():
        empty_expected = functional.interpolate(torch.from_numpy(inputs["X"]), scale_factor=2, mode="nearest").numpy()
    save("resize_empty_roi", empty_roi_node, inputs, empty_roi, empty_expected,
         reference="torch.nn.functional.interpolate nearest CPU; ONNX zero-element ROI placeholder")
    empty_sizes = {"roi": np.empty((0,), dtype=np.float32), "scales": np.empty((0,), dtype=np.float32),
                   "sizes": np.asarray(shape, dtype=np.int64)}
    empty_sizes_node = helper.make_node("Resize", ["X", "roi", "scales", "sizes"], ["Y"],
        name="resize_empty_scales_sizes", mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    save("resize_empty_scales_sizes", empty_sizes_node, inputs, empty_sizes, empty_expected,
         reference="torch.nn.functional.interpolate nearest CPU; ONNX zero-element ROI/scales placeholders")
    for suffix, changes, diagnostic in (
        ("linear", {"mode": "linear"}, "nearest|mode"),
        ("half_pixel", {"coordinate_transformation_mode": "half_pixel"}, "asymmetric|coordinate"),
        ("ceil", {"nearest_mode": "ceil"}, "floor|nearest"),
    ):
        attrs = dict(mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")
        attrs.update(changes)
        bad = helper.make_node("Resize", list(node.input), ["Y"], name="resize_reject_" + suffix, **attrs)
        save("resize_reject_" + suffix, bad, inputs, constants, shape=shape, reject_regex=diagnostic)

    def msda(name, *, batch, heads, channels, queries, points, spatial, known_pixels=False, simx=False):
        spatial = np.asarray(spatial, dtype=np.int64)
        levels = len(spatial)
        elements = int(np.sum(spatial[:, 0] * spatial[:, 1]))
        inputs = {"value": random((batch, elements, heads, channels)),
                  "locations": rng.uniform(-0.25, 1.25, (batch, queries, heads, levels, points, 2)).astype(np.float32),
                  "attention_weights": rng.uniform(0.1, 1, (batch, queries, heads, levels, points)).astype(np.float32)}
        inputs["attention_weights"] /= inputs["attention_weights"].sum(axis=(-1, -2), keepdims=True)
        # Exact boundary and fully out-of-image samples; some weights are zero.
        inputs["locations"].reshape(-1, 2)[:4] = [[0, 0], [1, 1], [-1, 0.5], [2, 0.5]]
        inputs["attention_weights"][0, -1] = 0
        if known_pixels:
            assert (batch, heads, queries, levels, points) == (1, 1, 4, 1, 1)
            inputs["locations"][0, :, 0, 0, 0] = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
            inputs["attention_weights"].fill(1)
        constants = {"spatial_shapes": spatial}
        node = helper.make_node("MSDeformAttn", ["value", "spatial_shapes", "locations", "attention_weights"],
                                ["Y"], name=name, domain="gac")
        with torch.inference_mode():
            expected = msda_reference(torch.from_numpy(inputs["value"]), torch.from_numpy(spatial),
                torch.from_numpy(inputs["locations"]), torch.from_numpy(inputs["attention_weights"])).numpy()
        if known_pixels:
            np.testing.assert_array_equal(expected, inputs["value"].reshape(1, 4, channels))
        save(name, node, inputs, constants, expected,
             reference="torch grid_sample bilinear zeros align_corners=False + weighted sum",
             details={"spatial_shapes": spatial.tolist(), "heads": heads, "channels_per_head": channels,
                      "batch": batch, "points": points, "zero_attention_query": not known_pixels},
             simx_representative=simx)
        return node, inputs, constants, list(expected.shape)

    msda("msda_exact_pixel_centers", batch=1, heads=1, channels=4, queries=4, points=1,
         spatial=[(2, 2)], known_pixels=True)
    multi = msda("msda_multilevel_heads", batch=1, heads=2, channels=3, queries=3,
                 points=2, spatial=[(2, 3), (1, 2)], simx=True)
    msda("msda_batch_three_levels", batch=2, heads=3, channels=2, queries=4,
         points=3, spatial=[(3, 2), (1, 4), (2, 1)])
    node, inputs, constants, shape = multi
    bad_spatial = {"spatial_shapes": constants["spatial_shapes"].copy()}
    bad_spatial["spatial_shapes"][1, 1] += 1
    save("msda_reject_spatial_sum", node, inputs, bad_spatial, shape=shape,
         reject_regex="spatial|sum|value|length")
    bad_heads = {**inputs, "locations": inputs["locations"][:, :, :1].copy()}
    save("msda_reject_head_mismatch", node, bad_heads, constants, shape=shape,
         reject_regex="head|shape|dimension")

    # Independent end-to-end memory test: retain one returned branch while
    # differently sized intermediates die. No allocator offsets are prescribed.
    # The 1.125 KiB scratch budget is feasible by reusing adjacent dead ranges;
    # numerical checks on both outputs detect accidental reuse of the live fork.
    name = "memory_fragmented_fork"
    image = random((4, 16))
    constants = {"W0": random((16, 16)), "W1": random((16, 8)),
                 "W2": random((16, 12)), "bias": random((12,)),
                 "W3": random((24, 4))}
    nodes = [
        helper.make_node("MatMul", ["X", "W0"], ["retained"], name="retained_branch"),
        helper.make_node("MatMul", ["retained", "W1"], ["P"], name="small_projection"),
        helper.make_node("Relu", ["P"], ["Q"], name="fork_activation"),
        helper.make_node("Concat", ["P", "Q"], ["R"], axis=1, name="join_fork"),
        helper.make_node("MatMul", ["R", "W2"], ["S"], name="medium_projection"),
        helper.make_node("Add", ["S", "bias"], ["T"], name="biased_projection"),
        helper.make_node("Concat", ["T", "T"], ["U"], axis=1, name="large_intermediate"),
        helper.make_node("MatMul", ["U", "W3"], ["Y"], name="final_projection"),
    ]
    with torch.inference_mode():
        weights = {key: torch.from_numpy(value) for key, value in constants.items()}
        retained = torch.from_numpy(image) @ weights["W0"]
        projected = retained @ weights["W1"]
        joined = torch.cat((projected, torch.relu(projected)), dim=1)
        biased = joined @ weights["W2"] + weights["bias"]
        final = torch.cat((biased, biased), dim=1) @ weights["W3"]
    graph = helper.make_graph(nodes, name,
        [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [4, 16])],
        [helper.make_tensor_value_info("retained", onnx.TensorProto.FLOAT, [4, 16]),
         helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 4])],
        [numpy_helper.from_array(value, name=key) for key, value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9,
                              producer_name="gac-native-base-ops-reference-tests")
    onnx.checker.check_model(model)
    folder = output / name
    folder.mkdir(exist_ok=True)
    onnx.save(model, folder / "model.onnx")
    np.savez(folder / "inputs.npz", X=image)
    np.savez(folder / "expected.npz", retained=retained.numpy(), Y=final.numpy())
    cases.append({"name": name, "kind": "StaticMemoryForkDAG", "valid": True,
                  "reference": "PyTorch eager matmul/relu/cat/add; two observable outputs",
                  "maximum_arena_bytes": 1152,
                  "details": {"retained_output_bytes": 256,
                              "intermediate_bytes": [128, 128, 256, 192, 192, 384, 64],
                              "constraint": "Both fork output and final output must remain correct within the scratch budget"},
                  "simx_representative": False})

    report = {"schema_version": 1,
              "purpose": "Independent base-operator numerical fixtures; no compiler implementation in Python",
              "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                           "onnx": onnx.__version__, "numpy": np.__version__},
              "cases": cases}
    (output / "cases.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} cases ({sum(case['valid'] for case in cases)} numerical references) in {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
