"""Explicit eager references and ONNX contracts for the original BEVFormer.

These adapters belong to model export/reference generation, not the compiler.
The exported custom operators are subsequently imported and lowered by C++ MLIR.
No trained operator is replaced by an identity or by a precomputed activation.
"""

from __future__ import annotations

import importlib
import sys

import torch
import torch.nn.functional as F


def torchvision_modulated_deform_conv_forward(
        input, weight, bias, columns, offset, mask, output, ones, *,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, group, deformable_group, with_bias):
    """Implement MMCV's eager extension ABI using real torchvision CPU code.

Both implementations address offset channels as
    deform_group * (2 * Kh * Kw) + 2 * (kh * Kw + kw) + {dy: 0, dx: 1}.
MMCV's original autograd Function and its ONNX symbolic remain in control.
"""
    from torchvision.ops import deform_conv2d

    if input.device.type != "cpu":
        raise ValueError("The torchvision MMCV export adapter is CPU-only")
    if tuple(weight.shape[-2:]) != (kernel_h, kernel_w):
        raise ValueError("MMCV DCNv2 kernel metadata disagrees with weights")
    if input.shape[1] != weight.shape[1] * group:
        raise ValueError("MMCV DCNv2 groups disagree with input/weight channels")
    if offset.shape[1] != 2 * deformable_group * kernel_h * kernel_w:
        raise ValueError("MMCV DCNv2 offset channel count is inconsistent")
    if mask.shape[1] != deformable_group * kernel_h * kernel_w:
        raise ValueError("MMCV DCNv2 modulation mask channel count is inconsistent")
    value = deform_conv2d(input, offset, weight, bias if with_bias else None,
                         stride=(stride_h, stride_w), padding=(pad_h, pad_w),
                         dilation=(dilation_h, dilation_w), mask=mask)
    output.copy_(value)


class ModulatedDeformConvExport(torch.autograd.Function):
    """Trace-safe CPU reference, delegating ONNX semantics to MMCV's symbolic.

    MMCV's extension forward writes into a newly allocated output buffer. A
    Python copy_ adapter nested in that Function triggers a PyTorch 2.0 JIT
    peephole alias error. Returning torchvision's real result avoids that alias
    pattern without changing the trained convolution, offset, or mask.
    """

    @staticmethod
    def forward(ctx, x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups):
        from torchvision.ops import deform_conv2d
        if x.device.type != "cpu":
            raise ValueError("The DCNv2 export reference is CPU-only")
        if x.shape[1] != weight.shape[1] * groups or offset.shape[1] != 2 * deform_groups * weight.shape[2] * weight.shape[3]:
            raise ValueError("DCNv2 groups/offset shape disagrees with the original MMCV contract")
        return deform_conv2d(x.type_as(offset), offset, weight.type_as(offset), bias,
                             stride=stride, padding=padding, dilation=dilation, mask=mask)

    @staticmethod
    def symbolic(graph, x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups):
        from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dFunction
        return ModulatedDeformConv2dFunction.symbolic(
            graph, x, offset, mask, weight, bias, stride, padding, dilation, groups, deform_groups)


def install_dcn_export():
    """Keep original DCNv2 Pack layers/weights; adapt only the op call boundary."""
    module = importlib.import_module("mmcv.ops.modulated_deform_conv")
    module.modulated_deform_conv2d = ModulatedDeformConvExport.apply
    return {"domain": "mmcv", "op_type": "MMCVModulatedDeformConv2d", "opset": 1,
            "reference": "torchvision compiled CPU deform_conv2d",
            "symbolic": "delegates unchanged to MMCV ModulatedDeformConv2dFunction.symbolic",
            "reason": "Avoid PyTorch 2.0 tracing alias error in MMCV output-buffer copy adapter"}


def streaming_ms_deform_attn(value, spatial_shapes, sampling_locations,
                             attention_weights, *, query_chunk=256):
    """Same sampling equation as MMCV, bounded by query chunk and one level.

Inputs: value[B,S,H,D], shapes[L,2], locations[B,Q,H,L,P,2],
weights[B,Q,H,L,P]. Coordinates are normalized [0,1], with bilinear zero-padded
sampling and align_corners=False. Reduction is performed per level then summed;
FP32 rounding can differ slightly from MMCV's single flattened reduction.
"""
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    if any(t.dtype != torch.float32 for t in
           (value, sampling_locations, attention_weights)):
        raise ValueError("The high-level MSDeformAttn export contract is FP32")
    if spatial_shapes.dtype != torch.int64 or spatial_shapes.ndim != 2 or spatial_shapes.shape[1] != 2:
        raise ValueError("MSDeformAttn spatial_shapes must be int64[L,2]")
    batch, sequence, heads, channels = map(int, value.shape)
    if sampling_locations.ndim != 6 or sampling_locations.shape[-1] != 2:
        raise ValueError("MSDeformAttn locations must be [B,Q,H,L,P,2]")
    _, queries, _, levels, points, _ = map(int, sampling_locations.shape)
    shapes = [(int(height), int(width)) for height, width in spatial_shapes.tolist()]
    if (len(shapes) != levels or any(height <= 0 or width <= 0 for height, width in shapes)
            or sum(height * width for height, width in shapes) != sequence):
        raise ValueError("MSDeformAttn levels do not cover the value sequence")
    if tuple(sampling_locations.shape[:1]) != (batch,) or sampling_locations.shape[2] != heads:
        raise ValueError("MSDeformAttn batch/head dimensions disagree")
    if tuple(attention_weights.shape) != (batch, queries, heads, levels, points):
        raise ValueError("MSDeformAttn attention weight shape disagrees")

    feature_levels = []
    start = 0
    for height, width in shapes:
        length = height * width
        feature = value[:, start:start + length].flatten(2).transpose(1, 2)
        feature_levels.append(feature.reshape(batch * heads, channels, height, width))
        start += length
    output = value.new_empty(batch, queries, heads * channels)
    for begin in range(0, queries, query_chunk):
        end = min(begin + query_chunk, queries)
        accumulated = value.new_zeros(batch * heads, channels, end - begin)
        for level, feature in enumerate(feature_levels):
            grid = sampling_locations[:, begin:end, :, level]
            grid = (2 * grid - 1).transpose(1, 2).flatten(0, 1)
            samples = F.grid_sample(feature, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            weights = attention_weights[:, begin:end, :, level]
            weights = weights.transpose(1, 2).reshape(batch * heads, 1, end - begin, points)
            accumulated += (samples * weights).sum(-1)
        output[:, begin:end] = accumulated.reshape(batch, heads * channels, end - begin).transpose(1, 2)
    return output


class MSDeformAttnExport(torch.autograd.Function):
    query_chunk = 256

    @staticmethod
    def forward(ctx, value, spatial_shapes, sampling_locations, attention_weights):
        # The surrounding autograd Function already supplies a real ONNX
        # symbolic. Recording every operation of the chunked eager oracle only
        # builds a discarded inner trace and retains gigabytes at base size.
        # Pause that inner trace while still computing all actual CPU values;
        # restore it before the Function returns its four-input custom op.
        # This scoped PyTorch 2.0 adapter is covered by real ONNX export tests.
        state = torch._C._get_tracing_state()
        try:
            if state is not None:
                torch._C._set_tracing_state(None)
            return streaming_ms_deform_attn(
                value, spatial_shapes, sampling_locations, attention_weights,
                query_chunk=MSDeformAttnExport.query_chunk)
        finally:
            if state is not None:
                torch._C._set_tracing_state(state)

    @staticmethod
    def symbolic(graph, value, spatial_shapes, sampling_locations, attention_weights):
        result = graph.op("gac::MSDeformAttn", value, spatial_shapes,
                          sampling_locations, attention_weights,
                          version_i=1, mode_s="bilinear", padding_mode_s="zeros",
                          align_corners_i=0)
        # Retain static profile dimensions for the rest of the upstream graph.
        vs, ls = value.type().sizes(), sampling_locations.type().sizes()
        if vs is not None and ls is not None and all(v is not None for v in (vs[0], vs[2], vs[3], ls[1])):
            result.setType(value.type().with_sizes([vs[0], ls[1], vs[2] * vs[3]]))
        return result


def install_attention_export(*, query_chunk=256):
    """Redirect all three original CPU attention call sites to one contract."""
    modules = (
        "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention",
        "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention",
        "projects.mmdet3d_plugin.bevformer.modules.decoder",
    )
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    MSDeformAttnExport.query_chunk = query_chunk

    def exported(value, spatial_shapes, sampling_locations, attention_weights):
        return MSDeformAttnExport.apply(value, spatial_shapes, sampling_locations, attention_weights)

    patched = []
    for name in modules:
        module = sys.modules.get(name)
        if module is None or not hasattr(module, "multi_scale_deformable_attn_pytorch"):
            raise RuntimeError(f"The original attention CPU entry point is unavailable: {name}")
        module.multi_scale_deformable_attn_pytorch = exported
        patched.append(name + ".multi_scale_deformable_attn_pytorch")
    return {"domain": "gac", "op_type": "MSDeformAttn", "opset": 1,
            "query_chunk": query_chunk, "patched_original_call_sites": patched,
            "reference": "PyTorch GridSample, query chunks, one level at a time; no activation precomputation",
            "trace_boundary": "Outer custom Function symbolic retained; inner reference executes with tracing temporarily paused",
            "reduction": "sum points per level, then accumulate levels; FP32 rounding checked with tolerance"}


def _deform_grid_reference(x, offset, mask, weight, bias, *, stride, padding, dilation, groups, deform_groups):
    """Small test oracle; never used for the model forward or ONNX export."""
    batch, cin, height, width = x.shape
    cout, per_group, kh, kw = weight.shape
    ho, wo = offset.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(ho), torch.arange(wo), indexing="ij")
    output = x.new_zeros(batch, cout, ho, wo)
    for ci in range(cin):
        dg = ci // (cin // deform_groups)
        group = ci // per_group
        channels = slice(group * (cout // groups), (group + 1) * (cout // groups))
        for ky in range(kh):
            for kx in range(kw):
                point = ky * kw + kx
                channel = dg * kh * kw + point
                y = yy * stride[0] - padding[0] + ky * dilation[0] + offset[:, 2 * channel]
                z = xx * stride[1] - padding[1] + kx * dilation[1] + offset[:, 2 * channel + 1]
                grid = torch.stack((2 * (z + .5) / width - 1, 2 * (y + .5) / height - 1), -1)
                sample = F.grid_sample(x[:, ci:ci + 1], grid, align_corners=False,
                                       padding_mode="zeros", mode="bilinear")
                modulation = mask[:, channel:channel + 1]
                output[:, channels] += sample * modulation * weight[channels, ci % per_group, ky, kx][None, :, None, None]
    return output if bias is None else output + bias[None, :, None, None]


def validate_export_operator_contracts():
    """Numerically check adapters against independent existing implementations."""
    from torchvision.ops import deform_conv2d

    generator = torch.Generator().manual_seed(415)
    results = []
    with torch.no_grad():
        for groups, deform_groups, stride, dilation in ((1, 1, (1, 1), (1, 1)),
                                                       (2, 2, (2, 1), (1, 2))):
            x = torch.randn(1, 4, 5, 7, generator=generator)
            weight = torch.randn(4, 4 // groups, 3, 3, generator=generator)
            bias = torch.randn(4, generator=generator)
            padding = (1, 2)
            ho = (5 + 2 * padding[0] - dilation[0] * 2 - 1) // stride[0] + 1
            wo = (7 + 2 * padding[1] - dilation[1] * 2 - 1) // stride[1] + 1
            offset = torch.randn(1, deform_groups * 18, ho, wo, generator=generator) * .7
            mask = torch.sigmoid(torch.randn(1, deform_groups * 9, ho, wo, generator=generator))
            actual = deform_conv2d(x, offset, weight, bias, stride=stride,
                                   padding=padding, dilation=dilation, mask=mask)
            expected = _deform_grid_reference(x, offset, mask, weight, bias,
                                              stride=stride, padding=padding, dilation=dilation,
                                              groups=groups, deform_groups=deform_groups)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            results.append({"case": f"dcn_groups_{groups}_deform_groups_{deform_groups}",
                            "passed": True, "max_abs_error": float((actual - expected).abs().max())})

        # Keep the upstream MMCV implementation untouched as the independent
        # reference, even when the BEVFormer module-level aliases are patched.
        original = importlib.import_module("mmcv.ops.multi_scale_deform_attn").multi_scale_deformable_attn_pytorch
        for batch, queries, levels in ((2, 7, [(3, 4)]), (2, 11, [(3, 4), (2, 3), (1, 2), (1, 1)])):
            shapes = torch.tensor(levels, dtype=torch.int64)
            value = torch.randn(batch, sum(h * w for h, w in levels), 2, 3, generator=generator)
            locations = torch.rand(batch, queries, 2, len(levels), 4, 2, generator=generator) * 1.6 - .3
            weights = torch.randn(batch, queries, 2, len(levels), 4, generator=generator).softmax(-1)
            expected = original(value, shapes, locations, weights)
            actual = streaming_ms_deform_attn(value, shapes, locations, weights, query_chunk=3)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            results.append({"case": f"msda_{len(levels)}_levels_query_chunks", "passed": True,
                            "max_abs_error": float((actual - expected).abs().max())})
    return {"passed": True, "rtol": 2e-5, "atol": 2e-5, "cases": results,
            "dcn_offset_contract": "offset[:, deform_group*2*Kh*Kw + 2*(kh*Kw+kw) + {dy:0, dx:1}]"}
