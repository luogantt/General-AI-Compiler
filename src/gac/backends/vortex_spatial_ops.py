"""Correctness-first spatial kernels expressed as scalar MLIR, without CUDA.

Each Vortex thread owns one output element. Reductions stay in an FP32
register; input/output buffers use their declared FP16/FP32 storage type.
Padding uses control flow so an invalid coordinate never issues a load.
"""

from __future__ import annotations

import math
from typing import Callable

from ..ir import Node, TensorSpec
from .vortex_scalar_ir import ScalarBuilder, Value


def _inside(b: ScalarBuilder, x: Value, y: Value, width: int, height: int) -> Value:
    x_valid = b.binary("and", b.cmp("ge", x, 0), b.cmp("lt", x, width))
    y_valid = b.binary("and", b.cmp("ge", y, 0), b.cmp("lt", y, height))
    return b.binary("and", x_valid, y_valid)


def _bounded_floor(b: ScalarBuilder, value: Value) -> tuple[Value, Value]:
    """Floor an already bounds-checked coordinate without FP rounding ops.

    RISC-V floor intrinsics can expand to conditional target pseudos that VOLT
    cannot lower in divergent sampling code. Truncate and correct negative
    fractional coordinates with arithmetic/select operations instead.
    """
    truncated = b.cast(value, "i32")
    truncated_float = b.cast(truncated, "f32")
    correction = b.cast(b.cmp("lt", value, truncated_float), "i32")
    integer = b.sub(truncated, correction)
    return integer, b.cast(integer, "f32")


def _bilinear(
    b: ScalarBuilder,
    x: Value,
    y: Value,
    width: int,
    height: int,
    load: Callable[[Value, Value], Value],
) -> Value:
    # Reject distant / nonfinite coordinates before float-to-int conversion.
    valid_x = b.binary("and", b.cmp("gt", x, -1.0), b.cmp("lt", x, width))
    valid_y = b.binary("and", b.cmp("gt", y, -1.0), b.cmp("lt", y, height))

    def sample() -> Value:
        x0, fx = _bounded_floor(b, x)
        y0, fy = _bounded_floor(b, y)
        lx, ly = b.sub(x, fx), b.sub(y, fy)
        hx, hy = b.sub(b.const(1.0, "f32"), lx), b.sub(b.const(1.0, "f32"), ly)
        total = b.const(0.0, "f32")
        for sx, sy, wx, wy in (
            (x0, y0, hx, hy),
            (b.add(x0, 1), y0, lx, hy),
            (x0, b.add(y0, 1), hx, ly),
            (b.add(x0, 1), b.add(y0, 1), lx, ly),
        ):
            value = b.if_value(
                _inside(b, sx, sy, width, height),
                lambda sx=sx, sy=sy: load(sx, sy),
                lambda: b.const(0.0, "f32"),
            )
            total = b.add(total, b.mul(b.mul(value, wx), wy))
        return total

    return b.if_value(
        b.binary("and", valid_x, valid_y), sample, lambda: b.const(0.0, "f32")
    )


def _round_even(b: ScalarBuilder, value: Value) -> Value:
    """Nearest integer with ties to even, including negative half integers."""
    lower, lower_float = _bounded_floor(b, value)
    fraction = b.sub(value, lower_float)
    odd = b.cmp("ne", b.rem(lower, 2), 0)
    round_up = b.binary(
        "or", b.cmp("gt", fraction, 0.5),
        b.binary("and", b.cmp("eq", fraction, 0.5), odd),
    )
    return b.select(round_up, b.add(lower, 1), lower)


def emit_spatial_op(
    b: ScalarBuilder,
    node: Node,
    ptrs: tuple[str, ...],
    specs: tuple[TensorSpec, ...],
    out: str,
    out_spec: TensorSpec,
    tid: Value,
) -> bool:
    """Emit a supported spatial op; return False without emitting otherwise."""
    if node.op not in {"Conv2D", "BatchNorm", "MaxPool2D", "GridSample2D", "MSDeformableAttention"}:
        return False

    if node.op == "Conv2D":
        _, in_channels, height, width = specs[0].shape
        out_channels, channels_per_group, kernel_h, kernel_w = specs[1].shape
        batch, channel, oy, ox = b.coords(tid, out_spec.shape)
        sh, sw = node.attrs.get("strides", (1, 1))
        dh, dw = node.attrs.get("dilations", (1, 1))
        ph, pw, _, _ = node.attrs.get("pads", (0, 0, 0, 0))
        group = b.div(channel, out_channels // int(node.attrs.get("groups", 1)))
        channel_start = b.mul(group, channels_per_group)
        initial = b.f32(ptrs[2], channel, specs[2]) if len(ptrs) == 3 else b.const(0.0, "f32")

        def reduce(index: Value, args: tuple[Value, ...]) -> tuple[Value, ...]:
            ci, ky, kx = b.coords(index, (channels_per_group, kernel_h, kernel_w))
            sy = b.add(b.sub(b.mul(oy, sh), ph), b.mul(ky, dh))
            sx = b.add(b.sub(b.mul(ox, sw), pw), b.mul(kx, dw))

            def product() -> Value:
                source = b.linear((batch, b.add(channel_start, ci), sy, sx), specs[0].shape)
                weight = b.linear((channel, ci, ky, kx), specs[1].shape)
                return b.mul(b.f32(ptrs[0], source, specs[0]), b.f32(ptrs[1], weight, specs[1]))

            term = b.if_value(_inside(b, sx, sy, width, height), product, lambda: b.const(0.0, "f32"))
            return (b.add(args[0], term),)

        result, = b.loop(channels_per_group * kernel_h * kernel_w, (initial,), reduce)

    elif node.op == "BatchNorm":
        channel = b.rem(b.div(tid, math.prod(specs[0].shape[2:])), specs[0].shape[1])
        value = b.f32(ptrs[0], tid, specs[0])
        scale, bias, mean, variance = (b.f32(ptr, channel, spec) for ptr, spec in zip(ptrs[1:], specs[1:], strict=True))
        denominator = b.math("sqrt", b.add(variance, float(node.attrs.get("epsilon", 1e-5))))
        result = b.add(b.mul(b.div(b.sub(value, mean), denominator), scale), bias)

    elif node.op == "MaxPool2D":
        _, _, height, width = specs[0].shape
        batch, channel, oy, ox = b.coords(tid, out_spec.shape)
        kh, kw = node.attrs["kernel_shape"]
        sh, sw = node.attrs.get("strides", (kh, kw))
        dh, dw = node.attrs.get("dilations", (1, 1))
        ph, pw, _, _ = node.attrs.get("pads", (0, 0, 0, 0))

        def maximum(index: Value, args: tuple[Value, ...]) -> tuple[Value, ...]:
            ky, kx = b.coords(index, (kh, kw))
            sy = b.add(b.sub(b.mul(oy, sh), ph), b.mul(ky, dh))
            sx = b.add(b.sub(b.mul(ox, sw), pw), b.mul(kx, dw))
            value = b.if_value(
                _inside(b, sx, sy, width, height),
                lambda: b.f32(ptrs[0], b.linear((batch, channel, sy, sx), specs[0].shape), specs[0]),
                lambda: b.const(-math.inf, "f32"),
            )
            return (b.select(b.cmp("gt", value, args[0]), value, args[0]),)

        result, = b.loop(kh * kw, (b.const(-math.inf, "f32"),), maximum)

    elif node.op == "GridSample2D":
        _, _, height, width = specs[0].shape
        batch, channel, oy, ox = b.coords(tid, out_spec.shape)
        grid_index = b.linear((batch, oy, ox, b.const(0)), specs[1].shape)
        gx, gy = b.f32(ptrs[1], grid_index, specs[1]), b.f32(ptrs[1], b.add(grid_index, 1), specs[1])
        align = node.attrs.get("align_corners", False)
        x = b.mul(b.add(gx, 1.0), (width - 1 if align else width) * 0.5)
        y = b.mul(b.add(gy, 1.0), (height - 1 if align else height) * 0.5)
        if not align:
            x, y = b.sub(x, 0.5), b.sub(y, 0.5)

        def load(sx: Value, sy: Value) -> Value:
            return b.f32(ptrs[0], b.linear((batch, channel, sy, sx), specs[0].shape), specs[0])

        if node.attrs.get("mode", "bilinear") == "nearest":
            valid_x = b.binary("and", b.cmp("ge", x, -1.0), b.cmp("le", x, width))
            valid_y = b.binary("and", b.cmp("ge", y, -1.0), b.cmp("le", y, height))

            def nearest() -> Value:
                sx, sy = _round_even(b, x), _round_even(b, y)
                return b.if_value(_inside(b, sx, sy, width, height), lambda: load(sx, sy), lambda: b.const(0.0, "f32"))

            result = b.if_value(b.binary("and", valid_x, valid_y), nearest, lambda: b.const(0.0, "f32"))
        else:
            result = _bilinear(b, x, y, width, height, load)

    else:  # MSDeformableAttention
        batch_size, spatial_size, heads, head_dim = specs[0].shape
        _, queries, _, levels, points, _ = specs[1].shape
        batch, query, head, channel = b.coords(tid, (batch_size, queries, heads, head_dim))
        result = b.const(0.0, "f32")
        level_start = 0
        for level, (height, width) in enumerate(node.attrs["spatial_shapes"]):
            # Levels are static; points remain a runtime SCF loop.
            def reduce(point: Value, args: tuple[Value, ...]) -> tuple[Value, ...]:
                weight_index = b.linear((batch, query, head, b.const(level), point), specs[2].shape)
                location_index = b.mul(weight_index, 2)
                x = b.sub(b.mul(b.f32(ptrs[1], location_index, specs[1]), float(width)), 0.5)
                y = b.sub(b.mul(b.f32(ptrs[1], b.add(location_index, 1), specs[1]), float(height)), 0.5)

                def load(sx: Value, sy: Value) -> Value:
                    spatial = b.add(b.add(b.mul(sy, width), sx), level_start)
                    source = b.linear((batch, spatial, head, channel), specs[0].shape)
                    return b.f32(ptrs[0], source, specs[0])

                sample = _bilinear(b, x, y, width, height, load)
                weighted = b.mul(sample, b.f32(ptrs[2], weight_index, specs[2]))
                return (b.add(args[0], weighted),)

            result, = b.loop(points, (result,), reduce)
            level_start += height * width

    b.store(result, out, tid, out_spec.dtype)
    return True
