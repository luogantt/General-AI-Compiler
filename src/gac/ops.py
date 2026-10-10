"""Operation contracts, shape inference, and NumPy reference semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .errors import EvaluationError, ShapeInferenceError
from .ir import Dim, Node, TensorSpec


SUPPORTED_OPS = frozenset(
    {
        "MatMul",
        "Add",
        "AddReLU",
        "Sub",
        "Mul",
        "Div",
        "Greater",
        "Less",
        "ReLU",
        "GELU",
        "LayerNorm",
        "Softmax",
        "Identity",
        "MatMulBiasGELU",
        "MSDeformableAttention",
        "Cast",
        "Sigmoid",
        "Log",
        "Sin",
        "Cos",
        "Where",
        "Reshape",
        "Transpose",
        "Concat",
        "Expand",
        "Slice",
        "ReduceSum",
        "ReduceMean",
        "GridSample2D",
        "Conv2D",
        "BatchNorm",
        "MaxPool2D",
        "Clamp",
    }
)


def _result_dtype(*specs: TensorSpec) -> str:
    return np.result_type(*(np.dtype(spec.dtype) for spec in specs)).name


def _require_arity(node: Node, inputs: Sequence[TensorSpec], allowed: set[int]) -> None:
    if len(inputs) not in allowed:
        expected = " or ".join(str(value) for value in sorted(allowed))
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} expects {expected} inputs, got {len(inputs)}"
        )
    if len(node.outputs) != 1:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} expects one output, got {len(node.outputs)}"
        )


def _is_dynamic(dim: Dim) -> bool:
    return not isinstance(dim, int)


def _contract_compatible(left: Dim, right: Dim) -> bool:
    return left == right or _is_dynamic(left) or _is_dynamic(right)


def _broadcast_dim(left: Dim, right: Dim) -> Dim:
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    if _is_dynamic(left) and isinstance(right, int):
        return right
    if _is_dynamic(right) and isinstance(left, int):
        return left
    if _is_dynamic(left) and _is_dynamic(right):
        return None
    raise ShapeInferenceError(f"dimensions {left!r} and {right!r} do not broadcast")


def broadcast_shapes(*shapes: Sequence[Dim]) -> tuple[Dim, ...]:
    """Infer a NumPy-style broadcast shape, retaining useful dynamic dims."""

    result: tuple[Dim, ...] = ()
    for shape in shapes:
        current = tuple(shape)
        rank = max(len(result), len(current))
        padded_left = (1,) * (rank - len(result)) + result
        padded_right = (1,) * (rank - len(current)) + current
        result = tuple(
            _broadcast_dim(left, right)
            for left, right in zip(padded_left, padded_right, strict=True)
        )
    return result


def infer_matmul_shape(left: Sequence[Dim], right: Sequence[Dim]) -> tuple[Dim, ...]:
    """Infer ``numpy.matmul`` shape, including vector promotion rules."""

    left = tuple(left)
    right = tuple(right)
    if not left or not right:
        raise ShapeInferenceError("MatMul inputs must both have rank >= 1")

    left_was_vector = len(left) == 1
    right_was_vector = len(right) == 1
    promoted_left = (1, left[0]) if left_was_vector else left
    promoted_right = (right[0], 1) if right_was_vector else right
    if not _contract_compatible(promoted_left[-1], promoted_right[-2]):
        raise ShapeInferenceError(
            "MatMul contracting dimensions do not match: "
            f"{promoted_left[-1]!r} versus {promoted_right[-2]!r}"
        )
    batch = broadcast_shapes(promoted_left[:-2], promoted_right[:-2])
    result = batch + (promoted_left[-2], promoted_right[-1])
    if left_was_vector:
        result = result[:-2] + result[-1:]
    if right_was_vector:
        result = result[:-1]
    return result


def _normalize_axis(axis: int, rank: int, *, op: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if normalized < 0 or normalized >= rank:
        raise ShapeInferenceError(f"{op} axis {axis} is invalid for rank {rank}")
    return normalized


def _reshape_shape(node: Node, input_shape: Sequence[Dim]) -> tuple[Dim, ...]:
    raw = node.attrs.get("shape")
    if not isinstance(raw, (list, tuple)):
        raise ShapeInferenceError(
            f"Reshape node {node.id!r} requires a list-valued 'shape' attribute"
        )
    result: list[Dim] = []
    inferred_axis: int | None = None
    for axis, dimension in enumerate(raw):
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise ShapeInferenceError(
                f"Reshape node {node.id!r} shape dimensions must be integers"
            )
        dimension = int(dimension)
        if dimension == -1:
            if inferred_axis is not None:
                raise ShapeInferenceError(
                    f"Reshape node {node.id!r} allows at most one inferred dimension"
                )
            inferred_axis = axis
            result.append(None)
        elif dimension <= 0:
            raise ShapeInferenceError(
                f"Reshape node {node.id!r} dimensions must be positive or -1"
            )
        else:
            result.append(dimension)

    input_static = all(isinstance(value, int) for value in input_shape)
    known_output = int(np.prod([value for value in result if isinstance(value, int)], dtype=np.int64))
    if input_static:
        input_size = int(np.prod(input_shape, dtype=np.int64))
        if inferred_axis is not None:
            if known_output == 0 or input_size % known_output:
                raise ShapeInferenceError(
                    f"Reshape node {node.id!r} cannot infer a dimension for {tuple(input_shape)}"
                )
            result[inferred_axis] = input_size // known_output
        elif input_size != known_output:
            raise ShapeInferenceError(
                f"Reshape node {node.id!r} changes element count from {input_size} to {known_output}"
            )
    return tuple(result)


def _transpose_perm(node: Node, rank: int) -> tuple[int, ...]:
    raw = node.attrs.get("perm")
    if raw is None:
        return tuple(reversed(range(rank)))
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != rank
        or any(isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) for axis in raw)
    ):
        raise ShapeInferenceError(
            f"Transpose node {node.id!r} perm must contain exactly {rank} integer axes"
        )
    permutation = tuple(int(axis) for axis in raw)
    if sorted(permutation) != list(range(rank)):
        raise ShapeInferenceError(
            f"Transpose node {node.id!r} perm is not a permutation of 0..{rank - 1}"
        )
    return permutation


def _static_shape_attribute(node: Node, name: str) -> tuple[int, ...]:
    raw = node.attrs.get(name)
    if not isinstance(raw, (list, tuple)):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} requires a list-valued {name!r} attribute"
        )
    shape: list[int] = []
    for dimension in raw:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or int(dimension) <= 0
        ):
            raise ShapeInferenceError(
                f"{node.op} node {node.id!r} {name} must contain positive integers"
            )
        shape.append(int(dimension))
    return tuple(shape)


def _reduction_axes(node: Node, rank: int) -> tuple[int, ...]:
    raw = node.attrs.get("axes", tuple(range(rank)))
    if isinstance(raw, (int, np.integer)) and not isinstance(raw, bool):
        raw = (int(raw),)
    if not isinstance(raw, (list, tuple)) or any(
        isinstance(axis, bool) or not isinstance(axis, (int, np.integer))
        for axis in raw
    ):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} axes must be an integer or integer list"
        )
    axes = tuple(_normalize_axis(int(axis), rank, op=node.op) for axis in raw)
    if len(set(axes)) != len(axes):
        raise ShapeInferenceError(f"{node.op} node {node.id!r} axes must be unique")
    return tuple(sorted(axes))


def _keepdims(node: Node) -> bool:
    value = node.attrs.get("keepdims", True)
    if not isinstance(value, bool):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} keepdims must be boolean"
        )
    return value


def _slice_parameters(
    node: Node, input_shape: Sequence[Dim]
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    starts = node.attrs.get("starts")
    ends = node.attrs.get("ends")
    if not isinstance(starts, (list, tuple)) or not isinstance(ends, (list, tuple)):
        raise ShapeInferenceError(
            f"Slice node {node.id!r} requires list-valued starts and ends"
        )
    axes = node.attrs.get("axes", list(range(len(starts))))
    steps = node.attrs.get("steps", [1] * len(starts))
    if not isinstance(axes, (list, tuple)) or not isinstance(steps, (list, tuple)):
        raise ShapeInferenceError(
            f"Slice node {node.id!r} axes and steps must be lists"
        )
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        raise ShapeInferenceError(
            f"Slice node {node.id!r} starts/ends/axes/steps lengths must match"
        )
    normalized: list[tuple[int, int, int, int]] = []
    seen: set[int] = set()
    for start, end, raw_axis, step in zip(starts, ends, axes, steps, strict=True):
        values = (start, end, raw_axis, step)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise ShapeInferenceError(f"Slice node {node.id!r} attributes must contain integers")
        axis = _normalize_axis(int(raw_axis), len(input_shape), op="Slice")
        if axis in seen:
            raise ShapeInferenceError(f"Slice node {node.id!r} axes must be unique")
        seen.add(axis)
        if int(step) == 0:
            raise ShapeInferenceError(f"Slice node {node.id!r} step cannot be zero")
        dimension = input_shape[axis]
        if not isinstance(dimension, int):
            raise ShapeInferenceError(
                f"Slice node {node.id!r} requires a static sliced dimension"
            )
        begin, stop, stride = slice(int(start), int(end), int(step)).indices(dimension)
        normalized.append((axis, begin, stop, stride))
    slices = [slice(None)] * len(input_shape)
    output = list(input_shape)
    for axis, begin, stop, stride in normalized:
        slices[axis] = slice(begin, stop, stride)
        output[axis] = len(range(begin, stop, stride))
    return tuple(slices), tuple(int(value) if isinstance(value, int) else value for value in output)


def _integer_tuple_attr(
    node: Node,
    name: str,
    *,
    length: int,
    default: Sequence[int] | None = None,
    minimum: int = 1,
) -> tuple[int, ...]:
    raw = node.attrs.get(name, default)
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != length
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
            for value in raw
        )
    ):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} {name} must contain {length} integers >= {minimum}"
        )
    return tuple(int(value) for value in raw)


def _conv_output_dimension(
    input_size: Dim,
    kernel_size: int,
    before: int,
    after: int,
    stride: int,
    dilation: int,
) -> Dim:
    if not isinstance(input_size, int):
        return None
    effective_kernel = dilation * (kernel_size - 1) + 1
    output = (input_size + before + after - effective_kernel) // stride + 1
    if output <= 0:
        raise ShapeInferenceError("convolution or pooling produces a non-positive dimension")
    return output


def _require_floating(node: Node, spec: TensorSpec) -> None:
    if not np.issubdtype(np.dtype(spec.dtype), np.floating):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} requires floating point input, got {spec.dtype}"
        )


def _spatial_shapes(node: Node) -> tuple[tuple[int, int], ...]:
    """Return the static feature-map shapes carried by an MSDA node."""

    raw = node.attrs.get("spatial_shapes")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} requires non-empty spatial_shapes"
        )
    result: list[tuple[int, int]] = []
    for index, pair in enumerate(raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ShapeInferenceError(
                f"{node.op} node {node.id!r} spatial_shapes[{index}] must be [height, width]"
            )
        height, width = pair
        if (
            isinstance(height, bool)
            or isinstance(width, bool)
            or not isinstance(height, (int, np.integer))
            or not isinstance(width, (int, np.integer))
            or int(height) <= 0
            or int(width) <= 0
        ):
            raise ShapeInferenceError(
                f"{node.op} node {node.id!r} spatial shapes must contain positive integers"
            )
        result.append((int(height), int(width)))
    return tuple(result)


def _compatible_dimension(left: Dim, right: Dim, description: str) -> None:
    if not _contract_compatible(left, right):
        raise ShapeInferenceError(
            f"MSDeformableAttention {description} dimensions do not match: "
            f"{left!r} versus {right!r}"
        )


def _infer_ms_deformable_attention(
    node: Node, inputs: Sequence[TensorSpec]
) -> TensorSpec:
    """Infer the canonical BEVFormer multi-scale deformable-attention op.

    The layouts match MMCV's CUDA op after its value projection:
    value [B,S,H,D], locations [B,Q,H,L,P,2], weights [B,Q,H,L,P].
    Coordinates use grid_sample's align_corners=False convention.
    """

    _require_arity(node, inputs, {3})
    value, locations, weights = inputs
    for spec in inputs:
        _require_floating(node, spec)
    if len({spec.dtype for spec in inputs}) != 1:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} requires all inputs to have one dtype"
        )
    if value.rank != 4 or locations.rank != 6 or weights.rank != 5:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} expects ranks [4, 6, 5], got "
            f"[{value.rank}, {locations.rank}, {weights.rank}]"
        )
    if locations.shape[-1] != 2:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} sampling_locations last dimension must be 2"
        )
    shapes = _spatial_shapes(node)
    if locations.shape[3] != len(shapes) or weights.shape[3] != len(shapes):
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} level dimension must equal "
            f"len(spatial_shapes)={len(shapes)}"
        )
    expected_spatial = sum(height * width for height, width in shapes)
    if isinstance(value.shape[1], int) and value.shape[1] != expected_spatial:
        raise ShapeInferenceError(
            f"{node.op} node {node.id!r} value spatial dimension {value.shape[1]} "
            f"does not equal sum(H*W)={expected_spatial}"
        )
    _compatible_dimension(value.shape[0], locations.shape[0], "batch")
    _compatible_dimension(value.shape[0], weights.shape[0], "batch")
    _compatible_dimension(locations.shape[1], weights.shape[1], "query")
    _compatible_dimension(value.shape[2], locations.shape[2], "head")
    _compatible_dimension(value.shape[2], weights.shape[2], "head")
    _compatible_dimension(locations.shape[4], weights.shape[4], "point")
    heads, head_dim = value.shape[2], value.shape[3]
    channel_dim: Dim = heads * head_dim if isinstance(heads, int) and isinstance(head_dim, int) else None
    return TensorSpec(
        (value.shape[0], locations.shape[1], channel_dim),
        value.dtype,
    )


def infer_node(
    node: Node, value_specs: Mapping[str, TensorSpec]
) -> dict[str, TensorSpec]:
    """Infer output specs for one node from already-known input specs."""

    if node.op not in SUPPORTED_OPS:
        raise ShapeInferenceError(f"unsupported operation {node.op!r} at node {node.id!r}")
    try:
        inputs = [value_specs[name] for name in node.inputs]
    except KeyError as exc:
        raise ShapeInferenceError(
            f"node {node.id!r} has no spec for input {exc.args[0]!r}"
        ) from exc

    if node.op == "MatMul":
        _require_arity(node, inputs, {2})
        output = TensorSpec(
            infer_matmul_shape(inputs[0].shape, inputs[1].shape),
            _result_dtype(*inputs),
        )
    elif node.op in {"Add", "AddReLU", "Sub", "Mul", "Div", "Greater", "Less"}:
        _require_arity(node, inputs, {2})
        if node.op in {"Div", "Greater", "Less"}:
            for spec in inputs:
                _require_floating(node, spec)
        output = TensorSpec(
            broadcast_shapes(inputs[0].shape, inputs[1].shape),
            _result_dtype(*inputs),
        )
    elif node.op == "Where":
        _require_arity(node, inputs, {3})
        condition, when_true, when_false = inputs
        if condition.dtype != "int32":
            raise ShapeInferenceError(
                f"Where node {node.id!r} condition must have dtype int32"
            )
        if when_true.dtype != "float32" or when_false.dtype != "float32":
            raise ShapeInferenceError(
                f"Where node {node.id!r} choices must both have dtype float32"
            )
        output = TensorSpec(
            broadcast_shapes(
                condition.shape, when_true.shape, when_false.shape
            ),
            "float32",
        )
    elif node.op in {"ReLU", "Identity"}:
        _require_arity(node, inputs, {1})
        output = inputs[0]
    elif node.op in {"GELU", "Sigmoid", "Log", "Sin", "Cos", "Clamp"}:
        _require_arity(node, inputs, {1})
        _require_floating(node, inputs[0])
        if node.op == "Clamp":
            minimum = node.attrs.get("min")
            maximum = node.attrs.get("max")
            if minimum is None and maximum is None:
                raise ShapeInferenceError(f"Clamp node {node.id!r} requires min or max")
            for name, value in (("min", minimum), ("max", maximum)):
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ShapeInferenceError(
                        f"Clamp node {node.id!r} {name} must be a finite number"
                    )
            if minimum is not None and maximum is not None and float(minimum) > float(maximum):
                raise ShapeInferenceError(f"Clamp node {node.id!r} min exceeds max")
        output = inputs[0]
    elif node.op == "Reshape":
        _require_arity(node, inputs, {1})
        output = TensorSpec(
            _reshape_shape(node, inputs[0].shape),
            inputs[0].dtype,
            "contiguous",
        )
    elif node.op == "Transpose":
        _require_arity(node, inputs, {1})
        permutation = _transpose_perm(node, inputs[0].rank)
        output = TensorSpec(
            tuple(inputs[0].shape[axis] for axis in permutation),
            inputs[0].dtype,
            "contiguous",
        )
    elif node.op == "Concat":
        if not inputs:
            raise ShapeInferenceError(
                f"Concat node {node.id!r} expects at least one input"
            )
        _require_arity(node, inputs, set(range(1, len(inputs) + 1)))
        rank = inputs[0].rank
        if any(spec.rank != rank for spec in inputs):
            raise ShapeInferenceError(
                f"Concat node {node.id!r} requires inputs with equal rank"
            )
        if len({spec.dtype for spec in inputs}) != 1:
            raise ShapeInferenceError(
                f"Concat node {node.id!r} requires inputs with equal dtype"
            )
        axis_value = node.attrs.get("axis", 0)
        if isinstance(axis_value, bool) or not isinstance(axis_value, (int, np.integer)):
            raise ShapeInferenceError(f"Concat node {node.id!r} axis must be an integer")
        axis = _normalize_axis(int(axis_value), rank, op="Concat")
        shape = list(inputs[0].shape)
        for spec in inputs[1:]:
            for index, (left, right) in enumerate(zip(shape, spec.shape, strict=True)):
                if index != axis and not _contract_compatible(left, right):
                    raise ShapeInferenceError(
                        f"Concat node {node.id!r} dimension {index} does not match: "
                        f"{left!r} versus {right!r}"
                    )
        concatenated = [spec.shape[axis] for spec in inputs]
        shape[axis] = (
            sum(concatenated)
            if all(isinstance(value, int) for value in concatenated)
            else None
        )
        output = TensorSpec(tuple(shape), inputs[0].dtype, "contiguous")
    elif node.op == "Expand":
        _require_arity(node, inputs, {1})
        target = _static_shape_attribute(node, "shape")
        inferred = broadcast_shapes(inputs[0].shape, target)
        if inferred != target:
            raise ShapeInferenceError(
                f"Expand node {node.id!r} input shape {inputs[0].shape} cannot expand to {target}"
            )
        output = TensorSpec(target, inputs[0].dtype, "contiguous")
    elif node.op == "Slice":
        _require_arity(node, inputs, {1})
        _, shape = _slice_parameters(node, inputs[0].shape)
        output = TensorSpec(shape, inputs[0].dtype, "contiguous")
    elif node.op in {"ReduceSum", "ReduceMean"}:
        _require_arity(node, inputs, {1})
        _require_floating(node, inputs[0])
        axes = _reduction_axes(node, inputs[0].rank)
        keepdims = _keepdims(node)
        output_shape = tuple(
            1 if axis in axes else dimension
            for axis, dimension in enumerate(inputs[0].shape)
        ) if keepdims else tuple(
            dimension
            for axis, dimension in enumerate(inputs[0].shape)
            if axis not in axes
        )
        output = TensorSpec(output_shape, inputs[0].dtype, "contiguous")
    elif node.op == "GridSample2D":
        _require_arity(node, inputs, {2})
        value, grid = inputs
        for spec in inputs:
            _require_floating(node, spec)
        if value.rank != 4 or grid.rank != 4 or grid.shape[-1] != 2:
            raise ShapeInferenceError(
                f"GridSample2D node {node.id!r} expects NCHW value and [N,OH,OW,2] grid"
            )
        if not _contract_compatible(value.shape[0], grid.shape[0]):
            raise ShapeInferenceError(
                f"GridSample2D node {node.id!r} batch dimensions do not match"
            )
        if value.dtype != grid.dtype:
            raise ShapeInferenceError(
                f"GridSample2D node {node.id!r} requires equal input dtypes"
            )
        if node.attrs.get("mode", "bilinear") not in {"bilinear", "nearest"}:
            raise ShapeInferenceError(
                "GridSample2D supports only bilinear or nearest mode"
            )
        if node.attrs.get("padding_mode", "zeros") != "zeros":
            raise ShapeInferenceError("GridSample2D supports only zero padding")
        if not isinstance(node.attrs.get("align_corners", False), bool):
            raise ShapeInferenceError("GridSample2D align_corners must be boolean")
        output = TensorSpec(
            (value.shape[0], value.shape[1], grid.shape[1], grid.shape[2]),
            value.dtype,
            "contiguous",
        )
    elif node.op == "Conv2D":
        _require_arity(node, inputs, {2, 3})
        value, weight = inputs[:2]
        for spec in inputs:
            _require_floating(node, spec)
        if value.rank != 4 or weight.rank != 4:
            raise ShapeInferenceError(
                f"Conv2D node {node.id!r} expects NCHW input and OIHW weight"
            )
        if len({spec.dtype for spec in inputs}) != 1:
            raise ShapeInferenceError(f"Conv2D node {node.id!r} requires equal dtypes")
        groups = node.attrs.get("groups", 1)
        if isinstance(groups, bool) or not isinstance(groups, (int, np.integer)) or int(groups) <= 0:
            raise ShapeInferenceError(f"Conv2D node {node.id!r} groups must be positive")
        groups = int(groups)
        in_channels, out_channels = value.shape[1], weight.shape[0]
        if isinstance(in_channels, int) and in_channels % groups:
            raise ShapeInferenceError(f"Conv2D node {node.id!r} input channels are not divisible by groups")
        if isinstance(out_channels, int) and out_channels % groups:
            raise ShapeInferenceError(f"Conv2D node {node.id!r} output channels are not divisible by groups")
        expected_weight_channels = in_channels // groups if isinstance(in_channels, int) else None
        if not _contract_compatible(weight.shape[1], expected_weight_channels):
            raise ShapeInferenceError(f"Conv2D node {node.id!r} weight channels do not match groups")
        if len(inputs) == 3 and inputs[2].shape != (out_channels,):
            raise ShapeInferenceError(f"Conv2D node {node.id!r} bias must have shape [out_channels]")
        strides = _integer_tuple_attr(node, "strides", length=2, default=(1, 1))
        dilations = _integer_tuple_attr(node, "dilations", length=2, default=(1, 1))
        pads = _integer_tuple_attr(node, "pads", length=4, default=(0, 0, 0, 0), minimum=0)
        out_h = _conv_output_dimension(value.shape[2], weight.shape[2], pads[0], pads[2], strides[0], dilations[0])
        out_w = _conv_output_dimension(value.shape[3], weight.shape[3], pads[1], pads[3], strides[1], dilations[1])
        output = TensorSpec((value.shape[0], out_channels, out_h, out_w), value.dtype, "contiguous")
    elif node.op == "BatchNorm":
        _require_arity(node, inputs, {5})
        value, scale, bias, mean, variance = inputs
        for spec in inputs:
            _require_floating(node, spec)
        if value.rank < 2:
            raise ShapeInferenceError(f"BatchNorm node {node.id!r} input rank must be at least 2")
        if len({spec.dtype for spec in inputs}) != 1:
            raise ShapeInferenceError(f"BatchNorm node {node.id!r} requires equal dtypes")
        channel_shape = (value.shape[1],)
        if any(spec.shape != channel_shape for spec in (scale, bias, mean, variance)):
            raise ShapeInferenceError(f"BatchNorm node {node.id!r} parameters must have shape [C]")
        epsilon = node.attrs.get("epsilon", 1e-5)
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0
        ):
            raise ShapeInferenceError(f"BatchNorm node {node.id!r} epsilon must be finite and positive")
        output = TensorSpec(value.shape, value.dtype, "contiguous")
    elif node.op == "MaxPool2D":
        _require_arity(node, inputs, {1})
        value = inputs[0]
        _require_floating(node, value)
        if value.rank != 4:
            raise ShapeInferenceError(f"MaxPool2D node {node.id!r} expects NCHW input")
        kernel = _integer_tuple_attr(node, "kernel_shape", length=2)
        strides = _integer_tuple_attr(node, "strides", length=2, default=kernel)
        dilations = _integer_tuple_attr(node, "dilations", length=2, default=(1, 1))
        pads = _integer_tuple_attr(node, "pads", length=4, default=(0, 0, 0, 0), minimum=0)
        ceil_mode = node.attrs.get("ceil_mode", False)
        if ceil_mode is not False:
            raise ShapeInferenceError("MaxPool2D currently supports only ceil_mode=False")
        out_h = _conv_output_dimension(value.shape[2], kernel[0], pads[0], pads[2], strides[0], dilations[0])
        out_w = _conv_output_dimension(value.shape[3], kernel[1], pads[1], pads[3], strides[1], dilations[1])
        output = TensorSpec((value.shape[0], value.shape[1], out_h, out_w), value.dtype, "contiguous")
    elif node.op == "Softmax":
        _require_arity(node, inputs, {1})
        _require_floating(node, inputs[0])
        _normalize_axis(int(node.attrs.get("axis", -1)), inputs[0].rank, op="Softmax")
        output = inputs[0]
    elif node.op == "LayerNorm":
        _require_arity(node, inputs, {1, 2, 3})
        _require_floating(node, inputs[0])
        rank = inputs[0].rank
        if rank == 0:
            raise ShapeInferenceError("LayerNorm input must not be scalar")
        axis = _normalize_axis(int(node.attrs.get("axis", -1)), rank, op="LayerNorm")
        normalized_shape = inputs[0].shape[axis:]
        for parameter in inputs[1:]:
            try:
                inferred = broadcast_shapes(normalized_shape, parameter.shape)
            except ShapeInferenceError as exc:
                raise ShapeInferenceError(
                    f"LayerNorm parameter shape {parameter.shape} cannot broadcast "
                    f"to normalized shape {normalized_shape}"
                ) from exc
            if len(inferred) != len(normalized_shape):
                raise ShapeInferenceError(
                    f"LayerNorm parameter rank {parameter.rank} exceeds normalized rank"
                )
        output = TensorSpec(inputs[0].shape, _result_dtype(*inputs))
    elif node.op == "MSDeformableAttention":
        output = _infer_ms_deformable_attention(node, inputs)
    elif node.op == "Cast":
        _require_arity(node, inputs, {1})
        target = node.attrs.get("to")
        try:
            target_dtype = np.dtype(target).name
        except TypeError as exc:
            raise ShapeInferenceError(
                f"Cast node {node.id!r} requires a valid 'to' dtype"
            ) from exc
        if target_dtype not in {"float16", "float32"}:
            raise ShapeInferenceError(
                f"Cast node {node.id!r} supports only float16/float32, got {target_dtype}"
            )
        _require_floating(node, inputs[0])
        output = TensorSpec(inputs[0].shape, target_dtype, inputs[0].layout)
    else:  # MatMulBiasGELU
        _require_arity(node, inputs, {3})
        _require_floating(node, inputs[0])
        matmul_shape = infer_matmul_shape(inputs[0].shape, inputs[1].shape)
        output = TensorSpec(
            broadcast_shapes(matmul_shape, inputs[2].shape),
            _result_dtype(*inputs),
        )

    return {node.outputs[0]: output}


def _gelu_tanh(value: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(np.sqrt(2.0 / np.pi), dtype=value.dtype)
    cubic = np.asarray(0.044715, dtype=value.dtype)
    half = np.asarray(0.5, dtype=value.dtype)
    one = np.asarray(1.0, dtype=value.dtype)
    return half * value * (one + np.tanh(coefficient * (value + cubic * value**3)))


def _layer_norm(inputs: Sequence[np.ndarray], attrs: Mapping[str, Any]) -> np.ndarray:
    value = inputs[0]
    if value.ndim == 0:
        raise EvaluationError("LayerNorm input must not be scalar")
    axis = int(attrs.get("axis", -1))
    if axis < 0:
        axis += value.ndim
    if axis < 0 or axis >= value.ndim:
        raise EvaluationError(f"LayerNorm axis {attrs.get('axis', -1)} is invalid")
    axes = tuple(range(axis, value.ndim))
    epsilon = float(attrs.get("epsilon", 1e-5))
    if epsilon <= 0:
        raise EvaluationError("LayerNorm epsilon must be positive")
    mean = np.mean(value, axis=axes, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=axes, keepdims=True)
    result = (value - mean) / np.sqrt(variance + epsilon)
    if len(inputs) >= 2:
        result = result * inputs[1]
    if len(inputs) >= 3:
        result = result + inputs[2]
    return result


def _evaluation_spatial_shapes(attrs: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    raw = attrs.get("spatial_shapes")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise EvaluationError("MSDeformableAttention requires non-empty spatial_shapes")
    result: list[tuple[int, int]] = []
    for pair in raw:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in pair)
        ):
            raise EvaluationError(
                "MSDeformableAttention spatial_shapes must contain positive [height, width] pairs"
            )
        result.append((int(pair[0]), int(pair[1])))
    return tuple(result)


def _ms_deformable_attention(
    inputs: Sequence[np.ndarray], attrs: Mapping[str, Any]
) -> np.ndarray:
    """Vectorized FP32 oracle for MMCV MSDeformAttn sampling semantics."""

    if len(inputs) != 3:
        raise EvaluationError("MSDeformableAttention expects three inputs")
    value, locations, weights = inputs
    shapes = _evaluation_spatial_shapes(attrs)
    if value.ndim != 4 or locations.ndim != 6 or weights.ndim != 5:
        raise EvaluationError(
            "MSDeformableAttention expects value/location/weight ranks [4, 6, 5]"
        )
    batch, spatial_size, heads, head_dim = value.shape
    if locations.shape[-1] != 2:
        raise EvaluationError("MSDeformableAttention locations last dimension must be 2")
    if (
        locations.shape[0] != batch
        or weights.shape[0] != batch
        or locations.shape[1] != weights.shape[1]
        or locations.shape[2] != heads
        or weights.shape[2] != heads
        or locations.shape[3] != len(shapes)
        or weights.shape[3] != len(shapes)
        or locations.shape[4] != weights.shape[4]
        or spatial_size != sum(height * width for height, width in shapes)
    ):
        raise EvaluationError("MSDeformableAttention input shapes are inconsistent")
    if not all(np.issubdtype(array.dtype, np.floating) for array in inputs):
        raise EvaluationError("MSDeformableAttention inputs must be floating point")

    queries, points = locations.shape[1], locations.shape[4]
    accumulator = np.zeros((batch, queries, heads, head_dim), dtype=np.float32)
    batch_index = np.arange(batch)[:, None, None]
    head_index = np.arange(heads)[None, None, :]
    level_start = 0

    for level, (height, width) in enumerate(shapes):
        for point in range(points):
            coordinates = locations[:, :, :, level, point, :].astype(np.float32, copy=False)
            x = coordinates[..., 0] * width - np.float32(0.5)
            y = coordinates[..., 1] * height - np.float32(0.5)
            x0 = np.floor(x).astype(np.int64)
            y0 = np.floor(y).astype(np.int64)
            x1, y1 = x0 + 1, y0 + 1
            point_weight = weights[:, :, :, level, point].astype(np.float32, copy=False)[..., None]

            for ix, iy, wx, wy in (
                (x0, y0, x1 - x, y1 - y),
                (x1, y0, x - x0, y1 - y),
                (x0, y1, x1 - x, y - y0),
                (x1, y1, x - x0, y - y0),
            ):
                valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                safe_x = np.clip(ix, 0, width - 1)
                safe_y = np.clip(iy, 0, height - 1)
                spatial_index = level_start + safe_y * width + safe_x
                sampled = value[batch_index, spatial_index, head_index, :].astype(
                    np.float32, copy=False
                )
                coefficient = (wx * wy)[..., None].astype(np.float32, copy=False)
                accumulator += sampled * coefficient * point_weight * valid[..., None]
        level_start += height * width

    return accumulator.reshape(batch, queries, heads * head_dim).astype(value.dtype, copy=False)


def evaluate_op(
    op: str,
    inputs: Sequence[np.ndarray],
    attrs: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, ...]:
    """Execute one supported op using deterministic NumPy semantics."""

    attrs = attrs or {}
    arrays = tuple(np.asarray(value) for value in inputs)
    try:
        if op == "MatMul":
            if len(arrays) != 2:
                raise EvaluationError("MatMul expects two inputs")
            result = np.matmul(arrays[0], arrays[1])
        elif op == "Add":
            if len(arrays) != 2:
                raise EvaluationError("Add expects two inputs")
            result = np.add(arrays[0], arrays[1])
        elif op == "AddReLU":
            if len(arrays) != 2:
                raise EvaluationError("AddReLU expects two inputs")
            summed = np.add(arrays[0], arrays[1])
            result = np.maximum(summed, np.asarray(0, dtype=summed.dtype))
        elif op == "Sub":
            if len(arrays) != 2:
                raise EvaluationError("Sub expects two inputs")
            result = np.subtract(arrays[0], arrays[1])
        elif op == "Mul":
            if len(arrays) != 2:
                raise EvaluationError("Mul expects two inputs")
            result = np.multiply(arrays[0], arrays[1])
        elif op == "Div":
            if len(arrays) != 2:
                raise EvaluationError("Div expects two inputs")
            result = np.divide(arrays[0], arrays[1])
        elif op == "Greater":
            if len(arrays) != 2:
                raise EvaluationError("Greater expects two inputs")
            dtype = np.result_type(arrays[0].dtype, arrays[1].dtype)
            result = np.greater(arrays[0], arrays[1]).astype(dtype)
        elif op == "Less":
            if len(arrays) != 2:
                raise EvaluationError("Less expects two inputs")
            dtype = np.result_type(arrays[0].dtype, arrays[1].dtype)
            result = np.less(arrays[0], arrays[1]).astype(dtype)
        elif op == "Where":
            if len(arrays) != 3:
                raise EvaluationError("Where expects condition, true, and false inputs")
            condition, when_true, when_false = arrays
            if condition.dtype != np.int32:
                raise EvaluationError("Where condition must have dtype int32")
            if when_true.dtype != np.float32 or when_false.dtype != np.float32:
                raise EvaluationError("Where choices must both have dtype float32")
            result = np.where(condition != 0, when_true, when_false)
        elif op == "ReLU":
            if len(arrays) != 1:
                raise EvaluationError("ReLU expects one input")
            result = np.maximum(arrays[0], np.asarray(0, dtype=arrays[0].dtype))
        elif op == "GELU":
            if len(arrays) != 1:
                raise EvaluationError("GELU expects one input")
            result = _gelu_tanh(arrays[0])
        elif op == "Sigmoid":
            if len(arrays) != 1:
                raise EvaluationError("Sigmoid expects one input")
            # Stable form avoids overflow warnings for large negative values.
            positive = arrays[0] >= 0
            result = np.empty_like(arrays[0])
            result[positive] = 1 / (1 + np.exp(-arrays[0][positive]))
            exponential = np.exp(arrays[0][~positive])
            result[~positive] = exponential / (1 + exponential)
        elif op == "Log":
            if len(arrays) != 1:
                raise EvaluationError("Log expects one input")
            result = np.log(arrays[0])
        elif op == "Sin":
            if len(arrays) != 1:
                raise EvaluationError("Sin expects one input")
            result = np.sin(arrays[0])
        elif op == "Cos":
            if len(arrays) != 1:
                raise EvaluationError("Cos expects one input")
            result = np.cos(arrays[0])
        elif op == "Clamp":
            if len(arrays) != 1:
                raise EvaluationError("Clamp expects one input")
            minimum = attrs.get("min")
            maximum = attrs.get("max")
            if minimum is None and maximum is None:
                raise EvaluationError("Clamp requires min or max")
            result = np.clip(arrays[0], minimum, maximum)
        elif op == "Reshape":
            if len(arrays) != 1:
                raise EvaluationError("Reshape expects one input")
            raw_shape = attrs.get("shape")
            if not isinstance(raw_shape, (list, tuple)):
                raise EvaluationError("Reshape requires a list-valued 'shape' attribute")
            result = np.reshape(arrays[0], tuple(int(value) for value in raw_shape)).copy()
        elif op == "Transpose":
            if len(arrays) != 1:
                raise EvaluationError("Transpose expects one input")
            raw_perm = attrs.get("perm")
            permutation = None if raw_perm is None else tuple(int(value) for value in raw_perm)
            result = np.transpose(arrays[0], axes=permutation).copy()
        elif op == "Concat":
            if not arrays:
                raise EvaluationError("Concat expects at least one input")
            axis = int(attrs.get("axis", 0))
            result = np.concatenate(arrays, axis=axis)
        elif op == "Expand":
            if len(arrays) != 1:
                raise EvaluationError("Expand expects one input")
            raw_shape = attrs.get("shape")
            if not isinstance(raw_shape, (list, tuple)):
                raise EvaluationError("Expand requires a list-valued 'shape' attribute")
            result = np.broadcast_to(arrays[0], tuple(int(value) for value in raw_shape)).copy()
        elif op == "Slice":
            if len(arrays) != 1:
                raise EvaluationError("Slice expects one input")
            synthetic = Node("evaluation", "Slice", ("x",), ("y",), attrs)
            _slice_parameters(synthetic, arrays[0].shape)
            starts = attrs["starts"]
            ends = attrs["ends"]
            axes = attrs.get("axes", list(range(len(starts))))
            steps = attrs.get("steps", [1] * len(starts))
            slices = [slice(None)] * arrays[0].ndim
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
                normalized_axis = int(axis) + arrays[0].ndim if int(axis) < 0 else int(axis)
                slices[normalized_axis] = slice(int(start), int(end), int(step))
            result = arrays[0][tuple(slices)].copy()
        elif op in {"ReduceSum", "ReduceMean"}:
            if len(arrays) != 1:
                raise EvaluationError(f"{op} expects one input")
            synthetic = Node("evaluation", op, ("x",), ("y",), attrs)
            axes = _reduction_axes(synthetic, arrays[0].ndim)
            keepdims = _keepdims(synthetic)
            if op == "ReduceSum":
                result = np.sum(arrays[0], axis=axes, keepdims=keepdims, dtype=arrays[0].dtype)
            else:
                result = np.mean(arrays[0], axis=axes, keepdims=keepdims, dtype=arrays[0].dtype)
        elif op == "GridSample2D":
            if len(arrays) != 2:
                raise EvaluationError("GridSample2D expects two inputs")
            value, grid = arrays
            if value.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2:
                raise EvaluationError("GridSample2D expects NCHW value and [N,OH,OW,2] grid")
            mode = attrs.get("mode", "bilinear")
            if mode not in {"bilinear", "nearest"} or attrs.get("padding_mode", "zeros") != "zeros":
                raise EvaluationError(
                    "GridSample2D supports bilinear or nearest mode with zero padding"
                )
            align_corners = attrs.get("align_corners", False)
            if not isinstance(align_corners, bool):
                raise EvaluationError("GridSample2D align_corners must be boolean")
            batch, channels, height, width = value.shape
            if grid.shape[0] != batch:
                raise EvaluationError("GridSample2D batch dimensions do not match")
            out_h, out_w = grid.shape[1:3]
            result = np.zeros((batch, channels, out_h, out_w), dtype=value.dtype)
            for batch_index in range(batch):
                for out_y in range(out_h):
                    for out_x in range(out_w):
                        grid_x, grid_y = grid[batch_index, out_y, out_x].astype(np.float32)
                        if align_corners:
                            x = (grid_x + 1.0) * (width - 1) * 0.5
                            y = (grid_y + 1.0) * (height - 1) * 0.5
                        else:
                            x = (grid_x + 1.0) * width * 0.5 - 0.5
                            y = (grid_y + 1.0) * height * 0.5 - 0.5
                        if mode == "nearest":
                            # PyTorch grid_sample uses nearbyint for nearest
                            # interpolation, i.e. round-to-nearest-even under
                            # the default floating-point environment.
                            source_x, source_y = int(np.rint(x)), int(np.rint(y))
                            if 0 <= source_x < width and 0 <= source_y < height:
                                result[batch_index, :, out_y, out_x] = value[
                                    batch_index, :, source_y, source_x
                                ]
                        else:
                            x0, y0 = int(np.floor(x)), int(np.floor(y))
                            for source_x, source_y, coefficient in (
                                (x0, y0, (x0 + 1 - x) * (y0 + 1 - y)),
                                (x0 + 1, y0, (x - x0) * (y0 + 1 - y)),
                                (x0, y0 + 1, (x0 + 1 - x) * (y - y0)),
                                (x0 + 1, y0 + 1, (x - x0) * (y - y0)),
                            ):
                                if 0 <= source_x < width and 0 <= source_y < height:
                                    result[batch_index, :, out_y, out_x] += (
                                        value[batch_index, :, source_y, source_x]
                                        * np.asarray(coefficient, dtype=value.dtype)
                                    )
        elif op == "Conv2D":
            if len(arrays) not in {2, 3}:
                raise EvaluationError("Conv2D expects value, weight, and optional bias")
            value, weight = arrays[:2]
            synthetic = Node("evaluation", "Conv2D", tuple(f"x{i}" for i in range(len(arrays))), ("y",), attrs)
            specs = {f"x{i}": TensorSpec(array.shape, array.dtype) for i, array in enumerate(arrays)}
            output_spec = infer_node(synthetic, specs)["y"]
            strides = tuple(int(item) for item in attrs.get("strides", (1, 1)))
            dilations = tuple(int(item) for item in attrs.get("dilations", (1, 1)))
            pads = tuple(int(item) for item in attrs.get("pads", (0, 0, 0, 0)))
            groups = int(attrs.get("groups", 1))
            batch, in_channels, in_h, in_w = value.shape
            out_channels, channels_per_group, kernel_h, kernel_w = weight.shape
            _, _, out_h, out_w = output_spec.shape
            result = np.empty(output_spec.shape, dtype=value.dtype)
            outputs_per_group = out_channels // groups
            for n in range(batch):
                for out_channel in range(out_channels):
                    group = out_channel // outputs_per_group
                    channel_start = group * channels_per_group
                    for out_y in range(out_h):
                        for out_x in range(out_w):
                            accumulator = float(arrays[2][out_channel]) if len(arrays) == 3 else 0.0
                            for channel_offset in range(channels_per_group):
                                for kernel_y in range(kernel_h):
                                    source_y = out_y * strides[0] - pads[0] + kernel_y * dilations[0]
                                    if source_y < 0 or source_y >= in_h:
                                        continue
                                    for kernel_x in range(kernel_w):
                                        source_x = out_x * strides[1] - pads[1] + kernel_x * dilations[1]
                                        if 0 <= source_x < in_w:
                                            accumulator += float(value[n, channel_start + channel_offset, source_y, source_x]) * float(weight[out_channel, channel_offset, kernel_y, kernel_x])
                            result[n, out_channel, out_y, out_x] = accumulator
        elif op == "BatchNorm":
            if len(arrays) != 5:
                raise EvaluationError("BatchNorm expects value, scale, bias, mean, variance")
            value, scale, bias, mean, variance = arrays
            epsilon = float(attrs.get("epsilon", 1e-5))
            parameter_shape = (1, value.shape[1]) + (1,) * (value.ndim - 2)
            result = (
                (value - mean.reshape(parameter_shape))
                / np.sqrt(variance.reshape(parameter_shape) + epsilon)
                * scale.reshape(parameter_shape)
                + bias.reshape(parameter_shape)
            ).astype(value.dtype, copy=False)
        elif op == "MaxPool2D":
            if len(arrays) != 1:
                raise EvaluationError("MaxPool2D expects one input")
            value = arrays[0]
            synthetic = Node("evaluation", "MaxPool2D", ("x",), ("y",), attrs)
            output_spec = infer_node(synthetic, {"x": TensorSpec(value.shape, value.dtype)})["y"]
            kernel = tuple(int(item) for item in attrs["kernel_shape"])
            strides = tuple(int(item) for item in attrs.get("strides", kernel))
            dilations = tuple(int(item) for item in attrs.get("dilations", (1, 1)))
            pads = tuple(int(item) for item in attrs.get("pads", (0, 0, 0, 0)))
            batch, channels, in_h, in_w = value.shape
            _, _, out_h, out_w = output_spec.shape
            result = np.empty(output_spec.shape, dtype=value.dtype)
            for n in range(batch):
                for channel in range(channels):
                    for out_y in range(out_h):
                        for out_x in range(out_w):
                            maximum = -np.inf
                            for kernel_y in range(kernel[0]):
                                source_y = out_y * strides[0] - pads[0] + kernel_y * dilations[0]
                                for kernel_x in range(kernel[1]):
                                    source_x = out_x * strides[1] - pads[1] + kernel_x * dilations[1]
                                    if 0 <= source_y < in_h and 0 <= source_x < in_w:
                                        maximum = max(maximum, float(value[n, channel, source_y, source_x]))
                            result[n, channel, out_y, out_x] = maximum
        elif op == "LayerNorm":
            if len(arrays) not in {1, 2, 3}:
                raise EvaluationError("LayerNorm expects one to three inputs")
            result = _layer_norm(arrays, attrs)
        elif op == "Softmax":
            if len(arrays) != 1:
                raise EvaluationError("Softmax expects one input")
            axis = int(attrs.get("axis", -1))
            shifted = arrays[0] - np.max(arrays[0], axis=axis, keepdims=True)
            exponent = np.exp(shifted)
            result = exponent / np.sum(exponent, axis=axis, keepdims=True)
        elif op == "Identity":
            if len(arrays) != 1:
                raise EvaluationError("Identity expects one input")
            result = arrays[0].copy()
        elif op == "MatMulBiasGELU":
            if len(arrays) != 3:
                raise EvaluationError("MatMulBiasGELU expects three inputs")
            result = _gelu_tanh(np.matmul(arrays[0], arrays[1]) + arrays[2])
        elif op == "MSDeformableAttention":
            result = _ms_deformable_attention(arrays, attrs)
        elif op == "Cast":
            if len(arrays) != 1:
                raise EvaluationError("Cast expects one input")
            try:
                target_dtype = np.dtype(attrs.get("to")).name
            except TypeError as exc:
                raise EvaluationError("Cast requires a valid 'to' dtype") from exc
            if target_dtype not in {"float16", "float32"}:
                raise EvaluationError("Cast supports only float16/float32")
            result = arrays[0].astype(target_dtype)
        else:
            raise EvaluationError(f"unsupported operation {op!r}")
    except EvaluationError:
        raise
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise EvaluationError(f"failed to evaluate {op}: {exc}") from exc
    return (np.asarray(result),)


def evaluate_node(
    node: Node, values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Execute a node and associate its output names with result arrays."""

    try:
        inputs = [values[name] for name in node.inputs]
    except KeyError as exc:
        raise EvaluationError(
            f"node {node.id!r} is missing input value {exc.args[0]!r}"
        ) from exc
    results = evaluate_op(node.op, inputs, node.attrs)
    if len(results) != len(node.outputs):
        raise EvaluationError(
            f"node {node.id!r} produced {len(results)} results for "
            f"{len(node.outputs)} outputs"
        )
    return dict(zip(node.outputs, results, strict=True))
