"""Portable, correctness-first MLIR kernels for tensor and vector operations.

Each invocation writes one output element. The launch distributes these elements
over Vortex threads; reduction axes remain sequential inside each thread. All
floating-point math accumulates in f32, including f16 storage. This deliberately
simple baseline can later be replaced by tiled LMEM/TCU implementations without
changing graph semantics or the buffer ABI.
"""

from __future__ import annotations

import math

from ..ir import Node, TensorSpec
from ..ops import _reduction_axes, _slice_parameters, _transpose_perm, infer_matmul_shape
from .vortex_scalar_ir import ScalarBuilder, Value


def _read(b: ScalarBuilder, ptr: str, index: Value, spec: TensorSpec) -> Value:
    value = b.load(ptr, index, spec.dtype)
    return b.cast(value, "f32") if spec.dtype.startswith("float") else value


def _relu(b: ScalarBuilder, value: Value) -> Value:
    # Ordered comparison preserves NaN: a NaN input takes the unchanged branch.
    return b.select(b.cmp("lt", value, 0), b.const(0, value.type), value)


def _sigmoid(b: ScalarBuilder, value: Value) -> Value:
    positive = b.cmp("ge", value, 0)
    absolute = b.select(positive, value, b.sub(b.const(0, "f32"), value))
    exponential = b.math("exp", b.mul(absolute, -1.0))
    denominator = b.add(exponential, 1.0)
    return b.select(positive, b.div(b.const(1.0, "f32"), denominator), b.div(exponential, denominator))


def _gelu(b: ScalarBuilder, value: Value) -> Value:
    cubic = b.mul(b.mul(value, value), value)
    argument = b.mul(b.add(value, b.mul(cubic, 0.044715)), math.sqrt(2.0 / math.pi))
    positive = b.cmp("ge", argument, 0)
    absolute = b.select(positive, argument, b.mul(argument, -1.0))
    exponential = b.math("exp", b.mul(absolute, -2.0))
    magnitude = b.div(b.sub(b.const(1.0, "f32"), exponential), b.add(exponential, 1.0))
    tangent = b.select(positive, magnitude, b.mul(magnitude, -1.0))
    return b.mul(b.mul(value, 0.5), b.add(tangent, 1.0))


def _matmul(
    b: ScalarBuilder, ptrs: tuple[str, ...], specs: tuple[TensorSpec, ...],
    tid: Value, output_shape: tuple[int, ...],
) -> Value:
    left, right = specs[:2]
    left_vector, right_vector = left.rank == 1, right.rank == 1
    left_shape = (1, left.shape[0]) if left_vector else left.shape
    right_shape = (right.shape[0], 1) if right_vector else right.shape
    matmul_shape = infer_matmul_shape(left.shape, right.shape)
    index = b.broadcast_index(tid, output_shape, matmul_shape)
    coords = b.coords(index, matmul_shape)
    batch_rank = len(matmul_shape) - (not left_vector) - (not right_vector)
    batch = coords[:batch_rank]
    row = b.const(0) if left_vector else coords[batch_rank]
    column = b.const(0) if right_vector else coords[-1]

    def batch_coords(shape: tuple[int, ...]) -> tuple[Value, ...]:
        aligned = batch[len(batch) - len(shape):] if shape else ()
        return tuple(b.const(0) if dimension == 1 else coord
                     for coord, dimension in zip(aligned, shape, strict=True))

    left_batch, right_batch = batch_coords(left_shape[:-2]), batch_coords(right_shape[:-2])
    compute_type = "f32" if any(spec.dtype.startswith("float") for spec in specs[:2]) else left.dtype

    def product(k: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        left_index = k if left_vector else b.linear((*left_batch, row, k), left_shape)
        right_index = k if right_vector else b.linear((*right_batch, k, column), right_shape)
        a = b.cast(_read(b, ptrs[0], left_index, left), compute_type)
        c = b.cast(_read(b, ptrs[1], right_index, right), compute_type)
        return (b.add(accumulators[0], b.mul(a, c)),)

    return b.loop(left_shape[-1], (b.const(0, compute_type),), product)[0]


def _reduction(
    b: ScalarBuilder, node: Node, ptr: str, spec: TensorSpec,
    tid: Value, output_spec: TensorSpec,
) -> Value:
    axes = _reduction_axes(node, spec.rank)
    keepdims = node.attrs.get("keepdims", True)
    output_coords = b.coords(tid, output_spec.shape)
    fixed: list[Value] = []
    output_axis = 0
    for axis in range(spec.rank):
        if axis in axes:
            fixed.append(b.const(0))
            output_axis += int(keepdims)
        else:
            fixed.append(output_coords[output_axis])
            output_axis += 1
    reduction_shape = tuple(spec.shape[axis] for axis in axes)
    count = math.prod(reduction_shape)

    def body(index: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        coords = list(fixed)
        for axis, coord in zip(axes, b.coords(index, reduction_shape), strict=True):
            coords[axis] = coord
        return (b.add(accumulators[0], b.f32(ptr, b.linear(coords, spec.shape), spec)),)

    result = b.loop(count, (b.const(0, "f32"),), body)[0]
    return b.div(result, float(count)) if node.op == "ReduceMean" else result


def _layer_norm(
    b: ScalarBuilder, node: Node, ptrs: tuple[str, ...], specs: tuple[TensorSpec, ...], tid: Value,
) -> Value:
    spec = specs[0]
    axis = int(node.attrs.get("axis", -1)) % spec.rank
    count = math.prod(spec.shape[axis:])
    base = b.mul(b.div(tid, count), count)

    def sum_body(index: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        return (b.add(accumulators[0], b.f32(ptrs[0], b.add(base, index), spec)),)

    mean = b.div(b.loop(count, (b.const(0, "f32"),), sum_body)[0], float(count))

    def variance_body(index: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        difference = b.sub(b.f32(ptrs[0], b.add(base, index), spec), mean)
        return (b.add(accumulators[0], b.mul(difference, difference)),)

    variance = b.div(b.loop(count, (b.const(0, "f32"),), variance_body)[0], float(count))
    result = b.div(b.sub(b.f32(ptrs[0], tid, spec), mean),
                   b.math("sqrt", b.add(variance, float(node.attrs.get("epsilon", 1e-5)))))
    for parameter_index, operation in ((1, b.mul), (2, b.add)):
        if len(specs) > parameter_index:
            parameter = specs[parameter_index]
            index = b.broadcast_index(tid, spec.shape, parameter.shape)
            result = operation(result, b.f32(ptrs[parameter_index], index, parameter))
    return result


def _softmax(b: ScalarBuilder, node: Node, ptr: str, spec: TensorSpec, tid: Value) -> Value:
    axis = int(node.attrs.get("axis", -1)) % spec.rank
    count, stride = spec.shape[axis], math.prod(spec.shape[axis + 1:])
    base = b.sub(tid, b.mul(b.rem(b.div(tid, stride), count), stride))

    def read(index: Value) -> Value:
        return b.f32(ptr, b.add(base, b.mul(index, stride)), spec)

    def maximum(index: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        value = read(index)
        greater = b.select(b.cmp("gt", value, accumulators[0]), value, accumulators[0])
        return (b.select(b.cmp("ne", value, value), value, greater),)

    max_value = b.loop(count, (b.const(float("-inf"), "f32"),), maximum)[0]

    def denominator(index: Value, accumulators: tuple[Value, ...]) -> tuple[Value, ...]:
        return (b.add(accumulators[0], b.math("exp", b.sub(read(index), max_value))),)

    total = b.loop(count, (b.const(0, "f32"),), denominator)[0]
    return b.div(b.math("exp", b.sub(b.f32(ptr, tid, spec), max_value)), total)


def emit_scalar_op(
    b: ScalarBuilder, node: Node, ptrs: tuple[str, ...], specs: tuple[TensorSpec, ...],
    out: str, out_spec: TensorSpec, tid: Value,
) -> bool:
    """Emit one element's computation; return False for spatial/custom ops.

    Shapes and arity must already have passed ``infer_node`` validation. Pointer
    parameters refer to contiguous global-memory buffers, with physical storage
    matching each TensorSpec. Boolean comparisons store numeric 0/1 in the
    declared output dtype, matching GAC's operation contract.
    """
    op = node.op

    def read(index: int, at: Value | None = None) -> Value:
        spec = specs[index]
        offset = b.broadcast_index(tid, out_spec.shape, spec.shape) if at is None else at
        return _read(b, ptrs[index], offset, spec)

    if op in {"Identity", "Reshape", "Cast"}:
        result = read(0, tid)
    elif op in {"Add", "AddReLU", "Sub", "Mul", "Div", "Greater", "Less"}:
        left, right = read(0), read(1)
        if left.type != right.type:
            compute_type = "f32" if out_spec.dtype.startswith("float") else out_spec.dtype
            left, right = b.cast(left, compute_type), b.cast(right, compute_type)
        if op in {"Greater", "Less"}:
            result = b.cmp("gt" if op == "Greater" else "lt", left, right)
        else:
            operation = {"Add": b.add, "AddReLU": b.add, "Sub": b.sub, "Mul": b.mul, "Div": b.div}[op]
            result = operation(left, right)
            if op == "AddReLU":
                result = _relu(b, result)
    elif op == "Where":
        condition = b.cmp("ne", read(0), 0)
        result = b.select(condition, read(1), read(2))
    elif op in {"ReLU", "GELU", "Sigmoid", "Log", "Sin", "Cos", "Clamp"}:
        result = read(0)
        if op == "ReLU":
            result = _relu(b, result)
        elif op == "GELU":
            result = _gelu(b, result)
        elif op == "Sigmoid":
            result = _sigmoid(b, result)
        elif op in {"Log", "Sin", "Cos"}:
            result = b.math(op.lower(), result)
        else:
            for name, predicate in (("min", "lt"), ("max", "gt")):
                if node.attrs.get(name) is not None:
                    limit = b.const(float(node.attrs[name]), "f32")
                    result = b.select(b.cmp(predicate, result, limit), limit, result)
    elif op in {"MatMul", "MatMulBiasGELU"}:
        result = _matmul(b, ptrs, specs, tid, out_spec.shape)
        if op == "MatMulBiasGELU":
            result = _gelu(b, b.add(result, read(2)))
    elif op == "Transpose":
        coords = b.coords(tid, out_spec.shape)
        original = [b.const(0) for _ in specs[0].shape]
        for axis, input_axis in enumerate(_transpose_perm(node, specs[0].rank)):
            original[input_axis] = coords[axis]
        result = read(0, b.linear(original, specs[0].shape))
    elif op == "Expand":
        result = read(0)
    elif op == "Slice":
        slices, _ = _slice_parameters(node, specs[0].shape)
        coords = list(b.coords(tid, out_spec.shape))
        for axis, item in enumerate(slices):
            start = 0 if item.start is None else item.start
            step = 1 if item.step is None else item.step
            coords[axis] = b.add(b.mul(coords[axis], step), start)
        result = read(0, b.linear(coords, specs[0].shape))
    elif op == "Concat":
        axis = int(node.attrs.get("axis", 0)) % out_spec.rank
        coords = b.coords(tid, out_spec.shape)

        def part(index: int, start: int) -> Value:
            input_coords = list(coords)
            input_coords[axis] = b.sub(coords[axis], start)
            selected = lambda: read(index, b.linear(input_coords, specs[index].shape))
            if index == len(specs) - 1:
                return selected()
            end = start + specs[index].shape[axis]
            # Conditional loads are required: select(load(A), load(B)) would
            # perform an out-of-bounds read from the unselected concat input.
            return b.if_value(b.cmp("lt", coords[axis], end), selected, lambda: part(index + 1, end))

        result = part(0, 0)
    elif op in {"ReduceSum", "ReduceMean"}:
        result = _reduction(b, node, ptrs[0], specs[0], tid, out_spec)
    elif op == "LayerNorm":
        result = _layer_norm(b, node, ptrs, specs, tid)
    elif op == "Softmax":
        result = _softmax(b, node, ptrs[0], specs[0], tid)
    else:
        return False
    b.store(result, out, tid, out_spec.dtype)
    return True
