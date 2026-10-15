"""Static FP32 CUDA source generation for the RTX 3070 / SM 8.6 target.

Compilation here emits CUDA C and launch metadata, without importing CuPy or
requiring a GPU. NVRTC compilation to device code happens when a Session opens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from gac.core.errors import BackendError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.memory import infer_zero_copy_view_specs, infer_zero_copy_views
from gac.compiler.passes import (
    AddReLUFusionPass,
    MatMulBiasGELUFusionPass,
    PassManager,
    ShapeInferencePass,
    default_pass_manager,
)


TARGET = "cuda-sm86"
NVRTC_OPTIONS = ("--std=c++11", "--fmad=false")
SUPPORTED_OPS = frozenset({
    "MatMul", "Add", "AddReLU", "Sub", "Mul", "Div", "Greater", "Less", "Where", "GELU", "Sigmoid", "Log", "Sin", "Cos", "ReLU",
    "Identity", "LayerNorm", "Softmax", "MatMulBiasGELU",
    "MSDeformableAttention", "Cast",
    "Reshape", "Transpose", "Concat",
    "Expand", "Slice", "ReduceSum", "ReduceMean", "GridSample2D",
    "Conv2D", "BatchNorm", "MaxPool2D",
    "Clamp",
})


@dataclass(frozen=True)
class KernelLaunch:
    entry_point: str
    node_id: str
    inputs: tuple[str, ...]
    output: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    source_nodes: tuple[str, ...]
    implementation_candidates: tuple[str, ...]


@dataclass
class CudaProgram:
    graph: Graph
    reference_graph: Graph
    source: str
    launches: tuple[KernelLaunch, ...]
    pass_reports: tuple[dict[str, Any], ...]

    def describe(self) -> dict[str, Any]:
        zero_copy_views = infer_zero_copy_view_specs(self.graph)
        return {
            "schema_version": 1,
            "target": TARGET,
            "compute_capability": [8, 6],
            "artifact_kind": "cuda-source-with-launch-metadata",
            "device_code_compiled": False,
            "nvrtc_options": list(NVRTC_OPTIONS),
            "precision": "float32 compute with int32 control tensors and explicit float16/float32 Cast; no TF32 or Tensor Core path",
            "name": self.graph.name,
            "inputs": {name: spec.to_dict() for name, spec in self.graph.inputs.items()},
            "outputs": {name: self.graph.value_specs[name].to_dict() for name in self.graph.outputs},
            "launches": [asdict(launch) for launch in self.launches],
            "zero_copy_views": [
                {"value": value, "source": source, "byte_offset": byte_offset}
                for value, (source, byte_offset) in zero_copy_views.items()
            ],
            "pass_reports": list(self.pass_reports),
            "tensor_bytes_upper_bound": sum(spec.nbytes for spec in self.graph.value_specs.values()),
        }


def _reject(node: Node, reason: str) -> None:
    raise BackendError(f"{TARGET}: node {node.id!r} ({node.op}): {reason}")


def _implementation_candidates(
    node: Node, specs: dict[str, TensorSpec]
) -> tuple[str, ...]:
    """Record deterministic Runtime dispatch choices in launch metadata."""
    if node.op == "Conv2D" and len(node.inputs) in {2, 3}:
        pads = tuple(int(value) for value in node.attrs.get("pads", (0, 0, 0, 0)))
        tensors = [specs[name] for name in (*node.inputs[:2], node.outputs[0])]
        if (
            len(pads) == 4
            and pads[0] == pads[2]
            and pads[1] == pads[3]
            and all(spec.dtype == "float32" for spec in tensors)
            and all(len(spec.shape) == 4 for spec in tensors)
        ):
            return ("cudnn", "generated-cuda")
    if node.op == "MatMul" and len(node.inputs) == 2:
        tensors = [specs[name] for name in (*node.inputs, node.outputs[0])]
        if all(
            spec.dtype == "float32" and len(spec.shape) >= 2
            for spec in tensors
        ):
            return ("cupy-cublas", "generated-cuda")
    return ("generated-cuda",)


def _check_node(node: Node, specs: dict[str, TensorSpec]) -> None:
    if node.op not in SUPPORTED_OPS:
        _reject(node, "operation is not supported")
    if len(node.outputs) != 1:
        _reject(node, "exactly one output is required")
    allowed = {
        "GELU": {"approximation"}, "MatMulBiasGELU": {"approximation"},
        "LayerNorm": {"axis", "epsilon"}, "Softmax": {"axis"},
        "MSDeformableAttention": {"spatial_shapes"},
        "Cast": {"to"},
        "Reshape": {"shape"}, "Transpose": {"perm"}, "Concat": {"axis"},
        "Expand": {"shape"},
        "Slice": {"starts", "ends", "axes", "steps"},
        "ReduceSum": {"axes", "keepdims"}, "ReduceMean": {"axes", "keepdims"},
        "GridSample2D": {"mode", "padding_mode", "align_corners"},
        "Conv2D": {"strides", "pads", "dilations", "groups"},
        "BatchNorm": {"epsilon"},
        "MaxPool2D": {"kernel_shape", "strides", "pads", "dilations", "ceil_mode"},
        "Clamp": {"min", "max"},
    }.get(node.op, set())
    if set(node.attrs) - allowed:
        _reject(node, f"unsupported attributes: {sorted(set(node.attrs) - allowed)}")
    if node.op in {"GELU", "MatMulBiasGELU"} and node.attrs.get("approximation", "tanh") != "tanh":
        _reject(node, "only tanh GELU is supported")
    inputs = [specs[name] for name in node.inputs]
    outputs = [specs[name] for name in node.outputs]
    if node.op == "Where":
        if (
            len(inputs) != 3
            or inputs[0].dtype != "int32"
            or any(spec.dtype != "float32" for spec in (*inputs[1:], *outputs))
        ):
            _reject(node, "requires int32 condition and float32 choices/output")
    elif node.op not in {"Cast"} and any(
        spec.dtype != "float32" for spec in (*inputs, *outputs)
    ):
        _reject(node, "this operation currently requires float32 tensors")
    if node.op == "Cast":
        if len(inputs) != 1 or len(outputs) != 1:
            _reject(node, "Cast requires one input and one output")
        if inputs[0].dtype not in {"float16", "float32"} or outputs[0].dtype not in {"float16", "float32"}:
            _reject(node, "Cast supports only float16 and float32")
        if node.attrs.get("to") != outputs[0].dtype:
            _reject(node, "Cast output dtype must match the 'to' attribute")
    if node.op in {"MatMul", "MatMulBiasGELU"}:
        if inputs[0].rank < 2 or inputs[1].rank < 2:
            _reject(node, "MatMul inputs must have rank >= 2")
        if node.op == "MatMulBiasGELU" and (inputs[0].rank != 2 or inputs[1].rank != 2):
            _reject(node, "the fused MatMulBiasGELU epilogue requires rank-2 inputs")
        if node.op == "MatMulBiasGELU" and inputs[2].shape != (inputs[1].shape[1],):
            _reject(node, "fused bias must have shape [N]; use --no-fuse for other broadcasts")
        if inputs[0].rank == 2 and math.ceil(inputs[0].shape[0] / 16) > 65535:
            _reject(node, "MatMul exceeds the supported grid.y range")
    if node.op in {"LayerNorm", "Softmax"}:
        axis = node.attrs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in {-1, inputs[0].rank - 1}:
            _reject(node, "only the last normalization axis is supported")
        if inputs[0].rank == 0:
            _reject(node, "normalization requires rank >= 1")
    if node.op == "LayerNorm":
        epsilon = node.attrs.get("epsilon", 1e-5)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            _reject(node, "epsilon must be finite and positive")
        if not np.isfinite(np.float32(epsilon)) or np.float32(epsilon) == 0:
            _reject(node, "epsilon must be representable as positive float32")
        if any(spec.shape != (inputs[0].shape[-1],) for spec in inputs[1:]):
            _reject(node, "scale and bias must have shape [last_dimension]")
    if node.op == "MSDeformableAttention":
        # ShapeInferencePass has already validated ranks and cross-input dimensions.
        shapes = node.attrs["spatial_shapes"]
        if len(shapes) > 32:
            _reject(node, "at most 32 feature levels are supported")
        if inputs[1].shape[4] > 64:
            _reject(node, "at most 64 sampling points per level are supported")
    if node.op == "Concat" and len(inputs) < 1:
        _reject(node, "at least one input is required")
    if node.op == "GridSample2D":
        if node.attrs.get("mode", "bilinear") not in {"bilinear", "nearest"}:
            _reject(node, "only bilinear or nearest mode is supported")
        if node.attrs.get("padding_mode", "zeros") != "zeros":
            _reject(node, "only zero padding is supported")
    if node.op == "BatchNorm":
        epsilon = node.attrs.get("epsilon", 1e-5)
        if not np.isfinite(np.float32(epsilon)) or np.float32(epsilon) <= 0:
            _reject(node, "epsilon must be representable as positive float32")
    if node.op == "MaxPool2D" and node.attrs.get("ceil_mode", False) is not False:
        _reject(node, "only ceil_mode=False is supported")


def _check_specs(graph: Graph) -> None:
    for name, spec in graph.value_specs.items():
        if not spec.is_static or any(dim <= 0 for dim in spec.shape):
            raise BackendError(f"{TARGET}: {name!r} requires static, positive dimensions")
        # ``layout`` describes logical axis order.  Runtime binding separately
        # enforces C-contiguous storage, so a dense NCHW tensor is just as valid
        # as an unlabeled contiguous tensor for the direct Conv2D kernels.
        floating_storage = spec.dtype in {"float16", "float32"} and spec.layout in {
            "contiguous", "NCHW"
        }
        integer_control = spec.dtype == "int32" and spec.layout == "contiguous"
        if not (floating_storage or integer_control):
            raise BackendError(
                f"{TARGET}: {name!r} requires C-contiguous float16/float32 "
                "storage with contiguous or NCHW logical layout, or contiguous "
                "int32 control storage"
            )
        if math.prod(spec.shape) > 2**31 - 1:
            raise BackendError(f"{TARGET}: {name!r} exceeds the supported 32-bit indexing range")
    for name, value in graph.constants.items():
        if not np.all(np.isfinite(value)):
            raise BackendError(f"{TARGET}: constant {name!r} must contain finite values")


def _broadcast_index(shape: tuple, output_shape: tuple) -> str:
    padded = (1,) * (len(output_shape) - len(shape)) + shape
    terms = []
    for axis, dimension in enumerate(padded):
        if dimension != 1:
            output_stride = math.prod(output_shape[axis + 1:])
            input_stride = math.prod(padded[axis + 1:])
            terms.append(f"((i / {output_stride}) % {dimension}) * {input_stride}")
    return " + ".join(terms) or "0"


def _matmul_batch_offset(shape: tuple, output_batch: tuple, matrix_size: int) -> str:
    """Map a flattened output batch index to one broadcast input matrix."""

    input_batch = shape[:-2]
    padded = (1,) * (len(output_batch) - len(input_batch)) + input_batch
    terms = []
    for axis, dimension in enumerate(padded):
        if dimension != 1:
            output_stride = math.prod(output_batch[axis + 1:])
            input_stride = math.prod(padded[axis + 1:])
            terms.append(
                f"((batch_index / {output_stride}) % {dimension}) * {input_stride}"
            )
    matrix_index = " + ".join(terms) or "0"
    return f"({matrix_index}) * {matrix_size}"


_PRELUDE = r'''// Generated by GAC: mixed-storage baseline for compute capability 8.6.
// Inputs and outputs are non-aliasing, contiguous device buffers.
#include <cuda_fp16.h>
__device__ float gac_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gac_msda_load(
    const float* value, int batch, int spatial_size, int heads, int head_dim,
    int level_start, int height, int width, int y, int x, int head, int channel) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
    int spatial = level_start + y * width + x;
    int index = (((batch * spatial_size + spatial) * heads + head) * head_dim + channel);
    return value[index];
}
__device__ __forceinline__ float gac_grid_load(
    const float* value, int batch, int channels, int height, int width,
    int channel, int y, int x) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
    return value[((batch * channels + channel) * height + y) * width + x];
}
'''


def _emit_kernel(node: Node, graph: Graph, index: int) -> tuple[str, KernelLaunch]:
    specs = graph.value_specs
    shape = specs[node.outputs[0]].shape
    size = math.prod(shape)
    entry = f"gac_kernel_{index}"
    cuda_type = {"float16": "half", "float32": "float", "int32": "int"}
    args = [
        f"const {cuda_type[specs[name].dtype]}* p{i}"
        for i, name in enumerate(node.inputs)
    ] + [f"{cuda_type[specs[node.outputs[0]].dtype]}* out"]
    header = f'extern "C" __global__ void {entry}({", ".join(args)}) {{\n'
    grid, block = ((size + 255) // 256, 1, 1), (256, 1, 1)
    if node.op == "Cast":
        source_dtype = specs[node.inputs[0]].dtype
        target_dtype = specs[node.outputs[0]].dtype
        if source_dtype == target_dtype:
            expression = "p0[i]"
        elif source_dtype == "float16":
            expression = "__half2float(p0[i])"
        else:
            expression = "__float2half_rn(p0[i])"
        body = f"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < {size}) out[i] = {expression};\n"
    elif node.op == "Reshape":
        body = f"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < {size}) out[i] = p0[i];\n"
    elif node.op == "Transpose":
        input_shape = specs[node.inputs[0]].shape
        permutation = tuple(node.attrs.get("perm", reversed(range(len(input_shape)))))
        input_strides = [math.prod(input_shape[axis + 1:]) for axis in range(len(input_shape))]
        output_strides = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
        terms = []
        for input_axis, input_stride in enumerate(input_strides):
            output_axis = permutation.index(input_axis)
            output_dimension = shape[output_axis]
            terms.append(
                f"((i / {output_strides[output_axis]}) % {output_dimension}) * {input_stride}"
            )
        input_index = " + ".join(terms) or "0"
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}) out[i] = p0[{input_index}];
'''
    elif node.op == "Concat":
        axis = int(node.attrs.get("axis", 0))
        if axis < 0:
            axis += len(shape)
        inner = math.prod(shape[axis + 1:])
        axis_total = shape[axis]
        offset = 0
        branches = []
        for input_index, input_name in enumerate(node.inputs):
            axis_length = specs[input_name].shape[axis]
            keyword = "if" if input_index == 0 else "else if"
            branches.append(
                f"    {keyword} (axis_index < {offset + axis_length}) {{\n"
                f"        int source_index = outer * {axis_length * inner} + "
                f"(axis_index - {offset}) * {inner} + inner_index;\n"
                f"        out[i] = p{input_index}[source_index];\n"
                f"    }}"
            )
            offset += axis_length
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int inner_index = i % {inner};
    int axis_index = (i / {inner}) % {axis_total};
    int outer = i / {axis_total * inner};
{chr(10).join(branches)}
'''
    elif node.op == "Expand":
        source_index = _broadcast_index(specs[node.inputs[0]].shape, shape)
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}) out[i] = p0[{source_index}];
'''
    elif node.op == "Slice":
        input_shape = specs[node.inputs[0]].shape
        starts = node.attrs["starts"]
        ends = node.attrs["ends"]
        axes = node.attrs.get("axes", list(range(len(starts))))
        steps = node.attrs.get("steps", [1] * len(starts))
        normalized = {
            (int(axis) + len(input_shape) if int(axis) < 0 else int(axis)):
            slice(int(start), int(end), int(step)).indices(
                input_shape[int(axis) + len(input_shape) if int(axis) < 0 else int(axis)]
            )
            for start, end, axis, step in zip(starts, ends, axes, steps, strict=True)
        }
        input_strides = [math.prod(input_shape[axis + 1:]) for axis in range(len(input_shape))]
        output_strides = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
        terms = []
        for axis, input_stride in enumerate(input_strides):
            coordinate = f"((i / {output_strides[axis]}) % {shape[axis]})"
            if axis in normalized:
                begin, _, step = normalized[axis]
                coordinate = f"({begin} + {coordinate} * {step})"
            terms.append(f"{coordinate} * {input_stride}")
        source_index = " + ".join(terms) or "0"
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < {size}) out[i] = p0[{source_index}];
'''
    elif node.op in {"ReduceSum", "ReduceMean"}:
        input_shape = specs[node.inputs[0]].shape
        raw_axes = node.attrs.get("axes", list(range(len(input_shape))))
        if isinstance(raw_axes, (int, np.integer)) and not isinstance(raw_axes, bool):
            raw_axes = [raw_axes]
        axes = tuple(sorted(
            int(axis) + len(input_shape) if int(axis) < 0 else int(axis)
            for axis in raw_axes
        ))
        keepdims = node.attrs.get("keepdims", True)
        non_axes = tuple(axis for axis in range(len(input_shape)) if axis not in axes)
        input_strides = [math.prod(input_shape[axis + 1:]) for axis in range(len(input_shape))]
        output_strides = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
        base_terms = []
        for output_axis, input_axis in enumerate(non_axes):
            if keepdims:
                output_axis = input_axis
            coordinate = f"((i / {output_strides[output_axis]}) % {shape[output_axis]})"
            base_terms.append(f"{coordinate} * {input_strides[input_axis]}")
        base_index = " + ".join(base_terms) or "0"
        reduce_shape = tuple(input_shape[axis] for axis in axes)
        reduction_size = math.prod(reduce_shape)
        reduction_terms = []
        for reduced_axis, input_axis in enumerate(axes):
            reduction_stride = math.prod(reduce_shape[reduced_axis + 1:])
            reduction_terms.append(
                f"((r / {reduction_stride}) % {input_shape[input_axis]}) * {input_strides[input_axis]}"
            )
        reduction_offset = " + ".join(reduction_terms) or "0"
        result = f"accumulator / {reduction_size}.0f" if node.op == "ReduceMean" else "accumulator"
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int base_index = {base_index};
    float accumulator = 0.0f;
    for (int r = 0; r < {reduction_size}; ++r) {{
        int reduction_offset = {reduction_offset};
        accumulator += p0[base_index + reduction_offset];
    }}
    out[i] = {result};
'''
    elif node.op == "GridSample2D":
        batch, channels, height, width = specs[node.inputs[0]].shape
        _, out_h, out_w, _ = specs[node.inputs[1]].shape
        align_corners = node.attrs.get("align_corners", False)
        if align_corners:
            x_expression = f"(grid_x + 1.0f) * {width - 1}.0f * 0.5f"
            y_expression = f"(grid_y + 1.0f) * {height - 1}.0f * 0.5f"
        else:
            x_expression = f"(grid_x + 1.0f) * {width}.0f * 0.5f - 0.5f"
            y_expression = f"(grid_y + 1.0f) * {height}.0f * 0.5f - 0.5f"
        common = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int cursor = i;
    int out_x = cursor % {out_w}; cursor /= {out_w};
    int out_y = cursor % {out_h}; cursor /= {out_h};
    int channel = cursor % {channels};
    int batch_index = cursor / {channels};
    int grid_index = ((batch_index * {out_h} + out_y) * {out_w} + out_x) * 2;
    float grid_x = p1[grid_index], grid_y = p1[grid_index + 1];
    float x = {x_expression}, y = {y_expression};
'''
        if node.attrs.get("mode", "bilinear") == "nearest":
            body = common + f'''    int source_x = (int)nearbyintf(x), source_y = (int)nearbyintf(y);
    out[i] = gac_grid_load(
        p0, batch_index, {channels}, {height}, {width}, channel, source_y, source_x);
'''
        else:
            body = common + f'''    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;
    float lx = x - (float)x0, ly = y - (float)y0;
    float hx = 1.0f - lx, hy = 1.0f - ly;
    out[i] =
        gac_grid_load(p0, batch_index, {channels}, {height}, {width}, channel, y0, x0) * hx * hy +
        gac_grid_load(p0, batch_index, {channels}, {height}, {width}, channel, y0, x1) * lx * hy +
        gac_grid_load(p0, batch_index, {channels}, {height}, {width}, channel, y1, x0) * hx * ly +
        gac_grid_load(p0, batch_index, {channels}, {height}, {width}, channel, y1, x1) * lx * ly;
'''
    elif node.op == "Conv2D":
        batch, in_channels, in_h, in_w = specs[node.inputs[0]].shape
        out_channels, channels_per_group, kernel_h, kernel_w = specs[node.inputs[1]].shape
        _, _, out_h, out_w = shape
        stride_h, stride_w = tuple(node.attrs.get("strides", (1, 1)))
        dilation_h, dilation_w = tuple(node.attrs.get("dilations", (1, 1)))
        pad_top, pad_left, _, _ = tuple(node.attrs.get("pads", (0, 0, 0, 0)))
        groups = int(node.attrs.get("groups", 1))
        outputs_per_group = out_channels // groups
        bias = "p2[out_channel]" if len(node.inputs) == 3 else "0.0f"
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int cursor = i;
    int out_x = cursor % {out_w}; cursor /= {out_w};
    int out_y = cursor % {out_h}; cursor /= {out_h};
    int out_channel = cursor % {out_channels};
    int batch_index = cursor / {out_channels};
    int group = out_channel / {outputs_per_group};
    int channel_start = group * {channels_per_group};
    float accumulator = {bias};
    for (int channel_offset = 0; channel_offset < {channels_per_group}; ++channel_offset) {{
        for (int kernel_y = 0; kernel_y < {kernel_h}; ++kernel_y) {{
            int source_y = out_y * {stride_h} - {pad_top} + kernel_y * {dilation_h};
            if (source_y < 0 || source_y >= {in_h}) continue;
            for (int kernel_x = 0; kernel_x < {kernel_w}; ++kernel_x) {{
                int source_x = out_x * {stride_w} - {pad_left} + kernel_x * {dilation_w};
                if (source_x < 0 || source_x >= {in_w}) continue;
                int value_index = ((batch_index * {in_channels} + channel_start + channel_offset) * {in_h} + source_y) * {in_w} + source_x;
                int weight_index = ((out_channel * {channels_per_group} + channel_offset) * {kernel_h} + kernel_y) * {kernel_w} + kernel_x;
                accumulator += p0[value_index] * p1[weight_index];
            }}
        }}
    }}
    out[i] = accumulator;
'''
    elif node.op == "BatchNorm":
        channels = specs[node.inputs[0]].shape[1]
        inner = math.prod(specs[node.inputs[0]].shape[2:])
        epsilon = f"{float(node.attrs.get('epsilon', 1e-5)):.9e}f"
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int channel = (i / {inner}) % {channels};
    out[i] = (p0[i] - p3[channel]) * rsqrtf(p4[channel] + {epsilon}) * p1[channel] + p2[channel];
'''
    elif node.op == "MaxPool2D":
        batch, channels, in_h, in_w = specs[node.inputs[0]].shape
        _, _, out_h, out_w = shape
        kernel_h, kernel_w = tuple(node.attrs["kernel_shape"])
        stride_h, stride_w = tuple(node.attrs.get("strides", (kernel_h, kernel_w)))
        dilation_h, dilation_w = tuple(node.attrs.get("dilations", (1, 1)))
        pad_top, pad_left, _, _ = tuple(node.attrs.get("pads", (0, 0, 0, 0)))
        body = f'''    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int cursor = i;
    int out_x = cursor % {out_w}; cursor /= {out_w};
    int out_y = cursor % {out_h}; cursor /= {out_h};
    int channel = cursor % {channels};
    int batch_index = cursor / {channels};
    float maximum = -__int_as_float(0x7f800000);
    for (int kernel_y = 0; kernel_y < {kernel_h}; ++kernel_y) {{
        int source_y = out_y * {stride_h} - {pad_top} + kernel_y * {dilation_h};
        if (source_y < 0 || source_y >= {in_h}) continue;
        for (int kernel_x = 0; kernel_x < {kernel_w}; ++kernel_x) {{
            int source_x = out_x * {stride_w} - {pad_left} + kernel_x * {dilation_w};
            if (source_x < 0 || source_x >= {in_w}) continue;
            int source_index = ((batch_index * {channels} + channel) * {in_h} + source_y) * {in_w} + source_x;
            maximum = fmaxf(maximum, p0[source_index]);
        }}
    }}
    out[i] = maximum;
'''
    elif node.op == "MSDeformableAttention":
        batch, spatial_size, heads, head_dim = specs[node.inputs[0]].shape
        _, queries, _, levels, points, _ = specs[node.inputs[1]].shape
        level_blocks = []
        level_start = 0
        for level, (height, width) in enumerate(node.attrs["spatial_shapes"]):
            level_blocks.append(f'''
    for (int point = 0; point < {points}; ++point) {{
        int location_index = (((((batch_index * {queries} + query) * {heads} + head) * {levels} + {level}) * {points} + point) * 2);
        int weight_index = ((((batch_index * {queries} + query) * {heads} + head) * {levels} + {level}) * {points} + point);
        float x = p1[location_index] * {width}.0f - 0.5f;
        float y = p1[location_index + 1] * {height}.0f - 0.5f;
        int x0 = (int)floorf(x), y0 = (int)floorf(y);
        int x1 = x0 + 1, y1 = y0 + 1;
        float lx = x - (float)x0, ly = y - (float)y0;
        float hx = 1.0f - lx, hy = 1.0f - ly;
        float sample =
            gac_msda_load(p0, batch_index, {spatial_size}, {heads}, {head_dim}, {level_start}, {height}, {width}, y0, x0, head, channel) * hx * hy +
            gac_msda_load(p0, batch_index, {spatial_size}, {heads}, {head_dim}, {level_start}, {height}, {width}, y0, x1, head, channel) * lx * hy +
            gac_msda_load(p0, batch_index, {spatial_size}, {heads}, {head_dim}, {level_start}, {height}, {width}, y1, x0, head, channel) * hx * ly +
            gac_msda_load(p0, batch_index, {spatial_size}, {heads}, {head_dim}, {level_start}, {height}, {width}, y1, x1, head, channel) * lx * ly;
        accumulator += sample * p2[weight_index];
    }}
''')
            level_start += height * width
        body = f'''
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int cursor = i;
    int channel = cursor % {head_dim}; cursor /= {head_dim};
    int head = cursor % {heads}; cursor /= {heads};
    int query = cursor % {queries};
    int batch_index = cursor / {queries};
    float accumulator = 0.0f;
{''.join(level_blocks)}
    out[i] = accumulator;
'''
    elif node.op in {"MatMul", "MatMulBiasGELU"}:
        m, k = specs[node.inputs[0]].shape[-2:]
        _, n = specs[node.inputs[1]].shape[-2:]
        if specs[node.inputs[0]].rank == 2 and specs[node.inputs[1]].rank == 2:
            grid, block = ((n + 15) // 16, (m + 15) // 16, 1), (16, 16, 1)
            result = "gac_gelu(acc + p2[col])" if node.op == "MatMulBiasGELU" else "acc"
            body = f'''
    __shared__ float a[16][16], b[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty, col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int tile = 0; tile < {(k + 15) // 16}; ++tile) {{
        int ak = tile * 16 + tx, bk = tile * 16 + ty;
        a[ty][tx] = (row < {m} && ak < {k}) ? p0[row * {k} + ak] : 0.0f;
        b[ty][tx] = (bk < {k} && col < {n}) ? p1[bk * {n} + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc += a[ty][j] * b[j][tx];
        __syncthreads();
    }}
    if (row < {m} && col < {n}) out[row * {n} + col] = {result};
'''
        else:
            left_shape = specs[node.inputs[0]].shape
            right_shape = specs[node.inputs[1]].shape
            output_batch = shape[:-2]
            m, k, n = left_shape[-2], left_shape[-1], right_shape[-1]
            left_base = _matmul_batch_offset(left_shape, output_batch, m * k)
            right_base = _matmul_batch_offset(right_shape, output_batch, k * n)
            body = f'''
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= {size}) return;
    int col = i % {n};
    int row = (i / {n}) % {m};
    int batch_index = i / {m * n};
    int left_base = {left_base};
    int right_base = {right_base};
    float accumulator = 0.0f;
    for (int inner = 0; inner < {k}; ++inner) {{
        accumulator += p0[left_base + row * {k} + inner] * p1[right_base + inner * {n} + col];
    }}
    out[i] = accumulator;
'''
    elif node.op in {"LayerNorm", "Softmax"}:
        width = shape[-1]
        rows = size // width
        grid = (rows, 1, 1)
        common = f'''
    __shared__ float scratch[256];
    int t = threadIdx.x, row = blockIdx.x;
    const float* x = p0 + row * {width};
    float* y = out + row * {width};
'''
        if node.op == "LayerNorm":
            epsilon = f"{float(node.attrs.get('epsilon', 1e-5)):.9e}f"
            result = f"(x[j] - mean) / sqrtf(variance + {epsilon})"
            if len(node.inputs) >= 2:
                result = f"({result}) * p1[j]"
            if len(node.inputs) == 3:
                result = f"({result}) + p2[j]"
            body = common + f'''
    float sum = 0.0f;
    for (int j = t; j < {width}; j += 256) sum += x[j];
    scratch[t] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (t < s) scratch[t] += scratch[t + s];
        __syncthreads();
    }}
    float mean = scratch[0] / {width}.0f;
    __syncthreads();
    sum = 0.0f;
    for (int j = t; j < {width}; j += 256) {{ float d = x[j] - mean; sum += d * d; }}
    scratch[t] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (t < s) scratch[t] += scratch[t + s];
        __syncthreads();
    }}
    float variance = scratch[0] / {width}.0f;
    for (int j = t; j < {width}; j += 256) y[j] = {result};
'''
        else:
            body = common + f'''
    float maximum = -__int_as_float(0x7f800000);
    for (int j = t; j < {width}; j += 256) maximum = fmaxf(maximum, x[j]);
    scratch[t] = maximum;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (t < s) scratch[t] = fmaxf(scratch[t], scratch[t + s]);
        __syncthreads();
    }}
    maximum = scratch[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = t; j < {width}; j += 256) sum += expf(x[j] - maximum);
    scratch[t] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s /= 2) {{
        if (t < s) scratch[t] += scratch[t + s];
        __syncthreads();
    }}
    float denominator = scratch[0];
    for (int j = t; j < {width}; j += 256) y[j] = expf(x[j] - maximum) / denominator;
'''
    else:
        expressions = {
            "Identity": "p0[i]",
            "ReLU": "(p0[i] < 0.0f ? 0.0f : p0[i])",
            "GELU": "gac_gelu(p0[i])",
            "Sigmoid": "(p0[i] >= 0.0f ? 1.0f / (1.0f + expf(-p0[i])) : expf(p0[i]) / (1.0f + expf(p0[i])))",
            "Log": "logf(p0[i])",
            "Sin": "sinf(p0[i])",
            "Cos": "cosf(p0[i])",
        }
        if node.op == "Clamp":
            expression = "p0[i]"
            if node.attrs.get("min") is not None:
                expression = f"fmaxf({expression}, {float(node.attrs['min']):.9e}f)"
            if node.attrs.get("max") is not None:
                expression = f"fminf({expression}, {float(node.attrs['max']):.9e}f)"
        elif node.op in {"Add", "AddReLU", "Sub", "Mul", "Div", "Greater", "Less"}:
            left = _broadcast_index(specs[node.inputs[0]].shape, shape)
            right = _broadcast_index(specs[node.inputs[1]].shape, shape)
            if node.op in {"Greater", "Less"}:
                operator = {"Greater": ">", "Less": "<"}[node.op]
                expression = f"(p0[{left}] {operator} p1[{right}] ? 1.0f : 0.0f)"
            else:
                operator = {
                    "Add": "+", "AddReLU": "+", "Sub": "-", "Mul": "*", "Div": "/"
                }[node.op]
                expression = f"p0[{left}] {operator} p1[{right}]"
        elif node.op == "Where":
            condition = _broadcast_index(specs[node.inputs[0]].shape, shape)
            when_true = _broadcast_index(specs[node.inputs[1]].shape, shape)
            when_false = _broadcast_index(specs[node.inputs[2]].shape, shape)
            expression = (
                f"(p0[{condition}] != 0 ? "
                f"p1[{when_true}] : p2[{when_false}])"
            )
        else:
            expression = expressions[node.op]
        if node.op == "AddReLU":
            body = (
                "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                f"    if (i < {size}) {{\n"
                f"        float value = {expression};\n"
                "        out[i] = value < 0.0f ? 0.0f : value;\n"
                "    }\n"
            )
        else:
            body = f"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < {size}) out[i] = {expression};\n"
    launch = KernelLaunch(
        entry,
        node.id,
        node.inputs,
        node.outputs[0],
        grid,
        block,
        node.source_nodes,
        _implementation_candidates(node, specs),
    )
    return header + body + "}\n", launch


def compile_cuda(graph: Graph, *, fuse: bool = True) -> CudaProgram:
    """Validate semantics, optimize, and emit source. No device access occurs."""
    reference = PassManager([ShapeInferencePass()]).run(graph).graph
    _check_specs(reference)
    for node in reference.nodes:
        _check_node(node, reference.value_specs)
    passes = default_pass_manager().passes
    if not fuse:
        passes = tuple(
            p
            for p in passes
            if not isinstance(p, (MatMulBiasGELUFusionPass, AddReLUFusionPass))
        )
    result = PassManager(passes).run(reference)
    _check_specs(result.graph)
    if fuse:
        for node in result.graph.nodes:
            try:
                _check_node(node, result.graph.value_specs)
            except BackendError as exc:
                # Keep a supported unfused graph when the candidate fusion is
                # outside the CUDA epilogue ABI (for example a matrix bias).
                unfused = compile_cuda(graph, fuse=False)
                unfused.pass_reports += ({"name": "cuda-fusion-rejected", "changed": False, "details": {"reason": str(exc)}},)
                return unfused
    sources, launches = [_PRELUDE], []
    zero_copy_views = infer_zero_copy_views(result.graph)
    for index, node in enumerate(result.graph.nodes):
        _check_node(node, result.graph.value_specs)
        if (
            node.op in {"Reshape", "Slice"}
            and len(node.inputs) == 1
            and len(node.outputs) == 1
            and zero_copy_views.get(node.outputs[0]) == node.inputs[0]
        ):
            continue
        source, launch = _emit_kernel(node, result.graph, index)
        sources.append(source)
        launches.append(launch)
    return CudaProgram(result.graph, reference, "\n".join(sources), tuple(launches), tuple(r.to_dict() for r in result.reports))
