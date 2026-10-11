"""Executable MLIR runtime-call lowering for the BEVFormer FFN on CUDA SM86.

The tensor/linalg backend remains the canonical, reviewable MLIR frontend.  This
module adds the first executable lowering slice: two ``MatMul`` operations are
made explicit cuBLAS runtime calls, while their surrounding pointwise work is
lowered to two generated CUDA kernels.  The final kernel fuses bias, residual
addition and LayerNorm and performs one block-wide reduction per row.

The emitted ``source`` is standard ``func``/``memref`` MLIR.  Private function
symbols describe the runtime ABI; :class:`gac.mlir_runtime.MlirCudaSession`
provides those implementations with CuPy/cuBLAS and NVRTC.  There is no hidden
fallback to the older CUDA graph backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Sequence

import numpy as np

from ..errors import BackendError
from ..ir import Graph, Node, TensorSpec
from .mlir import MlirProgram, compile_mlir


TARGET = "mlir-cuda-sm86"
COMPUTE_CAPABILITY = (8, 6)
NVRTC_OPTIONS = ("--std=c++11",)
RUNTIME_ABI_VERSION = 2

# Both generated kernels deliberately use signed 32-bit element indexes.  Keep
# the accepted ABI inside that domain instead of relying on C++ overflow or a
# driver-specific launch failure for oversized static tensors.  SM86 supports
# at most 2**31 - 1 blocks in grid.x.
CUDA_KERNEL_INDEX_MAX = (1 << 31) - 1
CUDA_GRID_X_MAX = (1 << 31) - 1

_FC1_SYMBOL = "gac_cublas_fc1_matmul_f32"
_BIAS_RELU_SYMBOL = "gac_bias_relu_inplace_f32"
_FC2_SYMBOL = "gac_cublas_fc2_matmul_f32"
_FINAL_SYMBOL = "gac_bias_residual_layer_norm_f32"
_RUNTIME_SYMBOLS = frozenset(
    {_FC1_SYMBOL, _BIAS_RELU_SYMBOL, _FC2_SYMBOL, _FINAL_SYMBOL}
)


@dataclass(frozen=True, slots=True)
class MlirCudaCall:
    """One explicit dispatch in the MLIR CUDA runtime-call plan."""

    abi_version: int
    symbol: str
    kind: str
    implementation: str
    inputs: tuple[str, ...]
    output: str
    node_ids: tuple[str, ...]
    source_nodes: tuple[str, ...]
    dtype: str
    layout: str
    grid: tuple[int, int, int] | None = None
    block: tuple[int, int, int] | None = None
    aliases_input: int | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    transpose_a: bool | None = None
    transpose_b: bool | None = None
    alpha: float | None = None
    beta: float | None = None
    rows: int | None = None
    channels: int | None = None
    elements: int | None = None
    epsilon: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "abi_version": self.abi_version,
            "symbol": self.symbol,
            "kind": self.kind,
            "implementation": self.implementation,
            "inputs": list(self.inputs),
            "output": self.output,
            "node_ids": list(self.node_ids),
            "source_nodes": list(self.source_nodes),
            "dtype": self.dtype,
            "layout": self.layout,
        }
        if self.grid is not None:
            result["grid"] = list(self.grid)
        if self.block is not None:
            result["block"] = list(self.block)
        if self.aliases_input is not None:
            result["aliases_input"] = self.aliases_input
        parameters = {
            name: value
            for name, value in (
                ("m", self.m),
                ("n", self.n),
                ("k", self.k),
                ("transpose_a", self.transpose_a),
                ("transpose_b", self.transpose_b),
                ("alpha", self.alpha),
                ("beta", self.beta),
                ("rows", self.rows),
                ("channels", self.channels),
                ("elements", self.elements),
                ("epsilon", self.epsilon),
            )
            if value is not None
        }
        result["parameters"] = parameters
        return result


@dataclass(frozen=True, slots=True)
class _FfnPlan:
    query: str
    fc1_weight: str
    fc1_bias: str
    fc1_matmul: str
    activated: str
    fc2_weight: str
    fc2_bias: str
    fc2_matmul: str
    norm_weight: str
    norm_bias: str
    output: str
    epsilon: float
    rows: int
    channels: int
    hidden_channels: int
    node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MlirCudaProgram:
    """Executable hybrid artifact produced from standard MLIR semantics.

    ``source`` is a verifier-friendly ABI manifest.  It is deliberately not
    parsed or executed by the Python runtime.  Execution is driven by ``calls``
    after the runtime deterministically regenerates and checks every artifact
    component before touching CUDA.
    """

    abi_version: int
    source_graph: Graph
    graph: Graph
    reference_graph: Graph
    source: str
    tensor_source: str
    kernel_source: str
    calls: tuple[MlirCudaCall, ...]
    pass_reports: tuple[dict[str, Any], ...]
    integrity_sha256: str
    executable: bool
    mlir_source_role: str
    mlir_source_executed: bool
    runtime_dispatch_source: str

    @property
    def target(self) -> str:
        """Return the executable backend name used by runtime selectors."""

        return TARGET

    @property
    def text(self) -> str:
        return self.source

    def describe(self) -> dict[str, Any]:
        integrity = {
            "algorithm": "sha256",
            "integrity_sha256": self.integrity_sha256,
            "mlir_source_sha256": _sha256_text(self.source),
            "tensor_source_sha256": _sha256_text(self.tensor_source),
            "kernel_source_sha256": _sha256_text(self.kernel_source),
            "call_plan_sha256": _sha256_json(
                [call.to_dict() for call in self.calls]
            ),
        }
        return {
            "schema_version": self.abi_version,
            "runtime_abi_version": self.abi_version,
            "compiler_backend": TARGET,
            "target": TARGET,
            "target_intent": "cuda-sm86",
            "compute_capability": list(COMPUTE_CAPABILITY),
            "lowering_stage": "runtime-call-plan",
            "executable": self.executable,
            "artifact_kind": "mlir-abi-manifest-with-validated-runtime-call-plan",
            "mlir_source_role": self.mlir_source_role,
            "mlir_source_executed": self.mlir_source_executed,
            "runtime_dispatch_source": self.runtime_dispatch_source,
            "integrity_validation": (
                "deterministic recompilation and exact artifact comparison before CUDA access"
            ),
            "integrity_sha256": self.integrity_sha256,
            "integrity": integrity,
            "dialects": ["builtin", "func", "memref"],
            "runtime_requirements": ["CUDA", "cuBLAS", "NVRTC", "CuPy"],
            "matmul_lowering": "cupy-cublas",
            "precision": {
                "storage": "float32",
                "cublas_compute_policy": "runtime-default",
                "tensor_core_policy": "not-forced-by-compiler",
            },
            "kernel_fusions": [
                "bias+relu",
                "bias+residual+layernorm",
            ],
            "layer_norm_lowering": "one CUDA block per row with block reduction",
            "name": self.graph.name,
            "inputs": {
                name: spec.to_dict() for name, spec in self.graph.inputs.items()
            },
            "outputs": {
                name: self.graph.value_specs[name].to_dict()
                for name in self.graph.outputs
            },
            "calls": [call.to_dict() for call in self.calls],
            "pass_reports": list(self.pass_reports),
        }


def _reject(reason: str) -> None:
    raise BackendError(f"{TARGET}: {reason}")


def _other_input(node: Node, value: str) -> str:
    if len(node.inputs) != 2 or node.inputs.count(value) != 1:
        _reject(f"node {node.id!r} must consume {value!r} exactly once")
    return node.inputs[1] if node.inputs[0] == value else node.inputs[0]


def _require_constant(graph: Graph, name: str, purpose: str) -> None:
    if name not in graph.constants:
        _reject(f"{purpose} {name!r} must be a resident graph constant")


def _match_bevformer_ffn(program: MlirProgram) -> _FfnPlan:
    graph = program.graph
    nodes = tuple(graph.nodes)
    expected_ops = ("MatMul", "AddReLU", "MatMul", "Add", "Add", "LayerNorm")
    actual_ops = tuple(node.op for node in nodes)
    if actual_ops != expected_ops:
        _reject(
            "the executable slice currently requires the fused BEVFormer FFN "
            f"pattern {list(expected_ops)}, got {list(actual_ops)}"
        )
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        _reject("the executable FFN slice requires one input and one output")

    fc1, bias_relu, fc2, bias2_add, residual_add, layer_norm = nodes
    query = next(iter(graph.inputs))
    output = graph.outputs[0]
    if fc1.inputs[0] != query or len(fc1.inputs) != 2:
        _reject("the first MatMul must consume the graph query as its left input")
    fc1_weight = fc1.inputs[1]
    fc1_matmul = fc1.outputs[0]
    fc1_bias = _other_input(bias_relu, fc1_matmul)
    activated = bias_relu.outputs[0]
    if fc2.inputs[0] != activated or len(fc2.inputs) != 2:
        _reject("the second MatMul must consume the fused bias/ReLU output")
    fc2_weight = fc2.inputs[1]
    fc2_matmul = fc2.outputs[0]
    fc2_bias = _other_input(bias2_add, fc2_matmul)
    projected = bias2_add.outputs[0]
    residual = _other_input(residual_add, projected)
    if residual != query:
        _reject("the executable FFN slice currently requires query residual identity")
    ffn_output = residual_add.outputs[0]
    if not layer_norm.inputs or layer_norm.inputs[0] != ffn_output:
        _reject("LayerNorm must directly consume the residual result")
    if len(layer_norm.inputs) != 3 or layer_norm.outputs != (output,):
        _reject("the fused LayerNorm requires scale, bias and the graph output")
    norm_weight, norm_bias = layer_norm.inputs[1:]

    for name, purpose in (
        (fc1_weight, "FC1 weight"),
        (fc1_bias, "FC1 bias"),
        (fc2_weight, "FC2 weight"),
        (fc2_bias, "FC2 bias"),
        (norm_weight, "LayerNorm scale"),
        (norm_bias, "LayerNorm bias"),
    ):
        _require_constant(graph, name, purpose)

    query_spec = graph.value_specs[query]
    fc1_weight_spec = graph.value_specs[fc1_weight]
    fc2_weight_spec = graph.value_specs[fc2_weight]
    if query_spec.rank < 2:
        _reject("query rank must be at least two")
    layer_norm_axis = int(layer_norm.attrs.get("axis", -1))
    if layer_norm_axis < 0:
        layer_norm_axis += query_spec.rank
    if layer_norm_axis != query_spec.rank - 1:
        _reject("fused LayerNorm currently requires normalization over the last axis")
    rows = math.prod(query_spec.shape[:-1])
    channels = query_spec.shape[-1]
    hidden_channels = fc1_weight_spec.shape[-1]
    if fc1_weight_spec.shape != (channels, hidden_channels):
        _reject("FC1 weight shape does not match query channels")
    if fc2_weight_spec.shape != (hidden_channels, channels):
        _reject("FC2 weight shape does not restore query channels")
    expected_vector_shapes = {
        fc1_bias: (hidden_channels,),
        fc2_bias: (channels,),
        norm_weight: (channels,),
        norm_bias: (channels,),
    }
    for name, expected_shape in expected_vector_shapes.items():
        if graph.value_specs[name].shape != expected_shape:
            _reject(
                f"runtime kernel requires {name!r} shape {expected_shape}, got "
                f"{graph.value_specs[name].shape}"
            )
    if graph.value_specs[output].shape != query_spec.shape:
        _reject("FFN output shape must match the query shape")
    if channels > 4096:
        _reject("fused LayerNorm supports at most 4096 channels")

    hidden_elements = rows * hidden_channels
    output_elements = rows * channels
    bias_relu_grid_x = (hidden_elements + 255) // 256
    for purpose, elements in (
        ("fused bias/ReLU", hidden_elements),
        ("fused residual/LayerNorm", output_elements),
    ):
        if elements > CUDA_KERNEL_INDEX_MAX:
            _reject(
                f"{purpose} requires {elements} elements, exceeding the signed "
                f"32-bit kernel index limit {CUDA_KERNEL_INDEX_MAX}"
            )
    for purpose, grid_x in (
        ("fused bias/ReLU", bias_relu_grid_x),
        ("fused residual/LayerNorm", rows),
    ):
        if grid_x > CUDA_GRID_X_MAX:
            _reject(
                f"{purpose} requires grid.x={grid_x}, exceeding the SM86 limit "
                f"{CUDA_GRID_X_MAX}"
            )

    epsilon = float(layer_norm.attrs.get("epsilon", 1.0e-5))
    epsilon_f32 = np.float32(epsilon)
    if (
        not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not np.isfinite(epsilon_f32)
        or epsilon_f32 == 0.0
    ):
        _reject("fused LayerNorm epsilon must be finite, positive and float32")
    return _FfnPlan(
        query=query,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc1_matmul=fc1_matmul,
        activated=activated,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        fc2_matmul=fc2_matmul,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        output=output,
        epsilon=float(epsilon_f32),
        rows=rows,
        channels=channels,
        hidden_channels=hidden_channels,
        node_ids=tuple(node.id for node in nodes),
    )


def _symbol_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_$.]", "_", name)
    if not clean or not re.match(r"[A-Za-z_$]", clean[0]):
        clean = "gac_" + clean
    return clean


def _entry_symbol_name(name: str) -> str:
    """Return an entry symbol in a namespace disjoint from runtime symbols."""

    entry = "gac_entry_" + _symbol_name(name)
    while entry in _RUNTIME_SYMBOLS:
        entry = "gac_entry_" + entry
    return entry


def _mlir_string_literal(value: str) -> str:
    """Encode arbitrary text as one MLIR string literal.

    Provenance originates outside this backend and can contain quotes,
    backslashes or control characters.  Escaping UTF-8 bytes keeps it data and
    prevents malformed attributes or injected operations.
    """

    encoded: list[str] = []
    for byte in value.encode("utf-8"):
        if byte == 0x22:
            encoded.append(r'\"')
        elif byte == 0x5C:
            encoded.append(r"\\")
        elif 0x20 <= byte <= 0x7E:
            encoded.append(chr(byte))
        else:
            encoded.append(f"\\{byte:02X}")
    return '"' + "".join(encoded) + '"'


def _provenance_literal(values: Sequence[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return _mlir_string_literal(payload)


def _cuda_float_literal(value: float) -> str:
    """Return a finite CUDA ``float`` token, including for integral values."""

    normalized = float(np.float32(value))
    if not math.isfinite(normalized):
        _reject(f"CUDA float literal must be finite, got {value!r}")
    token = format(normalized, ".9g")
    if "." not in token and "e" not in token.lower():
        token += ".0"
    return token + "f"


def _memref_type(spec: TensorSpec | Sequence[int]) -> str:
    shape = spec.shape if isinstance(spec, TensorSpec) else tuple(spec)
    if not shape:
        return "memref<f32>"
    return "memref<" + "x".join(str(int(dim)) for dim in shape) + "xf32>"


def _call(
    result: list[str],
    symbol: str,
    arguments: Sequence[str],
    types: Sequence[str],
    *,
    attributes: str | None = None,
) -> None:
    args = ", ".join(arguments)
    signature = ", ".join(types)
    suffix = f" {{{attributes}}}" if attributes else ""
    result.append(
        f"    func.call @{symbol}({args}){suffix} : ({signature}) -> ()"
    )


def _common_abi_attributes(call: MlirCudaCall) -> str:
    attributes = [
        f"gac.abi_version = {call.abi_version} : i64",
        f"gac.dtype = {_mlir_string_literal(call.dtype)}",
        f"gac.layout = {_mlir_string_literal(call.layout)}",
    ]
    for name in ("m", "n", "k", "rows", "channels", "elements"):
        value = getattr(call, name)
        if value is not None:
            attributes.append(f"gac.{name} = {value} : i64")
    for name in ("transpose_a", "transpose_b"):
        value = getattr(call, name)
        if value is not None:
            attributes.append(f"gac.{name} = {'true' if value else 'false'}")
    for name in ("alpha", "beta", "epsilon"):
        value = getattr(call, name)
        if value is not None:
            attributes.append(f"gac.{name} = {_cuda_float_literal(value)[:-1]} : f32")
    return ", ".join(attributes)


def _call_provenance_attributes(call: MlirCudaCall) -> str:
    return ", ".join(
        (
            f"gac.node_ids = {_provenance_literal(call.node_ids)}",
            f"gac.source_nodes = {_provenance_literal(call.source_nodes)}",
        )
    )


def _emit_runtime_call_mlir(
    graph: Graph, plan: _FfnPlan, calls: tuple[MlirCudaCall, ...]
) -> str:
    fc1_call, bias_relu_call, fc2_call, final_call = calls

    query_type = _memref_type(graph.value_specs[plan.query])
    fc1_weight_type = _memref_type(graph.value_specs[plan.fc1_weight])
    fc1_bias_type = _memref_type(graph.value_specs[plan.fc1_bias])
    hidden_type = _memref_type(graph.value_specs[plan.fc1_matmul])
    fc2_weight_type = _memref_type(graph.value_specs[plan.fc2_weight])
    fc2_bias_type = _memref_type(graph.value_specs[plan.fc2_bias])
    projected_type = _memref_type(graph.value_specs[plan.fc2_matmul])
    norm_weight_type = _memref_type(graph.value_specs[plan.norm_weight])
    norm_bias_type = _memref_type(graph.value_specs[plan.norm_bias])
    output_type = _memref_type(graph.value_specs[plan.output])

    abi_names = [*graph.inputs, *sorted(graph.constants), *graph.outputs]
    arguments = {name: f"%arg{index}" for index, name in enumerate(abi_names)}
    signature = ", ".join(
        f"{arguments[name]}: {_memref_type(graph.value_specs[name])}"
        for name in abi_names
    )
    entry = _entry_symbol_name(graph.name)

    lines = [
        "module {",
        (
            f"  func.func private @{fc1_call.symbol}({query_type}, {fc1_weight_type}, {hidden_type}) "
            f"attributes {{{_common_abi_attributes(fc1_call)}, "
            'gac.runtime_binding = "cupy.matmul", gac.vendor = "cublas"}'
        ),
        (
            f"  func.func private @{bias_relu_call.symbol}({hidden_type}, {fc1_bias_type}) "
            f"attributes {{{_common_abi_attributes(bias_relu_call)}, "
            'gac.fusion = "bias+relu", gac.runtime_binding = "nvrtc-cuda"}'
        ),
        (
            f"  func.func private @{fc2_call.symbol}({hidden_type}, {fc2_weight_type}, {projected_type}) "
            f"attributes {{{_common_abi_attributes(fc2_call)}, "
            'gac.runtime_binding = "cupy.matmul", gac.vendor = "cublas"}'
        ),
        (
            f"  func.func private @{final_call.symbol}({projected_type}, {fc2_bias_type}, "
            f"{query_type}, {norm_weight_type}, {norm_bias_type}, {output_type}) "
            f"attributes {{{_common_abi_attributes(final_call)}, "
            'gac.fusion = "bias+residual+layernorm", gac.runtime_binding = "nvrtc-cuda"}'
        ),
        f"  func.func @{entry}({signature}) {{",
        f"    %hidden = memref.alloc() : {hidden_type}",
        f"    %projected = memref.alloc() : {projected_type}",
    ]
    _call(
        lines,
        fc1_call.symbol,
        (arguments[plan.query], arguments[plan.fc1_weight], "%hidden"),
        (query_type, fc1_weight_type, hidden_type),
        attributes=_call_provenance_attributes(fc1_call),
    )
    _call(
        lines,
        bias_relu_call.symbol,
        ("%hidden", arguments[plan.fc1_bias]),
        (hidden_type, fc1_bias_type),
        attributes=_call_provenance_attributes(bias_relu_call),
    )
    _call(
        lines,
        fc2_call.symbol,
        ("%hidden", arguments[plan.fc2_weight], "%projected"),
        (hidden_type, fc2_weight_type, projected_type),
        attributes=_call_provenance_attributes(fc2_call),
    )
    _call(
        lines,
        final_call.symbol,
        (
            "%projected",
            arguments[plan.fc2_bias],
            arguments[plan.query],
            arguments[plan.norm_weight],
            arguments[plan.norm_bias],
            arguments[plan.output],
        ),
        (
            projected_type,
            fc2_bias_type,
            query_type,
            norm_weight_type,
            norm_bias_type,
            output_type,
        ),
        attributes=_call_provenance_attributes(final_call),
    )
    lines.extend(
        (
            f"    memref.dealloc %projected : {projected_type}",
            f"    memref.dealloc %hidden : {hidden_type}",
            "    return",
            "  }",
            "}",
        )
    )
    return "\n".join(lines) + "\n"


def _emit_cuda_kernels(plan: _FfnPlan) -> str:
    elements = plan.rows * plan.hidden_channels
    epsilon = _cuda_float_literal(plan.epsilon)
    return f'''// Generated runtime kernels for the MLIR CUDA call plan on SM86.
// MatMul is deliberately absent: both matrix products are cuBLAS calls.
#include <cuda_runtime.h>

extern "C" __global__ void gac_bias_relu_inplace_f32(
    float* values, const float* bias) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < {elements}) {{
    const float value = values[index] + bias[index % {plan.hidden_channels}];
    values[index] = isnan(value) ? value : fmaxf(value, 0.0f);
  }}
}}

__device__ __forceinline__ float gac_warp_sum(float value) {{
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {{
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }}
  return value;
}}

extern "C" __global__ void gac_bias_residual_layer_norm_f32(
    const float* projected,
    const float* bias,
    const float* residual,
    const float* gamma,
    const float* beta,
    float* output) {{
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  const int lane = thread & 31;
  const int warp = thread >> 5;
  __shared__ float warp_sums[8];
  __shared__ float mean_shared;
  __shared__ float inv_std_shared;

  float local_sum = 0.0f;
  for (int column = thread; column < {plan.channels}; column += 256) {{
    const int index = row * {plan.channels} + column;
    local_sum += projected[index] + bias[column] + residual[index];
  }}
  local_sum = gac_warp_sum(local_sum);
  if (lane == 0) warp_sums[warp] = local_sum;
  __syncthreads();
  if (warp == 0) {{
    float block_sum = lane < 8 ? warp_sums[lane] : 0.0f;
    block_sum = gac_warp_sum(block_sum);
    if (lane == 0) mean_shared = block_sum / {plan.channels}.0f;
  }}
  __syncthreads();

  const float mean = mean_shared;
  float local_square_sum = 0.0f;
  for (int column = thread; column < {plan.channels}; column += 256) {{
    const int index = row * {plan.channels} + column;
    const float value = projected[index] + bias[column] + residual[index];
    const float centered = value - mean;
    local_square_sum += centered * centered;
  }}
  local_square_sum = gac_warp_sum(local_square_sum);
  if (lane == 0) warp_sums[warp] = local_square_sum;
  __syncthreads();
  if (warp == 0) {{
    float block_square_sum = lane < 8 ? warp_sums[lane] : 0.0f;
    block_square_sum = gac_warp_sum(block_square_sum);
    if (lane == 0) {{
      inv_std_shared = rsqrtf(block_square_sum / {plan.channels}.0f + {epsilon});
    }}
  }}
  __syncthreads();

  const float inv_std = inv_std_shared;
  for (int column = thread; column < {plan.channels}; column += 256) {{
    const int index = row * {plan.channels} + column;
    const float value = projected[index] + bias[column] + residual[index];
    output[index] = (value - mean) * inv_std * gamma[column] + beta[column];
  }}
}}
'''


def _build_calls(
    graph: Graph, plan: _FfnPlan
) -> tuple[MlirCudaCall, MlirCudaCall, MlirCudaCall, MlirCudaCall]:
    hidden_elements = plan.rows * plan.hidden_channels
    nodes = tuple(graph.nodes)

    def source_nodes(*indexes: int) -> tuple[str, ...]:
        ordered: list[str] = []
        for index in indexes:
            for name in nodes[index].source_nodes:
                if name not in ordered:
                    ordered.append(name)
        return tuple(ordered)

    calls = (
        MlirCudaCall(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=_FC1_SYMBOL,
            kind="matmul",
            implementation="cupy-cublas",
            inputs=(plan.query, plan.fc1_weight),
            output=plan.fc1_matmul,
            node_ids=(plan.node_ids[0],),
            source_nodes=source_nodes(0),
            dtype="float32",
            layout="contiguous",
            m=plan.rows,
            n=plan.hidden_channels,
            k=plan.channels,
            transpose_a=False,
            transpose_b=False,
            alpha=1.0,
            beta=0.0,
        ),
        MlirCudaCall(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=_BIAS_RELU_SYMBOL,
            kind="fused-kernel",
            implementation="nvrtc-cuda",
            inputs=(plan.fc1_matmul, plan.fc1_bias),
            output=plan.activated,
            node_ids=(plan.node_ids[1],),
            source_nodes=source_nodes(1),
            dtype="float32",
            layout="contiguous",
            grid=((hidden_elements + 255) // 256, 1, 1),
            block=(256, 1, 1),
            aliases_input=0,
            rows=plan.rows,
            channels=plan.hidden_channels,
            elements=hidden_elements,
        ),
        MlirCudaCall(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=_FC2_SYMBOL,
            kind="matmul",
            implementation="cupy-cublas",
            inputs=(plan.activated, plan.fc2_weight),
            output=plan.fc2_matmul,
            node_ids=(plan.node_ids[2],),
            source_nodes=source_nodes(2),
            dtype="float32",
            layout="contiguous",
            m=plan.rows,
            n=plan.channels,
            k=plan.hidden_channels,
            transpose_a=False,
            transpose_b=False,
            alpha=1.0,
            beta=0.0,
        ),
        MlirCudaCall(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=_FINAL_SYMBOL,
            kind="fused-kernel",
            implementation="nvrtc-cuda",
            inputs=(
                plan.fc2_matmul,
                plan.fc2_bias,
                plan.query,
                plan.norm_weight,
                plan.norm_bias,
            ),
            output=plan.output,
            node_ids=(plan.node_ids[3], plan.node_ids[4], plan.node_ids[5]),
            source_nodes=source_nodes(3, 4, 5),
            dtype="float32",
            layout="contiguous",
            grid=(plan.rows, 1, 1),
            block=(256, 1, 1),
            rows=plan.rows,
            channels=plan.channels,
            elements=plan.rows * plan.channels,
            epsilon=plan.epsilon,
        ),
    )
    return calls


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _graph_sha256(graph: Graph) -> str:
    graph.validate()
    digest = hashlib.sha256()
    metadata = graph.to_dict(include_constant_data=False)
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(graph.constants):
        value = graph.constants[name]
        if not value.flags.c_contiguous:
            _reject(f"constant {name!r} must remain C-contiguous")
        digest.update(name.encode("utf-8"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _integrity_sha256(
    *,
    source_graph: Graph,
    graph: Graph,
    reference_graph: Graph,
    source: str,
    tensor_source: str,
    kernel_source: str,
    calls: tuple[MlirCudaCall, ...],
    pass_reports: tuple[dict[str, Any], ...],
) -> str:
    payload = {
        "abi_version": RUNTIME_ABI_VERSION,
        "source_graph_sha256": _graph_sha256(source_graph),
        "graph_sha256": _graph_sha256(graph),
        "reference_graph_sha256": _graph_sha256(reference_graph),
        "source_sha256": _sha256_text(source),
        "tensor_source_sha256": _sha256_text(tensor_source),
        "kernel_source_sha256": _sha256_text(kernel_source),
        "calls": [call.to_dict() for call in calls],
        "pass_reports": list(pass_reports),
        "executable": True,
        "mlir_source_role": "verifiable-abi-manifest",
        "mlir_source_executed": False,
        "runtime_dispatch_source": "validated-call-records",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compile_mlir_cuda_unchecked(graph: Graph) -> MlirCudaProgram:
    source_graph = graph.clone()
    tensor_program = compile_mlir(graph, fuse=True)
    plan = _match_bevformer_ffn(tensor_program)
    calls = _build_calls(tensor_program.graph, plan)
    source = _emit_runtime_call_mlir(tensor_program.graph, plan, calls)
    kernel_source = _emit_cuda_kernels(plan)
    integrity_sha256 = _integrity_sha256(
        source_graph=source_graph,
        graph=tensor_program.graph,
        reference_graph=tensor_program.reference_graph,
        source=source,
        tensor_source=tensor_program.source,
        kernel_source=kernel_source,
        calls=calls,
        pass_reports=tensor_program.pass_reports,
    )
    return MlirCudaProgram(
        abi_version=RUNTIME_ABI_VERSION,
        source_graph=source_graph,
        graph=tensor_program.graph,
        reference_graph=tensor_program.reference_graph,
        source=source,
        tensor_source=tensor_program.source,
        kernel_source=kernel_source,
        calls=calls,
        pass_reports=tensor_program.pass_reports,
        integrity_sha256=integrity_sha256,
        executable=True,
        mlir_source_role="verifiable-abi-manifest",
        mlir_source_executed=False,
        runtime_dispatch_source="validated-call-records",
    )


def compile_mlir_cuda(graph: Graph) -> MlirCudaProgram:
    """Lower one static BEVFormer FFN to cuBLAS calls and fused CUDA kernels.

    This M2 slice is deliberately strict.  Unsupported graphs fail during
    compilation instead of falling back to generated CUDA or NumPy.
    """

    return _compile_mlir_cuda_unchecked(graph)


def validate_mlir_cuda_program(program: MlirCudaProgram) -> None:
    """Fail closed unless every program component is canonically consistent.

    This function is CPU-only.  The runtime calls it before device discovery,
    CuPy import, NVRTC compilation, allocation, or any other CUDA operation.
    """

    if not isinstance(program, MlirCudaProgram):
        raise BackendError("MLIR runtime artifact must be an MlirCudaProgram")
    if (
        program.abi_version != RUNTIME_ABI_VERSION
        or program.executable is not True
        or program.mlir_source_role != "verifiable-abi-manifest"
        or program.mlir_source_executed is not False
        or program.runtime_dispatch_source != "validated-call-records"
    ):
        raise BackendError("MLIR runtime artifact execution metadata is invalid")

    try:
        expected = _compile_mlir_cuda_unchecked(program.source_graph)
        current_integrity = _integrity_sha256(
            source_graph=program.source_graph,
            graph=program.graph,
            reference_graph=program.reference_graph,
            source=program.source,
            tensor_source=program.tensor_source,
            kernel_source=program.kernel_source,
            calls=program.calls,
            pass_reports=program.pass_reports,
        )
    except Exception as exc:
        if isinstance(exc, BackendError):
            raise
        raise BackendError(f"MLIR runtime artifact cannot be validated: {exc}") from exc

    comparisons = (
        (
            "source graph",
            _graph_sha256(program.source_graph),
            _graph_sha256(expected.source_graph),
        ),
        (
            "reference graph",
            _graph_sha256(program.reference_graph),
            _graph_sha256(expected.reference_graph),
        ),
        ("optimized graph", _graph_sha256(program.graph), _graph_sha256(expected.graph)),
        ("MLIR ABI source", program.source, expected.source),
        ("tensor MLIR source", program.tensor_source, expected.tensor_source),
        ("CUDA kernel source", program.kernel_source, expected.kernel_source),
        ("runtime call records", program.calls, expected.calls),
        ("pass reports", program.pass_reports, expected.pass_reports),
        ("integrity digest", program.integrity_sha256, current_integrity),
        ("canonical integrity digest", current_integrity, expected.integrity_sha256),
    )
    for component, actual, canonical in comparisons:
        if actual != canonical:
            raise BackendError(
                f"MLIR runtime artifact {component} differs from its canonical FFN plan"
            )


__all__ = [
    "COMPUTE_CAPABILITY",
    "CUDA_GRID_X_MAX",
    "CUDA_KERNEL_INDEX_MAX",
    "MlirCudaCall",
    "MlirCudaProgram",
    "NVRTC_OPTIONS",
    "RUNTIME_ABI_VERSION",
    "TARGET",
    "compile_mlir_cuda",
    "validate_mlir_cuda_program",
]
