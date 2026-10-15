"""Official MLIR GPU/NVVM/PTX lowering for one BEVFormer FFN profile.

This backend is intentionally separate from :mod:`gac.backends.mlir_cuda`.
The latter is the established fast hybrid path (cuBLAS plus NVRTC kernels),
whereas this module gives all device computation for this FFN slice to upstream
MLIR: Python emits registered GPU-dialect IR, ``mlir-opt`` lowers the module
through NVVM to textual SM86 PTX, and the runtime launches only entries from
that PTX module.

The first slice is deliberately narrow.  It accepts exactly the production
BEVFormer-tiny FFN profile ``[1, 2500, 256] -> 512 -> 256`` in FP32 and emits
three kernels.  The two matrix products use a simple one-thread-per-output
implementation.  That is a correctness baseline for this FFN compile, load and
execute path, not a claim that generated GEMM outperforms cuBLAS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import re
from typing import Any

from gac.core.errors import BackendError
from gac.core.ir import Graph
from gac.toolchains.mlir import DEFAULT_TIMEOUT_SECONDS, MlirPtxArtifact, extract_sm86_ptx
from gac.targets.cuda.mlir import compile_mlir_cuda


TARGET = "mlir-ptx-sm86"
COMPUTE_CAPABILITY = (8, 6)
RUNTIME_ABI_VERSION = 1

PRODUCTION_QUERY_SHAPE = (1, 2500, 256)
PRODUCTION_ROWS = 2500
PRODUCTION_CHANNELS = 256
PRODUCTION_HIDDEN_CHANNELS = 512
THREADS_PER_BLOCK = 256

FC1_SYMBOL = "gac_mlir_ptx_fc1_bias_relu_f32"
FC2_SYMBOL = "gac_mlir_ptx_fc2_f32"
FINAL_SYMBOL = "gac_mlir_ptx_bias_residual_layer_norm_f32"
PTX_ENTRY_SYMBOLS = (FC1_SYMBOL, FC2_SYMBOL, FINAL_SYMBOL)
_PTX_POINTER_PARAMETER_COUNTS = {
    FC1_SYMBOL: 4,
    FC2_SYMBOL: 3,
    FINAL_SYMBOL: 6,
}


def _reject(reason: str) -> None:
    raise BackendError(f"{TARGET}: {reason}")


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _validate_ptx_pointer_abi(ptx: str) -> None:
    """Require the exact bare-pointer ABI used by the CuPy launch records."""

    entry_pattern = re.compile(
        r"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+"
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_.$]*)[ \t]*\("
        r"(?P<parameters>.*?)\)(?P<header>[^{]*?)\{"
    )
    parameter_pattern = re.compile(
        r"\.param[ \t]+\.u64[ \t]+\.ptr[ \t]+\.global"
        r"(?:[ \t]+\.align[ \t]+[1-9][0-9]*)?"
        r"[ \t]+[A-Za-z_$][A-Za-z0-9_.$]*"
    )
    observed: dict[str, int] = {}
    for match in entry_pattern.finditer(ptx):
        name = match.group("name")
        if name in observed:
            raise BackendError(f"MLIR PTX entry {name!r} is declared more than once")
        parameters = match.group("parameters")
        declarations = tuple(parameter_pattern.finditer(parameters))
        remainder = parameter_pattern.sub("", parameters)
        if remainder.strip(" \t\r\n,"):
            raise BackendError(
                f"MLIR PTX entry {name!r} does not use the required global u64 pointer ABI"
            )
        maxntid = re.findall(
            r"(?m)^[ \t]*\.maxntid[ \t]+([0-9]+)[ \t]*,[ \t]*"
            r"([0-9]+)[ \t]*,[ \t]*([0-9]+)[ \t]*$",
            match.group("header"),
        )
        if maxntid != [("256", "1", "1")]:
            raise BackendError(
                f"MLIR PTX entry {name!r} requires .maxntid 256, 1, 1"
            )
        observed[name] = len(declarations)

    if set(observed) != set(_PTX_POINTER_PARAMETER_COUNTS):
        raise BackendError("MLIR PTX pointer ABI does not describe the exact FFN entries")
    for name, expected_count in _PTX_POINTER_PARAMETER_COUNTS.items():
        actual_count = observed[name]
        if actual_count != expected_count:
            raise BackendError(
                f"MLIR PTX entry {name!r} requires {expected_count} global u64 "
                f"pointer parameters, found {actual_count}"
            )


@dataclass(frozen=True, slots=True)
class MlirPtxLaunch:
    """One CUDA launch whose implementation is an entry in ``program.ptx``."""

    abi_version: int
    symbol: str
    inputs: tuple[str, ...]
    output: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    dtype: str
    layout: str
    node_ids: tuple[str, ...]
    source_nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "abi_version": self.abi_version,
            "symbol": self.symbol,
            "kind": "mlir-ptx-kernel",
            "implementation": "official-mlir-gpu-nvvm-ptx",
            "inputs": list(self.inputs),
            "output": self.output,
            "grid": list(self.grid),
            "block": list(self.block),
            "dtype": self.dtype,
            "layout": self.layout,
            "node_ids": list(self.node_ids),
            "source_nodes": list(self.source_nodes),
        }


@dataclass(frozen=True, slots=True)
class MlirPtxProgram:
    """A compiled FFN artifact containing the actual PTX executed at runtime."""

    abi_version: int
    source_graph: Graph
    graph: Graph
    reference_graph: Graph
    tensor_source: str
    gpu_source: str
    binary_mlir: str
    ptx: str
    launches: tuple[MlirPtxLaunch, ...]
    pass_reports: tuple[dict[str, Any], ...]
    toolchain: dict[str, Any]
    integrity_sha256: str
    executable: bool

    @property
    def target(self) -> str:
        return TARGET

    @property
    def compute_capability(self) -> tuple[int, int]:
        return COMPUTE_CAPABILITY

    @property
    def kernels(self) -> tuple[MlirPtxLaunch, ...]:
        """Compatibility alias for runtimes that call launch records kernels."""

        return self.launches

    @property
    def source(self) -> str:
        """The device MLIR that was actually lowered, rather than an ABI manifest."""

        return self.gpu_source

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.abi_version,
            "runtime_abi_version": self.abi_version,
            "compiler_backend": TARGET,
            "target": TARGET,
            "target_intent": "cuda-sm86",
            "compute_capability": list(COMPUTE_CAPABILITY),
            "lowering_stage": "ptx",
            "executable": self.executable,
            "artifact_kind": "official-mlir-gpu-nvvm-ptx",
            "mlir_source_role": "lowering-input-for-executed-ptx",
            "mlir_source_lowered": True,
            "mlir_source_executed": False,
            "executed_artifact": "textual-ptx",
            "runtime_dispatch_source": "validated-ptx-launch-records",
            "runtime_requirements": ["CUDA Driver", "CuPy"],
            "matmul_lowering": "mlir-gpu-serial-f32",
            "precision": {
                "storage": "float32",
                "accumulator": "float32",
                "tensor_core_policy": "disabled-in-correctness-baseline",
            },
            "profile": {
                "query_shape": list(PRODUCTION_QUERY_SHAPE),
                "rows": PRODUCTION_ROWS,
                "channels": PRODUCTION_CHANNELS,
                "hidden_channels": PRODUCTION_HIDDEN_CHANNELS,
            },
            "dispatch_count": len(self.launches),
            "launches": [launch.to_dict() for launch in self.launches],
            "toolchain": _json_value(self.toolchain),
            "integrity": {
                "algorithm": "sha256",
                "integrity_sha256": self.integrity_sha256,
                "tensor_mlir_sha256": _sha256_text(self.tensor_source),
                "gpu_mlir_sha256": _sha256_text(self.gpu_source),
                "binary_mlir_sha256": _sha256_text(self.binary_mlir),
                "ptx_sha256": _sha256_text(self.ptx),
                "launch_plan_sha256": _sha256_json(
                    [launch.to_dict() for launch in self.launches]
                ),
            },
            "pass_reports": list(self.pass_reports),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    tensor_source: str,
    gpu_source: str,
    binary_mlir: str,
    ptx: str,
    launches: tuple[MlirPtxLaunch, ...],
    pass_reports: tuple[dict[str, Any], ...],
    toolchain: Mapping[str, Any],
) -> str:
    payload = {
        "abi_version": RUNTIME_ABI_VERSION,
        "target": TARGET,
        "compute_capability": list(COMPUTE_CAPABILITY),
        "source_graph_sha256": _graph_sha256(source_graph),
        "graph_sha256": _graph_sha256(graph),
        "reference_graph_sha256": _graph_sha256(reference_graph),
        "tensor_source_sha256": _sha256_text(tensor_source),
        "gpu_source_sha256": _sha256_text(gpu_source),
        "binary_mlir_sha256": _sha256_text(binary_mlir),
        "ptx_sha256": _sha256_text(ptx),
        "launches": [launch.to_dict() for launch in launches],
        "pass_reports": list(pass_reports),
        "toolchain": _json_value(toolchain),
        "executable": True,
        "mlir_source_role": "lowering-input-for-executed-ptx",
        "mlir_source_lowered": True,
        "mlir_source_executed": False,
        "executed_artifact": "textual-ptx",
        "runtime_dispatch_source": "validated-ptx-launch-records",
    }
    return _sha256_json(payload)


def _production_hybrid(graph: Graph):
    """Reuse the established strict FFN matcher and return its canonical plan."""

    hybrid = compile_mlir_cuda(graph)
    query_name = next(iter(hybrid.graph.inputs))
    query_spec = hybrid.graph.inputs[query_name]
    if query_spec.shape != PRODUCTION_QUERY_SHAPE:
        _reject(
            "the first PTX slice requires query shape "
            f"{PRODUCTION_QUERY_SHAPE}, got {query_spec.shape}"
        )
    if query_spec.dtype != "float32" or query_spec.layout != "contiguous":
        _reject("the first PTX slice requires contiguous float32 query storage")

    fc1, bias_relu, fc2, final = hybrid.calls
    expected_matmuls = (
        (fc1.m, fc1.n, fc1.k),
        (fc2.m, fc2.n, fc2.k),
    )
    if expected_matmuls != (
        (PRODUCTION_ROWS, PRODUCTION_HIDDEN_CHANNELS, PRODUCTION_CHANNELS),
        (PRODUCTION_ROWS, PRODUCTION_CHANNELS, PRODUCTION_HIDDEN_CHANNELS),
    ):
        _reject(f"the production FFN MatMul contract is invalid: {expected_matmuls}")
    if (
        final.rows != PRODUCTION_ROWS
        or final.channels != PRODUCTION_CHANNELS
        or final.epsilon is None
    ):
        _reject("the production FFN LayerNorm contract is invalid")
    return hybrid


def _build_launches(hybrid) -> tuple[MlirPtxLaunch, ...]:
    fc1, bias_relu, fc2, final = hybrid.calls
    hidden_elements = PRODUCTION_ROWS * PRODUCTION_HIDDEN_CHANNELS

    first_sources = _ordered_unique((*fc1.source_nodes, *bias_relu.source_nodes))
    launches = (
        MlirPtxLaunch(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=FC1_SYMBOL,
            inputs=(fc1.inputs[0], fc1.inputs[1], bias_relu.inputs[1]),
            output=bias_relu.output,
            grid=((hidden_elements + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            dtype="float32",
            layout="contiguous",
            node_ids=(*fc1.node_ids, *bias_relu.node_ids),
            source_nodes=first_sources,
        ),
        MlirPtxLaunch(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=FC2_SYMBOL,
            inputs=fc2.inputs,
            output=fc2.output,
            grid=(
                (
                    PRODUCTION_ROWS * PRODUCTION_CHANNELS
                    + THREADS_PER_BLOCK
                    - 1
                )
                // THREADS_PER_BLOCK,
                1,
                1,
            ),
            block=(THREADS_PER_BLOCK, 1, 1),
            dtype="float32",
            layout="contiguous",
            node_ids=fc2.node_ids,
            source_nodes=fc2.source_nodes,
        ),
        MlirPtxLaunch(
            abi_version=RUNTIME_ABI_VERSION,
            symbol=FINAL_SYMBOL,
            inputs=final.inputs,
            output=final.output,
            grid=(PRODUCTION_ROWS, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            dtype="float32",
            layout="contiguous",
            node_ids=final.node_ids,
            source_nodes=final.source_nodes,
        ),
    )
    return launches


def _emit_gpu_mlir(hybrid) -> str:
    """Emit registered GPU dialect for the exact production FFN ABI.

    Static memrefs plus the toolchain's kernel bare-pointer convention produce
    one PTX pointer parameter per logical tensor.  The leading batch dimension
    is flattened into ``PRODUCTION_ROWS``; contiguous storage makes that a
    pointer-only ABI change, not a data transformation.
    """

    fc1, bias_relu, fc2, final = hybrid.calls
    query = fc1.inputs[0]
    projected = fc2.output
    projected_input, _fc2_bias, residual, _norm_weight, _norm_bias = final.inputs
    if projected_input != projected or residual != query:
        _reject("canonical FFN call plan changed while emitting GPU MLIR")
    epsilon = format(float(final.epsilon), ".9g")
    if "." not in epsilon and "e" not in epsilon.lower():
        epsilon += ".0"

    # SSA and symbol names are fixed.  User-controlled graph names and
    # provenance are deliberately absent from the emitted syntax.
    return f'''// GAC {TARGET}: production BEVFormer FFN device module.
module attributes {{gpu.container_module}} {{
  gpu.module @gac_bevformer_ffn_sm86 {{
    gpu.func @{FC1_SYMBOL}(
        %query: memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %weight: memref<{PRODUCTION_CHANNELS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>,
        %bias: memref<{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>,
        %output: memref<{PRODUCTION_ROWS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: {THREADS_PER_BLOCK}, 1, 1>}} {{
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index
      %count = arith.constant {PRODUCTION_ROWS * PRODUCTION_HIDDEN_CHANNELS} : index
      %in_bounds = arith.cmpi ult, %linear, %count : index
      scf.if %in_bounds {{
        %width = arith.constant {PRODUCTION_HIDDEN_CHANNELS} : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index
        %lower = arith.constant 0 : index
        %upper = arith.constant {PRODUCTION_CHANNELS} : index
        %step = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %k = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {{
          %lhs = memref.load %query[%row, %k]
            : memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
          %rhs = memref.load %weight[%k, %column]
            : memref<{PRODUCTION_CHANNELS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>
          %product = arith.mulf %lhs, %rhs : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }}
        %bias_value = memref.load %bias[%column]
          : memref<{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>
        %biased = arith.addf %sum, %bias_value : f32
        %relu = arith.maximumf %biased, %zero : f32
        memref.store %relu, %output[%row, %column]
          : memref<{PRODUCTION_ROWS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>
      }}
      gpu.return
    }}

    gpu.func @{FC2_SYMBOL}(
        %input: memref<{PRODUCTION_ROWS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>,
        %weight: memref<{PRODUCTION_HIDDEN_CHANNELS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %output: memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: {THREADS_PER_BLOCK}, 1, 1>}} {{
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index
      %count = arith.constant {PRODUCTION_ROWS * PRODUCTION_CHANNELS} : index
      %in_bounds = arith.cmpi ult, %linear, %count : index
      scf.if %in_bounds {{
        %width = arith.constant {PRODUCTION_CHANNELS} : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index
        %lower = arith.constant 0 : index
        %upper = arith.constant {PRODUCTION_HIDDEN_CHANNELS} : index
        %step = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %k = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {{
          %lhs = memref.load %input[%row, %k]
            : memref<{PRODUCTION_ROWS}x{PRODUCTION_HIDDEN_CHANNELS}xf32, #gpu.address_space<global>>
          %rhs = memref.load %weight[%k, %column]
            : memref<{PRODUCTION_HIDDEN_CHANNELS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
          %product = arith.mulf %lhs, %rhs : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }}
        memref.store %sum, %output[%row, %column]
          : memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      }}
      gpu.return
    }}

    gpu.func @{FINAL_SYMBOL}(
        %projected: memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %bias: memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %residual: memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %gamma: memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %beta: memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>,
        %output: memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: {THREADS_PER_BLOCK}, 1, 1>}} {{
      %row = gpu.block_id x
      %column = gpu.thread_id x
      %projected_value = memref.load %projected[%row, %column]
        : memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      %bias_value = memref.load %bias[%column]
        : memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      %residual_value = memref.load %residual[%row, %column]
        : memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      %biased = arith.addf %projected_value, %bias_value : f32
      %value = arith.addf %biased, %residual_value : f32
      %sum = gpu.all_reduce add %value uniform {{}} : (f32) -> (f32)
      %width = arith.constant 256.0 : f32
      %mean = arith.divf %sum, %width : f32
      %centered = arith.subf %value, %mean : f32
      %square = arith.mulf %centered, %centered : f32
      %square_sum = gpu.all_reduce add %square uniform {{}} : (f32) -> (f32)
      %variance = arith.divf %square_sum, %width : f32
      %epsilon = arith.constant {epsilon} : f32
      %adjusted = arith.addf %variance, %epsilon : f32
      %inv_stddev = nvvm.rsqrt %adjusted : f32
      %normalized = arith.mulf %centered, %inv_stddev : f32
      %scale = memref.load %gamma[%column]
        : memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      %shift = memref.load %beta[%column]
        : memref<{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      %scaled = arith.mulf %normalized, %scale : f32
      %result = arith.addf %scaled, %shift : f32
      memref.store %result, %output[%row, %column]
        : memref<{PRODUCTION_ROWS}x{PRODUCTION_CHANNELS}xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def _toolchain_metadata(artifact: MlirPtxArtifact) -> dict[str, Any]:
    return {
        "name": "mlir-opt",
        "command": list(artifact.command),
        "version": artifact.mlir_opt_version,
        "llvm_source_revision": artifact.mlir_opt_source_revision,
        "source_sha256": artifact.source_sha256,
        "chip": artifact.target_chip,
        "features": artifact.target_features,
        "ptx_version": artifact.ptx_version,
        "entries": list(artifact.entries),
    }


def _assemble_from_hybrid(hybrid: Any, artifact: MlirPtxArtifact) -> MlirPtxProgram:
    if not isinstance(artifact, MlirPtxArtifact):
        raise TypeError("artifact must be an MlirPtxArtifact")
    if not isinstance(artifact.binary_mlir, str) or not artifact.binary_mlir.strip():
        raise BackendError("MLIR PTX artifact binary MLIR must be non-empty text")
    if not isinstance(artifact.ptx, str) or not artifact.ptx.strip():
        raise BackendError("MLIR PTX artifact PTX must be non-empty text")
    if (
        not isinstance(artifact.source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact.source_sha256) is None
    ):
        raise BackendError("MLIR PTX artifact source SHA-256 is invalid")
    if (
        not isinstance(artifact.command, tuple)
        or not artifact.command
        or any(not isinstance(item, str) or not item for item in artifact.command)
    ):
        raise BackendError("MLIR PTX artifact compiler command is invalid")
    if not isinstance(artifact.mlir_opt_version, str) or not artifact.mlir_opt_version:
        raise BackendError("MLIR PTX artifact compiler version is invalid")
    if artifact.mlir_opt_source_revision is not None and (
        not isinstance(artifact.mlir_opt_source_revision, str)
        or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            artifact.mlir_opt_source_revision,
        )
        is None
    ):
        raise BackendError("MLIR PTX artifact compiler revision is invalid")
    if (
        not isinstance(artifact.ptx_version, str)
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", artifact.ptx_version) is None
    ):
        raise BackendError("MLIR PTX artifact PTX version is invalid")

    gpu_source = _emit_gpu_mlir(hybrid)
    expected_source_sha256 = _sha256_text(gpu_source)
    if not hmac.compare_digest(artifact.source_sha256, expected_source_sha256):
        raise BackendError(
            "MLIR PTX artifact was not produced from this canonical FFN GPU module"
        )
    if artifact.target_chip != "sm_86" or artifact.target_features != "+ptx71":
        raise BackendError("MLIR PTX artifact target metadata is not SM86/PTX 7.1")
    if artifact.entries != PTX_ENTRY_SYMBOLS:
        raise BackendError("MLIR PTX artifact entry metadata is inconsistent")

    # Re-extract from the lowered MLIR instead of trusting two independent
    # strings supplied by a caller.  The toolchain helper rejects ambiguous
    # gpu.binary objects, residual gpu.func operations, malformed escaping,
    # wrong targets, duplicate entries and unexpected entries.
    extracted_ptx = extract_sm86_ptx(
        artifact.binary_mlir, expected_entries=PTX_ENTRY_SYMBOLS
    )
    if not hmac.compare_digest(extracted_ptx, artifact.ptx):
        raise BackendError(
            "MLIR PTX artifact binary MLIR and extracted PTX do not match"
        )

    launches = _build_launches(hybrid)
    toolchain = _toolchain_metadata(artifact)
    pass_reports = tuple(hybrid.pass_reports)
    source_graph = hybrid.source_graph.clone()
    optimized_graph = hybrid.graph.clone()
    reference_graph = hybrid.reference_graph.clone()
    integrity = _integrity_sha256(
        source_graph=source_graph,
        graph=optimized_graph,
        reference_graph=reference_graph,
        tensor_source=hybrid.tensor_source,
        gpu_source=gpu_source,
        binary_mlir=artifact.binary_mlir,
        ptx=artifact.ptx,
        launches=launches,
        pass_reports=pass_reports,
        toolchain=toolchain,
    )
    program = MlirPtxProgram(
        abi_version=RUNTIME_ABI_VERSION,
        source_graph=source_graph,
        graph=optimized_graph,
        reference_graph=reference_graph,
        tensor_source=hybrid.tensor_source,
        gpu_source=gpu_source,
        binary_mlir=artifact.binary_mlir,
        ptx=artifact.ptx,
        launches=launches,
        pass_reports=pass_reports,
        toolchain=toolchain,
        integrity_sha256=integrity,
        executable=True,
    )
    validate_mlir_ptx_program(program)
    return program


def assemble_mlir_ptx_program(
    graph: Graph, artifact: MlirPtxArtifact
) -> MlirPtxProgram:
    """Bind a verified offline PTX artifact to its canonical FFN graph.

    This is the deployment boundary for GPU hosts that do not install LLVM or
    MLIR.  The source hash binds ``artifact`` to the deterministically emitted
    GPU module, and the lowered MLIR is re-parsed to prove that ``artifact.ptx``
    is the exact, single SM86 object produced offline.
    """

    return _assemble_from_hybrid(_production_hybrid(graph), artifact)


def compile_mlir_ptx(
    graph: Graph,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> MlirPtxProgram:
    """Compile the production FFN through official MLIR to executable PTX.

    ``mlir-opt`` is an offline compiler dependency.  The resulting runtime
    artifact contains PTX and no CUDA C source, cuBLAS call or NVRTC fallback.
    """

    hybrid = _production_hybrid(graph)
    gpu_source = _emit_gpu_mlir(hybrid)

    # Import lazily so source emission and matcher tests remain useful without
    # an installed MLIR toolchain, and so the offline dependency is explicit.
    from gac.toolchains.mlir import lower_gpu_mlir_to_ptx

    artifact = lower_gpu_mlir_to_ptx(
        gpu_source,
        expected_entries=PTX_ENTRY_SYMBOLS,
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    return _assemble_from_hybrid(hybrid, artifact)


def validate_mlir_ptx_program(program: MlirPtxProgram) -> None:
    """Validate an in-memory artifact without invoking MLIR or touching CUDA."""

    if not isinstance(program, MlirPtxProgram):
        raise BackendError("MLIR PTX runtime artifact must be an MlirPtxProgram")
    if program.abi_version != RUNTIME_ABI_VERSION or program.executable is not True:
        raise BackendError("MLIR PTX runtime artifact execution metadata is invalid")

    expected_hybrid = _production_hybrid(program.source_graph)
    expected_gpu_source = _emit_gpu_mlir(expected_hybrid)
    expected_launches = _build_launches(expected_hybrid)
    comparisons = (
        (
            "source graph",
            _graph_sha256(program.source_graph),
            _graph_sha256(expected_hybrid.source_graph),
        ),
        (
            "optimized graph",
            _graph_sha256(program.graph),
            _graph_sha256(expected_hybrid.graph),
        ),
        (
            "reference graph",
            _graph_sha256(program.reference_graph),
            _graph_sha256(expected_hybrid.reference_graph),
        ),
        ("tensor MLIR", program.tensor_source, expected_hybrid.tensor_source),
        ("GPU MLIR", program.gpu_source, expected_gpu_source),
        ("PTX launch plan", program.launches, expected_launches),
        ("pass reports", program.pass_reports, tuple(expected_hybrid.pass_reports)),
    )
    for component, actual, canonical in comparisons:
        if actual != canonical:
            raise BackendError(
                f"MLIR PTX runtime artifact {component} differs from its canonical FFN plan"
            )

    if not program.binary_mlir.strip() or "gpu.binary" not in program.binary_mlir:
        raise BackendError("MLIR PTX runtime artifact has no lowered gpu.binary module")
    if "\x00" in program.ptx:
        raise BackendError("MLIR PTX runtime artifact contains a NUL byte")

    # Re-parse the post-lowering MLIR on every runtime validation.  The
    # integrity digest detects accidental field changes, while this check also
    # proves that the PTX about to be loaded is exactly the sole assembly
    # object carried by the lowered MLIR rather than a second, unrelated text
    # field with a separately recomputed digest.
    try:
        extracted_ptx = extract_sm86_ptx(
            program.binary_mlir, expected_entries=PTX_ENTRY_SYMBOLS
        )
    except BackendError as exc:
        raise BackendError(
            f"MLIR PTX runtime artifact has invalid lowered MLIR: {exc}"
        ) from exc
    if not hmac.compare_digest(extracted_ptx, program.ptx):
        raise BackendError(
            "MLIR PTX runtime artifact binary MLIR and PTX do not match"
        )
    _validate_ptx_pointer_abi(program.ptx)

    directive_patterns = {
        "version": r"(?m)^[ \t]*\.version[ \t]+([^\s]+)[ \t]*$",
        "target": r"(?m)^[ \t]*\.target[ \t]+(sm_[0-9]+)(?:[ \t]*,[^\r\n]+)?[ \t]*$",
        "address size": r"(?m)^[ \t]*\.address_size[ \t]+([0-9]+)[ \t]*$",
    }
    directives = {
        name: re.findall(pattern, program.ptx)
        for name, pattern in directive_patterns.items()
    }
    for name, values in directives.items():
        if len(values) != 1:
            raise BackendError(
                f"MLIR PTX runtime artifact requires exactly one {name} directive"
            )
    if directives["target"][0] != "sm_86" or directives["address size"][0] != "64":
        raise BackendError("MLIR PTX runtime artifact target ABI is not SM86/64-bit")
    entries = tuple(
        re.findall(
            r"(?m)^[ \t]*(?:\.visible[ \t]+)?\.entry[ \t]+"
            r"([A-Za-z_$][A-Za-z0-9_$.]*)[ \t]*\(",
            program.ptx,
        )
    )
    if len(entries) != len(set(entries)) or set(entries) != set(PTX_ENTRY_SYMBOLS):
        raise BackendError(
            "MLIR PTX runtime artifact entry set is not the exact FFN kernel ABI"
        )
    if tuple(program.toolchain.get("entries", ())) != PTX_ENTRY_SYMBOLS:
        raise BackendError("MLIR PTX toolchain entry metadata is inconsistent")
    if program.toolchain.get("chip") != "sm_86":
        raise BackendError("MLIR PTX toolchain target metadata is inconsistent")
    if program.toolchain.get("features") != "+ptx71":
        raise BackendError("MLIR PTX toolchain feature metadata is inconsistent")
    if program.toolchain.get("source_sha256") != _sha256_text(program.gpu_source):
        raise BackendError("MLIR PTX toolchain source digest is inconsistent")
    if program.toolchain.get("ptx_version") != directives["version"][0]:
        raise BackendError("MLIR PTX version metadata is inconsistent")

    current_integrity = _integrity_sha256(
        source_graph=program.source_graph,
        graph=program.graph,
        reference_graph=program.reference_graph,
        tensor_source=program.tensor_source,
        gpu_source=program.gpu_source,
        binary_mlir=program.binary_mlir,
        ptx=program.ptx,
        launches=program.launches,
        pass_reports=program.pass_reports,
        toolchain=program.toolchain,
    )
    if current_integrity != program.integrity_sha256:
        raise BackendError("MLIR PTX runtime artifact integrity digest does not match")


__all__ = [
    "COMPUTE_CAPABILITY",
    "FC1_SYMBOL",
    "FC2_SYMBOL",
    "FINAL_SYMBOL",
    "MlirPtxLaunch",
    "MlirPtxProgram",
    "PTX_ENTRY_SYMBOLS",
    "RUNTIME_ABI_VERSION",
    "TARGET",
    "assemble_mlir_ptx_program",
    "compile_mlir_ptx",
    "validate_mlir_ptx_program",
]
