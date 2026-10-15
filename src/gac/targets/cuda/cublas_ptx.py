"""cuBLAS MatMul plus official MLIR-to-PTX static FFN lowering.

This backend is the production-oriented bridge between the existing fast
``mlir-cuda-sm86`` call plan and the fully generated ``mlir-ptx-sm86``
correctness path.  Both MatMul operations remain explicit CuPy/cuBLAS calls.
The two pointwise/reduction stages are emitted as registered GPU dialect,
lowered by upstream ``mlir-opt`` through NVVM, and executed from textual PTX.

The compiler is deliberately strict: it accepts the frozen production FP32
BEVFormer ``[1, 2500, 256]`` and UniAD ``[1, 40000, 256]`` FFN profiles, both
with ``256 -> 512 -> 256`` channels.  Unsupported graphs fail instead of
falling back to CUDA C, NVRTC, NumPy, or the legacy CUDA backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import os
import re
from typing import Any

from gac.core.errors import BackendError
from gac.core.ir import Graph
from gac.toolchains.mlir import DEFAULT_TIMEOUT_SECONDS, MlirPtxArtifact, extract_sm86_ptx
from gac.targets.cuda.mlir import COMPUTE_CAPABILITY, MlirCudaCall, RUNTIME_ABI_VERSION, compile_mlir_cuda


TARGET = "mlir-cublas-ptx-sm86"
PRODUCTION_QUERY_SHAPE = (1, 2500, 256)
PRODUCTION_ROWS = 2500
PRODUCTION_CHANNELS = 256
PRODUCTION_HIDDEN_CHANNELS = 512
UNIAD_QUERY_SHAPE = (1, 40000, 256)
SUPPORTED_PRODUCTION_QUERY_SHAPES = (
    PRODUCTION_QUERY_SHAPE,
    UNIAD_QUERY_SHAPE,
)
THREADS_PER_BLOCK = 256

FC1_SYMBOL = "gac_cublas_fc1_matmul_f32"
BIAS_RELU_SYMBOL = "gac_bias_relu_inplace_f32"
FC2_SYMBOL = "gac_cublas_fc2_matmul_f32"
FINAL_SYMBOL = "gac_bias_residual_layer_norm_f32"
PTX_ENTRY_SYMBOLS = (BIAS_RELU_SYMBOL, FINAL_SYMBOL)


@dataclass(frozen=True, slots=True)
class MlirCublasPtxProgram:
    """Validated hybrid artifact whose custom kernels are executable PTX."""

    abi_version: int
    source_graph: Graph
    graph: Graph
    reference_graph: Graph
    tensor_source: str
    gpu_source: str
    binary_mlir: str
    ptx: str
    calls: tuple[MlirCudaCall, ...]
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
    def source(self) -> str:
        """Return the GPU MLIR that produced the executed PTX."""

        return self.gpu_source

    def describe(self) -> dict[str, Any]:
        query_shape = tuple(next(iter(self.graph.inputs.values())).shape)
        fc1 = self.calls[0]
        return {
            "schema_version": self.abi_version,
            "runtime_abi_version": self.abi_version,
            "compiler_backend": TARGET,
            "target": TARGET,
            "target_intent": "cuda-sm86",
            "compute_capability": list(COMPUTE_CAPABILITY),
            "lowering_stage": "cublas-runtime-calls-plus-ptx",
            "executable": self.executable,
            "artifact_kind": "mlir-cublas-calls-with-official-mlir-nvvm-ptx",
            "mlir_source_role": "lowering-input-for-executed-fusion-ptx",
            "mlir_source_lowered": True,
            "executed_custom_artifact": "textual-ptx",
            "runtime_dispatch_source": "validated-hybrid-call-records",
            "runtime_requirements": ["CUDA Driver", "cuBLAS", "CuPy"],
            "nvrtc_required": False,
            "matmul_lowering": "cupy-cublas",
            "fusion_lowering": "mlir-gpu-nvvm-ptx",
            "precision": {
                "storage": "float32",
                "cublas_compute_policy": "runtime-default",
                "tensor_core_policy": "not-forced-by-compiler",
            },
            "profile": {
                "query_shape": list(query_shape),
                "rows": fc1.m,
                "channels": fc1.k,
                "hidden_channels": fc1.n,
            },
            "dispatch_count": len(self.calls),
            "cublas_dispatch_count": sum(
                call.implementation == "cupy-cublas" for call in self.calls
            ),
            "ptx_dispatch_count": sum(
                call.implementation == "mlir-ptx" for call in self.calls
            ),
            "kernel_fusions": ["bias+relu", "bias+residual+layernorm"],
            "calls": [call.to_dict() for call in self.calls],
            "toolchain": _json_value(self.toolchain),
            "integrity": {
                "algorithm": "sha256",
                "integrity_sha256": self.integrity_sha256,
                "tensor_mlir_sha256": _sha256_text(self.tensor_source),
                "gpu_mlir_sha256": _sha256_text(self.gpu_source),
                "binary_mlir_sha256": _sha256_text(self.binary_mlir),
                "ptx_sha256": _sha256_text(self.ptx),
                "call_plan_sha256": _sha256_json(
                    [call.to_dict() for call in self.calls]
                ),
            },
            "pass_reports": list(self.pass_reports),
        }


def _reject(reason: str) -> None:
    raise BackendError(f"{TARGET}: {reason}")


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
    digest.update(
        json.dumps(
            graph.to_dict(include_constant_data=False),
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


def _production_hybrid(graph: Graph):
    """Reuse the strict FFN matcher and validate one frozen production profile."""

    hybrid = compile_mlir_cuda(graph)
    if len(hybrid.graph.inputs) != 1:
        _reject("the production FFN requires exactly one input")
    query_name = next(iter(hybrid.graph.inputs))
    query_spec = hybrid.graph.inputs[query_name]
    if query_spec.shape not in SUPPORTED_PRODUCTION_QUERY_SHAPES:
        _reject(
            "the production hybrid slice requires query shape in "
            f"{SUPPORTED_PRODUCTION_QUERY_SHAPES}, got {query_spec.shape}"
        )
    if query_spec.dtype != "float32" or query_spec.layout != "contiguous":
        _reject("the production hybrid slice requires contiguous float32 query storage")

    fc1, bias_relu, fc2, final = hybrid.calls
    rows = query_spec.shape[0] * query_spec.shape[1]
    if tuple(call.symbol for call in hybrid.calls) != (
        FC1_SYMBOL,
        BIAS_RELU_SYMBOL,
        FC2_SYMBOL,
        FINAL_SYMBOL,
    ):
        _reject("the canonical FFN runtime symbol contract changed")
    if (fc1.m, fc1.n, fc1.k) != (
        rows,
        PRODUCTION_HIDDEN_CHANNELS,
        PRODUCTION_CHANNELS,
    ):
        _reject("the production FC1 MatMul contract is invalid")
    if (fc2.m, fc2.n, fc2.k) != (
        rows,
        PRODUCTION_CHANNELS,
        PRODUCTION_HIDDEN_CHANNELS,
    ):
        _reject("the production FC2 MatMul contract is invalid")
    if (
        bias_relu.rows != rows
        or bias_relu.channels != PRODUCTION_HIDDEN_CHANNELS
        or final.rows != rows
        or final.channels != PRODUCTION_CHANNELS
        or final.epsilon is None
    ):
        _reject("the production fusion contract is invalid")
    return hybrid


def _build_calls(hybrid: Any) -> tuple[MlirCudaCall, ...]:
    fc1, bias_relu, fc2, final = hybrid.calls
    return (
        fc1,
        replace(bias_relu, implementation="mlir-ptx"),
        fc2,
        replace(final, implementation="mlir-ptx"),
    )


def _float_literal(value: float) -> str:
    token = format(float(value), ".9g")
    if "." not in token and "e" not in token.lower():
        token += ".0"
    return token


def _emit_gpu_mlir(hybrid: Any) -> str:
    """Emit the two fusion kernels as registered MLIR GPU dialect."""

    _fc1, bias_relu, _fc2, final = hybrid.calls
    epsilon = _float_literal(float(final.epsilon))
    rows = int(final.rows)
    channels = int(final.channels)
    hidden_channels = int(bias_relu.channels)
    hidden_elements = rows * hidden_channels
    model_name = "BEVFormer" if rows == PRODUCTION_ROWS else "UniAD"
    module_name = (
        "gac_bevformer_ffn_fusions_sm86"
        if rows == PRODUCTION_ROWS
        else "gac_uniad_ffn_fusions_sm86"
    )
    return f'''// GAC {TARGET}: production {model_name} FFN fusion module.
module attributes {{gpu.container_module}} {{
  gpu.module @{module_name} {{
    gpu.func @{BIAS_RELU_SYMBOL}(
        %value: memref<{rows}x{hidden_channels}xf32, #gpu.address_space<global>>,
        %bias: memref<{hidden_channels}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: {THREADS_PER_BLOCK}, 1, 1>}} {{
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index
      %count = arith.constant {hidden_elements} : index
      %in_bounds = arith.cmpi ult, %linear, %count : index
      scf.if %in_bounds {{
        %width = arith.constant {hidden_channels} : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index
        %input = memref.load %value[%row, %column]
          : memref<{rows}x{hidden_channels}xf32, #gpu.address_space<global>>
        %bias_value = memref.load %bias[%column]
          : memref<{hidden_channels}xf32, #gpu.address_space<global>>
        %biased = arith.addf %input, %bias_value : f32
        %zero = arith.constant 0.0 : f32
        %relu = arith.maximumf %biased, %zero : f32
        memref.store %relu, %value[%row, %column]
          : memref<{rows}x{hidden_channels}xf32, #gpu.address_space<global>>
      }}
      gpu.return
    }}

    gpu.func @{FINAL_SYMBOL}(
        %projected: memref<{rows}x{channels}xf32, #gpu.address_space<global>>,
        %bias: memref<{channels}xf32, #gpu.address_space<global>>,
        %residual: memref<{rows}x{channels}xf32, #gpu.address_space<global>>,
        %gamma: memref<{channels}xf32, #gpu.address_space<global>>,
        %beta: memref<{channels}xf32, #gpu.address_space<global>>,
        %output: memref<{rows}x{channels}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: {THREADS_PER_BLOCK}, 1, 1>}} {{
      %row = gpu.block_id x
      %column = gpu.thread_id x
      %projected_value = memref.load %projected[%row, %column]
        : memref<{rows}x{channels}xf32, #gpu.address_space<global>>
      %bias_value = memref.load %bias[%column]
        : memref<{channels}xf32, #gpu.address_space<global>>
      %residual_value = memref.load %residual[%row, %column]
        : memref<{rows}x{channels}xf32, #gpu.address_space<global>>
      %biased = arith.addf %projected_value, %bias_value : f32
      %value = arith.addf %biased, %residual_value : f32
      %sum = gpu.all_reduce add %value uniform {{}} : (f32) -> (f32)
      %width = arith.constant {float(channels):.1f} : f32
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
        : memref<{channels}xf32, #gpu.address_space<global>>
      %shift = memref.load %beta[%column]
        : memref<{channels}xf32, #gpu.address_space<global>>
      %scaled = arith.mulf %normalized, %scale : f32
      %result = arith.addf %scaled, %shift : f32
      memref.store %result, %output[%row, %column]
        : memref<{rows}x{channels}xf32, #gpu.address_space<global>>
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


def _integrity_sha256(
    *,
    source_graph: Graph,
    graph: Graph,
    reference_graph: Graph,
    tensor_source: str,
    gpu_source: str,
    binary_mlir: str,
    ptx: str,
    calls: tuple[MlirCudaCall, ...],
    pass_reports: tuple[dict[str, Any], ...],
    toolchain: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
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
            "calls": [call.to_dict() for call in calls],
            "pass_reports": list(pass_reports),
            "toolchain": _json_value(toolchain),
            "executable": True,
            "runtime_dispatch_source": "validated-hybrid-call-records",
        }
    )


def _validate_artifact_fields(artifact: MlirPtxArtifact, gpu_source: str) -> None:
    if not isinstance(artifact, MlirPtxArtifact):
        raise TypeError("artifact must be an MlirPtxArtifact")
    if not artifact.binary_mlir.strip() or not artifact.ptx.strip():
        _reject("MLIR PTX artifact must contain binary MLIR and PTX")
    if artifact.source_sha256 != _sha256_text(gpu_source):
        _reject("MLIR PTX artifact is not bound to the canonical fusion module")
    if artifact.target_chip != "sm_86" or artifact.target_features != "+ptx71":
        _reject("MLIR PTX artifact target metadata is not SM86/PTX 7.1")
    if artifact.ptx_version != "7.1":
        _reject("MLIR PTX artifact must use PTX version 7.1")
    if artifact.entries != PTX_ENTRY_SYMBOLS:
        _reject("MLIR PTX artifact entry metadata is inconsistent")
    if not artifact.command or any(not value for value in artifact.command):
        _reject("MLIR PTX artifact compiler command is invalid")
    if not artifact.mlir_opt_version:
        _reject("MLIR PTX artifact compiler version is invalid")


def _entry_header(ptx: str, symbol: str) -> tuple[str, str]:
    pattern = re.compile(
        rf"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+{re.escape(symbol)}"
        rf"[ \t]*\((.*?)\)[ \t]*\r?\n[ \t]*\.maxntid[ \t]+([^\r\n]+)"
    )
    matches = pattern.findall(ptx)
    if len(matches) != 1:
        _reject(f"PTX entry {symbol!r} has no unambiguous ABI header")
    return matches[0]


def _validate_ptx_pointer_abi(ptx: str) -> None:
    expected_counts = {BIAS_RELU_SYMBOL: 2, FINAL_SYMBOL: 6}
    parameter = re.compile(
        r"^\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?"
        r"\s+[A-Za-z_$][A-Za-z0-9_.$]*,?$"
    )
    for symbol, count in expected_counts.items():
        header, maxntid = _entry_header(ptx, symbol)
        lines = [line.strip() for line in header.splitlines() if line.strip()]
        if len(lines) != count or any(parameter.fullmatch(line) is None for line in lines):
            _reject(
                f"PTX entry {symbol!r} requires exactly {count} bare global pointers"
            )
        dimensions = tuple(
            int(value) for value in re.findall(r"[0-9]+", maxntid)
        )
        if dimensions != (THREADS_PER_BLOCK, 1, 1):
            _reject(
                f"PTX entry {symbol!r} requires .maxntid "
                f"{THREADS_PER_BLOCK}, 1, 1"
            )


def _assemble_from_hybrid(
    hybrid: Any, artifact: MlirPtxArtifact
) -> MlirCublasPtxProgram:
    gpu_source = _emit_gpu_mlir(hybrid)
    _validate_artifact_fields(artifact, gpu_source)
    extracted = extract_sm86_ptx(
        artifact.binary_mlir, expected_entries=PTX_ENTRY_SYMBOLS
    )
    if not hmac.compare_digest(extracted, artifact.ptx):
        _reject("binary MLIR and extracted PTX do not match")
    _validate_ptx_pointer_abi(artifact.ptx)

    source_graph = hybrid.source_graph.clone()
    graph = hybrid.graph.clone()
    reference_graph = hybrid.reference_graph.clone()
    calls = _build_calls(hybrid)
    pass_reports = tuple(hybrid.pass_reports)
    toolchain = _toolchain_metadata(artifact)
    integrity = _integrity_sha256(
        source_graph=source_graph,
        graph=graph,
        reference_graph=reference_graph,
        tensor_source=hybrid.tensor_source,
        gpu_source=gpu_source,
        binary_mlir=artifact.binary_mlir,
        ptx=artifact.ptx,
        calls=calls,
        pass_reports=pass_reports,
        toolchain=toolchain,
    )
    program = MlirCublasPtxProgram(
        abi_version=RUNTIME_ABI_VERSION,
        source_graph=source_graph,
        graph=graph,
        reference_graph=reference_graph,
        tensor_source=hybrid.tensor_source,
        gpu_source=gpu_source,
        binary_mlir=artifact.binary_mlir,
        ptx=artifact.ptx,
        calls=calls,
        pass_reports=pass_reports,
        toolchain=toolchain,
        integrity_sha256=integrity,
        executable=True,
    )
    validate_mlir_cublas_ptx_program(program)
    return program


def assemble_mlir_cublas_ptx_program(
    graph: Graph, artifact: MlirPtxArtifact
) -> MlirCublasPtxProgram:
    """Bind an offline MLIR PTX artifact to the canonical production graph."""

    return _assemble_from_hybrid(_production_hybrid(graph), artifact)


def compile_mlir_cublas_ptx(
    graph: Graph,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> MlirCublasPtxProgram:
    """Compile fusion kernels to PTX and retain MatMuls as cuBLAS calls."""

    hybrid = _production_hybrid(graph)
    gpu_source = _emit_gpu_mlir(hybrid)
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


def validate_mlir_cublas_ptx_program(program: MlirCublasPtxProgram) -> None:
    """Validate the complete artifact without invoking MLIR or touching CUDA."""

    if not isinstance(program, MlirCublasPtxProgram):
        raise BackendError(
            "MLIR cuBLAS/PTX runtime artifact must be an MlirCublasPtxProgram"
        )
    if (
        isinstance(program.abi_version, bool)
        or not isinstance(program.abi_version, int)
        or program.abi_version != RUNTIME_ABI_VERSION
        or program.executable is not True
    ):
        raise BackendError("MLIR cuBLAS/PTX artifact execution metadata is invalid")

    expected_hybrid = _production_hybrid(program.source_graph)
    expected_calls = _build_calls(expected_hybrid)
    expected_gpu_source = _emit_gpu_mlir(expected_hybrid)
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
        ("fusion GPU MLIR", program.gpu_source, expected_gpu_source),
        ("runtime call plan", program.calls, expected_calls),
        ("pass reports", program.pass_reports, tuple(expected_hybrid.pass_reports)),
    )
    for component, actual, canonical in comparisons:
        if actual != canonical:
            raise BackendError(
                f"MLIR cuBLAS/PTX artifact {component} differs from its canonical plan"
            )

    try:
        extracted = extract_sm86_ptx(
            program.binary_mlir, expected_entries=PTX_ENTRY_SYMBOLS
        )
    except BackendError as exc:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact has invalid lowered MLIR: {exc}"
        ) from exc
    if not hmac.compare_digest(extracted, program.ptx):
        raise BackendError("MLIR cuBLAS/PTX artifact binary MLIR and PTX do not match")
    _validate_ptx_pointer_abi(program.ptx)

    if tuple(program.toolchain.get("entries", ())) != PTX_ENTRY_SYMBOLS:
        raise BackendError("MLIR cuBLAS/PTX toolchain entry metadata is inconsistent")
    if (
        program.toolchain.get("chip") != "sm_86"
        or program.toolchain.get("features") != "+ptx71"
        or program.toolchain.get("ptx_version") != "7.1"
    ):
        raise BackendError("MLIR cuBLAS/PTX toolchain target metadata is inconsistent")
    if program.toolchain.get("source_sha256") != _sha256_text(program.gpu_source):
        raise BackendError("MLIR cuBLAS/PTX toolchain source digest is inconsistent")

    current = _integrity_sha256(
        source_graph=program.source_graph,
        graph=program.graph,
        reference_graph=program.reference_graph,
        tensor_source=program.tensor_source,
        gpu_source=program.gpu_source,
        binary_mlir=program.binary_mlir,
        ptx=program.ptx,
        calls=program.calls,
        pass_reports=program.pass_reports,
        toolchain=program.toolchain,
    )
    if current != program.integrity_sha256:
        raise BackendError("MLIR cuBLAS/PTX artifact integrity digest does not match")


__all__ = [
    "BIAS_RELU_SYMBOL",
    "COMPUTE_CAPABILITY",
    "FC1_SYMBOL",
    "FC2_SYMBOL",
    "FINAL_SYMBOL",
    "MlirCublasPtxProgram",
    "PTX_ENTRY_SYMBOLS",
    "PRODUCTION_QUERY_SHAPE",
    "RUNTIME_ABI_VERSION",
    "SUPPORTED_PRODUCTION_QUERY_SHAPES",
    "TARGET",
    "UNIAD_QUERY_SHAPE",
    "assemble_mlir_cublas_ptx_program",
    "compile_mlir_cublas_ptx",
    "validate_mlir_cublas_ptx_program",
]
