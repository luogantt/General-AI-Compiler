"""Executable three-dispatch MLIR/PTX BEVFormer FFN.

The first dispatch is a shared-memory tiled FP32 MatMul with its bias and
ReLU epilogue fused into the generated kernel.  The second dispatch is the
same tiled MatMul core without an epilogue.  The third dispatch reuses the
MLIR-generated bias, residual and LayerNorm PTX from the cuBLAS/PTX backend.

Every tuning field is executable state: ``tile_m``, ``tile_n`` and ``tile_k``
are used to emit both MatMul modules, and the recorded CUDA block is derived
from those modules.  Validation reconstructs that relationship before CUDA is
touched; metadata-only tile substitutions are rejected.
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

from .backends.mlir_cublas_ptx import (
    COMPUTE_CAPABILITY,
    FINAL_SYMBOL,
    MlirCublasPtxProgram,
    compile_mlir_cublas_ptx,
    validate_mlir_cublas_ptx_program,
)
from .backends.mlir_cuda import MlirCudaCall, RUNTIME_ABI_VERSION
from .errors import BackendError
from .ir import Graph
from .mlir_tiled_matmul import (
    TiledMatmulConfig,
    TiledMatmulPtxProgram,
    _validate_tiled_matmul_ptx_abi,
    bevformer_fc1_tiled_config,
    bevformer_fc2_tiled_config,
    emit_tiled_matmul_gpu_mlir,
    inspect_tiled_matmul_ptx,
    lower_tiled_matmul_to_ptx,
)
from .mlir_toolchain import DEFAULT_TIMEOUT_SECONDS, MlirPtxArtifact, extract_sm86_ptx


TARGET = "mlir-tiled-ffn-sm86"
IMPLEMENTATION = "mlir-tiled-ptx"
DEFAULT_TILE = (16, 16, 16)
SUPPORTED_PRODUCTION_TILES = ((8, 8, 16), (16, 16, 16))


def _reject(reason: str) -> None:
    raise BackendError(f"{TARGET}: {reason}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _graph_sha256(graph: Graph) -> str:
    graph.validate()
    digest = hashlib.sha256(
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
            _reject(f"constant {name!r} must be C-contiguous")
        digest.update(name.encode("utf-8"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _artifact_record(program: TiledMatmulPtxProgram) -> dict[str, Any]:
    artifact = program.artifact
    return {
        "config": {
            "symbol": program.config.symbol,
            "m": program.config.m,
            "n": program.config.n,
            "k": program.config.k,
            "tile_m": program.config.tile_m,
            "tile_n": program.config.tile_n,
            "tile_k": program.config.tile_k,
            "block": list(program.config.block),
            "grid": list(program.config.grid),
            "fuse_bias_relu": program.config.fuse_bias_relu,
        },
        "gpu_source_sha256": _sha256_text(program.gpu_source),
        "binary_mlir_sha256": _sha256_text(artifact.binary_mlir),
        "ptx_sha256": _sha256_text(artifact.ptx),
        "toolchain": {
            "command": list(artifact.command),
            "version": artifact.mlir_opt_version,
            "llvm_source_revision": artifact.mlir_opt_source_revision,
            "source_sha256": artifact.source_sha256,
            "chip": artifact.target_chip,
            "features": artifact.target_features,
            "ptx_version": artifact.ptx_version,
            "entries": list(artifact.entries),
        },
        "evidence": {
            "shared_declarations": program.evidence.shared_declarations,
            "static_shared_bytes": program.evidence.static_shared_bytes,
            "shared_loads": program.evidence.shared_loads,
            "shared_stores": program.evidence.shared_stores,
            "global_stores": program.evidence.global_stores,
            "vector_shared_loads": program.evidence.vector_shared_loads,
            "barriers": program.evidence.barriers,
            "fp32_multiplies": program.evidence.fp32_multiplies,
            "fp32_adds": program.evidence.fp32_adds,
            "fma_instructions": program.evidence.fma_instructions,
            "mma_instructions": program.evidence.mma_instructions,
            "async_copy_instructions": program.evidence.async_copy_instructions,
        },
    }


def _merge_provenance(*calls: MlirCudaCall) -> tuple[tuple[str, ...], tuple[str, ...]]:
    node_ids: list[str] = []
    sources: list[str] = []
    for call in calls:
        for node_id in call.node_ids:
            if node_id not in node_ids:
                node_ids.append(node_id)
        for source in call.source_nodes:
            if source not in sources:
                sources.append(source)
    return tuple(node_ids), tuple(sources)


def _build_calls(
    base: MlirCublasPtxProgram,
    fc1: TiledMatmulPtxProgram,
    fc2: TiledMatmulPtxProgram,
) -> tuple[MlirCudaCall, MlirCudaCall, MlirCudaCall]:
    base_fc1, bias_relu, base_fc2, final = base.calls
    fc1_nodes, fc1_sources = _merge_provenance(base_fc1, bias_relu)
    tiled_fc1 = replace(
        base_fc1,
        symbol=fc1.config.symbol,
        kind="fused-matmul",
        implementation=IMPLEMENTATION,
        inputs=(base_fc1.inputs[0], base_fc1.inputs[1], bias_relu.inputs[1]),
        output=bias_relu.output,
        node_ids=fc1_nodes,
        source_nodes=fc1_sources,
        grid=fc1.grid,
        block=fc1.block,
    )
    tiled_fc2 = replace(
        base_fc2,
        symbol=fc2.config.symbol,
        implementation=IMPLEMENTATION,
        grid=fc2.grid,
        block=fc2.block,
    )
    return tiled_fc1, tiled_fc2, final


def _integrity_sha256(
    base: MlirCublasPtxProgram,
    fc1: TiledMatmulPtxProgram,
    fc2: TiledMatmulPtxProgram,
    calls: tuple[MlirCudaCall, ...],
) -> str:
    return _sha256_json(
        {
            "abi_version": RUNTIME_ABI_VERSION,
            "target": TARGET,
            "compute_capability": list(COMPUTE_CAPABILITY),
            "base_integrity_sha256": base.integrity_sha256,
            "fc1": _artifact_record(fc1),
            "fc2": _artifact_record(fc2),
            "calls": [call.to_dict() for call in calls],
            "dispatch_count": 3,
            "executable": True,
        }
    )


@dataclass(frozen=True, slots=True)
class MlirTiledFfnProgram:
    """Three executable PTX dispatches bound to one production FFN graph."""

    abi_version: int
    base_program: MlirCublasPtxProgram
    fc1: TiledMatmulPtxProgram
    fc2: TiledMatmulPtxProgram
    calls: tuple[MlirCudaCall, MlirCudaCall, MlirCudaCall]
    integrity_sha256: str
    executable: bool

    @property
    def target(self) -> str:
        return TARGET

    @property
    def graph(self) -> Graph:
        return self.base_program.graph

    @property
    def source_graph(self) -> Graph:
        return self.base_program.source_graph

    @property
    def reference_graph(self) -> Graph:
        return self.base_program.reference_graph

    @property
    def matmul_tile(self) -> tuple[int, int, int]:
        config = self.fc1.config
        return config.tile_m, config.tile_n, config.tile_k

    @property
    def thread_block(self) -> tuple[int, int, int]:
        return self.fc1.block

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.abi_version,
            "runtime_abi_version": self.abi_version,
            "compiler_backend": TARGET,
            "target": TARGET,
            "target_intent": "cuda-sm86",
            "compute_capability": list(COMPUTE_CAPABILITY),
            "artifact_kind": "three-mlir-nvvm-ptx-dispatches",
            "executable": self.executable,
            "nvrtc_required": False,
            "precision": "fp32",
            "dispatch_count": len(self.calls),
            "matmul_dispatch_count": 2,
            "ptx_dispatch_count": 3,
            "matmul_lowering": "explicit-shared-memory-tiled-mlir-gpu",
            "fusion_lowering": "fc1-bias-relu-and-final-bias-residual-layernorm",
            "matmul_tile": list(self.matmul_tile),
            "thread_block": list(self.thread_block),
            "single_buffered_shared_memory": True,
            "async_double_buffered": False,
            "calls": [call.to_dict() for call in self.calls],
            "components": {
                "fc1": _artifact_record(self.fc1),
                "fc2": _artifact_record(self.fc2),
                "final_fusion": {
                    "program_integrity_sha256": self.base_program.integrity_sha256,
                    "ptx_sha256": _sha256_text(self.base_program.ptx),
                    "executed_entry": FINAL_SYMBOL,
                },
            },
            "integrity": {
                "algorithm": "sha256",
                "integrity_sha256": self.integrity_sha256,
                "base_integrity_sha256": self.base_program.integrity_sha256,
                "call_plan_sha256": _sha256_json(
                    [call.to_dict() for call in self.calls]
                ),
            },
        }


def _validate_component(
    program: TiledMatmulPtxProgram, expected: TiledMatmulConfig
) -> None:
    if not isinstance(program, TiledMatmulPtxProgram):
        _reject("MatMul component has the wrong program type")
    if program.config != expected:
        _reject(f"MatMul component {expected.symbol!r} has different tile metadata")
    canonical_source = emit_tiled_matmul_gpu_mlir(expected)
    if program.gpu_source != canonical_source:
        _reject(f"MatMul component {expected.symbol!r} has non-canonical GPU MLIR")
    artifact = program.artifact
    if not isinstance(artifact, MlirPtxArtifact):
        _reject(f"MatMul component {expected.symbol!r} has no MLIR PTX artifact")
    if (
        not isinstance(artifact.binary_mlir, str)
        or not artifact.binary_mlir.strip()
        or not isinstance(artifact.ptx, str)
        or not artifact.ptx.strip()
        or not isinstance(artifact.command, tuple)
        or not artifact.command
        or any(not isinstance(value, str) or not value for value in artifact.command)
        or not isinstance(artifact.mlir_opt_version, str)
        or not artifact.mlir_opt_version
        or (
            artifact.mlir_opt_source_revision is not None
            and (
                not isinstance(artifact.mlir_opt_source_revision, str)
                or not artifact.mlir_opt_source_revision
            )
        )
    ):
        _reject(f"MatMul component {expected.symbol!r} toolchain artifact is invalid")
    if artifact.source_sha256 != _sha256_text(canonical_source):
        _reject(f"MatMul component {expected.symbol!r} source digest differs")
    if (
        artifact.target_chip != "sm_86"
        or artifact.target_features != "+ptx71"
        or artifact.ptx_version != "7.1"
        or artifact.entries != (expected.symbol,)
    ):
        _reject(f"MatMul component {expected.symbol!r} target metadata differs")
    try:
        extracted = extract_sm86_ptx(
            artifact.binary_mlir, expected_entries=(expected.symbol,)
        )
    except BackendError as exc:
        raise BackendError(
            f"{TARGET}: MatMul component {expected.symbol!r} has invalid binary MLIR: {exc}"
        ) from exc
    if not hmac.compare_digest(extracted, artifact.ptx):
        _reject(f"MatMul component {expected.symbol!r} binary MLIR and PTX differ")
    _validate_tiled_matmul_ptx_abi(artifact.ptx, expected)
    evidence = inspect_tiled_matmul_ptx(artifact.ptx)
    if evidence != program.evidence:
        _reject(f"MatMul component {expected.symbol!r} PTX evidence differs")
    if (
        evidence.shared_declarations != 2
        or evidence.static_shared_bytes != expected.shared_memory_bytes
        or evidence.shared_loads == 0
        or evidence.shared_stores == 0
        or evidence.barriers < 2
        or not evidence.computes_real_matmul
    ):
        _reject(
            f"MatMul component {expected.symbol!r} lost its tiled MatMul computation"
        )
    if evidence.async_copy_instructions != 0:
        _reject(f"MatMul component {expected.symbol!r} unexpectedly changed buffering policy")
    if re.search(r"\bmalloc\b|\blocal_depot", artifact.ptx):
        _reject(f"MatMul component {expected.symbol!r} uses local allocation")


def assemble_tiled_matmul_program(
    config: TiledMatmulConfig, artifact: MlirPtxArtifact
) -> TiledMatmulPtxProgram:
    """Reconstruct and validate a tiled component from an offline artifact."""

    source = emit_tiled_matmul_gpu_mlir(config)
    program = TiledMatmulPtxProgram(
        config=config,
        gpu_source=source,
        artifact=artifact,
        evidence=inspect_tiled_matmul_ptx(artifact.ptx),
    )
    _validate_component(program, config)
    return program


def _require_common_tile(
    fc1: TiledMatmulPtxProgram, fc2: TiledMatmulPtxProgram
) -> tuple[int, int, int]:
    first = (fc1.config.tile_m, fc1.config.tile_n, fc1.config.tile_k)
    second = (fc2.config.tile_m, fc2.config.tile_n, fc2.config.tile_k)
    if first != second:
        _reject("FC1 and FC2 must use the same autotuned tile")
    if fc1.config.block != fc2.config.block:
        _reject("FC1 and FC2 must use the same CUDA thread block")
    return first


def assemble_mlir_tiled_ffn_program(
    base_program: MlirCublasPtxProgram,
    fc1: TiledMatmulPtxProgram,
    fc2: TiledMatmulPtxProgram,
) -> MlirTiledFfnProgram:
    """Bind validated tiled MatMuls to an existing MLIR final-fusion artifact."""

    validate_mlir_cublas_ptx_program(base_program)
    tile = _require_common_tile(fc1, fc2)
    expected_fc1 = bevformer_fc1_tiled_config(
        tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
    )
    expected_fc2 = bevformer_fc2_tiled_config(
        tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
    )
    _validate_component(fc1, expected_fc1)
    _validate_component(fc2, expected_fc2)
    calls = _build_calls(base_program, fc1, fc2)
    program = MlirTiledFfnProgram(
        abi_version=RUNTIME_ABI_VERSION,
        base_program=base_program,
        fc1=fc1,
        fc2=fc2,
        calls=calls,
        integrity_sha256=_integrity_sha256(base_program, fc1, fc2, calls),
        executable=True,
    )
    validate_mlir_tiled_ffn_program(program)
    return program


def compile_mlir_tiled_ffn(
    graph: Graph,
    *,
    tile_m: int = DEFAULT_TILE[0],
    tile_n: int = DEFAULT_TILE[1],
    tile_k: int = DEFAULT_TILE[2],
    base_program: MlirCublasPtxProgram | None = None,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> MlirTiledFfnProgram:
    """Compile two real tiled MatMuls and the final MLIR PTX fusion."""

    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    base = base_program
    if base is None:
        base = compile_mlir_cublas_ptx(
            graph,
            mlir_opt=mlir_opt,
            command=command,
            use_wsl=use_wsl,
            wsl_distribution=wsl_distribution,
            timeout=timeout,
            env=env,
            llvm_root=llvm_root,
        )
    else:
        validate_mlir_cublas_ptx_program(base)
        # The base artifact carries the graph constants and exact optimized
        # plan; a caller cannot pair it with a different source graph.
        if _graph_sha256(base.source_graph) != _graph_sha256(graph):
            _reject("supplied base program belongs to a different graph")

    kwargs = {
        "mlir_opt": mlir_opt,
        "command": command,
        "use_wsl": use_wsl,
        "wsl_distribution": wsl_distribution,
        "timeout": timeout,
        "env": env,
        "llvm_root": llvm_root,
    }
    fc1 = lower_tiled_matmul_to_ptx(
        bevformer_fc1_tiled_config(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
        ),
        **kwargs,
    )
    fc2 = lower_tiled_matmul_to_ptx(
        bevformer_fc2_tiled_config(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k
        ),
        **kwargs,
    )
    return assemble_mlir_tiled_ffn_program(base, fc1, fc2)


def validate_mlir_tiled_ffn_program(program: MlirTiledFfnProgram) -> None:
    """Validate the complete three-dispatch artifact without CUDA access."""

    if not isinstance(program, MlirTiledFfnProgram):
        _reject("runtime artifact must be an MlirTiledFfnProgram")
    if (
        isinstance(program.abi_version, bool)
        or not isinstance(program.abi_version, int)
        or program.abi_version != RUNTIME_ABI_VERSION
        or program.executable is not True
        or not isinstance(program.integrity_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", program.integrity_sha256) is None
    ):
        _reject("execution metadata is invalid")
    validate_mlir_cublas_ptx_program(program.base_program)
    tile = _require_common_tile(program.fc1, program.fc2)
    _validate_component(
        program.fc1,
        bevformer_fc1_tiled_config(
            tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
        ),
    )
    _validate_component(
        program.fc2,
        bevformer_fc2_tiled_config(
            tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
        ),
    )
    expected_calls = _build_calls(program.base_program, program.fc1, program.fc2)
    if program.calls != expected_calls:
        _reject("runtime call plan differs from the compiled tiled artifacts")
    if tuple(call.implementation for call in program.calls) != (
        IMPLEMENTATION,
        IMPLEMENTATION,
        "mlir-ptx",
    ):
        _reject("runtime implementation order is invalid")
    current = _integrity_sha256(
        program.base_program, program.fc1, program.fc2, program.calls
    )
    if not hmac.compare_digest(current, program.integrity_sha256):
        _reject("artifact integrity digest does not match")


__all__ = [
    "DEFAULT_TILE",
    "IMPLEMENTATION",
    "MlirTiledFfnProgram",
    "SUPPORTED_PRODUCTION_TILES",
    "TARGET",
    "assemble_mlir_tiled_ffn_program",
    "assemble_tiled_matmul_program",
    "compile_mlir_tiled_ffn",
    "validate_mlir_tiled_ffn_program",
]
