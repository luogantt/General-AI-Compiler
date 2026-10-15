"""Whole-model BEVFormer analysis, partitioning, and lowering plans.

The complete BEVFormer streaming graph remains the source of truth.  This
module discovers the FFN islands supported by the Vortex backend, assigns all
remaining CNN, Attention, state, geometry, and detection work to the complete
CUDA fallback, builds a real dispatch DAG, and attaches global buffer/runtime
and per-region schedule metadata.

The plan is deliberately honest about the current execution boundary.  CUDA
subgraphs can be emitted as CUDA C and each accepted FFN can be emitted as
standard Tensor/Linalg MLIR, but a unified CUDA/Vortex device session is not
implemented yet.  Cross-target values are therefore explicit runtime transfer
requirements rather than an implied shared device pointer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .backends.cuda import compile_cuda
from .backends.mlir import compile_mlir
from .backends.vortex_buffered_mlir import VortexBufferPlan
from .bevformer_partition import (
    BEVFORMER_CUDA_TARGET,
    BEVFORMER_VORTEX_TARGET,
    BevFormerPartitionReport,
    analyze_bevformer_partitions,
    extract_bevformer_ffn_region,
)
from .bevformer_step_pipeline import (
    BevFormerStepPipelineConfig,
    build_bevformer_step_pipeline_graph,
)
from .dispatch_runtime import DispatchRuntimePlan, build_dispatch_runtime_plan
from .errors import ValidationError
from .ir import Graph
from .memory import MemoryPlan
from .partition import Dispatch, DispatchDAG, NodePlacement, partition_graph


BEVFORMER_WHOLE_MODEL_PLAN_SCHEMA_VERSION = 1


def _semantic_kind(group: str | None) -> str:
    if group is None:
        return "utility"
    return group.split(":", 1)[0]


@dataclass(frozen=True, slots=True)
class DispatchStrategy:
    """Tiling, local-buffer, schedule, and lowering intent for one dispatch."""

    dispatch_id: str
    target: str
    kind: str
    tile_axes: tuple[str, ...]
    schedule: str
    local_buffer: str
    lowering: str
    implementation_state: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "target": self.target,
            "kind": self.kind,
            "tile_axes": list(self.tile_axes),
            "schedule": self.schedule,
            "local_buffer": self.local_buffer,
            "lowering": self.lowering,
            "implementation_state": self.implementation_state,
            "parameters": dict(self.parameters),
        }


def _strategy(dispatch: Dispatch, buffer_plan: VortexBufferPlan) -> DispatchStrategy:
    kind = _semantic_kind(dispatch.group)
    if kind == "cnn":
        return DispatchStrategy(
            dispatch.id,
            dispatch.target,
            kind,
            ("N", "C_out", "H", "W", "C_in"),
            "blocked-convolution-and-pointwise-fusion",
            "backend-managed shared memory/register tiles",
            "Canonical Graph -> CUDA C/NVRTC; eligible Conv2D may select cuDNN",
            "CUDA source lowering implemented; Vortex CNN lowering not implemented",
            {"selection": "cudnn-or-generated-cuda"},
        )
    if kind == "attention":
        return DispatchStrategy(
            dispatch.id,
            dispatch.target,
            kind,
            ("query", "head", "level", "sampling_point", "channel"),
            "attention-projection-msda-softmax-projection",
            "Q/K/V/context temporaries are dispatch-local",
            "Canonical Graph -> CUDA C with cuBLAS candidates for MatMul",
            "CUDA Attention/MSDA lowering implemented; Vortex Attention lowering not implemented",
            {"softmax_axis": "last", "msda": "generated-cuda"},
        )
    if kind == "ffn" and dispatch.target == BEVFORMER_VORTEX_TARGET:
        return DispatchStrategy(
            dispatch.id,
            dispatch.target,
            kind,
            ("M/rows", "N/output", "K/reduction"),
            "dxa-double-buffer",
            "two LMEM stages for A/B tiles; C accumulator remains in registers",
            "Canonical FFN -> Tensor/Linalg MLIR -> LLVM Dialect MLIR -> VOLT",
            "canonical MLIR and direct Vortex lowering implemented per FFN; "
            "unified mixed-device session pending",
            buffer_plan.to_dict(),
        )
    return DispatchStrategy(
        dispatch.id,
        dispatch.target,
        kind,
        ("linear element index",),
        "topological generated-kernel schedule",
        "dispatch-local temporaries; global values follow the runtime plan",
        "Canonical Graph -> CUDA C/NVRTC",
        "CUDA source lowering implemented",
        {"block_threads": 256},
    )


@dataclass(frozen=True, slots=True)
class BevFormerWholeModelPlan:
    """Validated offline plan for a complete BEVFormer graph."""

    graph: Graph
    capability_report: BevFormerPartitionReport
    dispatch_dag: DispatchDAG
    runtime_plan: DispatchRuntimePlan
    strategies: tuple[DispatchStrategy, ...]
    vortex_ffn_enabled: bool
    cuda_fallback_verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategies", tuple(self.strategies))
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.graph, Graph):
            raise ValidationError("whole-model plan graph must be a Graph")
        if not isinstance(self.vortex_ffn_enabled, bool):
            raise ValidationError("vortex_ffn_enabled must be boolean")
        if not isinstance(self.cuda_fallback_verified, bool):
            raise ValidationError("cuda_fallback_verified must be boolean")
        self.graph.validate()
        if self.dispatch_dag.source_graph is not self.graph:
            raise ValidationError("dispatch DAG must retain the planned source graph")
        self.dispatch_dag.validate()
        self.runtime_plan.validate()
        if self.runtime_plan.dag.to_dict() != self.dispatch_dag.to_dict():
            raise ValidationError("runtime plan and dispatch DAG disagree")
        strategy_ids = tuple(strategy.dispatch_id for strategy in self.strategies)
        if strategy_ids != self.dispatch_dag.schedule:
            raise ValidationError("every dispatch must have one ordered strategy")
        for strategy in self.strategies:
            dispatch = self.dispatch_dag.dispatch(strategy.dispatch_id)
            if strategy.target != dispatch.target:
                raise ValidationError(
                    f"strategy {strategy.dispatch_id!r} changed the dispatch target"
                )
            if strategy.kind != _semantic_kind(dispatch.group):
                raise ValidationError(
                    f"strategy {strategy.dispatch_id!r} changed the semantic kind"
                )
        placement_ids = tuple(item.node_id for item in self.capability_report.placements)
        if placement_ids != tuple(node.id for node in self.graph.nodes):
            raise ValidationError("capability report does not cover the complete graph")
        placements = {
            placement.node_id: placement
            for placement in self.capability_report.placements
        }
        for dispatch in self.dispatch_dag.dispatches:
            kinds = {placements[node_id].kind for node_id in dispatch.node_ids}
            if len(kinds) != 1:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} mixes semantic kinds {sorted(kinds)}"
                )
            if dispatch.target == BEVFORMER_VORTEX_TARGET:
                region_ids = {
                    placements[node_id].region_id for node_id in dispatch.node_ids
                }
                if None in region_ids or len(region_ids) != 1:
                    raise ValidationError(
                        f"Vortex dispatch {dispatch.id!r} is not one strict FFN region"
                    )

    @property
    def mixed_target(self) -> bool:
        return len({dispatch.target for dispatch in self.dispatch_dag.dispatches}) > 1

    @property
    def physical_runtime_ready(self) -> bool:
        # The existing monolithic CUDA Session remains executable.  This new
        # plan intentionally refuses to claim a mixed CUDA/Vortex Session
        # until a Vortex RuntimeDriver and copy bridge implement its steps.
        return False

    @property
    def cross_target_edge_instances(self) -> int:
        """Count producer/consumer pairs which need a physical copy bridge."""

        count = 0
        for edge in self.dispatch_dag.edges:
            if edge.producer is None:
                continue
            producer = self.dispatch_dag.dispatch(edge.producer)
            count += sum(
                self.dispatch_dag.dispatch(consumer).target != producer.target
                for consumer in edge.consumers
            )
        return count

    @property
    def memory_plan(self) -> MemoryPlan:
        """The authoritative static allocation plan for every source value."""

        return self.runtime_plan.memory_plan

    def vortex_graph_for_dispatch(self, dispatch_id: str) -> Graph:
        """Return the exact canonical FFN Graph accepted by ``vortex-v3``."""

        dispatch = self.dispatch_dag.dispatch(dispatch_id)
        if dispatch.target != BEVFORMER_VORTEX_TARGET:
            raise ValidationError(
                f"dispatch {dispatch_id!r} is not a current Vortex FFN region"
            )
        region = next(
            (
                candidate
                for candidate in self.capability_report.regions
                if candidate.node_ids == dispatch.node_ids
            ),
            None,
        )
        if region is None:
            raise ValidationError(
                f"dispatch {dispatch_id!r} has no strict FFN region record"
            )
        return extract_bevformer_ffn_region(self.graph, region)

    def coverage(self) -> dict[str, Any]:
        total_nodes = len(self.graph.nodes)
        vortex_nodes = sum(
            len(dispatch.node_ids)
            for dispatch in self.dispatch_dag.dispatches
            if dispatch.target == BEVFORMER_VORTEX_TARGET
        )
        return {
            "cuda_fallback_verified": self.cuda_fallback_verified,
            "full_model_native_vortex": total_nodes > 0 and vortex_nodes == total_nodes,
            "vortex": {
                "target": BEVFORMER_VORTEX_TARGET,
                "supported_scope": "strict static contiguous FP32 BEVFormer FFN only",
                "region_count": sum(
                    dispatch.target == BEVFORMER_VORTEX_TARGET
                    for dispatch in self.dispatch_dag.dispatches
                ),
                "node_count": vortex_nodes,
                "lowering": (
                    "Tensor/Linalg MLIR -> LLVM Dialect device MLIR -> "
                    "VOLT -> RV32 Vortex vxbin"
                ),
                "not_supported_yet": [
                    "ResNet/FPN CNN",
                    "temporal and spatial Attention",
                    "decoder self/cross Attention",
                    "geometry, detection, state, and utility operators",
                ],
            },
            "cuda": {
                "target": BEVFORMER_CUDA_TARGET,
                "fallback_node_count": total_nodes - vortex_nodes,
                "role": "complete static BEVFormer fallback and non-FFN execution",
            },
        }

    def to_dict(self) -> dict[str, Any]:
        targets: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for dispatch in self.dispatch_dag.dispatches:
            targets[dispatch.target] = targets.get(dispatch.target, 0) + 1
            kind = _semantic_kind(dispatch.group)
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "schema_version": BEVFORMER_WHOLE_MODEL_PLAN_SCHEMA_VERSION,
            "kind": "bevformer-whole-model-heterogeneous-offline-plan",
            "graph": self.graph.name,
            "node_count": len(self.graph.nodes),
            "graph_inputs": list(self.graph.inputs),
            "graph_outputs": list(self.graph.outputs),
            "vortex_ffn_enabled": self.vortex_ffn_enabled,
            "cuda_fallback_verified": self.cuda_fallback_verified,
            "capability_analysis": self.capability_report.to_dict(),
            "dispatch_count": len(self.dispatch_dag.dispatches),
            "dispatches_by_target": dict(sorted(targets.items())),
            "dispatches_by_kind": dict(sorted(kinds.items())),
            "runtime_plan": self.runtime_plan.to_dict(),
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "coverage": self.coverage(),
            "execution_status": {
                "physical_runtime_ready": self.physical_runtime_ready,
                "existing_monolithic_cuda_path": True,
                "mixed_target": self.mixed_target,
                "cross_target_edge_instances": self.cross_target_edge_instances,
                "blockers": [
                    "plan-driven Vortex RuntimeDriver is not implemented",
                    "CUDA/Vortex cross-device copy bridge is not implemented",
                    "CNN and Attention do not yet have Vortex lowering",
                ],
            },
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def plan_bevformer_graph(
    graph: Graph,
    *,
    enable_vortex_ffn: bool = True,
    alignment: int = 256,
    vortex_buffer_plan: VortexBufferPlan | None = None,
    verify_cuda_fallback: bool = True,
) -> BevFormerWholeModelPlan:
    """Analyze and partition one canonical BEVFormer graph.

    Strict FFN regions use ``vortex-v3`` when enabled.  Every remaining node
    uses the existing complete ``cuda-sm86`` backend.  Semantic groups are
    kept as compilation boundaries even when adjacent groups share CUDA.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("plan_bevformer_graph requires a Graph")
    if not isinstance(enable_vortex_ffn, bool):
        raise ValidationError("enable_vortex_ffn must be boolean")
    if not isinstance(verify_cuda_fallback, bool):
        raise ValidationError("verify_cuda_fallback must be boolean")
    graph.validate()
    if verify_cuda_fallback:
        # This performs static graph checks and CUDA source generation.  It
        # does not import a CUDA Runtime, invoke NVRTC, or execute on a GPU.
        compile_cuda(graph)
    report = analyze_bevformer_partitions(graph)
    placement: dict[str, NodePlacement] = {}
    for item in report.placements:
        target = item.target if enable_vortex_ffn else BEVFORMER_CUDA_TARGET
        if item.region_id is not None:
            group = f"ffn:{item.region_id}"
        else:
            group = item.kind
        placement[item.node_id] = NodePlacement(target, group)
    dag = partition_graph(graph, placement)
    runtime_plan = build_dispatch_runtime_plan(dag, alignment=alignment)
    buffer_plan = vortex_buffer_plan or VortexBufferPlan()
    strategies = tuple(_strategy(dispatch, buffer_plan) for dispatch in dag.dispatches)
    return BevFormerWholeModelPlan(
        graph,
        report,
        dag,
        runtime_plan,
        strategies,
        enable_vortex_ffn,
        verify_cuda_fallback,
    )


def build_bevformer_whole_model_plan(
    encoder_weights: Mapping[str, Any],
    decoder_weights: Mapping[str, Any],
    config: BevFormerStepPipelineConfig | None = None,
    *,
    name: str = "bevformer_streaming_step",
    enable_vortex_ffn: bool = True,
    alignment: int = 256,
    vortex_buffer_plan: VortexBufferPlan | None = None,
    verify_cuda_fallback: bool = True,
) -> BevFormerWholeModelPlan:
    """Build the complete stateful BEVFormer Graph and heterogeneous plan."""

    graph = build_bevformer_step_pipeline_graph(
        encoder_weights,
        decoder_weights,
        config,
        name=name,
    )
    if graph.metadata.get("kind") != "bevformer-complete-stateful-streaming-step-static":
        raise ValidationError("BEVFormer step builder returned an unexpected graph kind")
    return plan_bevformer_graph(
        graph,
        enable_vortex_ffn=enable_vortex_ffn,
        alignment=alignment,
        vortex_buffer_plan=vortex_buffer_plan,
        verify_cuda_fallback=verify_cuda_fallback,
    )


@dataclass(frozen=True, slots=True)
class LoweredDispatchSource:
    dispatch_id: str
    target: str
    kind: str
    artifact_kind: str
    source_sha256: str
    source_bytes: int
    source_path: str | None
    kernel_count: int | None
    executable_binary: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "target": self.target,
            "kind": self.kind,
            "artifact_kind": self.artifact_kind,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "source_path": self.source_path,
            "kernel_count": self.kernel_count,
            "executable_binary": self.executable_binary,
        }


def lower_bevformer_dispatch_sources(
    plan: BevFormerWholeModelPlan,
    *,
    output_directory: str | Path | None = None,
) -> tuple[LoweredDispatchSource, ...]:
    """Emit reviewable CUDA or Tensor/Linalg MLIR for every dispatch.

    This is source lowering, not mixed-device execution.  Vortex dispatches
    stop at the canonical Tensor/Linalg MLIR boundary here; the existing
    ``compile_vortex_direct`` flow performs external ``mlir-opt`` verification
    and VOLT machine-code generation when its pinned toolchain is supplied.
    """

    if not isinstance(plan, BevFormerWholeModelPlan):
        raise ValidationError("lowering requires a BevFormerWholeModelPlan")
    plan.validate()
    destination = Path(output_directory) if output_directory is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    regions = {
        tuple(region.node_ids): region for region in plan.capability_report.regions
    }
    lowered: list[LoweredDispatchSource] = []
    for dispatch in plan.dispatch_dag.topological_dispatches:
        kind = _semantic_kind(dispatch.group)
        if dispatch.target == BEVFORMER_VORTEX_TARGET:
            region = regions.get(tuple(dispatch.node_ids))
            if region is None:
                raise ValidationError(
                    f"Vortex dispatch {dispatch.id!r} is not one canonical FFN region"
                )
            canonical = extract_bevformer_ffn_region(plan.graph, region)
            program = compile_mlir(canonical, fuse=True)
            source = program.source
            suffix = ".mlir"
            artifact_kind = "standard-tensor-linalg-mlir"
            kernel_count = None
        elif dispatch.target == BEVFORMER_CUDA_TARGET:
            program = compile_cuda(dispatch.graph)
            source = program.source
            suffix = ".cu"
            artifact_kind = "cuda-source-with-launch-metadata"
            kernel_count = len(program.launches)
        else:
            raise ValidationError(
                f"no source lowerer is registered for target {dispatch.target!r}"
            )
        payload = source.encode("utf-8")
        source_path: str | None = None
        if destination is not None:
            path = destination / f"{dispatch.id}-{kind}{suffix}"
            path.write_text(source, encoding="utf-8", newline="\n")
            source_path = str(path.resolve(strict=False))
        lowered.append(
            LoweredDispatchSource(
                dispatch.id,
                dispatch.target,
                kind,
                artifact_kind,
                sha256(payload).hexdigest(),
                len(payload),
                source_path,
                kernel_count,
            )
        )
    return tuple(lowered)


__all__ = [
    "BEVFORMER_WHOLE_MODEL_PLAN_SCHEMA_VERSION",
    "BevFormerWholeModelPlan",
    "DispatchStrategy",
    "LoweredDispatchSource",
    "build_bevformer_whole_model_plan",
    "lower_bevformer_dispatch_sources",
    "plan_bevformer_graph",
]
