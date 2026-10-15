"""Single-device Vortex implementations of the shared ten compiler stages.

The first template for each operation is a correctness-oriented software
kernel: independent output elements are distributed over SIMT threads and
reductions stay in FP32 registers. There is no implicit CPU/CUDA fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gac.core.errors import BackendError
from gac.core.ir import Graph, Node
from gac.compiler.memory import MemoryPlan, plan_memory
from gac.toolchains.mlir import verify_mlir
from gac.compiler.passes import ConstantFoldingPass, DeadCodeEliminationPass, ShapeInferencePass
from gac.compiler.pipeline import COMPILER_STAGES, PIPELINE_NAME, pipeline_manifest, run_pipeline
from gac.targets.vortex.device_mlir import (
    CTA_BLOCK_DIM_X_CSR,
    CTA_BLOCK_ID_X_CSR,
    CTA_THREAD_ID_X_CSR,
    _module,
)
from gac.targets.vortex.scalar_ir import ScalarBuilder, Value


TARGET = "vortex-v3"
LOWERING_PASSES = (
    "--convert-scf-to-cf", "--convert-math-to-llvm",
    "--convert-arith-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts",
)
SOFTWARE_OPS = frozenset({
    "MatMul", "MatMulBiasGELU", "Add", "AddReLU", "Sub", "Mul", "Div",
    "Greater", "Less", "Where", "ReLU", "GELU", "Sigmoid", "Log", "Sin",
    "Cos", "Clamp", "Identity", "Cast", "Reshape", "Transpose", "Concat",
    "Expand", "Slice", "ReduceSum", "ReduceMean", "Softmax", "LayerNorm",
    "Conv2D", "BatchNorm", "MaxPool2D", "GridSample2D", "MSDeformableAttention",
})
_BOOLEAN_LAYOUT_OPS = frozenset({"Identity", "Reshape", "Transpose", "Concat", "Expand", "Slice"})


def _validate_static_specs(graph: Graph) -> None:
    """Validate sizes with Python integers before inference or allocation.

    TensorSpec.numel uses NumPy's int64 product. That is convenient for ordinary
    shapes but can wrap malformed, enormous shapes to zero. RV32 validation must
    therefore establish the mathematical size before using memory-plan helpers.
    """
    for name, spec in graph.value_specs.items():
        if not spec.is_static or any(d <= 0 for d in spec.shape):
            raise BackendError(f"vortex-v3: {name!r} needs a positive static shape; specialize dynamic dimensions first")
        if spec.dtype not in {"float16", "float32", "int32", "bool"}:
            raise BackendError(f"vortex-v3: {name!r} has unsupported dtype {spec.dtype}")
        elements = math.prod(spec.shape)
        byte_size = elements * np.dtype(spec.dtype).itemsize
        if elements > (1 << 31) - 1 or byte_size > (1 << 31) - 1:
            raise BackendError(f"vortex-v3: {name!r} exceeds signed RV32 indexing")


def _validate_boolean_operations(graph: Graph) -> None:
    # Boolean tensors are supported as byte-valued layout data. Numeric Add
    # and MatMul require separate Boolean algebra, so they must not silently
    # pass through the integer arithmetic templates (e.g. i1 True+True wraps).
    for node in graph.nodes:
        if node.op not in _BOOLEAN_LAYOUT_OPS and any(
            graph.value_specs[name].dtype == "bool" for name in (*node.inputs, *node.outputs)
        ):
            raise BackendError(f"vortex-v3: node {node.id!r} ({node.op}) does not support bool arithmetic; "
                               "bool is supported only by layout/copy operations")


def _template(node: Node) -> str:
    if node.op in {"MatMul", "MatMulBiasGELU"}:
        return "simt-output-scalar-gemm"
    if node.op == "Conv2D":
        return "simt-direct-convolution"
    if node.op == "MSDeformableAttention":
        return "simt-multilevel-bilinear-reduction"
    if node.op == "GridSample2D":
        return "simt-grid-sample"
    if node.op in {"ReduceSum", "ReduceMean", "LayerNorm", "Softmax", "MaxPool2D"}:
        return "simt-output-serial-reduction"
    if node.op in {"Reshape", "Transpose", "Concat", "Expand", "Slice", "Identity"}:
        return "simt-layout-copy"
    return "simt-broadcast-pointwise"


def _kernel_body(graph: Graph, node: Node, kernel: str) -> str:
    from gac.targets.vortex.scalar_ops import emit_scalar_op
    from gac.targets.vortex.spatial_ops import emit_spatial_op

    pointers = tuple(f"%p{i}" for i in range(len(node.inputs)))
    output = "%out"
    args = [f"{ptr}: !llvm.ptr" for ptr in (*pointers, output)] + ["%tid: i32"]
    builder = ScalarBuilder()
    specs = tuple(graph.value_specs[name] for name in node.inputs)
    out_spec = graph.value_specs[node.outputs[0]]
    handled = emit_scalar_op(builder, node, pointers, specs, output, out_spec, Value("%tid", "i32"))
    if not handled:
        handled = emit_spatial_op(builder, node, pointers, specs, output, out_spec, Value("%tid", "i32"))
    if not handled:
        raise BackendError(f"{TARGET}: no software lowering for node {node.id!r} ({node.op})")
    builder.emit("llvm.return")
    # VOLT's divergent calling convention belongs to the kernel entry. Keep
    # scalar bodies internal and inline them before Vortex divergence lowering.
    return f'  llvm.func internal @{kernel}_compute({", ".join(args)}) attributes {{passthrough = ["alwaysinline"]}} {{\n' + "\n".join(builder.lines) + "\n  }"


def _entry(dispatch: Mapping[str, Any]) -> str:
    """The sole device-specific boundary: CTA CSRs and the RV32 pointer ABI."""
    name = dispatch["kernel"]
    builder = ScalarBuilder()
    builder.emit('%args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%blob) : (!llvm.ptr) -> !llvm.ptr')
    ptrs = []
    for index in range(len(dispatch["inputs"]) + 1):
        offset = builder.const(index * 8)
        address = builder.result(f"llvm.getelementptr %args[{offset}] : (!llvm.ptr, i32) -> !llvm.ptr, i8", "!llvm.ptr")
        raw = builder.result(f"llvm.load {address} : !llvm.ptr -> i64", "i64")
        low = builder.cast(raw, "i32")
        ptrs.append(builder.result(f"llvm.inttoptr {low} : i32 to !llvm.ptr", "!llvm.ptr"))
    ids = []
    for csr in (CTA_THREAD_ID_X_CSR, CTA_BLOCK_ID_X_CSR, CTA_BLOCK_DIM_X_CSR):
        number = builder.const(csr)
        value = builder.result(f'llvm.inline_asm "csrr $0, $1", "=r,i" {number} : (i32) -> i32', "i32")
        if csr != CTA_THREAD_ID_X_CSR:
            value = builder.result(f'llvm.call_intrinsic "llvm.riscv.vx.uniform"({value}) : (i32) -> i32', "i32")
        ids.append(value)
    tid = builder.add(builder.mul(ids[1], ids[2]), ids[0])
    valid = builder.cmp("lt", tid, dispatch["elements"])
    valid = builder.binary("and", valid, builder.cmp("ge", tid, 0))
    builder.emit(f"scf.if {valid} {{")
    builder.indent += 1
    values = ", ".join(str(v) for v in (*ptrs, tid))
    types = ", ".join(v.type for v in (*ptrs, tid))
    builder.emit(f"llvm.call @{name}_compute({values}) : ({types}) -> ()")
    builder.indent -= 1
    builder.emit("}")
    builder.emit("llvm.return")
    return (f'  llvm.func @{name}(%blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel"], '
            f'section = ".vx_entry.{name}"}} {{\n' + "\n".join(builder.lines) + "\n  }")


@dataclass
class VortexGraphProgram:
    graph: Graph
    memory_plan: MemoryPlan
    dispatches: list[dict[str, Any]]
    source: str
    native_source: str
    stages: list[dict[str, Any]]
    original_summary: dict[str, Any]
    pass_reports: list[dict[str, Any]]
    lowered_mlir: str | None = None
    host_source: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "compiler_backend": "vortex-whole-graph",
            "branch_design": f"{PIPELINE_NAME}, single Vortex device",
            "target": TARGET, "fallback_targets": [],
            "graph": self.graph.name,
            "operator_counts": dict(sorted(Counter(n.op for n in self.graph.nodes).items())),
            "dispatch_count": len(self.dispatches),
            "supported_software_ops": sorted(SOFTWARE_OPS),
            "device_abi": {"pointer_bits": 32, "argument_slot_bytes": 8,
                           "order": "input addresses, then one output address"},
            "memory": self.memory_plan.to_dict(),
            "schedule": self.dispatches, "stages": self.stages,
            "source_sha256": hashlib.sha256(self.source.encode()).hexdigest(),
            "mlir_verified_and_lowered": self.lowered_mlir is not None,
            "optimization": {"template": "scalar-software-correctness-baseline",
                             "lmem_bytes": 0, "dxa": False, "tcu": False,
                             "accumulation": "FP32", "buffer_reuse": "global arena lifetimes"},
            "validation": {"full_checkpoint_on_device": False,
                           "timing_note": "Simulator host wall time is not silicon inference latency."},
        }


@dataclass(frozen=True, slots=True)
class KernelPartition:
    """A legal graph region before template, tiling or schedule choices."""

    node: Node
    kernel: str
    elements: int
    depends_on: tuple[int, ...]
    logical_io_bytes: int


@dataclass
class VortexCompilationState:
    graph: Graph
    optimize: bool = True
    mlir_opt: str | None = None
    timeout: float = 180.0
    original_summary: dict[str, Any] = field(default_factory=dict)
    pass_reports: list[dict[str, Any]] = field(default_factory=list)
    partitions: tuple[KernelPartition, ...] = ()
    templates: dict[str, str] = field(default_factory=dict)
    tiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    memory_plan: MemoryPlan | None = None
    dispatches: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
    native_source: str = ""
    lowered_mlir: str | None = None
    host_source: str = ""


def stage_model_import(state: VortexCompilationState) -> VortexCompilationState:
    """Accept the frontend's Graph/weights ABI without mutating its owner."""
    state.graph.validate()
    _validate_static_specs(state.graph)
    state.original_summary = {
        "name": state.graph.name, "nodes": len(state.graph.nodes),
        "operator_counts": dict(sorted(Counter(n.op for n in state.graph.nodes).items())),
    }
    state.graph = state.graph.clone()
    return state


def stage_ir_conversion(state: VortexCompilationState) -> VortexCompilationState:
    """Establish typed tensor IR before graph rewrites consume any dimensions."""
    unsupported = [f"{n.id} ({n.op})" for n in state.graph.nodes if n.op not in SOFTWARE_OPS]
    if unsupported:
        raise BackendError("vortex-v3: unsupported software operators: " + ", ".join(unsupported))
    state.pass_reports.append(ShapeInferencePass().run(state.graph).to_dict())
    _validate_static_specs(state.graph)
    _validate_boolean_operations(state.graph)
    return state


def stage_graph_optimization(state: VortexCompilationState) -> VortexCompilationState:
    if state.optimize:
        for transform in (ConstantFoldingPass(), DeadCodeEliminationPass(), ShapeInferencePass()):
            state.pass_reports.append(transform.run(state.graph).to_dict())
    return state


def stage_legalization(state: VortexCompilationState) -> VortexCompilationState:
    state.graph.validate()
    _validate_static_specs(state.graph)
    if not state.graph.nodes:
        raise BackendError("vortex-v3: graph has no runtime operations after optimization")
    for node in state.graph.nodes:
        if len(node.outputs) != 1:
            raise BackendError(f"vortex-v3: node {node.id!r} must have one output")
    return state


def stage_subgraph_partition(state: VortexCompilationState) -> VortexCompilationState:
    """The software baseline uses one executable region per graph node."""
    graph = state.graph
    producers = graph.producer_map()
    node_indices = {node.id: index for index, node in enumerate(graph.nodes)}
    partitions = []
    for index, node in enumerate(graph.nodes):
        dependencies = sorted({node_indices[producers[name].id]
                               for name in node.inputs if name in producers})
        output = node.outputs[0]
        partitions.append(KernelPartition(
            node, f"gac_op_{index:04d}", graph.value_specs[output].numel,
            tuple(dependencies), sum(graph.value_specs[name].nbytes for name in (*node.inputs, output)),
        ))
    state.partitions = tuple(partitions)
    return state


def stage_template_selection(state: VortexCompilationState) -> VortexCompilationState:
    state.templates = {part.kernel: _template(part.node) for part in state.partitions}
    return state


def stage_tiling(state: VortexCompilationState) -> VortexCompilationState:
    # Each SIMT thread owns one output element. Reductions remain serial until
    # a later target template introduces LMEM or matrix-unit tiles.
    state.tiles = {
        part.kernel: {"output_elements_per_thread": 1, "reduction": "serial",
                      "lmem_bytes": 0, "buffer_stages": 1}
        for part in state.partitions
    }
    return state


def stage_schedule(state: VortexCompilationState) -> VortexCompilationState:
    memory = plan_memory(state.graph)
    if memory.arena_size + memory.resident_bytes + memory.external_input_bytes + memory.external_output_bytes >= (1 << 31):
        raise BackendError("vortex-v3: aggregate memory exceeds signed RV32 addressable budget")
    state.memory_plan = memory
    # Every dispatch waits for its predecessor, including independent nodes:
    # the static arena reuses bytes according to this exact topological order.
    state.dispatches = [
        {"kernel": part.kernel, "node_id": part.node.id, "op": part.node.op,
         "inputs": list(part.node.inputs), "output": part.node.outputs[0], "elements": part.elements,
         "depends_on": list(part.depends_on), "wait_for_dispatch": index - 1 if index else None,
         "target": TARGET, "template": state.templates[part.kernel], "tiling": state.tiles[part.kernel],
         "logical_io_bytes": part.logical_io_bytes}
        for index, part in enumerate(state.partitions)
    ]
    return state


def _native_entry(partition: KernelPartition) -> str:
    kernel, node = partition.kernel, partition.node
    builder = ScalarBuilder()
    pointers = [f"%p{i}" for i in range(len(node.inputs) + 1)]
    def body(tid, _args):
        builder.emit(f"llvm.call @{kernel}_compute({', '.join([*pointers, str(tid)])}) : ({', '.join(['!llvm.ptr'] * len(pointers) + ['i32'])}) -> ()")
        return ()
    builder.loop(partition.elements, (), body)
    builder.emit("llvm.return")
    return f"  llvm.func @{kernel}_native({', '.join(p + ': !llvm.ptr' for p in pointers)}) {{\n" + "\n".join(builder.lines) + "\n  }"


def stage_codegen(state: VortexCompilationState) -> VortexCompilationState:
    computes = [_kernel_body(state.graph, part.node, part.kernel) for part in state.partitions]
    entries = [_entry(dispatch) for dispatch in state.dispatches]
    native_entries = [_native_entry(part) for part in state.partitions]
    state.source = _module(tuple(d["kernel"] for d in state.dispatches), "\n\n".join(computes + entries))
    state.native_source = "module {\n" + "\n\n".join(computes + native_entries) + "\n}\n"
    if state.mlir_opt is not None:
        state.lowered_mlir = verify_mlir(state.source, command=(state.mlir_opt, *LOWERING_PASSES), timeout=state.timeout)
    return state


def stage_runtime(state: VortexCompilationState) -> VortexCompilationState:
    from gac.targets.vortex.host_runtime import emit_graph_host
    if state.memory_plan is None:
        raise BackendError("vortex-v3: runtime stage requires a scheduled memory plan")
    state.host_source = emit_graph_host(state.graph, state.memory_plan, state.dispatches)
    return state


def compile_vortex_graph(graph: Graph, *, optimize: bool = True,
                         mlir_opt: str | None = None, timeout: float = 180.0) -> VortexGraphProgram:
    """Run the ten layers with every executable region assigned to Vortex."""
    state = run_pipeline(VortexCompilationState(graph, optimize, mlir_opt, timeout), {
        "model-import": stage_model_import,
        "ir-conversion": stage_ir_conversion,
        "graph-optimization": stage_graph_optimization,
        "legalization": stage_legalization,
        "subgraph-partition": stage_subgraph_partition,
        "template-selection": stage_template_selection,
        "tiling": stage_tiling,
        "schedule": stage_schedule,
        "codegen": stage_codegen,
        "runtime": stage_runtime,
    })
    return VortexGraphProgram(
        state.graph, state.memory_plan, state.dispatches, state.source, state.native_source,
        pipeline_manifest(), state.original_summary, state.pass_reports, state.lowered_mlir, state.host_source,
    )


def write_vortex_graph_artifact(output: str | Path, program: VortexGraphProgram,
                                *, inputs: Mapping[str, np.ndarray] | None = None,
                                golden: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Emit the ten stage contracts, executable MLIR and one Vortex2 host runner."""
    from gac.targets.vortex.host_runtime import emit_graph_host, emit_graph_makefile, emit_graph_verifier
    from gac.targets.vortex.checks import emit_output_verifier

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise BackendError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    graph = program.graph
    if inputs is not None and set(inputs) != set(graph.inputs):
        raise BackendError("artifact inputs must match the graph input names exactly")
    def write_json(name, data):
        (root / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    def tensors(role, values, specs):
        result = []
        for index, (name, spec) in enumerate(specs.items()):
            filename = f"{role}/{role[:-1]}_{index:04d}.bin"
            result.append({"name": name, "file": filename, **spec.to_dict(), "size_bytes": spec.nbytes})
            if values is not None:
                value = np.asarray(values[name])
                if value.shape != spec.shape or value.dtype.name != spec.dtype:
                    raise BackendError(f"{role} tensor {name!r} differs from its ABI")
                path = root / filename
                path.parent.mkdir(exist_ok=True)
                np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<'), copy=False)).tofile(path)
        return result
    input_files = tensors("inputs", inputs, graph.inputs)
    from gac.core.ir import TensorSpec
    constant_specs = {name: TensorSpec(array.shape, array.dtype) for name, array in graph.constants.items()}
    constant_files = tensors("constants", graph.constants, constant_specs)
    output_specs = {name: graph.value_specs[name] for name in graph.outputs}
    output_files = tensors("outputs", None, output_specs)
    if golden is not None:
        if set(golden) != set(graph.outputs):
            raise BackendError("golden outputs must match graph outputs exactly")
        tensors("goldens", golden, output_specs)
    stage_outputs = {
        "model-import": {**program.original_summary, "inputs": input_files, "constants": constant_files},
        "ir-conversion": graph.to_dict(include_constant_data=False),
        "graph-optimization": program.pass_reports,
        "legalization": {"target": TARGET, "supported": sorted(SOFTWARE_OPS), "unsupported_nodes": [], "fallback": False},
        "subgraph-partition": {"target": TARGET, "dispatches": program.dispatches},
        "template-selection": [{"kernel": d["kernel"], "op": d["op"], "template": d["template"]} for d in program.dispatches],
        "tiling": [{"kernel": d["kernel"], **d["tiling"]} for d in program.dispatches],
        "schedule": {"memory": program.memory_plan.to_dict(), "dispatches": program.dispatches},
        "codegen": program.source,
        "runtime": {"inputs": input_files, "constants": constant_files, "outputs": output_files,
                    "device": TARGET, "host_fallback": False,
                    "state": "caller supplies prev_bev/has_prev_bev; read next_bev output"},
    }
    for stage in COMPILER_STAGES:
        payload = stage_outputs[stage.name]
        if stage.name == "codegen":
            (root / stage.artifact).write_text(payload, encoding="utf-8", newline="\n")
        else:
            write_json(stage.artifact, payload)
    if program.lowered_mlir:
        (root / "kernel.lowered.mlir").write_text(program.lowered_mlir, encoding="utf-8", newline="\n")
    (root / "main.cpp").write_text(program.host_source or emit_graph_host(graph, program.memory_plan, program.dispatches), encoding="utf-8", newline="\n")
    (root / "Makefile").write_text(emit_graph_makefile(), encoding="utf-8", newline="\n")
    (root / "verify_outputs.py").write_text(emit_output_verifier(), encoding="utf-8", newline="\n")
    (root / "verify_artifact.py").write_text(emit_graph_verifier(), encoding="utf-8", newline="\n")
    report = program.describe()
    report["tensor_files"] = {"inputs": input_files, "constants": constant_files, "outputs": output_files}
    report["files"] = {str(p.relative_to(root)).replace('\\', '/'): {"size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(root.rglob('*')) if p.is_file()}
    write_json("manifest.json", report)
    return report
