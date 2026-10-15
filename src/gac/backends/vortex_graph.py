"""Single-device, whole-graph Vortex lowering, following the V6 ten stages.

The first template for each operation is a correctness-oriented software
kernel: independent output elements are distributed over SIMT threads and
reductions stay in FP32 registers. There is no implicit CPU/CUDA fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..errors import BackendError
from ..ir import Graph, Node
from ..memory import MemoryPlan, plan_memory
from ..mlir_toolchain import verify_mlir
from ..passes import ConstantFoldingPass, DeadCodeEliminationPass, ShapeInferencePass
from .vortex_device_mlir import (
    CTA_BLOCK_DIM_X_CSR, CTA_BLOCK_ID_X_CSR, CTA_THREAD_ID_X_CSR,
    _module,
)
from .vortex_scalar_ir import ScalarBuilder, Value


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
    from .vortex_scalar_ops import emit_scalar_op
    from .vortex_spatial_ops import emit_spatial_op

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

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "compiler_backend": "vortex-whole-graph",
            "branch_design": "V6 ten-stage single-device compiler",
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


def compile_vortex_graph(graph: Graph, *, optimize: bool = True,
                         mlir_opt: str | None = None, timeout: float = 180.0) -> VortexGraphProgram:
    """Compile every node onto one Vortex device; unsupported graphs fail closed."""
    graph.validate()
    _validate_static_specs(graph)
    original = {"name": graph.name, "nodes": len(graph.nodes),
                "operator_counts": dict(sorted(Counter(n.op for n in graph.nodes).items()))}
    graph = graph.clone()
    unsupported = [f"{n.id} ({n.op})" for n in graph.nodes if n.op not in SOFTWARE_OPS]
    if unsupported:
        raise BackendError("vortex-v3: unsupported software operators: " + ", ".join(unsupported))
    reports = [ShapeInferencePass().run(graph).to_dict()]
    _validate_static_specs(graph)
    _validate_boolean_operations(graph)
    if optimize:
        for transform in (ConstantFoldingPass(), DeadCodeEliminationPass(), ShapeInferencePass()):
            reports.append(transform.run(graph).to_dict())
    graph.validate()
    _validate_static_specs(graph)
    if not graph.nodes:
        raise BackendError("vortex-v3: graph has no runtime operations after optimization")
    memory = plan_memory(graph)
    if memory.arena_size + memory.resident_bytes + memory.external_input_bytes + memory.external_output_bytes >= (1 << 31):
        raise BackendError("vortex-v3: aggregate memory exceeds signed RV32 addressable budget")
    dispatches = []
    computes = []
    entries = []
    native_entries = []
    producers = graph.producer_map()
    index_by_node = {node.id: i for i, node in enumerate(graph.nodes)}
    for index, node in enumerate(graph.nodes):
        if len(node.outputs) != 1:
            raise BackendError(f"vortex-v3: node {node.id!r} must have one output")
        kernel = f"gac_op_{index:04d}"
        output = node.outputs[0]
        elements = graph.value_specs[output].numel
        dependencies = sorted({index_by_node[producers[name].id] for name in node.inputs if name in producers})
        dispatch = {"kernel": kernel, "node_id": node.id, "op": node.op,
                    "inputs": list(node.inputs), "output": output, "elements": elements,
                    "depends_on": dependencies,
                    "wait_for_dispatch": index - 1 if index else None,
                    "target": TARGET, "template": _template(node),
                    "tiling": {"output_elements_per_thread": 1, "reduction": "serial",
                               "lmem_bytes": 0, "buffer_stages": 1},
                    "logical_io_bytes": sum(graph.value_specs[n].nbytes for n in (*node.inputs, output))}
        dispatches.append(dispatch)
        computes.append(_kernel_body(graph, node, kernel))
        entries.append(_entry(dispatch))
        b = ScalarBuilder()
        pointers = [f"%p{i}" for i in range(len(node.inputs) + 1)]
        def body(tid, _args, name=kernel, ptrs=pointers, builder=b):
            builder.emit(f"llvm.call @{name}_compute({', '.join([*ptrs, str(tid)])}) : ({', '.join(['!llvm.ptr'] * len(ptrs) + ['i32'])}) -> ()")
            return ()
        b.loop(elements, (), body)
        b.emit("llvm.return")
        native_entries.append(f"  llvm.func @{kernel}_native({', '.join(p + ': !llvm.ptr' for p in pointers)}) {{\n" + "\n".join(b.lines) + "\n  }")
    source = _module(tuple(d["kernel"] for d in dispatches), "\n\n".join(computes + entries))
    native = "module {\n" + "\n\n".join(computes + native_entries) + "\n}\n"
    stage_defs = (
        ("model-import", "Model / weights / ABI", "01-import.json"),
        ("ir-conversion", "Typed Graph IR, axes and shapes", "02-graph.json"),
        ("graph-optimization", "Constant folding and dead-code elimination", "03-passes.json"),
        ("legalization", "Static shapes and software operator support", "04-capabilities.json"),
        ("subgraph-partition", "One software kernel per node, all on Vortex", "05-dispatch-dag.json"),
        ("template-selection", "SIMT scalar / reduction / spatial templates", "06-templates.json"),
        ("tiling", "One output per thread; serial reduction baseline", "07-tiling.json"),
        ("schedule", "Topological events and global buffer lifetime reuse", "08-schedule.json"),
        ("codegen", "SCF/Arith/Math/LLVM -> LLVM dialect -> LLVM IR -> VOLT", "kernel.mlir"),
        ("runtime", "Vortex2 resident weights, arena and dispatch execution", "10-runtime.json"),
    )
    stages = [{"stage": i, "name": name, "contract": contract, "artifact": artifact}
              for i, (name, contract, artifact) in enumerate(stage_defs, 1)]
    program = VortexGraphProgram(graph, memory, dispatches, source, native, stages, original, reports)
    if mlir_opt is not None:
        program.lowered_mlir = verify_mlir(source, command=(mlir_opt, *LOWERING_PASSES), timeout=timeout)
    return program


def write_vortex_graph_artifact(output: str | Path, program: VortexGraphProgram,
                                *, inputs: Mapping[str, np.ndarray] | None = None,
                                golden: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Emit the ten stage contracts, executable MLIR and one Vortex2 host runner."""
    from .vortex_graph_runtime import emit_graph_host, emit_graph_makefile, emit_graph_verifier
    from .vortex_graph_checks import emit_output_verifier

    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise BackendError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    graph = program.graph
    if inputs is not None and set(inputs) != set(graph.inputs):
        raise BackendError("artifact inputs must match the graph input names exactly")
    def write_json(name, data):
        (root / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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
    from ..ir import TensorSpec
    constant_specs = {name: TensorSpec(array.shape, array.dtype) for name, array in graph.constants.items()}
    constant_files = tensors("constants", graph.constants, constant_specs)
    output_specs = {name: graph.value_specs[name] for name in graph.outputs}
    output_files = tensors("outputs", None, output_specs)
    if golden is not None:
        if set(golden) != set(graph.outputs):
            raise BackendError("golden outputs must match graph outputs exactly")
        tensors("goldens", golden, output_specs)
    write_json("01-import.json", {**program.original_summary, "inputs": input_files, "constants": constant_files})
    write_json("02-graph.json", graph.to_dict(include_constant_data=False))
    write_json("03-passes.json", program.pass_reports)
    write_json("04-capabilities.json", {"target": TARGET, "supported": sorted(SOFTWARE_OPS), "unsupported_nodes": [], "fallback": False})
    write_json("05-dispatch-dag.json", {"target": TARGET, "dispatches": program.dispatches})
    write_json("06-templates.json", [{"kernel": d["kernel"], "op": d["op"], "template": d["template"]} for d in program.dispatches])
    write_json("07-tiling.json", [{"kernel": d["kernel"], **d["tiling"]} for d in program.dispatches])
    write_json("08-schedule.json", {"memory": program.memory_plan.to_dict(), "dispatches": program.dispatches})
    write_json("10-runtime.json", {"inputs": input_files, "constants": constant_files, "outputs": output_files,
                                   "device": TARGET, "host_fallback": False, "state": "caller supplies prev_bev/has_prev_bev; read next_bev output"})
    (root / "kernel.mlir").write_text(program.source, encoding="utf-8")
    if program.lowered_mlir:
        (root / "kernel.lowered.mlir").write_text(program.lowered_mlir, encoding="utf-8")
    (root / "main.cpp").write_text(emit_graph_host(graph, program.memory_plan, program.dispatches), encoding="utf-8")
    (root / "Makefile").write_text(emit_graph_makefile(), encoding="utf-8")
    (root / "verify_outputs.py").write_text(emit_output_verifier(), encoding="utf-8")
    (root / "verify_artifact.py").write_text(emit_graph_verifier(), encoding="utf-8")
    report = program.describe()
    report["tensor_files"] = {"inputs": input_files, "constants": constant_files, "outputs": output_files}
    report["files"] = {str(p.relative_to(root)).replace('\\', '/'): {"size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(root.rglob('*')) if p.is_file()}
    write_json("manifest.json", report)
    return report
