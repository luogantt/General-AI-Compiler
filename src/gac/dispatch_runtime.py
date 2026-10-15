"""Global storage bindings and host orchestration for a dispatch DAG.

The source :class:`~gac.ir.Graph` remains the single authority for tensor
lifetimes.  ``build_dispatch_runtime_plan`` runs the existing whole-graph
``plan_memory`` pass once, then projects those allocations onto every dispatch
input, output, constant, and private temporary.  A value crossing a dispatch
boundary therefore keeps one stable logical allocation instead of receiving a
different plan in each extracted subgraph.

In addition to the source-graph plan used by the reference host orchestrator,
the module builds a second *global boundary* plan.  That plan excludes values
private to a dispatch, records host/device and cross-target transfers as
explicit :class:`RuntimeStep` objects, and expresses each logical buffer's
lifetime in runtime-schedule coordinates.  Dispatch-private registers, CUDA
shared memory, and Vortex LMEM remain the responsibility of target lowering.

This module does not implement device memory or claim zero-copy transfers.
``execute_dispatch_plan`` is a deterministic reference orchestrator.  Concrete
CUDA, Vortex, or other runtimes remain responsible for interpreting logical
bindings, allocating native buffers, and submitting physical transfers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any, Literal

from .errors import ValidationError
from .ir import Graph, Node
from .memory import BufferAllocation, MemoryPlan, plan_memory
from .partition import Dispatch, DispatchDAG, DispatchEdge


DISPATCH_RUNTIME_PLAN_SCHEMA_VERSION = 1
HOST_TARGET = "host"
BindingRole = Literal["input", "output", "constant", "temporary"]
RuntimeStepKind = Literal["dispatch", "transfer"]
RuntimePhase = Literal["setup", "invoke"]


def _names(value: Any, field: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValidationError(f"{field} must be an array of names") from exc
    if not allow_empty and not result:
        raise ValidationError(f"{field} must not be empty")
    if any(not isinstance(name, str) or not name for name in result):
        raise ValidationError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValidationError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class DispatchBufferBinding:
    """One dispatch-local use of a whole-graph logical allocation."""

    value: str
    role: BindingRole
    allocation: BufferAllocation

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValidationError("dispatch buffer value must be a non-empty string")
        if self.role not in {"input", "output", "constant", "temporary"}:
            raise ValidationError(f"invalid dispatch buffer role {self.role!r}")
        if not isinstance(self.allocation, BufferAllocation):
            raise ValidationError("dispatch buffer binding needs a BufferAllocation")
        if self.allocation.name != self.value:
            raise ValidationError(
                f"binding value {self.value!r} does not match allocation "
                f"{self.allocation.name!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "role": self.role,
            "storage": self.allocation.storage,
            "offset": self.allocation.offset,
            "size_bytes": self.allocation.size_bytes,
            "alignment": self.allocation.alignment,
            "alias_of": self.allocation.alias_of,
            "view_of": self.allocation.view_of,
        }


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    """One ordered dispatch or logical data transfer.

    Transfer steps retain the SSA value name and distinguish physical copies
    by ``source_target`` and ``target``.  They describe required movement but
    deliberately do not prescribe DMA engines, streams, or device APIs.
    """

    id: str
    kind: RuntimeStepKind
    phase: RuntimePhase
    target: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    dispatch_id: str | None = None
    source_target: str | None = None
    source_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("runtime step id must be a non-empty string")
        if self.kind not in {"dispatch", "transfer"}:
            raise ValidationError(f"runtime step {self.id!r} has invalid kind")
        if self.phase not in {"setup", "invoke"}:
            raise ValidationError(f"runtime step {self.id!r} has invalid phase")
        if not isinstance(self.target, str) or not self.target:
            raise ValidationError(f"runtime step {self.id!r} has invalid target")
        object.__setattr__(self, "inputs", _names(self.inputs, f"{self.id}.inputs"))
        object.__setattr__(self, "outputs", _names(self.outputs, f"{self.id}.outputs"))
        object.__setattr__(
            self, "depends_on", _names(self.depends_on, f"{self.id}.depends_on")
        )
        object.__setattr__(
            self, "source_nodes", _names(self.source_nodes, f"{self.id}.source_nodes")
        )

        if self.kind == "dispatch":
            if self.phase != "invoke":
                raise ValidationError("dispatch runtime steps must be invocation steps")
            if not isinstance(self.dispatch_id, str) or not self.dispatch_id:
                raise ValidationError("dispatch runtime steps require dispatch_id")
            if self.source_target is not None:
                raise ValidationError("dispatch runtime steps cannot have source_target")
            if not self.outputs:
                raise ValidationError("dispatch runtime steps require outputs")
        else:
            if self.dispatch_id is not None or self.source_nodes:
                raise ValidationError(
                    "transfer runtime steps cannot carry dispatch provenance"
                )
            if not isinstance(self.source_target, str) or not self.source_target:
                raise ValidationError("transfer runtime steps require source_target")
            if self.source_target == self.target:
                raise ValidationError("transfer source and destination targets must differ")
            if len(self.inputs) != 1 or self.inputs != self.outputs:
                raise ValidationError(
                    "transfer runtime steps must copy exactly one identically named value"
                )

    @property
    def value(self) -> str | None:
        return self.inputs[0] if self.kind == "transfer" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "phase": self.phase,
            "target": self.target,
            "source_target": self.source_target,
            "dispatch_id": self.dispatch_id,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "depends_on": list(self.depends_on),
            "source_nodes": list(self.source_nodes),
        }


@dataclass(frozen=True, slots=True)
class BufferLifetime:
    """One globally visible value in runtime-schedule coordinates."""

    name: str
    storage: str
    size_bytes: int
    arena_offset: int | None
    first_step: int
    last_step: int
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("buffer lifetime name must be non-empty")
        if not isinstance(self.storage, str) or not self.storage:
            raise ValidationError(f"buffer lifetime {self.name!r} has invalid storage")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ValidationError(f"buffer lifetime {self.name!r} has invalid size")
        if self.arena_offset is not None and (
            isinstance(self.arena_offset, bool)
            or not isinstance(self.arena_offset, int)
            or self.arena_offset < 0
        ):
            raise ValidationError(
                f"buffer lifetime {self.name!r} has invalid arena offset"
            )
        if (
            isinstance(self.first_step, bool)
            or not isinstance(self.first_step, int)
            or self.first_step < -1
            or isinstance(self.last_step, bool)
            or not isinstance(self.last_step, int)
            or self.last_step < self.first_step
        ):
            raise ValidationError(f"buffer lifetime {self.name!r} has invalid interval")
        object.__setattr__(
            self,
            "targets",
            _names(self.targets, f"{self.name}.targets", allow_empty=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "storage": self.storage,
            "size_bytes": self.size_bytes,
            "arena_offset": self.arena_offset,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "targets": list(self.targets),
        }


@dataclass(frozen=True, slots=True)
class DispatchStep:
    """One deterministic host schedule step and its logical buffers."""

    dispatch_id: str
    target: str
    dependencies: tuple[str, ...]
    inputs: tuple[DispatchBufferBinding, ...]
    outputs: tuple[DispatchBufferBinding, ...]
    constants: tuple[DispatchBufferBinding, ...]
    temporaries: tuple[DispatchBufferBinding, ...]
    release_after: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dispatch_id, str) or not self.dispatch_id:
            raise ValidationError("dispatch step id must be a non-empty string")
        if not isinstance(self.target, str) or not self.target:
            raise ValidationError("dispatch step target must be a non-empty string")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "constants", tuple(self.constants))
        object.__setattr__(self, "temporaries", tuple(self.temporaries))
        object.__setattr__(self, "release_after", tuple(self.release_after))
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValidationError("dispatch step dependencies must be unique")
        if len(set(self.release_after)) != len(self.release_after):
            raise ValidationError("dispatch step release_after values must be unique")
        bindings = (*self.inputs, *self.outputs, *self.constants, *self.temporaries)
        if any(not isinstance(binding, DispatchBufferBinding) for binding in bindings):
            raise ValidationError("dispatch step contains a non-buffer binding")
        names = [binding.value for binding in bindings]
        if len(set(names)) != len(names):
            raise ValidationError("a dispatch value must have exactly one buffer role")

    @property
    def bindings(self) -> tuple[DispatchBufferBinding, ...]:
        return (*self.inputs, *self.outputs, *self.constants, *self.temporaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "target": self.target,
            "dependencies": list(self.dependencies),
            "inputs": [binding.to_dict() for binding in self.inputs],
            "outputs": [binding.to_dict() for binding in self.outputs],
            "constants": [binding.to_dict() for binding in self.constants],
            "temporaries": [binding.to_dict() for binding in self.temporaries],
            "release_after": list(self.release_after),
        }


@dataclass(frozen=True, slots=True)
class DispatchRuntimePlan:
    """Validated dispatch bindings, boundary storage, and serial schedule.

    ``memory_plan`` covers the complete source graph and keeps the original
    host reference executor compatible.  ``global_memory_plan`` collapses each
    dispatch to one node, so it contains only public ABI values, constants,
    and values which leave a dispatch.  Values absent from the latter plan are
    private to a backend and may be placed in registers, shared memory, or
    Vortex LMEM.
    """

    dag: DispatchDAG
    memory_plan: MemoryPlan
    steps: tuple[DispatchStep, ...]
    global_memory_plan: MemoryPlan
    runtime_steps: tuple[RuntimeStep, ...]
    buffer_lifetimes: tuple[BufferLifetime, ...]
    host_target: str = HOST_TARGET

    def __post_init__(self) -> None:
        if not isinstance(self.dag, DispatchDAG):
            raise ValidationError("dispatch runtime plan needs a DispatchDAG")
        if not isinstance(self.memory_plan, MemoryPlan):
            raise ValidationError("dispatch runtime plan needs a MemoryPlan")
        if not isinstance(self.global_memory_plan, MemoryPlan):
            raise ValidationError("dispatch runtime plan needs a global MemoryPlan")
        if not isinstance(self.host_target, str) or not self.host_target:
            raise ValidationError("dispatch runtime host_target must be non-empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "runtime_steps", tuple(self.runtime_steps))
        object.__setattr__(self, "buffer_lifetimes", tuple(self.buffer_lifetimes))
        self.validate()

    def step(self, dispatch_id: str) -> DispatchStep:
        for step in self.steps:
            if step.dispatch_id == dispatch_id:
                return step
        raise KeyError(dispatch_id)

    @property
    def setup_steps(self) -> tuple[RuntimeStep, ...]:
        return tuple(step for step in self.runtime_steps if step.phase == "setup")

    @property
    def invocation_steps(self) -> tuple[RuntimeStep, ...]:
        return tuple(step for step in self.runtime_steps if step.phase == "invoke")

    @property
    def transfer_steps(self) -> tuple[RuntimeStep, ...]:
        return tuple(step for step in self.runtime_steps if step.kind == "transfer")

    def validate(self) -> None:
        self.dag.validate()
        self.memory_plan.validate()
        self.global_memory_plan.validate()
        if self.memory_plan.graph_name != self.dag.source_graph.name:
            raise ValidationError("memory plan and dispatch DAG refer to different graphs")
        if self.global_memory_plan.graph_name != self.dag.source_graph.name:
            raise ValidationError(
                "global memory plan and dispatch DAG refer to different graphs"
            )
        expected_ids = self.dag.schedule
        actual_ids = tuple(step.dispatch_id for step in self.steps)
        if actual_ids != expected_ids:
            raise ValidationError("runtime steps must exactly follow the dispatch DAG schedule")

        allocation_names = {
            allocation.name for allocation in self.memory_plan.allocations
        }
        graph = self.dag.source_graph
        expected_values = (
            set(graph.inputs)
            | set(graph.constants)
            | {value for node in graph.nodes for value in node.outputs}
        )
        if allocation_names != expected_values:
            raise ValidationError(
                "whole-graph memory plan does not cover exactly the source graph values"
            )

        for step in self.steps:
            dispatch = self.dag.dispatch(step.dispatch_id)
            if step.target != dispatch.target:
                raise ValidationError(
                    f"runtime step {step.dispatch_id!r} changed its dispatch target"
                )
            if step.dependencies != self.dag.dependencies(step.dispatch_id):
                raise ValidationError(
                    f"runtime step {step.dispatch_id!r} has incorrect dependencies"
                )
            expected_roles = _dispatch_value_roles(dispatch)
            actual_roles = {
                binding.value: binding.role for binding in step.bindings
            }
            if actual_roles != expected_roles:
                raise ValidationError(
                    f"runtime step {step.dispatch_id!r} has incorrect buffer roles"
                )
            for binding in step.bindings:
                if binding.allocation != self.memory_plan.allocation(binding.value):
                    raise ValidationError(
                        f"runtime step {step.dispatch_id!r} does not use the global "
                        f"allocation for {binding.value!r}"
                    )

        expected_releases = _release_values_by_dispatch(self.dag, self.memory_plan)
        for step in self.steps:
            if step.release_after != expected_releases[step.dispatch_id]:
                raise ValidationError(
                    f"runtime step {step.dispatch_id!r} has incorrect release markers"
                )

        expected_global = plan_memory(
            _global_boundary_graph(self.dag), alignment=self.memory_plan.alignment
        )
        if self.global_memory_plan != expected_global:
            raise ValidationError(
                "global memory plan is not the canonical dispatch-boundary plan"
            )
        expected_runtime = _build_runtime_steps(
            self.dag, host_target=self.host_target
        )
        if self.runtime_steps != expected_runtime:
            raise ValidationError("runtime transfer/dispatch schedule is not canonical")
        expected_lifetimes = _buffer_lifetimes(
            self.global_memory_plan,
            self.runtime_steps,
            host_target=self.host_target,
        )
        if self.buffer_lifetimes != expected_lifetimes:
            raise ValidationError("global buffer lifetimes are not canonical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DISPATCH_RUNTIME_PLAN_SCHEMA_VERSION,
            "kind": "host-dispatch-orchestration-plan",
            "dispatch_dag": self.dag.to_dict(),
            "memory_plan": self.memory_plan.to_dict(),
            "global_memory_plan": self.global_memory_plan.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "runtime_steps": [step.to_dict() for step in self.runtime_steps],
            "buffer_lifetimes": [item.to_dict() for item in self.buffer_lifetimes],
            "host_target": self.host_target,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source_graph: Graph,
    ) -> "DispatchRuntimePlan":
        """Restore a manifest by rebuilding it from the authoritative graph."""

        if not isinstance(data, Mapping):
            raise ValidationError("dispatch runtime plan must be an object")
        expected = {
            "schema_version",
            "kind",
            "dispatch_dag",
            "memory_plan",
            "global_memory_plan",
            "steps",
            "runtime_steps",
            "buffer_lifetimes",
            "host_target",
        }
        if set(data) != expected:
            raise ValidationError("dispatch runtime plan fields do not match the schema")
        if data.get("schema_version") != DISPATCH_RUNTIME_PLAN_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported dispatch runtime schema {data.get('schema_version')!r}"
            )
        if data.get("kind") != "host-dispatch-orchestration-plan":
            raise ValidationError("invalid dispatch runtime plan kind")
        try:
            dag = DispatchDAG.from_dict(
                data["dispatch_dag"], source_graph=source_graph
            )
            alignment = data["memory_plan"]["alignment"]
            plan = build_dispatch_runtime_plan(
                dag,
                alignment=alignment,
                host_target=data["host_target"],
            )
        except ValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid dispatch runtime plan") from exc
        if plan.to_dict() != dict(data):
            raise ValidationError(
                "dispatch runtime manifest does not match the authoritative graph"
            )
        return plan

    @classmethod
    def from_json(
        cls,
        source: str | bytes,
        *,
        source_graph: Graph,
    ) -> "DispatchRuntimePlan":
        try:
            data = json.loads(source)
        except (TypeError, ValueError) as exc:
            raise ValidationError("invalid dispatch runtime JSON") from exc
        return cls.from_dict(data, source_graph=source_graph)


def _dispatch_value_roles(dispatch: Dispatch) -> dict[str, BindingRole]:
    roles: dict[str, BindingRole] = {}
    for value in dispatch.graph.inputs:
        roles[value] = "input"
    for value in dispatch.graph.outputs:
        roles[value] = "output"
    for value in dispatch.graph.constants:
        roles[value] = "constant"
    for node in dispatch.graph.nodes:
        for value in node.outputs:
            roles.setdefault(value, "temporary")
    return roles


def _bindings_for_role(
    dispatch: Dispatch,
    memory_plan: MemoryPlan,
    role: BindingRole,
) -> tuple[DispatchBufferBinding, ...]:
    return tuple(
        DispatchBufferBinding(value, value_role, memory_plan.allocation(value))
        for value, value_role in _dispatch_value_roles(dispatch).items()
        if value_role == role
    )


def _release_values_by_dispatch(
    dag: DispatchDAG,
    memory_plan: MemoryPlan,
) -> dict[str, tuple[str, ...]]:
    """Map arena lifetime ends to the dispatch containing the final use node."""

    node_dispatch = {
        node_id: dispatch.id
        for dispatch in dag.dispatches
        for node_id in dispatch.node_ids
    }
    node_ids = tuple(node.id for node in dag.source_graph.nodes)
    result: dict[str, list[str]] = {dispatch_id: [] for dispatch_id in dag.schedule}
    for allocation in memory_plan.allocations:
        if allocation.storage != "arena":
            continue
        if not 0 <= allocation.last_use < len(node_ids):
            raise ValidationError(
                f"arena value {allocation.name!r} has no schedulable final use"
            )
        dispatch_id = node_dispatch[node_ids[allocation.last_use]]
        result[dispatch_id].append(allocation.name)
    return {dispatch_id: tuple(values) for dispatch_id, values in result.items()}


def _dispatch_runtime_inputs(dispatch: Dispatch) -> tuple[str, ...]:
    """Boundary inputs followed by immutable constants used by a dispatch."""

    values = list(dispatch.graph.inputs)
    values.extend(name for name in sorted(dispatch.graph.constants) if name not in values)
    return tuple(values)


def _global_boundary_graph(dag: DispatchDAG) -> Graph:
    """Collapse each dispatch to one node for global DDR liveness planning."""

    source = dag.source_graph
    nodes: list[Node] = []
    global_values = set(source.inputs) | set(source.constants)
    for dispatch in dag.topological_dispatches:
        global_values.update(dispatch.graph.outputs)
        nodes.append(
            Node(
                id=f"global.{dispatch.id}",
                op="Dispatch",
                inputs=_dispatch_runtime_inputs(dispatch),
                outputs=dispatch.graph.outputs,
                source_nodes=dispatch.node_ids,
            )
        )
    graph = Graph(
        name=source.name,
        inputs=source.inputs,
        outputs=source.outputs,
        nodes=nodes,
        constants=source.constants,
        value_specs={
            name: source.value_specs[name]
            for name in global_values
            if name in source.value_specs
        },
        metadata={"kind": "dispatch-global-boundary-memory-plan"},
    )
    graph.validate()
    return graph


def _ordered_dependencies(
    candidates: set[str], step_positions: Mapping[str, int]
) -> tuple[str, ...]:
    return tuple(sorted(candidates, key=step_positions.__getitem__))


def _build_runtime_steps(
    dag: DispatchDAG,
    *,
    host_target: str,
) -> tuple[RuntimeStep, ...]:
    """Materialize a deterministic serial schedule with explicit copies."""

    dispatches = dag.topological_dispatches
    steps: list[RuntimeStep] = []
    step_positions: dict[str, int] = {}
    dispatch_step: dict[str, str] = {}
    # value -> target -> step which made the value ready.  ``None`` denotes a
    # host value resident before setup/invocation begins.
    residency: dict[str, dict[str, str | None]] = {
        name: {host_target: None}
        for name in (*dag.source_graph.inputs, *dag.source_graph.constants)
    }
    transfer_index = 0

    def append(step: RuntimeStep) -> None:
        if step.id in step_positions:
            raise ValidationError(f"duplicate generated runtime step {step.id!r}")
        step_positions[step.id] = len(steps)
        steps.append(step)

    def transfer(
        value: str,
        source_target: str,
        target: str,
        *,
        phase: RuntimePhase,
    ) -> str:
        nonlocal transfer_index
        ready = residency.get(value, {}).get(source_target)
        dependencies = () if ready is None else (ready,)
        step_id = f"transfer.{transfer_index:04d}"
        transfer_index += 1
        append(
            RuntimeStep(
                id=step_id,
                kind="transfer",
                phase=phase,
                target=target,
                source_target=source_target,
                inputs=(value,),
                outputs=(value,),
                depends_on=dependencies,
            )
        )
        residency.setdefault(value, {})[target] = step_id
        return step_id

    # Constants are immutable and uploaded once per target.  Keeping setup
    # transfers before invocation lets a runtime cache them across calls.
    for dispatch in dispatches:
        if dispatch.target == host_target:
            continue
        for name in sorted(dispatch.graph.constants):
            placements = residency.setdefault(name, {host_target: None})
            if dispatch.target not in placements:
                transfer(name, host_target, dispatch.target, phase="setup")

    producer_target = {
        edge.value: dag.dispatch(edge.producer).target
        for edge in dag.edges
        if edge.producer is not None
    }

    for ordinal, dispatch in enumerate(dispatches):
        dependency_steps: set[str] = set()
        for name in _dispatch_runtime_inputs(dispatch):
            placements = residency.get(name)
            if placements is None:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} input {name!r} has no residency"
                )
            if dispatch.target not in placements:
                preferred = producer_target.get(name, host_target)
                if preferred not in placements:
                    preferred = sorted(placements)[0]
                transfer(name, preferred, dispatch.target, phase="invoke")
            ready = residency[name][dispatch.target]
            if ready is not None:
                dependency_steps.add(ready)

        for dependency in dag.dependencies(dispatch.id):
            try:
                dependency_steps.add(dispatch_step[dependency])
            except KeyError as exc:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} has non-prior dependency {dependency!r}"
                ) from exc
        step_id = f"dispatch.{ordinal:04d}"
        append(
            RuntimeStep(
                id=step_id,
                kind="dispatch",
                phase="invoke",
                target=dispatch.target,
                dispatch_id=dispatch.id,
                inputs=_dispatch_runtime_inputs(dispatch),
                outputs=dispatch.graph.outputs,
                depends_on=_ordered_dependencies(dependency_steps, step_positions),
                source_nodes=dispatch.node_ids,
            )
        )
        dispatch_step[dispatch.id] = step_id
        for name in dispatch.graph.outputs:
            residency[name] = {dispatch.target: step_id}

    # Public outputs return to the host ABI.  Passthrough inputs/constants are
    # already host resident and therefore need no generated step.
    for name in dag.source_graph.outputs:
        placements = residency.setdefault(name, {host_target: None})
        if host_target in placements:
            continue
        preferred = producer_target.get(name)
        if preferred is None or preferred not in placements:
            preferred = sorted(placements)[0]
        transfer(name, preferred, host_target, phase="invoke")

    return tuple(steps)


def _buffer_lifetimes(
    memory_plan: MemoryPlan,
    steps: tuple[RuntimeStep, ...],
    *,
    host_target: str,
) -> tuple[BufferLifetime, ...]:
    """Project global allocations into serial runtime-step intervals."""

    accesses: dict[str, list[int]] = {
        allocation.name: [] for allocation in memory_plan.allocations
    }
    definitions: dict[str, int] = {
        allocation.name: -1
        for allocation in memory_plan.allocations
        if allocation.is_input or allocation.is_constant
    }
    targets: dict[str, set[str]] = {
        allocation.name: (
            {host_target}
            if allocation.is_input or allocation.is_output or allocation.is_constant
            else set()
        )
        for allocation in memory_plan.allocations
    }
    for index, step in enumerate(steps):
        for name in step.inputs:
            accesses[name].append(index)
            targets[name].add(step.target)
            if step.source_target is not None:
                targets[name].add(step.source_target)
        if step.kind == "dispatch":
            for name in step.outputs:
                definitions.setdefault(name, index)
                targets[name].add(step.target)

    result: list[BufferLifetime] = []
    for allocation in memory_plan.allocations:
        start = definitions.get(allocation.name, -1)
        end = max(accesses[allocation.name], default=start)
        result.append(
            BufferLifetime(
                name=allocation.name,
                storage=allocation.storage,
                size_bytes=allocation.size_bytes,
                arena_offset=allocation.offset,
                first_step=start,
                last_step=max(start, end),
                targets=tuple(sorted(targets[allocation.name])),
            )
        )
    return tuple(result)


def build_dispatch_runtime_plan(
    dag: DispatchDAG,
    *,
    alignment: int = 256,
    host_target: str = HOST_TARGET,
) -> DispatchRuntimePlan:
    """Build dispatch bindings and a serial, device-independent runtime plan."""

    if not isinstance(dag, DispatchDAG):
        raise ValidationError("build_dispatch_runtime_plan requires a DispatchDAG")
    dag.validate()
    if not isinstance(host_target, str) or not host_target:
        raise ValidationError("host_target must be a non-empty string")
    memory_plan = plan_memory(dag.source_graph, alignment=alignment)
    global_memory_plan = plan_memory(
        _global_boundary_graph(dag), alignment=alignment
    )
    runtime_steps = _build_runtime_steps(dag, host_target=host_target)
    releases = _release_values_by_dispatch(dag, memory_plan)
    steps = tuple(
        DispatchStep(
            dispatch_id=dispatch.id,
            target=dispatch.target,
            dependencies=dag.dependencies(dispatch.id),
            inputs=_bindings_for_role(dispatch, memory_plan, "input"),
            outputs=_bindings_for_role(dispatch, memory_plan, "output"),
            constants=_bindings_for_role(dispatch, memory_plan, "constant"),
            temporaries=_bindings_for_role(dispatch, memory_plan, "temporary"),
            release_after=releases[dispatch.id],
        )
        for dispatch in dag.topological_dispatches
    )
    return DispatchRuntimePlan(
        dag=dag,
        memory_plan=memory_plan,
        steps=steps,
        global_memory_plan=global_memory_plan,
        runtime_steps=runtime_steps,
        buffer_lifetimes=_buffer_lifetimes(
            global_memory_plan, runtime_steps, host_target=host_target
        ),
        host_target=host_target,
    )


DispatchExecutor = Callable[
    [Dispatch, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]
TransferAdapter = Callable[[Any, DispatchEdge, Dispatch, Dispatch], Any]


@dataclass(frozen=True, slots=True)
class DispatchExecutionRecord:
    """Observable host orchestration for one completed dispatch."""

    dispatch_id: str
    target: str
    dependencies: tuple[str, ...]
    adapted_inputs: tuple[str, ...]
    released_values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "target": self.target,
            "dependencies": list(self.dependencies),
            "adapted_inputs": list(self.adapted_inputs),
            "released_values": list(self.released_values),
        }


@dataclass(frozen=True, slots=True)
class DispatchExecutionResult:
    """Final public outputs and deterministic execution trace."""

    outputs: dict[str, Any]
    records: tuple[DispatchExecutionRecord, ...]


def _cross_target_edges(
    plan: DispatchRuntimePlan,
) -> tuple[tuple[DispatchEdge, Dispatch, Dispatch], ...]:
    result: list[tuple[DispatchEdge, Dispatch, Dispatch]] = []
    for edge in plan.dag.edges:
        if edge.producer is None:
            continue
        producer = plan.dag.dispatch(edge.producer)
        for consumer_id in edge.consumers:
            consumer = plan.dag.dispatch(consumer_id)
            if producer.target != consumer.target:
                result.append((edge, producer, consumer))
    return tuple(result)


def execute_dispatch_plan(
    plan: DispatchRuntimePlan,
    inputs: Mapping[str, Any],
    executors: Mapping[str, DispatchExecutor],
    *,
    transfer_adapters: Mapping[tuple[str, str], TransferAdapter] | None = None,
) -> DispatchExecutionResult:
    """Execute the DAG serially through explicit host-side runtime adapters.

    ``executors`` is keyed by dispatch id, rather than target, because each
    dispatch is a separately compiled unit.  ``transfer_adapters`` is keyed by
    ``(producer_target, consumer_target)`` and is mandatory for every
    cross-target edge.  The adapter may copy a host value, enqueue a real
    device transfer, or translate an opaque device handle; this generic layer
    deliberately makes no choice on its behalf.
    """

    if not isinstance(plan, DispatchRuntimePlan):
        raise ValidationError("execute_dispatch_plan requires a DispatchRuntimePlan")
    plan.validate()
    try:
        bound_inputs = dict(inputs)
        executor_map = dict(executors)
        adapter_map = dict(transfer_adapters or {})
    except (TypeError, ValueError) as exc:
        raise ValidationError("inputs, executors, and adapters must be mappings") from exc

    expected_inputs = set(plan.dag.source_graph.inputs)
    if set(bound_inputs) != expected_inputs:
        raise ValidationError(
            f"runtime input names must be {sorted(expected_inputs)}, "
            f"got {sorted(bound_inputs)}"
        )
    expected_executors = set(plan.dag.schedule)
    if set(executor_map) != expected_executors:
        raise ValidationError(
            f"executors must cover exactly the dispatch ids; "
            f"missing={sorted(expected_executors - set(executor_map))}, "
            f"unknown={sorted(set(executor_map) - expected_executors)}"
        )
    if any(not callable(executor) for executor in executor_map.values()):
        raise ValidationError("every dispatch executor must be callable")
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or any(not isinstance(item, str) or not item for item in key)
        or not callable(adapter)
        for key, adapter in adapter_map.items()
    ):
        raise ValidationError(
            "transfer adapters must map (producer_target, consumer_target) to callables"
        )

    cross_target = _cross_target_edges(plan)
    required_pairs = {
        (producer.target, consumer.target)
        for _, producer, consumer in cross_target
    }
    missing_pairs = sorted(required_pairs - set(adapter_map))
    if missing_pairs:
        raise ValidationError(
            f"cross-target dispatch edges require transfer adapters: {missing_pairs}"
        )
    transfer_by_consumer_value = {
        (consumer.id, edge.value): (edge, producer, consumer)
        for edge, producer, consumer in cross_target
    }

    values: dict[str, Any] = dict(plan.dag.source_graph.constants)
    values.update(bound_inputs)
    completed: set[str] = set()
    records: list[DispatchExecutionRecord] = []
    for step in plan.steps:
        if not set(step.dependencies).issubset(completed):
            raise ValidationError(
                f"dispatch {step.dispatch_id!r} was scheduled before its dependencies"
            )
        dispatch = plan.dag.dispatch(step.dispatch_id)
        missing_values = [value for value in dispatch.inputs if value not in values]
        if missing_values:
            raise ValidationError(
                f"dispatch {dispatch.id!r} is missing runtime values {missing_values}"
            )

        adapted: list[str] = []
        dispatch_inputs: dict[str, Any] = {}
        for value in dispatch.inputs:
            runtime_value = values[value]
            transfer = transfer_by_consumer_value.get((dispatch.id, value))
            if transfer is not None:
                edge, producer, consumer = transfer
                runtime_value = adapter_map[(producer.target, consumer.target)](
                    runtime_value, edge, producer, consumer
                )
                adapted.append(value)
            dispatch_inputs[value] = runtime_value
        constants = {
            value: values[value] for value in dispatch.graph.constants
        }
        produced = executor_map[dispatch.id](dispatch, dispatch_inputs, constants)
        if not isinstance(produced, Mapping):
            raise ValidationError(
                f"executor for {dispatch.id!r} must return an output mapping"
            )
        produced = dict(produced)
        expected_outputs = set(dispatch.outputs)
        if set(produced) != expected_outputs:
            raise ValidationError(
                f"executor for {dispatch.id!r} returned incorrect outputs; "
                f"missing={sorted(expected_outputs - set(produced))}, "
                f"unknown={sorted(set(produced) - expected_outputs)}"
            )
        values.update(produced)
        completed.add(dispatch.id)

        for value in step.release_after:
            if value in values:
                del values[value]
        records.append(
            DispatchExecutionRecord(
                dispatch_id=dispatch.id,
                target=dispatch.target,
                dependencies=step.dependencies,
                adapted_inputs=tuple(adapted),
                # Private temporaries are owned by the dispatch executor and
                # do not appear in ``values``.  Keep their static lifetime-end
                # markers in the trace as well as those removed from the host
                # value table.
                released_values=step.release_after,
            )
        )

    missing_outputs = [
        value for value in plan.dag.source_graph.outputs if value not in values
    ]
    if missing_outputs:
        raise ValidationError(f"dispatch execution did not produce {missing_outputs}")
    return DispatchExecutionResult(
        outputs={value: values[value] for value in plan.dag.source_graph.outputs},
        records=tuple(records),
    )


__all__ = [
    "BufferLifetime",
    "DISPATCH_RUNTIME_PLAN_SCHEMA_VERSION",
    "DispatchBufferBinding",
    "DispatchExecutionRecord",
    "DispatchExecutionResult",
    "DispatchExecutor",
    "DispatchRuntimePlan",
    "DispatchStep",
    "HOST_TARGET",
    "RuntimeStep",
    "TransferAdapter",
    "build_dispatch_runtime_plan",
    "execute_dispatch_plan",
]
