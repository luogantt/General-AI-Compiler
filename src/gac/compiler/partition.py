"""Deterministic graph partitioning and dispatch-DAG construction.

This module deliberately separates *where* a node executes from backend
compilation.  Callers may provide one target placement for every node or an
ordered list of backend capabilities.  Consecutive nodes with the same target
and semantic group form one dispatch.  Each dispatch owns a normal ``Graph``
whose value names are the original, global SSA names, so existing backends can
compile extracted regions without a second value-identity system.

Only values crossing a dispatch boundary (plus public graph inputs/outputs)
become :class:`DispatchEdge` objects.  Constants stay in the extracted
subgraph and are not mistaken for runtime transfers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import copy
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from gac.core.errors import BackendError, ValidationError
from gac.core.ir import Graph, Node, TensorSpec


DISPATCH_DAG_SCHEMA_VERSION = 1

BackendCheck = Callable[[Node, Graph], str | None]


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        # ``equal_nan`` is undefined for string and a few extension dtypes,
        # while ordinary element equality remains exact for them.
        return bool(np.array_equal(left, right))


def _ordered_nodes(graph: Graph, node_ids: Iterable[str]) -> tuple[Node, ...]:
    try:
        requested = tuple(node_ids)
    except TypeError as exc:
        raise ValidationError("subgraph node_ids must be iterable") from exc
    if not requested:
        raise ValidationError("an extracted subgraph must contain at least one node")
    if any(not isinstance(node_id, str) or not node_id for node_id in requested):
        raise ValidationError("subgraph node_ids must be non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValidationError("subgraph node_ids must not contain duplicates")
    known = {node.id for node in graph.nodes}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValidationError(f"subgraph refers to unknown nodes: {unknown}")
    selected = set(requested)
    return tuple(node for node in graph.nodes if node.id in selected)


def _subgraph_boundary(
    graph: Graph,
    nodes: tuple[Node, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Return inputs, constants, outputs, and all local values for ``nodes``."""

    selected = {node.id for node in nodes}
    producers = graph.producer_map()
    consumers = graph.consumers_map()

    inputs: list[str] = []
    constants: list[str] = []
    seen_inputs: set[str] = set()
    seen_constants: set[str] = set()
    for node in nodes:
        for value in node.inputs:
            if value in graph.constants:
                if value not in seen_constants:
                    constants.append(value)
                    seen_constants.add(value)
                continue
            producer = producers.get(value)
            if producer is None or producer.id not in selected:
                if value not in seen_inputs:
                    inputs.append(value)
                    seen_inputs.add(value)

    outputs: list[str] = []
    for node in nodes:
        for value in node.outputs:
            value_consumers = consumers.get(value, ())
            if (
                value in graph.outputs
                or not value_consumers
                or any(consumer.id not in selected for consumer in value_consumers)
            ):
                outputs.append(value)

    local_values = frozenset(
        (*inputs, *constants, *(value for node in nodes for value in node.outputs))
    )
    return tuple(inputs), tuple(constants), tuple(outputs), local_values


def extract_subgraph(
    graph: Graph,
    node_ids: Iterable[str],
    *,
    name: str | None = None,
) -> Graph:
    """Extract nodes into a valid graph while preserving global value names.

    A non-constant input whose producer is outside the selected region becomes
    a subgraph input.  A produced value becomes a subgraph output when it is a
    public output, has an outside consumer, or is a dead sink.  Constants used
    by the region are copied into the subgraph and therefore never become
    transfer edges.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("extract_subgraph requires a Graph")
    graph.validate()
    nodes = _ordered_nodes(graph, node_ids)
    inputs, constant_names, outputs, local_values = _subgraph_boundary(graph, nodes)

    input_specs: dict[str, TensorSpec] = {}
    for value in inputs:
        spec = graph.value_specs.get(value)
        if spec is None:
            raise ValidationError(
                f"subgraph boundary value {value!r} has no TensorSpec; run shape inference first"
            )
        input_specs[value] = spec

    selected_nodes = [
        Node(
            id=node.id,
            op=node.op,
            inputs=node.inputs,
            outputs=node.outputs,
            attrs=copy.deepcopy(node.attrs),
            source_nodes=node.source_nodes,
        )
        for node in nodes
    ]
    constants = {
        constant_name: graph.constants[constant_name].copy()
        for constant_name in constant_names
    }
    value_specs = {
        value: spec
        for value, spec in graph.value_specs.items()
        if value in local_values
    }
    metadata = copy.deepcopy(graph.metadata)
    metadata["extracted_partition"] = {
        "source_graph": graph.name,
        "node_ids": [node.id for node in nodes],
    }
    subgraph = Graph(
        name=name or f"{graph.name}.subgraph",
        inputs=input_specs,
        outputs=outputs,
        nodes=selected_nodes,
        constants=constants,
        value_specs=value_specs,
        metadata=metadata,
    )
    subgraph.validate()
    return subgraph


@dataclass(frozen=True, slots=True)
class NodePlacement:
    """Backend and semantic group selected for one source node.

    ``group`` controls dispatch fusion independently of ``target``.  For
    example, CNN and Attention nodes may both execute on ``cuda-sm86`` while
    remaining separate compilation units.  A missing group means that target
    identity alone controls fusion, preserving the original simple mapping
    behaviour.
    """

    target: str
    group: str | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.target, "placement target")
        if self.group is not None:
            _non_empty_text(self.group, "placement group")

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "group": self.group}


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """One ordered node-placement rule.

    ``supported_ops`` is the cheap registration gate.  ``check`` may then
    inspect shapes, attributes, provenance, or the surrounding graph and
    returns ``None`` when the node is accepted or a concrete rejection reason.
    Capabilities are tried in caller-provided order, so a specialised backend
    can precede a general fallback backend deterministically.
    """

    target: str
    supported_ops: frozenset[str] | None = None
    check: BackendCheck | None = None
    group: str | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.target, "backend capability target")
        if self.group is not None:
            _non_empty_text(self.group, "backend capability group")
        if self.supported_ops is not None:
            try:
                operations = frozenset(self.supported_ops)
            except TypeError as exc:
                raise ValidationError("supported_ops must be an iterable of strings") from exc
            if any(not isinstance(op, str) or not op for op in operations):
                raise ValidationError("supported_ops must contain non-empty strings")
            object.__setattr__(self, "supported_ops", operations)
        if self.check is not None and not callable(self.check):
            raise ValidationError("backend capability check must be callable")

    def rejection_reason(self, node: Node, graph: Graph) -> str | None:
        """Return ``None`` when this capability accepts ``node``."""

        if self.supported_ops is not None and node.op not in self.supported_ops:
            return "operation is not registered"
        if self.check is None:
            return None
        reason = self.check(node, graph)
        if reason is None:
            return None
        if not isinstance(reason, str) or not reason:
            raise ValidationError(
                f"capability check for {self.target!r} must return None or a "
                "non-empty rejection reason"
            )
        return reason


def assign_backends(
    graph: Graph,
    capabilities: Iterable[BackendCapability],
    *,
    fallback_target: str | None = None,
    fallback_group: str | None = None,
) -> dict[str, NodePlacement]:
    """Assign every node to the first accepting backend capability.

    The returned mapping follows source topological order.  If no capability
    accepts a node, an explicit fallback may be supplied; otherwise the
    function raises :class:`BackendError` with all rejection reasons.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("assign_backends requires a Graph")
    graph.validate()
    try:
        ordered = tuple(capabilities)
    except TypeError as exc:
        raise ValidationError("capabilities must be iterable") from exc
    if any(not isinstance(item, BackendCapability) for item in ordered):
        raise ValidationError("capabilities must contain BackendCapability values")
    fallback = None
    if fallback_target is not None:
        fallback = NodePlacement(fallback_target, fallback_group)
    elif fallback_group is not None:
        raise ValidationError("fallback_group requires fallback_target")

    placements: dict[str, NodePlacement] = {}
    for node in graph.nodes:
        rejected: list[str] = []
        for capability in ordered:
            reason = capability.rejection_reason(node, graph)
            if reason is None:
                placements[node.id] = NodePlacement(
                    capability.target, capability.group
                )
                break
            rejected.append(f"{capability.target}: {reason}")
        else:
            if fallback is not None:
                placements[node.id] = fallback
                continue
            details = "; ".join(rejected) if rejected else "no capabilities configured"
            raise BackendError(
                f"no backend can execute node {node.id!r} ({node.op}): {details}"
            )
    return placements


@dataclass(frozen=True, slots=True)
class RegionAssignment:
    """One consecutive source-graph region assigned to a target."""

    target: str
    node_ids: tuple[str, ...]
    group: str | None = None

    def __post_init__(self) -> None:
        _non_empty_text(self.target, "region target")
        node_ids = tuple(self.node_ids)
        if not node_ids or any(
            not isinstance(node_id, str) or not node_id for node_id in node_ids
        ):
            raise ValidationError("region node_ids must contain non-empty strings")
        if len(set(node_ids)) != len(node_ids):
            raise ValidationError("region node_ids must be unique")
        if self.group is not None:
            _non_empty_text(self.group, "region group")
        object.__setattr__(self, "node_ids", node_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "group": self.group,
            "node_ids": list(self.node_ids),
        }


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One backend compilation unit extracted from the source graph."""

    id: str
    target: str
    node_ids: tuple[str, ...]
    graph: Graph

    def __post_init__(self) -> None:
        _non_empty_text(self.id, "dispatch id")
        _non_empty_text(self.target, "dispatch target")
        node_ids = tuple(self.node_ids)
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValidationError("dispatch node_ids must be non-empty and unique")
        if not isinstance(self.graph, Graph):
            raise ValidationError("dispatch graph must be a Graph")
        object.__setattr__(self, "node_ids", node_ids)

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.graph.inputs)

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.graph.outputs

    @property
    def group(self) -> str | None:
        partition = self.graph.metadata.get("extracted_partition", {})
        if not isinstance(partition, Mapping):
            return None
        value = partition.get("group")
        return value if isinstance(value, str) and value else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "group": self.group,
            "node_ids": list(self.node_ids),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "constants": list(self.graph.constants),
        }


@dataclass(frozen=True, slots=True)
class DispatchEdge:
    """One global SSA value visible at a dispatch or public ABI boundary."""

    value: str
    spec: TensorSpec
    producer: str | None
    consumers: tuple[str, ...]
    is_graph_input: bool = False
    is_graph_output: bool = False
    is_constant: bool = False

    def __post_init__(self) -> None:
        _non_empty_text(self.value, "dispatch edge value")
        if self.producer is not None:
            _non_empty_text(self.producer, "dispatch edge producer")
        consumers = tuple(self.consumers)
        if any(not isinstance(item, str) or not item for item in consumers):
            raise ValidationError("dispatch edge consumers must be non-empty strings")
        if len(set(consumers)) != len(consumers):
            raise ValidationError("dispatch edge consumers must be unique")
        if not isinstance(self.spec, TensorSpec):
            raise ValidationError("dispatch edge spec must be a TensorSpec")
        if any(
            not isinstance(flag, bool)
            for flag in (self.is_graph_input, self.is_graph_output, self.is_constant)
        ):
            raise ValidationError("dispatch edge role flags must be booleans")
        if self.is_graph_input and self.is_constant:
            raise ValidationError("a dispatch edge cannot be both input and constant")
        object.__setattr__(self, "consumers", consumers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "spec": self.spec.to_dict(),
            "producer": self.producer,
            "consumers": list(self.consumers),
            "is_graph_input": self.is_graph_input,
            "is_graph_output": self.is_graph_output,
            "is_constant": self.is_constant,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DispatchEdge":
        try:
            return cls(
                value=data["value"],
                spec=TensorSpec.from_dict(data["spec"]),
                producer=data.get("producer"),
                consumers=tuple(data.get("consumers", ())),
                is_graph_input=data.get("is_graph_input", False),
                is_graph_output=data.get("is_graph_output", False),
                is_constant=data.get("is_constant", False),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"invalid dispatch edge: {data!r}") from exc


def _dispatch_edges(
    graph: Graph,
    dispatches: tuple[Dispatch, ...],
) -> tuple[DispatchEdge, ...]:
    dispatch_by_node = {
        node_id: dispatch.id
        for dispatch in dispatches
        for node_id in dispatch.node_ids
    }
    dispatch_order = {dispatch.id: index for index, dispatch in enumerate(dispatches)}
    consumers = graph.consumers_map()

    def consumer_dispatches(value: str, producer: str | None) -> tuple[str, ...]:
        result: set[str] = set()
        for node in consumers.get(value, ()):
            dispatch_id = dispatch_by_node[node.id]
            if dispatch_id != producer:
                result.add(dispatch_id)
        return tuple(sorted(result, key=dispatch_order.__getitem__))

    edges: list[DispatchEdge] = []
    for value in graph.inputs:
        value_consumers = consumer_dispatches(value, None)
        if value_consumers or value in graph.outputs:
            edges.append(
                DispatchEdge(
                    value=value,
                    spec=graph.value_specs[value],
                    producer=None,
                    consumers=value_consumers,
                    is_graph_input=True,
                    is_graph_output=value in graph.outputs,
                )
            )

    # A constant is normally embedded in each dispatch graph, not transferred.
    # A public constant output still needs an ABI edge so it cannot disappear.
    for value in graph.constants:
        if value in graph.outputs:
            edges.append(
                DispatchEdge(
                    value=value,
                    spec=graph.value_specs[value],
                    producer=None,
                    consumers=consumer_dispatches(value, None),
                    is_graph_output=True,
                    is_constant=True,
                )
            )

    for node in graph.nodes:
        producer = dispatch_by_node[node.id]
        for value in node.outputs:
            value_consumers = consumer_dispatches(value, producer)
            if value_consumers or value in graph.outputs:
                spec = graph.value_specs.get(value)
                if spec is None:
                    raise ValidationError(
                        f"dispatch boundary value {value!r} has no TensorSpec; "
                        "run shape inference first"
                    )
                edges.append(
                    DispatchEdge(
                        value=value,
                        spec=spec,
                        producer=producer,
                        consumers=value_consumers,
                        is_graph_output=value in graph.outputs,
                    )
                )
    return tuple(edges)


@dataclass(frozen=True, slots=True)
class DispatchDAG:
    """A validated partition of one source graph into ordered dispatches."""

    source_graph: Graph
    dispatches: tuple[Dispatch, ...]
    edges: tuple[DispatchEdge, ...]
    schedule: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_graph, Graph):
            raise ValidationError("dispatch DAG source_graph must be a Graph")
        object.__setattr__(self, "dispatches", tuple(self.dispatches))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "schedule", tuple(self.schedule))
        self.validate()

    def dispatch(self, dispatch_id: str) -> Dispatch:
        for dispatch in self.dispatches:
            if dispatch.id == dispatch_id:
                return dispatch
        raise KeyError(dispatch_id)

    def dependencies(self, dispatch_id: str) -> tuple[str, ...]:
        self.dispatch(dispatch_id)
        order = {item: index for index, item in enumerate(self.schedule)}
        dependencies = {
            edge.producer
            for edge in self.edges
            if dispatch_id in edge.consumers and edge.producer is not None
        }
        return tuple(sorted(dependencies, key=order.__getitem__))

    @property
    def topological_dispatches(self) -> tuple[Dispatch, ...]:
        return tuple(self.dispatch(dispatch_id) for dispatch_id in self.schedule)

    def validate(self) -> None:
        graph = self.source_graph
        graph.validate()
        if any(not isinstance(dispatch, Dispatch) for dispatch in self.dispatches):
            raise ValidationError("dispatch DAG contains a non-Dispatch value")
        dispatch_ids = [dispatch.id for dispatch in self.dispatches]
        if len(set(dispatch_ids)) != len(dispatch_ids):
            raise ValidationError("dispatch ids must be unique")
        if len(self.schedule) != len(dispatch_ids) or set(self.schedule) != set(dispatch_ids):
            raise ValidationError("dispatch schedule must contain every dispatch exactly once")

        source_node_ids = tuple(node.id for node in graph.nodes)
        covered_node_ids = tuple(
            node_id for dispatch in self.dispatches for node_id in dispatch.node_ids
        )
        if covered_node_ids != source_node_ids:
            raise ValidationError(
                "dispatches must cover every source node exactly once in source order"
            )

        source_nodes = {node.id: node for node in graph.nodes}
        for dispatch in self.dispatches:
            dispatch.graph.validate()
            if tuple(node.id for node in dispatch.graph.nodes) != dispatch.node_ids:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} graph nodes do not match node_ids"
                )
            expected_nodes = tuple(source_nodes[node_id] for node_id in dispatch.node_ids)
            if tuple(node.to_dict() for node in dispatch.graph.nodes) != tuple(
                node.to_dict() for node in expected_nodes
            ):
                raise ValidationError(
                    f"dispatch {dispatch.id!r} graph changed source node semantics"
                )
            expected_inputs, expected_constants, expected_outputs, local_values = (
                _subgraph_boundary(graph, expected_nodes)
            )
            if tuple(dispatch.graph.inputs) != expected_inputs:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} has incorrect boundary inputs"
                )
            if tuple(dispatch.graph.constants) != expected_constants:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} has incorrect constants"
                )
            if dispatch.graph.outputs != expected_outputs:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} has incorrect boundary outputs"
                )
            expected_specs = {
                value: spec
                for value, spec in graph.value_specs.items()
                if value in local_values
            }
            if dispatch.graph.value_specs != expected_specs:
                raise ValidationError(
                    f"dispatch {dispatch.id!r} has incorrect value specs"
                )
            for constant in expected_constants:
                actual = dispatch.graph.constants[constant]
                expected = graph.constants[constant]
                if not _arrays_equal(actual, expected):
                    raise ValidationError(
                        f"dispatch {dispatch.id!r} changed constant {constant!r}"
                    )

        if any(not isinstance(edge, DispatchEdge) for edge in self.edges):
            raise ValidationError("dispatch DAG contains a non-DispatchEdge value")
        edge_values = [edge.value for edge in self.edges]
        if len(set(edge_values)) != len(edge_values):
            raise ValidationError("dispatch edge values must be unique")
        expected_edges = _dispatch_edges(graph, self.dispatches)
        if self.edges != expected_edges:
            raise ValidationError("dispatch edges do not match source producer/consumer boundaries")

        position = {dispatch_id: index for index, dispatch_id in enumerate(self.schedule)}
        for edge in self.edges:
            if edge.producer is None:
                continue
            for consumer in edge.consumers:
                if position[edge.producer] >= position[consumer]:
                    raise ValidationError(
                        f"dispatch schedule is not topological for value {edge.value!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DISPATCH_DAG_SCHEMA_VERSION,
            "graph": self.source_graph.name,
            "inputs": {
                name: spec.to_dict() for name, spec in self.source_graph.inputs.items()
            },
            "outputs": list(self.source_graph.outputs),
            "dispatches": [dispatch.to_dict() for dispatch in self.dispatches],
            "edges": [edge.to_dict() for edge in self.edges],
            "schedule": list(self.schedule),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source_graph: Graph,
    ) -> "DispatchDAG":
        """Restore and validate a manifest against its source graph.

        Dispatch subgraphs and constants are reconstructed from
        ``source_graph``.  The serialized form therefore remains a small plan
        manifest and cannot silently replace executable graph semantics.
        """

        if not isinstance(data, Mapping):
            raise ValidationError("serialized dispatch DAG must be an object")
        if not isinstance(source_graph, Graph):
            raise ValidationError("source_graph must be a Graph")
        source_graph.validate()
        try:
            if data.get("schema_version") != DISPATCH_DAG_SCHEMA_VERSION:
                raise ValidationError(
                    f"unsupported dispatch DAG schema version "
                    f"{data.get('schema_version')!r}"
                )
            if data["graph"] != source_graph.name:
                raise ValidationError(
                    f"dispatch DAG graph {data['graph']!r} does not match "
                    f"{source_graph.name!r}"
                )
            serialized_inputs = {
                name: TensorSpec.from_dict(spec)
                for name, spec in data.get("inputs", {}).items()
            }
            if serialized_inputs != source_graph.inputs:
                raise ValidationError("dispatch DAG inputs do not match source graph")
            if tuple(data["outputs"]) != source_graph.outputs:
                raise ValidationError("dispatch DAG outputs do not match source graph")

            dispatches: list[Dispatch] = []
            for record in data["dispatches"]:
                node_ids = tuple(record["node_ids"])
                subgraph = extract_subgraph(
                    source_graph,
                    node_ids,
                    name=f"{source_graph.name}.{record['id']}",
                )
                group = record.get("group")
                if group is not None:
                    _non_empty_text(group, "serialized dispatch group")
                    subgraph.metadata["extracted_partition"]["group"] = group
                dispatch = Dispatch(
                    id=record["id"],
                    target=record["target"],
                    node_ids=node_ids,
                    graph=subgraph,
                )
                if dict(record) != dispatch.to_dict():
                    raise ValidationError(
                        f"serialized dispatch {dispatch.id!r} boundary metadata "
                        "does not match the source graph"
                    )
                dispatches.append(dispatch)
            edges = tuple(DispatchEdge.from_dict(record) for record in data["edges"])
            schedule = tuple(data["schedule"])
        except ValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid serialized dispatch DAG") from exc
        return cls(source_graph, tuple(dispatches), edges, schedule)

    @classmethod
    def from_json(
        cls,
        source: str | bytes,
        *,
        source_graph: Graph,
    ) -> "DispatchDAG":
        try:
            data = json.loads(source)
        except (TypeError, ValueError) as exc:
            raise ValidationError("invalid dispatch DAG JSON") from exc
        return cls.from_dict(data, source_graph=source_graph)


def partition_regions(
    graph: Graph,
    assignments: Iterable[RegionAssignment],
) -> DispatchDAG:
    """Build a dispatch DAG from explicit, ordered region assignments.

    This form preserves caller-selected boundaries even when adjacent regions
    use the same target.  It is useful after a domain-specific pattern matcher
    has recognized CNN, Attention, and FFN regions.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("partition_regions requires a Graph")
    graph.validate()
    try:
        regions = tuple(assignments)
    except TypeError as exc:
        raise ValidationError("assignments must be iterable") from exc
    if any(not isinstance(item, RegionAssignment) for item in regions):
        raise ValidationError("assignments must contain RegionAssignment values")
    source_node_ids = tuple(node.id for node in graph.nodes)
    assigned_node_ids = tuple(
        node_id for assignment in regions for node_id in assignment.node_ids
    )
    if assigned_node_ids != source_node_ids:
        raise ValidationError(
            "region assignments must cover every graph node exactly once in "
            "source order"
        )

    dispatches: list[Dispatch] = []
    for index, assignment in enumerate(regions):
        dispatch_id = f"dispatch_{index:04d}"
        subgraph = extract_subgraph(
            graph,
            assignment.node_ids,
            name=f"{graph.name}.{dispatch_id}",
        )
        if assignment.group is not None:
            subgraph.metadata["extracted_partition"]["group"] = assignment.group
        dispatches.append(
            Dispatch(
                id=dispatch_id,
                target=assignment.target,
                node_ids=assignment.node_ids,
                graph=subgraph,
            )
        )
    frozen_dispatches = tuple(dispatches)
    edges = _dispatch_edges(graph, frozen_dispatches)
    schedule = tuple(dispatch.id for dispatch in frozen_dispatches)
    return DispatchDAG(graph, frozen_dispatches, edges, schedule)


def _coerce_placement(value: object, node_id: str) -> NodePlacement:
    if isinstance(value, NodePlacement):
        return value
    if isinstance(value, str):
        return NodePlacement(value)
    raise ValidationError(
        f"placement for node {node_id!r} must be a target string or NodePlacement"
    )


def partition_graph(
    graph: Graph,
    placement: Mapping[str, str | NodePlacement],
) -> DispatchDAG:
    """Partition ``graph`` using an explicit placement for every source node.

    Consecutive nodes with the same target and group are coalesced.  A plain
    string mapping remains supported and coalesces solely by target.  The same
    placement on either side of a different placement remains two dispatches;
    this conservative rule keeps the contracted graph acyclic for every
    topological source graph.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("partition_graph requires a Graph")
    graph.validate()
    if not isinstance(placement, Mapping):
        raise ValidationError("placement must map every node id to a target")
    node_ids = tuple(node.id for node in graph.nodes)
    placement_keys = set(placement)
    expected_keys = set(node_ids)
    missing = sorted(expected_keys - placement_keys)
    unknown = sorted(placement_keys - expected_keys, key=repr)
    if missing or unknown:
        raise ValidationError(
            f"placement must cover exactly the graph nodes; missing={missing}, "
            f"unknown={unknown}"
        )

    assignments: list[RegionAssignment] = []
    for node_id in node_ids:
        selected = _coerce_placement(placement[node_id], node_id)
        if (
            assignments
            and assignments[-1].target == selected.target
            and assignments[-1].group == selected.group
        ):
            previous = assignments[-1]
            assignments[-1] = RegionAssignment(
                target=selected.target,
                node_ids=previous.node_ids + (node_id,),
                group=selected.group,
            )
        else:
            assignments.append(
                RegionAssignment(
                    target=selected.target,
                    node_ids=(node_id,),
                    group=selected.group,
                )
            )
    return partition_regions(graph, assignments)


def partition_by_capabilities(
    graph: Graph,
    capabilities: Iterable[BackendCapability],
    *,
    fallback_target: str | None = None,
    fallback_group: str | None = None,
) -> DispatchDAG:
    """Assign nodes using ordered capabilities, then build a dispatch DAG."""

    placements = assign_backends(
        graph,
        capabilities,
        fallback_target=fallback_target,
        fallback_group=fallback_group,
    )
    return partition_graph(graph, placements)


partition_by_placement = partition_graph


__all__ = [
    "BackendCapability",
    "BackendCheck",
    "DISPATCH_DAG_SCHEMA_VERSION",
    "Dispatch",
    "DispatchDAG",
    "DispatchEdge",
    "NodePlacement",
    "RegionAssignment",
    "assign_backends",
    "extract_subgraph",
    "partition_by_capabilities",
    "partition_by_placement",
    "partition_graph",
    "partition_regions",
]
