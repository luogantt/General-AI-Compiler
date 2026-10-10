"""Explicit backend capability reports and deterministic graph segmentation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from .errors import BackendError
from .ir import Graph, Node


CapabilityCheck = Callable[[Node, Graph], str | None]


@dataclass(frozen=True, slots=True)
class NodeSupport:
    node_id: str
    op: str
    supported: bool
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "supported": self.supported,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SupportSegment:
    supported: bool
    node_ids: tuple[str, ...]
    ops: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "node_ids": list(self.node_ids),
            "ops": list(self.ops),
        }


@dataclass(frozen=True, slots=True)
class GraphSupportReport:
    graph_name: str
    target: str
    nodes: tuple[NodeSupport, ...]
    segments: tuple[SupportSegment, ...]

    @property
    def fully_supported(self) -> bool:
        return all(node.supported for node in self.nodes)

    @property
    def unsupported(self) -> tuple[NodeSupport, ...]:
        return tuple(node for node in self.nodes if not node.supported)

    def require_supported(self) -> None:
        if self.fully_supported:
            return
        details = "; ".join(
            f"{node.node_id}({node.op}): {node.reason}" for node in self.unsupported
        )
        raise BackendError(f"{self.target}: graph {self.graph_name!r} is not supported: {details}")

    def to_dict(self) -> dict:
        counts = Counter(node.op for node in self.nodes)
        unsupported_counts = Counter(node.op for node in self.unsupported)
        return {
            "schema_version": 1,
            "graph": self.graph_name,
            "target": self.target,
            "fully_supported": self.fully_supported,
            "node_count": len(self.nodes),
            "op_counts": dict(sorted(counts.items())),
            "unsupported_op_counts": dict(sorted(unsupported_counts.items())),
            "nodes": [node.to_dict() for node in self.nodes],
            "segments": [segment.to_dict() for segment in self.segments],
        }


def analyze_support(
    graph: Graph,
    *,
    target: str,
    supported_ops: Iterable[str],
    check: CapabilityCheck | None = None,
) -> GraphSupportReport:
    """Analyze every node without mutating or silently partitioning the graph.

    ``check`` returns ``None`` for a supported node or a concrete rejection
    reason.  Contiguous segments are diagnostic only: the caller must make an
    explicit decision before any heterogeneous execution is allowed.
    """

    graph.validate()
    allowed = frozenset(supported_ops)
    nodes: list[NodeSupport] = []
    for node in graph.nodes:
        if node.op not in allowed:
            nodes.append(NodeSupport(node.id, node.op, False, "operation is not registered"))
            continue
        reason = check(node, graph) if check is not None else None
        nodes.append(NodeSupport(node.id, node.op, reason is None, reason))

    segments: list[SupportSegment] = []
    for node, support in zip(graph.nodes, nodes, strict=True):
        if segments and segments[-1].supported == support.supported:
            previous = segments[-1]
            segments[-1] = SupportSegment(
                previous.supported,
                previous.node_ids + (node.id,),
                previous.ops + (node.op,),
            )
        else:
            segments.append(SupportSegment(support.supported, (node.id,), (node.op,)))
    return GraphSupportReport(graph.name, target, tuple(nodes), tuple(segments))


__all__ = [
    "CapabilityCheck",
    "GraphSupportReport",
    "NodeSupport",
    "SupportSegment",
    "analyze_support",
]
