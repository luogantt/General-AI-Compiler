"""Deterministic composition of canonical graphs.

Component frontends keep their own small, testable graphs.  ``GraphComposer``
inlines those graphs into a larger static program while namespacing every node,
constant, and intermediate value.  It performs no implicit casts or layout
changes; incompatible bindings fail during final shape inference.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


class GraphComposer:
    """Accumulate nodes and safely inline already-canonical component graphs."""

    def __init__(
        self,
        name: str,
        inputs: Mapping[str, TensorSpec],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValidationError("composed graph name must be non-empty")
        if not isinstance(inputs, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(spec, TensorSpec)
            for key, spec in inputs.items()
        ):
            raise ValidationError("composed graph inputs must map names to TensorSpec values")
        self.name = name
        self.inputs = dict(inputs)
        self.nodes: list[Node] = []
        self.constants: dict[str, np.ndarray] = {}
        self.metadata = copy.deepcopy(dict(metadata or {}))
        self._known = set(self.inputs)
        self._node_ids: set[str] = set()
        self._prefixes: set[str] = set()

    @property
    def known_values(self) -> frozenset[str]:
        return frozenset(self._known)

    def add_constant(self, name: str, value: np.ndarray) -> str:
        if not isinstance(name, str) or not name or name in self._known:
            raise ValidationError(f"composed value {name!r} is invalid or already defined")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValidationError(f"composed constant {name!r} cannot use object dtype")
        self.constants[name] = np.ascontiguousarray(array).copy()
        self._known.add(name)
        return name

    def add_node(self, node: Node) -> tuple[str, ...]:
        if not isinstance(node, Node):
            raise ValidationError("composed nodes must be Node values")
        if node.id in self._node_ids:
            raise ValidationError(f"composed node id {node.id!r} is already defined")
        missing = [name for name in node.inputs if name not in self._known]
        if missing:
            raise ValidationError(f"composed node {node.id!r} has unknown inputs {missing}")
        duplicate = [name for name in node.outputs if name in self._known]
        if duplicate:
            raise ValidationError(f"composed node {node.id!r} redefines values {duplicate}")
        self.nodes.append(node)
        self._node_ids.add(node.id)
        self._known.update(node.outputs)
        return node.outputs

    def inline(
        self,
        graph: Graph,
        *,
        prefix: str,
        bindings: Mapping[str, str],
        output_bindings: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Inline ``graph`` and return its old-to-new output value mapping."""

        if not isinstance(graph, Graph):
            raise ValidationError("inline source must be a Graph")
        graph.validate()
        if (
            not isinstance(prefix, str)
            or not prefix
            or prefix in self._prefixes
            or any(character.isspace() for character in prefix)
        ):
            raise ValidationError("inline prefix must be unique, non-empty, and contain no whitespace")
        if not isinstance(bindings, Mapping) or set(bindings) != set(graph.inputs):
            raise ValidationError(
                f"inline bindings must name exactly {sorted(graph.inputs)}"
            )
        for source, destination in bindings.items():
            if not isinstance(destination, str) or destination not in self._known:
                raise ValidationError(
                    f"inline input {source!r} binds unknown value {destination!r}"
                )
        requested_outputs = dict(output_bindings or {})
        if set(requested_outputs) - set(graph.outputs):
            raise ValidationError("inline output bindings contain a non-output value")
        if any(not isinstance(name, str) or not name for name in requested_outputs.values()):
            raise ValidationError("inline output binding names must be non-empty strings")

        # Inlining is transactional: a late name or topology error must not
        # leave constants or nodes from half of a component in the parent.
        node_count = len(self.nodes)
        original_constants = set(self.constants)
        original_known = set(self._known)
        original_node_ids = set(self._node_ids)
        original_prefixes = set(self._prefixes)
        try:
            rename = dict(bindings)
            for constant_name, value in graph.constants.items():
                destination = f"{prefix}.{constant_name}"
                self.add_constant(destination, value)
                rename[constant_name] = destination

            for node in graph.nodes:
                outputs: list[str] = []
                for output in node.outputs:
                    destination = requested_outputs.get(output, f"{prefix}.{output}")
                    if destination in self._known:
                        raise ValidationError(
                            f"inline output {output!r} would redefine {destination!r}"
                        )
                    rename[output] = destination
                    outputs.append(destination)
                try:
                    node_inputs = tuple(rename[name] for name in node.inputs)
                except KeyError as exc:
                    raise ValidationError(
                        f"inline node {node.id!r} cannot resolve value {exc.args[0]!r}"
                    ) from exc
                self.add_node(
                    Node(
                        id=f"{prefix}.{node.id}",
                        op=node.op,
                        inputs=node_inputs,
                        outputs=tuple(outputs),
                        attrs=copy.deepcopy(node.attrs),
                        source_nodes=node.source_nodes,
                    )
                )
            self._prefixes.add(prefix)
            return {name: rename[name] for name in graph.outputs}
        except Exception:
            del self.nodes[node_count:]
            for name in set(self.constants) - original_constants:
                del self.constants[name]
            self._known = original_known
            self._node_ids = original_node_ids
            self._prefixes = original_prefixes
            raise

    def build(
        self,
        outputs: tuple[str, ...] | list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Graph:
        selected = tuple(outputs)
        if not selected or any(name not in self._known for name in selected):
            raise ValidationError("composed graph outputs must be known values")
        combined_metadata = copy.deepcopy(self.metadata)
        if metadata:
            combined_metadata.update(copy.deepcopy(dict(metadata)))
        graph = Graph(
            name=self.name,
            inputs=self.inputs,
            outputs=selected,
            nodes=self.nodes,
            constants=self.constants,
            metadata=combined_metadata,
        )
        ShapeInferencePass().run(graph)
        graph.validate()
        return graph


__all__ = ["GraphComposer"]
