"""Optimization and analysis passes for the canonical model IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from gac.core.errors import GACError, PassError
from gac.core.ir import Graph, Node, TensorSpec
from gac.core.ops import SUPPORTED_OPS, evaluate_node, infer_node


@dataclass(frozen=True, slots=True)
class PassReport:
    """Machine-readable result of one pass invocation."""

    name: str
    changed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "changed": self.changed,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    graph: Graph
    reports: tuple[PassReport, ...]

    @property
    def changed(self) -> bool:
        return any(report.changed for report in self.reports)


class GraphPass(Protocol):
    name: str

    def run(self, graph: Graph) -> PassReport:
        """Transform ``graph`` in place and return a report."""


class ShapeInferencePass:
    name = "shape-inference"

    def run(self, graph: Graph) -> PassReport:
        specs: dict[str, TensorSpec] = dict(graph.inputs)
        specs.update(
            {
                name: TensorSpec(tuple(value.shape), value.dtype)
                for name, value in graph.constants.items()
            }
        )
        inferred_values: list[str] = []
        for node in graph.nodes:
            inferred = infer_node(node, specs)
            specs.update(inferred)
            inferred_values.extend(inferred)
        changed = specs != graph.value_specs
        graph.value_specs = specs
        return PassReport(
            self.name,
            changed,
            {"inferred_values": inferred_values, "value_count": len(specs)},
        )


class ConstantFoldingPass:
    name = "constant-folding"

    def run(self, graph: Graph) -> PassReport:
        constant_values = dict(graph.constants)
        retained: list[Node] = []
        folded_nodes: list[str] = []
        provenance = graph.metadata.setdefault("provenance", {})
        if not isinstance(provenance, dict):
            raise PassError("graph metadata field 'provenance' must be an object")

        for node in graph.nodes:
            if node.op not in SUPPORTED_OPS or not all(
                name in constant_values for name in node.inputs
            ):
                retained.append(node)
                continue
            results = evaluate_node(node, constant_values)
            constant_values.update(results)
            for output, value in results.items():
                graph.constants[output] = value
                graph.value_specs[output] = TensorSpec(tuple(value.shape), value.dtype)
                provenance[output] = list(node.source_nodes)
            folded_nodes.append(node.id)

        graph.nodes = retained
        return PassReport(
            self.name,
            bool(folded_nodes),
            {
                "folded_nodes": folded_nodes,
                "created_constants": [
                    output
                    for node_id in folded_nodes
                    for node in graph.nodes + []
                    if node.id == node_id
                    for output in node.outputs
                ],
            },
        )


class DeadCodeEliminationPass:
    name = "dead-code-elimination"

    def run(self, graph: Graph) -> PassReport:
        live_values = set(graph.outputs)
        retained_reversed: list[Node] = []
        removed_nodes: list[str] = []
        for node in reversed(graph.nodes):
            if any(output in live_values for output in node.outputs):
                retained_reversed.append(node)
                live_values.update(node.inputs)
            else:
                removed_nodes.append(node.id)
        graph.nodes = list(reversed(retained_reversed))

        removed_constants = [
            name for name in graph.constants if name not in live_values
        ]
        for name in removed_constants:
            del graph.constants[name]

        defined_values = set(graph.inputs) | set(graph.constants)
        for node in graph.nodes:
            defined_values.update(node.outputs)
        graph.value_specs = {
            name: spec
            for name, spec in graph.value_specs.items()
            if name in defined_values
        }
        removed_nodes.reverse()
        return PassReport(
            self.name,
            bool(removed_nodes or removed_constants),
            {
                "removed_nodes": removed_nodes,
                "removed_constants": removed_constants,
            },
        )


def _single_consumer(
    consumers: dict[str, list[Node]], value: str, expected: Node
) -> bool:
    users = consumers.get(value, [])
    return len(users) == 1 and users[0].id == expected.id


def _merge_sources(*nodes: Node) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for node in nodes:
        for source in node.source_nodes:
            if source not in seen:
                seen.add(source)
                result.append(source)
    return tuple(result)


class MatMulBiasGELUFusionPass:
    """Fuse a single-use MatMul -> Add -> GELU chain."""

    name = "matmul-bias-gelu-fusion"

    @staticmethod
    def _unique_id(graph: Graph, base: str) -> str:
        occupied = {node.id for node in graph.nodes}
        if base not in occupied:
            return base
        suffix = 1
        while f"{base}_{suffix}" in occupied:
            suffix += 1
        return f"{base}_{suffix}"

    def run(self, graph: Graph) -> PassReport:
        producers = graph.producer_map()
        consumers = graph.consumers_map()
        output_names = set(graph.outputs)
        removed_ids: set[str] = set()
        replacements: dict[str, Node] = {}
        fusions: list[dict[str, Any]] = []

        for gelu in graph.nodes:
            if gelu.op != "GELU" or len(gelu.inputs) != 1 or len(gelu.outputs) != 1:
                continue
            add = producers.get(gelu.inputs[0])
            if add is None or add.op != "Add" or len(add.inputs) != 2 or len(add.outputs) != 1:
                continue
            matmul: Node | None = None
            bias: str | None = None
            for index, add_input in enumerate(add.inputs):
                candidate = producers.get(add_input)
                if candidate is not None and candidate.op == "MatMul":
                    matmul = candidate
                    bias = add.inputs[1 - index]
                    break
            if (
                matmul is None
                or bias is None
                or len(matmul.inputs) != 2
                or len(matmul.outputs) != 1
                or matmul.id in removed_ids
                or add.id in removed_ids
                or gelu.id in removed_ids
                or matmul.outputs[0] in output_names
                or add.outputs[0] in output_names
                or not _single_consumer(consumers, matmul.outputs[0], add)
                or not _single_consumer(consumers, add.outputs[0], gelu)
            ):
                continue

            fused_id = self._unique_id(
                graph, f"fused_{matmul.id}_{add.id}_{gelu.id}"
            )
            fused = Node(
                id=fused_id,
                op="MatMulBiasGELU",
                inputs=(matmul.inputs[0], matmul.inputs[1], bias),
                outputs=gelu.outputs,
                attrs={"approximation": "tanh"},
                source_nodes=_merge_sources(matmul, add, gelu),
            )
            removed_ids.update({matmul.id, add.id, gelu.id})
            replacements[gelu.id] = fused
            fusions.append(
                {
                    "fused_node": fused.id,
                    "source_nodes": list(fused.source_nodes),
                }
            )

        if not fusions:
            return PassReport(self.name, False, {"fusions": []})

        new_nodes: list[Node] = []
        for node in graph.nodes:
            if node.id in replacements:
                new_nodes.append(replacements[node.id])
            elif node.id not in removed_ids:
                new_nodes.append(node)
        graph.nodes = new_nodes

        live_definitions = set(graph.inputs) | set(graph.constants)
        for node in graph.nodes:
            live_definitions.update(node.outputs)
        graph.value_specs = {
            name: spec
            for name, spec in graph.value_specs.items()
            if name in live_definitions
        }
        history = graph.metadata.setdefault("fusions", [])
        if not isinstance(history, list):
            raise PassError("graph metadata field 'fusions' must be an array")
        history.extend(fusions)
        return PassReport(self.name, True, {"fusions": fusions})


class AddReLUFusionPass:
    """Fuse a single-use Add -> ReLU chain without exposing the sum."""

    name = "add-relu-fusion"

    def run(self, graph: Graph) -> PassReport:
        producers = graph.producer_map()
        consumers = graph.consumers_map()
        output_names = set(graph.outputs)
        removed_ids: set[str] = set()
        replacements: dict[str, Node] = {}
        fusions: list[dict[str, Any]] = []

        for relu in graph.nodes:
            if relu.op != "ReLU" or len(relu.inputs) != 1 or len(relu.outputs) != 1:
                continue
            add = producers.get(relu.inputs[0])
            if (
                add is None
                or add.op != "Add"
                or len(add.inputs) != 2
                or len(add.outputs) != 1
                or add.id in removed_ids
                or relu.id in removed_ids
                or add.outputs[0] in output_names
                or not _single_consumer(consumers, add.outputs[0], relu)
            ):
                continue

            fused_id = MatMulBiasGELUFusionPass._unique_id(
                graph, f"fused_{add.id}_{relu.id}"
            )
            fused = Node(
                id=fused_id,
                op="AddReLU",
                inputs=add.inputs,
                outputs=relu.outputs,
                source_nodes=_merge_sources(add, relu),
            )
            removed_ids.update({add.id, relu.id})
            replacements[relu.id] = fused
            fusions.append(
                {
                    "fused_node": fused.id,
                    "source_nodes": list(fused.source_nodes),
                }
            )

        if not fusions:
            return PassReport(self.name, False, {"fusions": []})

        graph.nodes = [
            replacements[node.id] if node.id in replacements else node
            for node in graph.nodes
            if node.id not in removed_ids or node.id in replacements
        ]
        defined_values = set(graph.inputs) | set(graph.constants)
        for node in graph.nodes:
            defined_values.update(node.outputs)
        graph.value_specs = {
            name: spec
            for name, spec in graph.value_specs.items()
            if name in defined_values
        }
        history = graph.metadata.setdefault("fusions", [])
        if not isinstance(history, list):
            raise PassError("graph metadata field 'fusions' must be an array")
        history.extend(fusions)
        return PassReport(self.name, True, {"fusions": fusions})


class ConvBatchNormFusionPass:
    """Fold inference BatchNorm constants into a single Conv2D."""

    name = "conv-batchnorm-folding"

    @staticmethod
    def _unique_value(graph: Graph, base: str) -> str:
        occupied = set(graph.inputs) | set(graph.constants) | set(graph.value_specs)
        occupied.update(output for node in graph.nodes for output in node.outputs)
        if base not in occupied:
            return base
        suffix = 1
        while f"{base}_{suffix}" in occupied:
            suffix += 1
        return f"{base}_{suffix}"

    def run(self, graph: Graph) -> PassReport:
        producers = graph.producer_map()
        consumers = graph.consumers_map()
        output_names = set(graph.outputs)
        removed_ids: set[str] = set()
        replacements: dict[str, Node] = {}
        fusions: list[dict[str, Any]] = []

        for batch_norm in graph.nodes:
            if batch_norm.op != "BatchNorm" or len(batch_norm.inputs) != 5:
                continue
            convolution = producers.get(batch_norm.inputs[0])
            if (
                convolution is None
                or convolution.op != "Conv2D"
                or len(convolution.inputs) not in {2, 3}
                or convolution.outputs[0] in output_names
                or not _single_consumer(consumers, convolution.outputs[0], batch_norm)
                or convolution.id in removed_ids
            ):
                continue
            weight_name = convolution.inputs[1]
            parameter_names = batch_norm.inputs[1:]
            required = (weight_name, *parameter_names)
            if len(convolution.inputs) == 3:
                required += (convolution.inputs[2],)
            if not all(name in graph.constants for name in required):
                continue

            weight = graph.constants[weight_name]
            scale, offset, mean, variance = (
                graph.constants[name] for name in parameter_names
            )
            epsilon = float(batch_norm.attrs.get("epsilon", 1e-5))
            denominator = variance.astype(np.float64) + epsilon
            if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
                raise PassError(
                    f"BatchNorm node {batch_norm.id!r} has invalid variance/epsilon"
                )
            factor = scale / np.sqrt(variance + np.asarray(epsilon, dtype=variance.dtype))
            old_bias = (
                graph.constants[convolution.inputs[2]]
                if len(convolution.inputs) == 3
                else np.zeros_like(scale)
            )
            folded_weight = (
                weight * factor.reshape((-1,) + (1,) * (weight.ndim - 1))
            ).astype(weight.dtype, copy=False)
            folded_bias = ((old_bias - mean) * factor + offset).astype(
                weight.dtype, copy=False
            )
            folded_weight_name = self._unique_value(
                graph, f"{batch_norm.id}__folded_weight"
            )
            graph.constants[folded_weight_name] = np.ascontiguousarray(folded_weight)
            graph.value_specs[folded_weight_name] = TensorSpec(
                folded_weight.shape, folded_weight.dtype
            )
            folded_bias_name = self._unique_value(
                graph, f"{batch_norm.id}__folded_bias"
            )
            graph.constants[folded_bias_name] = np.ascontiguousarray(folded_bias)
            graph.value_specs[folded_bias_name] = TensorSpec(
                folded_bias.shape, folded_bias.dtype
            )
            fused_id = MatMulBiasGELUFusionPass._unique_id(
                graph, f"fused_{convolution.id}_{batch_norm.id}"
            )
            fused = Node(
                id=fused_id,
                op="Conv2D",
                inputs=(convolution.inputs[0], folded_weight_name, folded_bias_name),
                outputs=batch_norm.outputs,
                attrs=dict(convolution.attrs),
                source_nodes=_merge_sources(convolution, batch_norm),
            )
            removed_ids.update({convolution.id, batch_norm.id})
            replacements[batch_norm.id] = fused
            fusions.append(
                {
                    "fused_node": fused.id,
                    "source_nodes": list(fused.source_nodes),
                    "folded_weight": folded_weight_name,
                    "folded_bias": folded_bias_name,
                }
            )

        if not fusions:
            return PassReport(self.name, False, {"fusions": []})
        graph.nodes = [
            replacements[node.id] if node.id in replacements else node
            for node in graph.nodes
            if node.id not in removed_ids or node.id in replacements
        ]
        defined_values = set(graph.inputs) | set(graph.constants)
        for node in graph.nodes:
            defined_values.update(node.outputs)
        graph.value_specs = {
            name: spec
            for name, spec in graph.value_specs.items()
            if name in defined_values
        }
        history = graph.metadata.setdefault("fusions", [])
        if not isinstance(history, list):
            raise PassError("graph metadata field 'fusions' must be an array")
        history.extend(fusions)
        return PassReport(self.name, True, {"fusions": fusions})


class PassManager:
    """Runs graph passes transactionally on a clone by default."""

    def __init__(self, passes: list[GraphPass] | tuple[GraphPass, ...]) -> None:
        self.passes = tuple(passes)

    def run(self, graph: Graph, *, in_place: bool = False) -> PipelineResult:
        current = graph if in_place else graph.clone()
        current.validate()
        reports: list[PassReport] = []
        for graph_pass in self.passes:
            try:
                report = graph_pass.run(current)
                current.validate()
            except GACError:
                raise
            except Exception as exc:  # pragma: no cover - defensive plugin boundary
                raise PassError(f"pass {graph_pass.name!r} failed: {exc}") from exc
            reports.append(report)
        history = current.metadata.setdefault("pass_history", [])
        if not isinstance(history, list):
            raise PassError("graph metadata field 'pass_history' must be an array")
        history.extend(report.to_dict() for report in reports)
        return PipelineResult(current, tuple(reports))


def default_pass_manager() -> PassManager:
    """Return the deterministic v0 pipeline used by the package compiler."""

    return PassManager(
        [
            ShapeInferencePass(),
            ConstantFoldingPass(),
            DeadCodeEliminationPass(),
            ConvBatchNormFusionPass(),
            MatMulBiasGELUFusionPass(),
            AddReLUFusionPass(),
            DeadCodeEliminationPass(),
            ShapeInferencePass(),
        ]
    )


def optimize(graph: Graph, *, in_place: bool = False) -> PipelineResult:
    return default_pass_manager().run(graph, in_place=in_place)
