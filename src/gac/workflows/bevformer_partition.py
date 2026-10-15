"""Deterministic BEVFormer region discovery for heterogeneous compilation.

The generic partitioner is intentionally kept separate from this module.  This
file recognizes the one production subgraph that the Vortex backend can compile
today: a Transformer FFN in either its seven-node pre-fusion form or six-node
``AddReLU`` form. Every other node remains an explicit CUDA placement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal

import numpy as np

from gac.core.errors import BackendError, ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


BEVFORMER_CUDA_TARGET = "cuda-sm86"
BEVFORMER_VORTEX_TARGET = "vortex-v3"
BevFormerNodeKind = Literal["cnn", "attention", "ffn", "utility"]

_CANONICAL_CONSTANTS = (
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
    "norm_weight",
    "norm_bias",
)


@dataclass(frozen=True, slots=True)
class BevFormerFFNRegion:
    """A validated FFN and its boundary/value bindings.

    Canonical BEVFormer graphs contain seven nodes before ``Add+ReLU`` fusion
    and six nodes afterwards.  Both forms describe the same Vortex contract;
    ``fused_bias_relu`` records which form was recognized so extraction can
    preserve, rather than guess, the source semantics.
    """

    region_id: str
    node_ids: tuple[str, ...]
    query: str
    output: str
    fc1_weight: str
    fc1_bias: str
    fc2_weight: str
    fc2_bias: str
    norm_weight: str
    norm_bias: str
    input_shape: tuple[int, ...]
    hidden_dim: int
    fused_bias_relu: bool = False

    @property
    def constants(self) -> tuple[str, ...]:
        return (
            self.fc1_weight,
            self.fc1_bias,
            self.fc2_weight,
            self.fc2_bias,
            self.norm_weight,
            self.norm_bias,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "node_ids": list(self.node_ids),
            "query": self.query,
            "output": self.output,
            "constants": list(self.constants),
            "input_shape": list(self.input_shape),
            "hidden_dim": self.hidden_dim,
            "fused_bias_relu": self.fused_bias_relu,
            "source_node_count": len(self.node_ids),
        }


@dataclass(frozen=True, slots=True)
class BevFormerNodePlacement:
    node_id: str
    kind: BevFormerNodeKind
    target: str
    region_id: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "target": self.target,
            "region_id": self.region_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BevFormerPartitionReport:
    graph_name: str
    regions: tuple[BevFormerFFNRegion, ...]
    placements: tuple[BevFormerNodePlacement, ...]
    rejected_ffn_candidates: tuple[tuple[str, str], ...]

    def placement(self, node_id: str) -> BevFormerNodePlacement:
        for placement in self.placements:
            if placement.node_id == node_id:
                return placement
        raise KeyError(node_id)

    def to_dict(self) -> dict[str, Any]:
        kind_counts = {
            kind: sum(placement.kind == kind for placement in self.placements)
            for kind in ("cnn", "attention", "ffn", "utility")
        }
        target_counts: dict[str, int] = {}
        for placement in self.placements:
            target_counts[placement.target] = target_counts.get(placement.target, 0) + 1
        return {
            "graph": self.graph_name,
            "node_count": len(self.placements),
            "kind_counts": kind_counts,
            "target_counts": target_counts,
            "vortex_ffn_regions": [region.to_dict() for region in self.regions],
            "placements": [placement.to_dict() for placement in self.placements],
            "rejected_ffn_candidates": [
                {"node_id": node_id, "reason": reason}
                for node_id, reason in self.rejected_ffn_candidates
            ],
        }


def _node_text(node: Node) -> str:
    return " ".join((node.id, *node.source_nodes)).lower()


def _looks_like_ffn(node: Node) -> bool:
    text = _node_text(node)
    return bool(
        re.search(r"(?:^|[._\[])ffn(?:$|[._\[])|feed.?forward", text)
        or "ffns.0" in text
    )


def _looks_like_fc1(node: Node) -> bool:
    if node.op != "MatMul" or not _looks_like_ffn(node):
        return False
    text = _node_text(node)
    return bool(
        "fc1" in text
        or "layers[0]" in text
        or "layers.0" in text
        or "layers.0.0" in text
    )


def classify_bevformer_node(node: Node) -> BevFormerNodeKind:
    """Classify a canonical BEVFormer node by semantic family.

    FFN naming is checked first because decoder FFNs are nested below a
    Transformer/attention scope.  The classifier is descriptive; only strict
    region discovery grants Vortex placement.
    """

    if not isinstance(node, Node):
        raise TypeError("node must be a Node")
    text = _node_text(node)
    node_id = node.id.lower()
    if _looks_like_ffn(node):
        return "ffn"
    # Geometry constructs reference points and visibility masks consumed by
    # spatial attention, but it is not itself an attention kernel.  Keep this
    # id-based guard ahead of source provenance such as
    # ``SpatialCrossAttention.query_visibility``.
    if node_id.startswith("encoder.geometry."):
        return "utility"
    if (
        node.op in {"Conv2D", "BatchNorm", "MaxPool2D"}
        or node_id in {"images_to_float32", "flatten_image_camera_batch"}
        or node_id.startswith(("backbone.", "feature_prep."))
        or "backbone" in text
        or "resnet" in text
        or "fpn" in text
    ):
        return "cnn"
    # Queue selection and the normalization immediately following temporal or
    # spatial attention belong to that layer's attention scheduling region.
    # Classifying them with the surrounding attention avoids artificial
    # one-node utility dispatches while leaving the post-FFN norm in the strict
    # FFN pattern below ``analyze_bevformer_partitions``.
    encoder_attention_scaffolding = bool(
        re.search(
            r"^encoder\.layer\d+\."
            r"(?:cold_temporal_value_concat|select_temporal_value|norm[01])$",
            node_id,
        )
    )
    decoder_attention_norm = bool(
        re.search(
            r"^decoder\.decoder\.layer\d+\.transformer\.norm1$",
            node_id,
        )
    )
    if (
        node.op == "MSDeformableAttention"
        or encoder_attention_scaffolding
        or decoder_attention_norm
        or "attention" in text
        or ".temporal." in text
        or ".spatial." in text
    ):
        return "attention"
    return "utility"


def _single_output(node: Node, role: str) -> str:
    if len(node.outputs) != 1:
        raise ValidationError(f"{role} must have exactly one output")
    return node.outputs[0]


def _only_consumer(
    consumers: dict[str, list[Node]], value: str, role: str
) -> Node:
    # ``consumers_map`` records one entry per operand.  A malformed binary op
    # can therefore mention the same value twice while still being one node;
    # deduplicate here and let the operation-specific check diagnose it.
    users = list({node.id: node for node in consumers.get(value, [])}.values())
    if len(users) != 1:
        raise ValidationError(
            f"{role} output {value!r} must have one internal consumer; got "
            f"{[node.id for node in users]}"
        )
    return users[0]


def _binary_other_input(node: Node, value: str, role: str) -> str:
    if len(node.inputs) != 2 or node.inputs.count(value) != 1:
        raise ValidationError(f"{role} must consume {value!r} exactly once")
    return node.inputs[1] if node.inputs[0] == value else node.inputs[0]


def _require_constant(graph: Graph, name: str, role: str) -> str:
    if name not in graph.constants:
        raise ValidationError(f"{role} {name!r} must be a graph constant")
    return name


def _require_static_f32(spec: TensorSpec | None, name: str) -> TensorSpec:
    if spec is None:
        raise ValidationError(f"FFN value {name!r} has no inferred TensorSpec")
    if spec.dtype != "float32":
        raise ValidationError(f"FFN value {name!r} must use float32")
    if spec.layout != "contiguous":
        raise ValidationError(f"FFN value {name!r} must use contiguous layout")
    if not spec.is_static or any(dimension <= 0 for dimension in spec.shape):
        raise ValidationError(f"FFN value {name!r} must have a positive static shape")
    return spec


def _region_id(graph: Graph, fc1: Node) -> str:
    node_id = fc1.id
    for suffix in (".fc1_matmul", ".ffn_fc1", "fc1_matmul"):
        if node_id.endswith(suffix):
            prefix = node_id[: -len(suffix)].rstrip("._")
            if suffix == ".ffn_fc1":
                prefix += ".ffn"
            if prefix:
                return prefix
    return f"{graph.name}.ffn"


def _match_ffn_region(
    graph: Graph,
    fc1: Node,
    consumers: dict[str, list[Node]],
) -> BevFormerFFNRegion:
    if len(fc1.inputs) != 2:
        raise ValidationError("FFN FC1 MatMul must have two inputs")
    query, fc1_weight = fc1.inputs
    if query in graph.constants:
        raise ValidationError("FFN query must be a runtime value")
    _require_constant(graph, fc1_weight, "FC1 weight")

    fc1_mm = _single_output(fc1, "FFN FC1 MatMul")
    bias_activation = _only_consumer(consumers, fc1_mm, "FFN FC1 MatMul")
    if bias_activation.op not in {"Add", "AddReLU"}:
        raise ValidationError("FFN FC1 MatMul must feed Add or fused AddReLU")
    fc1_bias = _require_constant(
        graph,
        _binary_other_input(
            bias_activation,
            fc1_mm,
            "FFN FC1 bias Add" if bias_activation.op == "Add" else "FFN FC1 AddReLU",
        ),
        "FC1 bias",
    )

    fused_bias_relu = bias_activation.op == "AddReLU"
    if fused_bias_relu:
        activated = _single_output(bias_activation, "FFN FC1 AddReLU")
        activation_nodes = (bias_activation,)
    else:
        fc1_added = _single_output(bias_activation, "FFN FC1 bias Add")
        relu = _only_consumer(consumers, fc1_added, "FFN FC1 bias Add")
        if relu.op != "ReLU" or relu.inputs != (fc1_added,):
            raise ValidationError("FFN FC1 bias Add must feed a unary ReLU")
        activated = _single_output(relu, "FFN ReLU")
        activation_nodes = (bias_activation, relu)
    fc2 = _only_consumer(consumers, activated, "FFN ReLU")
    if fc2.op != "MatMul" or len(fc2.inputs) != 2 or fc2.inputs[0] != activated:
        raise ValidationError("FFN ReLU must feed the left input of FC2 MatMul")
    fc2_weight = _require_constant(graph, fc2.inputs[1], "FC2 weight")

    fc2_mm = _single_output(fc2, "FFN FC2 MatMul")
    add2 = _only_consumer(consumers, fc2_mm, "FFN FC2 MatMul")
    if add2.op != "Add":
        raise ValidationError("FFN FC2 MatMul must feed an Add")
    fc2_bias = _require_constant(
        graph, _binary_other_input(add2, fc2_mm, "FFN FC2 bias Add"), "FC2 bias"
    )

    projected = _single_output(add2, "FFN FC2 bias Add")
    residual = _only_consumer(consumers, projected, "FFN FC2 bias Add")
    if residual.op != "Add" or len(residual.inputs) != 2:
        raise ValidationError("FFN projection must feed a residual Add")
    if residual.inputs.count(projected) != 1 or residual.inputs.count(query) != 1:
        raise ValidationError("FFN residual Add must reconnect the original query")

    residual_value = _single_output(residual, "FFN residual Add")
    norm = _only_consumer(consumers, residual_value, "FFN residual Add")
    if norm.op != "LayerNorm" or len(norm.inputs) != 3 or norm.inputs[0] != residual_value:
        raise ValidationError("FFN residual Add must feed a three-input LayerNorm")
    norm_weight = _require_constant(graph, norm.inputs[1], "LayerNorm weight")
    norm_bias = _require_constant(graph, norm.inputs[2], "LayerNorm bias")
    output = _single_output(norm, "FFN LayerNorm")

    constants = (
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        norm_weight,
        norm_bias,
    )
    if len(set(constants)) != len(_CANONICAL_CONSTANTS):
        raise ValidationError("FFN must use six distinct constants")

    nodes = (fc1, *activation_nodes, fc2, add2, residual, norm)
    expected_node_count = 6 if fused_bias_relu else 7
    if len({node.id for node in nodes}) != expected_node_count:
        raise ValidationError(
            f"FFN region must contain {expected_node_count} distinct nodes"
        )
    exposed_intermediates = sorted(
        set(graph.outputs).intersection(
            output_name for node in nodes[:-1] for output_name in node.outputs
        )
    )
    if exposed_intermediates:
        raise ValidationError(
            "FFN intermediate values cannot be graph outputs: "
            f"{exposed_intermediates}"
        )

    values = {query, output, *constants}
    for node in nodes:
        values.update(node.outputs)
    for name in values:
        _require_static_f32(graph.value_specs.get(name), name)

    query_spec = graph.value_specs[query]
    output_spec = graph.value_specs[output]
    if query_spec.rank != 3 or query_spec.shape[0] != 1:
        raise ValidationError("FFN query must use static [1, rows, channels] ABI")
    if output_spec != query_spec:
        raise ValidationError("FFN output must match the query TensorSpec")
    channels = query_spec.shape[-1]
    hidden = graph.value_specs[fc1_weight].shape
    expected_shapes = {
        fc1_weight: (channels, hidden[1] if len(hidden) == 2 else -1),
        fc1_bias: (hidden[1] if len(hidden) == 2 else -1,),
        fc2_weight: (hidden[1] if len(hidden) == 2 else -1, channels),
        fc2_bias: (channels,),
        norm_weight: (channels,),
        norm_bias: (channels,),
    }
    if len(hidden) != 2 or hidden[0] != channels:
        raise ValidationError("FFN FC1 weight must have shape [channels, hidden]")
    for name, expected in expected_shapes.items():
        if graph.value_specs[name].shape != expected:
            raise ValidationError(
                f"FFN constant {name!r} must have shape {expected}, got "
                f"{graph.value_specs[name].shape}"
            )
        if not np.all(np.isfinite(graph.constants[name])):
            raise ValidationError(
                f"Vortex backend contract rejected FFN: constant {name!r} "
                "must contain only finite float32 values"
            )

    # Capability analysis must enforce the same legality rules as the actual
    # lowering path.  Reuse MLIR's node checker for attributes (notably the
    # last-axis/finite-positive LayerNorm contract), then apply the RV32 index
    # and byte-size limits used by the Vortex device emitter.  Keeping this in
    # discovery prevents an apparently supported FFN from failing only after
    # it has already been assigned to Vortex.
    from gac.targets.mlir import _check_node as _check_mlir_node
    from gac.targets.vortex.device_mlir import validate_bevformer_ffn_rv32_extents

    try:
        for node in nodes:
            _check_mlir_node(node, graph.value_specs)
        validate_bevformer_ffn_rv32_extents(
            math.prod(query_spec.shape[:-1]), channels, hidden[1]
        )
    except BackendError as exc:
        raise ValidationError(f"Vortex backend contract rejected FFN: {exc}") from exc

    return BevFormerFFNRegion(
        region_id=_region_id(graph, fc1),
        node_ids=tuple(node.id for node in nodes),
        query=query,
        output=output,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        input_shape=query_spec.shape,
        hidden_dim=hidden[1],
        fused_bias_relu=fused_bias_relu,
    )


def _discover_with_rejections(
    graph: Graph,
) -> tuple[tuple[BevFormerFFNRegion, ...], tuple[tuple[str, str], ...], Graph]:
    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    graph.validate()
    shaped = graph.clone()
    ShapeInferencePass().run(shaped)
    consumers = shaped.consumers_map()
    regions: list[BevFormerFFNRegion] = []
    rejected: list[tuple[str, str]] = []
    claimed: set[str] = set()
    for node in shaped.nodes:
        if not _looks_like_fc1(node):
            continue
        try:
            region = _match_ffn_region(shaped, node, consumers)
        except ValidationError as exc:
            rejected.append((node.id, str(exc)))
            continue
        overlap = claimed.intersection(region.node_ids)
        if overlap:
            rejected.append(
                (node.id, f"FFN candidate overlaps nodes {sorted(overlap)}")
            )
            continue
        claimed.update(region.node_ids)
        regions.append(region)
    return tuple(regions), tuple(rejected), shaped


def discover_bevformer_ffn_regions(graph: Graph) -> tuple[BevFormerFFNRegion, ...]:
    """Return strict, non-overlapping pre- or post-fusion FFN regions."""

    regions, _, _ = _discover_with_rejections(graph)
    return regions


def extract_bevformer_ffn_region(
    graph: Graph, region: BevFormerFFNRegion
) -> Graph:
    """Rewrite a validated region to the exact ABI consumed by Vortex."""

    if not isinstance(region, BevFormerFFNRegion):
        raise TypeError("region must be a BevFormerFFNRegion")
    regions, _, shaped = _discover_with_rejections(graph)
    matches = [candidate for candidate in regions if candidate == region]
    if len(matches) != 1:
        raise ValidationError("FFN region does not belong to the supplied graph")

    by_id = {node.id: node for node in shaped.nodes}
    originals = [by_id[node_id] for node_id in region.node_ids]
    if region.fused_bias_relu:
        canonical_ids = (
            "fc1_matmul",
            "fc1_bias_relu",
            "fc2_matmul",
            "fc2_bias_add",
            "residual_add",
            "post_layer_norm",
        )
        canonical_inputs = (
            ("query", "fc1_weight"),
            ("fc1_mm", "fc1_bias"),
            ("activated", "fc2_weight"),
            ("fc2_mm", "fc2_bias"),
            ("query", "projected"),
            ("ffn_output", "norm_weight", "norm_bias"),
        )
        canonical_outputs = (
            ("fc1_mm",),
            ("activated",),
            ("fc2_mm",),
            ("projected",),
            ("ffn_output",),
            ("output",),
        )
    else:
        canonical_ids = (
            "fc1_matmul",
            "fc1_bias_add",
            "relu",
            "fc2_matmul",
            "fc2_bias_add",
            "residual_add",
            "post_layer_norm",
        )
        canonical_inputs = (
            ("query", "fc1_weight"),
            ("fc1_mm", "fc1_bias"),
            ("fc1",),
            ("activated", "fc2_weight"),
            ("fc2_mm", "fc2_bias"),
            ("query", "projected"),
            ("ffn_output", "norm_weight", "norm_bias"),
        )
        canonical_outputs = (
            ("fc1_mm",),
            ("fc1",),
            ("activated",),
            ("fc2_mm",),
            ("projected",),
            ("ffn_output",),
            ("output",),
        )
    nodes = [
        Node(
            id=canonical_id,
            op=original.op,
            # Add is commutative, but the Vortex matcher deliberately uses one
            # canonical operand order.  Encoder graphs spell the residual as
            # ``projected + query`` while decoder graphs use ``query +
            # projected``; normalize both here.
            inputs=inputs,
            outputs=outputs,
            attrs=dict(original.attrs),
            source_nodes=original.source_nodes,
        )
        for canonical_id, inputs, outputs, original in zip(
            canonical_ids,
            canonical_inputs,
            canonical_outputs,
            originals,
            strict=True,
        )
    ]
    constants = {
        canonical: np.ascontiguousarray(shaped.constants[source]).copy()
        for canonical, source in zip(
            _CANONICAL_CONSTANTS, region.constants, strict=True
        )
    }
    query_spec = shaped.value_specs[region.query]
    channels = query_spec.shape[-1]
    result = Graph(
        name=re.sub(r"[^A-Za-z0-9_.-]", "_", region.region_id),
        inputs={"query": query_spec},
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-transformer-ffn",
            "partition_source_graph": graph.name,
            "partition_source_nodes": list(region.node_ids),
            "profile": {
                "batch": 1,
                "rows": int(np.prod(query_spec.shape[:-1])),
                "embed_dim": channels,
                "feedforward_channels": region.hidden_dim,
            },
        },
    )
    ShapeInferencePass().run(result)
    result.validate()
    return result


def analyze_bevformer_partitions(graph: Graph) -> BevFormerPartitionReport:
    """Place strict FFNs on Vortex and every other node on CUDA."""

    regions, rejected, shaped = _discover_with_rejections(graph)
    region_by_node = {
        node_id: region
        for region in regions
        for node_id in region.node_ids
    }
    rejection_by_scope: dict[str, str] = {}
    for node_id, reason in rejected:
        prefix = node_id.rsplit(".", 1)[0] if "." in node_id else node_id
        rejection_by_scope[prefix] = reason

    placements: list[BevFormerNodePlacement] = []
    for node in shaped.nodes:
        kind = classify_bevformer_node(node)
        region = region_by_node.get(node.id)
        if region is not None:
            placements.append(
                BevFormerNodePlacement(
                    node.id, "ffn", BEVFORMER_VORTEX_TARGET, region.region_id
                )
            )
            continue
        reason = None
        if kind == "ffn":
            prefix = node.id.rsplit(".", 1)[0] if "." in node.id else node.id
            reason = rejection_by_scope.get(prefix)
            if reason is None:
                reason = "not covered by a canonical BEVFormer FFN region"
        placements.append(
            BevFormerNodePlacement(
                node.id,
                kind,
                BEVFORMER_CUDA_TARGET,
                reason=reason,
            )
        )
    return BevFormerPartitionReport(
        shaped.name, regions, tuple(placements), rejected
    )


def place_bevformer_nodes(graph: Graph) -> tuple[BevFormerNodePlacement, ...]:
    """Convenience wrapper returning only per-node placements."""

    return analyze_bevformer_partitions(graph).placements


__all__ = [
    "BEVFORMER_CUDA_TARGET",
    "BEVFORMER_VORTEX_TARGET",
    "BevFormerFFNRegion",
    "BevFormerNodeKind",
    "BevFormerNodePlacement",
    "BevFormerPartitionReport",
    "analyze_bevformer_partitions",
    "classify_bevformer_node",
    "discover_bevformer_ffn_regions",
    "extract_bevformer_ffn_region",
    "place_bevformer_nodes",
]
