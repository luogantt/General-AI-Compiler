"""Composition of one static BEVFormer encoder layer for inference.

The graph contains one TemporalSelfAttention module, one
SpatialCrossAttention module, the three post-normalization steps, and the
two-linear-layer FFN used by BEVFormer-tiny.  Camera reference projection is
also composed into the graph from a ``lidar2img`` input.  This is one encoder
layer only; stacking the three tiny-model layers remains an outer concern.

All linear weights use the compiler's canonical
``[input_channel, output_channel]`` layout.  PyTorch linear weights therefore
need one offline transpose before being supplied here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from .bevformer_geometry import BevProjectionConfig, build_bev_projection_graph
from .bevformer_spatial import (
    SpatialCrossAttentionConfig,
    build_spatial_cross_attention_graph,
)
from .bevformer_temporal import (
    TemporalSelfAttentionConfig,
    build_temporal_self_attention_graph,
    temporal_weight_shapes,
)
from .compose import GraphComposer
from .errors import ValidationError
from .ir import Graph, Node, TensorSpec


@dataclass(frozen=True, slots=True)
class BevFormerEncoderLayerConfig:
    """Mutually consistent static profiles for one encoder layer."""

    temporal: TemporalSelfAttentionConfig = field(
        default_factory=TemporalSelfAttentionConfig
    )
    spatial: SpatialCrossAttentionConfig = field(
        default_factory=SpatialCrossAttentionConfig
    )
    projection: BevProjectionConfig = field(default_factory=BevProjectionConfig)
    ffn_hidden_dim: int = 512
    norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.temporal, TemporalSelfAttentionConfig):
            raise ValidationError(
                "encoder layer temporal must be a TemporalSelfAttentionConfig"
            )
        if not isinstance(self.spatial, SpatialCrossAttentionConfig):
            raise ValidationError(
                "encoder layer spatial must be a SpatialCrossAttentionConfig"
            )
        if not isinstance(self.projection, BevProjectionConfig):
            raise ValidationError(
                "encoder layer projection must be a BevProjectionConfig"
            )
        if (
            isinstance(self.ffn_hidden_dim, bool)
            or not isinstance(self.ffn_hidden_dim, int)
            or self.ffn_hidden_dim <= 0
        ):
            raise ValidationError(
                "encoder layer ffn_hidden_dim must be a positive integer"
            )
        if (
            isinstance(self.norm_epsilon, bool)
            or not isinstance(self.norm_epsilon, (int, float))
            or not math.isfinite(float(self.norm_epsilon))
            or float(self.norm_epsilon) <= 0
            or np.float32(self.norm_epsilon) == 0
        ):
            raise ValidationError(
                "encoder layer norm_epsilon must be positive finite float32"
            )

        temporal = self.temporal
        spatial = self.spatial
        projection = self.projection
        mismatches: list[str] = []
        if temporal.batch != spatial.batch or temporal.batch != projection.batch:
            mismatches.append("batch")
        if temporal.queries != spatial.queries or temporal.queries != projection.queries:
            mismatches.append("queries/bev_h/bev_w")
        if temporal.embed_dim != spatial.embed_dim:
            mismatches.append("embed_dim")
        if temporal.heads != spatial.heads:
            mismatches.append("heads")
        if temporal.bev_h != projection.bev_h or temporal.bev_w != projection.bev_w:
            mismatches.append("temporal/projection bev_h/bev_w")
        if spatial.cameras != projection.cameras:
            mismatches.append("cameras")
        if spatial.z_anchors != projection.z_anchors:
            mismatches.append("z_anchors")
        if mismatches:
            raise ValidationError(
                "encoder layer component profiles disagree on: "
                + ", ".join(mismatches)
            )

    @property
    def batch(self) -> int:
        return self.temporal.batch

    @property
    def queries(self) -> int:
        return self.temporal.queries

    @property
    def embed_dim(self) -> int:
        return self.temporal.embed_dim


def encoder_layer_weight_shapes(
    config: BevFormerEncoderLayerConfig,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return the exact canonical tensor inventory for all weighted stages."""

    if not isinstance(config, BevFormerEncoderLayerConfig):
        raise ValidationError("config must be a BevFormerEncoderLayerConfig")
    channels = config.embed_dim
    spatial = config.spatial
    spatial_offsets = (
        spatial.heads
        * spatial.levels
        * spatial.sampling_points
        * 2
    )
    spatial_attention = (
        spatial.heads * spatial.levels * spatial.sampling_points
    )
    return {
        "temporal": temporal_weight_shapes(config.temporal),
        "spatial": {
            "value_weight": (channels, channels),
            "value_bias": (channels,),
            "offset_weight": (channels, spatial_offsets),
            "offset_bias": (spatial_offsets,),
            "attention_weight": (channels, spatial_attention),
            "attention_bias": (spatial_attention,),
            "output_weight": (channels, channels),
            "output_bias": (channels,),
        },
        "layer": {
            "norm0_weight": (channels,),
            "norm0_bias": (channels,),
            "norm1_weight": (channels,),
            "norm1_bias": (channels,),
            "ffn_fc1_weight": (channels, config.ffn_hidden_dim),
            "ffn_fc1_bias": (config.ffn_hidden_dim,),
            "ffn_fc2_weight": (config.ffn_hidden_dim, channels),
            "ffn_fc2_bias": (channels,),
            "norm2_weight": (channels,),
            "norm2_bias": (channels,),
        },
    }


def _validate_weight_group(
    scope: str,
    weights: Mapping[str, np.ndarray],
    expected: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError(
            f"encoder layer {scope} weights must be a string-to-NumPy-array mapping"
        )
    if any(not isinstance(name, str) for name in weights):
        raise ValidationError(f"encoder layer {scope} weight names must be strings")
    if set(weights) != set(expected):
        raise ValidationError(
            f"encoder layer {scope} weight names must be {sorted(expected)}, "
            f"got {sorted(weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"encoder layer {scope} weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"encoder layer {scope} weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"encoder layer {scope} weight {name!r} must be contiguous and finite"
            )
        result[name] = value
    return result


def _layer_node(
    node_id: str,
    op: str,
    inputs: tuple[str, ...],
    output: str,
    attrs: Mapping[str, object] | None,
    source: str,
) -> Node:
    return Node(
        id=node_id,
        op=op,
        inputs=inputs,
        outputs=(output,),
        attrs=dict(attrs or {}),
        source_nodes=(source,),
    )


def build_bevformer_encoder_layer_graph(
    temporal_weights: Mapping[str, np.ndarray],
    spatial_weights: Mapping[str, np.ndarray],
    layer_weights: Mapping[str, np.ndarray],
    config: BevFormerEncoderLayerConfig | None = None,
    *,
    name: str = "bevformer_encoder_layer",
) -> Graph:
    """Build one post-normalized BEVFormer encoder layer plus geometry.

    The data path is ``TemporalSelfAttention -> norm0 ->
    SpatialCrossAttention -> norm1 -> FFN residual -> norm2``.  The projection
    branch consumes ``lidar2img`` and supplies static-upper-bound reference
    points and visibility to spatial attention.
    """

    if config is None:
        config = BevFormerEncoderLayerConfig()
    if not isinstance(config, BevFormerEncoderLayerConfig):
        raise ValidationError("config must be a BevFormerEncoderLayerConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("encoder layer graph name must be non-empty")

    shapes = encoder_layer_weight_shapes(config)
    temporal_weights = _validate_weight_group(
        "temporal", temporal_weights, shapes["temporal"]
    )
    spatial_weights = _validate_weight_group(
        "spatial", spatial_weights, shapes["spatial"]
    )
    layer_weights = _validate_weight_group(
        "layer", layer_weights, shapes["layer"]
    )

    temporal_graph = build_temporal_self_attention_graph(
        temporal_weights, config.temporal
    )
    spatial_graph = build_spatial_cross_attention_graph(
        spatial_weights, config.spatial
    )
    projection_graph = build_bev_projection_graph(config.projection)

    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    levels = config.temporal.levels
    spatial = config.spatial
    projection = config.projection
    composer = GraphComposer(
        name,
        {
            "query": TensorSpec((batch, queries, channels), "float32"),
            "temporal_value": TensorSpec(
                (
                    batch * config.temporal.queue_length,
                    queries,
                    channels,
                ),
                "float32",
            ),
            "query_pos": TensorSpec((batch, queries, channels), "float32"),
            "spatial_query_pos": TensorSpec(
                (batch, queries, channels), "float32"
            ),
            "prev_reference_points": TensorSpec(
                (batch, queries, levels, 2), "float32"
            ),
            "reference_points": TensorSpec(
                (batch, queries, levels, 2), "float32"
            ),
            "image_value": TensorSpec(
                (
                    batch,
                    spatial.cameras,
                    spatial.spatial_size,
                    channels,
                ),
                "float32",
            ),
            "lidar2img": TensorSpec(
                (batch, projection.cameras, 4, 4), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-single-encoder-layer-static",
            "scope": "one-BEVFormerLayer-inference-call",
            "full_encoder_stack": False,
            "operation_order": [
                "temporal-self-attention",
                "norm0",
                "spatial-cross-attention",
                "norm1",
                "ffn-residual",
                "norm2",
            ],
            "geometry": "in-graph-lidar2img-projection",
            "dropout": "inference-identity",
            "temporal_value_contract": (
                "queue axis is [previous BEV, initial current BEV] and is supplied "
                "explicitly by the caller"
            ),
            "position_contract": {
                "query_pos": "temporal-self-attention-bev-pos",
                "spatial_query_pos": (
                    "explicit-spatial-position; bind zeros for official tiny encoder"
                ),
            },
            "profile": {
                "batch": batch,
                "queries": queries,
                "embed_dim": channels,
                "heads": config.temporal.heads,
                "cameras": spatial.cameras,
                "bev_hw": [config.temporal.bev_h, config.temporal.bev_w],
                "feature_shapes": [list(shape) for shape in spatial.spatial_shapes],
                "z_anchors": spatial.z_anchors,
                "ffn_hidden_dim": config.ffn_hidden_dim,
            },
        },
    )

    projected = composer.inline(
        projection_graph,
        prefix="geometry",
        bindings={"lidar2img": "lidar2img"},
        output_bindings={
            "reference_points": "camera_reference_points",
            "visibility": "camera_visibility",
        },
    )
    temporal = composer.inline(
        temporal_graph,
        prefix="temporal",
        bindings={
            "query": "query",
            "temporal_value": "temporal_value",
            "query_pos": "query_pos",
            "prev_reference_points": "prev_reference_points",
            "reference_points": "reference_points",
        },
        output_bindings={"output": "temporal_output"},
    )

    for weight_name, value in layer_weights.items():
        composer.add_constant(f"layer.{weight_name}", value)

    epsilon = float(config.norm_epsilon)
    composer.add_node(
        _layer_node(
            "layer.norm0",
            "LayerNorm",
            (
                temporal["output"],
                "layer.norm0_weight",
                "layer.norm0_bias",
            ),
            "norm0_output",
            {"axis": -1, "epsilon": epsilon},
            "BEVFormerLayer.forward.norm0",
        )
    )

    spatial_output = composer.inline(
        spatial_graph,
        prefix="spatial",
        bindings={
            "query": "norm0_output",
            "query_pos": "spatial_query_pos",
            "value": "image_value",
            "reference_points": projected["reference_points"],
            "visibility": projected["visibility"],
        },
        output_bindings={"output": "spatial_output"},
    )
    composer.add_node(
        _layer_node(
            "layer.norm1",
            "LayerNorm",
            (
                spatial_output["output"],
                "layer.norm1_weight",
                "layer.norm1_bias",
            ),
            "norm1_output",
            {"axis": -1, "epsilon": epsilon},
            "BEVFormerLayer.forward.norm1",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_fc1",
            "MatMul",
            ("norm1_output", "layer.ffn_fc1_weight"),
            "ffn_fc1_mm",
            None,
            "FFN.layers.0.linear",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_fc1_bias",
            "Add",
            ("ffn_fc1_mm", "layer.ffn_fc1_bias"),
            "ffn_fc1",
            None,
            "FFN.layers.0.linear",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_relu",
            "ReLU",
            ("ffn_fc1",),
            "ffn_hidden",
            None,
            "FFN.layers.0.relu",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_fc2",
            "MatMul",
            ("ffn_hidden", "layer.ffn_fc2_weight"),
            "ffn_fc2_mm",
            None,
            "FFN.layers.1.linear",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_fc2_bias",
            "Add",
            ("ffn_fc2_mm", "layer.ffn_fc2_bias"),
            "ffn_projected",
            None,
            "FFN.layers.1.linear",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.ffn_residual",
            "Add",
            ("ffn_projected", "norm1_output"),
            "ffn_residual",
            None,
            "FFN.forward.residual",
        )
    )
    composer.add_node(
        _layer_node(
            "layer.norm2",
            "LayerNorm",
            (
                "ffn_residual",
                "layer.norm2_weight",
                "layer.norm2_bias",
            ),
            "output",
            {"axis": -1, "epsilon": epsilon},
            "BEVFormerLayer.forward.norm2",
        )
    )
    return composer.build(
        ("output",),
        metadata={
            "component_graphs": {
                "geometry": projection_graph.metadata.get("component"),
                "temporal": temporal_graph.metadata.get("kind"),
                "spatial": spatial_graph.metadata.get("kind"),
            }
        },
    )


def validate_encoder_layer_inputs(
    graph: Graph,
    values: Mapping[str, np.ndarray],
) -> None:
    """Validate the composed static ABI before reference or device execution."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        actual = (
            sorted(repr(name) for name in values)
            if isinstance(values, Mapping)
            else type(values).__name__
        )
        raise ValidationError(
            f"encoder layer input names must be {sorted(graph.inputs)}, got {actual}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"encoder layer input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"encoder layer input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"encoder layer input {name!r} must be contiguous and finite"
            )
    for name in ("prev_reference_points", "reference_points"):
        reference = values[name]
        if np.any(reference < 0.0) or np.any(reference > 1.0):
            raise ValidationError(
                f"encoder layer input {name!r} must be normalized to [0, 1]"
            )


__all__ = [
    "BevFormerEncoderLayerConfig",
    "build_bevformer_encoder_layer_graph",
    "encoder_layer_weight_shapes",
    "validate_encoder_layer_inputs",
]
