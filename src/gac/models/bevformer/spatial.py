"""Static upper-bound lowering for BEVFormer spatial cross attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec


@dataclass(frozen=True, slots=True)
class SpatialCrossAttentionConfig:
    batch: int = 1
    cameras: int = 6
    queries: int = 2500
    embed_dim: int = 256
    heads: int = 8
    spatial_shapes: tuple[tuple[int, int], ...] = ((15, 25),)
    sampling_points: int = 8
    z_anchors: int = 4

    def __post_init__(self) -> None:
        for name in (
            "batch", "cameras", "queries", "embed_dim", "heads",
            "sampling_points", "z_anchors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"spatial attention {name} must be positive")
        if self.embed_dim % self.heads:
            raise ValidationError("spatial attention embed_dim must be divisible by heads")
        if self.batch != 1:
            raise ValidationError("spatial attention currently requires batch=1")
        if self.sampling_points % self.z_anchors:
            raise ValidationError("sampling_points must be divisible by z_anchors")
        if not isinstance(self.spatial_shapes, tuple) or not self.spatial_shapes:
            raise ValidationError("spatial_shapes must be a non-empty tuple")
        for shape in self.spatial_shapes:
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
            ):
                raise ValidationError("spatial_shapes must contain positive (height, width) pairs")

    @property
    def levels(self) -> int:
        return len(self.spatial_shapes)

    @property
    def spatial_size(self) -> int:
        return sum(height * width for height, width in self.spatial_shapes)

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads


def _weight_shapes(config: SpatialCrossAttentionConfig) -> dict[str, tuple[int, ...]]:
    channels = config.embed_dim
    offsets = config.heads * config.levels * config.sampling_points * 2
    attention = config.heads * config.levels * config.sampling_points
    return {
        "value_weight": (channels, channels),
        "value_bias": (channels,),
        "offset_weight": (channels, offsets),
        "offset_bias": (offsets,),
        "attention_weight": (channels, attention),
        "attention_bias": (attention,),
        "output_weight": (channels, channels),
        "output_bias": (channels,),
    }


def _validated_weights(
    config: SpatialCrossAttentionConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = _weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError(
            "spatial attention weights must be a string-to-array mapping"
        )
    if any(not isinstance(key, str) for key in weights):
        raise ValidationError("spatial attention weight names must be strings")
    if set(weights) != set(expected):
        actual = sorted(weights)
        raise ValidationError(
            f"spatial attention weights must be {sorted(expected)}, got {actual}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"spatial attention weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"spatial attention weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    normalizer = np.asarray(
        [[width, height] for height, width in config.spatial_shapes],
        dtype=np.float32,
    ).reshape(1, 1, 1, config.levels, 1, 1, 2)
    result["offset_normalizer"] = normalizer
    return result


def build_spatial_cross_attention_graph(
    weights: Mapping[str, np.ndarray],
    config: SpatialCrossAttentionConfig | None = None,
    *,
    name: str = "bevformer_spatial_cross_attention",
) -> Graph:
    """Build the static all-query variant of BEVFormer spatial attention.

    ``visibility`` is float32 ``[B,Cam,Q,1]`` containing 0 or 1.  Each element
    says whether any Z anchor for that camera/query pair is visible.  Keeping
    all queries gives the compiler a fixed memory bound; masked camera results
    are removed before aggregation, matching the reference gather/scatter
    result.  ``query_pos`` is used only for sampling-offset and attention
    prediction.  The residual path keeps the original ``query``, matching
    MMCV's ``SpatialCrossAttention.forward`` contract.
    """

    config = config or SpatialCrossAttentionConfig()
    if not isinstance(config, SpatialCrossAttentionConfig):
        raise ValidationError("config must be a SpatialCrossAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("spatial attention graph name must be non-empty")
    constants = _validated_weights(config, weights)
    b, cameras, q, channels = (
        config.batch, config.cameras, config.queries, config.embed_dim
    )
    heads, levels, points = config.heads, config.levels, config.sampling_points
    anchors, head_dim = config.z_anchors, config.head_dim
    n = b * cameras
    points_per_anchor = points // anchors
    spatial = config.spatial_size
    nodes = [
        Node(
            "query_position_add",
            "Add",
            ("query", "query_pos"),
            ("positioned_query",),
            source_nodes=("SpatialCrossAttention.forward.query_pos",),
        ),
        Node("query_camera_axis", "Reshape", ("positioned_query",), ("query_b1qc",), {"shape": [b, 1, q, channels]}),
        Node("query_camera_expand", "Expand", ("query_b1qc",), ("query_bcqc",), {"shape": [b, cameras, q, channels]}),
        Node("query_flatten_cameras", "Reshape", ("query_bcqc",), ("camera_query",), {"shape": [n, q, channels]}),
        Node("value_flatten_cameras", "Reshape", ("value",), ("camera_value",), {"shape": [n, spatial, channels]}),
        Node("value_projection", "MatMul", ("camera_value", "value_weight"), ("value_mm",)),
        Node("value_bias", "Add", ("value_mm", "value_bias"), ("value_projected",)),
        Node("value_heads", "Reshape", ("value_projected",), ("value_msda",), {"shape": [n, spatial, heads, head_dim]}),
        Node("offset_projection", "MatMul", ("camera_query", "offset_weight"), ("offset_mm",)),
        Node("offset_bias", "Add", ("offset_mm", "offset_bias"), ("offset_flat",)),
        Node("offset_shape", "Reshape", ("offset_flat",), ("offsets",), {"shape": [n, q, heads, levels, points_per_anchor, anchors, 2]}),
        Node("offset_normalize", "Div", ("offsets", "offset_normalizer"), ("normalized_offsets",)),
        Node("reference_flatten_cameras", "Reshape", ("reference_points",), ("reference_nqa2",), {"shape": [n, q, anchors, 2]}),
        Node("reference_axes", "Reshape", ("reference_nqa2",), ("reference_nq111a2",), {"shape": [n, q, 1, 1, 1, anchors, 2]}),
        Node("reference_expand", "Expand", ("reference_nq111a2",), ("reference_expanded",), {"shape": [n, q, heads, levels, points_per_anchor, anchors, 2]}),
        Node("location_add", "Add", ("reference_expanded", "normalized_offsets"), ("locations_split",)),
        Node("location_shape", "Reshape", ("locations_split",), ("sampling_locations",), {"shape": [n, q, heads, levels, points, 2]}),
        Node("attention_projection", "MatMul", ("camera_query", "attention_weight"), ("attention_mm",)),
        Node("attention_bias", "Add", ("attention_mm", "attention_bias"), ("attention_flat",)),
        Node("attention_head_shape", "Reshape", ("attention_flat",), ("attention_by_head",), {"shape": [n, q, heads, levels * points]}),
        Node("attention_softmax", "Softmax", ("attention_by_head",), ("attention_probability",), {"axis": -1}),
        Node("attention_shape", "Reshape", ("attention_probability",), ("attention_weights",), {"shape": [n, q, heads, levels, points]}),
        Node(
            "spatial_msda",
            "MSDeformableAttention",
            ("value_msda", "sampling_locations", "attention_weights"),
            ("camera_context",),
            {"spatial_shapes": [list(shape) for shape in config.spatial_shapes]},
            source_nodes=("SpatialCrossAttention.deformable_attention",),
        ),
        Node("camera_context_shape", "Reshape", ("camera_context",), ("context_bcqc",), {"shape": [b, cameras, q, channels]}),
        Node("visibility_apply", "Mul", ("context_bcqc", "visibility"), ("masked_context",)),
        Node("camera_sum", "ReduceSum", ("masked_context",), ("context_sum",), {"axes": [1], "keepdims": False}),
        Node("visibility_sum", "ReduceSum", ("visibility",), ("visible_count",), {"axes": [1], "keepdims": False}),
        Node("visibility_clamp", "Clamp", ("visible_count",), ("safe_count",), {"min": 1.0}),
        Node("camera_average", "Div", ("context_sum", "safe_count"), ("averaged_context",)),
        Node("output_projection", "MatMul", ("averaged_context", "output_weight"), ("output_mm",)),
        Node("output_bias", "Add", ("output_mm", "output_bias"), ("projected",)),
        Node("residual_add", "Add", ("projected", "query"), ("output",)),
    ]
    return Graph(
        name=name,
        inputs={
            "query": TensorSpec((b, q, channels), "float32"),
            "query_pos": TensorSpec((b, q, channels), "float32"),
            "value": TensorSpec((b, cameras, spatial, channels), "float32"),
            "reference_points": TensorSpec((b, cameras, q, anchors, 2), "float32"),
            "visibility": TensorSpec((b, cameras, q, 1), "float32"),
        },
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-spatial-cross-attention-static-upper-bound",
            "query_compaction": "all-queries-plus-visibility-mask",
            "batch_size": b,
            "camera_count": cameras,
            "query_count": q,
            "embed_dim": channels,
            "heads": heads,
            "spatial_shapes": [list(shape) for shape in config.spatial_shapes],
            "sampling_points": points,
            "z_anchors": anchors,
            "visibility_semantics": "one-iff-any-z-anchor-is-visible",
            "query_position": "explicit-input-residual-excludes-position",
            "official_tiny_query_pos": "all-zero (encoder passes None)",
        },
    )


def validate_spatial_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the extra binary-mask contract before a runtime invocation."""

    if not isinstance(values, Mapping):
        raise ValidationError("spatial attention inputs must be a mapping")
    if set(values) != set(graph.inputs):
        raise ValidationError("spatial attention input names do not match the graph")
    checked: dict[str, np.ndarray] = {}
    for name, spec in graph.inputs.items():
        value = np.asarray(values[name])
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(f"spatial input {name!r} does not match {spec.shape}/{spec.dtype}")
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(f"spatial input {name!r} must be contiguous and finite")
        checked[name] = value
    visibility = checked["visibility"]
    if not np.logical_or(visibility == 0, visibility == 1).all():
        raise ValidationError("visibility must contain only 0 or 1")


__all__ = [
    "SpatialCrossAttentionConfig",
    "build_spatial_cross_attention_graph",
    "validate_spatial_inputs",
]
