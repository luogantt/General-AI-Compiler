"""Static BEVFormer Temporal Self Attention lowering.

This module maps the inference path in BEVFormer's
``temporal_self_attention.py`` to canonical tensor operations.  The supported
contract is deliberately fixed to batch one and a two-entry BEV queue ordered
as ``[previous frame, current frame]``.  The graph keeps the previous and
current reference points separate at its ABI so that the temporal state
transition is explicit.

PyTorch ``Linear.weight`` tensors must be transposed offline to the canonical
``[input_channel, output_channel]`` convention used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class TemporalSelfAttentionConfig:
    """One fully static Temporal Self Attention specialization."""

    batch: int = 1
    queue_length: int = 2
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    heads: int = 8
    sampling_points: int = 4

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "queue_length",
            "bev_h",
            "bev_w",
            "embed_dim",
            "heads",
            "sampling_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"temporal attention {name} must be a positive integer")
        if self.batch != 1:
            raise ValidationError("temporal attention currently requires batch=1")
        if self.queue_length != 2:
            raise ValidationError("temporal attention currently requires queue_length=2")
        if self.embed_dim % self.heads:
            raise ValidationError("temporal attention embed_dim must be divisible by heads")

    @property
    def queries(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads

    @property
    def spatial_shapes(self) -> tuple[tuple[int, int], ...]:
        return ((self.bev_h, self.bev_w),)

    @property
    def levels(self) -> int:
        return 1


def temporal_weight_shapes(
    config: TemporalSelfAttentionConfig,
) -> dict[str, tuple[int, ...]]:
    """Return canonical (already-transposed) parameter shapes."""

    if not isinstance(config, TemporalSelfAttentionConfig):
        raise ValidationError("config must be a TemporalSelfAttentionConfig")
    channels = config.embed_dim
    query_channels = channels * config.queue_length
    offset_channels = (
        config.queue_length
        * config.heads
        * config.levels
        * config.sampling_points
        * 2
    )
    attention_channels = (
        config.queue_length
        * config.heads
        * config.levels
        * config.sampling_points
    )
    return {
        "value_weight": (channels, channels),
        "value_bias": (channels,),
        "offset_weight": (query_channels, offset_channels),
        "offset_bias": (offset_channels,),
        "attention_weight": (query_channels, attention_channels),
        "attention_bias": (attention_channels,),
        "output_weight": (channels, channels),
        "output_bias": (channels,),
    }


def _validated_weights(
    config: TemporalSelfAttentionConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = temporal_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("temporal attention weights must be a mapping")
    if set(weights) != set(expected):
        actual = sorted(repr(name) for name in weights)
        raise ValidationError(
            f"temporal attention weight names must be {sorted(expected)}, got {actual}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"temporal attention weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"temporal attention weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    result["offset_normalizer"] = np.asarray(
        [config.bev_w, config.bev_h], dtype=np.float32
    ).reshape(1, 1, 1, 1, 1, 2)
    return result


def _node(
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


def build_temporal_self_attention_graph(
    weights: Mapping[str, np.ndarray],
    config: TemporalSelfAttentionConfig | None = None,
    *,
    name: str = "bevformer_temporal_self_attention",
) -> Graph:
    """Lower BEVFormer's batch-first inference path to canonical operations.

    Inputs are the current layer ``query``, the explicit two-entry
    ``temporal_value`` queue, positional encoding, and one normalized
    reference-point tensor for each queue entry.  Keeping the value queue
    explicit is required for layers after the first: with history enabled,
    ``BEVFormerEncoder.forward`` constructs ``[previous BEV, initial current
    BEV]`` once before the encoder loop while the layer query keeps changing.
    """

    config = config or TemporalSelfAttentionConfig()
    if not isinstance(config, TemporalSelfAttentionConfig):
        raise ValidationError("config must be a TemporalSelfAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("temporal attention graph name must be non-empty")
    constants = _validated_weights(config, weights)
    b = config.batch
    queue = config.queue_length
    queries = config.queries
    channels = config.embed_dim
    heads = config.heads
    levels = config.levels
    points = config.sampling_points
    head_dim = config.head_dim
    queue_batch = b * queue
    nodes = [
        _node(
            "query_position_add",
            "Add",
            ("query", "query_pos"),
            "positioned_query",
            None,
            "TemporalSelfAttention.forward.query_pos",
        ),
        _node(
            "previous_value_slice",
            "Slice",
            ("temporal_value",),
            "previous_value",
            {"starts": [0], "ends": [b], "axes": [0], "steps": [1]},
            "TemporalSelfAttention.forward.value_first_batch",
        ),
        _node(
            "temporal_query_concat",
            "Concat",
            ("previous_value", "positioned_query"),
            "temporal_query",
            {"axis": -1},
            "TemporalSelfAttention.forward.concat_prev_query",
        ),
        _node(
            "value_projection",
            "MatMul",
            ("temporal_value", "value_weight"),
            "value_mm",
            None,
            "TemporalSelfAttention.forward.value_proj",
        ),
        _node(
            "value_bias_add",
            "Add",
            ("value_mm", "value_bias"),
            "value_projected",
            None,
            "TemporalSelfAttention.forward.value_proj",
        ),
        _node(
            "value_head_shape",
            "Reshape",
            ("value_projected",),
            "value_msda",
            {"shape": [queue_batch, queries, heads, head_dim]},
            "TemporalSelfAttention.forward.value_reshape",
        ),
        _node(
            "offset_projection",
            "MatMul",
            ("temporal_query", "offset_weight"),
            "offset_mm",
            None,
            "TemporalSelfAttention.forward.sampling_offsets",
        ),
        _node(
            "offset_bias_add",
            "Add",
            ("offset_mm", "offset_bias"),
            "offset_flat",
            None,
            "TemporalSelfAttention.forward.sampling_offsets",
        ),
        _node(
            "offset_queue_shape",
            "Reshape",
            ("offset_flat",),
            "offset_bqhqlp2",
            {"shape": [b, queries, heads, queue, levels, points, 2]},
            "TemporalSelfAttention.forward.offset_view",
        ),
        _node(
            "offset_queue_transpose",
            "Transpose",
            ("offset_bqhqlp2",),
            "offset_bqqhlp2",
            {"perm": [0, 3, 1, 2, 4, 5, 6]},
            "TemporalSelfAttention.forward.offset_permute",
        ),
        _node(
            "offset_queue_flatten",
            "Reshape",
            ("offset_bqqhlp2",),
            "sampling_offsets",
            {"shape": [queue_batch, queries, heads, levels, points, 2]},
            "TemporalSelfAttention.forward.offset_queue_flatten",
        ),
        _node(
            "offset_normalize",
            "Div",
            ("sampling_offsets", "offset_normalizer"),
            "normalized_offsets",
            None,
            "TemporalSelfAttention.forward.offset_normalizer",
        ),
        _node(
            "reference_queue_concat",
            "Concat",
            ("prev_reference_points", "reference_points"),
            "queue_reference",
            {"axis": 0},
            "BEVFormerEncoder.forward.hybrid_ref_2d",
        ),
        _node(
            "reference_broadcast_shape",
            "Reshape",
            ("queue_reference",),
            "reference_bq1l12",
            {"shape": [queue_batch, queries, 1, levels, 1, 2]},
            "TemporalSelfAttention.forward.reference_points",
        ),
        _node(
            "sampling_location_add",
            "Add",
            ("reference_bq1l12", "normalized_offsets"),
            "sampling_locations",
            None,
            "TemporalSelfAttention.forward.sampling_locations",
        ),
        _node(
            "attention_projection",
            "MatMul",
            ("temporal_query", "attention_weight"),
            "attention_mm",
            None,
            "TemporalSelfAttention.forward.attention_weights",
        ),
        _node(
            "attention_bias_add",
            "Add",
            ("attention_mm", "attention_bias"),
            "attention_flat",
            None,
            "TemporalSelfAttention.forward.attention_weights",
        ),
        _node(
            "attention_queue_shape",
            "Reshape",
            ("attention_flat",),
            "attention_bqhqk",
            {"shape": [b, queries, heads, queue, levels * points]},
            "TemporalSelfAttention.forward.attention_view",
        ),
        _node(
            "attention_softmax",
            "Softmax",
            ("attention_bqhqk",),
            "attention_probability",
            {"axis": -1},
            "TemporalSelfAttention.forward.attention_softmax",
        ),
        _node(
            "attention_level_shape",
            "Reshape",
            ("attention_probability",),
            "attention_bqhqlp",
            {"shape": [b, queries, heads, queue, levels, points]},
            "TemporalSelfAttention.forward.attention_level_view",
        ),
        _node(
            "attention_queue_transpose",
            "Transpose",
            ("attention_bqhqlp",),
            "attention_bqqhlp",
            {"perm": [0, 3, 1, 2, 4, 5]},
            "TemporalSelfAttention.forward.attention_permute",
        ),
        _node(
            "attention_queue_flatten",
            "Reshape",
            ("attention_bqqhlp",),
            "attention_weights",
            {"shape": [queue_batch, queries, heads, levels, points]},
            "TemporalSelfAttention.forward.attention_queue_flatten",
        ),
        _node(
            "temporal_msda",
            "MSDeformableAttention",
            ("value_msda", "sampling_locations", "attention_weights"),
            "queue_context",
            {"spatial_shapes": [[config.bev_h, config.bev_w]]},
            "TemporalSelfAttention.forward.ms_deformable_attention",
        ),
        _node(
            "context_queue_shape",
            "Reshape",
            ("queue_context",),
            "context_bqqc",
            {"shape": [b, queue, queries, channels]},
            "TemporalSelfAttention.forward.queue_view",
        ),
        _node(
            "context_queue_mean",
            "ReduceMean",
            ("context_bqqc",),
            "context",
            {"axes": [1], "keepdims": False},
            "TemporalSelfAttention.forward.mean_queue",
        ),
        _node(
            "output_projection",
            "MatMul",
            ("context", "output_weight"),
            "output_mm",
            None,
            "TemporalSelfAttention.forward.output_proj",
        ),
        _node(
            "output_bias_add",
            "Add",
            ("output_mm", "output_bias"),
            "projected",
            None,
            "TemporalSelfAttention.forward.output_proj",
        ),
        _node(
            "residual_add",
            "Add",
            ("projected", "query"),
            "output",
            None,
            "TemporalSelfAttention.forward.residual",
        ),
    ]
    graph = Graph(
        name=name,
        inputs={
            "query": TensorSpec((b, queries, channels), "float32"),
            "temporal_value": TensorSpec(
                (queue_batch, queries, channels), "float32"
            ),
            "query_pos": TensorSpec((b, queries, channels), "float32"),
            "prev_reference_points": TensorSpec((b, queries, levels, 2), "float32"),
            "reference_points": TensorSpec((b, queries, levels, 2), "float32"),
        },
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-temporal-self-attention-static",
            "source": "projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py",
            "batch": b,
            "queue_length": queue,
            "queue_order": ["previous", "current"],
            "temporal_value_contract": (
                "history path: [previous_bev, initial_current_bev]; cold path: "
                "[current_layer_query, current_layer_query]"
            ),
            "spatial_shapes": [[config.bev_h, config.bev_w]],
            "dropout": "inference-identity",
            "key_padding_mask": "unsupported-none-only",
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_temporal_inputs(
    graph: Graph,
    values: Mapping[str, np.ndarray],
) -> None:
    """Validate the static ABI and normalized-reference-point contract."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        actual = (
            sorted(repr(name) for name in values)
            if isinstance(values, Mapping)
            else type(values).__name__
        )
        raise ValidationError(
            f"temporal input names must be {sorted(graph.inputs)}, got {actual}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"temporal input {name!r} must be a NumPy ndarray")
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"temporal input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"temporal input {name!r} must be contiguous and finite"
            )
    for name in ("prev_reference_points", "reference_points"):
        value = values[name]
        if np.any(value < 0.0) or np.any(value > 1.0):
            raise ValidationError(f"temporal input {name!r} must be normalized to [0, 1]")


__all__ = [
    "TemporalSelfAttentionConfig",
    "build_temporal_self_attention_graph",
    "temporal_weight_shapes",
    "validate_temporal_inputs",
]
