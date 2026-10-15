"""Canonical BEVFormer decoder deformable cross-attention lowering.

This module covers one inference call to the ``CustomMSDeformableAttention``
module in BEVFormer's decoder.  The compiler ABI is batch-first even though
the upstream tiny model enters the module as query-first; that layout adapter
belongs at the surrounding decoder boundary.

PyTorch ``Linear.weight`` tensors must be transposed offline to the canonical
``[input_channel, output_channel]`` convention used by this graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class DecoderCrossAttentionConfig:
    """One static specialization of decoder ``CustomMSDeformableAttention``."""

    batch: int = 1
    queries: int = 900
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    heads: int = 8
    sampling_points: int = 4

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "queries",
            "bev_h",
            "bev_w",
            "embed_dim",
            "heads",
            "sampling_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(
                    f"decoder cross attention {name} must be a positive integer"
                )
        if self.batch != 1:
            raise ValidationError("decoder cross attention currently requires batch=1")
        if self.embed_dim % self.heads:
            raise ValidationError(
                "decoder cross attention embed_dim must be divisible by heads"
            )
        if self.sampling_points > 64:
            raise ValidationError(
                "decoder cross attention supports at most 64 sampling points"
            )

    @property
    def levels(self) -> int:
        return 1

    @property
    def spatial_size(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads

    @property
    def spatial_shapes(self) -> tuple[tuple[int, int], ...]:
        return ((self.bev_h, self.bev_w),)


def decoder_cross_attention_weight_shapes(
    config: DecoderCrossAttentionConfig,
) -> dict[str, tuple[int, ...]]:
    """Return shapes for canonical, already-transposed linear parameters."""

    if not isinstance(config, DecoderCrossAttentionConfig):
        raise ValidationError("config must be a DecoderCrossAttentionConfig")
    channels = config.embed_dim
    offset_channels = (
        config.heads * config.levels * config.sampling_points * 2
    )
    attention_channels = config.heads * config.levels * config.sampling_points
    return {
        "value_weight": (channels, channels),
        "value_bias": (channels,),
        "offset_weight": (channels, offset_channels),
        "offset_bias": (offset_channels,),
        "attention_weight": (channels, attention_channels),
        "attention_bias": (attention_channels,),
        "output_weight": (channels, channels),
        "output_bias": (channels,),
    }


def _validated_weights(
    config: DecoderCrossAttentionConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = decoder_cross_attention_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError(
            "decoder cross attention weights must be a string-to-NumPy-array mapping"
        )
    if any(not isinstance(name, str) for name in weights):
        raise ValidationError("decoder cross attention weight names must be strings")
    if set(weights) != set(expected):
        raise ValidationError(
            "decoder cross attention weight names must be "
            f"{sorted(expected)}, got {sorted(weights)}"
        )

    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder cross attention weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder cross attention weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder cross attention weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()

    result["offset_normalizer"] = np.asarray(
        [config.bev_w, config.bev_h], dtype=np.float32
    ).reshape(1, 1, 1, config.levels, 1, 2)
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


def build_decoder_cross_attention_graph(
    weights: Mapping[str, np.ndarray],
    config: DecoderCrossAttentionConfig | None = None,
    *,
    name: str = "bevformer_decoder_cross_attention",
) -> Graph:
    """Lower one decoder deformable cross-attention inference call.

    ``query_pos`` affects sampling offsets and attention weights, while the
    final residual uses the original ``query``.  Dropout is an identity during
    inference.  The value tensor is the flattened 50x50 BEV memory for the
    default tiny specialization.
    """

    if config is None:
        config = DecoderCrossAttentionConfig()
    if not isinstance(config, DecoderCrossAttentionConfig):
        raise ValidationError("config must be a DecoderCrossAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder cross attention graph name must be non-empty")

    constants = _validated_weights(config, weights)
    batch = config.batch
    queries = config.queries
    spatial = config.spatial_size
    channels = config.embed_dim
    heads = config.heads
    levels = config.levels
    points = config.sampling_points
    head_dim = config.head_dim

    nodes = [
        _node(
            "query_position_add",
            "Add",
            ("query", "query_pos"),
            "positioned_query",
            None,
            "CustomMSDeformableAttention.forward.query_pos",
        ),
        _node(
            "value_projection",
            "MatMul",
            ("value", "value_weight"),
            "value_mm",
            None,
            "CustomMSDeformableAttention.forward.value_proj",
        ),
        _node(
            "value_bias_add",
            "Add",
            ("value_mm", "value_bias"),
            "value_projected",
            None,
            "CustomMSDeformableAttention.forward.value_proj",
        ),
        _node(
            "value_head_shape",
            "Reshape",
            ("value_projected",),
            "value_msda",
            {"shape": [batch, spatial, heads, head_dim]},
            "CustomMSDeformableAttention.forward.value_view",
        ),
        _node(
            "offset_projection",
            "MatMul",
            ("positioned_query", "offset_weight"),
            "offset_mm",
            None,
            "CustomMSDeformableAttention.forward.sampling_offsets",
        ),
        _node(
            "offset_bias_add",
            "Add",
            ("offset_mm", "offset_bias"),
            "offset_flat",
            None,
            "CustomMSDeformableAttention.forward.sampling_offsets",
        ),
        _node(
            "offset_shape",
            "Reshape",
            ("offset_flat",),
            "sampling_offsets",
            {"shape": [batch, queries, heads, levels, points, 2]},
            "CustomMSDeformableAttention.forward.offset_view",
        ),
        _node(
            "offset_normalize",
            "Div",
            ("sampling_offsets", "offset_normalizer"),
            "normalized_offsets",
            None,
            "CustomMSDeformableAttention.forward.offset_normalizer",
        ),
        _node(
            "reference_broadcast_shape",
            "Reshape",
            ("reference_points",),
            "reference_bq1l12",
            {"shape": [batch, queries, 1, levels, 1, 2]},
            "CustomMSDeformableAttention.forward.reference_points_2d",
        ),
        _node(
            "sampling_location_add",
            "Add",
            ("reference_bq1l12", "normalized_offsets"),
            "sampling_locations",
            None,
            "CustomMSDeformableAttention.forward.sampling_locations",
        ),
        _node(
            "attention_projection",
            "MatMul",
            ("positioned_query", "attention_weight"),
            "attention_mm",
            None,
            "CustomMSDeformableAttention.forward.attention_weights",
        ),
        _node(
            "attention_bias_add",
            "Add",
            ("attention_mm", "attention_bias"),
            "attention_flat",
            None,
            "CustomMSDeformableAttention.forward.attention_weights",
        ),
        _node(
            "attention_head_shape",
            "Reshape",
            ("attention_flat",),
            "attention_by_head",
            {"shape": [batch, queries, heads, levels * points]},
            "CustomMSDeformableAttention.forward.attention_view",
        ),
        _node(
            "attention_softmax",
            "Softmax",
            ("attention_by_head",),
            "attention_probability",
            {"axis": -1},
            "CustomMSDeformableAttention.forward.attention_softmax",
        ),
        _node(
            "attention_level_shape",
            "Reshape",
            ("attention_probability",),
            "attention_weights",
            {"shape": [batch, queries, heads, levels, points]},
            "CustomMSDeformableAttention.forward.attention_level_view",
        ),
        _node(
            "decoder_msda",
            "MSDeformableAttention",
            ("value_msda", "sampling_locations", "attention_weights"),
            "context",
            {"spatial_shapes": [[config.bev_h, config.bev_w]]},
            "CustomMSDeformableAttention.forward.ms_deformable_attention",
        ),
        _node(
            "output_projection",
            "MatMul",
            ("context", "output_weight"),
            "output_mm",
            None,
            "CustomMSDeformableAttention.forward.output_proj",
        ),
        _node(
            "output_bias_add",
            "Add",
            ("output_mm", "output_bias"),
            "projected",
            None,
            "CustomMSDeformableAttention.forward.output_proj",
        ),
        _node(
            "residual_add",
            "Add",
            ("projected", "query"),
            "output",
            None,
            "CustomMSDeformableAttention.forward.residual",
        ),
    ]

    graph = Graph(
        name=name,
        inputs={
            "query": TensorSpec((batch, queries, channels), "float32"),
            "query_pos": TensorSpec((batch, queries, channels), "float32"),
            "value": TensorSpec((batch, spatial, channels), "float32"),
            "reference_points": TensorSpec(
                (batch, queries, levels, 2), "float32"
            ),
        },
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-decoder-custom-msda-cross-attention",
            "source": "projects/mmdet3d_plugin/bevformer/modules/decoder.py",
            "scope": "single-CustomMSDeformableAttention-inference-call",
            "full_decoder_stack": False,
            "batch_first_abi": True,
            "batch": batch,
            "queries": queries,
            "embed_dim": channels,
            "heads": heads,
            "spatial_shapes": [[config.bev_h, config.bev_w]],
            "sampling_points": points,
            "reference_point_width": 2,
            "dropout": "inference-identity",
            "key_padding_mask": "unsupported-none-only",
            "excluded_stages": [
                "decoder-self-attention",
                "layer-normalization",
                "feed-forward-network",
                "reference-point-refinement",
                "remaining-decoder-layers",
            ],
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_decoder_cross_attention_inputs(
    graph: Graph,
    values: Mapping[str, np.ndarray],
) -> None:
    """Validate the static decoder ABI and normalized 2D references."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        actual = (
            sorted(repr(name) for name in values)
            if isinstance(values, Mapping)
            else type(values).__name__
        )
        raise ValidationError(
            f"decoder cross attention input names must be "
            f"{sorted(graph.inputs)}, got {actual}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder cross attention input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder cross attention input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder cross attention input {name!r} must be contiguous and finite"
            )

    references = values["reference_points"]
    if np.any(references < 0.0) or np.any(references > 1.0):
        raise ValidationError(
            "decoder cross attention reference_points must be normalized to [0, 1]"
        )


__all__ = [
    "DecoderCrossAttentionConfig",
    "build_decoder_cross_attention_graph",
    "decoder_cross_attention_weight_shapes",
    "validate_decoder_cross_attention_inputs",
]
