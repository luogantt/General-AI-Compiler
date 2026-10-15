"""Composition of one static BEVFormer decoder transformer layer.

The graph is exactly one inference layer with operation order
``self_attn, norm0, cross_attn, norm1, ffn, norm2``.  It composes the existing
decoder self-attention (including norm0), deformable cross-attention, and FFN
(including norm2) graphs, with norm1 added at the layer boundary.  Reference
point refinement and the remaining decoder layers are outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import numpy as np

from gac.models.bevformer.decoder import (
    DecoderCrossAttentionConfig,
    build_decoder_cross_attention_graph,
    decoder_cross_attention_weight_shapes,
)
from gac.models.bevformer.decoder_self import (
    DecoderSelfAttentionConfig,
    build_decoder_self_attention_graph,
    decoder_self_attention_weight_shapes,
)
from gac.models.bevformer.ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec


def _default_ffn_config() -> BEVFormerFFNConfig:
    return BEVFormerFFNConfig(bev_h=1, bev_w=900)


@dataclass(frozen=True, slots=True)
class BevFormerDecoderLayerConfig:
    """Mutually consistent static profiles for one decoder layer."""

    self_attention: DecoderSelfAttentionConfig = field(
        default_factory=DecoderSelfAttentionConfig
    )
    cross_attention: DecoderCrossAttentionConfig = field(
        default_factory=DecoderCrossAttentionConfig
    )
    ffn: BEVFormerFFNConfig = field(default_factory=_default_ffn_config)
    norm1_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.self_attention, DecoderSelfAttentionConfig):
            raise ValidationError(
                "decoder layer self_attention must be a DecoderSelfAttentionConfig"
            )
        if not isinstance(self.cross_attention, DecoderCrossAttentionConfig):
            raise ValidationError(
                "decoder layer cross_attention must be a DecoderCrossAttentionConfig"
            )
        if not isinstance(self.ffn, BEVFormerFFNConfig):
            raise ValidationError("decoder layer ffn must be a BEVFormerFFNConfig")
        if not self.self_attention.post_layer_norm:
            raise ValidationError("decoder layer self_attention must include norm0")
        if not self.ffn.post_layer_norm:
            raise ValidationError("decoder layer ffn must include norm2")

        mismatches: list[str] = []
        if not (
            self.self_attention.batch
            == self.cross_attention.batch
            == self.ffn.batch
        ):
            mismatches.append("batch")
        if not (
            self.self_attention.queries
            == self.cross_attention.queries
            == self.ffn.queries
        ):
            mismatches.append("queries")
        if not (
            self.self_attention.embed_dim
            == self.cross_attention.embed_dim
            == self.ffn.embed_dim
        ):
            mismatches.append("embed_dim")
        if self.self_attention.heads != self.cross_attention.heads:
            mismatches.append("heads")
        if mismatches:
            raise ValidationError(
                "decoder layer component profiles disagree on: "
                + ", ".join(mismatches)
            )

        epsilon = self.norm1_epsilon
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
            or not np.isfinite(np.float32(epsilon))
            or np.float32(epsilon) == 0.0
        ):
            raise ValidationError(
                "decoder layer norm1_epsilon must be finite, positive, "
                "and representable as float32"
            )
        object.__setattr__(self, "norm1_epsilon", float(epsilon))

    @property
    def batch(self) -> int:
        return self.self_attention.batch

    @property
    def queries(self) -> int:
        return self.self_attention.queries

    @property
    def embed_dim(self) -> int:
        return self.self_attention.embed_dim


def decoder_layer_weight_shapes(
    config: BevFormerDecoderLayerConfig,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return the three exact weight-group contracts for one layer."""

    if not isinstance(config, BevFormerDecoderLayerConfig):
        raise ValidationError("config must be a BevFormerDecoderLayerConfig")
    channels = config.embed_dim
    hidden = config.ffn.hidden_dim
    return {
        "self_attention": decoder_self_attention_weight_shapes(
            config.self_attention
        ),
        "cross_attention": decoder_cross_attention_weight_shapes(
            config.cross_attention
        ),
        "layer": {
            "norm1_weight": (channels,),
            "norm1_bias": (channels,),
            "ffn_fc1_weight": (channels, hidden),
            "ffn_fc1_bias": (hidden,),
            "ffn_fc2_weight": (hidden, channels),
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
            f"decoder layer {scope} weights must be a string-to-NumPy-array mapping"
        )
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValidationError(
            f"decoder layer {scope} weight names must be non-empty strings"
        )
    if set(weights) != set(expected):
        raise ValidationError(
            f"decoder layer {scope} weight names must be {sorted(expected)}, "
            f"got {sorted(weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder layer {scope} weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder layer {scope} weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder layer {scope} weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    return result


def _ffn_weights(layer_weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "fc1_weight": layer_weights["ffn_fc1_weight"],
        "fc1_bias": layer_weights["ffn_fc1_bias"],
        "fc2_weight": layer_weights["ffn_fc2_weight"],
        "fc2_bias": layer_weights["ffn_fc2_bias"],
        "norm_weight": layer_weights["norm2_weight"],
        "norm_bias": layer_weights["norm2_bias"],
    }


def build_bevformer_decoder_layer_graph(
    self_attention_weights: Mapping[str, np.ndarray],
    cross_attention_weights: Mapping[str, np.ndarray],
    layer_weights: Mapping[str, np.ndarray],
    config: BevFormerDecoderLayerConfig | None = None,
    *,
    name: str = "bevformer_decoder_transformer_layer",
) -> Graph:
    """Build one post-normalized BEVFormer decoder transformer layer."""

    if config is None:
        config = BevFormerDecoderLayerConfig()
    if not isinstance(config, BevFormerDecoderLayerConfig):
        raise ValidationError("config must be a BevFormerDecoderLayerConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder layer graph name must be non-empty")

    shapes = decoder_layer_weight_shapes(config)
    self_attention_weights = _validate_weight_group(
        "self_attention", self_attention_weights, shapes["self_attention"]
    )
    cross_attention_weights = _validate_weight_group(
        "cross_attention", cross_attention_weights, shapes["cross_attention"]
    )
    layer_weights = _validate_weight_group(
        "layer", layer_weights, shapes["layer"]
    )

    self_graph = build_decoder_self_attention_graph(
        self_attention_weights, config.self_attention
    )
    cross_graph = build_decoder_cross_attention_graph(
        cross_attention_weights, config.cross_attention
    )
    ffn_graph = build_bevformer_ffn_graph(
        _ffn_weights(layer_weights), config.ffn
    )

    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    cross = config.cross_attention
    query_spec = TensorSpec((batch, queries, channels), "float32")
    composer = GraphComposer(
        name,
        {
            "query": query_spec,
            "query_pos": query_spec,
            "value": TensorSpec(
                (batch, cross.spatial_size, channels), "float32"
            ),
            "reference_points": TensorSpec(
                (batch, queries, cross.levels, 2), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-decoder-transformer-layer-static",
            "scope": "one-decoder-transformer-layer",
            "full_decoder": False,
            "operation_order": [
                "self_attn",
                "norm0",
                "cross_attn",
                "norm1",
                "ffn",
                "norm2",
            ],
            "dropout": "inference-identity",
            "batch_first_abi": True,
            "cross_attention_query_pos": "original external query_pos",
            "profile": {
                "batch": batch,
                "queries": queries,
                "bev_h": cross.bev_h,
                "bev_w": cross.bev_w,
                "embed_dim": channels,
                "heads": config.self_attention.heads,
                "sampling_points": cross.sampling_points,
                "ffn_hidden_dim": config.ffn.hidden_dim,
            },
            "excluded_stages": [
                "reference-point-refinement",
                "remaining-decoder-layers",
                "decoder-output-intermediate-stack",
            ],
        },
    )

    self_output = composer.inline(
        self_graph,
        prefix="self_attention",
        bindings={"query": "query", "query_pos": "query_pos"},
        output_bindings={"output": "norm0_output"},
    )
    cross_output = composer.inline(
        cross_graph,
        prefix="cross_attention",
        bindings={
            "query": self_output["output"],
            "query_pos": "query_pos",
            "value": "value",
            "reference_points": "reference_points",
        },
        output_bindings={"output": "cross_attention_output"},
    )

    composer.add_constant("norm1_weight", layer_weights["norm1_weight"])
    composer.add_constant("norm1_bias", layer_weights["norm1_bias"])
    composer.add_node(
        Node(
            id="norm1",
            op="LayerNorm",
            inputs=(
                cross_output["output"],
                "norm1_weight",
                "norm1_bias",
            ),
            outputs=("norm1_output",),
            attrs={"axis": -1, "epsilon": config.norm1_epsilon},
            source_nodes=(
                "mmcv.cnn.bricks.transformer.BaseTransformerLayer.forward.norms.1",
            ),
        )
    )
    ffn_output = composer.inline(
        ffn_graph,
        prefix="ffn",
        bindings={"query": "norm1_output"},
        output_bindings={"output": "output"},
    )
    return composer.build((ffn_output["output"],))


def validate_decoder_layer_inputs(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate the static one-layer ABI and normalized references."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(inputs, Mapping):
        raise ValidationError("decoder layer inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in inputs):
        raise ValidationError("decoder layer input names must be non-empty strings")
    if set(inputs) != set(graph.inputs):
        raise ValidationError(
            f"decoder layer input names must be {sorted(graph.inputs)}, "
            f"got {sorted(inputs)}"
        )
    for name, spec in graph.inputs.items():
        value = inputs[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder layer input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder layer input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder layer input {name!r} must be contiguous and finite"
            )
    references = inputs["reference_points"]
    if np.any(references < 0.0) or np.any(references > 1.0):
        raise ValidationError(
            "decoder layer reference_points must be normalized to [0, 1]"
        )


__all__ = [
    "BevFormerDecoderLayerConfig",
    "build_bevformer_decoder_layer_graph",
    "decoder_layer_weight_shapes",
    "validate_decoder_layer_inputs",
]
