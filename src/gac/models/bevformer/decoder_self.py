"""Static BEVFormer decoder self-attention inference subgraph.

This module covers only the ``self_attn[, norm]`` prefix of one tiny decoder
layer.  It follows MMCV ``MultiheadAttention.forward``: positional encoding is
added to Q and K, V remains the original query, and the projected result is
added to that original query.  The default final LayerNorm represents
``norms.0`` from the following ``norm`` operation in the decoder layer.

All linear matrices use the compiler's canonical ``[input, output]`` layout.
Dropout is absent because every configured dropout path is an identity during
evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.models.common.transformer import MultiHeadAttentionConfig, build_multihead_attention_graph


_ATTENTION_WEIGHT_NAMES = (
    "q_weight",
    "q_bias",
    "k_weight",
    "k_bias",
    "v_weight",
    "v_bias",
    "out_weight",
    "out_bias",
)


@dataclass(frozen=True, slots=True)
class DecoderSelfAttentionConfig:
    """One fully static decoder self-attention specialization."""

    batch: int = 1
    queries: int = 900
    embed_dim: int = 256
    heads: int = 8
    post_layer_norm: bool = True
    norm_epsilon: float = 1e-5
    dropout_probability: float = 0.1

    def __post_init__(self) -> None:
        for name in ("batch", "queries", "embed_dim", "heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(
                    f"decoder self attention {name} must be a positive integer"
                )
        if self.batch != 1:
            raise ValidationError("decoder self attention currently requires batch=1")
        if self.embed_dim % self.heads:
            raise ValidationError(
                "decoder self attention embed_dim must be divisible by heads"
            )
        if not isinstance(self.post_layer_norm, bool):
            raise ValidationError(
                "decoder self attention post_layer_norm must be boolean"
            )

        epsilon = self.norm_epsilon
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
            or not np.isfinite(np.float32(epsilon))
            or np.float32(epsilon) == 0.0
        ):
            raise ValidationError(
                "decoder self attention norm_epsilon must be finite, positive, "
                "and representable as float32"
            )
        dropout = self.dropout_probability
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float, np.integer, np.floating))
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValidationError(
                "decoder self attention dropout_probability must be in [0, 1)"
            )
        object.__setattr__(self, "norm_epsilon", float(epsilon))
        object.__setattr__(self, "dropout_probability", float(dropout))

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads


def decoder_self_attention_weight_shapes(
    config: DecoderSelfAttentionConfig,
) -> dict[str, tuple[int, ...]]:
    """Return the exact canonical weight inventory for this fragment."""

    if not isinstance(config, DecoderSelfAttentionConfig):
        raise ValidationError("config must be a DecoderSelfAttentionConfig")
    channels = config.embed_dim
    shapes: dict[str, tuple[int, ...]] = {}
    for prefix in ("q", "k", "v", "out"):
        shapes[f"{prefix}_weight"] = (channels, channels)
        shapes[f"{prefix}_bias"] = (channels,)
    if config.post_layer_norm:
        shapes["norm_weight"] = (channels,)
        shapes["norm_bias"] = (channels,)
    return shapes


def _validated_weights(
    config: DecoderSelfAttentionConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = decoder_self_attention_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError(
            "decoder self attention weights must be a string-to-NumPy-array mapping"
        )
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValidationError(
            "decoder self attention weight names must be non-empty strings"
        )
    if set(weights) != set(expected):
        raise ValidationError(
            "decoder self attention weight names must be "
            f"{sorted(expected)}, got {sorted(weights)}"
        )

    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder self attention weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder self attention weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder self attention weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    return result


def _node(
    node_id: str,
    op: str,
    inputs: tuple[str, ...],
    output: str,
    source: str,
    attrs: Mapping[str, object] | None = None,
) -> Node:
    return Node(
        id=node_id,
        op=op,
        inputs=inputs,
        outputs=(output,),
        attrs=dict(attrs or {}),
        source_nodes=(source,),
    )


def build_decoder_self_attention_graph(
    weights: Mapping[str, np.ndarray],
    config: DecoderSelfAttentionConfig | None = None,
    *,
    name: str = "bevformer_decoder_self_attention",
) -> Graph:
    """Compose the static ``self_attn[, norms.0]`` inference fragment."""

    if config is None:
        config = DecoderSelfAttentionConfig()
    if not isinstance(config, DecoderSelfAttentionConfig):
        raise ValidationError("config must be a DecoderSelfAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder self attention graph name must be non-empty")
    validated = _validated_weights(config, weights)

    channels = config.embed_dim
    tensor_spec = TensorSpec((config.batch, config.queries, channels), "float32")
    composer = GraphComposer(
        name,
        {"query": tensor_spec, "query_pos": tensor_spec},
        metadata={
            "kind": "bevformer-decoder-self-attention-static",
            "scope": "one-decoder-layer-self-attention-fragment",
            "full_decoder": False,
            "operation_order_fragment": (
                ["self_attn", "norms.0"]
                if config.post_layer_norm
                else ["self_attn"]
            ),
            "dropout": "inference-identity",
            "dropout_probability": config.dropout_probability,
            "masks": {"attn_mask": None, "key_padding_mask": None},
            "position_semantics": {
                "q": "query + query_pos",
                "k": "query + query_pos",
                "v": "query",
                "identity": "query",
            },
            "profile": {
                "batch": config.batch,
                "queries": config.queries,
                "embed_dim": channels,
                "heads": config.heads,
                "head_dim": config.head_dim,
            },
            "component_builder": (
                "transformer.build_multihead_attention_graph(residual=False)"
            ),
        },
    )
    composer.add_node(
        _node(
            "query_position_add",
            "Add",
            ("query", "query_pos"),
            "positioned_query",
            "mmcv.cnn.bricks.transformer.MultiheadAttention.forward.query_pos_key_pos",
        )
    )

    attention_config = MultiHeadAttentionConfig(
        batch=config.batch,
        query_length=config.queries,
        key_length=config.queries,
        embed_dim=channels,
        heads=config.heads,
        residual=False,
    )
    attention_graph = build_multihead_attention_graph(
        attention_config,
        {name: validated[name] for name in _ATTENTION_WEIGHT_NAMES},
        name="decoder_self_attention_core",
    )
    attention = composer.inline(
        attention_graph,
        prefix="attention",
        bindings={
            "query": "positioned_query",
            "key": "positioned_query",
            "value": "query",
        },
        output_bindings={"projected": "attention_projected"},
    )

    residual_output = "self_attention_output" if config.post_layer_norm else "output"
    composer.add_node(
        _node(
            "residual_add",
            "Add",
            (attention["projected"], "query"),
            residual_output,
            "mmcv.cnn.bricks.transformer.MultiheadAttention.forward.identity",
        )
    )
    if config.post_layer_norm:
        composer.add_constant("norm_weight", validated["norm_weight"])
        composer.add_constant("norm_bias", validated["norm_bias"])
        composer.add_node(
            _node(
                "post_layer_norm",
                "LayerNorm",
                ("self_attention_output", "norm_weight", "norm_bias"),
                "output",
                "BEVFormerDecoderLayer.operation_order.norms.0",
                {"axis": -1, "epsilon": config.norm_epsilon},
            )
        )
    return composer.build(("output",))


def validate_decoder_self_attention_inputs(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate the two-input static ABI without executing the graph."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(inputs, Mapping):
        raise ValidationError("decoder self attention inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in inputs):
        raise ValidationError(
            "decoder self attention input names must be non-empty strings"
        )
    if set(inputs) != set(graph.inputs):
        raise ValidationError(
            "decoder self attention input names must be "
            f"{sorted(graph.inputs)}, got {sorted(inputs)}"
        )
    for name, spec in graph.inputs.items():
        value = inputs[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder self attention input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder self attention input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder self attention input {name!r} must be contiguous and finite"
            )


__all__ = [
    "DecoderSelfAttentionConfig",
    "build_decoder_self_attention_graph",
    "decoder_self_attention_weight_shapes",
    "validate_decoder_self_attention_inputs",
]
