"""Canonical Transformer subgraph builders used by the BEVFormer decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .errors import ValidationError
from .ir import Graph, Node, TensorSpec


@dataclass(frozen=True, slots=True)
class MultiHeadAttentionConfig:
    batch: int
    query_length: int
    key_length: int
    embed_dim: int = 256
    heads: int = 8
    residual: bool = True

    def __post_init__(self) -> None:
        for name in ("batch", "query_length", "key_length", "embed_dim", "heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"attention {name} must be a positive integer")
        if self.embed_dim % self.heads:
            raise ValidationError("attention embed_dim must be divisible by heads")
        if not isinstance(self.residual, bool):
            raise ValidationError("attention residual must be boolean")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads


_WEIGHT_SHAPES = {
    "q_weight": "matrix",
    "q_bias": "bias",
    "k_weight": "matrix",
    "k_bias": "bias",
    "v_weight": "matrix",
    "v_bias": "bias",
    "out_weight": "matrix",
    "out_bias": "bias",
}


def _attention_weights(
    weights: Mapping[str, np.ndarray], embed_dim: int, head_dim: int
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError("attention weights must be a mapping")
    if set(weights) != set(_WEIGHT_SHAPES):
        raise ValidationError(
            f"attention weight names must be {sorted(_WEIGHT_SHAPES)}, got {sorted(weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, kind in _WEIGHT_SHAPES.items():
        value = np.asarray(weights[name])
        expected = (embed_dim, embed_dim) if kind == "matrix" else (embed_dim,)
        if value.shape != expected or value.dtype != np.float32:
            raise ValidationError(
                f"attention weight {name!r} requires {expected}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"attention weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    result["score_scale"] = np.asarray(
        1.0 / math.sqrt(head_dim), dtype=np.float32
    )
    return result


def build_multihead_attention_graph(
    config: MultiHeadAttentionConfig,
    weights: Mapping[str, np.ndarray],
    *,
    name: str = "multihead_attention",
) -> Graph:
    """Lower scaled dot-product attention to ordinary canonical tensor ops.

    Matrices use the canonical ``[input_channel, output_channel]`` convention.
    PyTorch ``Linear.weight`` values therefore need one offline transpose before
    calling this builder.
    """

    if not isinstance(config, MultiHeadAttentionConfig):
        raise ValidationError("config must be a MultiHeadAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("attention graph name must be non-empty")
    constants = _attention_weights(weights, config.embed_dim, config.head_dim)
    batch = config.batch
    queries = config.query_length
    keys = config.key_length
    channels = config.embed_dim
    heads = config.heads
    head_dim = config.head_dim
    nodes = [
        Node("q_linear", "MatMul", ("query", "q_weight"), ("q_mm",)),
        Node("q_bias_add", "Add", ("q_mm", "q_bias"), ("q_projected",)),
        Node("q_reshape", "Reshape", ("q_projected",), ("q_bqhd",), {"shape": [batch, queries, heads, head_dim]}),
        Node("q_transpose", "Transpose", ("q_bqhd",), ("q_heads",), {"perm": [0, 2, 1, 3]}),
        Node("k_linear", "MatMul", ("key", "k_weight"), ("k_mm",)),
        Node("k_bias_add", "Add", ("k_mm", "k_bias"), ("k_projected",)),
        Node("k_reshape", "Reshape", ("k_projected",), ("k_bkhd",), {"shape": [batch, keys, heads, head_dim]}),
        Node("k_transpose", "Transpose", ("k_bkhd",), ("k_heads",), {"perm": [0, 2, 3, 1]}),
        Node("v_linear", "MatMul", ("value", "v_weight"), ("v_mm",)),
        Node("v_bias_add", "Add", ("v_mm", "v_bias"), ("v_projected",)),
        Node("v_reshape", "Reshape", ("v_projected",), ("v_bkhd",), {"shape": [batch, keys, heads, head_dim]}),
        Node("v_transpose", "Transpose", ("v_bkhd",), ("v_heads",), {"perm": [0, 2, 1, 3]}),
        Node("score_matmul", "MatMul", ("q_heads", "k_heads"), ("raw_scores",)),
        Node("score_scale", "Mul", ("raw_scores", "score_scale"), ("scaled_scores",)),
        Node("score_softmax", "Softmax", ("scaled_scores",), ("probabilities",), {"axis": -1}),
        Node("context_matmul", "MatMul", ("probabilities", "v_heads"), ("context_heads",)),
        Node("context_transpose", "Transpose", ("context_heads",), ("context_bqhd",), {"perm": [0, 2, 1, 3]}),
        Node("context_reshape", "Reshape", ("context_bqhd",), ("context",), {"shape": [batch, queries, channels]}),
        Node("out_linear", "MatMul", ("context", "out_weight"), ("out_mm",)),
        Node("out_bias_add", "Add", ("out_mm", "out_bias"), ("projected",)),
    ]
    output = "projected"
    if config.residual:
        nodes.append(Node("residual_add", "Add", ("projected", "query"), ("output",)))
        output = "output"
    return Graph(
        name=name,
        inputs={
            "query": TensorSpec((batch, queries, channels), "float32"),
            "key": TensorSpec((batch, keys, channels), "float32"),
            "value": TensorSpec((batch, keys, channels), "float32"),
        },
        outputs=(output,),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "scaled-dot-product-multihead-attention",
            "heads": heads,
            "head_dim": head_dim,
            "source": "BEVFormer decoder attention lowering",
        },
    )


__all__ = ["MultiHeadAttentionConfig", "build_multihead_attention_graph"]
