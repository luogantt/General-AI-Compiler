"""Backend-independent residual transformer FFN graph shared by model adapters.

The graph uses canonical [input_channel, output_channel] f32 weights. It can
represent arbitrary positive hidden widths; model wrappers impose their own
checkpoint contracts and supply model-specific metadata and node provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class TransformerFFNConfig:
    batch: int
    tokens: int
    embed_dim: int
    hidden_dim: int
    post_layer_norm: bool = True
    layer_norm_epsilon: float = 1e-5
    ffn_dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("batch", "tokens", "embed_dim", "hidden_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"transformer FFN {name} must be a positive integer")
        if not isinstance(self.post_layer_norm, bool):
            raise ValidationError("transformer FFN post_layer_norm must be boolean")
        epsilon = self.layer_norm_epsilon
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating))
                or not math.isfinite(float(epsilon)) or float(epsilon) <= 0
                or not np.isfinite(np.float32(epsilon)) or np.float32(epsilon) == 0):
            raise ValidationError("transformer FFN epsilon must be positive and representable as float32")
        dropout = self.ffn_dropout
        if (isinstance(dropout, bool) or not isinstance(dropout, (int, float, np.integer, np.floating))
                or not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1):
            raise ValidationError("transformer FFN dropout must be in [0, 1)")
        object.__setattr__(self, "layer_norm_epsilon", float(epsilon))
        object.__setattr__(self, "ffn_dropout", float(dropout))


def ffn_weight_shapes(config: TransformerFFNConfig) -> dict[str, tuple[int, ...]]:
    if not isinstance(config, TransformerFFNConfig):
        raise ValidationError("config must be a TransformerFFNConfig")
    shapes = {
        "fc1_weight": (config.embed_dim, config.hidden_dim),
        "fc1_bias": (config.hidden_dim,),
        "fc2_weight": (config.hidden_dim, config.embed_dim),
        "fc2_bias": (config.embed_dim,),
    }
    if config.post_layer_norm:
        shapes.update(norm_weight=(config.embed_dim,), norm_bias=(config.embed_dim,))
    return shapes


def _copy_weights(config: TransformerFFNConfig, weights: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected = ffn_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("transformer FFN weights must be a mapping")
    if set(weights) != set(expected):
        raise ValidationError(f"transformer FFN weight names must be {sorted(expected)}, got {sorted(map(repr, weights))}")
    result = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(f"transformer FFN weight {name!r} requires {shape}/float32, got {value.shape}/{value.dtype}")
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(f"transformer FFN weight {name!r} must be contiguous and finite")
        result[name] = value.copy()
    return result


def build_transformer_ffn_graph(
    weights: Mapping[str, np.ndarray], config: TransformerFFNConfig, *,
    name: str = "transformer_ffn", metadata: Mapping[str, Any] | None = None,
    sources: Mapping[str, str] | None = None,
) -> Graph:
    """Build MatMul→Bias→ReLU→MatMul→Bias→Residual→optional LayerNorm.

    Dropout has inference identity semantics. ``metadata`` replaces the generic
    graph metadata when supplied, allowing existing model contracts to remain
    byte-for-byte compatible while sharing the tensor graph implementation.
    """
    if not isinstance(config, TransformerFFNConfig):
        raise ValidationError("config must be a TransformerFFNConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("transformer FFN graph name must be non-empty")
    constants = _copy_weights(config, weights)
    provenance = dict(sources or {})
    nodes = []

    def node(node_id, op, inputs, output, source, attrs=None):
        nodes.append(Node(node_id, op, inputs, (output,), dict(attrs or {}),
                          (provenance.get(node_id, source),)))

    node("fc1_matmul", "MatMul", ("query", "fc1_weight"), "fc1_mm", "transformer.FFN.fc1")
    node("fc1_bias_add", "Add", ("fc1_mm", "fc1_bias"), "fc1", "transformer.FFN.fc1")
    node("relu", "ReLU", ("fc1",), "activated", "transformer.FFN.activation")
    node("fc2_matmul", "MatMul", ("activated", "fc2_weight"), "fc2_mm", "transformer.FFN.fc2")
    node("fc2_bias_add", "Add", ("fc2_mm", "fc2_bias"), "projected", "transformer.FFN.fc2")
    residual_output = "ffn_output" if config.post_layer_norm else "output"
    node("residual_add", "Add", ("query", "projected"), residual_output, "transformer.FFN.residual")
    if config.post_layer_norm:
        node("post_layer_norm", "LayerNorm", ("ffn_output", "norm_weight", "norm_bias"), "output",
             "transformer.FFN.post_norm", {"axis": -1, "epsilon": config.layer_norm_epsilon})
    default_metadata = {
        "kind": "transformer-residual-ffn",
        "profile": {"batch": config.batch, "tokens": config.tokens, "embed_dim": config.embed_dim,
                    "feedforward_channels": config.hidden_dim},
        "operation_order_fragment": ["ffn", "norm"] if config.post_layer_norm else ["ffn"],
        "dropout": "inference-identity", "ffn_dropout_probability": config.ffn_dropout,
        "weight_layout": "[input_channel, output_channel]",
    }
    graph = Graph(name, {"query": TensorSpec((config.batch, config.tokens, config.embed_dim), "float32")},
                  ("output",), nodes, constants, metadata=dict(metadata) if metadata is not None else default_metadata)
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


__all__ = ["TransformerFFNConfig", "ffn_weight_shapes", "build_transformer_ffn_graph"]
