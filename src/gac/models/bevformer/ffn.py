"""Static BEVFormer tiny feed-forward-network lowering.

The default specialization follows ``bevformer_tiny.py``: a ``[1, 2500,
256]`` BEV tensor, two fully-connected layers with 512 hidden channels, ReLU,
and ``ffn_dropout=0.1``.  Dropout is deliberately absent from the inference
graph because PyTorch/MMCV dropout is the identity in evaluation mode.

MMCV stores ``Linear.weight`` as ``[output_channel, input_channel]``.  This
module's canonical contract is ``[input_channel, output_channel]``, so an
offline checkpoint importer must transpose both matrices.  It must also
resolve the enclosing encoder-layer prefix.  The optional final LayerNorm is
the ``norm`` immediately following ``ffn`` in BEVFormer's operation order; it
is not part of :class:`mmcv.cnn.bricks.transformer.FFN` itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph
from gac.models.common.ffn import (
    TransformerFFNConfig,
    build_transformer_ffn_graph,
    ffn_weight_shapes as _shared_ffn_weight_shapes,
)


@dataclass(frozen=True, slots=True)
class BEVFormerFFNConfig:
    """One fully static BEVFormer FFN specialization.

    The hidden width is fixed to twice ``embed_dim`` to match the tiny model's
    ``feedforward_channels = _dim_ * 2`` configuration.
    """

    batch: int = 1
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    post_layer_norm: bool = True
    layer_norm_epsilon: float = 1e-5
    ffn_dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("batch", "bev_h", "bev_w", "embed_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"BEVFormer FFN {name} must be a positive integer")
        if not isinstance(self.post_layer_norm, bool):
            raise ValidationError("BEVFormer FFN post_layer_norm must be boolean")

        epsilon = self.layer_norm_epsilon
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float, np.integer, np.floating))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
            or not np.isfinite(np.float32(epsilon))
            or np.float32(epsilon) == 0.0
        ):
            raise ValidationError(
                "BEVFormer FFN layer_norm_epsilon must be finite, positive, "
                "and representable as float32"
            )
        dropout = self.ffn_dropout
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float, np.integer, np.floating))
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValidationError("BEVFormer FFN ffn_dropout must be in [0, 1)")
        object.__setattr__(self, "layer_norm_epsilon", float(epsilon))
        object.__setattr__(self, "ffn_dropout", float(dropout))

    @property
    def queries(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def hidden_dim(self) -> int:
        return self.embed_dim * 2


def _common_config(config: BEVFormerFFNConfig) -> TransformerFFNConfig:
    if not isinstance(config, BEVFormerFFNConfig):
        raise ValidationError("config must be a BEVFormerFFNConfig")
    return TransformerFFNConfig(
        batch=config.batch, tokens=config.queries, embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim, post_layer_norm=config.post_layer_norm,
        layer_norm_epsilon=config.layer_norm_epsilon, ffn_dropout=config.ffn_dropout,
    )


def ffn_weight_shapes(config: BEVFormerFFNConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact canonical parameter contract for this model profile."""
    return _shared_ffn_weight_shapes(_common_config(config))


def build_bevformer_ffn_graph(
    weights: Mapping[str, np.ndarray], config: BEVFormerFFNConfig | None = None, *,
    name: str = "bevformer_tiny_ffn",
) -> Graph:
    """Build the shared residual FFN with BEVFormer's contract and provenance."""
    if config is None:
        config = BEVFormerFFNConfig()
    common = _common_config(config)
    if not isinstance(name, str) or not name:
        raise ValidationError("BEVFormer FFN graph name must be non-empty")
    source = "mmcv.cnn.bricks.transformer.FFN"
    sources = {
        "fc1_matmul": f"{source}.layers[0].Linear",
        "fc1_bias_add": f"{source}.layers[0].Linear",
        "relu": f"{source}.layers[0].ReLU",
        "fc2_matmul": f"{source}.layers[1].Linear",
        "fc2_bias_add": f"{source}.layers[1].Linear",
        "residual_add": f"{source}.forward.identity",
        "post_layer_norm": "BEVFormerLayer.operation_order.norm_after_ffn",
    }
    metadata = {
        "kind": "bevformer-transformer-ffn",
        "profile": {
            "batch": config.batch, "bev_h": config.bev_h, "bev_w": config.bev_w,
            "embed_dim": config.embed_dim, "feedforward_channels": config.hidden_dim,
        },
        "operation_order_fragment": ["ffn", "norm"] if config.post_layer_norm else ["ffn"],
        "dropout": "inference-identity",
        "ffn_dropout_probability": config.ffn_dropout,
        "weight_layout": "[input_channel, output_channel]",
        "checkpoint_mapping_boundary": (
            "The enclosing encoder layer/checkpoint prefix is resolved by the "
            "checkpoint importer; transpose PyTorch Linear weights before build."
        ),
        "checkpoint_key_suffixes": {
            "fc1_weight": "ffns.0.layers.0.0.weight (transpose)",
            "fc1_bias": "ffns.0.layers.0.0.bias",
            "fc2_weight": "ffns.0.layers.1.weight (transpose)",
            "fc2_bias": "ffns.0.layers.1.bias",
            "norm_weight": "following operation_order norm weight",
            "norm_bias": "following operation_order norm bias",
        },
    }
    return build_transformer_ffn_graph(weights, common, name=name, metadata=metadata, sources=sources)


def validate_ffn_inputs(
    graph: Graph,
    inputs: Mapping[str, np.ndarray],
) -> None:
    """Validate the static input ABI without running the graph."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(inputs, Mapping):
        raise ValidationError("BEVFormer FFN inputs must be a mapping")
    actual_names = set(inputs)
    if actual_names != set(graph.inputs):
        actual = sorted(repr(name) for name in actual_names)
        raise ValidationError(
            f"BEVFormer FFN input names must be {sorted(graph.inputs)}, got {actual}"
        )
    for name, spec in graph.inputs.items():
        value = np.asarray(inputs[name])
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"BEVFormer FFN input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"BEVFormer FFN input {name!r} must be contiguous and finite"
            )


__all__ = [
    "BEVFormerFFNConfig",
    "build_bevformer_ffn_graph",
    "ffn_weight_shapes",
    "validate_ffn_inputs",
]
