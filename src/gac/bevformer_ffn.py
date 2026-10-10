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

from .errors import ValidationError
from .ir import Graph, Node, TensorSpec
from .passes import ShapeInferencePass


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


def ffn_weight_shapes(config: BEVFormerFFNConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact canonical parameter contract for ``config``."""

    if not isinstance(config, BEVFormerFFNConfig):
        raise ValidationError("config must be a BEVFormerFFNConfig")
    channels = config.embed_dim
    hidden = config.hidden_dim
    shapes: dict[str, tuple[int, ...]] = {
        "fc1_weight": (channels, hidden),
        "fc1_bias": (hidden,),
        "fc2_weight": (hidden, channels),
        "fc2_bias": (channels,),
    }
    if config.post_layer_norm:
        shapes.update(
            {
                "norm_weight": (channels,),
                "norm_bias": (channels,),
            }
        )
    return shapes


def _validated_weights(
    config: BEVFormerFFNConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = ffn_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("BEVFormer FFN weights must be a mapping")
    actual_names = set(weights)
    if actual_names != set(expected):
        actual = sorted(repr(name) for name in actual_names)
        raise ValidationError(
            f"BEVFormer FFN weight names must be {sorted(expected)}, got {actual}"
        )

    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"BEVFormer FFN weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"BEVFormer FFN weight {name!r} must be contiguous and finite"
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


def build_bevformer_ffn_graph(
    weights: Mapping[str, np.ndarray],
    config: BEVFormerFFNConfig | None = None,
    *,
    name: str = "bevformer_tiny_ffn",
) -> Graph:
    """Build the static inference graph for one ``ffn[, norm]`` fragment.

    ``query`` is both the FFN input and its default residual, matching MMCV
    ``FFN.forward(x, identity=None)`` for BEVFormer's post-normalization order.
    """

    if config is None:
        config = BEVFormerFFNConfig()
    if not isinstance(config, BEVFormerFFNConfig):
        raise ValidationError("config must be a BEVFormerFFNConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("BEVFormer FFN graph name must be non-empty")

    constants = _validated_weights(config, weights)
    nodes = [
        _node(
            "fc1_matmul",
            "MatMul",
            ("query", "fc1_weight"),
            "fc1_mm",
            "mmcv.cnn.bricks.transformer.FFN.layers[0].Linear",
        ),
        _node(
            "fc1_bias_add",
            "Add",
            ("fc1_mm", "fc1_bias"),
            "fc1",
            "mmcv.cnn.bricks.transformer.FFN.layers[0].Linear",
        ),
        _node(
            "relu",
            "ReLU",
            ("fc1",),
            "activated",
            "mmcv.cnn.bricks.transformer.FFN.layers[0].ReLU",
        ),
        _node(
            "fc2_matmul",
            "MatMul",
            ("activated", "fc2_weight"),
            "fc2_mm",
            "mmcv.cnn.bricks.transformer.FFN.layers[1].Linear",
        ),
        _node(
            "fc2_bias_add",
            "Add",
            ("fc2_mm", "fc2_bias"),
            "projected",
            "mmcv.cnn.bricks.transformer.FFN.layers[1].Linear",
        ),
    ]

    residual_output = "ffn_output" if config.post_layer_norm else "output"
    nodes.append(
        _node(
            "residual_add",
            "Add",
            ("query", "projected"),
            residual_output,
            "mmcv.cnn.bricks.transformer.FFN.forward.identity",
        )
    )
    if config.post_layer_norm:
        nodes.append(
            _node(
                "post_layer_norm",
                "LayerNorm",
                ("ffn_output", "norm_weight", "norm_bias"),
                "output",
                "BEVFormerLayer.operation_order.norm_after_ffn",
                {"axis": -1, "epsilon": config.layer_norm_epsilon},
            )
        )

    graph = Graph(
        name=name,
        inputs={
            "query": TensorSpec(
                (config.batch, config.queries, config.embed_dim), "float32"
            )
        },
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-transformer-ffn",
            "profile": {
                "batch": config.batch,
                "bev_h": config.bev_h,
                "bev_w": config.bev_w,
                "embed_dim": config.embed_dim,
                "feedforward_channels": config.hidden_dim,
            },
            "operation_order_fragment": (
                ["ffn", "norm"] if config.post_layer_norm else ["ffn"]
            ),
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
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


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
