"""Learned BEV queries and static positional inputs for BEVFormer-tiny."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class BevFormerEncoderInputConfig:
    batch: int = 1
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    can_bus_dim: int = 18
    can_bus_hidden_dim: int = 128
    norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "bev_h",
            "bev_w",
            "embed_dim",
            "can_bus_dim",
            "can_bus_hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"encoder input {name} must be positive")
        if self.batch != 1:
            raise ValidationError("encoder input preparation currently requires batch=1")
        if self.embed_dim % 2:
            raise ValidationError("encoder input embed_dim must be even")
        value = self.norm_epsilon
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
            or np.float32(value) == 0.0
        ):
            raise ValidationError(
                "encoder input norm_epsilon must be positive finite float32"
            )
        object.__setattr__(self, "norm_epsilon", float(value))

    @property
    def queries(self) -> int:
        return self.bev_h * self.bev_w


def encoder_input_weight_shapes(
    config: BevFormerEncoderInputConfig,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(config, BevFormerEncoderInputConfig):
        raise ValidationError("config must be a BevFormerEncoderInputConfig")
    return {
        "bev_embedding": (config.queries, config.embed_dim),
        "can_bus_fc0_weight": (config.can_bus_dim, config.can_bus_hidden_dim),
        "can_bus_fc0_bias": (config.can_bus_hidden_dim,),
        "can_bus_fc1_weight": (config.can_bus_hidden_dim, config.embed_dim),
        "can_bus_fc1_bias": (config.embed_dim,),
        "can_bus_norm_weight": (config.embed_dim,),
        "can_bus_norm_bias": (config.embed_dim,),
        "row_embedding": (config.bev_h, config.embed_dim // 2),
        "col_embedding": (config.bev_w, config.embed_dim // 2),
    }


def _validated_weights(
    config: BevFormerEncoderInputConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = encoder_input_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("encoder input weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != set(expected):
        raise ValidationError(
            f"encoder input weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"encoder input weight {name!r} must be an ndarray")
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"encoder input weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"encoder input weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    return result


def _learned_position(
    config: BevFormerEncoderInputConfig,
    row_embedding: np.ndarray,
    col_embedding: np.ndarray,
) -> np.ndarray:
    """Reproduce MMDetection ``LearnedPositionalEncoding`` exactly.

    MMDetection concatenates the column (x) embedding before the row (y)
    embedding, returns NCHW, and BEVFormer then flattens H/W.  The equivalent
    batch-first representation is materialized once here as a constant.
    """

    x = np.broadcast_to(
        col_embedding.reshape(1, config.bev_w, config.embed_dim // 2),
        (config.bev_h, config.bev_w, config.embed_dim // 2),
    )
    y = np.broadcast_to(
        row_embedding.reshape(config.bev_h, 1, config.embed_dim // 2),
        (config.bev_h, config.bev_w, config.embed_dim // 2),
    )
    position = np.concatenate((x, y), axis=-1)
    return np.ascontiguousarray(position.reshape(1, config.queries, config.embed_dim))


def _reference_grid(config: BevFormerEncoderInputConfig) -> np.ndarray:
    y = np.linspace(
        0.5, config.bev_h - 0.5, config.bev_h, dtype=np.float32
    ) / np.float32(config.bev_h)
    x = np.linspace(
        0.5, config.bev_w - 0.5, config.bev_w, dtype=np.float32
    ) / np.float32(config.bev_w)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, config.queries, 1, 2)
    return np.ascontiguousarray(grid)


def build_bevformer_encoder_input_graph(
    weights: Mapping[str, np.ndarray],
    config: BevFormerEncoderInputConfig | None = None,
    *,
    name: str = "bevformer_encoder_input_preparation",
) -> Graph:
    """Build learned BEV query, CAN-bus conditioning, position, and 2D grid."""

    config = config or BevFormerEncoderInputConfig()
    if not isinstance(config, BevFormerEncoderInputConfig):
        raise ValidationError("config must be a BevFormerEncoderInputConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("encoder input graph name must be non-empty")
    constants = _validated_weights(config, weights)
    constants["bev_embedding_batched"] = constants.pop("bev_embedding").reshape(
        1, config.queries, config.embed_dim
    )
    constants["query_pos"] = _learned_position(
        config,
        constants.pop("row_embedding"),
        constants.pop("col_embedding"),
    )
    constants["spatial_query_pos"] = np.zeros(
        (config.batch, config.queries, config.embed_dim), dtype=np.float32
    )
    constants["reference_points"] = _reference_grid(config)
    nodes = [
        Node(
            "can_bus_fc0_matmul",
            "MatMul",
            ("can_bus_delta", "can_bus_fc0_weight"),
            ("can_bus_fc0_mm",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.0",),
        ),
        Node(
            "can_bus_fc0_bias",
            "Add",
            ("can_bus_fc0_mm", "can_bus_fc0_bias"),
            ("can_bus_fc0",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.0",),
        ),
        Node(
            "can_bus_fc0_relu",
            "ReLU",
            ("can_bus_fc0",),
            ("can_bus_hidden",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.1",),
        ),
        Node(
            "can_bus_fc1_matmul",
            "MatMul",
            ("can_bus_hidden", "can_bus_fc1_weight"),
            ("can_bus_fc1_mm",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.2",),
        ),
        Node(
            "can_bus_fc1_bias",
            "Add",
            ("can_bus_fc1_mm", "can_bus_fc1_bias"),
            ("can_bus_fc1",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.2",),
        ),
        Node(
            "can_bus_fc1_relu",
            "ReLU",
            ("can_bus_fc1",),
            ("can_bus_activated",),
            source_nodes=("PerceptionTransformer.can_bus_mlp.3",),
        ),
        Node(
            "can_bus_norm",
            "LayerNorm",
            ("can_bus_activated", "can_bus_norm_weight", "can_bus_norm_bias"),
            ("can_bus_embedding",),
            {"axis": -1, "epsilon": config.norm_epsilon},
            source_nodes=("PerceptionTransformer.can_bus_mlp.norm",),
        ),
        Node(
            "can_bus_query_axis",
            "Reshape",
            ("can_bus_embedding",),
            ("can_bus_b1c",),
            {"shape": [config.batch, 1, config.embed_dim]},
            source_nodes=("PerceptionTransformer.get_bev_features.can_bus",),
        ),
        Node(
            "condition_bev_query",
            "Add",
            ("bev_embedding_batched", "can_bus_b1c"),
            ("query",),
            source_nodes=("PerceptionTransformer.get_bev_features.bev_queries",),
        ),
    ]
    graph = Graph(
        name=name,
        inputs={
            "can_bus_delta": TensorSpec(
                (config.batch, config.can_bus_dim), "float32"
            )
        },
        outputs=("query", "query_pos", "spatial_query_pos", "reference_points"),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-encoder-input-preparation",
            "can_bus_mlp": "Linear-ReLU-Linear-ReLU-LayerNorm",
            "bev_position": (
                "MMDetection LearnedPositionalEncoding using checkpoint "
                "col_embed then row_embed"
            ),
            "spatial_query_pos": "all-zero because official encoder passes None",
            "reference_grid": "cell centers normalized to [0,1]",
            "profile": {
                "batch": config.batch,
                "bev_h": config.bev_h,
                "bev_w": config.bev_w,
                "embed_dim": config.embed_dim,
                "can_bus_dim": config.can_bus_dim,
            },
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_encoder_input_values(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != {"can_bus_delta"}:
        raise ValidationError("encoder input preparation requires can_bus_delta")
    value = values["can_bus_delta"]
    spec = graph.inputs["can_bus_delta"]
    if not isinstance(value, np.ndarray):
        raise ValidationError("can_bus_delta must be a NumPy ndarray")
    if value.shape != spec.shape or value.dtype.name != spec.dtype:
        raise ValidationError(
            f"can_bus_delta requires {spec.shape}/{spec.dtype}, "
            f"got {value.shape}/{value.dtype}"
        )
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValidationError("can_bus_delta must be contiguous and finite")


__all__ = [
    "BevFormerEncoderInputConfig",
    "build_bevformer_encoder_input_graph",
    "encoder_input_weight_shapes",
    "validate_encoder_input_values",
]
