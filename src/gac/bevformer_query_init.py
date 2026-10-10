"""Static BEVFormer object-query and decoder-reference initialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .ir import Graph, Node
from .passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class DecoderQueryInitConfig:
    batch: int = 1
    queries: int = 900
    embed_dim: int = 256

    def __post_init__(self) -> None:
        for name in ("batch", "queries", "embed_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"decoder query init {name} must be positive")
        if self.batch != 1:
            raise ValidationError("decoder query init currently requires batch=1")


def decoder_query_init_weight_shapes(
    config: DecoderQueryInitConfig,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(config, DecoderQueryInitConfig):
        raise ValidationError("config must be a DecoderQueryInitConfig")
    channels = config.embed_dim
    return {
        "query_embedding": (config.queries, channels * 2),
        "reference_weight": (channels, 3),
        "reference_bias": (3,),
    }


def _validated_weights(
    config: DecoderQueryInitConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = decoder_query_init_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("decoder query init weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != set(expected):
        raise ValidationError(
            f"decoder query init weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder query init weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder query init weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    return result


def build_decoder_query_init_graph(
    weights: Mapping[str, np.ndarray],
    config: DecoderQueryInitConfig | None = None,
    *,
    name: str = "bevformer_decoder_query_init",
) -> Graph:
    """Materialize learned query/query-pos tensors and initial 3D references."""

    config = config or DecoderQueryInitConfig()
    if not isinstance(config, DecoderQueryInitConfig):
        raise ValidationError("config must be a DecoderQueryInitConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder query init graph name must be non-empty")
    constants = _validated_weights(config, weights)
    queries, channels = config.queries, config.embed_dim
    source = "PerceptionTransformer.forward.object_query_embed"
    nodes = [
        Node(
            "embedding_batch_axis",
            "Reshape",
            ("query_embedding",),
            ("batched_embedding",),
            {"shape": [config.batch, queries, channels * 2]},
            source_nodes=(source,),
        ),
        Node(
            "query_pos_slice",
            "Slice",
            ("batched_embedding",),
            ("query_pos",),
            {"starts": [0], "ends": [channels], "axes": [-1], "steps": [1]},
            source_nodes=("PerceptionTransformer.forward.split.query_pos",),
        ),
        Node(
            "query_slice",
            "Slice",
            ("batched_embedding",),
            ("query",),
            {
                "starts": [channels],
                "ends": [channels * 2],
                "axes": [-1],
                "steps": [1],
            },
            source_nodes=("PerceptionTransformer.forward.split.query",),
        ),
        Node(
            "reference_matmul",
            "MatMul",
            ("query_pos", "reference_weight"),
            ("reference_mm",),
            source_nodes=("PerceptionTransformer.reference_points",),
        ),
        Node(
            "reference_bias",
            "Add",
            ("reference_mm", "reference_bias"),
            ("reference_logits",),
            source_nodes=("PerceptionTransformer.reference_points",),
        ),
        Node(
            "reference_sigmoid",
            "Sigmoid",
            ("reference_logits",),
            ("reference_points",),
            source_nodes=("PerceptionTransformer.forward.reference_points.sigmoid",),
        ),
    ]
    graph = Graph(
        name=name,
        inputs={},
        outputs=("query", "query_pos", "reference_points"),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-decoder-query-initialization",
            "profile": {
                "batch": config.batch,
                "queries": queries,
                "embed_dim": channels,
            },
            "embedding_split_order": ["query_pos", "query"],
            "reference_dimensions": ["x", "y", "z"],
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


__all__ = [
    "DecoderQueryInitConfig",
    "build_decoder_query_init_graph",
    "decoder_query_init_weight_shapes",
]
