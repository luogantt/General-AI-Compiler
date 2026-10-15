"""Prepare one BEVFormer-tiny FPN level for spatial cross-attention.

The official transformer receives an NCHW feature for each camera, flattens
the spatial axes, moves channels to the last axis, and adds learned camera and
feature-level embeddings.  Keeping that transformation in the canonical graph
removes a framework-side tensor preparation step from the compiled boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class BevFormerFeaturePrepConfig:
    """Static single-level feature profile used by BEVFormer-tiny."""

    batch: int = 1
    cameras: int = 6
    embed_dim: int = 256
    feature_h: int = 15
    feature_w: int = 25

    def __post_init__(self) -> None:
        for name in ("batch", "cameras", "embed_dim", "feature_h", "feature_w"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"feature preparation {name} must be positive")
        if self.batch != 1:
            raise ValidationError("feature preparation currently requires batch=1")

    @property
    def backbone_batch(self) -> int:
        return self.batch * self.cameras

    @property
    def spatial_size(self) -> int:
        return self.feature_h * self.feature_w


def feature_prep_weight_shapes(
    config: BevFormerFeaturePrepConfig,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(config, BevFormerFeaturePrepConfig):
        raise ValidationError("config must be a BevFormerFeaturePrepConfig")
    return {
        "camera_embeddings": (config.cameras, config.embed_dim),
        "level_embedding": (1, config.embed_dim),
    }


def _validated_weights(
    config: BevFormerFeaturePrepConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = feature_prep_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("feature preparation weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != set(expected):
        raise ValidationError(
            f"feature preparation weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"feature preparation weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"feature preparation weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"feature preparation weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    return result


def build_bevformer_feature_prep_graph(
    weights: Mapping[str, np.ndarray],
    config: BevFormerFeaturePrepConfig | None = None,
    *,
    name: str = "bevformer_single_level_feature_preparation",
) -> Graph:
    """Lower FPN NCHW output to ``[B, camera, HW, C]`` encoder memory."""

    config = config or BevFormerFeaturePrepConfig()
    if not isinstance(config, BevFormerFeaturePrepConfig):
        raise ValidationError("config must be a BevFormerFeaturePrepConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("feature preparation graph name must be non-empty")
    constants = _validated_weights(config, weights)
    b, cameras, channels = config.batch, config.cameras, config.embed_dim
    height, width, spatial = config.feature_h, config.feature_w, config.spatial_size
    constants["camera_embeddings_bc"] = constants.pop("camera_embeddings").reshape(
        1, cameras, 1, channels
    )
    constants["level_embedding_bc"] = constants.pop("level_embedding").reshape(
        1, 1, 1, channels
    )
    nodes = [
        Node(
            "restore_batch_camera_axes",
            "Reshape",
            ("fpn_feature",),
            ("feature_bcc_hw",),
            {"shape": [b, cameras, channels, height, width]},
            source_nodes=("PerceptionTransformer.get_bev_features.mlvl_feats[0]",),
        ),
        Node(
            "channels_last",
            "Transpose",
            ("feature_bcc_hw",),
            ("feature_bchw_c",),
            {"perm": [0, 1, 3, 4, 2]},
            source_nodes=("PerceptionTransformer.get_bev_features.permute",),
        ),
        Node(
            "flatten_spatial_axes",
            "Reshape",
            ("feature_bchw_c",),
            ("feature_bcsc",),
            {"shape": [b, cameras, spatial, channels]},
            source_nodes=("PerceptionTransformer.get_bev_features.flatten",),
        ),
        Node(
            "add_camera_embeddings",
            "Add",
            ("feature_bcsc", "camera_embeddings_bc"),
            ("camera_positioned",),
            source_nodes=("PerceptionTransformer.cams_embeds",),
        ),
        Node(
            "add_level_embedding",
            "Add",
            ("camera_positioned", "level_embedding_bc"),
            ("image_value",),
            source_nodes=("PerceptionTransformer.level_embeds",),
        ),
    ]
    graph = Graph(
        name=name,
        inputs={
            "fpn_feature": TensorSpec(
                (config.backbone_batch, channels, height, width), "float32"
            )
        },
        outputs=("image_value",),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-single-level-feature-preparation",
            "source_layout": "(batch*camera)-C-H-W",
            "target_layout": "B-camera-HW-C",
            "feature_levels": 1,
            "camera_embeddings": True,
            "level_embeddings": True,
            "profile": {
                "batch": b,
                "cameras": cameras,
                "embed_dim": channels,
                "feature_shape": [height, width],
            },
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_feature_prep_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        raise ValidationError("feature preparation input names do not match the graph")
    value = values["fpn_feature"]
    spec = graph.inputs["fpn_feature"]
    if not isinstance(value, np.ndarray):
        raise ValidationError("feature preparation input must be a NumPy ndarray")
    if value.shape != spec.shape or value.dtype.name != spec.dtype:
        raise ValidationError(
            f"feature preparation input requires {spec.shape}/{spec.dtype}, "
            f"got {value.shape}/{value.dtype}"
        )
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValidationError("feature preparation input must be contiguous and finite")


__all__ = [
    "BevFormerFeaturePrepConfig",
    "build_bevformer_feature_prep_graph",
    "feature_prep_weight_shapes",
    "validate_feature_prep_inputs",
]
