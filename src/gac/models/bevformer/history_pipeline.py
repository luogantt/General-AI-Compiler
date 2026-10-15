"""Warm-history image-to-BEV encoder pipeline for BEVFormer-tiny.

The graph deliberately starts after motion compensation.  Callers provide an
already aligned previous BEV and its normalized reference points.  The graph
then constructs the official shared temporal queue
``[aligned previous BEV, initial current BEV query]`` exactly once and reuses
it in all three encoder layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gac.models.bevformer.encoder_inputs import (
    BevFormerEncoderInputConfig,
    build_bevformer_encoder_input_graph,
    encoder_input_weight_shapes,
)
from gac.models.bevformer.encoder_layer import encoder_layer_weight_shapes
from gac.models.bevformer.encoder_stack import (
    BevFormerEncoderStackConfig,
    build_bevformer_encoder_stack_graph,
)
from gac.models.bevformer.feature_prep import (
    BevFormerFeaturePrepConfig,
    build_bevformer_feature_prep_graph,
    feature_prep_weight_shapes,
)
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.models.bevformer.frontends.backbone import (
    BEVFORMER_R50_FPN_PROFILE,
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
    required_checkpoint_specs,
)
from gac.core.ir import Graph, Node, TensorSpec


def _default_history_encoder_stack() -> BevFormerEncoderStackConfig:
    return BevFormerEncoderStackConfig(history_mode=True)


@dataclass(frozen=True, slots=True)
class BevFormerHistoryPipelineConfig:
    """Mutually consistent image, query, and history encoder profiles."""

    backbone: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE
    feature_prep: BevFormerFeaturePrepConfig = field(
        default_factory=BevFormerFeaturePrepConfig
    )
    encoder_inputs: BevFormerEncoderInputConfig = field(
        default_factory=BevFormerEncoderInputConfig
    )
    encoder_stack: BevFormerEncoderStackConfig = field(
        default_factory=_default_history_encoder_stack
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, ResNetFpnProfile):
            raise ValidationError("history pipeline backbone profile is invalid")
        if not isinstance(self.feature_prep, BevFormerFeaturePrepConfig):
            raise ValidationError("history pipeline feature_prep profile is invalid")
        if not isinstance(self.encoder_inputs, BevFormerEncoderInputConfig):
            raise ValidationError("history pipeline encoder_inputs profile is invalid")
        if not isinstance(self.encoder_stack, BevFormerEncoderStackConfig):
            raise ValidationError("history pipeline encoder_stack profile is invalid")
        if not self.encoder_stack.history_mode:
            raise ValidationError("history pipeline requires history_mode=True")

        prep = self.feature_prep
        inputs = self.encoder_inputs
        stack = self.encoder_stack
        layer = stack.layer
        temporal = layer.temporal
        spatial = layer.spatial
        projection = layer.projection
        mismatches: list[str] = []
        if self.backbone.batch != prep.backbone_batch:
            mismatches.append("backbone batch versus batch*cameras")
        if self.backbone.fpn_channels != prep.embed_dim:
            mismatches.append("backbone/feature embed_dim")
        if self.backbone.output_hw != (prep.feature_h, prep.feature_w):
            mismatches.append("backbone/feature spatial shape")
        if self.backbone.input_hw != (projection.image_h, projection.image_w):
            mismatches.append("backbone/projection image shape")
        if prep.batch != inputs.batch or inputs.batch != stack.batch:
            mismatches.append("feature/input/stack batch")
        if prep.cameras != spatial.cameras:
            mismatches.append("feature/stack cameras")
        if prep.embed_dim != inputs.embed_dim or inputs.embed_dim != stack.embed_dim:
            mismatches.append("feature/input/stack embed_dim")
        if spatial.spatial_shapes != ((prep.feature_h, prep.feature_w),):
            mismatches.append("feature/stack level shape")
        if (inputs.bev_h, inputs.bev_w) != (temporal.bev_h, temporal.bev_w):
            mismatches.append("input/stack BEV shape")
        if inputs.queries != stack.queries:
            mismatches.append("input/stack BEV queries")
        if temporal.levels != 1:
            mismatches.append("encoder input reference levels")
        if mismatches:
            raise ValidationError(
                "history pipeline component profiles disagree on: "
                + ", ".join(mismatches)
            )


def history_pipeline_weight_shapes(
    config: BevFormerHistoryPipelineConfig,
) -> dict[str, Any]:
    """Return the complete grouped learned-tensor inventory."""

    if not isinstance(config, BevFormerHistoryPipelineConfig):
        raise ValidationError("config must be a BevFormerHistoryPipelineConfig")
    layer_shapes = encoder_layer_weight_shapes(config.encoder_stack.layer)
    return {
        "backbone": {
            spec.key: spec.shape for spec in required_checkpoint_specs(config.backbone)
        },
        "feature_prep": feature_prep_weight_shapes(config.feature_prep),
        "encoder_inputs": encoder_input_weight_shapes(config.encoder_inputs),
        "temporal": tuple(
            dict(layer_shapes["temporal"])
            for _ in range(config.encoder_stack.num_layers)
        ),
        "spatial": tuple(
            dict(layer_shapes["spatial"])
            for _ in range(config.encoder_stack.num_layers)
        ),
        "layer": tuple(
            dict(layer_shapes["layer"])
            for _ in range(config.encoder_stack.num_layers)
        ),
    }


def build_bevformer_history_pipeline_graph(
    backbone_weights: Mapping[str, np.ndarray],
    feature_prep_weights: Mapping[str, np.ndarray],
    encoder_input_weights: Mapping[str, np.ndarray],
    temporal_weights: Sequence[Mapping[str, np.ndarray]],
    spatial_weights: Sequence[Mapping[str, np.ndarray]],
    layer_weights: Sequence[Mapping[str, np.ndarray]],
    config: BevFormerHistoryPipelineConfig | None = None,
    *,
    name: str = "bevformer_history_image_to_bev_encoder",
) -> Graph:
    """Compose images and an externally aligned previous BEV into next BEV."""

    config = config or BevFormerHistoryPipelineConfig()
    if not isinstance(config, BevFormerHistoryPipelineConfig):
        raise ValidationError("config must be a BevFormerHistoryPipelineConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("history pipeline graph name must be non-empty")

    backbone = build_bevformer_backbone_graph(backbone_weights, config.backbone)
    feature_prep = build_bevformer_feature_prep_graph(
        feature_prep_weights, config.feature_prep
    )
    encoder_inputs = build_bevformer_encoder_input_graph(
        encoder_input_weights, config.encoder_inputs
    )
    encoder = build_bevformer_encoder_stack_graph(
        temporal_weights,
        spatial_weights,
        layer_weights,
        config.encoder_stack,
    )

    stack = config.encoder_stack
    layer = stack.layer
    levels = layer.temporal.levels
    composer = GraphComposer(
        name,
        {
            "images": TensorSpec(
                (
                    config.feature_prep.batch,
                    config.feature_prep.cameras,
                    config.backbone.input_channels,
                    *config.backbone.input_hw,
                ),
                config.backbone.input_dtype,
            ),
            "lidar2img": TensorSpec(
                (stack.batch, layer.projection.cameras, 4, 4), "float32"
            ),
            "can_bus_delta": TensorSpec(
                (config.encoder_inputs.batch, config.encoder_inputs.can_bus_dim),
                "float32",
            ),
            "aligned_prev_bev": TensorSpec(
                (stack.batch, stack.queries, stack.embed_dim), "float32"
            ),
            "prev_reference_points": TensorSpec(
                (stack.batch, stack.queries, levels, 2), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-history-image-to-bev-encoder-pipeline",
            "full_encoder_stack": True,
            "history_mode": True,
            "num_encoder_layers": stack.num_layers,
            "image_input_layout": "B-camera-C-H-W",
            "temporal_value_contract": (
                "one shared [aligned previous BEV, initial current query] queue"
            ),
            "history_alignment": (
                "external; aligned_prev_bev and prev_reference_points are runtime inputs"
            ),
            "stages": [
                "resnet50-fpn",
                "camera-and-level-feature-preparation",
                "learned-bev-query-and-position-preparation",
                "shared-history-queue-construction",
                "three-layer-bevformer-history-encoder",
            ],
            "profile": {
                "backbone": config.backbone.name,
                "batch": stack.batch,
                "cameras": config.feature_prep.cameras,
                "image_shape": list(config.backbone.input_hw),
                "feature_shape": [
                    config.feature_prep.feature_h,
                    config.feature_prep.feature_w,
                ],
                "bev_shape": [
                    config.encoder_inputs.bev_h,
                    config.encoder_inputs.bev_w,
                ],
                "embed_dim": stack.embed_dim,
            },
        },
    )
    composer.add_node(
        Node(
            "flatten_image_camera_batch",
            "Reshape",
            ("images",),
            ("backbone_images",),
            {"shape": list(config.backbone.input_shape)},
            source_nodes=("BEVFormer.extract_img_feat.flatten_batch_camera",),
        )
    )
    backbone_output = composer.inline(
        backbone,
        prefix="backbone",
        bindings={"images": "backbone_images"},
        output_bindings={"fpn_c5": "fpn_feature"},
    )
    prepared = composer.inline(
        feature_prep,
        prefix="feature_prep",
        bindings={"fpn_feature": backbone_output["fpn_c5"]},
        output_bindings={"image_value": "image_value"},
    )
    initialized = composer.inline(
        encoder_inputs,
        prefix="encoder_inputs",
        bindings={"can_bus_delta": "can_bus_delta"},
        output_bindings={
            "query": "initial_bev_query",
            "query_pos": "bev_query_pos",
            "spatial_query_pos": "spatial_query_pos",
            "reference_points": "bev_reference_points",
        },
    )
    composer.add_node(
        Node(
            "build_shared_temporal_queue",
            "Concat",
            ("aligned_prev_bev", initialized["query"]),
            ("shared_temporal_value",),
            {"axis": 0},
            source_nodes=("BEVFormerEncoder.forward.prev_bev_queue",),
        )
    )
    encoded = composer.inline(
        encoder,
        prefix="encoder",
        bindings={
            "query": initialized["query"],
            "query_pos": initialized["query_pos"],
            "spatial_query_pos": initialized["spatial_query_pos"],
            "reference_points": initialized["reference_points"],
            "image_value": prepared["image_value"],
            "lidar2img": "lidar2img",
            "temporal_value": "shared_temporal_value",
            "prev_reference_points": "prev_reference_points",
        },
        output_bindings={"output": "next_bev"},
    )
    return composer.build((encoded["output"],))


def validate_history_pipeline_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the explicit warm-history runtime ABI."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping):
        raise ValidationError("history pipeline inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValidationError(
            "history pipeline input names must be non-empty strings"
        )
    if set(values) != set(graph.inputs):
        raise ValidationError(
            f"history pipeline input names must be {sorted(graph.inputs)}, "
            f"got {sorted(repr(name) for name in values)}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"history pipeline input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"history pipeline input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous:
            raise ValidationError(
                f"history pipeline input {name!r} must be C-contiguous"
            )
        if not np.isfinite(value).all():
            raise ValidationError(
                f"history pipeline input {name!r} must contain only finite values"
            )

    # Ego-motion shift is added to the normalized base grid before temporal
    # attention. Border cells may therefore legitimately lie outside [0, 1];
    # MSDA applies zero padding to those samples.


__all__ = [
    "BevFormerHistoryPipelineConfig",
    "build_bevformer_history_pipeline_graph",
    "history_pipeline_weight_shapes",
    "validate_history_pipeline_inputs",
]
