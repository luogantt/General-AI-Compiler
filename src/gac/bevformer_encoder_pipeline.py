"""Cold-start image-to-BEV encoder pipeline for the frozen tiny profile."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .bevformer_encoder_inputs import (
    BevFormerEncoderInputConfig,
    build_bevformer_encoder_input_graph,
)
from .bevformer_encoder_stack import (
    BevFormerEncoderStackConfig,
    build_bevformer_encoder_stack_graph,
)
from .bevformer_feature_prep import (
    BevFormerFeaturePrepConfig,
    build_bevformer_feature_prep_graph,
)
from .compose import GraphComposer
from .errors import ValidationError
from .frontends.bevformer_backbone import (
    BEVFORMER_R50_FPN_PROFILE,
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
)
from .ir import Graph, Node, TensorSpec


def _default_cold_encoder_stack() -> BevFormerEncoderStackConfig:
    return BevFormerEncoderStackConfig(history_mode=False)


@dataclass(frozen=True, slots=True)
class BevFormerColdEncoderPipelineConfig:
    """Mutually consistent Backbone, adapter, and three-layer encoder profiles."""

    backbone: ResNetFpnProfile = BEVFORMER_R50_FPN_PROFILE
    feature_prep: BevFormerFeaturePrepConfig = field(
        default_factory=BevFormerFeaturePrepConfig
    )
    encoder_inputs: BevFormerEncoderInputConfig = field(
        default_factory=BevFormerEncoderInputConfig
    )
    encoder_stack: BevFormerEncoderStackConfig = field(
        default_factory=_default_cold_encoder_stack
    )

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, ResNetFpnProfile):
            raise ValidationError("encoder pipeline backbone profile is invalid")
        if not isinstance(self.feature_prep, BevFormerFeaturePrepConfig):
            raise ValidationError("encoder pipeline feature_prep profile is invalid")
        if not isinstance(self.encoder_inputs, BevFormerEncoderInputConfig):
            raise ValidationError("encoder pipeline encoder_inputs profile is invalid")
        if not isinstance(self.encoder_stack, BevFormerEncoderStackConfig):
            raise ValidationError("encoder pipeline encoder_stack profile is invalid")
        if self.encoder_stack.history_mode:
            raise ValidationError("cold encoder pipeline requires history_mode=False")

        prep = self.feature_prep
        inputs = self.encoder_inputs
        stack = self.encoder_stack
        spatial = stack.layer.spatial
        mismatches: list[str] = []
        if self.backbone.batch != prep.backbone_batch:
            mismatches.append("backbone batch versus batch*cameras")
        if self.backbone.fpn_channels != prep.embed_dim:
            mismatches.append("backbone/feature embed_dim")
        if self.backbone.output_hw != (prep.feature_h, prep.feature_w):
            mismatches.append("backbone/feature spatial shape")
        if inputs.batch != stack.batch:
            mismatches.append("encoder input/stack batch")
        if inputs.queries != stack.queries:
            mismatches.append("encoder input/stack BEV shape")
        if inputs.embed_dim != stack.embed_dim:
            mismatches.append("encoder input/stack embed_dim")
        if prep.batch != stack.batch or prep.cameras != spatial.cameras:
            mismatches.append("feature/stack batch or cameras")
        if prep.embed_dim != spatial.embed_dim:
            mismatches.append("feature/stack embed_dim")
        if spatial.spatial_shapes != ((prep.feature_h, prep.feature_w),):
            mismatches.append("feature/stack level shape")
        if mismatches:
            raise ValidationError(
                "encoder pipeline component profiles disagree on: "
                + ", ".join(mismatches)
            )


def build_bevformer_cold_encoder_pipeline_graph(
    backbone_weights: Mapping[str, np.ndarray],
    feature_prep_weights: Mapping[str, np.ndarray],
    encoder_input_weights: Mapping[str, np.ndarray],
    temporal_weights: Sequence[Mapping[str, np.ndarray]],
    spatial_weights: Sequence[Mapping[str, np.ndarray]],
    layer_weights: Sequence[Mapping[str, np.ndarray]],
    config: BevFormerColdEncoderPipelineConfig | None = None,
    *,
    name: str = "bevformer_cold_image_to_bev_encoder",
) -> Graph:
    """Compose images, learned inputs, and all three encoder layers."""

    config = config or BevFormerColdEncoderPipelineConfig()
    if not isinstance(config, BevFormerColdEncoderPipelineConfig):
        raise ValidationError("config must be a BevFormerColdEncoderPipelineConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("encoder pipeline graph name must be non-empty")

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

    stack_layer = config.encoder_stack.layer
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
                (
                    config.encoder_stack.batch,
                    stack_layer.projection.cameras,
                    4,
                    4,
                ),
                "float32",
            ),
            "can_bus_delta": TensorSpec(
                (
                    config.encoder_inputs.batch,
                    config.encoder_inputs.can_bus_dim,
                ),
                "float32",
            ),
        },
        metadata={
            "kind": "bevformer-cold-image-to-bev-encoder-pipeline",
            "full_encoder_stack": True,
            "cold_start_only": True,
            "history_state": "outside-this-graph",
            "stages": [
                "resnet50-fpn",
                "camera-and-level-feature-preparation",
                "learned-bev-query-and-position-preparation",
                "three-layer-bevformer-encoder",
            ],
            "profile": {
                "backbone": config.backbone.name,
                "cameras": config.feature_prep.cameras,
                "feature_shape": [
                    config.feature_prep.feature_h,
                    config.feature_prep.feature_w,
                ],
                "bev_shape": [
                    config.encoder_inputs.bev_h,
                    config.encoder_inputs.bev_w,
                ],
                "embed_dim": config.encoder_inputs.embed_dim,
            },
            "image_input_layout": "B-camera-C-H-W",
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
        },
        output_bindings={"output": "next_bev"},
    )
    return composer.build((encoded["output"],))


__all__ = [
    "BevFormerColdEncoderPipelineConfig",
    "build_bevformer_cold_encoder_pipeline_graph",
]
