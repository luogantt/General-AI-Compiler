"""Integrated cold-start BEVFormer-tiny image-to-raw-detections graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bevformer_decoder_pipeline import (
    BevFormerDecoderPipelineConfig,
    build_bevformer_decoder_pipeline_graph,
    decoder_pipeline_weight_shapes,
)
from .bevformer_encoder_inputs import encoder_input_weight_shapes
from .bevformer_encoder_layer import encoder_layer_weight_shapes
from .bevformer_encoder_pipeline import (
    BevFormerColdEncoderPipelineConfig,
    build_bevformer_cold_encoder_pipeline_graph,
)
from .bevformer_feature_prep import feature_prep_weight_shapes
from .compose import GraphComposer
from .errors import ValidationError
from .frontends.bevformer_backbone import required_checkpoint_specs
from .ir import Graph, TensorSpec


_ENCODER_WEIGHT_KEYS = frozenset(
    {
        "backbone",
        "feature_prep",
        "encoder_inputs",
        "temporal",
        "spatial",
        "layer",
    }
)
_DECODER_WEIGHT_KEYS = frozenset(
    {
        "query_init",
        "self_attention",
        "cross_attention",
        "layer",
        "refinement",
        "detection",
    }
)


@dataclass(frozen=True, slots=True)
class BevFormerColdPipelineConfig:
    """Consistent static profiles for a complete cold-frame core."""

    encoder: BevFormerColdEncoderPipelineConfig = field(
        default_factory=BevFormerColdEncoderPipelineConfig
    )
    decoder: BevFormerDecoderPipelineConfig = field(
        default_factory=BevFormerDecoderPipelineConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, BevFormerColdEncoderPipelineConfig):
            raise ValidationError(
                "cold pipeline encoder must be a BevFormerColdEncoderPipelineConfig"
            )
        if not isinstance(self.decoder, BevFormerDecoderPipelineConfig):
            raise ValidationError(
                "cold pipeline decoder must be a BevFormerDecoderPipelineConfig"
            )
        cross = self.decoder.decoder.layer.cross_attention
        mismatches: list[str] = []
        if self.encoder.encoder_stack.batch != self.decoder.batch:
            mismatches.append("batch")
        if self.encoder.encoder_stack.embed_dim != self.decoder.embed_dim:
            mismatches.append("embed_dim")
        if self.encoder.encoder_stack.queries != cross.spatial_size:
            mismatches.append("encoder BEV queries/decoder BEV memory")
        if (
            self.encoder.encoder_stack.layer.temporal.bev_h != cross.bev_h
            or self.encoder.encoder_stack.layer.temporal.bev_w != cross.bev_w
        ):
            mismatches.append("encoder/decoder BEV shape")
        if mismatches:
            raise ValidationError(
                "cold pipeline component profiles disagree on: "
                + ", ".join(mismatches)
            )


def cold_pipeline_weight_shapes(
    config: BevFormerColdPipelineConfig,
) -> dict[str, Any]:
    """Return the complete nested learned-tensor inventory."""

    if not isinstance(config, BevFormerColdPipelineConfig):
        raise ValidationError("config must be a BevFormerColdPipelineConfig")
    encoder_layer_shapes = encoder_layer_weight_shapes(
        config.encoder.encoder_stack.layer
    )
    return {
        "encoder": {
            "backbone": {
                spec.key: spec.shape
                for spec in required_checkpoint_specs(config.encoder.backbone)
            },
            "feature_prep": feature_prep_weight_shapes(
                config.encoder.feature_prep
            ),
            "encoder_inputs": encoder_input_weight_shapes(
                config.encoder.encoder_inputs
            ),
            "temporal": tuple(
                dict(encoder_layer_shapes["temporal"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
            "spatial": tuple(
                dict(encoder_layer_shapes["spatial"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
            "layer": tuple(
                dict(encoder_layer_shapes["layer"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
        },
        "decoder": decoder_pipeline_weight_shapes(config.decoder),
    }


def _weight_group(
    scope: str,
    weights: Mapping[str, Any],
    expected: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(weights, Mapping):
        raise ValidationError(f"cold pipeline {scope} weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != expected:
        raise ValidationError(
            f"cold pipeline {scope} weight groups must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    return weights


def build_bevformer_cold_pipeline_graph(
    encoder_weights: Mapping[str, Any],
    decoder_weights: Mapping[str, Any],
    config: BevFormerColdPipelineConfig | None = None,
    *,
    name: str = "bevformer_cold_image_to_raw_detections",
) -> Graph:
    """Compose Backbone, three-layer Encoder, six-layer Decoder, and raw head."""

    config = config or BevFormerColdPipelineConfig()
    if not isinstance(config, BevFormerColdPipelineConfig):
        raise ValidationError("config must be a BevFormerColdPipelineConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("cold pipeline graph name must be non-empty")
    encoder_weights = _weight_group(
        "encoder", encoder_weights, _ENCODER_WEIGHT_KEYS
    )
    decoder_weights = _weight_group(
        "decoder", decoder_weights, _DECODER_WEIGHT_KEYS
    )

    encoder_graph = build_bevformer_cold_encoder_pipeline_graph(
        encoder_weights["backbone"],
        encoder_weights["feature_prep"],
        encoder_weights["encoder_inputs"],
        encoder_weights["temporal"],
        encoder_weights["spatial"],
        encoder_weights["layer"],
        config.encoder,
    )
    decoder_graph = build_bevformer_decoder_pipeline_graph(
        decoder_weights["query_init"],
        decoder_weights["self_attention"],
        decoder_weights["cross_attention"],
        decoder_weights["layer"],
        decoder_weights["refinement"],
        decoder_weights["detection"],
        config.decoder,
    )

    projection = config.encoder.encoder_stack.layer.projection
    encoder_inputs = config.encoder.encoder_inputs
    composer = GraphComposer(
        name,
        {
            "images": TensorSpec(
                (
                    config.encoder.feature_prep.batch,
                    config.encoder.feature_prep.cameras,
                    config.encoder.backbone.input_channels,
                    *config.encoder.backbone.input_hw,
                ),
                config.encoder.backbone.input_dtype,
            ),
            "lidar2img": TensorSpec(
                (
                    config.encoder.encoder_stack.batch,
                    projection.cameras,
                    4,
                    4,
                ),
                "float32",
            ),
            "can_bus_delta": TensorSpec(
                (encoder_inputs.batch, encoder_inputs.can_bus_dim), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-cold-image-to-raw-detections-static",
            "cold_single_frame_core_connected": True,
            "full_encoder_stack": True,
            "full_decoder_stack": True,
            "full_raw_detection_head": True,
            "box_representation": (
                "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
                "sin-yaw,cos-yaw,vx,vy]; decode_nms_free converts size/yaw"
            ),
            "num_encoder_layers": config.encoder.encoder_stack.num_layers,
            "num_decoder_layers": config.decoder.decoder.num_layers,
            "compiled_target": "cuda-sm86",
            "postprocess": (
                "explicit host bevformer_postprocess.decode_nms_free until "
                "TopK/Gather CUDA lowering is added"
            ),
            "not_a_streaming_step": True,
            "excluded_stages": [
                "warm-history-prev-bev-rotation-and-shift",
                "stream-state-selection",
                "in-graph-top-k-and-variable-length-record-formatting",
            ],
            "profile": {
                "batch": config.decoder.batch,
                "cameras": projection.cameras,
                "image_shape": list(config.encoder.backbone.input_hw),
                "bev_shape": [projection.bev_h, projection.bev_w],
                "object_queries": config.decoder.queries,
                "embed_dim": config.decoder.embed_dim,
                "classes": config.decoder.classes,
                "box_code_size": config.decoder.code_size,
            },
            "image_input_layout": "B-camera-C-H-W",
        },
    )
    encoded = composer.inline(
        encoder_graph,
        prefix="image_to_bev",
        bindings={
            "images": "images",
            "lidar2img": "lidar2img",
            "can_bus_delta": "can_bus_delta",
        },
        output_bindings={"next_bev": "next_bev"},
    )
    decoded = composer.inline(
        decoder_graph,
        prefix="bev_to_detections",
        bindings={"value": encoded["next_bev"]},
        output_bindings={
            "all_class_logits": "all_class_logits",
            "all_class_scores": "all_class_scores",
            "all_boxes": "all_boxes",
            "final_class_logits": "final_class_logits",
            "final_class_scores": "final_class_scores",
            "final_boxes": "final_boxes",
            "initial_reference_points": "initial_reference_points",
            "intermediate_reference_points": "intermediate_reference_points",
        },
    )
    return composer.build(
        (
            decoded["all_class_logits"],
            decoded["all_class_scores"],
            decoded["all_boxes"],
            decoded["final_class_logits"],
            decoded["final_class_scores"],
            decoded["final_boxes"],
            encoded["next_bev"],
            decoded["initial_reference_points"],
            decoded["intermediate_reference_points"],
        )
    )


def validate_cold_pipeline_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the cold image-to-detections runtime input ABI."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping):
        raise ValidationError("cold pipeline inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValidationError(
            "cold pipeline input names must be non-empty strings"
        )
    if set(values) != set(graph.inputs):
        raise ValidationError(
            f"cold pipeline input names must be {sorted(graph.inputs)}, "
            f"got {sorted(values)}"
        )

    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"cold pipeline input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"cold pipeline input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous:
            raise ValidationError(
                f"cold pipeline input {name!r} must be C-contiguous"
            )
        if not np.isfinite(value).all():
            raise ValidationError(
                f"cold pipeline input {name!r} must contain only finite values"
            )


__all__ = [
    "BevFormerColdPipelineConfig",
    "build_bevformer_cold_pipeline_graph",
    "cold_pipeline_weight_shapes",
    "validate_cold_pipeline_inputs",
]
