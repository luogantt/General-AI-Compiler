"""Complete stateful BEVFormer-tiny image-to-raw-detections step graph.

The graph has one stable ABI for both the first frame and later frames.  A
``has_prev_bev`` flag selects either the learned initial BEV query/reference
grid or the motion-compensated previous state.  Warm execution shares the
official ``[previous BEV, initial query]`` queue across all encoder layers;
cold execution constructs ``[current query, current query]`` inside each
layer.  The encoded BEV is fed to
the complete six-layer decoder and raw detection head; NMS-free Top-K decoding
remains an explicit host boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from .bevformer_decoder_pipeline import (
    BevFormerDecoderPipelineConfig,
    build_bevformer_decoder_pipeline_graph,
    decoder_pipeline_weight_shapes,
)
from .bevformer_encoder_inputs import (
    build_bevformer_encoder_input_graph,
    encoder_input_weight_shapes,
)
from .bevformer_encoder_layer import encoder_layer_weight_shapes
from .bevformer_encoder_stack import build_bevformer_encoder_stack_graph
from .bevformer_feature_prep import (
    build_bevformer_feature_prep_graph,
    feature_prep_weight_shapes,
)
from .bevformer_history_pipeline import BevFormerHistoryPipelineConfig
from .bevformer_motion import BevFormerMotionConfig, build_bevformer_motion_graph
from .compose import GraphComposer
from .errors import ValidationError
from .frontends.bevformer_backbone import (
    build_bevformer_backbone_graph,
    required_checkpoint_specs,
)
from .ir import Graph, Node, TensorSpec


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
class BevFormerStepPipelineConfig:
    """Mutually consistent encoder, motion, decoder, and state profiles."""

    encoder: BevFormerHistoryPipelineConfig = field(
        default_factory=BevFormerHistoryPipelineConfig
    )
    motion: BevFormerMotionConfig = field(default_factory=BevFormerMotionConfig)
    decoder: BevFormerDecoderPipelineConfig = field(
        default_factory=BevFormerDecoderPipelineConfig
    )
    state_dtype: str = "float16"

    def __post_init__(self) -> None:
        if not isinstance(self.encoder, BevFormerHistoryPipelineConfig):
            raise ValidationError(
                "step pipeline encoder must be a BevFormerHistoryPipelineConfig"
            )
        if not isinstance(self.motion, BevFormerMotionConfig):
            raise ValidationError(
                "step pipeline motion must be a BevFormerMotionConfig"
            )
        if not isinstance(self.decoder, BevFormerDecoderPipelineConfig):
            raise ValidationError(
                "step pipeline decoder must be a BevFormerDecoderPipelineConfig"
            )
        if self.state_dtype != "float16":
            raise ValidationError("step pipeline state_dtype must be float16")
        if self.encoder.backbone.input_dtype != "float16":
            raise ValidationError("step pipeline image input dtype must be float16")

        stack = self.encoder.encoder_stack
        inputs = self.encoder.encoder_inputs
        projection = stack.layer.projection
        cross = self.decoder.decoder.layer.cross_attention
        motion = self.motion
        mismatches: list[str] = []
        if not stack.history_mode:
            mismatches.append("encoder history_mode")
        if stack.batch != motion.batch or stack.batch != self.decoder.batch:
            mismatches.append("batch")
        if stack.embed_dim != motion.embed_dim or stack.embed_dim != self.decoder.embed_dim:
            mismatches.append("embed_dim")
        if (inputs.bev_h, inputs.bev_w) != (motion.bev_h, motion.bev_w):
            mismatches.append("encoder input/motion BEV shape")
        if (
            stack.layer.temporal.bev_h,
            stack.layer.temporal.bev_w,
        ) != (motion.bev_h, motion.bev_w):
            mismatches.append("encoder stack/motion BEV shape")
        if (cross.bev_h, cross.bev_w) != (motion.bev_h, motion.bev_w):
            mismatches.append("decoder/motion BEV shape")
        if cross.spatial_size != stack.queries:
            mismatches.append("encoder BEV queries/decoder BEV memory")
        if inputs.can_bus_dim != motion.can_bus_dim:
            mismatches.append("CAN-bus dimension")
        if motion.can_bus_dim != 18:
            mismatches.append("official 18-element CAN-bus contract")
        if self.decoder.code_size != 10:
            mismatches.append("official 10-channel box code")

        pc = tuple(float(value) for value in projection.point_cloud_range)
        x_extent = pc[3] - pc[0]
        y_extent = pc[4] - pc[1]
        if not math.isclose(
            motion.grid_length_x * motion.bev_w,
            x_extent,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            mismatches.append("motion grid_length_x/point-cloud x extent")
        if not math.isclose(
            motion.grid_length_y * motion.bev_h,
            y_extent,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            mismatches.append("motion grid_length_y/point-cloud y extent")
        detection_pc = tuple(
            float(value)
            for value in self.decoder.detection.head.point_cloud_range
        )
        if any(
            not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
            for left, right in zip(pc, detection_pc, strict=True)
        ):
            mismatches.append("encoder/detection point-cloud range")
        if mismatches:
            raise ValidationError(
                "step pipeline component profiles disagree on: "
                + ", ".join(mismatches)
            )


def step_pipeline_weight_shapes(
    config: BevFormerStepPipelineConfig,
) -> dict[str, Any]:
    """Return the complete grouped learned-tensor inventory."""

    if not isinstance(config, BevFormerStepPipelineConfig):
        raise ValidationError("config must be a BevFormerStepPipelineConfig")
    layer_shapes = encoder_layer_weight_shapes(config.encoder.encoder_stack.layer)
    decoder_shapes = decoder_pipeline_weight_shapes(config.decoder)
    decoder_stack_shapes = decoder_shapes["decoder"]
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
                dict(layer_shapes["temporal"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
            "spatial": tuple(
                dict(layer_shapes["spatial"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
            "layer": tuple(
                dict(layer_shapes["layer"])
                for _ in range(config.encoder.encoder_stack.num_layers)
            ),
        },
        # Mirror the exact mapping accepted by
        # ``build_bevformer_step_pipeline_graph``.  The lower-level decoder
        # inventory groups its five transformer stages under ``decoder``;
        # the complete step ABI exposes those groups beside ``detection`` so
        # callers can materialize the returned inventory without a second
        # schema translation.
        "decoder": {
            "query_init": decoder_stack_shapes["query_init"],
            "self_attention": decoder_stack_shapes["self_attention"],
            "cross_attention": decoder_stack_shapes["cross_attention"],
            "layer": decoder_stack_shapes["layer"],
            "refinement": decoder_stack_shapes["refinement"],
            "detection": decoder_shapes["detection"],
        },
    }


def _weight_group(
    scope: str,
    weights: Mapping[str, Any],
    expected: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(weights, Mapping):
        raise ValidationError(f"step pipeline {scope} weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != expected:
        raise ValidationError(
            f"step pipeline {scope} weight groups must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    return weights


def build_bevformer_step_pipeline_graph(
    encoder_weights: Mapping[str, Any],
    decoder_weights: Mapping[str, Any],
    config: BevFormerStepPipelineConfig | None = None,
    *,
    name: str = "bevformer_streaming_step",
) -> Graph:
    """Build one cold-or-warm frame into detections and the next BEV state."""

    config = config or BevFormerStepPipelineConfig()
    if not isinstance(config, BevFormerStepPipelineConfig):
        raise ValidationError("config must be a BevFormerStepPipelineConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("step pipeline graph name must be non-empty")
    encoder_weights = _weight_group(
        "encoder", encoder_weights, _ENCODER_WEIGHT_KEYS
    )
    decoder_weights = _weight_group(
        "decoder", decoder_weights, _DECODER_WEIGHT_KEYS
    )

    # Cast the public five-dimensional FP16 image tensor before flattening the
    # camera axis.  This keeps every layout-only node FP32 for the first CUDA
    # backend while remaining numerically identical to the backbone's normal
    # input cast.
    execution_backbone_profile = replace(
        config.encoder.backbone, input_dtype="float32"
    )
    backbone = build_bevformer_backbone_graph(
        encoder_weights["backbone"], execution_backbone_profile
    )
    feature_prep = build_bevformer_feature_prep_graph(
        encoder_weights["feature_prep"], config.encoder.feature_prep
    )
    encoder_inputs = build_bevformer_encoder_input_graph(
        encoder_weights["encoder_inputs"], config.encoder.encoder_inputs
    )
    motion = build_bevformer_motion_graph(config.motion)
    encoder = build_bevformer_encoder_stack_graph(
        encoder_weights["temporal"],
        encoder_weights["spatial"],
        encoder_weights["layer"],
        config.encoder.encoder_stack,
        select_history_at_runtime=True,
    )
    decoder = build_bevformer_decoder_pipeline_graph(
        decoder_weights["query_init"],
        decoder_weights["self_attention"],
        decoder_weights["cross_attention"],
        decoder_weights["layer"],
        decoder_weights["refinement"],
        decoder_weights["detection"],
        config.decoder,
    )

    history = config.encoder
    stack = history.encoder_stack
    projection = stack.layer.projection
    composer = GraphComposer(
        name,
        {
            "images": TensorSpec(
                (
                    history.feature_prep.batch,
                    history.feature_prep.cameras,
                    history.backbone.input_channels,
                    *history.backbone.input_hw,
                ),
                "float16",
            ),
            "lidar2img": TensorSpec(
                (stack.batch, projection.cameras, 4, 4), "float32"
            ),
            "can_bus_delta": TensorSpec(
                (stack.batch, config.motion.can_bus_dim), "float32"
            ),
            "prev_bev": TensorSpec(
                (stack.batch, stack.queries, stack.embed_dim),
                config.state_dtype,
            ),
            "has_prev_bev": TensorSpec((stack.batch,), "int32"),
        },
        metadata={
            "kind": "bevformer-complete-stateful-streaming-step-static",
            "streaming_step": True,
            "cold_and_warm": True,
            "full_encoder_stack": True,
            "full_decoder_stack": True,
            "full_raw_detection_head": True,
            "state_selection": (
                "has_prev_bev selects motion-compensated history or the "
                "learned initial query/reference grid"
            ),
            "cold_input_contract": (
                "can_bus_delta[:3] and can_bus_delta[-1] must be zero wherever "
                "has_prev_bev is zero; orientation and absolute heading remain"
            ),
            "image_input_layout": "B-camera-C-H-W",
            "state_layout": "B-(bev_h*bev_w)-embed_dim",
            "state_dtype": config.state_dtype,
            "temporal_value_contract": (
                "warm shares [motion-compensated previous BEV, initial current "
                "query]; cold builds [layer input query, layer input query] "
                "inside every encoder layer"
            ),
            "box_representation": (
                "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
                "sin-yaw,cos-yaw,vx,vy]; decode_nms_free converts size/yaw"
            ),
            "postprocess": (
                "explicit host bevformer_postprocess.decode_nms_free until "
                "TopK/Gather CUDA lowering is added"
            ),
            "profile": {
                "batch": stack.batch,
                "cameras": history.feature_prep.cameras,
                "image_shape": list(history.backbone.input_hw),
                "bev_shape": [config.motion.bev_h, config.motion.bev_w],
                "object_queries": config.decoder.queries,
                "embed_dim": stack.embed_dim,
                "classes": config.decoder.classes,
                "box_code_size": config.decoder.code_size,
                "point_cloud_range": list(projection.point_cloud_range),
            },
        },
    )

    composer.add_node(
        Node(
            "images_to_float32",
            "Cast",
            ("images",),
            ("images_float32",),
            {"to": "float32"},
            source_nodes=("BEVFormer.extract_img_feat.img",),
        )
    )
    composer.add_node(
        Node(
            "flatten_image_camera_batch",
            "Reshape",
            ("images_float32",),
            ("backbone_images",),
            {"shape": list(execution_backbone_profile.input_shape)},
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
            "reference_points": "base_reference_points",
        },
    )
    composer.add_node(
        Node(
            "previous_state_to_float32",
            "Cast",
            ("prev_bev",),
            ("prev_bev_float32",),
            {"to": "float32"},
            source_nodes=("BEVFormer.streaming.prev_bev",),
        )
    )
    moved = composer.inline(
        motion,
        prefix="motion",
        bindings={
            "prev_bev": "prev_bev_float32",
            "can_bus_delta": "can_bus_delta",
        },
        output_bindings={
            "rotated_prev_bev": "motion_compensated_prev_bev",
            "shifted_reference_points": "motion_shifted_reference_points",
        },
    )
    # Batch is deliberately fixed to one.  Its [1] flag therefore broadcasts
    # as a scalar over both branch shapes and avoids a layout op on int32.
    composer.add_node(
        Node(
            "select_previous_bev",
            "Where",
            (
                "has_prev_bev",
                moved["rotated_prev_bev"],
                initialized["query"],
            ),
            ("selected_prev_bev",),
            source_nodes=("BEVFormer.streaming.has_prev_bev",),
        )
    )
    composer.add_node(
        Node(
            "select_previous_reference_points",
            "Where",
            (
                "has_prev_bev",
                moved["shifted_reference_points"],
                initialized["reference_points"],
            ),
            ("selected_prev_reference_points",),
            source_nodes=("BEVFormer.streaming.has_prev_bev",),
        )
    )
    composer.add_node(
        Node(
            "build_shared_temporal_queue",
            "Concat",
            ("selected_prev_bev", initialized["query"]),
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
            "prev_reference_points": "selected_prev_reference_points",
            "has_prev_bev": "has_prev_bev",
        },
        output_bindings={"output": "encoded_next_bev"},
    )
    decoded = composer.inline(
        decoder,
        prefix="decoder",
        bindings={"value": encoded["output"]},
        output_bindings={
            "final_class_logits": "cls_logits",
            "final_boxes": "bbox_regression",
        },
    )
    composer.add_node(
        Node(
            "next_state_to_float16",
            "Cast",
            (encoded["output"],),
            ("next_bev",),
            {"to": config.state_dtype},
            source_nodes=("BEVFormer.streaming.next_bev",),
        )
    )
    return composer.build(
        (
            decoded["final_class_logits"],
            decoded["final_boxes"],
            "next_bev",
        )
    )


def validate_step_pipeline_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the exact stateful runtime ABI and cold-frame invariant."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping):
        raise ValidationError("step pipeline inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValidationError("step pipeline input names must be non-empty strings")
    if set(values) != set(graph.inputs):
        raise ValidationError(
            f"step pipeline input names must be {sorted(graph.inputs)}, "
            f"got {sorted(repr(name) for name in values)}"
        )

    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"step pipeline input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"step pipeline input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous:
            raise ValidationError(
                f"step pipeline input {name!r} must be C-contiguous"
            )
        if not np.isfinite(value).all():
            raise ValidationError(
                f"step pipeline input {name!r} must contain only finite values"
            )

    flags = values["has_prev_bev"]
    if np.any((flags != 0) & (flags != 1)):
        raise ValidationError("step pipeline has_prev_bev values must be 0 or 1")
    cold_can_bus = values["can_bus_delta"][flags == 0]
    if np.any(cold_can_bus[:, :3] != np.float32(0.0)) or np.any(
        cold_can_bus[:, -1] != np.float32(0.0)
    ):
        raise ValidationError(
            "step pipeline cold rows require can_bus_delta[:3] and "
            "can_bus_delta[-1] to be zero"
        )


__all__ = [
    "BevFormerStepPipelineConfig",
    "build_bevformer_step_pipeline_graph",
    "step_pipeline_weight_shapes",
    "validate_step_pipeline_inputs",
]
