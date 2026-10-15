"""Integrated BEVFormer decoder and raw detection projection pipeline.

The graph connects learned object-query initialization, all six decoder
layers, iterative reference refinement, and all six raw classification/box
branches.  Its only runtime input is encoded BEV memory.  Top-K query
selection and the final NMS-free coder policy remain outside this graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from gac.models.bevformer.decoder_stack import (
    BevFormerDecoderStackConfig,
    build_bevformer_decoder_stack_graph,
    decoder_stack_weight_shapes,
)
from gac.models.bevformer.detection_stack import (
    DetectionStackConfig,
    build_detection_stack_graph,
    detection_stack_weight_shapes,
)
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, TensorSpec


@dataclass(frozen=True, slots=True)
class BevFormerDecoderPipelineConfig:
    """Consistent static profiles for decoder and raw detection stages."""

    decoder: BevFormerDecoderStackConfig = field(
        default_factory=BevFormerDecoderStackConfig
    )
    detection: DetectionStackConfig = field(default_factory=DetectionStackConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.decoder, BevFormerDecoderStackConfig):
            raise ValidationError(
                "decoder pipeline decoder must be a BevFormerDecoderStackConfig"
            )
        if not isinstance(self.detection, DetectionStackConfig):
            raise ValidationError(
                "decoder pipeline detection must be a DetectionStackConfig"
            )
        mismatches: list[str] = []
        if self.decoder.num_layers != self.detection.num_layers:
            mismatches.append("num_layers")
        if self.decoder.batch != self.detection.batch:
            mismatches.append("batch")
        if self.decoder.queries != self.detection.queries:
            mismatches.append("queries")
        if self.decoder.embed_dim != self.detection.embed_dim:
            mismatches.append("embed_dim")
        if self.decoder.code_size != self.detection.code_size:
            mismatches.append("code_size")
        if mismatches:
            raise ValidationError(
                "decoder pipeline component profiles disagree on: "
                + ", ".join(mismatches)
            )

    @property
    def batch(self) -> int:
        return self.decoder.batch

    @property
    def queries(self) -> int:
        return self.decoder.queries

    @property
    def embed_dim(self) -> int:
        return self.decoder.embed_dim

    @property
    def classes(self) -> int:
        return self.detection.classes

    @property
    def code_size(self) -> int:
        return self.decoder.code_size


def decoder_pipeline_weight_shapes(
    config: BevFormerDecoderPipelineConfig,
) -> dict[str, object]:
    """Return an explicitly grouped inventory for decoder and raw head."""

    if not isinstance(config, BevFormerDecoderPipelineConfig):
        raise ValidationError("config must be a BevFormerDecoderPipelineConfig")
    decoder_shapes = decoder_stack_weight_shapes(config.decoder)
    return {
        "decoder": {
            "query_init": decoder_shapes["query_init"],
            "self_attention": decoder_shapes["self_attention"],
            "cross_attention": decoder_shapes["cross_attention"],
            "layer": decoder_shapes["layer"],
            "refinement": decoder_shapes["refinement"],
        },
        "detection": detection_stack_weight_shapes(config.detection),
    }


def build_bevformer_decoder_pipeline_graph(
    query_init_weights: Mapping[str, np.ndarray],
    self_attention_weights: Sequence[Mapping[str, np.ndarray]],
    cross_attention_weights: Sequence[Mapping[str, np.ndarray]],
    layer_weights: Sequence[Mapping[str, np.ndarray]],
    refinement_weights: Sequence[Mapping[str, np.ndarray]],
    detection_weights: Sequence[Mapping[str, np.ndarray]],
    config: BevFormerDecoderPipelineConfig | None = None,
    *,
    name: str = "bevformer_decoder_raw_detection_pipeline",
) -> Graph:
    """Connect the complete decoder stack directly to all raw head branches."""

    config = config or BevFormerDecoderPipelineConfig()
    if not isinstance(config, BevFormerDecoderPipelineConfig):
        raise ValidationError("config must be a BevFormerDecoderPipelineConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder pipeline graph name must be non-empty")

    decoder_graph = build_bevformer_decoder_stack_graph(
        query_init_weights,
        self_attention_weights,
        cross_attention_weights,
        layer_weights,
        refinement_weights,
        config.decoder,
    )
    detection_graph = build_detection_stack_graph(
        detection_weights, config.detection
    )

    cross = config.decoder.layer.cross_attention
    composer = GraphComposer(
        name,
        {
            "value": TensorSpec(
                (config.batch, cross.spatial_size, config.embed_dim), "float32"
            )
        },
        metadata={
            "kind": "bevformer-decoder-and-raw-detection-pipeline-static",
            "full_decoder_stack": True,
            "full_raw_detection_head": True,
            "decoder_to_raw_head_connected": True,
            "num_decoder_layers": config.decoder.num_layers,
            "num_detection_layers": config.detection.num_layers,
            "runtime_inputs": ["encoded-bev-value"],
            "batch_first_abi": True,
            "golden_audit_outputs": [
                "initial_reference_points",
                "intermediate_reference_points",
            ],
            "raw_output_semantics": (
                "all decoder queries before NMS-free Top-K selection"
            ),
            "box_representation": (
                "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
                "sin-yaw,cos-yaw,vx,vy]; decode_nms_free converts size/yaw"
            ),
            "topk": {"status": "outside-this-graph", "k": 300},
            "nms_free_decode": "outside-this-graph",
            "profile": {
                "batch": config.batch,
                "queries": config.queries,
                "bev_h": cross.bev_h,
                "bev_w": cross.bev_w,
                "embed_dim": config.embed_dim,
                "classes": config.classes,
                "code_size": config.code_size,
                "point_cloud_range": list(
                    config.detection.head.point_cloud_range
                ),
            },
            "excluded_stages": [
                "encoder",
                "top-k-300-query-selection",
                "nms-free-final-record-formatting",
            ],
        },
    )

    decoder = composer.inline(
        decoder_graph,
        prefix="decoder",
        bindings={"value": "value"},
        output_bindings={
            "intermediate_states": "decoder_intermediate_states",
            "initial_reference_points": "initial_reference_points",
            "intermediate_reference_points": "intermediate_reference_points",
            "regressions": "decoder_regressions",
        },
    )
    detection = composer.inline(
        detection_graph,
        prefix="detection",
        bindings={
            "intermediate_states": decoder["intermediate_states"],
            "initial_reference_points": decoder["initial_reference_points"],
            "intermediate_reference_points": decoder[
                "intermediate_reference_points"
            ],
            "regressions": decoder["regressions"],
        },
        output_bindings={
            "all_class_logits": "all_class_logits",
            "all_class_scores": "all_class_scores",
            "all_boxes": "all_boxes",
            "final_class_logits": "final_class_logits",
            "final_class_scores": "final_class_scores",
            "final_boxes": "final_boxes",
        },
    )
    return composer.build(
        (
            detection["all_class_logits"],
            detection["all_class_scores"],
            detection["all_boxes"],
            detection["final_class_logits"],
            detection["final_class_scores"],
            detection["final_boxes"],
            decoder["initial_reference_points"],
            decoder["intermediate_reference_points"],
        )
    )


def validate_decoder_pipeline_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the single encoded-BEV input contract."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping):
        raise ValidationError("decoder pipeline inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValidationError(
            "decoder pipeline input names must be non-empty strings"
        )
    if set(values) != set(graph.inputs):
        raise ValidationError(
            f"decoder pipeline input names must be {sorted(graph.inputs)}, "
            f"got {sorted(repr(name) for name in values)}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder pipeline input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder pipeline input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder pipeline input {name!r} must be contiguous and finite"
            )


__all__ = [
    "BevFormerDecoderPipelineConfig",
    "build_bevformer_decoder_pipeline_graph",
    "decoder_pipeline_weight_shapes",
    "validate_decoder_pipeline_inputs",
]
