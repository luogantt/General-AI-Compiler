"""Composable BEVFormer encoder-stage graphs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gac.models.bevformer.geometry import BevProjectionConfig, build_bev_projection_graph
from gac.models.bevformer.spatial import SpatialCrossAttentionConfig, build_spatial_cross_attention_graph
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, TensorSpec


def _validate_matching_profiles(
    projection: BevProjectionConfig,
    spatial: SpatialCrossAttentionConfig,
) -> None:
    mismatches = []
    for name, left, right in (
        ("batch", projection.batch, spatial.batch),
        ("cameras", projection.cameras, spatial.cameras),
        ("queries", projection.queries, spatial.queries),
        ("z_anchors", projection.z_anchors, spatial.z_anchors),
    ):
        if left != right:
            mismatches.append(f"{name}: projection={left}, spatial={right}")
    if mismatches:
        raise ValidationError(
            "BEV projection and spatial attention profiles disagree: "
            + "; ".join(mismatches)
        )


def build_single_frame_spatial_stage_graph(
    weights: Mapping[str, np.ndarray],
    *,
    projection_config: BevProjectionConfig | None = None,
    spatial_config: SpatialCrossAttentionConfig | None = None,
    name: str = "bevformer_single_frame_spatial_stage",
) -> Graph:
    """Fuse point projection with one fixed-upper-bound spatial attention.

    This is the geometry plus cross-attention core of one encoder layer.  It
    does not include Temporal Self Attention, layer normalization, or the FFN,
    and its name/reporting deliberately preserve that boundary.
    """

    projection = projection_config or BevProjectionConfig()
    spatial = spatial_config or SpatialCrossAttentionConfig()
    if not isinstance(projection, BevProjectionConfig):
        raise ValidationError("projection_config must be a BevProjectionConfig")
    if not isinstance(spatial, SpatialCrossAttentionConfig):
        raise ValidationError("spatial_config must be a SpatialCrossAttentionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("single-frame spatial stage name must be non-empty")
    _validate_matching_profiles(projection, spatial)

    composer = GraphComposer(
        name,
        {
            "query": TensorSpec(
                (spatial.batch, spatial.queries, spatial.embed_dim), "float32"
            ),
            "query_pos": TensorSpec(
                (spatial.batch, spatial.queries, spatial.embed_dim), "float32"
            ),
            "value": TensorSpec(
                (
                    spatial.batch,
                    spatial.cameras,
                    spatial.spatial_size,
                    spatial.embed_dim,
                ),
                "float32",
            ),
            "lidar2img": TensorSpec(
                (projection.batch, projection.cameras, 4, 4), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-single-frame-spatial-stage",
            "scope": "point-projection-plus-one-spatial-cross-attention",
            "full_encoder_layer": False,
        },
    )
    geometry_outputs = composer.inline(
        build_bev_projection_graph(projection),
        prefix="projection",
        bindings={"lidar2img": "lidar2img"},
        output_bindings={
            "reference_points": "reference_points",
            "visibility": "visibility",
        },
    )
    spatial_outputs = composer.inline(
        build_spatial_cross_attention_graph(weights, spatial),
        prefix="spatial_attention",
        bindings={
            "query": "query",
            "query_pos": "query_pos",
            "value": "value",
            "reference_points": geometry_outputs["reference_points"],
            "visibility": geometry_outputs["visibility"],
        },
        output_bindings={"output": "output"},
    )
    return composer.build(
        (spatial_outputs["output"],),
        metadata={
            "projection_profile": {
                "image_hw": [projection.image_h, projection.image_w],
                "point_cloud_range": list(projection.point_cloud_range),
            },
            "attention_profile": {
                "spatial_shapes": [list(shape) for shape in spatial.spatial_shapes],
                "sampling_points": spatial.sampling_points,
                "heads": spatial.heads,
            },
        },
    )


__all__ = ["build_single_frame_spatial_stage_graph"]
