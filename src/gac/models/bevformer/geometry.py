"""Static BEV reference grid projection for BEVFormer-tiny.

The original model constructs a 3-D pillar grid in Python and projects it
through each camera's ``lidar2img`` matrix.  This module freezes the grid as a
constant and lowers the remaining work into canonical tensor operations, so
camera visibility is computed on the device without Python dictionaries,
NumPy callbacks, or data-dependent allocation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class BevProjectionConfig:
    """Static projection profile for the official BEVFormer pillar grid.

    The official encoder calls ``get_reference_points`` with ``Z`` equal to
    the physical height of ``point_cloud_range``.  For the tiny profile that
    height is ``3 - (-5) == 8``.  It then places ``z_anchors`` samples with
    ``linspace(0.5, Z - 0.5, z_anchors) / Z``; this is deliberately different
    from dividing the height into ``z_anchors`` equal cells.
    """

    batch: int = 1
    cameras: int = 6
    bev_h: int = 50
    bev_w: int = 50
    z_anchors: int = 4
    image_h: int = 480
    image_w: int = 800
    point_cloud_range: tuple[float, float, float, float, float, float] = (
        -51.2,
        -51.2,
        -5.0,
        51.2,
        51.2,
        3.0,
    )
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "batch",
            "cameras",
            "bev_h",
            "bev_w",
            "z_anchors",
            "image_h",
            "image_w",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"BEV projection {name} must be a positive integer")
        if (
            not isinstance(self.point_cloud_range, tuple)
            or len(self.point_cloud_range) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.point_cloud_range
            )
        ):
            raise ValidationError("point_cloud_range must contain six finite numbers")
        low = self.point_cloud_range[:3]
        high = self.point_cloud_range[3:]
        if any(float(maximum) <= float(minimum) for minimum, maximum in zip(low, high, strict=True)):
            raise ValidationError("point_cloud_range maxima must exceed minima")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float))
            or not math.isfinite(float(self.epsilon))
            or float(self.epsilon) <= 0
            or np.float32(self.epsilon) == 0
        ):
            raise ValidationError("BEV projection epsilon must be positive finite float32")

    @property
    def queries(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def reference_z_extent(self) -> float:
        """Return the official ``Z`` used to normalize pillar anchors."""

        return float(self.point_cloud_range[5]) - float(self.point_cloud_range[2])


def make_reference_grid(config: BevProjectionConfig) -> np.ndarray:
    """Return physical homogeneous pillar centers as ``[D,Q,4]`` FP32."""

    if not isinstance(config, BevProjectionConfig):
        raise ValidationError("config must be a BevProjectionConfig")
    x = (np.arange(config.bev_w, dtype=np.float32) + np.float32(0.5)) / np.float32(config.bev_w)
    y = (np.arange(config.bev_h, dtype=np.float32) + np.float32(0.5)) / np.float32(config.bev_h)
    # Match BEVFormerEncoder.get_reference_points exactly.  ``Z`` is the
    # physical z extent (8 for the official tiny profile), rather than the
    # number of anchors.  With four anchors this yields normalized heights
    # [0.0625, 0.354166..., 0.645833..., 0.9375].
    reference_z_extent = np.float32(config.reference_z_extent)
    z = np.linspace(
        np.float32(0.5),
        reference_z_extent - np.float32(0.5),
        config.z_anchors,
        dtype=np.float32,
    ) / reference_z_extent
    normalized = np.empty(
        (config.z_anchors, config.bev_h, config.bev_w, 3), dtype=np.float32
    )
    normalized[..., 0] = x.reshape(1, 1, config.bev_w)
    normalized[..., 1] = y.reshape(1, config.bev_h, 1)
    normalized[..., 2] = z.reshape(config.z_anchors, 1, 1)
    minimum = np.asarray(config.point_cloud_range[:3], dtype=np.float32)
    extent = np.asarray(config.point_cloud_range[3:], dtype=np.float32) - minimum
    physical = normalized.reshape(config.z_anchors, config.queries, 3) * extent + minimum
    ones = np.ones((config.z_anchors, config.queries, 1), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate((physical, ones), axis=-1))


def build_bev_projection_graph(
    config: BevProjectionConfig | None = None,
    *,
    name: str = "bevformer_reference_projection",
) -> Graph:
    """Build lidar-to-image projection and fixed-upper-bound visibility mask."""

    config = config or BevProjectionConfig()
    if not isinstance(config, BevProjectionConfig):
        raise ValidationError("config must be a BevProjectionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("BEV projection graph name must be non-empty")
    b, cameras, depth, queries = (
        config.batch,
        config.cameras,
        config.z_anchors,
        config.queries,
    )
    constants = {
        "reference_homogeneous": make_reference_grid(config).reshape(
            1, 1, depth, queries, 4, 1
        ),
        "image_normalizer": np.asarray(
            [config.image_w, config.image_h], dtype=np.float32
        ).reshape(1, 1, 1, 1, 2),
        "zero": np.asarray(0.0, dtype=np.float32),
        "one": np.asarray(1.0, dtype=np.float32),
        "epsilon": np.asarray(config.epsilon, dtype=np.float32),
    }
    nodes = [
        Node(
            "lidar2img_batch_axes",
            "Reshape",
            ("lidar2img",),
            ("camera_matrices",),
            {"shape": [b, cameras, 1, 1, 4, 4]},
            source_nodes=("BEVFormerEncoder.point_sampling.lidar2img",),
        ),
        Node(
            "project_reference_points",
            "MatMul",
            ("camera_matrices", "reference_homogeneous"),
            ("projected_column",),
            source_nodes=("BEVFormerEncoder.point_sampling.matmul",),
        ),
        Node(
            "projected_vector",
            "Reshape",
            ("projected_column",),
            ("projected",),
            {"shape": [b, cameras, depth, queries, 4]},
        ),
        Node(
            "projected_xy",
            "Slice",
            ("projected",),
            ("xy",),
            {"starts": [0], "ends": [2], "axes": [4]},
        ),
        Node(
            "projected_depth",
            "Slice",
            ("projected",),
            ("camera_depth",),
            {"starts": [2], "ends": [3], "axes": [4]},
        ),
        Node(
            "positive_depth",
            "Greater",
            ("camera_depth", "epsilon"),
            ("depth_mask",),
            source_nodes=("BEVFormerEncoder.point_sampling.bev_mask",),
        ),
        Node(
            "safe_depth",
            "Clamp",
            ("camera_depth",),
            ("clamped_depth",),
            {"min": float(config.epsilon)},
        ),
        Node("perspective_divide", "Div", ("xy", "clamped_depth"), ("pixel_xy",)),
        Node(
            "normalize_image_coordinates",
            "Div",
            ("pixel_xy", "image_normalizer"),
            ("normalized_xy",),
            source_nodes=("BEVFormerEncoder.point_sampling.img_shape",),
        ),
        Node(
            "x_coordinate",
            "Slice",
            ("normalized_xy",),
            ("x",),
            {"starts": [0], "ends": [1], "axes": [4]},
        ),
        Node(
            "y_coordinate",
            "Slice",
            ("normalized_xy",),
            ("y",),
            {"starts": [1], "ends": [2], "axes": [4]},
        ),
        Node("x_above_zero", "Greater", ("x", "zero"), ("x_positive",)),
        Node("x_below_one", "Less", ("x", "one"), ("x_in_upper_bound",)),
        Node("y_above_zero", "Greater", ("y", "zero"), ("y_positive",)),
        Node("y_below_one", "Less", ("y", "one"), ("y_in_upper_bound",)),
        Node("x_visible", "Mul", ("x_positive", "x_in_upper_bound"), ("x_mask",)),
        Node("y_visible", "Mul", ("y_positive", "y_in_upper_bound"), ("y_mask",)),
        Node("xy_visible", "Mul", ("x_mask", "y_mask"), ("xy_mask",)),
        Node("anchor_visible", "Mul", ("xy_mask", "depth_mask"), ("anchor_mask",)),
        Node(
            "query_anchor_count",
            "ReduceSum",
            ("anchor_mask",),
            ("visible_anchor_count_dcq1",),
            {"axes": [2], "keepdims": False},
        ),
        Node(
            "query_visible",
            "Greater",
            ("visible_anchor_count_dcq1", "zero"),
            ("visibility",),
            source_nodes=("SpatialCrossAttention.query_visibility",),
        ),
        Node(
            "reference_layout",
            "Transpose",
            ("normalized_xy",),
            ("reference_points",),
            {"perm": [0, 1, 3, 2, 4]},
            source_nodes=("BEVFormerEncoder.point_sampling.reference_points_cam",),
        ),
    ]
    graph = Graph(
        name=name,
        inputs={"lidar2img": TensorSpec((b, cameras, 4, 4), "float32")},
        outputs=("reference_points", "visibility"),
        nodes=nodes,
        constants=constants,
        metadata={
            "component": "BEVFormerEncoder.point_sampling",
            "profile": {
                "batch": b,
                "cameras": cameras,
                "bev_hw": [config.bev_h, config.bev_w],
                "z_anchors": depth,
                "reference_z_extent": config.reference_z_extent,
                "reference_z_normalization": (
                    "linspace(0.5, Z - 0.5, z_anchors) / Z"
                ),
                "image_hw": [config.image_h, config.image_w],
                "point_cloud_range": list(config.point_cloud_range),
            },
            "mask_representation": "float32-binary-fixed-upper-bound",
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_projection_inputs(graph: Graph, values: Mapping[str, np.ndarray]) -> None:
    """Validate the projection ABI before runtime submission."""

    if set(values) != set(graph.inputs):
        raise ValidationError("projection input names do not match the graph")
    value = values["lidar2img"]
    if not isinstance(value, np.ndarray):
        raise ValidationError("lidar2img must be a NumPy array")
    spec = graph.inputs["lidar2img"]
    if value.shape != spec.shape or value.dtype.name != spec.dtype:
        raise ValidationError(f"lidar2img must match {spec.shape}/{spec.dtype}")
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValidationError("lidar2img must be contiguous and finite")


__all__ = [
    "BevProjectionConfig",
    "build_bev_projection_graph",
    "make_reference_grid",
    "validate_projection_inputs",
]
