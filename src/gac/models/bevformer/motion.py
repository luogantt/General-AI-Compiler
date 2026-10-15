"""Warm-history ego-motion compensation for the BEVFormer-tiny profile.

The graph reproduces the two motion paths in the official
``PerceptionTransformer.get_bev_features`` implementation:

* rotate ``prev_bev`` with torchvision 0.10.1 tensor-rotate coordinates; and
* add the ego-translation shift to the normalized 2-D BEV reference grid.

The official call leaves ``interpolation`` unspecified, so its effective mode
is nearest-neighbour with zero fill and ``align_corners=False``.  Rotation is
therefore represented with ``GridSample2D(mode="nearest")`` rather than a
bilinear approximation.
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
class BevFormerMotionConfig:
    """Static official-tiny motion profile.

    ``can_bus_delta[..., 0:2]`` is the frame-to-frame XY translation,
    ``can_bus_delta[..., -2]`` is the absolute ego heading in radians, and
    ``can_bus_delta[..., -1]`` is the frame-to-frame rotation in degrees.
    """

    batch: int = 1
    bev_h: int = 50
    bev_w: int = 50
    embed_dim: int = 256
    can_bus_dim: int = 18
    # BEVFormerHead passes real_h / bev_h and real_w / bev_w.  The tiny
    # profile spans 102.4 m on each axis with a 50 x 50 BEV grid.
    grid_length_y: float = 2.048
    grid_length_x: float = 2.048
    rotate_center: tuple[float, float] = (100.0, 100.0)

    def __post_init__(self) -> None:
        for name in ("batch", "bev_h", "bev_w", "embed_dim", "can_bus_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"BEVFormer motion {name} must be positive")
        if self.batch != 1:
            raise ValidationError("BEVFormer motion currently requires batch=1")
        if self.can_bus_dim != 18:
            raise ValidationError(
                "BEVFormer motion requires the official 18-element CAN-bus vector"
            )
        float32_max = float(np.finfo(np.float32).max)
        float32_min_positive = float(
            np.nextafter(np.float32(0.0), np.float32(1.0))
        )
        for name in ("grid_length_y", "grid_length_x"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                or float(value) < float32_min_positive
                or float(value) > float32_max
            ):
                raise ValidationError(
                    f"BEVFormer motion {name} must be positive finite float32"
                )
            object.__setattr__(self, name, float(value))
        for name, cells in (
            ("grid_length_y", self.bev_h),
            ("grid_length_x", self.bev_w),
        ):
            if getattr(self, name) > float32_max / cells:
                raise ValidationError(
                    f"BEVFormer motion {name} * grid size must be positive finite float32"
                )
        center = self.rotate_center
        if (
            not isinstance(center, (tuple, list))
            or len(center) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or abs(float(value)) > float32_max
                for value in center
            )
        ):
            raise ValidationError(
                "BEVFormer motion rotate_center must contain two finite float32 numbers"
            )
        object.__setattr__(
            self, "rotate_center", (float(center[0]), float(center[1]))
        )

    @property
    def queries(self) -> int:
        return self.bev_h * self.bev_w


def _scalar(value: float) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _reference_grid(config: BevFormerMotionConfig) -> np.ndarray:
    y = (
        np.arange(config.bev_h, dtype=np.float32) + np.float32(0.5)
    ) / np.float32(config.bev_h)
    x = (
        np.arange(config.bev_w, dtype=np.float32) + np.float32(0.5)
    ) / np.float32(config.bev_w)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.ascontiguousarray(
        np.stack((grid_x, grid_y), axis=-1).reshape(1, config.queries, 1, 2)
    )


def _centered_pixel_grid(config: BevFormerMotionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return torchvision 0.10.1 ``_gen_affine_grid`` pixel centers."""

    x = (
        np.arange(config.bev_w, dtype=np.float32)
        + np.float32(0.5)
        - np.float32(config.bev_w * 0.5)
    )
    y = (
        np.arange(config.bev_h, dtype=np.float32)
        + np.float32(0.5)
        - np.float32(config.bev_h * 0.5)
    )
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return (
        np.ascontiguousarray(grid_x.reshape(1, config.bev_h, config.bev_w, 1)),
        np.ascontiguousarray(grid_y.reshape(1, config.bev_h, config.bev_w, 1)),
    )


def _node(
    node_id: str,
    op: str,
    inputs: tuple[str, ...],
    output: str,
    attrs: Mapping[str, object] | None = None,
    *,
    source: str,
) -> Node:
    return Node(
        node_id,
        op,
        inputs,
        (output,),
        dict(attrs or {}),
        source_nodes=(source,),
    )


def build_bevformer_motion_graph(
    config: BevFormerMotionConfig | None = None,
    *,
    name: str = "bevformer_warm_history_motion",
) -> Graph:
    """Build previous-BEV rotation and shifted-reference preparation.

    The returned tensors are batch-first.  ``rotated_prev_bev`` has shape
    ``[1, H*W, C]`` and ``shifted_reference_points`` has shape
    ``[1, H*W, 1, 2]`` in normalized XY order.

    The coordinate equations match torchvision 0.10.1 tensor ``rotate`` with
    ``expand=False``, its default nearest interpolation, zero fill, and the
    configured upper-left-origin center.  Floating-point evaluation is not
    bitwise promised at the nearest-neighbour half-pixel discontinuity:
    torchvision constructs its matrix with host ``math`` and samples through
    PyTorch, while this graph evaluates static FP32 CUDA nodes.  Away from an
    half-pixel boundary within floating-point rounding both select the same
    source pixel.
    """

    config = config or BevFormerMotionConfig()
    if not isinstance(config, BevFormerMotionConfig):
        raise ValidationError("config must be a BevFormerMotionConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("BEVFormer motion graph name must be non-empty")

    base_x, base_y = _centered_pixel_grid(config)
    # torchvision translates a user center from upper-left pixel coordinates
    # into coordinates whose origin is at the image center.
    center_x = config.rotate_center[0] - config.bev_w * 0.5
    center_y = config.rotate_center[1] - config.bev_h * 0.5
    constants = {
        "radians_per_degree": _scalar(math.pi / 180.0),
        "negative_one": _scalar(-1.0),
        "negative_center_x": _scalar(-center_x),
        "negative_center_y": _scalar(-center_y),
        "center_x": _scalar(center_x),
        "center_y": _scalar(center_y),
        "half_width": _scalar(config.bev_w * 0.5),
        "half_height": _scalar(config.bev_h * 0.5),
        "shift_x_denominator": _scalar(config.grid_length_x * config.bev_w),
        "shift_y_denominator": _scalar(config.grid_length_y * config.bev_h),
        "rotation_base_x": base_x,
        "rotation_base_y": base_y,
        "reference_points": _reference_grid(config),
    }
    source_shift = "PerceptionTransformer.get_bev_features.shift"
    source_rotate = "PerceptionTransformer.get_bev_features.rotate_prev_bev"
    nodes: list[Node] = [
        _node(
            "slice_delta_x", "Slice", ("can_bus_delta",), "delta_x",
            {"starts": [0], "ends": [1], "axes": [1]}, source=source_shift,
        ),
        _node(
            "slice_delta_y", "Slice", ("can_bus_delta",), "delta_y",
            {"starts": [1], "ends": [2], "axes": [1]}, source=source_shift,
        ),
        _node(
            "slice_ego_heading", "Slice", ("can_bus_delta",), "ego_heading",
            {"starts": [config.can_bus_dim - 2], "ends": [config.can_bus_dim - 1], "axes": [1]},
            source=source_shift,
        ),
        _node("ego_heading_sin", "Sin", ("ego_heading",), "heading_sin", source=source_shift),
        _node("ego_heading_cos", "Cos", ("ego_heading",), "heading_cos", source=source_shift),
        _node("shift_y_dx_cos", "Mul", ("delta_x", "heading_cos"), "shift_y_dx", source=source_shift),
        _node("shift_y_dy_sin", "Mul", ("delta_y", "heading_sin"), "shift_y_dy", source=source_shift),
        _node("shift_y_numerator", "Add", ("shift_y_dx", "shift_y_dy"), "shift_y_numerator_value", source=source_shift),
        _node("normalize_shift_y", "Div", ("shift_y_numerator_value", "shift_y_denominator"), "shift_y", source=source_shift),
        _node("shift_x_dx_sin", "Mul", ("delta_x", "heading_sin"), "shift_x_dx", source=source_shift),
        _node("shift_x_dy_cos", "Mul", ("delta_y", "heading_cos"), "shift_x_dy", source=source_shift),
        _node("shift_x_numerator", "Sub", ("shift_x_dx", "shift_x_dy"), "shift_x_numerator_value", source=source_shift),
        _node("normalize_shift_x", "Div", ("shift_x_numerator_value", "shift_x_denominator"), "shift_x", source=source_shift),
        _node(
            "reshape_shift_x", "Reshape", ("shift_x",), "shift_x_4d",
            {"shape": [config.batch, 1, 1, 1]}, source=source_shift,
        ),
        _node(
            "reshape_shift_y", "Reshape", ("shift_y",), "shift_y_4d",
            {"shape": [config.batch, 1, 1, 1]}, source=source_shift,
        ),
        _node(
            "pack_shift_xy", "Concat", ("shift_x_4d", "shift_y_4d"), "shift_xy",
            {"axis": 3}, source=source_shift,
        ),
        _node(
            "shift_reference_points", "Add", ("reference_points", "shift_xy"),
            "shifted_reference_points", source=source_shift,
        ),
        _node(
            "slice_rotation_degrees", "Slice", ("can_bus_delta",), "rotation_degrees",
            {"starts": [config.can_bus_dim - 1], "ends": [config.can_bus_dim], "axes": [1]},
            source=source_rotate,
        ),
        _node(
            "rotation_to_radians", "Mul", ("rotation_degrees", "radians_per_degree"),
            "rotation_radians", source=source_rotate,
        ),
        _node("rotation_sin", "Sin", ("rotation_radians",), "rotation_sin_value", source=source_rotate),
        _node("rotation_cos", "Cos", ("rotation_radians",), "rotation_cos_value", source=source_rotate),
        _node("negative_rotation_sin", "Mul", ("rotation_sin_value", "negative_one"), "negative_rotation_sin_value", source=source_rotate),
        # torchvision's inverse affine matrix for rotate(angle) is
        # [cos, -sin, tx; sin, cos, ty].  Preserve its construction order.
        _node("translation_x_cos", "Mul", ("rotation_cos_value", "negative_center_x"), "translation_x_cos_value", source=source_rotate),
        _node("translation_x_sin", "Mul", ("negative_rotation_sin_value", "negative_center_y"), "translation_x_sin_value", source=source_rotate),
        _node("translation_x_sum", "Add", ("translation_x_cos_value", "translation_x_sin_value"), "translation_x_partial", source=source_rotate),
        _node("translation_x_center", "Add", ("translation_x_partial", "center_x"), "translation_x", source=source_rotate),
        _node("translation_y_sin", "Mul", ("rotation_sin_value", "negative_center_x"), "translation_y_sin_value", source=source_rotate),
        _node("translation_y_cos", "Mul", ("rotation_cos_value", "negative_center_y"), "translation_y_cos_value", source=source_rotate),
        _node("translation_y_sum", "Add", ("translation_y_sin_value", "translation_y_cos_value"), "translation_y_partial", source=source_rotate),
        _node("translation_y_center", "Add", ("translation_y_partial", "center_y"), "translation_y", source=source_rotate),
    ]

    for value in (
        "rotation_cos_value",
        "negative_rotation_sin_value",
        "rotation_sin_value",
        "translation_x",
        "translation_y",
    ):
        nodes.append(
            _node(
                f"reshape_{value}",
                "Reshape",
                (value,),
                f"{value}_4d",
                {"shape": [config.batch, 1, 1, 1]},
                source=source_rotate,
            )
        )

    nodes.extend(
        [
            _node("grid_x_cos", "Mul", ("rotation_base_x", "rotation_cos_value_4d"), "grid_x_cos_value", source=source_rotate),
            _node("grid_x_sin", "Mul", ("rotation_base_y", "negative_rotation_sin_value_4d"), "grid_x_sin_value", source=source_rotate),
            _node("grid_x_sum", "Add", ("grid_x_cos_value", "grid_x_sin_value"), "grid_x_partial", source=source_rotate),
            _node("grid_x_translate", "Add", ("grid_x_partial", "translation_x_4d"), "grid_x_pixels", source=source_rotate),
            _node("grid_x_normalize", "Div", ("grid_x_pixels", "half_width"), "grid_x", source=source_rotate),
            _node("grid_y_sin", "Mul", ("rotation_base_x", "rotation_sin_value_4d"), "grid_y_sin_value", source=source_rotate),
            _node("grid_y_cos", "Mul", ("rotation_base_y", "rotation_cos_value_4d"), "grid_y_cos_value", source=source_rotate),
            _node("grid_y_sum", "Add", ("grid_y_sin_value", "grid_y_cos_value"), "grid_y_partial", source=source_rotate),
            _node("grid_y_translate", "Add", ("grid_y_partial", "translation_y_4d"), "grid_y_pixels", source=source_rotate),
            _node("grid_y_normalize", "Div", ("grid_y_pixels", "half_height"), "grid_y", source=source_rotate),
            _node("pack_rotation_grid", "Concat", ("grid_x", "grid_y"), "rotation_grid", {"axis": 3}, source=source_rotate),
            _node("prev_bev_channels_first", "Transpose", ("prev_bev",), "prev_bev_bcq", {"perm": [0, 2, 1]}, source=source_rotate),
            _node(
                "prev_bev_image", "Reshape", ("prev_bev_bcq",), "prev_bev_nchw",
                {"shape": [config.batch, config.embed_dim, config.bev_h, config.bev_w]},
                source=source_rotate,
            ),
            _node(
                "rotate_prev_bev", "GridSample2D", ("prev_bev_nchw", "rotation_grid"),
                "rotated_prev_bev_nchw",
                {"mode": "nearest", "padding_mode": "zeros", "align_corners": False},
                source=source_rotate,
            ),
            _node("rotated_prev_bev_channels_last", "Transpose", ("rotated_prev_bev_nchw",), "rotated_prev_bev_nhwc", {"perm": [0, 2, 3, 1]}, source=source_rotate),
            _node(
                "flatten_rotated_prev_bev", "Reshape", ("rotated_prev_bev_nhwc",),
                "rotated_prev_bev",
                {"shape": [config.batch, config.queries, config.embed_dim]},
                source=source_rotate,
            ),
        ]
    )

    graph = Graph(
        name=name,
        inputs={
            "prev_bev": TensorSpec(
                (config.batch, config.queries, config.embed_dim), "float32"
            ),
            "can_bus_delta": TensorSpec(
                (config.batch, config.can_bus_dim), "float32"
            ),
        },
        outputs=("rotated_prev_bev", "shifted_reference_points"),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-warm-history-motion-compensation",
            "source_semantics": "BEVFormer PerceptionTransformer + torchvision 0.10.1 tensor rotate",
            "rotation": {
                "angle": "can_bus_delta[-1] degrees counter-clockwise",
                "center_xy_upper_left_origin": list(config.rotate_center),
                "interpolation": "nearest (official torchvision default)",
                "padding_mode": "zeros",
                "align_corners": False,
                "expand": False,
            },
            "shift": {
                "translation": "can_bus_delta[0:2] frame delta",
                "ego_heading": "can_bus_delta[-2] absolute radians",
                "reference_order": "normalized xy",
                "grid_length_yx": [config.grid_length_y, config.grid_length_x],
            },
            "numeric_boundary": (
                "mathematically equivalent affine coordinates; FP32 graph evaluation "
                "may select the adjacent pixel within rounding distance of a nearest "
                "half-pixel boundary"
            ),
            "profile": {
                "batch": config.batch,
                "bev_h": config.bev_h,
                "bev_w": config.bev_w,
                "queries": config.queries,
                "embed_dim": config.embed_dim,
                "can_bus_dim": config.can_bus_dim,
            },
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_bevformer_motion_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate the strict warm-motion runtime ABI."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    required = {"prev_bev", "can_bus_delta"}
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValidationError(
            f"BEVFormer motion inputs must be exactly {sorted(required)}"
        )
    for name in sorted(required):
        value = values[name]
        spec = graph.inputs[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"BEVFormer motion input {name!r} must be an ndarray")
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"BEVFormer motion input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"BEVFormer motion input {name!r} must be contiguous and finite"
            )


__all__ = [
    "BevFormerMotionConfig",
    "build_bevformer_motion_graph",
    "validate_bevformer_motion_inputs",
]
