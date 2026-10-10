"""BEVFormer per-decoder-layer classification and box-code projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .errors import ValidationError
from .ir import Graph, Node, TensorSpec
from .passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class DetectionHeadConfig:
    batch: int = 1
    queries: int = 900
    embed_dim: int = 256
    classes: int = 10
    code_size: int = 10
    norm_epsilon: float = 1e-5
    inverse_sigmoid_epsilon: float = 1e-5
    point_cloud_range: tuple[float, float, float, float, float, float] = (
        -51.2,
        -51.2,
        -5.0,
        51.2,
        51.2,
        3.0,
    )

    def __post_init__(self) -> None:
        for name in ("batch", "queries", "embed_dim", "classes", "code_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"detection head {name} must be positive")
        if self.batch != 1:
            raise ValidationError("detection head currently requires batch=1")
        if self.code_size < 5:
            raise ValidationError("detection head code_size must be at least 5")
        for name in ("norm_epsilon", "inverse_sigmoid_epsilon"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                or np.float32(value) == 0.0
            ):
                raise ValidationError(f"detection head {name} must be positive finite float32")
            object.__setattr__(self, name, float(value))
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
            raise ValidationError("detection head point_cloud_range must contain six finite numbers")
        low = self.point_cloud_range[:3]
        high = self.point_cloud_range[3:]
        if any(float(upper) <= float(lower) for lower, upper in zip(low, high, strict=True)):
            raise ValidationError("detection head point_cloud_range maxima must exceed minima")


def detection_head_weight_shapes(
    config: DetectionHeadConfig,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(config, DetectionHeadConfig):
        raise ValidationError("config must be a DetectionHeadConfig")
    channels = config.embed_dim
    return {
        "cls0_weight": (channels, channels),
        "cls0_bias": (channels,),
        "cls0_norm_weight": (channels,),
        "cls0_norm_bias": (channels,),
        "cls1_weight": (channels, channels),
        "cls1_bias": (channels,),
        "cls1_norm_weight": (channels,),
        "cls1_norm_bias": (channels,),
        "cls2_weight": (channels, config.classes),
        "cls2_bias": (config.classes,),
    }


def _validated_weights(
    config: DetectionHeadConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = detection_head_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("detection head weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != set(expected):
        raise ValidationError(
            f"detection head weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"detection head weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"detection head weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    x0, y0, z0, x1, y1, z1 = map(float, config.point_cloud_range)
    result.update(
        {
            "one": np.asarray([1.0], dtype=np.float32),
            "xy_scale": np.asarray([x1 - x0, y1 - y0], dtype=np.float32),
            "xy_offset": np.asarray([x0, y0], dtype=np.float32),
            "z_scale": np.asarray([z1 - z0], dtype=np.float32),
            "z_offset": np.asarray([z0], dtype=np.float32),
        }
    )
    return result


def _node(node_id, op, inputs, output, source, attrs=None):
    return Node(
        node_id,
        op,
        tuple(inputs),
        (output,),
        dict(attrs or {}),
        source_nodes=(source,),
    )


def build_detection_head_graph(
    weights: Mapping[str, np.ndarray],
    config: DetectionHeadConfig | None = None,
    *,
    name: str = "bevformer_detection_head_layer",
) -> Graph:
    """Build one class branch and its reference-aware ten-channel box code.

    Center x/y/z are converted to point-cloud coordinates as in
    ``BEVFormerHead.forward``. Size stays logarithmic and rotation stays in
    sine/cosine form until the explicit NMS-free postprocessor.
    """

    config = config or DetectionHeadConfig()
    if not isinstance(config, DetectionHeadConfig):
        raise ValidationError("config must be a DetectionHeadConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("detection head graph name must be non-empty")
    constants = _validated_weights(config, weights)
    eps = config.inverse_sigmoid_epsilon
    cls_source = "BEVFormerHead.cls_branches[layer]"
    bbox_source = "BEVFormerHead.forward.reference_aware_bbox"
    nodes = [
        _node("cls0_matmul", "MatMul", ("query", "cls0_weight"), "cls0_mm", cls_source),
        _node("cls0_bias", "Add", ("cls0_mm", "cls0_bias"), "cls0", cls_source),
        _node("cls0_norm", "LayerNorm", ("cls0", "cls0_norm_weight", "cls0_norm_bias"), "cls0_normalized", cls_source, {"axis": -1, "epsilon": config.norm_epsilon}),
        _node("cls0_relu", "ReLU", ("cls0_normalized",), "cls_hidden0", cls_source),
        _node("cls1_matmul", "MatMul", ("cls_hidden0", "cls1_weight"), "cls1_mm", cls_source),
        _node("cls1_bias", "Add", ("cls1_mm", "cls1_bias"), "cls1", cls_source),
        _node("cls1_norm", "LayerNorm", ("cls1", "cls1_norm_weight", "cls1_norm_bias"), "cls1_normalized", cls_source, {"axis": -1, "epsilon": config.norm_epsilon}),
        _node("cls1_relu", "ReLU", ("cls1_normalized",), "cls_hidden1", cls_source),
        _node("cls2_matmul", "MatMul", ("cls_hidden1", "cls2_weight"), "class_mm", cls_source),
        _node("cls2_bias", "Add", ("class_mm", "cls2_bias"), "class_logits", cls_source),
        _node("class_sigmoid", "Sigmoid", ("class_logits",), "class_scores", "NMSFreeCoder.decode.sigmoid"),
        _node("reference_unit", "Clamp", ("reference_points",), "reference_clamped", bbox_source, {"min": 0.0, "max": 1.0}),
        _node("reference_numerator", "Clamp", ("reference_clamped",), "reference_num", bbox_source, {"min": eps}),
        _node("reference_complement", "Sub", ("one", "reference_clamped"), "reference_one_minus", bbox_source),
        _node("reference_denominator", "Clamp", ("reference_one_minus",), "reference_den", bbox_source, {"min": eps}),
        _node("reference_ratio", "Div", ("reference_num", "reference_den"), "reference_odds", bbox_source),
        _node("reference_log", "Log", ("reference_odds",), "reference_logits", bbox_source),
        _node("reg_xy", "Slice", ("regression",), "regression_xy", bbox_source, {"starts": [0], "ends": [2], "axes": [-1], "steps": [1]}),
        _node("reg_middle", "Slice", ("regression",), "regression_middle", bbox_source, {"starts": [2], "ends": [4], "axes": [-1], "steps": [1]}),
        _node("reg_z", "Slice", ("regression",), "regression_z", bbox_source, {"starts": [4], "ends": [5], "axes": [-1], "steps": [1]}),
        _node("reg_tail", "Slice", ("regression",), "regression_tail", bbox_source, {"starts": [5], "ends": [config.code_size], "axes": [-1], "steps": [1]}),
        _node("ref_xy", "Slice", ("reference_logits",), "reference_xy", bbox_source, {"starts": [0], "ends": [2], "axes": [-1], "steps": [1]}),
        _node("ref_z", "Slice", ("reference_logits",), "reference_z", bbox_source, {"starts": [2], "ends": [3], "axes": [-1], "steps": [1]}),
        _node("xy_logit_add", "Add", ("regression_xy", "reference_xy"), "xy_logits", bbox_source),
        _node("xy_sigmoid", "Sigmoid", ("xy_logits",), "xy_normalized", bbox_source),
        _node("xy_scale", "Mul", ("xy_normalized", "xy_scale"), "xy_scaled", bbox_source),
        _node("xy_offset", "Add", ("xy_scaled", "xy_offset"), "xy_physical", bbox_source),
        _node("z_logit_add", "Add", ("regression_z", "reference_z"), "z_logits", bbox_source),
        _node("z_sigmoid", "Sigmoid", ("z_logits",), "z_normalized", bbox_source),
        _node("z_scale", "Mul", ("z_normalized", "z_scale"), "z_scaled", bbox_source),
        _node("z_offset", "Add", ("z_scaled", "z_offset"), "z_physical", bbox_source),
        _node("bbox_concat", "Concat", ("xy_physical", "regression_middle", "z_physical", "regression_tail"), "boxes", bbox_source, {"axis": -1}),
    ]
    graph = Graph(
        name=name,
        inputs={
            "query": TensorSpec((config.batch, config.queries, config.embed_dim), "float32"),
            "reference_points": TensorSpec((config.batch, config.queries, 3), "float32"),
            "regression": TensorSpec((config.batch, config.queries, config.code_size), "float32"),
        },
        outputs=("class_logits", "class_scores", "boxes"),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-per-layer-detection-head",
            "outputs_are_raw_queries": True,
            "nms_free_topk": "outside-this-graph",
            "box_representation": (
                "[physical-x,physical-y,log-w,log-l,physical-z,log-h,"
                "sin-yaw,cos-yaw,vx,vy]; final denormalization outside graph"
            ),
            "classes": config.classes,
            "code_size": config.code_size,
            "point_cloud_range": list(config.point_cloud_range),
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_detection_head_inputs(graph: Graph, values: Mapping[str, np.ndarray]) -> None:
    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        raise ValidationError("detection head input names do not match the graph")
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"detection head input {name!r} must be an ndarray")
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"detection head input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(f"detection head input {name!r} must be contiguous and finite")
    reference = values["reference_points"]
    if np.any(reference < 0.0) or np.any(reference > 1.0):
        raise ValidationError("detection head reference_points must be normalized to [0, 1]")


__all__ = [
    "DetectionHeadConfig",
    "build_detection_head_graph",
    "detection_head_weight_shapes",
    "validate_detection_head_inputs",
]
