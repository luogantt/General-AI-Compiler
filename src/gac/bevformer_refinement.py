"""BEVFormer decoder regression branch and reference-point refinement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np

from .errors import ValidationError
from .ir import Graph, Node, TensorSpec
from .passes import ShapeInferencePass


@dataclass(frozen=True, slots=True)
class DecoderReferenceRefinementConfig:
    """Static BEVFormer-tiny decoder refinement specialization."""

    batch: int = 1
    queries: int = 900
    embed_dim: int = 256
    code_size: int = 10
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("batch", "queries", "embed_dim", "code_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(f"decoder refinement {name} must be positive")
        if self.code_size < 5:
            raise ValidationError("decoder refinement code_size must be at least 5")
        if (
            isinstance(self.epsilon, bool)
            or not isinstance(self.epsilon, (int, float, np.integer, np.floating))
            or not math.isfinite(float(self.epsilon))
            or not 0.0 < float(self.epsilon) < 0.5
            or np.float32(self.epsilon) == 0.0
        ):
            raise ValidationError(
                "decoder refinement epsilon must be finite, representable, and in (0, 0.5)"
            )
        object.__setattr__(self, "epsilon", float(self.epsilon))


def decoder_refinement_weight_shapes(
    config: DecoderReferenceRefinementConfig,
) -> dict[str, tuple[int, ...]]:
    """Return the canonical two-hidden-layer regression-branch contract."""

    if not isinstance(config, DecoderReferenceRefinementConfig):
        raise ValidationError("config must be a DecoderReferenceRefinementConfig")
    channels = config.embed_dim
    return {
        "fc0_weight": (channels, channels),
        "fc0_bias": (channels,),
        "fc1_weight": (channels, channels),
        "fc1_bias": (channels,),
        "fc2_weight": (channels, config.code_size),
        "fc2_bias": (config.code_size,),
    }


def _validated_weights(
    config: DecoderReferenceRefinementConfig,
    weights: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    expected = decoder_refinement_weight_shapes(config)
    if not isinstance(weights, Mapping):
        raise ValidationError("decoder refinement weights must be a mapping")
    if any(not isinstance(name, str) for name in weights) or set(weights) != set(expected):
        raise ValidationError(
            f"decoder refinement weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = np.asarray(weights[name])
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder refinement weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder refinement weight {name!r} must be contiguous and finite"
            )
        result[name] = value.copy()
    # Keep constants at rank >= 1 because the portable HXP NPY subset rejects
    # zero-dimensional array headers.  Shape [1] broadcasts identically here.
    result["one"] = np.asarray([1.0], dtype=np.float32)
    return result


def _node(
    node_id: str,
    op: str,
    inputs: tuple[str, ...],
    output: str,
    source: str,
    attrs: Mapping[str, object] | None = None,
) -> Node:
    return Node(
        node_id,
        op,
        inputs,
        (output,),
        dict(attrs or {}),
        source_nodes=(source,),
    )


def build_decoder_reference_refinement_graph(
    weights: Mapping[str, np.ndarray],
    config: DecoderReferenceRefinementConfig | None = None,
    *,
    name: str = "bevformer_decoder_reference_refinement",
) -> Graph:
    """Lower one decoder regression branch and its 3D reference update.

    The graph returns all ten regression channels as well as the next
    normalized ``(x, y, z)`` reference.  As in the official decoder, channels
    0/1 refine x/y and channel 4 refines z.
    """

    config = config or DecoderReferenceRefinementConfig()
    if not isinstance(config, DecoderReferenceRefinementConfig):
        raise ValidationError("config must be a DecoderReferenceRefinementConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder refinement graph name must be non-empty")
    constants = _validated_weights(config, weights)
    eps = config.epsilon
    nodes = [
        _node("fc0_matmul", "MatMul", ("query", "fc0_weight"), "fc0_mm", "BEVFormerHead.reg_branches[layer].0"),
        _node("fc0_bias", "Add", ("fc0_mm", "fc0_bias"), "fc0", "BEVFormerHead.reg_branches[layer].0"),
        _node("fc0_relu", "ReLU", ("fc0",), "hidden0", "BEVFormerHead.reg_branches[layer].1"),
        _node("fc1_matmul", "MatMul", ("hidden0", "fc1_weight"), "fc1_mm", "BEVFormerHead.reg_branches[layer].2"),
        _node("fc1_bias", "Add", ("fc1_mm", "fc1_bias"), "fc1", "BEVFormerHead.reg_branches[layer].2"),
        _node("fc1_relu", "ReLU", ("fc1",), "hidden1", "BEVFormerHead.reg_branches[layer].3"),
        _node("fc2_matmul", "MatMul", ("hidden1", "fc2_weight"), "fc2_mm", "BEVFormerHead.reg_branches[layer].4"),
        _node("fc2_bias", "Add", ("fc2_mm", "fc2_bias"), "regression", "BEVFormerHead.reg_branches[layer].4"),
        _node("reference_unit_clamp", "Clamp", ("reference_points",), "reference_unit", "DetectionTransformerDecoder.inverse_sigmoid.clamp01", {"min": 0.0, "max": 1.0}),
        _node("reference_numerator_clamp", "Clamp", ("reference_unit",), "reference_numerator", "DetectionTransformerDecoder.inverse_sigmoid.clamp_min", {"min": eps}),
        _node("reference_complement", "Sub", ("one", "reference_unit"), "reference_one_minus", "DetectionTransformerDecoder.inverse_sigmoid.one_minus"),
        _node("reference_denominator_clamp", "Clamp", ("reference_one_minus",), "reference_denominator", "DetectionTransformerDecoder.inverse_sigmoid.clamp_min", {"min": eps}),
        _node("reference_odds", "Div", ("reference_numerator", "reference_denominator"), "reference_ratio", "DetectionTransformerDecoder.inverse_sigmoid.ratio"),
        _node("reference_logit", "Log", ("reference_ratio",), "reference_logits", "DetectionTransformerDecoder.inverse_sigmoid.log"),
        _node("regression_xy", "Slice", ("regression",), "delta_xy", "DetectionTransformerDecoder.forward.tmp_xy", {"starts": [0], "ends": [2], "axes": [-1], "steps": [1]}),
        _node("regression_z", "Slice", ("regression",), "delta_z", "DetectionTransformerDecoder.forward.tmp_z", {"starts": [4], "ends": [5], "axes": [-1], "steps": [1]}),
        _node("reference_xy", "Slice", ("reference_logits",), "reference_xy_logits", "DetectionTransformerDecoder.forward.reference_xy", {"starts": [0], "ends": [2], "axes": [-1], "steps": [1]}),
        _node("reference_z", "Slice", ("reference_logits",), "reference_z_logit", "DetectionTransformerDecoder.forward.reference_z", {"starts": [2], "ends": [3], "axes": [-1], "steps": [1]}),
        _node("refine_xy_add", "Add", ("delta_xy", "reference_xy_logits"), "refined_xy_logits", "DetectionTransformerDecoder.forward.refine_xy"),
        _node("refine_xy_sigmoid", "Sigmoid", ("refined_xy_logits",), "refined_xy", "DetectionTransformerDecoder.forward.refine_xy"),
        _node("refine_z_add", "Add", ("delta_z", "reference_z_logit"), "refined_z_logit", "DetectionTransformerDecoder.forward.refine_z"),
        _node("refine_z_sigmoid", "Sigmoid", ("refined_z_logit",), "refined_z", "DetectionTransformerDecoder.forward.refine_z"),
        _node("next_reference_concat", "Concat", ("refined_xy", "refined_z"), "next_reference_points", "DetectionTransformerDecoder.forward.new_reference_points", {"axis": -1}),
    ]
    graph = Graph(
        name=name,
        inputs={
            "query": TensorSpec(
                (config.batch, config.queries, config.embed_dim), "float32"
            ),
            "reference_points": TensorSpec(
                (config.batch, config.queries, 3), "float32"
            ),
        },
        outputs=("regression", "next_reference_points"),
        nodes=nodes,
        constants=constants,
        metadata={
            "kind": "bevformer-decoder-reference-refinement",
            "regression_branch": "Linear-ReLU-Linear-ReLU-Linear",
            "reference_channels": {"xy": [0, 1], "z": 4},
            "detach": "inference-no-op",
            "full_detection_head": False,
        },
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


def validate_decoder_refinement_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        raise ValidationError("decoder refinement input names do not match the graph")
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"decoder refinement input {name!r} must be an ndarray")
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder refinement input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder refinement input {name!r} must be contiguous and finite"
            )
    reference = values["reference_points"]
    if np.any(reference < 0.0) or np.any(reference > 1.0):
        raise ValidationError(
            "decoder refinement reference_points must be normalized to [0, 1]"
        )


__all__ = [
    "DecoderReferenceRefinementConfig",
    "build_decoder_reference_refinement_graph",
    "decoder_refinement_weight_shapes",
    "validate_decoder_refinement_inputs",
]
