"""Six-layer BEVFormer classification and reference-aware box-code stack.

This graph consumes the layer-major outputs of
``bevformer_decoder_stack``.  It follows ``BEVFormerHead.forward`` exactly:
layer zero decodes boxes relative to the initial reference, while every later
layer uses the post-refinement reference emitted by the preceding decoder
layer.  Query selection (Top-K/NMS-free decoding) remains a separate stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field

import numpy as np

from gac.models.bevformer.detection_head import (
    DetectionHeadConfig,
    build_detection_head_graph,
    detection_head_weight_shapes,
)
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec


_NUM_LAYERS = 6


@dataclass(frozen=True, slots=True)
class DetectionStackConfig:
    """Static profile for all six BEVFormer-tiny detection projections."""

    head: DetectionHeadConfig = field(default_factory=DetectionHeadConfig)
    num_layers: int = _NUM_LAYERS

    def __post_init__(self) -> None:
        if not isinstance(self.head, DetectionHeadConfig):
            raise ValidationError(
                "detection stack head must be a DetectionHeadConfig"
            )
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers != _NUM_LAYERS
        ):
            raise ValidationError("detection stack requires exactly num_layers=6")

    @property
    def batch(self) -> int:
        return self.head.batch

    @property
    def queries(self) -> int:
        return self.head.queries

    @property
    def embed_dim(self) -> int:
        return self.head.embed_dim

    @property
    def classes(self) -> int:
        return self.head.classes

    @property
    def code_size(self) -> int:
        return self.head.code_size


def detection_stack_weight_shapes(
    config: DetectionStackConfig,
) -> tuple[dict[str, tuple[int, ...]], ...]:
    """Return six independent classification-branch tensor contracts."""

    if not isinstance(config, DetectionStackConfig):
        raise ValidationError("config must be a DetectionStackConfig")
    shapes = detection_head_weight_shapes(config.head)
    return tuple(copy.deepcopy(shapes) for _ in range(config.num_layers))


def _validate_weight_group(
    scope: str,
    weights: Mapping[str, np.ndarray],
    expected: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError(f"detection stack {scope} weights must be a mapping")
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValidationError(
            f"detection stack {scope} weight names must be non-empty strings"
        )
    if set(weights) != set(expected):
        raise ValidationError(
            f"detection stack {scope} weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"detection stack {scope} weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"detection stack {scope} weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"detection stack {scope} weight {name!r} must be contiguous and finite"
            )
        result[name] = value
    return result


def _validate_weight_sequence(
    weights: Sequence[Mapping[str, np.ndarray]],
    expected: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], ...]:
    if not isinstance(weights, (list, tuple)) or len(weights) != _NUM_LAYERS:
        length = len(weights) if isinstance(weights, (list, tuple)) else "non-sequence"
        raise ValidationError(
            "detection stack weights must contain exactly 6 layers, "
            f"got {length}"
        )
    return tuple(
        _validate_weight_group(f"layer[{index}]", group, expected)
        for index, group in enumerate(weights)
    )


def _node(
    node_id: str,
    op: str,
    inputs: tuple[str, ...],
    output: str,
    attrs: Mapping[str, object] | None,
    source: str,
) -> Node:
    return Node(
        id=node_id,
        op=op,
        inputs=inputs,
        outputs=(output,),
        attrs=dict(attrs or {}),
        source_nodes=(source,),
    )


def _slice_layer(
    composer: GraphComposer,
    *,
    source_value: str,
    index: int,
    output: str,
    shape: tuple[int, ...],
    source: str,
) -> str:
    """Select one layer-major entry and remove its leading unit axis."""

    sliced = f"{output}.with_layer_axis"
    composer.add_node(
        _node(
            f"{output}.slice",
            "Slice",
            (source_value,),
            sliced,
            {"starts": [index], "ends": [index + 1], "axes": [0], "steps": [1]},
            source,
        )
    )
    composer.add_node(
        _node(
            f"{output}.reshape",
            "Reshape",
            (sliced,),
            output,
            {"shape": list(shape)},
            source,
        )
    )
    return output


def build_detection_stack_graph(
    weights: Sequence[Mapping[str, np.ndarray]],
    config: DetectionStackConfig | None = None,
    *,
    name: str = "bevformer_detection_projection_stack",
) -> Graph:
    """Build six classification branches and reference-aware box-code projections."""

    config = config or DetectionStackConfig()
    if not isinstance(config, DetectionStackConfig):
        raise ValidationError("config must be a DetectionStackConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("detection stack graph name must be non-empty")
    expected = detection_head_weight_shapes(config.head)
    checked_weights = _validate_weight_sequence(weights, expected)

    layers = config.num_layers
    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    classes = config.classes
    code_size = config.code_size
    composer = GraphComposer(
        name,
        {
            "intermediate_states": TensorSpec(
                (layers, batch, queries, channels), "float32"
            ),
            "initial_reference_points": TensorSpec(
                (batch, queries, 3), "float32"
            ),
            "intermediate_reference_points": TensorSpec(
                (layers, batch, queries, 3), "float32"
            ),
            "regressions": TensorSpec(
                (layers, batch, queries, code_size), "float32"
            ),
        },
        metadata={
            "kind": "bevformer-six-layer-detection-projection-stack-static",
            "full_detection_projection_stack": True,
            "num_layers": layers,
            "batch_first_abi": True,
            "input_layouts": {
                "intermediate_states": ["layer", "batch", "query", "channel"],
                "intermediate_reference_points": [
                    "layer",
                    "batch",
                    "query",
                    "xyz",
                ],
                "regressions": ["layer", "batch", "query", "code"],
            },
            "reference_selection": {
                "layer0": "initial_reference_points",
                "layers1_to_5": "intermediate_reference_points[layer - 1]",
            },
            "box_coordinates": (
                "x/y/z converted to point-cloud range; size is logarithmic "
                "and rotation is sin/cos until NMS-free denormalization"
            ),
            "classification": "independent branch per decoder layer",
            "outputs_are_all_raw_queries": True,
            "nms_free_topk": "outside-this-graph",
            "topk_and_final_nms_free_decode": "outside-this-graph",
            "profile": {
                "batch": batch,
                "queries": queries,
                "embed_dim": channels,
                "classes": classes,
                "code_size": code_size,
                "point_cloud_range": list(config.head.point_cloud_range),
            },
        },
    )

    layer_logits: list[str] = []
    layer_scores: list[str] = []
    layer_boxes: list[str] = []
    for index in range(layers):
        prefix = f"layer{index}"
        query = _slice_layer(
            composer,
            source_value="intermediate_states",
            index=index,
            output=f"{prefix}.query",
            shape=(batch, queries, channels),
            source="BEVFormerHead.forward.hs[layer]",
        )
        regression = _slice_layer(
            composer,
            source_value="regressions",
            index=index,
            output=f"{prefix}.regression",
            shape=(batch, queries, code_size),
            source="BEVFormerHead.forward.reg_branches[layer](hs[layer])",
        )
        if index == 0:
            reference = "initial_reference_points"
        else:
            reference = _slice_layer(
                composer,
                source_value="intermediate_reference_points",
                index=index - 1,
                output=f"{prefix}.reference_points",
                shape=(batch, queries, 3),
                source="BEVFormerHead.forward.inter_references[layer - 1]",
            )

        head_graph = build_detection_head_graph(
            checked_weights[index], config.head
        )
        if index == layers - 1:
            output_bindings = {
                "class_logits": "final_class_logits",
                "class_scores": "final_class_scores",
                "boxes": "final_boxes",
            }
        else:
            output_bindings = {
                "class_logits": f"{prefix}.class_logits",
                "class_scores": f"{prefix}.class_scores",
                "boxes": f"{prefix}.boxes",
            }
        projected = composer.inline(
            head_graph,
            prefix=f"{prefix}.head",
            bindings={
                "query": query,
                "reference_points": reference,
                "regression": regression,
            },
            output_bindings=output_bindings,
        )
        layer_logits.append(projected["class_logits"])
        layer_scores.append(projected["class_scores"])
        layer_boxes.append(projected["boxes"])

    stacked_outputs = (
        ("class_logits", layer_logits, classes),
        ("class_scores", layer_scores, classes),
        ("boxes", layer_boxes, code_size),
    )
    for output_name, values, width in stacked_outputs:
        flat = f"all_{output_name}_flat"
        composer.add_node(
            _node(
                f"all_{output_name}.concat",
                "Concat",
                tuple(values),
                flat,
                {"axis": 0},
                f"BEVFormerHead.forward.stack.{output_name}",
            )
        )
        composer.add_node(
            _node(
                f"all_{output_name}.reshape",
                "Reshape",
                (flat,),
                f"all_{output_name}",
                {"shape": [layers, batch, queries, width]},
                f"BEVFormerHead.forward.stack.{output_name}",
            )
        )

    return composer.build(
        (
            "all_class_logits",
            "all_class_scores",
            "all_boxes",
            "final_class_logits",
            "final_class_scores",
            "final_boxes",
        )
    )


def validate_detection_stack_inputs(
    graph: Graph, values: Mapping[str, np.ndarray]
) -> None:
    """Validate decoder-stack tensors before detection projection."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping):
        raise ValidationError("detection stack inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValidationError(
            "detection stack input names must be non-empty strings"
        )
    if set(values) != set(graph.inputs):
        raise ValidationError(
            f"detection stack input names must be {sorted(graph.inputs)}, "
            f"got {sorted(repr(name) for name in values)}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"detection stack input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"detection stack input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"detection stack input {name!r} must be contiguous and finite"
            )
    for name in ("initial_reference_points", "intermediate_reference_points"):
        reference = values[name]
        if np.any(reference < 0.0) or np.any(reference > 1.0):
            raise ValidationError(
                f"detection stack {name} must be normalized to [0, 1]"
            )


__all__ = [
    "DetectionStackConfig",
    "build_detection_stack_graph",
    "detection_stack_weight_shapes",
    "validate_detection_stack_inputs",
]
