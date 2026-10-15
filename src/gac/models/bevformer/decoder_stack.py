"""Static six-layer BEVFormer-tiny decoder stack for inference.

The stack owns learned object-query initialization, executes six independent
decoder transformer layers against one fixed BEV memory tensor, and refines a
three-dimensional reference after every layer.  Its public ABI is batch-first;
the returned intermediate tensors retain the official layer-major semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field

import numpy as np

from gac.models.bevformer.decoder_layer import (
    BevFormerDecoderLayerConfig,
    build_bevformer_decoder_layer_graph,
    decoder_layer_weight_shapes,
)
from gac.models.bevformer.query_init import (
    DecoderQueryInitConfig,
    build_decoder_query_init_graph,
    decoder_query_init_weight_shapes,
)
from gac.models.bevformer.refinement import (
    DecoderReferenceRefinementConfig,
    build_decoder_reference_refinement_graph,
    decoder_refinement_weight_shapes,
)
from gac.compiler.compose import GraphComposer
from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec


_NUM_LAYERS = 6


@dataclass(frozen=True, slots=True)
class BevFormerDecoderStackConfig:
    """Mutually consistent static profile for the tiny six-layer decoder."""

    query_init: DecoderQueryInitConfig = field(
        default_factory=DecoderQueryInitConfig
    )
    layer: BevFormerDecoderLayerConfig = field(
        default_factory=BevFormerDecoderLayerConfig
    )
    refinement: DecoderReferenceRefinementConfig = field(
        default_factory=DecoderReferenceRefinementConfig
    )
    num_layers: int = _NUM_LAYERS

    def __post_init__(self) -> None:
        if not isinstance(self.query_init, DecoderQueryInitConfig):
            raise ValidationError(
                "decoder stack query_init must be a DecoderQueryInitConfig"
            )
        if not isinstance(self.layer, BevFormerDecoderLayerConfig):
            raise ValidationError(
                "decoder stack layer must be a BevFormerDecoderLayerConfig"
            )
        if not isinstance(self.refinement, DecoderReferenceRefinementConfig):
            raise ValidationError(
                "decoder stack refinement must be a "
                "DecoderReferenceRefinementConfig"
            )
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers != _NUM_LAYERS
        ):
            raise ValidationError("decoder stack requires exactly num_layers=6")

        mismatches: list[str] = []
        if not (
            self.query_init.batch
            == self.layer.batch
            == self.refinement.batch
        ):
            mismatches.append("batch")
        if not (
            self.query_init.queries
            == self.layer.queries
            == self.refinement.queries
        ):
            mismatches.append("queries")
        if not (
            self.query_init.embed_dim
            == self.layer.embed_dim
            == self.refinement.embed_dim
        ):
            mismatches.append("embed_dim")
        if self.layer.cross_attention.levels != 1:
            mismatches.append("cross_attention.levels")
        if mismatches:
            raise ValidationError(
                "decoder stack component profiles disagree on: "
                + ", ".join(mismatches)
            )

    @property
    def batch(self) -> int:
        return self.layer.batch

    @property
    def queries(self) -> int:
        return self.layer.queries

    @property
    def embed_dim(self) -> int:
        return self.layer.embed_dim

    @property
    def code_size(self) -> int:
        return self.refinement.code_size


WeightShapeMap = dict[str, tuple[int, ...]]


def decoder_stack_weight_shapes(
    config: BevFormerDecoderStackConfig,
) -> dict[str, WeightShapeMap | tuple[WeightShapeMap, ...]]:
    """Return the exact learned-tensor inventory for all six layers."""

    if not isinstance(config, BevFormerDecoderStackConfig):
        raise ValidationError("config must be a BevFormerDecoderStackConfig")
    layer_shapes = decoder_layer_weight_shapes(config.layer)
    return {
        "query_init": decoder_query_init_weight_shapes(config.query_init),
        "self_attention": tuple(
            copy.deepcopy(layer_shapes["self_attention"])
            for _ in range(config.num_layers)
        ),
        "cross_attention": tuple(
            copy.deepcopy(layer_shapes["cross_attention"])
            for _ in range(config.num_layers)
        ),
        "layer": tuple(
            copy.deepcopy(layer_shapes["layer"])
            for _ in range(config.num_layers)
        ),
        "refinement": tuple(
            copy.deepcopy(decoder_refinement_weight_shapes(config.refinement))
            for _ in range(config.num_layers)
        ),
    }


def _validate_weight_group(
    scope: str,
    weights: Mapping[str, np.ndarray],
    expected: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError(f"decoder stack {scope} weights must be a mapping")
    if any(not isinstance(name, str) or not name for name in weights):
        raise ValidationError(
            f"decoder stack {scope} weight names must be non-empty strings"
        )
    if set(weights) != set(expected):
        raise ValidationError(
            f"decoder stack {scope} weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder stack {scope} weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"decoder stack {scope} weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder stack {scope} weight {name!r} must be contiguous and finite"
            )
        result[name] = value
    return result


def _validate_weight_sequence(
    scope: str,
    weights: Sequence[Mapping[str, np.ndarray]],
    expected: Mapping[str, tuple[int, ...]],
) -> tuple[dict[str, np.ndarray], ...]:
    if not isinstance(weights, (list, tuple)) or len(weights) != _NUM_LAYERS:
        length = len(weights) if isinstance(weights, (list, tuple)) else "non-sequence"
        raise ValidationError(
            f"decoder stack {scope} weights must contain exactly 6 layers, "
            f"got {length}"
        )
    return tuple(
        _validate_weight_group(f"{scope}[{index}]", group, expected)
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


def build_bevformer_decoder_stack_graph(
    query_init_weights: Mapping[str, np.ndarray],
    self_attention_weights: Sequence[Mapping[str, np.ndarray]],
    cross_attention_weights: Sequence[Mapping[str, np.ndarray]],
    layer_weights: Sequence[Mapping[str, np.ndarray]],
    refinement_weights: Sequence[Mapping[str, np.ndarray]],
    config: BevFormerDecoderStackConfig | None = None,
    *,
    name: str = "bevformer_decoder_stack",
) -> Graph:
    """Build learned initialization and all six refined decoder layers.

    The only runtime input is the encoded BEV memory.  Each layer slices the
    current 3D reference to ``xy`` and reshapes it to ``[B, Q, 1, 2]`` for
    deformable cross-attention.  The corresponding regression branch then
    produces the next detached reference; detach is an inference no-op.
    """

    config = config or BevFormerDecoderStackConfig()
    if not isinstance(config, BevFormerDecoderStackConfig):
        raise ValidationError("config must be a BevFormerDecoderStackConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("decoder stack graph name must be non-empty")

    shapes = decoder_stack_weight_shapes(config)
    query_shapes = shapes["query_init"]
    self_shapes = shapes["self_attention"]
    cross_shapes = shapes["cross_attention"]
    layer_shapes = shapes["layer"]
    refinement_shapes = shapes["refinement"]
    assert isinstance(query_shapes, dict)
    assert isinstance(self_shapes, tuple)
    assert isinstance(cross_shapes, tuple)
    assert isinstance(layer_shapes, tuple)
    assert isinstance(refinement_shapes, tuple)

    checked_query_init = _validate_weight_group(
        "query_init", query_init_weights, query_shapes
    )
    checked_self = _validate_weight_sequence(
        "self_attention", self_attention_weights, self_shapes[0]
    )
    checked_cross = _validate_weight_sequence(
        "cross_attention", cross_attention_weights, cross_shapes[0]
    )
    checked_layer = _validate_weight_sequence(
        "layer", layer_weights, layer_shapes[0]
    )
    checked_refinement = _validate_weight_sequence(
        "refinement", refinement_weights, refinement_shapes[0]
    )

    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    code_size = config.code_size
    cross = config.layer.cross_attention
    composer = GraphComposer(
        name,
        {
            "value": TensorSpec(
                (batch, cross.spatial_size, channels), "float32"
            )
        },
        metadata={
            "kind": "bevformer-six-layer-decoder-stack-static",
            "full_decoder": True,
            "full_decoder_stack": True,
            "num_layers": config.num_layers,
            "return_intermediate": True,
            "batch_first_abi": True,
            "query_initialization": "learned object-query embedding split once",
            "bev_value_contract": "one fixed encoder BEV memory shared by all layers",
            "reference_flow": (
                "initial sigmoid xyz; each layer consumes current xy and emits "
                "post-regression sigmoid xyz"
            ),
            "reference_detach": "inference-no-op between layers",
            "intermediate_state_layout": [
                "layer",
                "batch",
                "query",
                "channel",
            ],
            "intermediate_reference_layout": [
                "layer",
                "batch",
                "query",
                "xyz",
            ],
            "intermediate_reference_semantics": (
                "post-refinement reference from the corresponding decoder layer"
            ),
            "layer_operation_order": [
                "self_attn",
                "norm0",
                "cross_attn",
                "norm1",
                "ffn",
                "norm2",
                "regression",
                "reference_refinement",
            ],
            "profile": {
                "batch": batch,
                "queries": queries,
                "bev_h": cross.bev_h,
                "bev_w": cross.bev_w,
                "embed_dim": channels,
                "heads": config.layer.self_attention.heads,
                "sampling_points": cross.sampling_points,
                "ffn_hidden_dim": config.layer.ffn.hidden_dim,
                "code_size": code_size,
            },
            "excluded_stages": [
                "classification-branches",
                "metric-box-decoding",
                "nms-free-top-k-selection",
            ],
        },
    )

    query_init_graph = build_decoder_query_init_graph(
        checked_query_init, config.query_init
    )
    initialized = composer.inline(
        query_init_graph,
        prefix="query_init",
        bindings={},
        output_bindings={
            "query": "initial_query",
            "query_pos": "query_pos",
            "reference_points": "initial_reference_points",
        },
    )

    current_query = initialized["query"]
    current_reference = initialized["reference_points"]
    intermediate_states: list[str] = []
    intermediate_references: list[str] = []
    regressions: list[str] = []

    for index in range(config.num_layers):
        prefix = f"layer{index}"
        reference_xy = f"{prefix}.reference_xy"
        composer.add_node(
            _node(
                f"{prefix}.reference_xy_slice",
                "Slice",
                (current_reference,),
                reference_xy,
                {"starts": [0], "ends": [2], "axes": [-1], "steps": [1]},
                "DetectionTransformerDecoder.forward.reference_points[..., :2]",
            )
        )
        cross_reference = f"{prefix}.cross_reference_points"
        composer.add_node(
            _node(
                f"{prefix}.reference_level_axis",
                "Reshape",
                (reference_xy,),
                cross_reference,
                {"shape": [batch, queries, cross.levels, 2]},
                "DetectionTransformerDecoder.forward.reference_points_input.unsqueeze",
            )
        )

        layer_graph = build_bevformer_decoder_layer_graph(
            checked_self[index],
            checked_cross[index],
            checked_layer[index],
            config.layer,
        )
        layer_output_name = f"{prefix}.output"
        layer_output = composer.inline(
            layer_graph,
            prefix=f"{prefix}.transformer",
            bindings={
                "query": current_query,
                "query_pos": initialized["query_pos"],
                "value": "value",
                "reference_points": cross_reference,
            },
            output_bindings={"output": layer_output_name},
        )["output"]

        refinement_graph = build_decoder_reference_refinement_graph(
            checked_refinement[index], config.refinement
        )
        regression_name = f"{prefix}.regression"
        next_reference_name = f"{prefix}.next_reference_points"
        refined = composer.inline(
            refinement_graph,
            prefix=f"{prefix}.refinement",
            bindings={
                "query": layer_output,
                "reference_points": current_reference,
            },
            output_bindings={
                "regression": regression_name,
                "next_reference_points": next_reference_name,
            },
        )

        current_query = layer_output
        current_reference = refined["next_reference_points"]
        intermediate_states.append(layer_output)
        intermediate_references.append(current_reference)
        regressions.append(refined["regression"])

    composer.add_node(
        _node(
            "intermediate_states_concat",
            "Concat",
            tuple(intermediate_states),
            "intermediate_states_flat",
            {"axis": 0},
            "DetectionTransformerDecoder.forward.torch.stack.intermediate",
        )
    )
    composer.add_node(
        _node(
            "intermediate_states_stack",
            "Reshape",
            ("intermediate_states_flat",),
            "intermediate_states",
            {"shape": [config.num_layers, batch, queries, channels]},
            "DetectionTransformerDecoder.forward.torch.stack.intermediate",
        )
    )
    composer.add_node(
        _node(
            "intermediate_references_concat",
            "Concat",
            tuple(intermediate_references),
            "intermediate_reference_points_flat",
            {"axis": 0},
            "DetectionTransformerDecoder.forward.torch.stack.references",
        )
    )
    composer.add_node(
        _node(
            "intermediate_references_stack",
            "Reshape",
            ("intermediate_reference_points_flat",),
            "intermediate_reference_points",
            {"shape": [config.num_layers, batch, queries, 3]},
            "DetectionTransformerDecoder.forward.torch.stack.references",
        )
    )
    composer.add_node(
        _node(
            "regressions_concat",
            "Concat",
            tuple(regressions),
            "regressions_flat",
            {"axis": 0},
            "BEVFormerHead.reg_branches",
        )
    )
    composer.add_node(
        _node(
            "regressions_stack",
            "Reshape",
            ("regressions_flat",),
            "regressions",
            {"shape": [config.num_layers, batch, queries, code_size]},
            "BEVFormerHead.reg_branches",
        )
    )

    return composer.build(
        (
            "intermediate_states",
            "initial_reference_points",
            "intermediate_reference_points",
            "regressions",
            current_query,
            current_reference,
        ),
        metadata={
            "final_query_output": current_query,
            "final_reference_output": current_reference,
        },
    )


def validate_decoder_stack_inputs(
    graph: Graph, inputs: Mapping[str, np.ndarray]
) -> None:
    """Validate the fixed encoded-BEV runtime ABI."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(inputs, Mapping):
        raise ValidationError("decoder stack inputs must be a mapping")
    if any(not isinstance(name, str) or not name for name in inputs):
        raise ValidationError("decoder stack input names must be non-empty strings")
    if set(inputs) != set(graph.inputs):
        raise ValidationError(
            f"decoder stack input names must be {sorted(graph.inputs)}, "
            f"got {sorted(repr(name) for name in inputs)}"
        )
    for name, spec in graph.inputs.items():
        value = inputs[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"decoder stack input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"decoder stack input {name!r} requires {spec.shape}/{spec.dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"decoder stack input {name!r} must be contiguous and finite"
            )


__all__ = [
    "BevFormerDecoderStackConfig",
    "build_bevformer_decoder_stack_graph",
    "decoder_stack_weight_shapes",
    "validate_decoder_stack_inputs",
]
