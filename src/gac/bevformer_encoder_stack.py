"""Static three-layer BEVFormer-tiny encoder stack for inference.

The stack computes camera projection once, then executes three independent
``TemporalSelfAttention -> norm0 -> SpatialCrossAttention -> norm1 -> FFN ->
norm2`` layers.  In history mode every layer consumes the same explicit queue
``[previous BEV, initial current BEV]``.  In cold mode each layer constructs a
two-entry queue from its own input query inside the graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import copy

import numpy as np

from .bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    encoder_layer_weight_shapes,
)
from .bevformer_geometry import build_bev_projection_graph
from .bevformer_spatial import build_spatial_cross_attention_graph
from .bevformer_temporal import build_temporal_self_attention_graph
from .compose import GraphComposer
from .errors import ValidationError
from .ir import Graph, Node, TensorSpec


_NUM_LAYERS = 3


@dataclass(frozen=True, slots=True)
class BevFormerEncoderStackConfig:
    """Compile-time profile for the fixed three-layer tiny encoder."""

    layer: BevFormerEncoderLayerConfig = field(
        default_factory=BevFormerEncoderLayerConfig
    )
    num_layers: int = _NUM_LAYERS
    history_mode: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layer, BevFormerEncoderLayerConfig):
            raise ValidationError(
                "encoder stack layer must be a BevFormerEncoderLayerConfig"
            )
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers != _NUM_LAYERS
        ):
            raise ValidationError("encoder stack requires exactly num_layers=3")
        if not isinstance(self.history_mode, bool):
            raise ValidationError("encoder stack history_mode must be bool")

    @property
    def batch(self) -> int:
        return self.layer.batch

    @property
    def queries(self) -> int:
        return self.layer.queries

    @property
    def embed_dim(self) -> int:
        return self.layer.embed_dim


def encoder_stack_weight_shapes(
    config: BevFormerEncoderStackConfig,
) -> tuple[dict[str, dict[str, tuple[int, ...]]], ...]:
    """Return one exact nested tensor inventory for each of three layers."""

    if not isinstance(config, BevFormerEncoderStackConfig):
        raise ValidationError("config must be a BevFormerEncoderStackConfig")
    inventory = encoder_layer_weight_shapes(config.layer)
    return tuple(copy.deepcopy(inventory) for _ in range(config.num_layers))


def _validate_weight_group(
    scope: str,
    weights: Mapping[str, np.ndarray],
    expected: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError(f"encoder stack {scope} weights must be a mapping")
    if any(not isinstance(name, str) for name in weights):
        raise ValidationError(f"encoder stack {scope} weight names must be strings")
    if set(weights) != set(expected):
        raise ValidationError(
            f"encoder stack {scope} weight names must be {sorted(expected)}, "
            f"got {sorted(repr(name) for name in weights)}"
        )
    result: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"encoder stack {scope} weight {name!r} must be a NumPy ndarray"
            )
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"encoder stack {scope} weight {name!r} requires "
                f"{shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"encoder stack {scope} weight {name!r} must be contiguous and finite"
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
            f"encoder stack {scope} weights must contain exactly 3 layers, got {length}"
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


def build_bevformer_encoder_stack_graph(
    temporal_weights: Sequence[Mapping[str, np.ndarray]],
    spatial_weights: Sequence[Mapping[str, np.ndarray]],
    layer_weights: Sequence[Mapping[str, np.ndarray]],
    config: BevFormerEncoderStackConfig | None = None,
    *,
    name: str = "bevformer_encoder_stack",
    select_history_at_runtime: bool = False,
) -> Graph:
    """Build the fixed three-layer BEVFormer-tiny encoder stack.

    History mode exposes one shared ``temporal_value`` input.  Its queue axis
    must be ordered as ``[previous BEV, initial current BEV]``.  Cold mode has
    no temporal queue input; each layer concatenates its own input query with
    itself before temporal attention.  Camera geometry is projected exactly
    once and shared by all three spatial-attention calls.
    """

    config = config or BevFormerEncoderStackConfig()
    if not isinstance(config, BevFormerEncoderStackConfig):
        raise ValidationError("config must be a BevFormerEncoderStackConfig")
    if not isinstance(name, str) or not name:
        raise ValidationError("encoder stack graph name must be non-empty")
    if not isinstance(select_history_at_runtime, bool):
        raise ValidationError("select_history_at_runtime must be bool")
    if select_history_at_runtime and not config.history_mode:
        raise ValidationError(
            "select_history_at_runtime requires encoder stack history_mode=True"
        )

    expected = encoder_layer_weight_shapes(config.layer)
    checked_temporal = _validate_weight_sequence(
        "temporal", temporal_weights, expected["temporal"]
    )
    checked_spatial = _validate_weight_sequence(
        "spatial", spatial_weights, expected["spatial"]
    )
    checked_layer = _validate_weight_sequence(
        "layer", layer_weights, expected["layer"]
    )

    layer_config = config.layer
    temporal_config = layer_config.temporal
    spatial_config = layer_config.spatial
    projection_config = layer_config.projection
    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    levels = temporal_config.levels

    inputs: dict[str, TensorSpec] = {
        "query": TensorSpec((batch, queries, channels), "float32"),
        "query_pos": TensorSpec((batch, queries, channels), "float32"),
        "spatial_query_pos": TensorSpec((batch, queries, channels), "float32"),
        "reference_points": TensorSpec((batch, queries, levels, 2), "float32"),
        "image_value": TensorSpec(
            (
                batch,
                spatial_config.cameras,
                spatial_config.spatial_size,
                channels,
            ),
            "float32",
        ),
        "lidar2img": TensorSpec(
            (batch, projection_config.cameras, 4, 4), "float32"
        ),
    }
    if config.history_mode:
        inputs["temporal_value"] = TensorSpec(
            (
                batch * temporal_config.queue_length,
                queries,
                channels,
            ),
            "float32",
        )
        inputs["prev_reference_points"] = TensorSpec(
            (batch, queries, levels, 2), "float32"
        )
        if select_history_at_runtime:
            inputs["has_prev_bev"] = TensorSpec((batch,), "int32")

    stage_order = [
        "temporal-self-attention",
        "norm0",
        "spatial-cross-attention",
        "norm1",
        "ffn-residual",
        "norm2",
    ]
    composer = GraphComposer(
        name,
        inputs,
        metadata={
            "kind": "bevformer-three-layer-encoder-stack-static",
            "full_encoder_stack": True,
            "num_layers": config.num_layers,
            "history_mode": config.history_mode,
            "temporal_mode": (
                "runtime-history-or-cold"
                if select_history_at_runtime
                else ("history" if config.history_mode else "cold")
            ),
            "temporal_value_contract": (
                "warm uses shared [previous BEV, initial current BEV]; cold uses "
                "each layer's [layer input query, layer input query]"
                if select_history_at_runtime
                else (
                    "shared explicit [previous BEV, initial current BEV]"
                    if config.history_mode
                    else "each layer concatenates [layer input query, layer input query]"
                )
            ),
            "geometry": "single in-graph lidar2img projection shared by all layers",
            "geometry_evaluations": 1,
            "layer_operation_order": stage_order,
            "dropout": "inference-identity",
            "position_contract": {
                "query_pos": "shared temporal-self-attention BEV position",
                "spatial_query_pos": (
                    "shared explicit spatial position; bind zeros for official tiny encoder"
                ),
            },
            "profile": {
                "batch": batch,
                "queries": queries,
                "embed_dim": channels,
                "heads": temporal_config.heads,
                "cameras": spatial_config.cameras,
                "bev_hw": [temporal_config.bev_h, temporal_config.bev_w],
                "feature_shapes": [
                    list(shape) for shape in spatial_config.spatial_shapes
                ],
                "z_anchors": spatial_config.z_anchors,
                "ffn_hidden_dim": layer_config.ffn_hidden_dim,
            },
        },
    )

    projection_graph = build_bev_projection_graph(projection_config)
    projected = composer.inline(
        projection_graph,
        prefix="geometry",
        bindings={"lidar2img": "lidar2img"},
        output_bindings={
            "reference_points": "camera_reference_points",
            "visibility": "camera_visibility",
        },
    )

    current_query = "query"
    for index in range(config.num_layers):
        prefix = f"layer{index}"
        temporal_graph = build_temporal_self_attention_graph(
            checked_temporal[index], temporal_config
        )
        spatial_graph = build_spatial_cross_attention_graph(
            checked_spatial[index], spatial_config
        )

        if config.history_mode:
            temporal_value = "temporal_value"
            previous_reference = "prev_reference_points"
            if select_history_at_runtime:
                cold_temporal_value = f"{prefix}.cold_temporal_value"
                composer.add_node(
                    _node(
                        f"{prefix}.cold_temporal_value_concat",
                        "Concat",
                        (current_query, current_query),
                        cold_temporal_value,
                        {"axis": 0},
                        "BEVFormerEncoder.forward.cold_prev_bev",
                    )
                )
                selected_temporal_value = f"{prefix}.selected_temporal_value"
                composer.add_node(
                    _node(
                        f"{prefix}.select_temporal_value",
                        "Where",
                        (
                            "has_prev_bev",
                            temporal_value,
                            cold_temporal_value,
                        ),
                        selected_temporal_value,
                        None,
                        "BEVFormerEncoder.forward.prev_bev",
                    )
                )
                temporal_value = selected_temporal_value
        else:
            temporal_value = f"{prefix}.cold_temporal_value"
            composer.add_node(
                _node(
                    f"{prefix}.cold_temporal_value_concat",
                    "Concat",
                    (current_query, current_query),
                    temporal_value,
                    {"axis": 0},
                    "BEVFormerEncoder.forward.cold_prev_bev",
                )
            )
            previous_reference = "reference_points"

        temporal_output_name = f"{prefix}.temporal_output"
        temporal = composer.inline(
            temporal_graph,
            prefix=f"{prefix}.temporal",
            bindings={
                "query": current_query,
                "temporal_value": temporal_value,
                "query_pos": "query_pos",
                "prev_reference_points": previous_reference,
                "reference_points": "reference_points",
            },
            output_bindings={"output": temporal_output_name},
        )

        for weight_name, value in checked_layer[index].items():
            composer.add_constant(f"{prefix}.{weight_name}", value)

        epsilon = float(layer_config.norm_epsilon)
        norm0_output = f"{prefix}.norm0_output"
        composer.add_node(
            _node(
                f"{prefix}.norm0",
                "LayerNorm",
                (
                    temporal["output"],
                    f"{prefix}.norm0_weight",
                    f"{prefix}.norm0_bias",
                ),
                norm0_output,
                {"axis": -1, "epsilon": epsilon},
                f"BEVFormerLayer[{index}].forward.norm0",
            )
        )

        spatial_output_name = f"{prefix}.spatial_output"
        spatial = composer.inline(
            spatial_graph,
            prefix=f"{prefix}.spatial",
            bindings={
                "query": norm0_output,
                "query_pos": "spatial_query_pos",
                "value": "image_value",
                "reference_points": projected["reference_points"],
                "visibility": projected["visibility"],
            },
            output_bindings={"output": spatial_output_name},
        )

        norm1_output = f"{prefix}.norm1_output"
        composer.add_node(
            _node(
                f"{prefix}.norm1",
                "LayerNorm",
                (
                    spatial["output"],
                    f"{prefix}.norm1_weight",
                    f"{prefix}.norm1_bias",
                ),
                norm1_output,
                {"axis": -1, "epsilon": epsilon},
                f"BEVFormerLayer[{index}].forward.norm1",
            )
        )

        ffn_fc1_mm = f"{prefix}.ffn_fc1_mm"
        composer.add_node(
            _node(
                f"{prefix}.ffn_fc1",
                "MatMul",
                (norm1_output, f"{prefix}.ffn_fc1_weight"),
                ffn_fc1_mm,
                None,
                f"BEVFormerLayer[{index}].ffn.layers.0.linear",
            )
        )
        ffn_fc1 = f"{prefix}.ffn_fc1"
        composer.add_node(
            _node(
                f"{prefix}.ffn_fc1_bias_add",
                "Add",
                (ffn_fc1_mm, f"{prefix}.ffn_fc1_bias"),
                ffn_fc1,
                None,
                f"BEVFormerLayer[{index}].ffn.layers.0.linear",
            )
        )
        ffn_hidden = f"{prefix}.ffn_hidden"
        composer.add_node(
            _node(
                f"{prefix}.ffn_relu",
                "ReLU",
                (ffn_fc1,),
                ffn_hidden,
                None,
                f"BEVFormerLayer[{index}].ffn.layers.0.relu",
            )
        )
        ffn_fc2_mm = f"{prefix}.ffn_fc2_mm"
        composer.add_node(
            _node(
                f"{prefix}.ffn_fc2",
                "MatMul",
                (ffn_hidden, f"{prefix}.ffn_fc2_weight"),
                ffn_fc2_mm,
                None,
                f"BEVFormerLayer[{index}].ffn.layers.1.linear",
            )
        )
        ffn_projected = f"{prefix}.ffn_projected"
        composer.add_node(
            _node(
                f"{prefix}.ffn_fc2_bias_add",
                "Add",
                (ffn_fc2_mm, f"{prefix}.ffn_fc2_bias"),
                ffn_projected,
                None,
                f"BEVFormerLayer[{index}].ffn.layers.1.linear",
            )
        )
        ffn_residual = f"{prefix}.ffn_residual"
        composer.add_node(
            _node(
                f"{prefix}.ffn_residual_add",
                "Add",
                (ffn_projected, norm1_output),
                ffn_residual,
                None,
                f"BEVFormerLayer[{index}].ffn.residual",
            )
        )

        layer_output = "output" if index == config.num_layers - 1 else f"{prefix}.output"
        composer.add_node(
            _node(
                f"{prefix}.norm2",
                "LayerNorm",
                (
                    ffn_residual,
                    f"{prefix}.norm2_weight",
                    f"{prefix}.norm2_bias",
                ),
                layer_output,
                {"axis": -1, "epsilon": epsilon},
                f"BEVFormerLayer[{index}].forward.norm2",
            )
        )
        current_query = layer_output

    return composer.build(
        ("output",),
        metadata={
            "component_graphs": {
                "geometry": projection_graph.metadata.get("component"),
                "temporal": "three independently weighted temporal attention calls",
                "spatial": "three independently weighted spatial attention calls",
            }
        },
    )


def validate_encoder_stack_inputs(
    graph: Graph,
    values: Mapping[str, np.ndarray],
) -> None:
    """Validate the selected history/cold ABI before execution."""

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    if not isinstance(values, Mapping) or set(values) != set(graph.inputs):
        actual = (
            sorted(repr(name) for name in values)
            if isinstance(values, Mapping)
            else type(values).__name__
        )
        raise ValidationError(
            f"encoder stack input names must be {sorted(graph.inputs)}, got {actual}"
        )
    for name, spec in graph.inputs.items():
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(
                f"encoder stack input {name!r} must be a NumPy ndarray"
            )
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(
                f"encoder stack input {name!r} requires "
                f"{spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"encoder stack input {name!r} must be contiguous and finite"
            )

    if "has_prev_bev" in graph.inputs:
        flags = values["has_prev_bev"]
        if np.any((flags != 0) & (flags != 1)):
            raise ValidationError(
                "encoder stack has_prev_bev values must be 0 or 1"
            )

    # The unshifted current-frame grid is normalized by construction.  Ego
    # translation is added to the previous-frame grid before temporal MSDA,
    # so border coordinates in ``prev_reference_points`` may legitimately
    # fall outside [0, 1]; zero padding handles those samples.
    reference = values["reference_points"]
    if np.any(reference < 0.0) or np.any(reference > 1.0):
        raise ValidationError(
            "encoder stack input 'reference_points' must be normalized to [0, 1]"
        )

    if graph.metadata.get("history_mode") is True:
        batch = values["query"].shape[0]
        current_queue_value = values["temporal_value"][batch : batch * 2]
        if not np.array_equal(current_queue_value, values["query"]):
            raise ValidationError(
                "history temporal_value second queue entry must equal the initial query"
            )


__all__ = [
    "BevFormerEncoderStackConfig",
    "build_bevformer_encoder_stack_graph",
    "encoder_stack_weight_shapes",
    "validate_encoder_stack_inputs",
]
