"""Canonical graph for Galaxea G0.5 Action Expert layer-zero MLP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from gac.compiler.passes import ShapeInferencePass
from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.models.galaxeavla.contract import (
    CANONICAL_WEIGHT_NAMES,
    DOWN_PROJ_KEY,
    GATE_PROJ_KEY,
    GalaxeaQwen35MLPContract,
    SOURCE_PREFIX,
    UP_PROJ_KEY,
)


_SOURCE_TO_CANONICAL = (
    (GATE_PROJ_KEY, "gate_proj_weight"),
    (UP_PROJ_KEY, "up_proj_weight"),
    (DOWN_PROJ_KEY, "down_proj_weight"),
)


def _selected_contract(
    contract: GalaxeaQwen35MLPContract | None,
) -> GalaxeaQwen35MLPContract:
    selected = GalaxeaQwen35MLPContract() if contract is None else contract
    if not isinstance(selected, GalaxeaQwen35MLPContract):
        raise ValidationError("contract must be a GalaxeaQwen35MLPContract")
    return selected


def canonicalize_qwen35_mlp_weights(
    loaded: object,
    contract: GalaxeaQwen35MLPContract | None = None,
) -> dict[str, np.ndarray]:
    """Select and transpose the three verified PyTorch-layout tensors."""

    from gac.artifacts.weights import LoadedWeights

    if not isinstance(loaded, LoadedWeights):
        raise ValidationError("Qwen3.5 MLP weights must be verified LoadedWeights")
    selected = _selected_contract(contract)
    missing = [name for name in selected.source_weight_shapes if name not in loaded]
    if missing:
        raise ValidationError(f"Qwen3.5 MLP weights are missing checkpoint keys: {missing}")

    canonical: dict[str, np.ndarray] = {}
    for source_name, canonical_name in _SOURCE_TO_CANONICAL:
        value = loaded[source_name]
        expected = selected.source_weight_shapes[source_name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"checkpoint weight {source_name!r} must be a NumPy ndarray")
        if value.shape != expected or value.dtype != np.float32:
            raise ValidationError(
                f"checkpoint weight {source_name!r} requires {expected}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(
                f"checkpoint weight {source_name!r} must be contiguous and finite"
            )
        canonical[canonical_name] = np.ascontiguousarray(value.T).copy()
    return canonical


def _copy_canonical_weights(
    weights: Mapping[str, np.ndarray], contract: GalaxeaQwen35MLPContract
) -> dict[str, np.ndarray]:
    if not isinstance(weights, Mapping):
        raise ValidationError("Qwen3.5 MLP canonical weights must be a mapping")
    if set(weights) != set(CANONICAL_WEIGHT_NAMES):
        raise ValidationError(
            "Qwen3.5 MLP canonical weight names must be "
            f"{sorted(CANONICAL_WEIGHT_NAMES)}, got {sorted(map(repr, weights))}"
        )
    copied: dict[str, np.ndarray] = {}
    for name, shape in contract.canonical_weight_shapes.items():
        value = weights[name]
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"canonical weight {name!r} must be a NumPy ndarray")
        if value.shape != shape or value.dtype != np.float32:
            raise ValidationError(
                f"canonical weight {name!r} requires {shape}/float32, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValidationError(f"canonical weight {name!r} must be contiguous and finite")
        copied[name] = value.copy()
    return copied


def build_qwen35_mlp_graph(
    weights: Mapping[str, np.ndarray],
    contract: GalaxeaQwen35MLPContract | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Graph:
    """Build ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    SiLU is deliberately decomposed into ``x * sigmoid(x)`` so this first
    slice uses only the existing canonical MatMul, Sigmoid and Mul operators.
    """

    selected = _selected_contract(contract)
    constants = _copy_canonical_weights(weights, selected)
    prefix = SOURCE_PREFIX
    nodes = [
        Node(
            "gate_proj",
            "MatMul",
            ("hidden_states", "gate_proj_weight"),
            ("gate",),
            source_nodes=(f"{prefix}.gate_proj",),
        ),
        Node(
            "gate_sigmoid",
            "Sigmoid",
            ("gate",),
            ("gate_sigmoid",),
            source_nodes=(f"{prefix}.activation",),
        ),
        Node(
            "gate_silu",
            "Mul",
            ("gate", "gate_sigmoid"),
            ("gate_activated",),
            source_nodes=(f"{prefix}.activation",),
        ),
        Node(
            "up_proj",
            "MatMul",
            ("hidden_states", "up_proj_weight"),
            ("up",),
            source_nodes=(f"{prefix}.up_proj",),
        ),
        Node(
            "swiglu_mul",
            "Mul",
            ("gate_activated", "up"),
            ("gated_up",),
            source_nodes=(f"{prefix}.swiglu",),
        ),
        Node(
            "down_proj",
            "MatMul",
            ("gated_up", "down_proj_weight"),
            ("output",),
            source_nodes=(f"{prefix}.down_proj",),
        ),
    ]
    default_metadata: dict[str, Any] = {
        "kind": selected.kind,
        "model_adapter": selected.model_adapter,
        "model_family": "galaxeavla",
        "model_scope": selected.model_scope,
        "upstream_repository": selected.upstream_repository,
        "upstream_source_revision": selected.upstream_source_revision,
        "upstream_license_url": selected.upstream_license_url,
        "upstream_usage_notice": selected.upstream_usage_notice,
        "component": selected.component,
        "layer": selected.layer,
        "full_model": False,
        "activation": selected.activation,
        "official_g05_dimensions": selected.official_g05_dimensions,
        "source_weight_layout": selected.source_weight_layout,
        "weight_layout": selected.canonical_weight_layout,
        "contract": selected.to_dict(),
    }
    if metadata is not None:
        default_metadata.update(dict(metadata))
    graph = Graph(
        name="galaxeavla_qwen35_action_expert_layer0_mlp",
        inputs={"hidden_states": TensorSpec(selected.input_shape, selected.dtype)},
        outputs=("output",),
        nodes=nodes,
        constants=constants,
        metadata=default_metadata,
    )
    ShapeInferencePass().run(graph)
    graph.validate()
    return graph


build_galaxeavla_qwen35_mlp_graph = build_qwen35_mlp_graph


__all__ = [
    "build_galaxeavla_qwen35_mlp_graph",
    "build_qwen35_mlp_graph",
    "canonicalize_qwen35_mlp_weights",
]
