"""Pure NumPy checkpoint contract for a UniAD BEV encoder FFN.

UniAD 2.0 builds each BEV encoder feed-forward block with MMCV 1.6.1's
``FFN``.  PyTorch stores the two Linear weights as ``[out, in]`` while the
General AI Compiler consumes matrices as ``[in, out]``.  This module resolves
one of the six encoder layers, validates its exact stage-2 shape contract, and
returns detached, C-contiguous arrays in the compiler's canonical layout.

The input is an already unwrapped checkpoint ``state_dict`` represented as a
``Mapping[str, numpy.ndarray]``.  Loading ``.pth`` files remains outside this
module so importing it does not require PyTorch or MMCV.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


ENCODER_LAYER_COUNT = 6
EMBED_DIM = 256
FEEDFORWARD_CHANNELS = 512

_ENCODER_PREFIX = "pts_bbox_head.transformer.encoder.layers"

# Suffixes are relative to ``{_ENCODER_PREFIX}.{layer}.``.  The third norm in
# the post-norm operation order is the LayerNorm immediately following FFN.
_CHECKPOINT_SUFFIXES = MappingProxyType(
    {
        "fc1_weight": "ffns.0.layers.0.0.weight",
        "fc1_bias": "ffns.0.layers.0.0.bias",
        "fc2_weight": "ffns.0.layers.1.weight",
        "fc2_bias": "ffns.0.layers.1.bias",
        "norm_weight": "norms.2.weight",
        "norm_bias": "norms.2.bias",
    }
)

_CHECKPOINT_SHAPES = MappingProxyType(
    {
        "fc1_weight": (FEEDFORWARD_CHANNELS, EMBED_DIM),
        "fc1_bias": (FEEDFORWARD_CHANNELS,),
        "fc2_weight": (EMBED_DIM, FEEDFORWARD_CHANNELS),
        "fc2_bias": (EMBED_DIM,),
        "norm_weight": (EMBED_DIM,),
        "norm_bias": (EMBED_DIM,),
    }
)

CANONICAL_WEIGHT_SHAPES = MappingProxyType(
    {
        "fc1_weight": (EMBED_DIM, FEEDFORWARD_CHANNELS),
        "fc1_bias": (FEEDFORWARD_CHANNELS,),
        "fc2_weight": (FEEDFORWARD_CHANNELS, EMBED_DIM),
        "fc2_bias": (EMBED_DIM,),
        "norm_weight": (EMBED_DIM,),
        "norm_bias": (EMBED_DIM,),
    }
)


class CheckpointContractError(ValueError):
    """Raised when a UniAD encoder FFN checkpoint violates the contract."""


def _validate_layer(layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise CheckpointContractError("encoder layer must be an integer in [0, 5]")
    result = int(layer)
    if not 0 <= result < ENCODER_LAYER_COUNT:
        raise CheckpointContractError("encoder layer must be in [0, 5]")
    return result


def encoder_ffn_checkpoint_keys(layer: int) -> dict[str, str]:
    """Return canonical-name to UniAD checkpoint-key mapping for ``layer``."""

    layer = _validate_layer(layer)
    base = f"{_ENCODER_PREFIX}.{layer}"
    return {
        canonical_name: f"{base}.{suffix}"
        for canonical_name, suffix in _CHECKPOINT_SUFFIXES.items()
    }


def extract_encoder_ffn_weights(
    state_dict: Mapping[str, np.ndarray],
    layer: int,
) -> dict[str, np.ndarray]:
    """Extract one UniAD stage-2 encoder ``FFN -> LayerNorm`` parameter set.

    The returned mapping is accepted directly by GAC's
    ``build_bevformer_ffn_graph``.  Both Linear weights are transposed from
    PyTorch's ``[out, in]`` layout.  Every returned array owns a finite,
    C-contiguous ``float32`` copy, so later mutation of the checkpoint cannot
    alter a compiled graph.
    """

    if not isinstance(state_dict, Mapping):
        raise CheckpointContractError(
            "state_dict must be a mapping from checkpoint keys to numpy arrays"
        )

    keys = encoder_ffn_checkpoint_keys(layer)
    validated: dict[str, np.ndarray] = {}
    for canonical_name, checkpoint_key in keys.items():
        if checkpoint_key not in state_dict:
            raise CheckpointContractError(
                f"missing UniAD checkpoint parameter {checkpoint_key!r}"
            )
        value = state_dict[checkpoint_key]
        if not isinstance(value, np.ndarray):
            raise CheckpointContractError(
                f"checkpoint parameter {checkpoint_key!r} must be a numpy array"
            )
        expected_shape = _CHECKPOINT_SHAPES[canonical_name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise CheckpointContractError(
                f"checkpoint parameter {checkpoint_key!r} requires "
                f"{expected_shape}/float32, got {value.shape}/{value.dtype}"
            )
        if not np.isfinite(value).all():
            raise CheckpointContractError(
                f"checkpoint parameter {checkpoint_key!r} must be finite"
            )
        validated[canonical_name] = value

    result: dict[str, np.ndarray] = {}
    for canonical_name, value in validated.items():
        if canonical_name in ("fc1_weight", "fc2_weight"):
            value = value.T
        canonical = np.array(value, dtype=np.float32, order="C", copy=True)
        expected_shape = CANONICAL_WEIGHT_SHAPES[canonical_name]
        if canonical.shape != expected_shape:  # Defensive invariant.
            raise CheckpointContractError(
                f"internal canonical shape mismatch for {canonical_name!r}: "
                f"expected {expected_shape}, got {canonical.shape}"
            )
        result[canonical_name] = canonical

    return result


__all__ = [
    "CANONICAL_WEIGHT_SHAPES",
    "CheckpointContractError",
    "EMBED_DIM",
    "ENCODER_LAYER_COUNT",
    "FEEDFORWARD_CHANNELS",
    "encoder_ffn_checkpoint_keys",
    "extract_encoder_ffn_weights",
]
