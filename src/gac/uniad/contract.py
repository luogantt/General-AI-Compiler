"""Dependency-light data contract for one UniAD stage-2 encoder FFN.

The production profile follows ``projects/configs/stage2_e2e/base_e2e.py``:
one batch, a 200 by 200 BEV grid, 256 embedding channels and 512 hidden
channels.  Smaller positive dimensions are accepted for protocol tests, but
the hidden width is always exactly twice the embedding width.

This module intentionally uses syntax supported by Python 3.9 and depends only
on the standard library.  It can therefore be imported in the official UniAD
environment without importing PyTorch, MMCV, CuPy or GAC.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Tuple


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "uniad-stage2-bev-encoder-ffn"

CANONICAL_WEIGHT_NAMES = (
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
    "norm_weight",
    "norm_bias",
)

SAMPLE_NAMES = ("query", "output")

_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "stage",
        "component",
        "layer",
        "batch",
        "bev_h",
        "bev_w",
        "embed_dim",
        "hidden_dim",
        "dtype",
        "layer_norm_epsilon",
        "operation_order",
        "dropout",
        "weight_layout",
    }
)


class ContractError(ValueError):
    """Raised when a serialized or in-memory FFN contract is invalid."""


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("{} must be a positive integer".format(name))
    return value


@dataclass(frozen=True)
class UniADFFNContract:
    """Static ABI for one post-normalized UniAD BEV encoder FFN.

    ``hidden_dim`` is explicit in the serialized representation so accidental
    producer/consumer disagreements are detected.  Validation requires it to
    equal ``2 * embed_dim``.
    """

    layer: int = 0
    batch: int = 1
    bev_h: int = 200
    bev_w: int = 200
    embed_dim: int = 256
    hidden_dim: int = 512
    layer_norm_epsilon: float = 1e-5

    schema_version: ClassVar[int] = CONTRACT_SCHEMA_VERSION
    kind: ClassVar[str] = CONTRACT_KIND
    stage: ClassVar[str] = "stage2_e2e"
    component: ClassVar[str] = "bev_encoder_ffn"
    dtype: ClassVar[str] = "float32"
    operation_order: ClassVar[Tuple[str, ...]] = (
        "MatMul",
        "BiasAdd",
        "ReLU",
        "MatMul",
        "BiasAdd",
        "ResidualAdd",
        "LayerNorm",
    )
    dropout: ClassVar[str] = "inference-identity"
    weight_layout: ClassVar[str] = "input_channel_output_channel"

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or not 0 <= self.layer < 6
        ):
            raise ContractError("layer must be an integer in [0, 5]")
        _positive_integer("batch", self.batch)
        _positive_integer("bev_h", self.bev_h)
        _positive_integer("bev_w", self.bev_w)
        _positive_integer("embed_dim", self.embed_dim)
        _positive_integer("hidden_dim", self.hidden_dim)
        if self.hidden_dim != self.embed_dim * 2:
            raise ContractError("hidden_dim must equal 2 * embed_dim")
        epsilon = self.layer_norm_epsilon
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or float(epsilon) <= 0.0
        ):
            raise ContractError("layer_norm_epsilon must be finite and positive")
        object.__setattr__(self, "layer_norm_epsilon", float(epsilon))

    @property
    def tokens(self) -> int:
        return self.bev_h * self.bev_w

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.batch, self.tokens, self.embed_dim)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return self.input_shape

    @property
    def weight_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return MappingProxyType(
            {
                "fc1_weight": (self.embed_dim, self.hidden_dim),
                "fc1_bias": (self.hidden_dim,),
                "fc2_weight": (self.hidden_dim, self.embed_dim),
                "fc2_bias": (self.embed_dim,),
                "norm_weight": (self.embed_dim,),
                "norm_bias": (self.embed_dim,),
            }
        )

    @property
    def sample_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return MappingProxyType(
            {"query": self.input_shape, "output": self.output_shape}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete JSON representation of the contract."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "stage": self.stage,
            "component": self.component,
            "layer": self.layer,
            "batch": self.batch,
            "bev_h": self.bev_h,
            "bev_w": self.bev_w,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "dtype": self.dtype,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "operation_order": list(self.operation_order),
            "dropout": self.dropout,
            "weight_layout": self.weight_layout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UniADFFNContract":
        """Parse a strict contract mapping and reject unknown fields."""

        if not isinstance(value, Mapping):
            raise ContractError("contract must be a mapping")
        actual_fields = set(value)
        if actual_fields != _CONTRACT_FIELDS:
            missing = sorted(_CONTRACT_FIELDS - actual_fields)
            extra = sorted(actual_fields - _CONTRACT_FIELDS)
            raise ContractError(
                "contract fields differ from schema; missing={}, extra={}".format(
                    missing, extra
                )
            )
        fixed = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": CONTRACT_KIND,
            "stage": cls.stage,
            "component": cls.component,
            "dtype": cls.dtype,
            "operation_order": list(cls.operation_order),
            "dropout": cls.dropout,
            "weight_layout": cls.weight_layout,
        }
        for name, expected in fixed.items():
            if type(value[name]) is not type(expected) or value[name] != expected:
                raise ContractError(
                    "contract field {!r} must equal {!r}".format(name, expected)
                )
        return cls(
            layer=value["layer"],
            batch=value["batch"],
            bev_h=value["bev_h"],
            bev_w=value["bev_w"],
            embed_dim=value["embed_dim"],
            hidden_dim=value["hidden_dim"],
            layer_norm_epsilon=value["layer_norm_epsilon"],
        )


__all__ = [
    "CANONICAL_WEIGHT_NAMES",
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "ContractError",
    "SAMPLE_NAMES",
    "UniADFFNContract",
]
