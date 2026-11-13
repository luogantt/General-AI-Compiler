"""Static contract for the first Galaxea G0.5 compiler slice.

The slice is exactly the bias-free SwiGLU MLP in Qwen3.5 Action Expert layer
zero.  Checkpoint tensors retain PyTorch's ``[out_features, in_features]``
layout at this boundary; the adapter transposes them before graph assembly.
This module is intentionally dependency-light and imports only the standard
library.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


CONTRACT_SCHEMA_VERSION = 1
CONTRACT_KIND = "galaxeavla-qwen35-action-expert-layer0-swiglu-mlp"
MODEL_ADAPTER_NAME = "galaxeavla-qwen35-mlp"
MODEL_SCOPE = "action-expert-layer0-qwen35-swiglu-mlp"
UPSTREAM_REPOSITORY = "https://github.com/OpenGalaxea/GalaxeaVLA"
UPSTREAM_SOURCE_REVISION = "89f2322b4ad016e192437adc1a2c253b05bab246"
UPSTREAM_LICENSE_URL = (
    "https://github.com/OpenGalaxea/GalaxeaVLA/blob/"
    f"{UPSTREAM_SOURCE_REVISION}/LICENSE-G0.5"
)
UPSTREAM_USAGE_NOTICE = (
    "G0.5 Community License applies; non-commercial and internal-evaluation "
    "restrictions must be reviewed before use"
)

SOURCE_PREFIX = "model.action_expert.layers.0.mlp"
GATE_PROJ_KEY = f"{SOURCE_PREFIX}.gate_proj.weight"
UP_PROJ_KEY = f"{SOURCE_PREFIX}.up_proj.weight"
DOWN_PROJ_KEY = f"{SOURCE_PREFIX}.down_proj.weight"
SOURCE_WEIGHT_KEYS = (GATE_PROJ_KEY, UP_PROJ_KEY, DOWN_PROJ_KEY)

CANONICAL_WEIGHT_NAMES = (
    "gate_proj_weight",
    "up_proj_weight",
    "down_proj_weight",
)

_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "model_adapter",
        "model_scope",
        "upstream_repository",
        "upstream_source_revision",
        "upstream_license_url",
        "upstream_usage_notice",
        "component",
        "layer",
        "batch",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "dtype",
        "activation",
        "operation_order",
        "source_weight_layout",
        "canonical_weight_layout",
        "official_g05_dimensions",
        "full_model",
    }
)


class ContractError(ValueError):
    """Raised when a Qwen3.5 MLP contract is malformed."""


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class GalaxeaQwen35MLPContract:
    """Tensor ABI for Action Expert layer zero's bias-free SwiGLU MLP."""

    batch: int = 1
    tokens: int = 32
    hidden_size: int = 1024
    intermediate_size: int = 4096

    schema_version: ClassVar[int] = CONTRACT_SCHEMA_VERSION
    kind: ClassVar[str] = CONTRACT_KIND
    model_adapter: ClassVar[str] = MODEL_ADAPTER_NAME
    model_scope: ClassVar[str] = MODEL_SCOPE
    upstream_repository: ClassVar[str] = UPSTREAM_REPOSITORY
    upstream_source_revision: ClassVar[str] = UPSTREAM_SOURCE_REVISION
    upstream_license_url: ClassVar[str] = UPSTREAM_LICENSE_URL
    upstream_usage_notice: ClassVar[str] = UPSTREAM_USAGE_NOTICE
    component: ClassVar[str] = SOURCE_PREFIX
    layer: ClassVar[int] = 0
    dtype: ClassVar[str] = "float32"
    activation: ClassVar[str] = "silu"
    operation_order: ClassVar[tuple[str, ...]] = (
        "MatMul",
        "Sigmoid",
        "Mul",
        "MatMul",
        "Mul",
        "MatMul",
    )
    source_weight_layout: ClassVar[str] = "[out_features, in_features]"
    canonical_weight_layout: ClassVar[str] = "[input_channel, output_channel]"
    full_model: ClassVar[bool] = False

    def __post_init__(self) -> None:
        _positive_integer("batch", self.batch)
        _positive_integer("tokens", self.tokens)
        _positive_integer("hidden_size", self.hidden_size)
        _positive_integer("intermediate_size", self.intermediate_size)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.batch, self.tokens, self.hidden_size)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.input_shape

    @property
    def official_g05_dimensions(self) -> bool:
        """Whether widths match the published G0.5 Action Expert profile."""

        return self.hidden_size == 1024 and self.intermediate_size == 4096

    @property
    def source_weight_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Expected PyTorch checkpoint shapes before transposition."""

        return MappingProxyType(
            {
                GATE_PROJ_KEY: (self.intermediate_size, self.hidden_size),
                UP_PROJ_KEY: (self.intermediate_size, self.hidden_size),
                DOWN_PROJ_KEY: (self.hidden_size, self.intermediate_size),
            }
        )

    @property
    def canonical_weight_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Graph constant shapes after converting to ``[input, output]``."""

        return MappingProxyType(
            {
                "gate_proj_weight": (self.hidden_size, self.intermediate_size),
                "up_proj_weight": (self.hidden_size, self.intermediate_size),
                "down_proj_weight": (self.intermediate_size, self.hidden_size),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "model_adapter": self.model_adapter,
            "model_scope": self.model_scope,
            "upstream_repository": self.upstream_repository,
            "upstream_source_revision": self.upstream_source_revision,
            "upstream_license_url": self.upstream_license_url,
            "upstream_usage_notice": self.upstream_usage_notice,
            "component": self.component,
            "layer": self.layer,
            "batch": self.batch,
            "tokens": self.tokens,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dtype": self.dtype,
            "activation": self.activation,
            "operation_order": list(self.operation_order),
            "source_weight_layout": self.source_weight_layout,
            "canonical_weight_layout": self.canonical_weight_layout,
            "official_g05_dimensions": self.official_g05_dimensions,
            "full_model": self.full_model,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GalaxeaQwen35MLPContract":
        if not isinstance(value, Mapping):
            raise ContractError("contract must be a mapping")
        actual_fields = set(value)
        if actual_fields != _CONTRACT_FIELDS:
            missing = sorted(_CONTRACT_FIELDS - actual_fields)
            extra = sorted(actual_fields - _CONTRACT_FIELDS)
            raise ContractError(
                f"contract fields differ from schema; missing={missing}, extra={extra}"
            )
        fixed = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "kind": CONTRACT_KIND,
            "model_adapter": MODEL_ADAPTER_NAME,
            "model_scope": MODEL_SCOPE,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_source_revision": UPSTREAM_SOURCE_REVISION,
            "upstream_license_url": UPSTREAM_LICENSE_URL,
            "upstream_usage_notice": UPSTREAM_USAGE_NOTICE,
            "component": SOURCE_PREFIX,
            "layer": 0,
            "dtype": "float32",
            "activation": "silu",
            "operation_order": list(cls.operation_order),
            "source_weight_layout": cls.source_weight_layout,
            "canonical_weight_layout": cls.canonical_weight_layout,
            "full_model": False,
        }
        for name, expected in fixed.items():
            if type(value[name]) is not type(expected) or value[name] != expected:
                raise ContractError(f"contract field {name!r} must equal {expected!r}")
        result = cls(
            batch=value["batch"],
            tokens=value["tokens"],
            hidden_size=value["hidden_size"],
            intermediate_size=value["intermediate_size"],
        )
        if (
            type(value["official_g05_dimensions"]) is not bool
            or value["official_g05_dimensions"] != result.official_g05_dimensions
        ):
            raise ContractError(
                "contract field 'official_g05_dimensions' does not match its widths"
            )
        return result


# Short and fully-qualified aliases keep the public contract discoverable.
Qwen35MLPContract = GalaxeaQwen35MLPContract
GalaxeaVLAQwen35MLPContract = GalaxeaQwen35MLPContract


__all__ = [
    "CANONICAL_WEIGHT_NAMES",
    "CONTRACT_KIND",
    "CONTRACT_SCHEMA_VERSION",
    "ContractError",
    "DOWN_PROJ_KEY",
    "GATE_PROJ_KEY",
    "GalaxeaQwen35MLPContract",
    "GalaxeaVLAQwen35MLPContract",
    "MODEL_ADAPTER_NAME",
    "MODEL_SCOPE",
    "Qwen35MLPContract",
    "SOURCE_PREFIX",
    "SOURCE_WEIGHT_KEYS",
    "UP_PROJ_KEY",
    "UPSTREAM_REPOSITORY",
    "UPSTREAM_LICENSE_URL",
    "UPSTREAM_SOURCE_REVISION",
    "UPSTREAM_USAGE_NOTICE",
]
