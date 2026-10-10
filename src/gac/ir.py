"""Small, serializable model IR used by the GAC prototype.

The IR intentionally models graph semantics, not a particular device.  It is
strict enough to catch malformed or non-topological graphs while remaining
easy to inspect in an ``.hxp`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import numpy as np

from .errors import ValidationError


Dim: TypeAlias = int | str | None
_SCHEMA_VERSION = 1


def _normalize_dim(dim: object) -> Dim:
    if dim is None:
        return None
    if isinstance(dim, bool):
        raise ValidationError("tensor dimensions cannot be booleans")
    if isinstance(dim, (int, np.integer)):
        value = int(dim)
        if value < 0:
            raise ValidationError(f"tensor dimensions must be non-negative, got {value}")
        return value
    if isinstance(dim, str) and dim:
        return dim
    raise ValidationError(
        f"tensor dimension must be a non-negative integer, symbol, or null; got {dim!r}"
    )


def _json_compatible(value: Any) -> Any:
    """Convert common NumPy/Python values to data accepted by ``json``."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValidationError(f"value of type {type(value).__name__} is not JSON-compatible")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Logical tensor type.

    A dimension is either a static non-negative integer, a named symbolic
    dimension, or ``None`` for an anonymous dynamic dimension.  Consequently
    ``nbytes`` is known only when every dimension is static.
    """

    shape: tuple[Dim, ...]
    dtype: str | np.dtype[Any]
    layout: str = "contiguous"

    def __post_init__(self) -> None:
        try:
            normalized_dtype = np.dtype(self.dtype).name
        except TypeError as exc:
            raise ValidationError(f"unsupported dtype {self.dtype!r}") from exc
        normalized_shape = tuple(_normalize_dim(dim) for dim in self.shape)
        if not isinstance(self.layout, str) or not self.layout:
            raise ValidationError("tensor layout must be a non-empty string")
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "dtype", normalized_dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_static(self) -> bool:
        return all(isinstance(dim, int) for dim in self.shape)

    @property
    def numel(self) -> int | None:
        if not self.is_static:
            return None
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    @property
    def nbytes(self) -> int | None:
        count = self.numel
        return None if count is None else count * np.dtype(self.dtype).itemsize

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSpec":
        try:
            return cls(
                shape=tuple(data["shape"]),
                dtype=data["dtype"],
                layout=data.get("layout", "contiguous"),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"invalid tensor spec: {data!r}") from exc


@dataclass(frozen=True, slots=True)
class Node:
    """One side-effect-free operation in a topologically ordered graph."""

    id: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    source_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValidationError("node id must be a non-empty string")
        if not isinstance(self.op, str) or not self.op:
            raise ValidationError(f"node {self.id!r} has an invalid op name")
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if any(not isinstance(name, str) or not name for name in inputs):
            raise ValidationError(f"node {self.id!r} has an invalid input name")
        if not outputs or any(not isinstance(name, str) or not name for name in outputs):
            raise ValidationError(f"node {self.id!r} must have named outputs")
        if len(set(outputs)) != len(outputs):
            raise ValidationError(f"node {self.id!r} defines an output more than once")
        sources = tuple(self.source_nodes) or (self.id,)
        if any(not isinstance(source, str) or not source for source in sources):
            raise ValidationError(f"node {self.id!r} has invalid provenance")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "attrs", dict(self.attrs))
        object.__setattr__(self, "source_nodes", sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _json_compatible(self.attrs),
            "source_nodes": list(self.source_nodes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        try:
            return cls(
                id=data["id"],
                op=data["op"],
                inputs=tuple(data.get("inputs", ())),
                outputs=tuple(data["outputs"]),
                attrs=dict(data.get("attrs", {})),
                source_nodes=tuple(data.get("source_nodes", ())),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError(f"invalid node: {data!r}") from exc


@dataclass(slots=True)
class Graph:
    """A topologically ordered, device-independent tensor graph."""

    name: str
    inputs: dict[str, TensorSpec]
    outputs: tuple[str, ...]
    nodes: list[Node]
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    value_specs: dict[str, TensorSpec] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.inputs = dict(self.inputs)
        self.outputs = tuple(self.outputs)
        self.nodes = list(self.nodes)
        self.constants = {
            name: np.asarray(value) for name, value in self.constants.items()
        }
        self.value_specs = dict(self.value_specs)
        self.metadata = copy.deepcopy(dict(self.metadata))
        for name, spec in self.inputs.items():
            self.value_specs.setdefault(name, spec)
        for name, value in self.constants.items():
            self.value_specs.setdefault(
                name, TensorSpec(tuple(value.shape), value.dtype)
            )

    def clone(self) -> "Graph":
        return Graph(
            name=self.name,
            inputs=dict(self.inputs),
            outputs=self.outputs,
            nodes=list(self.nodes),
            constants={name: value.copy() for name, value in self.constants.items()},
            value_specs=dict(self.value_specs),
            metadata=copy.deepcopy(self.metadata),
        )

    def validate(self) -> None:
        """Validate names, SSA-like definitions, specs, and topological order."""

        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("graph name must be a non-empty string")
        if not self.outputs:
            raise ValidationError(f"graph {self.name!r} must expose at least one output")

        known: set[str] = set()
        for name, spec in self.inputs.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("graph input names must be non-empty strings")
            if not isinstance(spec, TensorSpec):
                raise ValidationError(f"input {name!r} does not have a TensorSpec")
            if name in known:
                raise ValidationError(f"value {name!r} is defined more than once")
            known.add(name)

        for name, value in self.constants.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("constant names must be non-empty strings")
            if name in known:
                raise ValidationError(f"value {name!r} is both an input and a constant")
            if not isinstance(value, np.ndarray):
                raise ValidationError(f"constant {name!r} is not a NumPy array")
            known.add(name)
            spec = self.value_specs.get(name)
            actual = TensorSpec(tuple(value.shape), value.dtype)
            if spec is not None and (spec.shape != actual.shape or spec.dtype != actual.dtype):
                raise ValidationError(
                    f"constant {name!r} has spec {spec.shape}/{spec.dtype}, "
                    f"but data is {actual.shape}/{actual.dtype}"
                )

        node_ids: set[str] = set()
        for node in self.nodes:
            if not isinstance(node, Node):
                raise ValidationError("graph nodes must be Node instances")
            if node.id in node_ids:
                raise ValidationError(f"node id {node.id!r} is defined more than once")
            node_ids.add(node.id)
            missing = [value for value in node.inputs if value not in known]
            if missing:
                raise ValidationError(
                    f"node {node.id!r} uses values before definition: {missing}"
                )
            for output in node.outputs:
                if output in known:
                    raise ValidationError(f"value {output!r} is defined more than once")
                known.add(output)

        missing_outputs = [name for name in self.outputs if name not in known]
        if missing_outputs:
            raise ValidationError(f"graph outputs are undefined: {missing_outputs}")
        unknown_specs = sorted(set(self.value_specs) - known)
        if unknown_specs:
            raise ValidationError(f"value specs refer to undefined values: {unknown_specs}")
        for name, spec in self.value_specs.items():
            if not isinstance(spec, TensorSpec):
                raise ValidationError(f"value {name!r} does not have a TensorSpec")

    def producer_map(self) -> dict[str, Node]:
        return {output: node for node in self.nodes for output in node.outputs}

    def consumers_map(self) -> dict[str, list[Node]]:
        consumers: dict[str, list[Node]] = {}
        for node in self.nodes:
            for input_name in node.inputs:
                consumers.setdefault(input_name, []).append(node)
        return consumers

    def to_dict(self, *, include_constant_data: bool = True) -> dict[str, Any]:
        constants: dict[str, dict[str, Any]] = {}
        for name, value in self.constants.items():
            record: dict[str, Any] = {
                "dtype": value.dtype.name,
                "shape": list(value.shape),
            }
            if include_constant_data:
                record["data"] = value.tolist()
            constants[name] = record
        return {
            "schema_version": _SCHEMA_VERSION,
            "name": self.name,
            "inputs": {name: spec.to_dict() for name, spec in self.inputs.items()},
            "outputs": list(self.outputs),
            "nodes": [node.to_dict() for node in self.nodes],
            "constants": constants,
            "value_specs": {
                name: spec.to_dict() for name, spec in self.value_specs.items()
            },
            "metadata": _json_compatible(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        external_constants: Mapping[str, np.ndarray] | None = None,
    ) -> "Graph":
        version = data.get("schema_version", _SCHEMA_VERSION)
        if version != _SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported graph schema version {version!r}; expected {_SCHEMA_VERSION}"
            )
        supplied = dict(external_constants or {})
        constants: dict[str, np.ndarray] = {}
        try:
            for name, record in data.get("constants", {}).items():
                if "data" in record:
                    value = np.asarray(record["data"], dtype=record["dtype"])
                    value = value.reshape(tuple(record["shape"]))
                elif name in supplied:
                    value = np.asarray(supplied[name])
                    expected_shape = tuple(record["shape"])
                    expected_dtype = np.dtype(record["dtype"]).name
                    if value.shape != expected_shape or value.dtype.name != expected_dtype:
                        raise ValidationError(
                            f"external constant {name!r} does not match serialized metadata"
                        )
                else:
                    raise ValidationError(
                        f"constant {name!r} contains metadata only; provide external_constants"
                    )
                constants[name] = value

            graph = cls(
                name=data["name"],
                inputs={
                    name: TensorSpec.from_dict(spec)
                    for name, spec in data.get("inputs", {}).items()
                },
                outputs=tuple(data["outputs"]),
                nodes=[Node.from_dict(node) for node in data.get("nodes", ())],
                constants=constants,
                value_specs={
                    name: TensorSpec.from_dict(spec)
                    for name, spec in data.get("value_specs", {}).items()
                },
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("invalid serialized graph") from exc
        graph.validate()
        return graph

    def to_json(self, *, include_constant_data: bool = True, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_constant_data=include_constant_data),
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(
        cls,
        source: str | bytes | Path,
        *,
        external_constants: Mapping[str, np.ndarray] | None = None,
    ) -> "Graph":
        if isinstance(source, Path):
            payload = json.loads(source.read_text(encoding="utf-8"))
        else:
            payload = json.loads(source)
        if not isinstance(payload, Mapping):
            raise ValidationError("serialized graph root must be an object")
        return cls.from_dict(payload, external_constants=external_constants)
