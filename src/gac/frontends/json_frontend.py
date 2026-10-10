"""A small, deterministic JSON frontend used by the prototype and tests.

The JSON format is deliberately an interchange/debug format, not a proposed
replacement for StableHLO or ONNX.  It keeps the end-to-end compiler usable
without heavyweight framework dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gac.ir import Graph, Node, TensorSpec


class JsonFrontendError(ValueError):
    """Raised when a JSON model is malformed."""


_MAX_INITIALIZER_ELEMENTS = 100_000_000
_SUPPORTED_DTYPES = {
    "float16": np.dtype("float16"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "int8": np.dtype("int8"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "bool": np.dtype("bool"),
}


def _dtype(name: str) -> np.dtype[Any]:
    try:
        return _SUPPORTED_DTYPES[name]
    except KeyError as exc:
        raise JsonFrontendError(f"unsupported dtype {name!r}") from exc


def _shape(value: Any, *, allow_dynamic: bool) -> tuple[int | None, ...]:
    if not isinstance(value, list):
        raise JsonFrontendError("shape must be a JSON array")
    result: list[int | None] = []
    for dim in value:
        if dim is None and allow_dynamic:
            result.append(None)
        elif isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0:
            result.append(dim)
        else:
            raise JsonFrontendError(f"invalid shape dimension {dim!r}")
    return tuple(result)


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
        if count > _MAX_INITIALIZER_ELEMENTS:
            raise JsonFrontendError(
                f"initializer exceeds {_MAX_INITIALIZER_ELEMENTS:,} elements"
            )
    return count


def _constant(name: str, raw: Any) -> np.ndarray[Any, Any]:
    if not isinstance(raw, Mapping):
        # Concise form: nested JSON values, defaulting to float32.
        return np.asarray(raw, dtype=np.float32)

    dtype_name = str(raw.get("dtype", "float32"))
    dtype = _dtype(dtype_name)
    if "data" in raw:
        result = np.asarray(raw["data"], dtype=dtype)
        if "shape" in raw:
            shape = _shape(raw["shape"], allow_dynamic=False)
            _element_count(shape)  # type: ignore[arg-type]
            try:
                result = result.reshape(shape)
            except ValueError as exc:
                raise JsonFrontendError(
                    f"constant {name!r} data does not match shape {shape}"
                ) from exc
        return np.ascontiguousarray(result)

    kind = raw.get("initializer")
    if not isinstance(kind, str):
        raise JsonFrontendError(
            f"constant {name!r} needs either 'data' or an 'initializer'"
        )
    shape_any = _shape(raw.get("shape"), allow_dynamic=False)
    shape = tuple(int(dim) for dim in shape_any)
    _element_count(shape)

    if kind == "zeros":
        return np.zeros(shape, dtype=dtype)
    if kind == "ones":
        return np.ones(shape, dtype=dtype)
    if kind == "full":
        return np.full(shape, raw.get("value", 0), dtype=dtype)
    if kind == "arange":
        count = _element_count(shape)
        return np.arange(count, dtype=dtype).reshape(shape)
    if kind in {"random_normal", "random_uniform"}:
        seed = raw.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise JsonFrontendError(f"constant {name!r} seed must be an integer")
        rng = np.random.default_rng(seed)
        if kind == "random_normal":
            values = rng.normal(
                float(raw.get("mean", 0.0)),
                float(raw.get("std", 1.0)),
                size=shape,
            )
        else:
            values = rng.uniform(
                float(raw.get("low", -1.0)),
                float(raw.get("high", 1.0)),
                size=shape,
            )
        return np.ascontiguousarray(values.astype(dtype))
    raise JsonFrontendError(f"constant {name!r} has unknown initializer {kind!r}")


def graph_from_dict(document: Mapping[str, Any]) -> Graph:
    """Parse and validate a graph from a decoded JSON mapping."""

    name = document.get("name", "model")
    if not isinstance(name, str) or not name:
        raise JsonFrontendError("model name must be a non-empty string")

    raw_inputs = document.get("inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise JsonFrontendError("inputs must be a non-empty object")
    inputs: dict[str, TensorSpec] = {}
    for input_name, raw_spec in raw_inputs.items():
        if not isinstance(input_name, str) or not isinstance(raw_spec, Mapping):
            raise JsonFrontendError("each input must map a name to a tensor spec")
        inputs[input_name] = TensorSpec(
            shape=_shape(raw_spec.get("shape"), allow_dynamic=True),
            dtype=str(raw_spec.get("dtype", "float32")),
            layout=str(raw_spec.get("layout", "contiguous")),
        )

    raw_constants = document.get("constants", {})
    if not isinstance(raw_constants, Mapping):
        raise JsonFrontendError("constants must be an object")
    constants = {str(key): _constant(str(key), value) for key, value in raw_constants.items()}

    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise JsonFrontendError("nodes must be an array")
    nodes: list[Node] = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            raise JsonFrontendError(f"node {index} must be an object")
        node_id = raw_node.get("id", f"node_{index}")
        op = raw_node.get("op")
        node_inputs = raw_node.get("inputs", [])
        node_outputs = raw_node.get("outputs", [])
        attrs = raw_node.get("attrs", {})
        if not isinstance(node_id, str) or not isinstance(op, str):
            raise JsonFrontendError(f"node {index} needs string id and op")
        if not isinstance(node_inputs, list) or not all(isinstance(v, str) for v in node_inputs):
            raise JsonFrontendError(f"node {node_id!r} inputs must be strings")
        if not isinstance(node_outputs, list) or not all(isinstance(v, str) for v in node_outputs):
            raise JsonFrontendError(f"node {node_id!r} outputs must be strings")
        if not isinstance(attrs, Mapping):
            raise JsonFrontendError(f"node {node_id!r} attrs must be an object")
        raw_sources = raw_node.get("source_nodes", [node_id])
        if not isinstance(raw_sources, list) or not all(isinstance(v, str) for v in raw_sources):
            raise JsonFrontendError(f"node {node_id!r} source_nodes must be strings")
        nodes.append(
            Node(
                id=node_id,
                op=op,
                inputs=list(node_inputs),
                outputs=list(node_outputs),
                attrs=dict(attrs),
                source_nodes=tuple(raw_sources),
            )
        )

    raw_outputs = document.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs or not all(
        isinstance(v, str) for v in raw_outputs
    ):
        raise JsonFrontendError("outputs must be a non-empty array of value names")

    graph = Graph(
        name=name,
        inputs=inputs,
        outputs=list(raw_outputs),
        nodes=nodes,
        constants=constants,
        metadata=dict(document.get("metadata", {})),
    )
    graph.validate()
    return graph


def load_graph(path: str | Path) -> Graph:
    """Load a UTF-8 JSON model from *path*."""

    model_path = Path(path)
    try:
        document = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JsonFrontendError(f"cannot read {model_path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise JsonFrontendError("top-level JSON value must be an object")
    return graph_from_dict(document)

