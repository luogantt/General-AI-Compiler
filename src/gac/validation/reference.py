"""Explicit NumPy validation; independent of device compilers and runtimes."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph
from gac.core.ops import evaluate_node


def bind_inputs(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate the graph ABI without copying or selecting an execution device."""
    if set(inputs) != set(graph.inputs):
        raise ValidationError(f"input names must be {sorted(graph.inputs)}, got {sorted(inputs)}")
    bound = {}
    for name, spec in graph.inputs.items():
        value = np.asarray(inputs[name])
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(f"input {name!r} requires {spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}")
        if not value.flags.c_contiguous:
            raise ValidationError(f"input {name!r} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValidationError(f"input {name!r} must contain finite values")
        bound[name] = value
    return bound


def reference_run(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Produce offline golden outputs; never used as a device fallback."""
    values = dict(graph.constants)
    values.update(bind_inputs(graph, inputs))
    for node in graph.nodes:
        values.update(evaluate_node(node, values))
    return {name: values[name].copy() for name in graph.outputs}
