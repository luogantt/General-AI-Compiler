"""Lazy model adapters: model data enters the compiler as a common Graph.

Adapters own model contracts and graph assembly. They do not choose targets,
compile machine code or execute inference. New model families register a
factory here and reuse the existing compiler/target/runtime layers.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gac.core.ir import Graph


@runtime_checkable
class ModelAdapter(Protocol):
    name: str
    scope: str

    def load(self, source: str | Path, **options: object) -> Graph:
        """Verify model data from a file and construct the common Graph."""
        ...

    def build(self, data: object, **options: object) -> Graph:
        """Construct the common Graph from an adapter's in-memory data type."""
        ...


# Strings keep registry inspection free of model, NumPy and target imports.
_FACTORIES: dict[str, Callable[[], ModelAdapter] | tuple[str, str]] = {
    "bevformer-step": ("gac.models.bevformer.adapter", "BEVFormerStepAdapter"),
    "uniad-ffn": ("gac.models.uniad.adapter", "UniADFFNAdapter"),
}


def _name(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_-]*", value) is None:
        raise ValueError("model adapter name must use lowercase letters, digits, '_' or '-'")
    return value


def list_model_adapters() -> tuple[str, ...]:
    """Return registered names without importing any model implementation."""
    return tuple(sorted(_FACTORIES))


def register_model_adapter(name: str, factory: Callable[[], ModelAdapter], *, replace: bool = False) -> None:
    """Register a lazy factory; replacing an existing name must be explicit."""
    name = _name(name)
    if not callable(factory):
        raise TypeError("model adapter factory must be callable")
    if name in _FACTORIES and not replace:
        raise ValueError(f"model adapter {name!r} is already registered")
    _FACTORIES[name] = factory


def get_model_adapter(name: str) -> ModelAdapter:
    """Load only the selected model adapter and return a fresh instance."""
    name = _name(name)
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown model adapter {name!r}; available: {', '.join(list_model_adapters())}") from exc
    if isinstance(factory, tuple):
        module, attribute = factory
        factory = getattr(import_module(module), attribute)
    adapter = factory()
    if (not isinstance(adapter, ModelAdapter) or adapter.name != name or not isinstance(adapter.scope, str)
            or not adapter.scope or not callable(adapter.load) or not callable(adapter.build)):
        raise TypeError(f"factory for {name!r} did not return a matching ModelAdapter")
    return adapter


def load_model_graph(name: str, source: str | Path, **options: object) -> Graph:
    """Construct a selected model's graph; compilation is a separate step."""
    from gac.core.ir import Graph

    graph = get_model_adapter(name).load(source, **options)
    if not isinstance(graph, Graph):
        raise TypeError(f"model adapter {name!r} did not return a Graph")
    graph.validate()
    return graph


__all__ = ["ModelAdapter", "get_model_adapter", "list_model_adapters", "load_model_graph", "register_model_adapter"]
