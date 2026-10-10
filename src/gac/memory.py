"""Deterministic static storage planning for a validated :mod:`gac.ir` graph.

The planner deliberately has no device dependency.  Inputs and produced graph
outputs are caller-owned ABI buffers, constants are immutable resident buffers,
and only internal values are placed in a reusable arena.  Lifetimes use an
inclusive node-step interval: an input consumed by node ``i`` cannot share the
same bytes with that node's output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import numpy as np

from .errors import ValidationError
from .ir import Graph, Node, TensorSpec


MEMORY_PLAN_SCHEMA_VERSION = 1
StorageClass = Literal["external-input", "external-output", "resident-constant", "arena"]


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{field} must be >= {minimum}")
    return value


def _validate_alignment(alignment: int) -> int:
    alignment = _require_int(alignment, "alignment", minimum=1)
    if alignment & (alignment - 1):
        raise ValidationError("alignment must be a positive power of two")
    return alignment


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True, slots=True)
class BufferAllocation:
    """Storage and liveness assigned to one SSA value.

    ``defined_at`` is ``-1`` for graph inputs and constants. ``first_use`` and
    ``last_use`` include the graph-exit step (``len(graph.nodes)``) when the
    value is a graph output.  An unused dead value has ``first_use=None`` and a
    live interval consisting only of its definition step. ``alias_of`` records
    conservative reuse after another value dies; ``view_of`` records a live,
    compiler-proven dense view into an earlier arena allocation.
    """

    name: str
    spec: TensorSpec
    size_bytes: int
    storage: StorageClass
    alignment: int
    offset: int | None
    defined_at: int
    first_use: int | None
    last_use: int
    is_input: bool = False
    is_output: bool = False
    is_constant: bool = False
    alias_of: str | None = None
    view_of: str | None = None

    @property
    def live_start(self) -> int:
        return self.defined_at

    @property
    def live_end(self) -> int:
        return self.last_use

    @property
    def external(self) -> bool:
        return self.storage.startswith("external-")

    @property
    def resident(self) -> bool:
        return self.storage == "resident-constant"

    @property
    def arena_end(self) -> int | None:
        return None if self.offset is None else self.offset + self.size_bytes

    def overlaps_lifetime(self, other: "BufferAllocation") -> bool:
        return not (self.live_end < other.live_start or other.live_end < self.live_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "size_bytes": self.size_bytes,
            "storage": self.storage,
            "alignment": self.alignment,
            "offset": self.offset,
            "defined_at": self.defined_at,
            "first_use": self.first_use,
            "last_use": self.last_use,
            "is_input": self.is_input,
            "is_output": self.is_output,
            "is_constant": self.is_constant,
            "alias_of": self.alias_of,
            "view_of": self.view_of,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BufferAllocation":
        try:
            allocation = cls(
                name=data["name"],
                spec=TensorSpec.from_dict(data["spec"]),
                size_bytes=_require_int(data["size_bytes"], "size_bytes", minimum=0),
                storage=data["storage"],
                alignment=_validate_alignment(data["alignment"]),
                offset=data.get("offset"),
                defined_at=_require_int(data["defined_at"], "defined_at", minimum=-1),
                first_use=data.get("first_use"),
                last_use=_require_int(data["last_use"], "last_use", minimum=-1),
                is_input=data.get("is_input", False),
                is_output=data.get("is_output", False),
                is_constant=data.get("is_constant", False),
                alias_of=data.get("alias_of"),
                view_of=data.get("view_of"),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError("invalid memory allocation record") from exc
        if allocation.first_use is not None:
            _require_int(allocation.first_use, "first_use", minimum=-1)
        return allocation


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """A complete static storage contract for one graph."""

    graph_name: str
    alignment: int
    arena_size: int
    peak_live_bytes: int
    resident_bytes: int
    external_input_bytes: int
    external_output_bytes: int
    allocations: tuple[BufferAllocation, ...]
    aliases_allowed: bool = False

    def allocation(self, name: str) -> BufferAllocation:
        for allocation in self.allocations:
            if allocation.name == name:
                return allocation
        raise KeyError(name)

    @property
    def buffer_offsets(self) -> dict[str, int]:
        return {
            allocation.name: allocation.offset
            for allocation in self.allocations
            if allocation.storage == "arena" and allocation.offset is not None
        }

    def validate(self) -> None:
        alignment = _validate_alignment(self.alignment)
        if not isinstance(self.graph_name, str) or not self.graph_name:
            raise ValidationError("memory plan graph_name must be a non-empty string")
        if not isinstance(self.aliases_allowed, bool):
            raise ValidationError("aliases_allowed must be a boolean")
        _require_int(self.arena_size, "arena_size", minimum=0)
        _require_int(self.peak_live_bytes, "peak_live_bytes", minimum=0)
        names: set[str] = set()
        by_name: dict[str, BufferAllocation] = {}
        arena: list[BufferAllocation] = []
        valid_storage = {"external-input", "external-output", "resident-constant", "arena"}
        for item in self.allocations:
            if not isinstance(item, BufferAllocation):
                raise ValidationError("memory plan allocations must be BufferAllocation instances")
            if not isinstance(item.name, str) or not item.name or item.name in names:
                raise ValidationError(f"duplicate or empty allocation name {item.name!r}")
            names.add(item.name)
            by_name[item.name] = item
            if item.storage not in valid_storage:
                raise ValidationError(f"allocation {item.name!r} has invalid storage {item.storage!r}")
            if item.alignment != alignment:
                raise ValidationError(f"allocation {item.name!r} has inconsistent alignment")
            if not isinstance(item.spec, TensorSpec):
                raise ValidationError(f"allocation {item.name!r} has no TensorSpec")
            _require_int(item.size_bytes, f"{item.name}.size_bytes", minimum=0)
            _require_int(item.defined_at, f"{item.name}.defined_at", minimum=-1)
            _require_int(item.last_use, f"{item.name}.last_use", minimum=-1)
            if item.first_use is not None:
                _require_int(item.first_use, f"{item.name}.first_use", minimum=-1)
            if any(not isinstance(flag, bool) for flag in (item.is_input, item.is_output, item.is_constant)):
                raise ValidationError(f"allocation {item.name!r} role flags must be booleans")
            if item.alias_of is not None and (not isinstance(item.alias_of, str) or not item.alias_of):
                raise ValidationError(f"allocation {item.name!r} has an invalid alias source")
            if item.view_of is not None and (not isinstance(item.view_of, str) or not item.view_of):
                raise ValidationError(f"allocation {item.name!r} has an invalid view source")
            if item.alias_of is not None and item.view_of is not None:
                raise ValidationError(f"allocation {item.name!r} cannot be both an alias and a view")
            if item.spec.nbytes is None or item.size_bytes != item.spec.nbytes:
                raise ValidationError(f"allocation {item.name!r} size does not match its tensor spec")
            if item.live_end < item.live_start:
                raise ValidationError(f"allocation {item.name!r} has an invalid liveness interval")
            if item.first_use is not None and not (item.live_start <= item.first_use <= item.live_end):
                raise ValidationError(f"allocation {item.name!r} has first_use outside its liveness interval")
            if item.storage == "arena":
                if item.offset is None or isinstance(item.offset, bool) or item.offset < 0:
                    raise ValidationError(f"arena allocation {item.name!r} needs a non-negative offset")
                if (item.view_of is None and item.offset % alignment) or item.alignment != alignment:
                    raise ValidationError(f"arena allocation {item.name!r} is not aligned")
                if item.offset + item.size_bytes > self.arena_size:
                    raise ValidationError(f"arena allocation {item.name!r} exceeds the arena")
                arena.append(item)
            elif item.offset is not None:
                raise ValidationError(f"non-arena allocation {item.name!r} cannot have an arena offset")

        expected_arena_size = max(
            (item.offset + item.size_bytes for item in arena if item.offset is not None),
            default=0,
        )
        if self.arena_size != expected_arena_size:
            raise ValidationError("arena_size does not match allocation offsets")
        starts = {item.live_start for item in arena}
        expected_peak = max(
            (
                sum(
                    item.size_bytes
                    for item in arena
                    if item.view_of is None and item.live_start <= step <= item.live_end
                )
                for step in starts
            ),
            default=0,
        )
        if self.peak_live_bytes != expected_peak:
            raise ValidationError("peak_live_bytes does not match allocation lifetimes")
        expected_totals = {
            "resident_bytes": sum(item.size_bytes for item in self.allocations if item.resident),
            "external_input_bytes": sum(
                item.size_bytes for item in self.allocations if item.storage == "external-input"
            ),
            "external_output_bytes": sum(
                item.size_bytes for item in self.allocations if item.storage == "external-output"
            ),
        }
        for field, expected in expected_totals.items():
            actual = getattr(self, field)
            _require_int(actual, field, minimum=0)
            if actual != expected:
                raise ValidationError(f"{field} does not match allocations")

        view_roots: dict[str, str] = {}
        for item in self.allocations:
            if item.view_of is None:
                view_roots[item.name] = item.name
                continue
            source = by_name.get(item.view_of)
            if source is None or source.storage != "arena":
                raise ValidationError(f"view source {item.view_of!r} is not an arena value")
            if item.storage != "arena":
                raise ValidationError(f"view {item.name!r} must be an internal arena value")
            if source.name not in view_roots:
                raise ValidationError(f"view source {source.name!r} must be defined before {item.name!r}")
            assert source.offset is not None and item.offset is not None
            if (
                source.spec.dtype != item.spec.dtype
                or item.offset < source.offset
                or item.offset + item.size_bytes > source.offset + source.size_bytes
            ):
                raise ValidationError(f"view {item.name!r} is outside the storage of {source.name!r}")
            if source.live_end < item.live_end:
                raise ValidationError(f"view source {source.name!r} does not cover {item.name!r}'s lifetime")
            view_roots[item.name] = view_roots[source.name]

        for index, left in enumerate(arena):
            for right in arena[index + 1 :]:
                byte_overlap = (
                    left.size_bytes > 0
                    and right.size_bytes > 0
                    and left.offset < right.offset + right.size_bytes
                    and right.offset < left.offset + left.size_bytes
                )
                same_view_storage = view_roots[left.name] == view_roots[right.name]
                if byte_overlap and left.overlaps_lifetime(right) and not same_view_storage:
                    raise ValidationError(
                        f"arena allocations {left.name!r} and {right.name!r} overlap in bytes and lifetime"
                    )
        for item in arena:
            if item.alias_of is None:
                continue
            if not self.aliases_allowed:
                raise ValidationError("memory plan contains an alias while aliases are disabled")
            source = by_name.get(item.alias_of)
            if source is None or source.storage != "arena":
                raise ValidationError(f"alias source {item.alias_of!r} is not an arena value")
            if source.offset != item.offset or source.spec != item.spec:
                raise ValidationError(f"alias {item.name!r} does not exactly match {source.name!r}")
            if source.overlaps_lifetime(item):
                raise ValidationError(f"alias {item.name!r} overlaps the lifetime of {source.name!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MEMORY_PLAN_SCHEMA_VERSION,
            "graph_name": self.graph_name,
            "alignment": self.alignment,
            "arena_size": self.arena_size,
            "peak_live_bytes": self.peak_live_bytes,
            "resident_bytes": self.resident_bytes,
            "external_input_bytes": self.external_input_bytes,
            "external_output_bytes": self.external_output_bytes,
            "aliases_allowed": self.aliases_allowed,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryPlan":
        if data.get("schema_version") != MEMORY_PLAN_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported memory-plan schema {data.get('schema_version')!r}; "
                f"expected {MEMORY_PLAN_SCHEMA_VERSION}"
            )
        try:
            plan = cls(
                graph_name=data["graph_name"],
                alignment=_validate_alignment(data["alignment"]),
                arena_size=_require_int(data["arena_size"], "arena_size", minimum=0),
                peak_live_bytes=_require_int(data["peak_live_bytes"], "peak_live_bytes", minimum=0),
                resident_bytes=_require_int(data["resident_bytes"], "resident_bytes", minimum=0),
                external_input_bytes=_require_int(data["external_input_bytes"], "external_input_bytes", minimum=0),
                external_output_bytes=_require_int(data["external_output_bytes"], "external_output_bytes", minimum=0),
                allocations=tuple(BufferAllocation.from_dict(item) for item in data["allocations"]),
                aliases_allowed=data.get("aliases_allowed", False),
            )
        except (KeyError, TypeError) as exc:
            raise ValidationError("invalid memory plan") from exc
        plan.validate()
        return plan


def _storage_class(graph: Graph, name: str) -> StorageClass:
    if name in graph.inputs:
        return "external-input"
    if name in graph.constants:
        return "resident-constant"
    if name in graph.outputs:
        return "external-output"
    return "arena"


def _contiguous_slice_offset(node: Node, source: TensorSpec, destination: TensorSpec) -> int | None:
    """Return a Slice byte offset when its result is one dense C interval."""

    try:
        starts = node.attrs["starts"]
        ends = node.attrs["ends"]
        axes = node.attrs.get("axes", list(range(len(starts))))
        steps = node.attrs.get("steps", [1] * len(starts))
        if not all(isinstance(values, list) for values in (starts, ends, axes, steps)):
            return None
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            return None
        rank = source.rank
        if any(
            isinstance(axis, bool) or not isinstance(axis, (int, np.integer))
            for axis in axes
        ):
            return None
        normalized_axes = [int(axis) + rank if int(axis) < 0 else int(axis) for axis in axes]
        if len(set(normalized_axes)) != len(normalized_axes) or any(
            axis < 0 or axis >= rank for axis in normalized_axes
        ):
            return None
        selections = [(0, int(dimension), 1) for dimension in source.shape]
        for raw_start, raw_end, axis, raw_step in zip(
            starts, ends, normalized_axes, steps, strict=True
        ):
            if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (raw_start, raw_end, raw_step)):
                return None
            step = int(raw_step)
            if step != 1:
                return None
            selections[axis] = slice(int(raw_start), int(raw_end), step).indices(
                int(source.shape[axis])
            )
    except (KeyError, TypeError, ValueError):
        return None

    lengths = tuple(len(range(start, stop, step)) for start, stop, step in selections)
    if lengths != destination.shape or any(length <= 0 for length in lengths):
        return None
    partial_axes = [
        axis
        for axis, ((start, stop, _), dimension) in enumerate(zip(selections, source.shape, strict=True))
        if start != 0 or stop != dimension
    ]
    if partial_axes:
        pivot = partial_axes[-1]
        if any(lengths[axis] != 1 for axis in range(pivot)):
            return None
        if any(
            selections[axis] != (0, int(source.shape[axis]), 1)
            for axis in range(pivot + 1, source.rank)
        ):
            return None
    strides = [math.prod(source.shape[axis + 1 :]) for axis in range(source.rank)]
    element_offset = sum(selection[0] * stride for selection, stride in zip(selections, strides, strict=True))
    return element_offset * np.dtype(source.dtype).itemsize


def infer_zero_copy_view_specs(graph: Graph) -> dict[str, tuple[str, int]]:
    """Return internal dense views as ``destination -> (source, byte offset)``.

    Reshape is always a view when it preserves a static dense internal buffer.
    Slice is a view only when static unit-step indexing selects one contiguous
    C-order byte interval. Public ABI tensors and constants remain independent.
    """

    views: dict[str, tuple[str, int]] = {}
    dense_layouts = {"contiguous", "NCHW"}
    for node in graph.nodes:
        if node.op not in {"Reshape", "Slice"} or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue
        source, destination = node.inputs[0], node.outputs[0]
        if _storage_class(graph, source) != "arena" or _storage_class(graph, destination) != "arena":
            continue
        source_spec = graph.value_specs.get(source)
        destination_spec = graph.value_specs.get(destination)
        if source_spec is None or destination_spec is None:
            continue
        if (
            not source_spec.is_static
            or not destination_spec.is_static
            or source_spec.nbytes is None
            or source_spec.nbytes <= 0
            or source_spec.dtype != destination_spec.dtype
            or source_spec.layout not in dense_layouts
            or destination_spec.layout not in dense_layouts
        ):
            continue
        if node.op == "Reshape":
            if source_spec.nbytes != destination_spec.nbytes:
                continue
            byte_offset = 0
        else:
            byte_offset = _contiguous_slice_offset(node, source_spec, destination_spec)
            if byte_offset is None:
                continue
        views[destination] = (source, byte_offset)
    return views


def infer_zero_copy_views(graph: Graph) -> dict[str, str]:
    """Return the source value for each compiler-proven internal storage view."""

    return {
        destination: source
        for destination, (source, _) in infer_zero_copy_view_specs(graph).items()
    }


def _first_fit(size: int, alignment: int, conflicts: list[BufferAllocation]) -> int:
    if size == 0:
        return 0
    candidate = 0
    for item in sorted(conflicts, key=lambda value: (value.offset, value.name)):
        assert item.offset is not None
        candidate = _align_up(candidate, alignment)
        if candidate + size <= item.offset:
            return candidate
        if candidate < item.offset + item.size_bytes:
            candidate = item.offset + item.size_bytes
    return _align_up(candidate, alignment)


def plan_memory(
    graph: Graph,
    *,
    alignment: int = 256,
    aliases: Mapping[str, str] | None = None,
    alias_policy: Literal["forbid", "validated"] = "forbid",
    allow_aliases: bool | None = None,
) -> MemoryPlan:
    """Build a deterministic first-fit arena plan.

    Aliasing is forbidden by default.  ``alias_policy="validated"`` permits a
    destination to reuse the exact offset of an earlier, identically typed
    arena value only when their inclusive lifetimes do not overlap.  This
    conservative rule never assumes that an operation is safe in-place.
    ``allow_aliases`` is a convenience spelling retained for API callers.

    Independently of that opt-in reuse policy, compiler-proven internal
    Reshape values and contiguous Slice intervals are represented by
    ``view_of`` allocations. Their storage owner remains live through every
    view consumer, and views do not add to physical peak-live bytes.
    """

    graph.validate()
    alignment = _validate_alignment(alignment)
    if allow_aliases is not None:
        if not isinstance(allow_aliases, bool):
            raise ValidationError("allow_aliases must be a boolean")
        requested = "validated" if allow_aliases else "forbid"
        if alias_policy != "forbid" and alias_policy != requested:
            raise ValidationError("conflicting alias_policy and allow_aliases values")
        alias_policy = requested
    if not isinstance(alias_policy, str) or alias_policy not in {"forbid", "validated"}:
        raise ValidationError("alias_policy must be 'forbid' or 'validated'")
    try:
        alias_map = dict(aliases or {})
    except (TypeError, ValueError) as exc:
        raise ValidationError("aliases must be a mapping of destination to source names") from exc
    if any(
        not isinstance(destination, str)
        or not destination
        or not isinstance(source, str)
        or not source
        for destination, source in alias_map.items()
    ):
        raise ValidationError("aliases must map non-empty value names to non-empty value names")
    if alias_map and alias_policy == "forbid":
        raise ValidationError("buffer aliases are forbidden by this memory plan")

    definition: dict[str, int] = {name: -1 for name in graph.inputs}
    definition.update({name: -1 for name in graph.constants})
    order = sorted(graph.inputs) + sorted(graph.constants)
    for step, node in enumerate(graph.nodes):
        for output in node.outputs:
            definition[output] = step
            order.append(output)

    uses: dict[str, list[int]] = {name: [] for name in definition}
    for step, node in enumerate(graph.nodes):
        for input_name in node.inputs:
            uses[input_name].append(step)
    exit_step = len(graph.nodes)
    for output in graph.outputs:
        uses[output].append(exit_step)

    view_specs = infer_zero_copy_view_specs(graph)
    view_map = {
        destination: source
        for destination, (source, _) in view_specs.items()
    }
    conflicting_view_aliases = sorted(set(view_map) & set(alias_map))
    if conflicting_view_aliases:
        raise ValidationError(
            f"values cannot be both storage aliases and zero-copy views: {conflicting_view_aliases}"
        )
    effective_last_use = {
        name: max(value_uses) if value_uses else definition[name]
        for name, value_uses in uses.items()
    }
    for destination in reversed(order):
        source = view_map.get(destination)
        if source is not None:
            effective_last_use[source] = max(
                effective_last_use[source], effective_last_use[destination]
            )

    unknown_aliases = sorted((set(alias_map) | set(alias_map.values())) - set(definition))
    if unknown_aliases:
        raise ValidationError(f"aliases refer to unknown values: {unknown_aliases}")

    allocations: list[BufferAllocation] = []
    arena_allocations: list[BufferAllocation] = []
    by_name: dict[str, BufferAllocation] = {}
    for name in order:
        spec = graph.value_specs.get(name)
        if spec is None:
            raise ValidationError(f"value {name!r} has no TensorSpec; run shape inference before memory planning")
        if not spec.is_static or spec.nbytes is None:
            raise ValidationError(f"value {name!r} requires a static shape for memory planning")
        size = spec.nbytes
        value_uses = uses[name]
        first_use = min(value_uses) if value_uses else None
        last_use = effective_last_use[name]
        storage = _storage_class(graph, name)
        alias_of = alias_map.get(name)
        view_of = view_map.get(name)
        offset: int | None = None

        if view_of is not None:
            source = by_name.get(view_of)
            if source is None:
                raise ValidationError(f"view source {view_of!r} must be defined before {name!r}")
            if source.storage != "arena" or storage != "arena":
                raise ValidationError(f"view {name!r} and its source must be internal arena values")
            if source.spec.dtype != spec.dtype or source.size_bytes < size:
                raise ValidationError(f"view {name!r} must fit within source storage with the same dtype")
            relative_offset = view_specs[name][1]
            assert source.offset is not None
            offset = source.offset + relative_offset
            if offset + size > source.offset + source.size_bytes:
                raise ValidationError(f"view {name!r} exceeds source storage")
        elif alias_of is not None:
            if storage != "arena":
                raise ValidationError(f"alias destination {name!r} must be an internal arena value")
            source = by_name.get(alias_of)
            if source is None:
                raise ValidationError(f"alias source {alias_of!r} must be defined before {name!r}")
            if source.storage != "arena":
                raise ValidationError(f"alias source {alias_of!r} must be an internal arena value")
            if source.spec != spec:
                raise ValidationError(f"alias {name!r} must have the same tensor spec as {alias_of!r}")
            if not (source.last_use < definition[name] or last_use < source.defined_at):
                raise ValidationError(f"alias {name!r} overlaps the lifetime of {alias_of!r}")
            offset = source.offset

        if storage == "arena" and offset is None:
            conflicts = [
                item
                for item in arena_allocations
                if not (item.last_use < definition[name] or last_use < item.defined_at)
            ]
            offset = _first_fit(size, alignment, conflicts)
        elif storage == "arena" and view_of is None:
            assert offset is not None
            for item in arena_allocations:
                assert item.offset is not None
                life_overlap = not (item.last_use < definition[name] or last_use < item.defined_at)
                byte_overlap = (
                    size > 0
                    and item.size_bytes > 0
                    and offset < item.offset + item.size_bytes
                    and item.offset < offset + size
                )
                if life_overlap and byte_overlap:
                    raise ValidationError(
                        f"alias {name!r} would overlap live arena value {item.name!r}"
                    )

        allocation = BufferAllocation(
            name=name,
            spec=spec,
            size_bytes=size,
            storage=storage,
            alignment=alignment,
            offset=offset,
            defined_at=definition[name],
            first_use=first_use,
            last_use=last_use,
            is_input=name in graph.inputs,
            is_output=name in graph.outputs,
            is_constant=name in graph.constants,
            alias_of=alias_of,
            view_of=view_of,
        )
        allocations.append(allocation)
        by_name[name] = allocation
        if storage == "arena":
            arena_allocations.append(allocation)

    arena_size = max(
        (item.offset + item.size_bytes for item in arena_allocations if item.offset is not None),
        default=0,
    )
    peak_live = 0
    for step in range(-1, exit_step + 1):
        peak_live = max(
            peak_live,
            sum(
                item.size_bytes
                for item in arena_allocations
                if item.view_of is None and item.live_start <= step <= item.live_end
            ),
        )

    plan = MemoryPlan(
        graph_name=graph.name,
        alignment=alignment,
        arena_size=arena_size,
        peak_live_bytes=peak_live,
        resident_bytes=sum(item.size_bytes for item in allocations if item.resident),
        external_input_bytes=sum(item.size_bytes for item in allocations if item.storage == "external-input"),
        external_output_bytes=sum(item.size_bytes for item in allocations if item.storage == "external-output"),
        allocations=tuple(allocations),
        aliases_allowed=alias_policy == "validated",
    )
    plan.validate()
    return plan


# A descriptive alias for callers that treat planning as a compiler pass.
plan_static_memory = plan_memory
