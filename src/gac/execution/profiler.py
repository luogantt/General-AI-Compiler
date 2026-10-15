"""Runtime tracing with stable model-to-dispatch provenance identifiers."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterator


@dataclass(frozen=True)
class TraceEvent:
    sequence: int
    kind: str
    name: str
    device: str
    start_ns: int
    duration_ns: int
    source_nodes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_nodes"] = list(self.source_nodes)
        value["duration_us"] = self.duration_us
        return value


class Profiler:
    """Thread-safe event collector for one or more session invocations."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._events: list[TraceEvent] = []
        self._lock = Lock()
        self._next_sequence = 0

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._next_sequence = 0

    def add(
        self,
        *,
        kind: str,
        name: str,
        device: str,
        start_ns: int,
        duration_ns: int,
        source_nodes: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            self._events.append(
                TraceEvent(
                    sequence=sequence,
                    kind=kind,
                    name=name,
                    device=device,
                    start_ns=start_ns,
                    duration_ns=max(0, duration_ns),
                    source_nodes=source_nodes,
                    metadata=dict(metadata or {}),
                )
            )

    @contextmanager
    def record(
        self,
        kind: str,
        name: str,
        device: str,
        *,
        source_nodes: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            self.add(
                kind=kind,
                name=name,
                device=device,
                start_ns=start_ns,
                duration_ns=end_ns - start_ns,
                source_nodes=source_nodes,
                metadata=metadata,
            )

    def summary(self) -> dict[str, Any]:
        events = self.events
        by_kind: dict[str, dict[str, float | int]] = {}
        by_device: dict[str, dict[str, float | int]] = {}
        for event in events:
            for table, key in ((by_kind, event.kind), (by_device, event.device)):
                entry = table.setdefault(key, {"count": 0, "duration_us": 0.0})
                entry["count"] = int(entry["count"]) + 1
                entry["duration_us"] = float(entry["duration_us"]) + event.duration_us
        total_ns = sum(event.duration_ns for event in events)
        return {
            "event_count": len(events),
            "total_recorded_us": total_ns / 1_000.0,
            "by_kind": by_kind,
            "by_device": by_device,
        }

    def cuda_launch_profile(self) -> dict[str, Any]:
        """Return stable, JSON-safe CUDA Event timings grouped by graph op.

        Dispatch ``duration_ns`` values describe host submission time.  A CUDA
        session adds ``gpu_elapsed_ms`` to dispatch metadata only after the
        stream has completed, so this report deliberately aggregates that
        device-side clock instead of the host trace duration.
        """

        dispatches = [event for event in self.events if event.kind == "dispatch"]
        launches: list[dict[str, Any]] = []
        samples_by_op: dict[str, list[float]] = {}
        samples_by_implementation: dict[str, list[float]] = {}
        untimed = 0
        for event in dispatches:
            elapsed = event.metadata.get("gpu_elapsed_ms")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
            ):
                untimed += 1
                continue
            elapsed_ms = float(elapsed)
            op = str(event.metadata.get("op", "unknown"))
            implementation = str(
                event.metadata.get("implementation", "unknown")
            )
            samples_by_op.setdefault(op, []).append(elapsed_ms)
            samples_by_implementation.setdefault(implementation, []).append(
                elapsed_ms
            )
            launches.append(
                {
                    "sequence": event.sequence,
                    "node_id": str(event.metadata.get("node_id", "")),
                    "op": op,
                    "implementation": implementation,
                    "entry_point": str(event.metadata.get("entry_point", event.name)),
                    "source_nodes": list(event.source_nodes),
                    "grid": list(event.metadata.get("grid", ())),
                    "block": list(event.metadata.get("block", ())),
                    "output_bytes": int(event.metadata.get("output_bytes", 0)),
                    "storage": str(event.metadata.get("storage", "")),
                    "arena_offset": event.metadata.get("arena_offset"),
                    "gpu_elapsed_ms": elapsed_ms,
                }
            )

        total_ms = sum(item["gpu_elapsed_ms"] for item in launches)
        by_op: dict[str, dict[str, float | int]] = {}
        for op in sorted(samples_by_op):
            samples = samples_by_op[op]
            op_total = sum(samples)
            by_op[op] = {
                "launch_count": len(samples),
                "total_gpu_ms": op_total,
                "mean_gpu_ms": op_total / len(samples),
                "min_gpu_ms": min(samples),
                "max_gpu_ms": max(samples),
                "share_percent": (op_total * 100.0 / total_ms) if total_ms else 0.0,
            }

        by_implementation: dict[str, dict[str, float | int]] = {}
        for implementation in sorted(samples_by_implementation):
            samples = samples_by_implementation[implementation]
            implementation_total = sum(samples)
            by_implementation[implementation] = {
                "launch_count": len(samples),
                "total_gpu_ms": implementation_total,
                "mean_gpu_ms": implementation_total / len(samples),
                "min_gpu_ms": min(samples),
                "max_gpu_ms": max(samples),
                "share_percent": (
                    implementation_total * 100.0 / total_ms
                    if total_ms
                    else 0.0
                ),
            }

        return {
            "schema_version": 1,
            "clock": "cuda-event",
            "time_unit": "ms",
            "launch_count": len(dispatches),
            "timed_launch_count": len(launches),
            "untimed_launch_count": untimed,
            "total_gpu_ms": total_ms,
            "by_op": by_op,
            "by_implementation": by_implementation,
            "launches": launches,
        }

    def chrome_trace(self) -> dict[str, Any]:
        return {
            "traceEvents": [
                {
                    "name": event.name,
                    "cat": event.kind,
                    "ph": "X",
                    "ts": event.start_ns / 1_000.0,
                    "dur": event.duration_us,
                    "pid": 1,
                    "tid": event.device,
                    "args": {
                        **event.metadata,
                        "source_nodes": list(event.source_nodes),
                        "sequence": event.sequence,
                    },
                }
                for event in self.events
            ],
            "displayTimeUnit": "ms",
        }

    def export_chrome_trace(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.chrome_trace(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
