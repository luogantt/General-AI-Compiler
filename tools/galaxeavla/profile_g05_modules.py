"""Temporary, synchronized module timing for a separate G0.5 diagnostic run.

This utility accepts an already loaded policy and does not import torch or load
weights. Warm up the selected backend before attaching it. Do not mix these
serialized, instrumented durations with the uninstrumented latency benchmark;
wrappers may also change compiler guards. Prefer a separate profiling process.

Example (after model loading and normal backend warmup)::

    with ModuleProfiler(policy, runtime.synchronize) as profile:
        with profile.phase("request_0"):
            state = policy.prefill(batch["samples"], batch["pixel_values"])
            result = policy.generate_action(
                state, batch["samples"], action_dim_is_pad=batch["action_dim_is_pad"]
            )
    report = profile.summary()  # All original methods are already restored.

Use the same inference_mode/autocast context as the benchmark. The caller owns
that context; this profiler changes no inputs, outputs, precision, or model flags.
"""
from __future__ import annotations

import contextlib
import functools
import statistics
import time
from typing import Any, Callable


_MARKER = "_g05_module_profiler_wrapper"


class ModuleProfiler:
    """Record six coarse module boundaries, restoring method ownership on exit.

    Existing instance overrides (including torch.compile forward functions) are
    restored exactly. Originally class-owned methods are restored by deleting
    the temporary instance attribute. Later replacements made by someone else
    are preserved. This object is intended for one thread/stream of requests.
    """

    def __init__(self, policy: Any, synchronize: Callable[[], None], *,
                 enabled: bool = True, clock: Callable[[], float] = time.perf_counter):
        self.policy = policy
        self.synchronize = synchronize
        self.enabled = enabled
        self.clock = clock
        self.records: list[dict[str, Any]] = []
        self._replacements: list[tuple[Any, str, Any, bool, Any]] = []
        self._active = False
        self._phase = "unscoped"
        self._depth = 0

    def _targets(self):
        model = self.policy.model
        expert = model.action_expert
        # Pinned MixtureQwen35 names the action projections embed/decode.
        return (
            ("vision_tower.forward", model.vision_tower, "forward", "vision"),
            ("vlm.forward", model.vlm, "forward", "vlm"),
            ("action_expert.forward", expert, "forward", "action_expert"),
            ("action_expert.embed", expert, "embed", "embed_action"),
            ("action_expert.decode", expert, "decode", "decode_action"),
            ("action_expert.encode_time", expert, "encode_time", "encode_time"),
        )

    def _wrap(self, original: Callable, name: str, role: str):
        @functools.wraps(original)
        def timed(*args, **kwargs):
            self.synchronize()  # Exclude already queued work from this boundary.
            started = self.clock()
            phase, depth = self._phase, self._depth
            self._depth += 1
            error_type = None
            try:
                return original(*args, **kwargs)
            except BaseException as error:
                error_type = type(error).__name__
                raise
            finally:
                self._depth -= 1
                self.synchronize()
                elapsed = (self.clock() - started) * 1000.0
                self.records.append({
                    "module": name, "role": role, "phase": phase,
                    "duration_ms": elapsed, "nested_depth": depth,
                    "error_type": error_type,
                })

        setattr(timed, _MARKER, True)
        return timed

    def attach(self):
        if self._active:
            raise RuntimeError("Module profiler is already attached")
        if not self.enabled:
            self._active = True
            return self
        if not callable(self.synchronize):
            raise TypeError("synchronize must be a zero-argument callable")
        # Resolve everything before mutating a module, so an unexpected policy
        # API fails without leaving partially installed wrappers.
        resolved = []
        for name, module, attribute, role in self._targets():
            original = getattr(module, attribute)
            if not callable(original):
                raise TypeError(f"{name} is not callable")
            if getattr(original, _MARKER, False):
                raise RuntimeError(f"Another module profiler is already attached to {name}")
            resolved.append((name, module, attribute, role, original, attribute in vars(module)))
        try:
            for name, module, attribute, role, original, owned in resolved:
                installed = self._wrap(original, name, role)
                setattr(module, attribute, installed)
                self._replacements.append((module, attribute, original, owned, installed))
        except BaseException:
            self.close()
            raise
        self._active = True
        return self

    def close(self):
        for module, attribute, original, owned, installed in reversed(self._replacements):
            if getattr(module, attribute) is installed:
                if owned:
                    setattr(module, attribute, original)
                else:
                    delattr(module, attribute)
        self._replacements.clear()
        self._active = False

    def __enter__(self):
        return self.attach()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @contextlib.contextmanager
    def phase(self, name: str):
        previous = self._phase
        self._phase = name
        try:
            yield self
        finally:
            self._phase = previous

    def summary(self) -> dict[str, Any]:
        modules = {}
        # Include missing/uncalled targets: a zero count exposes skipped stages.
        for name, _, _, role in self._targets():
            rows = [record for record in self.records if record["module"] == name]
            durations = [row["duration_ms"] for row in rows]
            modules[name] = {
                "role": role, "calls": len(rows),
                "errors": sum(row["error_type"] is not None for row in rows),
                "total_ms": sum(durations),
                "mean_ms": statistics.fmean(durations) if durations else None,
                "p50_ms": statistics.median(durations) if durations else None,
                "min_ms": min(durations) if durations else None,
                "max_ms": max(durations) if durations else None,
                "durations_ms": durations,
            }
        return {
            "schema_version": 1, "enabled": self.enabled, "attached": self._active,
            "scope": "Synchronized module diagnostics; not an end-to-end latency benchmark",
            "timing_boundary": "Synchronize before clock; call original function; synchronize before stopping clock",
            "limitations": [
                "Synchronization serializes execution and adds host/device overhead",
                "Inclusive times can overlap when wrapped boundaries are nested; do not sum them as end-to-end latency",
                "Does not time policy preprocessing, action-mask/KV preparation or FM Euler updates outside these methods",
                "Compile/capture costs are included if they occur; warm up before attaching and use a separate diagnostic run",
            ],
            "modules": modules, "calls": [dict(record) for record in self.records],
        }
