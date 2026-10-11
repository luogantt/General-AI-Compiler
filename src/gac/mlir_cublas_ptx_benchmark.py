"""Real-session autotune adapter for the two BEVFormer FFN hybrid paths."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backends.mlir_cublas_ptx import (
    SUPPORTED_PRODUCTION_QUERY_SHAPES,
    TARGET as CUBLAS_PTX_TARGET,
)
from .backends.mlir_cuda import TARGET as MLIR_CUDA_TARGET
from .errors import BackendError, ValidationError
from .mlir_ptx_autotune import AutotuneCandidate
from .mlir_tiled_ffn import (
    TARGET as TILED_FFN_TARGET,
    MlirTiledFfnProgram,
    validate_mlir_tiled_ffn_program,
)


# AutotuneCandidate currently requires a tile tuple even for library-managed
# MatMuls.  (1, 1, 1) is an explicit non-tunable sentinel; no claim is made
# that either cuBLAS path executes a 1x1x1 tile.
LIBRARY_MANAGED_TILE = (1, 1, 1)
FUSION_BLOCK = (256, 1, 1)
EXECUTABLE_STRATEGIES = (CUBLAS_PTX_TARGET, MLIR_CUDA_TARGET)


class UnsupportedFfnCandidate(ValidationError):
    """A candidate has no executable session/code-generation mapping."""


def tiled_ffn_candidate(program: MlirTiledFfnProgram) -> AutotuneCandidate:
    """Derive tuning metadata from one validated executable tiled program."""

    if not isinstance(program, MlirTiledFfnProgram):
        raise ValidationError("tiled candidate requires an MlirTiledFfnProgram")
    validate_mlir_tiled_ffn_program(program)
    return AutotuneCandidate(
        backend_strategy=TILED_FFN_TARGET,
        matmul_tile=program.matmul_tile,
        thread_block=program.thread_block,
        precision="fp32",
        fusion_strategy="full-ffn",
    )


def executable_ffn_candidates(
    tiled_program: MlirTiledFfnProgram | None = None,
) -> tuple[AutotuneCandidate, ...]:
    """Return candidates backed by the supplied real full-FFN sessions."""

    candidates = tuple(
        AutotuneCandidate(
            backend_strategy=strategy,
            matmul_tile=LIBRARY_MANAGED_TILE,
            thread_block=FUSION_BLOCK,
            precision="fp32",
            fusion_strategy="full-ffn",
        )
        for strategy in EXECUTABLE_STRATEGIES
    )
    if tiled_program is not None:
        candidates += (tiled_ffn_candidate(tiled_program),)
    return candidates


def unwired_tiled_candidates() -> tuple[AutotuneCandidate, ...]:
    """Describe research candidates that must be skipped by this adapter.

    These records are useful in benchmark reports, but are deliberately never
    sent to the tuner because no full-FFN runtime session consumes their tile
    metadata yet.
    """

    return (
        AutotuneCandidate(
            backend_strategy="mlir-ptx-tiled",
            matmul_tile=(64, 64, 32),
            thread_block=(256, 1, 1),
            precision="fp32",
            fusion_strategy="full-ffn",
        ),
        AutotuneCandidate(
            backend_strategy="mlir-nvgpu-mma",
            matmul_tile=(128, 64, 32),
            thread_block=(128, 1, 1),
            precision="fp16",
            fusion_strategy="full-ffn",
        ),
    )


def skipped_tiled_candidate_report() -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate.to_dict(),
            "candidate_sha256": candidate.sha256,
            "status": "skipped",
            "reason": (
                "no full-FFN runtime session consumes this tile/precision policy; "
                "metadata-only candidates are not executed or timed"
            ),
        }
        for candidate in unwired_tiled_candidates()
    ]


def _device_projection(info: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "device_id",
        "name",
        "compute_capability",
        "driver_version",
        "runtime_version",
        "pci_bus_id",
        "uuid",
    )
    result = {key: info[key] for key in keys if key in info}
    capability = result.get("compute_capability")
    if isinstance(capability, tuple):
        result["compute_capability"] = list(capability)
    return result


class FfnSessionAutotuneRunner:
    """Map candidates to validated live full-FFN sessions and CUDA Event timings.

    Both session inputs and all weights stay resident.  Each latency sample is
    one Event interval around ``iterations`` complete FFN runs,
    divided by ``iterations`` and returned in milliseconds.
    """

    def __init__(
        self,
        query: np.ndarray,
        sessions: Mapping[str, Any],
    ) -> None:
        allowed = set(EXECUTABLE_STRATEGIES) | {TILED_FFN_TARGET}
        if (
            not set(EXECUTABLE_STRATEGIES).issubset(sessions)
            or not set(sessions).issubset(allowed)
        ):
            raise ValidationError(
                "autotune sessions must contain mlir-cublas-ptx-sm86 and "
                "mlir-cuda-sm86, with at most one mlir-tiled-ffn-sm86 session"
            )
        strategies = EXECUTABLE_STRATEGIES + (
            (TILED_FFN_TARGET,) if TILED_FFN_TARGET in sessions else ()
        )
        self._sessions = dict(sessions)
        self._strategies = strategies
        self._tiled_candidate: AutotuneCandidate | None = None
        if TILED_FFN_TARGET in self._sessions:
            self._tiled_candidate = tiled_ffn_candidate(
                self._sessions[TILED_FFN_TARGET].program
            )
        self._submits: dict[str, Callable[[], None]] = {}
        expected_query_shape: tuple[int, ...] | None = None
        for strategy, session in self._sessions.items():
            program = getattr(session, "program", None)
            if getattr(program, "target", None) != strategy:
                raise ValidationError(
                    f"session for {strategy!r} has a different compiler target"
                )
            graph = getattr(program, "graph", None)
            inputs = getattr(graph, "inputs", None)
            if not isinstance(inputs, Mapping) or set(inputs) != {"query"}:
                raise ValidationError(
                    f"session for {strategy!r} has no canonical query graph input"
                )
            spec = inputs["query"]
            shape = tuple(getattr(spec, "shape", ()))
            if (
                shape not in SUPPORTED_PRODUCTION_QUERY_SHAPES
                or getattr(spec, "dtype", None) != "float32"
                or getattr(spec, "layout", None) != "contiguous"
            ):
                raise ValidationError(
                    f"session for {strategy!r} has an unsupported FFN query profile"
                )
            if expected_query_shape is None:
                expected_query_shape = shape
            elif shape != expected_query_shape:
                raise ValidationError(
                    "autotune sessions must use the same FFN query profile"
                )
            submit = getattr(session, "_execute_calls", None)
            if not callable(submit):
                raise ValidationError(
                    f"session for {strategy!r} has no full-FFN submitter"
                )
            for attribute in (
                "_cp",
                "_device",
                "_stream",
                "_inputs",
                "_lock",
                "_closed",
                "_failed",
                "info",
            ):
                if not hasattr(session, attribute):
                    raise ValidationError(
                        f"session for {strategy!r} lacks {attribute}"
                    )
            self._submits[strategy] = submit

        assert expected_query_shape is not None
        contiguous = np.ascontiguousarray(query)
        if contiguous.shape != expected_query_shape or contiguous.dtype != np.float32:
            rendered = "[" + ",".join(str(value) for value in expected_query_shape) + "]"
            raise ValidationError(
                f"autotune query must be contiguous float32 {rendered}"
            )
        self._query = contiguous.copy()

        descriptions = {
            strategy: _device_projection(session.info)
            for strategy, session in self._sessions.items()
        }
        first = descriptions[EXECUTABLE_STRATEGIES[0]]
        if any(description != first for description in descriptions.values()):
            raise ValidationError("autotune sessions must use the same CUDA device")
        self._device = {
            **first,
            "resident_timing": "cuda-events",
            "candidate_backends": list(self._strategies),
        }
        identity_payload = {
            "device": self._device,
            "sessions": {
                key: id(self._sessions[key]) for key in self._strategies
            },
            "streams": {
                key: id(self._sessions[key]._stream)
                for key in self._strategies
            },
        }
        self._identity = hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for strategy in self._strategies:
            self._initialize_resident_input(strategy)

    def _initialize_resident_input(self, strategy: str) -> None:
        """Upload under the Session lock and honor its lifecycle contract."""

        session = self._sessions[strategy]
        if not session._lock.acquire(blocking=False):
            raise BackendError(
                "concurrent autotune initialization/run use of one FFN session "
                "is not supported"
            )
        try:
            if session._closed or session._failed:
                raise BackendError(
                    "FFN autotune session is closed or failed; create a new session"
                )
            self._upload(strategy)
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            session._failed = True
            try:
                with session._device:
                    session._stream.synchronize()
            except Exception:
                pass
            raise BackendError(
                f"FFN resident input initialization failed: {exc}"
            ) from exc
        finally:
            session._lock.release()

    def unsupported_reason(self, candidate: AutotuneCandidate) -> str | None:
        if not isinstance(candidate, AutotuneCandidate):
            return "candidate is not an AutotuneCandidate"
        if candidate.backend_strategy not in self._sessions:
            return "backend strategy has no executable full-FFN session"
        if candidate.backend_strategy == TILED_FFN_TARGET:
            if self._tiled_candidate is None:
                return "backend strategy has no executable tiled FFN session"
            if candidate != self._tiled_candidate:
                return (
                    "tile, block, precision or fusion metadata differs from the "
                    "compiled tiled FFN artifact"
                )
            return None
        if candidate.matmul_tile != LIBRARY_MANAGED_TILE:
            return "tile metadata is not wired to either cuBLAS runtime"
        if candidate.thread_block != FUSION_BLOCK:
            return "fusion block metadata differs from the compiled kernels"
        if candidate.precision != "fp32":
            return "the two executable sessions currently use FP32 storage"
        if candidate.fusion_strategy != "full-ffn":
            return "candidate does not represent the complete FFN pipeline"
        return None

    def _require_candidate(self, candidate: AutotuneCandidate) -> str:
        reason = self.unsupported_reason(candidate)
        if reason is not None:
            raise UnsupportedFfnCandidate(reason)
        return candidate.backend_strategy

    def _upload(self, strategy: str) -> None:
        session = self._sessions[strategy]
        try:
            query = session._inputs["query"]
        except (KeyError, TypeError) as exc:
            raise ValidationError(
                f"session for {strategy!r} has no resident query input"
            ) from exc
        with session._device, session._stream:
            query.set(self._query, stream=session._stream)
            session._stream.synchronize()

    def session_identity(self) -> str:
        return self._identity

    def describe_device(self) -> Mapping[str, Any]:
        return dict(self._device)

    def run(self, candidate: AutotuneCandidate) -> Mapping[str, Any]:
        strategy = self._require_candidate(candidate)
        result = self._sessions[strategy].run({"query": self._query})
        outputs = getattr(result, "outputs", None)
        if not isinstance(outputs, Mapping):
            raise ValidationError("full-FFN session returned no output mapping")
        return dict(outputs)

    def benchmark(
        self,
        candidate: AutotuneCandidate,
        *,
        warmup: int,
        iterations: int,
        repeats: int,
    ) -> Sequence[float]:
        strategy = self._require_candidate(candidate)
        for name, value, minimum in (
            ("warmup", warmup, 0),
            ("iterations", iterations, 1),
            ("repeats", repeats, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValidationError(f"{name} must be an integer >= {minimum}")

        session = self._sessions[strategy]
        submit = self._submits[strategy]
        if not session._lock.acquire(blocking=False):
            raise BackendError(
                "concurrent benchmark/run use of one FFN session is not supported"
            )
        try:
            if session._closed or session._failed:
                raise BackendError(
                    "FFN benchmark session is closed or failed; create a new session"
                )
            self._upload(strategy)
            cp = session._cp
            with session._device, session._stream:
                for _ in range(warmup):
                    submit()
                session._stream.synchronize()

            samples: list[float] = []
            for _ in range(repeats):
                with session._device, session._stream:
                    start = cp.cuda.Event()
                    end = cp.cuda.Event()
                    start.record(session._stream)
                    for _ in range(iterations):
                        submit()
                    end.record(session._stream)
                    end.synchronize()
                    elapsed_ms = (
                        float(cp.cuda.get_elapsed_time(start, end)) / iterations
                    )
                if not np.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                    raise ValidationError(
                        "CUDA Event returned a non-positive latency"
                    )
                samples.append(elapsed_ms)
            return tuple(samples)
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            session._failed = True
            try:
                with session._device:
                    session._stream.synchronize()
            except Exception:
                pass
            raise BackendError(f"FFN CUDA Event benchmark failed: {exc}") from exc
        finally:
            session._lock.release()


__all__ = [
    "EXECUTABLE_STRATEGIES",
    "FUSION_BLOCK",
    "FfnSessionAutotuneRunner",
    "LIBRARY_MANAGED_TILE",
    "UnsupportedFfnCandidate",
    "executable_ffn_candidates",
    "skipped_tiled_candidate_report",
    "tiled_ffn_candidate",
    "unwired_tiled_candidates",
]
