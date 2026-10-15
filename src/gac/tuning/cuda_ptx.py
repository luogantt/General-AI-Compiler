"""Deterministic, correctness-gated autotuning for MLIR/PTX FFN kernels.

The tuner is deliberately independent from CuPy.  A CUDA adapter owns one live
session and implements :class:`AutotuneRunner`; CPU-only tests can inject the
same interface.  Candidate output is checked before that candidate is timed.

The JSON cache is an integrity-checked local performance cache, not a digital
signature.  Only caches produced in a trusted build environment should be
used.  A cache is bound to the graph, compiler source, device, candidate space,
and measurement configuration and is published with an atomic replace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from gac.core.errors import BackendError, ValidationError


AUTOTUNE_CACHE_KIND = "gac.mlir-ptx-ffn-autotune"
AUTOTUNE_CACHE_SCHEMA_VERSION = 1
MAX_CACHE_BYTES = 4 * 1024 * 1024
MAX_CANDIDATES = 256

SUPPORTED_PRECISIONS = frozenset({"fp32", "tf32", "fp16"})
SUPPORTED_FUSION_STRATEGIES = frozenset(
    {"none", "bias-relu", "bias-residual-layernorm", "full-ffn"}
)

_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_STRATEGY_RE = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
_REJECTION_REASONS = frozenset(
    {
        "execution-error",
        "invalid-output",
        "incorrect-output",
        "benchmark-error",
        "invalid-timing",
    }
)


class AutotuneError(BackendError):
    """The candidate space could not produce a valid tuned implementation."""


class AutotuneCacheError(AutotuneError):
    """An autotune cache is corrupt or does not match the requested workload."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int_tuple(name: str, value: object, length: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValidationError(f"{name} must contain exactly {length} integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise ValidationError(f"{name} must contain only integers")
        integer = int(item)
        if integer <= 0:
            raise ValidationError(f"{name} dimensions must be positive")
        result.append(integer)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AutotuneCandidate:
    """One complete FFN code-generation policy.

    ``matmul_tile`` is ``(M, N, K)`` and ``thread_block`` is CUDA ``(x, y, z)``.
    The backend strategy is an inert identifier such as ``"cublaslt"``,
    ``"mlir-ptx-tiled"``, or ``"mlir-nvgpu-mma"``; this module never executes
    it as code.
    """

    backend_strategy: str
    matmul_tile: tuple[int, int, int]
    thread_block: tuple[int, int, int]
    precision: str
    fusion_strategy: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.backend_strategy, str)
            or _STRATEGY_RE.fullmatch(self.backend_strategy) is None
        ):
            raise ValidationError(
                "backend_strategy must be a lowercase strategy identifier"
            )
        tile = _positive_int_tuple("matmul_tile", self.matmul_tile, 3)
        block = _positive_int_tuple("thread_block", self.thread_block, 3)
        if any(dimension > 65536 for dimension in tile):
            raise ValidationError("matmul_tile dimensions must not exceed 65536")
        if any(dimension > 1024 for dimension in block):
            raise ValidationError("thread_block dimensions must not exceed 1024")
        if math.prod(block) > 1024:
            raise ValidationError("thread_block must not exceed 1024 threads")
        if (
            not isinstance(self.precision, str)
            or self.precision not in SUPPORTED_PRECISIONS
        ):
            raise ValidationError(
                f"unsupported precision {self.precision!r}; expected one of "
                f"{sorted(SUPPORTED_PRECISIONS)}"
            )
        if (
            not isinstance(self.fusion_strategy, str)
            or self.fusion_strategy not in SUPPORTED_FUSION_STRATEGIES
        ):
            raise ValidationError(
                f"unsupported fusion_strategy {self.fusion_strategy!r}; expected "
                f"one of {sorted(SUPPORTED_FUSION_STRATEGIES)}"
            )
        object.__setattr__(self, "matmul_tile", tile)
        object.__setattr__(self, "thread_block", block)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_strategy": self.backend_strategy,
            "matmul_tile": list(self.matmul_tile),
            "thread_block": list(self.thread_block),
            "precision": self.precision,
            "fusion_strategy": self.fusion_strategy,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "AutotuneCandidate":
        if not isinstance(value, Mapping):
            raise AutotuneCacheError("cached candidate must be a JSON object")
        expected = {
            "backend_strategy",
            "matmul_tile",
            "thread_block",
            "precision",
            "fusion_strategy",
        }
        if set(value) != expected:
            raise AutotuneCacheError("cached candidate has unexpected fields")
        try:
            return cls(
                backend_strategy=value["backend_strategy"],
                matmul_tile=value["matmul_tile"],
                thread_block=value["thread_block"],
                precision=value["precision"],
                fusion_strategy=value["fusion_strategy"],
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise AutotuneCacheError(f"cached candidate is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AutotuneConfig:
    """Correctness tolerances and event-timing sample counts."""

    warmup: int = 10
    iterations: int = 50
    repeats: int = 9
    atol: float = 1e-4
    rtol: float = 1e-4

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("warmup", self.warmup, 0, 100_000),
            ("iterations", self.iterations, 1, 1_000_000),
            ("repeats", self.repeats, 1, 10_001),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationError(f"{name} must be an integer")
            if not lower <= value <= upper:
                raise ValidationError(f"{name} must be in [{lower}, {upper}]")
        for name, value in (("atol", self.atol), ("rtol", self.rtol)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValidationError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "iterations": self.iterations,
            "repeats": self.repeats,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AutotuneConfig":
        if not isinstance(value, Mapping) or set(value) != {
            "warmup",
            "iterations",
            "repeats",
            "atol",
            "rtol",
        }:
            raise AutotuneCacheError("cached autotune configuration is invalid")
        try:
            return cls(**dict(value))
        except (TypeError, ValidationError) as exc:
            raise AutotuneCacheError(
                f"cached autotune configuration is invalid: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Correctness and timing outcome for one candidate."""

    candidate: AutotuneCandidate
    status: str
    rejection_reason: str | None
    max_abs_error: float | None
    max_rel_error: float | None
    latency_samples_ms: tuple[float, ...]
    median_latency_ms: float | None

    @property
    def candidate_sha256(self) -> str:
        return self.candidate.sha256

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "candidate_sha256": self.candidate_sha256,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "latency_samples_ms": list(self.latency_samples_ms),
            "median_latency_ms": self.median_latency_ms,
        }


@dataclass(frozen=True, slots=True)
class AutotuneResult:
    """Validated winner and the audit trail for the complete candidate space."""

    graph_sha256: str
    source_sha256: str
    device: Mapping[str, Any]
    device_sha256: str
    candidate_space_sha256: str
    config: AutotuneConfig
    selected_candidate: AutotuneCandidate
    evaluations: tuple[CandidateEvaluation, ...]
    cache_hit: bool = False

    @property
    def selected_candidate_sha256(self) -> str:
        return self.selected_candidate.sha256

    @property
    def median_latency_ms(self) -> float:
        for evaluation in self.evaluations:
            if evaluation.candidate_sha256 == self.selected_candidate_sha256:
                assert evaluation.median_latency_ms is not None
                return evaluation.median_latency_ms
        raise AssertionError("selected candidate is absent from evaluations")

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "kind": AUTOTUNE_CACHE_KIND,
            "schema_version": AUTOTUNE_CACHE_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "source_sha256": self.source_sha256,
            "device": dict(self.device),
            "device_sha256": self.device_sha256,
            "candidate_space_sha256": self.candidate_space_sha256,
            "config": self.config.to_dict(),
            "selected_candidate": self.selected_candidate.to_dict(),
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "evaluations": [item.to_dict() for item in self.evaluations],
        }


@runtime_checkable
class AutotuneRunner(Protocol):
    """Adapter around one already-open CUDA session.

    ``run`` returns host-visible NumPy-compatible output arrays. ``benchmark``
    performs warmup and returns exactly one elapsed-millisecond sample per
    repeat.  Both methods must execute in the session identified by
    ``session_identity``; the tuner verifies that identity and device metadata
    remain unchanged around every call.
    """

    def session_identity(self) -> str: ...

    def describe_device(self) -> Mapping[str, Any]: ...

    def run(self, candidate: AutotuneCandidate) -> Mapping[str, Any]: ...

    def benchmark(
        self,
        candidate: AutotuneCandidate,
        *,
        warmup: int,
        iterations: int,
        repeats: int,
    ) -> Sequence[float]: ...


def _normalize_json(value: object, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValidationError("device description is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValidationError("device description contains a non-finite number")
        return number
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValidationError(
                    "device description keys must be non-empty strings"
                )
            normalized[key] = _normalize_json(item, depth=depth + 1)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (tuple, list)):
        return [_normalize_json(item, depth=depth + 1) for item in value]
    raise ValidationError(
        f"device description contains unsupported {type(value).__name__}"
    )


def _normalize_device(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError("runner device description must be a mapping")
    normalized = _normalize_json(value)
    assert isinstance(normalized, dict)
    name = normalized.get("name")
    capability = normalized.get("compute_capability")
    if not isinstance(name, str) or not name:
        raise ValidationError("runner device description requires a non-empty name")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in capability
        )
        or any(item < 0 for item in capability)
    ):
        raise ValidationError(
            "runner device description requires compute_capability [major, minor]"
        )
    if len(_canonical_json_bytes(normalized)) > 64 * 1024:
        raise ValidationError("runner device description is too large")
    return normalized


def _session_state(runner: AutotuneRunner) -> tuple[str, dict[str, Any]]:
    identity = runner.session_identity()
    if not isinstance(identity, str) or not identity or len(identity) > 256:
        raise ValidationError("runner session_identity must be a non-empty string")
    device = _normalize_device(runner.describe_device())
    return identity, device


def _assert_same_session(
    runner: AutotuneRunner, identity: str, device: Mapping[str, Any]
) -> None:
    current_identity, current_device = _session_state(runner)
    if current_identity != identity:
        raise AutotuneError("autotune runner changed CUDA session during tuning")
    if current_device != device:
        raise AutotuneError("autotune runner changed CUDA device during tuning")


def _candidate_space(candidates: Sequence[AutotuneCandidate]) -> tuple[
    tuple[AutotuneCandidate, ...], str
]:
    if isinstance(candidates, (str, bytes)):
        raise ValidationError(
            "candidates must be a sequence of AutotuneCandidate values"
        )
    try:
        values = tuple(candidates)
    except TypeError as exc:
        raise ValidationError(
            "candidates must be an iterable of AutotuneCandidate values"
        ) from exc
    if not values:
        raise ValidationError("autotune requires at least one candidate")
    if len(values) > MAX_CANDIDATES:
        raise ValidationError(f"autotune supports at most {MAX_CANDIDATES} candidates")
    if any(not isinstance(candidate, AutotuneCandidate) for candidate in values):
        raise ValidationError("all candidates must be AutotuneCandidate values")
    ordered = tuple(sorted(values, key=lambda item: item.sha256))
    hashes = [candidate.sha256 for candidate in ordered]
    if len(set(hashes)) != len(hashes):
        raise ValidationError("autotune candidate space contains duplicates")
    digest = _sha256_json([candidate.to_dict() for candidate in ordered])
    return ordered, digest


def _reference_arrays(reference_outputs: object) -> dict[str, np.ndarray]:
    if not isinstance(reference_outputs, Mapping) or not reference_outputs:
        raise ValidationError("reference_outputs must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for name, value in reference_outputs.items():
        if not isinstance(name, str) or not name:
            raise ValidationError("reference output names must be non-empty strings")
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValidationError(
                f"reference output {name!r} is not NumPy-compatible"
            ) from exc
        if array.size == 0 or array.dtype.kind != "f":
            raise ValidationError(
                f"reference output {name!r} must be a non-empty floating array"
            )
        if not np.all(np.isfinite(array)):
            raise ValidationError(f"reference output {name!r} is not finite")
        converted = np.asarray(array, dtype=np.float64)
        if not np.all(np.isfinite(converted)):
            raise ValidationError(
                f"reference output {name!r} cannot be represented as float64"
            )
        result[name] = converted
    return result


def _check_outputs(
    actual_outputs: object,
    reference: Mapping[str, np.ndarray],
    *,
    config: AutotuneConfig,
) -> tuple[str | None, float | None, float | None]:
    if not isinstance(actual_outputs, Mapping):
        return "invalid-output", None, None
    try:
        names_match = set(actual_outputs) == set(reference)
    except Exception:
        return "invalid-output", None, None
    if not names_match:
        return "invalid-output", None, None
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for name in sorted(reference):
        try:
            actual = np.asarray(actual_outputs[name])
        except Exception:
            return "invalid-output", None, None
        expected = reference[name]
        if actual.shape != expected.shape or actual.dtype.kind != "f":
            return "invalid-output", None, None
        if not np.all(np.isfinite(actual)):
            return "invalid-output", None, None
        actual64 = np.asarray(actual, dtype=np.float64)
        if not np.all(np.isfinite(actual64)):
            return "invalid-output", None, None
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(actual64 - expected)
        absolute = float(np.max(difference))
        maximum_absolute = max(
            maximum_absolute,
            absolute if math.isfinite(absolute) else np.finfo(np.float64).max,
        )
        # A tolerance-sized floor keeps this diagnostic finite for a zero
        # reference.  Correctness itself is decided by np.allclose below.
        denominator_floor = max(config.atol, np.finfo(np.float64).eps)
        denominator = np.maximum(np.abs(expected), denominator_floor)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            relative = float(np.max(difference / denominator))
        maximum_relative = max(
            maximum_relative,
            relative if math.isfinite(relative) else np.finfo(np.float64).max,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            close = np.allclose(
                actual64,
                expected,
                atol=config.atol,
                rtol=config.rtol,
                equal_nan=False,
            )
        if not close:
            return "incorrect-output", maximum_absolute, maximum_relative
    return None, maximum_absolute, maximum_relative


def _rejected(
    candidate: AutotuneCandidate,
    reason: str,
    *,
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate=candidate,
        status="rejected",
        rejection_reason=reason,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        latency_samples_ms=(),
        median_latency_ms=None,
    )


def _timing_samples(value: object, repeats: int) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(raw) != repeats:
        return None
    samples: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
            return None
        sample = float(item)
        if not math.isfinite(sample) or sample <= 0.0:
            return None
        samples.append(sample)
    return tuple(samples)


def autotune_mlir_ptx_ffn(
    candidates: Sequence[AutotuneCandidate],
    runner: AutotuneRunner,
    reference_outputs: Mapping[str, Any],
    *,
    graph_sha256: str,
    source_sha256: str,
    config: AutotuneConfig | None = None,
    cache_path: str | os.PathLike[str] | None = None,
) -> AutotuneResult:
    """Correctness-gate, time, and deterministically select an FFN candidate.

    Candidates are evaluated in candidate-hash order.  The winner has the
    smallest median latency; an exact tie is broken by candidate SHA-256.  If a
    matching cache exists, it is validated and returned without executing the
    runner.  A mismatched or damaged cache raises instead of being overwritten.
    """

    graph_digest = _require_sha256("graph_sha256", graph_sha256)
    source_digest = _require_sha256("source_sha256", source_sha256)
    tuning_config = config if config is not None else AutotuneConfig()
    if not isinstance(tuning_config, AutotuneConfig):
        raise ValidationError("config must be an AutotuneConfig")
    ordered, space_digest = _candidate_space(candidates)
    reference = _reference_arrays(reference_outputs)
    identity, device = _session_state(runner)

    if cache_path is not None and Path(cache_path).exists():
        return load_autotune_cache(
            cache_path,
            graph_sha256=graph_digest,
            source_sha256=source_digest,
            device=device,
            candidates=ordered,
            config=tuning_config,
        )

    evaluations: list[CandidateEvaluation] = []
    for candidate in ordered:
        _assert_same_session(runner, identity, device)
        try:
            actual_outputs = runner.run(candidate)
        except Exception:
            _assert_same_session(runner, identity, device)
            evaluations.append(_rejected(candidate, "execution-error"))
            continue
        _assert_same_session(runner, identity, device)
        try:
            reason, maximum_absolute, maximum_relative = _check_outputs(
                actual_outputs, reference, config=tuning_config
            )
        except Exception:
            reason, maximum_absolute, maximum_relative = (
                "invalid-output",
                None,
                None,
            )
        if reason is not None:
            evaluations.append(
                _rejected(
                    candidate,
                    reason,
                    max_abs_error=maximum_absolute,
                    max_rel_error=maximum_relative,
                )
            )
            continue

        try:
            raw_samples = runner.benchmark(
                candidate,
                warmup=tuning_config.warmup,
                iterations=tuning_config.iterations,
                repeats=tuning_config.repeats,
            )
        except Exception:
            _assert_same_session(runner, identity, device)
            evaluations.append(
                _rejected(
                    candidate,
                    "benchmark-error",
                    max_abs_error=maximum_absolute,
                    max_rel_error=maximum_relative,
                )
            )
            continue
        _assert_same_session(runner, identity, device)
        samples = _timing_samples(raw_samples, tuning_config.repeats)
        if samples is None:
            evaluations.append(
                _rejected(
                    candidate,
                    "invalid-timing",
                    max_abs_error=maximum_absolute,
                    max_rel_error=maximum_relative,
                )
            )
            continue
        evaluations.append(
            CandidateEvaluation(
                candidate=candidate,
                status="eligible",
                rejection_reason=None,
                max_abs_error=maximum_absolute,
                max_rel_error=maximum_relative,
                latency_samples_ms=samples,
                median_latency_ms=float(statistics.median(samples)),
            )
        )

    eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
    if not eligible:
        reasons = sorted(
            {evaluation.rejection_reason or "unknown" for evaluation in evaluations}
        )
        raise AutotuneError(
            "no autotune candidate passed correctness and timing validation "
            f"(reasons: {', '.join(reasons)})"
        )
    winner = min(
        eligible,
        key=lambda item: (item.median_latency_ms, item.candidate_sha256),
    )
    result = AutotuneResult(
        graph_sha256=graph_digest,
        source_sha256=source_digest,
        device=device,
        device_sha256=_sha256_json(device),
        candidate_space_sha256=space_digest,
        config=tuning_config,
        selected_candidate=winner.candidate,
        evaluations=tuple(evaluations),
        cache_hit=False,
    )
    if cache_path is not None:
        write_autotune_cache(cache_path, result)
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise AutotuneCacheError("refusing to replace a symlink autotune cache")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        if os.name != "nt":
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        finally:
            raise


def write_autotune_cache(
    path: str | os.PathLike[str], result: AutotuneResult
) -> None:
    """Atomically publish a deterministic, self-checking tuning cache."""

    if not isinstance(result, AutotuneResult):
        raise ValidationError("result must be an AutotuneResult")
    payload = result.to_cache_dict()
    payload["payload_sha256"] = _sha256_json(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_CACHE_BYTES:
        raise AutotuneCacheError("autotune cache exceeds the size limit")
    _atomic_write(Path(path), encoded)


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise AutotuneCacheError(f"autotune cache repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AutotuneCacheError(f"autotune cache contains invalid number {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AutotuneCacheError("refusing to load a symlink autotune cache")
    try:
        stat = path.stat()
        if not path.is_file():
            raise AutotuneCacheError("autotune cache is not a regular file")
        if stat.st_size > MAX_CACHE_BYTES:
            raise AutotuneCacheError("autotune cache exceeds the size limit")
        raw = path.read_bytes()
    except AutotuneCacheError:
        raise
    except OSError as exc:
        raise AutotuneCacheError(f"could not read autotune cache: {exc}") from exc
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AutotuneCacheError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AutotuneCacheError(f"autotune cache is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AutotuneCacheError("autotune cache root must be an object")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutotuneCacheError(f"cached {name} is invalid")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AutotuneCacheError(f"cached {name} is invalid")
    return number


def _evaluation_from_dict(
    value: object, *, repeats: int
) -> CandidateEvaluation:
    expected_fields = {
        "candidate",
        "candidate_sha256",
        "status",
        "rejection_reason",
        "max_abs_error",
        "max_rel_error",
        "latency_samples_ms",
        "median_latency_ms",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise AutotuneCacheError("cached candidate evaluation has unexpected fields")
    candidate = AutotuneCandidate.from_dict(value["candidate"])
    if value["candidate_sha256"] != candidate.sha256:
        raise AutotuneCacheError("cached candidate digest does not match")
    status = value["status"]
    reason = value["rejection_reason"]
    if status == "eligible":
        if reason is not None:
            raise AutotuneCacheError("eligible cached candidate has a rejection reason")
        absolute = _finite_nonnegative(value["max_abs_error"], "max_abs_error")
        relative = _finite_nonnegative(value["max_rel_error"], "max_rel_error")
        samples = _timing_samples(value["latency_samples_ms"], repeats)
        if samples is None:
            raise AutotuneCacheError("cached latency samples are invalid")
        median = _finite_nonnegative(value["median_latency_ms"], "median latency")
        expected_median = float(statistics.median(samples))
        if median != expected_median:
            raise AutotuneCacheError("cached median does not match latency samples")
        return CandidateEvaluation(
            candidate, status, None, absolute, relative, samples, median
        )
    if (
        status != "rejected"
        or not isinstance(reason, str)
        or reason not in _REJECTION_REASONS
    ):
        raise AutotuneCacheError("cached candidate status is invalid")
    if value["latency_samples_ms"] != [] or value["median_latency_ms"] is not None:
        raise AutotuneCacheError("rejected cached candidate contains timing data")
    absolute_value = value["max_abs_error"]
    relative_value = value["max_rel_error"]
    absolute = (
        None
        if absolute_value is None
        else _finite_nonnegative(absolute_value, "max_abs_error")
    )
    relative = (
        None
        if relative_value is None
        else _finite_nonnegative(relative_value, "max_rel_error")
    )
    return CandidateEvaluation(candidate, status, reason, absolute, relative, (), None)


def load_autotune_cache(
    path: str | os.PathLike[str],
    *,
    graph_sha256: str,
    source_sha256: str,
    device: Mapping[str, Any],
    candidates: Sequence[AutotuneCandidate],
    config: AutotuneConfig | None = None,
) -> AutotuneResult:
    """Load a cache only when every workload and environment binding matches."""

    expected_graph = _require_sha256("graph_sha256", graph_sha256)
    expected_source = _require_sha256("source_sha256", source_sha256)
    expected_device = _normalize_device(device)
    expected_candidates, expected_space = _candidate_space(candidates)
    expected_config = config if config is not None else AutotuneConfig()
    if not isinstance(expected_config, AutotuneConfig):
        raise ValidationError("config must be an AutotuneConfig")

    value = _load_json(Path(path))
    expected_fields = {
        "kind",
        "schema_version",
        "graph_sha256",
        "source_sha256",
        "device",
        "device_sha256",
        "candidate_space_sha256",
        "config",
        "selected_candidate",
        "selected_candidate_sha256",
        "evaluations",
        "payload_sha256",
    }
    if set(value) != expected_fields:
        raise AutotuneCacheError("autotune cache has unexpected fields")
    payload_digest = value["payload_sha256"]
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    if (
        not isinstance(payload_digest, str)
        or _HASH_RE.fullmatch(payload_digest) is None
        or payload_digest != _sha256_json(unsigned)
    ):
        raise AutotuneCacheError("autotune cache payload digest does not match")
    if value["kind"] != AUTOTUNE_CACHE_KIND:
        raise AutotuneCacheError("autotune cache kind does not match")
    if (
        isinstance(value["schema_version"], bool)
        or not isinstance(value["schema_version"], int)
        or value["schema_version"] != AUTOTUNE_CACHE_SCHEMA_VERSION
    ):
        raise AutotuneCacheError("autotune cache schema version does not match")
    if value["graph_sha256"] != expected_graph:
        raise AutotuneCacheError("autotune cache graph digest does not match")
    if value["source_sha256"] != expected_source:
        raise AutotuneCacheError("autotune cache source digest does not match")
    try:
        cached_device = _normalize_device(value["device"])
    except ValidationError as exc:
        raise AutotuneCacheError(
            f"cached device description is invalid: {exc}"
        ) from exc
    if cached_device != expected_device:
        raise AutotuneCacheError("autotune cache device does not match")
    if value["device_sha256"] != _sha256_json(cached_device):
        raise AutotuneCacheError("autotune cache device digest does not match")
    if value["candidate_space_sha256"] != expected_space:
        raise AutotuneCacheError("autotune cache candidate space does not match")
    cached_config = AutotuneConfig.from_dict(value["config"])
    if cached_config != expected_config:
        raise AutotuneCacheError(
            "autotune cache measurement configuration does not match"
        )
    raw_evaluations = value["evaluations"]
    if not isinstance(raw_evaluations, list):
        raise AutotuneCacheError("autotune cache evaluations must be a list")
    evaluations = tuple(
        _evaluation_from_dict(item, repeats=expected_config.repeats)
        for item in raw_evaluations
    )
    expected_hashes = [candidate.sha256 for candidate in expected_candidates]
    actual_hashes = [evaluation.candidate_sha256 for evaluation in evaluations]
    if actual_hashes != expected_hashes:
        raise AutotuneCacheError(
            "autotune cache evaluations do not cover the candidate space exactly"
        )
    selected = AutotuneCandidate.from_dict(value["selected_candidate"])
    if value["selected_candidate_sha256"] != selected.sha256:
        raise AutotuneCacheError(
            "autotune cache selected candidate digest does not match"
        )
    eligible = [evaluation for evaluation in evaluations if evaluation.eligible]
    if not eligible:
        raise AutotuneCacheError("autotune cache has no eligible candidate")
    winner = min(
        eligible,
        key=lambda item: (item.median_latency_ms, item.candidate_sha256),
    )
    if selected.sha256 != winner.candidate_sha256:
        raise AutotuneCacheError("autotune cache selected candidate is not the winner")
    return AutotuneResult(
        graph_sha256=expected_graph,
        source_sha256=expected_source,
        device=cached_device,
        device_sha256=value["device_sha256"],
        candidate_space_sha256=expected_space,
        config=cached_config,
        selected_candidate=selected,
        evaluations=evaluations,
        cache_hit=True,
    )


__all__ = [
    "AUTOTUNE_CACHE_KIND",
    "AUTOTUNE_CACHE_SCHEMA_VERSION",
    "SUPPORTED_FUSION_STRATEGIES",
    "SUPPORTED_PRECISIONS",
    "AutotuneCacheError",
    "AutotuneCandidate",
    "AutotuneConfig",
    "AutotuneError",
    "AutotuneResult",
    "AutotuneRunner",
    "CandidateEvaluation",
    "autotune_mlir_ptx_ffn",
    "load_autotune_cache",
    "write_autotune_cache",
]
