"""Deterministic stage-wise golden bundles and numerical comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from .errors import ValidationError


_EPOCH = (1980, 1, 1, 0, 0, 0)


def _entry(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=_EPOCH)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _stage_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError("golden stage names must be non-empty strings")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(not part for part in path.parts):
        raise ValidationError(f"unsafe golden stage name {name!r}")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/" for character in name):
        raise ValidationError(f"golden stage name contains unsupported characters: {name!r}")
    return name


def _array_bytes(value: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


@dataclass(frozen=True, slots=True)
class TensorTolerance:
    atol: float = 1e-5
    rtol: float = 1e-4
    cosine_min: float = 0.99999

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.atol, self.rtol, self.cosine_min)):
            raise ValidationError("golden tolerances must be finite")
        if self.atol < 0 or self.rtol < 0 or not -1 <= self.cosine_min <= 1:
            raise ValidationError("invalid golden tolerance range")


@dataclass(frozen=True, slots=True)
class TensorComparison:
    shape: tuple[int, ...]
    dtype: str
    max_abs: float
    max_rel: float
    cosine_similarity: float
    close_fraction: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "cosine_similarity": self.cosine_similarity,
            "close_fraction": self.close_fraction,
            "passed": self.passed,
        }


def compare_tensor(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: TensorTolerance | None = None,
) -> TensorComparison:
    tolerance = tolerance or TensorTolerance()
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValidationError(
            f"golden shape mismatch: actual {actual.shape}, expected {expected.shape}"
        )
    if not np.issubdtype(actual.dtype, np.number) or not np.issubdtype(expected.dtype, np.number):
        raise ValidationError("golden comparison requires numeric arrays")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValidationError("golden comparison requires finite arrays")

    left = actual.astype(np.float64, copy=False).reshape(-1)
    right = expected.astype(np.float64, copy=False).reshape(-1)
    if left.size == 0:
        max_abs = max_rel = 0.0
        cosine = 1.0
        close_fraction = 1.0
    else:
        difference = np.abs(left - right)
        max_abs = float(np.max(difference))
        denominator = np.maximum(np.abs(right), np.finfo(np.float64).tiny)
        max_rel = float(np.max(difference / denominator))
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 and right_norm == 0.0:
            cosine = 1.0
        elif left_norm == 0.0 or right_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(np.dot(left, right) / (left_norm * right_norm))
            cosine = max(-1.0, min(1.0, cosine))
        close = difference <= tolerance.atol + tolerance.rtol * np.abs(right)
        close_fraction = float(np.mean(close))
    passed = close_fraction == 1.0 and cosine >= tolerance.cosine_min
    return TensorComparison(
        tuple(actual.shape), actual.dtype.name, max_abs, max_rel, cosine,
        close_fraction, passed,
    )


@dataclass(frozen=True, slots=True)
class GoldenBundle:
    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        normalized: dict[str, np.ndarray] = {}
        for name, value in self.tensors.items():
            name = _stage_name(name)
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValidationError(f"golden tensor {name!r} cannot have object dtype")
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValidationError(f"golden tensor {name!r} must contain finite values")
            normalized[name] = array.copy()
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("golden metadata must be finite JSON data") from exc
        object.__setattr__(self, "tensors", normalized)
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata)))


def write_golden(path: str | Path, bundle: GoldenBundle) -> None:
    destination = Path(path)
    if destination.suffix != ".gacgolden":
        raise ValidationError("golden bundle path must end in .gacgolden")
    destination.parent.mkdir(parents=True, exist_ok=True)
    array_entries: dict[str, bytes] = {}
    tensor_manifest: dict[str, Any] = {}
    for name in sorted(bundle.tensors):
        payload = _array_bytes(bundle.tensors[name])
        entry_name = f"tensors/{name}.npy"
        array_entries[entry_name] = payload
        tensor_manifest[name] = {
            "entry": entry_name,
            "shape": list(bundle.tensors[name].shape),
            "dtype": bundle.tensors[name].dtype.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "kind": "gac-stage-golden",
        "metadata": bundle.metadata,
        "tensors": tensor_manifest,
    }
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with ZipFile(destination, "w") as archive:
        archive.writestr(_entry("manifest.json"), manifest_bytes)
        for name, payload in sorted(array_entries.items()):
            archive.writestr(_entry(name), payload)


def read_golden(path: str | Path, *, max_tensor_bytes: int = 2**31) -> GoldenBundle:
    try:
        with ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise ValidationError("golden bundle has duplicate entries or no manifest")
            if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
                raise ValidationError("golden bundle contains an unsafe path")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema_version") != 1 or manifest.get("kind") != "gac-stage-golden":
                raise ValidationError("unsupported golden bundle schema")
            tensors: dict[str, np.ndarray] = {}
            expected_entries = {"manifest.json"}
            for raw_name, record in manifest.get("tensors", {}).items():
                name = _stage_name(raw_name)
                entry_name = record["entry"]
                if entry_name != f"tensors/{name}.npy":
                    raise ValidationError(f"golden tensor {name!r} has an invalid entry")
                expected_entries.add(entry_name)
                info = archive.getinfo(entry_name)
                if info.file_size > max_tensor_bytes:
                    raise ValidationError(f"golden tensor {name!r} exceeds the size limit")
                payload = archive.read(entry_name)
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise ValidationError(f"golden tensor {name!r} failed checksum validation")
                array = np.load(BytesIO(payload), allow_pickle=False)
                if list(array.shape) != record["shape"] or array.dtype.name != record["dtype"]:
                    raise ValidationError(f"golden tensor {name!r} disagrees with its manifest")
                tensors[name] = array
            if set(names) != expected_entries:
                raise ValidationError("golden bundle contains undeclared entries")
            return GoldenBundle(tensors, dict(manifest.get("metadata", {})))
    except ValidationError:
        raise
    except (BadZipFile, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read golden bundle: {exc}") from exc


def compare_bundle(
    actual: Mapping[str, np.ndarray],
    golden: GoldenBundle,
    tolerances: Mapping[str, TensorTolerance] | None = None,
) -> dict[str, TensorComparison]:
    if set(actual) != set(golden.tensors):
        raise ValidationError(
            f"golden stages must be {sorted(golden.tensors)}, got {sorted(actual)}"
        )
    tolerances = tolerances or {}
    unknown = set(tolerances) - set(golden.tensors)
    if unknown:
        raise ValidationError(f"tolerances name unknown stages: {sorted(unknown)}")
    return {
        name: compare_tensor(actual[name], golden.tensors[name], tolerances.get(name))
        for name in sorted(golden.tensors)
    }


__all__ = [
    "GoldenBundle",
    "TensorComparison",
    "TensorTolerance",
    "compare_bundle",
    "compare_tensor",
    "read_golden",
    "write_golden",
]
