"""Safe, deterministic interchange files for framework-independent weights.

``.gacweights`` is a data-only, uncompressed ZIP format.  It contains one
``manifest.json`` and one NumPy ``.npy`` member per named tensor.  Readers
verify the complete member set, sizes, SHA256 digests, NPY headers and resource
limits before returning immutable arrays.  Pickled/object arrays are never
written or loaded.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
import hmac
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np

from gac.core.errors import ValidationError


WEIGHTS_SCHEMA_VERSION = 1
WEIGHTS_FORMAT = "gacweights"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_MAX_WEIGHT_COUNT = 100_000
_ALLOWED_DTYPE_KINDS = frozenset("biufc")


@dataclass(frozen=True, slots=True)
class LoadedWeights(Mapping[str, np.ndarray]):
    """Verified read-only weights plus their checkpoint provenance."""

    weights: Mapping[str, np.ndarray]
    source_checkpoint_sha256: str
    manifest: Mapping[str, Any]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.weights[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.weights)

    def __len__(self) -> int:
        return len(self.weights)


def _normalize_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{field} must be a 64-character hexadecimal SHA256")
    return value.lower()


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"weights manifest is not canonical JSON: {exc}") from exc


def _reject_json_constant(token: str) -> None:
    raise ValidationError(f"non-finite JSON number {token!r} is not allowed")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_manifest(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("manifest.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("weights manifest root must be an object")
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name or ":" in name or "\x00" in name or name.startswith("/"):
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _array_bytes(value: np.ndarray, name: str) -> tuple[np.ndarray, bytes]:
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"weight {name!r} must be a NumPy ndarray")
    if value.dtype.hasobject:
        raise ValidationError(f"weight {name!r} has an object dtype")
    if value.dtype.fields is not None or value.dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValidationError(f"weight {name!r} has unsupported dtype {value.dtype}")
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValidationError(f"weight {name!r} contains non-finite values")
    contiguous = np.ascontiguousarray(value)
    stream = io.BytesIO()
    np.save(stream, contiguous, allow_pickle=False)
    return contiguous, stream.getvalue()


def _read_npy_header(data: bytes, name: str) -> tuple[tuple[int, ...], np.dtype[Any], int]:
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValidationError(f"weight {name!r} uses unsupported NPY version {version}")
    except ValidationError:
        raise
    except (EOFError, SyntaxError, TypeError, ValueError) as exc:
        raise ValidationError(f"weight {name!r} has an invalid NPY header") from exc
    if fortran_order:
        raise ValidationError(f"weight {name!r} must use C-contiguous NPY storage")
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValidationError(f"weight {name!r} has an unsafe or unsupported NPY dtype")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape):
        raise ValidationError(f"weight {name!r} has an invalid NPY shape")
    raw_size = (math.prod(shape) if shape else 1) * dtype.itemsize
    if len(data) - stream.tell() != raw_size:
        raise ValidationError(f"weight {name!r} NPY payload length is inconsistent")
    return shape, dtype, stream.tell()


def _load_array(data: bytes, name: str, record: Mapping[str, Any]) -> np.ndarray:
    expected_shape = record.get("shape")
    expected_dtype = record.get("dtype")
    if not isinstance(expected_shape, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in expected_shape
    ):
        raise ValidationError(f"weight {name!r} has an invalid manifest shape")
    if not isinstance(expected_dtype, str):
        raise ValidationError(f"weight {name!r} has an invalid manifest dtype")
    try:
        declared_dtype = np.dtype(expected_dtype)
    except TypeError as exc:
        raise ValidationError(f"weight {name!r} has an invalid manifest dtype") from exc
    if declared_dtype.hasobject or declared_dtype.fields is not None or declared_dtype.kind not in _ALLOWED_DTYPE_KINDS:
        raise ValidationError(f"weight {name!r} has an unsafe or unsupported manifest dtype")
    expected_raw_size = (math.prod(expected_shape) if expected_shape else 1) * declared_dtype.itemsize
    if expected_raw_size > len(data):
        raise ValidationError(f"weight {name!r} declares more data than its member contains")
    shape, dtype, _ = _read_npy_header(data, name)
    if shape != tuple(expected_shape) or dtype.str != declared_dtype.str:
        raise ValidationError(f"weight {name!r} NPY header does not match its manifest")
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValidationError(f"weight {name!r} is not a safe NumPy array") from exc
    if not isinstance(value, np.ndarray):
        raise ValidationError(f"weight {name!r} did not decode to an ndarray")
    value = np.ascontiguousarray(value)
    if value.dtype.kind in "fc" and not np.isfinite(value).all():
        raise ValidationError(f"weight {name!r} contains non-finite values")
    # Back with immutable ``bytes`` so callers cannot re-enable WRITEABLE via
    # ``setflags`` on an owning NumPy allocation.
    frozen = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
    frozen.setflags(write=False)
    return frozen


def write_weights(
    path: str | os.PathLike[str],
    weights: Mapping[str, np.ndarray],
    *,
    source_checkpoint_sha256: str,
) -> Path:
    """Write named NumPy weights as a deterministic ``.gacweights`` ZIP."""

    digest = _normalize_digest(source_checkpoint_sha256, "source_checkpoint_sha256")
    if not isinstance(weights, Mapping):
        raise ValidationError("weights must be a mapping of names to NumPy arrays")
    names = list(weights)
    if not names:
        raise ValidationError("at least one weight is required")
    if any(not isinstance(name, str) or not name or "\x00" in name for name in names):
        raise ValidationError("weight names must be non-empty NUL-free strings")
    if len(set(names)) != len(names):
        raise ValidationError("weights mapping yielded a duplicate name")

    payloads: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(names)):
        array, data = _array_bytes(weights[name], name)
        member = f"weights/{index:06d}.npy"
        member_digest = sha256(data).hexdigest()
        payloads[member] = data
        records[name] = {
            "path": member,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "size": len(data),
            "sha256": member_digest,
        }

    manifest = {
        "schema_version": WEIGHTS_SCHEMA_VERSION,
        "format": WEIGHTS_FORMAT,
        "source_checkpoint_sha256": digest,
        "weights": records,
    }
    manifest_bytes = _canonical_json(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(_zip_info("manifest.json"), manifest_bytes)
            for member in sorted(payloads):
                archive.writestr(_zip_info(member), payloads[member])
        os.replace(temporary_name, destination)
        temporary_name = None
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot write weights file {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def _inspect_archive(
    archive: zipfile.ZipFile,
    *,
    max_uncompressed_bytes: int,
    max_weight_count: int,
) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    total = 0
    all_infos = archive.infolist()
    if len(all_infos) > max_weight_count + 1:
        raise ValidationError("weights archive contains too many members")
    for info in all_infos:
        if info.filename in infos:
            raise ValidationError(f"weights archive contains duplicate member {info.filename!r}")
        if not _safe_member_name(info.filename):
            raise ValidationError(f"weights archive contains unsafe member path {info.filename!r}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if info.is_dir() or stat.S_ISLNK(mode):
            raise ValidationError(f"weights archive member {info.filename!r} is not a regular file")
        if info.flag_bits & 0x1:
            raise ValidationError("encrypted weights members are not supported")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValidationError(f"weights member {info.filename!r} uses unsupported compression")
        total += info.file_size
        if info.file_size < 0 or total > max_uncompressed_bytes:
            raise ValidationError("weights archive exceeds the uncompressed-size limit")
        infos[info.filename] = info
    return infos


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        return archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot read weights member {info.filename!r}") from exc


def load_weights(
    path: str | os.PathLike[str],
    *,
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES,
    max_weight_count: int = _DEFAULT_MAX_WEIGHT_COUNT,
) -> LoadedWeights:
    """Verify and load immutable arrays without extracting archive members."""

    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes <= 0
    ):
        raise ValidationError("max_uncompressed_bytes must be a positive integer")
    if isinstance(max_weight_count, bool) or not isinstance(max_weight_count, int) or max_weight_count <= 0:
        raise ValidationError("max_weight_count must be a positive integer")
    source = Path(path)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            infos = _inspect_archive(
                archive,
                max_uncompressed_bytes=max_uncompressed_bytes,
                max_weight_count=max_weight_count,
            )
            manifest_info = infos.get("manifest.json")
            if manifest_info is None:
                raise ValidationError("weights archive has no manifest.json")
            if manifest_info.file_size > _MAX_MANIFEST_BYTES:
                raise ValidationError("weights manifest is too large")
            manifest = _parse_manifest(_read_member(archive, manifest_info))

            if manifest.get("schema_version") != WEIGHTS_SCHEMA_VERSION:
                raise ValidationError(
                    f"unsupported weights schema {manifest.get('schema_version')!r}; "
                    f"expected {WEIGHTS_SCHEMA_VERSION}"
                )
            if manifest.get("format") != WEIGHTS_FORMAT:
                raise ValidationError(f"unsupported weights format {manifest.get('format')!r}")
            if set(manifest) != {
                "schema_version",
                "format",
                "source_checkpoint_sha256",
                "weights",
            }:
                raise ValidationError("weights manifest fields do not match the schema")
            source_digest = _normalize_digest(
                manifest.get("source_checkpoint_sha256"), "source_checkpoint_sha256"
            )
            records = manifest.get("weights")
            if not isinstance(records, dict) or not records:
                raise ValidationError("weights manifest must contain at least one named weight")
            if len(records) > max_weight_count:
                raise ValidationError("weights manifest exceeds the weight-count limit")

            paths: set[str] = set()
            for name, record in records.items():
                if not isinstance(name, str) or not name or "\x00" in name or not isinstance(record, dict):
                    raise ValidationError("weights manifest contains an invalid weight record")
                if set(record) != {"path", "shape", "dtype", "size", "sha256"}:
                    raise ValidationError(f"weight {name!r} has unexpected manifest fields")
                member = record.get("path")
                if (
                    not isinstance(member, str)
                    or not member.startswith("weights/")
                    or not member.endswith(".npy")
                    or not _safe_member_name(member)
                ):
                    raise ValidationError(f"weight {name!r} has an invalid member path")
                if member in paths:
                    raise ValidationError(f"weight member {member!r} is referenced more than once")
                paths.add(member)
                size = record.get("size")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValidationError(f"weight {name!r} has an invalid member size")
                record_digest = record.get("sha256")
                if not isinstance(record_digest, str) or _SHA256_RE.fullmatch(record_digest) is None:
                    raise ValidationError(f"weight {name!r} has an invalid SHA256")

            expected_members = {"manifest.json", *paths}
            if set(infos) != expected_members:
                missing = sorted(expected_members - set(infos))
                extra = sorted(set(infos) - expected_members)
                raise ValidationError(
                    f"weights member set differs from manifest; missing={missing}, extra={extra}"
                )

            arrays: dict[str, np.ndarray] = {}
            for name in sorted(records):
                record = records[name]
                member = record["path"]
                info = infos[member]
                if info.file_size != record["size"]:
                    raise ValidationError(f"weight {name!r} has the wrong member size")
                data = _read_member(archive, info)
                actual_digest = sha256(data).hexdigest()
                if not hmac.compare_digest(actual_digest, record["sha256"].lower()):
                    raise ValidationError(f"weight {name!r} failed SHA256 verification")
                arrays[name] = _load_array(data, name, record)
    except ValidationError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot open weights archive {source}: {exc}") from exc

    return LoadedWeights(
        weights=MappingProxyType(arrays),
        source_checkpoint_sha256=source_digest,
        manifest=_deep_freeze(manifest),
    )
