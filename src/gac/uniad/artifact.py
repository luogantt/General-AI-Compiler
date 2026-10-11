"""Safe, deterministic bundle I/O for the UniAD FFN compiler boundary.

A bundle is one ZIP file with exactly these members::

    contract.json
    weights.npz
    sample.npz        # optional: query plus expected output
    manifest.json

The manifest checks every payload for byte-level integrity and consistency by
size and SHA256.  It is not a signature or proof of provenance.  Loading is
data-only: JSON rejects duplicate keys and non-finite constants, NumPy archives
are opened with ``allow_pickle=False``, and all tensor names, shapes, dtypes and
values are checked against :class:`UniADFFNContract`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple, Union
import zipfile

import numpy as np

from .contract import (
    CANONICAL_WEIGHT_NAMES,
    SAMPLE_NAMES,
    ContractError,
    UniADFFNContract,
)


BUNDLE_SCHEMA_VERSION = 1
BUNDLE_FORMAT = "uniad-compiler-ffn-bundle"

_CONTRACT_MEMBER = "contract.json"
_WEIGHTS_MEMBER = "weights.npz"
_SAMPLE_MEMBER = "sample.npz"
_MANIFEST_MEMBER = "manifest.json"
_REQUIRED_PAYLOADS = frozenset({_CONTRACT_MEMBER, _WEIGHTS_MEMBER})
_ALLOWED_PAYLOADS = frozenset(
    {_CONTRACT_MEMBER, _WEIGHTS_MEMBER, _SAMPLE_MEMBER}
)
_MAX_JSON_BYTES = 1024 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
_SHA256_HEX = frozenset("0123456789abcdef")


class ArtifactError(ValueError):
    """Raised when a UniAD compiler bundle violates its data contract."""


@dataclass(frozen=True)
class UniADFFNBundle:
    """Verified in-memory contents of one compiler bundle."""

    contract: UniADFFNContract
    weights: Mapping[str, np.ndarray]
    sample: Optional[Mapping[str, np.ndarray]]
    manifest: Mapping[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("value is not canonical JSON: {}".format(exc)) from exc
    return encoded + b"\n"


def _reject_json_constant(token: str) -> None:
    raise ArtifactError("JSON contains non-finite constant {!r}".format(token))


def _object_without_duplicates(pairs: list) -> Dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ArtifactError("JSON contains duplicate key {!r}".format(name))
        result[name] = value
    return result


def _parse_json(data: bytes, member: str) -> Dict[str, Any]:
    if len(data) > _MAX_JSON_BYTES:
        raise ArtifactError("{} exceeds the JSON size limit".format(member))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError("{} is not UTF-8".format(member)) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except ArtifactError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactError("{} is not valid JSON: {}".format(member, exc)) from exc
    if not isinstance(value, dict):
        raise ArtifactError("{} must contain a JSON object".format(member))
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_record(data: bytes) -> Dict[str, Any]:
    return {"sha256": _sha256(data), "size_bytes": len(data)}


def _validated_arrays(
    values: Mapping[str, np.ndarray],
    expected_shapes: Mapping[str, Tuple[int, ...]],
    label: str,
) -> Dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise ArtifactError("{} must be a tensor mapping".format(label))
    actual_names = set(values)
    expected_names = set(expected_shapes)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names, key=repr)
        raise ArtifactError(
            "{} keys differ from contract; missing={}, extra={}".format(
                label, missing, extra
            )
        )
    result = {}
    for name in expected_shapes:
        value = values[name]
        if not isinstance(value, np.ndarray):
            raise ArtifactError("{} {!r} must be a numpy array".format(label, name))
        expected_shape = expected_shapes[name]
        if value.shape != expected_shape or value.dtype != np.float32:
            raise ArtifactError(
                "{} {!r} requires {}/float32, got {}/{}".format(
                    label, name, expected_shape, value.shape, value.dtype
                )
            )
        if not value.flags.c_contiguous:
            raise ArtifactError("{} {!r} must be C-contiguous".format(label, name))
        if not np.isfinite(value).all():
            raise ArtifactError("{} {!r} must be finite".format(label, name))
        copied = np.array(value, dtype=np.float32, order="C", copy=True)
        copied.setflags(write=False)
        result[name] = copied
    return result


def _npz_bytes(values: Mapping[str, np.ndarray], names: Tuple[str, ...]) -> bytes:
    output = BytesIO()
    ordered = {name: values[name] for name in names}
    np.savez(output, **ordered)
    return output.getvalue()


def _inspect_npz(data: bytes, expected_names: Tuple[str, ...], label: str) -> None:
    expected_members = {"{}.npy".format(name) for name in expected_names}
    try:
        with zipfile.ZipFile(BytesIO(data), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactError("{} contains duplicate members".format(label))
            if set(names) != expected_members:
                raise ArtifactError(
                    "{} members differ from contract".format(label)
                )
            total = 0
            for info in infos:
                if info.is_dir() or info.flag_bits & 0x1:
                    raise ArtifactError(
                        "{} contains an unsupported member".format(label)
                    )
                if info.compress_type not in (
                    zipfile.ZIP_STORED,
                    zipfile.ZIP_DEFLATED,
                ):
                    raise ArtifactError(
                        "{} uses unsupported compression".format(label)
                    )
                total += info.file_size
                if total > _MAX_NPZ_BYTES:
                    raise ArtifactError("{} exceeds the size limit".format(label))
    except ArtifactError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ArtifactError("{} is not a valid NPZ archive".format(label)) from exc


def _load_npz(
    data: bytes,
    expected_shapes: Mapping[str, Tuple[int, ...]],
    names: Tuple[str, ...],
    label: str,
) -> Dict[str, np.ndarray]:
    if len(data) > _MAX_NPZ_BYTES:
        raise ArtifactError("{} exceeds the size limit".format(label))
    _inspect_npz(data, names, label)
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if set(archive.files) != set(names):
                raise ArtifactError("{} tensor keys differ from the schema".format(label))
            loaded = {name: archive[name] for name in names}
    except ArtifactError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArtifactError("cannot decode {}: {}".format(label, exc)) from exc
    return _validated_arrays(loaded, expected_shapes, label)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _validate_manifest(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    if set(manifest) != {"schema_version", "format", "files"}:
        raise ArtifactError("manifest fields differ from the schema")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION
    ):
        raise ArtifactError("unsupported bundle schema version")
    if manifest["format"] != BUNDLE_FORMAT:
        raise ArtifactError("unsupported bundle format")
    records = manifest["files"]
    if not isinstance(records, dict) or set(records) != set(payloads):
        raise ArtifactError("manifest payload set differs from the archive")
    for name, data in payloads.items():
        record = records[name]
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
            raise ArtifactError("invalid manifest record for {!r}".format(name))
        size = record["size_bytes"]
        digest = record["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactError("invalid payload size for {!r}".format(name))
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _SHA256_HEX for character in digest)
        ):
            raise ArtifactError("invalid SHA256 for {!r}".format(name))
        if size != len(data):
            raise ArtifactError("payload {!r} has the wrong size".format(name))
        if digest != _sha256(data):
            raise ArtifactError("payload {!r} failed SHA256 verification".format(name))


def _read_archive(path: Path) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    try:
        with zipfile.ZipFile(str(path), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ArtifactError("bundle contains duplicate members")
            allowed = set(_ALLOWED_PAYLOADS) | {_MANIFEST_MEMBER}
            if not _REQUIRED_PAYLOADS.issubset(names) or set(names) - allowed:
                raise ArtifactError("bundle member set differs from the schema")
            if _MANIFEST_MEMBER not in names:
                raise ArtifactError("bundle has no manifest.json")
            total = 0
            info_by_name = {}
            for info in infos:
                if info.is_dir() or info.flag_bits & 0x1:
                    raise ArtifactError("bundle contains an unsupported member")
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ArtifactError("bundle members must use stored compression")
                limit = (
                    _MAX_JSON_BYTES
                    if info.filename.endswith(".json")
                    else _MAX_NPZ_BYTES
                )
                if info.file_size > limit:
                    raise ArtifactError(
                        "bundle member {!r} exceeds its size limit".format(
                            info.filename
                        )
                    )
                total += info.file_size
                if total > _MAX_TOTAL_BYTES:
                    raise ArtifactError("bundle exceeds the total size limit")
                info_by_name[info.filename] = info
            members = {
                name: archive.read(info_by_name[name])
                for name in names
                if name != _MANIFEST_MEMBER
            }
            manifest_data = archive.read(info_by_name[_MANIFEST_MEMBER])
    except ArtifactError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ArtifactError("cannot open bundle {!s}: {}".format(path, exc)) from exc
    manifest = _parse_json(manifest_data, _MANIFEST_MEMBER)
    return members, manifest


def load_bundle(path: Union[str, Path]) -> UniADFFNBundle:
    """Load and fully verify a data-only UniAD FFN compiler bundle."""

    source = Path(path)
    payloads, manifest = _read_archive(source)
    _validate_manifest(manifest, payloads)
    contract_mapping = _parse_json(payloads[_CONTRACT_MEMBER], _CONTRACT_MEMBER)
    try:
        contract = UniADFFNContract.from_dict(contract_mapping)
    except ContractError as exc:
        raise ArtifactError("invalid contract.json: {}".format(exc)) from exc
    weights = _load_npz(
        payloads[_WEIGHTS_MEMBER],
        contract.weight_shapes,
        CANONICAL_WEIGHT_NAMES,
        _WEIGHTS_MEMBER,
    )
    sample = None
    if _SAMPLE_MEMBER in payloads:
        sample = _load_npz(
            payloads[_SAMPLE_MEMBER],
            contract.sample_shapes,
            SAMPLE_NAMES,
            _SAMPLE_MEMBER,
        )
    frozen_manifest = MappingProxyType(
        {
            "schema_version": manifest["schema_version"],
            "format": manifest["format"],
            "files": MappingProxyType(
                {
                    name: MappingProxyType(dict(record))
                    for name, record in manifest["files"].items()
                }
            ),
        }
    )
    return UniADFFNBundle(
        contract=contract,
        weights=MappingProxyType(weights),
        sample=None if sample is None else MappingProxyType(sample),
        manifest=frozen_manifest,
    )


def write_bundle(
    path: Union[str, Path],
    contract: UniADFFNContract,
    weights: Mapping[str, np.ndarray],
    sample: Optional[Mapping[str, np.ndarray]] = None,
) -> UniADFFNBundle:
    """Atomically write a bundle, reload it, and return verified contents."""

    if not isinstance(contract, UniADFFNContract):
        raise ArtifactError("contract must be a UniADFFNContract")
    checked_weights = _validated_arrays(weights, contract.weight_shapes, "weights")
    checked_sample = None
    if sample is not None:
        checked_sample = _validated_arrays(sample, contract.sample_shapes, "sample")

    payloads = {
        _CONTRACT_MEMBER: _canonical_json(contract.to_dict()),
        _WEIGHTS_MEMBER: _npz_bytes(checked_weights, CANONICAL_WEIGHT_NAMES),
    }
    if checked_sample is not None:
        payloads[_SAMPLE_MEMBER] = _npz_bytes(checked_sample, SAMPLE_NAMES)
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "format": BUNDLE_FORMAT,
        "files": {name: _file_record(data) for name, data in payloads.items()},
    }
    manifest_data = _canonical_json(manifest)

    destination = Path(path)
    temporary_name = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".{}-".format(destination.name),
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, mode="w") as archive:
            for name in (_CONTRACT_MEMBER, _WEIGHTS_MEMBER, _SAMPLE_MEMBER):
                if name in payloads:
                    archive.writestr(_zip_info(name), payloads[name])
            archive.writestr(_zip_info(_MANIFEST_MEMBER), manifest_data)
        load_bundle(temporary_name)
        os.replace(temporary_name, str(destination))
        temporary_name = None
        # Exercise the published path as part of the write contract.  The
        # temporary archive was already verified before it atomically replaced
        # the destination, so this second load also catches path-level races.
        return load_bundle(destination)
    except ArtifactError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArtifactError("cannot write bundle {!s}: {}".format(destination, exc)) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


__all__ = [
    "ArtifactError",
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "UniADFFNBundle",
    "load_bundle",
    "write_bundle",
]
