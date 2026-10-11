"""Portable directory format for the BEVFormer MLIR-generated PTX slice.

The deployment artifact intentionally contains device code and provenance, but
not tensor-level MLIR: the current tensor emitter inlines model constants and
would otherwise copy checkpoint weights into a diagnostic text file.  Weights
remain in the caller's validated :class:`~gac.ir.Graph` and are bound as device
buffers by :class:`~gac.mlir_ptx_runtime.MlirPtxSession`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .backends.mlir_ptx import (
    PTX_ENTRY_SYMBOLS,
    TARGET,
    MlirPtxProgram,
    assemble_mlir_ptx_program,
    validate_mlir_ptx_program,
)
from .errors import BackendError
from .ir import Graph
from .mlir_toolchain import MlirPtxArtifact


ARTIFACT_KIND = "mlir-nvvm-ptx-directory"
ARTIFACT_FILES = {
    "gpu_mlir": "gpu.mlir",
    "binary_mlir": "binary.mlir",
    "ptx": "kernels.ptx",
}
DEPLOYMENT_BOUNDARY = (
    "device program and launch ABI only; model weights remain in the "
    "separately validated graph constants"
)
_MAX_FILE_BYTES = {
    "gpu_mlir": 1 << 20,
    "binary_mlir": 8 << 20,
    "ptx": 4 << 20,
}
_MAX_MANIFEST_BYTES = 1 << 20
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(value)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_limited(path: Path, maximum: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise BackendError(f"MLIR PTX artifact cannot read {label}: {exc}") from exc
    if size < 1 or size > maximum:
        raise BackendError(
            f"MLIR PTX artifact {label} size {size} is outside 1..{maximum} bytes"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BackendError(f"MLIR PTX artifact cannot read {label}: {exc}") from exc


def _decode_text(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendError(f"MLIR PTX artifact {label} is not UTF-8 text") from exc


def _manifest_file(
    directory: Path, key: str, entry: Any
) -> tuple[str, dict[str, Any]]:
    expected_name = ARTIFACT_FILES[key]
    if not isinstance(entry, dict) or set(entry) != {
        "file",
        "size_bytes",
        "sha256",
    }:
        raise BackendError(f"MLIR PTX artifact has invalid {key} file metadata")
    if entry["file"] != expected_name:
        raise BackendError(f"MLIR PTX artifact {key} filename is invalid")
    size = entry["size_bytes"]
    digest = entry["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise BackendError(f"MLIR PTX artifact {key} size metadata is invalid")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise BackendError(f"MLIR PTX artifact {key} SHA-256 metadata is invalid")
    payload = _read_limited(directory / expected_name, _MAX_FILE_BYTES[key], key)
    if len(payload) != size:
        raise BackendError(f"MLIR PTX artifact {key} size does not match its manifest")
    if _sha256_bytes(payload) != digest:
        raise BackendError(f"MLIR PTX artifact {key} SHA-256 does not match")
    return _decode_text(payload, key), entry


def write_mlir_ptx_artifact(
    directory: str | os.PathLike[str], program: MlirPtxProgram
) -> dict[str, Any]:
    """Write a validated, weight-free PTX deployment directory.

    The destination may be new or an empty directory.  Rejecting a non-empty
    directory prevents a failed refresh from leaving an old manifest paired
    with a mixture of old and new files.
    """

    validate_mlir_ptx_program(program)
    destination = Path(directory)
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("MLIR PTX artifact output must be a directory")
        try:
            if any(destination.iterdir()):
                raise ValueError("MLIR PTX artifact output directory must be empty")
        except OSError as exc:
            raise ValueError(f"cannot inspect MLIR PTX artifact output: {exc}") from exc
    else:
        destination.mkdir(parents=True)

    contents = {
        "gpu_mlir": program.gpu_source,
        "binary_mlir": program.binary_mlir,
        "ptx": program.ptx,
    }
    files: dict[str, dict[str, Any]] = {}
    for key, filename in ARTIFACT_FILES.items():
        payload = contents[key].encode("utf-8")
        if not payload or len(payload) > _MAX_FILE_BYTES[key]:
            raise BackendError(f"MLIR PTX artifact {key} exceeds its size limit")
        _write_bytes_atomic(destination / filename, payload)
        files[key] = {
            "file": filename,
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    manifest = {
        **program.describe(),
        "artifact": ARTIFACT_KIND,
        "files": files,
        "deployment_boundary": DEPLOYMENT_BOUNDARY,
    }
    encoded_manifest = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded_manifest) > _MAX_MANIFEST_BYTES:
        raise BackendError("MLIR PTX artifact manifest exceeds its size limit")
    _write_bytes_atomic(destination / "manifest.json", encoded_manifest)
    return manifest


def load_mlir_ptx_artifact(
    directory: str | os.PathLike[str], graph: Graph
) -> MlirPtxProgram:
    """Load, verify and bind an offline PTX directory to ``graph``."""

    root = Path(directory)
    if not root.is_dir():
        raise BackendError("MLIR PTX artifact path must be a directory")
    manifest_payload = _read_limited(
        root / "manifest.json", _MAX_MANIFEST_BYTES, "manifest"
    )
    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackendError(f"MLIR PTX artifact manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("artifact") != ARTIFACT_KIND:
        raise BackendError("directory is not a GAC MLIR/NVVM/PTX artifact")
    if manifest.get("compiler_backend") != TARGET:
        raise BackendError("MLIR PTX artifact compiler backend is invalid")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise BackendError("MLIR PTX artifact manifest has an unexpected file set")
    loaded = {
        key: _manifest_file(root, key, files[key])[0]
        for key in ARTIFACT_FILES
    }

    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise BackendError("MLIR PTX artifact has no toolchain metadata")
    command = toolchain.get("command")
    entries = toolchain.get("entries")
    revision = toolchain.get("llvm_source_revision")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise BackendError("MLIR PTX artifact compiler command metadata is invalid")
    if (
        not isinstance(entries, list)
        or any(not isinstance(value, str) for value in entries)
    ):
        raise BackendError("MLIR PTX artifact entry metadata is invalid")
    if revision is not None and not isinstance(revision, str):
        raise BackendError("MLIR PTX artifact source revision metadata is invalid")

    artifact = MlirPtxArtifact(
        binary_mlir=loaded["binary_mlir"],
        ptx=loaded["ptx"],
        source_sha256=str(toolchain.get("source_sha256", "")),
        command=tuple(command),
        mlir_opt_version=str(toolchain.get("version", "")),
        mlir_opt_source_revision=revision,
        target_chip=str(toolchain.get("chip", "")),
        target_features=str(toolchain.get("features", "")),
        ptx_version=str(toolchain.get("ptx_version", "")),
        entries=tuple(entries),
    )
    if artifact.entries != PTX_ENTRY_SYMBOLS:
        raise BackendError("MLIR PTX artifact entry order is invalid")
    program = assemble_mlir_ptx_program(graph, artifact)
    if loaded["gpu_mlir"] != program.gpu_source:
        raise BackendError("artifact GPU MLIR does not match the canonical graph")

    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise BackendError("MLIR PTX artifact has no integrity metadata")
    expected_integrity = integrity.get("integrity_sha256")
    if expected_integrity != program.integrity_sha256:
        raise BackendError("MLIR PTX artifact is not bound to this graph and launch plan")
    expected_manifest = {
        **program.describe(),
        "artifact": ARTIFACT_KIND,
        "files": files,
        "deployment_boundary": DEPLOYMENT_BOUNDARY,
    }
    if manifest != expected_manifest:
        raise BackendError(
            "MLIR PTX artifact manifest metadata does not match the validated program"
        )
    return program


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_KIND",
    "DEPLOYMENT_BOUNDARY",
    "load_mlir_ptx_artifact",
    "write_mlir_ptx_artifact",
]
