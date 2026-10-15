"""Weight-free deployment directories for ``mlir-cublas-ptx-sm86``.

The directory contains only the MLIR fusion module, its post-lowering binary
MLIR container, textual PTX, and a manifest.  Model constants stay in the
separately validated :class:`~gac.ir.Graph` supplied by the caller.  Loading
reconstructs the canonical call plan from that graph and requires the complete
manifest to match the reconstructed program before CUDA can be touched.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping

from gac.targets.cuda.cublas_ptx import (
    PTX_ENTRY_SYMBOLS,
    TARGET,
    MlirCublasPtxProgram,
    assemble_mlir_cublas_ptx_program,
    validate_mlir_cublas_ptx_program,
)
from gac.core.errors import BackendError
from gac.core.ir import Graph
from gac.toolchains.mlir import MlirPtxArtifact


ARTIFACT_KIND = "gac.mlir-cublas-ptx-sm86.directory"
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FILES = {
    "gpu_mlir": "fusions.gpu.mlir",
    "binary_mlir": "fusions.binary.mlir",
    "ptx": "fusions.ptx",
}
MANIFEST_FILE = "manifest.json"
DEPLOYMENT_BOUNDARY = (
    "fusion device code and validated four-call ABI only; model weights remain "
    "in the separately validated graph constants"
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_staged(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact_contents(program: MlirCublasPtxProgram) -> dict[str, str]:
    return {
        "gpu_mlir": program.gpu_source,
        "binary_mlir": program.binary_mlir,
        "ptx": program.ptx,
    }


def _manifest_without_digest(
    program: MlirCublasPtxProgram, files: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        **program.describe(),
        "artifact": ARTIFACT_KIND,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "deployment_boundary": DEPLOYMENT_BOUNDARY,
        "files": {key: dict(value) for key, value in files.items()},
    }


def _manifest(
    program: MlirCublasPtxProgram, files: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result = _manifest_without_digest(program, files)
    result["manifest_payload_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def write_mlir_cublas_ptx_artifact(
    directory: str | os.PathLike[str], program: MlirCublasPtxProgram
) -> dict[str, Any]:
    """Atomically publish a validated, weight-free deployment directory.

    The final path must not exist.  All members are written and synced in a
    private sibling directory, then the complete directory is renamed into
    place.  A failed write therefore exposes neither a partial manifest nor a
    mixture of artifact generations.
    """

    validate_mlir_cublas_ptx_program(program)
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise ValueError("MLIR cuBLAS/PTX artifact output must not already exist")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError("MLIR cuBLAS/PTX artifact parent must be a directory")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=parent)
    )
    published = False
    try:
        files: dict[str, dict[str, Any]] = {}
        for key, filename in ARTIFACT_FILES.items():
            payload = _artifact_contents(program)[key].encode("utf-8")
            if not payload or len(payload) > _MAX_FILE_BYTES[key]:
                raise BackendError(
                    f"MLIR cuBLAS/PTX artifact {key} exceeds its size limit"
                )
            _write_staged(staging / filename, payload)
            files[key] = {
                "file": filename,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }

        manifest = _manifest(program, files)
        manifest_payload = (
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(manifest_payload) > _MAX_MANIFEST_BYTES:
            raise BackendError("MLIR cuBLAS/PTX manifest exceeds its size limit")
        _write_staged(staging / MANIFEST_FILE, manifest_payload)
        _fsync_directory(staging)
        os.replace(staging, destination)
        published = True
        _fsync_directory(parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _read_regular_limited(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink():
        raise BackendError(f"MLIR cuBLAS/PTX artifact {label} must not be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact cannot read {label}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise BackendError(f"MLIR cuBLAS/PTX artifact {label} is not a regular file")
    if metadata.st_size < 1 or metadata.st_size > maximum:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact {label} size {metadata.st_size} is outside "
            f"1..{maximum} bytes"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact cannot read {label}: {exc}"
        ) from exc
    if len(payload) != metadata.st_size:
        raise BackendError(f"MLIR cuBLAS/PTX artifact {label} changed while reading")
    return payload


def _decode_text(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact {label} is not UTF-8 text"
        ) from exc


def _load_member(
    root: Path, key: str, entry: object
) -> tuple[str, dict[str, Any]]:
    expected_name = ARTIFACT_FILES[key]
    if not isinstance(entry, dict) or set(entry) != {
        "file",
        "size_bytes",
        "sha256",
    }:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact has invalid {key} file metadata"
        )
    if entry["file"] != expected_name:
        raise BackendError(f"MLIR cuBLAS/PTX artifact {key} filename is invalid")
    size = entry["size_bytes"]
    digest = entry["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise BackendError(f"MLIR cuBLAS/PTX artifact {key} size is invalid")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise BackendError(f"MLIR cuBLAS/PTX artifact {key} SHA-256 is invalid")
    payload = _read_regular_limited(root / expected_name, _MAX_FILE_BYTES[key], key)
    if len(payload) != size:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact {key} size does not match its manifest"
        )
    if _sha256_bytes(payload) != digest:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact {key} SHA-256 does not match"
        )
    return _decode_text(payload, key), entry


def _toolchain_artifact(
    manifest: Mapping[str, Any], loaded: Mapping[str, str]
) -> MlirPtxArtifact:
    toolchain = manifest.get("toolchain")
    if not isinstance(toolchain, dict):
        raise BackendError("MLIR cuBLAS/PTX artifact has no toolchain metadata")
    command = toolchain.get("command")
    entries = toolchain.get("entries")
    revision = toolchain.get("llvm_source_revision")
    version = toolchain.get("version")
    source_sha256 = toolchain.get("source_sha256")
    chip = toolchain.get("chip")
    features = toolchain.get("features")
    ptx_version = toolchain.get("ptx_version")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise BackendError("MLIR cuBLAS/PTX compiler command metadata is invalid")
    if not isinstance(entries, list) or any(
        not isinstance(value, str) or not value for value in entries
    ):
        raise BackendError("MLIR cuBLAS/PTX entry metadata is invalid")
    if revision is not None and not isinstance(revision, str):
        raise BackendError("MLIR cuBLAS/PTX source revision metadata is invalid")
    for label, value in (
        ("compiler version", version),
        ("source SHA-256", source_sha256),
        ("target chip", chip),
        ("target features", features),
        ("PTX version", ptx_version),
    ):
        if not isinstance(value, str) or not value:
            raise BackendError(f"MLIR cuBLAS/PTX {label} metadata is invalid")
    if _SHA256_PATTERN.fullmatch(source_sha256) is None:
        raise BackendError("MLIR cuBLAS/PTX source SHA-256 metadata is invalid")
    if tuple(entries) != PTX_ENTRY_SYMBOLS:
        raise BackendError("MLIR cuBLAS/PTX artifact entry order is invalid")
    return MlirPtxArtifact(
        binary_mlir=loaded["binary_mlir"],
        ptx=loaded["ptx"],
        source_sha256=source_sha256,
        command=tuple(command),
        mlir_opt_version=version,
        mlir_opt_source_revision=revision,
        target_chip=chip,
        target_features=features,
        ptx_version=ptx_version,
        entries=tuple(entries),
    )


def load_mlir_cublas_ptx_artifact(
    directory: str | os.PathLike[str], graph: Graph
) -> MlirCublasPtxProgram:
    """Verify an offline directory and bind it to ``graph`` without MLIR/CUDA."""

    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise BackendError(
            "MLIR cuBLAS/PTX artifact path must be a non-symlink directory"
        )
    expected_members = {MANIFEST_FILE, *ARTIFACT_FILES.values()}
    try:
        actual_members = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise BackendError(f"cannot inspect MLIR cuBLAS/PTX artifact: {exc}") from exc
    if actual_members != expected_members:
        raise BackendError(
            "MLIR cuBLAS/PTX artifact directory has an unexpected file set"
        )

    payload = _read_regular_limited(
        root / MANIFEST_FILE, _MAX_MANIFEST_BYTES, "manifest"
    )
    try:
        manifest = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackendError(
            f"MLIR cuBLAS/PTX artifact manifest is invalid: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BackendError("MLIR cuBLAS/PTX artifact manifest root must be an object")
    if manifest.get("artifact") != ARTIFACT_KIND:
        raise BackendError("directory is not a GAC MLIR cuBLAS/PTX artifact")
    artifact_schema_version = manifest.get("artifact_schema_version")
    if (
        isinstance(artifact_schema_version, bool)
        or not isinstance(artifact_schema_version, int)
        or artifact_schema_version != ARTIFACT_SCHEMA_VERSION
    ):
        raise BackendError("MLIR cuBLAS/PTX artifact schema version is invalid")
    if manifest.get("compiler_backend") != TARGET:
        raise BackendError("MLIR cuBLAS/PTX compiler backend is invalid")
    digest = manifest.get("manifest_payload_sha256")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise BackendError("MLIR cuBLAS/PTX manifest payload SHA-256 is invalid")
    unsigned = dict(manifest)
    del unsigned["manifest_payload_sha256"]
    if _sha256_bytes(_canonical_json_bytes(unsigned)) != digest:
        raise BackendError("MLIR cuBLAS/PTX manifest payload SHA-256 does not match")

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise BackendError("MLIR cuBLAS/PTX manifest has an unexpected file set")
    loaded = {
        key: _load_member(root, key, files[key])[0] for key in ARTIFACT_FILES
    }
    artifact = _toolchain_artifact(manifest, loaded)
    program = assemble_mlir_cublas_ptx_program(graph, artifact)
    if loaded["gpu_mlir"] != program.gpu_source:
        raise BackendError(
            "MLIR cuBLAS/PTX GPU MLIR does not match the canonical graph"
        )

    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise BackendError("MLIR cuBLAS/PTX artifact has no integrity metadata")
    if integrity.get("integrity_sha256") != program.integrity_sha256:
        raise BackendError(
            "MLIR cuBLAS/PTX artifact is not bound to this graph and call plan"
        )
    expected = _manifest(program, files)
    # Python mapping equality considers True == 1 and 1.0 == 1.  Compare the
    # canonical JSON representation so the validated manifest also preserves
    # the exact JSON scalar types reconstructed from the program.
    if _canonical_json_bytes(manifest) != _canonical_json_bytes(expected):
        raise BackendError(
            "MLIR cuBLAS/PTX manifest does not match the validated graph and call plan"
        )
    return program


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "DEPLOYMENT_BOUNDARY",
    "MANIFEST_FILE",
    "load_mlir_cublas_ptx_artifact",
    "write_mlir_cublas_ptx_artifact",
]
