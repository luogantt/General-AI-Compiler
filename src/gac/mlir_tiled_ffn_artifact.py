"""Weight-free offline artifact directories for ``mlir-tiled-ffn-sm86``."""

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

from .backends.mlir_cublas_ptx import assemble_mlir_cublas_ptx_program
from .errors import BackendError
from .ir import Graph
from .mlir_tiled_ffn import (
    TARGET,
    MlirTiledFfnProgram,
    assemble_mlir_tiled_ffn_program,
    assemble_tiled_matmul_program,
    infer_tiled_ffn_rows,
    validate_mlir_tiled_ffn_program,
)
from .mlir_tiled_matmul import (
    BEVFORMER_ROWS,
    bevformer_fc1_tiled_config,
    bevformer_fc2_tiled_config,
)
from .mlir_toolchain import MlirPtxArtifact


ARTIFACT_KIND = "gac.mlir-tiled-ffn-sm86.directory"
ARTIFACT_SCHEMA_VERSION = 2
LEGACY_ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = {
    "fusion_gpu_mlir": "fusion.gpu.mlir",
    "fusion_binary_mlir": "fusion.binary.mlir",
    "fusion_ptx": "fusion.ptx",
    "fc1_gpu_mlir": "fc1.gpu.mlir",
    "fc1_binary_mlir": "fc1.binary.mlir",
    "fc1_ptx": "fc1.ptx",
    "fc2_gpu_mlir": "fc2.gpu.mlir",
    "fc2_binary_mlir": "fc2.binary.mlir",
    "fc2_ptx": "fc2.ptx",
}
DEPLOYMENT_BOUNDARY = (
    "three validated PTX dispatches only; model constants remain in the separately "
    "validated graph"
)
_MAX_MEMBER_BYTES = 8 << 20
_MAX_MANIFEST_BYTES = 2 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _toolchain(artifact: MlirPtxArtifact) -> dict[str, Any]:
    return {
        "command": list(artifact.command),
        "version": artifact.mlir_opt_version,
        "llvm_source_revision": artifact.mlir_opt_source_revision,
        "source_sha256": artifact.source_sha256,
        "chip": artifact.target_chip,
        "features": artifact.target_features,
        "ptx_version": artifact.ptx_version,
        "entries": list(artifact.entries),
    }


def _contents(program: MlirTiledFfnProgram) -> dict[str, str]:
    base = program.base_program
    return {
        "fusion_gpu_mlir": base.gpu_source,
        "fusion_binary_mlir": base.binary_mlir,
        "fusion_ptx": base.ptx,
        "fc1_gpu_mlir": program.fc1.gpu_source,
        "fc1_binary_mlir": program.fc1.artifact.binary_mlir,
        "fc1_ptx": program.fc1.ptx,
        "fc2_gpu_mlir": program.fc2.gpu_source,
        "fc2_binary_mlir": program.fc2.artifact.binary_mlir,
        "fc2_ptx": program.fc2.ptx,
    }


def _manifest(
    program: MlirTiledFfnProgram, files: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    unsigned = {
        **program.describe(),
        "artifact": ARTIFACT_KIND,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "deployment_boundary": DEPLOYMENT_BOUNDARY,
        "files": {name: dict(record) for name, record in files.items()},
        "toolchains": {
            "fusion": _toolchain(
                MlirPtxArtifact(
                    binary_mlir=program.base_program.binary_mlir,
                    ptx=program.base_program.ptx,
                    source_sha256=program.base_program.toolchain["source_sha256"],
                    command=tuple(program.base_program.toolchain["command"]),
                    mlir_opt_version=program.base_program.toolchain["version"],
                    mlir_opt_source_revision=program.base_program.toolchain.get(
                        "llvm_source_revision"
                    ),
                    target_chip=program.base_program.toolchain["chip"],
                    target_features=program.base_program.toolchain["features"],
                    ptx_version=program.base_program.toolchain["ptx_version"],
                    entries=tuple(program.base_program.toolchain["entries"]),
                )
            ),
            "fc1": _toolchain(program.fc1.artifact),
            "fc2": _toolchain(program.fc2.artifact),
        },
    }
    return {
        **unsigned,
        "manifest_payload_sha256": _sha256(_canonical(unsigned)),
    }


def _manifest_for_schema(
    program: MlirTiledFfnProgram,
    files: Mapping[str, Mapping[str, Any]],
    schema_version: int,
) -> dict[str, Any]:
    """Render the canonical manifest for a supported on-disk schema.

    Schema 1 predates the explicit FFN profile.  It can describe only the
    original 2,500-row BEVFormer artifact.  Schema 2 makes the profile
    mandatory so a 40,000-row UniAD artifact cannot be confused with it.
    """

    manifest = _manifest(program, files)
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        return manifest
    if (
        schema_version != LEGACY_ARTIFACT_SCHEMA_VERSION
        or program.rows != BEVFORMER_ROWS
    ):
        raise BackendError("unsupported legacy MLIR tiled FFN artifact profile")
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256")
    unsigned.pop("profile")
    unsigned["artifact_schema_version"] = LEGACY_ARTIFACT_SCHEMA_VERSION
    return {
        **unsigned,
        "manifest_payload_sha256": _sha256(_canonical(unsigned)),
    }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_mlir_tiled_ffn_artifact(
    directory: str | os.PathLike[str], program: MlirTiledFfnProgram
) -> dict[str, Any]:
    """Atomically publish all three weight-free MLIR/PTX components."""

    validate_mlir_tiled_ffn_program(program)
    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise ValueError("MLIR tiled FFN artifact output must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        )
    )
    published = False
    try:
        files: dict[str, dict[str, Any]] = {}
        contents = _contents(program)
        for key, filename in ARTIFACT_FILES.items():
            payload = contents[key].encode("utf-8")
            if not payload or len(payload) > _MAX_MEMBER_BYTES:
                raise BackendError(f"MLIR tiled FFN artifact {key} is outside its size limit")
            path = staging / filename
            with path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            files[key] = {
                "file": filename,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
        manifest = _manifest(program, files)
        payload = (
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(payload) > _MAX_MANIFEST_BYTES:
            raise BackendError("MLIR tiled FFN manifest exceeds its size limit")
        with (staging / MANIFEST_FILE).open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        os.replace(staging, destination)
        published = True
        _fsync_directory(destination.parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _read(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink():
        raise BackendError(f"MLIR tiled FFN artifact {label} must not be a symlink")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise BackendError(f"cannot read MLIR tiled FFN artifact {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
        raise BackendError(f"MLIR tiled FFN artifact {label} is not a valid regular file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BackendError(
            f"cannot read MLIR tiled FFN artifact {label}: {exc}"
        ) from exc
    if len(payload) != metadata.st_size:
        raise BackendError(f"MLIR tiled FFN artifact {label} changed while reading")
    return payload


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_files(root: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ARTIFACT_FILES):
        raise BackendError("MLIR tiled FFN manifest has an unexpected file set")
    loaded: dict[str, str] = {}
    for key, filename in ARTIFACT_FILES.items():
        record = files[key]
        if not isinstance(record, dict) or set(record) != {
            "file",
            "size_bytes",
            "sha256",
        }:
            raise BackendError(f"MLIR tiled FFN {key} file metadata is invalid")
        size, digest = record["size_bytes"], record["sha256"]
        if (
            record["file"] != filename
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise BackendError(f"MLIR tiled FFN {key} file metadata is invalid")
        payload = _read(root / filename, _MAX_MEMBER_BYTES, key)
        if len(payload) != size or _sha256(payload) != digest:
            raise BackendError(f"MLIR tiled FFN {key} does not match its manifest")
        try:
            loaded[key] = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackendError(f"MLIR tiled FFN {key} is not UTF-8") from exc
    return loaded


def _artifact(
    toolchain: object,
    *,
    binary_mlir: str,
    ptx: str,
    expected_entries: tuple[str, ...],
) -> MlirPtxArtifact:
    if not isinstance(toolchain, dict):
        raise BackendError("MLIR tiled FFN toolchain metadata is invalid")
    expected = {
        "command",
        "version",
        "llvm_source_revision",
        "source_sha256",
        "chip",
        "features",
        "ptx_version",
        "entries",
    }
    if set(toolchain) != expected:
        raise BackendError("MLIR tiled FFN toolchain metadata has unexpected fields")
    command, entries = toolchain["command"], toolchain["entries"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
        or not isinstance(entries, list)
        or tuple(entries) != expected_entries
    ):
        raise BackendError("MLIR tiled FFN toolchain command or entries are invalid")
    source_digest = toolchain["source_sha256"]
    if not isinstance(source_digest, str) or _SHA256.fullmatch(source_digest) is None:
        raise BackendError("MLIR tiled FFN toolchain source digest is invalid")
    for key in ("version", "chip", "features", "ptx_version"):
        if not isinstance(toolchain[key], str) or not toolchain[key]:
            raise BackendError(f"MLIR tiled FFN toolchain {key} is invalid")
    revision = toolchain["llvm_source_revision"]
    if revision is not None and not isinstance(revision, str):
        raise BackendError("MLIR tiled FFN LLVM revision is invalid")
    return MlirPtxArtifact(
        binary_mlir=binary_mlir,
        ptx=ptx,
        source_sha256=source_digest,
        command=tuple(command),
        mlir_opt_version=toolchain["version"],
        mlir_opt_source_revision=revision,
        target_chip=toolchain["chip"],
        target_features=toolchain["features"],
        ptx_version=toolchain["ptx_version"],
        entries=tuple(entries),
    )


def load_mlir_tiled_ffn_artifact(
    directory: str | os.PathLike[str], graph: Graph
) -> MlirTiledFfnProgram:
    """Verify and reconstruct a tiled FFN artifact without MLIR or CUDA."""

    if not isinstance(graph, Graph):
        raise TypeError("graph must be a Graph")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise BackendError("MLIR tiled FFN artifact must be a non-symlink directory")
    try:
        members = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise BackendError(f"cannot inspect MLIR tiled FFN artifact: {exc}") from exc
    if members != {
        MANIFEST_FILE,
        *ARTIFACT_FILES.values(),
    }:
        raise BackendError("MLIR tiled FFN artifact directory has unexpected members")
    payload = _read(root / MANIFEST_FILE, _MAX_MANIFEST_BYTES, "manifest")
    try:
        manifest = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackendError(f"MLIR tiled FFN manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BackendError("MLIR tiled FFN manifest root must be an object")
    version = manifest.get("artifact_schema_version")
    if (
        manifest.get("artifact") != ARTIFACT_KIND
        or manifest.get("compiler_backend") != TARGET
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version not in {
            LEGACY_ARTIFACT_SCHEMA_VERSION,
            ARTIFACT_SCHEMA_VERSION,
        }
    ):
        raise BackendError("directory is not a supported MLIR tiled FFN artifact")
    digest = manifest.get("manifest_payload_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise BackendError("MLIR tiled FFN manifest digest is invalid")
    unsigned = dict(manifest)
    del unsigned["manifest_payload_sha256"]
    if _sha256(_canonical(unsigned)) != digest:
        raise BackendError("MLIR tiled FFN manifest digest does not match")
    loaded = _load_files(root, manifest)
    toolchains = manifest.get("toolchains")
    if not isinstance(toolchains, dict) or set(toolchains) != {"fusion", "fc1", "fc2"}:
        raise BackendError("MLIR tiled FFN component toolchains are invalid")

    fusion_entries = ("gac_bias_relu_inplace_f32", "gac_bias_residual_layer_norm_f32")
    fusion = _artifact(
        toolchains["fusion"],
        binary_mlir=loaded["fusion_binary_mlir"],
        ptx=loaded["fusion_ptx"],
        expected_entries=fusion_entries,
    )
    base = assemble_mlir_cublas_ptx_program(graph, fusion)
    if loaded["fusion_gpu_mlir"] != base.gpu_source:
        raise BackendError("MLIR tiled FFN fusion source is not canonical")
    rows = infer_tiled_ffn_rows(base)

    tile = manifest.get("matmul_tile")
    if (
        not isinstance(tile, list)
        or len(tile) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in tile)
    ):
        raise BackendError("MLIR tiled FFN tile metadata is invalid")
    configs = (
        bevformer_fc1_tiled_config(
            rows=rows, tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
        ),
        bevformer_fc2_tiled_config(
            rows=rows, tile_m=tile[0], tile_n=tile[1], tile_k=tile[2]
        ),
    )
    components = []
    for name, config in zip(("fc1", "fc2"), configs, strict=True):
        artifact = _artifact(
            toolchains[name],
            binary_mlir=loaded[f"{name}_binary_mlir"],
            ptx=loaded[f"{name}_ptx"],
            expected_entries=(config.symbol,),
        )
        component = assemble_tiled_matmul_program(config, artifact)
        if loaded[f"{name}_gpu_mlir"] != component.gpu_source:
            raise BackendError(f"MLIR tiled FFN {name} source is not canonical")
        components.append(component)
    program = assemble_mlir_tiled_ffn_program(base, components[0], components[1])
    expected = _manifest_for_schema(program, manifest["files"], version)
    if _canonical(manifest) != _canonical(expected):
        raise BackendError("MLIR tiled FFN manifest does not match the reconstructed program")
    return program


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_KIND",
    "ARTIFACT_SCHEMA_VERSION",
    "LEGACY_ARTIFACT_SCHEMA_VERSION",
    "DEPLOYMENT_BOUNDARY",
    "MANIFEST_FILE",
    "load_mlir_tiled_ffn_artifact",
    "write_mlir_tiled_ffn_artifact",
]
