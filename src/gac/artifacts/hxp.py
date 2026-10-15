"""Deterministic, data-only model packages.

An ``.hxp`` file is a ZIP container with a canonical manifest, graph metadata,
CUDA source, and one NumPy ``.npy`` payload per constant.  Loading verifies the
complete byte manifest before parsing arrays and returns the source as text; it
never compiles, imports, evaluates, or executes package content.
"""

from __future__ import annotations

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
from typing import Any, Mapping
import zipfile

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph
from gac.compiler.memory import MemoryPlan, plan_memory


PACKAGE_SCHEMA_VERSION = 1
PACKAGE_FORMAT = "gac-model-package"
DEFAULT_TARGET = "cuda-sm86"
_GRAPH_SCHEMA_VERSION = 1
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class LoadedPackage:
    """Verified package contents; ``program_source`` remains inert text."""

    graph: Graph
    program_source: str
    memory_plan: MemoryPlan
    manifest: dict[str, Any]

    @property
    def target(self) -> str:
        return self.manifest["target"]

    @property
    def program(self) -> str:
        return self.program_source


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
        raise ValidationError(f"package metadata is not canonical JSON: {exc}") from exc


def _reject_constant(token: str) -> None:
    raise ValidationError(f"non-finite JSON number {token!r} is not allowed")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{name} root must be a JSON object")
    return value


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def _file_record(data: bytes) -> dict[str, Any]:
    return {"sha256": _hash(data), "size": len(data)}


def _npy_bytes(value: np.ndarray) -> bytes:
    if value.dtype.hasobject:
        raise ValidationError("object-dtype constants cannot be stored in a model package")
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _load_npy(data: bytes, name: str, expected_shape: Any, expected_dtype: Any) -> np.ndarray:
    """Inspect an NPY header before allowing NumPy to allocate the array."""

    if not isinstance(expected_shape, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
        for dimension in expected_shape
    ):
        raise ValidationError(f"constant {name!r} has an invalid manifest shape")
    if not isinstance(expected_dtype, str):
        raise ValidationError(f"constant {name!r} has an invalid manifest dtype")
    try:
        declared_dtype = np.dtype(expected_dtype)
    except TypeError as exc:
        raise ValidationError(f"constant {name!r} has an invalid manifest dtype") from exc
    if declared_dtype.hasobject:
        raise ValidationError(f"constant {name!r} has an unsupported dtype")
    expected_elements = math.prod(expected_shape) if expected_shape else 1
    expected_bytes = expected_elements * declared_dtype.itemsize
    if expected_bytes > len(data):
        raise ValidationError(f"constant {name!r} declares more data than its payload contains")

    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValidationError(f"constant {name!r} uses unsupported NPY version {version}")
    except ValidationError:
        raise
    except (EOFError, ValueError) as exc:
        raise ValidationError(f"constant {name!r} has an invalid NPY header") from exc
    if dtype.hasobject or fortran_order:
        raise ValidationError(f"constant {name!r} must be a C-contiguous non-object array")
    if tuple(expected_shape) != shape or dtype.name != declared_dtype.name:
        raise ValidationError(f"constant {name!r} NPY header does not match its manifest")
    if len(data) - stream.tell() != expected_bytes:
        raise ValidationError(f"constant {name!r} NPY payload length is inconsistent")
    stream.seek(0)
    try:
        value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValidationError(f"constant {name!r} is not a safe NumPy array") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValidationError(f"constant {name!r} has an unsupported dtype")
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


def _validate_target(target: Any) -> str:
    if not isinstance(target, str) or _TARGET_RE.fullmatch(target) is None:
        raise ValidationError(f"invalid package target {target!r}")
    return target


def _validate_plan_for_graph(plan: MemoryPlan, graph: Graph) -> None:
    plan.validate()
    if plan.graph_name != graph.name:
        raise ValidationError("memory plan graph name does not match the package graph")
    expected = set(graph.inputs) | set(graph.constants)
    expected.update(output for node in graph.nodes for output in node.outputs)
    actual = {allocation.name for allocation in plan.allocations}
    if actual != expected:
        raise ValidationError("memory plan does not cover exactly the package graph values")
    for allocation in plan.allocations:
        if graph.value_specs.get(allocation.name) != allocation.spec:
            raise ValidationError(f"memory plan spec for {allocation.name!r} does not match the graph")
    aliases = {
        allocation.name: allocation.alias_of
        for allocation in plan.allocations
        if allocation.alias_of is not None
    }
    expected_plan = plan_memory(
        graph,
        alignment=plan.alignment,
        aliases=aliases,
        alias_policy="validated" if plan.aliases_allowed else "forbid",
    )
    if plan != expected_plan:
        raise ValidationError("memory plan does not match deterministic planning for the graph")


def write_model_package(
    path: str | os.PathLike[str],
    *,
    graph: Graph,
    program_source: str,
    target: str = DEFAULT_TARGET,
    memory_plan: MemoryPlan | None = None,
    program_metadata: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a deterministic package and atomically replace ``path``.

    The same graph, source, target, metadata and NumPy version produce identical
    bytes.  Constant data is omitted from ``graph.json`` and exists only in the
    individually checksummed ``constants/*.npy`` entries.
    """

    if not isinstance(graph, Graph):
        raise ValidationError("graph must be a Graph")
    graph.validate()
    target = _validate_target(target)
    if not isinstance(program_source, str) or "\x00" in program_source:
        raise ValidationError("program_source must be NUL-free text")
    plan = memory_plan or plan_memory(graph)
    _validate_plan_for_graph(plan, graph)

    graph_bytes = _canonical_json(graph.to_dict(include_constant_data=False))
    program_bytes = program_source.encode("utf-8")
    payloads: dict[str, bytes] = {
        "graph.json": graph_bytes,
        "program.cu": program_bytes,
    }
    constant_manifest: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(sorted(graph.constants)):
        value = np.asarray(graph.constants[name])
        member = f"constants/{index:04d}.npy"
        data = _npy_bytes(value)
        payloads[member] = data
        constant_manifest[name] = {
            "path": member,
            "dtype": value.dtype.name,
            "shape": list(value.shape),
            **_file_record(data),
        }

    file_manifest = {name: _file_record(payloads[name]) for name in sorted(payloads)}
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "format": PACKAGE_FORMAT,
        "target": target,
        "graph_schema_version": _GRAPH_SCHEMA_VERSION,
        "graph_name": graph.name,
        "files": file_manifest,
        "constants": constant_manifest,
        "memory_plan": plan.to_dict(),
        "program": dict(program_metadata or {}),
        "metadata": dict(metadata or {}),
    }
    manifest_bytes = _canonical_json(manifest)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(_zip_info("manifest.json"), manifest_bytes)
            for name in ("graph.json", "program.cu"):
                archive.writestr(_zip_info(name), payloads[name])
            for name in sorted(payloads):
                if name.startswith("constants/"):
                    archive.writestr(_zip_info(name), payloads[name])
        os.replace(temporary_name, destination)
        temporary_name = None
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot write model package {destination}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def write_package(
    path: str | os.PathLike[str],
    program: Any,
    *,
    memory_plan: MemoryPlan | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Convenience writer for a compiled program object.

    The object must expose ``graph`` and ``source``.  If it has ``describe()``,
    that data is recorded as inert program metadata.
    """

    graph = getattr(program, "graph", None)
    source = getattr(program, "source", None)
    if not isinstance(graph, Graph) or not isinstance(source, str):
        raise ValidationError("program must expose Graph 'graph' and text 'source' fields")
    description = program.describe() if callable(getattr(program, "describe", None)) else {}
    if not isinstance(description, Mapping):
        raise ValidationError("program.describe() must return a mapping")
    target = description.get("target", DEFAULT_TARGET)
    return write_model_package(
        path,
        graph=graph,
        program_source=source,
        target=target,
        memory_plan=memory_plan,
        program_metadata=description,
        metadata=metadata,
    )


def _inspect_archive(
    archive: zipfile.ZipFile,
    *,
    max_uncompressed_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in archive.infolist():
        if info.filename in infos:
            raise ValidationError(f"package contains duplicate member {info.filename!r}")
        if not _safe_member_name(info.filename):
            raise ValidationError(f"package contains unsafe member path {info.filename!r}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if info.is_dir() or stat.S_ISLNK(mode):
            raise ValidationError(f"package member {info.filename!r} is not a regular file")
        if info.flag_bits & 0x1:
            raise ValidationError("encrypted ZIP members are not supported")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValidationError(f"package member {info.filename!r} uses unsupported compression")
        total += info.file_size
        if info.file_size < 0 or total > max_uncompressed_bytes:
            raise ValidationError("package exceeds the uncompressed-size limit")
        infos[info.filename] = info
    if len(infos) > 100_000:
        raise ValidationError("package contains too many files")
    return infos


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        return archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot read package member {info.filename!r}") from exc


def _validated_file_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise ValidationError("manifest files must be an object")
    result: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if not isinstance(name, str) or not _safe_member_name(name) or name == "manifest.json":
            raise ValidationError(f"manifest contains invalid payload path {name!r}")
        if not isinstance(record, dict):
            raise ValidationError(f"manifest record for {name!r} must be an object")
        if set(record) != {"sha256", "size"}:
            raise ValidationError(f"manifest record for {name!r} has unexpected fields")
        digest, size = record.get("sha256"), record.get("size")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValidationError(f"manifest record for {name!r} has an invalid SHA256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValidationError(f"manifest record for {name!r} has an invalid size")
        result[name] = {"sha256": digest, "size": size}
    if "graph.json" not in result or "program.cu" not in result:
        raise ValidationError("manifest must list graph.json and program.cu")
    return result


def load_package(
    path: str | os.PathLike[str],
    *,
    expected_target: str | None = DEFAULT_TARGET,
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> LoadedPackage:
    """Load and verify a package without executing its program source."""

    if isinstance(max_uncompressed_bytes, bool) or not isinstance(max_uncompressed_bytes, int) or max_uncompressed_bytes <= 0:
        raise ValidationError("max_uncompressed_bytes must be a positive integer")
    if expected_target is not None:
        expected_target = _validate_target(expected_target)
    package_path = Path(path)
    try:
        with zipfile.ZipFile(package_path, mode="r") as archive:
            infos = _inspect_archive(archive, max_uncompressed_bytes=max_uncompressed_bytes)
            if "manifest.json" not in infos:
                raise ValidationError("package has no manifest.json")
            if infos["manifest.json"].file_size > _MAX_MANIFEST_BYTES:
                raise ValidationError("package manifest is too large")
            manifest = _parse_json(_read_member(archive, infos["manifest.json"]), "manifest.json")

            if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
                raise ValidationError(
                    f"unsupported package schema {manifest.get('schema_version')!r}; "
                    f"expected {PACKAGE_SCHEMA_VERSION}"
                )
            if manifest.get("format") != PACKAGE_FORMAT:
                raise ValidationError(f"unsupported package format {manifest.get('format')!r}")
            manifest_fields = {
                "schema_version",
                "format",
                "target",
                "graph_schema_version",
                "graph_name",
                "files",
                "constants",
                "memory_plan",
                "program",
                "metadata",
            }
            if set(manifest) != manifest_fields:
                raise ValidationError("manifest fields do not match the package schema")
            target = _validate_target(manifest.get("target"))
            if expected_target is not None and target != expected_target:
                raise ValidationError(f"package target {target!r} does not match expected {expected_target!r}")
            if manifest.get("graph_schema_version") != _GRAPH_SCHEMA_VERSION:
                raise ValidationError("unsupported graph schema declared by package")
            if not isinstance(manifest.get("program"), dict) or not isinstance(manifest.get("metadata"), dict):
                raise ValidationError("manifest program and metadata fields must be objects")
            program_target = manifest["program"].get("target")
            if program_target is not None and program_target != target:
                raise ValidationError("program metadata target does not match package target")

            records = _validated_file_records(manifest)
            expected_members = {"manifest.json", *records}
            if set(infos) != expected_members:
                missing = sorted(expected_members - set(infos))
                extra = sorted(set(infos) - expected_members)
                raise ValidationError(f"package payload set differs from manifest; missing={missing}, extra={extra}")

            payloads: dict[str, bytes] = {}
            for name, record in records.items():
                info = infos[name]
                if info.file_size != record["size"]:
                    raise ValidationError(f"package member {name!r} has the wrong size")
                data = _read_member(archive, info)
                if not hmac.compare_digest(_hash(data), record["sha256"]):
                    raise ValidationError(f"package member {name!r} failed SHA256 verification")
                payloads[name] = data
    except ValidationError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"cannot open model package {package_path}: {exc}") from exc

    constants_record = manifest.get("constants")
    if not isinstance(constants_record, dict):
        raise ValidationError("manifest constants must be an object")
    constants: dict[str, np.ndarray] = {}
    used_paths: set[str] = set()
    for name, record in constants_record.items():
        if not isinstance(name, str) or not name or not isinstance(record, dict):
            raise ValidationError("manifest contains an invalid constant record")
        if set(record) != {"path", "dtype", "shape", "sha256", "size"}:
            raise ValidationError(f"constant {name!r} has unexpected manifest fields")
        member = record.get("path")
        if not isinstance(member, str) or member not in payloads or not member.startswith("constants/"):
            raise ValidationError(f"constant {name!r} has an invalid payload path")
        if member in used_paths:
            raise ValidationError(f"constant payload {member!r} is referenced more than once")
        used_paths.add(member)
        file_record = records[member]
        if record.get("sha256") != file_record["sha256"] or record.get("size") != file_record["size"]:
            raise ValidationError(f"constant {name!r} checksum metadata is inconsistent")
        expected_shape = record.get("shape")
        expected_dtype = record.get("dtype")
        value = _load_npy(payloads[member], name, expected_shape, expected_dtype)
        value = np.ascontiguousarray(value)
        value.setflags(write=False)
        constants[name] = value
    constant_payloads = {name for name in records if name.startswith("constants/")}
    if used_paths != constant_payloads:
        raise ValidationError("manifest contains an unreferenced constant payload")

    graph_data = _parse_json(payloads["graph.json"], "graph.json")
    serialized_constants = graph_data.get("constants")
    if not isinstance(serialized_constants, dict) or any(
        not isinstance(record, dict) or "data" in record for record in serialized_constants.values()
    ):
        raise ValidationError("graph.json constants must contain metadata only")
    graph = Graph.from_dict(graph_data, external_constants=constants)
    if graph.name != manifest.get("graph_name"):
        raise ValidationError("manifest graph name does not match graph.json")
    if set(graph.constants) != set(constants_record):
        raise ValidationError("manifest constants do not match graph.json")
    try:
        program_source = payloads["program.cu"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("program.cu is not UTF-8 text") from exc
    if "\x00" in program_source:
        raise ValidationError("program.cu contains a NUL byte")

    memory_data = manifest.get("memory_plan")
    if not isinstance(memory_data, dict):
        raise ValidationError("manifest has no valid memory plan")
    memory_plan = MemoryPlan.from_dict(memory_data)
    _validate_plan_for_graph(memory_plan, graph)
    return LoadedPackage(graph, program_source, memory_plan, manifest)


# Symmetric, explicit aliases for callers.
load_model_package = load_package
save_package = write_model_package


def rebuild_cuda_program(package: LoadedPackage) -> Any:
    """Rebuild and verify the CUDA program carried by a loaded package.

    Loading a package keeps ``program.cu`` inert.  This explicit transition to
    an executable program recompiles the verified graph with the local GAC
    compiler and requires a byte-for-byte source and launch-plan match.  A
    package therefore cannot smuggle arbitrary CUDA text into NVRTC while
    claiming it was generated from its graph.
    """

    if not isinstance(package, LoadedPackage):
        raise ValidationError("package must be a verified LoadedPackage")
    from gac.targets.cuda.compiler import TARGET, compile_cuda

    if package.target != TARGET:
        raise ValidationError(
            f"package target {package.target!r} cannot be rebuilt by {TARGET!r}"
        )
    recorded = package.manifest.get("program")
    if not isinstance(recorded, dict) or not recorded:
        raise ValidationError("package has no executable CUDA program metadata")

    matches: list[Any] = []
    for fuse in (True, False):
        candidate = compile_cuda(package.graph, fuse=fuse)
        description = json.loads(_canonical_json(candidate.describe()))
        if (
            candidate.source == package.program_source
            and description["launches"] == recorded.get("launches")
        ):
            matches.append(candidate)
    if not matches:
        raise ValidationError(
            "package CUDA source or launch metadata does not match its verified graph"
        )
    program = matches[0]
    generated = json.loads(_canonical_json(program.describe()))
    if set(recorded) != set(generated):
        raise ValidationError("package CUDA program metadata fields are incompatible")
    for name, expected in generated.items():
        if name == "pass_reports":
            if not isinstance(recorded[name], list):
                raise ValidationError("package pass_reports must be an array")
            continue
        if recorded[name] != expected:
            raise ValidationError(
                f"package CUDA program metadata field {name!r} does not match its graph"
            )
    return program


def load_cuda_program(
    path: str | os.PathLike[str],
    *,
    max_uncompressed_bytes: int = _DEFAULT_MAX_UNCOMPRESSED_BYTES,
) -> Any:
    """Verify an SM 8.6 package and rebuild its executable CUDA program."""

    package = load_package(
        path,
        expected_target=DEFAULT_TARGET,
        max_uncompressed_bytes=max_uncompressed_bytes,
    )
    return rebuild_cuda_program(package)
