"""Compile GAC standard TTIR artifacts with the active Triton-Ascend target.

The module deliberately has no import-time dependency on PyTorch or Triton.
Those packages are loaded only after the complete GAC manifest and every TTIR
source have passed a fail-closed preflight.  Compilation is staged beside the
destination and the directory is published with one atomic rename.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from gac.core.errors import BackendError, DeviceUnavailableError, ValidationError


_SAFE_TTIR_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.ttir\Z")
_SUPPORTED_ARCH = re.compile(
    r"(?:Ascend\s*)?910B[A-Za-z0-9_-]*\Z|"
    r"(?:Ascend\s*)?910_93[A-Za-z0-9_-]*\Z|"
    r"(?:Atlas\s*)?A[23][A-Za-z0-9_-]*\Z|"
    r"(?:Ascend\s*)?950(?:PR|DT)?[A-Za-z0-9_-]*\Z",
    re.IGNORECASE,
)
_FORBIDDEN_ROOT_FIELDS = frozenset(
    {"double_buffer", "lmem_bytes", "vortex_matmul_schedule"}
)
_FORBIDDEN_DISPATCH_FIELDS = frozenset(
    {"triton_exclusion", "vortex", "vortex_schedule"}
)
_FORBIDDEN_TRITON_FIELDS = frozenset(
    {"abi", "block_size", "lanes", "num_warps"}
)


@dataclass(frozen=True)
class AscendTTIRDispatch:
    """One source that passed the standard-TTIR hand-off contract."""

    index: int
    kernel: str
    ttir_file: str
    source_path: Path
    source_bytes: bytes
    source_sha256: str


@dataclass(frozen=True)
class ValidatedAscendManifest:
    """A preflighted manifest and immutable snapshots of all of its sources."""

    path: Path
    payload: dict[str, Any]
    sha256: str
    dispatches: tuple[AscendTTIRDispatch, ...]


@dataclass(frozen=True)
class _VendorStack:
    torch_version: str
    torch_npu_version: str
    triton_version: str
    triton_ascend_version: str
    ir: Any
    buffer_ir: Any
    ascend_ir: Any
    make_backend: Any
    driver: Any
    compile: Any


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read {description} {path}: {exc}") from exc


def _require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{description} must be a JSON object")
    return value


def _safe_ttir_name(value: Any, *, dispatch: int) -> str:
    if not isinstance(value, str) or not _SAFE_TTIR_NAME.fullmatch(value):
        raise ValidationError(
            f"dispatch {dispatch} has an unsafe TTIR filename {value!r}; "
            "use one portable basename ending in .ttir"
        )
    if "/" in value or "\\" in value or Path(value).name != value:
        raise ValidationError(
            f"dispatch {dispatch} TTIR filename must not contain a directory: {value!r}"
        )
    return value


def validate_ascend_manifest(
    manifest: str | Path,
    ttir_dir: str | Path,
) -> ValidatedAscendManifest:
    """Validate an Ascend artifact-only manifest and snapshot every TTIR file.

    The source snapshots are used for compilation, so a file replacement after
    preflight cannot change the bytes that are sent to Triton.
    """

    manifest_path = Path(manifest).expanduser().resolve()
    source_directory = Path(ttir_dir).expanduser().resolve()
    if not source_directory.is_dir():
        raise ValidationError(f"TTIR directory does not exist: {source_directory}")

    manifest_bytes = _read_bytes(manifest_path, "Ascend manifest")
    try:
        decoded = manifest_bytes.decode("utf-8")
        payload = _require_object(json.loads(decoded), "Ascend manifest")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid UTF-8 JSON manifest {manifest_path}: {exc}") from exc

    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValidationError("Ascend manifest schema_version must be the JSON integer 1")

    required_root = {
        "target": "ascend",
        "artifact_format": "standard-ttir",
        "execution": "artifact-only",
    }
    for field, expected in required_root.items():
        if payload.get(field) != expected:
            raise ValidationError(
                f"Ascend manifest {field!r} must be {expected!r}, "
                f"got {payload.get(field)!r}"
            )
    if payload.get("runtime_available") is not False:
        raise ValidationError("Ascend manifest runtime_available must be the JSON boolean false")
    forbidden_root = sorted(_FORBIDDEN_ROOT_FIELDS.intersection(payload))
    forbidden_root.extend(
        sorted(key for key in payload if key.startswith("vortex_") and key not in forbidden_root)
    )
    if forbidden_root:
        raise ValidationError(
            "Ascend standard-TTIR manifest contains target-specific fields: "
            + ", ".join(forbidden_root)
        )

    dispatch_rows = payload.get("dispatches")
    if not isinstance(dispatch_rows, list) or not dispatch_rows:
        raise ValidationError("Ascend manifest dispatches must be a non-empty JSON array")
    dispatch_count = payload.get("triton_dispatches")
    if type(dispatch_count) is not int or dispatch_count != len(dispatch_rows):
        raise ValidationError(
            "Ascend manifest triton_dispatches must exactly equal the dispatch count"
        )

    dispatches: list[AscendTTIRDispatch] = []
    names: set[str] = set()
    kernels: set[str] = set()
    for index, raw_row in enumerate(dispatch_rows):
        row = _require_object(raw_row, f"dispatch {index}")
        if row.get("lowering") != "linalg-to-triton":
            raise ValidationError(
                f"dispatch {index} is not an all-Triton linalg-to-triton dispatch"
            )
        leaked = sorted(_FORBIDDEN_DISPATCH_FIELDS.intersection(row))
        if leaked:
            raise ValidationError(
                f"dispatch {index} contains fallback/Vortex fields: {', '.join(leaked)}"
            )

        kernel = row.get("kernel")
        if not isinstance(kernel, str) or not kernel.strip():
            raise ValidationError(f"dispatch {index} must have a non-empty kernel name")
        folded_kernel = kernel.casefold()
        if folded_kernel in kernels:
            raise ValidationError(f"duplicate Ascend dispatch kernel name: {kernel!r}")
        kernels.add(folded_kernel)

        compute_template = _require_object(
            row.get("compute_template"), f"dispatch {index} compute_template"
        )
        template_id = compute_template.get("id")
        if not isinstance(template_id, str) or not template_id.startswith(
            "triton.standard."
        ):
            raise ValidationError(
                f"dispatch {index} must select a triton.standard.* compute template"
            )
        if compute_template.get("selected") is not True:
            raise ValidationError(
                f"dispatch {index} standard Triton compute template must be selected"
            )

        contract = _require_object(row.get("triton"), f"dispatch {index} triton contract")
        if contract.get("artifact_format") != "standard-ttir":
            raise ValidationError(
                f"dispatch {index} triton artifact_format must be 'standard-ttir'"
            )
        signature = contract.get("ttir_signature")
        if not isinstance(signature, str) or not signature.strip():
            raise ValidationError(f"dispatch {index} has no standard TTIR signature")
        leaked = sorted(_FORBIDDEN_TRITON_FIELDS.intersection(contract))
        if leaked:
            raise ValidationError(
                f"dispatch {index} contains backend-specific Triton fields: "
                + ", ".join(leaked)
            )

        filename = _safe_ttir_name(contract.get("ttir_file"), dispatch=index)
        folded_name = filename.casefold()
        if folded_name in names:
            raise ValidationError(f"duplicate Ascend TTIR filename: {filename!r}")
        names.add(folded_name)

        candidate = source_directory / filename
        if candidate.is_symlink():
            raise ValidationError(f"TTIR source must not be a symbolic link: {candidate}")
        try:
            source_path = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValidationError(f"TTIR source does not exist: {candidate}") from exc
        if source_path.parent != source_directory or not source_path.is_file():
            raise ValidationError(f"TTIR source escapes its artifact directory: {candidate}")
        source_bytes = _read_bytes(source_path, "TTIR source")
        if not source_bytes.strip():
            raise ValidationError(f"TTIR source is empty: {source_path}")
        lowered_source = source_bytes.lower()
        if b"vortex" in lowered_source or b"npubin" in lowered_source:
            raise ValidationError(
                f"standard TTIR source contains a backend-specific artifact claim: {source_path}"
            )
        dispatches.append(
            AscendTTIRDispatch(
                index=index,
                kernel=kernel,
                ttir_file=filename,
                source_path=source_path,
                source_bytes=source_bytes,
                source_sha256=_digest(source_bytes),
            )
        )

    return ValidatedAscendManifest(
        path=manifest_path,
        payload=payload,
        sha256=_digest(manifest_bytes),
        dispatches=tuple(dispatches),
    )


def canonical_launcher_type(value: str) -> str:
    """Canonicalize Triton 3.2's IRSource type spelling for its launcher."""

    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"invalid Triton launcher type: {value!r}")
    scalar = value.strip()
    pointer = scalar.startswith("*")
    if pointer:
        scalar = scalar[1:]
    elif scalar.startswith("!tt.ptr<"):
        pointer = True
        scalar = scalar[len("!tt.ptr<") :]
    # Triton 3.2 can retain the MLIR pointer's closing angle bracket and a
    # trailing source location in the signature extracted by IRSource.
    scalar = scalar.split()[0].rstrip(">")
    scalar = {"f16": "fp16", "f32": "fp32", "f64": "fp64"}.get(
        scalar, scalar
    )
    if not scalar:
        raise ValidationError(f"invalid Triton launcher type: {value!r}")
    return ("*" if pointer else "") + scalar


def _load_vendor_stack() -> _VendorStack:
    """Load the matching torch_npu/Triton-Ascend stack in the required order."""

    try:
        torch = importlib.import_module("torch")
        # torch_npu must register the NPU runtime before importing Triton's
        # active driver or the process can select an unusable target.
        torch_npu = importlib.import_module("torch_npu")
        triton = importlib.import_module("triton")
        libtriton = importlib.import_module("triton._C.libtriton")
        ascend = importlib.import_module("triton._C.libtriton.ascend")
        compiler = importlib.import_module("triton.compiler.compiler")
        triton_compiler = importlib.import_module("triton.compiler")
        runtime_driver = importlib.import_module("triton.runtime.driver")
        return _VendorStack(
            torch_version=str(getattr(torch, "__version__", "unknown")),
            torch_npu_version=str(getattr(torch_npu, "__version__", "unknown")),
            triton_version=str(getattr(triton, "__version__", "unknown")),
            triton_ascend_version=_distribution_version("triton-ascend"),
            ir=libtriton.ir,
            buffer_ir=libtriton.buffer_ir,
            ascend_ir=ascend.ir,
            make_backend=compiler.make_backend,
            driver=runtime_driver.driver,
            compile=triton_compiler.compile,
        )
    # A version-skewed torch_npu/CANN stack often raises RuntimeError while
    # loading its shared libraries or registering the NPU driver.  Keep every
    # vendor initialization failure inside GAC's actionable CLI error boundary
    # instead of leaking a traceback.  Exception deliberately excludes
    # KeyboardInterrupt and SystemExit.
    except Exception as exc:
        raise DeviceUnavailableError(
            "Ascend TTIR compilation requires a matching Linux CANN, torch, "
            "torch_npu and Triton-Ascend installation (the validated image uses "
            "CANN 9.0, torch/torch_npu 2.7.1 and Triton-Ascend 3.2.1); "
            f"the vendor stack could not be loaded: {type(exc).__name__}: {exc}"
        ) from exc


def _distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _active_target(vendor: _VendorStack) -> tuple[Any, dict[str, Any]]:
    try:
        target = vendor.driver.active.get_current_target()
    except Exception as exc:
        raise DeviceUnavailableError(
            f"Triton-Ascend has no active NPU target: {type(exc).__name__}: {exc}"
        ) from exc
    backend = str(getattr(target, "backend", ""))
    arch = str(getattr(target, "arch", ""))
    if backend.casefold() != "npu":
        raise DeviceUnavailableError(
            f"active Triton target backend must be 'npu', got {backend!r}"
        )
    if _SUPPORTED_ARCH.fullmatch(arch.strip()) is None:
        raise DeviceUnavailableError(
            "active Triton NPU architecture is not an officially supported "
            f"910B/A2/A3 (including Ascend910_93xx)/950 SKU: {arch!r}"
        )
    warp_size = getattr(target, "warp_size", None)
    return target, {
        "backend": backend,
        "arch": arch,
        "warp_size": warp_size if isinstance(warp_size, int) else str(warp_size),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if hasattr(value, "backend") and hasattr(value, "arch"):
        return {
            "backend": str(value.backend),
            "arch": str(value.arch),
            "warp_size": _json_value(getattr(value, "warp_size", None)),
        }
    fields = getattr(value, "_fields", None)
    if fields is not None:
        return {str(field): _json_value(getattr(value, field)) for field in fields}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _metadata_object(metadata: Any) -> dict[str, Any]:
    converted = _json_value(metadata)
    if isinstance(converted, dict):
        return converted
    namespace = getattr(metadata, "__dict__", None)
    if isinstance(namespace, dict):
        return {str(key): _json_value(value) for key, value in namespace.items()}
    return {"value": converted}


def _write_json(path: Path, value: Any) -> bytes:
    try:
        payload = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        return payload
    except (OSError, TypeError, ValueError) as exc:
        raise ValidationError(f"cannot write JSON artifact {path}: {exc}") from exc


def _artifact(root: Path, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "file": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": _digest(payload),
    }


def _normalize_ttir(source: Path, destination: Path, vendor: _VendorStack, target: Any) -> bytes:
    context = None
    try:
        context = vendor.ir.context()
        vendor.ir.load_dialects(context)
        vendor.buffer_ir.load_dialects(context)
        vendor.ascend_ir.load_dialects(context)
        backend = vendor.make_backend(target)
        backend.load_dialects(context)
        module = vendor.ir.parse_mlir_module(str(source), context)
        # Keep the pybind context alive while producing pretty TTIR.
        module.context = context
        payload = str(module).encode("utf-8")
        if not payload.strip():
            raise BackendError(f"Triton normalized {source} to an empty module")
        destination.write_bytes(payload)
        return payload
    finally:
        if context is not None:
            disable = getattr(context, "disable_multithreading", None)
            if callable(disable):
                disable()


def _compile_vendor_dispatch(
    dispatch: AscendTTIRDispatch,
    output: Path,
    root: Path,
    vendor: _VendorStack,
    target: Any,
) -> dict[str, Any]:
    source_copy = output / "source.generic.ttir"
    normalized_path = output / "normalized.ttir"
    source_copy.write_bytes(dispatch.source_bytes)
    try:
        normalized = _normalize_ttir(source_copy, normalized_path, vendor, target)
    finally:
        try:
            source_copy.unlink()
        except FileNotFoundError:
            pass

    kernel = vendor.compile(str(normalized_path))
    asm = getattr(kernel, "asm", None)
    if not isinstance(asm, Mapping):
        raise BackendError(f"{dispatch.kernel}: Triton compiler returned no assembly mapping")
    try:
        npubin = bytes(asm["npubin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendError(f"{dispatch.kernel}: Triton compiler returned no npubin") from exc
    if not npubin:
        raise BackendError(f"{dispatch.kernel}: Triton compiler returned an empty npubin")
    ttadapter_value = asm.get("ttadapter")
    if isinstance(ttadapter_value, str):
        ttadapter = ttadapter_value.encode("utf-8")
    elif isinstance(ttadapter_value, (bytes, bytearray, memoryview)):
        ttadapter = bytes(ttadapter_value)
    else:
        raise BackendError(f"{dispatch.kernel}: Triton compiler returned no ttadapter")
    if not ttadapter:
        raise BackendError(f"{dispatch.kernel}: Triton compiler returned an empty ttadapter")

    source_object = getattr(kernel, "src", None)
    parsed_signature = dict(getattr(source_object, "signature", {}) or {})
    if not parsed_signature:
        raise BackendError(
            f"{dispatch.kernel}: Triton compiler returned an empty launcher signature"
        )
    launch_signature = {
        key: canonical_launcher_type(value) for key, value in parsed_signature.items()
    }
    if source_object is not None:
        source_object.signature = launch_signature

    raw_metadata = getattr(kernel, "metadata", None)
    metadata = _metadata_object(raw_metadata)
    kernel_name_value = metadata.get("kernel_name") or getattr(
        raw_metadata, "kernel_name", None
    )
    if not isinstance(kernel_name_value, str) or kernel_name_value != dispatch.kernel:
        raise BackendError(
            f"{dispatch.kernel}: compiled kernel name does not match the manifest: "
            f"{kernel_name_value!r}"
        )
    kernel_name = kernel_name_value

    adapter_path = output / "kernel.ttadapter.mlir"
    binary_path = output / "kernel.npubin"
    metadata_path = output / "metadata.json"
    adapter_path.write_bytes(ttadapter)
    binary_path.write_bytes(npubin)
    metadata_bytes = _write_json(metadata_path, metadata)
    report = {
        "dispatch_index": dispatch.index,
        "manifest_kernel": dispatch.kernel,
        "kernel_name": kernel_name,
        "source": {
            "file": dispatch.ttir_file,
            "sha256": dispatch.source_sha256,
        },
        "irsource_signature": {str(key): value for key, value in parsed_signature.items()},
        "launcher_signature": {str(key): value for key, value in launch_signature.items()},
        "asm_keys": sorted(str(key) for key in asm),
        "artifacts": {
            "normalized_ttir": _artifact(root, normalized_path, normalized),
            "ttadapter": _artifact(root, adapter_path, ttadapter),
            "npubin": _artifact(root, binary_path, npubin),
            "metadata": _artifact(root, metadata_path, metadata_bytes),
        },
    }
    report_path = output / "compile.json"
    report_bytes = _write_json(report_path, report)
    report["compile_report"] = _artifact(root, report_path, report_bytes)
    return report


def compile_ascend_ttir_manifest(
    manifest: str | Path,
    ttir_dir: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    always_compile: bool = False,
) -> dict[str, Any]:
    """Compile every dispatch and atomically publish one complete output tree."""

    validated = validate_ascend_manifest(manifest, ttir_dir)
    destination_input = Path(output_dir).expanduser()
    if destination_input.is_symlink():
        raise ValidationError(
            "Ascend compilation output must not be a symbolic link: "
            f"{destination_input}"
        )
    destination = destination_input.resolve()
    inherited_cache = os.environ.get("TRITON_CACHE_DIR")
    cache_value: str | Path | None = cache_dir
    if cache_value is None and inherited_cache and inherited_cache.strip():
        cache_value = inherited_cache
    resolved_cache = (
        Path(cache_value).expanduser().resolve() if cache_value is not None else None
    )
    if resolved_cache is not None and (
        resolved_cache == destination or destination in resolved_cache.parents
    ):
        raise ValidationError(
            "Triton cache directory must be outside the atomically published "
            f"output directory: cache={resolved_cache}, output={destination}"
        )
    if destination.exists():
        raise ValidationError(
            f"Ascend compilation output already exists; refusing to overwrite it: {destination}"
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValidationError(
            f"cannot create Ascend compilation output parent {destination.parent}: {exc}"
        ) from exc

    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
        )
    except OSError as exc:
        raise ValidationError(
            f"cannot create Ascend compilation staging directory beside {destination}: {exc}"
        ) from exc
    previous_cache = os.environ.get("TRITON_CACHE_DIR")
    previous_always = os.environ.get("TRITON_ALWAYS_COMPILE")
    try:
        if resolved_cache is not None:
            os.environ["TRITON_CACHE_DIR"] = str(resolved_cache)
        if always_compile:
            os.environ["TRITON_ALWAYS_COMPILE"] = "1"

        vendor = _load_vendor_stack()
        target, target_report = _active_target(vendor)
        dispatch_reports: list[dict[str, Any]] = []
        for dispatch in validated.dispatches:
            stem = Path(dispatch.ttir_file).stem
            dispatch_output = staging / "dispatches" / f"{dispatch.index:04d}-{stem}"
            dispatch_output.mkdir(parents=True)
            try:
                dispatch_report = _compile_vendor_dispatch(
                    dispatch, dispatch_output, staging, vendor, target
                )
            except (ValidationError, BackendError, DeviceUnavailableError):
                raise
            except Exception as exc:
                raise BackendError(
                    f"Triton-Ascend failed to compile dispatch {dispatch.index} "
                    f"({dispatch.kernel}): {type(exc).__name__}: {exc}"
                ) from exc
            dispatch_reports.append(dispatch_report)

        report = {
            "schema_version": 1,
            "kind": "gac-ascend-triton-compilation",
            "source_manifest": {
                "path": str(validated.path),
                "sha256": validated.sha256,
            },
            "output_dir": str(destination),
            "target": target_report,
            "versions": {
                "torch": vendor.torch_version,
                "torch_npu": vendor.torch_npu_version,
                "triton": vendor.triton_version,
                "triton_ascend": vendor.triton_ascend_version,
            },
            "dispatch_count": len(dispatch_reports),
            "dispatches": dispatch_reports,
        }
        _write_json(staging / "compile-manifest.json", report)
        if destination.exists():
            raise ValidationError(
                f"Ascend compilation output appeared during compilation: {destination}"
            )
        os.replace(staging, destination)
        staging = None  # type: ignore[assignment]
        return report
    except (ValidationError, BackendError, DeviceUnavailableError):
        raise
    except OSError as exc:
        raise ValidationError(
            f"cannot atomically publish Ascend compilation output {destination}: {exc}"
        ) from exc
    finally:
        if previous_cache is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous_cache
        if previous_always is None:
            os.environ.pop("TRITON_ALWAYS_COMPILE", None)
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = previous_always
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


# Concise public spelling for callers that do not need to mention the manifest
# implementation detail in their own API.
compile_ascend_ttir = compile_ascend_ttir_manifest


__all__ = [
    "AscendTTIRDispatch",
    "ValidatedAscendManifest",
    "canonical_launcher_type",
    "compile_ascend_ttir",
    "compile_ascend_ttir_manifest",
    "validate_ascend_manifest",
]
