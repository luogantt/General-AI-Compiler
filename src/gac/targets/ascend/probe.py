"""Fail-closed Ascend 910 environment discovery.

No import of ``torch`` or ``torch_npu`` occurs in the compiler process.  The
runtime check is isolated in a child interpreter so an incompatible CANN wheel
cannot corrupt graph construction or silently select a CPU fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


_OUTPUT_LIMIT = 4096
_SUPPORTED_PRODUCT_RE = re.compile(
    r"\b(?:Ascend\s*)?910B\w*\b|\bAtlas\s+A[23]\b|"
    r"\b(?:Ascend\s*)?950(?:PR|DT)\w*\b",
    re.IGNORECASE,
)
_ASCEND_PRODUCT_RE = re.compile(
    r"\b(?:Ascend\s*)?(?:910B\w*|910A\w*|910|950)\b|\bAtlas\s+A[23]\b",
    re.IGNORECASE,
)


def _bounded(value: str | None) -> str:
    return (value or "").strip()[:_OUTPUT_LIMIT]


def _command_probe(name: str, arguments: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if executable is None:
        return {
            "available": False,
            "ok": False,
            "executable": None,
            "returncode": None,
            "stdout": "",
            "stderr": "",
        }
    try:
        completed = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True,
            "ok": False,
            "executable": executable,
            "returncode": None,
            "stdout": "",
            "stderr": _bounded(f"{type(exc).__name__}: {exc}"),
        }
    return {
        "available": True,
        "ok": completed.returncode == 0,
        "executable": executable,
        "returncode": completed.returncode,
        "stdout": _bounded(completed.stdout),
        "stderr": _bounded(completed.stderr),
    }


_RUNTIME_PROBE = r"""
import json

result = {
    "importable": False,
    "torch_version": None,
    "torch_npu_version": None,
    "device_count": 0,
    "device_names": [],
    "triton_importable": False,
    "triton_version": None,
    "triton_target": None,
    "error": None,
}
try:
    import torch
    result["torch_version"] = str(torch.__version__)
    import torch_npu
    result["torch_npu_version"] = str(getattr(torch_npu, "__version__", "unknown"))
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch.npu is unavailable after importing torch_npu")
    result["device_count"] = int(npu.device_count())
    result["device_names"] = [str(npu.get_device_name(index)) for index in range(result["device_count"])]
    result["importable"] = True
    try:
        import triton
        result["triton_importable"] = True
        result["triton_version"] = str(getattr(triton, "__version__", "unknown"))
        target = triton.runtime.driver.active.get_current_target()
        result["triton_target"] = {
            "backend": str(getattr(target, "backend", "")),
            "arch": str(getattr(target, "arch", "")),
            "warp_size": int(getattr(target, "warp_size", 0)),
        }
    except Exception as exc:
        result["triton_error"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
"""


def _python_runtime_probe(executable: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [executable, "-c", _RUNTIME_PROBE],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "importable": False,
            "device_count": 0,
            "error": _bounded(f"{type(exc).__name__}: {exc}"),
        }
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        payload = {
            "importable": False,
            "device_count": 0,
            "error": "runtime probe returned no JSON object",
        }
    payload["probe_returncode"] = completed.returncode
    payload["probe_stderr"] = _bounded(completed.stderr)
    return payload


def _existing_toolkit_roots(environ: Mapping[str, str]) -> tuple[str, ...]:
    candidates = [
        environ.get("ASCEND_HOME_PATH"),
        environ.get("ASCEND_TOOLKIT_HOME"),
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/latest",
    ]
    roots: list[str] = []
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            resolved = str(Path(candidate).resolve())
            if resolved not in roots:
                roots.append(resolved)
    return tuple(roots)


def _products(*text: str) -> tuple[str, ...]:
    found = {
        match.group(0).strip()
        for value in text
        for match in _ASCEND_PRODUCT_RE.finditer(value or "")
    }
    return tuple(sorted(found, key=str.casefold))


def probe_ascend_environment(
    *,
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible, non-mutating Ascend discovery report.

    ``triton_ascend_discoverable`` is deliberately conservative: it requires a
    visible NPU runtime, a Triton target whose backend identifies as NPU/Ascend,
    and a product family explicitly advertised by upstream Triton-Ascend.  It
    is discovery, not a compile or execution smoke test.  A device reported
    only as ``Ascend 910`` remains unconfirmed until its exact SKU is known.
    """

    environment = dict(os.environ if environ is None else environ)
    npu_smi = _command_probe("npu-smi", ("info",))
    atc = _command_probe("atc", ("--version",))
    runtime = _python_runtime_probe(python_executable or sys.executable)
    toolkit_roots = _existing_toolkit_roots(environment)

    product_names = _products(
        npu_smi.get("stdout", ""),
        "\n".join(str(value) for value in runtime.get("device_names", ())),
    )
    supported_product = any(_SUPPORTED_PRODUCT_RE.search(name) for name in product_names)
    runtime_target = runtime.get("triton_target")
    target_backend = (
        str(runtime_target.get("backend", "")).lower()
        if isinstance(runtime_target, dict)
        else ""
    )
    hardware_present = bool(npu_smi.get("ok")) and (
        bool(product_names) or int(runtime.get("device_count", 0) or 0) > 0
    )
    cann_present = bool(toolkit_roots) or bool(atc.get("ok"))
    torch_npu_discoverable = (
        platform.system() == "Linux"
        and hardware_present
        and cann_present
        and runtime.get("importable") is True
        and int(runtime.get("device_count", 0) or 0) > 0
    )
    triton_discoverable = (
        torch_npu_discoverable
        and runtime.get("triton_importable") is True
        and target_backend in {"npu", "ascend"}
        and supported_product
    )

    blockers: list[str] = []
    if platform.system() != "Linux":
        blockers.append("Ascend deployment requires a Linux host")
    if not hardware_present:
        blockers.append("npu-smi did not confirm an available Ascend device")
    if not cann_present:
        blockers.append("CANN toolkit was not found")
    if runtime.get("importable") is not True:
        blockers.append("torch_npu could not be imported in the selected Python environment")
    elif int(runtime.get("device_count", 0) or 0) <= 0:
        blockers.append("torch_npu reported zero NPU devices")
    if not supported_product:
        blockers.append("exact Triton-Ascend-supported product SKU is not confirmed")
    if runtime.get("triton_importable") is not True or target_backend not in {"npu", "ascend"}:
        blockers.append("active Triton target is not the Ascend NPU backend")
    if torch_npu_discoverable:
        blockers.append(
            "torch_npu execution is unverified; this discovery-only probe does not launch an NPU operator"
        )
    if triton_discoverable:
        blockers.append(
            "Triton-Ascend compilation/execution is unverified; no kernel smoke test was run"
        )
    blockers.append("GAC Ascend npubin compilation and physical launcher are not implemented")

    return {
        "schema_version": 1,
        "kind": "gac-ascend-environment-probe",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_executable": python_executable or sys.executable,
            "python_version": platform.python_version(),
        },
        "cann": {
            "toolkit_roots": list(toolkit_roots),
            "ascend_opp_path_set": bool(environment.get("ASCEND_OPP_PATH")),
            "atc": atc,
        },
        "hardware": {
            "npu_smi": npu_smi,
            "products": list(product_names),
            "present": hardware_present,
            "triton_supported_product_confirmed": supported_product,
            "exact_sku_confirmation_required": not supported_product,
        },
        "python_runtime": runtime,
        "readiness": {
            "torch_npu_discoverable": torch_npu_discoverable,
            "triton_ascend_discoverable": triton_discoverable,
            "torch_npu_execution_verified": False,
            "triton_ascend_compile_verified": False,
            "triton_ascend_execution_verified": False,
            "gac_ttir_export_only": True,
            "gac_npubin_compiler_ready": False,
            "gac_physical_runtime_ready": False,
            "full_galaxeavla_ready": False,
        },
        "blockers": blockers,
    }


__all__ = ["probe_ascend_environment"]
