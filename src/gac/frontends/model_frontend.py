"""Unified, lazy model frontend dispatch."""

from __future__ import annotations

from pathlib import Path

from gac.ir import Graph


class ModelFrontendError(ValueError):
    """Raised when the source model format is unknown or inconsistent."""


_FORMAT_ALIASES = {
    "json": "json",
    "gac-json": "json",
    "onnx": "onnx",
    "torch-export": "torch-export",
    "torch_export": "torch-export",
    "pt2": "torch-export",
}


def detect_model_format(path: str | Path) -> str:
    """Return the frontend name implied by a model file suffix."""

    suffix = Path(path).suffix.lower()
    try:
        return {".json": "json", ".onnx": "onnx", ".pt2": "torch-export"}[suffix]
    except KeyError as exc:
        raise ModelFrontendError(
            f"cannot infer model format from suffix {suffix or '<none>'!r}; "
            "use format='json', 'onnx', or 'torch-export'"
        ) from exc


def load_model(
    path: str | Path,
    *,
    format: str | None = None,
    strict: bool = True,
    trusted: bool = False,
    expected_sha256: str | None = None,
) -> Graph:
    """Load JSON, ONNX, or a trusted ``torch.export`` archive.

    Heavy framework packages are imported only inside the selected frontend.
    ``strict=False`` is intended for import/support inspection: unsupported
    operators remain qualified source operators and are not compilable yet.
    """

    model_format = detect_model_format(path) if format is None else _FORMAT_ALIASES.get(format.lower())
    if model_format is None:
        raise ModelFrontendError(
            f"unknown model format {format!r}; expected json, onnx, or torch-export"
        )
    if model_format == "json":
        if expected_sha256 is not None:
            raise ModelFrontendError("expected_sha256 applies only to torch-export archives")
        from .json_frontend import load_graph

        return load_graph(path)
    if model_format == "onnx":
        if expected_sha256 is not None:
            raise ModelFrontendError("expected_sha256 applies only to torch-export archives")
        from .onnx_frontend import load_onnx

        return load_onnx(path, strict=strict)

    from .torch_export_frontend import load_exported_program

    return load_exported_program(
        path,
        trusted=trusted,
        expected_sha256=expected_sha256,
        strict=strict,
    )


# Compatibility name for callers that historically imported a JSON-only
# ``load_graph``.  It now accepts every registered model source format.
load_graph = load_model
