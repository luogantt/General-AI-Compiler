#!/usr/bin/env python3
"""Render an inspected Graph IR as an offline network and real-weight viewer.

The JSON and NPZ come from ``inspect_onnx_ir.py``.  They are reloaded together
through Graph.from_dict, so the drawing and weight previews describe the same
validated graph as graph.ir.txt.  This does not parse text, run the model, or
invent activation values.  Large tensors are sampled, never expanded in HTML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gac.core.ir import Graph


_TEMPLATES = Path(__file__).resolve().parent / "templates"


def _json_value(value: Any) -> Any:
    """Keep JSON strict and script-safe without losing non-finite evidence."""
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        # JavaScript cannot represent every int64 as a Number.  Slice sentinels
        # such as INT64_MAX must remain exact, rather than silently rounding.
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} in Graph IR view")


def _script_json(value: Any) -> str:
    encoded = json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    # JSON is embedded in a script element.  Escaping '<' also prevents a tensor
    # name containing '</script>' from terminating that element.
    for source, escaped in (("&", "\\u0026"), ("<", "\\u003c"),
                            (">", "\\u003e"), ("\u2028", "\\u2028"),
                            ("\u2029", "\\u2029")):
        encoded = encoded.replace(source, escaped)
    return encoded


def _sample_indices(size: int, limit: int) -> list[int]:
    """Evenly spaced actual coordinates, including both ends when present."""
    if size <= 0:
        return []
    if size <= limit:
        return list(range(size))
    return [(index * (size - 1)) // (limit - 1) for index in range(limit)]


def _matrix(array: np.ndarray, leading: tuple[int, ...] = ()) -> dict[str, Any]:
    """A true last-two-axis plane; rank-zero and vectors have explicit axes."""
    if array.ndim == 0:
        plane = array.reshape(1, 1)
        axes: list[int | None] = [None, None]
        description = "标量：按 1×1 显示唯一真实数值。"
    elif array.ndim == 1:
        plane = array.reshape(1, array.shape[0])
        axes = [None, 0]
        description = "向量：按 1×N 显示，列号对应轴 0 的原始索引。"
    else:
        plane = array[leading] if leading else array
        axes = [array.ndim - 2, array.ndim - 1]
        description = f"行对应轴 {axes[0]}，列对应轴 {axes[1]}。"
        if leading:
            description += "前置轴由切片索引固定。"
    rows, cols = map(int, plane.shape)
    row_indices = _sample_indices(rows, 16)
    col_indices = _sample_indices(cols, 16)
    values = plane[np.ix_(row_indices, col_indices)].tolist()
    return {
        "rows": rows, "cols": cols,
        "row_indices": row_indices, "col_indices": col_indices,
        "values": _json_value(values), "axes": axes,
        "leading_indices": list(leading),
        "sampled": len(row_indices) < rows or len(col_indices) < cols,
        "description": description + "均为真实常量值；每轴最多均匀采样 16 项，不补零、不取平均。",
    }


def _constant_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    numeric = value.dtype.kind in "buifc"
    finite = np.isfinite(value) if numeric else None
    finite_count = int(np.count_nonzero(finite)) if finite is not None else None
    finite_values = value[finite] if finite_count else None
    prefix_shape = value.shape[:-2] if value.ndim > 2 else ()
    prefix_count = math.prod(prefix_shape)
    prefixes = [tuple(int(item) for item in np.unravel_index(index, prefix_shape))
                for index in _sample_indices(prefix_count, 8)] if prefix_shape else [()]
    planes = [
        {"leading_indices": list(prefix),
         "label": "[" + ", ".join(map(str, prefix)) + "]" if prefix else "all",
         "matrix": _matrix(value, prefix)}
        for prefix in prefixes
    ]
    # A zero-length leading axis has no plane to select.  Expose its real final
    # two-axis dimensions with zero displayed rows, never a fabricated slice.
    matrix = planes[0]["matrix"] if planes else {
        "rows": int(value.shape[-2]), "cols": int(value.shape[-1]),
        "row_indices": [], "col_indices": [], "values": [],
        "axes": [value.ndim - 2, value.ndim - 1], "leading_indices": [],
        "sampled": False, "description": "空张量：前置轴长度为零，没有可显示的真实切片。",
    }
    summary = {
        "shape": list(value.shape), "dtype": value.dtype.name,
        "nbytes": int(value.nbytes), "numel": int(value.size),
        "finite_count": finite_count,
        "min": _json_value(finite_values.min()) if finite_values is not None else None,
        "max": _json_value(finite_values.max()) if finite_values is not None else None,
        "statistics_scope": "finite elements of the entire constant array",
        "preview": _json_value(value.reshape(-1)[:8].tolist()),
        "matrix": matrix, "planes": planes,
        "plane_count": int(prefix_count), "sampled_plane_count": len(planes),
    }
    if value.ndim >= 2:
        # Explicit dimensions also handle an empty leading or trailing axis;
        # reshape(rows, -1) would be ambiguous when rows == 0.
        flat = value.reshape(value.shape[0], math.prod(value.shape[1:]))
        flattened = _matrix(flat)
        flattened.pop("leading_indices")
        flattened["axes"] = [0, f"flatten(1..{value.ndim - 1})"]
        flattened["description"] = (
            "行对应原张量第 0 维，列按行主序展平其余维度。"
            "均为真实常量值；行列号是展平后原始索引，每轴最多均匀采样 16 项。"
        )
        summary["flattened_matrix"] = flattened
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_link(path: Path, output: Path) -> str:
    try:
        relative = os.path.relpath(path, output.parent).replace(os.sep, "/")
    except ValueError:  # Windows files on different drives cannot be relative.
        return path.as_uri()
    return quote(relative, safe="/")


def _constant_aliases(graph: Graph) -> dict[str, dict[str, Any]]:
    """Follow only exact Identity chains; never evaluate or rewrite the graph.

    Exporters commonly share LayerNorm weights through Identity nodes.  Their
    output is a proven copy of a stored array, so the viewer can show that array
    with an explicit alias path.  Arbitrary constant expressions and activation
    identities remain ordinary tensors with unavailable runtime values.
    """
    aliases: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op != "Identity" or len(node.inputs) != 1 or len(node.outputs) != 1:
            continue
        source = node.inputs[0]
        if source in graph.constants:
            aliases[node.outputs[0]] = {"source": source, "via": [node.id]}
        elif source in aliases:
            previous = aliases[source]
            aliases[node.outputs[0]] = {"source": previous["source"],
                                       "via": [*previous["via"], node.id]}
    return aliases


def build_payload(input_dir: str | Path, output_file: str | Path) -> dict[str, Any]:
    """Load and validate genuine inspection artifacts; keep embedded data small."""
    source = Path(input_dir).resolve()
    output = Path(output_file).resolve()
    graph_path = source / "canonical.graph.json"
    constants_path = source / "constants.npz"
    serialized = json.loads(graph_path.read_text(encoding="utf-8"))
    with np.load(constants_path, allow_pickle=False) as archive:
        constants = {name: archive[name] for name in serialized.get("constants", {})}
    graph = Graph.from_dict(serialized, external_constants=constants)
    report_path = source / "import-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    artifacts = [graph_path, constants_path, source / "graph.ir.txt", report_path]
    provenance = {
        "input_dir": str(source),
        "graph_json_sha256": _sha256(graph_path),
        "graph_ir_txt_sha256": _sha256(source / "graph.ir.txt") if (source / "graph.ir.txt").exists() else None,
        "artifact_links": {path.name: _artifact_link(path, output)
                           for path in artifacts if path.exists()},
        "data_source": "The same validated Graph JSON and real constants.npz used by graph.ir.txt; not a text parser.",
        "activation_values_available": False,
        "weight_display": "Real constant matrices, sampled to 16×16 coordinates and at most 8 leading-axis planes per tensor.",
    }
    return {"graph": graph.to_dict(include_constant_data=False), "report": report,
            "constants": {name: _constant_summary(value) for name, value in graph.constants.items()},
            "constant_aliases": _constant_aliases(graph),
            "provenance": provenance}


def render_graph_ir(input_dir: str | Path, output_file: str | Path, *,
                    template_path: str | Path | None = None,
                    layout_path: str | Path | None = None) -> dict[str, Any]:
    """Write a self-contained HTML page; no server, CDN or NPZ fetch is needed."""
    output = Path(output_file).resolve()
    payload = build_payload(input_dir, output)
    template = Path(template_path or _TEMPLATES / "graph-ir-network.html").read_text(encoding="utf-8")
    layout = Path(layout_path or _TEMPLATES / "graph-ir-layout.js").read_text(encoding="utf-8")
    for marker in ("__GRAPH_IR_DATA__", "__GRAPH_IR_LAYOUT__"):
        if template.count(marker) != 1:
            raise ValueError(f"template must contain exactly one {marker} marker")
    rendered = template.replace("__GRAPH_IR_LAYOUT__", layout).replace("__GRAPH_IR_DATA__", _script_json(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return {"output": str(output), "node_count": len(payload["graph"]["nodes"]),
            "constant_count": len(payload["constants"]), "bytes": output.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory produced by inspect_onnx_ir.py")
    parser.add_argument("--output", required=True, type=Path, help="Standalone HTML output (for example graph-network.html)")
    args = parser.parse_args(argv)
    print(json.dumps(render_graph_ir(args.input, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
