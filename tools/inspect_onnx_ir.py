#!/usr/bin/env python3
"""Inspect an ONNX file through the real GAC frontend, without device execution.

Example::

    python tools/inspect_onnx_ir.py --model build/bevformer/model.onnx \
        --output build/bevformer/inspection

The JSON contains the serializable Canonical Graph; actual constant arrays live
in ``constants.npz``.  The text and HTML views deliberately call this a Python
Graph IR, not MLIR: this command does not invoke ``mlir-opt`` or a GPU backend.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gac.core.ir import Graph, TensorSpec
from gac.frontends.onnx_frontend import _onnx_module, load_onnx


_ARTIFACTS = (
    "canonical.graph.json", "constants.npz", "constants.index.json",
    "source.onnx.txt", "graph.ir.txt", "import-report.json", "index.html",
)


def _json_value(value: Any) -> Any:
    """Produce small, safe display values; tensor payloads get summaries."""

    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _array_summary(array: np.ndarray, *, preview: int = 8) -> dict[str, Any]:
    value = np.asarray(array)
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.name,
        "nbytes": int(value.nbytes),
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B") if value.nbytes else b"").hexdigest(),
        "preview": [_json_value(item) for item in value.reshape(-1)[:preview]],
        "preview_count": min(preview, int(value.size)),
        "numel": int(value.size),
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_constants(path: Path, constants: dict[str, np.ndarray]) -> None:
    # np.savez(path, **constants) cannot represent a perfectly valid tensor named
    # "file" because it collides with that function's first parameter.  Writing
    # the standard NPZ format directly keeps every original ONNX tensor name.
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, value in constants.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)


def _qualified_op(node: Any) -> str:
    return f"{node.domain or 'ai.onnx'}::{node.op_type}"


def _onnx_attribute(attribute: Any, onnx: Any) -> Any:
    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, onnx.TensorProto):
        # Tensor attributes may contain large payloads.  The source listing shows
        # their type/shape; the imported array is inspected through constants.npz.
        return {
            "onnx_tensor": True,
            "dtype": onnx.TensorProto.DataType.Name(value.data_type),
            "shape": list(value.dims),
            "name": value.name,
        }
    if isinstance(value, onnx.GraphProto):
        return {"nested_graph": value.name, "node_count": len(value.node)}
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], onnx.TensorProto):
        return [
            {"onnx_tensor": True, "shape": list(item.dims),
             "dtype": onnx.TensorProto.DataType.Name(item.data_type)}
            for item in value
        ]
    return _json_value(value)


def _source_model(model: Any, onnx: Any) -> dict[str, Any]:
    def value_info(value: Any) -> dict[str, str]:
        return {"name": value.name, "type": onnx.helper.printable_type(value.type)}

    return {
        "name": model.graph.name,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "ir_version": int(model.ir_version),
        "opsets": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "inputs": [value_info(value) for value in model.graph.input],
        "outputs": [value_info(value) for value in model.graph.output],
        "value_info": [value_info(value) for value in model.graph.value_info],
        "initializers": [
            {"name": value.name, "shape": list(value.dims),
             "dtype": onnx.TensorProto.DataType.Name(value.data_type)}
            for value in model.graph.initializer
        ],
        "nodes": [
            {
                "index": index,
                "name": node.name or f"{node.op_type}_{index}",
                "op": _qualified_op(node),
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attrs": {
                    attribute.name: _onnx_attribute(attribute, onnx)
                    for attribute in node.attribute
                },
            }
            for index, node in enumerate(model.graph.node)
        ],
    }


def _source_text(source: dict[str, Any]) -> str:
    """A readable ONNX listing with named tensors, never weight payloads."""

    lines = [
        "# Source ONNX Graph (not GAC IR or MLIR)",
        f"# graph: {source['name']}",
        f"# opsets: {json.dumps(source['opsets'], ensure_ascii=False)}",
        "# Empty optional ONNX inputs are printed as <omitted>.",
        "",
    ]
    for kind in ("inputs", "initializers"):
        for value in source[kind]:
            detail = value.get("type", f"{value.get('dtype')}{value.get('shape')}")
            lines.append(f"{kind[:-1]} %{json.dumps(value['name'], ensure_ascii=False)} : {detail}")
    lines.append("")
    for node in source["nodes"]:
        refs = lambda values: ", ".join(
            "%" + json.dumps(name, ensure_ascii=False) if name else "<omitted>"
            for name in values
        )
        lines.append(
            f"{refs(node['outputs'])} = {node['op']}({refs(node['inputs'])}) "
            f"{json.dumps(node['attrs'], ensure_ascii=False, sort_keys=True)} "
            f"  # {node['index']}: {node['name']}"
        )
    lines.extend(["", "return " + ", ".join("%" + json.dumps(item["name"], ensure_ascii=False) for item in source["outputs"])])
    return "\n".join(lines) + "\n"


def _spec_text(spec: TensorSpec | None) -> str:
    if spec is None:
        return "type unknown (not inferred)"
    dims = ",".join("?" if dim is None else str(dim) for dim in spec.shape)
    return f"TensorSpec(shape=[{dims}], dtype={spec.dtype}, layout={spec.layout})"


def _graph_text(graph: Graph, constants: dict[str, Any]) -> str:
    ref = lambda name: "%" + json.dumps(name, ensure_ascii=False)
    lines = [
        "# GAC Canonical Graph: human-readable SSA-style display",
        "# This is NOT MLIR syntax and is not an executable model.",
        "# Nodes reference names. TensorSpec describes types; it stores no tensor data.",
        "# Real weights: Graph.constants[name] -> NumPy ndarray -> constants.npz[name].",
        f"graph {json.dumps(graph.name, ensure_ascii=False)} {{",
    ]
    for name, spec in graph.inputs.items():
        lines.append(f"  input {ref(name)} : {_spec_text(spec)}")
    for name, value in constants.items():
        lines.append(
            f"  constant {ref(name)} : {_spec_text(graph.value_specs.get(name))} "
            f"= constants.npz[{json.dumps(name, ensure_ascii=False)}] "
            f"# {value['nbytes']} bytes, preview={json.dumps(value['preview'], ensure_ascii=False)}"
        )
    lines.append("")
    for node in graph.nodes:
        lines.append(
            f"  {', '.join(ref(name) for name in node.outputs)} = {node.op}"
            f"({', '.join(ref(name) for name in node.inputs)}) "
            f"{json.dumps(_json_value(node.attrs), ensure_ascii=False, sort_keys=True)}"
        )
        lines.append(
            f"    # id={json.dumps(node.id, ensure_ascii=False)} "
            f"source_nodes={json.dumps(node.source_nodes, ensure_ascii=False)}"
        )
        for name in node.outputs:
            lines.append(f"    # {ref(name)} : {_spec_text(graph.value_specs.get(name))}")
    lines.extend(["", "  return " + ", ".join(ref(name) for name in graph.outputs), "}"])
    return "\n".join(lines) + "\n"


def _html(output: Path, report: dict[str, Any], source: dict[str, Any] | None,
          graph: Graph | None, constants: dict[str, Any]) -> None:
    data = {
        "report": report,
        "source": source,
        "graph": graph.to_dict(include_constant_data=False) if graph else None,
        "constants": constants,
    }
    payload = json.dumps(_json_value(data), ensure_ascii=False, allow_nan=False)
    # Node names and attributes are untrusted model data.  Prevent closing the
    # JSON script element; the viewer renders their values with textContent.
    for character, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                    ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(character, replacement)
    template = Path(__file__).with_name("templates").joinpath("onnx-ir-inspector.html")
    output.write_text(template.read_text(encoding="utf-8").replace("__INSPECTION_DATA__", payload), encoding="utf-8")


def inspect_onnx_ir(
    model_path: str | Path,
    output_dir: str | Path,
    *,
    export_report: str | Path | None = None,
) -> dict[str, Any]:
    """Write frontend artifacts, recording genuine failures instead of faking IR.

    The diagnostic import uses ``strict=False`` so unknown ONNX nodes remain
    visible.  A separate strict import tests the actual frontend contract; its
    success still says nothing about any particular device backend's coverage.
    """

    model_path, output_dir = Path(model_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in _ARTIFACTS:
        stale = output_dir / filename
        if stale.is_file():
            stale.unlink()
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "frontend": "gac.frontends.onnx_frontend.load_onnx",
        "diagnostic_import": {"strict": False, "success": False},
        "strict_import": {"strict": True, "success": False, "attempted": False},
        "mlir": {"generated": False, "reason": "This tool inspects frontend Graph IR only; no MLIR lowering or verification is performed."},
        "device_execution": False,
    }
    source, graph, constants = None, None, {}
    phase = "read-source"
    try:
        onnx = _onnx_module()
        report["onnx_version"] = onnx.__version__
        report["model_sha256"] = _file_hash(model_path)
        if export_report is not None:
            phase = "validate-export-report"
            export_path = Path(export_report).resolve()
            export_details = json.loads(export_path.read_text(encoding="utf-8"))
            expected_hash = export_details.get("onnx", {}).get("sha256")
            if not isinstance(expected_hash, str) or expected_hash.lower() != report["model_sha256"]:
                raise ValueError("export report ONNX SHA-256 does not match the inspected model")
            report["export_report"] = {
                "path": str(export_path),
                "model_sha256_verified": True,
                "details": export_details,
            }
        phase = "read-source"
        model = onnx.load(str(model_path), load_external_data=False)
        source = _source_model(model, onnx)
        del model
        report["source_operator_counts"] = dict(sorted(Counter(node["op"] for node in source["nodes"]).items()))
        report["source_node_count"] = len(source["nodes"])
        (output_dir / "source.onnx.txt").write_text(_source_text(source), encoding="utf-8")

        phase = "diagnostic-import"
        graph = load_onnx(model_path, strict=False)
        report["diagnostic_import"]["success"] = True
        report["canonical_node_count"] = len(graph.nodes)
        report["canonical_operator_counts"] = dict(sorted(Counter(node.op for node in graph.nodes).items()))
        report["unsupported_nodes"] = graph.metadata.get("import", {}).get("unsupported_nodes", [])
        report["input_count"] = len(graph.inputs)
        report["output_count"] = len(graph.outputs)
        report["constant_count"] = len(graph.constants)
        report["constant_bytes"] = sum(int(value.nbytes) for value in graph.constants.values())

        phase = "strict-import"
        report["strict_import"]["attempted"] = True
        try:
            strict_graph = load_onnx(model_path, strict=True)
            report["strict_import"].update(success=True, canonical_node_count=len(strict_graph.nodes))
            # Prefer additional types validated by the strict pass, while keeping
            # the diagnostic graph and its import metadata faithful to its call.
            graph.value_specs.update(strict_graph.value_specs)
            del strict_graph
        except Exception as exc:
            report["strict_import"].update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())

        report["values_without_inferred_types"] = sorted(
            {name for node in graph.nodes for name in node.outputs} - graph.value_specs.keys()
        )

        phase = "write-artifacts"
        constants = {name: _array_summary(value) for name, value in graph.constants.items()}
        _write_json(output_dir / "canonical.graph.json", graph.to_dict(include_constant_data=False))
        _write_constants(output_dir / "constants.npz", graph.constants)
        _write_json(output_dir / "constants.index.json", constants)
        (output_dir / "graph.ir.txt").write_text(_graph_text(graph, constants), encoding="utf-8")
        report["graph_roundtrip"] = {"success": False}
        with np.load(output_dir / "constants.npz", allow_pickle=False) as arrays:
            restored = Graph.from_dict(json.loads((output_dir / "canonical.graph.json").read_text(encoding="utf-8")), external_constants=arrays)
            report["graph_roundtrip"] = {"success": True, "node_count": len(restored.nodes), "constant_count": len(restored.constants)}
            del restored
    except Exception as exc:
        report["error"] = {"phase": phase, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["success"] = bool(report["diagnostic_import"]["success"] and "error" not in report)
    report["artifacts"] = [name for name in _ARTIFACTS if (output_dir / name).is_file() or name in {"import-report.json", "index.html"}]
    _write_json(output_dir / "import-report.json", report)
    _html(output_dir / "index.html", report, source, graph, constants)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Actual ONNX model, including any external tensor data.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for inspectable Graph IR, constants, report and HTML.")
    parser.add_argument("--export-report", type=Path, help="Optional export provenance/scope JSON; its onnx.sha256 must match --model.")
    args = parser.parse_args(argv)
    report = inspect_onnx_ir(args.model, args.output, export_report=args.export_report)
    print(json.dumps({key: report.get(key) for key in ("success", "source_node_count", "canonical_node_count", "constant_count", "constant_bytes", "strict_import", "error", "output_dir")}, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
