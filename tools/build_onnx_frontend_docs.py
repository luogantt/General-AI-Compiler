#!/usr/bin/env python3
"""Build the offline ONNX frontend walkthrough from a real, tiny import.

Run from any directory with the project's optional ONNX dependencies installed:
    python tools/build_onnx_frontend_docs.py

The example artifacts stay in build/onnx-frontend-walkthrough/. Only the HTML,
template and generator are source-controlled. No inference or external MLIR
toolchain is invoked. Source excerpts are copied with their actual line numbers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gac.core.ir import Graph
from gac.frontends.onnx_frontend import _onnx_module, load_onnx
from gac.targets.mlir import compile_mlir
from gac.targets.vortex.compiler import compile_vortex_graph


def excerpt(path: str, start: str, stop: str | None = None, *, lines: int = 20) -> dict:
    """Select exact source, not a hand-maintained imitation of the importer."""
    text = (ROOT / path).read_text(encoding="utf-8")
    source = text.splitlines()
    begin = next(i for i, line in enumerate(source) if start in line)
    end = next((i for i in range(begin + 1, len(source)) if stop and stop in source[i]),
               min(begin + lines, len(source)))
    return {"path": path, "start": begin + 1, "end": end,
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "text": "\n".join(source[begin:end])}


def function_source(path: str, name: str) -> dict:
    text = (ROOT / path).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return {"path": path, "start": node.lineno, "end": node.end_lineno,
            "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            "text": "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])}


def build_payload(output: Path) -> dict:
    onnx = _onnx_module()
    h, t = onnx.helper, onnx.TensorProto
    w = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([0.5, -0.5], dtype=np.float32)
    proto = h.make_model(h.make_graph([
        h.make_node("Constant", [], ["b"], name="bias_constant",
                    value=onnx.numpy_helper.from_array(b)),
        h.make_node("MatMul", ["X", "W"], ["M"], name="matmul"),
        h.make_node("Add", ["M", "b"], ["Z"], name="bias_add"),
        h.make_node("Relu", ["Z"], ["Y"], name="relu"),
    ], "frontend_demo", [h.make_tensor_value_info("X", t.FLOAT, [1, 2])],
        [h.make_tensor_value_info("Y", t.FLOAT, [1, 2])],
        initializer=[onnx.numpy_helper.from_array(w, name="W")]),
        opset_imports=[h.make_opsetid("", 17)], ir_version=8,
        producer_name="GAC frontend walkthrough")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "example.onnx"
    onnx.save_model(proto, str(path))
    graph = load_onnx(path, strict=True)
    assert [n.op for n in graph.nodes] == ["MatMul", "Add", "ReLU"]
    assert set(graph.constants) == {"W", "b"}
    np.testing.assert_array_equal(graph.constants["W"], w)
    np.testing.assert_array_equal(graph.constants["b"], b)
    inferred = onnx.shape_inference.infer_shapes(
        proto, check_type=True, strict_mode=False, data_prop=True)
    structure = graph.to_dict(include_constant_data=False)
    (output / "canonical.graph.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez(output / "constants.npz", **graph.constants)
    with np.load(output / "constants.npz", allow_pickle=False) as arrays:
        restored = Graph.from_dict(structure, external_constants=arrays)
        assert restored.to_dict() == graph.to_dict()
    mlir = compile_mlir(graph, fuse=False).source
    (output / "example.tensor.mlir").write_text(mlir, encoding="utf-8")
    vortex = compile_vortex_graph(graph, optimize=False)
    (output / "example.vortex.mlir").write_text(vortex.source, encoding="utf-8")
    summary = json.loads((ROOT / "docs/validation/onnx/bevformer-tiny-cold-2026-10-15/summary.json")
                         .read_text(encoding="utf-8"))
    frontend = "src/gac/frontends/onnx_frontend.py"
    ir = "src/gac/core/ir.py"
    return {
        "example": {"onnx_version": onnx.__version__, "ir_version": proto.ir_version,
                    "opset": 17, "file_bytes": path.stat().st_size,
                    "source": onnx.printer.to_text(proto),
                    "inferred": onnx.printer.to_text(inferred),
                    "graph": graph.to_dict(), "mlir": mlir,
                    "vortex_dispatches": len(vortex.dispatches),
                    "vortex_mlir": vortex.source,
                    "verified_with_mlir_opt": False, "device_execution": False},
        "bevformer": {"onnx": summary["onnx"],
                      "import": {k: summary["import"][k] for k in
                                 ("source_node_count", "canonical_node_count", "constant_count", "constant_bytes")}},
        "sources": {
            "load": function_source(frontend, "load_onnx"),
            "check": excerpt(frontend, "    # 阶段 1：", "    graph_proto = model.graph"),
            "weights": excerpt(frontend, "    constants: dict[str, np.ndarray[Any, Any]] = {}", "    declared_input_names"),
            "types": excerpt(frontend, "    value_specs: dict[str, TensorSpec] = {}", "    occupied_values ="),
            "spec": function_source(frontend, "_tensor_spec"),
            "constant": excerpt(frontend, '    if proto.op_type == "Constant":', "    direct ="),
            "node": function_source(ir, "Node"),
            "translate": excerpt(frontend, "            lowered, unsupported = _translate_node(", "        except _UnsupportedOnnxSemantics"),
            "direct": excerpt(frontend, "_DIRECT_OPS = {", "def _onnx_module"),
            "graph": excerpt(frontend, "    graph = Graph(", "def load_onnx"),
            "serialize": excerpt("tools/inspect_onnx_ir.py", '        phase = "write-artifacts"', "    except Exception as exc:"),
            "mlir": function_source("src/gac/targets/mlir.py", "compile_mlir"),
            "emitter": excerpt("src/gac/targets/mlir.py", "        for name in sorted(self.graph.constants):", "def compile_mlir"),
            "vortex": function_source("src/gac/targets/vortex/compiler.py", "compile_vortex_graph"),
            "toolchain": excerpt("src/gac/toolchains/mlir.py", "def verify_mlir(", lines=20),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check whether the committed HTML is current")
    args = parser.parse_args()
    payload = build_payload(ROOT / "build/onnx-frontend-walkthrough")
    template = (ROOT / "tools/templates/onnx-frontend-walkthrough.html").read_text(encoding="utf-8")
    assert template.count("__PAYLOAD__") == 1
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    rendered = template.replace("__PAYLOAD__", data)
    destination = ROOT / "docs/onnx-frontend-walkthrough.html"
    if args.check:
        if not destination.exists() or destination.read_text(encoding="utf-8") != rendered:
            print("ONNX walkthrough needs regeneration")
            return 1
    else:
        destination.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"{'Checked' if args.check else 'Wrote'} {destination}")
    print("Real ONNX import + constant round-trip + both MLIR emitters passed; no mlir-opt/device execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
