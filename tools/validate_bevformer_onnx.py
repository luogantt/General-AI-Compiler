#!/usr/bin/env python3
"""Compare the exported cold-frame ONNX outputs with saved real PyTorch outputs.

Standard ONNX executes with ONNX Runtime. Base's explicit custom operators
require a native backend: --actual-outputs compares that backend's saved NPZ,
without claiming that this script executed it. This checks a fixed fixture,
not dataset accuracy or device performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_path(directory: Path, model: Path | None) -> Path:
    if model is not None:
        return model
    report_path = directory / "export-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        filename = report.get("onnx", {}).get("file")
        if filename:
            return directory / filename
    candidates = list(directory.glob("*-cold.onnx"))
    if len(candidates) != 1:
        raise ValueError("Specify --model: expected one cold ONNX or an export-report ONNX path")
    return candidates[0]


def validate(directory: Path, *, threads: int = 4,
             rtol: float = 1e-3, atol: float = 1e-4,
             model: Path | None = None, actual_outputs: Path | None = None) -> dict:
    report = {
        "success": False,
        "rtol": rtol, "atol": atol,
        "execution": "compare supplied backend outputs" if actual_outputs else "onnxruntime CPU",
        "scope": "Single seeded image fixture, fixed calibration/CAN-bus, cold raw heads; no dataset accuracy or performance claim.",
    }
    try:
        if threads < 1 or rtol < 0 or atol < 0:
            raise ValueError("threads must be positive and tolerances nonnegative")
        model = _model_path(directory, model)
        graph = onnx.load(model, load_external_data=False).graph
        report["model"] = str(model)
        report["model_sha256"] = _sha256(model)
        report["reference_sha256"] = _sha256(directory / "pytorch-outputs.npz")
        report["images_sha256"] = _sha256(directory / "images.npy")
        names = [item.name for item in graph.output]
        if actual_outputs is None:
            custom = sorted({f"{node.domain}::{node.op_type}" for node in graph.node
                             if node.domain not in ("", "ai.onnx", "ai.onnx.ml")})
            if custom:
                raise ValueError("Stock ONNX Runtime cannot execute these project custom ops: "
                                 + ", ".join(custom)
                                 + ". Run the native MLIR backend and use --actual-outputs with its NPZ.")
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = ort.InferenceSession(str(model), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            images = np.load(directory / "images.npy", allow_pickle=False)
            values = session.run(names, {"images": images})
            actual = dict(zip(names, values))
            report.update(onnxruntime=ort.__version__, providers=session.get_providers())
        else:
            with np.load(actual_outputs, allow_pickle=False) as saved:
                actual = {key: saved[key] for key in saved.files}
            report["actual_outputs"] = str(actual_outputs)
            report["actual_outputs_sha256"] = _sha256(actual_outputs)
            report["provenance_limit"] = "NPZ comparison alone does not prove which compiler/backend produced the supplied values."
        report.update(_compare(directory, actual, names=names, rtol=rtol, atol=atol))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    filename = "supplied-outputs-correctness.json" if actual_outputs else "onnx-correctness.json"
    (directory / filename).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _compare(directory: Path, actual: dict, *, names: list[str], rtol: float, atol: float) -> dict:
    comparisons = {}
    with np.load(directory / "pytorch-outputs.npz", allow_pickle=False) as expected:
        if set(expected.files) != set(names) or set(actual) != set(names):
            raise ValueError("ONNX, actual output and saved PyTorch reference names must match exactly")
        for name in names:
            value, reference = actual[name], expected[name]
            same_shape = value.shape == reference.shape
            same_dtype = value.dtype == reference.dtype
            finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
            passed = same_shape and same_dtype and finite and np.allclose(value, reference, rtol=rtol, atol=atol)
            error = np.abs(value.astype(np.float64) - reference.astype(np.float64)) if same_shape and finite else None
            comparisons[name] = {
                "shape": list(value.shape), "expected_shape": list(reference.shape),
                "dtype": str(value.dtype), "expected_dtype": str(reference.dtype),
                "finite": finite, "allclose": bool(passed),
                "max_abs_error": float(error.max()) if error is not None and error.size else None,
                "mean_abs_error": float(error.mean()) if error is not None and error.size else None,
            }
    return {
        "success": bool(comparisons) and all(item["allclose"] for item in comparisons.values()),
        "outputs": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--model", type=Path, help="Default: ONNX path from export-report.json")
    parser.add_argument("--actual-outputs", type=Path,
                        help="Compare separately executed backend NPZ; this option does not execute the backend")
    args = parser.parse_args()
    report = validate(args.directory, threads=args.threads, rtol=args.rtol, atol=args.atol,
                      model=args.model, actual_outputs=args.actual_outputs)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success"] else 1)
