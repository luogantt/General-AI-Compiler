#!/usr/bin/env python3
"""Compare the exported cold-frame ONNX outputs with saved real PyTorch outputs.

Requires numpy and onnxruntime. This checks one fixed input/metadata fixture;
it does not measure dataset accuracy or device performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def validate(directory: Path, *, threads: int = 4,
             rtol: float = 1e-3, atol: float = 1e-4) -> dict:
    model = directory / "bevformer-tiny-cold.onnx"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "success": False,
        "model_sha256": digest.hexdigest(), "onnxruntime": ort.__version__,
        "rtol": rtol, "atol": atol,
        "scope": "Single seeded image fixture, fixed calibration/CAN-bus, cold raw heads; no dataset accuracy or performance claim.",
    }
    try:
        report.update(_compare(directory, model, threads=threads, rtol=rtol, atol=atol))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    (directory / "onnx-correctness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _compare(directory: Path, model: Path, *, threads: int, rtol: float, atol: float) -> dict:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    images = np.load(directory / "images.npy", allow_pickle=False)
    actual = session.run(None, {"images": images})
    comparisons = {}
    with np.load(directory / "pytorch-outputs.npz", allow_pickle=False) as expected:
        if set(expected.files) != {output.name for output in session.get_outputs()}:
            raise ValueError("ONNX output names do not match the saved PyTorch reference")
        for output, value in zip(session.get_outputs(), actual):
            reference = expected[output.name]
            same_shape = value.shape == reference.shape
            finite = bool(np.isfinite(value).all() and np.isfinite(reference).all())
            passed = same_shape and finite and np.allclose(value, reference, rtol=rtol, atol=atol)
            error = np.abs(value.astype(np.float64) - reference.astype(np.float64)) if same_shape and finite else None
            comparisons[output.name] = {
                "shape": list(value.shape), "expected_shape": list(reference.shape),
                "finite": finite, "allclose": bool(passed),
                "max_abs_error": float(error.max()) if error is not None else None,
                "mean_abs_error": float(error.mean()) if error is not None else None,
            }
    return {
        "success": bool(comparisons) and all(item["allclose"] for item in comparisons.values()),
        "providers": session.get_providers(),
        "outputs": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    report = validate(args.directory, threads=args.threads, rtol=args.rtol, atol=args.atol)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["success"] else 1)
