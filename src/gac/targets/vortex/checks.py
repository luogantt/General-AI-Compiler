"""Standalone output verification shipped with Vortex whole-graph artifacts."""


def emit_output_verifier() -> str:
    return '''#!/usr/bin/env python3
"""Compare device-produced raw outputs with optional offline golden tensors."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-4)
    parser.add_argument("--report")
    args = parser.parse_args()
    if not np.isfinite(args.rtol) or not np.isfinite(args.atol) or args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be finite and nonnegative")
    root = Path(args.data_dir)
    contract = json.loads((root / "10-runtime.json").read_text())
    results = []
    for index, record in enumerate(contract["outputs"]):
        actual_path = root / record["file"]
        golden_path = root / f"goldens/golden_{index:04d}.bin"
        dtype = np.dtype(record["dtype"]).newbyteorder("<")
        expected_bytes = record["size_bytes"]
        for path in (actual_path, golden_path):
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"missing or incorrectly sized tensor: {path}")
        actual = np.fromfile(actual_path, dtype=dtype).reshape(record["shape"])
        expected = np.fromfile(golden_path, dtype=dtype).reshape(record["shape"])
        if dtype.kind in "biu":
            passed = bool(np.array_equal(actual, expected))
        else:
            passed = bool(np.allclose(actual, expected, rtol=args.rtol, atol=args.atol, equal_nan=False))
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        # Equal signed infinities are valid results of padded MaxPool, for example.
        difference = np.where(actual == expected, 0.0, difference)
        error = float(difference.max()) if difference.size else 0.0
        results.append({"name": record["name"], "passed": passed,
                        "max_abs_error": error if np.isfinite(error) else None})
    report = {"passed": all(row["passed"] for row in results), "rtol": args.rtol,
              "atol": args.atol, "outputs": results}
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\\n")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
'''
