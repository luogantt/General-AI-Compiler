#!/usr/bin/env python3
"""Run normal native regression checks, then exact Range constant ABI checks.

Pass the same arguments as run_tests.py. This extra check deliberately requires
the actual machine-code manifest's output constant, including its byte payload,
rather than inferring success from a pretty-printed MLIR literal or allclose.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="selected_cases")
    args, _ = parser.parse_known_args()
    directory = args.fixtures.resolve()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    compiler = args.compiler.resolve()
    provenance = {"compiler": str(compiler), "compiler_sha256_before": digest(compiler),
                  "runner_sha256": digest(Path(__file__)),
                  "fixture_cases_sha256": digest(directory / "cases.json"),
                  "command": [sys.executable, *sys.argv]}
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_tests.py")),
                                 *sys.argv[1:]], check=False)
        provenance["regression_exit_code"] = result.returncode
    finally:
        provenance["compiler_sha256_after"] = digest(compiler)
        provenance["compiler_unchanged"] = provenance["compiler_sha256_before"] == provenance["compiler_sha256_after"]
        (directory / "compiler-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                             encoding="utf-8")
    assert provenance["compiler_unchanged"], "Compiler binary changed during validation"
    result.check_returncode()
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))["cases"]
    exact = []
    for case in cases:
        if args.selected_cases and case["name"] not in args.selected_cases:
            continue
        if not case.get("check_constant_payload_exact"):
            continue
        folder = directory / case["name"]
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        tensors = {item["name"]: item for item in plan["tensors"]}
        with np.load(folder / "expected.npz", allow_pickle=False) as expected:
            for name in plan["outputs"]:
                item = tensors[name]
                assert item["storage"] == "constant", (case["name"], item)
                path = Path(item["file"])
                if not path.is_absolute():
                    candidates = (folder / path, folder / "constants" / path,
                                  folder / "constants" / path.name)
                    path = next((p for p in candidates if p.is_file()), candidates[0])
                data = np.fromfile(path, dtype=np.dtype(item["dtype"])).reshape(item["shape"])
                reference = expected[name]
                assert data.dtype == reference.dtype and data.shape == reference.shape
                assert data.tobytes() == reference.tobytes(), (case["name"], data, reference)
        exact.append({"name": case["name"], "constant_payload_bit_exact": True})
        print(f"EXACT {case['name']}", flush=True)
    (directory / "exact-results.json").write_text(json.dumps({"cases": exact}, indent=2) + "\n",
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
