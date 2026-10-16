#!/usr/bin/env python3
"""Compile base-operator fixtures through native C++ MLIR and execute machine code.

The reference fixture generator needs PyTorch/torchvision; this runner only needs
NumPy. It never computes an operator instead of calling the generated kernel.
Optional SimX checks go through VOLT and the separate C++ Vortex runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from run_tests import execute, run
from run_full_model import load_plan


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_simx(args, folder):
    artifact = folder / "vortex-simx"
    environment = os.environ.copy()
    environment.update(GAC_NATIVE_BIN=str(args.compiler.resolve().parent),
                       LLVM_PATH=str(args.llvm_bin.resolve().parent),
                       VORTEX_BUILD=str(args.vortex_build.resolve()))
    result = run(["bash", args.compile_script.resolve(), folder / "model.onnx", artifact],
                 env=environment, timeout=args.timeout)
    (artifact / "build.log").write_text(result.stdout + result.stderr)
    plan = load_plan(artifact / "plan.json")
    with np.load(folder / "inputs.npz", allow_pickle=False) as inputs:
        for index, tensor in enumerate(plan["tensors"]):
            if tensor["storage"] == "input":
                np.ascontiguousarray(inputs[tensor["name"]]).tofile(artifact / f"input_{index}.bin")
    environment["VORTEX_DRIVER"] = "simx"
    environment["LD_LIBRARY_PATH"] = str(args.vortex_build.resolve() / "sw/runtime") + ":" + environment.get("LD_LIBRARY_PATH", "")
    output = artifact / "outputs"
    result = run([args.runner.resolve(), artifact / "plan.json", artifact / "kernel.vxbin", artifact, output],
                 env=environment, cwd=artifact, timeout=args.timeout)
    (artifact / "run.log").write_text(result.stdout + result.stderr)
    tensors = {item["name"]: item for item in plan["tensors"]}
    errors = {}
    with np.load(folder / "expected.npz", allow_pickle=False) as reference:
        for index, name in enumerate(plan["outputs"]):
            tensor = tensors[name]
            actual = np.fromfile(output / f"output_{index}.bin", dtype=tensor["dtype"]).reshape(tensor["shape"])
            np.testing.assert_allclose(actual, reference[name], rtol=2e-5, atol=2e-5,
                                       err_msg=f"SimX {folder.name}: {name}")
            errors[name] = float(np.max(np.abs(actual.astype(np.float64) - reference[name].astype(np.float64)), initial=0))
    return {"passed": True, "physical_device_execution": False, "max_absolute_errors": errors,
            "live_buffer_overlap_check": "passed",
            "vxbin_sha256": sha256(artifact / "kernel.vxbin")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--llvm-bin", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--compile-script", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--vortex-build", type=Path)
    parser.add_argument("--simx-case", action="append", dest="simx_cases",
                        help="Run selected positive cases in SimX; repeat as needed")
    args = parser.parse_args()
    if args.simx_cases and not all((args.compile_script, args.runner, args.vortex_build)):
        parser.error("--simx-case needs --compile-script, --runner and --vortex-build")
    args.fixtures = args.fixtures.resolve()
    specification = json.loads((args.fixtures / "cases.json").read_text())
    cases = specification["cases"]
    available = {case["name"] for case in cases}
    if set(args.selected or []) - available or set(args.simx_cases or []) - available:
        parser.error("Unknown case name")
    if args.selected:
        cases = [case for case in cases if case["name"] in args.selected]
    if set(args.simx_cases or []) - {case["name"] for case in cases if case["valid"]}:
        parser.error("SimX cases must be selected positive cases")
    report = {"scope": "C++ ONNX importer + MLIR passes + actual generated native CPU kernels",
              "python_operator_fallback": False, "compiler": str(args.compiler.resolve()),
              "compiler_sha256_at_start": sha256(args.compiler),
              "reference_versions": specification["versions"], "cases": []}
    report_path = args.fixtures / ("selected-results.json" if args.selected else "results.json")
    for case in cases:
        folder = args.fixtures / case["name"]
        entry = {"name": case["name"], "kind": case["kind"], "reference": case["reference"],
                 "model_sha256": sha256(folder / "model.onnx"), "passed": False}
        start = time.perf_counter()
        llvm_mlir, manifest = folder / "llvm.mlir", folder / "plan.json"
        constants = folder / "constants"
        command = [args.compiler.resolve(), folder / "model.onnx", "--target=native", "--emit=llvm",
                   "-o", llvm_mlir, "--manifest", manifest, "--constants-dir", constants]
        try:
            if not case["valid"]:
                rejected = run(command, succeeds=False, timeout=args.timeout)
                diagnostic = rejected.stderr.strip()
                assert diagnostic, "Rejected inputs need a diagnostic"
                assert "no registered GAC import mapping" not in diagnostic, diagnostic
                assert "unsupported ONNX domain" not in diagnostic, diagnostic
                assert "no registered import contract" not in diagnostic, diagnostic
                assert re.search(case["expected_rejection_regex"], diagnostic, re.IGNORECASE), diagnostic
                entry["expected_rejection"] = True
                entry["diagnostic"] = diagnostic
            else:
                run(command, timeout=args.timeout)
                plan = load_plan(manifest)
                entry["arena_bytes"] = plan["arena_bytes"]
                entry["live_buffer_overlap_check"] = "passed"
                if "maximum_arena_bytes" in case:
                    assert plan["arena_bytes"] <= case["maximum_arena_bytes"], (
                        f"Arena {plan['arena_bytes']} exceeds fixture memory budget {case['maximum_arena_bytes']}")
                llvm_ir = folder / "kernel.ll"
                run([args.llvm_bin / "mlir-translate", "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir], timeout=args.timeout)
                run([args.llvm_bin / "opt", "-passes=verify", "-disable-output", llvm_ir], timeout=args.timeout)
                library = folder / "kernels.so"
                run([args.llvm_bin / "clang", "-shared", "-fPIC", "-O2", llvm_ir, "-lm", "-o", library], timeout=args.timeout)
                entry.update(execute(manifest, constants, library, folder))
                entry["library_sha256"] = sha256(library)
                if case["name"] in (args.simx_cases or []):
                    entry["simx"] = run_simx(args, folder)
            entry["passed"] = True
        except (AssertionError, subprocess.TimeoutExpired, OSError, ValueError) as failure:
            entry["failure"] = str(failure)
        entry["test_wall_seconds"] = time.perf_counter() - start
        report["cases"].append(entry)
        report["passed"] = all(item["passed"] for item in report["cases"])
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {case['name']}", flush=True)
        if not entry["passed"]:
            print(entry["failure"], flush=True)
    report["compiler_sha256_at_end"] = sha256(args.compiler)
    report["compiler_stable_during_run"] = report["compiler_sha256_at_start"] == report["compiler_sha256_at_end"]
    report["passed"] = report["passed"] and report["compiler_stable_during_run"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["compiler_stable_during_run"]:
        print("Compiler changed during testing: this run is not a reproducible regression result.", flush=True)
    print(f"{sum(case['passed'] for case in report['cases'])}/{len(report['cases'])} passed; {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
