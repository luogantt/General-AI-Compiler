#!/usr/bin/env python3
"""Verify native ONNX -> MLIR -> VOLT machine-code artifacts on small fixtures.

Python only orchestrates test commands and checks their outputs. The ONNX
importer, graph passes, device lowering and dispatch planning are C++.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from run_tests import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--compile-script", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--vortex-build", type=Path)
    parser.add_argument("--device-case", action="append", dest="device_cases",
                        help="Case to run in SimX; defaults to MatMul, Conv and Softmax")
    parser.add_argument("--case", action="append", dest="selected_cases",
                        help="Restrict object validation to named cases; repeat as needed")
    args = parser.parse_args()
    if any((args.compile_script, args.runner, args.vortex_build)) and not all(
            (args.compile_script, args.runner, args.vortex_build)):
        parser.error("SimX execution needs --compile-script, --runner and --vortex-build together")
    fixtures, toolchain = args.fixtures.resolve(), args.llvm_bin.resolve()
    cases = json.loads((fixtures / "cases.json").read_text(encoding="utf-8"))["cases"]
    if args.selected_cases:
        unknown = set(args.selected_cases) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.selected_cases]
    report = {"target": "vortex", "scope": "native MLIR to VOLT RV32 object validation",
              "physical_device_execution": False, "simx_execution_requested": bool(args.runner),
              "cases": []}
    selected = set(args.device_cases or ["matmul_bias_relu", "conv_bias_relu", "softmax"])
    for case in cases:
        if not case["valid"] or case.get("reject_device"):
            continue
        directory = fixtures / case["name"]
        artifact = directory / "vortex"
        artifact.mkdir(exist_ok=True)
        llvm_mlir, llvm_ir = artifact / "kernel.mlir", artifact / "kernel.ll"
        object_file, manifest = artifact / "kernel.o", artifact / "plan.json"
        run([args.compiler, directory / "model.onnx", "--target=vortex", "--emit=llvm",
             "-o", llvm_mlir, "--manifest", manifest,
             "--constants-dir", artifact / "constants"])
        run([toolchain / "mlir-translate", "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir])
        run([toolchain / "opt", "-passes=verify", "-disable-output", llvm_ir])
        run([toolchain / "clang++", "--target=riscv32-unknown-elf",
             "-Xclang", "-target-feature", "-Xclang", "+xvortex",
             "-Xclang", "-target-feature", "-Xclang", "+zicond",
             "-mllvm", "-disable-loop-idiom-all", "-march=rv32imaf", "-mabi=ilp32f",
             "-O3", "-mcmodel=medany", "-fdata-sections", "-ffunction-sections",
             "-c", llvm_ir, "-o", object_file])
        header = run([toolchain / "llvm-readelf", "-h", object_file]).stdout
        symbols = run([toolchain / "llvm-readelf", "-s", "-W", object_file]).stdout
        sections = run([toolchain / "llvm-readelf", "-S", "-W", object_file]).stdout
        assert "ELF32" in header and "RISC-V" in header
        plan = json.loads(manifest.read_text(encoding="utf-8"))
        assert plan["target"] == "vortex"
        for dispatch in plan["dispatches"]:
            assert "__vx_kentry_" + dispatch["kernel"] in symbols, dispatch
            assert ".vx_entry." + dispatch["kernel"] in sections, dispatch
        (artifact / "object-header.txt").write_text(header, encoding="utf-8")
        (artifact / "symbols.txt").write_text(symbols, encoding="utf-8")
        (artifact / "sections.txt").write_text(sections, encoding="utf-8")
        entry = {"name": case["name"], "passed": True,
                 "dispatches": len(plan["dispatches"]),
                 "object_bytes": object_file.stat().st_size}
        if args.runner and case["name"] in selected:
            device_artifact = directory / "vortex-simx"
            environment = os.environ.copy()
            environment.update(GAC_NATIVE_BIN=str(args.compiler.resolve().parent),
                               LLVM_PATH=str(toolchain.parent),
                               VORTEX_BUILD=str(args.vortex_build.resolve()))
            compiled = run(["bash", args.compile_script, directory / "model.onnx", device_artifact],
                           env=environment)
            (device_artifact / "build.log").write_text(compiled.stdout + compiled.stderr,
                                                       encoding="utf-8")
            device_plan = json.loads((device_artifact / "plan.json").read_text(encoding="utf-8"))
            with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
                for index, info in enumerate(device_plan["tensors"]):
                    if info["storage"] == "input":
                        np.ascontiguousarray(inputs[info["name"]]).tofile(
                            device_artifact / f"input_{index}.bin")
            environment["VORTEX_DRIVER"] = "simx"
            environment["LD_LIBRARY_PATH"] = (str(args.vortex_build.resolve() / "sw/runtime")
                                               + ":" + environment.get("LD_LIBRARY_PATH", ""))
            outputs = device_artifact / "outputs"
            executed = run([args.runner.resolve(), device_artifact / "plan.json", device_artifact / "kernel.vxbin",
                            device_artifact, outputs], env=environment, timeout=240,
                           cwd=device_artifact)
            (device_artifact / "run.log").write_text(executed.stdout + executed.stderr, encoding="utf-8")
            tensor_info = {info["name"]: info for info in device_plan["tensors"]}
            errors = {}
            with np.load(directory / "expected.npz", allow_pickle=False) as expected:
                for index, name in enumerate(device_plan["outputs"]):
                    info = tensor_info[name]
                    actual = np.fromfile(outputs / f"output_{index}.bin", dtype=info["dtype"])
                    actual = actual.reshape(info["shape"])
                    np.testing.assert_allclose(actual, expected[name], rtol=2e-5, atol=2e-5,
                                               err_msg=f"SimX {case['name']}: {name}")
                    errors[name] = float(np.max(np.abs(actual.astype(np.float64)
                                                      - expected[name].astype(np.float64)), initial=0))
            entry["simx"] = {"passed": True, "max_absolute_errors": errors}
            print(f"PASS SimX numerical {case['name']}", flush=True)
        report["cases"].append(entry)
        (fixtures / "vortex-object-results.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"PASS Vortex object {case['name']}", flush=True)
    print(f"{len(report['cases'])} Vortex object cases passed")


if __name__ == "__main__":
    main()
