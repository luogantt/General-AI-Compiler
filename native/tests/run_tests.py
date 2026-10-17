#!/usr/bin/env python3
"""Exercise the C++ ONNX importer, native MLIR passes and emitted host kernels.

This is a test harness, not a compiler or deployment fallback. Numerical tests
run an explicitly selected native CPU target to validate the same scalar MLIR
lowering independently of Vortex device availability.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def run(command, *, succeeds=True, env=None, timeout=120, cwd=None):
    result = subprocess.run([str(arg) for arg in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, timeout=timeout, cwd=cwd)
    if (result.returncode == 0) != succeeds:
        raise AssertionError(f"Unexpected exit {result.returncode}: {' '.join(map(str, command))}\n"
                             f"{result.stdout}\n{result.stderr}")
    return result


def execute(manifest_path: Path, constants_dir: Path, library: Path,
            case_dir: Path) -> dict:
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == 1
    assert plan["compiler_backend"] == "native-mlir"
    assert plan["target"] == "native", "These ctypes checks execute only the native test target"
    arena = np.zeros(max(1, plan["arena_bytes"]), dtype=np.uint8)
    tensors = {}
    with np.load(case_dir / "inputs.npz", allow_pickle=False) as inputs:
        for item in plan["tensors"]:
            shape, dtype = tuple(item["shape"]), np.dtype(item["dtype"])
            count = int(np.prod(shape, dtype=np.int64))
            assert count * dtype.itemsize == item["bytes"], item
            if item["storage"] == "input":
                tensors[item["name"]] = np.ascontiguousarray(inputs[item["name"]], dtype=dtype)
                assert tensors[item["name"]].shape == shape
            elif item["storage"] == "constant":
                source = Path(item["file"])
                if not source.is_absolute():
                    candidates = [manifest_path.parent / source, constants_dir / source,
                                  constants_dir / source.name]
                    source = next((path for path in candidates if path.is_file()), candidates[0])
                values = np.fromfile(source, dtype=dtype)
                assert values.size == count, item
                tensors[item["name"]] = values.reshape(shape)
            else:
                assert item["storage"] == "arena", item
                tensors[item["name"]] = np.ndarray(shape, dtype=dtype, buffer=arena,
                                                     offset=item["offset"])
    module = ctypes.CDLL(str(library.resolve()))
    for index, dispatch in enumerate(plan["dispatches"]):
        assert all(prior < index for prior in dispatch.get("depends_on", []))
        arguments = [tensors[name] for name in dispatch["inputs"]]
        arguments.append(tensors[dispatch["output"]])
        kernel = getattr(module, dispatch["kernel"])
        kernel.argtypes = [ctypes.c_void_p] * len(arguments)
        kernel.restype = None
        kernel(*(ctypes.c_void_p(array.ctypes.data) for array in arguments))
    errors = {}
    with np.load(case_dir / "expected.npz", allow_pickle=False) as expected:
        assert set(expected.files) == set(plan["outputs"])
        for name in plan["outputs"]:
            actual, reference = tensors[name], expected[name]
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5,
                                       err_msg=f"{case_dir.name}: {name}")
            errors[name] = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64)), initial=0))
    return {"dispatches": len(plan["dispatches"]), "max_absolute_errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--optimizer", type=Path)
    parser.add_argument("--mlir-translate", type=Path, required=True)
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--case", action="append", dest="selected_cases",
                        help="Run one named case; repeat to select several")
    args = parser.parse_args()
    fixtures = args.fixtures.resolve()
    if args.regenerate:
        from generate_models import generate
        generate(fixtures)
    cases = json.loads((fixtures / "cases.json").read_text(encoding="utf-8"))["cases"]
    if args.selected_cases:
        unknown = set(args.selected_cases) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.selected_cases]
    report = {"compiler": str(args.compiler.resolve()), "target": "native",
              "scope": "C++ ONNX import + MLIR passes + native scalar numerical validation",
              "vortex_device_execution": False, "cases": []}
    for case in cases:
        directory = fixtures / case["name"]
        base = [args.compiler, directory / "model.onnx", "--target=native"]
        entry = {"name": case["name"]}
        if not case["valid"]:
            failure = run(base + ["--emit=optimized", "-o", directory / "rejected.mlir"],
                          succeeds=False)
            entry["diagnostic"] = failure.stderr.strip()
            assert entry["diagnostic"], "Rejected models need an actionable diagnostic"
            if case.get("expected_diagnostic"):
                assert case["expected_diagnostic"] in entry["diagnostic"], entry["diagnostic"]
            if case["unsupported"]:
                output = directory / "inspection.mlir"
                run(base + ["--emit=import", "--allow-unsupported", "-o", output])
                assert "Mystery" in output.read_text(encoding="utf-8")
                entry["inspection_preserved_unsupported"] = True
        else:
            imported, optimized = directory / "import.mlir", directory / "optimized.mlir"
            run(base + ["--emit=import", "-o", imported])
            run(base + ["--emit=optimized", "-o", optimized])
            before, after = imported.read_text(encoding="utf-8"), optimized.read_text(encoding="utf-8")
            assert "func.func" in before and "gac." in before
            for fragment in case.get("expect_optimized_contains", []):
                assert fragment in after, f"{case['name']}: optimized IR is missing {fragment}"
            for fragment in case.get("expect_optimized_absent", []):
                assert fragment not in after, f"{case['name']}: optimized IR still contains {fragment}"
            if case["expect_fusion"]:
                count_before = len(re.findall(r'"gac\.[^\"]+"\(', before))
                count_after = len(re.findall(r'"gac\.[^\"]+"\(', after))
                assert count_after < count_before, f"{case['name']}: expected a native fusion rewrite"
                entry["operation_counts"] = [count_before, count_after]
            if case.get("expect_no_fusion"):
                for operation in ("add_relu", "matmul_bias_relu", "conv2d_relu"):
                    assert f'"gac.{operation}"(' not in after, f"Unexpected low-precision fusion: {operation}"
                entry["rounding_boundaries_preserved"] = True
            if case.get("expect_shape_fold"):
                for operation in ("shape", "gather", "unsqueeze", "concat", "cast"):
                    assert f'"gac.{operation}"(' not in after, f"Shape operation was not folded: {operation}"
                assert '"gac.reshape"(' in after
                entry["shape_chain_folded"] = True
            if case.get("expect_cse"):
                entry["cse_operation_counts"] = {}
                for operation, expected_counts in case["expect_cse"].items():
                    counts = [text.count(f'"gac.{operation}"(') for text in (before, after)]
                    assert counts == expected_counts, f"{case['name']}: CSE counts {counts}, expected {expected_counts}"
                    entry["cse_operation_counts"][operation] = counts
            if args.optimizer:
                run([args.optimizer, imported,
                     "--pass-pipeline=builtin.module(gac-infer-shapes,gac-legalize-tensor-ops,gac-infer-shapes,gac-normalize-provenance,canonicalize,cse,gac-fuse,gac-normalize-provenance,canonicalize,cse,gac-propagate-layout)",
                     "-o", directory / "optimizer.mlir"])
                if case.get("expect_cse"):
                    standalone = (directory / "optimizer.mlir").read_text(encoding="utf-8")
                    for operation, expected_counts in case["expect_cse"].items():
                        assert standalone.count(f'"gac.{operation}"(') == expected_counts[1]
            llvm_mlir, llvm_ir = directory / "llvm.mlir", directory / "kernel.ll"
            manifest, constants = directory / "plan.json", directory / "constants"
            command = base + ["--emit=llvm", "-o", llvm_mlir, "--manifest", manifest,
                              "--constants-dir", constants]
            if case.get("reject_device"):
                rejected = run(command, succeeds=False)
                assert rejected.stderr.strip(), "Target rejection must explain the unsupported contract"
                entry["device_rejection"] = rejected.stderr.strip()
                if case.get("expected_diagnostic"):
                    assert case["expected_diagnostic"] in entry["device_rejection"], entry["device_rejection"]
            else:
                run(command)
                run([args.mlir_translate, "--mlir-to-llvmir", llvm_mlir, "-o", llvm_ir])
                library = directory / "kernels.so"
                run([args.clang, "-shared", "-fPIC", "-O2", llvm_ir, "-lm", "-o", library])
                entry.update(execute(manifest, constants, library, directory))
        entry["passed"] = True
        report["cases"].append(entry)
        print(f"PASS {case['name']}", flush=True)
        report_path = fixtures / ("selected-results.json" if args.selected_cases else "results.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(report['cases'])} native regression cases passed; report: {report_path}")


if __name__ == "__main__":
    main()
