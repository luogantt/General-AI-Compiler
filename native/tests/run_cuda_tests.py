#!/usr/bin/env python3
"""Validate native C++ MLIR -> NVPTX artifacts and, explicitly, real CUDA execution.

Reuse existing ONNX fixtures or supply a whole model, inputs and reference NPZ.
NumPy only reads/writes tensors and compares results; there is no CPU operator
fallback. --compile-only cannot report a numerical pass or a GPU latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from run_full_model import compare, constant_path, load_plan, read_inputs, sha256, write_json


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def command(argv, log: Path, timeout: int, environment=None) -> subprocess.CompletedProcess:
    """Save diagnostics even when a compiler/driver invocation fails."""
    try:
        result = subprocess.run([str(arg) for arg in argv], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        stderr = error.stderr or b""
        log.write_bytes((output.encode() if isinstance(output, str) else output) +
                        (stderr.encode() if isinstance(stderr, str) else stderr))
        raise
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


def require_success(result, log: Path) -> None:
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log}")


def validate_artifact(directory: Path) -> dict:
    plan = load_plan(directory / "plan.json")
    if plan["target"] != "cuda":
        raise ValueError("CUDA validation requires a CUDA manifest")
    abi = plan.get("cuda", {})
    if (abi.get("kernel_abi") != "direct-pointers" or abi.get("pointer_bits") != 64
            or abi.get("index_bits") != 32 or abi.get("math_library") != "libdevice"):
        raise ValueError("Unsupported CUDA kernel ABI")
    if plan.get("lmem_bytes") != 0 or plan.get("double_buffer") is not False:
        raise ValueError("CUDA artifact cannot carry a Vortex LMEM contract")
    ptx = (directory / "kernel.ptx").read_text(encoding="utf-8")
    if not re.search(r"\.address_size\s+64\b", ptx):
        raise ValueError("Expected 64-bit NVPTX addressing")
    entries = set(re.findall(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(", ptx))
    for dispatch in plan["dispatches"]:
        if dispatch["kernel"] not in entries:
            raise ValueError(f"Missing PTX entry: {dispatch['kernel']}")
        for field in ("grid_size", "block_size"):
            size = dispatch[field]
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Invalid CUDA {field}: {size}")
        if (dispatch["block_size"] > 1024 or dispatch["grid_size"] > 2147483647
                or dispatch.get("shared_memory_bytes") != 0):
            raise ValueError("Invalid CUDA launch resources")
        schedule = dispatch.get("cuda_schedule")
        if schedule is None:
            expected_grid = (dispatch["elements"] + dispatch["block_size"] - 1) // dispatch["block_size"]
        else:
            if (not isinstance(schedule, dict) or
                    schedule.get("block_size") != dispatch["block_size"] or
                    schedule.get("grid_size") != dispatch["grid_size"] or
                    not 0 <= schedule.get("static_shared_memory_bytes", -1) <= 48 * 1024):
                raise ValueError("Invalid CUDA schedule resources")
            if schedule.get("schedule") == "shared-tiled":
                m, n, batches = (schedule[field] for field in ("m", "n", "batches"))
                tm, tn, tk, rm, rn = (schedule[field] for field in (
                    "tile_m", "tile_n", "tile_k", "thread_tile_m", "thread_tile_n"))
                if (min(m, n, batches, tm, tn, tk, rm, rn) <= 0 or
                        tm % rm or tn % rn or (tm // rm) * (tn // rn) != dispatch["block_size"] or
                        4 * tk * (tm + tn) != schedule["static_shared_memory_bytes"] or
                        m * n * batches != dispatch["elements"]):
                    raise ValueError("Invalid CUDA matrix tile geometry")
                expected_grid = ((m + tm - 1) // tm) * ((n + tn - 1) // tn) * batches
            elif schedule.get("schedule") == "channel-tiled":
                tile = schedule["channel_tile"]
                if (tile not in (2, 4, 8) or dispatch["elements"] % tile or
                        schedule["logical_threads"] != dispatch["elements"] // tile or
                        schedule["static_shared_memory_bytes"] != 0):
                    raise ValueError("Invalid CUDA channel tile geometry")
                expected_grid = (schedule["logical_threads"] + dispatch["block_size"] - 1) // dispatch["block_size"]
            elif schedule.get("schedule") == "row-reduction":
                expected_grid = schedule["rows"]
                if (schedule["width"] <= 0 or expected_grid * schedule["width"] != dispatch["elements"] or
                        schedule["static_shared_memory_bytes"] != dispatch["block_size"] * 4):
                    raise ValueError("Invalid CUDA row reduction geometry")
            else:
                raise ValueError("Unknown CUDA compute schedule")
        if dispatch["grid_size"] != expected_grid:
            raise ValueError("Invalid CUDA launch geometry")
    return plan


def compare_outputs(plan: dict, directory: Path, reference_path: Path,
                    rtol: float, atol: float) -> dict:
    tensors = {item["name"]: item for item in plan["tensors"]}
    results = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        if set(reference.files) != set(plan["outputs"]):
            raise ValueError("Reference names must match every model output exactly")
        for index, name in enumerate(plan["outputs"]):
            info = tensors[name]
            output = directory / f"output_{index}.bin"
            if output.stat().st_size != info["bytes"]:
                raise ValueError(f"Output byte count mismatch: {name}")
            value = np.fromfile(output, dtype=info["dtype"]).reshape(info["shape"])
            expected = reference[name]
            # Integer, bool and shape outputs require exact equality. Floating
            # point comparisons preserve explicit dtype/rounding boundaries.
            tolerance = (rtol, atol) if value.dtype.kind == "f" else (0, 0)
            result = compare(value, expected, *tolerance)
            result["dtype_matches"] = value.dtype == expected.dtype
            if value.dtype.kind in "iub":
                # Avoid float64 losing low integer bits above 2**53.
                result["passed"] = bool(np.array_equal(value, expected))
                result["mismatched_elements"] = int(np.count_nonzero(value != expected)) if value.shape == expected.shape else None
                if value.shape == expected.shape and result["dtype_matches"]:
                    mismatches = value != expected
                    maximum, total = 0, 0
                    for actual_item, expected_item in zip(value[mismatches], expected[mismatches]):
                        error = abs(int(actual_item) - int(expected_item))
                        maximum, total = max(maximum, error), total + error
                    result["max_abs_error"] = maximum
                    result["mean_abs_error"] = total / value.size if value.size else None
            result["passed"] = bool(result["passed"] and result["dtype_matches"])
            result.update(rtol=tolerance[0], atol=tolerance[1], artifact=identity(output))
            results[name] = result
    return results


def execute_case(args, case: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    entry = {"name": case["name"], "status": "started", "gpu_executed": False,
             "numerical_passed": None, "rtol": args.rtol, "atol": args.atol,
             "validation_report": str(directory / "validation.json"),
             "reference_description": case.get("reference_description"),
             "reference_versions": case.get("reference_versions"),
             "artifacts": {}}
    begin = time.perf_counter()
    try:
        if case.get("model"):
            entry["artifacts"]["model"] = identity(case["model"])
        artifact = case.get("artifact") or directory / "artifact"
        if not case.get("artifact"):
            artifact.mkdir(exist_ok=True)
            log = directory / "compile.log"
            environment = os.environ.copy()
            environment.update(GAC_NATIVE_BIN=str(args.compiler.resolve().parent),
                               LLVM_PATH=str(args.llvm_bin.resolve().parent),
                               GAC_CUDA_ARCH=args.arch,
                               GAC_CONSTANT_FOLD_PROFILE=args.constant_fold_profile)
            result = command(["bash", args.compile_script.resolve(), case["model"], artifact],
                             log, args.compile_timeout, environment)
            require_success(result, log)
        plan = validate_artifact(artifact)
        entry["compile_status"] = "ptx_generated_and_manifest_validated"
        entry["dispatch_count"] = len(plan["dispatches"])
        entry["arena_bytes"] = plan["arena_bytes"]
        if case.get("maximum_arena_bytes") is not None:
            if plan["arena_bytes"] > case["maximum_arena_bytes"]:
                raise ValueError("Static arena exceeds fixture memory budget")
        for name in ("plan.json", "kernel.ptx", "kernel.ll", "kernel.mlir", "kernel.cubin", "compile-provenance.txt"):
            if (artifact / name).is_file():
                entry["artifacts"][name] = identity(artifact / name)
        entry["constants"] = {item["name"]: identity(constant_path(item, artifact / "plan.json", artifact))
                              for item in plan["tensors"] if item["storage"] == "constant"}
        for name in ("inputs", "reference"):
            if case.get(name):
                entry["artifacts"][name] = identity(case[name])
        if args.compile_only:
            entry["status"] = "compiled_only"
            return entry
        inputs = read_inputs(case["inputs"], plan)
        for index, item in enumerate(plan["tensors"]):
            if item["storage"] == "input":
                inputs[item["name"]].tofile(artifact / f"input_{index}.bin")
        # A fresh directory prevents old GPU output files from being mistaken
        # for output of a failed or incomplete invocation.
        output = directory / "device-output"
        output.mkdir(exist_ok=False)
        log = directory / "runtime.log"
        result = command([args.runner.resolve(), artifact / "plan.json", artifact / "kernel.ptx",
                          artifact, output, "--device", args.device, "--warmup", args.warmup,
                          "--repeat", args.repeat] + (["--trace-dispatches"] if args.trace_dispatches else []),
                         log, args.run_timeout)
        runtime_report = output / "runtime-report.json"
        if runtime_report.is_file():
            entry["runtime"] = json.loads(runtime_report.read_text(encoding="utf-8"))
        require_success(result, log)
        runtime = entry.get("runtime", {})
        if (runtime.get("status") != "completed" or runtime.get("execution") != "cuda-device"
                or runtime.get("completed_dispatches") != len(plan["dispatches"])
                or runtime.get("host_operator_fallback") is not False):
            raise ValueError("Missing or incomplete real CUDA runtime execution evidence")
        entry["gpu_executed"] = True
        entry["outputs"] = compare_outputs(plan, output, case["reference"], args.rtol, args.atol)
        entry["numerical_passed"] = all(item["passed"] for item in entry["outputs"].values())
        entry["status"] = "gpu_passed" if entry["numerical_passed"] else "numerical_failed"
    except Exception as error:
        entry["status"] = "failed"
        entry["error"] = f"{type(error).__name__}: {error}"
    finally:
        entry["validation_wall_seconds"] = time.perf_counter() - begin
        write_json(directory / "validation.json", entry)
    return entry


def collect_cases(args, parser) -> tuple[list, list]:
    cases, excluded = [], []
    if args.fixtures:
        for suite in args.fixtures:
            suite = suite.resolve()
            specification = json.loads((suite / "cases.json").read_text(encoding="utf-8"))
            for item in specification["cases"]:
                name = f"{suite.name}/{item['name']}"
                if args.selected and item["name"] not in args.selected and name not in args.selected:
                    continue
                if not item.get("valid", True) or item.get("reject_device"):
                    excluded.append({"name": name, "reason": "Existing fixture is an intentional rejection case; not a numerical test"})
                    continue
                folder = suite / item["name"]
                cases.append({**item, "name": name, "model": folder / "model.onnx",
                              "artifact": (args.skip_compile.resolve() / name / "artifact") if args.skip_compile else None,
                              "inputs": folder / "inputs.npz", "reference": folder / "expected.npz",
                              "reference_description": item.get("reference", specification.get("reference")),
                              "reference_versions": specification.get("versions")})
        if args.selected:
            known = {item["name"] for item in cases + excluded}
            unknown = {name for name in args.selected if name not in known and not any(
                candidate.endswith("/" + name) for candidate in known)}
            if unknown:
                parser.error(f"Unknown cases: {sorted(unknown)}")
    else:
        if not args.compile_only and not (args.inputs and args.reference):
            parser.error("Whole-model GPU validation needs --inputs and --reference")
        cases.append({"name": "full-model", "model": args.model.resolve() if args.model else None,
                      "artifact": args.artifact.resolve() if args.artifact else None,
                      "inputs": args.inputs.resolve() if args.inputs else None,
                      "reference": args.reference.resolve() if args.reference else None,
                      "reference_description": args.reference_description})
    if not cases:
        parser.error("No positive numerical fixtures selected")
    if len({case["name"] for case in cases}) != len(cases):
        parser.error("Fixture suite/case names must be unique")
    return cases, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixtures", type=Path, action="append", help="Repeat for existing fixture suites")
    source.add_argument("--model", type=Path, help="Whole-model ONNX to compile and validate")
    source.add_argument("--artifact", type=Path, help="Existing CUDA artifact directory; skip compilation")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-description", default="User-supplied independent reference outputs")
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--skip-compile", type=Path,
                        help="With --fixtures, reuse artifacts under this previous validation output directory")
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--llvm-bin", type=Path)
    parser.add_argument("--compile-script", type=Path,
                        default=Path(__file__).resolve().parents[1] / "tools/compile-cuda.sh")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--runner", type=Path, help="Real CUDA Driver runtime executable; no CPU fallback")
    parser.add_argument("--output-dir", type=Path, required=True, help="New report directory; existing reports are preserved")
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--constant-fold-profile", default="default", choices=("default", "bevformer-base"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--trace-dispatches", action="store_true",
                        help="Synchronize and record each dispatch for diagnosis; timing is not normal throughput")
    parser.add_argument("--compile-timeout", type=int, default=1800)
    parser.add_argument("--run-timeout", type=int, default=1800)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--atol", type=float, default=2e-5)
    args = parser.parse_args()
    if not args.artifact and not args.skip_compile and not (args.compiler and args.llvm_bin):
        parser.error("Compilation requires --compiler and --llvm-bin")
    if args.skip_compile and not args.fixtures:
        parser.error("--skip-compile is for fixture suites; use --artifact for a whole model")
    if args.selected and not args.fixtures:
        parser.error("--case is only valid with --fixtures")
    if (args.device < 0 or args.warmup < 0 or args.repeat <= 0 or min(args.compile_timeout, args.run_timeout) <= 0
            or not all(math.isfinite(n) and n >= 0 for n in (args.rtol, args.atol))):
        parser.error("Invalid device, iteration, timeout or tolerance value")
    for path in (args.compiler, args.runner):
        if path is not None and not path.is_file():
            parser.error(f"Executable does not exist: {path}")
    cases, excluded = collect_cases(args, parser)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "target": "cuda", "compiler_backend": "native-mlir",
              "mode": "compile_only" if args.compile_only else "physical_cuda_validation",
              "python_operator_fallback": False,
              "requested_arch": args.arch if not (args.artifact or args.skip_compile) else None,
              "excluded_cases": excluded, "cases": [], "gpu_executed": False, "numerical_passed": None}
    for name in ("compiler", "runner"):
        path = getattr(args, name)
        if path:
            report[name] = identity(path)
    for case in cases:
        entry = execute_case(args, case, args.output_dir / case["name"])
        report["cases"].append(entry)
        write_json(args.output_dir / "cuda-validation.json", report)
        print(f"{entry['status']}: {case['name']}" + (f" ({entry['error']})" if "error" in entry else ""), flush=True)
    success = all(item["status"] == ("compiled_only" if args.compile_only else "gpu_passed") for item in report["cases"])
    report["status"] = ("compiled_only" if args.compile_only else "gpu_passed") if success else "failed"
    if not args.compile_only:
        report["gpu_executed"] = all(item["gpu_executed"] for item in report["cases"])
        report["numerical_passed"] = success
    write_json(args.output_dir / "cuda-validation.json", report)
    print(f"{report['status']}; report: {args.output_dir / 'cuda-validation.json'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
