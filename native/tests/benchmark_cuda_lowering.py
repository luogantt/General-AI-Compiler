#!/usr/bin/env python3
"""Measure reproducible ONNX -> PTX builds for legacy and Linalg lowering.

This harness only orchestrates native compiler executables. It does not run a
GPU, generate reference answers, or include CUDA Driver JIT in compile time.
Run with fresh local output storage; source/model hashes and copying are outside
the measured stages. All variants export their constants on every repetition.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


VARIANTS = {
    "legacy-scalar": ["--regular-lowering=legacy"],
    "linalg": ["--regular-lowering=linalg"],
    "legacy-optimized": ["--regular-lowering=legacy", "--cuda-matmul-schedule=auto",
                         "--cuda-reduction-schedule=cooperative",
                         "--cuda-conv-channel-tile=8"],
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": sha256(path)}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_stage(name, command, directory, timeout):
    command = [str(item) for item in command]
    print(f"{utc()} {directory.name}: {name} started", flush=True)
    record = {"name": name, "command": command, "started_utc": utc()}
    stdout_path, stderr_path = directory / f"{name}.stdout.log", directory / f"{name}.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
            record["exit_code"] = completed.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=None, error=f"timeout after {timeout} seconds")
        record["wall_seconds"] = time.perf_counter() - started
    record.update(stdout_log=str(stdout_path), stderr_log=str(stderr_path), finished_utc=utc())
    print(f"{utc()} {directory.name}: {name} {record['wall_seconds']:.3f} s, "
          f"exit={record['exit_code']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-path", type=Path, required=True)
    parser.add_argument("--libdevice", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-compiler-sha256")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--append-variants", nargs="+", choices=tuple(VARIANTS),
                        help="append clean replacement measurements to an existing report")
    parser.add_argument("--exclude-sample", action="append", default=[],
                        metavar="ROUND:VARIANT:REASON", help="keep a raw sample but exclude it from aggregates")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=bool(args.append_variants))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    tools = {name: args.llvm_path / "bin" / name
             for name in ("mlir-translate", "llvm-link", "opt", "llc")}
    for path in (args.compiler, args.model, args.libdevice, *tools.values()):
        if not path.is_file():
            parser.error(f"missing input/tool: {path}")
    compiler_hash = sha256(args.compiler)
    if args.expected_compiler_sha256 and compiler_hash != args.expected_compiler_sha256:
        parser.error("compiler SHA-256 does not match the requested frozen build")
    # Freeze all input artifacts before timing. Do not accidentally benchmark a
    # changed compiler while another task rebuilds its original executable.
    frozen = args.output / "inputs"
    compiler, model, libdevice = frozen / "gac-compile", frozen / "model.onnx", frozen / "libdevice.10.bc"
    if not args.append_variants:
        frozen.mkdir()
        for source, destination in ((args.compiler, compiler), (args.model, model), (args.libdevice, libdevice)):
            shutil.copy2(source, destination)
    report = {
        "kind": "sequential ONNX to PTX wall-clock comparison; no GPU execution",
        "started_utc": utc(), "status": "running",
        "compiler": identity(compiler), "model": identity(model), "libdevice": identity(libdevice),
        "source_paths": {"compiler": str(args.compiler), "model": str(args.model),
                         "libdevice": str(args.libdevice)},
        "tools": {name: identity(path) for name, path in tools.items()},
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu_count": os.cpu_count(), "cwd": str(Path.cwd()),
                        "lscpu": subprocess.check_output(["lscpu"], text=True)},
        "measurement": {
            "clock": "time.perf_counter subprocess wall time",
            "included": ["ONNX import and C++ MLIR passes", "constant .bin export", "LLVM IR translation",
                         "libdevice linking", "LLVM O2 optimization", "LLVM NVPTX PTX generation"],
            "excluded": ["ONNX export", "setup and file copies", "SHA-256", "reporting",
                         "network transfer", "offline ptxas", "CUDA Driver JIT", "GPU execution"],
            "cache_policy": "OS file caches are not flushed; two balanced-order rounds, not cold-cache compilation",
            "target": "sm_86 +ptx80; FP32 model; fp-contract off; nvptx-fma-level 0",
            "variants": VARIANTS,
            "order": [["legacy-scalar", "linalg", "legacy-optimized"],
                      ["legacy-optimized", "linalg", "legacy-scalar"]],
        }, "runs": [],
    }
    order = report["measurement"]["order"]
    first_round = 1
    if args.append_variants:
        report = json.loads((args.output / "compile-report.json").read_text())
        if report["status"] == "running":
            parser.error("cannot append while the original benchmark is still running")
        for name, path in (("compiler", compiler), ("model", model), ("libdevice", libdevice)):
            if sha256(path) != report[name]["sha256"]:
                parser.error(f"frozen {name} changed; cannot append comparable samples")
        for name, path in tools.items():
            if sha256(path) != report["tools"][name]["sha256"]:
                parser.error(f"tool {name} changed; cannot append comparable samples")
        first_round = max(run["round"] for run in report["runs"]) + 1
        order = [args.append_variants]
        report["measurement"]["order"].extend(order)
        report.setdefault("append_sessions", []).append({"started_utc": utc(), "first_round": first_round,
                                                        "variants": args.append_variants})
        report["status"] = "running"
        report.pop("summary", None)

    def exclude_samples():
        for exclusion in args.exclude_sample:
            try:
                round_text, variant, reason = exclusion.split(":", 2)
                round_number = int(round_text)
            except ValueError:
                parser.error("--exclude-sample expects ROUND:VARIANT:REASON")
            matched = [run for run in report["runs"] if run["round"] == round_number and run["variant"] == variant]
            if len(matched) != 1 or not reason.strip():
                parser.error(f"exclusion does not identify one sample with a reason: {exclusion}")
            matched[0].update(excluded_from_aggregate=True, exclusion_reason=reason)
            artifact_report = Path(matched[0]["directory"]) / "compile-result.json"
            if artifact_report.exists():
                atomic_json(artifact_report, matched[0])

    if args.append_variants:
        exclude_samples()

    def publish():
        atomic_json(args.output / "compile-report.json", report)
        atomic_json(args.report_dir / "compile-report.json", report)

    publish()
    for round_index, variants_in_round in enumerate(order, first_round):
        for variant in variants_in_round:
            directory = args.output / f"round-{round_index}-{variant}"
            directory.mkdir()
            run = {"round": round_index, "variant": variant, "directory": str(directory),
                   "status": "running", "steps": []}
            report["runs"].append(run)
            publish()
            pipeline = [
                ("compile", [compiler, model, "--target=cuda", "--emit=llvm",
                             "--constant-fold-profile=bevformer-base", *VARIANTS[variant],
                             "--manifest=" + str(directory / "plan.json"),
                             "--constants-dir=" + str(directory), "-o", directory / "kernel.mlir"]),
                ("translate", [tools["mlir-translate"], "--mlir-to-llvmir", directory / "kernel.mlir",
                               "-o", directory / "kernel.ll"]),
                ("link", [tools["llvm-link"], "--only-needed", "--internalize", directory / "kernel.ll",
                          libdevice, "-o", directory / "kernel-linked.bc"]),
                ("optimize", [tools["opt"], "-passes=default<O2>", directory / "kernel-linked.bc",
                              "-o", directory / "kernel-optimized.bc"]),
                ("ptx", [tools["llc"], "-march=nvptx64", "-mcpu=sm_86", "-mattr=+ptx80", "-O2",
                         "-fp-contract=off", "-nvptx-fma-level=0", directory / "kernel-optimized.bc",
                         "-o", directory / "kernel.ptx"]),
            ]
            for name, command in pipeline:
                result = run_stage(name, command, directory, args.timeout)
                run["steps"].append(result)
                publish()
                if result["exit_code"] != 0:
                    run["status"] = "failed"
                    break
            else:
                run["status"] = "passed"
            run["total_wall_seconds"] = sum(step["wall_seconds"] for step in run["steps"])
            if run["status"] == "passed":
                plan = json.loads((directory / "plan.json").read_text())
                run["plan_summary"] = {
                    "dispatch_count": len(plan["dispatches"]),
                    "lowering_counts": dict(Counter(item.get("lowering", "unspecified") for item in plan["dispatches"])),
                    "arena_bytes": plan["arena_bytes"],
                    "linalg_fused_producers": plan.get("linalg_fused_producers", 0),
                    "inputs": plan.get("inputs"), "outputs": plan.get("outputs"),
                }
                # Hashes prove execution uses these exact outputs, but are never
                # included in the measured pipeline time.
                run["artifacts"] = {name: identity(directory / name) for name in ("kernel.ptx", "plan.json")}
                run["constants"] = {path.name: identity(path) for path in sorted(directory.glob("tensor_*.bin"))}
                atomic_json(directory / "compile-result.json", run)
                shutil.copy2(directory / "plan.json", args.report_dir / f"compile-round-{round_index}-{variant}-plan.json")
                print(f"ARTIFACT_READY {variant} {round_index} {directory}", flush=True)
            for log in directory.glob("*.log"):
                shutil.copy2(log, args.report_dir / f"compile-round-{round_index}-{variant}-{log.name}")
            publish()
    report["compiler_unchanged"] = sha256(compiler) == compiler_hash
    report["tools_unchanged"] = all(sha256(tools[name]) == data["sha256"] for name, data in report["tools"].items())
    report["status"] = "passed" if report["compiler_unchanged"] and report["tools_unchanged"] and all(
        run["status"] == "passed" for run in report["runs"]) else "failed"
    exclude_samples()
    report["summary"] = {}
    for variant in VARIANTS:
        runs = [run for run in report["runs"] if run["variant"] == variant and run["status"] == "passed"
                and not run.get("excluded_from_aggregate", False)]
        if runs:
            report["summary"][variant] = {
                "samples": len(runs), "total_median_seconds": statistics.median(run["total_wall_seconds"] for run in runs),
                "total_min_seconds": min(run["total_wall_seconds"] for run in runs),
                "total_max_seconds": max(run["total_wall_seconds"] for run in runs),
                "stage_median_seconds": {name: statistics.median(next(step["wall_seconds"] for step in run["steps"]
                    if step["name"] == name) for run in runs) for name, _ in pipeline},
            }
    report["finished_utc"] = utc()
    publish()
    print(json.dumps(report["summary"], indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
