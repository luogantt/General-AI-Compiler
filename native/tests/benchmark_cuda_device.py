#!/usr/bin/env python3
"""Compare precompiled CUDA artifacts with fresh JIT and balanced run order.

Run on the GPU host. This script uses only Python's standard library, never
compiles a model, and never treats matching output hashes as numerical proof.
Every measured process has one warmup and three timed whole-model sequences.
The CUDA disk cache is disabled for these child processes; it is not deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time


VARIANTS = ("legacy-scalar", "linalg", "legacy-optimized")
ORDER = (*VARIANTS, *reversed(VARIANTS))
GPU_FIELDS = ("index", "uuid", "name", "temperature.gpu", "clocks.sm",
              "clocks.mem", "utilization.gpu", "utilization.memory",
              "memory.used", "memory.total", "power.draw", "pstate")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def command_snapshot(argv: list[str], timeout: float = 10) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "error": str(error)}


def gpu_snapshot(device: int) -> dict:
    gpu = command_snapshot([
        "nvidia-smi", f"--id={device}", "--query-gpu=" + ",".join(GPU_FIELDS),
        "--format=csv,noheader,nounits"], timeout=2)
    rows = []
    if gpu.get("returncode") == 0:
        for row in csv.reader(io.StringIO(gpu["stdout"])):
            if len(row) == len(GPU_FIELDS):
                rows.append(dict(zip(GPU_FIELDS, (value.strip() for value in row))))
    processes = command_snapshot([
        "nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"], timeout=2)
    return {"unix_seconds": time.time(), "gpus": rows,
            "gpu_query": gpu, "compute_processes": processes}


def temperature(snapshot: dict) -> float | None:
    try:
        return float(snapshot["gpus"][0]["temperature.gpu"])
    except (IndexError, KeyError, ValueError):
        return None


def cooldown(device: int) -> dict:
    """Wait at most 60 s; record inability to attain 55 C instead of hiding it."""
    started = time.monotonic()
    samples = []
    while True:
        current = gpu_snapshot(device)
        samples.append(current)
        temp = temperature(current)
        if temp is None:
            reason = "temperature_unavailable"
            break
        if temp <= 55:
            reason = "temperature_at_or_below_55C"
            break
        remaining = 60 - (time.monotonic() - started)
        if remaining <= 0:
            reason = "60s_limit_reached"
            break
        time.sleep(min(5, remaining))
    return {"wall_seconds": time.monotonic() - started,
            "reason": reason, "samples": samples}


def monitor_gpu(device: int, destination: Path, stop: threading.Event) -> None:
    with destination.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            started = time.monotonic()
            stream.write(json.dumps(gpu_snapshot(device)) + "\n")
            stream.flush()
            stop.wait(max(0, 1 - (time.monotonic() - started)))


def tensor_signature(tensor: dict) -> dict:
    return {key: tensor[key] for key in ("name", "dtype", "shape", "bytes")}


def prepare_artifacts(root: Path, source: Path) -> tuple[dict, dict, dict]:
    """Validate all variants before creating any input symlinks."""
    common_input = identity(source)
    plans, artifacts = {}, {}
    signature = None
    for variant in VARIANTS:
        folder = root / variant
        plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
        if plan.get("target") != "cuda" or not plan.get("dispatches"):
            raise ValueError(f"{variant}: expected nonempty CUDA device plan")
        tensors = plan["tensors"]
        inputs = [(index, item) for index, item in enumerate(tensors)
                  if item["storage"] == "input"]
        if len(inputs) != 1 or inputs[0][0] != 0:
            raise ValueError(f"{variant}: requires exactly one input at tensor index 0")
        if inputs[0][1]["bytes"] != common_input["bytes"]:
            raise ValueError(f"{variant}: input byte count does not match plan")
        by_name = {item["name"]: item for item in tensors}
        if len(by_name) != len(tensors):
            raise ValueError(f"{variant}: duplicate tensor names")
        outputs = [by_name[name] for name in plan["outputs"]]
        current = {"input": tensor_signature(inputs[0][1]),
                   "outputs": [tensor_signature(item) for item in outputs]}
        if signature is not None and current != signature:
            raise ValueError(f"{variant}: model input/output names, shapes or dtypes differ")
        signature = current
        constants = []
        for item in tensors:
            if item["storage"] != "constant":
                continue
            path = (folder / item["file"]).resolve(strict=True)
            if not path.is_relative_to(folder.resolve()):
                raise ValueError(f"{variant}: constant must be inside its artifact directory")
            record = identity(path)
            if record["bytes"] != item["bytes"]:
                raise ValueError(f"{variant}: constant byte count mismatch: {item['name']}")
            constants.append({"tensor": tensor_signature(item), "file": record})
        target = folder / "input_0.bin"
        if target.exists() or target.is_symlink():
            if identity(target)["sha256"] != common_input["sha256"]:
                raise ValueError(f"{variant}: existing input_0.bin differs; never overwritten")
        artifacts[variant] = {
            "plan": identity(folder / "plan.json"),
            "ptx": identity(folder / "kernel.ptx"), "constants": constants,
            "dispatches": len(plan["dispatches"]),
            "arena_bytes": plan.get("arena_bytes"), "io": current}
        plans[variant] = plan
    for variant in VARIANTS:
        target = root / variant / "input_0.bin"
        if not target.exists() and not target.is_symlink():
            target.symlink_to(source.resolve())
    return plans, artifacts, common_input


def stats(values: list[float]) -> dict:
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def telemetry_warnings(path: Path, runtime_pid: int | None) -> list[str]:
    """Compute-process telemetry is advisory; display-only processes are excluded."""
    others = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            query = sample.get("compute_processes", {})
            if query.get("returncode") != 0:
                continue
            for row in csv.reader(io.StringIO(query.get("stdout", ""))):
                if row and row[0].strip().isdigit():
                    pid = int(row[0].strip())
                    if pid != runtime_pid:
                        others.add(pid)
    return (["Other GPU compute PIDs observed; timings may include contention: " +
             ", ".join(str(pid) for pid in sorted(others))] if others else [])


def validate_runtime(runtime: dict, plan: dict) -> None:
    if (runtime.get("status") != "completed" or
            runtime.get("execution") != "cuda-device" or
            runtime.get("host_operator_fallback") is not False or
            runtime.get("completed_dispatches") != len(plan["dispatches"]) or
            runtime.get("dispatches") != len(plan["dispatches"])):
        raise ValueError("Runtime lacks complete CUDA execution evidence")
    if runtime.get("warmup_runs") != 1 or runtime.get("measured_runs") != 3:
        raise ValueError("Unexpected warmup/repeat count")
    if runtime.get("trace_dispatches") or runtime.get("profile_dispatches"):
        raise ValueError("Instrumented dispatch timing cannot be compared here")
    samples = runtime.get("gpu_dispatch_samples_ms", [])
    if len(samples) != 3 or any(not isinstance(value, (int, float)) or
                               not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Expected three positive finite GPU timings")
    phases = runtime.get("startup_timing_ms", {})
    if "module_load_ms" not in phases:
        raise ValueError("Runner lacks separate module-load/JIT instrumentation")
    for key in ("CUDA_CACHE_DISABLE", "CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING"):
        expected = "1" if key == "CUDA_CACHE_DISABLE" else "EAGER"
        if runtime.get("cuda_environment", {}).get(key) != expected:
            raise ValueError(f"Runtime did not confirm {key}={expected}")


def execute(args: argparse.Namespace, index: int, variant: str, plan: dict) -> dict:
    directory = args.output / f"{index + 1:02d}-{variant}"
    directory.mkdir(exist_ok=False)
    output = directory / "outputs"
    output.mkdir()
    entry = {"variant": variant, "sequence_index": index,
             "numerical_validation": "pending_reference_comparison",
             "cooldown": cooldown(args.device)}
    artifact = args.root / variant
    argv = [str(args.runner), str(artifact / "plan.json"),
            str(artifact / "kernel.ptx"), str(artifact), str(output),
            "--device", str(args.device), "--warmup", "1", "--repeat", "3"]
    entry["command"] = argv
    environment = os.environ.copy()
    environment.update(CUDA_MODULE_LOADING="EAGER", CUDA_MODULE_DATA_LOADING="EAGER",
                       CUDA_CACHE_DISABLE="1")
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu,
                               args=(args.device, directory / "gpu.jsonl", stop), daemon=True)
    monitor.start()
    started = time.monotonic()
    process = None
    try:
        with (directory / "runtime.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            entry["pid"] = process.pid
            try:
                entry["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                entry["timed_out"] = True
                entry["returncode"] = process.returncode
        entry["process_wall_seconds"] = time.monotonic() - started
        report = output / "runtime-report.json"
        if report.is_file():
            entry["runtime"] = json.loads(report.read_text(encoding="utf-8"))
        if entry["returncode"]:
            raise RuntimeError(f"Runtime exited {entry['returncode']}; see {directory / 'runtime.log'}")
        validate_runtime(entry.get("runtime", {}), plan)
        by_name = {item["name"]: item for item in plan["tensors"]}
        entry["outputs"] = []
        for output_index, name in enumerate(plan["outputs"]):
            record = identity(output / f"output_{output_index}.bin")
            if record["bytes"] != by_name[name]["bytes"]:
                raise ValueError(f"Output byte count mismatch: {name}")
            entry["outputs"].append({"name": name, **record})
        entry["status"] = "measured_pending_numerical_validation"
    except Exception as error:
        entry.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        stop.set()
        monitor.join(timeout=6)
        if not monitor.is_alive():
            entry["warnings"] = telemetry_warnings(directory / "gpu.jsonl", entry.get("pid"))
        else:
            entry["warnings"] = ["GPU telemetry thread did not finish within timeout"]
        write_json(directory / "run.json", entry)
    return entry


def summarize(runs: list[dict]) -> dict:
    result = {}
    for variant in VARIANTS:
        records = [item for item in runs if item["variant"] == variant and
                   item["status"] == "measured_pending_numerical_validation"]
        if not records:
            continue
        reports = [item["runtime"] for item in records]
        timings = {"gpu_dispatch_samples_ms": stats([
            sample for report in reports for sample in report["gpu_dispatch_samples_ms"]])}
        wall_samples = [sample for report in reports
                        for sample in report.get("dispatch_wall_samples_ms", [])]
        if wall_samples:
            timings["dispatch_wall_samples_ms"] = stats(wall_samples)
        for field in ("initialization_ms", "driver_reported_jit_wall_ms",
                      "first_dispatch_gpu_ms", "first_dispatch_host_wall_ms",
                      "upload_ms", "download_ms", "input_file_io_ms",
                      "total_host_wall_ms"):
            values = [report[field] for report in reports if field in report]
            if values:
                timings[field] = stats(values)
        for field in reports[0]["startup_timing_ms"]:
            values = [report["startup_timing_ms"][field] for report in reports
                      if field in report["startup_timing_ms"]]
            timings["startup." + field] = stats(values)
        output_hashes = [[(item["name"], item["sha256"]) for item in record["outputs"]]
                         for record in records]
        consistent = (all(value == output_hashes[0] for value in output_hashes)
                      if len(records) > 1 else None)
        result[variant] = {"processes": len(records), "timings_ms": timings,
                           "same_variant_final_output_hashes_equal": consistent,
                           "output_hash_note": "Only the final outputs of each process are saved. "
                           "Equal hashes prove repeatability, not reference accuracy; "
                           "individual timed iterations are not downloaded.",
                           "numerical_validation": "pending_reference_comparison"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Contains legacy-scalar/, linalg/, legacy-optimized/ artifacts")
    parser.add_argument("--runner", type=Path, required=True, help="Instrumented gac-cuda-run")
    parser.add_argument("--input", type=Path, required=True, help="Shared raw input_0.bin")
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Maximum seconds per runtime process (default: 1800)")
    args = parser.parse_args()
    if args.device < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--device must be nonnegative and --timeout positive and finite")
    for name in ("root", "runner", "input", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "preparing", "order": list(ORDER),
              "warmup_per_process": 1, "timed_sequences_per_process": 3,
              "cache_policy": {"CUDA_CACHE_DISABLE": "1", "CUDA_MODULE_LOADING": "EAGER",
                               "CUDA_MODULE_DATA_LOADING": "EAGER",
                               "existing_driver_cache_deleted": False},
              "numerical_validation": "pending_reference_comparison", "runs": []}
    try:
        report["runner"] = identity(args.runner)
        plans, artifacts, source = prepare_artifacts(args.root, args.input)
        report.update(artifacts=artifacts, input=source)
        report["environment"] = {
            "uname": command_snapshot(["uname", "-a"]),
            "lscpu": command_snapshot(["lscpu"]),
            "nvidia_smi": command_snapshot(["nvidia-smi"]),
            "initial_gpu": gpu_snapshot(args.device)}
        for index, variant in enumerate(ORDER):
            print(f"[{index + 1}/{len(ORDER)}] {variant}: fresh process/JIT, warmup=1 repeat=3", flush=True)
            record = execute(args, index, variant, plans[variant])
            report["runs"].append(record)
            report["summary"] = summarize(report["runs"])
            report["status"] = "running" if record["status"] != "failed" else "failed"
            write_json(args.output / "comparison.json", report)
            if record["status"] == "failed":
                raise RuntimeError(record["error"])
        report["status"] = "measured_pending_numerical_validation"
        if any(not item["same_variant_final_output_hashes_equal"]
               for item in report["summary"].values()):
            report["status"] = "output_repeatability_discrepancy"
        # Outside all timed processes: make an artifact change a hard error.
        _, final_artifacts, final_input = prepare_artifacts(args.root, args.input)
        unchanged = (final_artifacts == artifacts and final_input == source and
                     identity(args.runner) == report["runner"])
        report["input_runner_and_artifacts_unchanged_after_runs"] = unchanged
        if not unchanged:
            raise ValueError("Input, runner or compilation artifacts changed during the benchmark")
        report["warnings"] = [f"Run {run['sequence_index'] + 1}: {warning}"
                              for run in report["runs"] for warning in run.get("warnings", [])]
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    write_json(args.output / "comparison.json", report)
    print(f"{report['status']}: {args.output / 'comparison.json'}", flush=True)
    return 0 if report["status"] == "measured_pending_numerical_validation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
