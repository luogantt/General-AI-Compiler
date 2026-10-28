#!/usr/bin/env python3
"""Reproduce the six BEVFormer custom-op checks on a TensorRT 10 GPU.

Supply independent ONNX/NumPy fixtures matching the six CASES below. Each
fixture is adapted, built and compared to its NumPy oracle, sequentially. The
default is the validated FP32 policy without per-plugin precision constraints.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .adapt_onnx import adapt_file


CASES = (
    "dcn_mask_bias",
    "dcn_zero_offset_conv_equivalence",
    "dcn_fractional_zero_padding",
    "msda_exact_pixel_centers",
    "msda_multilevel_heads",
    "msda_batch_three_levels",
)
TOLERANCE = "2e-5"


def write_summary(output, summary):
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def invoke(stage, command, directory, environment, record):
    log = directory / f"{stage}.log"
    entry = {"stage": stage, "command": command, "log": str(log)}
    record["steps"].append(entry)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    entry["returncode"] = process.returncode
    if process.returncode:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4000))
            entry["error_log_tail"] = stream.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{stage} exited with {process.returncode}; see {log}")


def verify(args):
    fixtures, output = args.fixtures.resolve(), args.output.resolve()
    for case in CASES:
        for filename in ("model.onnx", "inputs.npz", "expected.npz"):
            if not (fixtures / case / filename).is_file():
                raise FileNotFoundError(fixtures / case / filename)
    # Reject even an empty existing directory: an earlier run is never replaced.
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, NVIDIA_TF32_OVERRIDE="0")
    summary = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(fixtures), "output": str(output),
        "precision": args.precision, "plugins_fp32": args.plugins_fp32,
        "environment_overrides": {"NVIDIA_TF32_OVERRIDE": "0"},
        "rtol": float(TOLERANCE), "atol": float(TOLERANCE), "warmup": 1, "repeat": 3,
        "scope": "Six custom-op fixtures only; not whole-model validation or performance",
        "cases": [],
    }
    write_summary(output, summary)
    try:
        for case in CASES:
            print(f"[{len(summary['cases']) + 1}/{len(CASES)}] {case}", flush=True)
            source = fixtures / case
            directory = output / case
            directory.mkdir()
            adapted, engine = directory / "model.onnx", directory / "engine.plan"
            adaptation_report = directory / "adaptation.json"
            record = {"case": case, "status": "running", "steps": []}
            summary["cases"].append(record)
            write_summary(output, summary)
            try:
                record["steps"].append({
                    "stage": "adapt", "api": "adapt_onnx.adapt_file",
                    "source": str(source / "model.onnx"), "destination": str(adapted),
                    "report": str(adaptation_report),
                })
                report = adapt_file(source / "model.onnx", adapted, adaptation_report)
                record["converted"] = report["converted"]
                command = [sys.executable, "-m", "bevformer_trt", "build", "--onnx", str(adapted),
                           "--engine", str(engine), "--precision", args.precision,
                           "--report", str(directory / "build.json"), "--device", str(args.device)]
                if args.plugins_fp32:
                    command.append("--plugins-fp32")
                invoke("build", command, directory, environment, record)
                write_summary(output, summary)
                invoke("run", [
                    sys.executable, "-m", "bevformer_trt", "run", "--engine", str(engine),
                    "--inputs", str(source / "inputs.npz"), "--reference", str(source / "expected.npz"),
                    "--output", str(directory / "run"), "--warmup", "1", "--repeat", "3",
                    "--rtol", TOLERANCE, "--atol", TOLERANCE, "--device", str(args.device),
                ], directory, environment, record)
                run_report = json.loads((directory / "run" / "report.json").read_text(encoding="utf-8"))
                record["validation"] = run_report["validation"]
                record["cuda_event_ms"] = run_report["cuda_event_ms"]
                if record["validation"]["status"] != "passed":
                    raise RuntimeError("Runner did not confirm numerical validation")
                record["status"] = "passed"
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"  FAILED: {record['error']}", file=sys.stderr, flush=True)
            write_summary(output, summary)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        write_summary(output, summary)
        return 130
    passed = sum(record["status"] == "passed" for record in summary["cases"])
    summary.update(status="passed" if passed == len(CASES) else "failed", passed=passed,
                   total=len(CASES), finished_utc=datetime.now(timezone.utc).isoformat())
    write_summary(output, summary)
    print(f"{passed}/{len(CASES)} passed; report: {output / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are rejected")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--plugins-fp32", action="store_true", help="Also constrain plugins to FP32 (optional policy experiment)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return verify(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
