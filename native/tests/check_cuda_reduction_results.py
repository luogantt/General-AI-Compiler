#!/usr/bin/env python3
"""Validate CUDA reduction outputs, including explicitly exceptional fixtures.

The generic whole-model validator deliberately requires finite outputs. This
separate checker allows NaN/Inf only in fixtures marked expected_nonfinite and
requires their exact masks to match. It never relaxes normal model validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from run_cuda_tests import compare_outputs, validate_artifact
from run_full_model import sha256, write_json


def validate(metadata_path: Path, result_root: Path, fixtures: Path) -> dict:
    metadata = json.loads(metadata_path.read_text())
    suite = json.loads((result_root / "suite.json").read_text())
    specification = json.loads((fixtures / "cases.json").read_text())
    fixture_info = {case["name"]: case for case in specification["cases"]}
    if (metadata["deployment_id"] != suite["deployment_id"] or
            metadata["runner_sha256"] != suite["runner_sha256"]):
        raise ValueError("GPU report does not belong to the requested deployment")
    runtime_cases = {case["name"]: case for case in suite["cases"]}
    if set(runtime_cases) != {case["name"] for case in metadata["cases"]}:
        raise ValueError("GPU case names differ from the deployment")
    results = []
    for case in metadata["cases"]:
        runtime = runtime_cases[case["name"]]
        rt = runtime["runtime"]
        if (not runtime["gpu_executed"] or not runtime["artifact_hashes_verified"] or
                rt["status"] != "completed" or rt["execution"] != "cuda-device" or
                rt["host_operator_fallback"] is not False or
                rt["completed_dispatches"] != case["dispatches"]):
            raise ValueError("GPU execution or artifact identity is incomplete")
        artifact, reference = Path(case["artifact"]), Path(case["reference"])
        if (sha256(artifact / "plan.json") != case["manifest_sha256"] or
                sha256(reference) != case["reference_sha256"] or
                sha256(Path(case["inputs"])) != case["inputs_sha256"]):
            raise ValueError("Compiler inputs, manifest or reference changed")
        plan = validate_artifact(artifact)
        output_directory = result_root / "cases" / case["name"] / "output"
        exceptional = fixture_info[case["name"].split("/")[-1]].get("expected_nonfinite", False)
        if not exceptional:
            outputs = compare_outputs(plan, output_directory, reference,
                                      metadata["rtol"], metadata["atol"])
        else:
            outputs = {}
            tensors = {item["name"]: item for item in plan["tensors"]}
            with np.load(reference, allow_pickle=False) as expected_file:
                if set(expected_file.files) != set(plan["outputs"]):
                    raise ValueError("Reference must contain every declared output")
                for index, name in enumerate(plan["outputs"]):
                    info = tensors[name]
                    path = output_directory / f"output_{index}.bin"
                    if path.stat().st_size != info["bytes"]:
                        raise ValueError("GPU output byte count mismatch")
                    actual = np.fromfile(path, dtype=info["dtype"]).reshape(info["shape"])
                    expected = expected_file[name]
                    if actual.shape != expected.shape or actual.dtype != expected.dtype:
                        raise ValueError("GPU output shape or dtype mismatch")
                    nan = bool(np.array_equal(np.isnan(actual), np.isnan(expected)))
                    positive = bool(np.array_equal(np.isposinf(actual), np.isposinf(expected)))
                    negative = bool(np.array_equal(np.isneginf(actual), np.isneginf(expected)))
                    finite = np.isfinite(expected)
                    close = bool(np.allclose(actual[finite], expected[finite],
                                             rtol=metadata["rtol"], atol=metadata["atol"]))
                    outputs[name] = {"passed": nan and positive and negative and close,
                                     "nan_mask_equal": nan, "positive_inf_mask_equal": positive,
                                     "negative_inf_mask_equal": negative,
                                     "finite_values_allclose": close,
                                     "finite_elements": int(finite.sum()),
                                     "output_sha256": sha256(path),
                                     "rtol": metadata["rtol"], "atol": metadata["atol"]}
        results.append({"name": case["name"], "expected_nonfinite": exceptional,
                        "numerical_passed": all(out["passed"] for out in outputs.values()),
                        "manifest_sha256": case["manifest_sha256"],
                        "reference_sha256": case["reference_sha256"],
                        "outputs": outputs, "gpu_name": rt["gpu_name"],
                        "kernel_resources": rt["kernel_resources"]})
    return {"status": "passed" if all(case["numerical_passed"] for case in results) else "failed",
            "cases": results, "runner_sha256": suite["runner_sha256"],
            "deployment_id": suite["deployment_id"],
            "metadata_sha256": sha256(metadata_path),
            "gpu_report_sha256": sha256(result_root / "suite.json"),
            "nonfinite_policy": "Exact NaN and signed Inf masks only for explicitly marked fixtures; finite values retain normal tolerance"}


def paired_summary(path: Path) -> dict:
    report = json.loads(path.read_text())
    cases = []
    for name in dict.fromkeys(case["name"] for case in report["rounds"]):
        modes = {}
        for mode in ("scalar", "optimized"):
            runs = [case for case in report["rounds"] if case["name"] == name and case["mode"] == mode]
            samples = [sample for run in runs for sample in run["runtime"]["gpu_dispatch_samples_ms"]]
            if len(runs) != 4 or len(samples) != 120:
                raise ValueError("Expected four paired rounds with 30 samples each")
            modes[mode] = {"median_ms": statistics.median(samples), "samples_ms": samples,
                           "round_medians_ms": [statistics.median(run["runtime"]["gpu_dispatch_samples_ms"]) for run in runs],
                           "output_hashes_identical": len({run["output_sha256"] for run in runs}) == 1,
                           "kernel_resources": runs[0]["runtime"]["kernel_resources"]}
        cases.append({"name": name, "modes": modes,
                      "speedup": modes["scalar"]["median_ms"] / modes["optimized"]["median_ms"]})
    return {"measurement": report["timing_note"], "raw_report_sha256": sha256(path),
            "cases": cases, "whole_model_claim": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--paired-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.metadata, args.results, args.fixtures)
    if args.paired_report:
        result["paired_benchmark"] = paired_summary(args.paired_report)
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "report": str(args.output)}))
    raise SystemExit(0 if result["status"] == "passed" else 1)
