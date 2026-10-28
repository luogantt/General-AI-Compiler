#!/usr/bin/env python3
"""Validate the timing contract of an actual gac-cuda-run JSON report.

This checks measurement consistency, not model accuracy or benchmark stability.
Use --require-cold-jit for runs with driver cache disabled and eager loading.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate(report: dict, require_cold_jit: bool = False) -> None:
    if report.get("status") != "completed" or report.get("execution") != "cuda-device":
        raise ValueError("Report does not describe completed CUDA execution")

    def number(value, label):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid nonnegative timing: {label}={value!r}")
        return value

    phases = report["startup_timing_ms"]
    expected = {
        "manifest_validation_ms", "driver_library_and_symbol_load_ms",
        "driver_init_and_device_query_ms", "context_creation_ms",
        "stream_and_event_creation_ms", "ptx_file_read_ms", "module_load_ms",
        "module_log_and_memory_query_ms", "device_buffer_allocation_ms",
        "kernel_lookup_resources_and_arguments_ms",
    }
    if set(phases) != expected:
        raise ValueError(f"Unexpected startup phases: {set(phases) ^ expected}")
    phase_sum = sum(number(value, key) for key, value in phases.items())
    startup = number(report["initialization_ms"], "initialization_ms")
    if phase_sum > startup + 0.01:
        raise ValueError("Exclusive startup phases exceed inclusive startup time")
    number(report["driver_reported_jit_wall_ms"], "driver_reported_jit_wall_ms")

    samples = report["gpu_dispatch_samples_ms"]
    host_samples = report["dispatch_wall_samples_ms"]
    warmups = report["warmup_gpu_samples_ms"]
    if len(samples) != report["measured_runs"] or len(host_samples) != len(samples):
        raise ValueError("Measured sample count disagrees with measured_runs")
    if not samples or len(warmups) != report["warmup_runs"]:
        raise ValueError("Missing samples or incorrect warmup count")
    for kind, values in (("gpu", samples), ("host", host_samples), ("warmup", warmups)):
        for index, value in enumerate(values):
            number(value, f"{kind}[{index}]")
    if not math.isclose(report["gpu_dispatch_ms"], sum(samples) / len(samples), rel_tol=1e-7):
        raise ValueError("GPU sample mean disagrees with gpu_dispatch_ms")
    if sum(host_samples) > number(report["dispatch_wall_ms"], "dispatch_wall_ms") + 0.01:
        raise ValueError("Exclusive frame wall times exceed dispatch interval")
    first = warmups[0] if warmups else samples[0]
    if not math.isclose(report["first_dispatch_gpu_ms"], first, rel_tol=1e-7):
        raise ValueError("First dispatch is not the first warmup/measured sample")
    number(report["first_dispatch_host_wall_ms"], "first_dispatch_host_wall_ms")
    number(report["warmup_host_wall_ms"], "warmup_host_wall_ms")
    if require_cold_jit:
        environment = report["cuda_environment"]
        if environment["CUDA_CACHE_DISABLE"] != "1" or environment["CUDA_MODULE_LOADING"] != "EAGER":
            raise ValueError("Cold JIT comparison needs cache disabled and eager loading")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--require-cold-jit", action="store_true")
    args = parser.parse_args()
    validate(json.loads(args.report.read_text(encoding="utf-8")), args.require_cold_jit)
    print(f"CUDA timing contract passed: {args.report}")


if __name__ == "__main__":
    main()
