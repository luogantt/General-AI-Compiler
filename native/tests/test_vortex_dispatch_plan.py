#!/usr/bin/env python3
"""Reject inconsistent tiled manifests before loading any Vortex driver.

The end-to-end MatMul suite covers accepted manifests and actual device results.
This suite deliberately supplies an unavailable driver: each error must instead
be detected by the shared C++ manifest parser before device initialization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def plan(schedule="lmem-tiled"):
    tcu = schedule == "tcu"
    dxa = schedule == "dxa-double-buffer"
    stages = 0 if tcu else 2 if dxa else 1
    lmem = stages * 128
    return {
        "schema_version": 1, "target": "vortex",
        "compiler_backend": "native-mlir", "arena_bytes": 128,
        "embedded_constant_bytes": 0, "lmem_bytes": lmem,
        "double_buffer": dxa, "inputs": ["A", "B"], "outputs": ["Y"],
        "tensors": [
            {"name": "A", "dtype": "float32", "storage": "input",
             "bytes": 128, "offset": 0, "shape": [8, 4]},
            {"name": "B", "dtype": "float32", "storage": "input",
             "bytes": 64, "offset": 0, "shape": [4, 4]},
            {"name": "Y", "dtype": "float32", "storage": "arena",
             "bytes": 128, "offset": 0, "shape": [8, 4]},
        ],
        "dispatches": [{
            "kernel": "matmul", "wait_for_dispatch": -1, "depends_on": [],
            "inputs": ["A", "B"], "output": "Y", "elements": 32,
            "vortex": {"schedule": schedule, "m": 8, "n": 4, "k": 4,
                       "a_shape": [8, 4], "b_shape": [4, 4], "bias_shape": [],
                       **({"arithmetic": "tf32x3-fp32-accumulate"} if tcu else {}),
                       "batches": 1, "tile_m": 8 if tcu else 4,
                       "tile_n": 4, "tile_k": 4, "lanes": 4,
                       "block_size": 4 if tcu else 16,
                       "grid_size": 1 if tcu else 2,
                       "lmem_bytes": lmem, "stages": stages},
        }],
    }


def cases():
    rows = []

    def add(name, expected, mutate, schedule="lmem-tiled"):
        value = plan(schedule)
        mutate(value, value["dispatches"][0]["vortex"])
        rows.append((name, expected, value))

    add("zero_tile", "invalid Vortex schedule dimension", lambda p, v: v.update(tile_k=0))
    add("oversized_tile", "tile dimensions exceed", lambda p, v: v.update(tile_k=2147483647))
    add("wrong_grid", "tile/grid/lane", lambda p, v: v.update(grid_size=3))
    add("wrong_lmem", "LMEM allocation", lambda p, v: v.update(lmem_bytes=64))
    add("zero_stages", "LMEM allocation", lambda p, v: v.update(stages=0))
    add("wrong_root_lmem", "root LMEM/buffer summary", lambda p, v: p.update(lmem_bytes=0))
    add("wrong_root_double_buffer", "root LMEM/buffer summary", lambda p, v: p.update(double_buffer=False), "dxa-double-buffer")
    add("unknown_schedule", "unknown Vortex compute schedule", lambda p, v: v.update(schedule="implicit-fallback"))
    add("wrong_matrix_size", "matrix dimensions disagree", lambda p, v: v.update(n=5))
    add("wrong_inner_dimension", "compatible FP32 matrix", lambda p, v: v.update(k=5))
    add("wrong_dtype", "compatible FP32 matrix", lambda p, v: p["tensors"][0].update(dtype="int32"))
    add("missing_logical_shape", "missing array: a_shape", lambda p, v: v.pop("a_shape"))
    add("logical_shape_too_large", "logical operand shape exceeds physical storage", lambda p, v: v.update(a_shape=[8, 5]))
    add("logical_shape_too_small", "logical operand shape differs from physical byte count", lambda p, v: v.update(a_shape=[4, 4]))
    add("unexpected_bias_shape", "without a bias cannot carry a bias shape", lambda p, v: v.update(bias_shape=[4]))
    add("wrong_projection_output_rank", "flattened projection output shape", lambda p, v: p["tensors"][2].update(shape=[2, 4, 4]))
    add("wrong_tcu_stages", "unsupported Vortex FP32 WMMA", lambda p, v: v.update(stages=1), "tcu")
    add("wrong_tcu_lanes", "unsupported Vortex FP32 WMMA", lambda p, v: v.update(lanes=2), "tcu")
    add("wrong_tcu_arithmetic", "requires explicit TF32x3 arithmetic", lambda p, v: v.update(arithmetic="fp32"), "tcu")
    add("missing_tcu_arithmetic", "missing string: arithmetic", lambda p, v: v.pop("arithmetic"), "tcu")
    add("insufficient_dxa_lanes", "DXA requires", lambda p, v: v.update(lanes=2), "dxa-double-buffer")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []
    env = dict(os.environ, VORTEX_DRIVER="unavailable-parser-test-driver")
    with tempfile.TemporaryDirectory(prefix="gac-vortex-plan-") as temporary:
        directory = Path(temporary)
        for name, expected, fixture in cases():
            path = directory / f"{name}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            result = subprocess.run(
                [str(args.runner.resolve()), str(path), "missing.vxbin",
                 str(directory), str(directory / "outputs")],
                env=env, capture_output=True, text=True, timeout=20)
            if result.returncode != 1 or expected not in result.stderr:
                raise AssertionError(f"{name}: expected parser error {expected!r}, "
                                     f"got {result.returncode}: {result.stderr}")
            results.append({"name": name, "passed": True,
                            "diagnostic": result.stderr.strip()})
    report = {"passed": True, "cases": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
