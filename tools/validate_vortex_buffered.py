#!/usr/bin/env python3
"""Build and numerically compare scalar and DXA FFN artifacts in SimX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gac.targets.vortex.ffn import compile_vortex_direct, write_vortex_artifact
from gac.targets.vortex.buffered_mlir import VortexBufferPlan
from gac.models.bevformer.ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph, ffn_weight_shapes


def fixture(rows: int, channels: int):
    config = BEVFormerFFNConfig(bev_h=1, bev_w=rows, embed_dim=channels)
    rng = np.random.default_rng(3070)
    weights = {}
    for name, shape in ffn_weight_shapes(config).items():
        value = rng.uniform(0.8, 1.2, shape) if name == "norm_weight" else rng.normal(0, 0.1, shape)
        weights[name] = np.asarray(value, dtype=np.float32)
    return build_bevformer_ffn_graph(weights, config, name=f"buffer_validation_{rows}_{channels}")


def run(command: list[str], log: Path, *, cwd: Path | None = None, env=None, timeout=120):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return log.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vortex-build", required=True, type=Path)
    parser.add_argument("--llvm-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", default="smoke,tails,reuse,compare")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = {"smoke": (2, 4), "tails": (3, 19), "reuse": (5, 35),
             "compare": (8, 32), "wide": (8, 128),
             "production-width": (8, 256), "multiwarp-tail": (3, 19)}
    env = os.environ | {"VORTEX_DRIVER": "simx", "LD_LIBRARY_PATH": str(args.vortex_build / "sw/runtime")}
    results = []
    for name in args.cases.split(","):
        rows, channels = cases[name]
        graph = fixture(rows, channels)
        schedules = {"dxa-2x2x16": VortexBufferPlan()}
        if name == "compare":
            schedules = {"scalar": None, "dxa-2x2x16": VortexBufferPlan(),
                         "dxa-4x4x16": VortexBufferPlan(4, 4, 16),
                         "dxa-4x4x32": VortexBufferPlan(4, 4, 32)}
        elif name == "wide":
            schedules = {"scalar": None, "dxa-4x4x64": VortexBufferPlan(4, 4, 64),
                         "dxa-2x8x64": VortexBufferPlan(2, 8, 64),
                         "dxa-1x16x64": VortexBufferPlan(1, 16, 64)}
        elif name == "production-width":
            schedules = {"scalar": None, "dxa-4x4x64": VortexBufferPlan(4, 4, 64),
                         "dxa-4x4x128": VortexBufferPlan(4, 4, 128)}
        elif name == "multiwarp-tail":
            schedules = {"dxa-4x4x16": VortexBufferPlan(4, 4, 16)}
        for schedule, plan in schedules.items():
            tag = f"{name}-{schedule}"
            destination = args.vortex_build / "tests/regression" / f"gac_buffer_{tag}"
            program = compile_vortex_direct(graph, mlir_opt=str(args.llvm_path / "bin/mlir-opt"), buffer_plan=plan)
            write_vortex_artifact(destination, program)
            run(["make", "-C", str(destination), "all", f"LLVM_PATH={args.llvm_path}",
                 "CONFIGS=-DVX_CFG_EXT_DXA_ENABLE"], args.output / f"{tag}-build.log")
            text = run([str(destination / "gac_bevformer_vortex"), "--data-dir", ".",
                        "--kernel", "kernel.vxbin", "--warmup", "1", "--repeat", "3"],
                       args.output / f"{tag}-simx.log", cwd=destination, env=env)
            report, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
            if not report["verified"]:
                raise RuntimeError(f"numerical mismatch for {tag}")
            counters = re.search(r"PERF: instrs=(\d+), cycles=(\d+)", text)
            report.update(case=name, schedule=schedule, rows=rows, channels=channels,
                          buffer_plan=plan.to_dict() if plan else None,
                          simulator_instructions=int(counters[1]) if counters else None,
                          simulator_cycles=int(counters[2]) if counters else None)
            results.append(report)
            (args.output / "results.json").write_text(json.dumps({
                "scope": "synthetic FFN SimX; timings are simulator host time, not hardware latency",
                "results": results}, indent=2) + "\n", encoding="utf-8")
            print(f"{tag}: verified, error={report['max_abs_error']}, cycles={report['simulator_cycles']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
