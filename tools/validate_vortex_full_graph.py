#!/usr/bin/env python3
"""Reproduce tiny BEVFormer full-graph MLIR/native/SimX correctness validation.

The synthetic model keeps the complete streaming graph topology with smaller
tensor shapes; it is not a nuScenes accuracy or production latency benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from gac.targets.vortex.compiler import SOFTWARE_OPS, compile_vortex_graph, write_vortex_graph_artifact
from gac.targets.vortex.validation import NativeVortexGraphSession
from gac.models.bevformer.step_pipeline import build_bevformer_step_pipeline_graph
from gac.core.ir import Node
from gac.validation.reference import reference_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm-path", type=Path, required=True)
    parser.add_argument("--vortex-build", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--all-ops", action="store_true", help="append independent tiny outputs exercising the four ops absent from BEVFormer")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be empty or new")
    args.output.mkdir(parents=True, exist_ok=True)
    from tests.test_bevformer_step_pipeline import small_config, make_weights, runtime_inputs
    config = small_config()
    encoder, decoder = make_weights(config)
    graph = build_bevformer_step_pipeline_graph(encoder, decoder, config)
    # Debug outputs make the history path observable even when small random
    # learned weights attenuate its effect on final detection tensors.
    graph.outputs += ("selected_prev_bev", "motion_compensated_prev_bev")
    if args.all_ops:
        graph.constants["coverage_weight"] = np.full((18, 2), 0.03, np.float32)
        graph.constants["coverage_bias"] = np.asarray([0.01, -0.02], np.float32)
        for op in ("Identity", "GELU", "AddReLU", "MatMulBiasGELU"):
            names = ("can_bus_delta",)
            if op == "AddReLU":
                names = ("can_bus_delta", "can_bus_delta")
            elif op == "MatMulBiasGELU":
                names = ("can_bus_delta", "coverage_weight", "coverage_bias")
            output = "coverage_" + op
            graph.nodes.append(Node(output, op, names, (output,)))
            graph.outputs += (output,)
    program = compile_vortex_graph(graph, mlir_opt=str(args.llvm_path / "bin/mlir-opt"))
    if args.all_ops:
        assert {n.op for n in program.graph.nodes} == SOFTWARE_OPS
    report = {"kind": "synthetic-tiny-whole-BEVFormer", "original_nodes": len(graph.nodes),
              "dispatches": len(program.dispatches), "operator_types": sorted({n.op for n in program.graph.nodes}),
              "arena_bytes": program.memory_plan.arena_size, "native": [], "simx": []}
    cold_expected = None
    with NativeVortexGraphSession(program, llvm_bin=args.llvm_path / "bin") as native:
        prior_state = None
        for warm in (False, True):
            values = runtime_inputs(config, warm=warm)
            if warm:
                values["prev_bev"] = prior_state
            expected = reference_run(graph, values)
            actual = native.run(values)
            for name in graph.outputs:
                np.testing.assert_allclose(actual[name], expected[name], rtol=2e-3, atol=3e-4)
            report["native"].append({"frame": "warm" if warm else "cold", "passed": True})
            prior_state = actual["next_bev"]
            if not warm:
                cold_expected = expected
        probe = runtime_inputs(config, warm=True)
        probe["prev_bev"] = np.arange(probe["prev_bev"].size, dtype=np.float16).reshape(probe["prev_bev"].shape) + np.float16(3)
        probe["can_bus_delta"][0, -1] = 0  # identity rotation exposes exact history values
        probe_actual = native.run(probe)
        probe_expected = reference_run(graph, probe)
        for name in graph.outputs:
            np.testing.assert_allclose(probe_actual[name], probe_expected[name], rtol=2e-3, atol=3e-4)
        np.testing.assert_array_equal(probe_actual["selected_prev_bev"], probe["prev_bev"].astype(np.float32))
        report["native"].append({"frame": "history-route-probe", "passed": True})
    artifact = args.output / "artifact"
    cold_inputs = runtime_inputs(config, warm=False)
    write_vortex_graph_artifact(artifact, program, inputs=cold_inputs, golden=cold_expected)
    def run(command, log, *, env=None, timeout=300):
        start = time.monotonic()
        with (args.output / log).open("w") as output:
            completed = subprocess.run([str(x) for x in command], cwd=artifact, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"command failed ({completed.returncode}); see {args.output / log}")
        return time.monotonic() - start
    if args.vortex_build:
        run(["make", "all", f"ROOT_DIR={args.vortex_build}", f"LLVM_PATH={args.llvm_path}", "CONFIGS=-DVX_CFG_EXT_DXA_ENABLE"], "build.log")
        env = os.environ | {"VORTEX_DRIVER": "simx", "LD_LIBRARY_PATH": str(args.vortex_build / "sw/runtime")}
        prior_state = None
        cold_actual = None
        for label in ("cold", "warm", "history-route-probe"):
            warm = label != "cold"
            values = runtime_inputs(config, warm=warm)
            if label == "warm":
                values["prev_bev"] = prior_state
            elif label == "history-route-probe":
                values = probe
            expected = reference_run(graph, values)
            # Invocation data may change between frames; compiled source and weights do not.
            for i, name in enumerate(program.graph.inputs):
                values[name].tofile(artifact / f"inputs/input_{i:04d}.bin")
            for i, name in enumerate(program.graph.outputs):
                expected[name].tofile(artifact / f"goldens/golden_{i:04d}.bin")
                (artifact / f"outputs/output_{i:04d}.bin").unlink(missing_ok=True)
            wall = run([artifact / "gac_bevformer_full_vortex", "--data-dir", artifact,
                        "--kernel", artifact / "kernel.vxbin", "--threads", args.threads,
                        "--warmup", 0, "--repeat", 1], f"{label}-simx.log", env=env)
            run([sys.executable, artifact / "verify_outputs.py", "--data-dir", artifact,
                 "--report", args.output / f"{label}-correctness.json"], f"{label}-verify.log")
            actual = {name: np.fromfile(artifact / f"outputs/output_{i:04d}.bin", dtype=program.graph.value_specs[name].dtype).reshape(program.graph.value_specs[name].shape)
                      for i, name in enumerate(program.graph.outputs)}
            np.savez(args.output / f"{label}-outputs.npz", **actual)
            prior_state = actual["next_bev"]
            if label == "cold":
                cold_actual = actual
            if label == "history-route-probe":
                np.testing.assert_array_equal(actual["selected_prev_bev"], probe["prev_bev"].astype(np.float32))
            report["simx"].append({"frame": label, "passed": True, "simulator_host_wall_seconds": wall})
        # Restore the frozen artifact fixture so its hashes remain reproducible.
        for i, name in enumerate(program.graph.inputs):
            cold_inputs[name].tofile(artifact / f"inputs/input_{i:04d}.bin")
        cold_golden = reference_run(graph, cold_inputs)
        for i, name in enumerate(program.graph.outputs):
            cold_golden[name].tofile(artifact / f"goldens/golden_{i:04d}.bin")
            cold_actual[name].tofile(artifact / f"outputs/output_{i:04d}.bin")
    report["state_feedback"] = "previous invocation next_bev -> host -> next invocation prev_bev"
    report["state_probe"] = "selected_prev_bev matches controlled FP16 history exactly"
    report["oracle"] = "original unoptimized Graph / independent NumPy evaluator"
    report["production_checkpoint_validated"] = False
    report["silicon_latency_measured"] = False
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
