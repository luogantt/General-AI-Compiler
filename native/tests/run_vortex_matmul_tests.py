#!/usr/bin/env python3
"""Compare actual native-MLIR Vortex MatMul templates on one SimX profile.

NumPy supplies independent golden results, never the implementation under test.
Every measured result comes from a compiled ELF/vxbin and the C++ runtime.
Simulator cycle counts are not board latency or a whole-BEVFormer benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCHEDULES = ("scalar", "lmem-tiled", "dxa-double-buffer", "tcu")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures(root):
    """Tail tiles, repeated stages, real FFN projection ranks and dependencies."""
    rng = np.random.default_rng(20261017)
    cases = []
    configurations = [
        ("tail", (5, 9), (9, 7), "bias_relu"),
        ("aligned", (16, 64), (64, 32), "plain"),
        ("projection_3d", (2, 3, 19), (19, 9), "bias"),
        ("batched", (2, 3, 9), (2, 9, 7), "plain"),
        ("broadcast", (2, 1, 3, 9), (1, 2, 9, 7), "plain"),
        ("ffn_chain", (8, 32), (32, 64), "chain"),
        ("ffn_width", (8, 256), (256, 512), "bias_relu"),
        ("reshape_projection", (8, 4), (8, 4), "reshape"),
        ("nonfinite", (4, 9), (9, 4), "nonfinite"),
    ]
    for name, a_shape, b_shape, mode in configurations:
        directory = root / name
        directory.mkdir(parents=True)
        a = rng.normal(0, .2, a_shape).astype(np.float32)
        b = rng.normal(0, .2, b_shape).astype(np.float32)
        if mode == "nonfinite":
            a[0, 0], a[1, 1] = np.inf, np.nan
            b[0, :3] = [1, -1, 0]
        constants = [numpy_helper.from_array(b, "B")]
        nodes = []
        matrix_input, logical_a = "A", a
        if mode == "reshape":
            constants.append(numpy_helper.from_array(np.array([4, 8], np.int64), "shape"))
            nodes.append(helper.make_node("Reshape", ["A", "shape"], ["view"]))
            matrix_input, logical_a = "view", a.reshape(4, 8)
        nodes.append(helper.make_node("MatMul", [matrix_input, "B"], ["mm"]))
        with np.errstate(invalid="ignore"):
            y = np.matmul(logical_a, b)
        last = "mm"
        if mode in ("bias", "bias_relu", "chain"):
            bias = rng.normal(0, .1, (b_shape[-1],)).astype(np.float32)
            constants.append(numpy_helper.from_array(bias, "bias"))
            nodes.append(helper.make_node("Add", [last, "bias"], ["biased"]))
            y = y + bias
            last = "biased"
        if mode in ("bias_relu", "chain"):
            nodes.append(helper.make_node("Relu", [last], ["activated"]))
            y = np.maximum(y, np.float32(0))
            last = "activated"
        if mode == "chain":
            weight2 = rng.normal(0, .2, (b_shape[-1], a_shape[-1])).astype(np.float32)
            constants.append(numpy_helper.from_array(weight2, "weight2"))
            nodes.append(helper.make_node("MatMul", [last, "weight2"], ["projected"]))
            nodes.append(helper.make_node("Add", ["projected", "A"], ["residual"]))
            y = np.matmul(y, weight2) + a
            last = "residual"
        nodes.append(helper.make_node("Identity", [last], ["Y"]))
        graph = helper.make_graph(nodes, name,
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, a_shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, y.shape)], constants)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx.save(model, directory / "model.onnx")
        np.savez(directory / "inputs.npz", A=a)
        np.savez(directory / "expected.npz", Y=y)
        cases.append({"name": name, "directory": directory,
                      "dxa_eligible": len(b_shape) == 2})
    return cases


def run(argv, log, env, timeout=300):
    started = time.monotonic()
    try:
        result = subprocess.run([str(arg) for arg in argv], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=timeout, text=True, cwd=log.parent)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log.write_text(output + "\nTIMEOUT\n", encoding="utf-8")
        raise
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}), see {log}")
    return {"argv": list(map(str, argv)), "exit_code": result.returncode,
            "host_wall_seconds": time.monotonic() - started, "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--vortex-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--schedule", action="append", choices=SCHEDULES)
    parser.add_argument("--tile-m", type=int, default=4)
    parser.add_argument("--tile-n", type=int, default=4)
    parser.add_argument("--tile-k", type=int, default=16)
    parser.add_argument("--thread-tile-m", type=int, default=1)
    parser.add_argument("--thread-tile-n", type=int, default=1)
    parser.add_argument("--tcu-combined", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be new or empty to keep evidence unambiguous")
    output.mkdir(parents=True, exist_ok=True)
    schedules = args.schedule or list(SCHEDULES)
    cases = fixtures(output / "fixtures")
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.case]
    native = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(GAC_NATIVE_BIN=str(args.compiler.resolve().parent),
               LLVM_PATH=str(args.llvm_bin.resolve().parent),
               VORTEX_BUILD=str(args.vortex_build.resolve()),
               VORTEX_DRIVER="simx", GAC_VORTEX_LANES="4",
               GAC_TILE_M=str(args.tile_m), GAC_TILE_N=str(args.tile_n),
               GAC_TILE_K=str(args.tile_k), GAC_LMEM_BUDGET="16384")
    env.update(GAC_VORTEX_THREAD_TILE_M=str(args.thread_tile_m),
               GAC_VORTEX_THREAD_TILE_N=str(args.thread_tile_n))
    env["LD_LIBRARY_PATH"] = str(args.vortex_build.resolve() / "sw/runtime") + ":" + env.get("LD_LIBRARY_PATH", "")
    report = {"scope": "Vortex SimX MatMul templates: independent numerical comparison and modeled counters",
              "physical_device_execution": False, "full_model_execution": False,
              "compiler_sha256": sha256(args.compiler), "runner_sha256": sha256(args.runner),
              "simulator_sha256": sha256(args.vortex_build / "sw/runtime/libsimx.so"),
              "tile": [args.tile_m, args.tile_n, args.tile_k], "cases": [],
              "thread_tile": [args.thread_tile_m, args.thread_tile_n],
              "tcu_call_mode": "combined" if args.tcu_combined else "separate",
              "performance_note": "Same profile and inputs; cycles include kernel startup and synchronization. Simulator wall time is not hardware latency."}
    report_file = output / "results.json"
    try:
        for case in cases:
            for schedule in schedules:
                artifact = output / case["name"] / schedule
                artifact.parent.mkdir(parents=True, exist_ok=True)
                env["GAC_VORTEX_MATMUL_SCHEDULE"] = schedule
                env["GAC_VORTEX_TCU_COMBINED"] = "1" if args.tcu_combined and schedule == "tcu" else "0"
                entry = {"name": case["name"], "schedule": schedule, "passed": False,
                         "model_sha256": sha256(case["directory"] / "model.onnx")}
                report["cases"].append(entry)
                entry["compile"] = run(["bash", native / "tools/compile-vortex.sh",
                    case["directory"] / "model.onnx", artifact],
                    artifact.parent / (schedule + "-compile.log"), env, args.timeout)
                plan = json.loads((artifact / "plan.json").read_text())
                selected = [d["vortex"]["schedule"] for d in plan["dispatches"] if "vortex" in d]
                entry["selected_templates"] = selected
                entry["arithmetic"] = [d["vortex"].get("arithmetic", "fp32-scalar")
                                       for d in plan["dispatches"] if "vortex" in d]
                if schedule == "scalar":
                    assert not selected, selected
                elif schedule == "dxa-double-buffer" and not case["dxa_eligible"]:
                    assert not selected and any(d.get("fallback_reason") for d in plan["dispatches"])
                else:
                    assert selected and all(item == schedule for item in selected), selected
                with np.load(case["directory"] / "inputs.npz", allow_pickle=False) as inputs:
                    for index, info in enumerate(plan["tensors"]):
                        if info["storage"] == "input":
                            np.ascontiguousarray(inputs[info["name"]]).tofile(artifact / f"input_{index}.bin")
                entry["execute"] = run([args.runner.resolve(), artifact / "plan.json",
                    artifact / "kernel.vxbin", artifact, artifact / "outputs", "--profile-counters"],
                    artifact / "execute.log", env, args.timeout)
                runtime = json.loads((artifact / "outputs/runtime-report.json").read_text())
                entry["device_profile"] = runtime.get("device_profile")
                entry["device_counters"] = runtime.get("device_counters")
                errors = {}
                info_by_name = {t["name"]: t for t in plan["tensors"]}
                with np.load(case["directory"] / "expected.npz", allow_pickle=False) as expected:
                    assert set(expected.files) == set(plan["outputs"])
                    for index, name in enumerate(plan["outputs"]):
                        info = info_by_name[name]
                        actual = np.fromfile(artifact / f"outputs/output_{index}.bin", dtype=info["dtype"]).reshape(info["shape"])
                        np.testing.assert_allclose(actual, expected[name], atol=2e-5, rtol=2e-5, equal_nan=True)
                        finite = np.isfinite(expected[name])
                        assert np.isfinite(actual[finite]).all()
                        errors[name] = float(np.max(np.abs(actual[finite].astype(np.float64) - expected[name][finite].astype(np.float64)), initial=0))
                entry["max_absolute_errors"] = errors
                entry["artifact_sha256"] = {name: sha256(artifact / name) for name in ("kernel.mlir", "kernel.ll", "kernel.elf", "kernel.vxbin", "plan.json")}
                entry["passed"] = True
                report_file.write_text(json.dumps(report, indent=2) + "\n")
                print(f"PASS {case['name']} {schedule}: {errors}", flush=True)
        baselines = {entry["name"]: entry["device_counters"]["max_core_cycles"]
                     for entry in report["cases"]
                     if entry["schedule"] == "scalar" and entry["device_counters"].get("available")}
        for entry in report["cases"]:
            counters = entry["device_counters"]
            if entry["name"] in baselines and counters.get("available"):
                entry["speedup_vs_scalar_cycles"] = baselines[entry["name"]] / counters["max_core_cycles"]
        report["compiler_unchanged"] = report["compiler_sha256"] == sha256(args.compiler)
        report["runner_unchanged"] = report["runner_sha256"] == sha256(args.runner)
        assert report["compiler_unchanged"] and report["runner_unchanged"]
        report["passed"] = True
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report_file.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
