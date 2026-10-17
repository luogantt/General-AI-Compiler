#!/usr/bin/env python3
"""Check CUDA auto selection through real ONNX import and verified MLIR lowering.

Selection is a conservative shape heuristic, not a performance guarantee. These
tests check flattened versus batched M, broadcasts, tail geometry, and explicit
schedule overrides. Numerical GPU tests remain a separate required check.
"""
import argparse
import json
import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("flattened_projection", [2, 64, 64], [64, 64], [2, 64, 64], "auto", True),
        ("independent_batches", [2, 64, 64], [2, 64, 64], [2, 64, 64], "auto", False),
        ("broadcast_batches", [2, 1, 128, 64], [1, 2, 64, 64], [2, 2, 128, 64], "auto", True),
        ("attention_short_k", [8, 128, 32], [8, 32, 256], [8, 128, 256], "auto", False),
        ("tail_dimensions", [129, 65], [65, 67], [129, 67], "auto", True),
        ("small_rows", [127, 64], [64, 64], [127, 64], "auto", False),
        ("explicit_scalar", [256, 256], [256, 256], [256, 256], "scalar", False),
        ("explicit_small_tile", [8, 64], [64, 64], [8, 64], "shared-tiled", True),
    ]
    results = []
    for name, a, b, output, schedule, optimized in cases:
        directory = args.output / name
        directory.mkdir()
        model = helper.make_model(helper.make_graph(
            [helper.make_node("MatMul", ["A", "B"], ["Y"])], name,
            [helper.make_tensor_value_info("A", TensorProto.FLOAT, a),
             helper.make_tensor_value_info("B", TensorProto.FLOAT, b)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output)]),
            opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        onnx.checker.check_model(model)
        onnx.save(model, directory / "model.onnx")
        command = [str(args.compiler.resolve()), str(directory / "model.onnx"),
                   "--target=cuda", "--emit=llvm", "--cuda-matmul-schedule=" + schedule,
                   "--constants-dir=" + str(directory), "--manifest=" + str(directory / "plan.json"),
                   "-o", str(directory / "kernel.mlir")]
        completed = subprocess.run(command, text=True, capture_output=True)
        (directory / "compile.log").write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0, (name, completed.stderr)
        plan = json.loads((directory / "plan.json").read_text())
        dispatch = plan["dispatches"][0]
        actual = dispatch.get("cuda_schedule")
        assert bool(actual) == optimized, (name, dispatch)
        if actual:
            assert actual["schedule"] == "shared-tiled"
            expected_tile = 64 if schedule == "auto" else 32
            expected_register_tile = 4 if schedule == "auto" else 2
            assert (actual["tile_m"], actual["tile_n"], actual["tile_k"]) == (expected_tile, expected_tile, 16)
            assert (actual["thread_tile_m"], actual["thread_tile_n"]) == (expected_register_tile, expected_register_tile)
            assert actual["block_size"] == 256
        elif schedule == "auto":
            assert "heuristic keeps scalar" in dispatch["fallback_reason"]
        results.append({"case": name, "requested": schedule, "optimized": optimized,
                        "selected": actual or "scalar", "passed": True})
    (args.output / "results.json").write_text(json.dumps({"passed": True, "cases": results}, indent=2))
    print(f"Passed {len(results)} CUDA MatMul selection checks")


if __name__ == "__main__":
    main()
