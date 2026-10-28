#!/usr/bin/env python3
"""Validate the template registry through real ONNX -> device/LLVM compilation.

This is a code-generation/launch-contract regression, not a GPU performance or
numerical test. An optional pre-refactor compiler checks byte-identical emitted
IR and unchanged Runtime launch fields. Each run requires a fresh output folder.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

import onnx
from onnx import TensorProto, helper


FAMILIES = {
    "cuda.matmul.shared", "cuda.reduction.row", "cuda.convolution.channels",
    "vortex.matmul.lmem", "vortex.matmul.dxa", "vortex.matmul.tcu",
}
# Runtime still consumes these fields. New inspection metadata is additive;
# diagnostic fallback text is deliberately not part of the execution ABI.
LAUNCH_FIELDS = (
    "kernel", "op", "inputs", "output", "elements", "depends_on",
    "wait_for_dispatch", "block_size", "grid_size", "shared_memory_bytes",
    "cuda_schedule", "vortex",
)


def write_model(path, op, inputs, output, attributes=None, dtype=TensorProto.FLOAT):
    tensors = [helper.make_tensor_value_info(name, dtype, shape)
               for name, shape in inputs]
    graph = helper.make_graph(
        [helper.make_node(op, [name for name, _ in inputs], ["Y"],
                          **(attributes or {}))], path.stem, tensors,
        [helper.make_tensor_value_info("Y", dtype, output)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)


def fixtures(root):
    root.mkdir()
    write_model(root / "matmul.onnx", "MatMul", [("A", [5, 9]), ("B", [9, 7])], [5, 7])
    write_model(root / "fp16.onnx", "MatMul", [("A", [5, 9]), ("B", [9, 7])],
                [5, 7], dtype=TensorProto.FLOAT16)
    write_model(root / "batched.onnx", "MatMul",
                [("A", [2, 3, 9]), ("B", [2, 9, 7])], [2, 3, 7])
    write_model(root / "softmax.onnx", "Softmax", [("X", [2, 5, 3])],
                [2, 5, 3], {"axis": 1})
    write_model(root / "layer_norm.onnx", "LayerNormalization",
                [("X", [2, 3, 7]), ("scale", [7]), ("bias", [7])], [2, 3, 7],
                {"axis": -1, "epsilon": 1e-5})
    for name, channels in (("conv", 8), ("conv_fallback", 6)):
        write_model(root / f"{name}.onnx", "Conv",
                    [("X", [1, 4, 5, 5]), ("W", [channels, 2, 3, 3])],
                    [1, channels, 5, 5], {"group": 2, "pads": [1, 1, 1, 1]})


def compile_model(compiler, model, directory, target, options, stage, expect_error=None):
    directory.mkdir(parents=True)
    command = [str(compiler), str(model), "--no-optimize", f"--target={target}",
               f"--emit={stage}", "--constants-dir=" + str(directory),
               "--manifest=" + str(directory / "plan.json"),
               "-o", str(directory / "kernel.mlir"), *options]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "compile.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if expect_error is not None:
        assert completed.returncode != 0, (directory.name, "invalid options accepted")
        assert expect_error in completed.stderr, (directory.name, completed.stderr)
        return None
    assert completed.returncode == 0, (directory, completed.stderr)
    return json.loads((directory / "plan.json").read_text())


def validate_dispatch(dispatch, expected_id):
    selected = dispatch["compute_template"]
    assert selected["id"] == expected_id, dispatch
    assert selected["collective"] == (expected_id in FAMILIES - {"cuda.convolution.channels"}), dispatch
    cuda = dispatch.get("cuda_schedule")
    if cuda:
        assert dispatch["shared_memory_bytes"] == 0, "static scratch is not dynamic launch memory"
        assert (cuda["block_size"], cuda["grid_size"]) == (dispatch["block_size"], dispatch["grid_size"])
        if expected_id == "cuda.matmul.shared":
            assert cuda["static_shared_memory_bytes"] == 4 * cuda["tile_k"] * (cuda["tile_m"] + cuda["tile_n"])
            assert cuda["block_size"] == (cuda["tile_m"] // cuda["thread_tile_m"]) * (cuda["tile_n"] // cuda["thread_tile_n"])
        elif expected_id == "cuda.reduction.row":
            assert cuda["static_shared_memory_bytes"] == 4 * cuda["block_size"]
            assert cuda["grid_size"] == cuda["rows"]
            assert cuda["width"] * cuda["rows"] == dispatch["elements"]
        elif expected_id == "cuda.convolution.channels":
            assert cuda["static_shared_memory_bytes"] == 0
            assert cuda["logical_threads"] * cuda["channel_tile"] == dispatch["elements"]
    vortex = dispatch.get("vortex")
    if vortex:
        assert vortex["block_size"] % vortex["lanes"] == 0
        if expected_id == "vortex.matmul.tcu":
            assert vortex["stages"] == vortex["lmem_bytes"] == 0
            assert vortex["arithmetic"] == selected["arithmetic"] == "tf32x3-fp32-accumulate"
        else:
            assert vortex["stages"] == (2 if expected_id == "vortex.matmul.dxa" else 1)
            assert vortex["lmem_bytes"] == vortex["stages"] * 4 * vortex["tile_k"] * (vortex["tile_m"] + vortex["tile_n"])


def compare_reference(new_dir, old_dir, plan, reference):
    current = (new_dir / "kernel.mlir").read_text()
    original = (old_dir / "kernel.mlir").read_text()
    if current != original:
        diff = "".join(difflib.unified_diff(original.splitlines(True), current.splitlines(True),
                                           fromfile="reference", tofile="registry"))
        (new_dir / "ir.diff").write_text(diff, encoding="utf-8")
        raise AssertionError(f"emitted IR changed: {new_dir / 'ir.diff'}")
    launch = lambda p: [{key: row[key] for key in LAUNCH_FIELDS if key in row}
                        for row in p["dispatches"]]
    assert launch(plan) == launch(reference), (new_dir, "Runtime launch contract changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--reference-compiler", type=Path,
                        help="Frozen pre-refactor binary; enables strict IR/launch equivalence")
    parser.add_argument("--catalog-tool", type=Path,
                        help="gac-template-list; defaults to the build templates/ subdirectory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve()
    reference = args.reference_compiler.resolve() if args.reference_compiler else None
    catalog_tool = (args.catalog_tool or compiler.parent / "templates/gac-template-list").resolve()
    catalog = json.loads(subprocess.check_output([str(catalog_tool)], text=True, timeout=30))
    assert {row["id"] for row in catalog["templates"]} == FAMILIES
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    fixture_dir = output / "fixtures"
    fixtures(fixture_dir)
    cases = [
        ("cuda_matmul", "matmul", "cuda", ["--cuda-matmul-schedule=shared-tiled"], "cuda.matmul.shared", None),
        ("cuda_softmax", "softmax", "cuda", ["--cuda-reduction-schedule=cooperative"], "cuda.reduction.row", None),
        ("cuda_layer_norm", "layer_norm", "cuda", ["--cuda-reduction-schedule=cooperative"], "cuda.reduction.row", None),
        ("cuda_conv", "conv", "cuda", ["--cuda-conv-channel-tile=4"], "cuda.convolution.channels", None),
        ("vortex_lmem", "matmul", "vortex", ["--vortex-matmul-schedule=lmem-tiled"], "vortex.matmul.lmem", None),
        ("vortex_dxa", "matmul", "vortex", ["--vortex-matmul-schedule=dxa-double-buffer"], "vortex.matmul.dxa", None),
        ("vortex_tcu", "matmul", "vortex", ["--vortex-matmul-schedule=tcu"], "vortex.matmul.tcu", None),
        ("cuda_fp16_fallback", "fp16", "cuda", ["--cuda-matmul-schedule=shared-tiled"], "gac.scalar", "fallback_reason"),
        ("cuda_channel_fallback", "conv_fallback", "cuda", ["--cuda-conv-channel-tile=4"], "gac.scalar", "convolution_fallback_reason"),
        ("vortex_budget_fallback", "matmul", "vortex", ["--vortex-matmul-schedule=lmem-tiled", "--lmem-budget=128"], "gac.scalar", "fallback_reason"),
        ("vortex_dxa_batch_fallback", "batched", "vortex", ["--vortex-matmul-schedule=dxa-double-buffer"], "gac.scalar", "fallback_reason"),
        ("vortex_tcu_lanes_fallback", "matmul", "vortex", ["--vortex-matmul-schedule=tcu", "--vortex-lanes=8"], "gac.scalar", "fallback_reason"),
    ]
    report = {"scope": "template registry codegen/launch contracts; no hardware execution",
              "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
              "reference_compiler_sha256": hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None,
              "passed": False, "cases": []}
    try:
        for name, fixture, target, options, template_id, fallback in cases:
            for stage in ("device", "llvm"):
                run_dir = output / name / stage
                flags = ["--regular-lowering=legacy", *options]
                plan = compile_model(compiler, fixture_dir / f"{fixture}.onnx", run_dir / "new", target, flags, stage)
                assert len(plan["dispatches"]) == 1, (name, plan["dispatches"])
                dispatch = plan["dispatches"][0]
                validate_dispatch(dispatch, template_id)
                if fallback:
                    assert dispatch.get(fallback), (name, dispatch)
                if reference:
                    old = compile_model(reference, fixture_dir / f"{fixture}.onnx", run_dir / "reference", target, flags, stage)
                    compare_reference(run_dir / "new", run_dir / "reference", plan, old)
                report["cases"].append({"name": name, "emit": stage, "template": template_id,
                                        "reference_ir_and_launch_equal": bool(reference), "passed": True})
        # Default compilation must remain the Linalg route, even now that a
        # shared-memory template is available in the same linked executable.
        plan = compile_model(compiler, fixture_dir / "matmul.onnx", output / "default_linalg", "cuda", [], "device")
        assert all(d["compute_template"]["id"] == "linalg.output-thread" for d in plan["dispatches"]), plan
        assert all("cuda_schedule" not in d for d in plan["dispatches"]), plan
        report["cases"].append({"name": "default_linalg", "passed": True})
        for name, target, flags, error in (
            ("linalg_no_bypass", "cuda", ["--cuda-matmul-schedule=shared-tiled"], "require --regular-lowering=legacy"),
            ("invalid_cuda_tile", "cuda", ["--regular-lowering=legacy", "--cuda-matmul-schedule=shared-tiled", "--cuda-thread-tile-m=3"], "invalid CUDA"),
        ):
            compile_model(compiler, fixture_dir / "matmul.onnx", output / name, target, flags, "device", error)
            report["cases"].append({"name": name, "correctly_rejected": True, "passed": True})
        report["passed"] = True
    finally:
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Passed {len(report['cases'])} registry checks; reference comparison: {bool(reference)}")


if __name__ == "__main__":
    main()
