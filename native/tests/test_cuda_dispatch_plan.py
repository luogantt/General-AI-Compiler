#!/usr/bin/env python3
"""Check CUDA launch contracts before PTX loading or device allocations.

Accepted fixtures deliberately reference missing PTX: reaching the driver/JIT
stage proves parser acceptance without executing a kernel. Rejected fixtures
must fail in manifest-validation with the intended diagnostic. GPU execution
and numerical checks belong to run_cuda_tests.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile


def tensor(name, shape, storage="input"):
    return {"name": name, "dtype": "float32", "storage": storage,
            "bytes": 4 * math.prod(shape), "offset": 0, "shape": shape}


def plan(inputs, output, op, schedule):
    return {
        "schema_version": 1, "target": "cuda", "compiler_backend": "native-mlir",
        "arena_bytes": output["bytes"], "embedded_constant_bytes": 0,
        "lmem_bytes": 0, "double_buffer": False,
        "cuda": {"kernel_abi": "direct-pointers", "pointer_bits": 64,
                 "index_bits": 32, "math_library": "libdevice"},
        "inputs": [value["name"] for value in inputs], "outputs": [output["name"]],
        "tensors": inputs + [output],
        "dispatches": [{"kernel": "parser_fixture", "op": op,
                        "source_node": "parser-test", "wait_for_dispatch": -1,
                        "depends_on": [], "inputs": [value["name"] for value in inputs],
                        "output": output["name"], "elements": math.prod(output["shape"]),
                        "block_size": schedule["block_size"],
                        "grid_size": schedule["grid_size"], "shared_memory_bytes": 0,
                        "cuda_schedule": schedule}],
    }


def matrix(batched=False, bias=False):
    a = [2, 1, 7, 5] if batched else [7, 5]
    b = [1, 3, 5, 9] if batched else [5, 9]
    y = [2, 3, 7, 9] if batched else [7, 9]
    batches = 6 if batched else 1
    inputs = [tensor("A", a), tensor("B", b)]
    if bias:
        inputs.append(tensor("bias", [9]))
    return plan(inputs, tensor("Y", y, "arena"),
                "gac.matmul_bias_relu" if bias else "gac.matmul", {
                    "schedule": "shared-tiled", "m": 7, "n": 9, "k": 5,
                    "batches": batches, "tile_m": 16, "tile_n": 16, "tile_k": 16,
                    "thread_tile_m": 2, "thread_tile_n": 2,
                    "block_size": 64, "grid_size": batches,
                    "static_shared_memory_bytes": 2048,
                    "a_shape": a, "b_shape": b, "bias_shape": [9] if bias else [],
                })


def reduction(layer_norm=False):
    shape = [2, 5, 7]
    inputs = [tensor("X", shape)]
    if layer_norm:
        inputs += [tensor("scale", [5, 7]), tensor("bias", [5, 7])]
    kind = "layer_norm" if layer_norm else "softmax"
    return plan(inputs, tensor("Y", shape, "arena"), "gac." + kind, {
        "schedule": "row-reduction", "kind": kind, "input_shape": shape,
        "axis": 1, "width": 35 if layer_norm else 5,
        "stride": 1 if layer_norm else 7, "rows": 2 if layer_norm else 14,
        "block_size": 128, "grid_size": 2 if layer_norm else 14,
        "static_shared_memory_bytes": 512,
        "scale_shape": [5, 7] if layer_norm else [],
        "bias_shape": [5, 7] if layer_norm else [],
    })


def convolution(deform=False):
    shapes = [[2, 8, 7, 9], [8, 4, 3, 3], [8]]
    if deform:
        shapes = [shapes[0], [2, 36, 7, 9], [2, 18, 7, 9], shapes[1], shapes[2]]
    output = [2, 8, 7, 9]
    kind = "deform_conv2d" if deform else "conv2d_relu"
    threads = math.prod(output) // 4
    return plan([tensor(f"input_{i}", s) for i, s in enumerate(shapes)],
                tensor("Y", output, "arena"), "gac." + kind, {
        "schedule": "channel-tiled", "kind": kind, "channel_tile": 4,
        "groups": 2, "channels_per_group": 4, "logical_threads": threads,
        "block_size": 256, "grid_size": (threads + 255) // 256,
        "static_shared_memory_bytes": 0, "input_shapes": shapes, "output_shape": output,
    })


def cases():
    rows = []

    def add(name, base, mutate=None, expected=None):
        value = copy.deepcopy(base)
        dispatch = value["dispatches"][0]
        if mutate:
            mutate(value, dispatch, dispatch["cuda_schedule"])
        rows.append((name, expected, value))

    mat, batched, fused = matrix(), matrix(batched=True), matrix(bias=True)
    softmax, layer_norm = reduction(), reduction(layer_norm=True)
    add("matrix_tail", mat)
    add("batch_broadcast", batched)
    add("fused_bias", fused)
    add("view_physical_shape", mat, lambda p, d, s: p["tensors"][0].update(shape=[5, 7]))
    add("softmax_strided_axis", softmax)
    add("layer_norm_multi_axis", layer_norm)
    add("zero_tile", mat, lambda p, d, s: s.update(tile_k=0), "invalid CUDA schedule dimension")
    add("oversized_tile", mat, lambda p, d, s: s.update(tile_m=256), "matrix tile/storage geometry")
    add("nondivisible_thread_tile", mat, lambda p, d, s: s.update(thread_tile_m=3), "matrix tile/storage geometry")
    add("wrong_thread_count", mat, lambda p, d, s: (d.update(block_size=32), s.update(block_size=32)), "matrix tile/storage geometry")
    add("partial_warp_matrix", mat,
        lambda p, d, s: (d.update(block_size=16),
                         s.update(block_size=16, thread_tile_m=4, thread_tile_n=4)),
        "matrix tile/storage geometry")
    add("wrong_grid", mat, lambda p, d, s: (d.update(grid_size=2), s.update(grid_size=2)), "matrix tile/storage geometry")
    add("row_schedule_launch_mismatch", mat, lambda p, d, s: s.update(grid_size=2), "resource/launch contract")
    add("wrong_shared_bytes", mat, lambda p, d, s: s.update(static_shared_memory_bytes=1024), "matrix tile/storage geometry")
    add("excessive_shared_bytes", mat, lambda p, d, s: s.update(static_shared_memory_bytes=65536), "resource/launch contract")
    add("dynamic_shared_bytes", mat, lambda p, d, s: d.update(shared_memory_bytes=16), "dynamic shared allocation")
    add("wrong_matrix_n", mat, lambda p, d, s: s.update(n=8), "matrix tile/storage geometry")
    add("wrong_matrix_k", mat, lambda p, d, s: s.update(k=6), "matrix operand dimensions")
    add("wrong_matrix_op", mat, lambda p, d, s: d.update(op="gac.conv2d"), "requires MatMul")
    add("missing_matrix_op", mat, lambda p, d, s: d.pop("op"), "requires MatMul")
    add("missing_operand", mat, lambda p, d, s: d.update(inputs=["A"]), "requires MatMul")
    add("logical_storage_overflow", mat, lambda p, d, s: s.update(a_shape=[8, 5]), "logical shape exceeds physical storage")
    add("logical_storage_shortfall", mat, lambda p, d, s: s.update(a_shape=[5, 5]), "byte-identical FP32")
    add("matrix_dtype", mat, lambda p, d, s: p["tensors"][0].update(dtype="int32"), "byte-identical FP32")
    add("output_dtype", mat, lambda p, d, s: p["tensors"][-1].update(dtype="int32"), "resource/launch contract")
    add("batch_broadcast_mismatch", batched, lambda p, d, s: s.update(a_shape=[1, 2, 7, 5]), "batch broadcasting")
    add("wrong_batch_count", batched, lambda p, d, s: s.update(batches=3), "matrix tile/storage geometry")
    add("wrong_projection_rank", mat, lambda p, d, s: p["tensors"][-1].update(shape=[1, 7, 9]), "projection shape")
    add("unexpected_bias", mat, lambda p, d, s: s.update(bias_shape=[9]), "without bias has a bias shape")
    add("incompatible_bias", fused, lambda p, d, s: s.update(bias_shape=[3, 3]), "cannot broadcast to output")
    add("unknown_schedule", mat, lambda p, d, s: s.update(schedule="pretend-fused"), "unknown CUDA compute schedule")
    add("vortex_schedule_spoof", mat, lambda p, d, s: d.update(vortex={}), "cannot carry a Vortex schedule")
    add("vortex_root_memory_spoof", mat, lambda p, d, s: p.update(lmem_bytes=16), "cannot use a Vortex LMEM")
    add("reduction_wrong_kind", softmax, lambda p, d, s: s.update(kind="layer_norm"), "does not match its operator")
    add("reduction_wrong_op", softmax, lambda p, d, s: d.update(op="gac.matmul"), "does not match its operator")
    add("reduction_negative_axis", softmax, lambda p, d, s: s.update(axis=-1), "reduction axis or shape")
    add("reduction_axis_out_of_rank", softmax, lambda p, d, s: s.update(axis=3), "reduction axis or shape")
    add("reduction_shape_mismatch", softmax, lambda p, d, s: s.update(input_shape=[2, 7, 5]), "reduction axis or shape")
    add("reduction_width_mismatch", softmax, lambda p, d, s: s.update(width=7), "row/launch geometry")
    add("reduction_stride_mismatch", softmax, lambda p, d, s: s.update(stride=1), "row/launch geometry")
    add("reduction_rows_mismatch", softmax, lambda p, d, s: s.update(rows=10), "row/launch geometry")
    add("reduction_grid_mismatch", softmax, lambda p, d, s: (d.update(grid_size=7), s.update(grid_size=7)), "row/launch geometry")
    add("reduction_thread_count", softmax, lambda p, d, s: (d.update(block_size=64), s.update(block_size=64, static_shared_memory_bytes=256)), "row/launch geometry")
    add("reduction_shared_mismatch", softmax, lambda p, d, s: s.update(static_shared_memory_bytes=256), "row/launch geometry")
    add("reduction_absent_scale", softmax, lambda p, d, s: s.update(scale_shape=[5]), "absent affine input")
    add("layer_norm_wrong_width", layer_norm, lambda p, d, s: s.update(width=5), "row/launch geometry")
    add("layer_norm_affine_shape", layer_norm, lambda p, d, s: s.update(scale_shape=[7, 5]), "cannot broadcast to output")
    conv, dcn = convolution(), convolution(deform=True)
    add("channel_tiled_grouped_conv", conv)
    add("channel_tiled_grouped_dcn", dcn)
    add("channel_tile_group_crossing", conv, lambda p, d, s: s.update(channel_tile=8), "channel tile/group/launch")
    add("channel_wrong_groups", conv, lambda p, d, s: s.update(groups=1), "channel tile/group/launch")
    add("channel_wrong_thread_count", conv, lambda p, d, s: s.update(logical_threads=1), "channel tile/group/launch")
    add("channel_wrong_kind", conv, lambda p, d, s: s.update(kind="softmax"), "does not match convolution")
    add("channel_wrong_shared", conv, lambda p, d, s: s.update(static_shared_memory_bytes=1024), "does not match convolution")
    add("channel_wrong_shape", conv, lambda p, d, s: s["input_shapes"].__setitem__(0, [2, 4, 14, 9]), "channel/group dimensions")
    add("channel_wrong_output_shape", conv, lambda p, d, s: s.update(output_shape=[2, 8, 9, 7]), "output shape disagrees")
    add("channel_wrong_bias", conv, lambda p, d, s: s["input_shapes"].__setitem__(-1, [2, 4]), "bias shape disagrees")
    add("dcn_wrong_sampling", dcn, lambda p, d, s: s["input_shapes"].__setitem__(1, [2, 18, 14, 9]), "sampling shapes disagree")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="gac-cuda-plan-") as temporary:
        directory = Path(temporary)
        for name, expected, fixture in cases():
            path = directory / f"{name}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            output = directory / name
            result = subprocess.run(
                [str(args.runner.resolve()), str(path), str(directory / "missing.ptx"),
                 str(directory), str(output)], capture_output=True, text=True, timeout=20)
            report = json.loads((output / "runtime-report.json").read_text(encoding="utf-8"))
            stage = "manifest-validation" if expected else "driver-initialization-and-ptx-jit"
            if (result.returncode != 1 or report.get("failure_stage") != stage or
                    (expected and expected not in report.get("error", ""))):
                raise AssertionError(f"{name}: expected {stage}: {expected!r}, "
                                     f"got {result.returncode}: {result.stderr}")
            results.append({"name": name, "passed": True,
                            "manifest_accepted": expected is None,
                            "failure_stage": stage, "diagnostic": report["error"]})
    report = {"passed": True, "scope": "manifest parser only; no GPU kernels executed",
              "cases": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
