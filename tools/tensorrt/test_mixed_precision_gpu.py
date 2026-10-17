"""GPU integration test with a tiny synthetic CNN, never BEVFormer calibration.

Usage: python tools/tensorrt/test_mixed_precision_gpu.py NEW_OUTPUT_DIRECTORY
Requires ONNX, NumPy, TensorRT 10.13.3.9 and CUDA PyTorch. Tests calibration,
cache reuse and explicit measured ranges, precision inspection and numerical
execution; synthetic tiny-model evidence only, with no application FPS claim.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import run_engine
from calibration_ranges import TEST_KIND, convert_bound_cache


def identity_manifest(model, model_path, build_report, build_report_path, output):
    """Describe the unchanged tiny graph, never pretend it is a Base backbone."""
    model_hash = run_engine.sha256(model_path)
    if build_report["onnx_file_sha256"] != model_hash:
        raise AssertionError("Precision audit belongs to a different tiny model")
    policy = build_report["precision_policy"]["mixed_layer_policy"]
    int8_names = [row["name"] for row in policy["layers"] if row.get("compute_precision") == "int8"]
    if len(int8_names) != 1 or int8_names[0] not in {node.name for node in model.graph.node}:
        raise AssertionError("Expected exactly the original tiny backbone Conv to be INT8")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def describe(value):
        tensor = value.type.tensor_type
        if not all(dim.HasField("dim_value") for dim in tensor.shape.dim):
            raise AssertionError(f"Tiny tensor has an unknown shape: {value.name}")
        return {"name": value.name, "dtype": str(helper.tensor_dtype_to_np_dtype(tensor.elem_type)),
                "shape": [dim.dim_value for dim in tensor.shape.dim]}

    tensors = {value.name: describe(value) for value in (
        *inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    producers = {value.name: "input" for value in model.graph.input}
    for weight in model.graph.initializer:
        tensors[weight.name] = {"name": weight.name,
            "dtype": str(helper.tensor_dtype_to_np_dtype(weight.data_type)), "shape": list(weight.dims)}
        producers[weight.name] = "initializer"
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node.name
    used_names = set(producers) | {name for node in model.graph.node for name in node.input if name}
    if set(tensors) != used_names:
        raise AssertionError("Identity manifest must describe every tiny-model tensor")
    manifest = {
        "schema_version": 1, "kind": TEST_KIND,
        "scope": "Unchanged synthetic three-Conv graph; not a production backbone extraction",
        "source_full_onnx_sha256": model_hash, "subgraph_onnx_sha256": model_hash,
        "policy_rule_fingerprint_sha256": policy["fingerprint_sha256"],
        "source_full_policy_fingerprint": build_report["policy_fingerprint"],
        "precision_plan_sha256": run_engine.sha256(build_report_path),
        "precision_plan_source": "Actual cached-build report containing applied precision constraints",
        "inputs": [describe(value) for value in model.graph.input],
        "outputs": [describe(value) for value in model.graph.output],
        "int8_convolution_names": int8_names,
        "coverage": {"expected_int8_convolutions": 1, "retained_int8_convolutions": 1,
                     "missing": [], "extra": []},
        "preservation": {key: True for key in ("node_protobufs_identical",
            "initializer_protobufs_identical", "input_value_info_identical", "tensor_names_identical")},
        "tensor_mapping": [{"source_name": name, "subgraph_name": name,
            "dtype": tensors[name]["dtype"], "shape": tensors[name]["shape"],
            "producer_name": producers[name]} for name in sorted(tensors)],
    }
    run_engine.write_json(output, manifest)
    return manifest


def inspect_precisions(path):
    layers = json.loads(path.read_text(encoding="utf-8"))["Layers"]
    actual = []
    for prefix, expected in (("/img_backbone/conv1/", "Half"),
                             ("/img_backbone/layer1/", "Int8"),
                             ("/pts_bbox_head/", "Half")):
        matches = [layer for layer in layers if layer.get("ParameterType") == "Convolution"
                   and layer["Name"].startswith(prefix)]
        if len(matches) != 1 or matches[0].get("Weights", {}).get("Type") != expected:
            raise AssertionError(f"Engine did not implement {prefix} with {expected} weights")
        layer = matches[0]
        if not all(expected in tensor["Format/Datatype"] for tensor in layer["Inputs"]):
            raise AssertionError(f"Engine input precision differs for {layer['Name']}")
        actual.append({"name": layer["Name"], "input_and_weight_type": expected,
                       "tactic": layer.get("TacticName"), "output_formats": [
                           tensor["Format/Datatype"] for tensor in layer["Outputs"]]})
    return actual


def test(directory):
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    rng = np.random.default_rng(731)
    nodes, weights = [], []
    current = "images"
    for label, scope, ci, co, kernel in (
        ("stem", "/img_backbone/conv1", 3, 8, 3),
        ("body", "/img_backbone/layer1/layer1.0/conv1", 8, 8, 3),
        ("head", "/pts_bbox_head/output", 8, 4, 1),
    ):
        weight = rng.normal(0, 0.05, (co, ci, kernel, kernel)).astype(np.float32)
        bias = rng.normal(0, 0.01, co).astype(np.float32)
        weights.extend([numpy_helper.from_array(weight, label + "_W"),
                        numpy_helper.from_array(bias, label + "_b")])
        out = label + "_conv"
        nodes.append(helper.make_node("Conv", [current, label + "_W", label + "_b"], [out],
                                      name=scope + "/Conv", pads=[kernel // 2] * 4))
        current = out
        if label != "head":
            current = label + "_relu"
            nodes.append(helper.make_node("Relu", [out], [current], name=scope + "/Relu"))
    graph = helper.make_graph(nodes, "synthetic_mixed_precision_interface_test",
        [helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(current, onnx.TensorProto.FLOAT, [1, 4, 8, 8])], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    model_path = directory / "model.onnx"
    onnx.save(model, model_path)
    calibration = directory / "calibration"
    calibration.mkdir()
    files = []
    for index in range(8):
        name = f"sample_{index}.npy"
        np.save(calibration / name, rng.uniform(-1, 1, (1, 3, 8, 8)).astype(np.float32))
        files.append(name)
    (calibration / "calibration.json").write_text(json.dumps({
        "schema_version": 1, "source_kind": "synthetic_interface_test",
        "dataset": "Seeded tiny CNN interface-test arrays, not camera data",
        "preprocessing": "Synthetic FP32 NCHW in [-1,1]; no model deployment calibration",
        "files": files,
    }), encoding="utf-8")
    held_out = rng.uniform(-1, 1, (1, 3, 8, 8)).astype(np.float32)
    np.save(directory / "input.npy", held_out)
    expected = ReferenceEvaluator(model).run(None, {"images": held_out})[0]
    np.savez(directory / "expected.npz", **{current: expected})
    args = argparse.Namespace(
        onnx=model_path, engine=directory / "first.engine", precision="mixed-int8",
        plan_only=False, calibration_dir=calibration, calibration_cache=directory / "ranges.cache",
        workspace_mib=64, max_aux_streams=0, plugins_fp32=False,
        report=directory / "first-build.json", device=0, verbose=False,
        synthetic_calibration_test=True, calibrated_ranges=None,  # Internal test only.
    )
    first, _ = run_engine.build(args)
    if first["calibration"]["consumed_batches"] != len(files):
        raise AssertionError("Calibration did not consume all synthetic test inputs")
    args.engine, args.report = directory / "cached.engine", directory / "cached-build.json"
    cached, _ = run_engine.build(args)
    if not cached["calibration"]["cache_hit"] or cached["calibration"]["consumed_batches"]:
        raise AssertionError("Second build did not reuse the correctly bound cache")
    actual_precisions = inspect_precisions(args.report.with_suffix(".layers.json"))
    cached_engine, cached_build_report = args.engine, args.report
    manifest_path, ranges_path = directory / "identity-subgraph.json", directory / "measured-ranges.json"
    identity_manifest(model, model_path, cached, cached_build_report, manifest_path)
    converted = convert_bound_cache(cache=args.calibration_cache, subgraph_manifest=manifest_path,
                                   output=ranges_path, allow_synthetic_for_testing=True)
    args.engine, args.report = directory / "ranges.engine", directory / "ranges-build.json"
    args.calibrated_ranges, args.calibration_dir, args.calibration_cache = ranges_path, None, None
    ranged, _ = run_engine.build(args)
    if (ranged["calibration"] is not None or not ranged["calibrated_ranges"]
            or ranged["calibrated_ranges"]["covered_int8_convolutions"] != 1):
        raise AssertionError("Third build must use measured ranges without a calibrator/cache")
    ranges_precisions = inspect_precisions(args.report.with_suffix(".layers.json"))
    report, code = run_engine.run(argparse.Namespace(
        engine=cached_engine, inputs=directory / "input.npy", reference=directory / "expected.npz",
        output=directory / "run", warmup=2, repeat=3, rtol=0.05, atol=0.002,
        device=0, verbose=False,
    ))
    ranges_report, ranges_code = run_engine.run(argparse.Namespace(
        engine=args.engine, inputs=directory / "input.npy", reference=directory / "expected.npz",
        output=directory / "ranges-run", warmup=2, repeat=3, rtol=0.05, atol=0.002,
        device=0, verbose=False,
    ))
    with np.load(directory / "ranges-run" / "outputs.npz", allow_pickle=False) as archive:
        ranges_outputs = {name: archive[name] for name in archive.files}
    comparison = run_engine.validate(ranges_outputs, directory / "run" / "outputs.npz", 1e-3, 1e-5, np)
    summary = {
        "scope": "Synthetic three-Conv GPU integration only; not BEVFormer calibration/accuracy/performance",
        "calibration_batches": first["calibration"]["consumed_batches"],
        "cache_reused": cached["calibration"]["cache_hit"],
        "validation": report["validation"],
        "inspection": str(cached_build_report.with_suffix(".layers.json")),
        "actual_precisions": actual_precisions,
        "ranges": {"artifact": str(ranges_path), "artifact_sha256": converted["artifact_sha256"],
                   "identity_manifest": str(manifest_path), "build": ranged["calibrated_ranges"],
                   "inspection": str(args.report.with_suffix(".layers.json")),
                   "actual_precisions": ranges_precisions,
                   "reference_validation": ranges_report["validation"]},
        "cache_vs_ranges": comparison,
        "reference_tolerances": {"rtol": 0.05, "atol": 0.002},
        "status": "passed" if code == ranges_code == 0 and comparison["status"] == "passed" else "failed",
    }
    run_engine.write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    raise SystemExit(test(parser.parse_args().output))
