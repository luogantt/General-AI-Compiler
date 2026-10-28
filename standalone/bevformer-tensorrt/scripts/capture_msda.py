#!/usr/bin/env python3
"""Capture one official MSDA's real five inputs and output from a full ONNX graph.

Build first without importing torch. Use --build-only and then --engine to run
in a fresh process when GPU memory is limited. Each invocation writes a new
experiment directory; the original ONNX/ranges/images are never rewritten.

Example (from the standalone project):
  python scripts/capture_msda.py --onnx models/adapted.onnx --images images.npy \
      --precision mixed-int8 --ranges ranges.json --output build/capture-build --build-only
  python scripts/capture_msda.py --engine build/capture-build/capture.engine \
      --images images.npy --output build/capture-run
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


# Support the checked-out standalone project as well as an installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevformer_trt import engine as engine_tools  # noqa: E402


KEYS = ("value", "shapes", "starts", "locations", "weights", "expected")
OFFICIAL_MSDA = "MultiscaleDeformableAttnPlugin_TRT"
SCHEMA = "gac.msda.real_capture.v1"
IMPACT = (
    "Parsed the original complete ONNX, applied precision/range policy before changing outputs, "
    "then exposed only one official MSDA's five inputs and output. Added constrained Identity "
    "boundaries and removed original output marks; TensorRT may prune downstream work and select "
    "different tactics/reformats. These are real-image activations from this capture engine, "
    "not a claim of bitwise equality to the original full-model engine."
)


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def source_record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": engine_tools.sha256(path)}


def verify_source(record, override=None):
    path = Path(override) if override is not None else Path(record["path"])
    if engine_tools.sha256(path) != record["sha256"]:
        raise ValueError(f"Source SHA-256 changed: {path}")
    return path


def select_node(path, substring):
    import onnx

    # All precision modes require embedded weights so the recorded ONNX digest
    # identifies the actual model, not just a protobuf referencing other files.
    engine_tools.require_embedded_weights(path)
    model = onnx.load(path, load_external_data=False)
    selected = [node for node in model.graph.node
                if node.op_type == OFFICIAL_MSDA and substring in node.name]
    if len(selected) != 1:
        candidates = [node.name for node in model.graph.node if node.op_type == OFFICIAL_MSDA]
        raise ValueError(f"Expected one official MSDA matching {substring!r}; found {len(selected)}. "
                         f"Available nodes: {candidates}")
    node = selected[0]
    if len(node.input) != 5 or len(node.output) != 1:
        raise ValueError("Selected official MSDA does not have the five-input, one-output ABI")
    attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
    version = attributes.get("plugin_version", b"1")
    namespace = attributes.get("plugin_namespace", b"")
    return {"name": node.name, "op_type": node.op_type, "domain": node.domain,
            "inputs": list(node.input), "outputs": list(node.output),
            "plugin_version": version.decode() if isinstance(version, bytes) else str(version),
            "plugin_namespace": namespace.decode() if isinstance(namespace, bytes) else str(namespace)}


def inspect_io(engine, trt):
    result = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"Capture requires nonempty static I/O: {name}: {shape}")
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ValueError(f"Capture requires LINEAR I/O: {name}")
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"Capture requires device I/O: {name}")
        result.append({"name": name, "shape": list(shape), "dtype": str(engine.get_tensor_dtype(name)),
                       "mode": "input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output"})
    return result


def check_capture_io(engine, trt, mapping, precision, *, keys=KEYS, integer_keys=("shapes", "starts")):
    io = inspect_io(engine, trt)
    outputs = {item["name"] for item in io if item["mode"] == "output"}
    if set(mapping) != set(keys) or outputs != set(mapping.values()) or len(outputs) != len(keys):
        raise ValueError("Engine outputs do not match the recorded plugin capture boundaries")
    target = trt.float32 if precision == "fp32" else trt.float16
    for key, name in mapping.items():
        expected = trt.int32 if key in integer_keys else target
        actual = engine.get_tensor_dtype(name)
        if actual != expected:
            raise ValueError(f"Capture boundary {key} has actual dtype {actual}, expected {expected}")
    inputs = [item for item in io if item["mode"] == "input"]
    if len(inputs) != 1:
        raise ValueError(f"A real images.npy capture requires one image input; got {inputs}")
    return io


def mark_capture_boundaries(network, layer, trt, precision, *, keys=KEYS,
                            integer_keys=("shapes", "starts"), prefix="gac_msda_capture", operation="msda"):
    target = trt.float32 if precision == "fp32" else trt.float16
    mapping, boundaries = {}, []
    original_outputs = [network.get_output(i).name for i in range(network.num_outputs)]
    while network.num_outputs:
        network.unmark_output(network.get_output(0))
    occupied = {network.get_input(i).name for i in range(network.num_inputs)}
    for index in range(network.num_layers):
        candidate = network.get_layer(index)
        occupied.update(candidate.get_output(i).name for i in range(candidate.num_outputs))
    for key in keys:
        name = f"{prefix}/{key}"
        if name in occupied:
            raise ValueError(f"Capture tensor name already exists: {name}")
        mapping[key] = name
    for index, key in enumerate(keys[:-1]):
        tensor = layer.get_input(index)
        expected = trt.int32 if key in integer_keys else target
        if tensor is None or (key in integer_keys and tensor.dtype != trt.int32):
            raise ValueError(f"Unexpected {operation} input {index}; metadata must be INT32")
        if key not in integer_keys and tensor.dtype not in (trt.float16, trt.float32):
            raise ValueError(f"Unexpected nonfloating {operation} input {index}")
        original = {"key": key, "source_tensor": tensor.name, "source_dtype": str(tensor.dtype)}
        identity = network.add_identity(tensor)
        if identity is None:
            raise RuntimeError(f"Could not create capture identity for {key}")
        identity.name = f"{prefix}/identity_{key}"
        identity.set_output_type(0, expected)
        captured = identity.get_output(0)
        captured.name = mapping[key]
        layer.set_input(index, captured)
        network.mark_output(captured)
        # Marking an intermediate output can otherwise retain its inferred FP32
        # network type despite a half output precision constraint on its producer.
        captured.dtype = expected
        boundaries.append({**original, "capture_tensor": captured.name,
                           "shape": list(captured.shape), "required_dtype": str(expected),
                           f"feeds_selected_{operation}_input": index})
    layer.set_output_type(0, target)
    source = layer.get_output(0)
    identity = network.add_identity(source)
    if identity is None:
        raise RuntimeError("Could not create capture output identity")
    identity.name = f"{prefix}/identity_expected"
    identity.set_output_type(0, target)
    captured = identity.get_output(0)
    captured.name = mapping["expected"]
    network.mark_output(captured)
    captured.dtype = target
    boundaries.append({"key": "expected", "source_tensor": source.name,
                       "capture_tensor": captured.name, "shape": list(captured.shape),
                       "required_dtype": str(target)})
    return mapping, boundaries, original_outputs


class MSDACaptureSpec:
    """Operation-specific metadata for the shared full-model capture workflow."""
    schema = SCHEMA
    impact = IMPACT
    prefix = "gac_msda_capture"
    operation = "msda"
    integer_keys = ("shapes", "starts")
    default_node_substring = "encoder/layers.0/attentions.1"
    precisions = ("fp32", "fp16", "mixed-int8")
    description = globals()["__doc__"]
    expected_description = "official NVIDIA MSDA output from capture engine"

    @staticmethod
    def select_node(path, substring):
        return select_node(path, substring)

    @staticmethod
    def keys(selected):
        return KEYS

    @staticmethod
    def validate_arrays(arrays, np, selected):
        return validate_arrays(arrays, np)


def build_capture(args, capture=None):
    import numpy as np

    capture = capture or MSDACaptureSpec()
    sources = {"onnx": source_record(args.onnx), "images": source_record(args.images),
               "ranges": source_record(args.ranges) if args.ranges else None}
    selected = capture.select_node(args.onnx, args.node_substring)
    keys = capture.keys(selected)
    gc.collect()
    images = np.load(args.images, allow_pickle=False, mmap_mode="r")
    if not isinstance(images, np.ndarray):
        raise ValueError("--images must be a single .npy array")
    image_spec = {"shape": list(images.shape), "dtype": str(images.dtype)}
    del images
    trt, logger, environment, driver = engine_tools.initialize_builder(args)
    if "torch" in sys.modules:
        raise RuntimeError("Capture build must run before importing torch; use a fresh process")
    print("BUILD_ENVIRONMENT " + json.dumps(environment), file=sys.stderr, flush=True)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file(str(args.onnx.resolve()))
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed or errors:
        raise RuntimeError("Full ONNX parse failed: " + "\n".join(errors))
    plugin_types = {getattr(trt.LayerType, name) for name in ("PLUGIN_V2", "PLUGIN_V3")
                    if hasattr(trt.LayerType, name)}
    candidates = [network.get_layer(i) for i in range(network.num_layers)
                  if network.get_layer(i).name == selected["name"] and network.get_layer(i).type in plugin_types]
    if len(candidates) != 1:
        raise ValueError(f"Parsed network lacks unique selected plugin layer: {selected['name']}")
    layer = candidates[0]
    if layer.num_inputs != len(keys) - 1 or layer.num_outputs != 1:
        raise ValueError("Parsed selected plugin has an incompatible ABI")
    if network.num_inputs != 1:
        raise ValueError("Capture currently requires the full model's single image input")
    image_input = network.get_input(0)
    if (list(image_input.shape) != image_spec["shape"]
            or np.dtype(trt.nptype(image_input.dtype)) != np.dtype(image_spec["dtype"])):
        raise ValueError(f"images.npy does not match full-model input {image_input.name}")
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.max_aux_streams = 0
    config.builder_optimization_level = args.builder_optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if args.tactic_dram_mib is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, args.tactic_dram_mib << 20)
        if config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) != args.tactic_dram_mib << 20:
            raise RuntimeError("TensorRT rejected --tactic-dram-mib")
    policy, ranges = None, None
    if args.precision != "fp32":
        config.set_flag(trt.BuilderFlag.FP16)
    if args.precision == "mixed-int8":
        from bevformer_trt.precision_policy import apply_precision_policy
        from bevformer_trt.calibration_ranges import apply_calibrated_ranges

        # These operate on the COMPLETE parsed network and original file digest.
        # No subgraph ONNX or altered checksum is substituted to bypass coverage.
        policy = apply_precision_policy(network, config, trt, plugins_fp32=False)
        ranges = apply_calibrated_ranges(network, trt, path=args.ranges,
                                        model_sha256=sources["onnx"]["sha256"], precision_policy=policy)
    mapping, boundaries, removed_outputs = mark_capture_boundaries(
        network, layer, trt, args.precision, keys=keys, integer_keys=capture.integer_keys,
        prefix=capture.prefix, operation=capture.operation)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT could not build the capture engine; inspect diagnostics")
    payload = bytes(plan)
    del plan, candidates, layer, image_input, parser, network, config, builder, driver
    gc.collect()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError("Capture engine could not be deserialized for dtype inspection")
    io = check_capture_io(engine, trt, mapping, args.precision, keys=keys, integer_keys=capture.integer_keys)
    inspector = engine.create_engine_inspector()
    inspection = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    engine_path = args.output / "capture.engine"
    report_path = engine_path.with_suffix(".build.json")
    for record in sources.values():
        if record is not None:
            verify_source(record)
    report = {"schema": capture.schema, "status": "built", "sources": sources, "selected_node": selected,
              "precision": args.precision, "image_input": image_spec, "capture_tensors": mapping,
              "capture_boundaries": boundaries, "removed_outputs": removed_outputs, "engine_io": io,
              "engine": str(engine_path.resolve()), "engine_sha256": hashlib.sha256(payload).hexdigest(),
              "engine_bytes": len(payload), "build_seconds": elapsed, "environment": environment,
              "precision_policy": policy, "calibrated_ranges": ranges,
              "workspace_mib": args.workspace_mib, "tactic_dram_mib": args.tactic_dram_mib,
              "builder_optimization_level": args.builder_optimization_level,
              "impact": capture.impact, "compute": "TensorRT; Python only controls loading, constraints and copies",
              "capture_dtype_validation": ("actual engine I/O matches requested floating precision and INT32 metadata"
                                           if capture.integer_keys else "actual engine I/O matches requested floating precision")}
    with engine_path.open("xb") as target:
        target.write(payload)
    with (args.output / "capture.layers.json").open("x", encoding="utf-8") as target:
        target.write(inspection)
    save_json(report_path, report)
    del inspector, engine, runtime
    gc.collect()
    return engine_path, report


def validate_arrays(arrays, np):
    value, shapes, starts, locations, weights, expected = (arrays[key] for key in KEYS)
    if value.ndim != 4 or shapes.ndim != 2 or starts.ndim != 1 or locations.ndim != 6 or weights.ndim != 5:
        raise ValueError("Captured MSDA ranks are invalid")
    b, s, h, c = value.shape
    if locations.shape[0] != b or locations.shape[2] != h or locations.shape[5] != 2:
        raise ValueError("Captured MSDA batch/head/coordinate dimensions disagree")
    _, q, _, levels, points, _ = locations.shape
    if (shapes.shape != (levels, 2) or starts.shape != (levels,)
            or weights.shape != (b, q, h, levels, points) or expected.shape != (b, q, h, c)
            or any(dimension <= 0 for dimension in (b, s, h, c, q, levels, points))):
        raise ValueError("Captured MSDA shapes are inconsistent or empty")
    if shapes.dtype != np.int32 or starts.dtype != np.int32:
        raise ValueError("Captured metadata must be INT32")
    if value.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("Captured values must be FP16 or FP32")
    if any(array.dtype != value.dtype for array in (locations, weights, expected)):
        raise ValueError("Captured floating tensor dtypes disagree")
    heights_widths = shapes.astype(np.int64)
    offsets = starts.astype(np.int64)
    areas = heights_widths[:, 0] * heights_widths[:, 1]
    if (np.any(heights_widths <= 0) or np.any(offsets < 0) or np.any(offsets >= s)
            or np.any(areas > s - offsets)):
        raise ValueError("Captured spatial_shapes/level_starts exceed the value sequence")
    finite = {key: bool(np.isfinite(arrays[key]).all()) for key in ("value", "locations", "weights", "expected")}
    if not all(finite.values()):
        raise ValueError(f"Captured MSDA contains nonfinite tensors: {finite}")
    return {"status": "passed", "metadata_in_bounds": True, "finite": finite,
            "spatial_shapes": shapes.tolist(), "level_starts": starts.tolist(),
            "levels_exhaust_sequence": int(areas.sum()) == s,
            "levels_contiguous": bool(np.array_equal(offsets, np.r_[0, np.cumsum(areas[:-1])])),
            "dimensions": {"B": b, "S": s, "H": h, "C": c, "Q": q, "L": levels, "P": points}}


def run_capture(args, engine_path, build_report, capture=None):
    capture = capture or MSDACaptureSpec()
    if build_report.get("schema") != capture.schema or build_report.get("status") != "built":
        raise ValueError("--engine requires this script's successful .build.json manifest")
    if engine_tools.sha256(engine_path) != build_report["engine_sha256"]:
        raise ValueError("Capture engine SHA-256 does not match its build manifest")
    sources = build_report["sources"]
    verify_source(sources["images"], args.images)
    verify_source(sources["onnx"], args.onnx)
    if sources["ranges"] is not None:
        verify_source(sources["ranges"], args.ranges)
    elif args.ranges is not None:
        raise ValueError("--ranges was supplied but this capture engine was not built with ranges")
    # initialize() imports torch only at execution time, after builder objects
    # have been released (or in this separate --engine process).
    np, trt, torch, logger = engine_tools.initialize(args)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize capture engine")
    mapping = build_report["capture_tensors"]
    io = check_capture_io(engine, trt, mapping, build_report["precision"],
                          keys=capture.keys(build_report["selected_node"]), integer_keys=capture.integer_keys)
    if io != build_report["engine_io"]:
        raise ValueError("Capture engine I/O differs from the build manifest")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not allocate capture execution context")
    arrays = engine_tools.load_inputs(args.images, [item["name"] for item in io if item["mode"] == "input"], np)
    stream = torch.cuda.Stream()
    buffers = {}
    with torch.cuda.stream(stream):
        for item in io:
            name, shape = item["name"], tuple(item["shape"])
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            if item["mode"] == "input":
                array = arrays[name]
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError(f"Real input {name} must be {shape} {dtype}, got {array.shape} {array.dtype}")
                buffers[name] = torch.from_numpy(np.array(array, copy=True, order="C")).to("cuda")
            else:
                buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
            if not context.set_tensor_address(name, buffers[name].data_ptr()):
                raise RuntimeError(f"Could not bind capture tensor {name}")
        missing = context.infer_shapes()
        if missing:
            raise ValueError(f"Capture shape inference is incomplete: {missing}")
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT capture execution failed")
    stream.synchronize()
    captured = {key: buffers[name].cpu().numpy().copy() for key, name in mapping.items()}
    validation = capture.validate_arrays(captured, np, build_report["selected_node"])
    verify_source(sources["images"], args.images)
    archive_path = args.output / "capture.npz"
    with archive_path.open("xb") as target:
        np.savez(target, **captured)
    report = {"schema": capture.schema, "status": "captured", "source_kind": "real_preprocessed_images",
              "sources": sources, "engine": str(engine_path.resolve()),
              "engine_sha256": build_report["engine_sha256"], "selected_node": build_report["selected_node"],
              "precision": build_report["precision"], "engine_io": io, "capture_tensors": mapping,
              "arrays": {key: {"shape": list(array.shape), "dtype": str(array.dtype), "bytes": array.nbytes}
                         for key, array in captured.items()},
              "archive": str(archive_path.resolve()), "archive_sha256": engine_tools.sha256(archive_path),
              "validation": validation, "environment": engine_tools.environment(trt, torch),
              "impact": capture.impact, "timing": "not measured", "expected": capture.expected_description,
              "compute": "TensorRT; torch supplies buffers/copies only; Python validates saved metadata"}
    save_json(args.output / "capture.report.json", report)
    return report


def main(argv=None, capture=None):
    capture = capture or MSDACaptureSpec()
    parser = argparse.ArgumentParser(description=capture.description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, help="Original complete adapted.onnx; required for building")
    parser.add_argument("--images", "--inputs", dest="images", type=Path, required=True,
                        help="Real preprocessed images.npy; no preprocessing or dtype casts are performed")
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent experiment directory")
    parser.add_argument("--engine", type=Path, help="Execute an existing capture.engine instead of building")
    parser.add_argument("--build-report", type=Path, help="Existing engine manifest; defaults to ENGINE.build.json")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--precision", choices=capture.precisions, default="fp16")
    parser.add_argument("--ranges", type=Path, help="Provenance-bound original-model calibrated ranges for mixed-int8")
    parser.add_argument("--node-substring", default=capture.default_node_substring)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=3)
    parser.add_argument("--tactic-dram-mib", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.images.suffix.lower() != ".npy":
            raise ValueError("--images requires a real preprocessed .npy file")
        if args.engine is not None and args.build_only:
            raise ValueError("--engine and --build-only cannot be combined")
        if args.engine is None:
            if args.onnx is None:
                raise ValueError("--onnx is required to build a capture engine")
            if (args.precision == "mixed-int8") != (args.ranges is not None):
                raise ValueError("--ranges is required exactly when building --precision mixed-int8")
            if args.build_report is not None:
                raise ValueError("--build-report applies only to --engine")
        if args.workspace_mib <= 0 or args.device < 0:
            raise ValueError("Workspace must be positive and device must be nonnegative")
        dram = args.tactic_dram_mib
        if dram is not None and (dram <= 0 or dram & (dram - 1)):
            raise ValueError("--tactic-dram-mib must be a positive power of two")
        args.output.mkdir(parents=True, exist_ok=False)
        if args.engine is None:
            engine_path, build_report = build_capture(args, capture)
        else:
            engine_path = args.engine
            build_report = json.loads((args.build_report or engine_path.with_suffix(".build.json")).read_text(encoding="utf-8"))
        if args.build_only:
            print(json.dumps({"status": "built", "engine": str(engine_path),
                              "manifest": str(engine_path.with_suffix('.build.json'))}))
        else:
            report = run_capture(args, engine_path, build_report, capture)
            print(json.dumps({key: report[key] for key in ("status", "archive", "precision", "selected_node")}, indent=2))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
