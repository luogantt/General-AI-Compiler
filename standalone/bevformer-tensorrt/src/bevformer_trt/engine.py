#!/usr/bin/env python3
"""Build static TensorRT 10 engines and benchmark them using PyTorch CUDA buffers.

TensorRT owns engine compilation and GPU execution; PyTorch supplies CUDA buffers. CUDA-event measurements
exclude input/output copies, preprocessing, loading, and engine construction.
"""

import argparse
import atexit
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


_CUDA_MATH_LIBRARIES = []
_PLUGIN_LIBRARIES = {}


def load_plugin_libraries(paths=()):
    """Load plugin registrars before parsing/deserialization and retain handles."""
    for value in paths or ():
        path = Path(value).resolve(strict=True)
        digest = sha256(path)
        key = str(path)
        if key in _PLUGIN_LIBRARIES:
            if _PLUGIN_LIBRARIES[key][1] != digest:
                raise RuntimeError(f"Plugin library changed after loading: {path}; start a new process")
            continue
        try:
            handle = ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise RuntimeError(f"Cannot load TensorRT --plugin-library {path}: {error}") from error
        _PLUGIN_LIBRARIES[key] = (handle, digest)
    return [{"path": path, "sha256": value[1]} for path, value in _PLUGIN_LIBRARIES.items()]


def plugin_error_hint():
    return (" Custom TensorRT plugins must be loaded with --plugin-library PATH before parsing "
            "or deserialization (GACMSDeformAttn version 1, empty namespace). "
            "Inspect TensorRT diagnostics for missing creators or ABI/version mismatch.")


def load_cuda_math_libraries():
    """Expose pip-installed cuBLAS to plugins without importing PyTorch.

    NVIDIA's DCNv2 plugin opens cuBLAS by SONAME. A CUDA Driver context alone
    does not add the nvidia-cublas wheel's directory to the OS loader search.
    Load Lt first because cuBLAS depends on it; retain handles for this process.
    """
    if os.name != "posix" or _CUDA_MATH_LIBRARIES:
        return
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        try:
            handle = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            try:
                distribution = package_metadata.distribution("nvidia-cublas-cu12")
            except package_metadata.PackageNotFoundError as error:
                raise RuntimeError(f"Cannot load {name}; configure CUDA library paths or install nvidia-cublas-cu12") from error
            matches = [distribution.locate_file(file) for file in distribution.files or () if file.name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Could not locate a unique {name} in nvidia-cublas-cu12")
            handle = ctypes.CDLL(str(matches[0]), mode=ctypes.RTLD_GLOBAL)
        _CUDA_MATH_LIBRARIES.append(handle)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_timing_cache(input_path=None, output_path=None, *, artifact_paths=(), plan_only=False):
    """Read/audit requested timing data and reject path hazards before CUDA init."""
    if input_path is None and output_path is None:
        return None, None, None
    if plan_only:
        raise ValueError("Timing-cache options require an engine build; they cannot be combined with --plan-only")
    input_path = Path(input_path) if input_path is not None else None
    output_path = Path(output_path) if output_path is not None else None
    paths = [Path(path).resolve() for path in (*artifact_paths, input_path, output_path) if path is not None]
    if len(paths) != len(set(paths)):
        raise ValueError("Timing-cache input/output and model/build artifact paths must all differ")
    # Resolving handles symlinks; samefile also handles existing hard-link aliases.
    for index, path in enumerate(paths):
        if path.exists() and any(other.exists() and path.samefile(other) for other in paths[:index]):
            raise ValueError("Timing-cache input/output must not alias another model/build artifact")
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(f"Timing-cache output already exists: {output_path}")
        if any(parent.exists() and not parent.is_dir() for parent in output_path.parents):
            raise ValueError(f"Timing-cache output parent is not a directory: {output_path}")
    payload, source = b"", None
    if input_path is not None:
        if not input_path.is_file():
            raise FileNotFoundError(f"Timing-cache input must name an existing file: {input_path}")
        payload = input_path.read_bytes()
        if not payload:
            raise ValueError("Timing-cache input is empty; omit --timing-cache-input to create a new cache")
        source = {"path": str(input_path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                  "bytes": len(payload)}
    audit = {"input": source, "output": None, "ignore_mismatch": False,
             "initialization": "serialized_input" if input_path is not None else "empty_cache",
             "scope": "Reuse TensorRT tactic timing measurements; does not guarantee identical tactic selection or numerical outputs"}
    return payload, audit, output_path


def attach_timing_cache(config, payload):
    """Return the live cache; its lifetime must exceed the builder's lifetime."""
    cache = config.create_timing_cache(payload)
    if cache is None:
        raise RuntimeError("TensorRT could not create the requested timing cache")
    if not config.set_timing_cache(cache, ignore_mismatch=False):
        raise RuntimeError("TensorRT rejected the timing cache; device/version mismatch is not ignored")
    return cache


def serialize_timing_cache(config):
    cache = config.get_timing_cache()
    if cache is None:
        raise RuntimeError("TensorRT returned no timing cache after the engine build")
    serialized = cache.serialize()
    if serialized is None:
        raise RuntimeError("TensorRT could not serialize the timing cache")
    payload = bytes(serialized)
    if not payload:
        raise RuntimeError("TensorRT serialized an empty timing cache")
    return payload


def environment_base(trt):
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,clocks.sm,power.limit",
         "--format=csv,noheader"], capture_output=True, text=True, check=False,
    )
    return {
        "utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
        "python": sys.version.split()[0], "tensorrt": trt.__version__,
        "nvidia_smi": query.stdout.strip() if query.returncode == 0 else query.stderr.strip(),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
        "plugin_libraries": [{"path": path, "sha256": value[1]}
                             for path, value in _PLUGIN_LIBRARIES.items()],
    }


def environment(trt, torch):
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    result = environment_base(trt)
    result.update({
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_initialization": "PyTorch; required for run buffers and events",
        "gpu": {"index": device, "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory},
    })
    return result


def initialize_trt(args):
    load_cuda_math_libraries()
    import tensorrt as trt

    if int(trt.__version__.split(".")[0]) != 10:
        raise ValueError("This runner targets TensorRT 10; precision flags differ in TensorRT 11")
    logger = trt.Logger(trt.Logger.INFO if args.verbose else trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("TensorRT built-in plugin registration failed")
    load_plugin_libraries(getattr(args, "plugin_library", ()))
    return trt, logger


def initialize(args):
    import numpy as np
    import torch

    torch.cuda.set_device(args.device)
    trt, logger = initialize_trt(args)
    return np, trt, torch, logger


def initialize_builder(args):
    """Select the CUDA primary context without importing PyTorch/CUDA kernels."""
    cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    pointer = ctypes.POINTER
    integer, size, context_type = ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p

    def call(name, types, *values):
        function = getattr(cuda, name)
        function.argtypes, function.restype = types, integer
        status = function(*values)
        if status != 0:
            raise RuntimeError(f"CUDA Driver {name} failed with CUresult={status}")

    call("cuInit", [ctypes.c_uint], 0)
    device, context = integer(), context_type()
    call("cuDeviceGet", [pointer(integer), integer], ctypes.byref(device), args.device)
    call("cuDevicePrimaryCtxRetain", [pointer(context_type), integer], ctypes.byref(context), device)
    # This CLI builds once and exits. Release after its TensorRT builder/parser
    # objects have been destroyed, not while exception tracebacks retain them.
    release = cuda.cuDevicePrimaryCtxRelease_v2
    release.argtypes, release.restype = [integer], integer
    atexit.register(release, device)
    call("cuCtxSetCurrent", [context_type], context)
    name = ctypes.create_string_buffer(256)
    major, minor, driver_version, total_device = integer(), integer(), integer(), size()
    call("cuDeviceGetName", [pointer(ctypes.c_char), integer, integer], name, len(name), device)
    call("cuDeviceComputeCapability", [pointer(integer), pointer(integer), integer],
         ctypes.byref(major), ctypes.byref(minor), device)
    call("cuDeviceTotalMem_v2", [pointer(size), integer], ctypes.byref(total_device), device)
    call("cuDriverGetVersion", [pointer(integer)], ctypes.byref(driver_version))
    trt, logger = initialize_trt(args)
    free, total = size(), size()
    call("cuMemGetInfo_v2", [pointer(size), pointer(size)], ctypes.byref(free), ctypes.byref(total))
    result = environment_base(trt)
    try:
        torch_version = package_metadata.version("torch")
    except package_metadata.PackageNotFoundError:
        torch_version = None
    result.update({
        "torch_installed_version": torch_version,
        "torch_imported": "torch" in sys.modules,
        "cuda_math_libraries": [handle._name for handle in _CUDA_MATH_LIBRARIES],
        "cuda_initialization": "ctypes CUDA Driver primary context; no PyTorch initialization",
        "cuda_driver_api_version": driver_version.value,
        "gpu": {"index": args.device, "name": name.value.decode("utf-8", errors="replace"),
                "compute_capability": [major.value, minor.value], "total_memory_bytes": total_device.value},
        "memory_snapshot_before_builder": {
            "free_bytes": free.value, "total_bytes": total.value,
            "device_used_bytes": total.value - free.value,
            "scope": "One device-wide cuMemGetInfo snapshot after CUDA/TRT initialization; not a peak",
        },
    })
    return trt, logger, result, CudaMemory(cuda)


class CudaMemory:
    """Small synchronous allocator for calibration; the primary context is selected above."""

    def __init__(self, library):
        self.library = library

    def call(self, name, types, *values):
        function = getattr(self.library, name)
        function.argtypes, function.restype = types, ctypes.c_int
        status = function(*values)
        if status:
            raise RuntimeError(f"CUDA Driver {name} failed with CUresult={status}")

    def malloc(self, nbytes):
        address = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
                  ctypes.byref(address), nbytes)
        return address.value

    def copy_host_to_device(self, address, array):
        if not array.flags.c_contiguous:
            raise ValueError("Calibration host buffer must be contiguous")
        self.call("cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
                  address, array.ctypes.data, array.nbytes)

    def free(self, address):
        self.call("cuMemFree_v2", [ctypes.c_uint64], address)


def require_static(name, shape):
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"Dynamic I/O is not supported: {name!r} has shape {list(shape)}")


def require_embedded_weights(path):
    """The calibration cache hashes one ONNX file, so require weights inside it."""
    import onnx
    pending = [onnx.load(path, load_external_data=False).graph]
    while pending:
        graph = pending.pop()
        tensors = list(graph.initializer)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
                if attribute.HasField("g"):
                    pending.append(attribute.g)
                pending.extend(attribute.graphs)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in tensors):
            raise ValueError("mixed-int8 requires self-contained ONNX weights; first run bevformer-trt adapt")


def build(args):
    ranges_path = getattr(args, "calibrated_ranges", None)
    dcn_path = getattr(args, "dcn_config", None)
    if dcn_path is not None:
        if args.precision == "fp32" or args.plugins_fp32:
            raise ValueError("Experimental DCN replacement requires FP16 plugin edges")
        if args.precision == "mixed-int8" and (ranges_path is None or args.calibration_dir or args.calibration_cache):
            raise ValueError("DCN A/B experiment requires frozen, source-bound --calibrated-ranges; recalibration is a separate experiment")
    if not args.plan_only and args.engine is None:
        raise ValueError("--engine is required unless --plan-only is selected")
    if (args.engine is not None and args.engine.exists()) or args.report.exists():
        raise FileExistsError("Engine/report already exists; select new output paths")
    if args.engine is not None and args.engine.resolve() == args.report.resolve():
        raise ValueError("Engine and report paths must differ")
    if not args.plan_only and args.report.with_suffix(".layers.json").exists():
        raise FileExistsError("Engine inspection report already exists; choose a new report path")
    if ranges_path and (args.calibration_dir or args.calibration_cache or args.plan_only):
        raise ValueError("--calibrated-ranges cannot be combined with calibration inputs/cache or --plan-only")
    if args.precision == "mixed-int8" and not args.plan_only and args.calibration_dir is None and ranges_path is None:
        raise ValueError("mixed-int8 requires real --calibration-dir or bound --calibrated-ranges; use --plan-only to review the policy first")
    if args.precision != "mixed-int8" and (args.calibration_dir or args.calibration_cache or ranges_path):
        raise ValueError("Calibration options apply only to --precision mixed-int8")
    if args.precision == "mixed-int8":
        require_embedded_weights(args.onnx)
    cache_path = ((args.calibration_cache or args.report.with_suffix(".calibration.cache"))
                  if args.precision == "mixed-int8" and not args.plan_only and ranges_path is None else None)
    artifacts = [args.onnx, args.report, args.report.with_suffix(".layers.json")]
    artifacts.extend(path for path in (args.engine, cache_path, ranges_path) if path is not None)
    if len({path.resolve() for path in artifacts}) != len(artifacts):
        raise ValueError("Model, engine, report, inspection and calibration cache paths must differ")
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    if args.max_aux_streams < 0:
        raise ValueError("--max-aux-streams must be nonnegative")
    optimization_level = getattr(args, "builder_optimization_level", 3)
    tactic_dram_mib = getattr(args, "tactic_dram_mib", None)
    if optimization_level not in range(6):
        raise ValueError("--builder-optimization-level must be between 0 and 5")
    if tactic_dram_mib is not None and (tactic_dram_mib <= 0 or tactic_dram_mib & (tactic_dram_mib - 1)):
        raise ValueError("--tactic-dram-mib must be a positive power of two for this TensorRT memory-pool API")
    timing_payload, timing_audit, timing_output = prepare_timing_cache(
        getattr(args, "timing_cache_input", None), getattr(args, "timing_cache_output", None),
        artifact_paths=artifacts, plan_only=args.plan_only)
    trt, logger, metadata, driver = initialize_builder(args)
    # Preserve the initial memory state in the build log even if construction
    # fails. Only successful construction writes engine/report artifacts.
    print("BUILD_ENVIRONMENT " + json.dumps(metadata), file=sys.stderr, flush=True)
    builder = trt.Builder(logger)
    # Explicit batch is always enabled in TRT 10. Deliberately use weak typing:
    # STRONGLY_TYPED disallows FP16 builder flags and per-layer precision hints.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    dcn_audit, parser_payload = None, None
    if dcn_path is not None:
        from .dcn import read_config, rewrite_for_parser
        source_payload = args.onnx.read_bytes()
        source_digest = hashlib.sha256(source_payload).hexdigest()
        parser_payload, dcn_audit = rewrite_for_parser(source_payload, read_config(dcn_path))
        del source_payload
        dcn_audit.update(source_onnx_sha256=source_digest,
                         transformed_onnx_sha256=hashlib.sha256(parser_payload).hexdigest(),
                         config_sha256=sha256(dcn_path),
                         calibration="Frozen original-model ranges for a controlled implementation A/B; no new calibration or accuracy claim")
        parsed = parser.parse(parser_payload)
    else:
        parsed = parser.parse_from_file(str(args.onnx.resolve()))
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed or errors:
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors or ["No parser diagnostic"])
                           + plugin_error_hint())
    io = []
    for mode, count, getter in (("INPUT", network.num_inputs, network.get_input),
                                 ("OUTPUT", network.num_outputs, network.get_output)):
        for index in range(count):
            tensor = getter(index)
            shape = tuple(tensor.shape)
            require_static(tensor.name, shape)
            io.append({"name": tensor.name, "mode": mode, "shape": list(shape), "dtype": str(tensor.dtype)})
    config = builder.create_builder_config()
    timing_cache = attach_timing_cache(config, timing_payload) if timing_audit is not None else None
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.builder_optimization_level = optimization_level
    if tactic_dram_mib is not None:
        # Profiling memory differs from execution workspace. A limit filters
        # candidate tactics; it is not a cap on the process's total GPU memory.
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, tactic_dram_mib << 20)
        if config.get_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM) != tactic_dram_mib << 20:
            raise RuntimeError("TensorRT did not accept the requested tactic DRAM limit")
    # Concurrent branches can extend activation lifetimes. Default to one
    # main stream for the 8 GB baseline; opt in to auxiliary streams explicitly.
    config.max_aux_streams = args.max_aux_streams
    # Clearing TF32 controls TensorRT tactics, not independent cuBLAS calls
    # inside plugins. NVIDIA_TF32_OVERRIDE=0 is recorded separately above.
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision in ("fp16", "mixed-int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    constrained = []
    layer_policy = None
    if args.precision == "mixed-int8":
        from .precision_policy import apply_precision_policy
        layer_policy = apply_precision_policy(network, config, trt, plugins_fp32=args.plugins_fp32)
    elif args.plugins_fp32:
        from .precision_policy import constrain_plugin_float_edges
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        plugin_types = {trt.LayerType.PLUGIN_V2, trt.LayerType.PLUGIN_V3}
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type in plugin_types:
                constrained.append(constrain_plugin_float_edges(network, layer, trt, precision="fp32"))
    model_hash = sha256(args.onnx)
    if dcn_audit is not None and dcn_audit["source_onnx_sha256"] != model_hash:
        raise ValueError("Source ONNX changed during DCN transformation")
    policy = {"requested": args.precision, "network_typing": "weak",
              "tf32_builder_flag": False, "fp16_allowed": args.precision != "fp32",
              "plugins_fp32": args.plugins_fp32, "fp32_plugin_layers": constrained,
              "mixed_layer_policy": layer_policy}
    policy_fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    if args.plan_only:
        report = {"status": "precision_plan_only", "scope": "Constraints only; no calibration, engine or inference",
                  "environment": metadata, "onnx": str(args.onnx.resolve()),
                  "onnx_file_sha256": model_hash, "io": io, "precision_policy": policy,
                  "policy_fingerprint": policy_fingerprint,
                  "dcn_implementation_experiment": dcn_audit}
        write_json(args.report, report)
        return report, 0
    calibrator, calibration, calibrated_ranges = None, None, None
    if ranges_path is not None:
        from .calibration_ranges import apply_calibrated_ranges
        calibrated_ranges = apply_calibrated_ranges(
            network, trt, path=ranges_path, model_sha256=model_hash,
            precision_policy=layer_policy,
            allow_synthetic_for_testing=getattr(args, "synthetic_calibration_test", False))
    elif args.precision == "mixed-int8":
        from .quantization import create_calibrator
        inputs = [item for item in io if item["mode"] == "INPUT"]
        if len(inputs) != 1 or inputs[0]["dtype"] != str(trt.float32):
            raise ValueError("Calibration currently requires one FP32 image input")
        calibrator, calibration = create_calibrator(
            trt, directory=args.calibration_dir, model_sha256=model_hash,
            policy_fingerprint=policy_fingerprint, input_name=inputs[0]["name"],
            input_shape=tuple(inputs[0]["shape"]), cache_path=cache_path, driver=driver,
            allow_synthetic_for_testing=getattr(args, "synthetic_calibration_test", False))
        config.int8_calibrator = calibrator
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    started = time.perf_counter()
    try:
        plan = builder.build_serialized_network(network, config)
        if calibrator is not None:
            calibrator.raise_if_error()
            if plan is not None:
                calibrator.assert_complete()
    finally:
        if calibrator is not None:
            calibrator.close()
            # A failed builder produces no engine/report. Preserve how far
            # calibration got in its log, without reporting a successful run.
            print("CALIBRATION_PROGRESS " + json.dumps({
                key: calibration.get(key) for key in
                ("consumed_batches", "cache_hit", "cache_written", "error")
            }), file=sys.stderr, flush=True)
    seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT could not build the engine; inspect builder diagnostics")
    payload = bytes(plan)
    timing_result = serialize_timing_cache(config) if timing_output is not None else None
    report = {
        "status": "built", "scope": "Engine construction only; no numerical validation",
        "environment": metadata, "onnx": str(args.onnx.resolve()),
        "onnx_file_sha256": model_hash,
        "hash_scope": "ONNX file itself; separate external weight files are not included",
        "engine": str(args.engine.resolve()), "engine_sha256": hashlib.sha256(payload).hexdigest(),
        "engine_bytes": len(payload), "build_seconds": seconds, "io": io,
        "precision_policy": policy, "policy_fingerprint": policy_fingerprint,
        "calibration": calibration,
        "calibrated_ranges": calibrated_ranges,
        "dcn_implementation_experiment": dcn_audit,
        "workspace_limit_bytes": args.workspace_mib << 20,
        "builder_optimization_level": optimization_level,
        "tactic_dram_limit_bytes": tactic_dram_mib << 20 if tactic_dram_mib is not None else None,
        "max_aux_streams": args.max_aux_streams,
        "workspace_note": "Workspace limit is not a limit on total GPU memory",
    }
    # Keep actual tactic/tensor-type evidence beside requested constraints.
    del parser, network, config, builder, parser_payload
    del timing_cache
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError("Built engine could not be loaded for precision inspection")
    inspector = engine.create_engine_inspector()
    inspection_path = args.report.with_suffix(".layers.json")
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    inspection_path.write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON), encoding="utf-8")
    report["engine_layer_inspection"] = str(inspection_path)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(payload)
    if timing_audit is not None:
        if timing_output is not None:
            timing_output.parent.mkdir(parents=True, exist_ok=True)
            with timing_output.open("xb") as output:
                output.write(timing_result)
            timing_audit["output"] = {"path": str(timing_output.resolve()),
                                      "sha256": hashlib.sha256(timing_result).hexdigest(), "bytes": len(timing_result)}
        timing_audit["status"] = "attached_and_built"
        report["timing_cache"] = timing_audit
    write_json(args.report, report)
    return report, 0


def load_inputs(path, names, np):
    if path.suffix.lower() == ".npy":
        if len(names) != 1:
            raise ValueError(f"A single .npy requires exactly one input; engine inputs: {names}")
        arrays = {names[0]: np.load(path, allow_pickle=False)}
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    else:
        raise ValueError("--inputs must be .npz (named tensors) or .npy (one engine input)")
    if set(arrays) != set(names):
        raise ValueError(f"Input names differ: expected {names}, received {sorted(arrays)}")
    return arrays


def validate(outputs, reference_path, rtol, atol, np):
    finite = {name: bool(np.isfinite(value).all()) for name, value in outputs.items()}
    result = {"status": "not_checked", "all_outputs_finite": all(finite.values()),
              "output_finite": finite, "scope": "Final output of the last measured execution only"}
    if reference_path is None:
        if not all(finite.values()):
            result["status"] = "failed_nonfinite"
        return result
    with np.load(reference_path, allow_pickle=False) as archive:
        if set(archive.files) != set(outputs):
            raise ValueError(f"Reference keys differ: engine {sorted(outputs)}, reference {archive.files}")
        comparisons = {}
        for name, actual in outputs.items():
            expected = archive[name]
            if actual.shape != expected.shape:
                raise ValueError(f"Reference shape differs for {name}: {expected.shape} vs {actual.shape}")
            ref_finite = bool(np.isfinite(expected).all())
            floating = actual.dtype.kind == "f" or expected.dtype.kind == "f"
            close = (np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
                     if floating else np.equal(actual, expected))
            passed = finite[name] and ref_finite and bool(close.all())
            detail = {"passed": passed, "shape": list(actual.shape),
                      "actual_dtype": str(actual.dtype), "reference_dtype": str(expected.dtype),
                      "reference_finite": ref_finite, "mismatched_elements": int((~close).sum()),
                      "elements": actual.size}
            if finite[name] and ref_finite and actual.size:
                difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                detail["max_absolute_error"] = float(difference.max())
                detail["max_relative_error_floor_1e_12"] = float(
                    (difference / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)).max())
            comparisons[name] = detail
    result.update(status="passed" if all(item["passed"] for item in comparisons.values()) else "failed",
                  reference=str(reference_path.resolve()), reference_sha256=sha256(reference_path),
                  rtol=rtol, atol=atol, outputs=comparisons)
    return result


def run(args):
    if args.repeat <= 0 or args.warmup < 0 or args.rtol < 0 or args.atol < 0:
        raise ValueError("repeat must be positive; warmup and tolerances must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new directory")
    np, trt, torch, logger = initialize(args)
    metadata = environment(trt, torch)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize engine on this GPU/TensorRT version" + plugin_error_hint())
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not allocate an execution context")
    io = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(engine.get_tensor_shape(name))
        require_static(name, shape)
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            raise ValueError(f"Non-LINEAR I/O requires packed buffers and is unsupported: {name}")
        io.append({"name": name, "shape": list(shape), "dtype": str(engine.get_tensor_dtype(name)),
                   "mode": "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT",
                   "location": str(engine.get_tensor_location(name))})
    names = [item["name"] for item in io if item["mode"] == "INPUT"]
    arrays = load_inputs(args.inputs, names, np)
    stream = torch.cuda.Stream()
    buffers = {}
    with torch.cuda.stream(stream):
        for item in io:
            name, shape = item["name"], tuple(item["shape"])
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            torch_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype
            host = engine.get_tensor_location(name) == trt.TensorLocation.HOST
            if item["mode"] == "INPUT":
                array = arrays[name]
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError(f"Input {name}: expected {shape} {dtype}, got {array.shape} {array.dtype}")
                # Copy before timing, on the same stream as TensorRT execution.
                tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
                buffers[name] = tensor.pin_memory() if host else tensor.to("cuda")
            else:
                buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cpu" if host else "cuda",
                                             pin_memory=host)
            if not context.set_tensor_address(name, buffers[name].data_ptr()):
                raise RuntimeError(f"Could not bind tensor {name}")
    stream.synchronize()
    missing = context.infer_shapes()
    if missing:
        raise ValueError(f"Shape inference needs values for: {missing}")
    # This is the context's enqueue-memory requirement, not weights, I/O,
    # driver state, or the process's total GPU allocation. Prefer TRT10's V2
    # accessor, which also accounts for weight-streaming scratch if enabled.
    memory_property = ("device_memory_size_v2" if hasattr(engine, "device_memory_size_v2")
                       else "device_memory_size")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory = {
        "context_device_memory_requirement_bytes": int(getattr(engine, memory_property)),
        "context_requirement_source": f"ICudaEngine.{memory_property}",
        "snapshot_after_context_and_io_binding": {
            "free_bytes": free_bytes, "total_bytes": total_bytes,
            "device_used_bytes": total_bytes - free_bytes,
        },
        "scope": "One device-wide cudaMemGetInfo snapshot before warmup; includes other processes; not peak memory",
        "context_note": "Context requirement excludes weights, I/O buffers and other runtime allocations",
    }

    def execute():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned failure")

    for _ in range(args.warmup):
        execute()
    stream.synchronize()
    gpu_ms, wall_ms = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Synchronized per-inference latency, not CUDA Graph replay or throughput.
    for _ in range(args.repeat):
        started = time.perf_counter()
        begin.record(stream)
        execute()
        end.record(stream)
        end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000)
        gpu_ms.append(begin.elapsed_time(end))
    # The synchronization above completes all writes before the CPU copy.
    outputs = {item["name"]: buffers[item["name"]].cpu().numpy().copy()
               for item in io if item["mode"] == "OUTPUT"}
    validation = validate(outputs, args.reference, args.rtol, args.atol, np)

    def statistics(values):
        return {"samples": values, "mean": float(np.mean(values)), "median": float(np.median(values)),
                "min": min(values), "max": max(values), "p95": float(np.percentile(values, 95))}

    report = {
        "status": "executed", "environment_before": metadata,
        "environment_after": environment(trt, torch), "engine": str(args.engine.resolve()),
        "engine_sha256": sha256(args.engine), "inputs": str(args.inputs.resolve()),
        "inputs_sha256": sha256(args.inputs), "io": io, "memory": memory,
        "engine_aux_streams": engine.num_aux_streams,
        "warmup": args.warmup, "repeat": args.repeat,
        "timing_scope": "One inference per sample; input/output copies, loading, preprocessing and warmup excluded",
        "cuda_event_note": "Stream elapsed time can include GPU idle gaps caused by host submission",
        "cuda_event_ms": statistics(gpu_ms),
        "host_submit_and_sync_ms": statistics(wall_ms), "validation": validation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "outputs.npz", **outputs)
    write_json(args.output / "report.json", report)
    return report, 1 if validation["status"].startswith("failed") else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Parse ONNX and build a static engine")
    build_parser.add_argument("--onnx", type=Path, required=True)
    build_parser.add_argument("--dcn-config", type=Path,
                              help="Experimental exact-node DCNv2 implementation selection; freezes original ranges, requires custom plugin library")
    build_parser.add_argument("--engine", type=Path, help="Required unless --plan-only is used")
    build_parser.add_argument("--precision", choices=("fp32", "fp16", "mixed-int8"), default="fp32")
    build_parser.add_argument("--builder-optimization-level", type=int, choices=range(6), default=3,
                              help="TensorRT build search level; 0 reduces search and can reduce performance")
    build_parser.add_argument("--tactic-dram-mib", type=int,
                              help="Profiling DRAM limit in power-of-two MiB; distinct from workspace and total memory")
    build_parser.add_argument("--timing-cache-input", type=Path,
                              help="Reuse serialized TensorRT tactic timings with strict device/version compatibility")
    build_parser.add_argument("--timing-cache-output", type=Path,
                              help="Write the resulting timing cache to a new file; never overwrite an existing cache")
    build_parser.add_argument("--plan-only", action="store_true", help="Parse and save precision constraints without calibration/build")
    build_parser.add_argument("--calibration-dir", type=Path, help="Real preprocessed inputs and calibration.json manifest")
    build_parser.add_argument("--calibration-cache", type=Path, help="Cache bound to model, precision policy and calibration data")
    build_parser.add_argument("--calibrated-ranges", type=Path,
                              help="Validated backbone-derived ranges bound to this full ONNX; no full-model calibrator")
    build_parser.add_argument("--workspace-mib", type=int, default=1024)
    build_parser.add_argument("--max-aux-streams", type=int, default=0,
                              help="Maximum TensorRT auxiliary streams (default: 0 for lower activation memory)")
    build_parser.add_argument("--plugins-fp32", action="store_true",
                              help="Require all plugin layers to compute/output floating tensors in FP32")
    build_parser.add_argument("--report", type=Path, required=True)
    run_parser = commands.add_parser("run", help="Run with NumPy inputs and optional numerical reference")
    run_parser.add_argument("--engine", type=Path, required=True)
    run_parser.add_argument("--inputs", type=Path, required=True)
    run_parser.add_argument("--reference", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--warmup", type=int, default=3)
    run_parser.add_argument("--repeat", type=int, default=10)
    run_parser.add_argument("--rtol", type=float, default=1e-3)
    run_parser.add_argument("--atol", type=float, default=1e-4)
    for subparser in (build_parser, run_parser):
        subparser.add_argument("--device", type=int, default=0)
        subparser.add_argument("--verbose", action="store_true")
        subparser.add_argument("--plugin-library", type=Path, action="append", default=[],
                               help="Load a custom plugin shared library before TensorRT parsing/loading; repeatable")
    args = parser.parse_args(argv)
    try:
        report, code = build(args) if args.command == "build" else run(args)
        summary = report if args.command == "run" else {
            "status": report["status"], "report": str(args.report),
            "precision": args.precision, "engine": report.get("engine"),
            "scope": report["scope"], "policy_fingerprint": report["policy_fingerprint"],
        }
        print(json.dumps(summary, indent=2, allow_nan=False))
        return code
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
