#!/usr/bin/env python3
r"""Profile a static TensorRT 10 engine without rebuilding it.

Layers mode separates unprofiled CUDA-event latency, profiled whole-engine
latency, and layer sampling. Trace mode supports Nsight Systems, for example:
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    --capture-range-end=stop --output=trace python profile_engine.py \
    --mode trace --engine model.engine --inputs input.npy --output-dir trace-data
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
import time

from run_engine import environment, initialize, load_inputs, require_static, sha256, validate, write_json


def summarize(values):
    """Milliseconds; p95 uses linear interpolation, as NumPy's default does."""
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timing samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower, upper = math.floor(position), math.ceil(position)
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values), "min_ms": ordered[0],
            "max_ms": ordered[-1],
            "p95_ms": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)}


def summarize_layers(iterations):
    """Require stable callback order; never combine repeated layer names."""
    samples = [item for item in iterations if item["phase"] == "layers"]
    if not samples or not samples[0]["callbacks"]:
        raise RuntimeError("TensorRT returned no layer profiling callbacks")
    signature = [(item["ordinal"], item["name"]) for item in samples[0]["callbacks"]]
    for sample in samples:
        current = [(item["ordinal"], item["name"]) for item in sample["callbacks"]]
        if current != signature:
            raise RuntimeError("Layer callback order changed; raw data preserved, aggregation refused")
    rows = []
    for index, (ordinal, name) in enumerate(signature):
        values = [sample["callbacks"][index]["milliseconds"] for sample in samples]
        rows.append({"ordinal": ordinal, "name": name, **summarize(values)})
    return rows


def make_profiler(trt):
    class Collector(trt.IProfiler):
        def __init__(self):
            trt.IProfiler.__init__(self)
            self.iterations = []
            self.current = None
            self.unassigned_callbacks = []

        def begin(self, phase, iteration):
            self.current = {"phase": phase, "iteration": iteration, "callbacks": []}
            self.iterations.append(self.current)

        def report_layer_time(self, layer_name, milliseconds):
            callbacks = (self.current["callbacks"] if self.current is not None
                         else self.unassigned_callbacks)
            callbacks.append({"ordinal": len(callbacks), "name": str(layer_name),
                              "milliseconds": float(milliseconds)})

        def finish(self):
            self.current = None
            if self.unassigned_callbacks:
                raise RuntimeError("Unexpected profiler callback outside an execution")

    return Collector()


def allocate_io(engine, np, trt, torch, input_path, stream):
    """Use the same static LINEAR host/device buffers as run_engine.py."""
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
    arrays = load_inputs(input_path, [item["name"] for item in io if item["mode"] == "INPUT"], np)
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
                tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
                buffers[name] = tensor.pin_memory() if host else tensor.to("cuda")
            else:
                buffers[name] = torch.empty(shape, dtype=torch_dtype, device="cpu" if host else "cuda",
                                             pin_memory=host)
    stream.synchronize()
    return io, buffers


def make_context(engine, buffers):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not allocate an execution context")
    for name, tensor in buffers.items():
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Could not bind tensor {name}")
    missing = context.infer_shapes()
    if missing:
        raise ValueError(f"Shape inference needs values for: {missing}")
    return context


def execute(context, stream):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT execute_async_v3 returned failure")


def collect_profile(context, profiler):
    # Caller synchronizes first; deferred reporting avoids enqueue's implicit sync.
    if not context.report_to_profiler():
        raise RuntimeError("TensorRT report_to_profiler returned failure")
    profiler.finish()


def warmup(context, stream, count, profiler=None):
    for iteration in range(count):
        if profiler is not None:
            profiler.begin("profiled_warmup", iteration)
        execute(context, stream)
        if profiler is not None:
            stream.synchronize()
            collect_profile(context, profiler)
    stream.synchronize()


def measure(context, stream, torch, count, profiler=None):
    gpu_ms, wall_ms, with_report_ms = [], [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for iteration in range(count):
        if profiler is not None:
            profiler.begin("profiled_whole_engine", iteration)
        started = time.perf_counter()
        begin.record(stream)
        execute(context, stream)
        end.record(stream)
        end.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000)
        gpu_ms.append(float(begin.elapsed_time(end)))
        if profiler is not None:
            collect_profile(context, profiler)
            with_report_ms.append((time.perf_counter() - started) * 1000)
    result = {"cuda_event": {**summarize(gpu_ms), "samples_ms": gpu_ms},
              "host_submit_and_sync": {**summarize(wall_ms), "samples_ms": wall_ms}}
    if profiler is not None:
        result["host_submit_sync_and_report"] = {**summarize(with_report_ms), "samples_ms": with_report_ms}
    return result


def finite_outputs(io, buffers, np):
    outputs = {item["name"]: buffers[item["name"]].cpu().numpy().copy()
               for item in io if item["mode"] == "OUTPUT"}
    return validate(outputs, None, 0, 0, np)


def capture_trace(context, engine, stream, torch, trt, warmup_count, repeat):
    # TensorRT permits lowering verbosity, but never raising it above the engine.
    engine_level = engine.profiling_verbosity
    context.nvtx_verbosity = (trt.ProfilingVerbosity.DETAILED
                              if engine_level == trt.ProfilingVerbosity.DETAILED else engine_level)
    warmup(context, stream, warmup_count)
    runtime = torch.cuda.cudart()
    torch.cuda.check_error(runtime.cudaProfilerStart())
    try:
        for iteration in range(repeat):
            torch.cuda.nvtx.range_push(f"frame_{iteration}")
            try:
                execute(context, stream)
                stream.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.check_error(runtime.cudaProfilerStop())
    return {"frames": repeat, "nvtx_verbosity": str(context.nvtx_verbosity),
            "capture_range": "cudaProfilerApi", "profiler_attached": False,
            "scope": "Synchronized inference frames after warmup; loading, copies and validation outside capture"}


def run(args):
    if args.repeat <= 0 or args.warmup < 0 or args.device < 0:
        raise ValueError("repeat must be positive; warmup and device must be nonnegative")
    if not args.engine.is_file() or not args.inputs.is_file():
        raise FileNotFoundError("--engine and --inputs must name existing files")
    # Reject even an empty existing directory, so retries cannot replace artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    np, trt, torch, logger = initialize(args)
    report = {"mode": args.mode, "environment_before": environment(trt, torch),
              "engine": str(args.engine.resolve()), "engine_sha256": sha256(args.engine),
              "inputs": str(args.inputs.resolve()), "inputs_sha256": sha256(args.inputs),
              "warmup": args.warmup, "repeat": args.repeat, "device": args.device}
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize engine on this GPU/TensorRT version")
    stream = torch.cuda.Stream()
    io, buffers = allocate_io(engine, np, trt, torch, args.inputs, stream)
    report.update(io=io, engine_aux_streams=engine.num_aux_streams,
                  engine_profiling_verbosity=str(engine.profiling_verbosity))
    inspection = engine.create_engine_inspector()
    if inspection is None:
        raise RuntimeError("Could not create engine inspector")
    write_json(args.output_dir / "engine.layers.json",
               json.loads(inspection.get_engine_information(trt.LayerInformationFormat.JSON)))
    del inspection
    context = make_context(engine, buffers)
    if args.mode == "trace":
        report["trace"] = capture_trace(context, engine, stream, torch, trt, args.warmup, args.repeat)
        report["validation"] = {"trace": finite_outputs(io, buffers, np)}
    else:
        warmup(context, stream, args.warmup)
        report["baseline"] = measure(context, stream, torch, args.repeat)
        checks = {"baseline": finite_outputs(io, buffers, np)}
        # Release the baseline context before allocating profiling workspace.
        del context
        context = make_context(engine, buffers)
        profiler = make_profiler(trt)
        context.profiler = profiler
        context.enqueue_emits_profile = False
        try:
            warmup(context, stream, args.warmup, profiler)
            report["profiled_whole_engine"] = measure(context, stream, torch, args.repeat, profiler)
            checks["profiled_whole_engine"] = finite_outputs(io, buffers, np)
            for iteration in range(args.repeat):
                profiler.begin("layers", iteration)
                execute(context, stream)
                stream.synchronize()
                collect_profile(context, profiler)
            checks["layers"] = finite_outputs(io, buffers, np)
        finally:
            write_json(args.output_dir / "layers.raw.json", {
                "ordinal_scope": "Zero-based callback order within one execution; names may repeat",
                "iterations": profiler.iterations,
                "unassigned_callbacks": profiler.unassigned_callbacks})
        rows = summarize_layers(profiler.iterations)
        write_json(args.output_dir / "layers.json", rows)
        with (args.output_dir / "layers.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        baseline = report["baseline"]["cuda_event"]["mean_ms"]
        profiled = report["profiled_whole_engine"]["cuda_event"]["mean_ms"]
        report["profiling_overhead"] = {
            "cuda_event_mean_delta_ms": profiled - baseline,
            "cuda_event_mean_ratio": profiled / baseline if baseline else None,
            "note": "Independent sequential passes; includes run-to-run variation and profiling perturbation"}
        report["layer_summary"] = {
            "callbacks_per_iteration": len(rows), "stable_ordinal_name_sequence": True,
            "sum_mean_ms": sum(row["mean_ms"] for row in rows),
            "note": "Sum of profiled layer means is not the unprofiled baseline or an end-to-end latency"}
        report["validation"] = checks
        report["timing_scope"] = "One synchronized inference per sample; copies, loading and warmup excluded"
        report["cuda_event_note"] = "Stream elapsed time can include GPU idle gaps caused by host submission"
        report["profiler"] = {"enqueue_emits_profile": False, "report_after_synchronization": True}
    finite = all(item["all_outputs_finite"] for item in report["validation"].values())
    report["status"] = "completed" if finite else "failed_nonfinite"
    report["validation_scope"] = "Final output of each measured pass only; no numerical reference comparison"
    report["environment_after"] = environment(trt, torch)
    write_json(args.output_dir / "report.json", report)
    return report, 0 if finite else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("layers", "trace"), default="layers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.output_created = False
    try:
        report, code = run(args)
        print(json.dumps({"status": report["status"], "mode": args.mode,
                          "report": str((args.output_dir / "report.json").resolve())}, indent=2))
        return code
    except Exception as error:
        if args.output_created:
            write_json(args.output_dir / "error.json", {"status": "failed", "type": type(error).__name__,
                                                        "error": str(error)})
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
