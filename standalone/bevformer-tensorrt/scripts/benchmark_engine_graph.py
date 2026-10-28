#!/usr/bin/env python3
r"""Probe full-engine CUDA Graph capture, then compare ordinary enqueue vs replay.

Run from an installed bevformer-trt environment (or set PYTHONPATH=src):
  python scripts/benchmark_engine_graph.py --engine model.engine \
    --inputs images.npy --plugin-library libgac_msda_plugin.so \
    --output-dir results/graph-probe

This never rebuilds an engine or falls back to a partial graph. Capture refusal
is reported as not_supported (exit 2); numerical or execution errors fail (1).
Only completed, bitwise-validated runs receive timing summaries (exit 0).
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from bevformer_trt.engine import environment, initialize, plugin_error_hint, sha256, write_json
from bevformer_trt.profile_engine import allocate_io, execute, make_context, summarize, warmup


def validate_args(args):
    if args.repeat < 2 or args.warmup < 1 or args.device < 0:
        raise ValueError("repeat must be at least 2, warmup at least 1, device nonnegative")
    for path in (args.engine, args.inputs, *args.plugin_library):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")


def capture_is_unsupported(errors):
    """Recognize capture refusal without treating unrelated GPU faults as support limits."""
    # A later invalidated-capture error cannot explain an earlier generic
    # enqueue failure (which may be allocation failure or a real GPU fault).
    messages = [item["error"].split("CUDA kernel errors might")[0].lower() for item in errors]
    if any("execute_async_v3 returned failure" in message for message in messages):
        return False
    text = "\n".join(messages)
    fatal = ("out of memory", "illegal memory", "illegal address", "device-side assert",
             "misaligned address", "launch failure", "unspecified launch failure")
    if any(marker in text for marker in fatal):
        return False
    markers = ("stream capture", "streamcapture", "capture is invalidated",
               "capturing stream", "when stream is capturing", "capture error",
               "previous error during capture",
               "operation not permitted when stream is capturing")
    return any(marker in text for marker in markers)


def compare_outputs(reference, actual, np):
    """Require identical shape, dtype and bytes, including signed zero; reject NaN/Inf."""
    if set(reference) != set(actual):
        raise ValueError("Engine output names changed")
    checks = {}
    for name, expected in reference.items():
        observed = actual[name]
        same_shape = observed.shape == expected.shape
        same_dtype = observed.dtype == expected.dtype
        finite = bool(np.isfinite(observed).all() and np.isfinite(expected).all())
        exact = bool(same_shape and same_dtype and
                     np.array_equal(np.ascontiguousarray(observed).view(np.uint8).reshape(-1),
                                    np.ascontiguousarray(expected).view(np.uint8).reshape(-1)))
        checks[name] = {"shape": list(observed.shape), "dtype": str(observed.dtype),
                        "finite": finite, "bitwise_equal": exact,
                        "max_abs": (float(np.max(np.abs(observed.astype(np.float64) -
                                                        expected.astype(np.float64)), initial=0))
                                    if finite and same_shape else None)}
    return {"passed": bool(checks) and all(item["finite"] and item["bitwise_equal"]
                                           for item in checks.values()), "outputs": checks}


def snapshot(io, buffers):
    return {item["name"]: buffers[item["name"]].cpu().numpy().copy()
            for item in io if item["mode"] == "OUTPUT"}


def poison_outputs(io, buffers, torch, stream):
    # Outside the graph and measurement: an empty/stale replay must not pass.
    with torch.cuda.stream(stream):
        for item in io:
            if item["mode"] == "OUTPUT":
                tensor = buffers[item["name"]]
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
                else:
                    tensor.bitwise_not_()
    stream.synchronize()


def measure_once(action, stream, begin, end):
    started = time.perf_counter()
    begin.record(stream)
    action()
    end.record(stream)
    end.synchronize()
    return {"cuda_event_ms": float(begin.elapsed_time(end)),
            "host_submit_and_sync_ms": (time.perf_counter() - started) * 1000}


def capture_full_engine(context, stream, torch):
    graph = torch.cuda.CUDAGraph()
    errors = []
    with torch.cuda.stream(stream):
        began = False
        try:
            graph.capture_begin(capture_error_mode="global")
            began = True
            execute(context, stream)
        except Exception as error:
            errors.append({"phase": "capture_enqueue", "type": type(error).__name__,
                           "error": str(error)})
        finally:
            if began:
                try:
                    graph.capture_end()
                except Exception as error:
                    errors.append({"phase": "capture_end", "type": type(error).__name__,
                                   "error": str(error)})
    return graph, errors


def run(args):
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": "full_engine_cuda_graph_probe",
              "engine": str(args.engine.resolve()), "engine_sha256": sha256(args.engine),
              "inputs": str(args.inputs.resolve()), "inputs_sha256": sha256(args.inputs),
              "script_sha256": sha256(Path(__file__)),
              "plugin_libraries": [{"path": str(path.resolve()), "sha256": sha256(path)}
                                   for path in args.plugin_library],
              "warmup": args.warmup, "repeat_pairs": args.repeat,
              "capture_scope": "One complete execute_async_v3; same engine, context, addresses and input bytes",
              "timing_scope": "One synchronized inference; excludes loading, copies, capture, validation and warmup",
              "cuda_event_note": "Stream elapsed time can include GPU idle gaps caused by host submission",
              "validation_scope": "Bitwise equality on this fixed input; does not measure dataset mAP/NDS",
              "fallback": None, "validation": [], "pairs": []}
    report_path = args.output_dir / "report.json"
    try:
        np, trt, torch, logger = initialize(args)
        report["environment_before"] = environment(trt, torch)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        if engine is None:
            raise RuntimeError("Could not deserialize engine" + plugin_error_hint())
        stream = torch.cuda.Stream()
        io, buffers = allocate_io(engine, np, trt, torch, args.inputs, stream)
        report.update(io=io, engine_aux_streams=engine.num_aux_streams)
        context = make_context(engine, buffers)
        # The first enqueue flushes TensorRT shape/resource updates before capture.
        warmup(context, stream, args.warmup)
        reference = snapshot(io, buffers)
        reference_check = compare_outputs(reference, reference, np)
        report["validation"].append({"phase": "ordinary_reference", **reference_check})
        if not reference_check["passed"]:
            raise AssertionError("Ordinary execution has nonfinite or missing outputs")
        np.savez(args.output_dir / "reference.outputs.npz", **reference)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report["memory_before_capture"] = {"free_bytes": free_bytes, "total_bytes": total_bytes,
                                            "torch_allocated_bytes": torch.cuda.memory_allocated(),
                                            "torch_reserved_bytes": torch.cuda.memory_reserved()}
        graph, capture_errors = capture_full_engine(context, stream, torch)
        report["capture"] = {"errors": capture_errors, "succeeded": not capture_errors}
        if capture_errors:
            if capture_is_unsupported(capture_errors):
                report["status"] = "not_supported"
                report["reason"] = "The complete engine refused CUDA stream capture; no timing or partial fallback"
                return report, 2
            raise RuntimeError("Full-engine capture failed: " + "; ".join(item["error"] for item in capture_errors))

        def ordinary():
            execute(context, stream)

        def replay():
            with torch.cuda.stream(stream):
                graph.replay()

        actions = {"ordinary": ordinary, "graph": replay}

        def validate_current(phase):
            checked = compare_outputs(reference, snapshot(io, buffers), np)
            report["validation"].append({"phase": phase, **checked})
            if not checked["passed"]:
                raise AssertionError(f"Outputs differ from ordinary execution: {phase}")

        for name, action in actions.items():
            poison_outputs(io, buffers, torch, stream)
            action()
            stream.synchronize()
            validate_current(f"initial_{name}")
        for _ in range(args.warmup):
            ordinary()
            replay()
        stream.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for iteration in range(args.repeat):
            order = ["ordinary", "graph"] if iteration % 2 == 0 else ["graph", "ordinary"]
            pair = {"iteration": iteration, "order": order}
            for name in order:
                poison_outputs(io, buffers, torch, stream)
                pair[name] = measure_once(actions[name], stream, begin, end)
                validate_current(f"pair_{iteration}_{name}")
            report["pairs"].append(pair)
        report["timings"] = {
            name: {metric: summarize([pair[name][metric] for pair in report["pairs"]])
                   for metric in ("cuda_event_ms", "host_submit_and_sync_ms")}
            for name in actions}
        report["paired_median_speedup"] = {
            metric: statistics.median([pair["ordinary"][metric] / pair["graph"][metric]
                                       for pair in report["pairs"]])
            for metric in ("cuda_event_ms", "host_submit_and_sync_ms")}
        report["environment_after"] = environment(trt, torch)
        report["status"] = "completed"
        return report, 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "error": str(error)})
        raise
    finally:
        write_json(report_path, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        report, code = run(args)
        print(json.dumps({"status": report["status"], "report": str(args.output_dir / "report.json"),
                          "paired_median_speedup": report.get("paired_median_speedup")}, indent=2))
        return code
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
