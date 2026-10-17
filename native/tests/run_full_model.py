#!/usr/bin/env python3
"""Validate whole-model artifacts without implementing or replacing any operator.

The host mode calls only compiler-generated native machine-code kernels through
ctypes. NumPy supplies storage and compares results; it never evaluates the model.
Vortex mode only reads results produced by the separate C++ device runtime.
Neither host wall time nor SimX wall time is reported as physical GPU latency.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("compiler_backend") != "native-mlir":
        raise ValueError("Expected a native-mlir schema_version 1 manifest")
    tensors = {item["name"]: item for item in plan["tensors"]}
    if len(tensors) != len(plan["tensors"]):
        raise ValueError("Duplicate tensor names")
    known = set()
    producers = {}
    last_use = {}
    arena_bytes = plan["arena_bytes"]
    if not isinstance(arena_bytes, int) or arena_bytes < 0:
        raise ValueError("Invalid arena byte count")
    for item in plan["tensors"]:
        shape, dtype = item["shape"], np.dtype(item["dtype"])
        if any(not isinstance(n, int) or n <= 0 for n in shape):
            raise ValueError(f"Expected positive static dimensions: {item['name']}")
        if math.prod(shape) * dtype.itemsize != item["bytes"] or dtype.hasobject:
            raise ValueError(f"Invalid byte count or dtype: {item['name']}")
        if item["storage"] == "arena":
            if item["offset"] < 0 or item["offset"] + item["bytes"] > arena_bytes:
                raise ValueError(f"Tensor outside arena: {item['name']}")
            if item["offset"] % dtype.alignment:
                raise ValueError(f"Misaligned arena tensor: {item['name']}")
        elif item["storage"] in ("input", "constant"):
            known.add(item["name"])
        else:
            raise ValueError(f"Unknown storage: {item['storage']}")
    for index, dispatch in enumerate(plan["dispatches"]):
        if not all(isinstance(prior, int) and 0 <= prior < index
                   for prior in dispatch.get("depends_on", [])):
            raise ValueError(f"Invalid dependency at dispatch {index}")
        for name in dispatch["inputs"]:
            if name not in known:
                raise ValueError(f"Read before definition: {name}")
            last_use[name] = index
        name = dispatch["output"]
        if name not in tensors or name in known or tensors[name]["storage"] != "arena":
            raise ValueError(f"Invalid dispatch output: {name}")
        known.add(name)
        producers[name] = index
        last_use.setdefault(name, index)
    for name in plan["outputs"]:
        if name not in known:
            raise ValueError(f"Undefined model output: {name}")
        last_use[name] = len(plan["dispatches"])
    if set(plan["inputs"]) != {item["name"] for item in plan["tensors"] if item["storage"] == "input"}:
        raise ValueError("Manifest input declarations disagree")
    # A sequential schedule permits address reuse only after the old value dies.
    active = []
    for name, birth in producers.items():
        active = [(old, end) for old, end in active if end >= birth]
        item = tensors[name]
        for old, end in active:
            previous = tensors[old]
            if max(item["offset"], previous["offset"]) < min(
                    item["offset"] + item["bytes"], previous["offset"] + previous["bytes"]):
                raise ValueError(f"Live arena overlap: {old} and {name} at dispatch {birth}")
        active.append((name, last_use[name]))
    return plan


def read_inputs(path: Path, plan: dict) -> dict:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        if len(plan["inputs"]) != 1:
            raise ValueError("A .npy file requires exactly one model input; use .npz otherwise")
        result = {plan["inputs"][0]: np.ascontiguousarray(loaded)}
    else:
        with loaded:
            result = {name: np.ascontiguousarray(loaded[name]) for name in loaded.files}
    if set(result) != set(plan["inputs"]):
        raise ValueError(f"Input names disagree: {list(result)} vs {plan['inputs']}")
    for item in plan["tensors"]:
        if item["storage"] == "input":
            value = result[item["name"]]
            if value.shape != tuple(item["shape"]) or value.dtype != np.dtype(item["dtype"]):
                raise ValueError(f"Input dtype/shape mismatch: {item['name']}")
    return result


def constant_path(item: dict, manifest: Path, directory: Path) -> Path:
    source = Path(item["file"])
    candidates = [source] if source.is_absolute() else [directory / source,
        directory / source.name, manifest.parent / source]
    for candidate in candidates:
        if candidate.is_file():
            if candidate.stat().st_size != item["bytes"]:
                raise ValueError(f"Constant size mismatch: {candidate}")
            return candidate
    raise FileNotFoundError(f"Missing constant: {source}")


def compare(actual, expected, rtol: float, atol: float) -> dict:
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape)}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    close = np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False)
    return {"passed": bool(finite and close.all()), "shape": list(actual.shape),
            "actual_dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
            "finite": finite, "mismatched_elements": int(np.count_nonzero(~close)),
            "max_abs_error": float(difference.max(initial=0)) if finite else None,
            "mean_abs_error": float(difference.mean()) if finite and difference.size else None}


def save_checkpoint(directory: Path, arena, next_index: int, identity: dict, timings: list) -> None:
    # Alternate slots: an interrupted new save cannot invalidate the old index.
    index_file = directory / "checkpoint.json"
    old = json.loads(index_file.read_text()) if index_file.exists() else {}
    slot = 1 - old.get("slot", 1)
    destination = directory / f"checkpoint-arena-{slot}.npy"
    with destination.open("wb") as stream:
        np.save(stream, arena, allow_pickle=False)
    write_json(index_file, {"slot": slot, "next_dispatch": next_index,
                           "identity": identity, "dispatches": timings})


def execute_host(args, plan, inputs, report):
    if plan["target"] != "native":
        raise ValueError("ctypes execution requires --target=native compiler artifacts")
    arena = np.zeros(max(1, plan["arena_bytes"]), dtype=np.uint8)
    tensors = dict(inputs)
    weights_digest = hashlib.sha256()
    for item in plan["tensors"]:
        if item["storage"] == "constant":
            path = constant_path(item, args.manifest, args.constants_dir)
            weights_digest.update((item["name"] + "\0" + sha256(path)).encode())
            tensors[item["name"]] = np.memmap(path, mode="r", dtype=item["dtype"],
                                             shape=tuple(item["shape"]))
        elif item["storage"] == "arena":
            tensors[item["name"]] = np.ndarray(tuple(item["shape"]), dtype=item["dtype"],
                                                buffer=arena, offset=item["offset"])
    identity = {"manifest_sha256": sha256(args.manifest), "library_sha256": sha256(args.library),
                "inputs_sha256": sha256(args.inputs), "constants_sha256": weights_digest.hexdigest()}
    report["artifact_identity"] = identity
    start_index, timings = 0, []
    if args.resume:
        checkpoint = json.loads((args.output_dir / "checkpoint.json").read_text())
        if checkpoint["identity"] != identity:
            raise ValueError("Checkpoint artifacts/inputs/constants differ from this invocation")
        saved = np.load(args.output_dir / f"checkpoint-arena-{checkpoint['slot']}.npy", allow_pickle=False)
        if saved.shape != arena.shape or saved.dtype != arena.dtype:
            raise ValueError("Checkpoint arena shape/dtype mismatch")
        arena[:] = saved
        start_index, timings = checkpoint["next_dispatch"], checkpoint["dispatches"]
        if not 0 <= start_index <= len(plan["dispatches"]):
            raise ValueError("Checkpoint dispatch index is outside this model")
    module = ctypes.CDLL(str(args.library.resolve()))
    dump_pattern = re.compile(args.dump_intermediates) if args.dump_intermediates else None
    references = np.load(args.intermediate_reference, allow_pickle=False) if args.intermediate_reference else None
    dumped_path = args.output_dir / "intermediates.json"
    dumped = json.loads(dumped_path.read_text()) if args.resume and dumped_path.exists() else []
    run_start = time.perf_counter()
    completed = start_index
    with (args.output_dir / "dispatch-progress.jsonl").open("a" if args.resume else "w", encoding="utf-8") as log:
        for index in range(start_index, len(plan["dispatches"])):
            dispatch = plan["dispatches"][index]
            entry = {"index": index, "kernel": dispatch["kernel"], "op": dispatch["op"],
                     "output": dispatch["output"], "source_node": dispatch.get("source_node"),
                     "source_output": dispatch.get("source_output")}
            log.write(json.dumps({**entry, "event": "begin"}) + "\n")
            log.flush()
            arguments = [tensors[name] for name in dispatch["inputs"]] + [tensors[dispatch["output"]]]
            kernel = getattr(module, dispatch["kernel"])
            kernel.argtypes, kernel.restype = [ctypes.c_void_p] * len(arguments), None
            begin = time.perf_counter()
            kernel(*(ctypes.c_void_p(array.ctypes.data) for array in arguments))
            entry["host_kernel_wall_seconds"] = time.perf_counter() - begin
            actual = arguments[-1]
            labels = [entry[key] for key in ("source_output", "source_node", "output") if entry.get(key)]
            if references is not None:
                key = next((name for name in labels if name in references.files), None)
                if key is not None:
                    entry["validation"] = compare(actual, references[key], args.rtol, args.atol)
                    entry["reference_key"] = key
            if dump_pattern and any(dump_pattern.search(name) for name in labels):
                filename = f"dispatch-{index:05d}.npy"
                np.save(args.output_dir / filename, actual, allow_pickle=False)
                dumped.append({**entry, "file": filename})
                write_json(dumped_path, dumped)
            timings.append(entry)
            completed = index + 1
            log.write(json.dumps({**entry, "event": "end"}, allow_nan=False) + "\n")
            log.flush()
            if index % args.progress_every == 0 or completed == len(plan["dispatches"]):
                print(f"[{completed}/{len(plan['dispatches'])}] {dispatch['op']} "
                      f"{entry['host_kernel_wall_seconds']:.3f}s {entry.get('source_node') or dispatch['output']}", flush=True)
            stopped = bool(args.stop_after and completed >= args.stop_after)
            failed = not entry.get("validation", {}).get("passed", True)
            if (args.checkpoint_every and completed % args.checkpoint_every == 0) or stopped or failed:
                save_checkpoint(args.output_dir, arena, completed, identity, timings)
            if stopped or failed:
                break
    if references is not None:
        references.close()
    report.update(completed_dispatches=completed, dispatches=timings,
                  this_invocation_wall_seconds=time.perf_counter() - run_start,
                  summed_host_kernel_wall_seconds=sum(item["host_kernel_wall_seconds"] for item in timings),
                  resumed_from_dispatch=start_index,
                  intermediates_passed=all(item.get("validation", {}).get("passed", True) for item in timings))
    if completed != len(plan["dispatches"]):
        return None
    # Copy outputs before releasing the shared arena; no operator computation.
    return {name: tensors[name].copy() for name in plan["outputs"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--constants-dir", type=Path)
    parser.add_argument("--inputs", type=Path, help="Exactly typed .npy (one input) or named .npz")
    parser.add_argument("--reference", type=Path, help="Named reference output .npz, e.g. PyTorch raw heads")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--library", type=Path, help="Execute generated native CPU kernels from this shared library")
    mode.add_argument("--vortex-results", type=Path, help="Read output_N.bin from the C++ Vortex runtime")
    mode.add_argument("--prepare-vortex-inputs", action="store_true", help="Write input_<tensor index>.bin only")
    mode.add_argument("--audit-only", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0, help="Absolute completed dispatch count; saves a checkpoint")
    parser.add_argument("--dump-intermediates", help="Regex over source node, source output and tensor name")
    parser.add_argument("--intermediate-reference", type=Path, help="Optional .npz keyed by source output/node or tensor")
    args = parser.parse_args()
    if args.progress_every <= 0 or args.checkpoint_every < 0 or args.stop_after < 0:
        parser.error("Progress interval must be positive; checkpoint/stop counts must be nonnegative")
    if (args.library or args.prepare_vortex_inputs) and not args.inputs:
        parser.error("Host execution or Vortex input preparation requires --inputs")
    args.constants_dir = args.constants_dir or args.manifest.parent / "constants"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_plan(args.manifest)
    sizes = {kind: sum(item["bytes"] for item in plan["tensors"] if item["storage"] == kind)
             for kind in ("input", "constant", "arena")}
    report = {"manifest": str(args.manifest.resolve()), "target": plan["target"],
              "physical_device_latency_ms": None, "python_role": "artifact I/O, ctypes invocation and numerical comparison only",
              "dispatch_count": len(plan["dispatches"]), "logical_tensor_bytes": sizes,
              "arena_bytes": plan["arena_bytes"],
              "embedded_constant_bytes": plan.get("embedded_constant_bytes", 0),
              "minimum_storage_bytes_excluding_code_runtime": sizes["input"] + sizes["constant"] + plan["arena_bytes"] + plan.get("embedded_constant_bytes", 0),
              "storage_budget_scope": "Includes embedded constant globals; excludes instructions, stack and runtime metadata",
              "rtol": args.rtol, "atol": args.atol, "full_model_executed": False,
              "output_numerical_validation": "not_run"}
    outputs = None
    if args.audit_only:
        report["mode"] = "manifest_resource_audit_only"
    elif args.prepare_vortex_inputs:
        if plan["target"] != "vortex":
            raise ValueError("Raw Vortex inputs require a Vortex manifest")
        inputs = read_inputs(args.inputs, plan)
        for index, item in enumerate(plan["tensors"]):
            if item["storage"] == "input":
                inputs[item["name"]].tofile(args.output_dir / f"input_{index}.bin")
        report["mode"] = "vortex_input_preparation_only"
    elif args.library:
        report["mode"] = "native_cpu_generated_kernels"
        outputs = execute_host(args, plan, read_inputs(args.inputs, plan), report)
    else:
        if plan["target"] != "vortex":
            raise ValueError("Device result validation requires a Vortex manifest")
        report["mode"] = "vortex_cpp_runtime_result_validation"
        report["execution_environment"] = "See runtime log; result files alone do not identify SimX vs physical hardware"
        tensors = {item["name"]: item for item in plan["tensors"]}
        outputs = {}
        for index, name in enumerate(plan["outputs"]):
            item = tensors[name]
            path = args.vortex_results / f"output_{index}.bin"
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"Device output size mismatch: {name}")
            outputs[name] = np.fromfile(path, dtype=item["dtype"]).reshape(item["shape"])
    if outputs is not None:
        report["full_model_executed"] = True
        np.savez(args.output_dir / "outputs.npz", **outputs)
        if args.reference:
            with np.load(args.reference, allow_pickle=False) as reference:
                if set(reference.files) != set(plan["outputs"]):
                    raise ValueError("Reference output names disagree with manifest")
                report["outputs"] = {name: compare(value, reference[name], args.rtol, args.atol)
                                     for name, value in outputs.items()}
            report["passed"] = all(item["passed"] for item in report["outputs"].values())
            report["output_numerical_validation"] = "passed" if report["passed"] else "failed"
    report["passed"] = report.get("passed", True) and report.get("intermediates_passed", True)
    write_json(args.output_dir / "full-model-report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "dispatches"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
