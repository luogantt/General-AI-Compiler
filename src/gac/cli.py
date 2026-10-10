"""Offline CUDA source generation and single-device validation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .backends.cuda import TARGET, compile_cuda
from .errors import GACError
from .frontends.json_frontend import JsonFrontendError, load_graph
from .package import load_package, rebuild_cuda_program, write_package
from .runtime import CudaSession, cuda_device_info, reference_run


def _write_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAC RTX 3070 / SM 8.6 prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="probe the physical CUDA device and NVRTC")
    doctor.add_argument("--device", type=int, default=0)
    doctor.add_argument("--report")
    compile_cmd = sub.add_parser("compile", help="emit CUDA source or a verified .hxp model package (no GPU required)")
    compile_cmd.add_argument("model")
    compile_cmd.add_argument("-o", "--output", required=True, help="output .cu or .hxp file")
    compile_cmd.add_argument("--target", choices=[TARGET], default=TARGET)
    compile_cmd.add_argument("--no-fuse", action="store_true")
    run = sub.add_parser("run", help="compile a JSON model in memory and execute it")
    run.add_argument("model")
    run.add_argument("--backend", choices=[TARGET, "numpy-cpu"], default=TARGET)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--no-fuse", action="store_true")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--repeat", type=int, default=10)
    run.add_argument(
        "--conv-backend",
        choices=["auto", "cudnn", "generated"],
        default="auto",
        help="CUDA Conv2D implementation policy",
    )
    run.add_argument(
        "--matmul-backend",
        choices=["auto", "cupy", "generated"],
        default="auto",
        help="CUDA MatMul implementation policy",
    )
    run.add_argument(
        "--cudnn-tensor-core",
        choices=["never", "auto", "always"],
        default="never",
        help="cuDNN Tensor Core policy; 'never' preserves the FP32 baseline",
    )
    run.add_argument("--verify", action="store_true", help="compare every measured run to the original unfused graph")
    run.add_argument("--inputs", help="optional NPZ with input names; otherwise use seeded normal inputs")
    run.add_argument("--outputs", help="save final outputs to NPZ")
    run.add_argument("--trace", help="save a separate instrumented invocation as Chrome Trace JSON")
    run.add_argument(
        "--profile",
        help="save per-launch CUDA Event timings and per-op aggregates as JSON",
    )
    run.add_argument("--report", help="save device, correctness and timing report")
    inspect = sub.add_parser("inspect", help="verify and describe an .hxp model package without a GPU")
    inspect.add_argument("package")
    inspect.add_argument("--report")
    return parser


def _stats(samples: list[float]) -> dict:
    return {"median": float(np.median(samples)), "min": min(samples), "max": max(samples)}


def _run(args) -> dict:
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup must be non-negative")
    if (args.trace or args.profile) and args.backend != TARGET:
        raise ValueError("--trace and --profile currently require the cuda-sm86 backend")
    started = time.perf_counter()
    model_path = Path(args.model)
    if model_path.suffix.lower() == ".hxp":
        if args.no_fuse:
            raise ValueError("--no-fuse cannot change an already compiled .hxp package")
        program = rebuild_cuda_program(load_package(model_path))
        preparation = "verified-package-load-and-codegen-check"
    else:
        program = compile_cuda(load_graph(model_path), fuse=not args.no_fuse)
        preparation = "json-source-generation"
    preparation_ms = (time.perf_counter() - started) * 1000
    input_source = args.inputs or "seeded normal"
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in archive.files}
    elif (
        {node.op for node in program.graph.nodes} == {"MSDeformableAttention"}
        and set(program.graph.inputs)
        == {"value", "sampling_locations", "attention_weights"}
    ):
        from .bevformer import make_msda_inputs

        inputs = make_msda_inputs(program.graph, seed=args.seed)
        input_source = "seeded BEVFormer MSDA domain inputs"
    else:
        rng = np.random.default_rng(args.seed)
        inputs = {name: rng.normal(size=spec.shape).astype(spec.dtype) for name, spec in program.graph.inputs.items()}
    expected = reference_run(program.reference_graph, inputs) if args.verify else None
    max_errors: dict[str, float] = {}

    def verify(outputs):
        if expected is not None:
            for name, reference in expected.items():
                np.testing.assert_allclose(outputs[name], reference, rtol=1e-4, atol=1e-5, equal_nan=False, err_msg=f"output {name}")
                error = float(np.max(np.abs(outputs[name] - reference)))
                max_errors[name] = max(max_errors.get(name, 0.0), error)

    report = {"schema_version": 1, "backend": args.backend, "model": program.graph.name,
              "source_sha256": hashlib.sha256(program.source.encode()).hexdigest(),
              "program_preparation": preparation, "program_preparation_ms": preparation_ms,
              "source_generation_ms": preparation_ms if preparation == "json-source-generation" else 0.0,
              "warmup": args.warmup, "repeat": args.repeat,
              "seed": args.seed, "input_source": input_source,
              "verification": "requested" if args.verify else "not requested",
              "rtol": 1e-4, "atol": 1e-5}
    wall, gpu = [], []
    if args.backend == TARGET:
        with CudaSession(
            program,
            device_id=args.device,
            conv_backend=args.conv_backend,
            matmul_backend=args.matmul_backend,
            cudnn_tensor_core=args.cudnn_tensor_core,
        ) as session:
            report["device"] = session.info
            report["session_setup_ms"] = session.setup_ms
            report["kernels"] = session.kernel_info
            first = session.run(inputs)
            report["first_invocation_ms"] = first.wall_ms
            verify(first.outputs)
            for _ in range(args.warmup):
                session.run(inputs)
            for _ in range(args.repeat):
                result = session.run(inputs)
                outputs = result.outputs
                wall.append(result.wall_ms)
                gpu.append(result.gpu_stream_ms)
                verify(outputs)
            if args.trace or args.profile:
                instrumented = session.run(
                    inputs,
                    trace=bool(args.trace),
                    profile=bool(args.profile),
                )
                verify(instrumented.outputs)
                launch_profile = instrumented.profiler.cuda_launch_profile()
                report["cuda_launch_profile"] = launch_profile
                report["instrumented_invocation_wall_ms"] = instrumented.wall_ms
                report["instrumented_gpu_stream_ms"] = instrumented.gpu_stream_ms
                report["instrumented_dispatch_sum_ms"] = instrumented.kernel_ms
                if args.trace:
                    Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
                    instrumented.profiler.export_chrome_trace(args.trace)
                    report["trace"] = args.trace
                if args.profile:
                    _write_json(args.profile, launch_profile)
                    report["profile"] = args.profile
        report["gpu_stream_ms"] = _stats(gpu)
        report["gpu_stream_scope"] = "H2D + allocations/submission gaps + kernels; excludes output D2H"
    else:
        report["device"] = {"name": "NumPy CPU reference", "physical_cuda_execution": False}
        for _ in range(args.warmup):
            reference_run(program.reference_graph, inputs)
        for _ in range(args.repeat):
            started = time.perf_counter()
            outputs = reference_run(program.reference_graph, inputs)
            wall.append((time.perf_counter() - started) * 1000)
            verify(outputs)
    report["wall_ms"] = _stats(wall)
    report["wall_scope"] = "input validation, transfers, allocations, execution and host outputs; excludes setup and verification"
    report["verification"] = "passed" if expected is not None else "not requested"
    report["max_abs_error"] = max_errors
    report["outputs"] = {name: {"shape": list(value.shape), "dtype": value.dtype.name} for name, value in outputs.items()}
    if args.outputs:
        Path(args.outputs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.outputs, **outputs)
    return report


def _inspect_package(path: str | Path) -> dict:
    package_path = Path(path)
    loaded = load_package(package_path)
    program = rebuild_cuda_program(loaded)
    plan = loaded.memory_plan
    op_counts: dict[str, int] = {}
    for node in loaded.graph.nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
    return {
        "schema_version": 1,
        "artifact": "hxp",
        "path": str(package_path),
        "size_bytes": package_path.stat().st_size,
        "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "target": loaded.target,
        "graph": loaded.graph.name,
        "node_count": len(loaded.graph.nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "inputs": {name: spec.to_dict() for name, spec in loaded.graph.inputs.items()},
        "outputs": {
            name: loaded.graph.value_specs[name].to_dict()
            for name in loaded.graph.outputs
        },
        "constant_count": len(loaded.graph.constants),
        "constant_bytes": sum(value.nbytes for value in loaded.graph.constants.values()),
        "codegen_verified": True,
        "source_sha256": hashlib.sha256(program.source.encode()).hexdigest(),
        "launches": program.describe()["launches"],
        "memory": {
            "alignment": plan.alignment,
            "arena_size": plan.arena_size,
            "peak_live_bytes": plan.peak_live_bytes,
            "resident_bytes": plan.resident_bytes,
            "external_input_bytes": plan.external_input_bytes,
            "external_output_bytes": plan.external_output_bytes,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = cuda_device_info(args.device)
            if report["supported_target"] is None:
                raise ValueError(f"device is not SM 8.6: {report}")
            from .runtime import _cupy
            report["nvrtc_version"] = list(_cupy().cuda.nvrtc.getVersion())
        elif args.command == "compile":
            program = compile_cuda(load_graph(args.model), fuse=not args.no_fuse)
            destination = Path(args.output)
            report = program.describe()
            report["source_sha256"] = hashlib.sha256(program.source.encode()).hexdigest()
            if destination.suffix.lower() == ".cu":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(program.source, encoding="utf-8")
                _write_json(destination.with_suffix(".plan.json"), report)
                report["artifact"] = "cuda-source"
                report["source_file"] = str(destination)
            elif destination.suffix.lower() == ".hxp":
                write_package(destination, program)
                report["artifact"] = "hxp"
                report["package_file"] = str(destination)
                report["package_size_bytes"] = destination.stat().st_size
                report["package_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            else:
                raise ValueError("compile output must end in .cu or .hxp")
        elif args.command == "inspect":
            report = _inspect_package(args.package)
        else:
            report = _run(args)
        if getattr(args, "report", None):
            _write_json(args.report, report)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (GACError, JsonFrontendError, OSError, ValueError, AssertionError) as exc:
        print(f"gac: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if args.command == "doctor":
            print(f"gac: CUDA/NVRTC probe failed: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
