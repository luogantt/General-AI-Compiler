"""Offline CUDA/MLIR compilation and single-device validation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from .backends.cuda import TARGET, compile_cuda
from .backends.mlir import compile_mlir
from .backends.mlir_cuda import TARGET as MLIR_CUDA_TARGET, compile_mlir_cuda
from .errors import GACError
from .frontends.json_frontend import JsonFrontendError, load_graph
from .mlir_toolchain import (
    DEFAULT_TIMEOUT_SECONDS,
    discover_mlir_opt,
    probe_mlir_opt_source_revision,
    probe_mlir_opt_version,
    verify_mlir,
)
from .package import load_package, rebuild_cuda_program, write_package
from .runtime import CudaSession, cuda_device_info, reference_run
from .mlir_runtime import MlirCudaSession


def _write_text_atomic(path: str | Path, value: str) -> None:
    """Publish a complete text file with one same-directory replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(value)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: str | Path, value: dict) -> None:
    serialized = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_text_atomic(path, serialized)


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
    mlir_compile = sub.add_parser(
        "mlir-compile",
        help="lower a JSON model to standard MLIR and verify it with mlir-opt",
    )
    mlir_compile.add_argument("model")
    mlir_compile.add_argument(
        "-o", "--output", required=True, help="output .mlir module"
    )
    mlir_compile.add_argument("--no-fuse", action="store_true")
    mlir_compile.add_argument(
        "--mlir-opt",
        help="mlir-opt executable, or its Linux path when --wsl is used",
    )
    mlir_compile.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"mlir-opt timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    mlir_compile.add_argument(
        "--emit-only",
        action="store_true",
        help="export generated MLIR without invoking mlir-opt",
    )
    mlir_compile.add_argument(
        "--wsl",
        action="store_true",
        help="run the Linux mlir-opt selected by --mlir-opt through WSL",
    )
    mlir_compile.add_argument(
        "--wsl-distribution",
        help="optional WSL distribution name, for example Ubuntu-24.04",
    )
    mlir_compile.add_argument("--report", help="save the compilation report as JSON")
    run = sub.add_parser("run", help="compile a JSON model in memory and execute it")
    run.add_argument("model")
    run.add_argument(
        "--backend",
        choices=[TARGET, MLIR_CUDA_TARGET, "numpy-cpu"],
        default=TARGET,
    )
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


def _paths_alias(first: str | Path, second: str | Path) -> bool:
    """Return whether two user-supplied paths identify the same destination."""

    first_path = Path(first).expanduser()
    second_path = Path(second).expanduser()
    try:
        if first_path.exists() and second_path.exists():
            return os.path.samefile(first_path, second_path)
    except OSError:
        # Resolving the normalized names below still handles non-existent output
        # paths and existing parent-directory symlinks.
        pass
    first_name = os.path.normcase(str(first_path.resolve(strict=False)))
    second_name = os.path.normcase(str(second_path.resolve(strict=False)))
    return first_name == second_name


def _compile_mlir_command(args) -> dict:
    destination = Path(args.output)
    report_destination = Path(args.report) if args.report else None
    if report_destination is not None and _paths_alias(destination, report_destination):
        raise ValueError("mlir-compile --output and --report must name different files")
    if destination.suffix.lower() != ".mlir":
        raise ValueError("mlir-compile output must end in .mlir")
    if args.wsl_distribution and not args.wsl:
        raise ValueError("--wsl-distribution requires --wsl")
    if args.wsl and not args.emit_only and not args.mlir_opt:
        raise ValueError("--wsl requires --mlir-opt with a Linux executable path")

    program = compile_mlir(load_graph(args.model), fuse=not args.no_fuse)
    description = program.describe()
    compiler_backend = description["compiler_backend"]
    target_intent = description["target_intent"]
    planned_codegen_backend = description["planned_codegen_backend"]
    lowering_stage = description["lowering_stage"]
    executable = bool(description["executable"])
    generated_source = program.source
    generated_sha256 = hashlib.sha256(generated_source.encode("utf-8")).hexdigest()
    mlir_opt_passes: list[str] = []
    verification_tool = None
    if args.emit_only:
        artifact_source = generated_source
        verified = False
        verification = "skipped-by-emit-only"
    else:
        verification_arguments = {
            "timeout": args.timeout,
        }
        if args.wsl:
            selected_mlir_opt = args.mlir_opt
            command_prefix = ["wsl.exe"]
            if args.wsl_distribution:
                command_prefix.extend(("-d", args.wsl_distribution))
            command_prefix.extend(("--", selected_mlir_opt))
            mlir_opt_version = probe_mlir_opt_version(
                command=(selected_mlir_opt,),
                use_wsl=True,
                wsl_distribution=args.wsl_distribution,
                **verification_arguments,
            )
            mlir_opt_source_revision = probe_mlir_opt_source_revision(
                selected_mlir_opt,
                use_wsl=True,
                wsl_distribution=args.wsl_distribution,
                **verification_arguments,
            )
            artifact_source = verify_mlir(
                generated_source,
                command=(selected_mlir_opt,),
                use_wsl=True,
                wsl_distribution=args.wsl_distribution,
                **verification_arguments,
            )
            verification = "mlir-opt-via-wsl"
        else:
            selected_mlir_opt = os.fspath(discover_mlir_opt(args.mlir_opt))
            command_prefix = [selected_mlir_opt]
            mlir_opt_version = probe_mlir_opt_version(
                command=(selected_mlir_opt,),
                **verification_arguments,
            )
            mlir_opt_source_revision = probe_mlir_opt_source_revision(
                selected_mlir_opt,
                **verification_arguments,
            )
            artifact_source = verify_mlir(
                generated_source,
                command=(selected_mlir_opt,),
                **verification_arguments,
            )
            verification = "mlir-opt"
        verified = True
        mlir_opt_passes = ["verify-each", "canonicalize", "cse"]
        verification_tool = {
            "name": "mlir-opt",
            "executable": selected_mlir_opt,
            "command_prefix": command_prefix,
            "arguments": ["--verify-each", "--canonicalize", "--cse"],
            "timeout_seconds": float(args.timeout),
            "version": mlir_opt_version,
            # This is populated only from the build marker adjacent to this
            # exact executable.  It remains null for system/unmanaged tools.
            "source_revision": mlir_opt_source_revision,
        }

    source_sha256 = hashlib.sha256(artifact_source.encode("utf-8")).hexdigest()
    report = {
        "schema_version": 1,
        "artifact": "mlir-module",
        "artifact_kind": description.get("artifact_kind", "textual-mlir-module"),
        "output_file": str(destination.resolve(strict=False)),
        "report_file": (
            str(report_destination.resolve(strict=False))
            if report_destination is not None
            else None
        ),
        "target_intent": target_intent,
        "compute_capability": [8, 6],
        "compiler_backend": compiler_backend,
        "planned_codegen_backend": planned_codegen_backend,
        "lowering_stage": lowering_stage,
        "device_code_compiled": executable,
        "graph": description.get("name", program.graph.name),
        "source_sha256": source_sha256,
        "generated_source_sha256": generated_sha256,
        "verified": verified,
        "verification": verification,
        "verification_tool": verification_tool,
        "dialects": description.get("dialects", []),
        "passes": {
            "gac": description.get("pass_reports", list(program.pass_reports)),
            "mlir_opt": mlir_opt_passes,
        },
    }
    # Metadata construction and external verification both finish before the
    # artifact is atomically published.  A requested report is then published
    # atomically by ``main``; if that second publication fails, the complete
    # artifact remains usable and any previous report remains intact.
    _write_text_atomic(destination, artifact_source)
    return report


def _run(args) -> dict:
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup must be non-negative")
    if (args.trace or args.profile) and args.backend != TARGET:
        raise ValueError("--trace and --profile currently require the cuda-sm86 backend")
    started = time.perf_counter()
    model_path = Path(args.model)
    if model_path.suffix.lower() == ".hxp":
        if args.backend == MLIR_CUDA_TARGET:
            raise ValueError(
                "HXP v1 packages do not carry the MLIR CUDA runtime-call plan; "
                "run mlir-cuda-sm86 from a JSON graph"
            )
        if args.no_fuse:
            raise ValueError("--no-fuse cannot change an already compiled .hxp package")
        program = rebuild_cuda_program(load_package(model_path))
        preparation = "verified-package-load-and-codegen-check"
    else:
        source_graph = load_graph(model_path)
        if args.backend == MLIR_CUDA_TARGET:
            if args.no_fuse:
                raise ValueError(
                    "mlir-cuda-sm86 requires its explicit FFN fusion plan"
                )
            if args.conv_backend != "auto":
                raise ValueError(
                    "--conv-backend does not apply to the MLIR CUDA FFN plan"
                )
            if args.cudnn_tensor_core != "never":
                raise ValueError(
                    "--cudnn-tensor-core does not apply to the MLIR CUDA FFN plan"
                )
            if args.matmul_backend not in {"auto", "cupy"}:
                raise ValueError(
                    "mlir-cuda-sm86 has an explicit cuBLAS MatMul lowering"
                )
            program = compile_mlir_cuda(source_graph)
        else:
            program = compile_cuda(source_graph, fuse=not args.no_fuse)
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
    elif args.backend == MLIR_CUDA_TARGET:
        with MlirCudaSession(program, device_id=args.device) as session:
            report["device"] = session.info
            report["session_setup_ms"] = session.setup_ms
            report["runtime_calls"] = session.kernel_info
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
        report["gpu_stream_ms"] = _stats(gpu)
        report["gpu_stream_scope"] = (
            "H2D + explicit cuBLAS/fused-kernel call plan; excludes output D2H"
        )
        report["lowering"] = program.describe()
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
        elif args.command == "mlir-compile":
            report = _compile_mlir_command(args)
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
