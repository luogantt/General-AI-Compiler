"""Compile an imported model entirely for one Vortex device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from gac.targets.vortex.compiler import compile_vortex_graph, write_vortex_graph_artifact
from gac.core.errors import GACError
from gac.frontends.model_frontend import load_model
from gac.models.api import list_model_adapters, load_model_graph
from gac.validation.reference import reference_run


def _compile_native_onnx(args: argparse.Namespace, model: Path) -> int:
    """Compatibility CLI launcher only; all IR construction/passes are C++.

    The standalone gac-compile executable also works without this Python CLI.
    A missing executable is an error, never an implicit Python Graph fallback.
    """
    compiler = args.native_compiler or os.environ.get("GAC_NATIVE_COMPILER") or shutil.which("gac-compile")
    if not compiler:
        raise ValueError("ONNX → Vortex now uses the C++ MLIR compiler. Build native/ and set "
                         "GAC_NATIVE_COMPILER or --native-compiler; see native/README.md. "
                         "--legacy-python-graph explicitly selects the previous implementation.")
    if args.inputs or args.golden or args.mlir_opt:
        raise ValueError("The native path uses its pinned MLIR library and raw Runtime buffers. "
                         "Use native/tools/compile-vortex.sh and gac-vortex-run; "
                         "--inputs/--golden/--mlir-opt belong to the legacy Graph artifact format.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("native artifact output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    command = [compiler, str(model), "--target=vortex", "--emit=llvm",
               "--save-stages", str(output), "-o", str(output / "kernel.mlir"),
               "--manifest", str(output / "plan.json"), "--constants-dir", str(output)]
    if args.no_optimize:
        command.append("--no-optimize")
    if args.constant_fold_profile != "default":
        command.extend(["--constant-fold-profile", args.constant_fold_profile])
    subprocess.run(command, check=True)
    makefile = Path(__file__).resolve().parents[3] / "native" / "tools" / "Vortex.mk"
    if makefile.is_file():
        shutil.copyfile(makefile, output / "Makefile")
    print(json.dumps({"compiler_backend": "native-mlir", "output": str(output),
                      "device_binary_built": False, "python_graph_passes": False}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="canonical Graph JSON, ONNX, trusted torch.export .pt2, or verified HXP graph")
    source.add_argument("--bevformer-weights", help="data-only .gacweights; construct the complete BEVFormer-tiny streaming graph")
    parser.add_argument("--adapter", choices=list_model_adapters(),
                        help="construct --model with a registered model adapter instead of a generic frontend")
    parser.add_argument("--expected-checkpoint-sha256", help="expected source checkpoint hash for --bevformer-weights")
    parser.add_argument("--output", "-o", required=True, help="new artifact directory")
    parser.add_argument("--mlir-opt", help="pinned VOLT LLVM20 mlir-opt; verify and lower device MLIR")
    parser.add_argument("--inputs", help="NPZ with the exact named graph inputs")
    parser.add_argument("--golden", action="store_true", help="compute NumPy golden files offline (never executed by device Runtime)")
    parser.add_argument("--trusted", action="store_true", help="allow loading a trusted torch.export archive")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--native-compiler", help="C++ gac-compile executable for the default ONNX → MLIR path")
    parser.add_argument("--constant-fold-profile", choices=("default", "bevformer-base"),
                        default="default", help="bounded constant geometry evaluation in native MLIR")
    parser.add_argument("--legacy-python-graph", action="store_true",
                        help="explicitly retain the previous ONNX → Python Graph compiler for comparison")
    args = parser.parse_args(argv)
    try:
        if args.golden and not args.inputs:
            raise ValueError("--golden requires --inputs")
        if args.adapter and args.bevformer_weights:
            raise ValueError("--adapter is used with --model; --bevformer-weights already selects bevformer-step")
        adapter = "bevformer-step" if args.bevformer_weights else args.adapter
        if args.expected_checkpoint_sha256 and adapter != "bevformer-step":
            raise ValueError("--expected-checkpoint-sha256 requires the bevformer-step adapter")
        path = Path(args.model or args.bevformer_weights)
        if path.suffix.lower() == ".onnx" and not adapter and not args.legacy_python_graph:
            return _compile_native_onnx(args, path)
        if args.constant_fold_profile != "default":
            raise ValueError("--constant-fold-profile applies only to the native ONNX compiler")
        if adapter:
            options = ({"expected_checkpoint_sha256": args.expected_checkpoint_sha256}
                       if adapter == "bevformer-step" else {})
            graph = load_model_graph(adapter, path, **options)
        elif path.suffix.lower() == ".hxp":
            from gac.artifacts.hxp import load_package
            graph = load_package(path).graph
        else:
            graph = load_model(path, trusted=args.trusted)
        inputs = None
        if args.inputs:
            with np.load(args.inputs, allow_pickle=False) as archive:
                inputs = {name: archive[name] for name in archive.files}
        program = compile_vortex_graph(graph, optimize=not args.no_optimize, mlir_opt=args.mlir_opt)
        golden = reference_run(program.graph, inputs) if args.golden else None
        report = write_vortex_graph_artifact(args.output, program, inputs=inputs, golden=golden)
        print(json.dumps({key: report[key] for key in ("target", "graph", "dispatch_count", "operator_counts", "mlir_verified_and_lowered")}, indent=2))
        return 0
    except (GACError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"gac-vortex-graph: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
