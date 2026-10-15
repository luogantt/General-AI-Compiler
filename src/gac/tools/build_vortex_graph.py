"""Compile an imported model entirely for one Vortex device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ..backends.vortex_graph import compile_vortex_graph, write_vortex_graph_artifact
from ..errors import GACError
from ..frontends.json_frontend import load_graph
from ..runtime import reference_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="canonical Graph JSON, ONNX, trusted torch.export .pt2, or verified HXP graph")
    source.add_argument("--bevformer-weights", help="data-only .gacweights; construct the complete BEVFormer-tiny streaming graph")
    parser.add_argument("--expected-checkpoint-sha256", help="expected source checkpoint hash for --bevformer-weights")
    parser.add_argument("--output", "-o", required=True, help="new artifact directory")
    parser.add_argument("--mlir-opt", help="pinned VOLT LLVM20 mlir-opt; verify and lower device MLIR")
    parser.add_argument("--inputs", help="NPZ with the exact named graph inputs")
    parser.add_argument("--golden", action="store_true", help="compute NumPy golden files offline (never executed by device Runtime)")
    parser.add_argument("--trusted", action="store_true", help="allow loading a trusted torch.export archive")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.golden and not args.inputs:
            raise ValueError("--golden requires --inputs")
        if args.expected_checkpoint_sha256 and not args.bevformer_weights:
            raise ValueError("--expected-checkpoint-sha256 requires --bevformer-weights")
        path = Path(args.model or args.bevformer_weights)
        if args.bevformer_weights:
            from ..frontends.bevformer_graph import load_bevformer_step_graph
            graph = load_bevformer_step_graph(path, expected_checkpoint_sha256=args.expected_checkpoint_sha256)
        elif path.suffix.lower() == ".hxp":
            from ..package import load_package
            graph = load_package(path).graph
        elif path.suffix.lower() == ".onnx":
            from ..frontends.onnx_frontend import load_onnx
            graph = load_onnx(path)
        elif path.suffix.lower() == ".pt2":
            from ..frontends.torch_export_frontend import load_exported_program
            graph = load_exported_program(path, trusted=args.trusted)
        else:
            graph = load_graph(path)
        inputs = None
        if args.inputs:
            with np.load(args.inputs, allow_pickle=False) as archive:
                inputs = {name: archive[name] for name in archive.files}
        program = compile_vortex_graph(graph, optimize=not args.no_optimize, mlir_opt=args.mlir_opt)
        golden = reference_run(program.graph, inputs) if args.golden else None
        report = write_vortex_graph_artifact(args.output, program, inputs=inputs, golden=golden)
        print(json.dumps({key: report[key] for key in ("target", "graph", "dispatch_count", "operator_counts", "mlir_verified_and_lowered")}, indent=2))
        return 0
    except (GACError, ValueError, OSError) as exc:
        print(f"gac-vortex-graph: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
