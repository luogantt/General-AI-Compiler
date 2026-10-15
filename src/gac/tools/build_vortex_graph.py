"""Compile an imported model entirely for one Vortex device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from gac.targets.vortex.compiler import compile_vortex_graph, write_vortex_graph_artifact
from gac.core.errors import GACError
from gac.frontends.model_frontend import load_model
from gac.models.api import list_model_adapters, load_model_graph
from gac.validation.reference import reference_run


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
    except (GACError, ValueError, OSError) as exc:
        print(f"gac-vortex-graph: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
