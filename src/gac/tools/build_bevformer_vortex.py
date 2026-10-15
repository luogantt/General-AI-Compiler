"""Build a reviewable BEVFormer FFN -> MLIR -> Vortex artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from gac.targets.vortex.ffn import (
    DIRECT_MLIR_LOWERING,
    LEGACY_CPP_LOWERING,
    compile_vortex,
    compile_vortex_direct,
    write_vortex_artifact,
)
from gac.targets.vortex.buffered_mlir import VortexBufferPlan
from gac.models.bevformer.ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph, ffn_weight_shapes
from gac.core.errors import GACError
from gac.frontends.json_frontend import JsonFrontendError, load_graph
from gac.models.bevformer.frontends.weights import extract_encoder_ffn_weights
from gac.toolchains.mlir import DEFAULT_TIMEOUT_SECONDS
from gac.artifacts.weights import load_weights


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "compile a BEVFormer FFN through external mlir-opt and emit a "
            "Vortex 3.0 regression project"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="canonical GAC JSON graph with FFN weights")
    source.add_argument(
        "--weights",
        help="verified .gacweights archive; extracts one encoder FFN directly",
    )
    source.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="build deterministic [1,2,4] correctness workload",
    )
    source.add_argument(
        "--synthetic-production",
        action="store_true",
        help="build deterministic [1,2500,256]->512->256 workload",
    )
    source.add_argument(
        "--synthetic-width-smoke",
        action="store_true",
        help="build one-row [1,1,256]->512->256 production-width workload",
    )
    parser.add_argument("--output", "-o", required=True, help="new artifact directory")
    parser.add_argument("--mlir-opt", required=True, help="external mlir-opt path")
    parser.add_argument(
        "--device-lowering",
        choices=(DIRECT_MLIR_LOWERING, LEGACY_CPP_LOWERING),
        default=LEGACY_CPP_LOWERING,
        help=(
            "device implementation: direct-mlir emits LLVM dialect MLIR and "
            "requires the pinned VOLT LLVM 20.1.8 tree; cpp-template keeps the "
            "validated compatibility bridge (default: cpp-template)"
        ),
    )
    parser.add_argument(
        "--wsl",
        action="store_true",
        help="invoke the Linux mlir-opt from a Windows host through WSL",
    )
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--matmul-schedule",
        choices=("scalar", "dxa-double-buffer"),
        default="scalar",
        help="direct-mlir MatMul schedule: scalar baseline or tiled LMEM/DXA pipeline",
    )
    parser.add_argument("--tile-m", type=int, default=2, help="buffered MatMul output tile rows")
    parser.add_argument("--tile-n", type=int, default=2, help="buffered MatMul output tile columns")
    parser.add_argument("--tile-k", type=int, default=16, help="buffered MatMul reduction chunk")
    parser.add_argument("--seed", type=int, default=3070)
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="encoder layer selected with --weights (default: 0)",
    )
    parser.add_argument(
        "--weights-rows",
        type=int,
        default=2500,
        help=(
            "flattened query rows selected with --weights; use 1 for a fast "
            "production-width SimX check (default: 2500)"
        ),
    )
    parser.add_argument("--report", help="optional JSON report outside the artifact")
    return parser


def _synthetic_graph(profile: str, seed: int):
    if profile == "production":
        config = BEVFormerFFNConfig()
    elif profile == "production-width-smoke":
        config = BEVFormerFFNConfig(bev_h=1, bev_w=1, embed_dim=256)
    else:
        config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, shape in ffn_weight_shapes(config).items():
        if name == "norm_weight":
            value = rng.uniform(0.8, 1.2, shape)
        elif name == "norm_bias":
            value = rng.uniform(-0.1, 0.1, shape)
        else:
            value = rng.normal(0.0, 0.03 if profile != "smoke" else 0.12, shape)
        weights[name] = np.ascontiguousarray(value, dtype=np.float32)
    graph = build_bevformer_ffn_graph(
        weights,
        config,
        name=(
            "bevformer_tiny_ffn_vortex_synthetic_production"
            if profile == "production"
            else (
                "bevformer_ffn_vortex_synthetic_production_width_smoke"
                if profile == "production-width-smoke"
                else "bevformer_ffn_vortex_synthetic_smoke"
            )
        ),
    )
    graph.metadata["weights_provenance"] = {
        "kind": "deterministic-synthetic",
        "seed": seed,
        "suitable_for_accuracy_evaluation": False,
    }
    return graph


def _paths_alias(first: str | Path, second: str | Path) -> bool:
    left = Path(first).resolve(strict=False)
    right = Path(second).resolve(strict=False)
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _validate_output_paths(args: argparse.Namespace) -> None:
    if not args.report:
        return
    output = Path(args.output).resolve(strict=False)
    report = Path(args.report).resolve(strict=False)
    if report == output or output in report.parents:
        raise ValueError("--report must be outside the Vortex artifact directory")
    for source in (args.model, args.weights):
        if source and _paths_alias(source, args.report):
            raise ValueError("--report must not overwrite the model or weights input")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.wsl_distribution and not args.wsl:
            raise ValueError("--wsl-distribution requires --wsl")
        if args.matmul_schedule != "scalar" and args.device_lowering != DIRECT_MLIR_LOWERING:
            raise ValueError("--matmul-schedule dxa-double-buffer requires --device-lowering direct-mlir")
        if args.matmul_schedule == "scalar" and (args.tile_m, args.tile_n, args.tile_k) != (2, 2, 16):
            raise ValueError("custom --tile-* parameters require --matmul-schedule dxa-double-buffer")
        _validate_output_paths(args)
        if args.model:
            graph = load_graph(args.model)
        elif args.weights:
            if args.layer < 0:
                raise ValueError("--layer must be non-negative")
            if args.weights_rows <= 0:
                raise ValueError("--weights-rows must be positive")
            loaded = load_weights(args.weights)
            config = BEVFormerFFNConfig(
                bev_h=1,
                bev_w=args.weights_rows,
                embed_dim=256,
            )
            selected = extract_encoder_ffn_weights(loaded, args.layer, config)
            graph = build_bevformer_ffn_graph(
                selected,
                config,
                name=(
                    f"bevformer_encoder_layer_{args.layer}_ffn_vortex"
                    if args.weights_rows == 2500
                    else (
                        f"bevformer_encoder_layer_{args.layer}_ffn_vortex_"
                        f"rows_{args.weights_rows}"
                    )
                ),
            )
            graph.metadata["weights_provenance"] = {
                "kind": "verified-gacweights",
                "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
                "encoder_layer": args.layer,
                "compiled_rows": args.weights_rows,
                "suitable_for_accuracy_evaluation": True,
            }
        else:
            profile = (
                "production"
                if args.synthetic_production
                else (
                    "production-width-smoke"
                    if args.synthetic_width_smoke
                    else "smoke"
                )
            )
            graph = _synthetic_graph(profile, args.seed)
        compiler = (
            compile_vortex_direct
            if args.device_lowering == DIRECT_MLIR_LOWERING
            else compile_vortex
        )
        schedule_args = {}
        if args.matmul_schedule == "dxa-double-buffer":
            schedule_args["buffer_plan"] = VortexBufferPlan(args.tile_m, args.tile_n, args.tile_k)
        program = compiler(
            graph,
            mlir_opt=args.mlir_opt,
            use_wsl=args.wsl,
            wsl_distribution=args.wsl_distribution,
            timeout=args.timeout,
            **schedule_args,
        )
        report = write_vortex_artifact(args.output, program)
        report["weights_provenance"] = graph.metadata.get("weights_provenance", "model")
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (GACError, JsonFrontendError, OSError, ValueError) as exc:
        print(f"gac-vortex: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
