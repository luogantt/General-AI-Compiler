#!/usr/bin/env python3
"""Compile a UniAD FFN bundle to CUDA/HXP and/or the MLIR CUDA call plan.

Run this command in the separate General AI Compiler Python 3.11 / CUDA 12
environment.  Compilation itself is CPU-only; executing the result requires an
SM86 GPU and CuPy/cuBLAS/NVRTC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from gac.backends.cuda import compile_cuda
from gac.backends.mlir_cuda import compile_mlir_cuda
from gac.bevformer_ffn import BEVFormerFFNConfig, build_bevformer_ffn_graph
from gac.package import write_package
from gac.uniad import load_bundle


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_mlir_artifact(output: Path, program: Any) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "abi.mlir": program.source.encode("utf-8"),
        "tensor.mlir": program.tensor_source.encode("utf-8"),
        "kernels.cu": program.kernel_source.encode("utf-8"),
        "calls.json": _canonical_json(
            {"calls": [call.to_dict() for call in program.calls]}
        ).encode("utf-8"),
    }
    records: dict[str, Any] = {}
    for name, payload in files.items():
        path = output / name
        path.write_bytes(payload)
        records[name] = {"sha256": _sha256_bytes(payload), "size": len(payload)}

    manifest = {
        "schema_version": 1,
        "format": "uniad-mlir-cuda-call-plan",
        "boundary": "one UniAD stage2 BEV encoder FFN; not end-to-end UniAD",
        "description": program.describe(),
        "files": records,
    }
    rendered = _canonical_json(manifest).encode("utf-8")
    (output / "manifest.json").write_bytes(rendered)
    return {
        "directory": str(output.resolve()),
        "manifest_sha256": _sha256_bytes(rendered),
        "integrity_sha256": program.integrity_sha256,
        "call_count": len(program.calls),
        "calls": [call.to_dict() for call in program.calls],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend",
        choices=("cuda-hxp", "mlir-cuda-sm86", "both"),
        default="both",
    )
    args = parser.parse_args()

    loaded = load_bundle(args.bundle)
    profile = loaded.contract
    bundle_sha256 = _sha256_file(args.bundle)
    if profile.hidden_dim != profile.embed_dim * 2:
        raise RuntimeError("current GAC FFN lowering requires hidden_dim == 2 * embed_dim")

    config = BEVFormerFFNConfig(
        batch=profile.batch,
        bev_h=profile.bev_h,
        bev_w=profile.bev_w,
        embed_dim=profile.embed_dim,
        post_layer_norm=True,
        layer_norm_epsilon=profile.layer_norm_epsilon,
        ffn_dropout=0.1,
    )
    graph = build_bevformer_ffn_graph(
        loaded.weights,
        config,
        name="uniad_stage2_encoder_layer_%d_ffn" % profile.layer,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "one UniAD stage2 BEV encoder FFN",
        "boundary": "compiler subgraph artifact; not end-to-end UniAD inference",
        "profile": profile.to_dict(),
        "source_bundle": {
            "path": str(args.bundle.resolve()),
            "sha256": bundle_sha256,
            "format": loaded.manifest["format"],
            "schema_version": loaded.manifest["schema_version"],
        },
        "backends": {},
    }

    if args.backend in ("cuda-hxp", "both"):
        started = time.perf_counter()
        cuda_program = compile_cuda(graph)
        compile_ms = (time.perf_counter() - started) * 1000.0
        hxp_path = args.output / (
            "uniad-stage2-encoder-layer-%d-sm86.hxp" % profile.layer
        )
        write_package(
            hxp_path,
            cuda_program,
            metadata={
                "model": "UniAD 2.0",
                "component": "stage2 BEV encoder FFN",
                "layer": profile.layer,
                "bundle_sha256": bundle_sha256,
            },
        )
        report["backends"]["cuda_hxp"] = {
            "path": str(hxp_path.resolve()),
            "sha256": _sha256_file(hxp_path),
            "compile_ms": compile_ms,
            "dispatch_count": len(cuda_program.launches),
            "matmul_lowering": "runtime-select cupy-cublas; generated CUDA fallback is explicit",
        }

    if args.backend in ("mlir-cuda-sm86", "both"):
        started = time.perf_counter()
        mlir_program = compile_mlir_cuda(graph)
        compile_ms = (time.perf_counter() - started) * 1000.0
        artifact = _write_mlir_artifact(args.output / "mlir-cuda-sm86", mlir_program)
        artifact["compile_ms"] = compile_ms
        report["backends"]["mlir_cuda_sm86"] = artifact

    report_path = args.output / "compile-report.json"
    report_path.write_text(_canonical_json(report), encoding="utf-8")
    print(report_path.resolve())


if __name__ == "__main__":
    main()
