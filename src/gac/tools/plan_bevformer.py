"""Build an offline heterogeneous plan from a verified full-step BEVFormer HXP.

The command performs package verification, graph analysis, automatic
partitioning, global buffer planning, and optional reviewable source emission.
It does not launch CUDA or Vortex code and does not claim that the mixed
CUDA/Vortex physical runtime is implemented.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

from gac.bevformer_compiler import (
    lower_bevformer_dispatch_sources,
    plan_bevformer_graph,
)
from gac.errors import GACError, ValidationError
from gac.package import load_package


_FULL_STEP_KIND = "bevformer-complete-stateful-streaming-step-static"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def build_plan_report(
    package_path: str | Path,
    *,
    cuda_only: bool = False,
    alignment: int = 256,
    verify_cuda_fallback: bool = True,
    lowered_source_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Verify one full-step HXP and return its deterministic offline plan."""

    source = Path(package_path)
    loaded = load_package(source)
    graph_kind = loaded.graph.metadata.get("kind")
    if graph_kind != _FULL_STEP_KIND:
        raise ValidationError(
            "input HXP must contain the complete stateful BEVFormer streaming "
            f"step graph; got graph metadata kind {graph_kind!r}"
        )
    plan = plan_bevformer_graph(
        loaded.graph,
        enable_vortex_ffn=not cuda_only,
        alignment=alignment,
        verify_cuda_fallback=verify_cuda_fallback,
    )
    result = plan.to_dict()
    result["input_package"] = {
        "format": loaded.manifest["format"],
        "schema_version": loaded.manifest["schema_version"],
        "target": loaded.target,
        "sha256": _file_sha256(source),
        "integrity_verified": True,
    }
    artifacts = ()
    if lowered_source_directory is not None:
        artifacts = lower_bevformer_dispatch_sources(
            plan,
            output_directory=lowered_source_directory,
        )
    source_records: list[dict[str, Any]] = []
    for artifact in artifacts:
        record = artifact.to_dict()
        source_path = record.pop("source_path")
        record["source_file"] = Path(source_path).name if source_path else None
        source_records.append(record)
    result["source_emission"] = {
        "requested": lowered_source_directory is not None,
        "artifact_count": len(source_records),
        "artifacts": source_records,
        "executable_binaries_emitted": False,
    }
    result["command_execution"] = {
        "physical_runtime_executed": False,
        "scope": "offline plan and reviewable source emission only",
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="verified full-step CUDA/HXP package")
    parser.add_argument("-o", "--output", required=True, type=Path, help="atomic JSON plan")
    parser.add_argument(
        "--cuda-only",
        action="store_true",
        help="keep every semantic subgraph on the existing CUDA backend",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=256,
        help="global arena alignment in bytes (positive power of two; default: 256)",
    )
    parser.add_argument(
        "--no-verify-cuda-fallback",
        action="store_true",
        help="skip offline CUDA source generation for the complete fallback graph",
    )
    parser.add_argument(
        "--lowered-source-dir",
        type=Path,
        help="optionally emit one reviewable .cu or .mlir source per dispatch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        report = build_plan_report(
            arguments.package,
            cuda_only=arguments.cuda_only,
            alignment=arguments.alignment,
            verify_cuda_fallback=not arguments.no_verify_cuda_fallback,
            lowered_source_directory=arguments.lowered_source_dir,
        )
        _write_json_atomic(arguments.output, report)
    except (GACError, OSError) as exc:
        print(f"gac-bevformer-plan: {exc}", file=sys.stderr)
        return 1
    json.dump(
        {
            "output": str(arguments.output),
            "dispatch_count": report["dispatch_count"],
            "physical_runtime_executed": False,
        },
        sys.stdout,
        ensure_ascii=False,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
