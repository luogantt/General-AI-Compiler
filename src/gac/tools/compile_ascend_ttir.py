"""Compile a GAC Ascend standard-TTIR manifest with Triton-Ascend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from gac.core.errors import GACError
from gac.targets.ascend.triton_compile import compile_ascend_ttir_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="GAC Ascend artifact manifest JSON")
    parser.add_argument(
        "--ttir-dir",
        type=Path,
        required=True,
        help="directory containing every standard TTIR source in the manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory to publish atomically (must not already exist)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional external Triton cache directory",
    )
    parser.add_argument(
        "--always-compile",
        action="store_true",
        help="set TRITON_ALWAYS_COMPILE=1 for this invocation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = compile_ascend_ttir_manifest(
            arguments.manifest,
            arguments.ttir_dir,
            arguments.output_dir,
            cache_dir=arguments.cache_dir,
            always_compile=arguments.always_compile,
        )
    except GACError as exc:
        print(f"gac-ascend-compile-ttir: error: {exc}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
