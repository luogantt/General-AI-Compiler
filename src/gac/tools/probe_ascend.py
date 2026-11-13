"""Inspect an Ascend host without compiling or launching device code."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from gac.targets.ascend.probe import probe_ascend_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python interpreter containing the matching torch/torch_npu environment",
    )
    parser.add_argument(
        "--require-torch-npu",
        action="store_true",
        help="exit non-zero unless a torch_npu device is discoverable (no op is run)",
    )
    parser.add_argument(
        "--require-triton-ascend",
        action="store_true",
        help=(
            "exit non-zero unless an Ascend Triton target/product is discoverable "
            "(no kernel is compiled or run)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = probe_ascend_environment(python_executable=arguments.python_executable)
    json.dump(report, fp=sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    print()
    readiness = report["readiness"]
    if (
        arguments.require_triton_ascend
        and not readiness["triton_ascend_discoverable"]
    ):
        return 2
    if arguments.require_torch_npu and not readiness["torch_npu_discoverable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
