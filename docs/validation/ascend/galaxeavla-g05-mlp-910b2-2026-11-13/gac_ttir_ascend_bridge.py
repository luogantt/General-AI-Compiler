#!/usr/bin/env python3
"""Compile one GAC generic TTIR module with Triton-Ascend 3.2.1.

This is a validation bridge, not a GAC runtime.  It normalizes generic MLIR
syntax to the pretty TTIR syntax expected by Triton's path-based IRSource,
compiles the module for the active NPU target, and saves reviewable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_launcher_type(value: str) -> str:
    """Translate Triton 3.2 IRSource's MLIR spellings to Ascend launcher ABI."""
    pointer = value.startswith("*")
    scalar = value[1:] if pointer else value
    # 3.2's pointer regex leaves the closing '>' and sometimes source locations.
    scalar = scalar.split()[0].rstrip(">")
    scalar = {
        "f16": "fp16",
        "f32": "fp32",
        "f64": "fp64",
    }.get(scalar, scalar)
    return ("*" if pointer else "") + scalar


def normalize_ttir(source: Path, destination: Path) -> None:
    # torch_npu must be loaded before the active Triton driver on this image so
    # libtorch_npu.so and the CANN runtime are visible to the launcher utility.
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    from triton._C.libtriton import buffer_ir, ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.compiler.compiler import make_backend
    from triton.runtime.driver import driver

    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    backend = make_backend(driver.active.get_current_target())
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    module.context = context
    destination.write_text(str(module), encoding="utf-8")
    context.disable_multithreading()


def serializable_metadata(metadata) -> dict:
    result = {}
    for key in metadata._fields:
        value = getattr(metadata, key)
        if key == "target":
            value = {"backend": value.backend, "arch": value.arch,
                     "warp_size": value.warp_size}
        elif isinstance(value, tuple):
            value = list(value)
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ttir", type=Path, help="GAC generic or pretty TTIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--always-compile", action="store_true")
    arguments = parser.parse_args()

    source = arguments.ttir.expanduser().resolve()
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = (arguments.cache_dir or output / "triton-cache").expanduser().resolve()
    os.environ["TRITON_CACHE_DIR"] = str(cache)
    if arguments.always_compile:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    normalized = output / f"{source.stem}.normalized.ttir"
    normalize_ttir(source, normalized)

    from triton.compiler import compile as triton_compile
    from triton.runtime.driver import driver

    kernel = triton_compile(str(normalized))
    parsed_signature = dict(kernel.src.signature)
    launch_signature = {
        index: canonical_launcher_type(value)
        for index, value in parsed_signature.items()
    }
    # This fixes only launcher metadata.  The compiled npubin is independent of
    # the buggy signature strings and has already been produced at this point.
    kernel.src.signature = launch_signature

    kernel_name = kernel.metadata.kernel_name
    npubin = bytes(kernel.asm["npubin"])
    ttadapter = kernel.asm.get("ttadapter")
    npubin_path = output / f"{kernel_name}.npubin"
    npubin_path.write_bytes(npubin)
    if isinstance(ttadapter, str):
        (output / f"{kernel_name}.ttadapter.mlir").write_text(
            ttadapter, encoding="utf-8")

    report = {
        "source": str(source),
        "source_sha256": sha256(source.read_bytes()),
        "normalized_ttir": str(normalized),
        "normalized_ttir_sha256": sha256(normalized.read_bytes()),
        "target": str(driver.active.get_current_target()),
        "kernel_name": kernel_name,
        "irsource_signature": parsed_signature,
        "launcher_signature": launch_signature,
        "npubin": str(npubin_path),
        "npubin_bytes": len(npubin),
        "npubin_sha256": sha256(npubin),
        "asm_keys": sorted(kernel.asm.keys()),
        "metadata": serializable_metadata(kernel.metadata),
    }
    report_path = output / f"{kernel_name}.compile.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

