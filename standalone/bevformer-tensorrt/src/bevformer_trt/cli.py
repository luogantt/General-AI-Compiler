"""One installed entry point; CPU tools do not import the CUDA runtime."""

import argparse
import importlib
import sys

from . import __version__


COMMANDS = {
    "check": ("check_environment", "Validate GPU, required plugins and a tiny engine"),
    "adapt": ("adapt_onnx", "Convert exported DCNv2/MSDA nodes to TensorRT plugin ABIs"),
    "msda-rewrite": ("msda", "Replace adapted NVIDIA MSDA nodes with configurable GAC plugins"),
    "msda-rebind-ranges": ("msda", "Prove MSDA-only changes and explicitly rebind backbone ranges"),
    "build": ("engine", "Compile static ONNX into a TensorRT engine"),
    "run": ("engine", "Execute an engine and save raw outputs and timing"),
    "prepare-calibration": ("prepare_nuscenes_calibration", "Prepare real nuScenes mini camera batches"),
    "backbone": ("backbone_calibration", "Extract the audited backbone calibration graph"),
    "calibration-ranges": ("calibration_ranges", "Convert a bound calibration cache to measured ranges"),
    "profile": ("profile_engine", "Record TensorRT layer timings or an Nsight capture range"),
    "verify-plugins": ("verify_plugins", "Compare six supplied plugin fixtures to numerical references"),
}


def main(argv=None):
    """Dispatch without modifying sys.path or relying on a source checkout."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="bevformer-trt",
        description="Build and run static BEVFormer TensorRT 10 engines.",
        epilog="Use bevformer-trt COMMAND --help for command arguments.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")
    for name, (_, help_text) in COMMANDS.items():
        commands.add_parser(name, help=help_text, add_help=False)
    # Command modules own their arguments, including --help. Do not parse them
    # here: that would consume options with meanings specific to those modules.
    if not arguments or arguments[0] in ("-h", "--help", "--version"):
        parser.parse_args(arguments)
        if not arguments:
            parser.print_help()
        return 0
    command, *remaining = arguments
    if command not in COMMANDS:
        parser.error(f"unknown command: {command}")
    module = importlib.import_module(f".{COMMANDS[command][0]}", __package__)
    if command in ("build", "run"):
        remaining.insert(0, command)
    if command in ("msda-rewrite", "msda-rebind-ranges"):
        remaining.insert(0, command)
    return module.main(remaining) or 0
