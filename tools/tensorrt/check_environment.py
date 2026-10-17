#!/usr/bin/env python3
"""Check TensorRT plugins and execute a tiny GPU network in the active environment."""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import tensorrt as trt
import torch


def check():
    logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("Could not initialize TensorRT plugins")
    creators = trt.get_plugin_registry().all_creators
    required = {"MultiscaleDeformableAttnPlugin_TRT", "ModulatedDeformConv2d"}
    available = {c.name for c in creators if c.plugin_version == "2" and c.plugin_namespace == ""}
    if required - available:
        raise RuntimeError(f"Missing plugin v2: {sorted(required - available)}")

    builder = trt.Builder(logger)
    network = builder.create_network()
    x = network.add_input("x", trt.float32, (2, 3))
    y = network.add_elementwise(x, x, trt.ElementWiseOperation.SUM).get_output(0)
    y.name = "y"
    network.mark_output(y)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Could not build the smoke-test engine")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        inputs = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
        outputs = torch.empty_like(inputs)
        if not context.set_tensor_address("x", inputs.data_ptr()):
            raise RuntimeError("Failed to bind input")
        if not context.set_tensor_address("y", outputs.data_ptr()):
            raise RuntimeError("Failed to bind output")
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("Failed to execute smoke-test engine")
    stream.synchronize()
    actual = outputs.cpu().tolist()
    expected = [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]
    if actual != expected:
        raise RuntimeError(f"Incorrect smoke-test output: {actual}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    return {
        "scope": "Environment and FP32 add smoke test only; not BEVFormer validation",
        "passed": True, "host": platform.node(), "python": sys.version.split()[0],
        "tensorrt": trt.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": gpu,
        "plugins": [{"name": name, "version": "2", "namespace": ""} for name in sorted(required)],
        "actual": actual, "expected": expected,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists; choose a new output path")
    try:
        report = check()
    except Exception as error:
        report = {"passed": False, "scope": "environment smoke test", "error": str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["passed"] else 1)
