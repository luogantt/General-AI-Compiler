"""Check TensorRT plugins and execute a tiny GPU network."""

import argparse
import json
from pathlib import Path

from .engine import environment, initialize, write_json


def check(args):
    _, trt, torch, logger = initialize(args)
    required = {"MultiscaleDeformableAttnPlugin_TRT", "ModulatedDeformConv2d"}
    available = {
        creator.name for creator in trt.get_plugin_registry().all_creators
        if creator.plugin_version == "2" and creator.plugin_namespace == ""
    }
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
    if engine is None:
        raise RuntimeError("Could not deserialize the smoke-test engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create the smoke-test execution context")
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
    return {
        "scope": "Environment and FP32 add smoke test only; not BEVFormer validation",
        "passed": True,
        "environment": environment(trt, torch),
        "plugins": [{"name": name, "version": "2", "namespace": ""} for name in sorted(required)],
        "actual": actual,
        "expected": expected,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("report already exists; choose a new output path")
    try:
        report = check(args)
    except Exception as error:
        report = {"passed": False, "scope": "environment smoke test", "error": str(error)}
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
