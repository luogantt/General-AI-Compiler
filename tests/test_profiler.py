"""Unit coverage for optional CUDA launch profiling without a physical GPU."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.cli import _parser, _run
from gac.ir import Graph, Node, TensorSpec
from gac.memory import plan_memory
from gac.profiler import Profiler
from gac.runtime import CudaSession, RunResult


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class _FakeStream(_Context):
    def __init__(self) -> None:
        self.clock_ms = 0.0

    def synchronize(self) -> None:
        return None


class _FakeEvent:
    def __init__(self) -> None:
        self.clock_ms = 0.0

    def record(self, stream: _FakeStream) -> None:
        self.clock_ms = stream.clock_ms


class _FakeCuda:
    def __init__(self) -> None:
        self.event_count = 0

    def Event(self) -> _FakeEvent:
        self.event_count += 1
        return _FakeEvent()

    @staticmethod
    def get_elapsed_time(start: _FakeEvent, end: _FakeEvent) -> float:
        return end.clock_ms - start.clock_ms


class _FakeArray:
    def __init__(self, shape, dtype) -> None:
        self.value = np.empty(shape, dtype=dtype)

    @property
    def nbytes(self) -> int:
        return self.value.nbytes

    def set(self, value, *, stream) -> None:
        del stream
        np.copyto(self.value, value)


class _FakeCupy:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    @staticmethod
    def asnumpy(value: _FakeArray, *, stream, blocking: bool):
        del stream, blocking
        return value.value.copy()


def _fake_session() -> CudaSession:
    graph = Graph(
        "profile_identity",
        {"x": TensorSpec((3,), "float32")},
        ("y",),
        [Node("identity", "Identity", ("x",), ("y",))],
    )
    program = compile_cuda(graph)
    session = CudaSession.__new__(CudaSession)
    session.program = program
    session._cp = _FakeCupy()
    session._device = _Context()
    session._stream = _FakeStream()
    session._lock = Lock()
    session._closed = False
    session._failed = False
    session._constants = {}
    session._inputs = {"x": _FakeArray((3,), "float32")}
    session._outputs = {"y": _FakeArray((3,), "float32")}
    session._arena_views = {}
    session._arena_memory = None
    session.memory_plan = plan_memory(program.graph)
    session._node_ops = {"identity": "Identity"}

    def identity_kernel(grid, block, args, *, stream) -> None:
        del grid, block
        np.copyto(args[-1].value, args[0].value)
        stream.clock_ms += 2.5

    session._kernels = {program.launches[0].entry_point: identity_kernel}
    return session


class ProfilerTests(unittest.TestCase):
    def test_cuda_launch_profile_is_json_safe_and_aggregated_by_op(self):
        profiler = Profiler()
        samples = [
            ("add_0", "Add", 1.25),
            ("matmul", "MatMul", 4.0),
            ("add_1", "Add", 0.75),
        ]
        for index, (node_id, op, elapsed) in enumerate(samples):
            profiler.add(
                kind="dispatch",
                name=f"kernel_{index}",
                device="cuda-sm86",
                start_ns=index * 100,
                duration_ns=25,
                source_nodes=(node_id,),
                metadata={
                    "node_id": node_id,
                    "op": op,
                    "implementation": "generated-cuda",
                    "entry_point": f"kernel_{index}",
                    "grid": (1, 2, 3),
                    "block": (32, 1, 1),
                    "output_bytes": 64,
                    "storage": "arena",
                    "arena_offset": index * 64,
                    "gpu_elapsed_ms": elapsed,
                },
            )
        profiler.add(
            kind="copy_h2d",
            name="inputs",
            device="cuda-sm86",
            start_ns=0,
            duration_ns=10,
        )

        profile = profiler.cuda_launch_profile()

        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["clock"], "cuda-event")
        self.assertEqual(profile["launch_count"], 3)
        self.assertEqual(profile["timed_launch_count"], 3)
        self.assertEqual(profile["untimed_launch_count"], 0)
        self.assertAlmostEqual(profile["total_gpu_ms"], 6.0)
        self.assertEqual(profile["by_op"]["Add"]["launch_count"], 2)
        self.assertAlmostEqual(profile["by_op"]["Add"]["total_gpu_ms"], 2.0)
        self.assertAlmostEqual(profile["by_op"]["Add"]["mean_gpu_ms"], 1.0)
        self.assertAlmostEqual(profile["by_op"]["Add"]["share_percent"], 100 / 3)
        self.assertEqual(
            profile["by_implementation"]["generated-cuda"]["launch_count"],
            3,
        )
        self.assertEqual(
            profile["launches"][0]["implementation"], "generated-cuda"
        )
        self.assertEqual(profile["launches"][0]["grid"], [1, 2, 3])
        json.dumps(profile, allow_nan=False)

    def test_default_run_does_not_create_per_launch_events(self):
        session = _fake_session()
        result = session.run({"x": np.arange(3, dtype=np.float32)})

        np.testing.assert_array_equal(result.outputs["y"], np.arange(3, dtype=np.float32))
        self.assertEqual(session._cp.cuda.event_count, 2)
        self.assertIsNone(result.kernel_ms)
        self.assertEqual(result.profiler.events, ())

    def test_profile_run_times_each_launch_and_identifies_its_op(self):
        session = _fake_session()
        result = session.run(
            {"x": np.arange(3, dtype=np.float32)},
            profile=True,
        )

        self.assertEqual(session._cp.cuda.event_count, 4)
        self.assertAlmostEqual(result.kernel_ms, 2.5)
        profile = result.profiler.cuda_launch_profile()
        self.assertEqual(profile["launch_count"], 1)
        self.assertEqual(profile["untimed_launch_count"], 0)
        self.assertEqual(profile["launches"][0]["node_id"], "identity")
        self.assertEqual(profile["launches"][0]["op"], "Identity")
        self.assertAlmostEqual(profile["by_op"]["Identity"]["total_gpu_ms"], 2.5)

    def test_cli_rejects_cuda_profile_for_cpu_backend(self):
        args = _parser().parse_args(
            ["run", "unused.json", "--backend", "numpy-cpu", "--profile", "out.json"]
        )
        with self.assertRaisesRegex(ValueError, "--trace and --profile"):
            _run(args)

    def test_cli_writes_machine_readable_profile(self):
        calls: list[tuple[bool, bool]] = []
        policies: list[dict[str, str]] = []

        class FakeSession:
            def __init__(self, program, *, device_id=0, **execution_policy) -> None:
                self.program = program
                self.execution_policy = execution_policy
                policies.append(execution_policy)
                self.info = {"device_id": device_id, "name": "fake-sm86"}
                self.setup_ms = 1.0
                self.kernel_info = []

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def run(self, inputs, *, trace=False, profile=False):
                del inputs
                calls.append((trace, profile))
                collector = Profiler(enabled=trace or profile)
                if trace or profile:
                    op_by_id = {
                        node.id: node.op for node in self.program.graph.nodes
                    }
                    for index, launch in enumerate(self.program.launches):
                        collector.add(
                            kind="dispatch",
                            name=launch.entry_point,
                            device="cuda-sm86",
                            start_ns=index,
                            duration_ns=1,
                            source_nodes=launch.source_nodes,
                            metadata={
                                "node_id": launch.node_id,
                                "op": op_by_id[launch.node_id],
                                "implementation": "generated-cuda",
                                "entry_point": launch.entry_point,
                                "grid": launch.grid,
                                "block": launch.block,
                                "gpu_elapsed_ms": 0.5,
                            },
                        )
                outputs = {
                    name: np.zeros(
                        self.program.graph.value_specs[name].shape,
                        dtype=self.program.graph.value_specs[name].dtype,
                    )
                    for name in self.program.graph.outputs
                }
                kernel_ms = (
                    0.5 * len(self.program.launches)
                    if trace or profile
                    else None
                )
                return RunResult(outputs, 2.0, 1.5, kernel_ms, collector)

        model = Path(__file__).resolve().parents[1] / "examples" / "heterogeneous_mlp.json"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "profile.json"
            args = _parser().parse_args(
                [
                    "run",
                    str(model),
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                    "--conv-backend",
                    "generated",
                    "--matmul-backend",
                    "generated",
                    "--profile",
                    str(destination),
                ]
            )
            with patch("gac.cli.CudaSession", FakeSession):
                report = _run(args)

            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved, report["cuda_launch_profile"])
            self.assertEqual(saved["timed_launch_count"], len(saved["launches"]))
            self.assertTrue(saved["by_op"])
            self.assertEqual(calls[-1], (False, True))
            self.assertEqual(sum(profile for _, profile in calls), 1)
            self.assertEqual(
                policies,
                [
                    {
                        "conv_backend": "generated",
                        "matmul_backend": "generated",
                        "cudnn_tensor_core": "never",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
