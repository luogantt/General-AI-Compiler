"""CPU-only checks for profiling aggregation and capture boundaries."""

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from bevformer_trt import profile_engine as runner


class ProfileTests(unittest.TestCase):
    def test_existing_output_directory_is_rejected_before_cuda_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, inputs = root / "model.engine", root / "input.npy"
            engine.touch()
            inputs.touch()
            with self.assertRaises(FileExistsError):
                runner.run(SimpleNamespace(engine=engine, inputs=inputs, output_dir=root,
                                           warmup=0, repeat=1, device=0))

    def test_duplicate_names_remain_separate_callbacks(self):
        profiler = runner.make_profiler(SimpleNamespace(IProfiler=object))
        for iteration in range(2):
            profiler.begin("layers", iteration)
            profiler.report_layer_time("same", 1 + iteration)
            profiler.report_layer_time("same", 10 + iteration)
            profiler.finish()
        rows = runner.summarize_layers(profiler.iterations)
        self.assertEqual([row["ordinal"] for row in rows], [0, 1])
        self.assertEqual([row["mean_ms"] for row in rows], [1.5, 10.5])
        self.assertEqual(rows[0]["p95_ms"], 1.95)

    def test_changed_callback_order_refuses_aggregation(self):
        profiler = runner.make_profiler(SimpleNamespace(IProfiler=object))
        for iteration, names in enumerate((("a", "b"), ("b", "a"))):
            profiler.begin("layers", iteration)
            for name in names:
                profiler.report_layer_time(name, 1)
            profiler.finish()
        with self.assertRaisesRegex(RuntimeError, "order changed"):
            runner.summarize_layers(profiler.iterations)

    def test_invalid_timing_refuses_summary(self):
        for values in ([], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                runner.summarize(values)

    def test_deferred_reporting_follows_synchronization(self):
        calls = []
        profiler = runner.make_profiler(SimpleNamespace(IProfiler=object))
        context = SimpleNamespace(
            execute_async_v3=lambda stream: calls.append("execute") or True,
            report_to_profiler=lambda: calls.append("report") or True)
        stream = SimpleNamespace(cuda_stream=1, synchronize=lambda: calls.append("sync"))
        runner.warmup(context, stream, 2, profiler)
        self.assertEqual(calls, ["execute", "sync", "report", "execute", "sync", "report", "sync"])

    def trace_fakes(self, fail=False):
        calls = []
        executions = 0

        def execute(stream):
            nonlocal executions
            executions += 1
            calls.append("execute")
            return not (fail and executions == 2)

        context = SimpleNamespace(execute_async_v3=execute)
        stream = SimpleNamespace(cuda_stream=1, synchronize=lambda: calls.append("sync"))
        runtime = SimpleNamespace(cudaProfilerStart=lambda: calls.append("start"),
                                  cudaProfilerStop=lambda: calls.append("stop"))
        cuda = SimpleNamespace(cudart=lambda: runtime, check_error=lambda code: None,
                               nvtx=SimpleNamespace(range_push=lambda name: calls.append(name),
                                                    range_pop=lambda: calls.append("pop")))
        return calls, context, stream, SimpleNamespace(cuda=cuda)

    def test_trace_capture_excludes_warmup_and_bounds_verbosity(self):
        calls, context, stream, torch = self.trace_fakes()
        result = runner.capture_trace(context, SimpleNamespace(profiling_verbosity="NAMES"), stream,
                                      torch, SimpleNamespace(ProfilingVerbosity=SimpleNamespace(DETAILED="DETAIL")),
                                      1, 2)
        self.assertEqual(calls, ["execute", "sync", "start", "frame_0", "execute", "sync", "pop",
                                 "frame_1", "execute", "sync", "pop", "stop"])
        self.assertEqual(context.nvtx_verbosity, "NAMES")
        self.assertFalse(result["profiler_attached"])
        self.assertFalse(hasattr(context, "profiler"))

    def test_trace_stops_capture_on_execution_failure(self):
        calls, context, stream, torch = self.trace_fakes(fail=True)
        with self.assertRaisesRegex(RuntimeError, "execute_async_v3"):
            runner.capture_trace(context, SimpleNamespace(profiling_verbosity="DETAIL"), stream,
                                  torch, SimpleNamespace(ProfilingVerbosity=SimpleNamespace(DETAILED="DETAIL")),
                                  1, 2)
        self.assertEqual(calls[-2:], ["pop", "stop"])


if __name__ == "__main__":
    unittest.main()
