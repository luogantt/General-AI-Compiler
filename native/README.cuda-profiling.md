# CUDA PTX dispatch profiling

The native runtime runs kernels emitted by the C++ MLIR compiler. Profiling does
not substitute CPU operators, cuBLAS, or cuDNN implementations.

```sh
gac-cuda-run dispatch.json kernel.ptx data/ output/ \
  --warmup 1 --repeat 3 --profile-dispatches
```

The command executes three distinct phases on the same CUDA stream:

1. Warmup: execute the dispatch sequence `--warmup` times.
2. Timing: execute it `--repeat` times, with only a start/end CUDA event around
   each complete sequence. These samples produce `gpu_dispatch_ms` and
   `gpu_dispatch_samples_ms`.
3. Profiling: allocate two CUDA events per dispatch and execute another
   `--repeat` sequences. Record events immediately before and after each kernel;
   synchronize only the final sequence event, then read all elapsed intervals.

This separation keeps per-dispatch event instrumentation out of the normal
timing samples. Profiling doubles the number of measured sequence executions;
use `--repeat 1` for an initial expensive whole-model investigation.

`runtime-report.json` contains:

| Field | Meaning |
| --- | --- |
| `gpu_dispatch_ms` | Mean uninstrumented sequence interval, excluding uploads, downloads, JIT, and warmup |
| `dispatch_profile.by_op` | Per-operator accumulated time per sequence and dispatch counts, sorted by time |
| `dispatch_profile.dispatches` | Per-dispatch mean/min/max time, invocation count, kernel name, and source node |
| `dispatch_profile.sequence_samples_ms` | Whole sequence intervals during the separate instrumented runs |
| `kernel_resources` | Driver-reported registers per thread, static shared bytes per block, local bytes per thread, and launch dimensions |

The profile is a hotspot locator. CUDA events measure elapsed time and can add
noticeable overhead to short kernels. Host submission gaps may be included in
event intervals. The sum of dispatch intervals need not equal the complete
sequence interval. Compare optimization candidates using the uninstrumented
samples, identical inputs/hardware, and numerical output checks.

Resource attributes describe the JIT-compiled function. A nonzero local-memory
allocation may represent stack, spills, or compiler-generated local arrays; it
does not count executed spill loads/stores. The runtime does not claim to measure
achieved occupancy, bank conflicts, DDR bandwidth, Tensor Core utilization, or
hardware cycles. Those require a device profiler and suitable performance
counters.

`--trace-dispatches` remains a debugging mode that waits after every kernel and
flushes a progress log. It cannot be combined with `--profile-dispatches`.
Its timing must not be used as a throughput result.

All parameter pointer arrays are constructed once at initialization and reused
for every launch. The ordering of dispatches and the static arena lifetime proof
are unchanged.
