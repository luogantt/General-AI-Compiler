# GAC MSDA CUDA / TensorRT plugin

`GACMSDeformAttn`, version `1`, empty namespace, is an independent TensorRT
10.13 `IPluginV3` implementation. It keeps the five-input MSDA ABI and does not
replace NVIDIA's `MultiscaleDeformableAttnPlugin_TRT` creator.

## Tensor contract

All tensors are contiguous `LINEAR` CUDA tensors. `T` is either FP32 or FP16,
and every floating input and the output must have the same type.

| Position | Tensor | Shape | Type |
| --- | --- | --- | --- |
| 0 | value | `[B,S,H,C]` | T |
| 1 | spatial shapes | `[L,2]` (height, width) | INT32 |
| 2 | level starts | `[L]` | INT32 |
| 3 | sampling locations | `[B,Q,H,L,P,2]` (x, y) | T |
| 4 | attention weights | `[B,Q,H,L,P]` | T |
| output | result | `[B,Q,H,C]` | T |

Only nonempty static shapes are supported. Dynamic dimensions, varying
MIN/OPT/MAX profiles, inconsistent ranks or linked dimensions, unsupported
types/formats, and unsafe tensor sizes are rejected. Channels may have tails;
they need not be divisible by `channels_per_thread`.

The producer must supply positive spatial sizes and nonnegative starts, with
`start[level] + height[level] * width[level] <= S`. BEVFormer normally supplies
contiguous cumulative starts and `sum(height[level] * width[level]) == S`;
the CUDA contract also permits in-bounds gaps and overlapping levels. Shapes
and starts are execution tensors on the GPU, not TensorRT shape inputs. The
plugin checks their tensor descriptors but never dereferences device pointers
on the CPU. `gac_msda_validate_metadata` validates their contents separately
before timing: its host return is the launch status, while the device status
is `0` for valid metadata or `1` for invalid metadata. The caller synchronizes
before reading that status. An invalid level makes the forward output NaN and
does not trigger value/location reads for that level.

Sampling uses `x * width - 0.5`, `y * height - 0.5`, with zero padding beyond
each level. Attention is supplied by the caller; this plugin does not apply
softmax. FP16 preserves the pairwise interpolation tree and sequential point
order, with explicit half fused multiply-add operations matching the measured
NVIDIA TensorRT **10.13.3 / SM86** binary. Spatial dimensions use half conversion
toward negative infinity. Coordinates use `hfma(location, extent, -0.5)`.
Writing `hmul` and `hadd` for separately rounded half operations, each point is:

```text
w0 = hmul(hy,hx); w1 = hmul(hy,lx)
w2 = hmul(ly,hx); w3 = hmul(ly,lx)
pair01 = hfma(w0,v0,hmul(w1,v1))
pair23 = hfma(w2,v2,hmul(w3,v3))
sampled = hadd(pair01,pair23)
acc = hfma(sampled,attention,acc)
```

The second product of each pair is rounded before the fused first product;
switching these operands changes results. Scalar half and half2 implement the
same per-channel arithmetic. Ordinary non-`_rn` CUDA half intrinsics permit
compiler contraction; this implementation uses explicit FMA plus `_rn`
operations to fix the measured rounding boundaries. A saved NVIDIA GPU fixture
distinguished 72 contraction models: only this model matched all 140 output
elements bit for bit. Validate additional shapes and any other TensorRT binary
or GPU target before accepting performance results. Accumulation stays in FP16.

## Serialized options

All four fields have TensorRT type `PluginFieldType.INT32` and length one.
The creator accepts omitted build-time fields using the defaults below;
runtime deserialization requires all fields. Unknown, duplicate, mistyped, or
out-of-range fields fail creation.

| Field | Values | Default |
| --- | --- | --- |
| `channels_per_thread` | `1`, `2`, `4` | `1` |
| `block_threads` | `128`, `256`, `512`, `768` | `256` |
| `specialize` | `0` generic, `1` automatic, `2` require specialization, `3` automatic 32-bit indices | `1` |
| `prefetch` | `0` off, `1` locations/weights, `2` locations/weights and Value neighbors | `0` |

Specialization currently targets `L=4, P=8, C=32`. Automatic mode falls back to
the generic kernel for other dimensions. Required mode rejects mismatches.
Mode `3` uses unsigned 32-bit element indices only for that specialized shape
and when every tensor's byte span is at most `INT32_MAX`; otherwise it follows
mode `1` with 64-bit indices. It also derives adjacent-neighbor offsets from
row/column strides. Host overflow checks and device metadata checks retain
64-bit arithmetic, and every neighbor is still independently masked.

The measured second-round SM86 spatial configuration is CPT=4, block=128,
specialize=3, prefetch=0. Whole fixed-graph latency decreased a further 2.60%
against the first-round binary in a separate controlled comparison. See the
[round-two report](../../docs/validation-msda-round2-3070.md). Prefetch 2 was
slower on the captured workload and remains experimental. The non-prefetch
hot path deliberately keeps coefficient broadcasts and interpolation local;
factoring them through the prefetch helpers caused a measured regression.

Prefetch mode `1` loads the next point's x/y/attention values before
interpolating the current point. Mode `2` additionally prepares the next
point's coordinates, coefficients, and four neighboring Value samples for
each channel pack. Two register-resident sample records alternate within the
same level; the last point never reads beyond the level's point count. Loads
are placed before the current interpolation in the source; achieved overlap
depends on compiler scheduling and must be measured. Point accumulation order
and the FP16 rounding boundaries above stay unchanged. No extra CUDA stream,
shared memory, workspace, or allocation is used.

Mode `2` deliberately disables point-loop unrolling to limit live state.
For four channels using half2, each sample record holds eight packed Value
entries plus coefficients, weight, and an active flag. Two live records
increase register pressure; scalar FP32 or scalar half holds sixteen scalar
Values per record. Actual register allocation depends on the compiler. The
768-thread launch bound may cause spills, so inspect
compiler register/spill reports and benchmark this option independently from
mode `3` before combining them. Modes `0`/`1` and all defaults are preserved;
neither new option is assumed to be faster without target-GPU measurements.

Options are serialized per plugin instance; clones own their field storage.
The implementation needs no workspace or device allocation. Enqueue launches
on TensorRT's stream and returns the CUDA launch status without synchronizing.
Asynchronous execution errors remain observable through the caller's CUDA
stream/error checks. Keep the library loaded until all engines and contexts
using the plugin are destroyed.

## Build

Build on a host with a CUDA toolkit, CMake 3.24+, a compatible C++17 compiler,
and TensorRT **10.13** headers and libraries. Header versions are checked at
compile time. An installed Python TensorRT runtime wheel may omit development
headers; use the matching SDK or the official `release/10.13` include tree.
No TensorRT OSS plugin library rebuild is needed.

```bash
cmake -S native/msda -B build/msda -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT-10.13/include \
  -DTENSORRT_LIBRARY_DIR=/path/to/site-packages/tensorrt_libs \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/msda --parallel
```

Run these commands from `standalone/bevformer-tensorrt`. For wheel layouts that
place libraries under `tensorrt_bindings`, pass that directory instead. If
there is no unversioned linker symlink, the build also finds `libnvinfer.so.10`.
For other layouts, pass the full file with
`-DTENSORRT_LIBRARY=/path/to/libnvinfer.so.10`. Windows builds require the
TensorRT import library (`nvinfer_10.lib`), not its DLL. `TENSORRT_ROOT` is an
optional SDK-root hint. `CUDAToolkit_ROOT` and `CMAKE_CUDA_COMPILER` are standard
CMake overrides for nonstandard CUDA installations.

The default CUDA architecture is `86` (RTX 3070); override
`CMAKE_CUDA_ARCHITECTURES` explicitly for other targets. CUDA compilation uses
`--fmad=false` and does not enable fast math. The explicit `__hfma`/`__hfma2`
operations above remain fused; the flag prevents unintended extra contractions.
The output is
`build/msda/libgac_msda_plugin.so` on Linux (`gac_msda_plugin.dll` on Windows).
The build embeds the selected TensorRT library directory in the build RPATH;
deployment must also make TensorRT and CUDA runtime dependencies discoverable.

## Load and test

Load the shared library before parsing ONNX or deserializing an engine, for
example with `ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)` on Linux. Retrieve the
creator as `("GACMSDeformAttn", "1", "")`. In direct TensorRT construction,
create with `TensorRTPhase.BUILD` and use `network.add_plugin_v3` with these
five execution inputs and an empty shape-input list.

`msda_kernel.h` is the authoritative C ABI for direct `ctypes` tests. The same
shared library exports `gac_msda_forward` with the five device input pointers,
an output device pointer, dimensions `B,S,H,C,L,Q,P` as 64-bit integers, dtype
`0` for FP32 or `1` for FP16, the four tuning values, and a `cudaStream_t`.
Callers own the buffers and stream. The function returns a CUDA status integer;
check it and synchronize outside timed regions when verifying results. Do not
pass NumPy host pointers as CUDA tensor pointers.

Before accepting a candidate, compare against a CPU/PyTorch MSDA reference and
the NVIDIA plugin on identical data, including channel tails, boundaries,
multiple levels/heads/batches, FP32 and FP16, and engine serialization/reload.
Performance claims require measurements on the target GPU; the source and its
tuning defaults alone do not establish a speedup.

## References and license

The plugin wrapper is independently written under the repository's Apache-2.0
license. Its public interface was checked against NVIDIA's
[TensorRT 10.13 runtime header](https://github.com/NVIDIA/TensorRT/blob/release/10.13/include/NvInferRuntime.h)
and [V3 MSDA declaration](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.h).
Those upstream files carry NVIDIA's Apache-2.0 notices; they are referenced,
not vendored here. See the CUDA source for any kernel-specific attribution.
