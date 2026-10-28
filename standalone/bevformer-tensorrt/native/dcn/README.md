# DCNv2 CUDA kernel and optional TensorRT plugin

There are two build targets. Neither replaces a model automatically.

| Target | Built by default | Responsibility | Dependencies |
| --- | --- | --- | --- |
| `gac_dcn_im2col` | Yes | Sampling/column construction through the C ABI; no plugin registration or GEMM | CUDA toolkit, C++17 |
| `gac_dcn_plugin` | No | `GACModulatedDeformConv2d` v1: the same sampling kernel followed by `cublasHgemm` | CUDA, TensorRT 10.13 headers/runtime, cuBLAS headers/runtime |

Run the following commands from `standalone/bevformer-tensorrt`.

## Build only the sampling library

The default configuration does not require TensorRT/cuBLAS development files:

```bash
cmake -S native/dcn -B build/dcn -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_FLAGS="-Xptxas=-v"
cmake --build build/dcn --parallel
```

Linux produces `build/dcn/libgac_dcn_im2col.so`; Windows produces `gac_dcn_im2col.dll`.
`dcn_im2col.h` specifies the exact function argument order, int64 dimension ABI,
FP32/FP16 enum, async return semantics, shape checks, and contiguous layout.

The output is `[C*kH*kW,batch,Hout,Wout]`. For batch 1 this is the same
column matrix passed to GEMM by NVIDIA's DCN plugin. For batch greater than 1
it is **not** a batch-major sequence of independently contiguous GEMM matrices.
Offset channels are `(dy,dx)` pairs; mask values are already modulations.
All buffers use the same type. No convolution weights or biases are consumed.

## Build the optional TensorRT plugin

Use matching CUDA/cuBLAS versions and TensorRT 10.13. Replace these example
paths with the actual SDK/wheel locations; the pip TensorRT runtime may not
contain development headers. Versioned `.so` files need no unversioned symlink.

```bash
TRT_HEADERS=/path/to/TensorRT-10.13/include
TRT_LIBRARY=/path/to/site-packages/tensorrt_libs/libnvinfer.so.10
CUBLAS_HEADERS=/path/to/site-packages/nvidia/cublas/include
CUBLAS_LIBRARY=/path/to/site-packages/nvidia/cublas/lib/libcublas.so.12

cmake -S native/dcn -B build/dcn-plugin -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DGAC_BUILD_DCN_PLUGIN=ON \
  -DTENSORRT_INCLUDE_DIR="$TRT_HEADERS" -DTENSORRT_LIBRARY="$TRT_LIBRARY" \
  -DCUBLAS_INCLUDE_DIR="$CUBLAS_HEADERS" -DCUBLAS_LIBRARY="$CUBLAS_LIBRARY"
cmake --build build/dcn-plugin --parallel
```

This produces both libraries, including `build/dcn-plugin/libgac_dcn_plugin.so`.
The plugin includes the sampling kernel; loading the separate im2col library
is unnecessary. The normal runner loads the custom plugin before parsing or
deserializing an engine through `--plugin-library`.

The plugin deliberately supports a narrower subset than the sampling C ABI:

- Four inputs: image, offset, mask, weight; no bias. All actual execution buffers
  are FP16, contiguous LINEAR, static NCHW, with batch size 1.
- A 3x3 kernel, `group=1`, `deformable_group=1`; positive symmetric stride and
  dilation, nonnegative symmetric padding. Fields use NVIDIA's `[width,height]`
  order. Tensor shapes and integer coordinate bounds are checked.
- `block_threads` is 128/256/512/1024; `specialize` is 1/3/4. Unsupported inputs
  fail explicitly; the plugin does not silently fall back or change precision.
- Weakly typed ONNX can initially describe FP32 tensors. That logical type is
  preserved during parsing; executable format negotiation remains FP16 only.

The output is NCHW FP16. TensorRT supplies a workspace of
`Cin * 3 * 3 * Hout * Wout * sizeof(half)` bytes for columns. The plugin performs
sampling and default-math-mode `cublasHgemm` on the caller's stream, without
allocating columns or synchronizing in `enqueue`.

## Select exact model nodes

Create `dcn-selection.json` with **exact names from the adapted ONNX**. Replace
the placeholder; this is an explicit subset, not a substring or a request to
replace every DCNv2 node:

```json
{
  "node_names": ["REPLACE_WITH_EXACT_SELECTED_DCN_NODE_NAME"],
  "block_threads": 128,
  "specialize": 4
}
```

The example configuration is a candidate, not a universal performance choice.
Select only nodes satisfying the plugin restrictions; all unselected nodes
retain their existing implementations. For a controlled mixed-precision build:

```bash
python -m bevformer_trt build \
  --onnx models/adapted.onnx --precision mixed-int8 \
  --calibrated-ranges models/ranges.json \
  --dcn-config dcn-selection.json \
  --plugin-library build/dcn-plugin/libgac_dcn_plugin.so \
  --workspace-mib 1024 --builder-optimization-level 3 --max-aux-streams 0 \
  --engine build/dcn-ab/model.engine --report build/dcn-ab/build.json

python -m bevformer_trt run \
  --engine build/dcn-ab/model.engine --inputs images.npy \
  --plugin-library build/dcn-plugin/libgac_dcn_plugin.so \
  --output build/dcn-ab/run --warmup 5 --repeat 20
```

If the source model also contains a custom MSDA plugin, supply its
`--plugin-library` as an additional argument for both commands. Output paths
must be new. Use `--precision fp16` without calibration arguments for an FP16
experiment. `--precision fp32` and `--plugins-fp32` are incompatible with this
plugin. Mixed INT8 requires frozen `--calibrated-ranges`; simultaneous
`--calibration-dir` or `--calibration-cache` is rejected.

`--dcn-config` transforms parser input in memory, preserving weights, tensor
edges, convolution attributes and the original ONNX file. The build audit
records selected nodes and source/transformed hashes. Calibration remains
bound to the original source model; no new calibration or accuracy claim is
implied. Omitting `--dcn-config` preserves the existing build path.

## Context resources and validation

Layer clones share a cuBLAS resource through a weak-reference registry keyed by
TensorRT resource-context identity and CUDA context. Different execution
contexts receive independent handles. A host mutex protects `setStream` plus
HGEMM submission when layers share a handle; it introduces no GPU wait. Creation
and release happen with the context resources, rather than once per inference
or once per layer enqueue. The registry does not keep an execution context alive.

Validate serialized plugin outputs against NVIDIA on captured inputs, then test
two simultaneously live execution contexts and streams. For complete engines,
use identical images, frozen calibration and comparable build/timing conditions;
verify all outputs and remeasure latency. Engine rebuilding can change unrelated
tactics. A passing sampling test or a faster single DCNv2 is not sufficient to
promote the custom implementation as the default. This opt-in path requires
separate whole-model acceptance.

## Sampling implementation

The controlled candidates use block sizes 128/256/512/1024. Specialization
0 has runtime kernel loops; 1 unrolls 3x3, otherwise falls back to generic;
2 requires 3x3. These original paths assign one channel/output position to each
thread and retain their implementation unchanged.

Modes 3 and 4 assign two and four input channels per thread for 3x3 kernels,
respectively, with generic fallback for other kernel sizes. Each deformable
group independently divides its channels into thread groups, including tails.
The thread reuses offset, mask, sampled coordinates, four interpolation weights,
and neighbor indices only within that deformable group. Neighboring warp lanes
still cover neighboring spatial positions in NCHW. Each channel uses the same
FP16 formula and writes its original column location; GEMM remains unchanged.
The independent channel kernels do not refactor the original one-channel path.

The generic implementation supports rectangular kernels, strides, dilation,
deformable groups, and batch sizes within the header's checked integer bounds.

Both paths share NVIDIA release/10.13's FP16 arithmetic: convert the integer
sampling base plus the offset through float back to half; compute fractional
parts and coefficients in half; use the same pairwise half interpolation tree;
multiply by the half mask. Native half intrinsics allow compiler contraction,
so the library explicitly uses `--fmad=true` without fast math. This is a
source-level contract, not proof of agreement with a precompiled TensorRT
binary. Require measured comparison before using its performance results.
Out-of-bounds/nonfinite coordinates are masked before floor/address creation.

Validate generic against specialized columns bit for bit on the GPU; compare
sampling plus the same `cublasHgemm` call against the installed NVIDIA plugin
on captured C4/C5 tensors. Include boundary samples, odd spatial sizes, groups,
batch > 1 layout, per-deformable-group channel tails (for example C=6/groups=2),
empty/invalid shapes, and overflow rejection. Check every
CUDA/cuBLAS status and synchronize outside timing. Record register/spill counts:
unrolling can increase register pressure under the 1024-thread launch bound.
Neither reduced node count nor faster im2col alone establishes whole-model gain.

Arithmetic/layout reference: [NVIDIA TensorRT release/10.13 DCN kernel](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/modulatedDeformConvPlugin/modulatedDeformConvPluginKernel.cu).
NVIDIA's [GEMM helper](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/modulatedDeformConvPlugin/modulatedDeformConvCudaHelper.cu)
binds the supplied stream and calls `cublasHgemm` for half (`cublasSgemm` for
float). The sampling-only library has no GEMM; standalone kernel benchmarks
provide that comparison themselves. The optional plugin's independently written
`plugin.cpp` owns its cuBLAS resource and submits `cublasHgemm` after sampling.
The CUDA file retains the NVIDIA and OpenMMLab Apache-2.0 notices for adapted
sampling arithmetic. Host validation and the C ABI are independently written.
