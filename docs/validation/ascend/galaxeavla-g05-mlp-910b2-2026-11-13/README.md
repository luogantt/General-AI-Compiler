# GalaxeaVLA G0.5 MLP / Ascend 910B2 validation

This directory records the 2026-11-13 phase-1 validation on one physical
Ascend 910B2.  Commit `2e281239499818b7005c3bf1893945d984d2965b` is the
native exporter/model-adapter baseline.  The formal TTIR hand-off, final
738-test run, and their evidence belong to the later commit that contains this
record; generated manifests retain their exact source hashes.  The image
contained Ubuntu 22.04, CANN 9.0.0, Python 3.11.15, PyTorch 2.7.1,
torch_npu 2.7.1.post4,
Triton-Ascend distribution 3.2.1, and imported `triton.__version__` 3.2.0.  A
stale `triton-3.5.0.dist-info` directory also makes `pip show triton` report
3.5.0; the imported backend code and active target, rather than that orphaned
metadata, are used as the execution identity.  The active
Triton target was `npu/Ascend910B2`.

## Scope

The test covers only the static FP32 Action Expert layer-0 SwiGLU MLP
contract `[1,32,1024] -> [1,32,1024]`, with intermediate width 4096.  It uses
deterministic synthetic weights and inputs.  It does **not** load a Galaxea
checkpoint, test the complete G0.5 model, establish checkpoint-license
authorization for Ascend, or claim an optimized GEMM implementation.

Three complementary paths were tested:

1. `run_g05_mlp_triton.py` expresses the three matrix multiplications and a
   fused SwiGLU pointwise step directly in Triton.  It establishes that this
   exact shape and computation compile and execute with the selected
   Triton-Ascend/CANN image.  It is not, by itself, proof of GAC TTIR
   interoperability.
2. The native GAC compiler imports the official-shape synthetic ONNX and
   exports one manifest plus four standard-TTIR dispatches (three MatMul and
   one fused pointwise).  `gac-ascend-compile-ttir` validates that complete
   hand-off and atomically compiles all four dispatches to `ttadapter` and
   `npubin`.  This is the repository's formal compile-only path.  A separate,
   deliberately narrow validation launcher then binds the manifest tensors
   and synthetic constants and executes the four-dispatch graph; it is not a
   general GAC Ascend runtime.
3. `gac_ttir_ascend_bridge.py` consumes standard TTIR previously emitted by
   GAC, normalizes generic MLIR syntax through Triton's own dialect parser,
   compiles it to `ttadapter` and `npubin`, and launches it on the NPU.  The
   checked-in MatMul and Softmax cases establish pointer-only and
   pointer-plus-scalar launcher ABI handling.  They are individual dispatch
   tests, not the complete MLP chain.

No `npubin`, checkpoint, synthetic weight archive, or 50 MB ONNX file is
committed.  JSON records retain the hashes and sizes needed to identify the
generated artifacts.

## Results

| Validation | Result |
| --- | --- |
| GAC Python suite | 738 run, 31 skipped, suite OK |
| Galaxea-specific tests | 21 passed |
| Fixed six-op ONNX vs GAC host reference | exact match (`max_abs = 0`) |
| Triton vector add | `max_abs = 0` |
| Triton Sigmoid | `max_abs = 1.19e-7` |
| Triton 64x64 MatMul | `max_abs = 8.94e-8` |
| Four-dispatch synthetic MLP | 100/100 passed, `rtol = atol = 1e-4`, global `max_abs = 1.86e-7` |
| Four-dispatch Triton latency | p50 0.899 ms, p95 0.918 ms; 50 warm-ups + 200 synchronized samples |
| torch_npu eager comparison | p50 0.165 ms, p95 0.187 ms under the same sampling procedure |
| Peak NPU allocation / reservation | 54,401,536 / 62,914,560 bytes |
| Native official-shape ONNX -> GAC TTIR contract | passed; 4/4 all-Triton dispatches, 5 fused producers |
| Formal manifest batch compile | passed; 4/4 `ttadapter` and `npubin` artifacts, atomic publication |
| Full GAC-generated 4-dispatch physical chain | passed at `rtol = atol = 1e-5`; final `max_abs = 2.01e-7` |
| GAC generic MatMul TTIR -> `npubin` -> NPU | passed, `max_abs = 9.54e-7` |
| GAC generic Softmax TTIR -> `npubin` -> NPU | passed, `max_abs = 5.96e-8` |
| GAC generic ReduceSum TTIR -> `npubin` -> NPU | passed, exact result |
| GAC generic LayerNorm TTIR -> `npubin` -> NPU | passed, `max_abs = 3.87e-7` |

The Triton MLP uses deliberately conservative 16x16x16 `tl.dot` tiles.  Its
latency is a correctness baseline and must not be presented as the expected
performance of a tuned Triton-Ascend kernel.  Relative-error maxima near zero
are also not used as the acceptance condition; the recorded absolute/relative
allclose test is `rtol = atol = 1e-4`.

The GAC-generated chain launches grids 512, 512, 1024 and 128 in manifest
order.  Gate, up-projection, fused SwiGLU and final output each pass the CPU
comparison at `rtol = atol = 1e-5`; their maximum absolute errors are
`2.03e-6`, `1.91e-6`, `1.04e-6` and `2.01e-7`, respectively.  Peak allocated
and reserved memory are 52,170,752 and 58,720,256 bytes.  The recorded 19.35 s
wall time includes first-launch compiler/launcher initialization and is
explicitly not a steady-state performance measurement.

The native binaries used for this evidence were built against the image's
preinstalled Triton LLVM/MLIR `20.0.0git` in a separate validation copy.  That
copy needed narrow source/API compatibility edits and `-fpermissive` for a
known LLVM header issue; those edits are not represented by the checked-in
native source, whose supported build contract remains LLVM/MLIR 20.1.8.  The
C++ and Python Ascend export contracts both passed in the validation copy.

## Evidence map

| File | Purpose |
| --- | --- |
| `ascend-probe.json` | Host, device, CANN, torch_npu, and Triton discovery snapshot |
| `synthetic-weights.json` | Synthetic-weight contract and archive digest |
| `onnx-export.json`, `onnx-reference.json` | Fixed ONNX structure/digest and reference comparison |
| `unittest-full.txt`, `unittest-galaxeavla.txt` | Test-suite transcripts |
| `triton_smoke.json`, `run_triton_smoke.py` | Small physical compile/launch checks |
| `g05_mlp_triton.json`, `run_g05_mlp_triton.py` | Official-shape four-dispatch accuracy, latency, HBM, and reproducer |
| `native-official/` | Native compiler manifest, four generic TTIR modules, and C++/CLI contract transcripts |
| `formal-compiled/` | Formal batch compile manifest plus normalized TTIR, `ttadapter`, metadata and per-dispatch reports; binaries omitted |
| `official-full-launch.json` | Initial manifest-ordered four-kernel physical launch and per-stage CPU comparisons |
| `run_official_gac_mlp.py`, `official-full-launch-repro.json` | Hash-pinned validation launcher and its independent repeat result |
| `generic_bridge_result.json` | GAC generic MatMul TTIR compile/launch result |
| `generic-softmax-launch.json` | GAC generic Softmax TTIR compile/launch result |
| `generic-reducesum-launch.json`, `generic-layernorm-launch.json` | Additional reduction and mixed pointer/scalar ABI launch results |
| `*.normalized.ttir`, `*.ttadapter.mlir` | Reviewable compiler intermediates; no device binary |
| `gac_ttir_ascend_bridge.py` | Reproducer for the external TTIR hand-off |

The discovery probe intentionally does not infer execution readiness from
installed packages.  Its static readiness fields remain conservative; the
separate smoke and launch records are the evidence of physical execution.
