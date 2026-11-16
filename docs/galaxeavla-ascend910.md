# GalaxeaVLA G0.5 on Ascend 910

## Status and boundary

This branch starts the port from `lg/bevformer-triton-vortex-deep`. It now has
two distinct execution paths: the GAC compiler validates an Action Expert MLP
slice through standard TTIR and Triton-Ascend `npubin`, while the
[TorchNPU source overlay](../integrations/galaxeavla/README.md) runs the complete
official G0.5 checkpoint in eager mode on Ascend 910B2. The latter includes
vision/VLM, CoT generation, ten flow-matching steps, discrete action AR and
ActionCodec. Full-model timing and its synthetic-input limitations are recorded
in the [2026-11-16 report](validation/ascend/galaxeavla-g05-full-910b2-2026-11-16/README.md).
For the separate real LIBERO episode used in the user's RTX 3090 benchmark,
the best archived run uses TorchAir GE for Action Expert and TorchAir's
`mode=npugraph_ex` ACLGraph for VLM, both with `frozen_parameter=true`, adds a
FP32 Triton GDN row recurrence and NPU fused AE attention, and omits codec work
whose action field has no consumer. It also compiles the fixed two-camera Vision
geometry, precomputes time conditioning with request-local step lookup, and uses
FP32 matrix partial RoPE and fused RMS normalization in AE, plus functional
MRoPE selection in AE/VLM. BF16 resident
weights retain the official FP32 islands. Complete Vision/VLM + FM10 latency is
**155.57 ms p50 / 156.79 ms p95**, versus **1011.55 ms native-noise eager (6.50×)**.
The Triton reduction tree and fused attention can change floating-point rounding;
12 same-device A/B/C/D/E/F checks have 0.2511% worst relative-L2 error. The earlier
163.91/172.50/200.70 ms native controls, 225.58 ms mixed-backend and 264.46 ms graph-only
combination and original-formula
297.89 ms variant with zero tested action error is separately retained.
The 120.48 ms CUDA target remains unmet; this is not cross-device or closed-loop validation.
[Optimization evidence and reproduction](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/README.md).
GAC still has no general Ascend buffer planner or physical kernel-launch runtime;
the complete model has not been compiled by GAC or Triton-Ascend.

| Capability | Current status |
| --- | --- |
| G0.5 Action Expert MLP graph import | Implemented, static FP32 slice |
| NumPy reference validation | Implemented |
| Ascend host/CANN/torch_npu discovery probe | Implemented, discovery-only and fail-closed |
| Standard TTIR review artifact | Implemented, all-Triton and fail-closed |
| Triton-Ascend `npubin` compilation | Implemented as an atomic compile-only hand-off |
| GAC Ascend physical runtime | Not implemented |
| Complete G0.5 inference | R1 Lite complete dual-head eager baseline; real LIBERO FM10 currently 155.57 ms native-noise with GE/ACLGraph, Triton GDN, fused AE attention/Norm, request time lookup, fixed Vision geometry, matrix partial RoPE and functional MRoPE; 12 same-device replay/state/task/independent-camera checks |
| Complete G0.5 GAC compilation | Not implemented |

The upstream source baseline used for this plan is GalaxeaVLA commit
`89f2322b4ad016e192437adc1a2c253b05bab246`.  Relevant primary sources are the
[G0.5 model configuration](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/configs/model/g05.yaml),
[Qwen3.5 mixture implementation](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/src/g05/models/g05/qwen35/mixture_qwen35.py),
and [upstream dependency declaration](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/pyproject.toml).

The LIBERO comparison retains each original report and measures the same
checkpoint, episode 0/frame 0, two 256×256 cameras, 159 tokens, batch 1,
`[1,32,20]` action output and ten FM steps. CoT and discrete AR are disabled
as in the user's original LIBERO run. Each Ascend row uses five warmups and
twenty measured requests. Native timing includes the original device `torch.randn`,
as in the historical CUDA timing boundary; shared timing clones fixed explicit noise.
Speedups compare the same noise mode. Correctness checks and archived `action.npy`
always use a separate shared-noise request, even when formal timing is native.

| Real LIBERO configuration | Noise timing | Prefill p50 | FM10 p50 | Total p50 | Total p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 910B2 BF16 eager | native | 265.29 ms | 746.76 ms | 1011.55 ms | 1022.70 ms |
| GE AE / ACLGraph VLM + triangular + unused-codec omission | native | 119.03 ms | 100.10 ms | 219.18 ms | 221.87 ms |
| Previous row + exact time-conditioning precompute, regression | native | 117.20 ms | 111.30 ms | 228.46 ms | 229.25 ms |
| GE AE / ACLGraph VLM + Triton GDN + fused AE + unused-codec omission | native | 103.04 ms | 97.67 ms | 200.70 ms | 203.10 ms |
| Previous row + request time lookup + fixed Vision geometry | native | 82.72 ms | 89.74 ms | 172.50 ms | 173.58 ms |
| 172.50 ms configuration + GE single_stream, no improvement | native | 83.41 ms | 89.63 ms | 173.09 ms | 173.58 ms |
| 172.50 ms configuration + AE FP32 matrix partial RoPE | native | 82.64 ms | 81.20 ms | 163.91 ms | 164.62 ms |
| Previous row + fused AE Norm + functional MRoPE | native | 83.36 ms | 72.18 ms | **155.57 ms** | **156.79 ms** |
| 910B2 BF16 eager, earlier baseline | shared | 266.32 ms | 747.07 ms | 1013.87 ms | 1026.25 ms |
| 910B2 BF16 TorchAir AE | shared | 257.39 ms | 108.03 ms | 365.39 ms | 369.58 ms |
| 910B2 BF16 `npugraph_ex` frozen AE + VLM, original formula | shared | 155.04 ms | 142.82 ms | 297.89 ms | 300.62 ms |
| 910B2 BF16 TorchAir frozen AE + VLM, compatible decompositions | shared | 1490.62 ms | 100.54 ms | 1590.90 ms | 1597.47 ms |
| 910B2 BF16 frozen graph AE/VLM + triangular + unused-codec omission | shared | 118.12 ms | 146.38 ms | 264.46 ms | 265.40 ms |
| 910B2 BF16 frozen GE AE / ACLGraph VLM + triangular + unused-codec omission | shared | 117.61 ms | 108.01 ms | 225.58 ms | 226.40 ms |
| User's RTX 3090 BF16 Inductor AE | original device randn | 75.42 ms | 45.03 ms | 120.48 ms | 125.33 ms |

The fastest native run's first request, including capture/compilation, is 153.65 seconds
and is excluded from steady-state statistics. Its 12 checks include A/B/A/B,
changed-state C and changed-task D, plus independent exterior-camera E and
wrist-camera F changes, each checked twice. Worst action error is 0.2511%
relative-L2 / 0.004586 maximum absolute error (E). Action and gripper signs match
100%, padding is zero, and C/D/E/F encoded tokens/masks match their respective
eager references. All image/state/task processing is recomputed:
39 calls actually remove the unused action field, without caching model inputs
or outputs. The original-formula 297.89 ms run retains zero tested action error
and a 153.82-second cold request. The earlier TorchAir AE result
and its 0.1337% worst relative-L2 replay error also remain archived. Its faster FM
stage cannot be combined with another run's faster prefill to invent an
unmeasured total. Module-profile timings come from a separate synchronized
diagnostic request and likewise do not replace the table's full-model samples.

Both mixed backends come from the same TorchAir package, avoiding registration
conflicts with the independent `npugraph_ex` backend package. The VLM uses
`AclConcreteGraph`; AE still uses GE. The complete 155.57 ms run is measured,
not assembled from different runs' stage minima. Its fastest individual sample
is 155.13 ms, which is not the p50. The earlier 172.50 ms and latest post-timing traces use an explicit
single-active-request Level1 schedule; the 163.91 ms report has no trace.
The older 225.58 ms trace's default-schedule warning is retained separately.
Neither trace replaces formal timing.

The native 228.46 ms time-conditioning experiment is a regression relative to
its native 219.18 ms control. It retains the original sequential FP32 timesteps
and cached 49 AdaLN projections, but exact lookup performs a device-to-CPU read
at each FM step. The 200.70 ms control does not enable that precomputation.
The 172.50 ms control and current run instead use request-local lookup inside the audited
original FM loop, with no per-step D2H; all 39 requests in the current report consumed ten steps.
Its Vision graph caches geometry only, reads the learned position embedding
inside the graph, validates geometry each request, and preserves two independent
256-token SDPA calls in each of 24 blocks. Together these two additions reduce
full-model p50 by 14.05% versus 200.70 ms; their individual contributions have
not been isolated. Standalone Triton inverse timings,
opaque-op capture tests and private-namespace guard fixes are linked in the
evidence; their gains are not added to the full-model timing.

Adding AE matrix partial RoPE to the 172.50 ms control produces the separately
measured 163.91 ms result. It retains FP32 arithmetic, a rotating width of 64 in
each 256-wide head, and current cos/sin inputs. GE `single_stream=true` on the
172.50 ms control instead measures 173.09 ms, so it is not selected.
The [RoPE microbenchmark](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/ae-rotary-npu.json)
passes its six original/sliced/matrix cases with zero tested error; the
[Norm GE microbenchmark](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/ae-norm-ge.json)
passes six shape/dtype cases with maximum absolute error 9.5367e-7 and relative-L2
4.6216e-8. Neither is a full-model speed measurement. The
[172.50 ms Level1 summary](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/profile-172ms-level1-summary.json)
reports 40 AI_CPU ScatterElements totaling 5.115 ms of operator durations, a
diagnostic lead rather than removable critical-path latency. Adding fused AE
Norm and functional MRoPE to the matrix RoPE configuration is now independently
measured at 155.57 ms, a 5.08% total-latency reduction from 163.91 ms. Norm keeps
FP32 math and original output casts; functional MRoPE replaces in-place selection
with two `where` operations consuming current frequencies, caching only Boolean
index masks. Its [real-configuration microcheck](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/functional-mrope-ge-acl.json)
has 24 exact-output checks across eager, GE and ACLGraph, separate from the
full-model regression checks. Related local tests, visualization-builder tests
and overlay checks pass. Reproduction flags and source identities are in
the [complete report](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/README.md).

The [final Level1 summary](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/profile-155ms-level1-summary.json)
and [exact CSV type comparison](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/kernel-type-comparison-172-vs-155.json)
record ScatterElements 40→0, StridedSliceD 2120→180 and Square 960→0 relative to
the 172.50 ms control, with 490 RmsNorm and 480 RotaryPositionEmbedding calls in
the final trace. Four AI_CPU tasks remain. Summed kernel durations are not wall
time, and individual changes are not added to claim the combination's speedup.

Peak BF16 throughput does not directly determine batch-one, 159-token latency.
The archived trace contains substantial slicing, transposes, casts, vector work
and AI_CPU operators, so memory traffic, dispatch and implementation choices
also matter. Compilation can reduce launches and intermediate transfers; it
does not move the complete model's weights into on-chip SRAM. We therefore use
matched complete requests, not unverified peak-throughput ratios, for comparison.

Earlier whole-VLM TorchAir GE attempts stopped before measurement on unimplemented
`aten.softplus.default` and then `aten.eye.default` converters. Backend-local
decompositions have separate CPU checks (48,892 Softplus values and 244 eye
cases). The subsequent complete GE run passed same-device A/B/A/B checks with
0.1956% worst relative-L2 and 0.003065 maximum absolute error, but regressed to
1590.90 ms, so it is archived as an experiment rather than selected as best.
Its cold request was 464.78 seconds, excluded from steady-state measurements.
The optional GDN triangular solve changes accumulation order: 84 CPU cases show finite but
not universally identical results. One NPU helper-only eager case measured
9.72 → 4.92 ms and its graph counterpart 4.29 → 3.14 ms, both with exact tested
output/state. Those microbenchmarks exclude projections, convolution, Vision
and the rest of the model. The 264.46 ms complete run separately validates the
combined triangular/unused-codec path; individual contributions cannot be
derived from the microbenchmarks. A warmed GE/manual NPUGraph
capture probe also failed on a small synthetic graph. These failures and partial
experiments are linked in the optimization evidence; future successful complete
runs must be appended with their own timing and replay checks.

The GE-compatible run also archived a real LIBERO
[workload.json](validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/libero-bf16-ge-frozen-compatible/workload.json):
**1,064,344,764,416 counted FLOPs per request** (1.064344764416 TFLOP), measured as partial
Linear/Conv/AdaLN arithmetic in one restored-eager request after formal timing.
This is a lower bound for the original eager path's selected arithmetic, not
the current optimized execution path. Time-conditioning precomputation and
unused-codec omission require new counting; subtracting a few components does
not yield a measured complete FLOP count. It retains the same checkpoint, input and explicit-noise identity, and the
diagnostic output exactly matches its eager reference. Attention/GDN functional
operations and other work remain uncounted; this is neither total model FLOPs,
compiled hardware instructions, utilization nor a peak compute requirement.
The original path includes 1,210,843,136 counted codec FLOPs, so a later variant
that skips codec work must be counted again. The R1 Lite count is not substituted.

## Upstream license

G0.5 code and weights use the
[G0.5 Community License](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/LICENSE-G0.5),
not this repository's Apache-2.0 license.  It restricts use to non-commercial
purposes and narrowly defined internal evaluation.  Its internal-PoC language
also requires “Authorized Hardware” and an “Authorized Image”, but the license
does not define those terms or state that Ascend 910 qualifies.

The technical measurements here do not establish a license grant or determine
whether a particular hardware/image combination is authorized. Users must review
the upstream terms for their use. Checkpoint data is not distributed in this
repository. The overlay carries an upstream license link, NOTICE and per-file
modification statements; this repository's Apache-2.0 license does not relicense it.

## Phase-1 model contract

The first slice is Action Expert decoder layer 0's bias-free SwiGLU MLP.  It is
bounded immediately after the layer's post-attention adaptive RMSNorm, so the
first phase does not pretend to implement AdaLN, attention, or cache semantics.

```text
hidden_states [1, 32, 1024]
  ├─ gate_proj [4096, 1024] ─ sigmoid ─ multiply-with-gate ─┐
  └─ up_proj   [4096, 1024] ────────────────────────────────×
                                                            └─ down_proj [1024, 4096]
                                                               output [1, 32, 1024]
```

The exact computation is:

```text
gate = x @ gate_proj.T
up = x @ up_proj.T
output = (gate * sigmoid(gate) * up) @ down_proj.T
```

The selected policy-state keys are:

```text
model.action_expert.layers.0.mlp.gate_proj.weight
model.action_expert.layers.0.mlp.up_proj.weight
model.action_expert.layers.0.mlp.down_proj.weight
```

PyTorch stores each linear weight as `[out_features, in_features]`; the adapter
copies and transposes it into GAC's `[input_channel, output_channel]` layout.
The canonical graph contains three `MatMul`, one `Sigmoid`, and two `Mul`
nodes.  SiLU is represented exactly as `x * sigmoid(x)`, so this slice needs no
new semantic operation.

Only structurally validated, data-only `.gacweights` input is accepted.  The
adapter does not load the upstream 11.4 GB pickle checkpoint.  Extract the
three tensors once in an isolated, trusted Python 3.10 environment after
pinning the Hugging Face revision and recording the source SHA-256; then write
the data-only interchange file.  Missing keys, wrong shapes, wrong dtypes, or
a source-digest attestation mismatch must fail rather than initialize
replacement weights.

After that trusted extraction boundary, export the fixed ONNX slice in the GAC
environment:

```bash
gac-galaxeavla-export-mlp-onnx \
  build/galaxeavla/action-expert-layer0.gacweights \
  --expected-checkpoint-sha256 <64-hex-source-checkpoint-sha256> \
  --output build/galaxeavla/action-expert-layer0.onnx
```

The required expected checkpoint SHA-256 is compared with the digest
attestation written by the trusted extractor.  This detects a mismatch in that
attestation, but it is **not** cryptographic proof that the three tensors came
from the named upstream checkpoint: the archive writer can supply both the
tensors and the attestation.  Likewise, archive-member hashes establish only
the internal integrity of that archive.  Trust in the source-to-tensor mapping
therefore remains at the controlled extractor boundary; record the extractor
version, its code digest, and its complete extraction manifest alongside the
artifact.

After those checks, the command transposes the three PyTorch-layout weights,
runs the ONNX checker, and publishes the model atomically.  Its JSON output
records both the source-weights and ONNX SHA-256, the attestation kind, static
shapes, initializer layouts, the exact six-operation sequence, and the G0.5
license URL/non-commercial usage warning.  These embedded notices do not grant
commercial use or establish that Ascend is Authorized Hardware.

With the native compiler built, the current Ascend boundary stops explicitly
at reviewable standard TTIR:

```bash
gac-compile build/galaxeavla/action-expert-layer0.onnx \
  --target=ascend \
  --regular-lowering=linalg \
  --emit=device \
  --triton-dir=build/galaxeavla/ttir \
  --constants-dir=build/galaxeavla/constants \
  --manifest=build/galaxeavla/ascend-plan.json \
  --save-stages=build/galaxeavla/stages \
  -o build/galaxeavla/ascend-device.mlir
```

This command does not create an NPU binary.  `--emit=llvm`, Vortex ABI fields,
native fallback dispatches, and backend launch geometry are rejected for the
Ascend target.  On the version-matched NPU image, compile the checked hand-off
without overwriting an existing output directory:

```bash
gac-ascend-compile-ttir build/galaxeavla/ascend-plan.json \
  --ttir-dir build/galaxeavla/ttir \
  --output-dir build/galaxeavla/ascend-compiled \
  --cache-dir /tmp/gac-triton-cache \
  --always-compile
```

The command validates the complete manifest and snapshots every source before
loading vendor packages.  It accepts only `target=ascend`,
`execution=artifact-only`, `artifact_format=standard-ttir`, an all-Triton
dispatch set, and an active supported NPU target.  Each dispatch gets
normalized TTIR, `ttadapter`, `npubin`, metadata and SHA-256 records in a
staging tree; the output directory is published atomically only after every
dispatch succeeds.  This is still compilation, not execution.  The launcher
signature recorded in the report does not allocate buffers, infer launch
geometry, or submit kernels.

## Environment separation

GalaxeaVLA declares Python `>=3.10.16,<3.11`, PyTorch 2.7.1 with a CUDA 12.8
index, and CUDA-oriented packages such as `flash-attn-4`,
`flash-linear-attention`, bitsandbytes, and Liger.  GAC declares Python
`>=3.11`.  Keep two environments:

1. A trusted Galaxea export/oracle environment on Python 3.10 with an exact
   PyTorch/torch_npu/CANN combination from the Ascend compatibility matrix.
2. A GAC Python 3.11 environment that consumes only structurally validated
   data artifacts produced at that controlled extraction boundary.

Do not install the upstream CUDA lock set unchanged on the NPU host.  CUDA
FlashAttention binaries cannot execute on Ascend.  The complete port must also
remove hard-coded CUDA autocast/device calls before it can serve as an NPU
oracle.

## Probe the 910 host first

Run the probe from the GAC environment and point it at the separate model
interpreter:

```bash
gac-ascend-probe \
  --python /opt/galaxeavla-py310/bin/python \
  --require-torch-npu
```

Use `--require-triton-ascend` only after confirming the exact product.  The
Triton-Ascend project currently advertises Atlas A2/A3 (and newer listed
products); an NPU reported only as `Ascend 910` is deliberately marked
unconfirmed.  Capture the full JSON report with every validation run.  The
probe checks `npu-smi`, CANN/ATC discovery, torch_npu device count, device name,
and the active Triton target in an isolated child process.  Its positive fields
are deliberately named `*_discoverable`: neither `--require-torch-npu` nor
`--require-triton-ascend` proves that an NPU tensor operation, a Triton compile,
or a Triton kernel launch succeeds.  Treat these flags as discovery gates only;
real readiness still requires a version-matched on-device smoke test and the
physical-NPU acceptance evidence below.  The probe never falls back to CPU and
never changes the host.

Primary compatibility references:

- [Ascend PyTorch compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)
- [Triton-Ascend README and supported products](https://github.com/triton-lang/triton-ascend/blob/main/README.md)
- [Triton-Ascend released image/version matrix](https://github.com/triton-lang/triton-ascend/blob/main/docker/OVERVIEW.md)
- [Triton-Ascend architecture](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)

## Physical 910B2 validation

The phase-1 evidence was collected on 2026-11-13 with one physical 64 GB
Ascend 910B2.  The image pairing exactly follows the published Triton-Ascend
3.2.1 image: Ubuntu 22.04, CANN 9.0.0, Python 3.11, PyTorch 2.7.1,
torch_npu 2.7.1.post4, and Triton-Ascend 3.2.1.  The imported implementation
reports `triton.__version__ == 3.2.0` and the active target reports
`npu/Ascend910B2`; both values are retained because distribution metadata and
the imported Triton implementation are not the same version field.

The checked-in [validation record](validation/ascend/galaxeavla-g05-mlp-910b2-2026-11-13/README.md)
contains the reports, compiler intermediates, hashes and reproducers.  The
main results are:

- all 738 GAC Python tests completed successfully on the AArch64 image (31
  skipped), including
  all 21 Galaxea-specific tests;
- the fixed six-operation ONNX result exactly matched the GAC host reference;
- vector add, Sigmoid and `tl.dot` MatMul each compiled to `ttadapter` and
  `npubin` and executed on the 910B2;
- the synthetic official-shape MLP ran as three Triton MatMul dispatches plus
  a fused SwiGLU dispatch: 100/100 deterministic FP32 cases passed at
  `rtol=atol=1e-4`, with global maximum absolute error `1.86e-7`;
- after 50 warm-ups, 200 synchronized Triton samples had p50 0.899 ms and p95
  0.918 ms, with 54,401,536 bytes peak allocated.  The conservative
  16x16x16 `tl.dot` kernels are a correctness baseline, not a tuned
  performance claim;
- independent generic TTIR emitted by GAC for MatMul, Softmax, ReduceSum and
  LayerNorm was normalized by the hand-off, compiled to `npubin`, launched,
  and matched its PyTorch reference;
- the official-shape synthetic ONNX was also imported by the native GAC
  compiler, fused to four standard-TTIR dispatches, atomically compiled to
  four `npubin` files, and executed in manifest order on the 910B2.  Every
  intermediate passed its CPU reference at `rtol=atol=1e-5`; the final output
  maximum absolute error was `2.01e-7`.  This closes the phase-1 MLP chain,
  but the narrow validation launcher is not a general GAC NPU runtime.

All of those **2026-11-13 phase-1 runs** use deterministic synthetic weights;
they did not download or load a G0.5 checkpoint. The later full-model run is
documented separately above. A separate
original `Ascend 910` (non-910B) host passed a torch_npu eager MLP baseline,
but it is not counted as Triton-Ascend evidence because the current project
support list names A2/910B rather than that original SKU.

The image's preinstalled Triton LLVM/MLIR identifies as `20.0.0git`, while the
native GAC build contract is LLVM/MLIR 20.1.8.  The on-image native build used
a separate validation copy with narrow API/printing compatibility edits and a
compiler permissiveness flag for an LLVM-header issue.  Those temporary edits
were not used as evidence of a new supported native toolchain; the exact
details are retained with the validation record.

## Remaining complete-model compiler work

G0.5 is not a conventional static Qwen graph.  The VLM has 24 layers: 18
GatedDeltaNet linear-attention layers and six full-attention layers.  The
Action Expert has 24 full-attention/AdaLN layers.  The vision tower adds a
24-layer Conv3D ViT and PatchMerger.  End-to-end inference also includes MRoPE,
SparseKVCache plus recurrent GDN state, up to 300 autoregressive token steps,
and ten flow-matching Euler steps.

The remaining compiler work includes BF16 with selected FP32 islands, RMSNorm
and adaptive RMSNorm, Conv3D, grouped attention and MRoPE, explicit state/cache
interfaces, GatedDeltaNet, and fixed-bound prefill/decode/action entry points.
The current native Vortex ABI, LLVM link path, and runtime are not reusable as
an Ascend binary ABI.  Standard TTIR selection and static graph planning are
reusable, but unsupported regions must fail closed until an explicit
CANN/Ascend partition exists.

## Phase-1 acceptance criteria

The slice is ready to advance only when all of the following are recorded:

1. Galaxea has confirmed the exact hardware/image license boundary.
2. Source repository commit, gated-model revision, file sizes, and SHA-256
   values are pinned; the trusted extractor version/code digest and complete
   extraction manifest are retained; no unsafe checkpoint is loaded by GAC.
3. The graph has exactly three MatMul, one Sigmoid, and two Mul operations,
   static input/output `[1,32,1024]`, and no unknown operation.
4. One physical NPU runs all inputs, constants, intermediates, and outputs;
   profiler evidence shows no CPU fallback.  The second card repeats the test
   independently rather than introducing distributed execution.
5. At least 100 deterministic FP32 cases match the official PyTorch oracle with
   `rtol <= 1e-4` and `atol <= 1e-4`, with maximum errors recorded.  A later
   BF16 run uses an explicitly documented tolerance and keeps required FP32
   islands.
6. After 50 warm-ups, 200 measured iterations report p50/p95 latency, peak HBM,
   and all software/firmware versions.  Phase 1 establishes a baseline; it does
   not promise a target throughput.
