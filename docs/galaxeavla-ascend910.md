# GalaxeaVLA G0.5 on Ascend 910

## Status and boundary

This branch starts the port from `lg/bevformer-triton-vortex-deep`.  It does
not yet claim that the complete GalaxeaVLA model runs on Ascend.  The first
reviewable unit is the real Qwen3.5 Action Expert SwiGLU MLP contract; the
native side may export standard TTIR for an Ascend hand-off, but GAC does not
yet compile `npubin` or launch physical NPU kernels.

| Capability | Current status |
| --- | --- |
| G0.5 Action Expert MLP graph import | Implemented, static FP32 slice |
| NumPy reference validation | Implemented |
| Ascend host/CANN/torch_npu discovery probe | Implemented, discovery-only and fail-closed |
| Standard TTIR review artifact | Phase-1 export only |
| Triton-Ascend `npubin` compilation | Not implemented |
| GAC Ascend physical runtime | Not implemented |
| Complete G0.5 inference | Not implemented |

The upstream source baseline used for this plan is GalaxeaVLA commit
`89f2322b4ad016e192437adc1a2c253b05bab246`.  Relevant primary sources are the
[G0.5 model configuration](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/configs/model/g05.yaml),
[Qwen3.5 mixture implementation](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/src/g05/models/g05/qwen35/mixture_qwen35.py),
and [upstream dependency declaration](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/pyproject.toml).

## License gate

G0.5 code and weights use the
[G0.5 Community License](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/LICENSE-G0.5),
not this repository's Apache-2.0 license.  It restricts use to non-commercial
purposes and narrowly defined internal evaluation.  Its internal-PoC language
also requires “Authorized Hardware” and an “Authorized Image”, but the license
does not define those terms or state that Ascend 910 qualifies.

Before downloading or executing the gated checkpoint on Ascend, obtain written
confirmation from Galaxea for the exact NPU SKU and software image, or obtain a
separate commercial license.  Do not commit checkpoint data to this repository.
Any distributed modification or derivative must retain the upstream license,
notices, and modification statements.  This is a release gate, not an optional
documentation item.

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
Ascend target.  A later tool must compile each checked TTIR module with the
matching Triton-Ascend/CANN stack and write an `npubin` plus launcher metadata.

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
- [Triton-Ascend architecture](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/architecture_design_and_core_features.md)

## Complete-model blockers

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
