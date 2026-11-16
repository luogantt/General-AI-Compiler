# GalaxeaVLA Ascend 910 / TorchNPU overlay

This directory carries a reviewable source patch for
[`OpenGalaxea/GalaxeaVLA`](https://github.com/OpenGalaxea/GalaxeaVLA) pinned to:

```text
89f2322b4ad016e192437adc1a2c253b05bab246
```

The patch SHA-256 is:

```text
3cfeedaa69f7d965c5601369082e3314c5f95b11b8ecc849675b8c132b8e047d
```

It is a source overlay, not a fork or a model distribution. No checkpoint,
Hugging Face credential, robot data, generated benchmark result, or local
machine path is included.

For the tested image, domestic PyPI installation, official model downloads and
complete inference command, see [runtime setup](runtime-setup.md). Measured
full-model latency is recorded separately in the
[910B2 validation report](../../docs/validation/ascend/galaxeavla-g05-full-910b2-2026-11-16/README.md).

## Safe application

The helper defaults to a read-only dry run. It validates the exact upstream
commit, validates the bundled patch digest, and runs `git apply --check`:

```bash
python tools/galaxeavla/apply_ascend_overlay.py <galaxeavla-worktree>
```

Apply only after reviewing the result:

```bash
python tools/galaxeavla/apply_ascend_overlay.py <galaxeavla-worktree> --apply
```

`--apply` refuses any tracked, staged, or untracked change in the upstream
worktree. It repeats the dry run before applying and does not stage or commit
the result. This protects an existing GalaxeaVLA checkout from accidental
overlap with user work.

## Patch scope

`patches/0001-ascend-torchnpu-runtime.patch` makes the minimum runtime changes needed
for an eager TorchNPU baseline:

- resolves autocast from the executing tensor/device instead of hard-coding CUDA;
- synchronizes CUDA or NPU correctly for reported phase timings;
- dispatches allocator cache cleanup to the active accelerator;
- makes ActionCodec precision contexts device-aware;
- keeps Qwen3.5 FP32 precision islands on the execution device;
- prevents non-CUDA tensors from entering CUDA-only FLA, causal-conv or
  FlashAttention paths;
- routes non-CUDA GatedDeltaNet and vision attention through the existing
  pure-PyTorch and SDPA fallbacks;
- rewrites Qwen3.5 temporal image patchification into rank-eight intermediate
  tensors while preserving the original rank-nine expression element-for-element;
- adds CPU dispatch/fallback tests.

The overlay does not change model dimensions, checkpoint keys, action routing,
flow-matching steps, autoregressive limits, or output contracts. It also does
not claim that the fallback kernels are performant. They establish a truthful
full-model eager baseline before Triton-Ascend/GAC optimization.

## Validation

The integration test creates a detached temporary worktree at the pinned
revision, runs the default dry run, applies the patch, runs `git diff --check`,
and parses the new device/fallback tests:

```bash
GALAXEAVLA_UPSTREAM=<galaxeavla-checkout> \
  python -m unittest tests.test_galaxeavla_ascend_overlay
```

The test is skipped when no matching upstream checkout is available. Runtime
device tests in the applied upstream tree are:

```bash
python -m pytest -q \
  tests/test_device_utils.py \
  tests/test_qwen35_device_fallbacks.py
```

## License boundary

The pinned upstream revision is in the G0.5 release line. Galaxea states that
G0.5 code, model weights, configurations and related materials are governed by
the **G0.5 Community License Agreement (Non-Commercial + Limited Patent
License)**. The Apache-2.0 license of the surrounding General-AI-Compiler
repository does not relicense GalaxeaVLA or this derivative source patch.

Before using or redistributing the overlay, review and comply with the pinned
upstream [LICENSE-G0.5](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/LICENSE-G0.5)
and [NOTICE](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/NOTICE).
Model access remains gated separately by the model host and the user's accepted
terms. See the local [NOTICE](NOTICE) for attribution and modification scope.
