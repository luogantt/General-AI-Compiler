# G0.5 R1 Lite full-model inference — Ascend 910B2 result template

> Status: **template only**. No latency, throughput, or memory number in this
> document is a measured result. Populate every `TBD` from the JSON emitted by
> `tools/galaxeavla/benchmark_g05_full.py`; do not estimate from the earlier
> single-layer MLP benchmark.

## Measurement identity

| Field | Required value |
|---|---|
| Hardware | 1 × Ascend 910B2, 64 GB HBM |
| Source | `OpenGalaxea/GalaxeaVLA@89f2322b4ad016e192437adc1a2c253b05bab246` plus recorded Ascend overlay hash |
| Checkpoint | `OpenGalaxea/G05`, pinned revision `aca2916ba2adaba3ce05bfa279cdc6b98da35cad`, `g05-base` |
| Checkpoint SHA-256 | `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327` |
| Input scope | Deterministic **synthetic** R1 Lite request, batch 1 |
| Images | 3 cameras × `[3, 720, 1280] uint8`, resized by the official processor to 256×256 |
| State | 14 raw dimensions, merged/padded to 27 model dimensions |
| Output horizon | 32 actions; service consumes 16 actions per recompute |
| Compile mode | Eager baseline, `torch.compile=false` |

The headline path must execute the checkpoint-authored full chain:

1. vision + VLM prefill;
2. CoT autoregressive generation (`predict_cot=true`, EOV/EOS early stop,
   AR budget 300);
3. continuous flow matching, 10 Euler steps;
4. discrete action autoregressive generation (`|`/EOS early stop, AR budget
   at least 300);
5. ActionCodec decode and processor postprocessing.

`max_text_tokens=160` is the processor/input-text limit. It is not the CoT
generation limit.

## Configuration compatibility disclosure

The released checkpoint's `.hydra/config.yaml` remains authoritative for model
architecture and CoT/FM/AR execution switches. Its saved `data` node references
legacy pretraining YAML files that are absent from the pinned public source.
The benchmark therefore replaces only the data/processor deployment fragments
with public `configs/data/r1lite.yaml` and the processor/tokenizer overrides from
`configs/task/r1lite.yaml`. The JSON report records hashes of all three config
inputs under `configuration_compatibility`.

The saved Cholesky asset is unavailable and applies only to `behavior_r1pro`.
The R1 Lite benchmark sets `fm.use_correlated_noise=false`; R1 Lite uses the
same standard-Gaussian noise branch with either setting.

## Headline results

| Metric | Cold | Warm p50 | Warm p95 | Warm max |
|---|---:|---:|---:|---:|
| End-to-end recompute request | TBD | TBD | TBD | TBD |
| CPU raw-build + preprocess | TBD | TBD | TBD | TBD |
| H2D | TBD | TBD | TBD | TBD |
| Vision + VLM prefill | TBD | TBD | TBD | TBD |
| CoT AR generation | TBD | TBD | TBD | TBD |
| FM action, 10 steps total | TBD | TBD | TBD | TBD |
| Discrete action AR generation | TBD | TBD | TBD | TBD |
| ActionCodec / AR processor decode | TBD | TBD | TBD | TBD |
| D2H + postprocess | TBD | TBD | TBD | TBD |

| Derived metric | Result |
|---|---:|
| Recomputes/s at warm p50 | TBD |
| Theoretical 16-action capacity (`16 / p50`) | TBD Hz |
| CoT generated tokens/request, p50 | TBD |
| CoT token/s, p50 | TBD |
| Raw action AR generated tokens/request, p50 | TBD |
| Parsed action tokens/request, p50 | TBD |

Parsed action tokens are the tokens exposed by upstream `decode_ar`; they are
not the complete raw action-AR sequence and must not be presented as decoder
token throughput. The first AR call may already generate action tokens; record
both calls' raw `generated_ids` lengths, including stop tokens. A one-token
action AR call can stop from the existing hidden state without another decoder
forward pass. These counts and the input must accompany the timing.

## Memory and validation

| Metric | Result |
|---|---:|
| Model load time | TBD |
| Cold peak HBM allocated | TBD |
| Cold peak HBM reserved | TBD |
| Warm measurement peak HBM allocated | TBD |
| Warm measurement peak HBM reserved | TBD |

Before publishing results, verify:

- checkpoint size and SHA-256 passed;
- JSON says `result_scope = synthetic-input full-model throughput`;
- configuration contract contains `predict_cot=true`, FM steps 10, both action
  heads enabled, and AR max-new-tokens 300;
- all measured requests returned the expected action-part shapes and finite
  continuous, decoded AR and served action values;
- NPU synchronization was enabled at every timing boundary;
- logs contain no runtime errors; explicitly state whether a profiler was used
  to check accelerator-operator CPU fallback (absence of a warning is not proof);
- warmup and measurement counts are disclosed.

## Command

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/galaxeavla/benchmark_g05_full.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --checkpoint /root/work/models/galaxeavla-g05/g05-base/checkpoints/model_state_dict.pt \
  --asset-root /root/work/models/galaxeavla-g05 \
  --device npu:0 \
  --warmup 5 \
  --iterations 30 \
  --action-steps 16 \
  --output /root/work/results/g05-base-r1lite-full-910b2.json
```

If the pure-PyTorch GatedDeltaNet fallback makes the first baseline too slow,
an initial `--warmup 3 --iterations 10` run may be reported, but its reduced
sample count must remain explicit. Never replace missing measurements with an
MLP extrapolation.
