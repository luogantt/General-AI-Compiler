# G0.5 R1 Lite full-model inference on Ascend 910B2

On 2026-11-16, the official G0.5 base checkpoint completed the full R1 Lite
inference path on one 64 GB Ascend 910B2 using TorchNPU eager execution. This
is a **synthetic-input full-model benchmark**, including image processing,
transfers, vision/VLM prefill, CoT generation, 10-step flow matching, the
discrete action AR call, ActionCodec decoding and output postprocessing.

With one host intra-op thread, the full request measured **3,036.794 ms p50**
and **3,130.296 ms p95** after five warmups and 30 measured requests. Peak
allocated NPU memory was **15.9311 GiB**. The full model call alone, excluding
CPU preprocessing/postprocessing and transfers, measured **2,994.278 ms p50**.

## Measurement identity

| Field | Value |
|---|---|
| Hardware | 1 × Ascend 910B2, 64 GB HBM; logical device `npu:0` |
| Software image | Ubuntu 22.04.5 LTS, AArch64; CANN 9.0.0; Triton-Ascend 3.2.1 image |
| Python / accelerator | Python 3.11.15; PyTorch `2.7.1+cpu`; torch_npu `2.7.1.post4` |
| Upstream source | `OpenGalaxea/GalaxeaVLA@89f2322b4ad016e192437adc1a2c253b05bab246` |
| Overlay SHA-256 | `3cfeedaa69f7d965c5601369082e3314c5f95b11b8ecc849675b8c132b8e047d` |
| Checkpoint | `OpenGalaxea/G05@aca2916ba2adaba3ce05bfa279cdc6b98da35cad`, `g05-base` |
| Checkpoint size | 11,440,519,964 bytes |
| Checkpoint SHA-256 | `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327` |
| Input | One deterministic synthetic R1 Lite request, batch 1, seed `20261113` |
| Images | Three `[3,720,1280]` uint8 images; official processor resizes each to `[3,256,256]` |
| State | 14 raw dimensions, merged/padded to 27 model dimensions |
| Prompt | `Pick up the object and place it in the target area.` |
| Input SHA-256 | `771f9a42991d0ee8a577b611bfdafba6f8ad19b7838c2209c0bb278bdded304b` |
| Outputs | 32-step action horizon; the service is configured to consume 16 steps between recomputes |
| Execution | TorchNPU eager with BF16 autocast and retained FP32 precision islands; `torch.compile=false` |
| Sampling | 1 cold request, 5 warmups, 30 measured requests in each process |

The model is loaded onto the NPU despite PyTorch's `+cpu` build suffix;
`torch_npu` supplies the accelerator backend. Triton-Ascend is installed in the
image, but these full-model numbers do not measure a GAC-compiled or a
Triton-compiled whole model. The separate phase-1 MLP result remains a
[compiler-slice validation](../galaxeavla-g05-mlp-910b2-2026-11-13/README.md).

## One intra-op host thread

This run sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before Python starts.
The JSON records PyTorch intra-op threads as 1 and inter-op threads as 256,
plus both environment variables. The setting controls host work; it does not
reduce NPU core count or set every host thread pool to one.

All times below are milliseconds, measured with NPU synchronization. Each
stage has its own sample distribution, so stage medians need not sum to the
request median. Cold results exclude model loading. Validation and JSON
writing are outside the timed request; preprocessing and H2D/D2H are included.

| Metric | Cold | Warm p50 | Warm p95 | Warm max |
|---|---:|---:|---:|---:|
| Full recompute request | 4,380.902 | 3,036.794 | 3,130.296 | 3,147.989 |
| CPU raw observation build + preprocessing | 41.227 | 38.683 | 39.346 | 39.781 |
| H2D | 1.284 | 1.045 | 1.102 | 1.143 |
| Vision + VLM prefill | 1,408.444 | 360.590 | 366.470 | 374.682 |
| CoT AR call | 1,823.865 | 1,772.120 | 1,843.057 | 1,843.594 |
| Flow matching, all 10 steps | 985.769 | 830.077 | 866.763 | 875.562 |
| Discrete action AR call | 2.424 | 1.834 | 2.023 | 2.081 |
| ActionCodec / AR processor decoding | 100.465 | 12.988 | 13.448 | 13.511 |
| D2H | 1.116 | 1.041 | 1.134 | 1.146 |
| Output postprocessing | 1.517 | 1.351 | 1.413 | 1.495 |

| Derived quantity | Result |
|---|---:|
| Recomputes/s, reciprocal of warm p50 | 0.3293 |
| `16 / warm p50 seconds`, theoretical action capacity | 5.2687 Hz |
| Raw IDs returned by the CoT AR call, per request | 20 |
| Raw IDs returned by the action AR call, per request | 1 |
| Action tokens parsed from the combined sequence, per request | 18 |

The theoretical 16-action capacity is arithmetic on the service chunk size.
It does not establish closed-loop robot control frequency, task success or
sustained throughput under concurrent requests.

## Default-thread comparison

The first full run used the host's default thread settings. A separate host
check reported 256 PyTorch intra-op threads; that run's JSON predates the
explicit `cpu_threads` metadata field. It is retained unchanged as
[result-default-threads.json](result-default-threads.json), with its
[execution log](run-default-threads.log).

| Metric | Default-thread p50 | Default-thread p95 | 1 intra-op thread p50 |
|---|---:|---:|---:|
| Full recompute request, ms | 5,401.077 | 6,033.327 | 3,036.794 |
| CPU build + preprocessing, ms | 1,306.485 | 1,637.444 | 38.683 |
| Vision + VLM prefill, ms | 1,507.505 | 1,869.611 | 360.590 |
| CoT AR call, ms | 1,764.292 | 1,922.923 | 1,772.120 |
| Flow matching, ms | 813.490 | 886.317 | 830.077 |
| Discrete action AR call, ms | 1.897 | 2.017 | 1.834 |
| ActionCodec / AR processor decoding, ms | 13.490 | 15.352 | 12.988 |

The default-thread full-request range was 4,665.244–6,325.806 ms, with mean
5,427.301 ms. Its cold request took 7,618.828 ms and model loading took
73,874.212 ms. The reciprocal-p50 recompute rate was 0.1851/s; the 16-action
capacity calculation was 2.9624 Hz. This is a sequential host-configuration
comparison on one machine, not an interleaved controlled kernel speedup trial.

## Autoregressive count semantics

Every request in both runs returned **20 raw IDs** from the
first, CoT-labelled AR call and **one raw ID** from the subsequent action AR
call. The processor parsed **18 action tokens** from the combined
input + CoT + action sequence. The counters record `generated_ids` lengths,
including stop tokens; they do not record token identities or count decoder
forward passes.

The first token in an AR call is sampled from the existing hidden state.
The one-token action call terminates immediately under the configured stop
rules (`|` or EOS); the precise terminal ID was not saved. Therefore its
roughly 1.8–1.9 ms median does not represent generating 18 fresh action tokens.
The earlier CoT-labelled call can already contain action tokens. Dividing the
18 parsed action tokens by that short action-AR interval would be invalid.
Latency depends on the generated sequence and input; this test does not
measure a 300-token worst case. The configured AR budget remains 300;
`max_text_tokens=160` is the input-text limit.

## Memory and validation

Memory figures below are PyTorch NPU allocator peaks, converted using
`1 GiB = 2^30 bytes`. They are not total card HBM occupancy reported by
`npu-smi` and do not include all driver/runtime allocations.

| Metric | Default threads | 1 intra-op thread |
|---|---:|---:|
| Model loading, ms | 73,874.212 | 81,945.780 |
| Cold peak allocated, GiB | 15.9257 | 15.9257 |
| Cold peak reserved, GiB | 16.9980 | 16.9980 |
| Warm measured peak allocated, GiB | 15.9311 | 15.9311 |
| Warm measured peak reserved, GiB | 17.0332 | 17.0332 |

Both runs passed size and SHA-256 validation for the checkpoint.
All 72 requests (36 per run: cold + warmups + measured) returned finite continuous and
decoded AR actions of shape `[1,32,27]`, and finite postprocessed left/right
arms `[1,32,6]` and grippers `[1,32,1]`.

The associated checks passed: 15 Python unittest cases for the benchmark and
overlay, 24 PowerShell Pester cases for downloading/staging, and eight upstream
device/fallback tests. These tests include output-validation failures and
preservation of patchification layout; they are not full-model accuracy tests.

The logs retain expected ActionCodec metadata, image-size override, storage and
NPU internal-format warnings. No runtime error was reported in either run.
No accelerator profiler was captured, so absence of a CPU
fallback warning is not proof that every accelerator operator executed on the
NPU. CPU preprocessing/tokenization/postprocessing is intentional. Full-model
numeric agreement against CUDA and real-data/robot task accuracy have not been
validated.

## Configuration and implementation changes

The overlay adds device-aware autocast, synchronization and allocator cleanup;
keeps non-CUDA GatedDeltaNet on the existing PyTorch backend and vision
attention on SDPA; and rewrites temporal image patchification to stay within
rank-eight intermediates while preserving element order. Model dimensions,
checkpoint parameter names and output horizons remain unchanged.

The released checkpoint's saved `data` node references legacy training YAMLs
absent from the pinned public source. The harness uses public R1 Lite data and
processor/tokenizer deployment fragments, recording configuration hashes in
each result. Checkpoint-authored CoT, continuous/discrete action switches and
10 FM steps remain enabled. The missing Cholesky asset applies only to
`behavior_r1pro`; `fm.use_correlated_noise=false` avoids loading it for this
`galaxea_r1lite` benchmark, which already uses standard Gaussian noise.
See the [configuration explanation](../../../../integrations/galaxeavla/runtime-setup.md#configuration-compatibility).

## Counted arithmetic and interactive view

The [offline network viewer](../../../visualizations/g05-ascend-compute-latency.html)
follows the supplied Netron-style layout, with searchable subgraphs, measured
stage timing and a request-frequency arithmetic calculator. Its 137 subgraphs
separate measured timing from counted arithmetic; internal modules have no
invented latency allocation. Download the HTML and open it locally.

A separate, instrumented one-request diagnostic on the same checkpoint, source,
input and configuration records actual module shapes and call counts in
[workload.json](workload.json). Its log is retained as
[workload-diagnostic.log](workload-diagnostic.log); **diagnostic timings are not
performance results** and are not used by the viewer.

| Phase | Counted GFLOP / request | Independent warm p50, ms |
|---|---:|---:|
| Prefill, including input encoding | 1120.402092 | 360.590 |
| CoT | 72.822571 | 1772.120 |
| Flow matching, 10 steps | 309.176566 | 830.077 |
| Action AR | 1.032966 | 1.834 |
| Final ActionCodec decoding | 1.714520 | 13.143 |
| Counted arithmetic / full request timing | **1505.148715** | **3036.794** |

The count covers Linear, ordinary Conv1d/2d/3d, and AdaptiveRMSNorm's functional
conditioning projection. One multiply-accumulate counts as two FLOPs. Every
actual call is counted; shared parameter storage is deduplicated only for
parameter totals. The 777 shape records represent 6133 calls. Linear uses
`2 * output.numel * in_features`; ordinary convolution uses
`2 * output.numel * (in_channels/groups) * kernel_volume`.

This is an **incomplete count of nominal algorithmic operations**, not full-model
FLOPs, hardware instruction counts or hardware peak requirements. It excludes
attention QK/PV products, GDN functional state operations and decode conv,
ConvTranspose, activations, norms, RoPE, DCT/IDCT, CPU processing and other
functional operations. A target of 1 request/s corresponds to 1.50515 TFLOP/s
of the **counted portion**; multiplying by a requested rate does not account
for bandwidth, scheduling, uncounted work or execution efficiency and does not
predict achievable latency.

The actual prefix is 234 tokens. The resized LM head has 252189 output entries;
the CoT call performs 19 incremental VLM forwards and 20 head calls, followed
by one head call in action AR. The AE runs ten times with a 253-token prefix
cache in its six cross-attention layers. Prefill also executes **1.816265 GFLOP
of action-placeholder encoding through Codec Encoder/RVQ**. Those 36 calls
belong to prefill, not to the final 13.143 ms decoder interval.

The loaded policy plus independent ActionCodec backend contains 2,902,159,843
deduplicated parameters, all resident FP32 in this run: 10.8114 GiB of parameter
tensors. BF16 autocast does not mean BF16-resident weights. This parameter-only
number excludes buffers, caches, activations and allocator/runtime overhead.

To reproduce the diagnostic in the runtime environment, use
`tools/galaxeavla/count_g05_workload.py` with the same source/checkpoint/assets
arguments as the benchmark and a separate `--output` path. It forces one smoke
request and writes no diagnostic latency into its JSON. Rebuild the viewer from
the checked-in evidence on any Python 3.11+ host:

```bash
python tools/galaxeavla/build_compute_view.py \
  --output docs/visualizations/g05-ascend-compute-latency.html
```

Six counter unit tests passed, and the observed formulas, sums and call counts
were independently reviewed. Browser checks covered all 137 reachable subgraphs
and parent/child arithmetic sums, navigation, search, budget calculation,
unchanged measured timing, evidence download and a 390-pixel mobile layout.

## Reproduction and evidence

The [runtime setup](../../../../integrations/galaxeavla/runtime-setup.md)
documents the fixed source, overlay, exact installed additions using a domestic
PyPI source, and official checkpoint acquisition. With the environment active
and its source/assets variables set, run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$G05_DIR/src:$G05_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python "$GAC_DIR/tools/galaxeavla/benchmark_g05_full.py" \
  --source-dir "$G05_DIR" \
  --checkpoint "$G05_ASSETS/g05-base/checkpoints/model_state_dict.pt" \
  --asset-root "$G05_ASSETS" --device npu:0 \
  --warmup 5 --iterations 30 --action-steps 16 \
  --output /data/results/g05-base-r1lite-full-910b2.json
```

- [One intra-op thread report](result.json) and [log](run.log): final 30-sample result.
- [Default-thread report](result-default-threads.json) and [log](run-default-threads.log): original 30-sample comparison.
- [Environment snapshot](environment.json): package versions, source-file hashes and `npu-smi`; its benchmark-script hash identifies the first run.
- [Execution manifest](execution-manifest.json): final overlay/script hashes, both commands and CPU quota details.

The two runs use the same model-computation source. The second harness adds
CPU-thread metadata to the JSON; this reporting change does not alter model
computation. G0.5 source and weights retain the upstream license described in
the [integration notice](../../../../integrations/galaxeavla/NOTICE).
