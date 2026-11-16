# G0.5 full-model TorchNPU runtime setup

This recipe reproduces the R1 Lite **synthetic-input eager benchmark** on one
64 GB Ascend 910B2. It uses the official G0.5 checkpoint and the
[Ascend source overlay](README.md); it does not compile the complete model with
GAC or Triton-Ascend. Model weights remain outside the source repository.

## Environment

The validation host used the Triton-Ascend 3.2.1 image with Ubuntu 22.04,
Python 3.11.15, CANN 9.0.0, PyTorch 2.7.1 and torch_npu 2.7.1.post4 on AArch64.
Keep the image's PyTorch, torch_npu and Triton installations. Package metadata
for `triton` and `triton-ascend` can coexist and need not match the version of
the imported Triton implementation.

The upstream package declares Python `<3.11` and CUDA/training/simulation
dependencies. This deployment instead imports the pinned source through
`PYTHONPATH` in the image's Python 3.11 environment. It does not run the
upstream full dependency installer. CUDA-only FLA, FlashAttention,
bitsandbytes and liger kernels are not needed for this eager inference path.

The following explicit additions match the benchmark virtual environment;
`--no-deps` preserves the vendor accelerator packages. The inherited image
also supplies NumPy 1.26.4, SciPy 1.13.1, PyYAML 6.0.3, pytest 8.3.2 and the
standard PyTorch dependencies. This is a recipe for that image, not a complete
lock file for an arbitrary Linux installation. The command uses the
[TUNA PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.11 -m venv --system-site-packages /workspace/venvs/g05-ascend
source /workspace/venvs/g05-ascend/bin/activate
python -m pip install --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple \
  accelerate==1.8.1 transformers==4.57.1 huggingface-hub==0.35.0 \
  safetensors==0.6.1 tokenizers==0.22.2 hf-xet==1.8.0 \
  hydra-core==1.3.2 hydra-colorlog==1.2.0 omegaconf==2.3.1 \
  antlr4-python3-runtime==4.9.3 colorlog==6.12.0 \
  torchvision==0.22.1 timm==0.9.10 einops==0.8.0 \
  vector-quantize-pytorch==1.27.6 einx==0.4.3 frozendict==2.5.1 \
  opencv-python-headless==4.10.0.84 pillow==10.2.0 \
  rootutils==1.0.7 python-dotenv==1.2.4 rich==13.9.4 termcolor==2.4.0 \
  markdown-it-py==4.3.0 mdurl==0.1.2 Pygments==2.21.0 \
  GitPython==3.1.45 gitdb==4.0.12 smmap==5.0.3 \
  msgpack==1.1.2 websockets==16.0 tqdm==4.70.1 regex==2026.11.9 \
  requests==2.34.2 urllib3==2.8.0 idna==3.20 \
  certifi==2026.10.23 charset-normalizer==3.5.2
```

## Pinned source and overlay

Set these paths for your host. `GAC_DIR` is this repository, already checked
out on the `GalaxeaVLA` branch. Apply the overlay to a clean upstream checkout.

```bash
export GAC_DIR=/workspace/General-AI-Compiler
export G05_DIR=/workspace/GalaxeaVLA
export G05_ASSETS=/data/models/galaxeavla-g05
git clone https://github.com/OpenGalaxea/GalaxeaVLA.git "$G05_DIR"
git -C "$G05_DIR" checkout --detach 89f2322b4ad016e192437adc1a2c253b05bab246
python "$GAC_DIR/tools/galaxeavla/apply_ascend_overlay.py" "$G05_DIR"
python "$GAC_DIR/tools/galaxeavla/apply_ascend_overlay.py" "$G05_DIR" --apply
export PYTHONPATH="$G05_DIR/src:$G05_DIR"
cd "$G05_DIR"
python -m pytest -q tests/test_device_utils.py tests/test_qwen35_device_fallbacks.py
```

## Official model assets

The downloader is pinned to `OpenGalaxea/G05` revision
`aca2916ba2adaba3ce05bfa279cdc6b98da35cad`. It needs access to the gated model
under the upstream terms; see the [license notice](README.md#license-boundary).
The 12-file base subset is about 12 GB:

```text
galaxeavla-g05/
  action_tokenizer.pt
  g05-base/
    .hydra/config.yaml
    dataset_stats.json
    checkpoints/model_state_dict.pt
  qwen3_5_2b_base_processor/  # all eight released processor/tokenizer files
```

Checkpoint size is `11440519964` bytes and its SHA-256 is
`072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327`.
Both downloaders verify large LFS objects against pinned official hashes.
The benchmark independently verifies the checkpoint by default.

On Windows, from the GAC repository, the complete downloader can prompt for
the HF token without placing it in a command:

```powershell
./tools/galaxeavla/download_g05_base.ps1 -Destination D:/models/galaxeavla-g05
```

Transfer that complete directory, including `.hydra`, to `$G05_ASSETS` on the
NPU host. When direct HF access from the server fails, the following alternative
downloads only sidecars locally and streams the three large objects from
official signed CDN URLs to the server. `ascend-host` is a placeholder SSH
config alias with a verified target key and key authentication; adjust the
port and paths to match your connection.

```powershell
$assetDir = 'D:/models/galaxeavla-g05'
./tools/galaxeavla/download_g05_base.ps1 -Destination $assetDir -SmallFilesOnly
ssh ascend-host 'mkdir -p /data/models'
scp -O -r $assetDir ascend-host:/data/models/
./tools/galaxeavla/stage_g05_base_remote.ps1 `
  -SshHost ascend-host -SshPort 22 -SshUser root -HostKeyAlias ascend-host `
  -RemoteRoot /data/models/galaxeavla-g05 -Segments 8
```

The small-file pass writes `g05-base-small-files-download-manifest.json`;
the complete local download writes `g05-base-download-manifest.json`.
The HF token stays on the local machine during remote staging. Model downloads
use the official HF/CDN endpoints; the domestic mirror above is for Python
packages, not the gated checkpoint.

## Full-model benchmark

Run after activating the environment and exporting the paths above. The
output directory is outside the upstream checkout, preserving the source audit.

```bash
mkdir -p /data/results
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$GAC_DIR/tools/galaxeavla/benchmark_g05_full.py" \
  --source-dir "$G05_DIR" \
  --checkpoint "$G05_ASSETS/g05-base/checkpoints/model_state_dict.pt" \
  --asset-root "$G05_ASSETS" --device npu:0 \
  --warmup 5 --iterations 30 --action-steps 16 \
  --output /data/results/g05-base-r1lite-full-910b2.json
```

An initial `--smoke-only` invocation runs one cold request and does not publish
warm statistics. The full invocation separates load, cold request, warmups and
measured requests; synchronizes the NPU at timing boundaries; records generated
token counts, memory and action shapes; and checks action finiteness outside the
timed request. Inputs are deterministic synthetic three-camera R1 Lite images
and state, batch 1. Outputs cover a 32-step horizon; `--action-steps 16` describes
the service chunk consumed between recomputes, not a shorter model horizon.
These timings are not robot-task accuracy or an achieved robot control rate.

The validation container exposed 256 logical CPUs but had a 32-CPU quota.
Its default PyTorch intra-op pool used 256 threads. The command above limits
CPU intra-op parallelism for this NPU inference workload; both the default
and one-thread measurements are retained in the
[result report](../../docs/validation/ascend/galaxeavla-g05-full-910b2-2026-11-16/README.md).
This environment setting does not compile or fuse model operators.

## Configuration compatibility

The saved checkpoint `.hydra/config.yaml` remains authoritative for model
architecture and CoT/FM/AR execution. Its `data` node references legacy training
YAML files absent from the pinned public source. The harness replaces that node
with public `configs/data/r1lite.yaml` and merges only processor/tokenizer
deployment fragments from `configs/task/r1lite.yaml`. The report records hashes
of all three configurations. In particular, it retains the checkpoint's
`predict_cot=true`, both action heads, 10 FM steps and AR budget 300.

The saved correlated-noise loader also references an unavailable Cholesky file.
Its configured embodiment list contains only `behavior_r1pro`; the upstream
noise sampler uses standard Gaussian noise for `galaxea_r1lite` regardless.
The harness sets `fm.use_correlated_noise=false` to avoid loading that unused
file. This override is specific to R1 Lite and is not a recipe for reproducing
Behavior R1 Pro inference.

Finally, the harness selects the existing PyTorch GatedDeltaNet backend and
disables `torch.compile`. The overlay keeps non-CUDA attention on its
PyTorch/SDPA paths and uses device-aware precision and synchronization. A
successful run establishes execution and finite outputs; agreement with CUDA
outputs and task-level accuracy require separate validation.
