# Native MLIR CUDA validation

`run_cuda_tests.py` compiles through the C++ ONNX importer and native MLIR passes,
then calls `compile-cuda.sh` for LLVM NVPTX code generation. Numerical validation
executes only `gac-cuda-run` and compares its raw outputs against independently
generated reference tensors. Python supplies test orchestration, tensor I/O and
comparison; it never evaluates a model operator as a fallback.

## Existing fixtures

Reuse the same small fixtures as the native CPU and Vortex tests. Keep their
`cases.json`, `model.onnx`, `inputs.npz` and `expected.npz` together.

| Fixture generator | Coverage |
| --- | --- |
| `generate_models.py` | Conv + ReLU, MatMul, Softmax, LayerNorm, GridSample, FP16 rounding boundaries |
| `generate_base_ops_models.py` | DCNv2, grouped/dilated cases, MSDA, retained branch and arena reuse |
| `generate_matmul_bias_models.py` | Bias fusion, broadcasting and multiple consumers |
| `generate_msda_offsets_models.py` | Coordinate fusion, retained outputs and incompatible view patterns |
| `generate_view_alias_models.py` | Views, aliases and live fork storage |
| `generate_scatter_legalization_models.py` | Scatter forms and exact shape/index semantics |

The harness can accept multiple `--fixtures` directories. `--case NAME` or
`--case SUITE/NAME` selects particular cases. Intentional rejection fixtures are
listed under `excluded_cases`, never counted as numerical passes. Reference
generation remains a separate step and may require PyTorch, torchvision or ONNX
Runtime; execution and comparison require only NumPy and the CUDA runner.

## Compile on a build machine

Run these commands in a Linux shell. Use a fresh output directory for each run.
The toolchain must match the LLVM/MLIR version used to build `gac-compile`.

```bash
export CUDA_LIBDEVICE=/path/to/cuda/nvvm/libdevice/libdevice.10.bc
export CUDA_PTXAS=/path/to/cuda/bin/ptxas  # Optional offline PTX assembly.
python3 native/tests/run_cuda_tests.py \
  --compiler /path/to/gac-compile \
  --llvm-bin /path/to/llvm/bin \
  --fixtures build/native-tests \
  --fixtures build/native-base-ops-tests \
  --case conv_bias_relu --case softmax --case layer_norm_axes_and_bias \
  --case dcn_mask_bias --case memory_fragmented_fork \
  --compile-only --arch sm_86 --output-dir build/cuda-compiled
```

Status `compiled_only` means PTX entries and the static manifest were checked.
It does **not** establish numerical correctness or physical GPU latency, even
when optional `ptxas` assembly also succeeds. The report leaves
`gpu_executed=false` and `numerical_passed=null`.

## Execute copied artifacts on the CUDA machine

Copy the output directory, selected fixture directories and native test scripts
to the CUDA machine. No LLVM toolchain is needed there for this mode. Preserve
the fixture suite directory names because they identify the compiled cases.

```bash
python3 native/tests/run_cuda_tests.py \
  --fixtures build/native-tests \
  --fixtures build/native-base-ops-tests \
  --case conv_bias_relu --case softmax --case layer_norm_axes_and_bias \
  --case dcn_mask_bias --case memory_fragmented_fork \
  --skip-compile build/cuda-compiled \
  --runner /path/to/gac-cuda-run --device 0 \
  --warmup 1 --repeat 5 --output-dir build/cuda-gpu-checked
```

Without `--skip-compile`, supplying compiler options and `--runner` compiles and
executes on the same machine. Missing NVIDIA Driver/device, JIT failure,
incomplete dispatch execution or a numerical mismatch returns a nonzero status.
There is no automatic CPU fallback or successful "skip" for these failures.

## Whole BEVFormer model

Use the exact fixed-input archive and independently captured PyTorch outputs
associated with the exported model. All declared outputs must be present.

```bash
python3 native/tests/run_cuda_tests.py \
  --artifact /path/to/full-model-cuda-artifact \
  --inputs /path/to/inputs.npz --reference /path/to/pytorch-reference.npz \
  --reference-description 'PyTorch BEVFormer-base FP32 fixed-input cold frame' \
  --runner /path/to/gac-cuda-run --output-dir build/cuda-base-checked \
  --rtol 1e-3 --atol 1e-4 --run-timeout 7200
```

Alternatively `--model model.onnx --compiler ... --llvm-bin ...` compiles first;
use `--constant-fold-profile bevformer-base` for that model's bounded constant
geometry evaluation. Default small-fixture tolerances are `rtol=atol=2e-5`.
Integer and boolean outputs always require exact equality, and floating-point
outputs must also preserve their expected dtype and shape.

## Evidence and timing

`cuda-validation.json` records each case's status and a reference to its
`validation.json`, compile/runtime logs and output files. Reports contain SHA-256
identities for models, PTX, manifest, constants, inputs, references and returned
tensors; compiler and runtime binaries are also hashed when supplied.

Only `gpu_passed` means the runtime completed every dispatch on CUDA and every
output passed comparison. `runtime-report.json` contains device identity and
CUDA Event timings. Upload, download, host wall time and device dispatch time
are distinct measurements. The checker wall time includes file I/O and hashing
and must not be presented as inference latency. A single-frame result does not
establish warm/history-frame accuracy or a sustained end-to-end FPS target.

Checker-only regression tests are separate from GPU numerical tests:

```bash
python3 native/tests/test_cuda_validation.py -v
```
