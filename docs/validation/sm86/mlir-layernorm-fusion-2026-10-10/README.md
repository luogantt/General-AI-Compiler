# SM86 MLIR LayerNorm 融合微基准

本目录冻结一个独立的、真实执行的 **MLIR GPU Dialect → NVVM → PTX**
LayerNorm 参考实现及 RTX 3080 测量结果。测试形状为 BEVFormer FFN 的
`[2500, 256]` FP32 尾部，归一化使用 `epsilon=1e-5`。

这个实验不属于当前产品运行路径。仓库当前的 `MlirCudaSession` 使用语义等价的
NVRTC CUDA C kernel 执行 Bias、Residual 和 LayerNorm 融合；它尚未加载本例生成的
PTX，也没有自动把产品 Tensor/Linalg IR 降低到本例。此结果不能表示完整
BEVFormer 推理速度。

## 产物

- [`examples/mlir_sm86_layernorm_fusion.mlir`](../../../../examples/mlir_sm86_layernorm_fusion.mlir)：
  含串行基线、每行一个 block 的归约实现，以及
  `LN(matmul_output + bias + residual)` 融合实现。
- [`tools/benchmark_mlir_layernorm_fusion.py`](../../../../tools/benchmark_mlir_layernorm_fusion.py)：
  从 `mlir-opt` 产物提取内嵌 PTX，用 CuPy 加载、校验和计时。
- [`benchmark-rtx3080.json`](benchmark-rtx3080.json)：使用当前 MLIR source、
  benchmark driver 和重新生成的 PTX 在 RTX 3080 上运行得到的原始报告。

三个 MLIR kernel 的 ABI 都使用裸设备指针。优化和融合 kernel 的启动配置是
`grid=(2500,1,1)`、`block=(256,1,1)`，一个 block 处理一行，一个线程处理一列。
`gpu.all_reduce` 经 NVVM lowering 后生成 warp shuffle 和 block reduction。串行基线
使用 `grid=(10,1,1)`、`block=(256,1,1)`，每个线程依次处理一整行的 256 个元素。

## 复现

先构建仓库所需的 MLIR/NVPTX 工具；构建脚本不会修改 `llvm-project`：

```bash
bash tools/build_mlir_wsl.sh
```

在仓库根目录执行实际 GPU lowering。`cubin-format=isa` 使 `gpu.binary` 保存文本
PTX，基准工具可直接提取并加载：

```bash
mkdir -p build

MLIR_OPT="${GAC_MLIR_OPT:-$HOME/.cache/gac-mlir/current-build/bin/mlir-opt}"
MLIR_BUILD="$(dirname "$(dirname "$(readlink -f "$MLIR_OPT")")")"
LLVM_SOURCE_REVISION="$(cat "$MLIR_BUILD/.gac-build-revision")"
MLIR_OPT_VERSION="$("$MLIR_OPT" --version)"
MLIR_OPT_SHA256="$(sha256sum "$MLIR_OPT" | awk '{print $1}')"
LOWERING_OPTIONS='--gpu-lower-to-nvvm-pipeline=kernel-bare-ptr-calling-convention=1 cubin-chip=sm_86 cubin-format=isa opt-level=3 allow-pattern-rollback=0'
"$MLIR_OPT" \
  examples/mlir_sm86_layernorm_fusion.mlir \
  "$LOWERING_OPTIONS" \
  -o build/mlir_sm86_layernorm_fusion.binary.mlir
```

在安装了 NumPy、CuPy 且具有 SM86 NVIDIA GPU 的 Python 环境中执行：

```bash
python tools/benchmark_mlir_layernorm_fusion.py \
  --binary-mlir build/mlir_sm86_layernorm_fusion.binary.mlir \
  --mlir-source examples/mlir_sm86_layernorm_fusion.mlir \
  --mlir-opt-version "$MLIR_OPT_VERSION" \
  --mlir-opt-sha256 "$MLIR_OPT_SHA256" \
  --llvm-source-revision "$LLVM_SOURCE_REVISION" \
  "--lowering-options=$LOWERING_OPTIONS" \
  --warmup 50 \
  --iterations 100 \
  --repeats 20 \
  --output build/mlir-layernorm-fusion-reproduced.json
```

脚本也接受已经提取的 PTX：把 `--binary-mlir PATH` 换成 `--ptx PATH`。随机输入的
固定种子为 `20261010`，NumPy bit generator 为 PCG64。脚本先触发 PTX 和 NVRTC
JIT，并将四条完整路径分别与 CuPy FP32 参考结果比较，再执行 50 次预热；每组通过
CUDA Event 计量 100 次调用，报告 20 组数据的中位数、P10、P90 和最小值。

## RTX 3080 结果

报告生成于 `2026-10-10T15:11:24.853932+00:00`。测试环境为 NVIDIA GeForce
RTX 3080、Compute Capability 8.6、CUDA Runtime `12090`、CUDA Driver `13000`、
Python `3.11.17`、NumPy `2.4.6`、CuPy `13.6.0`。输入、权重和输出均已驻留 GPU；
数据不包含 GEMM、主机和设备间传输、模块加载及首次 JIT。

| 路径 | GPU 中位延迟 | 说明 |
| --- | ---: | --- |
| 串行 MLIR LayerNorm | 174.121 μs | 每线程处理一行 |
| block reduction MLIR LayerNorm | **10.271 μs** | 相对串行实现快 **16.95×** |
| 两个独立 Add + block LayerNorm | 28.344 μs | 两个 Add 基线由脚本中的 NVRTC CUDA C kernel 执行 |
| MLIR Bias + Residual + LayerNorm 融合 | **13.604 μs** | 相对未融合尾部快 **2.08×** |

block reduction 和融合 kernel 相对 CuPy 参考的最大绝对误差均为
`7.152557373046875e-07`，`rtol=2e-4`、`atol=2e-4` 的校验通过。串行基线最大绝对
误差为 `1.9073486328125e-06`。

本次报告把受测输入和工具链绑定到以下 SHA-256：MLIR source
`140fac08e417580c114bd0ac9c013f5062e5848fc0526119fcf7ae96c5e51bb7`、benchmark
driver `e365a6efc96732c663473272712a595a83bdd8f6240d4a271c0a089defac5df1`、binary
MLIR `e32714a537d114e44e60bc68f27227513d6a1a625cb8acf9a1cdbdc7140f738d`、内嵌 PTX
`ce30bdc481637eaca88c4905d5783cf9edd5e844864005d25e30ca7b91697749`、`mlir-opt`
可执行文件 `08ff07313f22cb35f8c9ee559dd6489244200d8f1c540051a2ec592f16a8d55c`、lowering
options `df6db84d445de74086d08b6e560e36d2b820b02e24b57b06f282d788fc19a628`。
LLVM source revision 为 `765392a6b4498f1dc57233f03a31b87462fdef01`，工具报告
`LLVM 24.0.0git (Optimized build)`。binary MLIR 的摘要绑定此次受测文件；其中包含
lowering 耗时元数据，因此独立重建时以 PTX 摘要核对设备代码更稳定。完整原始样本、
seed、`rtol=atol=2e-4`、正确性误差和上述 provenance 均保存在 JSON 中。
