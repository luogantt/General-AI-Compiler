# BEVFormer FFN：官方 MLIR → NVVM → PTX 真机验证

本目录记录 `mlir-ptx-sm86` 第一版的编译与 RTX 3080 真机结果。验证对象是
一个生产形状 FP32 BEVFormer FFN：

```text
[1, 2500, 256]
  → MatMul(256, 512) + Bias + ReLU
  → MatMul(512, 256)
  → Bias + query residual + LayerNorm
```

这是完整的 **FFN 编译单元**，不是完整 BEVFormer 网络。CNN、Spatial/Temporal
Attention、MSDeformableAttention、Decoder、Detection Head 和跨帧状态仍由现有
CUDA/HXP 路径执行。

## 编译和执行边界

编译器从同一 Canonical Graph 独立生成 Tensor/Linalg 审阅 sidecar；实际设备分支
由 strict matcher 直接物化三个 `gpu.func`。官方 LLVM 24 `mlir-opt` 执行：

```text
GPU Dialect
  → SCF/Arith/MemRef lowering
  → NVVM Dialect
  → LLVM NVPTX backend
  → textual PTX 7.1, .target sm_86
```

固定 pipeline 是：

```text
--gpu-lower-to-nvvm-pipeline=
  kernel-bare-ptr-calling-convention=1
  cubin-triple=nvptx64-nvidia-cuda
  cubin-chip=sm_86
  cubin-features=+ptx71
  cubin-format=isa
  opt-level=3
  allow-pattern-rollback=0
```

实际工具链 revision 为
`765392a6b4498f1dc57233f03a31b87462fdef01`。Runtime 校验图、GPU MLIR、
launch ABI、PTX target、64 位地址、三个精确 entry 和完整性摘要后，通过
`CuPy RawModule(path=...)` 加载 PTX。该路径不调用 cuBLAS，不编译 CUDA C，
也没有 NVRTC、CPU 或旧 CUDA 后端回退。PTX 仍由 CUDA Driver 在模块加载时 JIT
为 SASS；本页的稳态 Kernel 延时不包含 Session 建立和 Driver JIT。

本目录保留了本次真实编译的三个阶段：

- `production-ffn.gpu.mlir`：实际进入官方 lowering pipeline 的 GPU Dialect；
- `production-ffn.binary.mlir`：包含唯一 `gpu.binary/#gpu.object` 的结果；
- `production-ffn.ptx`：Runtime 实际加载的三个 kernel；
- `SHA256SUMS`：以上产物摘要。

这些 SHA-256 用于复核本次证据和发现文件损坏，不是发布者签名。部署时只应加载由
可信编译环境生成，或由外部可信摘要确认的 PTX artifact。

## RTX 3080 结果

设备为 NVIDIA GeForce RTX 3080，Compute Capability 8.6；CUDA Runtime 12.9，
CuPy 13.6.0。两条路径在同一进程共驻留，每个 repeat 交换执行次序，每次路径结束
后同步 CUDA Event。分别以全 PTX 和 cuBLAS hybrid 作为第一次序再独立运行一轮。
每轮 20 个 repeat，每个 repeat 50 次 FFN。

| 首次测量次序 | 全 PTX 中位数 | cuBLAS hybrid 中位数 | 全 PTX 慢于 hybrid |
|---|---:|---:|---:|
| PTX first | 1101.814 µs | 209.029 µs | 5.265× |
| Hybrid first | 1092.239 µs | 214.467 µs | 5.150× |

两轮全 PTX 的 device-resident 延时为 **1.092～1.102 ms/FFN**，正确性最大绝对
误差为 `7.153e-7`。cuBLAS hybrid 为 **0.209～0.214 ms/FFN**。包含输入上传、
输出下载、校验和同步的公开调用 wall 延时分别为：

| 首次测量次序 | 全 PTX wall | cuBLAS hybrid wall |
|---|---:|---:|
| PTX first | 2545.028 µs | 1872.911 µs |
| Hybrid first | 2538.097 µs | 1799.507 µs |

原始逐 repeat 数据在：

- `benchmark-ptx-first-rtx3080.json`
- `benchmark-hybrid-first-rtx3080.json`

## 为什么没有自动变快

MLIR 到 PTX 只改变编译链，不能自动保证 kernel 调度质量。第一版两个 MatMul
采用“一线程计算一个输出、串行遍历 K”的正确性实现，没有 shared-memory tiling、
向量化、Tensor Core、流水化或 autotuning。它因此比高度优化的 cuBLAS 慢约
5.2～5.3 倍。

MLIR 的性能收益来自它承载的优化，而不是 PTX 这个终点。本仓库另一项同卡实验已
证明这一点：并行 block LayerNorm 为 10.271 µs，融合
`Bias + Residual + LayerNorm` 为 13.604 µs，而未融合尾部为 28.344 µs。后续应把
MatMul 降到 NVGPU/MMA Tensor Core，或在生产模式保留 cuBLAS/cuBLASLt，仅让 MLIR
生成和融合定制 epilogue。

## 复现

Windows + WSL 离线编译，不访问 CUDA：

```powershell
$env:PYTHONPATH = "src"
python tools/benchmark_mlir_ptx_ffn.py `
  --mlir-opt /home/thomas/gac-mlir-build-lf/bin/mlir-opt `
  --wsl --wsl-distribution Ubuntu-24.04 `
  --compile-only `
  --emit-artifact-dir build/mlir-ptx-artifact `
  --output build/mlir-ptx-compile.json
```

在 SM86 GPU 主机加载预编译产物并做配对基准：

```bash
PYTHONPATH=src python tools/benchmark_mlir_ptx_ffn.py \
  --artifact-dir build/mlir-ptx-artifact \
  --warmup 20 --iterations 50 --repeats 20 --public-samples 30 \
  --benchmark-order ptx-first \
  --output build/benchmark-ptx-first.json
```

将 `--benchmark-order` 改为 `hybrid-first` 可复现第二轮。
