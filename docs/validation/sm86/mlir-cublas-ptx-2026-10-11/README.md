# BEVFormer FFN cuBLAS + MLIR PTX 优化记录（2026-10-11）

本目录记录生产形状 FP32 BEVFormer FFN 优化分支的证据边界。被测编译单元固定为：

```text
[1,2500,256]
  → MatMul 512 → Bias + ReLU
  → MatMul 256 → Bias + query residual + LayerNorm
```

它不是完整 BEVFormer 推理，也不包含 CNN、Attention、MSDA、Decoder、Detection Head、输入预处理、输出 D2H 或 CPU NMS-free 后处理。

## 当前状态

`mlir-cublas-ptx-sm86`、`mlir-cuda-sm86` 与 `mlir-tiled-ffn-sm86` 的正式三候选真机验收已经完成；真实输入 MMA 与双缓冲 MatMul 的功能报告也已保存。正式报告绑定 Git 提交 `5794d2c7a54104874d6af93802276263896b8a49`（短 SHA `5794d2c`），记录 `git dirty=false`；报告中的 80 个源码 SHA-256 已与本地同名文件逐一核对，缺失和不匹配均为 0。设备是 **RTX 3070 Ti Laptop GPU、Compute Capability 8.6、8 GB**；它不是桌面 RTX 3070，也不是此前首版基准使用的 RTX 3080。

| 环境项 | 值 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.3 LTS |
| NVIDIA Driver | 580.178.04 |
| CUDA Driver API | 13000 |
| CUDA Runtime | 12.9 / API 12090 |
| CuPy | 13.6.0 |
| Python 环境 | Python 3.12 venv |

该 Python 3.12 venv 的 CUDA Runtime/NVRTC wheel 没有附带可加载的 cuBLAS wheel。验收没有隐藏这个环境修复：`LD_LIBRARY_PATH` 显式加入 Isaac Conda 环境里的 cuBLAS 目录，并加入当前 venv 的 CUDA Runtime/NVRTC wheel 目录：

```bash
export LD_LIBRARY_PATH="/home/robot/miniconda3/envs/isaac/lib/python3.11/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

这只是补齐运行时动态库搜索路径，没有改变被测 Kernel、权重、计时范围或精度策略。

历史数据继续保留为比较基线：RTX 3080 上，首版 `mlir-ptx-sm86` 为 `1.092～1.102 ms/FFN`，同轮旧 cuBLAS hybrid 为 `0.209～0.214 ms/FFN`。这些数字来自[首版全 PTX 验证目录](../mlir-ptx-2026-10-11/README.md)，不能当作本目录新后端或 RTX 3070 Ti Laptop 的结果。

## RTX 3070 Ti Laptop 实测结果

正式协议为 warmup 50、40 个 repeat、每个 repeat 100 次完整 FFN、public samples 30。M2c 与 M2a 在每个 repeat 内交换位置，并分别用 PTX-first 与 NVRTC-first 作为首轮顺序重跑；单缓冲 tiled 完整 Session 使用相同 warmup/iterations/repeats，并与两条基线共同进入 correctness gate 和 autotune。

| 首次测量次序 | M2c `mlir-cublas-ptx-sm86` | M2a `mlir-cuda-sm86` | `mlir-tiled-ffn-sm86` | M2c 位置平衡加速 |
| --- | ---: | ---: | ---: | ---: |
| PTX first | 221.915998 μs | 227.223997 μs | 892.597122 μs | 1.0248387378× |
| NVRTC first | 223.343678 μs | 229.038086 μs | 893.539200 μs | 1.0274777097× |

两轮中三个候选均为 `eligible`，autotuner 都选择 `mlir-cublas-ptx-sm86`。M2c 与 tiled 的 NumPy 对照均为 `allclose=True`，两者最大绝对误差都为 `7.152557e-7`。

公共 `run` 包含输入校验、H2D、四调用计算、D2H 和同步；其中 GPU 与 wall 中位数为：

| 首次测量次序 | 新路径 H2D+计算 GPU | 新路径含 D2H/同步 wall |
| --- | ---: | ---: |
| PTX first | 584.239990 μs | 1233.796007 μs |
| NVRTC first | 575.888008 μs | 1192.150492 μs |

这些数字只属于一个生产形状 FP32 FFN。它们不是完整 BEVFormer 延时，也不能外推为桌面 RTX 3070。RTX 3080 历史基线来自另一台设备，不能与这里的绝对延时直接横比。

本目录中的原始证据可直接审计：

| 文件 | 内容 |
| --- | --- |
| [`benchmark-ptx-first-rtx3070ti-laptop.json`](benchmark-ptx-first-rtx3070ti-laptop.json) | 固定提交 `5794d2c` 的 PTX-first 40×100 正式三候选样本、公共 `run`、正确性、设备、Git 状态和 80 个源码 SHA-256 |
| [`benchmark-nvrtc-first-rtx3070ti-laptop.json`](benchmark-nvrtc-first-rtx3070ti-laptop.json) | 同一固定提交的 NVRTC-first 正式独立重跑 |
| [`autotune-ptx-first-rtx3070ti-laptop.json`](autotune-ptx-first-rtx3070ti-laptop.json) | PTX-first 三个可执行 FFN 候选的正确性门、计时样本和选择结果 |
| [`autotune-nvrtc-first-rtx3070ti-laptop.json`](autotune-nvrtc-first-rtx3070ti-laptop.json) | NVRTC-first 三候选独立 autotune 结果 |
| [`compile-report-llvm24.json`](compile-report-llvm24.json) | M2c 与 tiled 两个无权重 artifact 的 LLVM 24 lowering 命令、工具链 revision、PTX entry 和摘要 |
| [`functional-mma-fp16-tf32-llvm24-sm86.json`](functional-mma-fp16-tf32-llvm24-sm86.json) | FP16/TF32 真实输入功能 MMA 的 LLVM 24 lowering、PTX 指令计数与 SHA-256 |
| [`functional-mma-fp16-rtx3070ti-laptop.json`](functional-mma-fp16-rtx3070ti-laptop.json) | FP16 功能 MMA 的真机输入、参考输出、逐样本计时与误差 |
| [`functional-mma-tf32-rtx3070ti-laptop.json`](functional-mma-tf32-rtx3070ti-laptop.json) | TF32 功能 MMA 的真机输入、参考输出、逐样本计时与误差 |
| [`async-tiled-matmul-llvm24-sm86.json`](async-tiled-matmul-llvm24-sm86.json) | 真实双 stage MatMul 的 LLVM 24 lowering、`cp.async`/commit/wait 与 shared-memory 证据 |
| [`async-tiled-matmul-rtx3070ti-laptop.json`](async-tiled-matmul-rtx3070ti-laptop.json) | 双 stage MatMul 的真机输入、逐元素正确性与逐样本计时 |
| [`async-double-buffer-probe-llvm24-sm86.json`](async-double-buffer-probe-llvm24-sm86.json) | 较早的最小异步调度 probe：2 条 `cp.async`、2 个 commit、`wait_group 1 → 0`；不执行 MatMul |

## 已实现的生产候选

`mlir-cublas-ptx-sm86` 固定执行四个调用：

| 顺序 | 实现 | 计算 |
| ---: | --- | --- |
| 1 | CuPy/cuBLAS | FC1 MatMul |
| 2 | 官方 MLIR→NVVM→PTX | Bias + ReLU |
| 3 | CuPy/cuBLAS | FC2 MatMul |
| 4 | 官方 MLIR→NVVM→PTX | Bias + query residual + LayerNorm |

编译器只接受完整、静态、连续 FP32 生产 FFN。两个融合 entry 固定为 PTX 7.1 / `sm_86` / 64 位裸指针 ABI；Runtime 在 CUDA 初始化前重建 graph 和 call plan，检查 entry、参数数量、`.maxntid 256,1,1`、binary MLIR/PTX 一致性和整体摘要。执行阶段不提交 CUDA C 给 NVRTC，也没有 NumPy、旧 CUDA 或全 PTX fallback。

`src/gac/mlir_cublas_ptx_artifact.py` 可把以下无权重目录原子发布并严格加载：

```text
fusions.gpu.mlir
fusions.binary.mlir
fusions.ptx
manifest.json
```

权重继续来自调用者提供并单独校验的 graph；加载器据此重建完整四调用计划。HXP v1 尚不承载该 artifact。MatMul 当前是 `cupy.matmul(..., out=...)` 的 cuBLAS 路径，不是 cuBLASLt。

在已有生产尺寸 JSON graph 上运行：

```bash
PYTHONPATH=src python -m gac.cli run build/bevformer-production-ffn.json \
  --backend mlir-cublas-ptx-sm86 \
  --mlir-opt /path/to/mlir-opt \
  --verify --warmup 50 --repeat 100 \
  --report build/mlir-cublas-ptx-sm86-run.json
```

`examples/bevformer_tiny_ffn.json` 是小尺寸 tensor/linalg smoke，不满足本后端的生产 shape，会被明确拒绝。

专用工具会自行构造确定性的生产图。先在有 LLVM/MLIR 的主机生成无权重 artifact：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --mlir-opt /path/to/mlir-opt \
  --emit-artifact-dir build/mlir-cublas-ptx-artifact \
  --compile-only \
  --output build/mlir-cublas-ptx-compile.json
```

GPU 主机无需安装 LLVM/MLIR；用相同确定性 graph 加载目录，并分别改变首轮顺序：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order ptx-first \
  --warmup 50 --iterations 100 --repeats 40 --public-samples 30 \
  --autotune --autotune-cache build/mlir-cublas-ptx-autotune-ptx-first.json \
  --output build/mlir-cublas-ptx-ptx-first.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order nvrtc-first \
  --warmup 50 --iterations 100 --repeats 40 --public-samples 30 \
  --autotune --autotune-cache build/mlir-cublas-ptx-autotune-nvrtc-first.json \
  --output build/mlir-cublas-ptx-nvrtc-first.json
```

`--repeats` 必须为偶数。工具在每个 repeat 内继续交换两条路径的位置；`--benchmark-order` 只决定第一轮先测哪条路径。

## tiled、NVGPU/MMA 与异步双 stage 证据

`src/gac/mlir_tiled_matmul.py` 为 FC1/FC2 生成显式 FP32 shared-memory tiled GPU module。16×16×16 配置的本地 LLVM 24/PTX 7.1 指令审计结果为：

| Kernel | block | shared memory | shared 声明 | shared load/store | barrier | vector shared load | `mma.sync` |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| FC1 + Bias + ReLU | `16×16×1` | 2048 B | 2 | 20 / 2 | 2 | 4 | 0 |
| FC2 | `16×16×1` | 2048 B | 2 | 20 / 2 | 2 | 4 | 0 |

这证明显式 workgroup tile 与 LLVM 生成的向量 shared load 已进入 PTX。Kernel 仍是单缓冲、一线程一个标量输出，但现在已由 `src/gac/mlir_tiled_ffn.py`、`mlir_tiled_ffn_runtime.py` 和 `mlir_tiled_ffn_artifact.py` 接成完整的三 dispatch `mlir-tiled-ffn-sm86` Session：FC1+Bias+ReLU、FC2、MLIR PTX Bias+Residual+LayerNorm。

固定 Git 提交 `5794d2c` 在 RTX 3070 Ti Laptop 上使用 warmup 50、每个 repeat 100 次完整 FFN、40 个 repeat、public samples 30 的正式三候选结果如下：

| 首次测量次序 | tiled | M2c cuBLAS+PTX | M2a cuBLAS+NVRTC | autotune |
| --- | ---: | ---: | ---: | --- |
| PTX first | 0.892597122 ms | 0.221915998 ms | 0.227223997 ms | 三者 eligible，选择 M2c |
| NVRTC first | 0.893539200 ms | 0.223343678 ms | 0.229038086 ms | 三者 eligible，选择 M2c |

两份 benchmark JSON 都保存 tiled 的逐次样本、正确性、artifact 摘要、完整 Git 提交和 80 个源码 SHA-256；两份独立 autotune JSON 保存三个候选的全部评估与最终选择。M2c 与 tiled 两轮均为 `allclose=True`，最大绝对误差都为 `7.152557e-7`。

同一 lowering 工具还生成两种**真实输入**的固定尺寸 NVGPU MMA 功能候选：

| 模式 | 计算 | 最终 PTX | global load/store | max abs | launch 中位数 |
| --- | --- | --- | --- | ---: | ---: |
| TF32 | `A[16,16] f32 × B[16,8] f32 → C[16,8] f32` | 4×`mma.sync.m16n8k4` | 12 / 4 | 0 | 3.879600 μs |
| FP16 | `A[16,32] f16 × B[32,8] f16 → C[16,8] f32` | 2×`mma.sync.m16n8k16` | 24 / 4 | 0 | 4.012688 μs |

真机协议为 warmup 50、30 个样本、每个样本 1000 次 launch。两者都加载测试生成的真实矩阵并与 CPU 参考比较；它们仍是固定尺寸功能候选，没有扩展到生产 FC1/FC2 shape、边界处理、占用率验收或完整 FFN。

`src/gac/mlir_async_tiled_matmul.py` 实现了真实输入 FP32 `A[16,64] × B[64,16] → C[16,16]` 双 stage MatMul 功能候选。LLVM 24→SM86 PTX 7.1 和真机结果为：

| `cp.async` | commit group | wait 深度 | barrier | shared memory | 正确性 | launch 中位数 |
| ---: | ---: | --- | ---: | ---: | --- | ---: |
| 8 | 4 | `[1,1,1,0]` | 7 | 4096 B | 256 个输出逐元素 exact，max abs 0 | 2.652160 μs |

真机协议为 warmup 20、30 个样本、每个样本 200 次 launch。它验证了真实双缓冲计算，但仍是固定尺寸功能候选，尚未接入生产 FC1/FC2 或完整 FFN。目录中较早的 [`async-double-buffer-probe-llvm24-sm86.json`](async-double-buffer-probe-llvm24-sm86.json) 继续记录 128 B、2 条 `cp.async`、2 个 commit、`wait_group 1 → 0` 和 PTX SHA-256 `b80c2ee9a03e9f46e1862eeac98661e5427a241bb1ea0c808c4ea29c74e646fd` 的最小调度证据；不要把该旧 probe 与新的功能 MatMul 混为一项性能结果。

复现 lowering：

```bash
PYTHONPATH=src python tools/probe_mlir_tiled_matmul.py \
  --output-dir build/mlir-tiled-probe \
  --mlir-opt /path/to/mlir-opt \
  --which both --tile-m 16 --tile-n 16 --tile-k 16 \
  --probe-functional-mma

PYTHONPATH=src python tools/run_mlir_functional_mma.py \
  --ptx build/mlir-tiled-probe/nvgpu-tf32-functional-matmul.ptx \
  --mode tf32 --warmup 50 --samples 30 --launches-per-sample 1000 \
  --output-json build/run-tf32.json

PYTHONPATH=src python tools/run_mlir_functional_mma.py \
  --ptx build/mlir-tiled-probe/nvgpu-fp16-functional-matmul.ptx \
  --mode fp16 --warmup 50 --samples 30 --launches-per-sample 1000 \
  --output-json build/run-fp16.json

PYTHONPATH=src python tools/probe_mlir_async_tiled_matmul.py \
  --output-dir build/mlir-async-tiled-matmul \
  --mlir-opt /path/to/mlir-opt

PYTHONPATH=src python tools/run_mlir_async_tiled_matmul.py \
  --ptx build/mlir-async-tiled-matmul/async-tiled-matmul.ptx \
  --warmup 20 --samples 30 --launches-per-sample 200 \
  --output-json build/run-async-tiled-matmul.json
```

编译输出目录必须不存在或为空；报告记录 GPU MLIR、binary MLIR、PTX、工具链 revision、指令审计和摘要。

## autotuning 边界

`src/gac/mlir_ptx_autotune.py` 已实现：

- `(backend, tile, block, precision, fusion)` 候选记录和确定性 SHA-256；
- 先验证输出，再允许候选进入计时；
- 按重复样本中位数选优，平局按候选 SHA-256 决定；
- 绑定 graph/source、设备、候选空间和计时配置的完整性缓存；
- 原子发布缓存，并拒绝损坏、重复 JSON key、非有限数和上下文变化。

`src/gac/mlir_cublas_ptx_benchmark.py` 已提供三个真实 CUDA Session adapter。提供基础 artifact 与 tiled artifact 后，`mlir-cublas-ptx-sm86`、`mlir-cuda-sm86` 和 `mlir-tiled-ffn-sm86` 都能进入相同正确性门和 CUDA Event 计时。固定尺寸 MMA 与双缓冲 MatMul 没有完整 FFN Session，不会把微内核延时填进 FFN 候选。

编译基础与 tiled 两个无权重 artifact，再执行三候选基准：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --mlir-opt /path/to/mlir-opt \
  --emit-artifact-dir build/mlir-cublas-ptx-artifact \
  --emit-tiled-artifact-dir build/mlir-tiled-ffn-artifact \
  --tiled-tile 16,16,16 --compile-only \
  --output build/mlir-three-candidate-compile.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --tiled-artifact-dir build/mlir-tiled-ffn-artifact \
  --warmup 50 --iterations 100 --repeats 40 --public-samples 30 \
  --autotune --autotune-cache build/mlir-three-candidate-autotune.json \
  --output build/mlir-three-candidate-benchmark.json
```

本目录正式报告已满足固定 Git commit、clean tree、设备、artifact hash、80 个源码 hash 和逐次原始样本的可追溯要求。

## 真机验收协议

新增路径的有效报告至少需要：

1. 保存设备名、显存、Compute Capability、Driver、CUDA Runtime、CuPy 和工具链 revision；
2. 在同一进程同时建立参与比较的完整 FFN Session；四调用正式报告比较 M2c/M2a，tiled 正式报告还必须加入 `mlir-tiled-ffn-sm86`；
3. 权重和 query 常驻设备，用 CUDA Event 包住候选的完整 call plan（三个或四个 dispatch），排除 H2D/D2H；
4. M2c/M2a 配对按 repeat 交换次序，并分别以新路径 first 和旧路径 first 重跑；tiled 保存相同 warmup/iterations/repeats 下的逐次样本；
5. 另报公共 `run` 的 H2D+计算 GPU 时间、含 D2H/同步的 wall 时间、setup 与首次调用；
6. 对同一 NumPy 参考保存最大绝对/相对误差；
7. 保存 artifact/call-plan/source hash 和逐次原始样本，禁止只保存汇总数字。

本目录的两份正式三候选 benchmark JSON、两份对应 autotune JSON、一份 M2c+tiled LLVM 24 compile report、三份功能 MMA 报告、两份真实异步 MatMul 报告和一份早期异步调度 probe report 是本轮可审计证据；汇总数字均能回溯到逐次样本、artifact/source manifest 与固定提交 `5794d2c`。

## 尚未完成

- cuBLASLt 及其 epilogue；
- 把已通过真实输入功能验证的异步双 stage `cp.async` MatMul 接入生产 FC1/FC2 与完整 FFN，并完成正式数值和性能验收；
- 把固定尺寸真实输入 FP16/TF32 NVGPU/MMA 扩展到生产 FC1/FC2 shape、边界处理和完整 FFN；
- HXP v2 混合执行计划；
- 完整 BEVFormer 的 MLIR 覆盖；
- 桌面 RTX 3070 的延时、显存和稳定性验收。
