# General AI Compiler（GAC）

> 面向边缘推理的 AI 编译器与 Runtime 原型；当前生产闭环是固定 Profile 的 BEVFormer-tiny raw step 和 CUDA SM86，同时提供 tensor-level MLIR emitter、cuBLAS 混合 FFN、全 PTX FFN，以及两次 cuBLAS MatMul 配合两个 MLIR→PTX 融合 Kernel 的 `mlir-cublas-ptx-sm86` 生产候选。

GAC 的 CUDA 路径把模型配置、权重和静态计算图转换为可校验的 HXP 包，在 Runtime 中完成设备检查、显存规划、实现选择、NVRTC 编译和 CUDA 执行。逐元素、布局、Attention 与 BEV 专用算子由项目生成 CUDA；生产尺寸 FP32 Conv 和 MatMul 可分别选择 cuDNN 与 CuPy/cuBLAS。MLIR 侧有五条边界不同的路径：`mlir-linalg` 生成标准 Tensor/Linalg MLIR；`mlir-cuda-sm86` 用两次 cuBLAS MatMul 和两个 NVRTC 融合 Kernel 执行固定 FFN；`mlir-ptx-sm86` 把同一生产形状 FFN 的三个 `gpu.func` 经官方 LLVM 24 `mlir-opt` 降到 SM86 PTX；`mlir-cublas-ptx-sm86` 保留两次 CuPy/cuBLAS MatMul，并把两个融合阶段改为 MLIR PTX；`mlir-tiled-ffn-sm86` 用两个单缓冲 tiled MatMul Kernel 和一个最终融合 PTX Kernel 执行三个 dispatch。

```text
JSON graph / BEVFormer config + checkpoint
                    │
                    ▼
      Frontend → Canonical Graph
                    │  shape / dtype / layout validation
                    ▼
   graph passes → fusion → static memory plan
                    │
                    ▼
 Generated CUDA + vendor candidates + metadata
                    │
                    ▼
        versioned, validated HXP package
                    │
                    ▼
 CudaSession → NVRTC / cuDNN / CuPy-cuBLAS → raw outputs
                    └────────────────────────→ trace / profile
```

当前 GPU 路径不会在 CUDA 缺失时静默改用 CPU。HXP 中的 `implementation_candidates` 供 Session 建立时选择；vendor 调用执行中失败不会逐节点自动切回 generated CUDA。NumPy 是显式选择的参考后端。

`mlir-cuda-sm86`、`mlir-ptx-sm86`、`mlir-cublas-ptx-sm86` 和 `mlir-tiled-ffn-sm86` 都不会静默回退到 NumPy 或旧 CUDA 图后端。它们当前直接从 JSON graph 建立计划；HXP v1 不保存这类产物，因此不能把 `.hxp` 作为输入。

## 当前已完成

截至 2026-10-11，GAC 已跑通单 GPU 的模型导入、图优化、静态内存规划、CUDA 代码生成、vendor library 选择、HXP 打包和真机执行闭环。MLIR M0/M1 已完成 tensor IR emitter 和外部 verifier；生产形状 BEVFormer FFN 已完成首版 GPU Dialect→NVVM→PTX 的真机正确性与性能基线。本轮又完成 `mlir-cublas-ptx-sm86`、单缓冲三 dispatch 的 `mlir-tiled-ffn-sm86`、真实输入 FP16/TF32 MMA、真实 `cp.async` 双缓冲 MatMul 功能候选，以及三候选 autotuning；三个完整 FFN Session 已在固定提交 `5794d2c` 上完成正式双顺序报告。完整 BEVFormer 仍未全部进入 MLIR 路径。

| 层次 | 已完成能力 |
| --- | --- |
| 编译器核心 | Canonical Graph、Shape/DType 推导、静态连续布局检查、常量折叠、死代码消除、MatMul/Add/GELU 融合、安全 Add+ReLU 融合 |
| MLIR M0/M1 | 确定性 textual MLIR、标准注册方言、外部 `mlir-opt` verifier 接口；覆盖手工构造的 BEVFormer-like tiny FFN JSON smoke 所需的 MatMul、广播 Add、ReLU、LayerNorm，以及 GELU、Identity 和融合 AddReLU/MatMulBiasGELU |
| MLIR CUDA FFN 切片 | `mlir-cuda-sm86` 严格匹配一个静态 FP32 BEVFormer FFN：两次 CuPy/cuBLAS MatMul、`Bias+ReLU` 融合 Kernel、`Bias+Residual+LayerNorm` 融合 Kernel；生成显式 runtime-call plan 并在 Compute Capability 8.6 上执行 |
| MLIR PTX FFN 切片 | `mlir-ptx-sm86` 严格匹配生产形状 `[1,2500,256]→512→256` FP32 FFN；生成三个 GPU Dialect kernel，调用官方 `mlir-opt` 降到 NVVM/PTX 7.1，校验产物后由 Runtime 直接加载 PTX；没有 cuBLAS/NVRTC fallback |
| MLIR cuBLAS+PTX 生产候选 | `mlir-cublas-ptx-sm86` 使用固定四调用计划：cuBLAS FC1、MLIR PTX `Bias+ReLU`、cuBLAS FC2、MLIR PTX `Bias+Residual+LayerNorm`；不提交 CUDA C 给 NVRTC；RTX 3070 Ti Laptop 正式 device-resident 中位数为 `221.915998～223.343678 μs/FFN` |
| MLIR tiled FFN 功能候选 | `mlir-tiled-ffn-sm86` 用单缓冲 16×16×16 FP32 tiled FC1+Bias+ReLU、tiled FC2 和 MLIR PTX 最终融合组成三个 dispatch；已有完整 Session、无权重 artifact 和三候选 autotune；RTX 3070 Ti Laptop 正式结果为 `0.892597122～0.893539200 ms/FFN` |
| MLIR Kernel 优化基础设施 | FP16 `16×8×32` 与 TF32 `16×8×16` 真实输入 MMA 已在 RTX 3070 Ti Laptop 功能运行；独立 `16×16×64` FP32 `cp.async` 双缓冲 MatMul 候选也已 exact 运行；两者均未接入生产 FFN。autotuner 已实现正确性门、median 选择和绑定设备/图/源码/候选空间的原子缓存 |
| CUDA SM86 后端 | 基础 Tensor/MLP、MHA、CNN、MSDeformableAttention、BEV geometry、Spatial/Temporal Attention、运动补偿及 step 边界所需 Cast/Where；不支持的类型、属性和布局会明确拒绝 |
| HXP 与 Runtime | 确定性打包、摘要校验、安全加载、源码与 launch metadata 复核、静态 arena、驻留权重、单 stream 执行、状态提交/回滚、Trace 和可选逐 launch CUDA Event Profile |
| BEVFormer-tiny | 官方 checkpoint 严格映射；ResNet-50 + 单层 FPN；feature/input preparation；三层 Encoder；六层 Decoder；六层 raw detection projection；cold/history image-to-BEV；统一 cold/warm 五输入三输出 step |
| 时序状态 | 历史 BEV rotate/shift、`has_prev_bev` 选择、scene/CAN-bus delta 管理；只有完整调用和三个输出校验成功后才提交 `next_bev` |
| 主机后处理 | 显式 CPU `decode_nms_free`、Top-K 300、box denormalization 与范围过滤；该阶段位于 HXP/CUDA 图外 |

冻结的 BEVFormer-tiny Profile 为 6 相机、`480×800` 图像、`50×50` BEV、256 embedding。统一 step 输入为 `images`、`lidar2img`、`can_bus_delta`、`prev_bev`、`has_prev_bev`，输出最后一层 `cls_logits`、raw `bbox_regression` 和 FP16 `next_bev`。公开图像与状态存储边界使用 FP16，Transformer/MSDA 等主要计算仍以 FP32 为基线。

## MLIR / `mlir-opt` 文件在哪里？

**`mlir-opt` 不是本仓库里的 Python 文件或动态库。** 它是 LLVM/MLIR 源码编译出来的外部命令行程序，体积、平台和 LLVM revision 都与构建环境有关，因此不提交到 General-AI-Compiler Git 仓库。GAC 通过子进程调用它，把 GPU Dialect MLIR 降到 NVVM 和 SM86 PTX；部署已经离线编译好的 PTX artifact 时不再需要 `mlir-opt`。

GitHub 上完整的 BEVFormer MLIR/PTX 实现位于 [`codex/bevformer-mlir-ptx-opt`](https://github.com/luogantt/General-AI-Compiler/tree/codex/bevformer-mlir-ptx-opt) 分支；默认 `main` 当前没有下面这些工具链、lowering 和验证产物，因此直接浏览 `main` 会找不到。查看代码时请先切换到该分支。

- [`tools/build_mlir_wsl.sh`](tools/build_mlir_wsl.sh) 从项目外部的 `llvm-project` checkout 构建 `mlir-opt`、`mlir-translate` 和 `llc`。默认稳定路径是 WSL 中的 `$HOME/.cache/gac-mlir/current-build/bin/mlir-opt`。
- 本次 BEVFormer/UniAD 验证实际使用的可执行文件是 WSL 中的 `/home/thomas/gac-mlir-build-lf/bin/mlir-opt`；它是本机编译产物，不在 Git 中。对应的外部 LLVM 源码入口是 `C:\Users\Thomas\Desktop\universal_ai_compile\llvm-project\mlir\tools\mlir-opt\mlir-opt.cpp`，同样位于 General-AI-Compiler 仓库之外。
- [`src/gac/mlir_toolchain.py`](src/gac/mlir_toolchain.py) 负责查找 `mlir-opt`、构造无 shell 的命令参数、执行官方 GPU→NVVM pipeline、提取 PTX，并记录 LLVM revision、版本、命令和摘要。
- [`src/gac/backends/mlir_ptx.py`](src/gac/backends/mlir_ptx.py) 是 BEVFormer 固定 FFN 的全 PTX lowering；[`src/gac/backends/mlir_cublas_ptx.py`](src/gac/backends/mlir_cublas_ptx.py) 保留两个 cuBLAS MatMul，并把两个融合 Kernel 由 MLIR 降到 PTX。
- Git 中可直接查看的 MLIR 示例是 [`examples/mlir_sm86_add_relu.mlir`](examples/mlir_sm86_add_relu.mlir) 和 [`examples/mlir_sm86_layernorm_fusion.mlir`](examples/mlir_sm86_layernorm_fusion.mlir)。保存的真实 lowering 样例位于 [`docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.gpu.mlir`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.gpu.mlir)、[`production-ffn.binary.mlir`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.binary.mlir) 和 [`production-ffn.ptx`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.ptx)。
- 普通编译生成的 `.mlir`、`.ptx`、manifest 和报告写入 `build/`；该目录不进入 Git。它们是编译产物，`src/gac/` 下的 Python 文件才是 emitter、lowering、artifact 校验和 Runtime 实现。

在 Windows PowerShell 中构建并取得实际路径：

```powershell
wsl -d Ubuntu-24.04 -- bash tools/build_mlir_wsl.sh
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
```

## MLIR 编译路径（M0/M1）

MLIR 后端复用 Canonical Graph 数据结构和图 Pass；当前接入的是普通 JSON graph。`examples/bevformer_tiny_ffn.json` 是一个手工构造的小尺寸 BEVFormer-like FFN smoke，用于检查 FFN 结构、广播和 LayerNorm lowering。它不读取官方 BEVFormer 代码或 checkpoint，也不是完整 BEVFormer importer。当前链路如下：

```text
JSON graph / hand-written BEVFormer-like tiny FFN smoke
                    │
                    ▼
             Canonical Graph
                    │  shape / dtype / broadcast validation
                    ▼
 func + tensor + linalg + arith + math
                    │
                    ▼
     mlir-opt --verify-each --canonicalize --cse
                    │
                    ▼
 textual .mlir（外部 verifier 成功后才标记 verified）
```

通用 `mlir-linalg` 产物把 `compiler_backend` 记录为已经实际使用的后端，把 `cuda-sm86` 记录为 `target_intent`，并把 `mlir-nvvm` 记录为计划的 codegen。它仍停在 tensor 层，`executable` 为 false。新增 `mlir-ptx-sm86` 则是一个独立、严格限定生产 FFN pattern 的可执行路径；它不是通用 Tensor/Linalg 自动 GPU 映射，也没有覆盖完整 BEVFormer。

Windows + WSL 的可复现工具链构建脚本是 [`tools/build_mlir_wsl.sh`](tools/build_mlir_wsl.sh)。它从项目旁的本地 `llvm-project` checkout 创建 LF 源码快照，并构建 `mlir-opt`、`mlir-translate` 和 `llc`：

```bash
wsl -d Ubuntu-24.04 -- bash tools/build_mlir_wsl.sh
```

从 Windows 调用 WSL 内构建出的 `mlir-opt` 验证上述 tiny FFN smoke：

```powershell
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
python -m gac.cli mlir-compile examples/bevformer_tiny_ffn.json `
  -o build/bevformer_tiny_ffn.mlir `
  --wsl `
  --wsl-distribution Ubuntu-24.04 `
  --mlir-opt $mlirOpt `
  --report build/bevformer_tiny_ffn.mlir.json
```

CLI 默认必须调用 `mlir-opt`；只有显式指定 `--emit-only` 才会跳过外部验证，并在报告中记录 `verified: false`。验证失败时不会写出目标 `.mlir` 文件。若命令在 WSL 内执行，可省略 `--wsl` 和 `--wsl-distribution`，把 `--mlir-opt` 直接指向 `$HOME/.cache/gac-mlir/current-build/bin/mlir-opt`。

本次已用本地 LLVM revision `765392a6b4498f1dc57233f03a31b87462fdef01` 构建的 `mlir-opt 24.0.0git` 真实执行上述命令，完整 tiny FFN 返回 `verified: true`。报告记录实际 executable、版本 banner、命令参数、Pass、超时和源码/产物摘要；使用仓库构建脚本生成的工具时，还会从受管 build marker 记录 LLVM source revision。

[`examples/mlir_sm86_add_relu.mlir`](examples/mlir_sm86_add_relu.mlir) 是独立的手写 GPU Dialect 工具链 smoke。以下命令已真实生成带 `#nvvm.target<chip = "sm_86">` 的 `gpu.binary`，其中内嵌 PTX 包含 `.target sm_86` 和 `add_relu_broadcast` entry：

```bash
"$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  examples/mlir_sm86_add_relu.mlir \
  "--gpu-lower-to-nvvm-pipeline=kernel-bare-ptr-calling-convention=1 cubin-chip=sm_86 cubin-format=isa allow-pattern-rollback=0" \
  -o build/mlir_sm86_add_relu.binary.mlir
```

这个 smoke 验证本地 stock MLIR/LLVM NVPTX 工具链能从手写 `gpu.module` 生成 SM86 PTX。`mlir-ptx-sm86` 已把 GAC Canonical Graph 中严格匹配的生产 FFN 连接到同一官方 GPU/NVVM 管线；通用图仍停在 tensor/linalg。

分层设计、里程碑和验收门槛见 [MLIR 编译器架构](docs/MLIR_ARCHITECTURE.md)。

## `mlir-cuda-sm86` 可执行 FFN 切片

`mlir-cuda-sm86` 是首个 MLIR-derived 可执行 runtime-call 切片，不是通用 MLIR GPU 后端。编译器先复用 tensor-level MLIR 的规范图和融合 Pass，然后只接受融合后的下列完整顺序：

```text
MatMul → AddReLU → MatMul → Add(bias) → Add(query residual) → LayerNorm
   │         │          │                      │
   ▼         ▼          ▼                      ▼
CuPy/     NVRTC      CuPy/          NVRTC Bias+Residual+
cuBLAS  Bias+ReLU    cuBLAS              LayerNorm
```

两次 MatMul 都是显式 `cupy.matmul(..., out=...)`，由 CuPy 调用 cuBLAS。第一个生成 Kernel 把 FC1 bias 与 ReLU 原地融合；第二个生成 Kernel 把 FC2 bias、原始 query residual 和末轴 LayerNorm 融合，并为每一行使用一个 CUDA block 做归约。MLIR source 记录相同的四调用 ABI，供审阅和 `mlir-opt` 验证；执行顺序来自编译器同步生成并由 Session 复核的四条 Python call record。Runtime 不解析 MLIR source，也不会按 op 名称猜测实现。

当前匹配器要求一个输入、一个输出、静态连续 FP32 Tensor、常驻的两组权重/偏置和 LayerNorm scale/bias、query identity residual，以及通道数不超过 4096。任何额外、缺失、重排或不匹配的节点都会在编译阶段被拒绝；它没有覆盖 GELU FFN、任意 MLP、完整 Encoder/Decoder、MSDA、CNN 或状态执行。

安装 CUDA 依赖后，可直接从 JSON graph 运行 tiny FFN smoke：

```bash
python -m pip install -e '.[cuda12]'
python -m gac.cli run examples/bevformer_tiny_ffn.json \
  --backend mlir-cuda-sm86 \
  --verify \
  --warmup 3 \
  --repeat 10 \
  --report build/bevformer_tiny_ffn.mlir-cuda.json
```

该路径要求 NVIDIA Compute Capability **恰好为 8.6**，并要求 CuPy、CUDA Runtime、NVRTC 和 cuBLAS 可用。两个融合 Kernel 在 Session 建立时通过 CuPy `RawModule`/NVRTC 编译；设备缺失、架构不匹配、cuBLAS/NVRTC 失败或 call plan 不匹配都会直接报错。`--trace` 和 `--profile` 仍只适用于 `cuda-sm86`；要测量这个 FFN 切片，请使用专用工具：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cuda_ffn.py \
  --warmup 50 \
  --iterations 100 \
  --repeats 20 \
  --public-samples 50 \
  --output build/mlir-cuda-cublas-ffn.json
```

基准工具只测一个生产尺寸 FFN 片段：`[1,2500,256] → 512 → 256 + ReLU + query residual + LayerNorm`，并把 device-resident 四调用 CUDA Event、公共 `run` 的 H2D+计算 GPU 时间、含 D2H/同步的 wall 时间、正确性、调用顺序和 artifact hash 分开记录。同进程共驻留的两条路径在每个 repeat 后交换先后位置，并分别用 MLIR-first 与 CUDA-first 作为首轮顺序重跑，以抵消初始化、热态和动态频率偏差。只有最终 JSON 中保存的真实设备样本可以作为数字来源，且不能外推为完整 BEVFormer、RTX 3070 或通用 MLIR/NVVM lowering；证据边界和两份原始报告见[真机验证说明](docs/validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)。

RTX 3080 配对交替实测中，同一进程内的两条路径按 repeat 轮换先后位置。以 MLIR-first / CUDA-first 作为首轮顺序的两份独立报告中，这个四调用切片的 device-resident 中位数分别为 **202.906 μs** 和 **209.848 μs**；同轮 CUDA codegen + cuBLAS 对照为 **220.313 μs** 和 **226.811 μs**。按两个测量位置平衡后的加速比分别为 **1.0858×** 和 **1.0816×**，两轮最大绝对误差均为 `9.5367431640625e-7`。这些数字只代表一个 FFN 子图。

仓库还提供一个独立的 [MLIR GPU LayerNorm 融合参考](examples/mlir_sm86_layernorm_fusion.mlir)，它真实经过 GPU Dialect → NVVM → SM86 PTX：block reduction LayerNorm 为 **10.271 μs**，相对 **174.121 μs** 的串行 MLIR 实现快 **16.95×**；Bias + Residual + LayerNorm 融合为 **13.604 μs**，相对 **28.344 μs** 的三个未融合 dispatch 快 **2.08×**。该 PTX 微基准尚未接入 `MlirCudaSession`，产品路径仍执行语义等价的 NVRTC Kernel；完整边界见[微基准说明](docs/validation/sm86/mlir-layernorm-fusion-2026-10-10/README.md)。

HXP v1 只携带旧 CUDA 后端的源码、launch metadata 和静态内存计划，不包含上述 MLIR runtime-call plan。因此下面的命令会被明确拒绝：

```bash
python -m gac.cli run model.hxp --backend mlir-cuda-sm86
```

`mlir-cuda-sm86` 仍是混合 runtime-call 路径，不能作为 MLIR→PTX 完成声明。

## `mlir-ptx-sm86` 官方 MLIR→PTX FFN 路径

`mlir-ptx-sm86` 复用相同的严格 FFN pattern matcher，但三个计算阶段都由 MLIR
GPU Dialect 表达并经官方 `mlir-opt` 降到 NVVM 和 PTX。当前 pattern matcher 与
GPU 调度器是 GAC 的 Python 编译 Pass：它直接根据 Canonical Graph 的类型和拓扑
合同物化显式 `gpu.func`；Tensor/Linalg 分支只复用相同合同，不是 GPU emitter 的
输入，也并非让 stock `mlir-opt` 自动猜测 Tensor/Linalg 的 GPU 调度。

```text
Canonical Graph
  ├→ Tensor/Linalg MLIR（独立审阅/验证产物）
  └→ strict FFN matcher + GAC GPU schedule
       → 3× gpu.func → NVVM → SM86 textual PTX 7.1
       → CuPy RawModule(path=PTX) → CUDA Driver JIT to SASS
       → 3× PTX kernel
```

Runtime 在访问 CUDA 前重建规范图和 launch plan，检查 GPU MLIR/source hash、唯一
`gpu.binary`、`.target sm_86`、`.address_size 64`、精确 entry 集合和整体 artifact
摘要。执行阶段没有 cuBLAS、NVRTC 或 CPU fallback。离线编译和 GPU 主机加载使用
同一 `MlirPtxArtifact`，GPU 主机不需要安装 LLVM/MLIR。部署目录只保存 `gpu.mlir`、
`binary.mlir`、`kernels.ptx` 和 `manifest.json`；tensor-level MLIR 会内联权重，因此不进入
部署目录。PTX 是中间设备代码，CUDA Driver 在模块加载时仍会 JIT 为 SASS。

用模型 JSON 离线编译一个可校验的生产尺寸 FFN 产物。输出目录必须尚不存在或为空，
报告文件必须位于该目录外：

```powershell
$env:PYTHONPATH = "src"
python -m gac.cli mlir-ptx-compile model.json `
  -o build/mlir-ptx-artifact `
  --mlir-opt /home/thomas/gac-mlir-build-lf/bin/mlir-opt `
  --wsl --wsl-distribution Ubuntu-24.04 `
  --report build/mlir-ptx-compile.json
```

在 GPU 主机上用同一生产 FFN JSON 提供权重，并加载离线目录：

```bash
PYTHONPATH=src python -m gac.cli run model.json \
  --backend mlir-ptx-sm86 \
  --mlir-ptx-artifact build/mlir-ptx-artifact
```

`manifest.json` 的摘要用于发现损坏和模型/launch-plan 不匹配，不是发布者签名；只加载由
可信离线编译环境生成或由外部可信摘要确认的 PTX 目录。

RTX 3080 配对交替实测中，全 PTX 路径为 **1.092～1.102 ms/FFN**，cuBLAS
混合路径为 **0.209～0.214 ms/FFN**；当前全 PTX 正确性基线慢 **5.150～5.265×**，
最大绝对误差为 `7.153e-7`。原因是首版两个 MatMul 采用每线程串行 K 循环，尚未
使用 shared-memory tiling、向量化或 Tensor Core。MLIR→PTX 本身不自动提高速度；
性能来自 scheduling、融合和目标专用 lowering。完整编译产物、原始报告和复现命令见
[MLIR→PTX 真机验证](docs/validation/sm86/mlir-ptx-2026-10-11/README.md)。

## `mlir-cublas-ptx-sm86` 四调用生产候选

`mlir-cublas-ptx-sm86` 面向同一个固定 FP32 生产 FFN，并复用严格 pattern matcher。它把成熟矩阵乘和项目可控融合边界组合成以下执行计划：

```text
FC1 MatMul                 CuPy/cuBLAS
Bias + ReLU               MLIR GPU Dialect → NVVM → PTX
FC2 MatMul                 CuPy/cuBLAS
Bias + residual + LayerNorm  MLIR GPU Dialect → NVVM → PTX
```

两个融合 entry 使用官方 LLVM 24 pipeline 生成 PTX 7.1；`MlirCublasPtxSession` 在 CUDA 初始化前重建并校验规范计划，再用 `RawModule(path=PTX)` 加载。`mlir_cublas_ptx_artifact.py` 可把 GPU MLIR、binary MLIR、PTX 和 manifest 原子发布为无权重目录；加载时用另行校验的 graph 常量重建完整四调用计划。这个 Runtime 不向 NVRTC 提交 CUDA C，也没有 NumPy、旧 CUDA 或全 PTX fallback。MatMul 当前是 `cupy.matmul(..., out=...)` 的 cuBLAS 路径，尚未接入 cuBLASLt epilogue。

在已有生产尺寸 JSON graph 上可直接编译并运行：

```bash
PYTHONPATH=src python -m gac.cli run build/bevformer-production-ffn.json \
  --backend mlir-cublas-ptx-sm86 \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  --verify --warmup 50 --repeat 100 \
  --report build/mlir-cublas-ptx-sm86.json
```

输入 JSON 必须恰好满足 `[1,2500,256] → 512 → 256`、FP32、连续布局及 query residual 合同；小尺寸 `examples/bevformer_tiny_ffn.json` 会被明确拒绝。

RTX 3070 Ti Laptop（SM86、8 GB、Driver 580.178.04、CUDA Runtime 12.9、CuPy 13.6.0）上，两轮各采用 warmup 50、40 个 repeat、每个 repeat 100 次完整 FFN，并在每轮内交换 M2c/M2a 测量顺序。PTX-first 报告中，M2c、M2a 与 tiled 的 device-resident 中位数分别为 **221.915998 / 227.223997 / 892.597122 μs**，M2c 位置平衡加速为 **1.0248387378×**；NVRTC-first 报告中分别为 **223.343678 / 229.038086 / 893.539200 μs**，加速为 **1.0274777097×**。两轮三个候选均为 `eligible`，autotuner 都选择 `mlir-cublas-ptx-sm86`；M2c 与 tiled 均为 `allclose=True`，最大绝对误差都为 `7.152557e-7`。

M2c 公共 `run` 的 H2D+计算 GPU 中位数为 **584.239990 / 575.888008 μs**，含 D2H 与同步的 wall 中位数为 **1233.796007 / 1192.150492 μs**（PTX-first / NVRTC-first）。正式报告绑定 clean Git 提交 `5794d2c`，并保存了已与本地逐一匹配的 80 个源码 SHA-256。这些数字只属于单个生产形状 FFN，不是完整 BEVFormer。上一版 RTX 3080 cuBLAS hybrid 的 `0.209～0.214 ms/FFN` 是另一台设备上的历史基线，不能与这台 Laptop GPU 的绝对延时直接横比。

专用工具会构造确定性的生产图、支持无权重 artifact，并保存逐次原始样本。两次运行只改变首轮顺序：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --mlir-opt /path/to/mlir-opt \
  --emit-artifact-dir build/mlir-cublas-ptx-artifact \
  --compile-only --output build/mlir-cublas-ptx-compile.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order ptx-first --warmup 50 --iterations 100 --repeats 40 \
  --public-samples 30 --autotune \
  --autotune-cache build/mlir-cublas-ptx-autotune-ptx-first.json \
  --output build/mlir-cublas-ptx-ptx-first.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order nvrtc-first --warmup 50 --iterations 100 --repeats 40 \
  --public-samples 30 --autotune \
  --autotune-cache build/mlir-cublas-ptx-autotune-nvrtc-first.json \
  --output build/mlir-cublas-ptx-nvrtc-first.json
```

## tiled、NVGPU/MMA、异步双 stage 与 autotuning

[`src/gac/mlir_tiled_ffn.py`](src/gac/mlir_tiled_ffn.py) 已把单缓冲 16×16×16 FP32 shared-memory tile 接成完整生产形状 FFN：FC1+Bias+ReLU、FC2、Bias+Residual+LayerNorm 三次 PTX dispatch，执行不经过 NVRTC。[`MlirTiledFfnSession`](src/gac/mlir_tiled_ffn_runtime.py) 与无权重 artifact 已接到 benchmark/autotune。RTX 3070 Ti Laptop 上使用 warmup 50、每个 repeat 100 次、40 个 repeat、public samples 30 的固定提交 `5794d2c` 正式结果如下：

| 完整 FFN 候选 | PTX-first | NVRTC-first | 正确性/选择状态 |
| --- | ---: | ---: | --- |
| `mlir-tiled-ffn-sm86` | 0.892597122 ms | 0.893539200 ms | `allclose=True`，max abs `7.152557e-7`，eligible |
| `mlir-cublas-ptx-sm86` | 0.221915998 ms | 0.223343678 ms | `allclose=True`，max abs `7.152557e-7`，eligible，autotuner 选择 |
| `mlir-cuda-sm86` | 0.227223997 ms | 0.229038086 ms | eligible |

[`src/gac/mlir_tiled_matmul.py`](src/gac/mlir_tiled_matmul.py) 还提供真实输入 Tensor Core 功能候选。TF32 路径计算 `A[16,16] × B[16,8] → C[16,8]`，PTX 含 4 条 `mma.sync.m16n8k4`、12 次 global load 和 4 次 store；FP16 路径计算 `A[16,32] × B[32,8] → C_f32[16,8]`，PTX 含 2 条 `mma.sync.m16n8k16`、24 次 global load 和 4 次 store。两者在 RTX 3070 Ti Laptop 上均为 `max_abs_error=0`，每次 launch 的 CUDA Event 中位数分别为 **3.8796 μs** 和 **4.0127 μs**。它们仍是固定尺寸功能候选，没有接入生产 FC1/FC2 或完整 FFN。

[`src/gac/mlir_async_tiled_matmul.py`](src/gac/mlir_async_tiled_matmul.py) 实现了真实输入 FP32 `16×16×64` 双缓冲 MatMul。LLVM 24→SM86 PTX 7.1 包含 8 条 `cp.async`、4 个 commit、`wait_group [1,1,1,0]` 和 4096 B shared memory；RTX 3070 Ti Laptop 上 256 个输出逐元素 exact，`max_abs_error=0`，每次 launch 中位数 **2.652160 μs**。这个固定尺寸候选尚未接入生产 FFN。

功能候选可以独立复现：

```bash
PYTHONPATH=src python tools/probe_mlir_tiled_matmul.py \
  --output-dir build/mlir-tiled-probe \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  --which fc1 --tile-m 16 --tile-n 16 --tile-k 16 \
  --probe-functional-mma

PYTHONPATH=src python tools/probe_mlir_async_tiled_matmul.py \
  --output-dir build/mlir-async-tiled-matmul \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
```

[`src/gac/mlir_ptx_autotune.py`](src/gac/mlir_ptx_autotune.py) 定义候选的 backend、`(M,N,K)` tile、CUDA block、FP32/TF32/FP16 精度和融合策略。它先对每个候选做输出正确性检查，再计时并按重复样本的中位数选择赢家；缓存绑定 graph/source SHA-256、设备、候选空间和计时配置，并用原子替换发布。[`src/gac/mlir_cublas_ptx_benchmark.py`](src/gac/mlir_cublas_ptx_benchmark.py) 现在可以让 `mlir-cublas-ptx-sm86`、`mlir-cuda-sm86` 和一个 `mlir-tiled-ffn-sm86` Session 同轮进入 CUDA Event 计时。真实输入 MMA 与 `cp.async` MatMul 仍未接到完整 FFN autotune。生产 Tensor Core FFN、双缓冲生产 FFN、cuBLASLt 和更多 tile 候选仍未完成。

本轮优化边界、真机结果和未接线候选见 [cuBLAS+PTX 优化验证说明](docs/validation/sm86/mlir-cublas-ptx-2026-10-11/README.md)。

## RTX 3070 / SM86 适配边界

首个目标是桌面 RTX 3070 对应的 `cuda-sm86`。最终全图验收实际使用 **RTX 3080 10 GB**；本轮 cuBLAS+MLIR PTX 单 FFN 使用 **RTX 3070 Ti Laptop 8 GB**。这些 GPU 与桌面 RTX 3070 的 Compute Capability 都是 8.6，因此结果验证了相同 SM86 编译目标下对应切片的代码生成、加载和 Runtime 契约。

这不等于桌面 RTX 3070 性能验收。不同卡的计算资源、带宽、显存容量和功耗不同，本文所有 BEVFormer 全图耗时仍属于 RTX 3080；RTX 3070 Ti Laptop 新数据只覆盖一个 FFN。桌面 RTX 3070 的全图延迟、实际峰值显存和稳定性尚未实测。

## RTX 3080 最终验收

验收环境为 Ubuntu 22.04、RTX 3080、CUDA SM86；GAC 使用 Python 3.11.17、CuPy 13.6.0、CUDA Runtime/NVRTC 12.9。官方 `bevformer_tiny_epoch_24.pth` 在独立 PyTorch/MMCV 参考环境中生成 golden。

下表是固定 Profile raw step 的稳态中位数：全生成路径各测 3 次，优化路径各测 10 次。GPU stream 与调用 wall time 均不包含 Session 构建、完整数据预处理或 CPU NMS-free 后处理。

| 路径 | 全生成 GPU / wall | 零拷贝 + FP32 cuDNN/cuBLAS GPU / wall | GPU / wall 加速比 |
| --- | ---: | ---: | ---: |
| cold | 425.325073 / 436.881591 ms | 57.624464 / 69.403093 ms | 7.38098× / 6.29484× |
| warm | 426.032532 / 437.718436 ms | 56.823263 / 68.747440 ms | 7.49750× / 6.36705× |

输出相对官方 PyTorch/MMCV golden 的最大绝对误差为：

| 路径 | `cls_logits` | `bbox_regression` | `next_bev` |
| --- | ---: | ---: | ---: |
| cold | 0.003891945 | 0.004856288 | 0.001953125 |
| warm | 0.000953674 | 0.000725389 | 0.001953125 |

其他最终验收数据：

- 独立进程完成 Session 构建后的首个 cold `run`：`277.443573 ms GPU / 289.668472 ms wall`。同一进程基准里的 `116.635651 ms` 发生在 Session 已初始化后，不是 fresh-process 首次 `run`。
- 带逐 launch CUDA Event 的独立 Profile 会扰动执行：该轮为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 Kernel Event 求和 `63.4391 ms`；其中 cuDNN 55 次 / `36.6409 ms`、CuPy/cuBLAS 118 次 / `8.0060 ms`、generated CUDA 530 次 / `18.7923 ms`。这些值用于归因，不与上表普通基准直接比较。
- 强制 CUDA 的完整测试：`Ran 287 tests in 18.881s`，`OK (skipped=1)`。
- 最终 HXP：`137,896,143` bytes，SHA-256 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`。
- 图优化规模：融合前 876 节点，39 组 Add+ReLU 融合后 837 个执行节点，零拷贝后 703 次设备 dispatch；内部 arena `552,960,000` bytes。
- Tensor 计划为 `705,177,100` bytes；加上 cuDNN 8 MiB 后，`713,565,708` bytes 是 Tensor + workspace 的计划预算/准入值，并非设备进程峰值，不包含 CUDA module/context、Event 和 vendor 临时分配。

显式 128 MiB cuDNN workspace 实验把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，并比当前 8 MiB budget 多占 120 MiB，因此本轮没有集成。下一步优先推进 CUDA Graph、批量 dispatch 和更多融合；当前 `cupyx.cudnn` 与 capture 不兼容，cuDNN 图捕获仍需要稳定的低层 API。

## 快速运行

离线编译与 NumPy 参考路径需要 Python 3.11+。Linux 或 WSL2 上的 CUDA 路径还需要 NVIDIA 驱动和项目的 `cuda12` 可选依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cuda12]'

.venv/bin/python -m gac.cli doctor
.venv/bin/python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
.venv/bin/python -m gac.cli inspect build/mlp-sm86.hxp
.venv/bin/python -m gac.cli run build/mlp-sm86.hxp \
  --verify \
  --trace build/trace.json \
  --profile build/profile.json \
  --report build/report.json
```

`--profile` 会额外执行一次带逐 launch Event 的调用；默认运行路径不创建这些 Event。`compile` 也可输出 `.cu` 和 `.plan.json` 调试产物。当前 HXP 保存规范图、常量、CUDA 源码、launch metadata 和静态内存计划，不是预编译 cubin；它与当前 codegen 绑定，优化前的 schema 1 包需要重建。

没有 NVIDIA GPU 时，可显式使用 NumPy 参考后端检查语义：

```bash
python -m pip install -e .
python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
python -m gac.cli inspect build/mlp-sm86.hxp
python -m gac.cli run build/mlp-sm86.hxp --backend numpy-cpu --verify
```

Windows 将解释器路径换成 `.\.venv\Scripts\python.exe`。原生 Windows 的离线编译与 CPU 路径已检查，CUDA 路径尚未实测。

BEVFormer 的合同检查、权重检查和子图/HXP 构建由 `gac-bevformer` 提供：

```bash
gac-bevformer --help
gac-bevformer step-pipeline --help
```

完整的官方 checkpoint 环境、golden 生成和远程复现步骤见 [BEVFormer 参考环境](docs/bevformer_reference_environment.md)与[远程验收记录](docs/bevformer_sm86_validation_2026-10-10.md)。

## UniAD 2.0 adapter

GAC 现在包含一个自包含的 UniAD 2.0 `stage2_e2e` BEV encoder FFN adapter。它从可信 checkpoint 精确映射第 0～5 层的两组 Linear、bias 和 post-FFN LayerNorm 参数，通过带 SHA-256 的 data-only bundle 跨越官方 UniAD Python 3.9/CUDA 11.8 环境与 GAC Python 3.11/CUDA 12 环境。bundle 加载禁用 pickle，并严格检查成员、摘要、shape、dtype、连续布局和有限值。

**这里编译的是一个 UniAD `stage2` BEV encoder layer 的 FFN，不是完整 UniAD。** 静态 FP32 Profile 是 `[1,40000,256] → 512 → 256`。attention、tracking、mapping、motion、occupancy、planning 和其余层仍在这个编译单元之外，因此单个 FFN 的延迟不能换算成完整 UniAD 帧率。

这个 FFN 目前有四条可核验路径：

- `cuda-hxp`：六 dispatch 基线，两个 MatMul 在 Runtime 中选择 CuPy/cuBLAS；
- `mlir-cuda-sm86`：旧四调用路径，两个 cuBLAS MatMul 加两个运行时 NVRTC 融合 Kernel；
- `mlir-cublas-ptx-sm86`：两个 cuBLAS MatMul 加两个由 `mlir-opt` 离线生成的融合 PTX Kernel，部署执行不需要 `mlir-opt` 或 NVRTC；
- `mlir-tiled-ffn-sm86`：FC1+Bias+ReLU、FC2、最终 Bias+Residual+LayerNorm 三个离线 PTX dispatch，部署执行同样不需要 `mlir-opt` 或 NVRTC。

`--backend both` 为兼容旧流程，只生成 HXP 和 `mlir-cuda-sm86`；`--backend all` 才会额外生成两种离线 PTX artifact。没有 checkpoint 时可生成生产 shape 的确定性 synthetic bundle。它只用于编译、正确性和性能链路验证，不包含真实模型权重，也不能用于完整 UniAD 精度或端到端结论：

```bash
python tools/uniad_make_synthetic_bundle.py build/uniad/layer0.uniadffn
```

在 Windows PowerShell 中调用 WSL 的 `mlir-opt`，一次生成全部四种产物：

```powershell
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
python tools/uniad_compile_ffn.py `
  build/uniad/layer0.uniadffn build/uniad/layer0-compiled `
  --backend all --tile 16,16,16 `
  --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt
```

若命令本身在 WSL/Linux 内运行，直接传 Linux 路径：

```bash
python3 tools/uniad_compile_ffn.py \
  build/uniad/layer0.uniadffn build/uniad/layer0-compiled \
  --backend all --tile 16,16,16 \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
```

`build/uniad/layer0-compiled/mlir-cublas-ptx-sm86/` 和 `mlir-tiled-ffn-sm86/` 是无内联权重、带完整性 manifest 的离线代码目录。部署机加载这些目录时不运行 `mlir-opt`，也不把 CUDA C 交给 NVRTC；CUDA Driver 仍会在加载 textual PTX 时完成 PTX→SASS JIT，cuBLAS+PTX 路径还需要 CuPy/cuBLAS。

如果只需要两种新 PTX artifact，也可以让正式 benchmark 工具执行纯离线编译；`--compile-only` 不访问 CUDA：

```powershell
python tools/benchmark_mlir_cublas_ptx_ffn.py `
  --uniad-bundle build/uniad/layer0.uniadffn --seed 20261011 `
  --emit-artifact-dir build/uniad/layer0-compiled/mlir-cublas-ptx-sm86 `
  --emit-tiled-artifact-dir build/uniad/layer0-compiled/mlir-tiled-ffn-sm86 `
  --compile-only --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt `
  --output build/uniad/layer0-compile.json
```

官方 UniAD 环境中的 checkpoint 导出使用同一 checkout wrapper，不需要把 Python 3.11 的完整 GAC 环境安装进去：

```bash
python tools/uniad_export_ffn.py \
  /path/to/UniAD-2.0/ckpts/uniad_base_e2e.pth \
  build/uniad/layer0.uniadffn --layer 0 --trusted-checkpoint \
  --with-synthetic-golden
```

导出器优先使用 Torch 的 `weights_only=True` 数据加载路径。`--trusted-checkpoint` 只允许不支持该参数的旧版 Torch 回退到 pickle 反序列化，因此只应对来源可信的本地 checkpoint 使用。

在 SM86 GPU 上，使用同一个 UniAD bundle 和已编译 artifact 做三候选配对测量与 autotune。`--repeats` 必须为偶数；resident GPU 时间与包含 H2D、D2H 和同步的 public 时间会分开报告：

```bash
python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --uniad-bundle build/uniad/layer0.uniadffn \
  --seed 20261011 \
  --artifact-dir build/uniad/layer0-compiled/mlir-cublas-ptx-sm86 \
  --tiled-artifact-dir build/uniad/layer0-compiled/mlir-tiled-ffn-sm86 \
  --device 0 --warmup 10 --iterations 30 --repeats 6 \
  --public-samples 3 --benchmark-order ptx-first \
  --autotune --autotune-cache build/uniad/layer0-autotune.json \
  --output build/uniad/layer0-benchmark.json
```

正式复现还要启动第二个独立进程，把 `--benchmark-order` 改为 `nvrtc-first`，并为该进程使用不同的 `--autotune-cache` 和 `--output` 文件。

`tools/uniad_benchmark_ffn.py`（安装入口 `gac-uniad-benchmark-ffn`）保留为旧 `cuda-hxp` 与 `mlir-cuda-sm86`/NVRTC 的基线对比；上面的 `benchmark_mlir_cublas_ptx_ffn.py --uniad-bundle` 才测量新的离线 PTX 和 tiled 候选。

最终 v3 在 RTX 3070 Ti Laptop（SM86）上使用真实 `uniad_base_e2e` checkpoint 的 layer-0 权重、seed `20261011` 的确定性 synthetic query 和 GAC NumPy reference。两个独立进程分别改变首次后端顺序，正式协议为 warmup 10、6 个 repeat、每个 repeat 30 次 FFN，以及 3 个 public `run()` 样本：

| 初始顺序 | cuBLAS+MLIR PTX resident | cuBLAS+NVRTC resident | tiled PTX resident | cuBLAS+MLIR PTX public wall | cuBLAS+NVRTC public wall | resident 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTX-first | 3.050785 ms | 3.150221 ms | 13.932249 ms | 20.710336 ms | 20.564843 ms | 1.035303× |
| NVRTC-first | 3.076282 ms | 3.182030 ms | 13.936361 ms | 21.537857 ms | 21.664099 ms | 1.032900× |

两份报告的 resident primary estimator 几何平均为 **1.034101×，即约 3.41% 优势**。public wall 包含参数检查、H2D、计算、D2H 和同步，两种顺序下 PTX 与 NVRTC 相差都不到 `0.15 ms`，方向还会随初始顺序变化，因此只能表述为基本持平。三个候选均通过 `rtol=1e-4`、`atol=1e-5` 的 `allclose`；两次 autotune 都选择 `mlir-cublas-ptx-sm86`。完整原始样本、源码摘要和 artifact manifest 见 [UniAD 单层 FFN 最终 MLIR/PTX 真机证据](docs/validation/sm86/uniad/mlir-ptx-2026-10-11/README.md)。

这只是一个 `[1,40000,256] → 512 → 256` FFN。bundle 不含 expected sample，真实的是 checkpoint 权重；输入仍是 synthetic query，正确性参考由 NumPy 计算。结果不代表完整 UniAD 的真实数据集精度、端到端延迟或 FPS。

迁移前历史基线是在 RTX 3080 上用合成权重和合成输入测得：旧四调用 MLIR/NVRTC 为 `1.635～1.645 ms`，同轮 CUDA/HXP + cuBLAS 为 `2.074～2.075 ms`。完整 UniAD native PyTorch 模型调用的历史均值为 `822.885 ms`。设备、权重和实现边界均不同，不能把这些历史数字与 v3 直接比较，也不能解释为 MLIR 整网加速。完整 native PyTorch 基线通过必选的 `--uniad-root` 连接外部 UniAD checkout；详细环境与边界见 [UniAD 编译器说明](docs/uniad_compiler.md)。

统一测试入口：

```bash
python -m unittest discover -s tests -v
GAC_REQUIRE_CUDA=1 python -m unittest discover -s tests -v
```

`GAC_REQUIRE_CUDA=1` 用于真机验收；CUDA 不可用时会失败，不会把 GPU 测试静默算作通过。

## 项目结构

```text
src/gac/ir.py, ops.py, passes.py     Canonical Graph、算子语义与图优化
src/gac/backends/cuda.py             SM86 支持检查、CUDA 生成与 launch metadata
src/gac/backends/mlir.py             Canonical Graph 到标准 tensor/linalg MLIR
src/gac/backends/mlir_cuda.py        严格 FFN pattern 到显式 cuBLAS/NVRTC call plan
src/gac/backends/mlir_ptx.py         严格生产 FFN 到 GPU Dialect/NVVM/PTX artifact
src/gac/backends/mlir_cublas_ptx.py  2×cuBLAS + 2×MLIR PTX 的严格四调用计划
src/gac/mlir_toolchain.py            mlir-opt 发现、WSL 调用、验证和诊断
src/gac/mlir_runtime.py              SM86 FFN call plan 的 CuPy/cuBLAS/NVRTC 执行
src/gac/mlir_ptx_runtime.py          校验并加载 MLIR 生成 PTX 的 SM86 Runtime
src/gac/mlir_cublas_ptx_runtime.py   校验并执行 cuBLAS+MLIR PTX 四调用计划
src/gac/mlir_cublas_ptx_artifact.py  无权重四调用 PTX 目录的原子写入与校验加载
src/gac/mlir_ptx_artifact.py         无内联权重的离线 PTX 目录写入、校验与加载
src/gac/mlir_tiled_matmul.py         shared-memory tiled PTX 与真实输入 FP16/TF32 MMA 功能候选
src/gac/mlir_async_tiled_matmul.py   真实输入 cp.async 双缓冲 MatMul 功能候选
src/gac/mlir_tiled_ffn.py            单缓冲 tiled FC1/FC2 的三 dispatch FFN 计划
src/gac/mlir_tiled_ffn_runtime.py    校验并执行三 dispatch tiled FFN
src/gac/mlir_tiled_ffn_artifact.py   tiled FFN 无权重目录的原子写入与校验加载
src/gac/mlir_ptx_autotune.py         正确性门、median 选择与完整性缓存
src/gac/mlir_cublas_ptx_benchmark.py 三个真实 FFN Session 的 CUDA Event adapter
src/gac/memory.py                    生命周期分析与静态 arena 规划
src/gac/package.py                   HXP 写入、摘要校验、安全加载与 codegen 复核
src/gac/runtime.py                   CudaSession、vendor 选择、传输、启动与计时
src/gac/profiler.py                  Trace 与 CUDA Event Profile 聚合
src/gac/frontends/                   JSON、BEVFormer 合同与 Backbone 导入
src/gac/uniad/                       UniAD FFN 合同、bundle 与 checkpoint 映射
src/gac/bevformer*.py                BEVFormer 算子、Encoder/Decoder/raw head 与状态 Runtime
src/gac/tools/build_bevformer.py     权重检查及 BEVFormer 子图/HXP 构建入口
src/gac/tools/uniad_*.py             UniAD 导出、编译、FFN 与端到端基准入口
src/gac/cli.py                       doctor / compile / mlir-compile / inspect / run
examples/                            MLP、MLIR FFN smoke、MSDA 示例与冻结合同
tools/build_mlir_wsl.sh              从本地 llvm-project 构建最小 MLIR 工具集
tools/benchmark_mlir_cuda_ffn.py     生产尺寸 FFN 切片的真机基准与正确性报告
tools/benchmark_mlir_cublas_ptx_ffn.py cuBLAS+MLIR PTX 的配对基准、artifact 与 autotune 报告
tools/probe_mlir_tiled_matmul.py     tiled MatMul 与 FP16/TF32 MMA lowering probe
tools/run_mlir_functional_mma.py    真实输入 MMA 的正确性与 CUDA Event 测量
tools/probe_mlir_async_tiled_matmul.py cp.async 双缓冲 MatMul lowering probe
tools/run_mlir_async_tiled_matmul.py 双缓冲 MatMul 的正确性与 CUDA Event 测量
tests/                               编译器、Runtime、HXP、BEVFormer 与物理 CUDA 测试
docs/                                架构、设计、后端约定、路线图和验收证据
```

更完整的实现映射见[架构说明](docs/architecture.md)，MLIR 分层与里程碑见 [MLIR 编译器架构](docs/MLIR_ARCHITECTURE.md)，BEVFormer 图与状态语义见[编译器设计](docs/bevformer_compiler_design.md)，后端扩展约定见[后端编写指南](docs/backend_authoring.md)，后续计划见[路线图](docs/roadmap.md)。

## 已知边界

- 当前支持声明限定为固定 Profile 的 BEVFormer-tiny **raw step**。CPU `decode_nms_free` 是公开的图外后处理边界，不包含在上述 GPU/wall 数字中。
- 正确性输入是冻结的合成、已归一化并 padding 的六相机 tensor 与固定几何输入；尚未完成真实 nuScenes 数据集的 mAP/NDS 验收。
- 完整 BEVFormer 只在 RTX 3080 SM86 上验收；桌面 RTX 3070 的延迟、峰值显存和稳定性仍待实测。
- 通用 `mlir-linalg` 路径仍只生成 tensor-level textual MLIR，并提供外部 verifier 接口。`mlir-ptx-sm86` 已完成一个生产形状 FFN 的 GPU/NVVM/PTX 闭环；`mlir-cublas-ptx-sm86` 只完成固定四调用生产候选。两者都不是通用 bufferization/GPU mapping，也没有完整 BEVFormer 或 RTX 3070 性能数据。
- MLIR 的 BEVFormer 相关输入仍是手写的 tiny FFN JSON smoke，真机 benchmark 使用相同构图器生成生产尺寸 FFN；尚无官方 checkpoint 导入、真实 BEVFormer 子图导出或 checkpoint-to-MLIR 路径。
- 完整 BEVFormer 尚未由 MLIR lowering 覆盖；MSDeformableAttention、GridSample、CNN 和状态执行仍使用现有 CUDA/HXP 路径。
- UniAD 编译覆盖当前只包含一个 stage2 BEV encoder FFN；tracking、mapping、motion、occupancy、planning 和 attention 尚未接入 HXP/MLIR。完整 UniAD 报告是外部 checkout 的 native PyTorch 基线，不是编译后整网数据。
- HXP v1 不携带 MLIR runtime-call plan 或 PTX artifact；`mlir-ptx-sm86` 可通过独立的已校验目录和 `--mlir-ptx-artifact` 加载，尚未并入 HXP。
- M0/M1 把小型 smoke 权重写成内联 `dense` 常量；完整模型需要资源化常量或外部权重 ABI，不能直接沿用这种文本承载方式。
- 只覆盖首条路径所需的静态 Shape 和连续布局；主要计算仍为 FP32。FP16/TF32 `mma.sync` 与双缓冲 `cp.async` 已有真实输入固定尺寸功能候选，但尚未扩展为生产 FC1/FC2。单缓冲 tiled 完整 FFN 已在固定提交 `5794d2c` 上完成三候选正式 autotune；生产 Tensor Core FFN、双缓冲生产 FFN 和 cuBLASLt 仍未完成。
- 当前使用单 CUDA stream，调用同步返回；设备侧跨帧状态驻留、多 stream、I/O 预绑定、CUDA Graph、批量 dispatch 和 DLPack 仍待实现。
- `713,565,708` bytes 只是已计划 Tensor 与 cuDNN workspace 的预算/准入值，不能当作 `nvidia-smi` 进程峰值。
- 当前 HXP 与本版 codegen 绑定，尚无预编译 cubin、多 target variant、签名认证或通用异构 Host Execution Plan。
- 通用后端注册、代价驱动划分、CPU/GPU 自动回退、跨设备 alloc/copy/event/free 计划和 NPU 后端尚未实现。
- Session 建立时选择 cuDNN、CuPy/cuBLAS 或 generated CUDA；vendor 调用执行中失败会报错，不会逐节点自动换实现。

## 验收证据

最终数字与历史阶段日志的边界见[优化验收证据索引](docs/validation/sm86/bevformer/optimization-2026-10-10/README.md)。主要原始文件：

- [最终 cold/warm 基准与精度](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-benchmark-final.json)
- [fresh-process 首次 cold run](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-first-frame-final.json)
- [最终 Profile 摘要](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final-summary.json)与[逐 launch Profile](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final.json)
- [最终强制 CUDA 测试日志](docs/validation/sm86/bevformer/optimization-2026-10-10/unittest-full-final.txt)
- [最终 HXP 构建日志](docs/validation/sm86/bevformer/optimization-2026-10-10/build-production-step-final.txt)、[大小](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.size.txt)与[SHA-256](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.sha256.txt)
- [128 MiB cuDNN workspace 实验](docs/validation/sm86/bevformer/optimization-2026-10-10/cudnn-explicit-workspace-benchmark.json)
- [`mlir-cuda-sm86` 生产尺寸 FFN 真机验证与配对交替原始报告](docs/validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)
- [`mlir-ptx-sm86` 官方 MLIR→NVVM→PTX 产物与真机配对报告](docs/validation/sm86/mlir-ptx-2026-10-11/README.md)
- [`mlir-cublas-ptx-sm86` 真机结果及 tiled/MMA/异步双 stage/autotune 的实现边界](docs/validation/sm86/mlir-cublas-ptx-2026-10-11/README.md)
- [UniAD 单层 FFN 最终 MLIR/PTX 真机证据](docs/validation/sm86/uniad/mlir-ptx-2026-10-11/README.md)
- [UniAD 单层 FFN 迁移前真机历史证据](docs/validation/sm86/uniad/ffn-2026-10-11/README.md)
- [UniAD native PyTorch 端到端历史基线](docs/validation/sm86/uniad/e2e-native-nuscenes-mini-2026-10-11/README.md)
- [完整远程验收说明](docs/bevformer_sm86_validation_2026-10-10.md)

## 参考项目与许可证

本项目的架构研究参考 IREE、Apache TVM、ONNX Runtime、TileLang 和 OpenXLA/XLA，但当前实现为本原型独立编写，没有从这些项目复制源码。未来若引入外部源码、头文件、生成物或二进制依赖，需要保留适用的版权、SPDX、`NOTICE` 和第三方许可证，并单独核对 CUDA、cuDNN、厂商 SDK 与模型权重的分发条款。
