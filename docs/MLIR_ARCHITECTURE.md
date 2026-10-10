# GAC MLIR 编译器架构

本文说明 GAC 如何从现有 Python Canonical Graph 逐步演进为基于 MLIR 的 AI 编译器。首个设备目标是 NVIDIA RTX 3070，对应 `cuda-sm86`；`mlir-nvvm` 表示计划中的最终通用代码生成路径。当前已经有标准 tensor/linalg IR emitter，以及一个严格限定 BEVFormer FFN pattern 的 `mlir-cuda-sm86` 可执行 runtime-call 切片。

本文严格区分已经落地的能力和后续路线图。M0/M1 包含 emitter、诊断和外部 verifier 接口；新增的可执行切片把两次 MatMul 显式绑定到 CuPy/cuBLAS，把 `Bias+ReLU` 与 `Bias+Residual+LayerNorm` 绑定到两个 NVRTC Kernel。它不是通用 bufferization/GPU/NVVM lowering，也不是完整 BEVFormer 后端。某次 textual MLIR 产物只有在对应 `mlir-opt` 调用成功后才能称为 verified；某次性能结论必须来自保存了设备、范围、协议和原始样本的真机报告。

## 1. 目标与边界

MLIR 在项目中的职责是承载逐层 lowering、做合法性验证并建立可扩展的 Pass 管线。它不替代模型前端、HXP 包格式、设备 Runtime，也不等同于 CUDA Kernel 实现。

```text
JSON（当前）/ BEVFormer 导出（路线图）
          │
          ▼
GAC Canonical Graph
  shape / dtype / layout / provenance
          │
          ▼
MLIR tensor IR                              当前 M0/M1
  func + tensor + linalg + arith + math
          │
          ├── mlir-opt 解析、验证、规范化
          │
          ├── 严格 FFN pattern matcher       当前可执行窄切片
          │       │
          │       ▼
          │   func + memref + 4 个 runtime calls
          │       ├── 2× CuPy/cuBLAS MatMul
          │       └── 2× NVRTC 融合 Kernel
          │
          ▼
通用 bufferization + memref                 路线图
          │
          ▼
scf/cf → gpu/nvgpu → NVVM                   路线图
          │
          ▼
LLVM IR → PTX → cubin                       路线图
          │
          ▼
HXP v2 Execution Plan → CUDA Runtime        路线图
```

三个名称描述不同维度：

- `cuda-sm86` 是设备 target，约束 NVIDIA Ampere Compute Capability 8.6 的指令集、地址空间、资源上限和产物兼容性；
- `mlir-cuda-sm86` 是当前窄范围的 compiler backend/runtime 组合，只接受一个固定 FFN pattern，并通过显式 runtime call 使用 cuBLAS 和 NVRTC；
- `mlir-nvvm` 是目标 compiler backend，描述计划从 MLIR 逐层降低到 NVVM/LLVM/PTX 的通用编译实现；当前 tensor/linalg emitter 和 FFN runtime-call 切片都没有完成这条路径。

这种拆分允许以后让同一个设备 target 使用不同编译后端，也允许同一个 MLIR 前端接 CPU、其他 GPU 或 NPU 后端。

## 2. 当前实现状态

| 里程碑 | 当前能力 | 可验证证据 | 明确未包含 |
| --- | --- | --- | --- |
| M0：MLIR 接入 | 从 GAC 静态 Tensor 图生成确定性的 textual MLIR；模块使用已注册的 `builtin`、`func`、`tensor`、`linalg`、`arith` 和 `math` 方言 | 单元测试覆盖文本生成、诊断和工具调用封装；产物可交给本地 LLVM 源码构建出的 `mlir-opt` 解析和验证，成功结果须由实际命令记录 | Python MLIR binding、二进制 bytecode、GPU 代码生成 |
| M1：首个算子切片 | 手工 JSON 图中的 BEVFormer-like tiny FFN/MLP smoke；覆盖 MatMul、Bias Add、GELU/ReLU、`AddReLU` 和末轴 LayerNorm，包含静态 shape 和广播约束 | 单元测试检查确定性、类型/shape 拒绝路径和 MLIR 文本；可选集成测试在配置实际 `mlir-opt` 后运行 | 官方 checkpoint importer、真实 BEVFormer 子图导出、完整 Encoder/Decoder、MSDA、GridSample、Conv 等全图算子覆盖 |
| M2a：SM86 FFN runtime-call 切片 | 严格匹配 `MatMul → AddReLU → MatMul → Add → Add(query residual) → LayerNorm`；发出两次 CuPy/cuBLAS MatMul 和两个 NVRTC 融合 Kernel 的显式 call plan | runtime-call MLIR ABI manifest 已由 LLVM 24 `mlir-opt --verify-each` 验证；独立的 Python call records 已在 RTX 3080 SM86 上完成输出正确性测试，最终比较基准要求带 artifact hash 的同进程配对交替测量 | 通用 bufferization、GPU/NVGPU/NVVM/PTX、HXP v1 打包、任意 MLP、完整 BEVFormer |

M0/M1 的价值不是“把 CUDA 字符串换成另一种字符串”。实际执行 `mlir-opt` 时，它会解析 IR 和方言，并检查 SSA 使用、类型、区域、block 参数和算子约束；只有该命令成功，特定产物才算通过 verifier。通用产物仍停在 tensor-level MLIR，因此不能把它称为已经生成 PTX。M2a 的 FFN 切片能在 SM86 上执行，但执行来自显式 CuPy/cuBLAS 与 NVRTC runtime binding，不等于 tensor/linalg 已自动降低为 GPU/NVVM。

BEVFormer 当前完整生产路径仍由已有 CUDA/HXP 后端承担。MLIR 样例首先覆盖 FFN 与 `AddReLU`，原因是这部分语义清楚、静态 shape 稳定，适合先检查 IR 边界、Pass 基础设施和 runtime-call ABI，再处理 BEV 专用算子。CLI 示例仍是手工编写的 tiny JSON 图；生产尺寸 benchmark 使用相同构图器生成一个 FFN 片段。两者都不来自官方 checkpoint importer，也没有证明真实 BEVFormer 模块或完整图已经接入 MLIR。

## 3. 分层设计

### 3.1 前端与 Canonical Graph

现有 CUDA/BEVFormer 路径的模型导入层负责权重映射、输入合同和 cold/warm 状态语义。Canonical Graph 保存 TensorSpec、节点属性、常量和来源标识，是框架无关的模型语义层。当前 MLIR M0/M1 直接接收 JSON 或内存中的 Canonical Graph；官方 checkpoint 到 MLIR 的 importer 和真实 BEVFormer 子图导出尚未实现。

在进入 MLIR 前，编译器必须完成：

- shape、dtype 与广播合法性检查；
- 静态 Profile 选择；完整 BEVFormer 未来需要复用现有固定 Profile 合同；
- 常量和输入的明确区分；
- 从原模型节点到 MLIR operation 的可追溯标识；
- 不支持算子的显式诊断，禁止静默丢节点或改用错误语义。

### 3.2 Tensor-level MLIR

当前 emitter 将一个 GAC Graph 映射为 `module` 和单个 `func.func`。函数参数和返回值使用 ranked tensor type；权重成为常量；矩阵乘使用 `linalg.matmul`；带广播的 Bias Add、激活和融合后的 `AddReLU` 使用 `linalg.generic`、`arith` 与 `math` 标量算子表达。

M0/M1 为了让测试产物完全自包含，把小型 smoke 权重写成 `dense` 常量。完整 BEVFormer 不能用巨大的文本常量承载全部 checkpoint；进入生产包前需要资源化 ElementsAttr 或外部权重 ABI，并继续由 HXP 做摘要、边界和生命周期校验。

这一层保留 Tensor 语义，不提前决定显存 offset、CUDA block 大小或具体库调用。这样规范化、CSE、融合、算子分解和数值检查可以与最终设备代码生成解耦。

### 3.3 `mlir-cuda-sm86` FFN runtime-call 切片

当前可执行切片由 [`src/gac/backends/mlir_cuda.py`](../src/gac/backends/mlir_cuda.py) 与 [`src/gac/mlir_runtime.py`](../src/gac/mlir_runtime.py) 实现。它先调用 tensor-level `compile_mlir(..., fuse=True)`，再检查融合后的图是否**完整且恰好**为：

```text
MatMul → AddReLU → MatMul → Add(fc2 bias) → Add(query residual) → LayerNorm
```

匹配器还要求一个输入、一个输出、静态连续 FP32 Tensor、常驻的两组权重/偏置和 LayerNorm scale/bias、残差必须是原始 query、输出 shape 必须等于 query，且通道数不超过 4096。任何多余节点、缺失节点、顺序变化、非常量参数或 shape 不匹配都会在编译期报错；不尝试拆分子图，也不会转到 NumPy 或旧 generated-CUDA 后端。

匹配成功后，后端生成标准 `func`/`memref` MLIR。四个 private function symbol 定义 runtime ABI，入口函数按固定顺序描述调用关系：

| 顺序 | Runtime symbol | 实现 | 融合边界 |
| ---: | --- | --- | --- |
| 1 | `gac_cublas_fc1_matmul_f32` | `cupy.matmul(..., out=...)` / cuBLAS | FC1 MatMul |
| 2 | `gac_bias_relu_inplace_f32` | CuPy `RawModule` / NVRTC Kernel | FC1 bias + ReLU，原地写 hidden buffer |
| 3 | `gac_cublas_fc2_matmul_f32` | `cupy.matmul(..., out=...)` / cuBLAS | FC2 MatMul |
| 4 | `gac_bias_residual_layer_norm_f32` | CuPy `RawModule` / NVRTC Kernel | FC2 bias + query residual + LayerNorm |

这份 MLIR source 是可审阅、可交给 `mlir-opt` 验证的 ABI manifest，并不由 Runtime 解析或执行。编译器同步生成结构化 Python call records；它们才是实际 dispatch 的唯一来源。`MlirCudaSession` 校验 call record 的 symbol、输入输出、grid/block 与 alias 合同后才运行，不会按 MLIR 文本或 graph op 名称推测调用。

末尾 Kernel 每行启动一个 256-thread block，用 block reduction 计算均值与方差。所有调用使用同一条 CuPy non-blocking stream。Session 建立时把常量驻留到设备，分配输入、输出和两个 FP32 中间 workspace，并通过 NVRTC 编译融合 Kernel。执行失败会使 Session 失败，不会逐调用替换实现。

命令行入口直接读取 JSON graph：

```bash
python -m gac.cli run examples/bevformer_tiny_ffn.json \
  --backend mlir-cuda-sm86 \
  --verify \
  --report build/bevformer_tiny_ffn.mlir-cuda.json
```

运行设备的 Compute Capability 必须恰好为 8.6，并要求 CuPy、CUDA Runtime、NVRTC 和 cuBLAS 可用。HXP v1 没有保存该 runtime-call plan，因而 `run model.hxp --backend mlir-cuda-sm86` 会被明确拒绝。当前 `--trace`/`--profile` 也只服务旧 `cuda-sm86` Session；FFN 切片的 device-resident 测量使用 [`tools/benchmark_mlir_cuda_ffn.py`](../tools/benchmark_mlir_cuda_ffn.py) 在整个四调用计划外放置一对 CUDA Event。

真机基准只覆盖生产尺寸 `[1,2500,256] → 512 → 256` 的单个 FFN 片段。报告必须分别保存 device-resident 四调用 CUDA Event、同步 wall、公共 `run` 的 H2D+计算 GPU 时间、含输出 D2H 的 wall 时间、NumPy 正确性、执行顺序，以及 MLIR manifest、call plan 和 Kernel source 的 hash。比较时在同一进程共驻留两个 Session，每个 repeat 后交换 MLIR/CUDA 的先后位置，并用两种首轮顺序独立重跑，避免把库初始化、热态或动态频率偏差误写成编译器收益。任何具体数字都必须直接来自最终原始 JSON，并且只属于报告中的设备和 FFN 片段，不是完整 BEVFormer、RTX 3070 或通用 NVVM lowering 的结果。证据边界和两份原始报告见[验证说明](validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)。

### 3.4 Pass 管线

Pass 管线按“每一阶段都有合法 IR 集合”的方式设计：

1. **Import cleanup**：canonicalize、CSE、常量规范化和死代码清理；
2. **Domain decomposition**：把组合算子分解或融合为后端支持的模式；
3. **Bufferization**：One-Shot Bufferize，把 Tensor SSA 值转换为显式 memref，并建立 alias/lifetime；
4. **Loop/GPU lowering**：将 `linalg` 降为并行循环，完成 tiling、fusion、vectorization 和 GPU mapping；
5. **NVVM lowering**：把 GPU/NVGPU 与向量操作降到 NVVM/LLVM 方言；
6. **Translation**：导出 LLVM IR，经过 NVPTX 目标生成 PTX，再由 CUDA 工具链生成可加载产物。

通用 lowering 当前只落地第 1 阶段及 M1 所需的 tensor-level 表达。第 2 至第 6 阶段仍属于路线图。M2a FFN 切片通过严格 pattern matcher 直接物化 runtime-call ABI，没有实现通用 One-Shot Bufferize、loop/GPU lowering 或 NVVM translation。每个通用阶段都应在边界运行 verifier，并检查上一层不允许残留的方言，避免把 lowering 缺口推迟到 Runtime 才暴露。

### 3.5 BEV 专用方言

完整 BEVFormer 有两个不能简单伪装成普通逐元素算子的核心操作：

- `bev.ms_deformable_attention`：表达多尺度特征、sampling location、attention weight 和 level metadata 的联合语义；
- `bev.grid_sample_2d`：表达坐标归一化、插值、padding 和边界行为。

它们计划进入独立 `bev` 方言，并定义 operation verifier、shape inference、canonicalization 和 lowering interface。早期可降低到 `linalg`/`scf` 的正确性实现；性能路径再匹配为专用 CUDA Kernel。方言本身只表达语义，不能把某个 CUDA 实现细节写进高层 op。

上述自定义方言尚未实现。当前 M1 不会用未注册 operation 假装完成完整 BEVFormer，也不会用 `allow-unregistered-dialect` 绕过验收。

### 3.6 Runtime 与包格式

编译器产物最终仍由 GAC Runtime 执行。规划中的 HXP v2 会保存多种 artifact、target requirements 和显式 Host Execution Plan，例如：

```text
load constant
allocate device buffers
launch MLIR-generated cubin
call cuBLAS / cuDNN
launch existing BEV CUDA kernel
record dependency/event
publish outputs and next_bev
```

这是一条混合执行路径：MLIR 负责适合通用 lowering 和融合的区域，vendor library 与已验证的专用 Kernel 继续执行更合适的区域。跨区域 tensor ABI、stream 顺序、workspace 和状态提交必须由计划明确描述。

HXP v2 混合执行尚未实现。现有 HXP v1 仍是完整 CUDA 路径的有效包格式和校验边界，但它只描述旧 CUDA backend 的源码、launch metadata 和静态内存计划，不能序列化 M2a 的 MLIR runtime-call plan。M2a 因此由 JSON graph 直接编译并在 `MlirCudaSession` 中运行。

## 4. 为什么保留 cuBLAS、cuDNN、Runtime 与 HXP v1

引入 MLIR 不等于重写所有已验证组件。

- **cuBLAS/cuDNN**：矩阵乘和卷积的高性能实现包含架构相关算法选择、workspace 管理和多年调优。M2a 已把两个 MatMul 显式降低为 CuPy/cuBLAS 调用；后续通用路径仍应通过 library-call lowering 或 HXP 调度复用 vendor library。
- **GAC Runtime**：M2a 的 `MlirCudaSession` 已负责输入绑定、驻留常量、stream、NVRTC module 和错误传播。后续由 MLIR/NVVM 生成的设备程序仍不负责完整 BEVFormer 跨帧 `prev_bev` 的提交/回滚和全图 profile；这些职责继续属于 Runtime。
- **HXP v1**：它已经定义常量、图、CUDA artifact、摘要、目标检查和安全加载合同。保留 v1 可以让 MLIR 研发期间的生产 CUDA 路径继续可运行，并为 v2 迁移提供可比较基线。
- **现有 CUDA BEV Kernel**：MSDA、GridSample、运动补偿等路径已有正确性证据。MLIR 版本必须先达到数值和设备验收门槛，才能逐块替换。

这样的迁移方式把编译器结构变化与模型可用性分开：一个区域完成验证后才进入混合计划，失败时编译阶段明确拒绝，不在运行中静默改变算法。

## 5. 本地 LLVM/MLIR 工具链

本次使用的源码不是系统中未知版本的预编译工具，而是项目目录旁的本地 LLVM checkout：

| 项目 | 值 |
| --- | --- |
| 源码目录 | `C:/Users/Thomas/Desktop/universal_ai_compile/llvm-project` |
| LLVM/MLIR 版本 | `24.0.0git` |
| Git revision | `765392a6b4498f1dc57233f03a31b87462fdef01` |
| 构建环境 | WSL Ubuntu 24.04、CMake 3.28、GCC 13 |
| LLVM targets | `X86;NVPTX` |
| 首批工具 | `mlir-opt`、`mlir-translate`、`llc` |

Windows 工作树里的部分脚本带 CRLF，因此本机验证使用同一 revision 的 LF 源码副本。副本不改变 Git 内容，也不构成新的源码版本。仓库提供了可复现脚本 [`tools/build_mlir_wsl.sh`](../tools/build_mlir_wsl.sh)，在 WSL 中运行：

```bash
cd /mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler
bash tools/build_mlir_wsl.sh
```

脚本构建成功后会提供 MLIR 解析、Pass 和 NVPTX 相关工具；工具存在不代表 M1 已经实现 NVVM lowering。脚本用 Git revision 隔离受管源码和 build cache，并写入 `.gac-build-revision` marker。编译报告记录实际 executable、`mlir-opt --version` banner、source revision（受管 marker 存在时）、命令参数、Pass pipeline、超时和源码/产物 SHA-256，避免只保存一个不可追溯的 `.mlir` 文件。

### 5.1 本次真实验证证据

本次不是只做字符串快照测试。使用上表 revision 构建的 `mlir-opt 24.0.0git` 后，进行了以下真实调用：

- 完整 `examples/bevformer_tiny_ffn.json` 经 `mlir-compile` 生成 MLIR，再执行 `--verify-each --canonicalize --cse`，CLI 报告为 `verified: true`；
- 在 WSL 内运行后端集成测试，完整 tiny FFN 以及 rank-3 MatMul、singleton broadcast、rank-1 LayerNorm、GELU、`MatMulBiasGELU` 和 scalar Add 均通过真实 parser/verifier；
- `mlir-cuda-sm86` 生成的 `func`/`memref` runtime-call ABI manifest 已由同一 LLVM 24 `mlir-opt --verify-each` 解析和验证；
- RTX 3080 SM86 真机测试验证四条 call record 的实际执行与 NumPy 参考结果；生产尺寸 FFN benchmark 工具记录 warm-up、样本分布、device-resident CUDA Event、公共 `run` 边界和同轮 CUDA codegen + cuBLAS 基线，比较报告保存 artifact hash，并在同进程内完成位置平衡的配对交替测量及两种首轮顺序重跑；
- [`examples/mlir_sm86_add_relu.mlir`](../examples/mlir_sm86_add_relu.mlir) 作为独立的手写 GPU Dialect smoke，执行 `gpu-lower-to-nvvm-pipeline`，目标 `cubin-chip=sm_86`、`cubin-format=isa`，成功生成一个 `gpu.binary`；内嵌 PTX 包含 `.target sm_86` 与 `add_relu_broadcast` entry，生成结果再次执行 `--verify-each` 通过。

最后一项只证明 stock MLIR 24 与 LLVM NVPTX 工具链能够把手写 `gpu.module` 降为 SM86 PTX。它没有把 GAC tensor/linalg 图自动接到 GPU/NVVM，也不属于 BEVFormer 性能证据；这段自动 lowering 仍属于通用 M2/M3。

## 6. RTX 3070 / SM86 适配原则

RTX 3070 的首个 target 固定为 `cuda-sm86`。MLIR M0/M1 验证设备无关的 tensor IR；M2a Runtime 会读取设备属性并要求 Compute Capability 恰好等于 8.6。后续通用 GPU lowering 必须把以下内容纳入 target profile：

- Compute Capability 8.6 与 NVPTX feature；
- 支持的 dtype、向量宽度、对齐和连续布局；
- block/thread mapping、shared memory 和 register pressure 限制；
- PTX 版本、CUDA Driver 最低要求和 artifact compatibility；
- FP32 基线，以及以后启用 TF32/FP16/Tensor Core 时独立的精度合同；
- 静态 workspace、常量驻留和跨 Kernel lifetime。

RTX 3080 与 RTX 3070 都属于 SM86，可以验证同一指令目标的兼容性；它们的核心数量、显存和带宽不同，因此 RTX 3080 数据不能作为 RTX 3070 的延迟或峰值显存结论。MLIR 路径目前没有 RTX 3070 性能结果。

## 7. 验收门槛

每个里程碑必须有可重复的证据，不能用“生成了文件”代替编译器验收。

### M0/M1：当前门槛

- 同一输入重复生成字节一致的 MLIR；
- `mlir-opt` 能解析并验证所有已声明支持的示例；
- MLIR 中只出现阶段允许的已注册方言；
- static shape、dtype、MatMul 维度和广播不合法时在编译阶段失败；
- FFN 与 `AddReLU` 的结构测试覆盖普通路径和拒绝路径；
- 不把 textual MLIR 验证结果描述为 GPU 执行或性能提升。

### M2a：当前 FFN runtime-call 切片门槛

- 只接受完整且严格的 BEVFormer FFN pattern，其他图在编译阶段明确失败；
- runtime-call MLIR manifest 只使用当前允许的 `func`/`memref`，并通过实际 `mlir-opt --verify-each`；
- Python call records 与四调用 ABI 一致性在 Session 建立时复核，Runtime 不解析或执行 MLIR source；
- 两次 MatMul 明确调用 CuPy/cuBLAS，两个融合 Kernel 明确由 NVRTC 编译，不允许隐式 fallback；
- 真机报告分别记录 setup、首次调用、device-resident CUDA Event、公共 `run` GPU/wall 和正确性，不外推完整模型或其他 GPU。

### M2b：通用 Bufferization 门槛

- tensor-to-memref 后无未预期 tensor op；
- alias、in-place 决策和 buffer lifetime 可检查；
- 峰值 workspace 有静态报告，并与现有 arena 规划的所有权规则一致；
- 数值结果与 NumPy/CUDA 参考路径在明确容差内一致。

### M3：NVVM/PTX 门槛

- GPU/NVGPU lowering 后不残留非法高层方言；
- `mlir-translate`/LLVM NVPTX 路径可生成 PTX，CUDA assembler/driver 可接受产物；
- artifact 明确记录 `cuda-sm86`、toolchain revision、ABI 和 hash；
- 在真实 RTX 3070 上完成加载、最小输入 smoke test、错误路径和重复运行稳定性验证；
- 性能使用 warm-up、GPU event、wall time 和峰值显存的独立记录，不外推其他 GPU 数据。

### M4：BEVFormer/HXP v2 门槛

- `bev.ms_deformable_attention` 和 `bev.grid_sample_2d` 有 verifier、shape inference、参考实现与边界用例；
- cold/warm raw outputs 及 `next_bev` 与官方 PyTorch/MMCV golden 对齐；
- 混合 plan 明确 MLIR artifact、cuBLAS/cuDNN 和已有 Kernel 的 stream 依赖与 tensor ABI；
- 执行失败不提交错误的下一帧状态；
- 完整图在 RTX 3070 上报告真实延迟、显存和精度，才能宣称 3070 适配完成。

## 8. 路线图

| 阶段 | 主要工作 | 产物 |
| --- | --- | --- |
| M0/M1（emitter 与验证接口已实现） | textual MLIR、`mlir-opt` 调用封装、手工 tiny FFN/MLP JSON 与 `AddReLU` smoke | `.mlir`、诊断，以及实际 verifier 成功后生成的验证报告 |
| M2a（严格 FFN runtime-call 切片已实现） | pattern matcher、四调用 ABI manifest/call records、CuPy/cuBLAS、NVRTC 融合 Kernel、SM86 Session | 可审阅/可验证的 MLIR ABI manifest、内存中 call plan、FFN 真机报告 |
| M2b | linalg canonicalization、One-Shot Bufferize、memref lifetime/arena 对接 | 通用 bufferized MLIR 与内存报告 |
| M3 | tiling/fusion/vectorization、GPU/NVGPU、NVVM/LLVM、NVPTX | `cuda-sm86` PTX/cubin |
| M4 | `bev` 方言、MSDA/GridSample lowering、库调用选择 | BEVFormer 编译区域与混合 artifact |
| M5 | HXP v2 Host Execution Plan、多 artifact 校验、Runtime 混合调度 | 可部署的完整模型包 |

路线图不是完成声明。M2a 的实现不能替代仍未完成的通用 bufferization、GPU/NVVM/PTX、自定义 `bev` 方言和 HXP v2。

## 9. 面试表述参考

可以这样简要介绍：

> 我把现有编译器设计成渐进式 MLIR 架构。M0/M1 能把 JSON/Canonical Graph 中的手工 tiny FFN smoke 生成 `func/tensor/linalg/arith/math` 标准 IR，并提供本地 LLVM 24 `mlir-opt` 的 verifier 调用入口。M2a 只针对一个严格 FFN pattern：生成可审阅、可验证的 `func`/`memref` runtime-call ABI manifest 和同步的 Python call records，实际执行由 call records 驱动两次 CuPy/cuBLAS MatMul 与两个 NVRTC 融合 Kernel；Runtime 不解析或执行 MLIR source。这个切片已在 RTX 3080 SM86 上真机测量，但不是官方 checkpoint importer，也没有覆盖完整 Encoder/Decoder。下一步仍是通用 One-Shot Bufferize、GPU/NVGPU、NVVM 和 PTX。完整 BEVFormer 的 MSDA 和 GridSample 计划进入 `bev` 自定义方言，再降低为通用循环或专用 CUDA Kernel。HXP v1 继续承载现有 CUDA 生产路径，但不能保存 M2a call plan；最终计划用 HXP v2 把 MLIR artifact、vendor 调用和现有 BEV Kernel 编成同一个显式混合执行计划。

被追问“是否已经用了 MLIR”时，准确回答是：已经实现标准 MLIR 文本 emitter、注册方言约束和 `mlir-opt` verifier 集成接口，并实现一个严格 FFN runtime-call 切片。该切片的 MLIR source 是 ABI manifest，Python call records 才是实际调度来源；其 RTX 3080 数字只能归因于这一个 CuPy/cuBLAS+NVRTC FFN 片段。尚未完成 checkpoint importer、通用 NVVM/PTX、HXP v2 和完整 BEVFormer 的 MLIR 执行。

相关背景可继续阅读[总架构说明](architecture.md)、[BEVFormer 编译器设计](bevformer_compiler_design.md)和[SM86 真机验收记录](bevformer_sm86_validation_2026-10-10.md)。
