# GAC MLIR 编译器架构

本文说明 GAC 如何从现有 Python Canonical Graph 逐步演进为基于 MLIR 的 AI 编译器。首个设备目标是 NVIDIA RTX 3070，对应 `cuda-sm86`；`mlir-nvvm` 表示计划中的最终代码生成路径，当前实现仍停在标准 tensor/linalg IR。

本文严格区分已经落地的能力和后续路线图。当前 MLIR 路径包含 M0/M1 emitter、诊断和外部 verifier 接口，还不是完整 BEVFormer 的 MLIR/NVVM 可执行后端，也没有新的 MLIR 性能数据。某次产物只有在对应 `mlir-opt` 调用成功后才能称为 verified。

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
MLIR tensor IR                         当前 M0/M1
  func + tensor + linalg + arith + math
          │
          ├── mlir-opt 解析、验证、规范化
          ▼
bufferization + memref                 路线图
          │
          ▼
scf/cf → gpu/nvgpu → NVVM              路线图
          │
          ▼
LLVM IR → PTX → cubin                  路线图
          │
          ▼
HXP v2 Execution Plan → CUDA Runtime   路线图
```

两个名称描述不同维度：

- `cuda-sm86` 是设备 target，约束 NVIDIA Ampere Compute Capability 8.6 的指令集、地址空间、资源上限和产物兼容性；
- `mlir-nvvm` 是目标 compiler backend，描述计划从 MLIR 逐层降低到 NVVM/LLVM/PTX 的编译实现；当前 tensor/linalg emitter 还没有完成这条路径。

这种拆分允许以后让同一个设备 target 使用不同编译后端，也允许同一个 MLIR 前端接 CPU、其他 GPU 或 NPU 后端。

## 2. 当前实现状态

| 里程碑 | 当前能力 | 可验证证据 | 明确未包含 |
| --- | --- | --- | --- |
| M0：MLIR 接入 | 从 GAC 静态 Tensor 图生成确定性的 textual MLIR；模块使用已注册的 `builtin`、`func`、`tensor`、`linalg`、`arith` 和 `math` 方言 | 单元测试覆盖文本生成、诊断和工具调用封装；产物可交给本地 LLVM 源码构建出的 `mlir-opt` 解析和验证，成功结果须由实际命令记录 | Python MLIR binding、二进制 bytecode、GPU 代码生成 |
| M1：首个算子切片 | 手工 JSON 图中的 BEVFormer-like tiny FFN/MLP smoke；覆盖 MatMul、Bias Add、GELU/ReLU、`AddReLU` 和末轴 LayerNorm，包含静态 shape 和广播约束 | 单元测试检查确定性、类型/shape 拒绝路径和 MLIR 文本；可选集成测试在配置实际 `mlir-opt` 后运行 | 官方 checkpoint importer、真实 BEVFormer 子图导出、完整 Encoder/Decoder、MSDA、GridSample、Conv 等全图算子覆盖 |

M0/M1 的价值不是“把 CUDA 字符串换成另一种字符串”。实际执行 `mlir-opt` 时，它会解析 IR 和方言，并检查 SSA 使用、类型、区域、block 参数和算子约束；只有该命令成功，特定产物才算通过 verifier。当前产物仍停在 tensor-level MLIR，因此不能把它称为已经生成 PTX 或已经在 RTX 3070 上执行。

BEVFormer 当前完整生产路径仍由已有 CUDA 后端承担。MLIR 样例首先覆盖 FFN 与 `AddReLU`，原因是这部分语义清楚、静态 shape 稳定，适合先检查 IR 边界和 Pass 基础设施，再处理 BEV 专用算子。这里的样例是手工编写的 tiny JSON 图，不来自官方 checkpoint，也没有证明真实 BEVFormer 模块已经接入 MLIR。

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

### 3.3 Pass 管线

Pass 管线按“每一阶段都有合法 IR 集合”的方式设计：

1. **Import cleanup**：canonicalize、CSE、常量规范化和死代码清理；
2. **Domain decomposition**：把组合算子分解或融合为后端支持的模式；
3. **Bufferization**：One-Shot Bufferize，把 Tensor SSA 值转换为显式 memref，并建立 alias/lifetime；
4. **Loop/GPU lowering**：将 `linalg` 降为并行循环，完成 tiling、fusion、vectorization 和 GPU mapping；
5. **NVVM lowering**：把 GPU/NVGPU 与向量操作降到 NVVM/LLVM 方言；
6. **Translation**：导出 LLVM IR，经过 NVPTX 目标生成 PTX，再由 CUDA 工具链生成可加载产物。

当前只落地第 1 阶段及 M1 所需的 tensor-level 表达。第 2 至第 6 阶段属于路线图。每个阶段都应在边界运行 verifier，并检查上一层不允许残留的方言，避免把 lowering 缺口推迟到 Runtime 才暴露。

### 3.4 BEV 专用方言

完整 BEVFormer 有两个不能简单伪装成普通逐元素算子的核心操作：

- `bev.ms_deformable_attention`：表达多尺度特征、sampling location、attention weight 和 level metadata 的联合语义；
- `bev.grid_sample_2d`：表达坐标归一化、插值、padding 和边界行为。

它们计划进入独立 `bev` 方言，并定义 operation verifier、shape inference、canonicalization 和 lowering interface。早期可降低到 `linalg`/`scf` 的正确性实现；性能路径再匹配为专用 CUDA Kernel。方言本身只表达语义，不能把某个 CUDA 实现细节写进高层 op。

上述自定义方言尚未实现。当前 M1 不会用未注册 operation 假装完成完整 BEVFormer，也不会用 `allow-unregistered-dialect` 绕过验收。

### 3.5 Runtime 与包格式

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

HXP v2 混合执行尚未实现。现有 HXP v1 仍是完整 CUDA 路径的有效包格式和校验边界。

## 4. 为什么保留 cuBLAS、cuDNN、Runtime 与 HXP v1

引入 MLIR 不等于重写所有已验证组件。

- **cuBLAS/cuDNN**：矩阵乘和卷积的高性能实现包含架构相关算法选择、workspace 管理和多年调优。当前 GAC 的真机结果也证明 vendor library 对生产 shape 有实际价值。MLIR 后续应通过 library-call lowering 或 HXP 调度复用它们，而不是先用朴素 Kernel 造成性能倒退。
- **GAC Runtime**：后续由 MLIR/NVVM 生成的设备程序仍不负责 BEVFormer 跨帧 `prev_bev` 的提交/回滚、输入绑定、stream、事件、错误传播和 profile。已有 Runtime 继续承担这些职责。
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
- [`examples/mlir_sm86_add_relu.mlir`](../examples/mlir_sm86_add_relu.mlir) 作为独立的手写 GPU Dialect smoke，执行 `gpu-lower-to-nvvm-pipeline`，目标 `cubin-chip=sm_86`、`cubin-format=isa`，成功生成一个 `gpu.binary`；内嵌 PTX 包含 `.target sm_86` 与 `add_relu_broadcast` entry，生成结果再次执行 `--verify-each` 通过。

第三项只证明 stock MLIR 24 与 LLVM NVPTX 工具链能够把手写 `gpu.module` 降为 SM86 PTX。它没有把 GAC tensor/linalg 图自动接到 GPU/NVVM，也不属于 BEVFormer 性能证据；这段自动 lowering 仍属于 M2/M3。

## 6. RTX 3070 / SM86 适配原则

RTX 3070 的首个 target 固定为 `cuda-sm86`。当前 MLIR M1 只验证设备无关的 tensor IR；后续 GPU lowering 必须把以下内容纳入 target profile：

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

### M2：Bufferization 门槛

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
| M2 | linalg canonicalization、One-Shot Bufferize、memref lifetime/arena 对接 | bufferized MLIR 与内存报告 |
| M3 | tiling/fusion/vectorization、GPU/NVGPU、NVVM/LLVM、NVPTX | `cuda-sm86` PTX/cubin |
| M4 | `bev` 方言、MSDA/GridSample lowering、库调用选择 | BEVFormer 编译区域与混合 artifact |
| M5 | HXP v2 Host Execution Plan、多 artifact 校验、Runtime 混合调度 | 可部署的完整模型包 |

路线图不是完成声明。尤其是 bufferization、GPU/NVVM/PTX、自定义 `bev` 方言和 HXP v2 目前都没有进入已完成范围。

## 9. 面试表述参考

可以这样简要介绍：

> 我把现有编译器设计成渐进式 MLIR 架构。现阶段 MLIR 后端能把 JSON/Canonical Graph 中的手工 tiny FFN smoke 生成 `func/tensor/linalg/arith/math` 标准 IR，并提供本地 LLVM 24 `mlir-opt` 的 verifier 调用入口；每次验证是否成功以命令结果和报告为准。这个 smoke 不是官方 BEVFormer checkpoint importer，也没有覆盖完整 Encoder/Decoder。下一步才是 One-Shot Bufferize、GPU/NVGPU、NVVM 和 PTX。设备 target 与最终代码生成路径分开建模：RTX 3070 是 `cuda-sm86`，计划中的后端是 `mlir-nvvm`。完整 BEVFormer 的 MSDA 和 GridSample 计划进入 `bev` 自定义方言，再降低为通用循环或专用 CUDA Kernel。Runtime、cuBLAS、cuDNN 和 HXP v1 继续保留，因为 MLIR 负责生成和优化程序，不替代状态管理、设备执行、vendor library 调优和包安全合同。最终计划用 HXP v2 把 MLIR 产物、vendor 调用和现有 BEV Kernel 编成同一个显式混合执行计划。

被追问“是否已经用了 MLIR”时，准确回答是：已经实现标准 MLIR 文本 emitter、注册方言约束和 `mlir-opt` verifier 集成接口；当前 BEVFormer 相关覆盖只是手工 tiny FFN JSON smoke。只有实际 `mlir-opt` 成功的产物才称为已验证；尚未完成 checkpoint importer、NVVM/PTX 和完整 BEVFormer 的 MLIR 执行，因此不能把现有 CUDA 性能数字归因给 MLIR。

相关背景可继续阅读[总架构说明](architecture.md)、[BEVFormer 编译器设计](bevformer_compiler_design.md)和[SM86 真机验收记录](bevformer_sm86_validation_2026-10-10.md)。
