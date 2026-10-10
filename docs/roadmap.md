# 开发路线图

> 状态核对：2026-10-10。官方 `bevformer_tiny_epoch_24.pth` 的统一 cold/warm 五输入三输出 step 已在 Ubuntu 22.04、RTX 3080（SM 8.6）上完成 HXP 真机数值验收。HXP 融合前图有 876 个节点；39 组安全 Add+ReLU 融合后为 837 个执行节点，零拷贝后实际设备 dispatch 为 703，内部 arena 保持 `552,960,000` bytes；最终强制 CUDA 测试结果为 `Ran 287 tests in 18.881s ... OK (skipped=1)`。首轮性能优化已完成内部 Reshape/连续 Slice 零拷贝、Add+ReLU 融合、FP32 cuDNN Conv、FP32 CuPy/cuBLAS MatMul 和逐 launch Profiler：cold 从 `425.325073 / 436.881591 ms` 降到 `57.624464 / 69.403093 ms`，warm 从 `426.032532 / 437.718436 ms` 降到 `56.823263 / 68.747440 ms`（GPU / wall）。RTX 3080 与目标 RTX 3070 的编译目标相同，但桌面 RTX 3070 的设备专属性能/显存仍待测试；显式 CPU NMS-free 后处理、真实 nuScenes 数据验收和通用异构基础设施仍是后续工作。

## 路线原则

路线图优先建立真实 GPU 的最小闭环，再补齐通用编译—Runtime 契约，最后扩展硬件形态。当前保留可执行示例、CPU 参考结果、HXP 往返和 Trace；增加更多算子不能替代官方 checkpoint、完整子图和物理 GPU 验证。

## 当前优先里程碑：SM 8.6 单卡适配

- 已完成：静态 FP32 CUDA C 生成、二维 tiled MatMul、Bias/GELU 融合、末轴 LayerNorm/Softmax、广播 Add 和基本逐元素算子。
- 已完成：NVRTC 模块编译、真实设备检查、驻留权重、单 stream 执行、CUDA Event 与主机时间分别计时；可选逐 launch Profiler 记录实际 implementation，并输出按 op / implementation 聚合 JSON，默认路径不承担逐 launch Event 开销。
- 历史真机已验证：离线 `compile`、`doctor`、JSON 模型 `run`，以及 MLP 与独立 MSDA；当时扩展后的 25 项测试在 RTX 3070 Ti Laptop 上通过。
- 已实现并真机运行：`.hxp` compile/inspect/run、安全加载、确定性静态 arena 计划，以及 CUDA Session 的单次 arena 分配和 offset view；内部 Reshape/连续 Slice 可直接复用源存储。生产尺寸统一 step 的 Tensor 计划为 `705,177,100` bytes；cuDNN 另设 8 MiB workspace budget，计划预算/准入值为 `713,565,708` bytes，已在 RTX 3080 上完成 cold/warm 执行；该值不包含 module、context、Event 和 vendor 临时分配。
- 已实现并纳入全图验收：通用 Graph Composer、MHA 规范图、CNN 与 Conv+BN folding、BEVFormer-tiny ResNet50-FPN 导入合同、feature-prep、learned encoder inputs、BEV 几何、Spatial/Temporal Attention、tiny FFN、完整三层 Encoder、完整六层 Decoder、六层 raw detection projection、cold/history image-to-BEV、Decoder+raw-head 和统一 cold/warm step pipeline，以及导出/官方权重映射/golden/状态合同。generated Conv/MatMul 是可检查基线，生产 FP32 路径已接入 cuDNN/cuBLAS。
- 已实现：`bevformer_postprocess.py` 的显式 CPU NMS-free Top-K 300、box denormalization 与范围过滤；该主机阶段不在 CUDA/HXP 图内。
- 已实现：`gac-bevformer` 的 22 个子命令，包括合同/权重检查、backbone、feature/input preparation、Encoder/Decoder 单层与 stack、raw detection 单层与 stack，以及 `history-encoder`、`decoder-pipeline`、`cold-pipeline`、`step-pipeline` 的 HXP 构建入口。
- 已实现并真机验收：官方 rotate/shift 运动补偿、`has_prev_bev` cold/warm 选择和 `BevFormerRuntime` 状态管理。warm 每层共享 `[motion prev, initial query]`，cold 每层动态构造 `[current query, current query]`。冷启动只清零 `can_bus_delta[:3]` 与 `can_bus_delta[-1]`，保留四元数、绝对 heading 与其他 CAN-bus 特征；只有 Session 执行和输出校验成功才提交状态。
- 已验收：官方 checkpoint 的 cold/warm raw step、生产 HXP 及物理 SM86 数值。性能版 cold 的 cls/bbox/bev max abs 为 `0.003891945` / `0.004856288` / `0.001953125`，warm 为 `0.000953674` / `0.000725389` / `0.001953125`。
- 已验收：全生成 cold/warm 中位数分别为 `425.325073 / 436.881591 ms` 与 `426.032532 / 437.718436 ms`；零拷贝 view + FP32 cuDNN/cuBLAS 分别为 `57.624464 / 69.403093 ms` 与 `56.823263 / 68.747440 ms`（GPU / wall），cold 加速 `7.38098× / 6.29484×`，warm 加速 `7.49750× / 6.36705×`。fresh-process 中 Session 构建后的首个 cold `run` 为 `277.443573 / 289.668472 ms`，Session 构建时间另计；同进程已预热调用不作为 fresh 数据。
- 已实现并进入切片真机验收：`mlir-cuda-sm86` 严格匹配一个静态 FP32 BEVFormer FFN，使用 2×CuPy/cuBLAS MatMul、NVRTC `Bias+ReLU` 和 `Bias+Residual+LayerNorm`。MLIR source 仅是可审阅/可验证的 runtime-call ABI manifest，实际 dispatch 由经校验的 Python call records 驱动；HXP v1 不承载该计划。生产尺寸单 FFN 基准必须保存 artifact hash，在同进程中按 repeat 交替 MLIR/CUDA 的测量位置，并用两种首轮顺序独立重跑；具体数字只从最终原始 JSON 引用，不能外推为完整 BEVFormer 或 RTX 3070 结果。证据目录见 [`validation/sm86/mlir-cuda-cublas-2026-10-10`](validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)。
- 下一步基础设施：固定 Buffer/Artifact/Driver 接口，加入通用执行计划和 CPU/GPU 候选选择；性能与产品验收继续完成混合精度/Tensor Core、真实 nuScenes、CPU 后处理 Trace 和桌面 RTX 3070 测量。
- 性能深化：以当前 FP32 vendor-library 基线为对照，优先推进 CUDA Graph、批量 dispatch 和更多融合，再评估 FP16/Tensor Core；每次都报告数值误差和含传输的端到端收益。显式 128 MiB cuDNN workspace 虽把 55 个 Conv p50 从 `35.030 ms` 降至 `25.781 ms`，整图 p50 却从 `57.924 ms` 变为 `58.231 ms`，且多占 120 MiB，本轮不集成；当前 `cupyx.cudnn` capture 不兼容，后续图捕获需稳定的低层 API。

当前 `.cu`/`.plan.json` 是调试中间产物；`.hxp` 已能保存规范图、权重、CUDA 源码、带 `implementation_candidates` 的 launch metadata 和静态内存计划，并支持校验加载。Runtime 把最终选择写入 `session.info`、`kernel_info` 和 profile。它还没有通用异构步骤表、CPU 回退、多 target variant 或预编译 cubin。参考 [架构说明第 10 节](architecture.md#10-rtx-3070-系列的首个实现) 和 [真机报告](validation/sm86/gpu-fused.json)。

版本号表示研究里程碑，不承诺生产级 API 稳定性。只有经过兼容性测试并发布独立 SDK 规范的接口，才可以称为稳定 ABI。

## v0.1：可执行编译—Runtime 闭环

目标：用最小算子集证明模型语义、设备划分、执行计划、打包和 Runtime 可以协同工作。

- JSON Frontend 与确定性常量初始化；
- Canonical Graph、静态 Shape/DType 检查；
- `MatMul`、`Add`、`GELU`、`LayerNorm` 演示链路；
- Python 后端协议与注册表；
- NumPy CPU 参考后端；
- 明确标注的 `sim-gpu` 模拟后端；
- 支持传输成本的启发式异构划分；
- Host Execution Plan；
- 版本化 `.hxp` 原型包；
- compile / inspect / run CLI；
- 确定性输入、输出摘要与 JSON Trace；
- unittest 单元测试和端到端测试。

完成标准：仓库中的 MLP 示例可以从 JSON 编译成 `.hxp`，重新加载后得到与 CPU 参考语义一致的输出，并能在 Trace 中看到节点放置和跨设备步骤。模拟 GPU 不能被描述为真实加速。

### v0.1 实施批次

第一条路径限定静态 Shape、FP32、连续 Tensor、纯函数图和同步执行。接口与包结构见 [架构设计第 8 节](architecture.md#8-首版数据契约)。每批提交能单独验证的实现，不先创建只有名称的模块。

| 顺序 | 实现工作 | 必须通过的检查 |
| --- | --- | --- |
| 1 | 固定现有图语义与 Pass 基线，建立 `tests/` | 广播/MatMul 维度、错误轴、非法拓扑；融合前后数值与 provenance 一致；导出的中间值不被错误融合 |
| 2 | 后端协议、注册表、`numpy-cpu`、`sim-gpu` | 支持范围与拒绝原因；重复注册/未知目标错误；模拟复制无数组别名；输入和常量不被修改 |
| 3 | 候选划分、Execution Plan 与校验器 | 高搬运成本保留 CPU；分支、共享常量、图输出回传；未初始化读、释放后读、依赖环被拒绝 |
| 4 | 编译入口与版本化 HXP | 融合可关闭/分解；保存加载往返；缺失 artifact、不兼容版本、损坏摘要被拒绝 |
| 5 | Session、Invocation、CPU 回退计划、Trace | 固定输入与独立 NumPy 公式对照；重复执行；缺设备回退或明确失败；节点/分区/步骤关联可追踪 |
| 6 | CLI、可复现示例与完整文档 | 三个独立进程完成 compile/inspect/run；真实生成包与 Trace；错误命令返回非零退出码 |

测试使用显式放置约束覆盖跨设备路径，同时保留自由放置时可能选择全 CPU 的正确行为。参考求值工具不是正式 Runtime；图 JSON 往返也不能代替 HXP 往返测试。

当前测试已覆盖 CUDA 子集的编译器契约、CLI、HXP 往返/安全拒绝和静态 arena；历史记录覆盖 MLP/MSDA 真机数值。上表第 4 批中的 HXP 子集已经完成，但通用后端协议、划分、跨设备 Execution Plan、CPU 回退及三个独立进程的完整异构闭环仍待实现，因此不能将这些批次整体视为完成。

## BEVFormer-tiny 首条支持路径

| 工作项 | 当前状态 | 尚缺的验收 |
| --- | --- | --- |
| MSDeformableAttention | 历史切片与本轮全图 SM86 真机已验证 | 保持数值回归与边界测试 |
| HXP 与静态 arena | 生产 step 包已在 RTX 3080 按 `705,177,100` bytes Tensor 计划完成执行；另加 cuDNN 8 MiB 后计划预算/准入值 `713,565,708` bytes | 桌面 RTX 3070 实际峰值与压力测试 |
| MHA、CNN、Conv+BN | 已实现并进入官方 checkpoint 全图 HXP 验收；生产 FP32 Conv/MatMul 已接入 cuDNN/cuBLAS | FP16/Tensor Core、更多融合与 Shape 策略 |
| ResNet50 + 单层 FPN | 规范图、严格导入合同、generated 基线和 FP32 cuDNN 真机路径已验证 | 混合精度与更细的性能归因 |
| 导出、官方权重映射、golden、时序状态合同 | 官方 checkpoint 映射、golden、cold/warm 状态执行已验收 | ONNX/`torch.export` 捕获仍待完成 |
| Feature-prep 与 learned encoder inputs | 已实现并进入官方 checkpoint 全图 HXP 验收 | 保持逐层回归 |
| BEV geometry、Spatial/Temporal Attention、tiny FFN | 完整三层 Encoder 已进入 cold/warm 真机验收 | 保持逐层回归与更多边界样例 |
| Cold image-to-BEV Encoder pipeline | cold 最终 raw 输出与 `next_bev` 已对齐官方 golden | 真实 nuScenes cold 样例 |
| History image-to-BEV 与运动补偿 | warm 最终 raw 输出与 `next_bev` 已对齐官方 golden | 真实连续帧与设备状态驻留优化 |
| Decoder | 完整六层 stack 已随最终 raw 输出通过官方 checkpoint 真机对比 | 真实 nuScenes 逐帧回归 |
| Raw detection 与后处理 | raw projection 已真机对齐；NMS-free Top-K/box denormalization 为显式 CPU 阶段 | 真实 nuScenes boxes/scores/labels 与 CPU 边界 Trace |
| 统一 BEVFormer step | 官方 checkpoint 的 cold/warm HXP 已在 RTX 3080 SM86 通过数值验收；FP32 优化 cold/warm GPU 加速为 `7.38098× / 7.49750×` | 桌面 RTX 3070、真实 nuScenes、更多分位数与混合精度报告 |

当前可声明固定 Profile 的 BEVFormer-tiny raw step 已通过官方 checkpoint 和真实 SM86 GPU 验收。只有真实 nuScenes 精度、显式 CPU 后处理边界和桌面 RTX 3070 的设备结果也完成后，才使用没有这些限定条件的完整支持表述。

## v0.2：真实异构推理

目标：用一个真实设备后端替换模拟执行路径，同时保持 v0.1 的编译—Runtime 契约。

- 首个真实 GPU 路径，优先 CUDA；也可根据设备条件选择 Vulkan/Metal；
- 将首版分开的 Compiler Backend / Runtime Driver 接口落实为可独立加载的真实硬件实现；
- 显式 Device、MemorySpace、Buffer/TensorView；
- alloc/copy/dispatch/event/wait/free 计划步骤；
- 异步队列、timeline/event 和错误传播；
- DLPack 导入导出与可验证的零拷贝条件；
- 真实设备能力、内存预算和传输带宽 Profile；
- Kernel/artifact 缓存和兼容性检查；
- 外部权重、mmap 和常量去重；
- I/O 预绑定，避免每次 Run 重复分配；
- 设备级 Trace 与基准工具。

完成标准：至少一个子图真正运行在非 CPU 设备上；Trace 和设备 Profiler 能证明 dispatch 发生在目标设备，跨设备复制次数可解释，关闭该后端时存在明确 CPU fallback 或清晰错误。

SM 8.6 已选择 CUDA C/NVRTC 与 vendor library 混合路径；MatMul/Add/GELU、LayerNorm、MSDA、CNN/MHA 以及官方 checkpoint 的统一 BEVFormer step 均已进入真机验证。静态 arena、零拷贝内部 view 及 CUDA Session 的一次性分配已在 RTX 3080 生产 HXP 中使用。逐 launch Profiler 已能用 CUDA Event 做按 op 和实际 implementation 归因；最终受扰动 Profile 为 `76.05085 ms GPU / 95.8321 ms wall`，Kernel Event 求和 `63.4391 ms`，与普通稳态基准分开解释。下一步继续扩展通用异步执行，并在桌面 RTX 3070 上核对实际峰值显存。性能工作必须分别记录编译时间、首次加载、fresh-process 首次 `run`、稳态端到端延迟和设备 Event 时间，并包含输入上传和输出回传；当前优化版 cold/warm 中位数为 `57.624464 / 56.823263 ms GPU` 与 `69.403093 / 68.747440 ms wall`，仍需 RTX 3070 实测。

## v0.3：跨 ISA 与插件化

目标：证明相同上层模型和 Execution Plan 语义可以覆盖不同 CPU ISA 与独立发布的设备插件。

- CPU AOT：LLVM lowering 和 x86-64/AArch64 产物；
- StableHLO 或版本化 MLIR 输入路径；
- 可演进的 Compiler Backend C ABI；
- 可演进的 Runtime Driver C ABI；
- 插件发现、版本协商、capability/extension negotiation；
- 多 variant 模型包与设备兼容性评分；
- 远程/交叉编译，host tools 与 target artifacts 分离；
- 动态 Shape bucket 与 Shape variant 选择；
- 量化语义、预打包权重和离线调优数据库；
- 可选的 Tile DSL/外部 Kernel provider；
- 后端 Conformance Suite 和 ABI 兼容矩阵。

完成标准：同一模型包或其可复现构建描述可以覆盖至少两种 CPU ISA 和一个加速器；插件可在不重新构建核心 Runtime 的情况下装载，旧 minor ABI 插件通过兼容性测试。

## v0.4：空间型 NPU 后端

目标：验证图划分和 Host Execution Plan 能扩展到具有计算 Tile、本地 SRAM 和显式 DMA 的空间架构。

- Spatial Mapping IR：compute tile、memory tile、DMA channel、route；
- 本地 SRAM 容量、bank、对齐和双缓冲约束；
- 算子 tiling、placement、routing 与 schedule 协同；
- MLIR-AIE 或厂商编译器 adapter；
- 多级内存和 host/device/片上 DMA 的统一传输成本；
- 静态死锁检查与事件依赖验证；
- NPU artifact、firmware/driver compatibility 和安全加载；
- CPU/GPU/NPU 三类设备的全局划分实验；
- Deadline、p99 延迟和能耗权重的 Cost Model。

完成标准：至少一个端到端模型子图在空间型 NPU 或可信仿真器上执行；编译报告能够解释 tile/DMA/内存选择，并能把运行时性能事件回溯到原始节点和映射决策。

## 跨版本持续工作

以下要求不等待某个“大版本”再补：

- 每个新算子先定义参考语义和失败边界；
- 每个模型包字段都有 schema version 和兼容性测试；
- 每个性能优化同时保留正确性与回归基线；
- Trace 贯穿 source node、IR、partition、kernel、dispatch；
- 对依赖和生成物执行许可证与供应链审查；
- 文档明确区分已实现、实验性和计划中能力；
- 不用模拟结果宣称真实硬件性能。

## 非目标

短期内不追求：

- 复制所有主流框架的完整算子集合；
- 自研 LLVM、GPU 驱动或厂商 NPU 工具链；
- 用单一 IR 抹平所有硬件差异；
- 在缺少设备测量的情况下宣称性能优于成熟系统；
- 在稳定 ABI、包签名和生产级威胁模型尚未完成时承诺生产部署。
