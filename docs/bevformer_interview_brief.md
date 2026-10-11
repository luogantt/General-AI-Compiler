# GAC / BEVFormer 编译器面试口述材料

> 适合项目介绍、简历追问和约 90 秒自述。文中的性能与精度数据均来自当前版本实测；RTX 3080 与 RTX 3070 同属 CUDA SM 8.6，但不能把 3080 的耗时直接当作 3070 的耗时。

## 一句话定位

GAC 是一款面向边缘推理的编译器与运行时原型：它把 BEVFormer-tiny 的模型配置和官方权重转换为统一静态图，完成图优化、显存规划和 CUDA SM86 代码生成，打包成可校验的 HXP，再由有状态 Runtime 执行连续帧推理；同时已为生产形状 FFN 建立独立的 MLIR GPU→NVVM→PTX 编译与执行切片。

## 编译器分层

```text
BEVFormer 配置 + 官方 checkpoint
            │
            ▼
前端适配器：安全读取权重、校验 key/shape/dtype、显式化图像元数据和时序状态
            │
            ▼
Canonical Graph IR：统一 Tensor、Shape、DType、Layout、算子和输入输出 ABI
            │
            ▼
优化 Pass：常量折叠、Conv-BN 折叠、视图规范化、算子识别、布局传播
            │
            ▼
内存规划：按张量生命周期复用静态 arena，分开管理输入、输出、常量和临时量
            │
            ▼
CUDA SM86 后端：能力检查、CUDA C 生成、cuDNN/cuBLAS 选择、NVRTC、launch metadata
            │
            ▼
HXP 模型包：图、权重、目标信息、内存计划、源码及摘要校验
            │
            ▼
Runtime：加载、绑定输入、执行、计时、输出校验、跨帧状态提交或失败回滚
```

这套分层把“模型语义”“设备代码”和“运行时状态”分开。前端不关心 CUDA block，后端不重新解析 PyTorch 对象，Runtime 只消费已经确定的包和执行信息，因此每一层都可以独立测试和替换。

## MLIR 到 PTX 的已实现范围

`mlir-ptx-sm86` 当前只编译一个生产形状 FP32 FFN：

```text
[1, 2500, 256]
  → MatMul(256, 512) + Bias + ReLU
  → MatMul(512, 256)
  → Bias + query residual + LayerNorm
```

编译器从 Canonical Graph 的独立分支严格检查 pattern，再直接物化三个 MLIR `gpu.func`；tensor-level MLIR 仅用于调试/审阅，不是 GPU emitter 的输入，也不进入部署 PTX artifact。官方 LLVM 24 `mlir-opt` 完成 GPU Dialect→NVVM lowering，LLVM NVPTX 后端生成 textual PTX 7.1 / `sm_86`。Runtime 校验 GPU MLIR 的 source hash、binary MLIR/PTX 一致性、64 位地址和三个精确 entry，随后加载 textual PTX，CUDA Driver 在模块加载时再 JIT 为 SASS。这个执行路径不调用 cuBLAS，不编译 CUDA C，也没有 NVRTC 或 CPU fallback。

这里的“全流程”限定在这三个 FFN Kernel。Tensor/Linalg 到任意 GPU 图的通用 bufferization、tiling 和 GPU mapping 还没有完成；CNN、Spatial/Temporal Attention、MSDA、Decoder、Detection Head 与跨帧状态仍由现有 CUDA/HXP 路径执行。

## BEVFormer 冷帧与暖帧数据流

六路经过归一化和 padding 的 FP16 图像先进入 ResNet-50 与单层 FPN，再完成相机布局转换、camera/level embedding、BEV query、位置编码和 CAN-bus 条件编码。三层 Encoder 通过 Temporal Self-Attention 与 Spatial Cross-Attention 生成 BEV，六层 Decoder 和六组分类/回归分支输出固定形状的 raw class、raw box 以及下一帧使用的 `next_bev`。

- **冷帧（cold）**：没有历史 BEV。每层 Temporal Self-Attention 按官方语义使用该层当前 query 构造冷启动 temporal value，然后继续空间注意力和 FFN。
- **暖帧（warm）**：Runtime 取上一帧 `prev_bev`，根据车辆旋转和平移做 BEV rotate 和 reference-point shift，再与当前 query 组成历史 temporal queue，送入三层 Encoder。
- **状态安全**：只有本帧执行成功，并且三个输出的 shape、dtype、连续性和有限值检查全部通过，Runtime 才提交 `next_bev`；执行失败时保留上一份状态。
- **后处理边界**：GPU/HXP 返回固定形状的 raw tensor；NMS-free Top-K、box denormalization 和范围过滤目前由显式 CPU API 完成。

## RTX 3070 / CUDA SM86 适配点

RTX 3070 的计算能力是 8.6，因此首版目标固定为 `cuda-sm86`。后端在编译前检查静态 shape、dtype、布局和算子约束：逐元素、Attention、GridSample、BEV 几何和 MSDA 等走 generated CUDA/NVRTC，生产尺寸 FP32 Conv 和 MatMul 分别选择 cuDNN 与 CuPy/cuBLAS。HXP 的 launch metadata 显式保存 vendor/generated 候选，Runtime 把实际选择写进会话、Kernel 信息和 profile，便于复现与归因。内部 Reshape 和连续 Slice 尽量成为零拷贝 view。公开的图像和状态 ABI 使用 FP16，Transformer/MSDA 等关键内部路径仍以 FP32 为主。

本轮真机使用 RTX 3080，它与 RTX 3070 同为 SM86，所以验证了指令目标、Kernel 生成、NVRTC 编译和运行时契约；显存规模、带宽和延迟仍需在实际 RTX 3070 上单独复测。

## 当前实测结果

| 项目 | 结果 |
| --- | --- |
| 真机环境 | RTX 3080，CUDA SM 8.6 |
| 自动化测试 | 最终强制 CUDA 运行 287 项，`18.881s`，结果 `OK`，其中 1 项跳过 |
| 官方权重 cold 最大绝对误差 | class `0.003891945`；box `0.004856288`；BEV `0.001953125` |
| 官方权重 warm 最大绝对误差 | class `0.000953674`；box `0.000725389`；BEV `0.001953125` |
| 生产 step 图 | 融合前 876 节点；Add+ReLU 融合后 837 节点；实际 703 次设备 dispatch |
| Runtime 规划显存 | Tensor `705,177,100` bytes；另加 cuDNN 8 MiB 后计划预算/准入值 `713,565,708` bytes，不代表进程设备峰值 |
| 全生成 FP32 基线 | cold `425.325073 / 436.881591 ms`；warm `426.032532 / 437.718436 ms`（GPU / wall） |
| cuDNN/cuBLAS 优化版 | cold `57.624464 / 69.403093 ms`；warm `56.823263 / 68.747440 ms`（GPU / wall） |
| 本轮收益 | cold `7.38098× / 6.29484×`；warm `7.49750× / 6.36705×`（GPU / wall） |
| fresh-process 中 Session 构建后的首个 cold `run` | `277.443573 ms` GPU；`289.668472 ms` wall |
| MLIR 全 PTX 生产形状 FFN | `1.092～1.102 ms/FFN` device-resident；max abs error `7.153e-7` |
| cuBLAS hybrid 同 FFN | `0.209～0.214 ms/FFN` device-resident；当前全 PTX 慢 `5.150～5.265×` |

这些数据证明官方权重的 cold/warm CUDA/HXP 路径已经在 SM86 真机上完成数值和首轮性能优化闭环。表中全生成与优化版分别是 3 次和 10 次稳态运行的中位数；fresh-process 数据单独启动进程，并在 Session 构建完成后测量首次 `run`，不包含 Session 构建时间，也不把同一进程已预热后的 `116.635651 ms` 混作首次调用。MLIR 行来自同进程共驻留、两种首次顺序、每个 repeat 交替位置的独立 FFN 基准。全图数字与 FFN 数字范围不同，都不应外推为 RTX 3070 延迟。

MLIR→PTX 并不会自动提高速度。离线 textual PTX 省去部署侧的 MLIR lowering 和 CUDA C/NVRTC 编译，但 Driver 的 PTX→SASS JIT 仍存在，稳态速度仍取决于 Kernel schedule。当前全 PTX 的两个 MatMul 是“一线程一个输出、串行遍历 K”的正确性实现，没有 shared-memory tiling、向量化、Tensor Core 或 autotuning，因此明显慢于 cuBLAS。下一步是 tiled/shared-memory MatMul、NVGPU/MMA Tensor Core、epilogue/LayerNorm 融合和 autotuning；生产版本也可让编译器选择 cuBLAS/cuBLASLt library-call lowering。

## 当前边界

- 输入是已经完成归一化和 padding 的 FP16 六相机张量，不包含图片解码和原始数据预处理。
- 输出是固定形状 raw class/box 与 FP16 `next_bev`；NMS-free 后处理仍在 CPU。
- 当前数字不包含图像预处理和 CPU NMS-free 后处理，因此尚未宣称完整业务链达到实时性能。
- FP32 cuDNN/cuBLAS 和首轮零拷贝已完成；后续仍需 Tensor Core/混合精度、更多融合、I/O 预绑定、CUDA Graph 和更少的 launch。
- MLIR→PTX 当前只覆盖生产形状 FFN，不包含完整 BEVFormer；通用 tensor/linalg lowering、HXP PTX artifact 和 Tensor Core 调度仍待完成。
- 已在同属 SM86 的 RTX 3080 验证；仍需在桌面 RTX 3070 上复测延迟、峰值显存和稳定性。

## 约 90 秒口述稿

我做的是一个面向边缘推理的 AI 编译器和 Runtime 原型，首个完整模型选择了 BEVFormer-tiny，首个 GPU 目标是 RTX 3070 对应的 CUDA SM86。编译侧先把官方配置和 checkpoint 转换成统一静态图，把图像元数据和跨帧状态改成明确的 Tensor ABI，再做 shape/dtype 推导、Conv-BN 折叠、布局规范化和静态显存规划。生产全图路径为 BEV 专用算子生成 SM86 CUDA，并为 FP32 Conv/MatMul 选择 cuDNN 和 cuBLAS，最后封装成可校验的 HXP；Runtime 只有在整帧成功后才提交下一帧 BEV。官方权重的 cold/warm 已在 RTX 3080 SM86 跑通，最大绝对误差为 `0.004856288`，cold 从全生成的 `425.325073 ms GPU` 优化到 `57.624464 ms`。此外，我为生产形状 FFN 做了独立的 MLIR 全 PTX 切片：三个 `gpu.func` 经官方 GPU→NVVM→LLVM NVPTX 生成 textual PTX，由 Runtime 校验并加载，再由 CUDA Driver JIT 为 SASS。它的最大绝对误差是 `7.153e-7`，但当前 `1.092～1.102 ms`，比 `0.209～0.214 ms` 的 cuBLAS hybrid 慢 `5.150～5.265×`，因为 MatMul 还没有 tiled、Tensor Core 和 autotuning。下一步会做 shared-memory tiling、NVGPU/MMA、融合和调优，并在真实 RTX 3070 上复测；MLIR 结果只代表这个 FFN 编译单元，不代表完整 BEVFormer 已全部降低到 PTX。

## 常见追问

### 1. 为什么它算编译器，而不只是调用 CUDA 的推理脚本？

它有明确的前端、统一 IR、优化 Pass、后端 lowering、静态内存计划和版本化模型包；Runtime 执行的是编译产物，而不是重新解释 PyTorch 模型。

### 2. BEVFormer 适配最难的部分是什么？

难点是六相机空间投影、自定义 MSDA、跨帧运动补偿以及 cold/warm 状态语义。它们原来混在 Python、MMCV CUDA 扩展和隐式字典状态里，需要改写成可验证的静态 Tensor 图和显式 Runtime 合同。

### 3. 为什么在 RTX 3080 上验证，也能说明 RTX 3070 的适配进展？

两者的计算能力都是 8.6，因此使用同一 SM86 代码生成和能力约束，能够验证代码可编译、Kernel 可执行和数值正确；但两张卡的资源和性能不同，所以仍需在 RTX 3070 上做最终性能与显存验收。

### 4. 现在的精度和性能应该怎样描述？

官方权重的 cold/warm 输出已经与参考实现完成真机数值对齐，最大绝对误差不超过 `0.004856288`。同一 RTX 3080 上，cold 的全生成与优化版中位数为 `425.325073 / 436.881591 ms` 和 `57.624464 / 69.403093 ms`（GPU / wall），提升 `7.38098× / 6.29484×`；warm 提升 `7.49750× / 6.36705×`。fresh-process 中 Session 构建后的首个 cold `run` 为 `277.443573 / 289.668472 ms`，Session 构建时间另计。完整预处理与 CPU 后处理不在这些数字内。独立的生产形状 FFN 上，全 PTX 为 `1.092～1.102 ms`，cuBLAS hybrid 为 `0.209～0.214 ms`，所以当前全 PTX 慢 `5.150～5.265×`；该数据不能与完整 step 延时混为一谈。

### 5. HXP 和有状态 Runtime 解决了什么问题？

HXP 把图、权重、目标、Kernel 和内存计划固定成可校验产物；有状态 Runtime 把 `prev_bev` 的生命周期和提交时机显式化，使连续帧可重放，并保证失败帧不会污染后续状态。

### 6. MLIR 是怎样真正参与编译的？为什么当前没有更快？

生产形状 FFN 后端生成三个标准 MLIR `gpu.func`，调用本地 LLVM 24 的 `mlir-opt` 将 GPU Dialect 降到 NVVM，再由 LLVM NVPTX 后端输出 textual PTX。编译器保存 GPU MLIR、含 `gpu.binary` 的 lowered MLIR、PTX、命令、revision 和摘要；部署侧校验后加载 PTX，CUDA Driver 再 JIT 为 SASS，所以它不是 Python 重写的“类 MLIR”，也不是只把 MLIR 当说明文档。当前 MatMul 的 GPU schedule 很朴素，成熟 cuBLAS 有 tiling、流水化和 Tensor Core 优化，因此 MLIR 版当前更慢。MLIR 的价值是提供可验证、可优化的 lowering 框架；真正提速仍要实现好的 schedule、融合和 autotuning。
