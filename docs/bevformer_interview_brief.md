# GAC / BEVFormer 编译器面试口述材料

> 适合项目介绍、简历追问和约 90 秒自述。文中的性能与精度数据均来自当前版本实测；RTX 3080 与 RTX 3070 同属 CUDA SM 8.6，但不能把 3080 的耗时直接当作 3070 的耗时。

## 一句话定位

GAC 是一款面向边缘推理的编译器与运行时原型：它把 BEVFormer-tiny 的模型配置和官方权重转换为统一静态图，完成图优化、显存规划和 CUDA SM86 代码生成，打包成可校验的 HXP，再由有状态 Runtime 执行连续帧推理。

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

这些数据证明官方权重的 cold/warm 路径已经在 SM86 真机上完成数值和首轮性能优化闭环。表中全生成与优化版分别是 3 次和 10 次稳态运行的中位数；fresh-process 数据单独启动进程，并在 Session 构建完成后测量首次 `run`，不包含 Session 构建时间，也不把同一进程已预热后的 `116.635651 ms` 混作首次调用。它们不应外推为 RTX 3070 延迟。

## 当前边界

- 输入是已经完成归一化和 padding 的 FP16 六相机张量，不包含图片解码和原始数据预处理。
- 输出是固定形状 raw class/box 与 FP16 `next_bev`；NMS-free 后处理仍在 CPU。
- 当前数字不包含图像预处理和 CPU NMS-free 后处理，因此尚未宣称完整业务链达到实时性能。
- FP32 cuDNN/cuBLAS 和首轮零拷贝已完成；后续仍需 Tensor Core/混合精度、更多融合、I/O 预绑定、CUDA Graph 和更少的 launch。
- 已在同属 SM86 的 RTX 3080 验证；仍需在桌面 RTX 3070 上复测延迟、峰值显存和稳定性。

## 约 90 秒口述稿

我做的是一个面向边缘推理的 AI 编译器和 Runtime 原型，首个完整模型选择了 BEVFormer-tiny，首个 GPU 目标是 RTX 3070 对应的 CUDA SM86。编译侧先把官方配置和 checkpoint 转换成统一静态图，同时把图像元数据和跨帧状态改成明确的 Tensor ABI；中间层做 shape/dtype 推导、常量与 Conv-BN 折叠、布局规范化和静态显存规划，内部 Reshape 与连续 Slice 可以直接成为零拷贝 view；后端为 BEV 专用算子生成 SM86 CUDA，也能为 FP32 Conv/MatMul 选择 cuDNN 和 cuBLAS，再把权重、计划和 Kernel 信息封装为可校验的 HXP。Runtime 管理上一帧 BEV，只有整帧成功才提交新状态，并用逐 launch CUDA Event Profiler 做按算子归因。现在已经在 RTX 3080 SM86 上用官方权重跑通 cold/warm，最大绝对误差为 `0.004856288`；全生成 cold 是 `425.325073 ms GPU`，接入零拷贝、cuDNN 和 cuBLAS 后是 `57.624464 ms`，加速 `7.38098×`。下一步先做 CUDA Graph、批量 dispatch 和更多融合，并在真实 RTX 3070 上复测。

## 常见追问

### 1. 为什么它算编译器，而不只是调用 CUDA 的推理脚本？

它有明确的前端、统一 IR、优化 Pass、后端 lowering、静态内存计划和版本化模型包；Runtime 执行的是编译产物，而不是重新解释 PyTorch 模型。

### 2. BEVFormer 适配最难的部分是什么？

难点是六相机空间投影、自定义 MSDA、跨帧运动补偿以及 cold/warm 状态语义。它们原来混在 Python、MMCV CUDA 扩展和隐式字典状态里，需要改写成可验证的静态 Tensor 图和显式 Runtime 合同。

### 3. 为什么在 RTX 3080 上验证，也能说明 RTX 3070 的适配进展？

两者的计算能力都是 8.6，因此使用同一 SM86 代码生成和能力约束，能够验证代码可编译、Kernel 可执行和数值正确；但两张卡的资源和性能不同，所以仍需在 RTX 3070 上做最终性能与显存验收。

### 4. 现在的精度和性能应该怎样描述？

官方权重的 cold/warm 输出已经与参考实现完成真机数值对齐，最大绝对误差不超过 `0.004856288`。同一 RTX 3080 上，cold 的全生成与优化版中位数为 `425.325073 / 436.881591 ms` 和 `57.624464 / 69.403093 ms`（GPU / wall），提升 `7.38098× / 6.29484×`；warm 提升 `7.49750× / 6.36705×`。fresh-process 中 Session 构建后的首个 cold `run` 为 `277.443573 / 289.668472 ms`，Session 构建时间另计。完整预处理与 CPU 后处理不在这些数字内。

### 5. HXP 和有状态 Runtime 解决了什么问题？

HXP 把图、权重、目标、Kernel 和内存计划固定成可校验产物；有状态 Runtime 把 `prev_bev` 的生命周期和提交时机显式化，使连续帧可重放，并保证失败帧不会污染后续状态。
