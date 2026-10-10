# BEVFormer 编译器设计：RTX 3070 / CUDA SM 86 首版

> 文档状态：设计基线与阶段实现记录；统一 cold/warm raw step 已用官方 checkpoint 完成 SM86 物理 GPU 数值验收。  
> 目标硬件：NVIDIA Ampere、Compute Capability 8.6，包括 RTX 3070 和 RTX 3070 Ti Laptop。  
> 首个模型：本地 `BEVFormer-master` 中的 BEVFormer-tiny 推理图。  
> 当前可运行主线：官方 BEVFormer-tiny checkpoint 的统一 cold/warm HXP；CPU NMS-free 后处理保持显式边界。

## 1. 性质与边界

本设计参考仓库根目录的《小鹏图灵 AI 编译器技术架构说明》，但该文档本身明确标注为**推测性参考架构**。小鹏并未公开确认其内部 Graph IR、MLIR 使用方式、机器指令、模型格式或 Runtime API。因此，本文是 GAC 项目针对 BEVFormer 和 RTX 3070 制定的工程方案，不能表述为小鹏真实编译器的复刻，也不能据此声称兼容小鹏图灵芯片。

本地 BEVFormer 参考代码位于：

```text
C:\Users\Thomas\Desktop\project\BEVFormer-master
```

首版以以下文件为语义依据：

```text
projects/configs/bevformer/bevformer_tiny.py
projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
projects/mmdet3d_plugin/bevformer/modules/transformer.py
projects/mmdet3d_plugin/bevformer/modules/encoder.py
projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
projects/mmdet3d_plugin/bevformer/modules/multi_scale_deformable_attn_function.py
```

首版只处理推理，不处理训练、反向传播、Hungarian Assigner 和损失函数。以下能力也不在首版范围内：BEVFormer-base、动态摄像头数量、动态分辨率、多 GPU、INT8、占用网络、训练时多帧队列以及未知算子的静默 CPU 回退。

## 2. 状态定义

本文用五个标签区分事实与计划：

- **历史真机已验证**：存在随仓库保存的物理 GPU 报告；仅覆盖报告所列源码与切片。
- **当前真机已验证**：当前源码与官方 checkpoint 已在物理 SM 8.6 GPU 上通过相应数值验收。
- **当前已实现**：仓库中已经存在代码并通过 CPU、代码生成或契约测试，但不自动表示性能或真实数据集精度已经验收。
- **正在完成**：已有实现工作，但尚未满足该阶段的完整验收条件。
- **后续里程碑**：尚未实现的扩展，不构成当前功能承诺。

### 2.1 当前已实现

历史真机证据已经覆盖：

- JSON/Canonical Graph 到 CUDA C、CuPy/NVRTC 单卡 Session 的 MLP 闭环；
- `MSDeformableAttention` 的 Shape/DType/属性校验、独立 NumPy 参考语义和 SM86 CUDA Kernel；
- 2026-10-10 在 RTX 3070 Ti Laptop GPU 上完成当时 25 项测试，以及真实 tiny 尺寸的 MSDA 数值、Trace 和性能验证。

当前仓库还已实现：

- 版本化 `.hxp` 的确定性打包、安全校验、加载、检查与运行入口；包内保存规范图、常量、CUDA 源码、静态内存计划，以及带 `implementation_candidates` 的 launch metadata；
- 基于存活期的静态 arena 规划、对齐和复用校验；CUDA Session 已一次性分配内部 arena，并按 allocation offset 建立中间 Tensor view，内部 Reshape/连续 Slice 可零拷贝复用源存储；
- MHA 规范图、参考求值和 CUDA 源码生成检查；
- Conv2D、BatchNorm、MaxPool2D 等 CNN 语义、Conv+BN 常量折叠、generated Conv/MatMul 正确性 Kernel，以及生产 FP32 cuDNN Conv/CuPy-cuBLAS MatMul；
- 可选逐 launch CUDA Event Profiler、按 op 与实际 implementation 聚合 JSON，以及每个 launch 的 vendor/generated 实现来源记录；
- BEVFormer-tiny ResNet-50 + 单层 FPN 的规范图导入合同、官方 key/shape 严格校验和缩小图参考执行；
- feature-prep 的 FPN layout 转换、camera/level embedding，以及 learned encoder inputs 的 BEV query/position、CAN-bus conditioning 和静态参考网格；
- BEVFormer 配置提取、纯 Tensor ABI、离线 manifest、官方 checkpoint 权重映射/安全权重文件、golden 数据，以及显式时序状态合同；
- BEV 几何参考点/投影/可见性 Mask、固定上界 Spatial Cross Attention 和 Temporal Self Attention 静态子图；
- tiny FFN 的 `C → 2C → C`、ReLU、推理态 dropout identity、残差和可选 post-LayerNorm 静态子图；
- 通用 Graph Composer、完整单层 Encoder 与完整三层 Encoder stack；stack 同时定义显式 history queue 和逐层 cold-start queue 两种合同；
- learned object-query/reference 初始化、Decoder self/cross attention、FFN/norm、reference refinement、完整单层 Decoder 与完整六层 Decoder stack；
- 六层 raw classification/box projection，以及把完整 Decoder 直接连接六层 raw head 的固定 Shape pipeline；
- Backbone→feature-prep→learned inputs→三层 Encoder 的 cold image-to-BEV pipeline，以及接收运动补偿历史的 history image-to-BEV pipeline；
- 与官方测试路径一致的 `prev_bev` rotate、reference-point shift，以及 `has_prev_bev` 对 cold/warm 历史与参考点的选择；
- 五输入三输出统一 streaming step：FP16 image/state 边界、三层 Encoder、六层 Decoder、raw head 和 FP16 `next_bev` 已贯通，并可由 `step-pipeline` 生成 HXP；warm 在每层共享 `[motion prev, initial query]`，cold 在每层动态构造 `[current query, current query]`；
- `BevFormerRuntime` 的 scene/CAN-bus delta、cold/warm flag、状态提交和失败回滚；当前状态保存在主机 NumPy 数组；
- `bevformer_postprocess.py` 的显式 NumPy CPU NMS-free Top-K 300、box denormalization、score threshold 与 post-center-range 过滤；coder 级输出保留中心 z，另以 `shift_detections_to_bottom_center` 显式对齐官方 `BEVFormerHead.get_bboxes` 的底面 z。

最终版本在 RTX 3080 上强制启用 CUDA 的全套结果为 `Ran 287 tests in 18.881s ... OK (skipped=1)`。官方 `bevformer_tiny_epoch_24.pth` 已完成严格权重映射、参考 golden 生成和生产尺寸 HXP cold/warm 真机对比。HXP 融合前图有 876 个节点；39 组安全 Add+ReLU 融合后为 837 个执行节点，零拷贝后实际设备 dispatch 为 703，内部 arena 保持 `552,960,000` bytes；随后还接入 FP32 CuPy/cuBLAS MatMul 和逐 launch Profiler。Tensor 计划总量为 `705,177,100` bytes，cuDNN 另有 8 MiB workspace budget，`713,565,708` bytes 是计划预算/准入值，不是严格设备峰值上界。全生成 cold/warm 中位数分别为 `425.325073 / 426.032532 ms GPU` 与 `436.881591 / 437.718436 ms wall`；库加速 cold/warm 分别为 `57.624464 / 56.823263 ms GPU` 与 `69.403093 / 68.747440 ms wall`。ONNX/`torch.export` 捕获、FP16/Tensor Core 主计算、真实 nuScenes 数据集精度和桌面 RTX 3070 的设备专属性能仍未完成。

因此当前准确描述是：“固定 Profile 的 BEVFormer-tiny cold/warm raw step 已用官方 checkpoint 在 RTX 3080 SM86 上通过 HXP 真机数值验收，并完成首轮 FP32 cuDNN/cuBLAS 性能优化；NMS-free Top-K/box denormalization 是显式 CPU 后处理。”这证明了目标 ISA 的编译、运行与首轮库加速闭环，不构成桌面 RTX 3070 性能结论。真实 nuScenes 指标、混合精度/Tensor Core 和桌面 RTX 3070 的设备专属显存/性能仍需继续完成。

### 2.2 当前实现重点

当前实现重点已经转向第二阶段性能深化和真实数据验收：以本轮 FP32 cuDNN/cuBLAS 结果作为新基线，引入混合精度/Tensor Core、进一步融合与 I/O 驻留，补充真实 nuScenes 数据集指标，并在桌面 RTX 3070 上测量显存与端到端性能。

当前只声明固定 Profile 的 raw step 已通过官方 checkpoint 的物理 SM86 数值验收；CPU 后处理边界、真实 nuScenes 精度以及桌面 RTX 3070 的性能结论均继续单独表述。

### 2.3 当前 `gac-bevformer` 命令边界

`build_bevformer.py` 当前实际暴露 22 个子命令：

- 合同与权重：`contract`、`inspect-weights`；
- 图像与 Encoder：`backbone`、`feature-prep`、`encoder-inputs`、`temporal-attention`、`spatial-attention`、`encoder-ffn`、`encoder-layer`、`encoder-stack`、`history-encoder`；
- Decoder 与 raw head：`decoder-cross-attention`、`decoder-self-attention`、`decoder-layer`、`decoder-stack`、`decoder-refinement`、`decoder-query-init`、`detection-head`、`detection-stack`、`decoder-pipeline`；
- 组合 pipeline：`cold-pipeline`、`step-pipeline`。

`encoder-stack` 的 `--mode history|cold` 选择三层 Encoder 的 value queue 合同；`history-encoder` 构建接收外部运动补偿历史的 image-to-BEV 图；`decoder-pipeline` 构建六层 Decoder+六层 raw head；`cold-pipeline` 构建 cold 单帧 image-to-raw core；`step-pipeline` 构建统一 cold/warm 五输入三输出 streaming step。`decode_nms_free` 仍是包外显式 CPU API，故 step HXP 输出固定 Shape raw tensor，不包含最终可变长检测记录。

## 3. BEVFormer-tiny 固定 Profile

### 3.1 模型 Profile

首版 Profile 命名为：

```text
bevformer-tiny-nuscenes-b1-cam6-480x800-bev50-fp16-sm86-v1
```

固定参数如下：

| 参数 | 首版值 |
| --- | --- |
| Batch | 1 |
| 摄像头数 | 6，顺序由 manifest 固定 |
| 缩放后图像内容尺寸 | 450 × 800 |
| Padding 后输入与投影尺寸 | 480 × 800，NCHW、连续布局 |
| Backbone | ResNet-50 |
| FPN level 数 | 1 |
| Embedding | 256 |
| BEV 网格 | 50 × 50，共 2500 个 query |
| Encoder | 3 层 |
| Decoder | 6 层 |
| Object query | 900 |
| 类别数 | 10 |
| Attention head | 8，单 head 32 维 |
| Temporal BEV queue | 当前帧和一个历史 BEV，共 2 份 value |
| 目标精度 | image/state 公开存储边界为 FP16；当前核心计算主要为 FP32，后续再验证 Tensor Core 混合精度 |
| 目标设备 | CUDA Compute Capability 8.6 |

`480 × 800` 是把原始 `1600 × 900` 图像缩放 0.5 后，再按 32 对齐 Padding 得到的编译输入尺寸。原始数据管线在 Padding 后把 `img_shape` 更新为 `480 × 800`，`point_sampling` 也使用这个尺寸归一化投影坐标；Export Adapter 必须保持该参考语义。`450 × 800` 作为缩放后、Padding 前的内容尺寸单独记录，不能与投影尺寸混用。

Profile 必须记录配置文件摘要、checkpoint 摘要、摄像头顺序、归一化参数、坐标系约定和编译器版本。只写“BEVFormer-tiny”不足以确定可复现模型。

### 3.2 纯计算图 ABI

导出适配器把 Python `img_metas`、NumPy 数组和可选 `prev_bev` 改写为显式 Tensor。首版纯计算图入口定义为：

```text
bevformer_tiny_step(
    images,              # f16[1, 6, 3, 480, 800]
    lidar2img,           # f32[1, 6, 4, 4]
    can_bus_delta,       # f32[1, 18]
    prev_bev,            # f16[1, 2500, 256]
    has_prev_bev         # i32[1]，只允许 0 或 1
) -> (
    cls_logits,          # f32[1, 900, 10]，只保留最后一层 decoder
    bbox_regression,     # f32[1, 900, 10]
    next_bev             # f16[1, 2500, 256]
)
```

合同说明：

- `images` 已完成 resize、颜色顺序转换、Normalize 和 Padding；首版不把 JPEG 解码和图像预处理编入模型图。
- 原始代码 Padding 后的 `img_shape` 在首版 Profile 中固定为 `480 × 800`，由编译期配置用于参考点投影归一化，不占用公开 Runtime 输入。
- `lidar2img` 和投影 MatMul 使用 FP32。原实现会主动关闭该路径的 TF32，首版保持这一数值语义。
- `can_bus_delta[:3]` 是相对上一帧的位置变化，`can_bus_delta[-1]` 是帧间旋转角；首次运行或 reset 后只清零这四个运动分量。四元数、`can_bus_delta[-2]` 的绝对航向及其他 CAN-bus 特征保留给 learned CAN-bus MLP，这与官方测试路径一致。
- `prev_bev` 始终存在，使用 `has_prev_bev` 区分有效和无效状态，避免 Optional Tensor 和数据相关图结构。
- Temporal Self Attention 保留官方逐层语义：warm 时三层 Encoder 共享 `[motion-compensated prev_bev, initial query]`；cold 时每层都从该层输入的 current query 动态构造 `[current query, current query]`，不能把进入 Encoder 前的 query 作为三层共享 cold queue。
- 外部 ABI 统一采用 `[B, HW, C]` 的 BEV 布局。导出适配器负责吸收原始 PyTorch 代码中 `[HW, B, C]` 的排列差异。
- 分类和回归结果先以固定 Shape 返回。`bbox_regression` 保留 `[physical-x, physical-y, log-w, log-l, physical-z, log-h, sin-yaw, cos-yaw, vx, vy]` raw 表示。显式 CPU `decode_nms_free` 已实现 Top-K 300、box denormalization 和范围过滤；step 图和 HXP metadata 均记录这条主机边界。
- 所有 Tensor 必须连续、地址满足后端对齐要求，Shape 与 DType 不匹配时应在提交前失败。

### 3.3 Runtime 状态 API

编译后的设备程序保持纯函数；`BevFormerRuntime` 在设备 Session 外管理跨帧状态。当前上层接口是：

```python
runtime = BevFormerRuntime(session)
runtime.reset(scene_token="scene-001")
result = runtime.step(
    images=images,
    lidar2img=lidar2img,
    can_bus=absolute_can_bus,
    scene_token="scene-001",
)
```

`BevFormerState` 至少保存：

```text
scene_token
valid
prev_bev       f16[1, 2500, 256]，当前为主机 NumPy 数组
last_position  f32[3]
last_angle     f32
frame_index
```

状态规则：

1. `reset()` 清零 `prev_bev`、上一位置和角度，并令 `valid=false`。
2. `step()` 比较 scene；scene 改变时自动建立该 scene 的 cold 状态，不沿用上一场景状态。
3. 第一次 `step()` 将 `can_bus_delta[:3]` 和 `can_bus_delta[-1]` 清零、保留其余绝对/辅助 CAN-bus 特征，并以 `has_prev_bev=0` 调用设备程序。
4. 后续帧由 Runtime 根据当前绝对位姿与上次位姿计算 delta，以 `has_prev_bev=1` 调用设备程序。
5. 只有 Session 执行和三个输出的 Shape/DType/连续性/有限值检查成功后，才用 `next_bev` 和当前位姿替换内部 state。异常在提交前传播，旧状态保持不变。
6. 当前 `prev_bev` 保存在 Runtime 主机内存并通过 Session 输入提交；显存驻留和跨 invocation I/O 预绑定是后续性能优化。
7. 一个 Runtime 只允许一个进行中的 invocation；并发场景使用不同 Runtime 实例。

该 API 把原始 `BEVFormer.prev_frame_info` 中隐含的 Python 状态变成可测试、可重放的 Runtime 合同。

## 4. 首版编译链

```text
PyTorch BEVFormer-tiny + checkpoint
            │
            ▼
BEVFormer Export Adapter
  - eval/inference only
  - img_metas Tensor 化
  - state 显式化
  - 自定义算子标记
            │
            ▼
Canonical Graph
  - Tensor/Shape/DType/Layout
  - source location
  - state input/output 标记
            │
            ▼
BEV Pass Pipeline
  - 规范化、常量折叠、模式识别
  - Layout、混合精度、能力检查
            │
            ▼
CUDA SM 86 Lowering
  - CUDA library call
  - generated CUDA C
  - BEV custom kernel
            │
            ▼
NVRTC artifact + launch metadata
            │
            ▼
BevFormerSession
  - resident weights/state
  - buffer plan
  - dispatch/trace/error
```

### 4.1 Export Adapter

原始 BEVFormer 代码包含 Python 字典/列表、NumPy 运动学计算、`torchvision.rotate`、按相机循环、数据相关 `nonzero/max_len` 和 MMCV `_ext` 自定义 CUDA 函数。普通 ONNX 导出无法自然保留全部语义。因此 Export Adapter 必须：

- 将 `img_shape`、`lidar2img`、`can_bus` 和状态改为具名 Tensor 输入；
- 将旋转和平移补偿规范化为 Tensor Op；
- 将多尺度可变形注意力标为 `gac.ms_deform_attn`，禁止展开成不可识别的 Python/CUDA 扩展调用；
- 将相机 Query 的 gather/rebatch/scatter 改写为规范算子；
- 首版用固定上界和 Mask 代替由 `nonzero` 产生的运行时 `max_len`；
- 删除训练分支，只保留最后一层检测输出和 `next_bev`；
- 保存源节点到 Canonical Graph 的映射，支持逐层对比。

原始项目依赖较老的 PyTorch/MMCV 组合。实现时可以先使用受控的 Python 捕获适配器生成 Canonical Graph，再增加 ONNX 或 `torch.export` 入口；不应为了声称支持标准格式而丢失时序和自定义算子语义。

### 4.2 Canonical Graph 与 BEV Op 集

首版完整模型所需的核心算子族包括：

- Conv2D、BatchNorm folding、ReLU、FPN 上采样和逐元素计算；
- Linear/MatMul、Bias、LayerNorm、Softmax、Sigmoid；
- Reshape、Transpose、Concat、Slice、Expand、Reduce、TopK；
- 参考点生成、4×4 投影、`Greater`/`Less` 比较与可见性 Mask；
- BEV Rotate/GridSample；
- Masked Gather、Camera Rebatch、Indexed Add/Scatter；
- `MSDeformableAttention`；
- Temporal BEV merge 和最终 box decode。

每个 Canonical Op 都必须有：Shape/DType 推导、参考语义、后端能力检查、失败诊断和来源信息。算子名称出现在 IR 中不等于 CUDA 后端已经支持该算子。

### 4.3 Pass Pipeline

BEVFormer Profile 的 Pass 顺序固定并版本化：

```text
1. ImportValidation
2. InferenceGraphPruning
3. MetadataTensorization
4. ExplicitStateLowering
5. ShapeAndTypeInference
6. ConstantFolding
7. CanonicalizeViewAndTranspose
8. FoldConvBatchNorm
9. RecognizeMSDeformableAttention
10. RewriteCameraQueryCompaction
11. LayoutPropagation
12. MixedPrecisionAssignment
13. FusionAndLibrarySelection
14. SM86CapabilityCheck
15. BufferLifetimeAndMemoryPlan
16. CUDAArtifactLowering
```

关键规则：

- BEV reference grid、`spatial_shapes`、`level_start_index`、camera/level embedding 等静态内容尽量离线常量化。
- Layout Pass 在子图范围内选择布局，避免每个算子前后反复 Transpose。
- FP16 不是全图强制转换。投影、Softmax、LayerNorm 和 deformable attention 累加根据逐层误差采用 FP32。
- Query compaction 的首版实现必须有固定内存上界。可以保留全部 2500 query 并用 Mask 屏蔽，不能把不可控的运行时长度带入静态内存计划。
- Capability Check 在生成代码前执行。未支持的组合必须带节点、Shape、DType 和拒绝原因失败。

### 4.4 CUDA SM 86 后端

后端由三类 Kernel provider 组成：

| Provider | 用途 | 首版策略 |
| --- | --- | --- |
| CUDA Library | Conv、GEMM 等成熟密集计算 | 已为生产 FP32 Conv 接入 cuDNN、为 FP32 MatMul 接入 CuPy/cuBLAS；继续补充版本/Shape 合同与混合精度候选 |
| Generated CUDA | 逐元素、Layout、简单归约、直接 Conv/MatMul | 延续现有 CUDA C + NVRTC；密集算子的 generated 实现保留为正确性基线与不支持 Shape 的明确候选 |
| BEV Custom Kernel | MSDeformableAttention、Query gather/scatter | 针对 SM 86 编写并单独验证 |

编译后端只接受 Compute Capability 8.6，不以设备名称推断能力。Artifact manifest 至少记录 target、CUDA/NVRTC 选项、入口、静态 Shape、DType、对齐、临时空间、源码摘要和 checkpoint 摘要；每个 `KernelLaunch` 还显式列出 `implementation_candidates`（如 `cudnn`、`cupy-cublas`、`generated-cuda`）。Runtime 加载时重新检查实际设备、驱动和所需显存，并把最终选择写入 `session.info`、`kernel_info` 和 profile。

首版保持单设备、单 invocation 串行语义。多 stream 只有在依赖图、buffer 生命周期和事件验证完成后再开启。

### 4.5 Runtime 与内存计划

Runtime 管理以下几类显存：

- 常驻权重和 embedding；
- 当前帧输入及 backbone/FPN 激活；
- `prev_bev` / `next_bev` 双缓冲；
- attention workspace；
- 固定输出和 Trace 元数据。

离线编译器已经通过生命周期分析生成确定性的静态 arena 计划，并把外部输入/输出、常驻常量和可复用中间值分开记录；HXP 加载会校验该计划。`CudaSession` 使用同一规划器计算所需字节数，在初始化时检查可用显存、一次性分配内部 arena，并按 allocation offset 创建中间 Tensor view；内部 Reshape 和可证明为连续区间的 Slice 直接复用源 allocation，不启动复制 Kernel。外部输入、外部输出和常量使用各自的独立存储类别。生产尺寸统一 step 已在 RTX 3080 上按 `705,177,100` bytes 的 Tensor 计划完成 HXP 加载和 cold/warm 执行；cuDNN 的 8 MiB（`8,388,608` bytes）workspace budget 单独列出，合计计划预算/准入值为 `713,565,708` bytes。该值不含 module、CUDA context、Event 和 vendor 临时分配，桌面 RTX 3070 的实际峰值仍须单独测量。超出预算必须明确失败，不能依赖 CUDA OOM 作为正常控制流；输入、权重、状态和临时 Buffer 的所有权必须可追踪。

## 5. 首个可运行切片：MSDeformableAttention

`MSDeformableAttention` 是首个实现切片，因为它同时覆盖 BEVFormer 最关键的非标准计算、复杂索引、双线性采样和 CUDA 后端扩展机制。当前已经完成 FP32 基线并通过 SM86 真机验证。参考源码在 CUDA 路径中始终选用带 `custom_fwd(cast_inputs=torch.float32)` 的函数，因此首版保持 MSDA 全部输入和累加为 FP32；公开图像/状态边界使用 FP16，当前 Conv/MatMul 主计算仍为 FP32。该切片通过验收也不代表 Backbone、Encoder、Decoder 或全模型已经可运行。

### 5.1 Canonical Op 合同

```text
MSDeformableAttention(
    value,               # T[N, S, H, D]
    sampling_locations,  # T[N, Q, H, L, P, 2]
    attention_weights;   # T[N, Q, H, L, P]
    spatial_shapes       # 编译期属性：[[H0,W0], ...]
) -> output              # T[N, Q, H * D]
```

其中：

- `T` 在 MSDA 边界固定为 FP32，与本地 BEVFormer 参考源码的强制转换一致；
- BEVFormer-tiny 固定 `H=8`、`D=32`、`L=1`；
- Spatial Cross Attention 使用 `P=8`，Temporal/Decoder 路径还需要 `P=4`；
- `N`、`S`、`Q` 在每个已编译 Profile 中为静态值；
- `sum(spatial_shapes[:,0] * spatial_shapes[:,1]) == S`；
- `level_start_index` 由编译器根据 `spatial_shapes` 静态推导，不占用 Runtime 输入；
- 所有位置坐标使用归一化 `[0,1]` 语义，越界样本贡献为零；
- 对第 `l` 层，像素坐标按 `x = loc_x * W_l - 0.5`、`y = loc_y * H_l - 0.5` 解释，再执行四邻域双线性插值；
- 输出对 level 和 point 维按 `attention_weights` 加权归约。

首个 Kernel 只实现 forward。Backward 不属于推理编译器范围。

### 5.2 切片编译路径

```text
JSON/内部测试图
  -> gac.ms_deform_attn Canonical Op
  -> Shape/DType/属性校验
  -> SM86 capability query
  -> CUDA source + launch metadata
  -> NVRTC compile
  -> CudaSession dispatch
  -> output + CUDA Event + Trace
```

该路径先使用独立算子测试图，不等待 BEVFormer 全图导出。当前 FP32 基线已经落实：

1. Canonical IR 用节点属性保存静态 `spatial_shapes`；
2. 独立 NumPy 定义覆盖中心点、四邻域、越界补零、多 level/head/point；
3. CUDA 后端生成 SM86 FP32 forward Kernel，并明确拒绝不支持的 DType/Shape；
4. launch metadata 记录输入、输出、grid/block 和源节点；
5. Session 保持无隐式 CPU fallback；可选逐 launch CUDA Event Profiler 把实际 implementation 明细，以及按 op / implementation 聚合写入机器可读 JSON，Trace 同时保留模型来源。

Spatial Cross Attention 的固定上界 mask/rebatch 子图及 Temporal Self Attention 静态子图已组合成三层 Encoder；对应的 self/cross attention、FFN/norm 和 refinement 也已组合成六层 Decoder，后接六层 raw detection projection。warm 多帧 rotate/shift、cold/warm 选择、FP16 image/state 边界及三层 Encoder→六层 Decoder→raw head 已在统一 step 图中贯通。官方 checkpoint 与 MMCV/PyTorch 参考环境生成的 cold/warm golden 已用于 RTX 3080 SM86 HXP 数值验收；cold 和 warm 分支分别保留逐层动态 queue 与共享 history queue 语义。

### 5.3 验收标准

首个切片只有同时满足以下条件才可标为“已实现”：

- 固定 seed 的随机样例、边界坐标、越界坐标和非整齐 Shape 均通过；
- 与本地 BEVFormer/MMCV 的 PyTorch 参考路径逐元素对比；
- FP32 初始门槛：最大绝对误差不大于 `1e-4`，Cosine Similarity 不小于 `0.99999`；
- 外围 FP16 路径进入 MSDA 前必须显式转换为 FP32，离开该算子后的降精度转换需要单独数值门槛；
- 在真实 SM 86 GPU 上执行，设备不满足时明确失败；
- Trace 能从 dispatch 回溯到 `gac.ms_deform_attn` 源节点，并记录 Shape、DType、grid/block、workspace 和 CUDA Event 时间；
- 重复运行结果稳定，输入不被修改，越界访问检查通过；
- 报告编译时间、首次运行、warm-up 后 Kernel 时间和端到端时间，不把独立算子数据宣称为全模型性能。

若业务层逐层误差或最终 mAP 证明上述门槛不足，应收紧精度或把更多中间量提升为 FP32，而不是放宽结果解释。

### 5.4 2026-10-10 真机结果

测试设备为 `NVIDIA GeForce RTX 3070 Ti Laptop GPU`，Compute Capability 8.6，显存 8,220,901,376 bytes。测试 Shape 为真实 tiny 上界切片：

```text
value                f32[6, 375, 8, 32]
sampling_locations   f32[6, 2500, 8, 1, 8, 2]
attention_weights    f32[6, 2500, 8, 1, 8]
output               f32[6, 2500, 256]
```

25 项本地语义、编译器和物理 CUDA 测试全部通过。固定 seed=3070，预热 10 次并测量 50 次：最大绝对误差为 `3.5762787e-7`；GPU stream 中位数 `3.561 ms`；包含输入校验、H2D、分配、执行和 D2H 的 wall time 中位数 `8.292 ms`。独立带 Trace 的 Kernel 时间为 `0.328 ms`。这些数字只代表一个 FP32 MSDA 切片，不是 BEVFormer 整模延迟。

原始证据保存在：

- `docs/validation/sm86/bevformer/bevformer-msda-sm86.json`
- `docs/validation/sm86/bevformer/bevformer-msda-trace.json`
- `docs/validation/sm86/bevformer/bevformer-tests.txt`
- `docs/validation/sm86/bevformer/reference-manifest.json`

原始 BEVFormer 参考栈的版本核对与本轮环境记录见 `docs/bevformer_reference_environment.md`。

### 5.5 2026-10-10 官方 checkpoint 全图结果

本轮在 Ubuntu 22.04、NVIDIA GeForce RTX 3080（Compute Capability 8.6）上，以官方 `bevformer_tiny_epoch_24.pth` 构建并运行统一 cold/warm step HXP。融合前图有 876 个节点；安全 Add+ReLU 融合 39 组后为 837 个执行节点，优化版再移除可证明安全的 Reshape/连续 Slice 复制 launch，实际设备 dispatch 为 703，内部 arena 保持 `552,960,000` bytes。最终强制 CUDA 的完整测试输出为 `Ran 287 tests in 18.881s ... OK (skipped=1)`，Conv/MatMul 分别选择 cuDNN 与 CuPy/cuBLAS vendor library。最终 HXP 为 `137,896,143` bytes，SHA-256 是 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`。

| 模式 | `cls_logits` max abs | `bbox_regression` max abs | `next_bev` max abs |
| --- | ---: | ---: | ---: |
| cold | 0.003891945 | 0.004856288 | 0.001953125 |
| warm | 0.000953674 | 0.000725389 | 0.001953125 |

| 路径 | 全生成 GPU / wall | 优化版 GPU / wall | GPU / wall 加速比 |
| --- | ---: | ---: | ---: |
| cold | 425.325073 / 436.881591 ms | 57.624464 / 69.403093 ms | 7.38098× / 6.29484× |
| warm | 426.032532 / 437.718436 ms | 56.823263 / 68.747440 ms | 7.49750× / 6.36705× |

误差以官方 PyTorch/MMCV golden 为基准。表中全生成与优化版分别是 3 次和 10 次稳态运行的中位数。fresh-process 中 Session 构建完成后的首个 cold `run` 为 `277.443573 ms GPU / 289.668472 ms wall`，不包含 Session 构建时间；同一进程已预热后的 `116.635651 ms` 不作为 fresh-process 数据。带逐 launch Event 的独立 Profile 为 `76.05085 ms GPU / 95.8321 ms wall`，Kernel Event 总和 `63.4391 ms`，用于归因而不与普通基准直接比较。NMS-free Top-K 与 box denormalization 仍在显式 CPU 主机边界。

显式 128 MiB cuDNN workspace 实验把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图的交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，没有端到端收益，并比当前 8 MiB budget 多占 120 MiB，因此本轮不集成。性能深化优先做 CUDA Graph、批量 dispatch 和更多融合；当前 `cupyx.cudnn` 与 capture 不兼容，未来若要捕获 cuDNN 路径，需要稳定的低层 API。

## 6. 从切片到全模型的里程碑

| 里程碑 | 当前状态 | 交付内容 | 完成标准 |
| --- | --- | --- | --- |
| M0：设计冻结 | 已完成 | 本文、固定 Profile、显式 State ABI | 文档与本地 tiny 配置一致；明确未实现项 |
| M1：MSDA 切片 | 历史真机已验证 | Canonical Op、参考实现、SM86 Kernel、真机测试 | 满足第 5.3 节；只宣称算子切片可运行 |
| M2：导出适配器 | 合同已离线实现，捕获未完成 | 配置/元数据 Tensor 化、state 显式化、custom op 捕获 | 可稳定导出相同节点/权重摘要；无 Python Runtime 语义泄漏 |
| M3：视觉 Backbone | 规范图、严格导入合同、generated 正确性路径及 FP32 cuDNN 真机执行已完成 | R50 + 单层 FPN + 基础 Tensor/CNN/Linear/Norm Op | 继续完成混合精度/Tensor Core 与更广 Shape 策略 |
| M4：单帧 BEV Encoder | cold image-to-BEV 已进入官方 checkpoint HXP 验收 | 投影、Mask、Spatial/Temporal Attention、FFN、3 层 Encoder | `has_prev_bev=0` 的最终 raw 输出与 `next_bev` 已和官方参考对齐 |
| M5：时序状态 | history Encoder、rotate/shift、逐层 warm queue、Runtime 状态提交/失败回滚已进入真机验收 | Rotate/shift、Temporal Self Attention、主机 `prev_bev` 状态 | warm 最终 raw 输出与 `next_bev` 已对齐；设备驻留优化待完成 |
| M6：Decoder 与检测 | 六层 Decoder 和 raw projection 已随 cold/warm HXP 通过最终 raw 输出对比；CPU NMS-free decode 显式存在 | 900 query、迭代 reference refinement、raw heads、Top-K 300/box decode | 真实 nuScenes boxes/scores/labels 及 CPU 边界 Trace 仍待完成 |
| M7：整模验收 | 官方 checkpoint 的五输入三输出 cold/warm HXP 已在 RTX 3080 SM86 通过数值验收 | step 模型包、arena 真机验收、完整 Trace、nuScenes 子集 | raw step 与首轮 FP32 性能优化已验收；桌面 RTX 3070、真实 nuScenes 与 P50/P95 仍待完成 |
| M8：性能深化 | 首轮已完成 | Reshape/连续 Slice 零拷贝、39 组 Add+ReLU 融合、FP32 cuDNN Conv、FP32 cuBLAS MatMul、逐 launch Profiler；后续 CUDA Graph、批量 dispatch、更多融合、FP16/Tensor Core、双缓冲、多 stream | cold GPU/wall `7.38098× / 6.29484×`，warm `7.49750× / 6.36705×`；每项后续优化继续保留数值回归 |

对外描述应限定为“固定 Profile 的 BEVFormer-tiny raw step 已在 RTX 3080 SM86 上通过官方 checkpoint 数值验收”。在真实 nuScenes 精度、显式 CPU 后处理边界和桌面 RTX 3070 结果补齐前，不将其简写为没有边界条件的“完整 BEVFormer 已支持”。

## 7. 全模型验证策略

验证分四层进行：

```text
PyTorch / MMCV Reference
        ↓
Canonical Op Reference
        ↓
Generated CUDA on SM86
        ↓
BEVFormer-tiny 多帧端到端结果
```

每个里程碑保存：

- 输入数据摘要、配置和 checkpoint hash；
- Pass pipeline 版本和编译器版本；
- 各关键层 Shape、DType、最大绝对/相对误差和 Cosine Similarity；
- warm-up 次数、测量次数、Kernel/传输/端到端耗时；
- 峰值显存、workspace、状态显存和权重显存；
- 实际 GPU 名称、Compute Capability、驱动、CUDA Runtime 和 NVRTC 版本；
- Trace 与失败诊断。

全模型最终还需检查 nuScenes NDS/mAP 的变化。单算子误差通过不代表检测精度通过；单帧结果通过也不代表跨帧状态正确。

## 8. 主要风险与处理

### 导出风险

原始模型依赖 Python 控制和 MMCV 自定义扩展。处理方式是使用专门 Export Adapter 和 Canonical Custom Op，不强求第一步得到纯标准 ONNX。

### 动态 Query 风险

按相机可见性做 `nonzero` 和动态 `max_len` 会破坏静态内存计划。首版使用固定 2500 query 上界和 Mask；在正确性稳定后，再评估 compaction 是否带来真实收益。

### 数值风险

Deformable attention、Softmax、LayerNorm 和多次归约对 FP16 敏感。首版采用选择性 FP32 累加，并以逐层误差和 mAP 决定精度，不采用全图统一降精度。

### 显存风险

BEVFormer-tiny 仍包含六路图像特征和 Transformer 激活。编译器必须生成生命周期明确的静态内存计划，Runtime 必须在加载时根据真实可用显存校验。未测量前不承诺固定峰值。

### 性能风险

独立 Kernel 更快不代表端到端更快。性能报告必须包含预处理边界、H2D/D2H、状态维护、同步和后处理；不与成熟框架做缺少同等条件的比较。

### 兼容性风险

RTX 3070、3070 Ti Laptop 和本轮使用的 RTX 3080 都是 SM 86，但显存、功耗和吞吐不同。编译兼容性由 Compute Capability 决定，性能和内存验收必须分别记录真实设备信息；RTX 3080 的数值验收不能替代桌面 RTX 3070 的设备专属测量。

## 9. 完成定义

只有满足以下全部条件，项目才可以声明“首版支持 BEVFormer-tiny”：

1. 固定 Profile 可从冻结配置和 checkpoint 可复现编译；
2. 六相机单帧及连续多帧均在真实 SM 86 GPU 上执行；
3. `reset`、scene 切换、失败回滚和状态驻留通过测试；
4. Backbone、Encoder、Temporal State、Decoder 和 Top-K 输出均无隐式 CPU 计算；若存在显式 CPU 后处理，文档和 Trace 必须准确标出；
5. 逐层数值与端到端 boxes/scores/labels 通过参考对比；
6. nuScenes 验证子集的精度变化在预先冻结的门槛内；
7. 模型加载、峰值显存、端到端延迟和 Trace 报告可重复生成；
8. 不支持的 Shape、DType、设备或模型版本能够明确拒绝。

当前仓库已完成该定义中的固定 Profile 编译、单帧/连续帧 SM86 执行、状态提交/回滚、官方 checkpoint raw 输出对比、生产 HXP/arena 真机运行及首轮 FP32 cuDNN/cuBLAS 优化。尚缺的是第 5、6、7 项中与真实 nuScenes boxes/scores/labels、数据集精度、完整 CPU 后处理 Trace 和桌面 RTX 3070 设备测量相关的部分。下一步以当前 vendor-library 基线推进混合精度与更多融合，并补齐这些设备和数据集验收；在此之前，支持声明继续限定为固定 Profile 的 cold/warm raw step。
