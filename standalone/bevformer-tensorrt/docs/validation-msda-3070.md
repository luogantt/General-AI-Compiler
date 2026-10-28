# MSDA 优化验证：RTX 3070 Ti Laptop

日期：2026-10-21。对象是完整 BEVFormer-base 固定形状、无历史帧的 TensorRT 路径。优化保持采样位置、点数、权重及网络参数不变。

本页保留首轮原始结果；后续 32 位索引优化、重构回归排查与 326.88 ms 的实测见[第二轮报告](validation-msda-round2-3070.md)。

**本轮结果：同输入单个 Spatial MSDA 加速 2.30 倍；完整引擎延时从 413.774 ms 降至 339.086 ms，下降 18.05%。受控整网测试的三个输出逐位一致。** 这只覆盖一个真实六相机输入，不代表 nuScenes 检测指标已经验收。

## 已确认的单算子结果

从真实六相机图像的第一层 Spatial MSDA 捕获五个输入，在同一输入上比较 NVIDIA 插件与 GAC 插件。最终独立复测使用 CUDA Graph replay，预热 10 次，交替测量 40 对：

| 实现 | 中位时间 | 数值结果 |
|---|---:|---|
| NVIDIA `MultiscaleDeformableAttnPlugin_TRT` v2 | 38.9508 ms | 与捕获的官方输出逐位一致 |
| GAC `GACMSDeformAttn` v1 | 16.9093 ms | 与同输入官方插件逐位一致 |

逐对时间比的中位数为 **2.300 倍**。这是一个 Spatial MSDA 的结果，不能直接换算为完整模型 FPS。复测期间 GPU 从 81°C 升至 87°C，SM 时钟降至 915 MHz；绝对时间反映这台笔记本的运行状态。初筛冷一些时约为 29/12 ms，不与本轮混用。

选中参数：

```json
{"channels_per_thread": 4, "block_threads": 128, "specialize": 1, "prefetch": 0}
```

每线程计算四个相邻通道，复用坐标、插值系数及索引，FP16 以两个 `half2` 处理；对 `C=32,L=4,P=8` 专门生成循环。预取仅指下一采样点的坐标和权重，本次选中关闭。没有减少采样点，也没有使用 shared-memory 双缓冲或 Tensor Core。

完整 FP16 网格为 48 组，全部通过本组输入的数值检查。另有 6 组 CPU oracle 对照、150 项 GPU 小形状/通道尾部/边界采样/插件序列化检查通过；FP32 按 `rtol=1e-5, atol=2e-6`，FP16 最终复测按位比较。结果只证明已测输入；更换形状、GPU 或 TensorRT 二进制需重测。

## 完整模型对照

仅将六层 Encoder Spatial MSDA 替换为 GAC，Temporal 和 Decoder 的 12 个 MSDA 保留 NVIDIA 实现。改写证明其余 ONNX 字段、权重和连接不变；校准范围保留原测量值及来源，使用显式派生绑定。

### 初次实验：未通过，因此不接受

第一轮两种引擎从相近温度启动，均预热 5 次、测量 20 次，不使用整网 CUDA Graph：

| 初次对照 | 中位延时 | 起始/结束温度 | 数值检查 |
|---|---:|---|---|
| 旧官方引擎 | 416.244 ms | 56 / 77°C | 比较基准 |
| 新自定义引擎 | 339.806 ms | 55 / 72°C | **未通过** `rtol=0.001, atol=0.001` |

虽然时间下降 18.36%，**这组结果不能作为已验收的整网加速**。两引擎各自重复运行输出一致，但彼此的 BEV/分类/框最大绝对误差为 0.07324/0.02734/0.203125。

层信息检查发现，重新构建还改变了 584 个共同名称层的 tactic，包括 31 个浮点 Conv；六层 Spatial MSDA 上游的 Value 投影也换了实现。实际 MSDA 输入输出的 shape、Half/Int32 类型保持一致，但输入数值可能因上游计算顺序改变。该对照不能把误差或全部时间变化单独归因于 MSDA。

### 受控复测：通过同输入数值检查

重新构建官方引擎并保存 timing cache，再使用该 cache 构建自定义引擎。两侧精度、workspace 1024 MiB、tactic DRAM 4096 MiB、优化级别 3、辅助流 0、TF32 关闭、CUDA lazy loading 一致。Timing cache 保存候选实现的计时信息，与 INT8 校准缓存不同，也不保证所有 tactic 自动保持一致。

按官方—优化—优化—官方顺序测试；每轮 GPU 降至 55°C 后再运行，预热 5 次、测量 20 次：

| 顺序 | 引擎 | 中位延时 | P95 | 结束温度 |
|---|---|---:|---:|---:|
| 1 | 官方 | 413.775 ms | 417.280 ms | 75°C |
| 2 | 优化 | 339.172 ms | 341.172 ms | 73°C |
| 3 | 优化 | 338.740 ms | 340.276 ms | 72°C |
| 4 | 官方 | 413.721 ms | 416.356 ms | 75°C |

合并各引擎两轮的 40 次样本：官方中位 **413.774 ms**，优化中位 **339.086 ms**，延时下降 **18.05%**、加速 **1.220 倍**。延时倒数约从 **2.42 FPS** 到 **2.95 FPS**。这里计时不含输入输出拷贝、预处理、加载、构建及框解码，没有整网 CUDA Graph。

`bev_embed`、`all_cls_scores`、`all_bbox_preds` 的比较均满足：

- 新自定义引擎与新官方引擎逐位一致，最大绝对误差为 0。
- 新官方引擎与原官方引擎逐位一致。
- 两种引擎各自重复运行逐位一致。

没有放宽数值阈值。四个运行报告的 `validation` 因未传 `--reference` 而是 `not_checked`，跨运行数值验收记录在单独的 [comparison.json](validation/msda-2026-10-21/controlled/comparison.json)，不能把命令返回成功当成数值验证。

层信息审计：两边均为 3816 层，全部 18 个 MSDA 的输入输出描述一致，其他层的 tactic、类型、布局及 stream 差异为 0；六个 Spatial 上游 Value 投影也保持相同。验证六个临时 tensor 重命名前后的所有生产者/消费者关系后，字段差异只余六个 MSDA 的插件类型/版本。另有 64 个 `NonZero`/`DeviceToShapeHost` 记录的 inspection 顺序不同，不能把层列表当成实际 GPU 指令轨迹。

这证明本轮相同输入下的完整引擎实现等价。GPU 未锁频且温度会变化，因此保留各次原始时间和环境；整网收益属于两份完整构建结果，不能把全部变化严格归因于一个 Kernel。没有对整个数据集或量化检测精度作出结论。

## 复现边界与证据

- GPU：RTX 3070 Ti Laptop，8 GB，SM86；驱动 580.178.04。
- TensorRT：10.13.3.9；Torch 2.7.0+cu126；自定义插件使用 CUDA 12.6 工具链和官方 `release/10.13` 头文件构建。
- 原图输入：`[1,6,3,928,1600]`。捕获 Value：`[6,30825,8,32]`；locations：`[6,40000,8,4,8,2]`；weights：`[6,40000,8,4,8]`。
- 虽然使用真实图像，导出图的相机几何、CAN-bus 和无历史帧限制不变。新增捕获输出也可能改变 TensorRT tactic，因此捕获引擎不是原整网引擎的无扰动观测。
- 未采集 Nsight Compute 硬件计数器；不能仅凭时间判断带宽、缓存命中或 occupancy 的具体瓶颈。
- 本次没有 nuScenes mAP/NDS 评估。原混合 INT8/FP16 引擎对 FP32 的精度差异也未由本次优化解决。

可复查的报告：

- [48 组搜索原始结果](validation/msda-2026-10-21/search-fp16.json)
- [最终独立复测、逐位误差及全部时间样本](validation/msda-2026-10-21/retest-fp16.json)
- [真实输入捕获与来源 SHA](validation/msda-2026-10-21/capture.json)
- [half FMA 舍入模式诊断](validation/msda-2026-10-21/half-arithmetic-analysis.json)
- [六节点改写与结构证明](validation/msda-2026-10-21/rewrite-spatial6.json)
- [初次旧引擎运行](validation/msda-2026-10-21/initial-stock-run.json) / [初次新引擎运行及失败误差](validation/msda-2026-10-21/initial-custom-run.json)
- [受控整网汇总](validation/msda-2026-10-21/controlled/summary.json) / [四轮运行协议](validation/msda-2026-10-21/controlled/protocol.json) / [跨运行数值比较](validation/msda-2026-10-21/controlled/comparison.json)
- [官方第 1 轮](validation/msda-2026-10-21/controlled/stock-1.json) / [优化第 1 轮](validation/msda-2026-10-21/controlled/custom-1.json) / [优化第 2 轮](validation/msda-2026-10-21/controlled/custom-2.json) / [官方第 2 轮](validation/msda-2026-10-21/controlled/stock-2.json)
- [受控层信息及 tactic 审计](validation/msda-2026-10-21/controlled/tactic-audit.json)；完整 build/layers 原始文件保留在项目 `build/msda-optimization-20261021/controlled-build-reports/`，SHA 写入审计记录。复现审计时，将四个 JSON 放入 [审计脚本](validation/msda-2026-10-21/controlled/audit_controlled_tactics.py) 旁的 `controlled-build-reports/` 再运行。

构建和复测命令见 [MSDA 实验流程](msda-optimization.md)。代码入口为 [CUDA kernel](../native/msda/msda_kernel.cu)、[TensorRT 插件](../native/msda/plugin.cpp)、[捕获脚本](../scripts/capture_msda.py)、[基准脚本](../scripts/benchmark_msda.py) 和 [图改写/校准范围证明](../src/bevformer_trt/msda.py)。
