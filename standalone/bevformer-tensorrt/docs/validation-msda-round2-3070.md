# MSDA 第二轮优化：从单算子到完整引擎

设备仍为 RTX 3070 Ti Laptop 8 GB，TensorRT 10.13.3.9、CUDA 12.6。继续使用同一份 BEVFormer-base 混合 INT8/FP16 固定图和真实六相机输入；相机几何固定、无历史帧，输出止于检测头张量。

**最终结果：修复重构引入的回归后，整网从本轮复测的 335.60 ms 降到 326.88 ms，延时再降低 2.60%，约 3.06 FPS。三个输出与第一轮原库及官方插件参考逐位一致。** 首轮的 18.05% 属于另一组对照，不与本轮百分比相加。[最终数据](validation/msda-2026-10-21/round2/final/summary.json)

## 最终采用的配置

```json
{"channels_per_thread":4,"block_threads":128,"specialize":3,"prefetch":0}
```

仅用于六个 Spatial MSDA；其余节点和精度策略保留。每轮预热 5 次、测试 30 次，运行前温度 ≤55°C、GPU 利用率 <10%。使用原二进制作基线，分别启动进程加载对应插件。

| 顺序 | 引擎／库 | 中位延时 |
|---|---|---:|
| A1 | 第一轮原引擎／原库 | 333.975 ms |
| X | 第一轮原引擎／修复库，原参数 | 336.732 ms |
| B1 | 32 位配置引擎／修复库 | 326.691 ms |
| B2 | 32 位配置引擎／修复库 | 326.972 ms |
| A2 | 第一轮原引擎／原库 | 337.154 ms |

A1+A2 与 B1+B2 各 60 个样本，中位数分别为 **335.596 / 326.880 ms**。X 是单独的默认路径恢复对照，不混入 A 的样本。X 落在 A1、A2 的波动范围内；两次 B 都快于两次 A。GPU 未锁频，保留每次环境和全部样本，结论只覆盖本次运行条件。

[运行协议](validation/msda-2026-10-21/round2/final/protocol.json) · [逐位比较](validation/msda-2026-10-21/round2/final/comparison.json) · [342 项 GPU 检查](validation/msda-2026-10-21/round2/final/checks.json) · [最终编译资源](validation/msda-2026-10-21/round2/final/resources.json)

修复后 CPT4/half2 无预取路径：64 位 49 个寄存器、32 位 40 个，均无 spill。最终插件库必须与引擎一起记录；本次修复沿用同一 32 位配置 engine，**只看 engine SHA 无法区分失败版本与修复版本**，库 SHA 见最终汇总及每轮 report。

## 先查当前热点

对第一轮已优化引擎重新 profiling，3816 条回调均按完整名称匹配 TensorRT Inspector；融合层不拆分计时。

| 执行层类别 | 回调 mean 合计 | 回调占比 |
|---|---:|---:|
| 6 个 Spatial MSDA | 76.84 ms | 22.94% |
| CaskConvolution | 64.89 ms | 19.37% |
| kgen 融合计算 | 61.88 ms | 18.47% |
| DCNv2 插件 | 57.09 ms | 17.04% |
| Reformat | 27.24 ms | 8.13% |
| GEMM | 23.78 ms | 7.10% |
| 其余 12 个 MSDA | 10.85 ms | 3.24% |

这些是 **profiler 回调占比**，不是无重叠的硬件时间分解。未附加 profiler 的整网 mean 为 333.11 ms，附加后为 337.39 ms。[完整分类与最慢节点](validation/msda-2026-10-21/round2/profile/README.md)

## 尝试的两个改动

1. **安全的 32 位索引。** `specialize=3` 仅在 `L=4/P=8/C=32` 且相关张量字节跨度均不超过 `INT32_MAX` 时启用；复用相邻像素步长。其他情况回退 64 位，元数据与溢出检查仍用 64 位。
2. **真实 Value 预取。** `prefetch=2` 提前读取下一采样点的四邻点特征，当前点与下一点暂存在寄存器中。没有改变采样位置、权重、点数或累加顺序。

搜索每线程 2/4 通道、128/256 线程块、64/32 位索引、无预取/Value 预取，共 16 组。数值始终与 NVIDIA 官方插件比较；性能直接比较第一轮赢家参数，而非再次使用较慢的官方插件作分母。[搜索配置](validation/msda-2026-10-21/round2/search-configs.json) · [全部候选数据](validation/msda-2026-10-21/round2/search.json)

Value 预取在这组输入上均未获益，保留为实验选项，不默认启用。源码提前发出读取不保证机器指令真正隐藏了访存等待。

## 第一版候选为什么没有采用

独立单算子复测 60 对：新库内的原参数为 16.940 ms，32 位参数为 15.132 ms，配对中位加速 1.1183 倍。但新库同时重构了插值 helper，这个比较尚未证明它优于第一轮的原库。[独立复测](validation/msda-2026-10-21/round2/initial-retest.json)

随后比较实际旧库/旧引擎与新库/新引擎：每轮预热 5 次、测量 30 次，开始前冷却至 ≤55°C，按旧—新—新—旧运行。

| 第一版候选的整网结果 | 中位延时 |
|---|---:|
| 第一轮原库＋原引擎，60 个样本 | 335.120 ms |
| 新库＋32 位配置引擎，60 个样本 | 343.546 ms |
| 结果 | **慢 2.51%，拒绝作为新默认** |

三个输出逐位一致；不是通过放宽容差换取性能。复用同一官方 timing cache，检查全部 3816 个 Inspector 记录。证明 3 个临时张量重命名保留所有生产者/消费者槽位后，其余字段完全一致；61 个记录的列出顺序改变，不代表设备执行顺序改变。[计时与精度](validation/msda-2026-10-21/round2/initial-whole/summary.json) · [逐输出比较](validation/msda-2026-10-21/round2/initial-whole/comparison.json) · [tactic 审计](validation/msda-2026-10-21/round2/tactic-audit.json)

再做三组 profiling，分离“代码重构”和“参数变更”：

| 引擎与库 | 6 个 Spatial MSDA mean 合计 |
|---|---:|
| 原引擎＋原库 | 77.079 ms |
| 原引擎＋重构库，仍使用原参数 | 94.213 ms |
| 新引擎＋重构库，启用 32 位 | 86.666 ms |

Conv、DCNv2、kgen、Reformat 合计基本稳定。退化定位到 MSDA 的插值 helper 重构：32 位索引挽回了部分耗时，却没抵消重构的损失。[对照数据](validation/msda-2026-10-21/round2/ablation.json)

修复方式是恢复原先直接 early return、循环外广播插值系数、局部变量与直写累加路径；32 位索引仍作为单独选项。最终收益必须依据修复后的整网测试，不能沿用上面的单算子百分比。

## CUDA Graph 和硬件指标

- 完整引擎 CUDA Graph 捕获失败：TensorRT 在 `allocateAsync` 报告申请 961536 字节失败，随后 capture 被作废。普通推理成功。**目前不能据此确定是物理显存不足还是捕获时的分配限制**；没有 Graph 加速结果。[原始报告](validation/msda-2026-10-21/round2/graph-probe.json)
- 已隔离安装 Nsight Compute 2024.3.2，但当前账号报 `ERR_NVGPUCTRPERM`。未修改驱动或计数器权限，未取得硬件计数器。不能声称已证明缓存命中率、occupancy、带宽或 stall 的具体原因。[日志](validation/msda-2026-10-21/round2/ncu.log)
- 首版候选的 CPT4/half2 编译报告：64 位无预取 51 寄存器，32 位无预取 38，64 位 Value 预取 64，32 位 Value 预取 54；均无 spill。**寄存器更少不等于整网更快，预取变慢也不能归因于不存在的 spill。** [编译资源](validation/msda-2026-10-21/round2/initial-resources.json)

## 代码和复现

- [CUDA kernel](../native/msda/msda_kernel.cu)：索引 guard、线程分工、插值和预取。
- [benchmark_msda.py](../scripts/benchmark_msda.py)：官方数值基准、342 项 GPU 检查、候选配置及配对测量。
- [summarize_profile.py](../scripts/summarize_profile.py)：Inspector 精确分类与热点统计。
- [benchmark_engine_graph.py](../scripts/benchmark_engine_graph.py)：完整 Graph 捕获、输出预填及逐位 replay 校验。
- [profile_msda_kernel.py](../scripts/profile_msda_kernel.py)：供 Nsight Compute 采集单个已验证 Kernel。
- [可执行复现命令](msda-optimization.md#7-继续优化与已经优化的配置比较)。

本轮 CPU 测试 134 项通过。GPU 检查覆盖 FP32/FP16、奇数通道、尾部通道、边界坐标、自动回退以及插件序列化，并保留 6 项独立 CPU oracle 对照。完整模型逐位比较只覆盖当前输入，不代替 nuScenes mAP/NDS 或混合量化对 FP32 的精度验收。单算子使用 CUDA Graph replay，整网使用普通 enqueue；两类绝对时间不能直接相加。
