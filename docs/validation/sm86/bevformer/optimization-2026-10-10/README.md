# BEVFormer-tiny SM86 最终优化验收证据

本目录冻结 2026-10-10 在 NVIDIA GeForce RTX 3080（Compute Capability 8.6）上的最终验收日志。下列“最终”文件是当前结论的数据源；同目录和相邻 `full-step-2026-10-10/` 中标为“历史”的文件只记录阶段进展，不能覆盖最终数字。

## 最终结论

| 项目 | cold | warm |
| --- | ---: | ---: |
| 全生成 FP32，GPU / wall 中位数 | 425.325073 / 436.881591 ms | 426.032532 / 437.718436 ms |
| 零拷贝 + FP32 cuDNN/cuBLAS，GPU / wall 中位数 | 57.624464 / 69.403093 ms | 56.823263 / 68.747440 ms |
| GPU / wall 加速比 | 7.38098× / 6.29484× | 7.49750× / 6.36705× |
| `cls_logits` / `bbox_regression` / `next_bev` max abs | 0.003891945 / 0.004856288 / 0.001953125 | 0.000953674 / 0.000725389 / 0.001953125 |

全生成数据各取 3 次稳态运行的中位数，优化版各取 10 次稳态运行的中位数，来源是 [gac-production-benchmark-final.json](gac-production-benchmark-final.json)。fresh-process 中 Session 构建完成后的首个 cold `run` 为 `277.443573 ms GPU / 289.668472 ms wall`，来源是 [gac-production-first-frame-final.json](gac-production-first-frame-final.json)。同一进程基准中的 `116.635651 ms` 是 Session 已预热后的调用，不是 fresh-process 首次 `run`；Session 构建时间另计。

最终强制 CUDA 测试为 `Ran 287 tests in 18.881s`、`OK (skipped=1)`，见 [unittest-full-final.txt](unittest-full-final.txt)。最终 HXP 大小为 `137,896,143` bytes，SHA-256 为 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`，分别见 [size](bevformer_tiny_step_sm86-final.size.txt)、[SHA-256](bevformer_tiny_step_sm86-final.sha256.txt)和[最终构建日志](build-production-step-final.txt)。融合前仍为 876 个节点，39 组 Add+ReLU 融合后为 837 个执行节点，实际 703 次设备 dispatch，内部 arena 为 `552,960,000` bytes。

## Profile 与 workspace 实验

[最终 Profile 摘要](gac-production-profile-final-summary.json)和[逐 launch 明细](gac-production-profile-final.json)来自带 CUDA Event 的独立调用，Event 会扰动执行，因此它们用于归因，不与无逐 launch Event 的普通基准直接比较。该调用为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 Kernel Event 求和 `63.4391 ms`：

| implementation | launch 数 | Kernel Event 总计 |
| --- | ---: | ---: |
| cuDNN | 55 | 36.6409 ms |
| CuPy/cuBLAS | 118 | 8.0060 ms |
| generated CUDA | 530 | 18.7923 ms |

[显式 cuDNN workspace 实验](cudnn-explicit-workspace-benchmark.json)显示，128 MiB workspace 把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图的交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，没有端到端收益，并比当前 8 MiB budget 多占 120 MiB。因此本轮不集成；下一步先做 CUDA Graph、批量 dispatch 和更多融合。当前 `cupyx.cudnn` 与 capture 不兼容，未来 cuDNN 图捕获需要稳定的低层 API。

Tensor 计划为 `705,177,100` bytes，另加 cuDNN 8 MiB 后的 `713,565,708` bytes 只表示 Tensor + cuDNN workspace 的计划预算/准入值，不是严格设备峰值上界；module、CUDA context、Event 和 vendor 临时分配不在其中。

## 包与实现选择边界

当前 HXP 与本版 codegen 绑定；优化前的 schema 1 包必须用当前版本重建。HXP 中的 `implementation_candidates` 供 Session 建立时选择，选择结果写入 `session.info`、`kernel_info` 和 Profile；vendor 调用执行中失败不会逐节点自动切换到 generated CUDA。

## 证据分层

最终结论使用：

- [最终普通基准与精度](gac-production-benchmark-final.json)
- [fresh-process 首次 `run`](gac-production-first-frame-final.json)
- [最终 Profile 摘要](gac-production-profile-final-summary.json)及[逐 launch Profile](gac-production-profile-final.json)
- [最终测试日志](unittest-full-final.txt)
- [最终构建日志](build-production-step-final.txt)、[HXP 大小](bevformer_tiny_step_sm86-final.size.txt)与[HXP SHA-256](bevformer_tiny_step_sm86-final.sha256.txt)
- [显式 128 MiB cuDNN workspace 实验](cudnn-explicit-workspace-benchmark.json)

历史阶段证据包括 [gac-production-benchmark-optimized.json](gac-production-benchmark-optimized.json)、[unittest-full-cudnn.txt](unittest-full-cudnn.txt)、[build-production-step-cudnn.txt](build-production-step-cudnn.txt)、[旧 HXP SHA-256](bevformer_tiny_step_sm86.sha256.txt)、[早期 cold Profile](gac-production-profile-cold.json)和相邻 [full-step-2026-10-10](../full-step-2026-10-10/) 目录。它们用于追踪演进，不作为最终验收数字。
