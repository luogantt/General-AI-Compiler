# UniAD 2.0 端到端基线：RTX 3080 / nuScenes mini

本目录保存 `2026-10-11` 对完整 UniAD 2.0 `stage2_e2e` 推理路径的原始基准证据。正式报告使用 NVIDIA GeForce RTX 3080（Compute Capability 8.6），加载真实 `uniad_base_e2e.pth` checkpoint 和 nuScenes `v1.0-mini` 数据，通过原生 PyTorch/MMCV 执行 tracking、mapping、motion、occupancy、planning 五个 head。checkpoint 以 `strict=True` 加载；报告中的 `compiled_subgraphs` 为空，因此这里测得的是后续编译优化所需的 **native PyTorch 端到端基线**，不是 MLIR 整网性能。

这份报告由迁移前放在外部 UniAD checkout 内的 benchmark 工具生成，Git/source provenance 指向当时的 UniAD 分支；原始 JSON、日志和 `SHA256SUMS` 在迁移时保持不变。GAC 现在通过 `tools/uniad_benchmark_e2e.py --uniad-root /path/to/UniAD-2.0 ...` 集成同一 native 基线，并在新报告中分别记录 GAC 与外部 UniAD checkout。下面的数字仍是有效的原始基线，但不是当前 GAC commit 上重新运行的编译器端到端结果。

## 正式 50 帧结果

测试按数据集顺序执行，batch size 为 1、`workers=0`、不 shuffle。前 10 帧用于 warmup，随后测量数据集索引 10～59 的 50 帧。测量覆盖 2 个 scene token：场景切换发生在测量索引 30（数据集索引 40），所以分组包含 1 个首帧和 49 个稳态帧。

| 计时项 | 帧数 | 均值 | 中位数 | P90 | P99 | 由均值换算的吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整模型调用 | 50 | 822.885 ms | 806.619 ms | 902.287 ms | 982.392 ms | 1.215 FPS |
| dataloader 等待 | 50 | 2914.904 ms | 2990.312 ms | 3616.270 ms | 4350.067 ms | — |
| 串行 input pipeline + 模型 | 50 | 3737.789 ms | 3784.887 ms | 4376.574 ms | 5303.153 ms | 0.268 FPS |

模型调用的场景分组如下。首帧只有一个样本，不能据此推断首帧通常比稳态更快。

| 分组 | 帧数 | 均值 | 中位数 | P90 | P99 | 由均值换算的吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 首帧 | 1 | 739.767 ms | 739.767 ms | 739.767 ms | 739.767 ms | 1.352 FPS |
| 稳态 | 49 | 824.581 ms | 807.731 ms | 904.118 ms | 982.523 ms | 1.213 FPS |

这里的“完整模型调用”从 dataloader 已生成 CPU batch 后开始，在调用 `MMDataParallel` 前同步 CUDA，然后计时 scatter/H2D、完整 `forward_test`、五个任务 head、Python 主机工作和 CUDA 执行，最后再次同步 CUDA 并停止计时。它是同步 wall time，不是只统计 GPU kernel 的 resident 时间。dataloader 的 `next()` 阻塞等待被单独计时；“串行 input pipeline + 模型”是同一帧两段时间之和。由于 `workers=0`，该数字描述本次未重叠的串行观测路径，不代表使用多进程预取的生产吞吐量。多样本指标聚合和结果序列化均不在模型计时内；原版 UniAD evaluation forward 中依赖 ground truth 的辅助逻辑仍在边界内。

测量期 PyTorch CUDA allocator 的峰值 allocated memory 为 `2,581,901,312` bytes（`2.405 GiB`），峰值 reserved memory 为 `3,586,129,920` bytes（`3.340 GiB`）。峰值统计在 warmup 后重置；track-query 遥测在模型计时结束后复制到 CPU，再进行筛选和计数。

## 输入和运行环境

- Git：`6c92fc4bbfa8e5d499da73b02807f14c9be2ecc0`，运行时 tracked files clean；基准脚本 SHA-256 为 `93fbb8333572f977cd51daf9e0f6482dc29aa6d5307027735ef56c3556dbd11e`。
- 配置：`projects/configs/stage2_e2e/base_e2e.py`，SHA-256 `2d45be9b330f8e82456203e2d6374053bd7b80171c5316412dfa88b32e187b8c`。
- checkpoint：`uniad_base_e2e.pth`，`996,773,953` bytes，SHA-256 `4dcdc34c09f964e377f608dbb764f003844a9ccc62b4a0469f94375302ed76d0`。
- annotation：`nuscenes_infos_temporal_val.pkl`，`5,311,119` bytes，SHA-256 `eb61b2fe66d5401f794568bae27702c38aa8d844ca74271c9c4c4b53468adf52`。
- 数据：nuScenes `v1.0-mini`；devkit 日志载入 10 个 scene、404 个原始 sample，配置后的 `NuScenesE2EDataset` test split 有 81 个顺序样本。本次 warmup 和测量共消费前 60 个。
- 软件：Python 3.9.25、PyTorch 2.0.1+cu118、CUDA runtime 11.8、cuDNN 8.7.0、MMCV 1.7.2、MMDetection 2.26.0、MMDetection3D 1.0.0rc6、NumPy 1.22.4；配置未启用 FP16。

以上设备属性、软件版本以及 config、checkpoint、annotation 的 SHA-256 和字节数均在远端原文件上再次独立核对。正式 JSON 中 `model_call_wall`、`dataloader_wait` 和 `observed_next_plus_model_wall` 的 all/first-frame/steady-state 分组统计，均已由 50 条原始 sample 重新计算；逐帧 `input pipeline + 模型` 加法、场景分组、五个 head、strict load、显存上下界和日志中的阶段/最终摘要也已复核。

## 证据文件与限制

- `uniad-e2e-rtx3080-50frames.json`：主报告，包含逐帧样本、分组统计、输入摘要、环境、显存和 Git/source provenance。
- `e2e_formal_50frames.log`：主报告对应的控制台日志；仅规范化了 warning 行的行末空白。
- `uniad-e2e-rtx3080.json`、`e2e_formal.log`：较早的 30 帧先导运行；日志同样仅规范化行末空白。30 帧都在同一个场景的稳态段，模型调用均值 `810.756 ms`、中位数 `803.858 ms`。主结论使用 50 帧报告。
- `SHA256SUMS`：上述四个原始证据文件的完整性摘要。

本次只覆盖 nuScenes mini 的 50 个测量样本，不是完整 nuScenes validation，不包含数据集精度指标，也没有执行多个独立进程的重复运行。当前 MLIR/CUDA 编译路径仍只覆盖 BEV encoder 的单层 FFN，尚未接入完整 UniAD 图，因此不能从本报告得出 MLIR 整网加速比。本结果也不是 RTX 3070 实测。
