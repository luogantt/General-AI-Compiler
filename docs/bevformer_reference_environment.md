# BEVFormer 原始参考环境基线

> 核对日期：2026-10-10  
> 用途：运行官方 BEVFormer-tiny、生成逐层 golden，并与 GAC/SM86 结果比较。  
> 这套环境是“模型语义参考环境”，不限制 GAC 自身以后使用更新的 CUDA。

## 结论

“Ubuntu 22.04 + CUDA 11.3”可以被改造成一套复现环境，但它不是 BEVFormer 官方仓库明确给出的原始组合。

BEVFormer 官方 `docs/install.md` 明确给出：

- Python 3.8；
- PyTorch 1.9.1 + CUDA 11.1；
- torchvision 0.10.1 + CUDA 11.1；
- torchaudio 0.9.1；
- mmcv-full 1.4.0，注释中的 wheel 索引为 `cu111/torch1.9.0`；
- mmdet 2.14.0；
- mmsegmentation 0.14.1；
- mmdet3d 0.17.1；
- GCC >= 5，示例为 GCC 6.2。

官方文档没有固定 Ubuntu 和 cuDNN 版本。NVIDIA 的 CUDA 11.1/11.3 Linux 支持表包含 Ubuntu 20.04，但不包含后来发布的 Ubuntu 22.04。因此，追求最少环境偏差时，应选择 Ubuntu 20.04。

## 推荐方案 A：忠实原版基线

| 项目 | 版本 |
| --- | --- |
| OS | Ubuntu 20.04 x86_64 |
| GPU | RTX 3070 / 3070 Ti，Compute Capability 8.6 |
| NVIDIA Driver | 可使用当前新驱动；新驱动可向后运行 CUDA 11.x Runtime |
| Python | 3.8 |
| PyTorch | 1.9.1+cu111 |
| torchvision | 0.10.1+cu111 |
| torchaudio | 0.9.1 |
| mmcv-full | 1.4.0，使用 cu111/torch1.9.0 wheel |
| mmdet | 2.14.0 |
| mmsegmentation | 0.14.1 |
| mmdet3d | 0.17.1 |
| GCC | 6 至 9；若需要源码编译 CUDA Op，固定实际编译器版本 |
| RAM | 建议 32 GB 或以上 |
| Disk | 建议至少 100 GB 可用空间 |

核心安装组合：

```bash
conda create -n bevformer-ref python=3.8 -y
conda activate bevformer-ref

pip install \
  torch==1.9.1+cu111 \
  torchvision==0.10.1+cu111 \
  torchaudio==0.9.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmdet==2.14.0 mmsegmentation==0.14.1
```

随后安装 mmdet3d v0.17.1 和 BEVFormer 官方文档列出的其余依赖。完整安装后必须保存 `collect_env` 输出、`pip freeze`、GPU 名称、驱动版本以及 MMCV CUDA Op 自检结果。

## 方案 B：坚持 CUDA 11.3

PyTorch 官方提供 PyTorch 1.9.1 + Conda `cudatoolkit=11.3` 的组合，因此“PyTorch 1.9.1 可以配 CUDA 11.3”是正确的。困难在于 MMCV 1.4 的官方预编译矩阵：

- `cu111 + torch1.9.0` 有预编译 wheel；
- `cu113` 表中对应的是 `torch1.10.0`；
- 没有列出 `cu113 + torch1.9.0` 的预编译 wheel。

因此 CUDA 11.3 有两条路径：

1. 保留 PyTorch 1.9.1，源码编译 mmcv-full 1.4.0；这更接近 BEVFormer 的 PyTorch 版本，但环境建设更脆弱。
2. 使用 PyTorch 1.10.x + CUDA 11.3 和 `cu113/torch1.10.0` 的 mmcv-full 1.4.0 wheel；安装更省事，但属于兼容性方案，不是官方 BEVFormer 原始组合。

若宿主必须是 Ubuntu 22.04，建议用 NVIDIA Container Toolkit，在容器里使用 Ubuntu 20.04 + CUDA 11.1/11.3 用户态，而不是让 CUDA 11.3、GCC 11、旧版 MMCV 直接混在 Ubuntu 22.04 系统 Python 中。

## `level_embeds` checkpoint 兼容语义

官方 BEVFormer-tiny 的 Transformer 将 `num_feature_levels` 保持为默认值 4，因此 checkpoint 中 `level_embeds` 的形状是 `(4, 256)`。但该配置的 FPN 只输出 1 个特征层，实际 forward 只读取第 0 行。GAC 在 official → canonical 映射时先完整验证原张量，再取 `level_embeds[0:1]` 作为单层运行时权重；导出归档仍保留 checkpoint 中未经裁剪的原始 `(4, 256)` 张量，便于追溯和复核。

## 历史机器与 region-41 验收记录

以下信息是旧机器的历史记录。该机器曾适合运行 GAC 的 SM86 CUDA 后端，但不适合作为原始 BEVFormer 参考环境，也不作为当前扩展源码的统一验收结论：

- Ubuntu 24.04.3，kernel 7.0；
- NVIDIA Driver 580.178.04；
- RTX 3070 Ti Laptop，8 GB；
- 系统没有 `nvcc` 或 CUDA Toolkit，只有驱动；
- GCC/G++ 13.3；
- Python 3.12.3；
- 没有 Conda、Docker、Podman；
- 当前 GAC venv 没有 torch、mmcv、mmdet、mmdet3d 或 onnx。

官方 BEVFormer-tiny checkpoint 曾下载到该机；路径与摘要用于迁移后的身份核对：

```text
/home/robot/gac-sm86-validation/checkpoints/bevformer_tiny_epoch_24.pth
size:   400785466 bytes
sha256: 7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12
```

2026-10-10 已在 region-41 完成此前计划的统一验证。宿主与两套隔离环境如下：

| 项目 | 本轮实际值 |
| --- | --- |
| Host OS | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 3080 10 GB，Compute Capability 8.6 |
| Driver / 系统 Toolkit | 580.142 / CUDA Toolkit 12.1 |
| 官方参考环境 | Python 3.8.20、PyTorch 1.9.1+cu111、torchvision 0.10.1+cu111、MMCV 1.4.0、mmdet 2.14.0、mmseg 0.14.1、mmdet3d 0.17.1、NumPy 1.19.5 |
| GAC 环境 | Python 3.11.17、CuPy 13.6、CUDA Runtime/NVRTC 12.9 |

官方 `bevformer_tiny_epoch_24.pth` 为 `400,785,466` bytes，SHA-256 为 `7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12`。参考环境能够导入 MMCV CUDA Ops；官方模型 643 个 tensor 全部严格加载，无 missing、unexpected 或 shape mismatch。GAC 使用同一 checkpoint 生成 golden，并完成生产尺寸统一 cold/warm step HXP 的真机数值验收：

| 项目 | 结果 |
| --- | --- |
| 完整强制 CUDA 测试 | 最终版本 `Ran 287 tests in 18.881s ... OK (skipped=1)` |
| 生产 HXP | `137,896,143` bytes，SHA-256 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`；融合前 876 节点、Add+ReLU 融合后 837 节点、703 次实际设备 dispatch；内部 arena `552,960,000` bytes；Tensor 计划 `705,177,100` bytes，另加 cuDNN 8 MiB 后计划预算/准入值 `713,565,708` bytes，不代表进程设备峰值 |
| 性能版 cold max abs（cls / bbox / BEV） | `0.003891945 / 0.004856288 / 0.001953125` |
| 性能版 warm max abs（cls / bbox / BEV） | `0.000953674 / 0.000725389 / 0.001953125` |
| 全生成 FP32 中位数 | cold `425.325073 / 436.881591 ms`；warm `426.032532 / 437.718436 ms`（GPU / wall） |
| 零拷贝 + FP32 cuDNN/cuBLAS 中位数 | cold `57.624464 / 69.403093 ms`；warm `56.823263 / 68.747440 ms`（GPU / wall） |
| fresh-process 中 Session 构建后的首个 cold `run` | `277.443573 / 289.668472 ms`（GPU / wall；Session 构建时间另计） |

warm 路径在每层共享 `[motion prev, initial query]`；cold 路径在每层用该层 current query 动态构造 `[current query, current query]`。性能版使用内部 Reshape/连续 Slice 零拷贝、FP32 cuDNN Conv 和 FP32 CuPy/cuBLAS MatMul，cold GPU/wall 加速 `7.38098× / 6.29484×`，warm 为 `7.49750× / 6.36705×`。HXP launch metadata 显式保存 vendor/generated 候选；Runtime 和逐 launch profile 记录实际 implementation，JSON 同时按 op 与 implementation 聚合。cuDNN/cuBLAS 首次调用包含初始化与算法缓存，必须与稳态中位数及 fresh-process 首次 `run` 分开。CPU NMS-free Top-K/box denormalization 仍是显式图外边界。

本轮 RTX 3080 与目标 RTX 3070 同属 SM 8.6，已经验证编译兼容性和固定 Profile 的 raw step 数值。桌面 RTX 3070 的实际显存/性能以及真实 nuScenes 数据集的 boxes/scores/labels 与精度验收仍待完成。

## 一手资料

- [BEVFormer 官方安装文档](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md)
- [PyTorch 历史版本安装矩阵](https://docs.pytorch.org/get-started/previous-versions/)
- [MMCV 1.4 安装与 CUDA/PyTorch wheel 矩阵](https://mmcv.readthedocs.io/en/v1.4.7/get_started/installation.html)
- [CUDA 11.3 Linux 安装支持表](https://docs.nvidia.com/cuda/archive/11.3.1/cuda-installation-guide-linux/index.html)
- [NVIDIA CUDA 向后兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html)
