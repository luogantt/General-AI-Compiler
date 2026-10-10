# BEVFormer-tiny / CUDA SM86 远程验收记录（2026-10-10）

## 1. 验收结论与适用范围

本轮在 Ubuntu 22.04 远程机器上，用 BEVFormer-tiny 官方 checkpoint 完成了 GAC 统一 cold/warm step 的权重导入、HXP 构建、CUDA SM86 真机执行和官方实现数值比较。最终自动化测试结果为 `Ran 287 tests in 18.881s`、`OK (skipped=1)`。性能版 cold 和 warm 输出最大绝对误差都不超过 `0.004856288`；cold GPU 中位数从全生成路径的 `425.325073 ms` 降到 `57.624464 ms`，warm 从 `426.032532 ms` 降到 `56.823263 ms`。

这说明当前的官方权重映射、BEVFormer 核心算子、cold/warm 时序语义、SM86 代码生成、Runtime 状态路径以及首轮 vendor library 派发在本次固定输入上完成了闭环。本轮实际显卡是 **RTX 3080 10 GB，Compute Capability 8.6**。目标 RTX 3070 与它同属 SM86，因此本结果可以验证 SM86 指令目标与 Kernel 兼容性，但**还没有在桌面版 RTX 3070 上实测**，不能把 3080 的耗时和显存余量当作 3070 的最终结论。

仓库内已保存本次验收的原始证据。最终数字统一由[优化验收索引](validation/sm86/bevformer/optimization-2026-10-10/README.md)及其链接的最终日志支撑；以下 `full-step-2026-10-10` 文件是首次正确性闭环的历史阶段证据：

- [设备与 CUDA doctor 输出](validation/sm86/bevformer/full-step-2026-10-10/doctor.txt)
- [官方 checkpoint 严格加载日志](validation/sm86/bevformer/full-step-2026-10-10/official-checkpoint-load.txt)
- [HXP 构建报告](validation/sm86/bevformer/full-step-2026-10-10/build-production-step.json)
- [cold 全图数值与耗时报告](validation/sm86/bevformer/full-step-2026-10-10/cold-parity.json)
- [warm 全图数值与耗时报告](validation/sm86/bevformer/full-step-2026-10-10/warm-parity.json)
- [cold Encoder 分段对齐报告](validation/sm86/bevformer/full-step-2026-10-10/encoder-cold-parity.json)
- [267 项测试完整日志](validation/sm86/bevformer/full-step-2026-10-10/test-all-sm86.txt)
- [证据文件 SHA256 清单](validation/sm86/bevformer/full-step-2026-10-10/SHA256SUMS)

## 2. 机器与软件环境

| 项目 | 本轮实际值 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 3080 10 GB |
| GPU 计算能力 | 8.6（SM86） |
| NVIDIA 驱动 | 580.142 |
| 系统 CUDA Toolkit | 12.1 |
| GAC 环境 | Python 3.11.17、CuPy 13.6.0、CUDA Runtime 12.9、NVRTC 12.9 |
| 官方 BEVFormer 参考环境 | Python 3.8.20、PyTorch 1.9.1+cu111、MMCV 1.4.0、MMDetection 2.14.0、MMSegmentation 0.14.1、MMDetection3D 0.17.1 |

GAC 进程实际使用 CuPy wheel 带入的 CUDA Runtime/NVRTC 12.9；系统 Toolkit 12.1 只是机器上的另一个安装项。官方参考环境的 PyTorch 是 `+cu111`，本文按实际执行环境记录，不把它改写为 CUDA 11.3。

## 3. 冻结的权重与编译产物

| 产物 | 路径 | 大小（bytes） | SHA256 / 关键属性 |
| --- | --- | ---: | --- |
| 官方 checkpoint | `/root/autodl-tmp/gac-validation/checkpoints/bevformer_tiny_epoch_24.pth` | 400,785,466 | `7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12` |
| GAC 权重包 | `/root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_epoch_24.gacweights` | 134,579,509 | `ee9776e968cce3e9dfbaa0b551c38d9cdec2a4b1649d898f377b2adc04fe6b26` |
| 最终 step HXP | `/root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_step_sm86.hxp` | 137,896,143 | `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4` |
| 历史首次正确性 step HXP | 同一路径的阶段产物，现已被最终包替换 | 137,896,493 | `a900b719c2deef016dd54c89bbd5177e14f47bb451999fc4b6bc78a48fc61be9` |

官方 checkpoint 在上述 BEVFormer 参考环境中完成严格加载：checkpoint 与模型各有 643 个 tensor，结果是 **643/643 匹配**，无 missing key、unexpected key 或 shape mismatch。可直接复核仓库中的[严格加载日志](validation/sm86/bevformer/full-step-2026-10-10/official-checkpoint-load.txt)；远端原始日志路径为：

```text
/root/autodl-tmp/gac-validation/logs/bevformer-official-load.txt
```

GAC 权重包保留 643 个 tensor；源 checkpoint 的 SHA256 被写入并在构建时校验。生产 HXP 的融合前图包含 876 个节点；39 组安全 Add+ReLU 融合后得到 837 个执行节点，再把内部 Reshape 和满足连续区间约束的 Slice 变成共享存储 view 后，实际设备 dispatch 为 **703**。39 组融合来自 ResNet 16、CAN-bus 2、Encoder 3、Decoder 18。FP32 Conv/MatMul 分别交给 cuDNN 和 CuPy/cuBLAS。`KernelLaunch` 在 HXP 中显式记录 `implementation_candidates`（`cudnn`、`cupy-cublas`、`generated-cuda`），Session 建立时选定实现并写入 `session.info`、`kernel_info` 和 profile；vendor 调用在执行中失败不会逐节点自动切回 generated CUDA。内部 arena 在融合前后均为 **552,960,000 bytes**；Tensor 计划总量为 **705,177,100 bytes**，cuDNN 的 8 MiB（**8,388,608 bytes**）workspace budget 单独管理，合计 **713,565,708 bytes** 是 Tensor + cuDNN workspace 的计划预算/准入值，不是严格设备峰值上界；它不含 module、CUDA context、Event 和 vendor 临时分配。当前 HXP 与本版 codegen 绑定，优化前的 schema 1 包需用当前版本重建，不能直接作为最终运行包。仓库中的 [HXP 构建报告](validation/sm86/bevformer/full-step-2026-10-10/build-production-step.json)冻结的是首次正确性闭环历史产物。

## 4. 验收输入与比较方法

本轮使用六相机、`480 x 800` 固定 profile。公开图像 ABI 是 FP16，输入形状为 `[1, 6, 3, 480, 800]`；输出为：

- `cls_logits`: `[1, 900, 10]`；
- `bbox_regression`: `[1, 900, 10]`；
- `next_bev`: `[1, 2500, 256]`，FP16 状态边界。

数值比较使用了**全零的合成、已归一化且已 padding 的 FP16 图像 tensor**，`lidar2img` 使用单位矩阵。cold 路径使用 `has_prev_bev=0`；warm 路径把同一份冻结的 FP16 `prev_bev` 同时交给官方实现和 GAC，使用 `has_prev_bev=1`。两端使用同一份输入和同一份权重，逐元素比较三个输出的绝对误差。

这组输入主要检查算子 lowering、权重映射、cold 每层 temporal queue、warm 历史 BEV 选择和 Runtime 状态语义。它不是 nuScenes 样本，没有执行数据集级的 mAP/NDS 评估，因此不能作为 BEVFormer 真实场景精度结论。

## 5. cold/warm 数值与执行结果

| 路径 | `cls_logits` max abs | `bbox_regression` max abs | `next_bev` max abs |
| --- | ---: | ---: | ---: |
| cold | `0.003891945` | `0.004856288` | `0.001953125` |
| warm | `0.000953674` | `0.000725389` | `0.001953125` |

| 路径 | 全生成 GPU / wall | 优化版 GPU / wall | GPU / wall 加速比 |
| --- | ---: | ---: | ---: |
| cold | 425.325073 / 436.881591 ms | 57.624464 / 69.403093 ms | 7.38098× / 6.29484× |
| warm | 426.032532 / 437.718436 ms | 56.823263 / 68.747440 ms | 7.49750× / 6.36705× |

表中全生成与优化版分别是 3 次和 10 次稳态运行的中位数。fresh-process 中 Session 构建完成后的首个 cold `run` 为 `277.443573 ms GPU / 289.668472 ms wall`；同一进程已预热后的 `116.635651 ms` 不能当作 fresh-process 首次 `run`。归档的 [cold 报告](validation/sm86/bevformer/full-step-2026-10-10/cold-parity.json)与 [warm 报告](validation/sm86/bevformer/full-step-2026-10-10/warm-parity.json)记录的是优化前首次正确性闭环，其中 `434.900 / 436.265 ms` 和旧误差字段只作为历史证据。Session setup 不包含在前述 `run` 时间中；首次 NVRTC 编译曾为 `10554.369 ms`（约 **10.55 s**），命中缓存后为 `740.448 ms`（约 **0.74 s**）。这些数字不外推为 RTX 3070 耗时。

Runtime 的可选逐 launch Profiler 使用 CUDA Event 记录设备耗时，JSON 同时提供每个 launch 的 node/op/source、grid/block、实际 implementation 和耗时，以及按 op 与 implementation 聚合的 count、total、mean、min、max 和占比。最终受 Event 扰动的独立 Profile 为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 Kernel Event 求和为 `63.4391 ms`：cuDNN 55 次 / `36.6409 ms`、CuPy/cuBLAS 118 次 / `8.0060 ms`、generated CUDA 530 次 / `18.7923 ms`。默认执行路径关闭逐 launch Event，因此这些归因值不与上表普通基准直接比较。

显式 128 MiB cuDNN workspace 实验把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，且比当前 8 MiB budget 多占 120 MiB，没有端到端收益，本轮不集成。下一步先推进 CUDA Graph、批量 dispatch 和更多融合；当前 `cupyx.cudnn` 与 capture 不兼容，未来 cuDNN 图捕获需稳定的低层 API。

## 6. 自动化测试

最终版本强制 CUDA 的全量测试结果为：

```text
Ran 287 tests in 18.881s
OK (skipped=1)
```

`GAC_REQUIRE_CUDA=1` 确保 CUDA 不可用时直接失败，因此物理 CUDA 测试没有被静默跳过。唯一 skip 是可选的本地 BEVFormer snapshot 测试：它在源码中指向 Windows 绝对路径，远程 Linux 机器上不存在该路径。最终日志是 [unittest-full-final.txt](validation/sm86/bevformer/optimization-2026-10-10/unittest-full-final.txt)；[267 项日志](validation/sm86/bevformer/full-step-2026-10-10/test-all-sm86.txt)与[零拷贝 + cuDNN 阶段日志](validation/sm86/bevformer/optimization-2026-10-10/unittest-full-cudnn.txt)只作为历史阶段证据。

```text
/root/autodl-tmp/gac-validation/logs/test-all-sm86-fixed.txt
```

## 7. 远程机器上的复现步骤

以下命令假定仓库位于 `/root/autodl-tmp/gac-validation/deploy-20261010/General-AI-Compiler`，并且上述两个 Conda 环境和冻结的官方参考输出仍然存在。

### 7.1 确认设备与 GAC CUDA 环境

```bash
cd /root/autodl-tmp/gac-validation/deploy-20261010/General-AI-Compiler
nvidia-smi
conda run -n gac-sm86 python -m gac.cli doctor
conda run -n gac-sm86 python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.runtimeGetVersion())"
```

`doctor` 应报告 Compute Capability `8.6`、目标 `cuda-sm86`、CUDA Runtime/NVRTC `12.9`。

### 7.2 校验 checkpoint 并导出 GAC 权重

```bash
cd /root/autodl-tmp/gac-validation
sha256sum checkpoints/bevformer_tiny_epoch_24.pth

cd deploy-20261010/General-AI-Compiler
conda run -n gac-sm86 python -m gac.tools.export_bevformer_checkpoint \
  --trusted-checkpoint /root/autodl-tmp/gac-validation/checkpoints/bevformer_tiny_epoch_24.pth \
  --expected-sha256 7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12 \
  -o /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_epoch_24.gacweights

conda run -n gac-sm86 python -m gac.tools.build_bevformer inspect-weights \
  /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_epoch_24.gacweights \
  --expected-checkpoint-sha256 7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12
```

checkpoint 只在已校验 SHA256 且来源可信时交给 PyTorch 反序列化。官方模型的 643/643 严格加载记录可从 `logs/bevformer-official-load.txt` 复核。

### 7.3 构建和检查生产 step HXP

```bash
conda run -n gac-sm86 python -m gac.tools.build_bevformer step-pipeline \
  --weights /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_epoch_24.gacweights \
  --expected-checkpoint-sha256 7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12 \
  -o /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_step_sm86.hxp

conda run -n gac-sm86 python -m gac.cli inspect \
  /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_step_sm86.hxp

sha256sum \
  /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_epoch_24.gacweights \
  /root/autodl-tmp/gac-validation/artifacts/bevformer_tiny_step_sm86.hxp
```

仓库内的结构化构建证据是 [build-production-step.json](validation/sm86/bevformer/full-step-2026-10-10/build-production-step.json)，远端原始构建日志位于 `/root/autodl-tmp/gac-validation/logs/build-production-step-fixed.txt`。该文件及 SHA256 对应首次正确性闭环产物；当前优化语义是融合前 876 节点、Add+ReLU 融合后 837 节点、703 次实际设备 dispatch，重新构建后的包摘要必须按新产物单独记录，不能沿用旧 SHA256。

### 7.4 运行全量测试

```bash
cd /root/autodl-tmp/gac-validation/deploy-20261010/General-AI-Compiler
GAC_REQUIRE_CUDA=1 conda run -n gac-sm86 \
  python -m unittest discover -s tests -v
```

### 7.5 重放 cold/warm 官方 golden 比较

```bash
conda run -n gac-sm86 python \
  /root/autodl-tmp/gac-validation/run_production_golden.py

conda run -n gac-sm86 python \
  /root/autodl-tmp/gac-validation/run_production_warm_golden.py
```

冻结输入和输出位于：

```text
/root/autodl-tmp/gac-validation/golden/official-zero-cold-inputs.npz
/root/autodl-tmp/gac-validation/golden/official-zero-cold-outputs.npz
/root/autodl-tmp/gac-validation/golden/official-zero-warm-outputs.npz
/root/autodl-tmp/gac-validation/golden/gac-zero-cold-outputs.npz
/root/autodl-tmp/gac-validation/golden/gac-zero-warm-outputs.npz
```

可机器读取的结果报告位于：

```text
/root/autodl-tmp/gac-validation/logs/gac-production-zero-cold.json
/root/autodl-tmp/gac-validation/logs/gac-production-zero-warm.json
```

## 8. 明确边界与下一步

1. **输入边界**：HXP 接收已归一化、已 padding 的 FP16 六相机 tensor；图像解码、resize、normalize 和 padding 不在本次 GPU 图内。
2. **后处理边界**：HXP 返回固定 shape 的 raw class、raw box 和 `next_bev`。NMS-free Top-K、box denormalization 及范围过滤目前由显式 **CPU `decode_nms_free`** API 完成，不在 GPU step 内；当前执行图为 837 个节点、703 次实际设备 dispatch，融合前语义图为 876 个节点。
3. **精度边界**：全零合成输入可以抓取图连接、权重、数值类型和状态语义错误，但不覆盖真实标定、场景分布和检测指标。
4. **性能边界**：FP32 cuDNN Conv、FP32 CuPy/cuBLAS MatMul、内部 Reshape/连续 Slice 零拷贝和逐 launch Profiler 已完成；Tensor Core/混合精度、更广算子融合、I/O 预绑定和 CUDA Graph 仍未完成。
5. **下一步验收**：用真实 nuScenes 连续帧生成官方/GAC 对照结果，执行完整 CPU NMS-free 后处理并比较 mAP/NDS；随后在实际桌面 RTX 3070 上重测首次编译、缓存 setup、预热后 P50/P95、端到端耗时、峰值显存和长时间连续帧稳定性。
