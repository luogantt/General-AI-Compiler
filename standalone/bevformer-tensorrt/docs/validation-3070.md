# 独立项目验证：RTX 3070 Ti Laptop 8 GB

日期：**2026-10-20**。

独立项目代码已在目标 GPU 上成功构建、执行完整固定形状 BEVFormer-base 到检测头输出。FP32 数值验证通过；混合 INT8/FP16 可以执行且更快，但未通过默认严格逐元素门限，尚未验收检测精度。

## 环境与独立性

| 项目 | 实际配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GB |
| 驱动 | 580.178.04 |
| Python | 3.11.16 |
| TensorRT | 10.13.3.9，CUDA 12 包 |
| PyTorch | 2.7.0+cu126 |
| CUDA 模块加载 | `CUDA_MODULE_LOADING=LAZY` |
| TF32 | builder flag 关闭，`NVIDIA_TF32_OVERRIDE=0` |
| 辅助 stream | `max_aux_streams=0` |

这不是桌面版 RTX 3070 或 RTX 3090。本次复用了测试机已有的 TensorRT/PyTorch/CUDA 依赖，没有重建一套完全隔离的基础环境。

整图构建和最初测速使用独立项目的 `src` 包路径；随后打成 wheel 安装，从源码目录外验证导入、CPU 测试和插件执行。模型构建和执行不依赖原编译器的 `gac` 包；TensorRT 执行网络，Torch 提供 CUDA I/O 和计时。构建、适配、精度策略、校准与范围转换的测试源文件和交付源文件逐一核对哈希，见 [源文件一致性](validation/3070-2026-10-20/source-identity.json)。

| 检查 | 结果 |
|---|---|
| wheel 构建、安装、源码目录外导入/CLI | 通过 |
| CPU 测试 | **97 项通过** |
| TensorRT 环境小网络 | 通过 |
| DCNv2/MSDA 六组插件 fixture | **6/6 通过** |
| 完整混合精度 engine 构建与执行 | 通过；严格输出比较未通过 |
| 完整 FP32 engine 构建与执行 | 通过；真实和合成输入严格比较均通过 |
| 安装后的 wheel 从 `/tmp` 执行完整 FP32 engine | 通过；配套合成输入和 PyTorch 参考严格比较通过 |

插件测试为 FP32、`rtol=2e-5, atol=2e-5`，验证小型算子样例，不替代整网验证。证据：[插件报告](validation/3070-2026-10-20/plugin-summary.json)、[测试汇总](validation/3070-2026-10-20/summary.json)。

安装后的整图回归取消了 `PYTHONPATH`，预热 1 次、执行 1 次，验证从安装包运行的独立性；
[回归报告](validation/3070-2026-10-20/installed-wheel-full.json)不替代下方 20 次采样的性能数据。

## 模型与计时边界

完整图为 ResNet-101＋DCNv2、FPN、六层 BEV Encoder、六层检测 Decoder 和检测头，输入 `float32 [1,6,3,928,1600]`，BEV 为 200×200。

图内相机投影/可见性仍为导出时固定合成元数据，CAN-bus 为零，无历史帧。输出是 `bev_embed`、`all_cls_scores`、`all_bbox_preds`，不含最终框解码。因此“完整图”指上述固定网络计算范围，不表示已完成真实视频感知部署。

每组预热 5 次、测量 20 次；每次一帧并同步。CUDA event 计时排除预处理、H2D/D2H、加载、构建和预热，无 CUDA Graph replay。区间可能包含主机提交造成的 GPU 空闲；FPS 是中位延时倒数，不是多帧并发吞吐。

## 实测结果

前两行使用**同一帧真实六相机输入**；合成输入单独列出：

| 精度 / 输入 | 中位延时 | P95 | 中位延时倒数 | `rtol=1e-3, atol=1e-4` |
|---|---:|---:|---:|---|
| 混合 INT8/FP16 / 真实图像 | **420.782 ms** | 423.458 ms | **2.377 FPS** | 未通过，输出均有限 |
| FP32 / 同帧真实图像 | **924.934 ms** | 986.490 ms | **1.081 FPS** | 通过 |
| FP32 / 固定合成输入 | **809.661 ms** | 820.623 ms | **1.235 FPS** | 对 PyTorch 参考通过 |

证据：[混合真实输入](validation/3070-2026-10-20/mixed-real.json)、[FP32 真实输入](validation/3070-2026-10-20/fp32-real.json)、[FP32 合成输入](validation/3070-2026-10-20/fp32-synthetic.json)。

真实输入的两组配置中，延时比约为 **2.20 倍**。运行时段、温度/频率与候选实现不同，不能将全部差异归因于 INT8。合成输入的 809.661 ms 也不能替换真实输入的 FP32 基线。

真实输入比较的相对 RMSE：

| 输出 | 混合精度相对 RMSE | FP32 相对 RMSE |
|---|---:|---:|
| BEV 特征 | 2.404% | 约 8.31×10⁻⁵% |
| 分类 logits | 0.528% | 约 1.99×10⁻⁵% |
| 框编码 | 0.468% | 约 3.89×10⁻⁵% |

真实输入的参考文件来自此前独立 **native-MLIR FP32** 路径，SHA-256 为 `2c293ca15c12e95796c843b13506bf8a7bd42175c3aea6f303a08bf455f84b98`。合成输入的参考来自上游 **PyTorch 原始模型模块**，SHA-256 为 `77dd2317d73aee213d99bcc1837a2570161e03241bfa4d8fd50cc7fc2ae67181`。两类参考不能混称为本次 PyTorch 真值。

以上比较读取已保存的参考输出，不要求运行时安装原编译器。相对 RMSE 不是 mAP/NDS 的下降比例。程序没有将混合精度失败改成成功，也没有通过放宽门限掩盖误差。

**本次尚未达到 30 FPS。** 真实图像仍经过固定合成几何图，不构成真实相机标定、历史状态或完整 nuScenes 检测指标验证。

## 构建尝试及成功参数

| 模式 | 优化级别 | workspace | tactic DRAM | 结果 |
|---|---:|---:|---:|---|
| 混合 INT8/FP16，首次 | 0 | 256 MiB | 2048 MiB | 构建失败，日志报 OOM |
| 混合 INT8/FP16，后续 | 3 | 1024 MiB | 4096 MiB | 成功 |
| FP32＋FP32 插件边界 | 3 | 4096 MiB | 4096 MiB | 成功 |

首轮日志的笼统 OOM **不能证明整张 8 GB 显存已满**。降低 workspace/搜索级别并不必然降低到足以构建的资源需求，也可能排除可用候选。本次扩大搜索和工作区后成功，不能继续沿用“这台机器放不下完整 Base”的旧结论。

| 成功构建的资源记录 | 混合 INT8/FP16 | FP32 |
|---|---:|---:|
| engine 构建耗时 | 263.02 s | 105.19 s |
| 构建及后续推理期间整卡显存采样最大值 | 4549 MiB | 4251 MiB |
| 推理 context＋I/O 绑定后整卡占用快照 | 2476.25 MiB | 4244.25 MiB |

采样值包含整卡占用，可能漏过瞬时峰值；绑定后的快照只是预热前一个时点。两者都不是单进程精确峰值，也不代表其他模型或配置的显存需求。详细记录见 [汇总证据](validation/3070-2026-10-20/summary.json)。

复现成功配置：

```bash
CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --precision mixed-int8 \
  --calibrated-ranges models/bevformer-base/ranges.json \
  --engine results/mixed/base.engine --report results/mixed/build.json \
  --workspace-mib 1024 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0

CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --precision fp32 --plugins-fp32 \
  --engine results/fp32/base.engine --report results/fp32/build.json \
  --workspace-mib 4096 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0
```

`workspace`、`tactic DRAM` 都不是进程总显存上限。新 GPU、不同 TensorRT 或不同输入图可能选择不同候选，engine 应重新构建。

## 交付物与复现限制

配套模型资产包提供已适配 ONNX、校准范围、固定合成输入和相应 PyTorch 参考输出。真实 nuScenes 输入/参考不放进公开源码或模型包；真实图像测试结果通过脱敏 JSON 报告保留。取得同一模型包，可复现合成输入的数值回归；真实图像速度必须使用对应的真实输入重新测量。

历史 3090 的 204.073 ms / 373.821 ms 另见 [历史性能](historical-results.md)。它们属于其他硬件和运行环境，没有冒充本次独立项目的测量结果。
