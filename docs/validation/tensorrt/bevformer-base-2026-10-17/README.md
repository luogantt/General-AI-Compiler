# BEVFormer-base TensorRT 验证（2026-10-17）

分支 `lg/bevformer-tensorrt`，独立 TensorRT 对照入口，不经过 GAC MLIR Pass。

同日补测 **全 FP32（关闭 TF32）**：同一 3090、完整模型和输入，中位延时
**373.821 ms，约 2.675 FPS**；混合精度为 **204.073 ms，约 4.900 FPS**，约快 1.832 倍。
FP32 使用 4 GiB workspace，混合精度使用 1 GiB，均为搜索级别 3；不是只改变精度的隔离实验。
FP32 输出通过同帧独立 native FP32 参考的严格逐元素检查，仍不代表检测精度验收。
构建失败与修正、显存和复现命令见 [FP32 对照报告](cloud-3090-fp32/README.md)。

同日已完成 **TensorRT IProfiler + Nsight Systems** 分析：
[交互报告](profiling/index.html)、[全量逐层 CSV](profiling/layers-ranked.csv)、
[采集说明与公开时间线](profiling/README.md)、[公开数据包](profiling/profile-data.zip)。
混合精度 / FP32 分别记录 3819 / 3260 条执行层回调，各预热 10 次、采样 30 次；
Nsight 各捕获 5 帧。混合引擎的 MSDA 插件合计 74.188 ms，占逐层均值之和的 35.08%；
Nsight 独立 trace 中对应采样 kernel 每帧累计 75.118 ms，占 kernel 累计时间的 37.90%。
两个分母和采集 pass 不同，不能混作无 profiler 的整网延时。图仍为无历史帧、合成投影条件；
这些统计不代表真实时序或检测精度验收。下载的 32 个采集文件均通过远端 SHA-256 校验。
公开前已从两个 SQLite 中删除进程 / 系统环境变量记录，并替换 `StringIds` 中的凭据后
执行 VACUUM。逐表行摘要确认除上述净化外，其余数据（包括计时和字符串 ID）不变；
可能包含环境凭据的原始 `.nsys-rep` 不纳入 Git 或 ZIP。
[净化记录](profiling/publication-sanitization.json)保留原文件与公开文件哈希；
`remote-manifest.json` / `download-verification.json` 描述原始下载，
`local-manifest.json` 描述净化后的公开文件，不能要求两组 SQLite 哈希相同。

## 性能分析与车企 / Orin 对比

### 本次 profiling 告诉了我们什么

这是与上方首次测速独立的一组实验。各引擎预热 10 次、采样 30 次，随后分别测量
未启用逐层 profiler 的 baseline、启用 profiler 的全图时间和逐层回调；Nsight 另捕获 5 帧。

| 引擎 | baseline 中位数 / P95 | 执行层回调数 | 逐层均值之和 |
| --- | ---: | ---: | ---: |
| 混合精度 | 204.224 / 232.416 ms | 3819 | 211.475 ms |
| FP32（关闭 TF32） | 375.967 / 398.980 ms | 3260 | 380.366 ms |

混合精度的主要执行层类别如下。占比分母是全部 IProfiler 回调的均值之和，不是
baseline 延时；融合层保持整体统计，不把同一条回调拆成多个原始 ONNX 算子重复计时。

| 执行层类别 | 均值之和 | 层占比 |
| --- | ---: | ---: |
| MSDA 插件 | 74.188 ms | 35.08% |
| Convolution | 37.380 ms | 17.68% |
| kgen 代码生成 / 融合层 | 29.505 ms | 13.95% |
| DCN 插件 | 28.635 ms | 13.54% |
| Reformat | 18.348 ms | 8.68% |
| MatrixMultiply | 11.231 ms | 5.31% |

Convolution 分类可能包括被映射为卷积的矩阵运算及融合 epilogue，不能全部解释成
CNN backbone。最慢的六条回调是六层 encoder 的空间 MSDeformAttn，每条约
11.38–11.73 ms。独立 Nsight trace 中，混合引擎每帧有 3929 次 Kernel 启动；
`ms_deformable_im2col_gpu_kernel<__half>` 每帧累计约 75.118 ms，占 Kernel 累计时间
37.90%。这些证据支持优先检查空间注意力的工作量和插件实现，但尚未采集 Nsight
Compute 硬件计数器，不能仅凭耗时判定显存带宽、缓存命中或计算利用率的根因。

### 当前几何条件为什么影响比较

本次导出图使用 synthetic all-visible 投影。空间 MSDA 的采样位置 shape 为
`[6, 40000, 8, 4, 8, 2]`，六个相机各处理全部 40,000 个 BEV 查询。
这不等于真实相机视野下的工作量。[官方 SpatialCrossAttention 实现](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py)
先用 `bev_mask` 为每个相机筛选可见查询，再按相机中最大的可见数量 `max_len` 补齐、
重组并执行注意力；不能简单把所有相机有效查询数相加就当成实际 Kernel 规模。

所以，约 204 ms 是当前固定图和输入条件下的设备序列延时，不能直接推断真实车载
BEVFormer 的延时，也不能用它证明所有量产 BEV 网络在 Orin 上都无法实时。
真实投影可能减少空间注意力的工作量，但历史帧、输入变化和实际导出方式也会影响性能，
具体收益需要重新测量，不能预先保证达到 30 FPS。

### 车企公开的网络不等于开源 BEVFormer-base

以下为官方披露的历史代际，核查日期为 2026-10-17，不代表厂商当前所有车型和版本。
BEV 是空间表示，Transformer 是网络组件；二者不能唯一确定具体 backbone、层数或权重。

| 厂商 / 披露时间 | 官方披露 | 可确认的边界 |
| --- | --- | --- |
| 小鹏 G6，2023-06-29 | 自研 BEV + Transformer 的 XNet，双 Orin-X | 没有据此公开可复现的完整网络配置和单帧延时，不能认定为官方 BEVFormer-base。[官方发布](https://www.xpeng.com/news/0189077394e388fa98d62c9e8d790065) |
| 理想 L7，2023-02-08 | 融合 BEV 算法框架；AD Max 双 Orin-X，AD Pro 征程 5 | 不能据此认定量产直接使用开源 BEVFusion 或 BEVFormer 的原配置、原权重。[官方发布](https://ir.lixiang.com/zh-hant/news-releases/news-release-details/li-auto-inc-launches-li-l7-five-seat-flagship-family-suv/) |

后续系统也持续演进：小鹏 2024 年披露 [XNet + XPlanner + XBrain](https://www.xiaopeng.com/news/company_news/5307.html)，
理想 2025 年初披露 [端到端模型 + VLM 双系统](https://ir.lixiang.com/zh-hant/news-releases/news-release-details/li-auto-inc-december-2024-delivery-update/)。
这些系统名称不能直接对应某个公开 checkpoint，也不提供与本实验同口径的 FPS。

### 可复核的 Orin 公开参考

[NVIDIA CUDA-BEVFusion](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion)
公开了 BEVFusion-Base、ResNet-50、相机输入 `256×704` 的以下结果：

| 公开配置 | Orin FPS | mAP / NDS |
| --- | ---: | ---: |
| TensorRT FP16 | 18 | 67.89 / 70.98 |
| TensorRT FP16 + INT8 PTQ | 25 | 67.66 / 70.81 |

官方说明使用 TensorRT 8.6、CUDA 11.4、cuDNN 8.6，在 nuScenes 6019 个验证样本上统计，
LiDAR 点数影响速度。该模型使用相机和 LiDAR，属于 **BEVFusion，不是 BEVFormer**；
表格不代表任何量产车型的整套智驾速度，也不能和我们的 `928×1600` 六相机 Base 图
做相同任务、相同精度的直接性能排名。

工程上可共同调整 backbone、输入分辨率、查询数量、层数、精度、时序复用和 Kernel
实现，以适配延时预算；结构调整通常需要重新训练并验证任务精度。摄像头帧率、感知
网络频率与控制循环频率也不是同一个指标；流水线提高吞吐，不自动消除单帧依赖延迟。
本项目下一步应先建立真实几何与连续帧的性能 / 精度基线，再按实测热点优化；若固定
官方 Base 配置，仅凭编译优化，目前没有证据保证 Orin 上达到 30 FPS。

## 最新结果：3090 完整混合精度引擎已执行

云机器实际为 **RTX 3090 24GB，并非 4090**。源代码提交
`cf34273c6d1853a302bbfc24c59d4afce46fbd16`；TensorRT 10.13.3.9、PyTorch
2.7.0+cu126、Python 3.11.15、驱动 580.126.18。使用修复权重引用后的完整 Base ONNX，
读取原机上 32 帧真实 mini_train 图像测得的 backbone 校准范围，在云端完成构建和推理。
环境依赖通过清华 PyPI 镜像及 NVIDIA 包源安装到独立 venv，未修改系统驱动。

| 检查 | 结果 |
|---|---|
| 完整引擎构建 | 成功；构建进程约 342 秒 |
| 实际 INT8 覆盖 | 462/462 个目标 backbone Conv 的激活和权重为 INT8，输出 FP16 |
| 测速方法 | 同一真实 mini_val 帧，预热 5 次、测量 20 次 |
| CUDA event 中位数 / P95 | **204.073 / 206.887 ms** |
| 中位延时倒数 | **约 4.90 帧/秒**；不是应用端到端吞吐 |
| 主机提交至同步完成中位数 | 204.091 ms |
| 构建时整卡显存采样峰值 | 9346 MiB，约 9.13 GiB |
| 推理进程期间整卡显存采样峰值 | 2570 MiB，约 2.51 GiB |
| 输出有限性 | 三路输出全部有限，无 NaN/Inf |

显存每 250 ms 采样一次，是整卡占用，包含基线和其他进程，不是本进程独占分配；
采样可能漏掉更短的峰值。上述构建需求不能直接当作推理需求，也不能据此保证
该引擎在其他 GPU 上可移植。计时排除模型加载、预热、输入输出复制、预处理和后处理，
没有使用 CUDA Graph replay。**当前没有达到 30 FPS（每帧约 33.3 ms）的目标。**

同一帧另用原机上的独立 GAC native MLIR FP32 实现生成参考，并检查输入、模型与常量身份：

| 输出 | 相对 RMSE | 余弦相似度 | 最大绝对误差 |
|---|---:|---:|---:|
| `bev_embed` | 2.419% | 0.999707 | 0.211466 |
| `all_cls_scores` | 0.539% | 0.999985 | 0.773867 |
| `all_bbox_preds` | 0.475% | 0.999989 | 1.104830 |

`rtol=1e-3, atol=1e-4` 的严格逐元素诊断阈值**未通过**。这不是新一次 PyTorch 真值运行，
差异包含混合精度及后端实现差异，不能全部归因于 INT8，也不能把特征误差解释为检测准确率下降。
此次执行的是完整固定形状 **cold raw-head** 导出图：R101/DCNv2、4 层 FPN、200×200 BEV、
6 层 encoder、六相机输入 `[1,6,3,928,1600]`。图内仍是合成全可见投影、zero CAN-bus，
没有历史帧或框解码。**真实几何/时序推理及 nuScenes mAP/NDS 尚未验收。**

证据：[汇总](cloud-3090/summary.json)、[构建](cloud-3090/full-build.json)、
[测速](cloud-3090/run.json)、[实际精度格式](cloud-3090/full-precision-audit.json)、
[逐输出误差](cloud-3090/native-comparison.json)、[输入来源](cloud-3090/held-out-input-provenance.json)、
[独立参考来源](cloud-3090/native-heldout-reference.provenance.json)、
[环境](cloud-3090/environment.json)、[依赖版本](cloud-3090/packages.txt)。
原始 run 报告未传入 `--reference`，保留 `not_checked`；数值对照由独立离线报告记录。
完整 inspector、日志和输出保留于本地 `build/tensorrt-cloud-20261017/`，大模型和 engine
保留于云端 `/root/gac-tensorrt-20261017/`；汇总记录文件哈希。

云端复现命令（模型和范围文件身份见汇总，先准备相应文件）：

```sh
cd /root/gac-tensorrt-20261017
NVIDIA_TF32_OVERRIDE=0 venv/bin/python tools/tensorrt/run_engine.py build \
  --onnx models/full.onnx --precision mixed-int8 --calibrated-ranges models/ranges.json \
  --engine results/mixed-01/full.engine --report results/mixed-01/full-build.json \
  --workspace-mib 1024 --builder-optimization-level 3 --max-aux-streams 0 --verbose
NVIDIA_TF32_OVERRIDE=0 venv/bin/python tools/tensorrt/run_engine.py run \
  --engine results/mixed-01/full.engine --inputs models/held-out-images.npy \
  --output results/mixed-01/heldout --warmup 5 --repeat 20
```

下文保留此前 **3070 Ti Laptop 8GB** 的校准、失败诊断与 backbone 测试过程，
不把不同 GPU、子图或不同参考的结果混为同一次实验。

## 原 3070 Ti 测试环境

测试机原地址 `robot@192.168.3.33`，当前为 `robot@192.168.3.24`，RTX 3070 Ti Laptop 8GB，TensorRT 10.13.3.9、
Python 3.11.16、PyTorch 2.7.0+cu126、驱动 580.178.04。
运行目录 `/home/robot/project/gac-tensorrt-20261017`；系统驱动与原 Isaac 环境未修改。

## 已完成的验证

- `environment.json`：实际构建并执行 FP32 加法 engine，确认两个 NVIDIA 内置 v2 插件存在。
- `plugin-validation.json`：三个 DCNv2、三个 MSDA fixture 在 GPU 执行后，全部通过
  `rtol=2e-5, atol=2e-5` 对照。参考来自 PyTorch/torchvision，不来自 TensorRT。
- `plugin-fp16-constrained.json`：允许 FP16、插件保持 FP32 的配置，也通过同六组测试；
  验证了混合类型插件的精度约束修复。它不代表完整模型的 FP16 精度已经通过。
- 87 项 CPU 单元测试通过：19 项适配器、10 项精度策略、12 项校准与缓存、
  8 项 nuScenes 预处理、20 项 backbone 提取、18 项范围转换测试，覆盖插件 ABI、
  精度例外、输入合法性、缓存错配、回调异常、完整消费检查、真实数据预处理契约
  和备用子图/范围路径。
- 完整 Base ONNX 已适配：22,412 → 22,538 节点，156 个 DCNv2、18 个 MSDA。
  额外节点只用于 MSDA shape 恢复；没有裁剪模型、预计算激活或 CPU 算子回退。
- 32 帧真实 mini 输入完成旧完整图校准并写出缓存；随后 engine 构建失败。
  失败 Conv 的常量权重别名已用 GPU 对照定位并在适配器中修复；后来在 3090 构建完整引擎，见上节。

## 混合精度：完整计划与小网络实测

修复权重别名前的完整 Base（ONNX SHA `0f24cc097292b6a56336175a992254b42086256169336bae4f2f22ea4c663552`）
的 TensorRT 解析结果为 60,935 个网络层，策略为插件浮点输入增加 678 个
边界层后共 61,613 层。显式约束 462 个 INT8 层和 2,391 个 FP16 计算/插件层。
backbone 中有 312 项例外：156 个 DCNv2 插件不支持 INT8，156 个 offset/mask
预测卷积保持浮点以保护采样坐标。另有 58,082 层因整数/非浮点输出或布局、shape 等
类别未施加显式计算精度约束。解析层数不等于 ONNX 算子数或 GPU kernel 数。

- [mixed-int8-plan-summary.json](mixed-int8-plan-summary.json)：环境、规则、各类计数、
  原始完整计划 SHA-256 及规则/策略指纹。
- [mixed-int8-layer-plan.csv](mixed-int8-layer-plan.csv)：2,853 个受约束层的名称、scope、
  计算/输出精度、浮点输入输出索引和例外原因。`constraint_index` 是约束清单序号，
  不是 TensorRT parser 层号；未约束层以摘要计数保留。

原始完整计划在本地 `build/tensorrt-session/mixed-int8-plan.json`，不纳入版本库。
这份计划本身没有执行校准、engine 构建或推理，不能证明实际精度或性能。
权重引用修复后的新模型需重新生成计划，不能把旧计划哈希当作新模型的证据。

独立三层小 CNN 的 GPU 集成测试已通过，输入为明确标记的合成数据 `[1,3,8,8]`：

| 检查 | 实测结果 | 证据 |
|---|---|---|
| 首次构建 | 实际消费 8 批校准输入 | [first-build.json](mixed-precision-interface/first-build.json) |
| 第二次构建 | 命中缓存，消费 0 批 | [cached-build.json](mixed-precision-interface/cached-build.json) |
| 实际输入/权重格式 | stem `Half`、backbone Conv `Int8`、head `Half`，三层浮点输出均为 `Half` | [cached-build.layers.json](mixed-precision-interface/cached-build.layers.json) |
| 最终输出对照 | 全部有限，256 个元素零超差，最大绝对误差约 `2.79e-4` | [summary.json](mixed-precision-interface/summary.json) |

该测试使用 `rtol=0.05, atol=0.002`，仅验证校准、精度约束、缓存和执行接口。
公开 I/O 保持 FP32。它不是 BEVFormer 的校准、精度或 FPS 结果。
新增范围接口 GPU 测试也完成了首次校准、缓存复用和独立显式范围构建。范围 engine 的
stem/body/head 仍为 `Half/Int8/Half`；与缓存 engine 的 256 个输出元素最大绝对差为 0。
见 [mixed-precision-ranges-interface/summary.json](mixed-precision-ranges-interface/summary.json)。
这一对照仍使用合成小 CNN；没有测量完整 Base 的精度或 FPS。
策略与数据契约见 [MIXED_PRECISION.md](../../../../tools/tensorrt/MIXED_PRECISION.md)。

备用 `backbone_calibration.py` 已在修复前的真实 Base 图上完成 CPU 提取与结构核对：5,516 个节点、
338 个 initializer、原 462 个待量化 Conv 全覆盖；只保留六个 `[1,2048,29,50]` 末端输出，
未把内部 Conv 输出全部标记为公开输出。保留节点/权重 protobuf 与输入、tensor 名称相同。
`calibration_ranges.py` 单独绑定子图缓存与范围产物，不能把子图缓存冒充全图缓存。
新 manifest 还记录 462 个 Conv 的 924 条 input0/output0 激活边；转换及全图应用都按精确
张量名称核对，必需范围缺失或无效时失败，不填默认值。子图提取已完成 CPU 验证，
范围应用接口已完成上述小网络 GPU 验证。修复后的 Base backbone 已完成真实校准与构建，见下节；后续 3090 完整模型执行及精度边界见本页开头。

## 真实 32 帧校准与权重别名修复

[full-calibration-summary.json](nuscenes-mini/full-calibration-summary.json) 记录旧完整图消费
全部 32 帧、写出缓存及 7,443 条范围记录。随后构建失败于
`/img_backbone/layer1/layer1.0/conv1_1/Conv`，错误为 `No supported formats`；
没有生成可执行的完整 engine，因此没有整网数值或延时结果。

这层输入/输出 `[1,64,232,400]`、权重 `[64,64,1,1]`，激活校准范围有效。
首相机的 77 个 INT8 Conv 使用直接 initializer，后五相机的 385 个使用共享权重的
Identity 别名。TensorRT parser 把后者导入为 tensor 权重路径。使用该失败层的真实权重、
shape 和已测范围进行 GPU 对照：Identity 版本失败，直接 initializer 版本构建成功，见
[conv-weight-alias-probe.json](nuscenes-mini/conv-weight-alias-probe.json)。

适配器现只重写标准 Conv 的确定常量 weight/bias 引用，共 390 个 Conv、780 条引用
（含 5 个 FP16 stem）；全部 462 个 INT8 Conv 已引用直接 initializer。原权重数值、
激活 input0、节点名称、输出、模型 IO 和 Identity 节点保留，动态或可覆盖权重不折叠。
CPU 随机小模型对照结果完全一致，真实 Base 的结构逐项核对通过；这不替代整网精度验证。
新适配模型 SHA-256 为 `ad34bc2e896916e719e76c50fb5136fecbbf1f8a0fe76543e8743f2e397669be`。

后续使用这个新模型重新解析计划、提取原六路 backbone，使用同一真实 32 帧重新校准，
再把独立范围产物应用到新完整图。旧完整图缓存继续保留原身份，不能重标成新模型或子图缓存。
新计划、精确子图提取和 32 帧校准已完成，backbone engine 为 124,234,100 B。
[backbone-built-summary.json](nuscenes-mini/backbone-built-summary.json) 保存构建和校准摘要，
[backbone-precision-audit.json](nuscenes-mini/backbone-precision-audit.json) 核对了缓存策略/模型绑定，
并逐个确认 462 个目标卷积的 INT8 输入/权重与 FP16 输出。此结论仅覆盖 backbone 引擎。

完整图最初仅传 Conv 两端范围，在 Conv+Add+ReLU 融合处失败；现在还传递已有实测范围的
backbone 激活，排除静态权重、shape 与未知 head。诊断构建已越过融合错误，但在后续
生成引擎时 OOM（新申请 655,520,000 B）。这是原 3070 Ti 上的构建失败，不是推理峰值测量；当时没有完整 FPS 结果。

正式范围传递实现再次使用 `builder_optimization_level=0`、`TACTIC_DRAM=4096 MiB`、
workspace 512 MiB、aux streams 0 构建完整图，仍在相同申请量处 OOM。
优化级别与候选显存限制均不改变模型/精度约束。这证明当前整图构建配置仍受阻，
不能据此断言任何分阶段执行方式都无法在 8GB 上运行。原始日志哈希与错误行见
[full-build-failure.json](nuscenes-mini/full-build-failure.json)。

## 未参与校准的真实帧：backbone 执行与误差

使用 `mini_val` 的 `scene-0103` 一帧六相机图像，比较同一 backbone ONNX 的
TensorRT FP32 与混合 INT8 引擎；该帧未参与 32 帧校准。两个进程顺序执行，均预热 2 次、
计时 5 次，排除加载、拷贝与预处理。**以下不是完整 BEVFormer 的时间。**

| 引擎 | CUDA event 延时中位数 | 输出 |
|---|---:|---|
| FP32 backbone | 396.02 ms | 六路末端特征均有限 |
| 混合 INT8 backbone | 146.72 ms | 六路末端特征均有限 |

FP32 构建搜索级别为 0，混合 INT8 为 3；这不是仅改变量化位宽的隔离性能实验。
FP32 禁用 FP16/TF32，插件保持 FP32；参考是同图 TensorRT 浮点引擎，不是另一次
独立 PyTorch 模型运行。六路输出余弦相似度为 0.98845–0.99343，
相对 RMSE 为 11.49%–15.21%。这是特征误差，**不是检测准确率下降的百分比**。
尚未设定任务精度验收线，也未测 nuScenes mAP/NDS，不能宣称量化精度达标。

证据：[比较与输入来源](nuscenes-mini/backbone-heldout-comparison.json)、
[混合 INT8 执行](nuscenes-mini/backbone-int8-run.json)、[FP32 执行](nuscenes-mini/backbone-fp32-run.json)。

## 早期整网构建结果

以下尝试来自会初始化 PyTorch CUDA 的旧 builder，均保留原始日志：

| 精度 | Workspace / aux streams | 结果 | 日志 |
|---|---|---|---|
| FP32 | 1024 MiB / TRT 默认 | OOM，单次申请 1,789,657,600 B | `base-fp32-build-oom.log` |
| FP32 | 256 MiB / 0 | attention fused region 的 tactic 至少申请 321,922,560 B，找不到可用实现 | `base-fp32-lowmem-build.log` |
| FP32 | 512 MiB / 0 | OOM，单次申请 1,247,040,000 B | `base-fp32-512-build.log` |
| FP16，插件 FP32 | 512 MiB / 0 | OOM，单次申请 666,080,000 B | `base-fp16-plugins-fp32-build.log` |

这些数字是失败的单次申请，**不是模型总显存需求或实测峰值**；workspace 也不是总显存上限。
现在 builder 已直接使用 CUDA Driver 初始化，不再导入 PyTorch CUDA；新路径已完成小网络
构建、完整模型计划解析与真实 32 帧校准。该 8GB 测试阶段没有完整 Base 构建成功的证据；
后续 24GB 3090 的成功构建不表示 8GB 构建 OOM 已解决。
静态张量容量与后续分块方向见 [memory-analysis.md](memory-analysis.md)。

## 模型与比较范围

使用此前原生 CUDA 测试的同一组文件，哈希在本地及远端核对一致：

| 文件 | SHA-256 |
|---|---|
| `bevformer-base-cold.onnx` | `85ce569241b6ac540d721244e91cd20ca3966631abb749ab82c98a54f7f32aed` |
| `images.npy` | `dae4f3b37e0d266b284d2268f23a6ae6f355aa0d96816fff85f8146a7e45ba28` |
| `pytorch-outputs.npz` | `77dd2317d73aee213d99bcc1837a2570161e03241bfa4d8fd50cc7fc2ae67181` |

Base：R101/DCNv2、4 层 FPN、200×200 BEV、6 层 encoder，输入 `[1,6,3,928,1600]`。
固定随机图像、合成全可见投影、zero CAN-bus、cold frame、camera_batch_chunk=1。
输出为 `bev_embed`、`all_cls_scores`、`all_bbox_preds`。
这是完整 cold raw-head 路径，不包含数据集预处理、历史帧、框解码及 NDS/mAP 验证。
原始导出限制见 [export-reference-contract.json](../../cuda/native-mlir-bevformer-base-2026-10-17/export-reference-contract.json)。

真实图像校准输入已从官方 nuScenes mini 准备完成：八个 `mini_train` 场景各选 4 帧，
共 32 帧、192 张相机图像；两个 `mini_val` 场景保留用于验证。预处理脚本核对原始 JSON 表关系、
相机顺序及图像尺寸，执行 BGR 减均值、std=1，归一化后底部补零至 928×1600；不缩放。
manifest 记录 sample token、原图与输出哈希及实际导出报告。证据为
[preparation-summary.json](nuscenes-mini/preparation-summary.json)、
[calibration.json](nuscenes-mini/calibration.json)、[dataset.json](nuscenes-mini/dataset.json)。
全部 32 帧已在旧完整图完成校准，但后续 engine 构建失败；以上既有随机输入对照文件
不会被标成真实校准结果。新模型的子图校准见上文，完整 engine 的 3090 构建结果见本页开头。

当前 ONNX 的相机投影仍为固定的 synthetic all-visible 几何，CAN-bus 仍为零。
真实图像不能修正这一导出边界，校准出的 head 激活统计不一定适用于真实几何和跨帧部署；
完整 nuScenes mAP/NDS 需要真实几何、时序输入与独立验证流程。

运行计时排除构建、加载、H2D/D2H、warmup 与预后处理。CUDA event 区间可包含主机提交空隙，
不能解释为各 kernel 纯执行时间之和。整网 FP32 对照固定 `rtol=1e-3, atol=1e-4`。

## 复现

用带 CPU PyTorch、torchvision、ONNX 的环境生成独立算子参考，然后把 fixture 传到 GPU 机：

```sh
python native/tests/generate_base_ops_models.py build/base-op-fixtures
python -m unittest discover -s tools/tensorrt -p 'test_*.py' -v
python tools/tensorrt/verify_plugins.py --fixtures build/base-op-fixtures \
  --output build/trt-plugin-fp32-new

python tools/tensorrt/adapt_onnx.py \
  build/bevformer-base-onnx/bevformer-base-cold.onnx \
  build/trt/base.onnx --report build/trt/adapt.json

# 独立 GPU 接口测试，使用合成小 CNN，不能作为 BEVFormer 校准数据。
python tools/tensorrt/test_mixed_precision_gpu.py build/trt-mixed-interface-new

# 官方 mini 解压后运行；输出目录必须新建。此步骤只准备输入，不构建 engine。
python tools/tensorrt/prepare_nuscenes_calibration.py \
  --data-root /home/robot/datasets/nuscenes \
  --output build/trt/mini-train-calibration --max-samples 32 \
  --export-report build/bevformer-base-onnx/export-report.json
```

精度与引擎构建/执行命令见 [README.runtime.md](../../../../tools/tensorrt/README.runtime.md)。
所有报告目录、engine 路径应新建；验证工具不会覆盖已有结果。

## 遇到并修正的接口问题

MSDA 同时接收浮点数据与 INT32 shape/index，不能把整个 plugin layer 的计算精度
统一强制为 FP32。旧配置会报 `No supported formats`。现在只约束浮点输入/输出，
保持整数边不变。FP32 baseline 禁用 FP16/TF32，并使用 `NVIDIA_TF32_OVERRIDE=0`；
这也限制插件内部独立 cuBLAS 调用的 TF32 行为。
