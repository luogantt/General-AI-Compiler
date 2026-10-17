# BEVFormer-base TensorRT 验证（2026-10-17）

分支 `lg/bevformer-tensorrt`，独立 TensorRT 对照入口，不经过 GAC MLIR Pass。
测试机 `robot@192.168.3.33`，RTX 3070 Ti Laptop 8GB，TensorRT 10.13.3.9、
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
  失败 Conv 的常量权重别名已用 GPU 对照定位并在适配器中修复，完整新 engine 待验收。

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
范围应用接口已完成上述小网络 GPU 验证。修复后的 Base backbone 已完成真实校准与构建，见下节；完整模型仍待验收。

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
生成引擎时 OOM（新申请 655,520,000 B）。这是构建失败，不是推理峰值测量；完整 FPS 尚无结果。

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
构建、完整模型计划解析与真实 32 帧校准，但尚无完整 Base 构建成功的证据，不能宣称 OOM 已解决。
尚不能由环境、单算子或小 CNN 验证推断完整模型已经通过或报告整网 FPS。
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
不会被标成真实校准结果。新模型将单独记录子图校准和完整 engine 构建结果。

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
