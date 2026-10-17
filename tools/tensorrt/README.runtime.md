# TensorRT 10 静态模型构建与测速

`run_engine.py` 使用 TensorRT 的 ONNX parser/builder/runtime；PyTorch 只提供
CUDA buffer、stream 和 event，不执行模型算子。此入口与 GAC 原生 MLIR 编译链独立。
FP32/FP16 构建需要 TensorRT 10 和相应驱动；混合 INT8 还需要 ONNX、NumPy 与校准文件。
执行需要 NumPy、可用 CUDA 的 PyTorch。
`build` 通过 `ctypes` 调用 CUDA Driver API 选择设备并绑定 primary context，不导入
PyTorch，不初始化其 CUDA 库。安装的 PyTorch 版本仅从 package metadata 读取。
这样避免给构建额外引入不需要的 PyTorch CUDA 占用，但不保证解决模型本身的显存不足。
构建日志记录 CUDA/TRT 初始化后的全卡显存快照，即使随后 OOM 也能查看。
仅接受静态 shape、
LINEAR 格式 I/O；包括 INT64 在内的张量 dtype 必须与 engine 一致。

ONNX 必须先完成自定义节点适配并经过插件数值验证。注册 NVIDIA 自带插件不会
自动使 `gac::MSDeformAttn` 或 MMCV 节点兼容 TensorRT。
适配器同时把标准 Conv 的常量 weight/bias Identity 链改为直接 initializer 引用；
不改激活边、权重数值或动态权重。报告的 `static_conv_weight_aliases` 保留逐条改写证据。
真实 Base 的 GPU 对照已确认：原 Identity 权重路径构建失败，直接 initializer 路径成功。

```sh
python tools/tensorrt/adapt_onnx.py original.onnx adapted.onnx --report adaptation.json
# fixtures 由 native/tests/generate_base_ops_models.py 生成；六组DCN/MSDA对照。
python tools/tensorrt/verify_plugins.py --fixtures build/base-op-fixtures \
  --output build/trt-plugin-validation-new
```

```sh
# 新路径：已有 engine/report 不会被覆盖。Workspace 并非总显存上限。
NVIDIA_TF32_OVERRIDE=0 python tools/tensorrt/run_engine.py build \
  --onnx adapted.onnx --engine fp32.engine --precision fp32 \
  --plugins-fp32 --workspace-mib 1024 --max-aux-streams 0 --report fp32-build.json

# npz 的 key 必须严格对应 engine I/O 名称；单输入也可传 images.npy。
NVIDIA_TF32_OVERRIDE=0 python tools/tensorrt/run_engine.py run \
  --engine fp32.engine --inputs inputs.npz --reference reference.npz \
  --output fp32-measurement --warmup 3 --repeat 10 --rtol 1e-3 --atol 1e-4
```

`build --precision fp16` 允许普通层选择 FP16，不能解释为所有层都使用 FP16。
`--plugins-fp32` 给 PluginV2/PluginV3 的浮点输入插入 FP32 Identity 边界，并约束
浮点输出为 FP32，启用 `OBEY_PRECISION_CONSTRAINTS`；无法满足时必须构建失败。
整数输入原样保留，尤其 MSDA 的 spatial shapes 和 level start index 必须为 INT32。
不能把混合类型插件的 `layer.precision` 整体设为 FP32。NVIDIA 的 MSDA/DCN 插件
要求浮点输入与输出同型，因此 FP32 输出约束也选择 FP32 计算；其它插件须单独核实契约。
为了使用这些
TRT10 精度控制，网络明确采用 weak typing，未启用 `STRONGLY_TYPED`。
所有构建精度模式都关闭 builder 的 TF32 flag。插件内部独立的 cuBLAS 调用未必受此 flag
影响，因此 FP32 对照同时设置并记录 `NVIDIA_TF32_OVERRIDE=0`。

## 混合 INT8 与逐层计划

`--precision mixed-int8` 对 backbone layer1–4 的普通卷积请求 INT8，对入口、head、
DCNv2 offset/mask 预测和插件浮点部分请求 FP16；公开 I/O 仍保持原来的 FP32。
插件可额外用 `--plugins-fp32` 保持 FP32。具体范围、例外、真实校准数据格式与命令见
[MIXED_PRECISION.md](MIXED_PRECISION.md)。

```sh
python tools/tensorrt/run_engine.py build \
  --onnx build/trt/base.onnx --precision mixed-int8 --plan-only \
  --report build/trt/mixed-plan.json
```

`--plan-only` 仍会在 TensorRT/CUDA 环境中解析模型，只保存逐层约束，不校准、不构建
engine、不推理。直接校准的混合构建必须提供 `--calibration-dir`，其中包含已预处理的真实
输入和 `calibration.json` 来源声明。程序不会用随机输入补齐；校准缓存绑定内嵌权重
ONNX、规则及逐层策略、TensorRT 版本、预处理声明和有序样本哈希。
此路径使用 TensorRT 10 已弃用但仍支持的 entropy PTQ 接口，不是显式 Q/DQ 路径。

构建阶段显存不足时，可单独试 `--builder-optimization-level 0 --tactic-dram-mib 4096`。
级别 0 减少搜索并关闭动态 Kernel 生成，可能降低最终性能；`TACTIC_DRAM` 限制优化器
试跑候选的显存，和 `--workspace-mib` 不同，也不是进程总显存上限。本机 TensorRT
10.13.3.9 的该 API 要求正的二次幂大小，6144 MiB 被拒绝；工具预检并读回确认生效。
这些选项保留原有模型和精度约束，不能保证 8GB GPU 可以构建任意完整模型。

官方 mini 解压后，可以生成 32 帧真实校准输入；该 CPU 预处理入口额外需要 OpenCV：

```sh
python tools/tensorrt/prepare_nuscenes_calibration.py \
  --data-root /home/robot/datasets/nuscenes \
  --output build/trt/mini-train-calibration --max-samples 32 \
  --export-report build/bevformer-base-onnx/export-report.json
```

脚本在八个 `mini_train` 场景内按时间分散采样，`mini_val` 的两个场景不参与校准。
输出 FP32 `[1,6,3,928,1600]` 和带样本/原图哈希的 manifest。相机顺序固定，BGR
减均值、std=1，不转换 RGB、不缩放；归一化后将 900×1600 底部补零至 928×1600。
`--export-report` 用实际导出报告核对 shape 与归一化契约。
当前 ONNX 内的 synthetic all-visible 几何和零 CAN-bus 保持不变；即使输入是真实图像，
head 的激活分布也未必代表真实几何/时序部署，不能代替 nuScenes mAP/NDS 验证。
本次 32 帧真实输入已准备完成：
[准备摘要](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/preparation-summary.json)、
[校准 manifest](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/calibration.json)、
[数据集记录](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/dataset.json)。

成功构建后同时保存 `<report-stem>.layers.json`，包含 engine inspector 的实际格式和
tactic。请求 INT8 的计划只能表达约束，实际采用的格式须查看这个文件。
当前 87 项 CPU 测试（含 19 项适配器、8 项前处理、20 项子图提取、18 项范围转换测试）及小 CNN 的
8 批校准、缓存复用、显式范围 GPU 测试通过。旧完整 Base 已完成 32 帧真实 mini 校准并写出缓存，
但 engine 在选择第 2 相机的首个 INT8 Conv 格式时失败；常量权重 Identity 别名问题已复现并修复。
修复后 390 个 Conv 的 780 条引用改变，新 ONNX 有独立 SHA，不能使用重标的旧缓存。
新的完整计划与 backbone 校准已通过，462 个目标 Conv 的实际 INT8 输入/权重已核对；完整 Base 的数值和 FPS 尚未验收。证据见
[验证报告](../../docs/validation/tensorrt/bevformer-base-2026-10-17/README.md)。

备用入口 [backbone_calibration.py](backbone_calibration.py) 保留六路 backbone 的末端
输出与原始 462 个 INT8 Conv；[calibration_ranges.py](calibration_ranges.py) 将严格绑定的
子图缓存转换成独立范围产物，供 `build --precision mixed-int8 --calibrated-ranges <JSON>`
使用。范围路径不能与校准目录、缓存或 `--plan-only` 混用，缺少任何必需的实测范围就失败。
子图 manifest 逐项记录全部 462 个 Conv 的 924 条 input0/output0 激活边，目标网络必须
保留这些精确名称；缺失、非有限或非正的必需范围不能用默认值补齐。
小 CNN 的显式范围 engine 已通过 GPU 输出对照，与缓存构建输出的最大绝对差为 0，见
[范围接口结果](../../docs/validation/tensorrt/bevformer-base-2026-10-17/mixed-precision-ranges-interface/summary.json)。
它不把子图缓存标为全图缓存，也不改变原完整模型的执行范围；真实 Base 的该路径仍待验收。

## 计时与数值范围

测速目录包含 `outputs.npz` 和 `report.json`，已有非空目录不会被覆盖。
输入 H2D、输出 D2H、预处理、模型加载、构建和 warmup 均在计时区间之外；
每个样本执行一次并同步，记录 CUDA event 时间和主机提交至同步完成的 wall time。
CUDA event 区间可能包含主机提交造成的 GPU 空闲，不能解释为纯 kernel 时间之和。
这里没有 CUDA Graph replay，也没有实现吞吐批处理或应用端到端测速。

只检查最后一次执行的输出，逐元素比较
`abs(actual-reference) <= atol + rtol*abs(reference)`，整数输出要求精确相等。
报告保留非有限检查、误差和失败元素数。无 `--reference` 时明确记录
`not_checked`，不能当作数值通过；非有限值或对照不通过返回非零退出码。
输出 key/shape 不一致立即报错，不进行猜测匹配或自动 reshape。
`--max-aux-streams` 默认 0，避免跨辅助 stream 并行分支延长激活张量的生命期；
它不改变模型数学语义，但会影响性能与显存需求。构建失败时可尝试缩小 workspace，
例如 `--workspace-mib 256`；这仍不能保证模型一定适合 8GB 显存。
报告记录 GPU、TensorRT/PyTorch 版本、精度策略、engine/input/reference hash；
构建报告中的 ONNX hash 覆盖 `.onnx` 本身。混合 INT8 路径要求所有权重内嵌，以使校准
缓存绑定完整模型；普通 FP32/FP16 路径的该哈希不包含独立 external weight 文件。

运行报告还记录 engine 查询到的 execution context device-memory requirement，
以及 context 和 I/O buffer 创建、绑定完成后的一次 `cudaMemGetInfo` 快照。
前者不包含权重、I/O 和其它运行时分配；后者是整张卡的全局 free/total，包含其他
进程的占用，且位于 warmup 之前。两者都不能当作本进程的显存峰值。

API 依据：[NVIDIA Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)、
[TRT10 迁移指南](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html)、
[精度控制](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)。
