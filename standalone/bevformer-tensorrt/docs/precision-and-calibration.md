# 混合精度与校准

最短复现路径是用配套的 `adapted.onnx` 与 `ranges.json` 构建混合精度 engine。重新校准只在更换模型、权重、预处理或代表性数据时进行，不能把随机测试输入用作真实 PTQ 校准。

## 精度规则

| 部分 | 请求的精度 |
|---|---|
| 六相机入口 stem | FP16 |
| backbone layer1–4 普通 Conv/downsample | INT8 输入与权重，浮点输出 FP16 |
| DCNv2 offset/mask 预测卷积 | FP16 |
| DCNv2/MSDA 插件浮点数据 | FP16，可用 `--plugins-fp32` 保留 FP32 |
| FPN、Transformer、检测头浮点算子 | FP16 |
| shape/index/bool | 保留整数/布尔类型 |
| 公开模型 I/O | 原始 FP32 |

`fp32`、`fp16` 和 `mixed-int8` 都关闭 TensorRT builder 的 TF32 flag。FP32 严格对照另设 `NVIDIA_TF32_OVERRIDE=0`，因为插件内部独立调用可能不受 builder flag 控制。

使用 TensorRT 10 的 weak typing＋`OBEY_PRECISION_CONSTRAINTS`。`--plugins-fp32` 约束插件浮点边界，不把混合整数/浮点输入的整个插件统一指定成 FP32。实际采用的格式和 tactic 记录于 `build.layers.json`，不能仅凭计划证明量化已生效。

当前 PTQ 依赖 TensorRT 10 的 `IInt8EntropyCalibrator2`，不是显式 Q/DQ 量化。该接口已弃用，本项目锁定 TensorRT **10.13.3.9**，不声称兼容 TensorRT 11。

## 1. 检查整图精度计划

```bash
python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --precision mixed-int8 --plan-only \
  --report results/calibration/full-plan.json
```

该命令需要 TensorRT/CUDA 环境。只解析并记录约束，不运行校准、不产生 engine。

## 2. 准备真实校准输入

```bash
python -m bevformer_trt prepare-calibration \
  --data-root data/nuscenes \
  --output data/calibration-mini-train --max-samples 32 \
  --export-report models/bevformer-base/export-report.json
```

在 8 个 `mini_train` 场景内分散选择时间点；2 个 `mini_val` 场景留作验证。输出每帧约 102 MiB，32 帧约 3.2 GiB，再加来源 manifest。图像处理规范见 [模型与数据契约](model-and-data.md)。

校准目录需要 `calibration.json`。例如：

```json
{
  "schema_version": 1,
  "source_kind": "real_preprocessed",
  "dataset": "真实数据集与校准划分说明",
  "preprocessing": "相机顺序、BGR均值归一化、padding等实际过程",
  "files": ["frame_000.npy", "frame_001.npy"]
}
```

文件清单只是格式示例。实际 `.npy` 必须是有限 FP32 `[1,6,3,928,1600]`；缺失、错误 shape/dtype 不会被随机数补齐。来源声明用于审计，不会自动证明样本真实或具有代表性。

## 3. 提取 backbone 并校准

```bash
python -m bevformer_trt backbone \
  --onnx models/bevformer-base/adapted.onnx \
  --precision-plan results/calibration/full-plan.json \
  --output results/calibration/backbone.onnx \
  --manifest results/calibration/backbone-manifest.json

NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx results/calibration/backbone.onnx --precision mixed-int8 \
  --calibration-dir data/calibration-mini-train \
  --calibration-cache results/calibration/backbone.cache \
  --engine results/calibration/backbone.engine \
  --report results/calibration/backbone-build.json \
  --workspace-mib 512 --max-aux-streams 0
```

提取器保留六路 backbone 的节点、参数和精确张量名称，输出六个末端特征图。校准子图与完整推理图不同：只在需要 INT8 的主体测量范围，推理仍执行完整 Base。

缓存绑定内嵌权重 ONNX、精度策略、TensorRT 版本、样本及预处理哈希。任何改变都需重新校验；**不能把子图缓存重命名成整图缓存**。

## 4. 转换范围并构建完整图

```bash
python -m bevformer_trt calibration-ranges convert \
  --cache results/calibration/backbone.cache \
  --subgraph-manifest results/calibration/backbone-manifest.json \
  --output results/calibration/ranges.json

NVIDIA_TF32_OVERRIDE=0 python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx --precision mixed-int8 \
  --calibrated-ranges results/calibration/ranges.json \
  --engine results/calibration/full.engine \
  --report results/calibration/full-build.json \
  --workspace-mib 1024 --max-aux-streams 0
```

范围产物检查完整源模型和策略绑定、必需激活边、有限正范围。缺失范围直接报错，不使用默认值伪造。`--calibrated-ranges` 不能与 `--calibration-dir`、`--calibration-cache` 或 `--plan-only` 混用。

完整图也可直接使用 `--calibration-dir` 与 `--calibration-cache` 校准，但构建/校准显存较大；并不因为只量化 backbone 就不执行其他部分。

## 5. 验证和性能分析

```bash
python -m bevformer_trt run \
  --engine results/calibration/full.engine \
  --inputs models/bevformer-base/images.npy \
  --reference models/bevformer-base/reference.npz \
  --output results/calibration/comparison --warmup 5 --repeat 20
```

上述比较保留默认严格逐元素门限。历史完整混合图未通过这一门限；应读取报告分析误差，而非把门限随意放宽到通过。之后还要用正确几何、时序和真实验证集验收检测指标。

显存不足时，可尝试 `--builder-optimization-level 0 --tactic-dram-mib 4096`。这会限制搜索，可能降低性能；`TACTIC_DRAM` 要求正的二次幂大小。它和 workspace 都不是进程总显存上限，不能保证任意 8 GB GPU 构建完整模型。
