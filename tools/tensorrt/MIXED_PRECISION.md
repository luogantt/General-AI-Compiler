# BEVFormer Base：FP16 入口与 head，INT8 backbone 卷积

这条路径由 TensorRT 10.13 执行 PTQ 校准、融合、tactic 选择与 engine 推理，
与 `native/` 的 MLIR 编译链独立。官方 nuScenes mini 已下载，32 帧真实六相机
输入已完成一次完整图校准并生成缓存；随后 engine 构建因共享常量权重的导入方式失败。
已通过 GPU 对照定位并修复 Conv 权重 Identity 别名问题。修正后的 backbone 已完成
32 帧真实校准并构建 engine，逐层检查确认 462 个目标 Conv 的输入和权重为 INT8。
尚未完成完整 BEVFormer 的 TensorRT engine、精度验收或 FPS 测量。
在一帧未参与校准的真实六相机输入上，混合 INT8 backbone 的延时中位数为 146.72 ms，
所有末端特征有限；这不包含 FPN/Transformer/head。与同图 FP32 backbone 对照的
特征相对 RMSE 为 11.49%–15.21%，尚不能判定检测精度达标。完整图在降低构建搜索开销后
仍因显存不足失败，详见[验证记录](../../docs/validation/tensorrt/bevformer-base-2026-10-17/README.md)。

## 精度落在什么位置

| 部分 | 当前策略 | Base 导出图/解析结果 |
|---|---|---:|
| 输入 stem：conv1、BN/ReLU、pool 等 | FP16 计算 | 6 个相机 stem Conv |
| backbone layer1–4 普通 Conv、downsample | INT8 输入/权重计算，浮点输出 FP16 | 462 个 Conv |
| DCNv2 的 offset/mask 预测卷积 | FP16，保护采样坐标 | 156 个 Conv |
| DCNv2 插件 | FP16；内置插件不支持 INT8 | 156 个插件 |
| FPN、Transformer、检测 head 的浮点计算 | FP16 | Conv、MatMul、归约、激活等 |
| MSDA 插件 | 浮点部分 FP16，shape/index 保持 INT32 | 18 个插件 |

`--plugins-fp32` 可单独把 DCNv2/MSDA 的浮点边界保留 FP32，用于进一步定位精度误差。
第一版量化的是 backbone 主体的**普通卷积计算**，没有声称每个 backbone 算子和
中间张量都为 INT8。卷积间可能有量化、反量化或布局转换，性能收益需要实测。
公开模型 I/O 与校准文件仍使用原导出契约的 FP32；入口算子在 engine 中转为 FP16。
内部累加精度由所选后端实现决定，不能从一个 FP16 属性推断全部中间算术精度。

对应代码：

- [precision_policy.py](precision_policy.py)：scope 分类、逐层精度约束、例外清单和规则版本。
- [quantization.py](quantization.py)：校准样本、CUDA buffer 生命周期、entropy 校准与缓存绑定。
- [run_engine.py](run_engine.py)：ONNX 解析、策略应用、构建、实际引擎层检查和推理验证。

所有计算精度约束启用 `OBEY_PRECISION_CONSTRAINTS`。不满足时构建应失败。
最终的 `.layers.json` 保存引擎实际格式、权重类型和 tactic，不能只用请求清单证明已量化。

## 没有数据时先看逐层计划

先用 [adapt_onnx.py](adapt_onnx.py) 适配自定义节点。然后：

```sh
python tools/tensorrt/run_engine.py build \
  --onnx build/trt/base.onnx --precision mixed-int8 --plan-only \
  --report build/trt/mixed-plan.json
```

此命令解析真实模型并生成每层约束，**不校准、不构建 engine、不推理**。
在本次 Base 上，TensorRT 解析为 60,935 个网络层，约束 462 个 INT8 计算层与
2,391 个 FP16 计算/插件层；大量其余层属于 shape、常量、布局处理等。
解析层数不等于 ONNX 算子数，更不等于 GPU kernel 数。
精简计划及逐层表见 [验证目录](../../docs/validation/tensorrt/bevformer-base-2026-10-17/README.md)。

## 有真实数据后校准

官方 mini 解压后，可直接按 Base 导出契约准备 32 帧：

```sh
python tools/tensorrt/prepare_nuscenes_calibration.py \
  --data-root /home/robot/datasets/nuscenes \
  --output build/trt/mini-train-calibration \
  --max-samples 32 --export-report build/bevformer-base-onnx/export-report.json
```

脚本按 mini_train 的八个场景均衡抽样，并在每个场景内均匀选择时间点；
mini_val 的两个场景留作验证。它根据原始 JSON 表关联六路相机，使用 BGR 均值
归一化，再将 900×1600 图像底部补零至 928×1600；不缩放图像。
输出约 3.42 GB，manifest 记录场景、sample token、图像哈希和预处理契约。
当前 ONNX 的相机几何仍是导出时固定的 synthetic all-visible 投影与零 CAN-bus；
真实图像校准不等于真实几何、时序完整的 nuScenes mAP/NDS 评估。

目录中的每个 `.npy` 文件是一帧已经预处理好的六相机输入，必须精确为
`float32 [1,6,3,928,1600]`；`.npz` 则必须只有 `images` 这个 key。
输入顺序、相机顺序、归一化、padding 应与模型导出一致，程序不猜测这些操作。
同目录放 `calibration.json`，明确列出文件和预处理过程，例如：

```json
{
  "schema_version": 1,
  "source_kind": "real_preprocessed",
  "dataset": "实际采集或数据集的校准子集说明",
  "preprocessing": "填写与该导出模型一致的相机顺序、归一化和padding流程",
  "files": ["frame_000.npy", "frame_001.npy"]
}
```

文件列表只是格式示例，实际数据应覆盖部署场景，校准集与精度评估集应分开。
来源声明用于记录，不会自动证明样本真实或有代表性。缺少文件、shape/dtype 不符、
NaN/Inf、缓存不匹配都会报错；不会用随机数补齐。

```sh
NVIDIA_TF32_OVERRIDE=0 python tools/tensorrt/run_engine.py build \
  --onnx build/trt/base.onnx --precision mixed-int8 \
  --calibration-dir /path/to/real-calibration \
  --calibration-cache build/trt/base-mixed.cache \
  --workspace-mib 512 --max-aux-streams 0 \
  --engine build/trt/base-mixed.engine --report build/trt/base-mixed-build.json
```

512 MiB 是 tactic 工作区上限，不是整网总显存上限，也不保证当前 8GB GPU 能构建完整 Base。
缓存绑定完整内嵌权重 ONNX 的哈希、精度策略、TensorRT 版本、预处理声明、
有序样本列表及文件哈希；任何一项改变都需要使用新缓存路径。
构建通过后再按 [运行说明](README.runtime.md) 做输出误差与真实验证集 NDS/mAP 检查。

实现使用 TensorRT 10 的 `IInt8EntropyCalibrator2` 隐式 PTQ 接口；此接口已弃用，
这不是显式 Q/DQ 路径。程序拒绝 TensorRT 11，未来迁移需单独实现 Q/DQ。

## 本次测试范围

```sh
python -m unittest discover -s tools/tensorrt -p 'test_*.py' -v
python tools/tensorrt/test_mixed_precision_gpu.py build/trt-mixed-interface-new
```

87 项 CPU 测试通过。GPU 测试使用明确标记为 `synthetic_interface_test` 的三层小 CNN，
验证 8 批校准、第二次构建复用缓存、第三次使用实测范围构建、数值执行，以及引擎中 FP16 stem/head、
INT8 backbone Conv 的真实权重/输入格式。其合成数据入口只在内部测试中显式启用，
普通构建 CLI 不开放。该测试不能用作 BEVFormer 校准数据、模型精度或速度结论。
小网络最大绝对误差约 `2.79e-4`，测试门限 `rtol=0.05, atol=0.002` 仅属于该接口测试。
从同一缓存转换的显式范围与缓存构建的小网络输出完全一致；这不代表完整 Base 的量化误差已验收。

## 共享权重与只校准 backbone

原导出模型通过 Identity 共享六路相机的 Conv 权重。TensorRT 10.13 的 parser 会把
这类间接权重作为 ITensor 输入；在本次 INT8 约束下，第二路相机的卷积无法选择格式。
GPU 上使用相同卷积、权重和实测范围，保留别名构建失败，直接引用 initializer 构建成功。
适配器现在只将确定为不可变 initializer 的 Conv 权重/bias Identity 链改成直接引用。
不改变激活输入或权重数值，也不把运行时输入、可覆盖权重或算术表达式折成常量。
Base 共修改 390 个 Conv 的 780 条引用，所有 462 个目标 INT8 Conv 均直接引用常量。

量化范围也可只在原模型的 backbone 上测量，避免统计保持 FP16 的 head。
提取器保留六路 backbone 的全部祖先节点、参数和精确 tensor 名称，输出六个末端特征图；
完整推理仍使用完整模型。子图缓存绑定子图，不可重标为完整模型缓存。

```sh
python tools/tensorrt/backbone_calibration.py \
  --onnx build/trt/base.onnx --precision-plan build/trt/mixed-plan.json \
  --output build/trt/backbone.onnx --manifest build/trt/backbone.json

NVIDIA_TF32_OVERRIDE=0 python tools/tensorrt/run_engine.py build \
  --onnx build/trt/backbone.onnx --precision mixed-int8 \
  --calibration-dir build/trt/mini-train-calibration \
  --calibration-cache build/trt/backbone.cache --engine build/trt/backbone.engine \
  --report build/trt/backbone-build.json --workspace-mib 512 --max-aux-streams 0

python tools/tensorrt/calibration_ranges.py convert \
  --cache build/trt/backbone.cache --subgraph-manifest build/trt/backbone.json \
  --output build/trt/backbone-ranges.json

NVIDIA_TF32_OVERRIDE=0 python tools/tensorrt/run_engine.py build \
  --onnx build/trt/base.onnx --precision mixed-int8 \
  --calibrated-ranges build/trt/backbone-ranges.json \
  --engine build/trt/base-ranges.engine --report build/trt/base-ranges-build.json \
  --workspace-mib 512 --max-aux-streams 0
```

范围转换仅支持已验证的 TensorRT **10.13.3.9** EntropyCalibration2 缓存格式。
它记录来源、源图/目标图哈希、样本、精度策略及每个边的名称映射；不是 TensorRT 的稳定公共缓存 ABI。
所有 INT8 Conv 的激活输入/输出都必须有有限、正的实测范围，否则构建前报错。
还会为匹配的 backbone 激活传递其余已测范围，例如残差 Add 与 ReLU 的输出；
仅覆盖 Conv 边不足以支持 TensorRT 的 Conv+Add+ReLU 融合。静态权重/常量及其别名、
shape 张量和未知 head 张量不会由这个入口填范围；全部候选验证通过后才写入网络。
无关的零值/Inf 缓存项只记录在审计清单，不分配任意默认范围。
转换后的范围独立保存，不改变原缓存的模型绑定，也不能证明完整模型检测精度。
