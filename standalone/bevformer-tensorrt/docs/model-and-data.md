# 模型与数据契约

本项目从**已经导出的固定形状 ONNX**开始，完成 TensorRT 适配、构建和执行。它不包含 PyTorch checkpoint → ONNX 导出器。配套模型包已经完成适配，因此拿到源码包和模型包后，可以直接 `build`。

## 获取与交付

| 文件或数据 | 来源与用途 |
|---|---|
| 配套 `adapted.onnx` | 从现有 BEVFormer-base checkpoint 导出并适配，内含权重；来源以配套 manifest/export report 为准 |
| `images.npy`、`reference.npz` | 同一次固定输入测试的合成输入与 PyTorch 输出，用于数值回归，不用于校准 |
| `ranges.json` | 既有真实图像 backbone 校准范围，与特定 ONNX 和精度规则绑定 |
| 原始 BEVFormer 代码/checkpoint | [官方仓库](https://github.com/fundamentalvision/BEVFormer)，用于另行训练或重新导出；官方 checkpoint 不能直接传给本项目 `build` |
| nuScenes mini | 从 [nuScenes 官网](https://www.nuscenes.org/nuscenes)按数据许可取得；不随源码包分发，不需要为一次已有范围的推理下载全量数据 |

上传独立源码仓库时，不提交 engine、大模型、原始数据或登录信息。需要复现固定模型时另外分发资产包，并保留其中的模型来源和许可说明。模型包解压后，先验证：

```bash
python scripts/verify_assets.py
```

该命令按 `models/bevformer-base/manifest.json` 的文件清单校验大小和 SHA-256。它验证交付文件的一致性，不证明检测精度。

## 固定模型范围

| 项目 | 契约 |
|---|---|
| 模型 | BEVFormer-base，ResNet-101＋DCNv2 |
| batch | 1 |
| 相机 | 6 路 |
| 图像张量 | `float32 [1,6,3,928,1600]` |
| BEV | 200×200、256 通道 |
| 时序 | cold：无历史 BEV |
| 几何 | 导出时固定的合成可见性/投影元数据 |
| CAN-bus | 固定零值 |
| 输出范围 | 原始 BEV、六层分类 logits、六层框编码 |
| 未包括 | 真实时序状态更新、真实相机几何输入、框解码、检测指标评估 |

更换六相机像素不会自动更新图内相机标定、姿态或历史状态。这是目前固定导出模型的边界，而非 TensorRT 本身不能接收这些输入。

## 输入文件

单输入模型可以传 `.npy`，也可传只有 `images` 键的 `.npz`：

```python
import numpy as np

images = np.load("models/bevformer-base/images.npy")
assert images.dtype == np.float32
assert images.shape == (1, 6, 3, 928, 1600)
```

真实图像预处理：

1. 相机顺序固定为 `CAM_FRONT`、`CAM_FRONT_RIGHT`、`CAM_FRONT_LEFT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`。
2. 每张原图为 900×1600，使用 BGR，不转 RGB、不缩放。
3. BGR 减均值 `[103.53,116.28,123.675]`，标准差为 1。
4. **归一化后**在底部补零，使高度从 900 变成 928。
5. 将 HWC 转成 CHW，按相机堆叠并增加 batch 维。

`prepare-calibration` 根据 nuScenes JSON 表匹配同一 sample 的六个关键帧图像，并记录来源、哈希和预处理契约。训练/校准数据与验证数据应分开。

```text
data/nuscenes/
├── v1.0-mini/
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── calibrated_sensor.json
│   └── sensor.json
└── samples/
    ├── CAM_FRONT/
    └── ...其余五路相机...
```

本项目只预处理六相机图像，不需要为此加载 lidarseg、nuImages 或地图扩展。

## 输出含义

`outputs.npz` 包含：

| 键 | 形状 | 说明 |
|---|---|---|
| `bev_embed` | `[40000,1,256]` | 展平的 BEV 网格特征 |
| `all_cls_scores` | `[6,1,900,10]` | 六层 Decoder 的 900 个候选、10 类 logits；尚未 sigmoid/Top-K |
| `all_bbox_preds` | `[6,1,900,10]` | `(cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy)` |

在本模型的输出契约中，中心坐标已经转换到物理坐标范围；尺寸仍是对数编码，朝向仍由 sin/cos 表达。最终框列表还需要选择最后层、分类分数处理、Top-K、范围过滤、尺寸指数还原、`atan2` 等处理。本项目目前保存原始输出，不悄悄加入另一套后处理。

## 适配自己的 ONNX

```bash
python -m bevformer_trt adapt \
  models/original.onnx models/custom-adapted.onnx \
  --report results/custom-adaptation.json
```

适配器针对既有导出器的节点 ABI，不承诺任意第三方 BEVFormer ONNX 都兼容：

| 原节点/问题 | 适配动作 |
|---|---|
| `gac::MSDeformAttn` | 改为 NVIDIA `MultiscaleDeformableAttnPlugin_TRT` v2；补 INT32 shape/index、level offset 和必要 reshape |
| MMCV DCNv2 | 改为 `ModulatedDeformConv2d` v2；按插件 field 契约转换属性 |
| 普通 Conv 的常量权重/bias 经 Identity 引用 | 仅把可证明不可变的 initializer 别名改为直接引用，保留权重数值 |

插件通过 `trt.init_libnvinfer_plugins` 注册。源码包不内嵌第三方 CUDA 插件源文件。适配器会拒绝不支持的契约，例如尚未验证的非对称 DCNv2 stride/padding/dilation；不会只换节点名后宣称正确。

## 数值验证

`run --reference` 要求参考文件 key、shape 与 engine 输出一致，同一输入才能比较。浮点逐元素条件是：

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

默认 `rtol=1e-3`、`atol=1e-4`，整数要求相等。NaN/Inf 或误差不通过时返回非零退出码。未传参考只检查有限性，并写入 `not_checked`。不能根据输入/输出形状正确、程序成功退出或一帧误差小，推出整个检测任务的 mAP/NDS 合格。
