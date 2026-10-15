# BEVFormer：从真实 PyTorch 模型到 ONNX 前端中间态

这次导出使用本机 `BEVFormer-master` 的原始 PyTorch 模块和真实 checkpoint，包含 ResNet50、FPN、三层 BEV encoder、六层 decoder 和检测头。不是把已有 GAC Graph 再写回 ONNX。

## 先看什么

本地生成目录为 `build/bevformer-onnx-inspect/`：

| 文件 | 作用 |
| --- | --- |
| `bevformer-tiny-cold.onnx` | 导出的模型图和权重，ONNX opset 17 |
| `inspection/index.html` | 点击算子，查看源 ONNX、前端节点、输入输出类型和权重摘要 |
| `inspection/graph.ir.txt` | 适合直接阅读的 GAC Graph IR 文本 |
| `inspection/canonical.graph.json` | 可回读的 Graph 结构，不把大权重展开成 JSON |
| `inspection/constants.npz` | 前端提取的真实常量数组 |
| `inspection/import-report.json` | 节点统计、未转换节点及严格导入结果 |
| `export-report.json` / `onnx-correctness.json` | 模型来源、导出边界及 PyTorch/ONNX 数值检查 |

HTML、JSON 和 IR 文本中的张量名称相互对应。浏览页面时可以搜索 `Conv`、`MatMul`、`GridSample`、`LayerNorm`，再点击输入或输出张量追踪来源。

## 中间态到底存了什么

`load_onnx()` 返回 Python `Graph` 对象：

```python
graph.inputs       # 输入名称 → TensorSpec(shape, dtype, layout)
graph.constants    # 权重/常量名称 → NumPy ndarray，真实数值在这里
graph.nodes        # 拓扑顺序的 Node；inputs/outputs 通过名称引用张量
graph.value_specs  # 中间张量名称 → TensorSpec，只含类型和形状
graph.outputs      # 模型输出名称
graph.metadata     # 源模型信息、算子转换统计及未支持原因
```

`graph.ir.txt` 用 `%张量名 = 算子(%输入名)` 的形式展示这些对象，**不是 MLIR 语法**。此次任务停在 ONNX 前端 Graph IR；尚未调用 `mlir-opt`，也没有编译或执行 Vortex 机器码。

例如本次真实提取出的第一个卷积节点是：

```python
Node(
    id="/img_backbone/conv1/Conv", op="Conv2D",
    inputs=("/Reshape_output_0", "onnx::Conv_12489", "onnx::Conv_12490"),
    outputs=("/img_backbone/conv1/Conv_output_0",),
    attrs={"strides": [2, 2], "dilations": [1, 1],
           "pads": [3, 3, 3, 3], "groups": 1},
)
```

这里三个输入依次引用图像、卷积权重和 bias。权重数值在 `graph.constants["onnx::Conv_12489"]`，而不在 Node 字段内；卷积输出的 `TensorSpec` 为 `float32[6,64,240,400]`。导入阶段只建立这条运算关系，尚未实际计算这块输出矩阵。

检查模式 `strict=False` 将暂时不能合法化的节点保留为 `ai.onnx::算子名`，记录原始边、属性和原因。它只用于查看及支持分析，不能据此断言整个模型已经能够编译。`strict=True` 仍会拒绝没有忠实转换的算子；非法 ONNX 和不支持的嵌套控制流不会被检查模式吞掉。

## 导出边界

- 输入为归一化/填充后的图像张量 `images[1,6,3,480,800]`，FP32。
- 冷帧 `prev_bev=None`，固定相机标定与 CAN-bus；不包含跨帧状态输入。
- 默认标定是用于查看完整计算路径的合成固定投影，图像是固定随机种子生成的输入；没有使用 nuScenes 样本，因此不代表检测精度验证。可用 `--metadata` 提供外部固定标定。
- 输出为 `bev_embed[2500,1,256]`、`all_cls_scores[6,1,900,10]`、`all_bbox_preds[6,1,900,10]`。最终 box 解码、阈值筛选和数据预处理在图外。
- CPU 路径使用原始 MMCV PyTorch deformable-attention 实现，导出为标准 ONNX 算子，包括 `GridSample`。缺少的 MMCV 二进制函数仅在导入时提供报错占位；如果实际执行到这些函数，会立刻失败，不会伪造结果。
- 旧版 PyTorch 的 ONNX tracer 不接受算术中的 Python bool。导出封装把原模型 `use_can_bus/use_shift` 开关转换成整数 0/1，并逐元素核对适配前后的三个真实输出。
- 旧导出器还会忽略 `linspace` 的 dtype，错误产生 FP64 参考点。封装在原始参考点函数返回处显式保留所请求的 dtype；适配前后 eager 输出完全一致，导出后 12 个 GridSample 的输入和采样网格均为 FP32。

## 本次实际结果（2026-10-15）

ONNX 完整格式/类型检查通过，模型为 200,663,543 bytes，9,129 个源节点、454 个 initializer。ONNX SHA-256 为 `5a36e46d80541d7f1d3c2d897c9db021da9f729d49b0c7da85ffc989571bb070`。

ONNX 前端检查模式提取出 5,511 个节点、4,088 个常量数组（200,403,776 bytes）。常量和结构外置保存后重新载入验证通过。节点减少主要因为 Constant 被移入常量表；这不是算子融合后的性能优化结果。

其中 2,996 个节点尚未转换成 GAC 规范算子，保留原始 ONNX 名称及信息；包括 Shape、Gather、ScatterND、NonZero、部分动态 shape 输入的 Reshape/Expand 等。严格导入仍在位置编码的 Expand 处拒绝缺少编译期 shape 常量。8 个中间输出的类型尚未推导完整，页面明确显示未知。**本次完成的是模型导出与前端中间态检查，不是这张 ONNX 图的全模型 Vortex 编译。**

ONNX Runtime CPU 与原始 PyTorch 对同一输入进行比较，三个输出均为有限数，`rtol=1e-3, atol=1e-4` 的逐元素检查通过：

| 输出 | 最大绝对误差 |
| --- | ---: |
| `bev_embed` | `3.278e-6` |
| `all_cls_scores` | `1.049e-5` |
| `all_bbox_preds` | `5.341e-5` |

可提交的小型证据见 `validation/onnx/bevformer-tiny-cold-2026-10-15/summary.json`；完整输出数组、模型和可交互页面保留在本地 `build/`。

## 复现

导出环境与 GAC 导入环境分开：本次原始模型使用 Python 3.10、PyTorch 2.0.1 CPU、torchvision 0.15.2、MMCV 1.4.0、MMDetection 2.14.0、NumPy 1.26.4；GAC 前端使用 Python 3.12。ONNX 为 1.21.0，数值验证使用 ONNX Runtime 1.23.2 CPU。

```powershell
# 在兼容原始 BEVFormer 依赖的环境中；checkpoint 必须来自可信来源。
python tools/export_bevformer_onnx.py `
  --source-root C:/Users/Thomas/Desktop/project/BEVFormer-master `
  --checkpoint build/checkpoints/bevformer_tiny_epoch_24.pth `
  --trusted-checkpoint --output-dir build/bevformer-onnx-inspect

python tools/validate_bevformer_onnx.py build/bevformer-onnx-inspect

# 切换到安装了 GAC 和 ONNX 的 Python >= 3.11 环境。
python tools/inspect_onnx_ir.py `
  --model build/bevformer-onnx-inspect/bevformer-tiny-cold.onnx `
  --output build/bevformer-onnx-inspect/inspection `
  --export-report build/bevformer-onnx-inspect/export-report.json
```

大模型及权重保存在被 Git 忽略的 `build/` 中；脚本和本文进入源码管理。
