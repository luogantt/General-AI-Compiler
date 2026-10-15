# Graph IR 网络图与矩阵查看器

把前端已提取的 Graph IR 显示成可逐层展开的网络图：模块之间有真实数据连线，点击算子可查看输入、输出、属性、上下游节点以及权重 / bias 矩阵。

## 打开实际结果

本地 BEVFormer 页面：`build/bevformer-onnx-inspect/inspection/graph-network.html`。

这个文件已嵌入实际图和矩阵采样数据，直接打开即可使用，无需服务或联网。`tools/templates/graph-ir-network.html` 是生成页面用的模板，不是实际模型图。

## 如何查看

1. 从“模型总览”开始，点击 `img_backbone`、`img_neck` 或 `pts_bbox_head` 展开到下一层。模块来自原始节点名称的 scope，包含的算子没有被删除。
2. 点击具体算子，右侧显示真实输入、输出、shape、属性和来源；点击“看上一层 / 下一层”查看张量依赖。分支、Residual 和多消费者关系使用实际连线表示，不假定所有算子串行排列。
3. 搜索 `Conv2D`、`MatMul`、`LayerNorm` 或完整节点名称，可直接定位算子。层内条目较多时使用“前一组 / 后一组”。
4. 在“权重 / bias / 常量矩阵”选择张量。二维数组显示行列网格；卷积等高维数组可选择空间切片或通道展平。空间切片固定前置维度；通道视图显示第 0 维 × 其余维度展平，`1×1` 卷积默认采用这一视图。切片索引、行列原始索引均可见。
5. 点击“放大矩阵”查看数值与颜色。颜色只按当前显示数值缩放；网格中的数值来自真实常量，不是用 shape 随机生成的示例。

大矩阵每个维度最多显示 16 个均匀选取的原始索引，高维数组最多显示 8 个代表性空间切片。没有显示的行列不补零；完整权重仍在 `constants.npz`。全数组有限值 min/max 与当前切片颜色范围是不同的统计量。超出 JavaScript 安全整数范围的整型常量按精确字符串显示。

若权重或 bias 经过 Identity 别名节点，查看器沿别名找到原始常量，并标出数组来源；原图中的 Identity 和连接仍然保留。它不会把任意常量表达式或运行时激活伪装成已计算的矩阵。

默认缩放保证节点标题可读；拖动画布可查看后续层级。“适应画布”可缩小查看整个模块，再放大查看局部。

中间激活在 Graph IR 中只有名称、类型和连线，没有实际执行值，因此页面不会伪造激活矩阵。橙色节点表示保留的源 ONNX 算子，尚未完成规范化；图能够显示不代表后端已经可以执行。

## 数据来源与生成

`graph.ir.txt`、`canonical.graph.json` 是同一 Graph 的两种表示。生成器读取结构化 JSON 和 NPZ，而不是从可读 TXT 中猜测运算或数字；页面记录源文件 SHA-256，保留原始 IR 下载链接。

```powershell
python tools/render_graph_ir.py `
  --input build/bevformer-onnx-inspect/inspection `
  --output build/bevformer-onnx-inspect/inspection/graph-network.html
```

代码分为三部分：

| 文件 | 职责 |
| --- | --- |
| `tools/render_graph_ir.py` | 校验 Graph + NPZ、提取真实矩阵切片、写入独立 HTML |
| `tools/templates/graph-ir-layout.js` | 构建层级、根据真实张量建立边、布局、模块分页与上下游查询 |
| `tools/templates/graph-ir-network.html` | 网络图、缩放与拖动、节点详情、矩阵切片及放大查看 |

模块聚合后若出现双向交叉引用，布局单独标注相关边；这不代表原始算子 DAG 存在循环。权重别名等根层辅助算子保留在辅助目录，直接参与数据路径的根层算子保留独立显示。
