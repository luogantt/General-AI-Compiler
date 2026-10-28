# BEVFormer-base：六相机特征、BEV Query 与空间对应关系

整理日期：2026-10-19

范围：从“六相机图像如何与 BEV 网格 Query 建立联系”开始，整理到 CNN 空间对应关系的讨论，以及完整 BEVFormer-base 的逐阶段张量维度。

本文保留讨论中的问题与修正后的解释，不把“特征还有二维索引”误写为“特征必然精确代表原位置的物体”。尺寸表依据本地官方 Base 配置、模型实现及当前训练配置逐层核对；本次文档整理没有额外启动一次模型前向来采集 Hook 日志。

## 1. 先分清三种不同的东西

| 名称 | 是什么 | BEVFormer-base 中的规模 |
| --- | --- | --- |
| 相机图像特征 | Backbone 和 FPN 从各相机图像中提取的多尺度特征图 | 六相机，每台相机四个尺度，每个位置 256 维 |
| BEV 网格 Query | 与地面网格位置绑定的可学习特征，负责收集该区域的信息 | 200×200＝40,000 个，每个 256 维 |
| 检测 Query / Object Query | 从 BEV 特征中读取信息、预测目标的查询向量 | 900 个，每个内容向量 256 维 |

**40,000 个 BEV Query 与 900 个检测 Query 是两组不同的 Query。**

模型的主线是：

```text
六路图像
   ↓ 六台相机共享 Backbone/FPN 权重，各自提取特征
六路、多尺度图像特征
   ↓ Spatial Cross-Attention 按几何参考位置采样并聚合
200×200 个 BEV 网格特征
   ↓ 900 个检测 Query 读取 BEV
目标类别、3D 框、速度
```

在这条主线之外，Temporal Self-Attention 还会引入历史 BEV 信息。

## 2. 一个 BEV 格子对应哪个实际位置？

### 2.1 网格位置由配置确定，不是网络随意猜测

Base 配置：

```python
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_h = 200
bev_w = 200
embed_dims = 256
```

水平方向覆盖 102.4 m × 102.4 m，每个格子宽、高均为：

```text
102.4 / 200 = 0.512 m
```

设行号为 `i`、列号为 `j`，均从 0 开始。格子中心为：

```text
x = -51.2 + (j + 0.5) × 0.512
y = -51.2 + (i + 0.5) × 0.512
Query 的展开索引 p = i × 200 + j
```

例如，`i=120, j=100`：

```text
p = 24100
x = 0.256 m
y = 10.496 m
```

因此，第 24,100 个 BEV Query 的网格位置是确定的。

**坐标系说明：**官方 nuScenes 路径中的这些参考点与 `lidar2img` 配套，位于当前关键帧的 LiDAR 坐标系。不能直接称为世界坐标，也不应忽略 LiDAR 与车体坐标系之间的外参差异。使用这个坐标系不意味着模型需要输入 LiDAR 点云；本配置仍是纯相机感知。

### 2.2 Query 的位置固定，内容可以学习和更新

初始 BEV Query 来自一个可学习参数表：

```python
nn.Embedding(200 * 200, 256)
```

需要分开理解：

- **格子坐标：**由网格尺寸、覆盖范围及索引决定。
- **Query 内容：**一个 256 维可学习向量，在 Encoder 中继续更新。
- **位置编码：**另外提供行、列位置信息，不能与 Query 内容混为一谈。

Query 向量的 256 个数并不是 `(x, y, z)` 坐标本身。代码通过相同的网格索引，把 Query、位置编码及参考点关联起来。

### 2.3 一个格子为什么需要多个高度参考点？

BEV 是俯视网格，但真实目标有高度。只在一个高度上投影，可能无法充分覆盖目标的图像证据。

Base 为每个格子设置 4 个高度参考点。当前源码使用：

```text
Z = 3 - (-5) = 8
归一化高度 = linspace(0.5, 7.5, 4) / 8
实际高度   = 归一化高度 × 8 - 5
```

得到：

```text
z ≈ [-4.5, -2.1667, 0.1667, 2.5] m
```

这四个值来自具体实现，不能直接替换成四个等分高度区间的中心。

因此，一个 BEV 格子对应的是一组共享 `(x,y)`、具有不同 `z` 的参考点。这些点提供采样锚点，不代表该格子中真实物体的高度已经被测量出来。

## 3. 三维参考点如何与六相机建立联系？

### 3.1 投影的是参考点，不是先把图像投影再送入 CNN

两条路径分别执行，之后汇合：

```text
图像路径：六相机图像 → Backbone → FPN → 六路多尺度特征图
                                                ↑
                                      在参考位置附近采样
                                                ↑
几何路径：BEV 格子 → 三维参考点 → 相机标定投影 → 二维参考坐标
```

图像通常整张输入 Backbone。没有要求先按每个 BEV 格子把图像裁成一小块，再分别运行网络。

### 3.2 相机标定把三维点投影到二维图像

对于第 `c` 台相机，使用对应的 `lidar2img[c]`。它结合了坐标变换与相机内参。

用齐次坐标表示：

```text
[a, b, d, ...]ᵀ = lidar2img[c] × [x, y, z, 1]ᵀ

u = a / d
v = b / d
```

`(u,v)` 是图像中的参考位置。实现还会按图像尺寸进行归一化，供后续多尺度特征采样使用。

需要使用与预处理一致的尺寸及坐标约定。当前 Base 的 Padding 会把高度补到 928；源码使用元信息中的 `img_shape` 归一化，不能一边用原始 900 高度归一化，一边按补齐后的特征网格采样。

### 3.3 一台相机可以服务很多个 BEV Query

相机与格子之间是多对多关系：

- 一台相机的视野可以覆盖很多 BEV 格子。
- 一个 BEV 格子可能被多台相机覆盖。
- 多个格子可以采样同一片或互相重叠的图像特征。

Backbone/FPN 对每台相机提取一次特征，之后多个 Query 共享这些特征图进行采样。

### 3.4 `bev_mask` 判断的是投影条件，不是遮挡关系

代码主要检查：

1. 投影深度为正。
2. 归一化后的横、纵坐标在有效坐标范围内。

这能排除相机后方和投影范围之外的点，但**不会直接判断某个点是否被前方车辆、建筑等遮挡**。因此不能把这个 mask 当成真实可见性的完整答案。

## 4. 图像经过 CNN，特征为什么还能按位置读取？

这是讨论中最重要、也最容易混淆的部分。

### 4.1 特征图仍然有空间维度

图像张量通常表示为：

```text
[颜色通道数, 高度, 宽度]
```

CNN 的中间特征图通常表示为：

```text
[特征通道数, 特征高度, 特征宽度]
```

特征的高、宽可能变小，通道数可能变大，但每个输出位置仍有行、列索引。BEVFormer 的图像分支没有在这里通过全局池化，把整张图压成唯一一个向量。

### 4.2 卷积是滑动窗口处理，不需要先裁成独立图像

用一维形式表示局部卷积：

```text
y[i] = activation(Σₖ w[k] × x[i+k] + bias)
```

每个输出从一个滑动窗口中获取输入；相邻输出的窗口通常重叠。多层卷积扩大信息可能依赖的区域，这个区域称为结构感受野。

这说明卷积具有受约束的空间连接结构，但**不能单凭这一点证明输出语义与输入位置严格对应**。

### 4.3 对“空间对应关系”的质疑成立在哪一层？

必须区分三个命题：

| 命题 | 能否保证 |
| --- | --- |
| 输出特征有确定的行、列索引 | 能，由张量结构确定 |
| 普通卷积某个输出可能依赖哪些输入位置 | 可以根据卷积核、步长、Padding 等逐层推导 |
| 某个输出一定准确描述其名义坐标处的物体 | 不能仅由网络结构保证，需要学习和验证 |

一个反例：步幅为 1、同分辨率、忽略边界时，卷积核可以只读取右邻居：

```text
第一层：y[i] = x[i+1]
第二层：z[i] = y[i+1] = x[i+2]
```

输出位置 `i` 的内容来自输入位置 `i+2`。从响应的位置看，原有响应向左移动了两格。

这种操作仍是局部卷积，也可以满足平移等变性。因此，“局部连接”或“平移等变”都不能推出“特征响应与原位置完全不变”。

**前面解释需要修正的地方，就是不能从“保留二维索引”直接跳到“该索引上的特征必然代表原图同一位置”。**

但这也不意味着 CNN 特征完全无法用于定位。它保留了可利用的空间结构；网络是否学到了有效的空间语义，需要由任务训练和检测结果验证。

### 4.4 DCNv2 让输入依赖更灵活，但不提供精确对应保证

普通卷积的采样网格由层参数决定。DCNv2 还会预测采样偏移和调制系数，使实际采样位置依赖输入。

它仍输出一个规则排列的特征张量，但该位置的内容可能来自经过偏移的区域。DCNv2 原论文也讨论了特征支持范围超出目标、受到无关背景影响的问题。

所以不能把“输出格点的位置”“实际采样的位置”“主要表达的物体位置”当成完全相同的概念。

### 4.5 特征尺寸变化不等于特征值的线性变化

- 单独的固定权重普通卷积，忽略偏置时是线性运算；带偏置时是仿射运算。
- 整个 Backbone 包含 ReLU 等非线性，整体通常不是线性变换。
- 下采样可能丢失信息，不能认为一定可以逆向恢复输入图像。
- 高、宽按步幅变化，只说明网格尺寸关系，不证明信息或语义保持不变。

## 5. BEVFormer 如何利用这种“不保证逐点对应”的特征？

### 5.1 几何参考点提供寻找信息的位置依据

BEVFormer 使用一个结构性先验：对某个三维区域有用的图像信息，通常可以从它的投影参考位置及相关区域的特征中提取。

这是一种用于组织学习和采样的设计，不是“每个投影点上的特征必定属于那个三维点”的数学证明。

### 5.2 Query 预测偏移和权重

空间交叉注意力中的核心关系为：

```python
sampling_offsets = offset_layer(query)
attention_weights = weight_layer(query).softmax(...)
sampling_locations = reference_points + normalized_sampling_offsets
```

其中偏移按特征图的宽、高进行归一化。模型在多个尺度、多个采样位置读取特征，并进行加权聚合。

这些偏移是学习如何获取有效信息的机制，**不是先精确测出 CNN 把某个像素搬到了哪里，再把它搬回来**。也不能保证恢复此前已经丢失的信息。

### 5.3 相机内加权与相机间聚合要分开看

在每台有效相机内部，多尺度可变形注意力根据学习到的采样位置和权重聚合特征。

随后，官方 Spatial Cross-Attention 将相机结果写回各自对应的 BEV Query 索引，并按该 Query 的有效相机数量求平均，经过输出投影和残差连接。

不能把它笼统描述为“六台相机各自有一个学习到的权重，然后统一加权求和”；当前这部分相机间聚合的代码是有效相机结果的平均。

### 5.4 Query 与网格的联系通过索引持续保持

下面是省略多头、尺度、batch 等细节后的逻辑示意，不是可直接运行的完整源码：

```python
for p in bev_grid_indices:
    q = bev_queries[p]
    reference_points = grid_reference_points[p]
    camera_results = []

    for camera in cameras_valid_for(p):
        reference_uv = project(reference_points, camera.lidar2img)
        offsets, weights = predict_sampling(q)
        result = sample_multiscale_features(
            camera.features, reference_uv, offsets, weights
        )
        camera_results.append(result)

    # 源码使用 slots 累加并除以有效相机数；无有效相机时避免除零。
    aggregate = average_valid_results_or_zero(camera_results)
    updated_queries[p] = q + output_projection(aggregate)
```

注意两点：

1. Spatial Cross-Attention 使用的 Query 可能已经经过当前层的 Temporal Self-Attention 等处理，不总是最初的参数表向量。
2. 聚合结果仍写回索引 `p`，所以它继续属于同一个 BEV 网格位置。

### 5.5 对应关系靠什么改善？

最终检测误差通过可微路径反向传播，调整可训练的图像特征提取、采样偏移、权重与后续预测模块。冻结的层不随这次训练更新。

训练使这些模块共同学习“哪些空间特征对正确检测有用”，但不能保证每个 Query 都能正确定位，也不能保证所有特征都保持严格的像素语义对应。需要通过验证集的检测精度、定位误差等结果检验。

## 6. BEVFormer-base 完整主线的实际特征维度

### 6.1 适用配置与记号

本文使用当前完整 Base 配置：

| 项目 | 数值 |
| --- | --- |
| 每卡 batch | 1 |
| 相机数 | 6 |
| 原始图像，高×宽 | 900×1600 |
| Padding 后，高×宽 | 928×1600 |
| Backbone | ResNet-101，Caffe 风格，后两个 Stage 使用 DCNv2 |
| FPN | 4 个尺度，每个 256 通道 |
| BEV | 200×200 |
| Encoder / Decoder | 6 层 / 6 层 |
| FFN 隐藏维度 | 512 |
| 检测 Query | 900 |
| 类别数 | 10 |

以下描述当前一帧的主干数据流。多卡数据并行不会改变每卡表中的尺寸。FP16、激活重计算也不会改变这些张量的逻辑形状。

### 6.2 输入与 Backbone

| 阶段 | 实际张量形状 | 说明 |
| --- | --- | --- |
| 六相机图像按模型布局组织 | `[1,6,3,900,1600]` | batch、相机、颜色通道、高、宽 |
| Normalize | `[1,6,3,900,1600]` | 数值改变，尺寸不变 |
| Pad 到 32 的倍数 | `[1,6,3,928,1600]` | 高度补齐到 928 |
| 合并 batch 与相机维 | `[6,3,928,1600]` | 实际送入 Backbone |
| Stem：7×7 Conv，stride=2；BN、ReLU | `[6,64,464,800]` | 64 通道 |
| MaxPool，stride=2 | `[6,64,232,400]` | 空间再次下采样 |
| layer1 / C2：3 个 Bottleneck | `[6,256,232,400]` | 普通卷积 |
| layer2 / C3：4 个 Bottleneck | `[6,512,116,200]` | 普通卷积 |
| layer3 / C4：23 个 Bottleneck | `[6,1024,58,100]` | 使用 DCNv2 |
| layer4 / C5：3 个 Bottleneck | `[6,2048,29,50]` | 使用 DCNv2 |

图像加载阶段的数据容器可以是图像列表及 HWC 数组；表中入口采用整理成模型张量后的布局，方便统一追踪维度。

配置 `out_indices=(1,2,3)` 选择的是 **C3、C4、C5** 三个 Backbone 输出。C2 不直接作为这个 FPN 的输入。

普通卷积的输出尺寸按实际参数计算，例如：

```text
Hout = floor((Hin + 2×padding - dilation×(kernel-1) - 1) / stride + 1)
```

这也解释了为什么不能对所有层都简单进行整数除法。

### 6.3 FPN 输出四个尺度

FPN 融合 C3、C4、C5，统一通道数，并从输出增加一个低分辨率尺度。恢复相机维后：

| 输出 | 实际形状 `[batch,相机,通道,高,宽]` | 每台相机的位置数 |
| --- | --- | ---: |
| `mlvl_feats[0]` / P3 | `[1,6,256,116,200]` | 23,200 |
| `mlvl_feats[1]` / P4 | `[1,6,256,58,100]` | 5,800 |
| `mlvl_feats[2]` / P5 | `[1,6,256,29,50]` | 1,450 |
| `mlvl_feats[3]` / P6 | `[1,6,256,15,25]` | 375 |

P6 使用额外的 3×3、stride=2、padding=1 卷积，因此高度从 29 变成 **15**。

例如：

```python
v = mlvl_feats[0][0, 2, :, 50, 80]
# v.shape == [256]
```

这是第 3 台相机、第一个尺度、第 50 行、第 80 列的 256 维特征向量。到这里六台相机仍分别保留，尚未融合成 BEV。

### 6.4 展开与拼接图像特征

每台相机的四个尺度共：

```text
116×200 + 58×100 + 29×50 + 15×25 = 30,825 个位置
```

具体形状变化：

```text
每个尺度：[1,6,256,H,W]
                   ↓ flatten 与 permute
          [6,1,H×W,256]
                   ↓ 沿空间序列拼接四个尺度
          [6,1,30825,256]
                   ↓ 再调整维度顺序
          [6,30825,1,256]
           相机 位置 batch 通道
```

同时保留：

```python
spatial_shapes = [[116, 200], [58, 100], [29, 50], [15, 25]]
level_start_index = [0, 23200, 29000, 30450]
```

有了尺寸、起始索引及确定的展开顺序，可以恢复每个位置所属的尺度、行、列。**Flatten 本身不是全局池化，不会因为变成序列就自动丢掉坐标索引。**

六台相机合计有 184,950 个尺度内位置，但代码在这里仍保留独立相机维。

### 6.5 BEV Encoder

| 阶段 | 实际形状 | 说明 |
| --- | --- | --- |
| BEV Query 参数表 | `[40000,256]` | 40,000 个网格 |
| BEV 位置编码 | `[1,256,200,200]` | 行、列编码组合 |
| 进入 Encoder 层内部 | `[1,40000,256]` | batch、网格、通道 |
| Temporal Self-Attention 输出 | `[1,40000,256]` | 融合时序信息 |
| Spatial Cross-Attention 输出 | `[1,40000,256]` | 融合六相机信息 |
| FFN 隐藏层 | `[1,40000,512]` | 临时扩展通道 |
| FFN 输出、LayerNorm 输出 | `[1,40000,256]` | 回到 256 通道 |
| 全部 6 层 Encoder 执行完 | `[1,40000,256]` | 每层主线尺寸均相同 |

在网格意义上，可以将最后的 BEV 表示为 `[1,200,200,256]`；进入 Decoder 时，代码还会调整为 `[40000,1,256]`。

**相机特征的 116×200、58×100 等网格，与 BEV 的 200×200 网格属于不同坐标域。200×200 不是把某张相机特征图直接 Resize 得到的。**

参考点相关张量也可以确定：

| 张量 | 形状 |
| --- | --- |
| 四个高度的三维参考点 | `[1,4,40000,3]` |
| 六相机投影后的二维参考点 | `[6,1,40000,4,2]` |
| 投影有效性 mask | `[6,1,40000,4]` |

每台相机筛选出来的有效 Query 数量与标定和投影结果有关，不能把所有内部临时张量都写成一个与输入无关的固定长度。

### 6.6 检测 Decoder

| 阶段 | 实际形状 | 说明 |
| --- | --- | --- |
| 检测 Query 参数表 | `[900,512]` | 内容与位置各占 256 维 |
| 拆分后的内容、位置向量 | 各 `[900,256]` | 不同于 BEV Query 参数表 |
| Decoder 内部 Query | `[900,1,256]` | 目标 Query、batch、通道 |
| Decoder 读取的 BEV | `[40000,1,256]` | 提供场景信息 |
| 每层 Decoder 输出 | `[900,1,256]` | 六层均保持主线形状 |
| FFN 隐藏层，按该布局表示 | `[900,1,512]` | 输出再回到 256 |
| 收集全部 6 层输出 | `[6,900,1,256]` | 第一维是层数 |
| 检测头调整顺序后 | `[6,1,900,256]` | 层数、batch、目标、通道 |

**此处第一个 6 是 Decoder 层数，不是相机数。**

### 6.7 检测头与最终结果

| 输出 | 实际形状 | 含义 |
| --- | --- | --- |
| `all_cls_scores` | `[6,1,900,10]` | 六层预测、10 类分类分数 |
| `all_bbox_preds` | `[6,1,900,10]` | 每个候选的 10 个框编码数值 |
| 最后一层分类分数 | `[1,900,10]` | 最终解码使用 |
| 最后一层框编码 | `[1,900,10]` | 最终解码使用 |
| 最终 3D 框 | `[K,9]`，`K≤300` | 中心、尺寸、角度、速度 |
| 最终分数与类别 | 各 `[K]` | 一条对应一个输出框 |

框的 10 维编码按源码排列为：

```text
[cx, cy, log(w), log(l), cz, log(h), sin(yaw), cos(yaw), vx, vy]
```

解码时，尺寸取指数，`sin/cos` 恢复成一个角度，得到 9 维框表示。最终 box 容器还遵循其坐标与中心约定。

类别数 10 与框编码维度 10 只是数值相同，含义不同。最终采用最后一层结果进行 Top-K 和范围过滤，最多保留 300 个框，不是固定输出 900 个框。

### 6.8 训练的四帧队列

当前训练 `queue_length=4`，输入额外带有时间维：

```text
[1,4,6,3,928,1600]
 batch 帧 相机 通道 高 宽
```

当前帧与历史帧有不同执行路径，历史帧用于产生历史 BEV。这个时间维不能理解为 24 台相机，也不意味着检测 Decoder 每次直接接收四帧所有图像特征。

## 7. 这段讨论中应当保留的准确表述

| 容易混淆的说法 | 更准确的表述 |
| --- | --- |
| 一台相机对应一个 BEV Query | 一台相机可服务很多 Query，一个 Query 也可关联多台相机 |
| 先把相机图像投影到 BEV，再送入 Backbone | 图像走 Backbone/FPN；BEV 三维参考点另行投影，决定图像特征采样参考位置 |
| CNN 后特征不再有位置 | 特征仍可有二维索引和受结构约束的输入依赖 |
| 有二维索引，就保证精确表达同位置的物体 | 索引和语义对应是两回事，后者没有这样的结构性保证 |
| 结构感受野证明特征主要描述哪个物体 | 它描述可能依赖的输入范围，不保证主要语义贡献 |
| 下采样 8 倍，每个输出只代表独立的 8×8 图块 | 网格间距与感受野不同，感受野可能更大且重叠 |
| 可学习偏移一定能纠正所有错位 | 偏移帮助适应采样位置，不保证纠正全部误差或恢复丢失信息 |
| 特征展平成序列，就丢失所有空间关系 | 保留尺寸、顺序和尺度起始索引后仍能恢复空间索引 |
| 六相机直接拼成一个大特征向量给每个 Query | 六路多尺度特征仍分别组织，通过投影和可变形注意力选择性读取 |
| BEV 200×200 是图像缩小后的尺寸 | 它是独立设定的地面网格，来自几何引导的特征聚合 |
| 40,000 个 Query 就是 40,000 个检测框 | 它们是 BEV 网格特征；900 个检测 Query 才用于候选目标预测 |

## 8. 对应代码与原始资料

以下路径相对于本地 BEVFormer 源码目录：

```text
C:/Users/Thomas/Desktop/project/BEVFormer-master/
```

| 内容 | 源码文件 | 本次核查入口 |
| --- | --- | --- |
| Base 网络与输入配置 | `projects/configs/bevformer/bevformer_base.py` | 第 11、31、43、54、63、78、106、166 行附近 |
| 六相机 Backbone/FPN 特征提取 | `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` | `extract_img_feat()`，第 100 行附近 |
| 图像 Padding | `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` | `PadMultiViewImage` |
| 相机标定矩阵组织 | `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py` | 第 129～148 行附近 |
| 参考点生成与投影 | `projects/mmdet3d_plugin/bevformer/modules/encoder.py` | `get_reference_points()`、`point_sampling()` |
| 图像特征展开与拼接 | `projects/mmdet3d_plugin/bevformer/modules/transformer.py` | 第 164～194 行附近 |
| 采样偏移、权重、相机间聚合 | `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` | `SpatialCrossAttention`、`MSDeformableAttention3D` |
| BEV 与检测 Query 参数表、预测头 | `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py` | `_init_layers()`、`forward()` |
| 六层 Decoder 输出收集 | `projects/mmdet3d_plugin/bevformer/modules/decoder.py` | `DetectionTransformerDecoder.forward()` |
| 框筛选与反编码 | `projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`；`projects/mmdet3d_plugin/core/bbox/util.py` | `decode_single()`、`denormalize_bbox()` |

本项目的训练配置快照：`docs/training/bevformer-base/base-trainval-config.py`。源码可能带有本地注释，因此上述行号仅作为本次快照的定位提示，以函数名为准。个别上游旧 docstring 与实际实现不一致，本文尺寸以配置和执行代码为准。

原始资料：

- [BEVFormer 论文](https://arxiv.org/abs/2203.17270)：BEV Query、空间交叉注意力与时序融合的设计。
- [官方 Base 配置](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py)：模型尺寸与模块参数。
- [官方参考点与投影实现](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/encoder.py)：网格参考点、相机投影及 mask。
- [官方空间交叉注意力实现](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py)：偏移、权重、特征采样与相机聚合。
- [MMDetection 2.14 ResNet](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py) 与 [FPN](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/necks/fpn.py)：对应版本的 Backbone 与多尺度特征实现。
- [DCNv2 论文](https://arxiv.org/abs/1811.11168)：可变形采样、调制机制及特征空间支持范围的讨论。

以上 GitHub `master` 链接用于阅读上游源码，后续可能变化；本次尺寸核对基于本地配置与源码快照。
