# BEVFormer Encoder：从相机特征到 BEV Query 的完整数学推导

本文只回答一个核心问题：

> **六相机的图像特征，究竟怎样通过数学运算进入某一个 BEV Query？**

推导以官方 **BEVFormer-base** 为准。为了让逻辑闭合，本文先推导一个 BEV 格子在一层 Spatial Cross-Attention 中的完整计算，再扩展到 `40,000` 个格子、六台相机和六层 Encoder。

---

## 1. 先给出结论

BEV Query 与图像特征之间不是简单相加，也不是普通 Transformer 的全局 `QK^T V`。

完整关系是：

```text
BEV 格子的三维位置
        ↓ 相机标定矩阵
图像中的几何参考位置
        ↓ 加上 Query 预测的偏移
图像特征图中的采样位置
        ↓ 双线性采样
采样到的特征向量
        ↓ Query 预测的权重
多尺度、多采样点加权求和
        ↓ 有效相机求平均
写回原来的 BEV Query
```

因此，一个 BEV Query 同时承担两个角色：

1. **控制器**：预测去哪里采样，以及各采样结果占多大权重；
2. **状态向量**：接收相机特征，并通过残差连接不断更新。

![从六相机特征到 BEV Query](assets/bevformer-query-guide/04-feature-to-query.png)

---

## 2. 符号与真实维度

| 符号 | 含义 | BEVFormer-base 中的值或形状 |
| --- | --- | --- |
| `B` | Batch size | 推理时通常为 `1` |
| `p` | 一个 BEV 格子编号 | `0 ... 39,999` |
| `c` | 相机编号 | `0 ... 5` |
| `l` | FPN 层级 | `0 ... 3` |
| `h` | Attention Head | `0 ... 7` |
| `k` | 每层每 Head 的采样点 | `0 ... 7` |
| `m` | 柱体高度参考点 | `0 ... 3` |
| $q_p$ | 第 `p` 个 BEV Query | $\mathbb{R}^{256}$ |
| $F_{c,l}$ | 相机 `c` 的第 `l` 层图像特征 | $H_l\times W_l\times256$ |
| $\rho_{p,c,m}$ | 三维参考点在相机中的二维投影 | $\mathbb{R}^{2}$ |
| $\Delta_{p,h,l,k}$ | Query 预测的采样偏移 | $\mathbb{R}^{2}$ |
| $\alpha_{p,h,l,k}$ | Query 预测的采样权重 | 标量 |

Base 的 BEV 网格为：

```text
200 × 200 = 40,000 个格子
每个格子绑定一个 256 维 Query
```

因此，一层 Encoder 的 Query 张量为：

$$
Q\in\mathbb{R}^{B\times40000\times256}.
$$

---

## 3. 图像先变成四层空间特征

六张图像经过共享的 ResNet-101 和 FPN。以 Base 的实际输入尺寸为例，四层特征为：

| FPN 层 | 六相机特征形状 |
| --- | --- |
| P3 | `[B, 6, 256, 116, 200]` |
| P4 | `[B, 6, 256, 58, 100]` |
| P5 | `[B, 6, 256, 29, 50]` |
| P6 | `[B, 6, 256, 15, 25]` |

每台相机共有：

$$
116\times200+58\times100+29\times50+15\times25=30825
$$

个特征位置。展平并拼接后，代码中的布局为：

```text
[6 camera, 30,825 position, B, 256]
```

在拼接之前，源码还会给特征加入相机身份编码和 FPN 层级编码：

$$
\widetilde F_{c,l}
=
F^{\mathrm{FPN}}_{c,l}
+e_c^{\mathrm{cam}}
+e_l^{\mathrm{level}}.
$$

因此，即使不同相机或不同 FPN 层出现相似的特征值，后续模块仍能区分特征来自哪台相机、哪一个尺度。

这些位置不是原始 RGB 像素。每个位置保存一个 `256` 维语义向量，但四层特征仍然保留二维行列结构，因此仍然可以按照坐标采样。

---

## 4. 一个 BEV 格子产生四个三维参考点

![BEV 网格与格子中心](assets/bevformer-query-guide/01-bev-grid.png)

BEVFormer-base 的空间范围是：

$$
x,y\in[-51.2,51.2].
$$

网格大小是 `200 × 200`，所以每格边长为：

$$
\Delta x=\Delta y=\frac{102.4}{200}=0.512\ \text{m}.
$$

第 `i` 行、第 `j` 列格子的中心为：

$$
x_p=-51.2+(j+0.5)\times0.512,
$$

$$
y_p=-51.2+(i+0.5)\times0.512.
$$

![同一格子的四个高度参考点](assets/bevformer-query-guide/02-four-height-points.png)

模型在这个格子的竖直柱体上设置四个高度：

$$
z_m\in\{-4.50,-2.17,0.17,2.50\}\ \text{m}.
$$

因此，一个格子对应四个齐次三维参考点：

$$
R_{p,m}=
\begin{bmatrix}
x_p\\
y_p\\
z_m\\
1
\end{bmatrix},
\qquad m=0,1,2,3.
$$

这些点只定义“到这根柱体的哪些高度附近寻找图像证据”，并不表示模型已经知道这里存在某个物体。

---

## 5. 三维参考点投影到六台相机

![四个三维参考点投影到六台相机](assets/bevformer-query-guide/03-camera-projection.png)

第 `c` 台相机具有投影矩阵 $M_c$，其中已经包含相机外参和内参。投影为：

$$
\begin{bmatrix}
\widetilde u\\
\widetilde v\\
d\\
*
\end{bmatrix}
=M_cR_{p,m}.
$$

执行透视除法：

$$
u=\frac{\widetilde u}{d},
\qquad
v=\frac{\widetilde v}{d}.
$$

再除以图像宽高，得到 `[0,1]` 范围内的归一化坐标：

$$
\rho_{p,c,m}=
\left(
\frac{u}{W_{\mathrm{img}}},
\frac{v}{H_{\mathrm{img}}}
\right).
$$

参考点有效需要满足：

$$
d>0,
\qquad
0<\rho_x<1,
\qquad
0<\rho_y<1.
$$

最终的投影张量形状为：

```text
reference_points_cam
[6 camera, B, 40,000 query, 4 height, 2 xy]
```

到这里，几何系统只完成了**基准地址生成**。真正的采样位置还要由 Query 修正。

---

## 6. Query 预测采样偏移

设当前格子的状态向量为：

$$
q_p\in\mathbb{R}^{256}.
$$

位置编码为 $e_p$。用于产生采样参数的向量是：

$$
\widetilde q_p=q_p+e_p.
$$

偏移由一个线性层生成：

$$
\Delta_p=W_{\Delta}\widetilde q_p+b_{\Delta}.
$$

其中 PyTorch 线性层的权重形状为：

$$
W_{\Delta}\in\mathbb{R}^{512\times256}.
$$

输出数量为：

$$
512=8_{\mathrm{head}}\times4_{\mathrm{level}}\times8_{\mathrm{point}}\times2_{xy}.
$$

reshape 后：

```text
sampling_offsets.shape = [8 Head, 4 Level, 8 Point, 2]
```

注意：这里是 `512` 个输出，不是 `4096` 个输出。

---

## 7. Query 同时预测采样权重

另一个线性层产生权重 Logit：

$$
a_p=W_{\alpha}\widetilde q_p+b_{\alpha}.
$$

输出数量为：

$$
256=8_{\mathrm{head}}\times4_{\mathrm{level}}\times8_{\mathrm{point}}.
$$

reshape 后：

```text
attention_logits.shape = [8 Head, 4 Level, 8 Point]
```

每个 Head 独立对自己的 `4 × 8 = 32` 个 Logit 做 Softmax：

$$
\alpha_{p,h,l,k}
=
\frac{\exp(a_{p,h,l,k})}
{\displaystyle
\sum_{l'=0}^{3}\sum_{k'=0}^{7}
\exp(a_{p,h,l',k'})}.
$$

因此，每个 Head 都满足：

$$
\sum_{l=0}^{3}\sum_{k=0}^{7}
\alpha_{p,h,l,k}=1.
$$

---

## 8. 四个高度与八个采样点的关系

配置中同时存在：

```python
num_points_in_pillar = 4
num_points = 8
```

这不表示每个高度采八个点。源码会把八个点重新解释为：

```text
8 Point = 2 Round × 4 Height
```

所以对于每个 Head、每个 FPN Level：

```text
高度 z0：2 个采样点
高度 z1：2 个采样点
高度 z2：2 个采样点
高度 z3：2 个采样点
合计：8 个采样点
```

按照源码的 reshape 顺序，采样点 `k` 对应的高度为：

$$
m(k)=k\bmod4.
$$

即：

```text
Point 0、4 → 高度 0
Point 1、5 → 高度 1
Point 2、6 → 高度 2
Point 3、7 → 高度 3
```

---

## 9. 计算最终采样位置

第 `l` 层特征图尺寸为 $H_l\times W_l$。Query 产生的偏移要按该层尺寸归一化：

$$
\widehat\Delta_{p,h,l,k}
=
\left(
\frac{\Delta x_{p,h,l,k}}{W_l},
\frac{\Delta y_{p,h,l,k}}{H_l}
\right).
$$

最终采样位置为：

$$
s_{p,c,h,l,k}
=
\rho_{p,c,m(k)}
+
\left(
\frac{\Delta x_{p,h,l,k}}{W_l},
\frac{\Delta y_{p,h,l,k}}{H_l}
\right).
$$

这一步的逻辑是：

```text
相机投影 ρ：提供几何基准位置
Query 偏移 Δ：在基准位置附近寻找更有效的特征
最终位置 s：真正执行特征读取的位置
```

---

## 10. 图像特征先投影为八个 Head

加入相机编码和层级编码后的特征向量 $\widetilde F_{c,l}(x,y)\in\mathbb{R}^{256}$ 先经过 Value 投影：

$$
V_{c,l}(x,y)=W_v\widetilde F_{c,l}(x,y)+b_v.
$$

其中：

$$
W_v\in\mathbb{R}^{256\times256}.
$$

随后将 256 维切分为八个 Head：

$$
V_{c,l}
=
\operatorname{Concat}
\left(
V_{c,l}^{(0)},
V_{c,l}^{(1)},
\ldots,
V_{c,l}^{(7)}
\right),
$$

其中：

$$
V_{c,l}^{(h)}(x,y)\in\mathbb{R}^{32},
\qquad
256=8\times32.
$$

---

## 11. 在非整数位置执行双线性采样

上一节得到的 $s=(s_x,s_y)$ 是 `[0,1]` 范围内的归一化坐标。按照 Deformable Attention 使用的 `align_corners=False` 约定，它在第 `l` 层特征图上的连续坐标是：

$$
x=W_ls_x-\frac{1}{2},
\qquad
y=H_ls_y-\frac{1}{2}.
$$

采样位置一般不是整数。定义：

$$
x_0=\lfloor x\rfloor,
\qquad
x_1=x_0+1,
$$

$$
y_0=\lfloor y\rfloor,
\qquad
y_1=y_0+1,
$$

$$
d_x=x-x_0,
\qquad
d_y=y-y_0.
$$

双线性采样函数 $\mathcal B$ 为：

$$
\begin{aligned}
\mathcal B(V,s)={}&
(1-d_x)(1-d_y)V[y_0,x_0]\\
&+d_x(1-d_y)V[y_0,x_1]\\
&+(1-d_x)d_yV[y_1,x_0]\\
&+d_xd_yV[y_1,x_1].
\end{aligned}
$$

这里四个 $V[y,x]$ 都是 32 维 Head 特征，所以采样结果也是 32 维向量。

---

## 12. 单个 Head 对多尺度采样结果加权

对于一个 BEV 格子 `p`、一台相机 `c` 和一个 Head `h`：

$$
g_{p,c}^{(h)}
=
\sum_{l=0}^{3}
\sum_{k=0}^{7}
\alpha_{p,h,l,k}\,
\mathcal B
\left(
V_{c,l}^{(h)},
s_{p,c,h,l,k}
\right).
$$

一个 Head 读取：

```text
4 个 FPN Level × 每层 8 个 Point = 32 个采样结果
```

每个采样结果是 32 维，加权求和后仍然是：

$$
g_{p,c}^{(h)}\in\mathbb{R}^{32}.
$$

八个 Head 拼接成一台相机的 256 维结果：

$$
g_{p,c}
=
\operatorname{Concat}
\left(
g_{p,c}^{(0)},
g_{p,c}^{(1)},
\ldots,
g_{p,c}^{(7)}
\right).
$$

因此：

$$
g_{p,c}\in\mathbb{R}^{256}.
$$

---

## 13. 融合六台相机

设能够看到格子 `p` 的有效相机集合为 $\mathcal V_p$，并定义：

$$
N_p=\max\left(1,\left|\mathcal V_p\right|\right).
$$

官方实现没有再学习一组跨相机 Softmax 权重，而是对有效相机结果做算术平均：

$$
\overline g_p
=
\frac{1}{N_p}
\sum_{c\in\mathcal V_p}g_{p,c}.
$$

如果某个格子只有前相机和前右相机可见，那么这里只平均这两台相机的结果。

---

## 14. 把相机特征写回 BEV Query

相机平均结果经过输出投影：

$$
o_p=W_o\overline g_p+b_o.
$$

执行 Dropout 和残差相加：

$$
\widehat q_p
=
q_p+\operatorname{Dropout}(o_p).
$$

随后执行 LayerNorm：

$$
q_p^{\mathrm{SCA}}
=
\operatorname{LayerNorm}(\widehat q_p).
$$

把这一阶段写成一条清晰的计算链：

```text
s = 几何投影位置 + Query 预测偏移
g_head = Σ(采样权重 × 双线性采样特征)
g_camera = Concat(8 个 Head)
g = 有效相机平均
q_new = LayerNorm(q_old + Dropout(output_proj(g)))
```

这里不再使用一个超长、深度嵌套的公式。上述五行与源码的数据依赖一一对应，也更容易检查每个中间变量的形状。

---

## 15. 用具体数字手算一次

为了可以手工验算，先缩小成：

```text
1 个相机
1 个 FPN Level
1 个 Head
2 个采样点
特征维度 2
```

假设 Query 为：

$$
q=[1,2].
$$

几何投影后的参考位置已经换算为特征图连续坐标：

$$
\rho=(1.2,1.4).
$$

Query 经过两个线性层后得到：

$$
\Delta_1=(0,0),
\qquad
\Delta_2=(1,0),
$$

$$
a=[0,\ln3].
$$

对 Logit 做 Softmax：

$$
\alpha=[0.25,0.75].
$$

两个最终采样位置为：

$$
s_1=(1.2,1.4),
\qquad
s_2=(2.2,1.4).
$$

假设局部特征图为：

```text
             x=1       x=2       x=3
y=1        [10,0]     [20,0]     [30,0]
y=2        [10,10]    [20,10]    [30,10]
```

双线性采样得到：

$$
\mathcal B(V,s_1)=[12,4],
$$

$$
\mathcal B(V,s_2)=[22,4].
$$

按权重求和：

$$
g
=0.25[12,4]+0.75[22,4]
=[19.5,4].
$$

暂时假设输出投影是单位矩阵，则残差结果为：

$$
\widehat q
=q+g
=[1,2]+[19.5,4]
=[20.5,6].
$$

如果为了演示取 LayerNorm 的 $\gamma=1$、$\beta=0$，则归一化结果约为：

$$
q^{\mathrm{SCA}}=[1,-1].
$$

这就是一次完整的“Query 控制采样，再接收图像特征”的计算。

真实 Base 只是把它扩展为：

```text
2 维      → 256 维
1 Head    → 8 Head
1 Level   → 4 Level
1 Camera  → 最多 6 Camera
2 Point   → 每个 Level 8 Point
1 Query   → 40,000 Query
```

数学结构没有变化。

---

## 16. 一层完整 BEV Encoder

Spatial Cross-Attention 负责从当前六相机中读取图像信息。完整 Encoder 层还包括历史 BEV 融合和 FFN：

```text
Temporal Self-Attention
        ↓
LayerNorm
        ↓
Spatial Cross-Attention
        ↓
LayerNorm
        ↓
FFN：256 → 512 → 256
        ↓
LayerNorm
```

设第 `r` 层的输入为 $Q^{(r)}$，上一帧对齐后的 BEV 为 $Q_{t-1}$。为了避免把残差加两次，下面用 $\Phi_{\mathrm{TSA}}$ 和 $\Phi_{\mathrm{SCA}}$ 专门表示两个 Attention 模块产生的**残差分支结果**。

Temporal Self-Attention 阶段可概括为：

$$
A^{(r)}
=
\operatorname{LayerNorm}
\left(
Q^{(r)}+
\Phi_{\mathrm{TSA}}(Q^{(r)},Q_{t-1})
\right).
$$

当前图像融合阶段为：

$$
B^{(r)}
=
\operatorname{LayerNorm}
\left(
A^{(r)}+
\Phi_{\mathrm{SCA}}(A^{(r)},F_t)
\right).
$$

FFN 阶段为：

$$
Q^{(r+1)}
=
\operatorname{LayerNorm}
\left(
B^{(r)}+
W_2\sigma(W_1B^{(r)}+b_1)+b_2
\right).
$$

Base 重复六层：

$$
Q^{(0)}\rightarrow Q^{(1)}\rightarrow\cdots\rightarrow Q^{(6)}.
$$

最终输出仍然是：

$$
Q^{(6)}\in\mathbb{R}^{B\times40000\times256}.
$$

它也可以重排成：

```text
[B, 256, 200, 200]
```

这是一张融合了当前六相机信息和历史信息的 BEV 特征图，还不是最终的三维检测框。

---

## 17. 为什么 CNN 之后的特征仍然能按位置采样

CNN 特征不能再解释成“这个值就是原图某个像素”，但它也不是一个失去位置的无序向量。

卷积和 FPN 输出仍有明确的二维布局：

```text
[Channel, Height, Width]
```

特征图的某个位置对应原图中的一个确定感受野。随着卷积、下采样和 DCNv2：

- 一个特征会综合周围区域的信息；
- 相邻位置的感受野会重叠；
- 更深层的特征具有更大的感受野；
- DCNv2 会学习感受野内部更灵活的采样位置。

所以准确说法是：

> 相机投影点对应的不是一个原始像素值，而是特征图上一个具有确定空间位置和感受野的语义向量。

几何投影提供基准位置，Deformable Attention 的偏移负责进一步修正。

---

## 18. 这些参数是怎样学会的

固定部分包括：

- `200 × 200` BEV 网格；
- 四个高度参考点；
- 相机标定矩阵；
- 透视投影公式；
- 双线性采样公式。

可学习部分包括：

- 初始 BEV Query；
- Backbone 与 FPN；
- $W_{\Delta}$：采样偏移预测；
- $W_{\alpha}$：采样权重预测；
- $W_v$：Value 投影；
- $W_o$：输出投影；
- FFN 和 LayerNorm 参数。

双线性采样不仅对特征值可导，也对采样坐标可导。因此检测损失可以沿下面的路径反向传播：

```text
检测损失
  ↓
BEV 检测输出
  ↓
更新后的 BEV Query
  ↓
采样权重 α 与采样坐标 s
  ↓
偏移预测 WΔ、权重预测 Wα
  ↓
FPN 与 Backbone
```

模型不需要人为给每个 BEV 格子标注“应该读取哪个图像特征”。三维几何先提供大致位置，检测损失再学习偏移与权重。

---

## 19. 它与普通 Attention 的区别

普通全局 Attention 为：

$$
Y=\operatorname{Softmax}
\left(
\frac{QK^{T}}{\sqrt d}
\right)V.
$$

它需要每个 Query 与大量 Key 做点积。

BEVFormer Spatial Cross-Attention 则：

- 不计算完整的 $QK^T$；
- 用相机几何产生参考地址；
- 用 Query 直接预测 Offset 和 Weight；
- 每个 Head 只在四层特征图上读取少量采样点；
- 把采样内容写回原 Query。

所以它更准确的名称是：

> **几何引导、Query 控制的多相机多尺度稀疏特征读取。**

---

## 20. 源码映射

| 计算阶段 | 源码位置 |
| --- | --- |
| Base 参数与 Encoder 配置 | [bevformer_base.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/configs/bevformer/bevformer_base.py:31) |
| BEV Query、CAN Bus 与四层特征展平 | [transformer.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/transformer.py:104) |
| 生成 2D/3D Reference Points | [encoder.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py:47) |
| 三维点投影到六台相机 | [encoder.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py:89) |
| 六层 Encoder 循环 | [encoder.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py:214) |
| 每层 TSA、SCA、FFN 顺序 | [encoder.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/encoder.py:356) |
| 有效 Query 按相机重排 | [spatial_cross_attention.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:136) |
| Query 产生 Offset 与 Weight | [spatial_cross_attention.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:338) |
| 四高度与八采样点 reshape | [spatial_cross_attention.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:360) |
| CUDA 多尺度双线性采样 | [spatial_cross_attention.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:390) |
| 有效相机求平均并写回 Query | [spatial_cross_attention.py](C:/Users/Thomas/Desktop/project/BEVFormer-master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py:165) |

---

## 21. 一句话记忆

```text
几何投影决定“从哪里开始找”；
BEV Query 决定“向哪里偏移、相信哪些点”；
图像特征提供“真正写进 BEV 的内容”；
残差连接把内容保存回原来的 BEV 格子。
```
