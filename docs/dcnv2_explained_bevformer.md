# DCNv2 可变形卷积：结构、数学原理与 BEVFormer 中的作用

本文按“普通卷积 → 改变采样位置 → 增加调制系数 → 张量形状 → 完整计算 → BEVFormer → 编译与硬件”的顺序展开。

范围：二维调制可变形卷积（modulated deformable convolution），结合官方 BEVFormer v1 的 Base / Tiny 配置，以及 MMCV 1.4.0、MMDetection 2.14.0 的实现。核对日期：2026-10-17。文中的小矩阵和数值是教学示例，不是模型实测激活。

## 1. 先把“形状”分成三个问题

**DCNv2 让卷积在不同位置，根据输入内容选择不同的采样坐标，并调节各采样点的贡献。**

“DCNv2”原本是一组网络改进的名称；在模型配置中，`DCNv2` 通常指其中的调制可变形卷积层。它可以嵌入 ResNet，不是必须独立使用的一整套检测网络。[论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html)

| 你问的“形状” | 实际含义 | DCNv2 中的情况 |
| --- | --- | --- |
| 网络结构 | 哪些层相连，哪些分支先算 | 增加预测 offset 和 mask 的分支，再执行采样与加权求和 |
| 张量 shape | 输入、权重、输出分别有几个维度 | 仍是规则张量，例如输入 `[N,C,H,W]`、权重 `[Cout,Cin,3,3]` |
| 空间采样形状 | 一个输出读取输入中的哪些位置 | 九个固定网格点可以各自移动，形成不规则采样分布 |

**变形的是采样分布。权重数组的尺寸不随每张图片变化，输出也仍然放在规则网格上。**

文中统一用 `(y,x)` 表示“行坐标、列坐标”。坐标单位是当前特征图上的格子，不直接等于原始图像像素或世界坐标的米。

## 2. 普通 3×3 卷积到底在做什么

先只看一个输入通道、一个输出通道，步长为 1。在输出位置 `(i,j)`，普通 3×3 卷积读取一个固定邻域：

```text
输入特征图上的固定采样点：

       j-1      j       j+1
 i-1    ●       ●        ●
 i      ●       ●        ●
 i+1    ●       ●        ●

九个输入值 × 对应的九个权重 → 求和 → 输出 Y[i,j]
```

忽略边界和 bias，可以写成：

$$
Y[i,j]=\sum_{a=-1}^{1}\sum_{b=-1}^{1}W[a,b]X[i+a,j+b].
$$

这里 `W[a,b]` 只是用相对坐标标记九个权重。程序中通常将它们存成下标从 0 开始的 3×3 数组。

如果有多个输入通道，需要把每个输入通道的贡献也加起来。不同输出通道使用不同的卷积核。

普通卷积的关键约束是：**每个输出位置使用相同的相对采样网格。** 卷积权重可以学习，但这九个相对位置本身是固定的。

例如，同一个车辆在远处很小、在近处很大，侧面和正面的投影形状也不同。固定网格能够通过多层网络学习这些变化，但单层卷积不能根据当前内容直接移动自己的采样点。

## 3. 从普通卷积到 DCNv1，再到 DCNv2

### 3.1 DCNv1：每个采样点都可以移动

对第 `k` 个采样点，增加一个二维偏移：

$$
\Delta p_k=(\Delta y_k,\Delta x_k).
$$

原来在 `p+r_k` 取值，现在在 `p+r_k+Δp_k` 取值。`p` 是当前输出对应的参考位置，`r_k` 是规则卷积网格中的相对位置。

偏移可以是小数，例如 `(0.25,0.50)`；九个点的偏移彼此不必相同。相邻输出位置也可以预测不同的九组偏移。

```text
普通卷积的一个点：    原定位置 (2,3)           → 读取 X[2,3]
可变形卷积的这个点：  原定位置 (2,3)+(0.25,0.5) → 在 (2.25,3.5) 插值取值
```

3×3 的九个权重仍然存在，每个权重仍对应自己的采样点。移动后，这九个点不必围成规则方形，也不必形成某个固定大小的矩形。

### 3.2 DCNv2：移动后，还决定这个点贡献多少

DCNv2 再增加一个调制系数 `m_k`：

$$
m_k=\operatorname{sigmoid}(z_k).
$$

有限实数下，`m_k` 在 0 与 1 之间。它不是必须取 0 或 1 的开关，也不要求九个系数之和等于 1。

| 内容 | 它决定什么 |
| --- | --- |
| 卷积权重 `W` | 各通道、各采样点的特征怎样组合 |
| 偏移 `Δp` | 去输入特征图的什么位置读取 |
| 调制系数 `m` | 这个采样值在当前位置贡献多少 |

单通道的核心计算是：

$$
Y(p)=\sum_{k=1}^{K}W_k\,\widetilde X(p+r_k+\Delta p_k)\,m_k+b,
\qquad K=K_hK_w.
$$

`X̃` 表示允许在小数坐标上插值取值的特征图。该设计允许网络调整取样范围，并减弱部分不相关采样的影响；不能保证每个点都恰好落在目标轮廓上。[DCNv2 论文](https://arxiv.org/abs/1811.11168)

## 4. offset 和 mask 从哪里来

它们通常由另一个普通卷积从当前输入特征中预测。以 `3×3、deform_groups=1` 的 Pack 模块为例：

```text
                         输入特征 X
                         /       \
                        /         \
             普通 Conv_offset      原始特征保留供采样
              输出 27 个通道              |
                    |                     |
             +------+------+              |
             |             |              |
        18 通道 offset   9 通道 logits     |
        每点两个坐标         |              |
             |           Sigmoid          |
             |             |              |
             |          9 通道 mask        |
             +-------------+--------------+
                           |
               按偏移坐标进行双线性采样
                           |
                 采样值 × mask × 权重 W
                           |
                  对通道和采样点求和
                           |
                      输出特征 Y
```

“27 个通道”来自：`9 个点 ×（2 个偏移坐标 + 1 个调制值）`。

这不意味着要输出 27 张检测结果。它们是这层内部用于控制采样的中间张量。

预测分支的权重是训练参数；它输出的 offset、mask 是运行时激活。推理时，预测分支的权重固定，但输入图片改变，offset 和 mask 仍会改变。

MMCV 的 `ModulatedDeformConv2dPack` 把预测分支和可变形卷积封装起来；底层 `ModulatedDeformConv2d` 则接收调用者提供的 `x、offset、mask`。零初始化的预测分支产生 `offset=0、mask=0.5`，因此不能说初始化后必然与原普通卷积数值完全相同；加载训练好的 checkpoint 后应以实际参数为准。[MMCV 实现](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/modulated_deform_conv.py)

## 5. 输入、权重、offset、mask 的张量形状

### 5.1 通用接口

以下使用 NCHW 布局。定义普通卷积分组数为 `G`，可变形分组数为 `D`，采样点数为 `K=Kh×Kw`。

| 对象 | Shape | 是权重还是运行时数据 |
| --- | --- | --- |
| 输入 `X` | `[N,Cin,Hin,Win]` | 运行时特征 |
| 主卷积权重 `W` | `[Cout,Cin/G,Kh,Kw]` | 训练参数 |
| 可选主卷积 bias | `[Cout]` | 训练参数 |
| offset | `[N,2×D×K,Hout,Wout]` | 预测出来的运行时数据 |
| mask | `[N,D×K,Hout,Wout]` | 预测出来并经过 sigmoid 的运行时数据 |
| 输出 `Y` | `[N,Cout,Hout,Wout]` | 计算结果 |

`G` 与 `D` 的用途不同：`G` 决定哪些输入通道参与某个输出通道的卷积求和；`D` 决定哪些输入通道共享一组采样偏移和调制系数。不能把两个参数混为一谈。

当 `D=1` 时，所有输入通道在同一输出位置共享九组坐标和 mask；各通道被采样的特征值不同，卷积权重也仍按输入、输出通道区分。

输出空间尺寸仍按普通卷积规则确定：

$$
H_{out}=\left\lfloor\frac{H_{in}+2P_h-D_h(K_h-1)-1}{S_h}\right\rfloor+1.
$$

宽度同理。这里 `D_h` 表示高度方向的 **dilation**，与可变形分组数 `D` 是不同变量。offset 不改变输出网格的数量。

### 5.2 一个能看清大小的例子

假设 `N=1、Cin=Cout=64、H=W=32、Kh=Kw=3、stride=1、padding=1、dilation=1、G=D=1`。

```text
X                  [1, 64, 32, 32]
主卷积 W           [64, 64, 3, 3]
预测分支的 W_offset [27, 64, 3, 3]
预测分支的 bias     [27]
预测分支输出         [1, 27, 32, 32]
offset             [1, 18, 32, 32]
mask               [1,  9, 32, 32]
Y                  [1, 64, 32, 32]
```

offset 和 mask 的数据量不乘 `Cout`，因为它们可以被多个输出通道复用；输出通道之间的差异由主卷积权重体现。

实现时还必须核对坐标排列。例如所核对的 MMCV CUDA Kernel 在每个可变形组内按 `dy0,dx0,dy1,dx1,…` 读取 offset。不能仅看到 Python 代码将预测结果切成三块，就认定前九个通道全部是 y、接着九个全部是 x。[坐标读取代码](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/common/cuda/modulated_deform_conv_cuda_kernel.cuh)

## 6. 小数坐标怎么读取：把双线性插值算一遍

假设某个通道上有以下四个值：

```text
                 x=3       x=4
         y=2      10        20
         y=3      30        40
```

现在需要读取 `(y,x)=(2.25,3.5)`。这个位置不对应一个独立存储的元素，需要由四个邻居算出来。

定义：

$$
y_0=\lfloor y\rfloor,\quad x_0=\lfloor x\rfloor,
\quad \alpha=y-y_0,\quad \beta=x-x_0.
$$

双线性插值为：

$$
\begin{aligned}
\widetilde X(y,x)=
&(1-\alpha)(1-\beta)X[y_0,x_0]\\
&+(1-\alpha)\beta X[y_0,x_0+1]\\
&+\alpha(1-\beta)X[y_0+1,x_0]\\
&+\alpha\beta X[y_0+1,x_0+1].
\end{aligned}
$$

代入 `α=0.25、β=0.5`：

$$
\widetilde X=0.375\times10+0.375\times20+0.125\times30+0.125\times40=20.
$$

假设这一个采样点的卷积权重是 `2`，mask 是 `0.8`：

$$
\text{该项贡献}=2\times20\times0.8=32.
$$

**32 还不是整个输出值。** 必须把其余采样点和其余输入通道的贡献也加上，最后再加可选 bias，才能得到一个输出元素。

边界处理通常采用零扩展：四个整数邻居中，落在图外的邻居按零处理。接近边界的小数坐标可能仍有有效邻居，不能一看到坐标小于零就把所有贡献清零。也不能未经约定，把越界坐标强行夹到边缘；那会改变算子语义。

## 7. 从一个采样点到完整二维 DCNv2

为了把实现关系写清楚，先取 `G=D=1`。令输出位置为 `(i,j)`，卷积核下标为 `a=0…Kh-1、b=0…Kw-1`。

第一步，计算规则卷积原本要访问的坐标：

$$
q_y^{base}=iS_h-P_h+aD_h,\qquad
q_x^{base}=jS_w-P_w+bD_w.
$$

第二步，加上这个位置、这个采样点的偏移：

$$
q_y=q_y^{base}+\Delta y[n,a,b,i,j],\qquad
q_x=q_x^{base}+\Delta x[n,a,b,i,j].
$$

第三步，插值、调制并累加：

$$
Y[n,o,i,j]=b_o+
\sum_{c=0}^{C_{in}-1}\sum_{a=0}^{K_h-1}\sum_{b=0}^{K_w-1}
W[o,c,a,b]\;m[n,a,b,i,j]\;\widetilde X[n,c,q_y,q_x].
$$

这里 `Δy、Δx、m` 用语义上的二维 kernel 下标展开书写；实际存储通常使用前述通道打包形式。求和下标 `b` 与偏置符号 `b_o` 含义不同。

与公式对应的教学伪代码如下，省略 batch 和 groups：

```python
for output_y, output_x in output_grid:
    for out_channel in output_channels:
        accumulator = bias[out_channel]  # 没有 bias 时为 0
        for in_channel in input_channels:
            for kernel_y, kernel_x in kernel_grid:
                sample_y = output_y * stride_y - pad_y + kernel_y * dilation_y
                sample_x = output_x * stride_x - pad_x + kernel_x * dilation_x
                sample_y += offset_y[kernel_y, kernel_x, output_y, output_x]
                sample_x += offset_x[kernel_y, kernel_x, output_y, output_x]
                feature = bilinear_sample_with_zero_padding(
                    X[in_channel], sample_y, sample_x)
                accumulator += (
                    W[out_channel, in_channel, kernel_y, kernel_x]
                    * mask[kernel_y, kernel_x, output_y, output_x]
                    * feature)
        Y[out_channel, output_y, output_x] = accumulator
```

这是语义解释，不是高性能实现。真实 Kernel 会并行计算、复用坐标和采样值，或者先展开采样结果再进行矩阵乘法。

三个退化情形有助于检查理解：

| 条件 | 结果 |
| --- | --- |
| `offset=0、mask=1` | 与相同参数的普通卷积一致 |
| `offset` 可变、`mask=1` | 是不带调制的可变形卷积形式 |
| `offset=0、mask` 可变 | 固定位置取样，但每个位置的采样贡献可变 |

第一个条件中的 `mask=1` 是数学上的退化条件；有限 sigmoid 输入通常不会恰好产生 1。

## 8. 没有人标注 offset，它是怎么学出来的

检测训练通常提供类别、位置等任务监督，不要求逐一标注每个 DCN 采样点应该去哪里。

```text
检测损失
   ↓ 反向传播
输出特征 Y
   ├── 更新主卷积权重 W
   ├── 经过 mask，更新调制预测参数
   └── 经过双线性插值，更新 offset 预测参数
```

双线性插值在每个网格单元内部对坐标可求导。例如：

$$
\frac{\partial\widetilde X}{\partial y}
=(1-\beta)(X[y_0+1,x_0]-X[y_0,x_0])
+\beta(X[y_0+1,x_0+1]-X[y_0,x_0+1]).
$$

因为 `q_y=规则坐标+Δy`，坐标梯度可以继续传给偏移预测分支。对于单个求和项，令 `m=sigmoid(z)`，则对 mask logit 的导数是：

$$
\frac{\partial (W\widetilde X m)}{\partial z}
=W\widetilde X\,m(1-m).
$$

多个通道、多个输出对共享变量的梯度需要累加；整数网格边界等非光滑位置由实现选定梯度处理方式。推理时只做前向计算，不运行反向传播或在线更新参数。

DCNv2 论文还讨论了训练策略；在某个 backbone 中使用 DCNv2 算子，不代表自动使用论文的所有训练方法。

## 9. 它在 BEVFormer-base 中具体放在哪里

### 9.1 先看全模型中的位置

```text
六路相机图像
    ↓
共享权重的 ResNet-101 图像 backbone
    ├── 较浅 stage：普通卷积
    └── 后两个 stage：部分卷积替换为 DCNv2
    ↓
多尺度 FPN 图像特征
    ↓
BEV Encoder：空间交叉注意力 + 时间自注意力
    ↓
BEV 特征
    ↓
检测 Decoder / Head
    ↓
类别、3D 框等输出
```

**DCNv2 工作在单相机的二维图像特征提取阶段。** 六个相机可以在实现中合并到 batch 维计算，但 DCNv2 本身不会因此跨相机求和或建立三维对应关系。

BEVFormer 的跨相机空间融合与历史 BEV 融合由后面的 Transformer 模块负责。[BEVFormer 论文](https://arxiv.org/abs/2203.17270)

### 9.2 哪些 stage 使用它

官方 Base 配置的关键设置为 `depth=101`、`style='caffe'`、`deform_groups=1`，并指定 `stage_with_dcn=(False, False, True, True)`。[Base 配置](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py)

对应的 ResNet-101 主体如下：

| Stage | 常用输出名 | Bottleneck 数 | Stage 输出通道 | 中间 3×3 卷积 |
| --- | --- | ---: | ---: | --- |
| `layer1` | C2 | 3 | 256 | 普通卷积 |
| `layer2` | C3 | 4 | 512 | 普通卷积 |
| `layer3` | C4 | 23 | 1024 | DCNv2 |
| `layer4` | C5 | 3 | 2048 | DCNv2 |

因此每个相机的 backbone 经过 `23+3=26` 个 DCNv2 模块。六相机共对应 `6×26=156` 次单相机模块计算；如果合并 batch，实际算子调用次数可以少于 156。网络模块数、图节点数和 GPU Kernel 数不能直接画等号。

一个采用 DCNv2 的 Bottleneck 主分支大致是：

```text
输入 ────────────────────────────────┐
  ↓                                  │ shortcut
1×1 Conv → BN → ReLU                 │ 必要时用 1×1 投影
  ↓                                  │
3×3 DCNv2 → BN → ReLU                │
  ↓                                  │
1×1 Conv → BN                        │
  ↓                                  │
  + ←────────────────────────────────┘
  ↓
ReLU
```

被替换的是中间的 `conv2`。例如 C4 内部 DCNv2 的通道是 `256→256`，最后一个 1×1 卷积才扩展到 1024 通道。不能把整个 stage 的输出通道直接当作内部 DCNv2 的通道数。`caffe` 风格将 stage 下采样的 stride 放在首个 1×1 卷积中。[MMDetection ResNet 实现](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py)

### 9.3 与我们 Base 输入形状对应起来

按本项目已使用的 padding 后输入 `[1,6,3,928,1600]`，展开相机维后可理解为 `[6,3,928,1600]`。根据上述 backbone 步长推导：

| 特征 | 相对输入下采样 | 张量 shape |
| --- | ---: | --- |
| C3 | 8 倍 | `[6,512,116,200]` |
| C4 | 16 倍 | `[6,1024,58,100]` |
| C5 | 32 倍 | `[6,2048,29,50]` |

例如 C4 中间的某个 DCNv2 接口是：

```text
输入       [6,256,58,100]
offset     [6,18,58,100]
mask       [6, 9,58,100]
DCN 输出   [6,256,58,100]
随后 1×1   [6,1024,58,100]
```

这些是按配置计算的逻辑形状；后端可以逐相机执行，物理 batch / 布局也可能不同。928 是此测试图的 padding 后高度，不能把它写成所有 BEVFormer 输入的固定高度。

### 9.4 对检测有什么作用

车辆大小、朝向、透视和遮挡都会改变图像中的有效特征分布。DCNv2 使 backbone 能按内容调整取样位置与贡献，为后续 FPN 和 BEV Encoder 提供更灵活的二维特征。

它不会直接把图像转换到鸟瞰坐标，也不会直接输出车辆速度或 3D 框。更灵活的特征表达是否带来多少精度收益，需要在相同训练和评测设置下做消融实验，不能仅凭使用 DCNv2 就给出固定提升幅度。

**官方 BEVFormer-tiny 使用普通 ResNet-50，没有启用这套 DCNv2 backbone 配置。** 因此 DCNv2 不是 BEVFormer 算法存在的必要条件，但删除 Base 中已训练的 DCNv2 不能当作语义等价的编译优化。[Tiny 配置](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py)

## 10. DCNv2 与 MSDeformAttn 为什么不是同一个算子

名字里都有“可变形”，是因为都涉及可学习的采样位置；但它们的计算和位置不同。

| 比较项 | DCNv2 | BEVFormer 中的 MSDeformAttn |
| --- | --- | --- |
| 所在模块 | 图像 backbone | Transformer 注意力模块 |
| 基本参考位置 | 卷积输出位置及规则 kernel 网格 | Query 相关的 reference points |
| 输出组织 | 规则二维特征图、输出通道 | Query 对应的特征、attention heads |
| 聚合方式 | 卷积权重 × 调制采样值，对通道及采样点求和 | 对采样的 value 按注意力权重聚合 |
| 常见权重约束 | mask 经 sigmoid，不要求和为 1 | 所核对的 MSDA 实现对采样权重做 softmax |

空间交叉注意力还要结合相机投影、可见性筛选与多尺度特征；这些不能由 backbone 中的 DCNv2 替代。[BEVFormer SpatialCrossAttention / MSDeformableAttention3D](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py)

## 11. 从编译器与硬件看，为什么它比较难加速

### 11.1 动态坐标导致读取位置不规则

普通卷积中，输出 tile 对应的输入区域通常可由 kernel、stride 和 dilation 预先推导。DCNv2 的实际地址还取决于运行时 offset。

因此，普通卷积中固定大小的 halo 不能无条件覆盖 DCNv2 的全部采样点。若模型没有约束偏移范围，编译器就不能为了让 tile 放进 SRAM，擅自截断 offset；那会改变模型输出。

可选择的实现包括直接 gather 全局内存、利用缓存、运行时分析访问范围，或在模型设计阶段约束采样范围。具体方案要同时看正确性、工作区大小和测量结果。

### 11.2 一个小数采样点通常需要访问四个邻居

每个采样点需要坐标计算、边界处理和插值。相邻线程的访问位置未必连续，数据复用也受 offset 影响。但这不等于耗时一定变成普通卷积的四倍：缓存、复用、矩阵乘效率和 Kernel 组织都会影响结果。

### 11.3 常见实现会使用展开 Buffer，再做 GEMM

一种实现思路是：

```text
输入 X + offset + mask
          ↓ 不规则读取、双线性插值、调制
采样展开 Buffer：每个输出位置需要的 Cin×Kh×Kw 个值
          ↓
与卷积权重做 GEMM
          ↓
输出 Y
```

在前面 `N=1、Cin=64、Hout=Wout=32、3×3` 的例子中，如果完整展开：

```text
输入：         64 × 32 × 32       =  65,536 个元素
展开 Buffer： (64 × 9) × (32 × 32) = 589,824 个元素
```

若这些中间数据用 FP16，输入占 128 KiB，展开 Buffer 占 1.125 MiB；这还不含权重、offset、mask 和输出。实现可以分块展开来减小工作区，也可以采用融合方案避免完整落地。

**这里的 Buffer 存的是计算期间的特征数据。** 主卷积权重存放在哪里、展开 Buffer 存放在哪里，是两个独立问题；Buffer 也不必全部位于 SRAM，可以位于显存、shared memory 或其他存储层级。

这种展开形式与所核对的 MMCV CUDA 路径一致：先生成经 mask 调制的采样列数据。不同后端和插件可以有不同实现，不能把这一模板当成 DCNv2 数学定义的一部分。[MMCV 采样实现](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/common/cuda/modulated_deform_conv_cuda_kernel.cuh)

### 11.4 编译器必须保留的语义

实现或 Lowering 时，至少需要对齐：

- 主权重、offset、mask 的形状和坐标通道顺序。
- stride、padding、dilation，以及普通 groups 与 deform_groups。
- 双线性插值的坐标单位、零扩展和边界行为。
- 哪些张量共享偏移，哪些通道参与同一个输出的求和。
- 数据类型、累加精度和容许的数值误差。

可以把规则的乘加部分映射到 GEMM / Tensor Core，但动态采样仍然要正确执行，不能直接用普通卷积替代整个算子。量化 offset 会改变采样位置，量化 mask 会改变贡献比例，所以 backbone 的普通 Conv 用 INT8，并不意味着 DCNv2 也应无条件使用 INT8。

### 11.5 对应我们已有的性能报告

本项目 2026-10-17 的 RTX 3090 混合精度测试中，IProfiler 将 156 条执行层回调归为 DCN 插件，均值合计约 **28.635 ms**，占全部层回调均值之和的 **13.54%**。

这只是该引擎、该输入和该采样方法的结果；预测 offset 的普通卷积等工作可能归入其他执行层，不能把这个数视为所有 DCNv2 相关计算的完整总成本。图使用合成投影、无历史帧，不能当作真实车载时序性能结论。[项目逐层报告（固定版本）](https://github.com/luogantt/General-AI-Compiler/blob/24baf60ca79318cb072881177613608c453a4ea7/docs/validation/tensorrt/bevformer-base-2026-10-17/profiling/README.md)

## 12. 几个常见误解

| 容易误解的说法 | 准确理解 |
| --- | --- |
| “3×3 卷积核变成了一个更大的权重矩阵。” | 权重 shape 仍可为 3×3；变的是九个采样点的位置。 |
| “offset 是 checkpoint 里固定存好的九个坐标。” | 通常由训练好的预测分支根据当前特征生成，随输入和位置变化。 |
| “mask 是是否保留这个物体的二值判断。” | 它是每个采样点的连续调制系数，不等于物体分割标签。 |
| “DCNv2 的输出是不规则排列的点云。” | 输出仍是规则特征张量；不规则的是读取输入的位置。 |
| “推理输出 Buffer 里存的就是卷积核。” | 卷积核是参数；输出 Buffer 存的是使用参数计算出的特征。 |
| “DCNv2 就是 BEVFormer 的注意力。” | 它在 backbone 提取图像特征；后面的可变形注意力是另一种算子。 |
| “采样点移动，是硬件把图片或权重搬动了。” | 主要是读取地址和插值计算改变，不要求重新排列整张图片或权重。 |
| “Base 换成普通卷积就等价，而且肯定更快。” | 需要重新训练或适配并验证精度；加速多少也必须测量。 |

阅读代码时，可以沿着以下数据关系逐项核对：

```text
当前输入特征
   → 预测 offset 与 mask
   → 规则坐标加 offset
   → 从输入特征中插值取值
   → 乘 mask 和主卷积权重
   → 沿输入通道及采样点累加
   → 写入规则输出特征图
```

## 参考源码与论文

1. [DCNv2 论文，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html)：调制可变形卷积的设计与实验。
2. [BEVFormer 论文，ECCV 2022](https://arxiv.org/abs/2203.17270)：图像特征、空间交叉注意力和时间自注意力的职责。
3. [BEVFormer-base 配置](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py)：ResNet-101、DCNv2 stage 和 FPN 设置。
4. [BEVFormer-tiny 配置](https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py)：普通 ResNet-50、单尺度特征。
5. [MMCV 1.4.0 的调制可变形卷积](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/modulated_deform_conv.py)：底层算子接口、Pack 预测分支及初始化。
6. [MMCV 1.4.0 CUDA 采样实现](https://github.com/open-mmlab/mmcv/blob/v1.4.0/mmcv/ops/csrc/common/cuda/modulated_deform_conv_cuda_kernel.cuh)：offset 排列、坐标和双线性插值。
7. [MMDetection 2.14.0 的 ResNet](https://github.com/open-mmlab/mmdetection/blob/v2.14.0/mmdet/models/backbones/resnet.py)：Bottleneck 结构、stage 数量和 DCN 替换位置。
