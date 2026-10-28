# ONNX 计算图与 GAC MLIR 计算图对应关系

## 1. 核心结论

ONNX 和当前图级 GAC MLIR 都可以表示计算图，但二者的定位不同：

> **ONNX 是用于保存和交换模型的 protobuf 计算图；MLIR 是用于编译、改写和逐层 Lowering 的 SSA 中间表示。**

当前 GAC 的 `func.func @main` 可以看成一张数据流计算图。但是 MLIR 本身还能表达循环、分支、函数、Region、Block 和内存操作，所以它比 ONNX 计算图更通用。

正确术语是 **Dialect（方言）**。`gac` 是方言命名空间，`gac.matmul` 是这个方言中的一个 Operation。

## 2. 整体结构

```text
ONNX ModelProto
└── GraphProto
    ├── graph.input
    ├── initializer
    ├── NodeProto[]
    ├── value_info
    └── graph.output

              ↓ OnnxImporter

MLIR ModuleOp
└── func.func @main
    └── Block
        ├── BlockArgument
        ├── gac.constant
        ├── gac.conv2d
        ├── gac.matmul
        ├── gac.softmax
        └── func.return
```

## 3. 名词一一对应

| 计算图概念 | ONNX 数据结构 | MLIR/GAC 数据结构 | 说明 |
|---|---|---|---|
| 整个模型文件 | `ModelProto` | `ModuleOp` | 二者都是顶层容器，但不是严格相同的数据格式 |
| 主计算图 | `GraphProto` | `func.func @main` 和入口 `Block` | 当前导入器把 ONNX 主图变成一个公开函数 |
| 图输入 | `graph.input: ValueInfoProto` | `BlockArgument` | 函数入口参数本身就是 SSA Value |
| 计算节点 | `NodeProto` | `Operation`，如 `gac.matmul` | 通常一个 ONNX 节点创建一个 GAC Operation |
| 节点输入 | `NodeProto.input[]` 字符串 | `Operation operands` | 从字符串名称转换成对前驱 SSA Value 的直接引用 |
| 节点输出 | `NodeProto.output[]` 字符串 | `OpResult` | 输出名称被绑定到 Operation 的结果 Value |
| 节点间连线 | 相同的 Tensor 名称 | SSA `Value` 的 use-def 关系 | MLIR 不再依靠字符串查找上下游 |
| Tensor 类型 | `ValueInfoProto` | `RankedTensorType` / `UnrankedTensorType` | 包含元素类型、rank 和 shape |
| 静态权重 | `initializer: TensorProto` | `gac.constant + DenseElementsAttr` | 常量的真实数值保存在 `DenseElementsAttr` 中 |
| 算子参数 | `AttributeProto` | MLIR `Attribute` | 例如 `axis`、`strides`、`pads` |
| 图输出 | `graph.output` | `func.return` | 输出名称被解析为最终 SSA Value |
| 算子版本 | `domain + opset` | 导入规则及 `gac.opset` 属性 | 决定某个 ONNX 算子的准确语义版本 |
| 节点调试名称 | `NodeProto.name` | `Location + gac.source_name` | 用于诊断和追踪，不是 SSA 名称 |
| 子图和控制流 | `GraphProto` 类型属性 | MLIR `Region + Block` | 当前 GAC 只支持很小的控制流子集 |

## 4. ONNX 和 MLIR 如何保存上下游关系

这是两者最关键的区别。

### 4.1 ONNX：通过字符串连接节点

```text
MatMul:
    input  = ["X", "W"]
    output = ["Y"]

Add:
    input  = ["Y", "B"]
    output = ["Z"]

Softmax:
    input  = ["Z"]
    output = ["P"]
```

`"Y"` 和 `"Z"` 是 Tensor 名称，也是 ONNX 计算图中连接生产者与消费者的边。

### 4.2 导入过程：名称转换为 SSA Value

导入器维护：

```cpp
llvm::StringMap<mlir::Value> values;
```

转换过程可以写成：

```text
values["X"] = %x
values["W"] = %w

创建 %0 = gac.matmul(%x, %w)
values["Y"] = %0

创建 %1 = gac.add(%0, %b)
values["Z"] = %1

创建 %2 = gac.softmax(%1)
values["P"] = %2
```

最终形成类似下面的 MLIR：

```mlir
%0 = "gac.matmul"(%x, %w)
    : (tensor<1x128x256xf32>,
       tensor<256x256xf32>)
    -> tensor<1x128x256xf32>

%1 = "gac.add"(%0, %b)
    : (tensor<1x128x256xf32>,
       tensor<256xf32>)
    -> tensor<1x128x256xf32>

%2 = "gac.softmax"(%1) {
    axis = -1 : i64
} : (tensor<1x128x256xf32>)
    -> tensor<1x128x256xf32>
```

因此可以记成：

```text
ONNX 的边 = 输入输出字符串名称
MLIR 的边 = operand 直接引用前驱产生的 Value
```

在 MLIR 中可以直接查询：

```cpp
value.getDefiningOp(); // 找到上游生产者
value.getUsers();      // 找到所有下游消费者
```

不需要另外维护“节点 A 的下游是节点 B”的邻接表。

## 5. 一个 MLIR Operation 保存什么

以 `gac.matmul` 为例：

```text
Operation
├── name          "gac.matmul"
├── operands      %x、%weight
├── results       %result
├── result types  tensor<1x128x256xf32>
├── attributes    transpose、opset、来源信息等
├── location      对应哪个 ONNX 节点
└── regions       当前 gac.matmul 没有 Region
```

真正创建 Operation 的代码位于 [`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L439)：

```cpp
OperationState state(location, name);
state.addTypes(results);
state.addOperands(operands);
state.addAttributes(attrs);
return builder.create(state);
```

所以：

```text
ONNX MatMul → gac.matmul
```

不是字符串替换。导入器至少还要完成：

1. 将输入名称解析成 SSA `Value`。
2. 将 ONNX dtype 和 shape 转为 MLIR `Type`。
3. 将 `AttributeProto` 转为 MLIR `Attribute`。
4. 创建结果 `OpResult`。
5. 检查 domain 和 opset。
6. 验证输入数量、类型和当前支持的语义子集。
7. 将 ONNX 输出名称绑定到新结果。
8. 保存源节点位置和名称。

## 6. 当前标准 ONNX 算子映射

当前映射表定义了 48 个标准 ONNX 算子到 GAC 高层 Operation 的候选映射，代码位于 [`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L523)。

这里的“直接映射”表示一个 ONNX Node 创建一个 GAC Operation，仍然必须转换 operands、results、types、attributes、opset 和来源信息。

### 6.1 基本计算和逻辑

```text
MatMul  → gac.matmul
Gemm    → gac.gemm
Add     → gac.add
Sub     → gac.sub
Mul     → gac.mul
Div     → gac.div
Pow     → gac.pow
Max     → gac.max
Greater → gac.greater
Less    → gac.less
Equal   → gac.equal
And     → gac.and
Where   → gac.where
```

### 6.2 激活和数学函数

```text
Relu     → gac.relu
Sigmoid  → gac.sigmoid
Log      → gac.log
Sin      → gac.sin
Cos      → gac.cos
Exp      → gac.exp
Sqrt     → gac.sqrt
Softmax  → gac.softmax
```

### 6.3 Tensor、形状和归约

```text
Identity        → gac.identity
Reshape         → gac.reshape
Transpose       → gac.transpose
Concat          → gac.concat
Slice           → gac.slice
Expand          → gac.expand
Cast            → gac.cast
ReduceSum       → gac.reduce_sum
ReduceMean      → gac.reduce_mean
Shape           → gac.shape
Gather          → gac.gather
Unsqueeze       → gac.unsqueeze
Squeeze         → gac.squeeze
Range           → gac.range
ConstantOfShape → gac.constant_of_shape
Resize          → gac.resize
Tile            → gac.tile
ScatterND       → gac.scatter_nd
NonZero         → gac.nonzero
Split           → gac.split
```

### 6.4 神经网络算子

```text
Conv               → gac.conv2d
Clip               → gac.clamp
LayerNormalization → gac.layer_norm
BatchNormalization → gac.batch_norm
MaxPool            → gac.max_pool2d
GridSample         → gac.grid_sample
```

## 7. 需要额外转换的情况

### 7.1 权重 Initializer

ONNX 权重原来存放在 `initializer: TensorProto` 中：

```text
TensorProto
    ↓ 读取 shape、dtype、raw_data/external_data
DenseElementsAttr
    ↓
gac.constant
    ↓
OpResult
```

权重会显式变成计算图中的常量节点。

中间激活的 SSA `Value` 不保存运行时 Tensor 数据，只描述：

- 谁产生这个 Tensor；
- Tensor 的类型和形状；
- 哪些 Operation 使用这个 Tensor。

真正的激活数据要等 Runtime 执行时才进入 GPU 显存、Vortex DDR、LMEM 或寄存器。

### 7.2 ONNX Constant

ONNX Constant 可以用多种格式保存：

```text
value
value_float
value_int
value_floats
value_ints
```

导入器统一转换为：

```mlir
%w = "gac.constant"() {
    value = dense<...>
} : () -> tensor<...>
```

### 7.3 特殊 If 子图

当前不支持通用 ONNX `If`，只识别下面这个严格模式：

```text
Shape(X)
   ↓
Gather(axis)
   ↓
Equal(1)
   ↓
If
├── true  → Squeeze(X, axis)
└── false → Identity(X)
```

整个结构转换为：

```text
gac.squeeze_if_unit_dim
```

这是“一个控制流结构变成一个 GAC Operation”，不是节点改名。

### 7.4 自定义 ONNX Domain

```text
mmcv::MMCVModulatedDeformConv2d
    → gac.deform_conv2d

gac::MSDeformAttn
    → gac.ms_deform_attn
```

自定义映射位于 [`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L580)。当前只接受显式注册的自定义算子契约。

### 7.5 可选输入

ONNX 使用空字符串表示某个可选输入不存在。导入器会移除空输入，同时使用：

```text
gac.input_positions
```

保存剩余 operand 原来处于第几个输入槽位。这样后续 Pass 才能区分：

```text
第 1 个参数存在、第 2 个参数缺失、第 3 个参数存在
```

### 7.6 多输出算子

`gac.split` 使用一个 Operation 和多个 `OpResult` 保存 ONNX Split 的多个输出。后续设备合法化阶段可以继续把静态 Split 转换成多个 Slice。

当前除 `Split` 外，已注册的导入路径一般只接受一个非空结果。因此 BatchNormalization、MaxPool 等算子的多输出变体仍可能被拒绝。

## 8. 图优化产生的新 GAC Operation

下面这些 Operation 不是从某个 ONNX 节点直接导入的，而是由后续图优化 Pass 创建：

```text
gac.matmul + gac.add
    → gac.matmul_bias

gac.matmul + gac.add + gac.relu
    → gac.matmul_bias_relu

gac.conv2d + gac.relu
    → gac.conv2d_relu
```

其他编译器内部 Operation 还包括：

```text
gac.add_relu
gac.matmul_bias_gelu
gac.ms_deform_attn_offsets
```

这些 Operation 的定义可见 [`Dialect.h`](../native/include/gac/Dialect.h#L94)，融合逻辑可见 [`GraphPasses.cpp`](../native/lib/GraphPasses.cpp#L273)。

图优化前后节点数量和拓扑结构可以改变，但必须保持模型的数学语义。

## 9. 当前不能直接转换的情况

下面的情况会在严格模式中报错，或者只在 inspection 模式中保存为 `gac.unsupported`：

- 映射表之外的算子，例如通用 `Loop`、`Scan`、`ConvTranspose`、`TopK` 等；
- 通用 ONNX `If`；
- 未注册的自定义 domain；
- Graph、SparseTensor、TypeProto 等复杂 Attribute；
- 标准 opset 13～18 之外的节点；
- 多输出算子，目前除 `Split` 外一般只允许一个非空输出；
- `Conv` 和 `MaxPool` 当前只支持二维 NCHW；
- `Max` 当前只支持两个输入；
- 训练模式算子。

`gac.unsupported` 只用于查看和诊断完整导入图，不能进入后端 Codegen。

## 10. GAC Dialect 不只是文本名字

`gac.matmul` 表达完整的矩阵乘数学语义，但此时还没有展开成逐元素乘加循环。

它定义的是：

```text
这是一个矩阵乘
输入和输出必须满足什么类型
Shape 如何推导
哪些属性合法
哪些 Pass 可以识别和改写它
以后应该进入哪条 Lowering
```

GAC Operation 定义在 [`Dialect.h`](../native/include/gac/Dialect.h#L27)，注册在 [`Dialect.cpp`](../native/lib/Dialect.cpp#L14)，类型和 Shape 验证入口在 [`Dialect.cpp`](../native/lib/Dialect.cpp#L507)。

完整 Lowering 关系是：

```text
gac.matmul
    ↓ 图优化和算子融合
结构化循环、算术和目标相关 IR
    ↓
Tiling、Schedule、线程映射、load/store、同步
    ↓
LLVM / Vortex / CUDA 相关 Dialect
    ↓
目标机器代码
```

因此，`gac.matmul` 既不是一个空名字，也不是最终机器实现。它是编译器中间层对矩阵乘的**语义契约**。

## 11. 总结

整个导入过程可以概括为：

```text
ONNX NodeProto
+ 输入输出字符串
+ ValueInfoProto
+ TensorProto
+ AttributeProto
+ domain/opset
        ↓
GAC Operation
+ SSA operands/results
+ MLIR TensorType
+ MLIR Attribute
+ Location
        ↓
图优化与算子融合
        ↓
Tiling / Schedule / Buffer
        ↓
LLVM / Vortex / CUDA Lowering
```

最需要记住的一句话是：

> **ONNX 用名称描述模型计算图；GAC MLIR 用带类型的 SSA 对象描述可验证、可改写、可 Lowering 的程序。**

## 12. 关键源码入口

- ONNX 导入总体说明：[`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L22)
- 创建 MLIR Operation：[`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L439)
- ONNX 算子映射表：[`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L523)
- 单节点导入过程：[`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L632)
- `ModuleOp` 和 `func.func` 构造：[`OnnxImporter.cpp`](../native/lib/OnnxImporter.cpp#L831)
- GAC Operation 定义：[`Dialect.h`](../native/include/gac/Dialect.h#L27)
- GAC Dialect 注册：[`Dialect.cpp`](../native/lib/Dialect.cpp#L14)
- GAC 类型和 Shape 验证：[`Dialect.cpp`](../native/lib/Dialect.cpp#L507)
- 图优化和融合：[`GraphPasses.cpp`](../native/lib/GraphPasses.cpp#L273)
