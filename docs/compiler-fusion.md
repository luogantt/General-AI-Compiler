# GAC 算子融合：规则、边界与验证

融合分为两层：**图层证明算子可以组合；设备层决定组合后如何计算。** 两层都在 C++ MLIR 路径里执行。

```text
ONNX → GAC MLIR
          ↓ gac-fuse：语义与使用关系检查
    融合后的 GAC Operation
          ↓
  Linalg 路径                     legacy 设备模板路径
  结构化循环、索引与归约            根据目标与资源选择计算模板
          ↓                             ↓
  同坐标 producer/consumer 合并     在 reduction 后执行 bias/ReLU
          └──────────→ 设备 IR → LLVM → 机器码
```

## 代码入口

| 位置 | 职责 |
|---|---|
| [`native/include/gac/Fusion.h`](../native/include/gac/Fusion.h) | 图融合规则的统一注册接口 `populateGraphFusionPatterns` |
| [`native/lib/fusion/GraphFusionPatterns.cpp`](../native/lib/fusion/GraphFusionPatterns.cpp) | 模式匹配、合法性证明、创建融合 Operation |
| [`native/lib/GraphPasses.cpp`](../native/lib/GraphPasses.cpp) | `gac-fuse` Pass 入口，调用 MLIR 贪心重写驱动 |
| [`native/lib/LinalgDevice.cpp`](../native/lib/LinalgDevice.cpp) | `isFusedLinalgProducer` 判断结构化计算是否可在同一输出坐标内执行 |
| [`native/tests/run_fusion_contract_tests.py`](../native/tests/run_fusion_contract_tests.py) | 新增 Conv 通道 bias 融合的数值与拒绝条件验证 |
| [`native/tests/generate_matmul_bias_models.py`](../native/tests/generate_matmul_bias_models.py) | MatMul 广播、消费者数量、FP16 舍入等既有独立参考样例 |
| [`native/tests/run_msda_offsets_tests.py`](../native/tests/run_msda_offsets_tests.py) | MSDeformAttn 坐标链融合的既有验证 |

新增规则放在 `lib/fusion/`，通过 `populateGraphFusionPatterns` 注册。不要在模型导入器中按网络节点名称进行融合，也不要让 Python 改写生产计算图。

## 已实现规则

| 原始计算 | 融合结果 | 主要限制 |
|---|---|---|
| `MatMul → Add` | `gac.matmul_bias` | FP32；MatMul 单使用；bias 不扩张 MatMul 输出域；保留 Add 左右顺序 |
| `MatMul → Add → ReLU` | `gac.matmul_bias_relu` | 同上；被删除的中间结果不能还有其他消费者 |
| `Conv → ReLU` | `gac.conv2d_relu` | FP32；保留 Conv 原有 stride、padding、dilation、groups 及可选 bias |
| `Add → ReLU` | `gac.add_relu` | FP32；Add 结果单使用，保持广播规则 |
| **`Conv → Add(channel bias) [→ ReLU]`** | `gac.conv2d [或 conv2d_relu]` 的 bias 参数 | 本次新增；具体条件见下节 |
| `Div → 可选 view → Add → 可选 view → MSDeformAttn` | `gac.ms_deform_attn_offsets` | FP32；保持除法、参考点相加、reshape 与广播含义；不跨越 Cast/Transpose |

`Canonicalize`、`CSE`、常量折叠与上述融合是不同优化。已有这些 Pass，不代表任意 Attention、Conv 链或完整 Encoder 都能自动融合成一个 Kernel。

## 新增：显式通道 bias 并入 Conv

```text
原来：Conv(X,W) → C[N,Cout,H,W] → Add(C,B) → ReLU → Y
现在：ConvReLU(X,W,reshape(B,[Cout])) → Y
```

必须同时满足：

1. Conv 原本没有 bias，输出为静态、未编码的 FP32 NCHW 张量。
2. Conv 输出只被这个 Add 使用，且 Add 不扩大输出形状。
3. Add 的左边是 Conv 结果，右边 bias 的形状严格为 `[Cout,1,1]` 或 `[1,Cout,1,1]`。
4. 可选 ReLU 的输入也满足单使用条件。

计算顺序仍是：**完成所有卷积乘加 → 加通道 bias → 可选 ReLU。** bias 的 reshape 只是连续数据的逻辑视图，运行时复用原指针。

以下情况保留原计算：

- `[Cout]` 不能直接视为 NCHW 的通道 bias。ONNX 从右往左对齐广播，这个形状实际对应 **W 轴**；即使 `Cout == W` 也不能混淆。
- 已有 Conv bias 再接 Add，不把两个 bias 提前相加。浮点数的结合顺序会改变结果，例如 `(1e10 - 1e10) + 1` 与 `1e10 + (-1e10 + 1)` 不同。
- FP16 不删除 Conv 输出的舍入边界。FP32 累加中间值先转 FP16、再加 bias，可能不同于先在 FP32 加 bias、最后转 FP16。
- Conv 中间结果是公开输出或被多处读取时，保留它。
- 逐像素 residual、逐 batch bias、反向 `Add(B,C)` 暂不吸收进 Conv。反向 Add 保留，是为了不在这个规则里引入浮点操作数交换。

这条规则让 legacy 的 Conv 模板能直接处理原来分开的通道 bias。Linalg 路径原本就能合并部分同坐标 epilogue，因此不能把新增图融合一概当作 Linalg 性能加速。

## Linalg 层目前融合到什么程度

当前检查单输出、单消费者、纯张量语义、静态同形状、无外部捕获和无副作用。普通消费者还需要全 parallel 迭代器，其输入索引必须与输出索引一致。零初始化有单独的受限规则。FP16 的中间转换仍保留。

因此它能把合适的 reduction 后 epilogue 放在同一次计算里；尚不等于通用的跨 Tile 融合搜索，也没有融合收益模型。广播、转置、跨输出复用和复杂归约边界仍需要专门方案。

## 怎么验证

在具备 LLVM/MLIR 20.1.8、ONNX、ONNX Runtime、NumPy 的 Linux 环境运行：

```bash
python native/tests/run_fusion_contract_tests.py \
  --compiler /path/to/gac-compile \
  --llvm-bin /path/to/llvm-build/bin \
  --fixtures build/fusion-contracts
```

脚本生成 13 组模型。每组分别执行 **legacy/Linalg × 图优化关闭/开启** 共四条路径：编译 LLVM IR、验证 LLVM IR、生成 CPU 动态库、调用生成的机器码，并与独立 ONNX 实现比较。还检查优化前后的有限数值完全一致、NaN/正负 Inf 分布一致、公开输出保留与 arena 活跃区间不重叠。

报告写入指定目录的 `results.json`，记录编译器 SHA-256、每条路径的 dispatch 数和 arena 大小。这是正确性与资源计划验证；实际 CUDA/Vortex 的性能收益需要另外实测。

2026-10-21 的实测：**13/13 组通过，52 条编译执行路径通过**。报告见 [fusion-contracts.json](validation/templates/2026-10-21/fusion-contracts.json)。例如：

| 样例 | legacy dispatch | Linalg dispatch |
|---|---:|---:|
| Conv + 运行时通道 bias + ReLU | 3 → 1 | 3 → 1 |
| Conv + 通道 bias，无激活 | 2 → 1 | 2 → 1 |
| 带 group、padding、stride、dilation 的 Conv 链 | 3 → 1 | 2 → 2，仍有独立 pad |
| FP16 舍入边界 | 3 → 3 | 3 → 3 |

“不吸收通道 bias”的样例仍可能使用原有的 `Add+ReLU` 融合，因此 dispatch 从 3 变 2 不表示忽略了拒绝条件。表中是小模型的实际执行计划，不能外推为 BEVFormer 整网的加速比例。

## 尚未承诺的能力

没有完整 FlashAttention 融合、通用多层 Conv 融合、任意 Residual+LayerNorm 专用融合模板，也没有基于实测代价的全图融合搜索。新增规则首先证明数学与内存语义不变，再验证目标模板是否减少 dispatch/搬运；不能只用“节点减少”推断速度提升。
