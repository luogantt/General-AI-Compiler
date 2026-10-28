# Triton / Vortex row-wise Softmax 验证记录

本目录保存 GAC 将 ONNX `Softmax` 合法化为 Linalg，并把符合合同的最后一轴
Softmax 交给 Triton/Vortex 后端的验证证据。验证目标是数值正确性、原子选择、
ABI 一致性和失败边界；这些结果不是硬件性能数据。

## 结论

2026-10-22 的完整 GAC 集成检查为 **92/92 通过**，其中 30 项直接覆盖
Softmax。后端独立检查同时得到 Softmax 专项 **28/28**、完整后端回归
**90/90**。

```text
ONNX Softmax(axis = last)
        ↓ C++ importer / legalize-to-Linalg
linalg.softmax
        ↓ 静态 shape 与物理 Buffer 合同检查
一个 row-wise TTIR kernel
        ↓ Triton/Vortex C++ lowering
VOLT → Vortex vxbin
        ↓ GAC Runtime
SimX 设备执行
```

当前 Triton 合同严格限制为：

- 静态、无编码 FP32 张量，`rank >= 1`；
- 归一化轴必须是最后一轴，负轴先规范化；
- `1 <= width <= 1024`；
- `rows = product(shape[:-1])`，rank-1 时 `rows = 1`；
- `rows * width <= INT32_MAX`，所有乘积使用 checked 整数计算；
- 输入与输出 shape 完全相同、物理 Buffer 不别名；
- dense View 的物理根 shape 必须与逻辑参数 shape 完全一致。

不满足这些条件时，Softmax 完整保留为一个 native Vortex dispatch；不会只把
内部步骤交给 Triton，也不会退到 CPU 或 CUDA。

## TTIR 与启动合同

每个 `program_id(0)` 处理一行：

```text
pid      = program_id(0)
rowBase  = pid * N
indices  = rowBase + [0, 1, ..., tile-1]
mask     = offsets < N

x        = masked_load(A[indices], other = -Inf)
rowMax   = reduce_maxnumf(x)
e        = exp(x - rowMax)
rowSum   = reduce_addf(e)
y        = e / rowSum
masked_store(Y[indices], y)
```

`tile=max(16,nextPowerOfTwo(width))`。Manifest 明确记录 `rows`、`width`、
`N`、`tile` 和 `grid_size=rows`；ABI 为：

```text
A pointer / Y pointer / N:i32
```

Runtime 为三个参数各保留 8 字节槽，`N` 使用第三个槽的低 32 位。Python
构建驱动和 C++ Runtime 分别从张量 shape 重算 rows、width、元素数、字节数、
tile 和 grid，并在打开设备前拒绝不一致的计划。

## 数值与边界结果

| 用例 | 路径 | 验证 | 最大绝对误差 |
|---|---:|---|---:|
| 旧 rank-1，shape `[19]` | 1 Triton / 0 native | SimX | `1.4901161193847656e-8` |
| rank-2，shape `[3,19]` | 1 Triton / 0 native | SimX | `1.4901161193847656e-8` |
| rank-3，shape `[2,3,19]` | 1 Triton / 0 native | SimX | `2.2351741790771484e-8` |
| BEV width 256，shape `[1,2,256]` | 1 Triton / 0 native | SimX | `9.313225746154785e-10` |
| width 1，shape `[2,3,1]` | 1 Triton / 0 native | SimX | `0` |
| width 1024，shape `[1,1024]` | 1 Triton / 0 native | 完整编译和选择 | 未执行 |
| 非末轴、width 1025、shape-changing View | 0 Triton / 1 native | 小形状完成 SimX；超宽为编译/选择验证 | 见 `validation.json` |
| 非末轴 Softmax + 独立 Add | 1 Triton sibling / 1 native Softmax | SimX | Softmax `5.960464477539063e-8`；Add `0` |

正例还检查：输入文件执行前后 SHA-256 不变、输出字节数精确、每行有限结果之和
为 1，以及每个 Triton/Vortex kernel 的 Vortex MLIR 恰好包含一对
`vx_split` / `vx_join`。

`selector-contract.log` 直接覆盖 rank-1、rank-3、width 1024，以及非末轴、
width 1025、prefix rows 溢出和总元素数溢出的选择边界。

12 组 Manifest 变异覆盖 grid、rows、缺失 rows、`rows*width` 溢出、width、
N、tile、ABI、输入 prefix shape、输出 shape、输出 bytes 和输出别名。Python
构建驱动与 C++ Runtime 均按各自层级拒绝这些错误。

## 复现命令

先构建编译器、Runtime 和选择合同测试：

```bash
cmake --build "$GAC_NATIVE_BUILD" \
  --target gac-compile gac-vortex-run \
           gac-triton-layernorm-selector-contract \
           gac-triton-softmax-selector-contract -j4
```

确认使用本次后端二进制，换用全新的 Triton cache，然后运行完整集成：

```bash
export TRITON_VORTEX_OPT=/path/to/triton/third_party/vortex/build/triton-vortex-opt
export PYTHONPATH=/path/to/triton/third_party/vortex/python:$PWD/src:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/gac-triton-rowsoftmax-cache
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime:${LD_LIBRARY_PATH:-}"

python native/tests/run_triton_vortex_tests.py \
  --compiler "$GAC_NATIVE_BUILD/gac-compile" \
  --runtime "$GAC_NATIVE_BUILD/gac-vortex-run" \
  --output /tmp/gac-triton-rowsoftmax-validation
```

本次运行使用的工具 SHA-256、源码 SHA-256 与原始输出位置见
[`evidence.json`](evidence.json)。

## 证据文件

- [`validation.json`](validation.json)：92 项逐项结果。
- [`legacy-rank1`](legacy-rank1/)：旧 rank-1 路径在统一 row-wise TTIR 下的证据。
- [`rank2-tail-19`](rank2-tail-19/) 与 [`rank3-tail-19`](rank3-tail-19/)：
  多行、非二次幂尾 lane 的 Linalg、TTIR、Vortex MLIR、plan 与 Runtime 报告。
- [`width-256`](width-256/) 与 [`width-1`](width-1/)：BEV 关键宽度和最小宽度。
- [`width-1024`](width-1024/)：合同上界的完整编译和选择证据。
- [`native-axis0`](native-axis0/)、[`native-too-wide`](native-too-wide/) 与
  [`native-shape-view`](native-shape-view/)：完整 native 回退。
- [`mixed-sibling`](mixed-sibling/)：回退不会阻止同图独立 pointwise Triton
  dispatch。
- [`rejections`](rejections/)：12 组非法 Manifest 与 Runtime 拒绝日志。
- [`selector-contract.log`](selector-contract.log)：shape、axis 与有符号 32 位
  索引边界。

width 1024 只做完整编译与选择验证，避免把小型 SimX 上的长串行运行误当作
性能结论。当前 Softmax 后端是正确性模板；本目录不声称性能提升，也不代表
完整 BEVFormer 已全部走 Triton。
