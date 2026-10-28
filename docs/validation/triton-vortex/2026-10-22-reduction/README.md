# Triton/Vortex ReduceSum 验证

2026-10-22 在 Vortex **SimX** 上完成首个真实 reduction 路径验证：

```text
ONNX ReduceSum
  → C++ OnnxImporter
  → Linalg sum-reduction
  → 标准 Triton 3.8 tt.reduce
  → Triton/Vortex C++ Lowering
  → LLVM IR
  → VOLT → Vortex vxbin
  → GAC Runtime → SimX
```

完整集成套件 **29/29 通过**。其中包含 18 个执行用例和 11 个非法 Manifest 拒绝用例；总结果见 [`validation.json`](validation.json)。这些结果验证编译链、ABI、数值和失败边界，不代表真实芯片性能。

独立 Triton/Vortex backend 回归也已通过：[`backend-reduction-validation.json`](backend-reduction-validation.json) 为 reduction 专项 **12/12**，[`backend-full-validation.json`](backend-full-validation.json) 为完整 backend **56/56**。

本次验证所对应的 GAC 与 backend 关键源码校验值记录在 [`source-sha256.txt`](source-sha256.txt)。

## ReduceSum 正例

| 用例 | ONNX 形状 | TTIR Tile | 后端 | 最大绝对误差 |
|---|---:|---:|---|---:|
| `triton_reduce_sum_scalar` | `[37] → []` | 64 | Triton/Vortex | 0 |
| `triton_reduce_sum_singleton` | `[19] → [1]` | 32 | Triton/Vortex | 0 |

37 和 19 都不是二次幂。编译器把 Tile 向上补齐为 64 和 32，用掩码限制有效元素，并让越界 Lane 读取显式的 `0.0`。输出通过标量 `tt.store` 写回。

标量用例的关键证据：

- [`02-linalg.mlir`](scalar/02-linalg.mlir)：带 `gac.linalg_role = "sum-reduction"` 的 Linalg 归约。
- [`gac_op_0.ttir`](scalar/gac_op_0.ttir)：GAC C++ 生成的标准 `tt.reduce`、`arith.addf`、`tt.reduce.return`。
- [`gac_op_0.triton.ttir`](scalar/gac_op_0.triton.ttir)：Triton 3.8 解析后的 TTIR。
- [`gac_op_0.vxmlir`](scalar/gac_op_0.vxmlir) 与 [`gac_op_0.ll`](scalar/gac_op_0.ll)：Vortex LLVM Dialect 和 LLVM IR。
- [`plan.json`](scalar/plan.json)：`kind=reduction`、`width=37`、`tile=64`、`grid_size=1` 和双指针 ABI。
- [`runtime-report.json`](scalar/runtime-report.json)：实际启动记录为 `backend=triton-vortex`，且无 Host 算子回退。
- [`kernel.dump`](scalar/kernel.dump)：最终 Vortex 程序反汇编。

保留单元素维度的用例证据位于 [`singleton/`](singleton/)。

## 选择合同

首版只选择以下形式：

- 一个静态、正长度、rank-1 FP32 输入；
- 输出是 rank-0 FP32 标量或 shape `[1]` 的 FP32 张量；
- 完整 axis-0 求和；
- Linalg 初始化值是 FP32 零；
- Region 中唯一 payload 是规范的 `arith.addf(input, accumulator)`；
- 唯一直接输入，不把其他生产者融合进 reduction；
- `width ≤ 1024`，`tile = max(16, next_power_of_two(width))`，`grid_size = 1`。

构建驱动和 Runtime 会分别重新核对张量形状、字节数、width、tile、grid、物理线程配置和 ABI。三份篡改后的 Manifest 都在设备初始化前被拒绝，日志位于 [`rejections/`](rejections/)。

## 明确回退

不符合合同的归约仍由 GAC 原生 Vortex 路径执行：

- [`fallback-rank2/plan.json`](fallback-rank2/plan.json)：`[3,17]` 沿 axis 1 求和，原因明确记录为首版只接受 rank-1 输入及标量/单元素输出。
- [`fallback-too-wide/plan.json`](fallback-too-wide/plan.json)：width 1025，原因明确记录为首版上限 1024。

两个回退用例也在 SimX 上与 ONNX ReferenceEvaluator 对齐。该路径没有宣称已经支持 Softmax、LayerNorm 或一般多轴归约。

## 性能边界

当前 Triton/Vortex backend 把 `tt.reduce` 实现为正确性优先的标量 FPU 归约循环。它还不是树形归约、Warp 协作归约、LMEM/DXA 双缓冲或 TCU 实现。SimX 报告中的主机墙钟时间包含模拟器、文件 I/O 和 Runtime 开销，不能当作芯片推理延时。
