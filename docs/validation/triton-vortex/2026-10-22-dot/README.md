# Triton / Vortex `tt.dot` 验证

> **历史首版记录：** 本目录冻结保存最初的 rank-2、`BK=K`、22/22
> 正确性基线，不再随当前实现改写。当前实现已经支持严格的 batched
> matched-prefix/shared-RHS 合同、K padding 和更完整的 Manifest/Runtime
> 校验；最终 118/118 GAC 集成与后端 20/20、108/108 证据见
> [batched MatMul 验证记录](../2026-10-22-batched-matmul/README.md)。

2026-10-22，GAC 的首版二维 FP32 MatMul 路径在 Vortex SimX 上完成 **22/22 项集成检查**。完整结果见 [validation.json](validation.json)。测试链路为：

```text
ONNX MatMul
  → GAC C++ MLIR / Linalg
  → 标准 TTIR tt.dot
  → Triton/Vortex C++ Lowering
  → LLVM IR
  → VOLT / vxbin
  → GAC Runtime / Vortex SimX
```

测试以 ONNX ReferenceEvaluator 为独立参考值。两个 `tt.dot` 用例都由设备路径执行，`host_operator_fallback=false`。

| 用例 | 形状 | Tile / grid | 最大绝对误差 |
|---|---|---|---:|
| aligned | `[16,16] × [16,16] → [16,16]` | `16×16×16` / 1 | `4.76837158203125e-7` |
| M/N tail | `[17,32] × [32,19] → [17,19]` | `16×16×32` / 4 | `9.5367431640625e-7` |

原始 TTIR 中只有一个标准 `"tt.dot"`。Triton 解析后的 TTIR 仍包含 `tt.dot`；最终 LLVM IR 中可以看到逐元素 FPU `fmul` / `fadd` 累加。M/N 尾块通过加载和写回 mask 保护。

## 选择与拒绝合同

首版只选择静态、无编码、行主序的 rank-2 FP32 MatMul：

```text
A[M,K] × B[K,N] → C[M,N]
BM = BN = 16
BK = K
K >= 16 且 K % 16 == 0
参数顺序：A 指针、B 指针、C 指针
```

`K=15` 的用例明确回退到 `linalg-to-scf`，见 [fallback-k15-plan.json](fallback-k15-plan.json)。Runtime 在初始化设备前重新校验 MatMul 形状、Tile、grid、kind 和 ABI；篡改合同的四个用例全部被拒绝：

- [错误 grid](rejections/invalid_dot_grid.log)
- [错误 K](rejections/invalid_dot_k.log)
- [错误 Tile](rejections/invalid_dot_tile.log)
- [未知 kind](rejections/invalid_dot_kind.log)

## 证据文件

每个 MatMul 用例保留以下纯文本产物：

| 文件 | 含义 |
|---|---|
| `plan.json` | 张量形状、参数顺序、Tile、grid 与 Runtime ABI |
| `gac_op_0.ttir` | GAC C++ 直接生成的原始 TTIR |
| `gac_op_0.triton.ttir` | Triton 解析后的标准 TTIR |
| `gac_op_0.ll` | Triton/Vortex Lowering 产生的 LLVM IR |
| `runtime-report.json` | SimX 启动配置及执行路径 |

对应目录：[aligned](aligned/) 和 [tail](tail/)。仓库没有复制模型、输入输出、缓存、ELF 或 vxbin 等二进制文件。

## 结论边界

这是 **FP32 标量 FPU correctness baseline**。它没有使用 LMEM、DXA、TCU，也没有证明性能提升。`runtime-report.json` 中的时间是 SimX 的主机墙钟时间，不能当作芯片延时或 FPS。

这次验证覆盖小型算子与混合 Dispatch 图，**不是完整 BEVFormer 推理验证**。
本目录生成时 Batch MatMul 尚未实现；它已在后续严格合同中完成。Bias/ReLU
epilogue、动态 Shape、LMEM 分块、双缓冲和 TCU 映射仍不属于本历史基线。
