# Triton / Vortex Softmax 接入验证

日期：2026-10-22  
范围：GAC 的 ONNX → MLIR Linalg → TTIR / native Vortex 选择、Manifest、Runtime ABI，以及 Triton/Vortex C++ Lowering → VOLT → SimX。这里只记录 correctness 路径，不代表硬件并行性能。

结论：**GAC 完整集成 48/48 通过；Triton/Vortex 后端 Softmax 专项 20/20 通过。**

## 首版合同

- 选择范围：静态 rank-1、FP32、axis 0、`1 <= width <= 1024`。
- 逻辑 tile：`max(16, nextPowerOfTwo(width))`，`grid_size=1`。
- TTIR：masked load 的 `other=-Inf`，随后执行 `maxnumf reduce → exp(x-max) → addf reduce → div → masked store`。
- Kernel 参数：`(input f32*, output f32*, N i32)`。
- Vortex 参数缓冲区：输入地址、输出地址、`N` 各占一个 8 字节槽；第三槽低 32 位为 N，高 32 位为零。
- Manifest ABI：`vortex-u64-pointers-i32-n`。
- rank-2、非末轴和 width 大于 1024 的形式明确保留 GAC native Vortex 路径。

## 编译链与中间表示

GAC C++ 编译器和 Runtime 已重新构建成功：

```text
[ 93%] Built target GACCompiler
[100%] Built target gac-compile
[100%] Built target gac-vortex-run
```

width 19 的 ONNX Softmax 已生成：

- [`static-linalg.mlir`](static-linalg.mlir)：`gac.softmax` 已转换为带 `gac.linalg_role = "softmax"` 的 `linalg.softmax`。
- [`static-plan.json`](static-plan.json)：`kind=softmax`、`width=n=19`、`tile=32`、`grid_size=1` 和独立 ABI 均已写入。
- [`static-softmax.ttir`](static-softmax.ttir)：函数签名保留真实的第三个 `i32 N`，含两个标准 `tt.reduce` Region、`arith.maxnumf`、`math.exp`、`arith.addf` 和 masked store。

静态 TTIR 的核心顺序为：

```text
make_range → cmp(index, N) → masked load(-Inf)
           → tt.reduce(maxnumf)
           → exp(x-max)
           → tt.reduce(addf)
           → numerator/denominator
           → masked store
```

## native fallback 已执行

rank-2 `[3,19]`、axis 1 的 Softmax 没有伪装成当前 Triton 合同，而是进入 native Linalg-to-SCF 路径：

- [`native-rank2-plan.json`](native-rank2-plan.json)：Dispatch 含明确 `triton_exclusion`。
- [`native-rank2-linalg.mlir`](native-rank2-linalg.mlir)：保留真实 `linalg.softmax dimension(1)`。
- [`native-rank2-runtime-report.json`](native-rank2-runtime-report.json)：Vortex SimX 执行，无 host operator fallback。

与独立 NumPy 稳定 Softmax 的最大绝对误差为：

```text
1.4901161193847656e-08
```

## GAC 完整集成套件

[`run_triton_vortex_tests.py`](../../../../native/tests/run_triton_vortex_tests.py) 的完整结果见 [`validation.json`](validation.json)，共 **48/48 通过**。Softmax 部分增加 19 项检查：

- 8 个 Triton Softmax 数值用例：width 1、普通、非二次幂、极大有限值、NaN、`+Inf`、全 `-Inf`、部分 `-Inf`。
- 3 个明确 native 回退：rank-2 last-axis、rank-2 axis 0、width 1025。
- 8 个 Manifest 拒绝：篡改 grid、width、N、tile、ABI、输入 shape、输出 shape、输出 bytes；构建驱动和 Runtime 都必须拒绝。

八类篡改已经独立执行构建驱动与 Runtime 两道检查，结果为 **16/16 通过**；Runtime 均在打开设备前拒绝。完整错误见 [`static-rejection-validation.json`](static-rejection-validation.json)，对应纯文本日志为 `reject-*.log`。

GAC 的 Triton Softmax 已实际在 SimX 通过 width 1、16、19、极大有限值、NaN、`+Inf`、全 `-Inf` 与部分 `-Inf`。width 1 的实际编译计划、TTIR 和 Runtime 记录见 [`width1-plan.json`](width1-plan.json)、[`width1-softmax.ttir`](width1-softmax.ttir) 和 [`width1-runtime-report.json`](width1-runtime-report.json)；非二次幂 width 19 的对应证据见 [`tail19-plan.json`](tail19-plan.json)、[`tail19-softmax.ttir`](tail19-softmax.ttir) 和 [`tail19-runtime-report.json`](tail19-runtime-report.json)。

rank-2 last-axis 与 axis-0 Softmax 都由 native Vortex 路径在 SimX 完成数值验证。width 1025 会正确选择 native fallback 并生成 vxbin，证据见 [`native-too-wide-plan.json`](native-too-wide-plan.json) 与 [`native-too-wide-compile-report.json`](native-too-wide-compile-report.json)。该标量 fallback 在当前小型 SimX 上 10 分钟仍未 retire，因此此项记录为 **compile/selection-only**，原始诊断见 [`native-too-wide-10-minute-timeout.log`](native-too-wide-10-minute-timeout.log)；没有把超时写成设备执行成功。

## Triton/Vortex 后端边界验证

独立后端专项 [`backend-softmax-validation.json`](backend-softmax-validation.json) 为 **20/20 通过**，其中 4 项是预期拒绝。真实设备正例覆盖：

- `1/16`、`16/16`、`13/16`、`23/32`、`32/32`、`37/64`、`64/64`；
- `65/128`、`256/256`；
- `1023/1024` 和 `1024/1024`；
- 极大正负有限值、NaN、`+Inf`、全 `-Inf` 和有限值混合 `-Inf`。

有限输出相对 FP32 稳定 Softmax 参考的最大绝对误差为 `2.9802322387695312e-08`。每个执行正例都从 Triton 编译缓存读取最终 LLVM IR，并自动断言 `softmax_f32` 恰好有一次 `llvm.riscv.vx.split.i32` call 和一次 `llvm.riscv.vx.join.i32` call。BLOCK 16/32/64 的最终 Vortex 反汇编还分别确认只有一对匹配的 `vx_split`/`vx_join`，见 [`backend-softmax-machine.txt`](backend-softmax-machine.txt)。

这仍是 CTA leader 串行实现：输出 buffer 暂作指数 scratch，随后累加分母并原位归一化。没有树形归约、warp shuffle、LMEM、DXA、TCU 或多行 row-wise Softmax，因此不能据此声明性能或 LayerNorm 已完成。

## 代码入口

| 层次 | 文件与入口 |
|---|---|
| ONNX Softmax → Linalg | `native/lib/LinalgLowering.cpp`：`lowerSoftmax()` |
| 合同选择和 TTIR | `native/lib/TritonBridge.cpp`：`selectTritonSoftmax()` / `emitTritonSoftmax()` |
| Dispatch 与 Manifest | `native/lib/DeviceLowering.cpp` |
| 构建期复核 | `native/tools/compile-triton-vortex.py`：`triton_dispatches()` |
| Runtime 复核 | `native/runtime/dispatch-plan.h`：`validateTritonSchedule()` |
| 参数槽构造 | `native/runtime/vortex-run.cpp` |
| 集成测试 | `native/tests/run_triton_vortex_tests.py` |

验证时的源码 SHA-256 见 [`source-sha256.txt`](source-sha256.txt)。其中集成测试脚本记录的是 48 项 Softmax 验证实际使用的快照；工作区脚本随后已进入独立的 LayerNorm 开发阶段。
