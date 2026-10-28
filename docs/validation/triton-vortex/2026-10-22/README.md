# GAC → Triton/Vortex 集成验证

2026-10-22，分支 `lg/bevformer-triton-vortex`。**11 项设备执行检查、4 项非法清单拒绝检查通过。**
原始结果见 [validation.json](validation.json)，复现入口是 [run_triton_vortex_tests.py](../../../../native/tests/run_triton_vortex_tests.py)。

环境：WSL Ubuntu 24.04、Triton 3.8.0、VOLT LLVM/MLIR 20.1.8、RV32 Vortex SimX；1 核、每核 4 Warp、每 Warp 4 Lane、16 KiB LMEM。本次逐元素内核没有使用 LMEM。

## 实际跑了什么

| 用例 | 验证内容 |
|---|---|
| 1 / 17 / 1003 元素加法 | 单元素轴、尾部 mask、多 Program |
| FP32 / i32 的 Add → Mul → ReLU | 三个算子进入一个 Triton 内核；逻辑三维张量展平；两个 Warp |
| A + A | 输入参数去重 |
| 含 NaN / ±Inf 的 ReLU | 特殊浮点值保持与参考一致 |
| Reshape → Add | View 共享原始设备存储，无额外拷贝 |
| 广播 Add | 明确保留原生 Vortex 实现 |
| Add → MatMul → Mul → ReLU | 两条编译路径共用一个设备模块、依赖计划和 Buffer |
| MatMul → ReLU，内部 M 不输出 | 含归约的融合块整体保留原生实现，不丢失已融合 producer |
| 错误 grid / BLOCK / ABI / 缺失元数据 | 打开设备前拒绝，避免错误线程映射 |

参考结果由 ONNX ReferenceEvaluator 计算；生成的程序由 `gac-vortex-run` 在 SimX 执行。测试路径没有 CPU/CUDA 算子回退。整数精确相等；浮点采用 `atol=rtol=2e-5`，显式检查 NaN/Inf。

## 混合执行实例

```text
A[5,7] + B[5,7]                 Triton gac_op_0
          ↓ S[5,7]
        S × W[7,9]              原生 Vortex gac_op_1
          ↓ M[5,9]
       ReLU(M ⊙ R[5,9])         Triton gac_op_2，Mul/ReLU 融合
          ↓ Y[5,9]
```

三个内核链接到同一 `kernel.vxbin`。M、Y 都作为模型输出检查，最大绝对误差分别为 `2.3841858e-7`、`1.1920929e-7`。

共享 arena 为 **384 B**，按 64 B 对齐分配：

| 张量 | 有效字节 | arena offset | 生命周期 |
|---|---:|---:|---|
| S | 140 | 0 | Add 写入，MatMul 读取后结束 |
| M | 180 | 192 | MatMul 写入，后续读取，保留为模型输出 |
| Y | 180 | 0 | 复用已经不再使用的 S 所在空间 |

这里的 Buffer 是设备全局内存中的分配区间；它与 LMEM 双缓冲是不同机制。中间张量不需要传回主机。

可以顺序查看真实产物：

1. [导入后的 GAC MLIR](mixed-matmul/01-import.mlir) → [图优化后](mixed-matmul/02-optimized.mlir) → [Linalg](mixed-matmul/02-linalg.mlir)。
2. [Dispatch 和 Buffer 计划](mixed-matmul/plan.json)：每项包含选中的路径；原生项记录 `triton_exclusion`。
3. [Triton 加法 TTIR](mixed-matmul/gac_op_0.ttir)、[Mul/ReLU TTIR](mixed-matmul/gac_op_2.ttir)、[原生 MatMul LLVM Dialect](mixed-matmul/native.mlir)。
4. [Triton 后端生成的 LLVM Dialect](mixed-matmul/gac_op_2.vxmlir) → [共同链接的 LLVM IR](mixed-matmul/kernel.ll) → [VOLT 机器码反汇编](mixed-matmul/kernel.dump)。
5. [Runtime 执行报告](mixed-matmul/runtime-report.json)、[编译阶段报告](mixed-matmul/compile-report.json)。

## 范围与复现

本次证明的是小型 ONNX 图的完整编译执行链路，**不是完整 BEVFormer 的 Triton 验收，也没有证明性能提升**。报告中的耗时是包含模拟器和主机开销的墙钟时间，不能换算成芯片 FPS。

[安装与运行说明](../../../../native/README.triton-vortex.md)列出构建环境和命令。[source-sha256.json](source-sha256.json)记录本次源码文件校验值。
另存有开发期间独立 C++ 清单解析检查的 [33 项结果](runtime-contracts.json)；它们不计入上述 15 项正式集成检查。
