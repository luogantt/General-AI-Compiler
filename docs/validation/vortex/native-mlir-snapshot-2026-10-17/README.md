# 原生 MLIR Vortex 版本存档

分支：`lg/bevformer-vortex-full-ops`。标签：`bevformer-base-mlir-vortex`。

本存档保存 CUDA 原生后端加入前的 Vortex 实现。此前这部分代码尚未单独提交；
本次从 Git 对象库中找到完整的旧源码树，恢复到独立工作区并提交。CUDA 分支及
`bevformer-base-mlir-cuda-ptx` 标签保持原来的提交不变。

## 源码身份

- 恢复来源树：`89df1527f7b10f3ef728d5334340c6b65643494a`。
- 原生编译器子树：`bd83dfbb7130d70ff843d74321c46140fa510b21`，共 42 个文件。
- `native/` 保留当时的完整文件布局，包括未拆分的 `VortexLowering.cpp` 和
  `vortex-run.cpp`；不是从 CUDA 版删去几个入口而得到的近似实现。
- C++ 编译器提供 `vortex` 与显式主机验证目标 `native`；新增的原生 CUDA/NVVM
  编译及 Runtime 文件不属于本快照。仓库中更早的 Python/CUDA 历史实现仍保留。

原构建的 `gac-compile` SHA-256 为
`050f89d4f3fff055c8db68bbb9c86b3c8b6c3d5f10e59aa2e378024fa50f666d`，
`gac-opt` 为 `537fc823a230f6b762d076ace2a37b8c6b9ccbfe2d543d3ef31674d57c73add8`。
仍保留的旧工具与历史验证记录中的身份一致。恢复后的构建与对照结果单独记录，
不会用新工具哈希覆盖旧报告。

## 本版本已经验证的范围

编译链为 `ONNX → C++ MLIR Operations/Passes → Vortex/LLVM Dialect → VOLT → ELF → vxbin`。
Buffer 规划和运行时均为 C++，没有 Python 算子回退。

| 项目 | 已有证据 |
|---|---|
| 小模型 Vortex 执行 | MatMul、Conv、Softmax 及部分索引/形状案例通过 SimX 数值测试 |
| 完整 BEVFormer-tiny 固定 cold 图 | 1,548 个 dispatch，主机机器码执行与参考对齐；完整 Vortex 产物已生成 |
| 完整 BEVFormer-base 固定 cold 图 | 3,224 个 dispatch，主机机器码执行与参考对齐；Vortex ELF/vxbin 及入口审核通过 |
| 完整模型 Vortex 运行速度 | **尚未验证**，不得使用 CUDA 版本的延时或显存数字代替 |

完整主机数值验证与完整 Vortex 设备执行是两项不同验收。上述整网测试仍限于固定
输入、固定标定的 cold-frame 原始输出，不包括数据集精度、历史帧或检测框解码。

原始记录：

- [原生 MLIR 与 SimX 小模型](../native-mlir-2026-10-16/README.md)
- [完整 BEVFormer-tiny](../native-mlir-bevformer-full-2026-10-16/README.md)
- [完整 BEVFormer-base](../native-mlir-bevformer-base-2026-10-16/README.md)

这些记录包含真实产物哈希、历史失败尝试及最终验证范围；归档时保留原始字节。

## 恢复后的独立检查

在独立目录重新构建 `gac-compile`、`gac-opt`、`gac-vortex-run` 成功。
重新构建的 `gac-compile` SHA-256 **与上面的历史二进制完全相同**。
没有覆盖或重新构建保存旧二进制的目录。

- 8 项基础主机回归和 DCNv2、MSDA、Resize、碎片 Buffer 复用 4 项代表性测试通过。
- 6 项 Vortex RV32 对象编译检查通过；这些检查不代表新增的完整 SimX 执行。
- 10 个代表性模型分别使用 `native`、`vortex` 目标，对照旧编译器和恢复后的编译器。
  20 组输出的 `plan.json`、`kernel.mlir`、`kernel.ll`，共 60 个文件全部逐字节相同。
- 暂存后的 `native/` Git 子树与历史子树 ID 相同，42 个文件完整保留。

证据文件：

| 文件 | 内容 |
|---|---|
| [source-recovery.json](source-recovery.json) | 恢复来源、各文件的 Git blob 与 SHA-256 |
| [preserved-compiler-comparison.json](preserved-compiler-comparison.json) | 旧/新编译器身份与 60 个产物的逐字节对照 |
| [native-regression-results.json](native-regression-results.json) | 基础主机回归 |
| [base-operator-regression-results.json](base-operator-regression-results.json) | DCNv2、MSDA、Resize 和 Buffer 案例 |
| [vortex-object-results.json](vortex-object-results.json) | RV32 对象编译验证 |
