# BEVFormer FFN → MLIR → Vortex 3.0 验证记录

本目录记录 `codex/bevformer-vortex-mlir` 的最终验证。编译器源代码固定在
`76bbb36d14c5ef20332358d295281311c3a8ba10`；Vortex 源码固定在
`14926ffc8755ceb664ec017028bba7f4a250096a`。每个结果 JSON 都保存两级 MLIR、
manifest、VOLT 产物、运行日志和目标配置的 SHA-256。

验证环境是 Vortex 3.0 RV32 build：1 core、每 core 4 warps、每 warp 4 threads，
TCU 未启用。`config.mk` 摘要为
`99bc7ddc92371790199a0b1039ad14fd7009d7c05be464ed6508bf29140cd7d9`，
`VX_config.toml` 摘要为
`ae535ee43b5b6aece2f3374b971b7b558e7b016de5da15ccf338cc395d61c436`。
VOLT 使用 clang 20.1.8，Vortex LLVM revision 为
`b2ecfe086df2d002333a87f3d3ebe6a74aba129e2`。MLIR 中端使用 LLVM 24
`mlir-opt` revision `765392a6b4498f1dc57233f03a31b87462fdef01`。

Vortex checkout 验证时不是干净状态：`third_party/cocogfx` 有修改，并存在未跟踪的
`start.sh`。二者都记录在原始日志和结果 JSON 中；本次构建没有修改或读取
`start.sh`，但复现实验时仍应核对这一差异。

## 结果

| Workload | 结果 | SimX kernel device profile | SimX 端到端 | 最大绝对误差 | cycles / instructions |
| --- | --- | ---: | ---: | ---: | ---: |
| 合成 `[1,2,4] → 8 → 4` | 3 次均通过 | `5.334338 ms` | `5.499720 ms` | `1.1920929e-7` | `2090 / 229` |
| 官方 checkpoint 第 0 层，一行 `[1,1,256] → 512 → 256` | 1 次通过 | `2319.042238 ms` | `2319.287289 ms` | `1.90734863e-6` | `58133 / 6025` |
| 官方 checkpoint 第 0 层，完整 `[1,2500,256] → 512 → 256` | 两级 MLIR、artifact 校验、host 与 VOLT 编译通过 | — | — | — | — |

两个执行用例均读取 artifact 中冻结的 `golden.bin`，容差为 `1e-3`，并输出
`finite=true`、`verified=true`。完整 2500-row 用例没有运行 SimX：当前 Kernel 是
标量功能基线，一行真实宽度已经需要约 2.319 秒的 simulator wall time；把 2500 行
SimX 时间当成 Vortex 硬件延时没有意义。

三份 Kernel 的 `kernel.vxbin` 都是 5010 bytes，SHA-256 为
`2896b0a623eecae2e3e2dd40c639c40b4ebcf33163429071004174ff422e418e`。
相同摘要是预期行为：Kernel 从参数结构读取 rows/channels/hidden，profile 差异位于
host 配置与数据文件中。

## 可追溯证据

| Workload | 结果记录 | Artifact manifest | 原始 build/run stdout |
| --- | --- | --- | --- |
| 合成 smoke | [`synthetic-smoke-simx.json`](synthetic-smoke-simx.json) | [`synthetic-smoke-manifest.json`](synthetic-smoke-manifest.json) | [`synthetic-smoke-build-and-simx.log`](synthetic-smoke-build-and-simx.log) |
| 官方权重一行 | [`official-checkpoint-row1-simx.json`](official-checkpoint-row1-simx.json) | [`official-checkpoint-row1-manifest.json`](official-checkpoint-row1-manifest.json) | [`official-checkpoint-row1-build-and-simx.log`](official-checkpoint-row1-build-and-simx.log) |
| 官方权重 2500 行 | [`official-checkpoint-production-compile.json`](official-checkpoint-production-compile.json) | [`official-checkpoint-production-manifest.json`](official-checkpoint-production-manifest.json) | [`official-checkpoint-production-build.log`](official-checkpoint-production-build.log) |

原始日志包含 commit、dirty status、VOLT revision、配置摘要、三份 MLIR 摘要、
`verify_artifact.py` 输出、实际编译命令、二进制摘要和完整 SimX JSON。结果 JSON
另外保存原始日志摘要，因此表格中的数值可以回溯到具体日志和 artifact。

## 解释边界

这些数据证明当前 BEVFormer FFN 切片完成了 Canonical Graph → 两级标准 MLIR →
受限 Runtime-call lowering → Vortex C++/VOLT → `.vxbin` → SimX 的正确性闭环。
SimX 是功能/指令级模拟器；上述毫秒数不能作为 FPGA/ASIC 性能，不能换算 FPS，
也不能与 RTX 3070 CUDA Event 延时直接比较。完整 BEVFormer 的 backbone、FPN、
MSDeformableAttention、Encoder/Decoder、检测头和时序状态仍未迁移到 Vortex。
