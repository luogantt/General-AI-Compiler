# Direct MLIR：DXA/LMEM 双缓冲验证

日期：2026-10-14。设备链为 `kernel.mlir → mlir-opt → mlir-translate → LLVM IR →
VOLT clang → ELF → vxbin → SimX`，没有生成设备 C++ Kernel。

本记录覆盖合成 FP32 FFN，包含 FC1/Bias/ReLU、FC2/Bias/Residual、LayerNorm，
不包含完整 BEVFormer 或 FPGA/ASIC 测量。固定 VOLT LLVM 20.1.8 revision 为
`b2ecfe086df2d00233a87f3d3ebe6a74aba129e2`。

## 数值与周期结果

以下周期均是 **一次 warm-up 加三次 measured runs 的 SimX 总周期**，包括三 Kernel
及启动/Runtime 开销；没有把软件模拟耗时转换成硬件毫秒。每组对照使用相同输入、
冻结 NumPy golden 和同一个启用 DXA 的 Runtime。LayerNorm 在两个版本中相同。

| Shape：rows×channels→hidden→channels | 调度 | 总周期 | 最大绝对误差 |
|---|---|---:|---:|
| 8×32→64→32 | scalar | 1,075,957 | 4.76837158e-7 |
| 8×32→64→32 | DXA 4×4×32 | 1,292,411 | 4.76837158e-7 |
| 8×128→256→128 | scalar | 18,161,147 | 1.1920929e-6 |
| 8×128→256→128 | DXA 4×4×64 | 16,435,476 | 1.1920929e-6 |
| 8×256→512→256 | scalar | 74,031,533 | 1.43051147e-6 |
| 8×256→512→256 | DXA 4×4×64 | 64,173,738 | 1.43051147e-6 |
| 8×256→512→256 | DXA 4×4×128 | 61,048,712 | 1.43051147e-6 |

生产宽度上的 4×4×128 配置减少约 **17.5%** 周期，对应同配置模拟周期比约 **1.21×**。
小尺寸反而更慢：分块、两 Stage 搬运和 collective helper 的开销可能超过访存收益。
因此调度是显式选项，尚无自动代价模型或 autotuning，也没有声称所有 Shape 都加速。

其余数值检查包括：2×4 smoke；3×19 的 M/N/K 尾部；5×35 的多次 Stage 复用；
4×4 Tile 的多 warp 尾部。所有记录均 `finite=true`、`verified=true`，误差低于
`1e-3` 门限。完整 15 条结果与 Runtime profile 在 [results.json](results.json)，
原始 `*-simx.log` 与 [evidence.sha256](evidence.sha256) 保留在本目录。

生产宽度日志含 Runtime 的“10 秒未退休命令”轮询告警；这些运行最终完成并返回
`verified=true`，原始告警也保留。当前软件模拟运行耗时较长，不以告警本身判定
SRAM 死锁，也不把该耗时当作硬件延迟。

## 验证配置与代码检查

- RV32IMAF / ILP32F，1 core、4 warps/core、4 lanes/warp；TCU 和 VM 关闭。
- 16 KiB LMEM、4 banks，L1 writeback 开启，L2/L3 关闭，DXA 开启。
- 两个 MatMul 共用分块模板，四个物理地址描述符，不共用 FC1/FC2 描述符槽。
- WGATHER+DXA 在同一 inline asm 中，固定 x31；VOLT 编译加 `-ffixed-x31`。
- Tiny collective helper 与调用点保留 convergent；尾部无效线程仍到达所有 barrier。
- Kernel event 依赖保留 Runtime 的 launch 后 cache flush，保证 FC1 输出对 FC2 DXA 可见。
- Host 编译要求至少两个独立 barrier slot；同配置构建 Host、Kernel 和 Runtime。

原生 WSL 工具链的 32 个 Vortex 测试通过，无 skip；Windows 的同组测试有 2 个环境
skip。真实工具链测试检查 MLIR、LLVM IR、O2 后 convergent、相邻 WGATHER/DXA 指令
及三个 `__vx_kentry_*` 入口。Host、startup、ELF link、vxbin 打包和数值执行另由本轮
完整 SimX 测试验证。可查看 [smoke-kernel.opt.mlir](smoke-kernel.opt.mlir)、
[smoke-kernel.ll](smoke-kernel.ll) 和 [smoke-kernel.dump](smoke-kernel.dump)。

最后用 `2×2×2` Tile 重新构建并执行 2×4 FFN：每 CTA 两组 Stage 共 64 B，
FC1/FC2 分别跨 2/4 个 K Tile，最大误差 `1.1920929e-7`。
修改 manifest 的 Buffer plan、配置仅一个 barrier slot 均被明确拒绝，见
[final-checks.json](final-checks.json) 和对应 `final-*-rejection.log`。
这条额外集成检查不计入上面的 15 条性能/尾部记录。

## 复现

从已配置的 RV32 Vortex build tree 先重新运行对应 checkout 的 configure；不要
复用另一配置的旧 driver。运行时和 Kernel 的配置必须一致：

```bash
make -C "$VORTEX_BUILD/sw/runtime/simx" -j4 CONFIGS=-DVX_CFG_EXT_DXA_ENABLE
make -C "$VORTEX_BUILD/sw/kernel" -j4 LLVM_PATH="$LLVM_PATH" \
  CONFIGS=-DVX_CFG_EXT_DXA_ENABLE

python3 tools/validate_vortex_buffered.py \
  --vortex-build "$VORTEX_BUILD" --llvm-path "$LLVM_PATH" \
  --output /path/to/new-result-directory \
  --cases smoke,tails,reuse,compare,wide,multiwarp-tail,production-width

PYTHONPATH=src GAC_VOLT_LLVM_PATH="$LLVM_PATH" \
  python3 -m unittest discover -s tests -p 'test_vortex*.py' -q
```

脚本在 build tree 的新 `tests/regression/gac_buffer_*` 目录生成源码，拒绝覆盖已存在
artifact。每次复现请使用新 build tree 或先归档原测试目录。模型 scope、Buffer 地址
和编译器/RTL 对应见[双缓冲设计](../../../designs/vortex_buffered_mlir.md)。
