# Vortex 寄存器分块与 TCU 调用合并验收

本轮将 LMEM/DXA 模板扩展为每线程 RM×RN 个累加器，并为 TCU 增加可选的三次
WMMA 合并调用。全部计算仍通过 C++ MLIR Operation → LLVM/VOLT → ELF/vxbin，
实际执行于同一 SimX 配置。没有使用 Python 计算代替设备执行。

## 可复查的结果

- [register-tiling.json](register-tiling.json)：60 组独立 NumPy 数值比较全部通过，
  含标量、1×1 基线、2×2、4×1 和 2×4 寄存器配置；记录实际模板、回退原因、
  模型/输入/输出/产物哈希、周期、指令数和部分 ELF 静态栈访问信息。
- [tcu-combined-comparison.json](tcu-combined-comparison.json)：9 个模型、两种 TCU
  调用模式，共 18 组数值检查通过；两种模式的输出二进制完全一致。
- [manifest-rejections.json](manifest-rejections.json)：27 个非法 manifest 在加载
  驱动前被拒绝，包括寄存器块大小、整除关系、线程数和 TCU 专用布局错误。
- [legacy-manifest.json](legacy-manifest.json)：删除 `thread_tile_m/n` 的旧 LMEM/DXA
  manifest 各实际执行一次，按 1×1 解读，输出与字段完整版本逐位相同。

正例误差阈值 `atol=rtol=2e-5`；NaN/Inf 通过非有限值用例单独覆盖。寄存器分块
60 组的最大有限绝对误差为 `1.1920929e-6`。NumPy 只生成独立参考。批量右矩阵
不能使用现有 DXA 二维描述符时，plan 明确记录标量回退，未把它计为 DXA 执行。

## 固定硬件、工具与输入

RV32、1 core、4 lanes、4 warps/core、16 KiB LMEM、8 barrier slots，启用 DXA、
TCU 和 TF32。TF32 与 barrier 数量属于构建配置声明，其余能力来自 Runtime 查询。

| 工具 | SHA-256 |
|---|---|
| gac-compile | `4c2538adf43e5fd8b6520c6b1ef2f4340189571c5fadcd34b19b9c933bdd85c1` |
| gac-vortex-run | `16201fb4fd406e8caa2273b50474a6a8937af4a029c90dc1f95b39c9de6bc08e` |
| libsimx.so | `162df881c1e73cb0f9fe8880c7434bc898d49170b6395bfbacaa4e96bac0a967` |

工具冻结于 `/home/thomas/gac-vortex-register-frozen-20261017`，硬件配置位于
`/home/thomas/vortex-gac-native-tf32-build32-20261017`。完整原始产物目录保留在 JSON
的 `artifacts` 中；报告不依赖历史测试的周期数字。

## FFN 投影性能对照

同一个 `[8,256] × [256,512] + Bias → ReLU`，确定性随机种子 `20261017`。
下表全部使用相同 ONNX、输入、编译器、Runtime 和 SimX；标量及 LMEM/DXA 输出
二进制完全一致。数字是整个小图 dispatch 序列的设备建模周期，包含启动、同步、
flush 开销，**不是实际芯片毫秒，也不是完整 BEVFormer 的帧延时**。

| 模板 | BM×BN×BK | RM×RN | 周期 | 相对标量速度 |
|---|---|---|---:|---:|
| Scalar | — | — | 7,782,295 | 1.000× |
| LMEM 原配置重测 | 4×4×16 | 1×1 | 12,845,850 | 0.606× |
| DXA 原配置重测 | 4×4×16 | 1×1 | 8,552,239 | 0.910× |
| LMEM 原 K=128 重测 | 4×4×128 | 1×1 | 15,113,886 | 0.515× |
| DXA 原 K=128 重测 | 4×4×128 | 1×1 | 7,594,572 | 1.025× |
| LMEM 寄存器分块 | 8×8×64 | 2×2 | 8,043,386 | 0.968× |
| DXA 寄存器分块 | 8×8×16 | 2×2 | 4,980,980 | 1.562× |
| DXA 寄存器分块 | 8×8×64 | 2×2 | 4,361,697 | 1.784× |
| DXA 寄存器分块 | 8×8×128 | 2×2 | 4,628,008 | 1.682× |
| LMEM 寄存器分块 | 8×16×64 | 2×4 | 6,317,970 | 1.232× |
| DXA 寄存器分块 | 8×16×64 | 2×4 | 3,578,862 | 2.175× |
| TCU 三次独立调用 | 固定 8×4×4 | 固定 fragment | 9,092,481 | 0.856× |
| TCU 一次合并调用 | 固定 8×4×4 | 固定 fragment | 6,704,995 | 1.161× |

2×4 DXA 相对本轮重测的旧最快 DXA（K=128）快约 2.122×。两组 Buffer 共占用
`2×4×64×(8+16)=12,288 B`，16 个线程各保留 8 个累加器；仍遵守读前等待和复用前
barrier。2×2 K 从 64 增加到 128 反而变慢；容量更大不等于速度更快。

TCU 合并调用只改变三次 WMMA 的函数边界，保持执行顺序和保护属性；相对旧 TCU
减少 26.26% 周期（1.356×），但仍慢于这里的 DXA 配置。caller 栈帧从 208 B 降至
144 B，helper 从 32 B 增至 112 B；不能据此声称动态 spill 流量减少。静态反汇编
确认每个 K 小块由三次 helper 调用变为一次，合并 helper 内含三条 WMMA。

小图并非全部获益：2×2、BK16 的 tail 例从标量 4,415 cycles 增至 DXA 5,586；
高维小投影从 7,665 增至 15,090。因此默认仍为 `scalar`，寄存器块与 TCU 合并
均由选项显式启用；没有把单例最快配置设为全模型统一策略。

## 复现

沿用 [native Vortex 说明](../../../../native/README.vortex-matmul.md) 的工具链配置：

```bash
export TOOLDIR=/home/thomas/vortex-gac-tools
export CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_NUM_THREADS=4"
python native/tests/run_vortex_matmul_tests.py \
  --compiler "$NATIVE_BIN/gac-compile" --llvm-bin "$LLVM_PATH/bin" \
  --runner "$NATIVE_BIN/gac-vortex-run" --vortex-build "$VORTEX_BUILD" \
  --output /tmp/vortex-register-new-run \
  --schedule scalar --schedule lmem-tiled --schedule dxa-double-buffer \
  --tile-m 8 --tile-n 8 --tile-k 16 --thread-tile-m 2 --thread-tile-n 2

# TCU 固定 fragment，保持寄存器选项默认 1×1。
python native/tests/run_vortex_matmul_tests.py \
  --compiler "$NATIVE_BIN/gac-compile" --llvm-bin "$LLVM_PATH/bin" \
  --runner "$NATIVE_BIN/gac-vortex-run" --vortex-build "$VORTEX_BUILD" \
  --output /tmp/vortex-tcu-combined-new-run --schedule tcu --tcu-combined

python native/tests/test_vortex_dispatch_plan.py \
  --runner "$NATIVE_BIN/gac-vortex-run" --report /tmp/vortex-plan-tests.json
```

每次使用新的 output 目录。改变 tile 或寄存器配置后应重新编译并验证；测试脚本
拒绝混用已有产物。当前没有实现 TCU＋LMEM＋DXA 合一模板，没有增加完整模型
Vortex 运行结果或物理硬件的 FPS；更细的 DDR、LMEM bank、stall 计数器尚未接入
本次 Runtime 报告，不能用周期/指令数替代瓶颈归因。
