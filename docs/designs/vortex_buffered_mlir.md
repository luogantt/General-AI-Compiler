# BEVFormer FFN：分块、LMEM 双缓冲与 Direct MLIR

对应分支 `lg/bevformer-vortex-direct-mlir`。本文说明 `dxa-double-buffer` 调度：
设备端由 GAC 生成 LLVM dialect MLIR，经原生 MLIR 工具和 VOLT 生成机器码；
主机端 C++ 负责分配内存、配置描述符和提交 Kernel。

当前范围是静态 Shape、连续 FP32、RV32 的两个 FFN MatMul；LayerNorm 复用标量
实现。它不是完整 BEVFormer，也没有启用 Tensor Core。

## 1. 一条编译链，两条数据路径

```text
模型/算子图
  → Tensor/Linalg MLIR → mlir-opt
  → 经过审计的 FFN 调用计划
  → LLVM dialect MLIR（线程、分块、DXA、LMEM、乘加）
  → mlir-opt → mlir-translate → LLVM IR
  → VOLT 指令选择/寄存器分配 → ELF → vxbin

输入：Device DDR → DXA → LMEM 中的 A/B Buffer → 操作数寄存器 → FPU
输出：FPU → C 累加寄存器 → LSU/缓存 → Device DDR
```

Tile 描述符记录地址、尺寸和 stride；它本身不装矩阵数据。LMEM 是物理 SRAM，
Buffer 是这块 SRAM 中分配的地址区域。C 的部分和保留在每个线程的累加值中，
由 VOLT 做寄存器分配；最终 C 不需要经过 A/B 的 LMEM Buffer。

`optimized.mlir → 调用计划 → kernel.mlir` 是 GAC Python 实现的受限 lowering。
原生 `mlir-opt` 负责解析、验证、canonicalize 和 CSE；当前没有新增通用的 C++
MLIR conversion pass，也没有自定义 Vortex dialect。

## 2. M、N、K 如何分块

矩阵乘 `C[M,N] = A[M,K] × B[K,N]`：M 是输出行，N 是输出列，K 是累加轴。
所有矩阵按 row-major 存放。FC1 的 `(M,N,K)=(rows,hidden,channels)`；
FC2 的 `(M,N,K)=(rows,channels,hidden)`。

一组 CTA 负责一个 `BM × BN` 的 C Tile，每个线程计算一个输出元素：

```text
local_row = thread_x / BN      local_col = thread_x % BN
row = row_base + local_row    col = col_base + local_col

沿 K 每次前进 BK：
  A[row_base : row_base+BM, k : k+BK]
  B[k : k+BK, col_base : col_base+BN]
  Cacc[row,col] += Σ A[row,j] * B[j,col]
```

DDR 中的原矩阵保持完整。所谓切分，是计算上述地址范围，并把所需数据搬到
LMEM。LMEM 内 A 的索引为 `local_row*BK+j`，B 的索引为 `j*BN+local_col`；
DDR 源行步长仍是原矩阵完整的 K 或 N。

教学图的 `C00 = A00×B00 + A01×B10`，就是同一个 C Tile 沿 K 的两轮累加。
实际 FFN 的 Cacc 从该列 bias 开始，FC1 完成后做 ReLU，FC2 完成后加 query residual。

## 3. 双 Buffer 的地址与时序

每个 CTA 申请两组 Stage，每组都包含一个 A Buffer 和一个 B Buffer：

```text
LMEM 起始地址
  +0                         Stage 0：A[BM,BK] | B[BK,BN]
  +stage_stride_bytes        Stage 1：A[BM,BK] | B[BK,BN]
```

FP32 的内存需求为：

```text
A Buffer = BM * BK * 4 B
B Buffer = BK * BN * 4 B
Stage    = (BM + BN) * BK * 4 B
双 Stage = 2 * (BM + BN) * BK * 4 B
```

默认 `BM=2, BN=2, BK=16`：4 个线程、每个 Stage 256 B、每 CTA 共 512 B。
Host 编译时检查至少两个独立 barrier slot，运行时用 `vx_check_occupancy()`
检查线程数和 LMEM，再设置 `lmem_size`；
不根据“必须把 SRAM 填满”选 Tile。

```text
1. loader warp 发起当前 K Tile 的 A/B 两次 DXA，目标 Stage 0。
2. 若还有下一轮，先发起下一 K Tile 的 DXA，目标另一个 Stage。
3. 全 CTA 等待当前 Stage 的两次 DXA 完成。
4. 全 CTA 读取当前 Stage，逐项乘加；下一 Stage 可同时接收数据。
5. 全 CTA 再同步，确保当前 Stage 已读完，允许后续覆盖。
6. cur ^= 1，K 前进 BK，重复 2～5；最后由有效线程写出 C。
```

尾部不整除时，DXA 越界位置零填充，乘加循环只遍历有效 K，最终写回检查 M/N。
即使某线程的输出越界，它也必须参加循环和两次 barrier，不能提前 return。
这避免某些线程退出、其他线程永远等待的同步错误。

### 描述符不是 Stage 编号

四个 DXA 源描述符由 Host 编程，两个 Stage 在 Device 端交替作为目标：

| Descriptor slot | 源矩阵 | Tile 尺寸（行×列） |
|---|---|---|
| 0 | FC1 query | `BM × BK` |
| 1 | FC1 weight | `BK × BN` |
| 2 | FC2 activation | `BM × BK` |
| 3 | FC2 weight | `BK × BN` |

DXA 绕过每核地址翻译，源描述符使用物理地址；生成的 Host 内存分配加
`VX_MEM_PHYS`。Buffer0/1 是 LMEM 的目标偏移，不是 descriptor slot0/1。

## 4. 从 MLIR 操作到指令与 RTL

GAC 代码入口采用函数名定位，避免后续修改导致行号失效：

| 工作 | GAC 源码入口 |
|---|---|
| CLI、Tile 参数 | [`build_bevformer_vortex.py`](../../src/gac/tools/build_bevformer_vortex.py)：`_parser()`、`main()` |
| 双 Stage 字节数、CTA Grid | [`vortex_buffered_mlir.py`](../../src/gac/backends/vortex_buffered_mlir.py)：`VortexBufferPlan` |
| LMEM 地址、循环、边界、累加 | 同文件：`_matmul_kernel()` |
| DXA/barrier 设备操作 | 同文件：`_copy()`、`_collective_helpers()` |
| 生成与审计 Device MLIR | 同文件：`emit_bevformer_ffn_buffered_device_mlir()`、`audit_bevformer_ffn_buffered_device_mlir()` |
| 接入编译链、Manifest | [`vortex.py`](../../src/gac/backends/vortex.py)：`compile_vortex_direct()`、`write_vortex_artifact()` |
| Host 描述符、LMEM 申请 | 同文件：`_buffered_host_source()` |
| 同版 MLIR/VOLT 构建 | 同文件：`_direct_makefile()` |

下表路径相对于外部的 `vortex-3.0/` 或 VOLT `llvm-project/` checkout：

| 动作 | Device IR/指令 | 软件或 RTL 入口 |
|---|---|---|
| 源二维地址与步长 | DXA descriptor | Vortex `sw/runtime/include/dxa.h`；`hw/rtl/dxa/VX_dxa_desc_table.sv`、`VX_dxa_addr_gen.sv` |
| 提交 A/B 搬运 | `llvm.inline_asm`：WGATHER + DXA 编码 | Vortex `sw/kernel/include/vx_dxa.h`；`hw/rtl/dxa/VX_dxa_dispatch.sv`、`VX_dxa_gmem_req.sv` |
| DDR 数据写入 LMEM | DXA 目标地址 | Vortex `hw/rtl/dxa/VX_dxa_smem_wr.sv`；`hw/rtl/mem/VX_local_mem.sv` 的 DMA port 和 banks |
| 等待搬运/保护 Stage 复用 | barrier 指令 | Vortex `hw/rtl/dxa/VX_dxa_completion.sv`；`hw/rtl/core/VX_scheduler.sv` |
| 从 LMEM 取数 | `llvm.load` → load 指令 | Vortex `hw/rtl/core/VX_lsu_unit.sv`；`hw/rtl/mem/VX_lmem_switch.sv` |
| 乘加和累加值 | `llvm.fmul`、`llvm.fadd` → 浮点指令 | Vortex `hw/rtl/fpu/VX_fpu_unit.sv`；`hw/rtl/core/VX_operands.sv`、`VX_opc_unit.sv` |
| 输出写回 | `llvm.store` → store 指令 | Vortex `hw/rtl/core/VX_lsu_unit.sv`；`hw/rtl/cache/VX_cache_data.sv` |
| Kernel ABI 与最终指令 | VOLT 后端 | LLVM `llvm/lib/Target/RISCV/RISCVVortexKEntry.cpp`、`RISCVInstrInfoVX.td`、`RISCVRegisterInfo.cpp` |

DXA 使用 WGATHER 收集前四个 lane 的参数，所以要求至少 4 个 CTA 线程和 4 个
硬件 lane。WGATHER 与 DXA 在同一内联汇编块中，通过 x31 传递打包参数。
生成的 Makefile 加 `-ffixed-x31`，防止普通寄存器分配破坏这一协议。

LLVM 20 的 `llvm.inline_asm` 不能直接附完整的 convergent 调用属性，因此 DXA
和 barrier 包装在很小的 `convergent + noinline + optnone` helper 中；调用点也
标记 convergent，内联汇编带 memory clobber。只有这些 collective helper 禁止
优化，MatMul 本体仍交给 VOLT 优化。它们保护参与线程集合和指令顺序。

FC1 的 activation 写回可能暂留在 dirty cache line。Host 提交
`FC1 event → FC2 event → LayerNorm event`；`vx_enqueue_launch()` 经
`sw/runtime/common/queue.cpp` 调用 `Device::cp_submit_launch()`，后者在
`sw/runtime/common/device.cpp` 中提交 `cp_submit_cache_flush()`。
这层缓存可见性保证必须保留，FC2 的 DXA 才能
读取 FC1 的最新结果。CTA barrier 只处理 CTA 内同步，不能替代 Kernel 间 flush。

## 5. 生成、验证与同 Shape 对照

先使用同一个 LLVM 20.1.8/VOLT revision 的 `mlir-opt`、`mlir-translate`、clang、lld。
工具链安装与构建方法见[完整架构说明](../bevformer_vortex_direct_mlir_architecture.md)。
以下从 GAC 根目录执行，`VORTEX_BUILD` 指向已配置的 RV32 Vortex build tree：

```bash
PYTHONPATH=src python3 tools/build_bevformer_vortex.py \
  --synthetic-width-smoke \
  --device-lowering direct-mlir \
  --matmul-schedule dxa-double-buffer \
  --tile-m 2 --tile-n 2 --tile-k 16 \
  --mlir-opt "$LLVM_PATH/bin/mlir-opt" \
  --output "$VORTEX_BUILD/tests/regression/gac_buffered"

make -C "$VORTEX_BUILD/tests/regression/gac_buffered" LLVM_PATH="$LLVM_PATH" \
  CONFIGS=-DVX_CFG_EXT_DXA_ENABLE
make -C "$VORTEX_BUILD/tests/regression/gac_buffered" LLVM_PATH="$LLVM_PATH" \
  CONFIGS=-DVX_CFG_EXT_DXA_ENABLE \
  run-simx OPTS="--data-dir . --kernel kernel.vxbin --warmup 1 --repeat 10"
```

标量对照保持相同 source profile 和 seed，将 `--matmul-schedule` 改为 `scalar`，
换一个 output 目录，执行相同 build/run 命令。真实权重使用
`--weights /path/model.gacweights --weights-rows 1` 替换 synthetic 参数。

首先确认 `verified=true`、有限值和误差阈值，再比较 instructions、cycles、
device profile 与端到端延迟。SimX 主机耗时不是 FPGA 或 ASIC 实际延迟。
本轮已通过完整 MLIR → LLVM IR → VOLT → ELF → vxbin → SimX，覆盖 M/N/K 尾部、
多 warp 和多轮 Stage 复用。合成生产宽度 `[1,8,256] → 512 → 256` 的
`BM=4, BN=4, BK=128` 使用 16 线程、8192 B LMEM，模拟总周期比同输入 scalar 少
17.5%，最大绝对误差 `1.43051147e-6`。小尺寸对照更慢，因此保留显式调度选择，
没有自动声称所有 Shape 都获益。[原始结果与复现步骤](../validation/vortex/buffered-mlir-2026-10-14/README.md)。

生成前确认 Vortex driver 和 Kernel 使用相同的 `CONFIGS=-DVX_CFG_EXT_DXA_ENABLE`；
从 build tree 重新执行 configure，再构建 `sw/runtime/simx`、`sw/kernel`。
本轮验证为 VM 关闭、1 core、4 warps、每 warp 4 lanes、16 KiB LMEM、TCU 关闭。
没有把软件模拟器的耗时当作真机推理延迟。

## 6. 没有这个 Buffer 会不会卡死

没有显式 LMEM Tile Buffer，仍可从 DDR/cache 把 A/B 元素读取到寄存器后计算。
原标量路径就是这种方式。它可能因重复访存和等待数据较慢，但不必然卡死。

双 Stage 增加 A/B 复用，并提供搬运与计算重叠的机会。收益还取决于 Tile 尺寸、
LMEM bank 冲突、CTA 并发数、DXA 开销和计算量；过大的 Tile 可能降低并发，
双缓冲也可能比标量更慢。barrier 参与者不一致或完成计数错误可能造成死锁；
覆盖尚未读完的 Stage 则会造成数据错误。它们需要正确的同步协议。
