# Vortex 原生 MLIR MatMul 模板

统一分支 `lg/bevformer-full-ops` 增加三个可显式选择的 Vortex 模板。
它们使用 C++ `OpBuilder` 创建真实 MLIR Operation，经 LLVM/VOLT 生成机器码；
没有生成 C++ 设备 Kernel，也没有调用 Python 数学实现代替设备执行。
默认仍为 `scalar`，CUDA 后端保持独立。

## 编译链与代码

```text
ONNX → OnnxImporter → gac.matmul / gac.matmul_bias / gac.matmul_bias_relu
     → 形状、常量和融合 Pass
     → DeviceLowering：SSA 依赖、全局 arena、模板选择
     → VortexMatmul：合法性检查、tile、LMEM、K 循环与同步
       或 VortexTcu：寄存器 fragment、TF32 分解与 WMMA
     → SCF / Arith / LLVM + Vortex CSR
     → LLVM IR → VOLT → ELF → vxbin
     → C++ Runtime：校验资源、设置 DXA descriptor、提交 dispatch
```

| 职责 | 源码入口 |
|---|---|
| 编译选项 | [DeviceLowering.h](include/gac/DeviceLowering.h)、[gac-compile.cpp](tools/gac-compile.cpp) |
| 模板选择和 dispatch 元数据 | [DeviceLowering.cpp](lib/DeviceLowering.cpp) |
| LMEM/DXA 算法与资源检查 | [VortexMatmul.cpp](lib/VortexMatmul.cpp)：`planVortexMatmul`、`emitVortexMatmul` |
| TCU 算术与寄存器布局 | [VortexTcu.cpp](lib/VortexTcu.cpp)：`configureVortexTcuMatmul`、`emitVortexTcuMatmul` |
| CSR、barrier、DXA 指令和 ABI | [VortexLowering.cpp](lib/VortexLowering.cpp) |
| manifest 校验 | [dispatch-plan.h](runtime/dispatch-plan.h) |
| 设备能力、搬运、提交、计数器 | [vortex-run.cpp](runtime/vortex-run.cpp) |
| 独立数值与周期比较 | [run_vortex_matmul_tests.py](tests/run_vortex_matmul_tests.py) |

## 三种模板的实际边界

| 模板选项 | 数据与执行方案 | 限制 |
|---|---|---|
| `lmem-tiled` | 一个 CTA 负责 C 的一个 M×N tile；协作搬入 A/B；每线程一个 FP32 累加器 | 完整 warp；静态连续 FP32；含 batch 广播 |
| `dxa-double-buffer` | 两组 LMEM A/B tile；计算当前组时预取下一组 | 当前限 4 lane；右矩阵 rank 2；M/N/K≤65535；DXA 扩展 |
| `tcu` | 4 lane warp 计算 8×4×4 tile；寄存器 fragment；三次 TF32 WMMA 累加 | TCU 和 TF32 必须启用；近似算术；此模板 LMEM=0 |

这些是独立模板。本版没有把 DXA、LMEM 和 TCU 合成 WGMMA 流水线。
Conv、DCNv2、Softmax、LayerNorm、MSDeformAttn 等继续使用现有设备 Lowering，
不会因为开启一个选项就自动变成 TCU 算子。

高维 A 与二维 B 的投影会把 A 的前导维展平为 M；批量 MatMul 按广播规则计算
batch 地址。Bias 在完整 K 归约后加入，ReLU 随后执行。M/N 尾块不写越界输出；
LMEM 模板的 K 尾块只归约有效元素，所有 CTA 线程仍参与 barrier。

## LMEM 和双缓冲的生命周期

每组缓冲区占用：`4 × BK × (BM + BN)` 字节；双缓冲需要两倍空间。
Buffer 是该 CTA 获配 LMEM 中的地址区间，C 的部分和保存在寄存器中。

```text
分配 LMEM 两组 → 发起 stage 0 的 A/B 搬运
每个 K tile：
  预取下一 tile 到另一组（若存在）
  → 等待当前组 DXA 完成且 CTA 全部到齐
  → 从当前组读取 A/B，累加 C
  → CTA barrier：确认所有线程停止读取当前组
  → 切换组，已释放的组才能用于之后的预取
全部 K 完成 → Bias/ReLU → global store C
```

DXA descriptor 0/1 分别描述 A/B 的二维形状、stride 和 tile。
Runtime 使用物理连续缓冲区；重编程 descriptor 前等待前一 Kernel 完成，
保留运行库的数据可见性处理。不会在前一个 dispatch 仍使用 descriptor 时修改它。

编译器记录每个 dispatch 的 grid、block、LMEM、stages 和逻辑 operand shape。
Reshape 等视图可以复用原 Buffer，但计算维度取逻辑 shape；Runtime 同时验证
逻辑 shape 的字节数与物理分配完全一致，避免把视图前的 shape 当成矩阵维度。

## TCU 的精度合同

当前 Vortex TCU 不支持直接 FP32×FP32 输入。`tcu` 明确使用 TF32 三项分解：

```text
A ≈ Ahi + Alo；B ≈ Bhi + Blo
C += Ahi×Bhi + Ahi×Blo + Alo×Bhi
```

每一项由真实 WMMA 指令完成，累加为 FP32。TF32 分量按 Vortex 的 19 位格式
打包；省略 `Alo×Blo`，因此不能保证与标量 FP32 逐位一致或任意输入都满足同一
误差阈值。manifest 必须记录 `arithmetic: tf32x3-fp32-accumulate`。

所有 WMMA 完成后，非有限累加结果会由设备 FPU 使用原始 A/B 完整重算，处理
分解中的 Inf/NaN 异常；没有 Host 算子回退。有限结果仍是近似值，完整模型启用
该模式后仍须逐层精度对齐和任务指标验证。默认标量模式不改变精度合同。

## 构建与使用

沿用 [主 README](README.md) 的 LLVM 20/VOLT 和 Vortex 构建环境。
使用 TCU 时，Vortex 硬件/SimX 构建需启用 DXA、TCU 和 TF32，Runtime 配套配置：

```bash
export CONFIGS="-DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TF32_ENABLE -DVX_CFG_NUM_THREADS=4"
cmake -S native -B build/native \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DVORTEX_BUILD="$VORTEX_BUILD" -DVORTEX_SOURCE="$VORTEX_HOME" \
  -DGAC_VORTEX_TCU_TF32=ON
cmake --build build/native -j4

GAC_NATIVE_BIN="$PWD/build/native" \
GAC_VORTEX_MATMUL_SCHEDULE=dxa-double-buffer \
GAC_TILE_M=4 GAC_TILE_N=4 GAC_TILE_K=128 \
  native/tools/compile-vortex.sh model.onnx build/model-dxa

VORTEX_DRIVER=simx LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime" \
  build/native/gac-vortex-run build/model-dxa/plan.json \
  build/model-dxa/kernel.vxbin build/model-dxa build/model-dxa/outputs \
  --profile-counters
```

将模板选项改成 `lmem-tiled` 或 `tcu` 可分别验证。TCU 的 tile 固定为 8×4×4，
实际 geometry 写入 manifest；不会使用命令中的 4×4×128。全模型 base 编译还需
`GAC_CONSTANT_FOLD_PROFILE=bevformer-base`。先运行测试模型，再逐层验证整网。

Runtime 动态查询 ISA、lane、warp 和 LMEM 能力，但当前 Vortex caps 无 TF32
格式和 barrier 数量查询接口。`GAC_VORTEX_TCU_TF32` 与 barrier 数量是明确的
构建配置声明，必须与加载的驱动、SimX 或 RTL 匹配；报告会区分两类信息。

不满足模板条件时，编译计划记录 `fallback_reason` 并保留设备标量实现。
manifest 不一致、设备扩展缺失或资源不足则明确报错，不静默伪造优化已执行。

## 验证和性能解释

本轮 36 组数值检查、K=128 对照、非法配置与 CUDA/CPU 回归的完整记录见
[2026-10-17 验收报告](../docs/validation/vortex/native-mlir-matmul-2026-10-17/README.md)。
测到 DXA K=128 在一个 FFN 投影上减少约 2.46% 周期；当前 LMEM 和 TCU 模板
在相同配置下仍慢于标量，未将它们设为默认。
完整 BEVFormer-base 三种模式均已生成并验证 LLVM IR：LMEM/TCU 各替换 181 个
MatMul，DXA 替换 145 个并生成完整 ELF/vxbin；整网新模板的设备执行仍待验收。

验证脚本从同一 ONNX 和输入生成四种机器码，在同一 SimX 配置执行，并与 NumPy
独立参考比较；NumPy 仅用于造数据和验收。测试覆盖尾块、多轮 K、批量广播、
高维投影、连续 MatMul/Residual、Reshape 视图和非有限值。

`device_counters.max_core_cycles` 是整个 dispatch 序列前后的设备计数器差值，
包括设备端启动、同步与 flush 开销；主机仿真耗时另列。不能把 SimX 主机耗时或
小 MatMul 的周期数说成完整 BEVFormer 的帧延时，也不能从这里推算实测 30 FPS。

LMEM 复用可能减少全局读取，但会增加协作搬运、地址运算和 barrier。DXA 双缓冲
能否获益取决于搬运与计算的比例。TCU 还受到 TF32 分解、fragment 重排和寄存器
压力影响。保留标量对照和慢于基线的结果，按真实 shape 测量后再选择模板。
