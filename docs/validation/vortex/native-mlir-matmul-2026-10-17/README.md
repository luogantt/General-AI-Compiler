# 原生 MLIR Vortex MatMul 模板验收

日期：2026-10-17。分支：`lg/bevformer-full-ops`。

新增 LMEM 分块、DXA 双缓冲、TCU 模板，均为 C++ MLIR Lowering。
验证执行的是 VOLT 编译出的 ELF/vxbin 和 C++ Runtime，NumPy 仅作独立参考。
设计、源码入口和运行命令见 [模板说明](../../../../native/README.vortex-matmul.md)。

## 数值与契约

| 检查 | 结果 | 原始证据 |
|---|---|---|
| 9 个用例 × 4 种 schedule | 36/36 数值通过 | [matmul-results.json](matmul-results.json) |
| FFN 投影，K tile=128，scalar/LMEM/DXA | 3/3 数值通过 | [matmul-k128-results.json](matmul-k128-results.json) |
| 非法 manifest | 21/21 在设备加载前拒绝 | [manifest-negative-results.json](manifest-negative-results.json) |
| 原有 CPU 数值回归 | 3/3 通过 | [regression-evidence.json](regression-evidence.json) |
| 原有 CUDA sm_86 PTX + ptxas | 3/3 离线编译通过，未重跑 GPU | [regression-evidence.json](regression-evidence.json) |
| TCU 指令与 LLVM 属性 | 真正 WMMA 编码及 convergent 调用存在 | [tcu-static-audit.json](tcu-static-audit.json) |

用例覆盖尾块、重复 K stage、三维投影、batched/broadcast MatMul、连续两个
MatMul＋Residual、较宽 FFN 投影、Reshape 视图，以及 NaN/Inf。
DXA 的 batched/broadcast 两个用例明确落到设备标量路径，并验证 fallback 原因；
不能把这两次执行算作 DXA 算子支持。其余支持用例检查了实际选中的模板。

有限值使用 `atol=2e-5, rtol=2e-5`，NaN/Inf 的位置和符号另由数值对照核验。
本批次最大有限值绝对误差：标量/LMEM/DXA 为 `1.1920929e-6`，TCU 为
`1.7881393e-6`。TCU 使用明确的 TF32×3 分解＋FP32 累加，不是原生 FP32 输入
矩阵乘法；非有限累加值由设备 FPU 重算。这里的误差不是任意模型的精度保证。

## 周期比较

统一配置：RV32、1 core、4 lanes、4 warps、16 KiB LMEM，DXA/TCU/TF32 开启。
配置、工具与库哈希见 [device-profile.json](device-profile.json)。
SimX、设备运行库与 `libvortex2` 使用对应配置。生成的 `.ll` 不依赖 C 预处理宏；
另以四种 tail 模板显式传入 DXA/TCU/TF32 的 `CONFIGS` 重编并执行，LLVM IR、
ELF、vxbin 哈希和周期均与原测试相同，见 [配置标志核查](config-flags-audit.json)。
计数来自整个 dispatch 序列前后的 SimX MPM 差值，包含设备端启动/同步/flush。
它是该模拟配置的周期模型，未用物理芯片或 RTL 仿真验收性能；主机 wall time
不作为推理延时。

同一输入的 FFN 投影：`[8,256] × [256,512] + Bias → ReLU`。

| 实现 | Tile | 设备周期 | 相对标量速度 |
|---|---|---:|---:|
| 标量 | 每线程一个输出 | 7,782,295 | 1.000× |
| LMEM | 4×4×16，单缓冲 | 12,878,576 | 0.604× |
| DXA | 4×4×16，双缓冲 | 8,521,022 | 0.913× |
| TCU | 8×4×4，TF32×3 | 9,092,481 | 0.856× |
| LMEM | 4×4×128，单缓冲 | 15,112,867 | 0.515× |
| DXA | 4×4×128，双缓冲 | 7,590,862 | 1.025× |

DXA K=128 的周期减少约 **2.46%**；其余上述优化方案慢于标量。
小矩阵同样存在明显额外开销，完整数据包含在报告中。分块和矩阵指令已经实际
执行，但不能据此宣称普遍加速。默认保留标量，下一阶段需减少 fragment/调用/
同步开销，并按真实 shape 和设备配置选择策略。

这里没有完整 BEVFormer 的设备执行或 30 FPS 结论；也没有把旧 Python FFN
路径或 CUDA 的性能数字移用到此次 C++ MLIR 实现。

## 完整 BEVFormer-base 编译接入

固定 cold ONNX 的三种模板均通过 C++ 导入、图优化、完整 LLVM IR 生成与验证。
三者均为 3,224 个 dispatch，图依赖、输入输出与全局 Buffer 规划一致。
命令、工具、模型及产物哈希见 [full-base-compile.json](full-base-compile.json)。

| 模板 | MatMul 采用新模板的数量 | 本轮最终产物 |
|---|---:|---|
| LMEM | 181 / 181 | 完整 LLVM IR，LLVM verifier 通过 |
| DXA 双缓冲 | 145 / 181 | 完整 ELF / vxbin，机器码构建通过 |
| TCU | 181 / 181 | 完整 LLVM IR，LLVM verifier 通过 |

DXA 有 24 个 MatMul 超出验证过的 16 位坐标范围、12 个使用批量右矩阵，明确保留
设备标量实现；3,043 个非 MatMul dispatch 不受本次模板替换影响。
全局 arena 为 1,554,944,000 字节，本次没有改变整网激活峰值规划。
整网新模板的 Vortex 数值和性能仍未执行验收；小模型数值通过不能替代这一步。

## 复现

编译器与 Runtime 使用相同 LLVM 20.1.8/VOLT 源码树；Runtime 构建时设置
`GAC_VORTEX_TCU_TF32=ON` 并使用配套 Vortex profile。模板说明中给出了构建命令。

```bash
python native/tests/run_vortex_matmul_tests.py \
  --compiler "$GAC_NATIVE_BIN/gac-compile" \
  --llvm-bin "$LLVM_PATH/bin" --runner "$GAC_NATIVE_BIN/gac-vortex-run" \
  --vortex-build "$VORTEX_BUILD" --output /tmp/gac-matmul-new

python native/tests/run_vortex_matmul_tests.py \
  --compiler "$GAC_NATIVE_BIN/gac-compile" \
  --llvm-bin "$LLVM_PATH/bin" --runner "$GAC_NATIVE_BIN/gac-vortex-run" \
  --vortex-build "$VORTEX_BUILD" --output /tmp/gac-matmul-k128-new \
  --case ffn_width --tile-k 128 \
  --schedule scalar --schedule lmem-tiled --schedule dxa-double-buffer

python native/tests/test_vortex_dispatch_plan.py \
  --runner "$GAC_NATIVE_BIN/gac-vortex-run"
```

输出目录必须全新。报告保存输入模型、编译器、Runtime、SimX 及产物的 SHA-256，
并确认测试前后编译器/Runtime 未被替换。不会上传模型权重或大型机器码产物。
