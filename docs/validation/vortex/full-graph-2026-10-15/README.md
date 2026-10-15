# BEVFormer 整图 Vortex 验证：2026-10-15

分支：`lg/bevformer-vortex-full-ops`。设计和十层代码映射见
[整图后端说明](../../../bevformer_vortex_full_graph.md)。

## 验证结论与范围

本轮验证从固定 FFN 扩展到现有完整 BEVFormer streaming Graph。网络拓扑包含
backbone/FPN、Encoder、时空 Attention、Decoder、原始检测头和历史状态。

| 项目 | 结果 | 证据 |
|---|---|---|
| 规范算子覆盖 | 33 类全部生成并执行软件 Kernel | [`local-summary.json`](local-summary.json) |
| 完整拓扑小尺寸模型 | 1,053 个网络节点，加 4 个独立算子覆盖探针；优化后 811 次 dispatch | 同上 |
| MLIR 编译 | 真正经过 SCF/Math/Arith/CF → LLVM Dialect → LLVM IR → VOLT → ELF → vxbin | [`local-build.log`](local-build.log) |
| 本机机器码数值验证 | 冷帧、回传历史帧、受控历史探针通过 | `local-summary.json` 的 native |
| WSL Vortex SimX | 同样三条路径通过 | [`local-cold-correctness.json`](local-cold-correctness.json)、[`local-warm-correctness.json`](local-warm-correctness.json)、[`local-history-route-probe-correctness.json`](local-history-route-probe-correctness.json) |
| 用户新机器 Vortex SimX | 三条路径全部通过，9/9 输出通过 | [`remote-summary.json`](remote-summary.json) |
| 真实 checkpoint、生产 Profile | 929 次 dispatch，编译/链接/封装成功；未执行生产尺寸推理 | [`production-build.json`](production-build.json)、[`production-compile.log`](production-compile.log) |
| 全仓库回归 | 651 项运行，无失败，32 项跳过 | [`local-unit-tests.log`](local-unit-tests.log) |
| 最终整图定向回归 | 7 项通过，包含标量输出 checker 的正确/错误数据检查 | [`final-graph-tests.log`](final-graph-tests.log) |

33 类包括 Conv2D、BatchNorm、MaxPool2D、MatMul、MSDeformableAttention、
GridSample2D、Softmax、LayerNorm，以及完整图需要的逐元素、布局、归约和精度转换。
额外四个探针用于 Identity、GELU、AddReLU、MatMulBiasGELU；它们不是擅自添加到
BEVFormer 网络结构中的层。另暴露两个历史路径中间张量作为验证输出。

## 历史状态验证方法

1. 冷帧用 `has_prev_bev=0` 执行，读取设备实际产生的 FP16 `next_bev`。
2. 下一次调用把该张量作为 `prev_bev`，设置 `has_prev_bev=1`，验证历史路径。
3. 单独注入 `arange + 3` 的非零 FP16 历史，并把增量旋转角设为零，验证
   `selected_prev_bev` 精确等于注入历史，防止随机小权重让错误路径被容差掩盖。

每次运行前移除已知输出文件，避免把上一次结果误认为新结果。WSL 主验证的 oracle
使用原始未优化 Graph 的 NumPy 语义，独立检查常量折叠与执行结果。远端还把输出
下载到本机再次比较。Runtime 不读取 golden，不执行 NumPy、CPU 或 CUDA 算子。

这证明两次调用之间通过 Host 反馈状态的正确性。当前远端 runner 每次进程重新打开
设备；未把它宣传为常驻跨帧 Session 或设备内部零拷贝状态链。

## 远端配置及误差

- 机器：`robot@192.168.115.128`，Ubuntu 22.04。
- Vortex：`/home/robot/project/vortex`，revision
  `14926ffc8755ceb664ec017028bba7f4a250096a`。
- 配置：RV32、1 core、4 warps、4 threads；不依赖 TCU 或 DXA。
- 产物：`/home/robot/project/vortex/build/tests/regression/gac_full_ops_20261015`。
- 工具：同 revision 的 VOLT LLVM 20.1.8；远端没有 MLIR 工具，设备机器码在 WSL
  编译，Host C++ 在远端重新编译以匹配 glibc。

所有远端帧的最大绝对误差为 `7.62939453125e-6`。冷帧 `next_bev` 与参考逐元素一致，
历史路径探针也精确一致。详细环境/ABI/Runtime 摘要见
[`remote-environment.json`](remote-environment.json)，命令见
[`remote-reproduce.md`](remote-reproduce.md)。

日志中约 3–5 秒的数字是 **SimX 在 Host 上的仿真耗时**，不能用来换算模型 FPS，
也不能当作 FPGA/ASIC 推理延时。

## 真实 checkpoint 的编译验证

输入是本地 `build/checkpoints/bevformer_tiny.gacweights`，源 checkpoint SHA-256：

```text
7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12
```

生产 Profile 的规划值：arena `516096000 B`，resident constants `136079360 B`，
external inputs `15104460 B`，external outputs `1352000 B`。
`.vxbin` 为 `193047 B`；MLIR、LLVM IR、ELF、vxbin 的摘要保存于
`production-build.json`。这些信息只证明真实权重/生产 Shape 的编译链可用。

尚未进行生产尺寸输入数值对齐、nuScenes mAP/NDS、真实 FPGA 执行或硅片性能测量。
Host 的 NMS-free TopK/阈值筛选/框包装仍在网络输出边界外。

## 复现

本地完整编译与仿真：

```bash
cd /mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler
PYTHONPATH=src python3 tools/validate_vortex_full_graph.py \
  --output /new/validation-directory \
  --llvm-path /home/thomas/gac-volt-llvm20/build-release \
  --vortex-build /home/thomas/vortex-gac-buffered-build32 --all-ops
```

生产 Shape 只编译：

```bash
PYTHONPATH=src python3 tools/build_vortex_graph.py \
  --bevformer-weights build/checkpoints/bevformer_tiny.gacweights \
  --mlir-opt /home/thomas/gac-volt-llvm20/build-release/bin/mlir-opt \
  --output /new/production-artifact
cd /home/thomas/vortex-gac-buffered-build32
make -C /new/production-artifact all \
  ROOT_DIR=/home/thomas/vortex-gac-buffered-build32 \
  LLVM_PATH=/home/thomas/gac-volt-llvm20/build-release \
  CONFIGS=-DVX_CFG_EXT_DXA_ENABLE
```

本机基线驱动启用了 DXA，所以编译参数与驱动保持一致；新通用 Kernel 本身没有
使用 DXA 指令。远端不启用 DXA 的驱动同样通过，证明这不是隐藏的硬件依赖。
