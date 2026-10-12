# BEVFormer FFN → MLIR → Vortex 3.0 验证记录

本目录记录 GAC 提交
`e38fff29a3b2096bbb54d0bcc2b88c63293fe3ad` 的 Vortex FFN 验证。Vortex
源码固定在 `14926ffc8755ceb664ec017028bba7f4a250096a`。三份结果均从本轮
`validation.raw.log` 和对应 `manifest.json` 重新生成，没有继承上一轮的计时或
原始日志摘要。

本轮上传到验证机的 artifact bundle SHA-256 是
`884be26ce4ae70a2192fefc349d0a05c733b419c38bbbc42407314fa686d22b5`；下载回本地的
validation bundle SHA-256 是
`0f6ec741440faec0cd36182ef967b58a02f09e8387061183b011b9916b4c4d60`。

## 环境

目标是 Vortex 3.0 RV32 build：1 core、每 core 4 warps、每 warp 4 threads，TCU
未启用。配置证据为：

- `config.mk` SHA-256：
  `99bc7ddc92371790199a0b1039ad14fd7009d7c05be464ed6508bf29140cd7d9`
- `VX_config.toml` SHA-256：
  `ae535ee43b5b6aece2f3374b971b7b558e7b016de5da15ccf338cc395d61c436`
- build `config.stamp` SHA-256：
  `1dbb1b64bdd09b6d9065762dbb7d4aa2378e1659bc9ec3fd5e6c5eecba11f3ea`
- VOLT clang：20.1.8，Vortex LLVM revision
  `b2ecfe086df2d00233a87f3d3ebe6a74aba129e2`
- MLIR 中端：LLVM 24 `mlir-opt` revision
  `765392a6b4498f1dc57233f03a31b87462fdef01`

Vortex checkout 验证时不是干净状态。原始 `git status --short` 内容是
` ? third_party/cocogfx` 和 `?? start.sh`；三份原始日志都保留了这一状态。

## 结果

| Workload | 验证结果 | SimX kernel device profile mean | SimX 端到端 mean | 最大绝对误差 | cycles / instructions |
| --- | --- | ---: | ---: | ---: | ---: |
| 合成 `[1,2,4] → 8 → 4` | warmup 1，执行 3 次，通过 | `5.405199 ms` | `5.596558 ms` | `1.1920929e-7` | `2090 / 229` |
| 官方 checkpoint 第 0 层，一行 `[1,1,256] → 512 → 256` | warmup 0，执行 1 次，通过 | `2344.678965 ms` | `2344.969531 ms` | `1.90734863e-6` | `58133 / 6025` |
| 官方 checkpoint 第 0 层，完整 `[1,2500,256] → 512 → 256` | artifact 校验、host 与 VOLT 编译通过；未运行 SimX | — | — | — | — |

两个执行用例都读取 artifact 中冻结的 `query.bin` 和 `golden.bin`，运行输出为
`finite=true`、`verified=true`，容差为 `1e-3`。真实权重来自
`verified-gacweights`，其源 checkpoint SHA-256 为
`7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12`。
这证明 FFN 算子的冻结输入数值对齐，不代表 nuScenes 数据集精度。

完整 2500-row 日志包含成功的 host/VOLT 构建、17-file artifact 校验以及
`kernel.vxbin` 摘要，但没有 `VORTEX_DRIVER` 执行记录。因此本目录不提供完整
2500-row SimX 时间，也不从 row1 时间外推它。

三个 workload 生成的 `kernel.vxbin` SHA-256 都是
`2896b0a623eecae2e3e2dd40c639c40b4ebcf33163429071004174ff422e418e`。各 workload
的 host runner 摘要不同，并分别记录在结果 JSON 和原始日志中：

| Workload | Host runner SHA-256 |
| --- | --- |
| 合成 smoke | `7f4d151e50e2945e2f8125469754e898c1f5da6bb7f7d26d3641793060bd062e` |
| 官方权重一行 | `6a3028c30441e65e5e836cd8a626e4a6e3a9a83f1383f58e01308d04d03b19cf` |
| 官方权重 2500 行 | `bb860e7db3114b15780a46f5dd2c557b1694971a8e3bfdd4f4be6a2c81bfb8e3` |

## 可追溯证据

| Workload | 结果记录 | Artifact manifest | 原始 build/run stdout |
| --- | --- | --- | --- |
| 合成 smoke | [`synthetic-smoke-simx.json`](synthetic-smoke-simx.json) | [`synthetic-smoke-manifest.json`](synthetic-smoke-manifest.json) | [`synthetic-smoke-build-and-simx.log`](synthetic-smoke-build-and-simx.log) |
| 官方权重一行 | [`official-checkpoint-row1-simx.json`](official-checkpoint-row1-simx.json) | [`official-checkpoint-row1-manifest.json`](official-checkpoint-row1-manifest.json) | [`official-checkpoint-row1-build-and-simx.log`](official-checkpoint-row1-build-and-simx.log) |
| 官方权重 2500 行 | [`official-checkpoint-production-compile.json`](official-checkpoint-production-compile.json) | [`official-checkpoint-production-manifest.json`](official-checkpoint-production-manifest.json) | [`official-checkpoint-production-build.log`](official-checkpoint-production-build.log) |

复制到本目录后的证据摘要如下；每份结果 JSON 也保存同一个 manifest/raw-log
摘要：

| Workload | Manifest SHA-256 | Raw log SHA-256 |
| --- | --- | --- |
| 合成 smoke | `346152dc3341177c74a6f124e6b672cb616fa0f2bf7e072bce4c74792d8559d6` | `3f2b136f73c91f85391badee9daa4ac2c3031fcd2c4505197dd0f652370a691b` |
| 官方权重一行 | `58dfeb7e4acf95b527a4621560c12ba758a6c61428f19f3b2410e34bcdd04ef2` | `4672fae0011f6646322bdb69c5834f311c661dd342b228322e188ef58e011f11` |
| 官方权重 2500 行 | `43be04a94fb68a527b7cee0328a90c4c86daeb8d57393863089707d7122379d1` | `7e67804d7ef29ea45392851fc4bc5ab9360a040b6e7d8f8992ce08c90d25a8e7` |

原始日志还包含 GAC/Vortex commit、dirty status、VOLT revision、目标配置、三份
MLIR 摘要、`verify_artifact.py` 输出、实际编译命令、host/kernel 摘要、两份执行
JSON 和 SimX PERF 行。manifest 保存两级 MLIR、标准 `func.call` Runtime plan、
六组外部权重和 17 个 payload 文件的摘要。

## 解释边界

这些证据证明当前 BEVFormer FFN 切片完成了 Canonical Graph → 两级标准 MLIR →
受限 Runtime-call lowering → Vortex C++/VOLT → `.vxbin` → SimX 的正确性闭环。
SimX 是功能/指令级模拟器；表中的毫秒数不能作为 FPGA/ASIC 性能，不能换算 FPS，
也不能与 RTX 3070 CUDA Event 延时直接比较。

这仍然只是一个 Encoder FFN。完整 BEVFormer 的 backbone、FPN、
MSDeformableAttention、其余 Encoder/Decoder、检测头和时序状态没有迁移到 Vortex；
本目录不包含完整模型端到端延时。
