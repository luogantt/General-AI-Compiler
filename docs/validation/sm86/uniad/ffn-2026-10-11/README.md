# UniAD encoder FFN：RTX 3080 / SM86 验证

本目录保存 `2026-10-11` 对一个 UniAD 2.0 stage2 BEV encoder FFN 的原始成对基准。测试形状是生产配置的 `[1, 40000, 256] -> 512 -> 256`，精度为 FP32，设备为 NVIDIA GeForce RTX 3080（Compute Capability 8.6）。权重 bundle 由迁移前的 `tools/uniad_compiler/make_synthetic_bundle.py` 使用 seed `20261011` 生成，SHA-256 为 `551788d2ed648a87fb884b63bd95b0b969bcff740d5d8a14b310a1514b708d5f`；基准输入使用同一个 seed。GPU 输出以 GAC NumPy reference 验证。测试没有使用 UniAD checkpoint 或 nuScenes，因此结果只代表这个 FFN 子图。

这是迁入 General-AI-Compiler 之前生成的历史证据。两个 JSON 的 source manifest 绑定当时的 UniAD adapter 文件和 GAC commit `c494d9e0a5bb841ea131d3397fb42691daf3f8f9`；为保持证据完整性，迁移时没有改写 JSON 或 `SHA256SUMS`。它可以证明迁移前实现的真机结果，不能冒充当前分支的新性能运行。迁移后的工具使用 `python tools/uniad_make_synthetic_bundle.py ...` 和 `python tools/uniad_benchmark_ffn.py ...`；当前分支的新真机报告应另建目录保存。

| 独立进程的首次顺序 | MLIR resident GPU | CUDA/HXP + cuBLAS resident GPU | MLIR 加速比 | MLIR - CUDA |
| --- | ---: | ---: | ---: | ---: |
| MLIR first | 1635.330 µs | 2074.368 µs | 1.26440× | -432.311 µs |
| CUDA first | 1645.388 µs | 2075.340 µs | 1.26061× | -428.928 µs |

两次运行的 MLIR 和 CUDA 输出都通过 `rtol=1e-4, atol=1e-5`；最大绝对误差均为 `9.5367431640625e-7`。CUDA 基线先写入、校验并重新加载同一个 HXP（SHA-256 `11dd77bdfad5a07d42dd1601c980bed118c21c6a08b4ba651377e40311574a2f`），两个 MatMul 均由 CuPy/cuBLAS 执行，整个基线有 6 次 dispatch。MLIR 路径有 4 次调用：两个 cuBLAS MatMul、融合 Bias+ReLU，以及融合 Bias+Residual+LayerNorm。

每个独立进程先 warmup 10 轮，再进行 10 个成对 repeat；每个 repeat 内执行 20 次 FFN。两个后端在每个 repeat 交换先后位置，前一条流完成并同步后才测另一条。表中加速比是 MLIR-first 位置和 CUDA-first 位置各自中位数比值的几何平均数。`public run` 的 H2D/D2H 结果也保存在 JSON 中，但其约 34 ms 的 wall time主要由约 41 MB 输入和 41 MB 输出的主机传输及同步决定，不用于上表的计算性能结论。

当前 MLIR 文件是可审查的 ABI manifest；实际执行由经过完整性校验的 call records 驱动，融合 kernel 由 NVRTC 编译。它不是整段 MLIR GPU -> NVVM -> PTX 执行。完整调用、环境、逐 repeat 样本和全部源文件 SHA-256 位于：

- `benchmark-mlir-first.json`
- `benchmark-cuda-first.json`
- `SHA256SUMS`

这些数字不能换算为完整 UniAD 帧率，也不能替代真实 checkpoint、数据集精度或 RTX 3070 实测。
