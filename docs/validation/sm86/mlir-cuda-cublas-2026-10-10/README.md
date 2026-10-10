# MLIR CUDA/cuBLAS FFN 真机验证（2026-10-10）

本次验证覆盖一个生产尺寸 BEVFormer FFN 片段：
`[1, 2500, 256] -> 512 -> 256 + ReLU + query residual + LayerNorm`。它不是完整
BEVFormer 的端到端延迟。

编译器生成 ABI v2 四调用计划：两次 FP32 MatMul 由 `cupy.matmul(..., out=...)`
调用 cuBLAS，另外两次是 NVRTC 编译的融合 Kernel。末尾 Kernel 融合 FC2 bias、
residual 和 LayerNorm，并使用每行一个 block 的归约。标准 `func`/`memref` MLIR
记录同一调用 ABI，供审阅和 `mlir-opt` 验证；Runtime 不解析该文本，而是在访问
CUDA 前确定性重编译并核对 graph、MLIR、Kernel source、call records、provenance
和 SHA-256 摘要，并在分配设备缓冲区后核对实际 tensor ABI 与 buffer alias。

## 环境与边界

- GPU：NVIDIA GeForce RTX 3080，Compute Capability 8.6，68 SM
- 显存：10,354,753,536 bytes
- CuPy：13.6.0
- CUDA Runtime：12.9（runtime API version 12090）
- CUDA Driver API version：13000
- 存储精度：FP32；cuBLAS 计算策略为 CuPy Runtime 默认值，编译器没有强制或禁用 TF32
- CUDA 对照组从相同 graph 在内存中执行旧 `cuda-sm86` codegen 和 CuPy/cuBLAS；没有做 HXP 写入/加载，包 I/O 不在计时边界内

## 命令

真机集成测试：

```bash
PYTHONPATH=src GAC_REQUIRE_CUDA=1 \
  /root/miniconda3/envs/gac-sm86/bin/python \
  -m unittest tests.test_mlir_cuda -v
```

为排除固定先后顺序带来的 GPU 热态或频率偏差，每次运行会在同一进程中同时保留
MLIR 与 CUDA 对照 Session，并在每个 repeat 后反转测量顺序。下面再用两个不同的
首轮顺序各运行一次，检查独立进程之间的稳定性：

```bash
PYTHONPATH=src /root/miniconda3/envs/gac-sm86/bin/python \
  tools/benchmark_mlir_cuda_ffn.py \
  --benchmark-order mlir-first \
  --warmup 50 --iterations 100 --repeats 20 --public-samples 50 \
  --output benchmark-mlir-first-rtx3080.json

PYTHONPATH=src /root/miniconda3/envs/gac-sm86/bin/python \
  tools/benchmark_mlir_cuda_ffn.py \
  --benchmark-order cuda-first \
  --warmup 50 --iterations 100 --repeats 20 --public-samples 50 \
  --output benchmark-cuda-first-rtx3080.json
```

## 结果

| 测量顺序 | MLIR device-resident GPU | CUDA codegen + cuBLAS | MLIR 加速 | MLIR 公共 `run` GPU / wall | 对照公共 `run` GPU / wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLIR-first | **202.906 us** | 220.313 us | **1.0858×** | 864.528 / 1699.902 us | 1013.680 / 1858.150 us |
| CUDA-first | **209.848 us** | 226.811 us | **1.0816×** | 880.480 / 1761.598 us | 1039.536 / 1955.314 us |

两份独立报告的配对平衡加速比分别为 **1.0858×** 和 **1.0816×**；对应的后端
中位数延迟分别降低约 **7.9%** 和 **7.5%**。每个 repeat 都先用一对 CUDA Event
包围一条路径的完整 FFN 调用计划并同步，再测另一条路径，下一组交换顺序；两种
测量位置各占一半。输入、权重和输出均已驻留 GPU。公共 `run` 在配对阶段之后按
后端分别测量，包含输入校验、query H2D、计算、输出 D2H 与同步；其中 GPU Event
在 H2D 和计算后结束，因此 D2H 只计入 wall 时间。

两轮 MLIR 输出都通过 NumPy 参考校验：`rtol=1e-4`、`atol=1e-5`，最大绝对误差
均为 `9.5367431640625e-7`。MLIR CUDA 专项真机测试共运行 22 项，21 项通过；远端
未配置 `mlir-opt`，因此 parser/verifier 集成项跳过。强制 CUDA 的全套回归共运行
352 项，结果为 `OK (skipped=9)`。本地 LLVM 24 已对普通和恶意字符串转义用例执行
`mlir-opt --verify-each` 并通过；本地全套回归共 331 项，结果为
[`OK (skipped=8)`](unittest-local-mlir-cublas-final.txt)。

两份报告记录的 artifact 完整性摘要相同：
`42bd41083a0d37f36d74a665fe2a56ca51c1cdd1b37d60f1711fc894dfa2b00f`。
远端目录没有 `.git`，所以报告中的 branch/commit 为 `null`；报告同时保存了完整
`src/gac` Python 包和 benchmark driver 共 57 个源文件的 SHA-256，以及 MLIR
source、tensor source、Kernel source 和 call plan 的独立 SHA-256，用于绑定实际
受测代码。

原始数据：

- [MLIR-first 报告](benchmark-mlir-first-rtx3080.json)
- [CUDA-first 报告](benchmark-cuda-first-rtx3080.json)
- [MLIR CUDA 专项真机测试日志](test-rtx3080.txt)
- [强制 CUDA 全套回归日志](unittest-full-rtx3080.txt)
- [本地 LLVM/MLIR 全套回归日志](unittest-local-mlir-cublas-final.txt)
