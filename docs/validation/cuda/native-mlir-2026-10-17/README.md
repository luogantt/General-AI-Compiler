# Native MLIR → CUDA PTX 算子验证（2026-10-17）

本记录证明 **62 个小型用例经过原生 C++ MLIR 编译、PTX 生成及 NVIDIA GPU 执行，73 路输出全部通过独立参考比对**。这份算子报告不代表完整 BEVFormer-base 已通过 GPU 验证，也不代表达到 30 FPS。

## 编译及执行链

```text
ONNX → C++ importer → GAC MLIR Operations / Passes
     → 设备任务、线程映射与静态 Buffer 规划
     → LLVM / NVVM → LLVM NVPTX → PTX
     → CUDA Driver JIT → cuLaunchKernel → GPU
```

- 编译工具链：LLVM/MLIR **20.1.8**，目标 `sm_86`，PTX 特性 `+ptx80`。
- 62 个用例均生成 PTX，并通过 `ptxas` 离线汇编；离线汇编本身不算数值验证。
- 实际设备：**NVIDIA GeForce RTX 3070 Ti Laptop GPU**，Compute Capability **8.6**。
- Runtime 报告 CUDA Driver API 版本为 `13000`。
- 执行方式：单 CUDA Stream，按 manifest 顺序提交；每个输出元素分配线程，归约使用串行循环。此阶段没有声称采用 Tensor Core、shared-memory tiling 或双 Buffer。
- 编译与 GPU 执行均无 Python/CPU 算子回退。Python 仅负责数据打包、读取预先生成的独立参考、比较结果。

编译器从独立只读目录运行，编译前后 `gac-compile`、`gac-opt` 哈希一致。此前被正在重链接的编译器中断的试跑不计入本记录。

## 62 个用例的范围

| 夹具集 | 数量 | 检查内容 |
| --- | ---: | --- |
| `native-tests` | 6 | MatMul + Bias + ReLU、多消费者、Conv + ReLU、Residual Add + ReLU、Softmax、形状链 |
| `native-base-ops-tests` | 14 | DCNv2、分组和偏移边界、FPN Resize、MSDA、多分支静态内存复用 |
| `native-msda-offsets-tests` | 16 | MSDA 坐标融合与不能融合的形式、多消费者及公开中间输出 |
| `native-view-alias-tests-v2` | 12 | Reshape/Squeeze/Unsqueeze、别名生命周期、公开 view、Transpose、FP16 Cast 往返 |
| `native-scatter-legalization-tests` | 14 | Scatter 合法化、不同索引顺序、负索引、部分更新及多阶段分支 |
| **合计** | **62** | **73 路输出全部通过** |

本组用例以 FP32 为主。FP16 Cast 往返通过不能据此推断完整 FP16 模型或 FP16 MatMul 已验证。完整模型、历史帧以及数据集检测精度需要各自的记录。

## 数值判据与证据

- 浮点容差统一为 `rtol=2e-5`、`atol=2e-5`，逐元素判断 `abs(actual-reference) <= atol + rtol*abs(reference)`。
- 输入、参考和输出名称、shape、dtype 必须吻合；NaN/Inf 不作为通过；整数/布尔输出采用精确比较。
- 全部输出中最大绝对误差为 **`1.9073486328125e-6`**；失配元素总数为 **0**。
- 部署时记录 PTX、manifest、权重和输入的 SHA-256，远端执行前验证这些文件；本地比对前核验参考及 manifest 身份。
- `gpu_passed` 必须同时满足真实 CUDA Runtime 完成全部 dispatch、没有 host operator fallback，以及全部输出通过数值比较。

| 文件 | 内容 |
| --- | --- |
| [operators-compile.json](operators-compile.json) | 编译器身份、62 个模型/IR/PTX/cubin/输入/参考/常量的哈希及静态 arena 大小 |
| [operators-gpu-validation.json](operators-gpu-validation.json) | Runtime 身份、实际设备、完成任务数、逐输出误差及输出哈希 |
| [provenance.json](provenance.json) | 原始报告、验证代码以及归档 JSON 的哈希 |

这些归档仅含元数据、误差与哈希，不包含权重、输入数组、输出数组或 PTX 内容。

## 计时和内存口径

每个小用例执行 **1 次 warmup、5 次测量**，没有开启逐 dispatch 同步的 trace 模式。报告中的 CUDA Event 时间覆盖整个 dispatch 序列，包含 GPU 工作和可能的主机提交空隙；不含 JIT、传输、文件 I/O 或 warmup。它不是完整 BEVFormer 延时，也不是 62 个小用例相加后的整网延时。

`static_allocated_tensor_bytes` 是显式申请的张量 Buffer 总量。`cuMemGetInfo` 记录的是设备在几个时点的空闲/已用内存，可能包含其他进程和驱动分配；**二者都不是本进程经过连续测量得到的显存峰值**。CUDA context、模块/JIT、内嵌常量及可能的 local-memory spill 不应混入静态 Buffer 量的定义。

## 复现入口

通用编译及 GPU 数值验证工具为 [`native/tests/run_cuda_tests.py`](../../../../native/tests/run_cuda_tests.py)，使用说明见 [`README.cuda.md`](../../../../native/tests/README.cuda.md)。测试机没有 NumPy 时，本次部署先用标准库脚本调用 C++ Runtime，再将原始结果下载到参考机，调用同一 `compare_outputs` 函数比对；该流程没有在远端执行参考模型。

完整 BEVFormer-base 的固定 cold-frame GPU 验证已另行归档于 [完整模型记录](../native-mlir-bevformer-base-2026-10-17/README.md)，其中单独标注模型、输入、参考来源、误差和计时范围；不与本目录的 62 个小用例混算。
