# BEVFormer-base：原生 MLIR → CUDA PTX 完整固定图验证

2026-10-17，**完整 BEVFormer-base 的固定输入、FP32、cold-frame 原始输出路径已在 RTX 3070 Ti Laptop GPU 执行全部 3,224 个 dispatch，三路输出与 PyTorch 参考比对通过**。这里验证的是下述明确导出边界，不包含真实数据集检测精度、连续历史帧或最终检测框解码。

首轮没有 warmup，CUDA Event 测得整个 dispatch 序列为 **17,050.46 ms（约 17.05 秒）**；随后预热 1 次、测量 3 次，平均为 **18,047.63 ms（约 18.05 秒）**。这说明当前实现已经可以正确执行，但**尚未满足每帧 ≤33.3 ms 的 30 FPS 目标**。少量测量不代表长时间稳定吞吐率。

## 模型、输入与参考边界

导出使用上游 BEVFormer 的实际 PyTorch 模块：ResNet-101/DCNv2、4 级 FPN 特征、200×200 BEV 和 6 层 Encoder。Checkpoint 严格加载 1,089 个状态 Tensor。并非从本项目 Graph 重建一个同名替代模型。

- 输入为固定 `images.npy`，FP32，shape **`[1, 6, 3, 928, 1600]`**。导出脚本生成固定随机输入；本次不是实拍数据集图像。
- `camera_batch_chunk=1`：Backbone/FPN 在固定 eval BatchNorm 统计量下，逐相机执行，再按相机顺序拼接。不能把这个结果当作六相机同时并行的调度性能。
- 标定元数据为 **synthetic fixed all-visible projection**，CAN-bus 全零。几何和可见性被固定；未验证任意相机标定或可见性模式。
- 输入不含历史 BEV，执行 cold-frame 路径。图像归一化、padding 和最终框解码在导出边界之外；eval 下的 GridMask 为 identity，导出时省略。
- 导出 ONNX opset 17，含 **22,412 个节点**，其中 `mmcv::MMCVModulatedDeformConv2d` 156 个，`gac::MSDeformAttn` 18 个。

参考来自上游 forward 在 ONNX tracing 期间的真实执行，并明确使用导出适配器：DCNv2 参考调用 torchvision 的 CPU deform-conv 算子；MSDA 参考由 PyTorch GridSample 和归约计算。相关算子合同测试另有记录。**此次没有独立执行“应用 tracer 兼容修改前后”的两次完整模型比较**，不能扩大为未经适配的原始 PyTorch 程序全输入等价证明。

完整导出约束、checkpoint/config/source 哈希、输入与参考身份见 [export-reference-contract.json](export-reference-contract.json)。

## 数值结果

比较容差为 `rtol=1e-3, atol=1e-4`，逐元素检查 `abs(actual-reference) <= atol + rtol*abs(reference)`。三路输出 dtype/shape 一致、全部有限，超差元素均为 0。

| 输出 | Shape | 最大绝对误差 |
| --- | --- | ---: |
| `bev_embed` | `[40000, 1, 256]` | `2.1189451e-5` |
| `all_cls_scores` | `[6, 1, 900, 10]` | `2.2888184e-5` |
| `all_bbox_preds` | `[6, 1, 900, 10]` | `6.1765313e-5` |

被测路径为 `ONNX → C++ Importer → MLIR Operations/Passes → 设备 Lowering → LLVM/NVVM → NVPTX → PTX → CUDA Driver JIT → GPU`。Runtime 仅管理 Buffer、提交 Kernel 和传输结果，没有 CPU/Python 算子回退。Python 在参考机读取返回文件并比较，GPU 测试机不需要 NumPy。

证据：[first-run-validation.json](first-run-validation.json)。该记录包含实际 Runtime 身份、3224 个已完成 dispatch、输出误差、返回文件哈希及参考哈希。归档时重新核验了返回文件的 SHA-256。

## 首轮计时

设备为 **NVIDIA GeForce RTX 3070 Ti Laptop GPU**，Compute Capability 8.6；Runtime 报告 CUDA Driver API 版本 `13000`。此轮 `warmup_runs=0`、`measured_runs=1`、`trace_dispatches=false`。

| 测量项 | 时间 | 口径 |
| --- | ---: | --- |
| CUDA dispatch 序列 | **17,050.46 ms** | CUDA Events；含设备计算和可能的主机提交空隙 |
| 初始化 | 17,358.99 ms | 含 manifest、设备/context、模块加载/JIT、Buffer/函数准备，不能全部称为纯 JIT 时间 |
| 输入及常量上传 | 58.88 ms | Host API 墙钟，包括上传完成等待 |
| 输出下载 | 5.65 ms | Host API 墙钟 |
| C++ Runtime 总墙钟 | 34,664.64 ms | 本次进程中的初始化、传输、计算、文件 I/O 等；不是稳态每帧延时 |

CUDA Event 时间不含初始化/JIT、传输、文件 I/O；因此也不是包括摄像头预处理和检测后处理的端到端应用延时。当前 Kernel 采用输出元素线程并行、串行归约和全局内存访问，没有宣称使用 cuBLAS、Tensor Core 或共享内存分块优化。

## 预热后的三次测量

使用同一部署包、输入和 Runtime，再次启动进程，预热 1 次后连续测量 3 次；没有逐 dispatch 同步，也没有重新上传每一帧输入。

| 测量 | CUDA dispatch 序列时间 |
| --- | ---: |
| 第 1 次 | 17,468.20 ms |
| 第 2 次 | 17,884.04 ms |
| 第 3 次 | 18,790.65 ms |
| **平均** | **18,047.63 ms** |

这些 CUDA Event 数字排除初始化/JIT、upload/download、文件 I/O 和预热。测试条件及三次样本见 [warmed-benchmark.json](warmed-benchmark.json)。初始化在本次进程中为 1,926.80 ms，单独记录，未算入上述平均值。

测量结束后保存的三路输出，其字节数与 SHA-256 **均与首轮已经通过参考比对的结果一致**。前两次测量未分别保存输出，不声称逐次单独完成了数值比对。

`warmup=1` 仅表示预热同一个固定 cold-frame 图，**不表示启用了 BEVFormer 的 warm/history 连续帧模式**。

## 内存口径

| 静态存储项 | 字节数 |
| --- | ---: |
| 输入 | 106,905,600 |
| 独立常量 Tensor | 368,518,572 |
| 复用后的 arena | 1,554,944,000 |
| 显式申请张量 Buffer 合计 | **2,030,368,172** |
| 内嵌常量表预算，另外计入模块 | 2,073,600 |

外部 `nvidia-smi` 监控在约 1 秒间隔的 **36 个采样点**中，观察到本进程最大占用 **2160 MiB（约 2.11 GiB）**。这是采样到的最大值，不是保证捕获全部瞬时分配的精确峰值。监控摘要见 [first-run-memory-monitor.json](first-run-memory-monitor.json)。

Runtime 的 `cuMemGetInfo` 是整个设备的时点快照，可能包含其他进程和驱动。静态 Buffer 合计不包含 CUDA context、模块/JIT 和可能的 local-memory spill；这三种内存指标不能相互替代。

## 编译产物与复现

编译器使用 LLVM/MLIR **20.1.8**，CUDA 目标 `sm_86` / `+ptx80`，数学函数链接 NVIDIA libdevice。3224 个 PTX 入口、launch geometry、SSA 依赖、arena 生命周期和常量文件尺寸均通过审计；`ptxas` 离线汇编通过。**实际执行加载的是 PTX，由 CUDA Driver JIT；cubin 是独立离线汇编检查产物。**

| 文件 | 内容 |
| --- | --- |
| [artifact-audit.json](artifact-audit.json) | 完整模型算子计数、设备资源与 PTX/cubin/manifest 身份 |
| [compile-provenance.txt](compile-provenance.txt) | 工具链版本和编译器/libdevice 哈希 |
| [export-reference-contract.json](export-reference-contract.json) | 输入、导出适配、参考边界和源文件身份 |
| [first-run-validation.json](first-run-validation.json) | 完整 GPU 执行、计时和三路数值比较 |
| [warmed-benchmark.json](warmed-benchmark.json) | 预热后 3 次计时及最终输出与已验证结果的哈希一致性 |
| [provenance.json](provenance.json) | 原始报告、监控样本、验证源码和归档文件哈希 |

归档仅保存文本证据，不包含模型权重、输入/输出数组或 PTX 本体。执行和比较入口见 [`native/tests/run_cuda_tests.py`](../../../../native/tests/run_cuda_tests.py) 与 [`README.cuda.md`](../../../../native/tests/README.cuda.md)。62 个独立小用例的真实 GPU 验证另外保存在 [算子验证记录](../native-mlir-2026-10-17/README.md)。
