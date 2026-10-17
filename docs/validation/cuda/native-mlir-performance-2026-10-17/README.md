# 原生 CUDA PTX：性能优化与整网复验

本轮在同一条 **ONNX → C++ MLIR Pass/Lowering → NVVM/NVPTX → PTX → CUDA Driver**
路径上增加计算模板和 profiling，没有用 cuDNN/cuBLAS 或 Python 算子替换设备计算。
优化组合已完整执行 BEVFormer-base 的 3,224 个 dispatch，三个输出与 PyTorch 参考对齐。

设备为 RTX 3070 Ti Laptop 8 GB，SM86。**30 FPS 尚未达到。** 最终组合
`auto MatMul + cooperative reduction + Conv channel8` 的四次设备序列为
**6.426–6.439 秒，中位数 6.430 秒**。同轮标量为 18.279–45.033 秒，存在明显
温度/频率变化，不能将两者样本比值当作受控固定频率加速比。
最初的 tile32/channel4 组合和全部较慢数据也保留在下文。

## 改动与代码位置

| 改动 | 实现 | 验证范围 |
|---|---|---|
| MatMul 共享内存与寄存器分块 | [CudaMatmul.cpp](../../../../native/lib/CudaMatmul.cpp) | FP32；A/B 协作加载、尾块、batch 广播、Bias/ReLU；每线程多个输出 |
| Softmax/LayerNorm 协作归约 | [CudaReduction.cpp](../../../../native/lib/CudaReduction.cpp) | 一个 CTA 一行；非末轴 Softmax；LayerNorm 两遍中心化方差 |
| Conv/DCNv2 通道分块 | [DeviceLowering.cpp](../../../../native/lib/DeviceLowering.cpp) 的 `emitConvolutionOutputs` | 同一输入样本复用于多个输出通道，块不能跨 group；保留标量 K 次序 |
| 逐 dispatch 耗时与资源 | [cuda-run.cpp](../../../../native/runtime/cuda-run.cpp) | 普通计时与 Event profiling 分开；查询寄存器、静态 shared、local memory |
| 调度与存储契约 | [dispatch-plan.h](../../../../native/runtime/dispatch-plan.h) | shape、广播、launch、完整 warp、shared memory、group 与 buffer 校验 |

参数选项见 [CUDA 后端说明](../../../../native/README.cuda.md)。默认仍为标量基线，
优化通过显式选项启用；小形状可能更慢，因此不对全部算子强制使用同一模板。
`auto` MatMul 是形状启发式，不是实测 autotuning。

## 整网实验的固定边界

沿用 [导出与参考契约](../native-mlir-bevformer-base-2026-10-17/export-reference-contract.json)：
真实 checkpoint、R101/DCNv2、四级 FPN、200×200 BEV、六层 Encoder，
固定随机图像 `[1,6,3,928,1600]`、固定合成标定、cold frame、raw-head 输出。
不包含预处理、检测框后处理、历史帧或真实数据集检测指标。

第一组在 GPU 完整验证的组合为：

```bash
export GAC_CUDA_MATMUL_SCHEDULE=shared-tiled
export GAC_CUDA_TILE_M=32 GAC_CUDA_TILE_N=32 GAC_CUDA_TILE_K=16
export GAC_CUDA_THREAD_TILE_M=2 GAC_CUDA_THREAD_TILE_N=2
export GAC_CUDA_REDUCTION_SCHEDULE=cooperative
export GAC_CUDA_CONV_CHANNEL_TILE=4
export GAC_CONSTANT_FOLD_PROFILE=bevformer-base
# 其余 LLVM、libdevice、SM86 环境设置见 native/README.cuda.md。
bash native/tools/compile-cuda.sh model.onnx build/model-cuda-optimized
```

该组合选择 666 个通道分块 dispatch、181 个 MatMul 分块、73 个协作归约；
其余 2,304 个保持原路径。图的输入/输出、SSA 依赖和全局 buffer 规划完全相同，
859 个外置常量文件逐个核对 SHA-256 一致。arena 仍为 1,554,944,000 字节。
见 [产物审计](full-artifact-audit.json)。优化没有删掉模型计算或缩小输入。

另已编译并在 GPU 执行 `MatMul=auto + Conv 通道块8 + cooperative` 的完整候选，见
[第二候选产物审计](full-auto-artifact-audit.json)。选中 144 个 MatMul 分块、666 个通道分块、
73 个协作归约，其余 2,341 个保持标量；小 MatMul 不再一律分块。
该候选三个输出与第一候选逐字节一致，从而满足同一份 PyTorch 参考检查，见
[输出哈希及运行记录](auto-output-validation.json)。首次普通测量为 **4,201.34 ms**，
初始化/JIT 等开销另计；重复对照结果与第一组分开保存。使用第二候选时，将上面的
`GAC_CUDA_MATMUL_SCHEDULE` 改为 `auto`、`GAC_CUDA_CONV_CHANNEL_TILE` 改为 `8`。

## 数值对齐

每个输出逐元素检查 `abs(actual-reference) ≤ 1e-4 + 1e-3*abs(reference)`，
dtype/shape 一致、全部有限、超差元素为零。

| 输出 | 最大绝对误差 |
|---|---:|
| `bev_embed` | 2.0623207e-5 |
| `all_cls_scores` | 2.0027161e-5 |
| `all_bbox_preds` | 5.9500337e-5 |

[输出验证记录](whole-model-output-validation.json) 保存输入、模型、PyTorch 参考和返回文件的哈希。
重复计时每个进程结束时保存的三路输出，均与相应版本已独立验证的输出哈希一致。
没有逐帧保存每个中间测量的输出，因此不声称逐帧进行了独立参考比较。

## 最终三版本对照

GPU 恢复连接后，用同一最终 Runtime 按 **标量 → tile32/channel4 → auto/channel8 →
auto/channel8 → tile32/channel4 → 标量** 的顺序执行。每个进程预热 1 次、测量 2 次，
每种共 4 个普通计时样本；profiling 关闭，所有进程最终输出通过哈希核对。

| 版本 | 中位数 / ms | 最小—最大 / ms |
|---|---:|---:|
| 标量基线 | 30,948.71 | 18,279.09—45,032.62 |
| tile32 + channel4 + cooperative | 9,566.70 | 8,371.33—11,228.71 |
| auto + channel8 + cooperative | **6,430.37** | **6,426.01—6,438.53** |

最终组合在本轮的四个样本均更快，且三个输出完全一致。相对第一组优化的样本
中位数比值为 1.49，相对标量为 4.81；**后一个比值明显混入了运行环境变化，不能
直接宣称编译器稳定加速 4.81×**。最后一组标量的时点 SM 频率由 915 降至 585 MHz，
两次样本约 40.79、45.03 秒；开始的两次标量为 18.28、21.11 秒。
时点温度最高 89°C。`power.draw` 曾返回不合理的 752 W，原值保留但不用于功耗分析。

[最终顺序平衡报告](whole-model-balanced-final.json) 包含命令、全部样本、工具/产物/输出
哈希以及 GPU 前后状态。四个样本不是持续运行可靠性验证，更不是包括预处理、
数据上传和框后处理的应用 FPS。`warmup` 只是重复固定 cold 图，不是 BEV 历史帧模式。

## 第一组组合的重复计时与温度影响

计时使用同一个新 Runtime 加载标量/优化产物，单 stream、无 trace、无逐 dispatch 等待。
顺序为 **标量 → 优化 → 优化 → 标量**；每个进程 warmup 1 次、测量 3 次，共每种 6 个样本。
CUDA Events 包围整个 dispatch 序列，包含可能的主机提交间隙，排除初始化/JIT、上传、下载。

| 本轮数据 | 标量 | 优化组合 |
|---|---:|---:|
| 中位数 | 24,098.54 ms | 15,830.36 ms |
| 均值 | 25,154.15 ms | 16,282.35 ms |
| 最小—最大 | 20,491.74—33,705.36 ms | 13,369.75—19,338.64 ms |

观察到的中位数比值约 **1.52×**，延时降低约 34.3%。但这不是锁定频率的受控加速比：
GPU 时点温度从 59°C 升到 88–89°C，SM 时钟出现 495–630 MHz 的低频时点。
记录只是各进程前后采样，不能重建每个 Kernel 的实时频率或把全部波动归于单一原因。
原始样本、设备身份、温度/频率和最终输出哈希见 [配对测量](whole-model-paired.json)。

更早同轮的各一次普通测量是标量 **18,573.28 ms**、优化 **6,131.02 ms**。
它们对应后续 profile 的普通计时阶段，初始化另计；只代表这两次执行，不作为稳定
3× 加速结论，也不能用历史 18.05 秒除以新版本任意最快样本来代替配对实验。

## 热点归因

`--profile-dispatches` 在普通计时之后另执行带逐 dispatch Event 的序列。
该测量会增加指令提交/Event 开销，且设备状态可能变化；只用于找热点，
不能把逐算子 profile 求和代替上面的普通计时。

标量 profile 中 DCNv2 约占 **58.5%**，普通 Conv 与 Conv+ReLU 合计约 **29.2%**，
MatMulBias 约 **7.1%**。因此本轮除 MatMul 外，也优化了 Conv/DCNv2 的输入采样复用。
优化后同类 profile 中 DCNv2 仍约占 **48.1%**，仍是后续重点。
详见 [两轮热点与资源摘要](whole-model-profiles.json)。

当前查询到的 `local_bytes_per_thread` 不是实际 spill 访问次数；本轮没有采集
DDR 带宽、bank conflict、stall 或实际 occupancy 计数。没有把非零 local allocation
直接解释成寄存器溢出瓶颈。

## 验证记录与下一步

- [小算子验证](README.kernels.md)：MatMul、Conv/DCNv2、协作归约、形状选择的独立覆盖与局部性能。
- [CUDA manifest 校验](cuda-parser-validation.json)：58 个用例；这是解析/拒绝验证，不执行 GPU Kernel。
- [共享标量语义回归](host-regression.json)：31 个 native 测试，包括合法化/拒绝和 CPU 机器码数值检查；不计作 GPU 验证。
- [首轮编译源码身份](cuda-source-identity.json)：绑定首次已测优化产物的源文件快照；后续 auto/Vortex 改动另行记录，不能冒充同一快照。
- [最终源码与重建核对](final-source-and-rebuild.json)：最终编译器以第一组参数重新编译完整模型，PTX 与 manifest 都与已测版本逐字节相同；保存最终源码和工具哈希。

当前仍未实现原生整网 CUDA Tensor Core、异步双缓冲、实测 autotuning、
Conv 的 shared-memory 空间分块或隐式 GEMM。接下来应先在可稳定访问和散热的设备上
完成候选的持续运行验收，再继续优化 DCNv2/Conv。Vortex 的寄存器分块与 TCU 调用开销实验
单独记录，不能将 SimX 周期与这里的 GPU 毫秒直接比较。
