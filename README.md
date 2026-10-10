# General AI Compiler（GAC）

> 面向边缘推理的 AI 编译器与 Runtime 原型；当前首条完整路径是固定 Profile 的 BEVFormer-tiny raw step 和 CUDA SM86。

GAC 把模型配置、权重和静态计算图转换为可校验的 HXP 包，在 Runtime 中完成设备检查、显存规划、实现选择、NVRTC 编译和 CUDA 执行。逐元素、布局、Attention 与 BEV 专用算子由项目生成 CUDA；生产尺寸 FP32 Conv 和 MatMul 可分别选择 cuDNN 与 CuPy/cuBLAS。

```text
JSON graph / BEVFormer config + checkpoint
                    │
                    ▼
      Frontend → Canonical Graph
                    │  shape / dtype / layout validation
                    ▼
   graph passes → fusion → static memory plan
                    │
                    ▼
 Generated CUDA + vendor candidates + metadata
                    │
                    ▼
        versioned, validated HXP package
                    │
                    ▼
 CudaSession → NVRTC / cuDNN / CuPy-cuBLAS → raw outputs
                    └────────────────────────→ trace / profile
```

当前 GPU 路径不会在 CUDA 缺失时静默改用 CPU。HXP 中的 `implementation_candidates` 供 Session 建立时选择；vendor 调用执行中失败不会逐节点自动切回 generated CUDA。NumPy 是显式选择的参考后端。

## 当前已完成

截至 2026-10-10，GAC 已跑通单 GPU 的模型导入、图优化、静态内存规划、CUDA 代码生成、vendor library 选择、HXP 打包和真机执行闭环。

| 层次 | 已完成能力 |
| --- | --- |
| 编译器核心 | Canonical Graph、Shape/DType 推导、静态连续布局检查、常量折叠、死代码消除、MatMul/Add/GELU 融合、安全 Add+ReLU 融合 |
| CUDA SM86 后端 | 基础 Tensor/MLP、MHA、CNN、MSDeformableAttention、BEV geometry、Spatial/Temporal Attention、运动补偿及 step 边界所需 Cast/Where；不支持的类型、属性和布局会明确拒绝 |
| HXP 与 Runtime | 确定性打包、摘要校验、安全加载、源码与 launch metadata 复核、静态 arena、驻留权重、单 stream 执行、状态提交/回滚、Trace 和可选逐 launch CUDA Event Profile |
| BEVFormer-tiny | 官方 checkpoint 严格映射；ResNet-50 + 单层 FPN；feature/input preparation；三层 Encoder；六层 Decoder；六层 raw detection projection；cold/history image-to-BEV；统一 cold/warm 五输入三输出 step |
| 时序状态 | 历史 BEV rotate/shift、`has_prev_bev` 选择、scene/CAN-bus delta 管理；只有完整调用和三个输出校验成功后才提交 `next_bev` |
| 主机后处理 | 显式 CPU `decode_nms_free`、Top-K 300、box denormalization 与范围过滤；该阶段位于 HXP/CUDA 图外 |

冻结的 BEVFormer-tiny Profile 为 6 相机、`480×800` 图像、`50×50` BEV、256 embedding。统一 step 输入为 `images`、`lidar2img`、`can_bus_delta`、`prev_bev`、`has_prev_bev`，输出最后一层 `cls_logits`、raw `bbox_regression` 和 FP16 `next_bev`。公开图像与状态存储边界使用 FP16，Transformer/MSDA 等主要计算仍以 FP32 为基线。

## RTX 3070 / SM86 适配边界

首个目标是桌面 RTX 3070 对应的 `cuda-sm86`。最终全图验收实际使用 **RTX 3080 10 GB**；RTX 3080 与 RTX 3070 的 Compute Capability 都是 8.6，因此这次结果验证了相同 SM86 编译目标下的代码生成、NVRTC、Kernel 执行和 Runtime 契约。

这不等于 RTX 3070 性能验收。两张卡的计算资源、带宽、显存容量和功耗不同，本文所有 BEVFormer 全图耗时都属于 RTX 3080。桌面 RTX 3070 的全图延迟、实际峰值显存和稳定性尚未实测。较早的 RTX 3070 Ti Laptop 记录只覆盖 MLP 与独立 MSDA 切片，不代表完整 BEVFormer。

## RTX 3080 最终验收

验收环境为 Ubuntu 22.04、RTX 3080、CUDA SM86；GAC 使用 Python 3.11.17、CuPy 13.6.0、CUDA Runtime/NVRTC 12.9。官方 `bevformer_tiny_epoch_24.pth` 在独立 PyTorch/MMCV 参考环境中生成 golden。

下表是固定 Profile raw step 的稳态中位数：全生成路径各测 3 次，优化路径各测 10 次。GPU stream 与调用 wall time 均不包含 Session 构建、完整数据预处理或 CPU NMS-free 后处理。

| 路径 | 全生成 GPU / wall | 零拷贝 + FP32 cuDNN/cuBLAS GPU / wall | GPU / wall 加速比 |
| --- | ---: | ---: | ---: |
| cold | 425.325073 / 436.881591 ms | 57.624464 / 69.403093 ms | 7.38098× / 6.29484× |
| warm | 426.032532 / 437.718436 ms | 56.823263 / 68.747440 ms | 7.49750× / 6.36705× |

输出相对官方 PyTorch/MMCV golden 的最大绝对误差为：

| 路径 | `cls_logits` | `bbox_regression` | `next_bev` |
| --- | ---: | ---: | ---: |
| cold | 0.003891945 | 0.004856288 | 0.001953125 |
| warm | 0.000953674 | 0.000725389 | 0.001953125 |

其他最终验收数据：

- 独立进程完成 Session 构建后的首个 cold `run`：`277.443573 ms GPU / 289.668472 ms wall`。同一进程基准里的 `116.635651 ms` 发生在 Session 已初始化后，不是 fresh-process 首次 `run`。
- 带逐 launch CUDA Event 的独立 Profile 会扰动执行：该轮为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 Kernel Event 求和 `63.4391 ms`；其中 cuDNN 55 次 / `36.6409 ms`、CuPy/cuBLAS 118 次 / `8.0060 ms`、generated CUDA 530 次 / `18.7923 ms`。这些值用于归因，不与上表普通基准直接比较。
- 强制 CUDA 的完整测试：`Ran 287 tests in 18.881s`，`OK (skipped=1)`。
- 最终 HXP：`137,896,143` bytes，SHA-256 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`。
- 图优化规模：融合前 876 节点，39 组 Add+ReLU 融合后 837 个执行节点，零拷贝后 703 次设备 dispatch；内部 arena `552,960,000` bytes。
- Tensor 计划为 `705,177,100` bytes；加上 cuDNN 8 MiB 后，`713,565,708` bytes 是 Tensor + workspace 的计划预算/准入值，并非设备进程峰值，不包含 CUDA module/context、Event 和 vendor 临时分配。

显式 128 MiB cuDNN workspace 实验把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，并比当前 8 MiB budget 多占 120 MiB，因此本轮没有集成。下一步优先推进 CUDA Graph、批量 dispatch 和更多融合；当前 `cupyx.cudnn` 与 capture 不兼容，cuDNN 图捕获仍需要稳定的低层 API。

## 快速运行

离线编译与 NumPy 参考路径需要 Python 3.11+。Linux 或 WSL2 上的 CUDA 路径还需要 NVIDIA 驱动和项目的 `cuda12` 可选依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cuda12]'

.venv/bin/python -m gac.cli doctor
.venv/bin/python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
.venv/bin/python -m gac.cli inspect build/mlp-sm86.hxp
.venv/bin/python -m gac.cli run build/mlp-sm86.hxp \
  --verify \
  --trace build/trace.json \
  --profile build/profile.json \
  --report build/report.json
```

`--profile` 会额外执行一次带逐 launch Event 的调用；默认运行路径不创建这些 Event。`compile` 也可输出 `.cu` 和 `.plan.json` 调试产物。当前 HXP 保存规范图、常量、CUDA 源码、launch metadata 和静态内存计划，不是预编译 cubin；它与当前 codegen 绑定，优化前的 schema 1 包需要重建。

没有 NVIDIA GPU 时，可显式使用 NumPy 参考后端检查语义：

```bash
python -m pip install -e .
python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
python -m gac.cli inspect build/mlp-sm86.hxp
python -m gac.cli run build/mlp-sm86.hxp --backend numpy-cpu --verify
```

Windows 将解释器路径换成 `.\.venv\Scripts\python.exe`。原生 Windows 的离线编译与 CPU 路径已检查，CUDA 路径尚未实测。

BEVFormer 的合同检查、权重检查和子图/HXP 构建由 `gac-bevformer` 提供：

```bash
gac-bevformer --help
gac-bevformer step-pipeline --help
```

完整的官方 checkpoint 环境、golden 生成和远程复现步骤见 [BEVFormer 参考环境](docs/bevformer_reference_environment.md)与[远程验收记录](docs/bevformer_sm86_validation_2026-10-10.md)。

统一测试入口：

```bash
python -m unittest discover -s tests -v
GAC_REQUIRE_CUDA=1 python -m unittest discover -s tests -v
```

`GAC_REQUIRE_CUDA=1` 用于真机验收；CUDA 不可用时会失败，不会把 GPU 测试静默算作通过。

## 项目结构

```text
src/gac/ir.py, ops.py, passes.py     Canonical Graph、算子语义与图优化
src/gac/backends/cuda.py             SM86 支持检查、CUDA 生成与 launch metadata
src/gac/memory.py                    生命周期分析与静态 arena 规划
src/gac/package.py                   HXP 写入、摘要校验、安全加载与 codegen 复核
src/gac/runtime.py                   CudaSession、vendor 选择、传输、启动与计时
src/gac/profiler.py                  Trace 与 CUDA Event Profile 聚合
src/gac/frontends/                   JSON、BEVFormer 合同与 Backbone 导入
src/gac/bevformer*.py                BEVFormer 算子、Encoder/Decoder/raw head 与状态 Runtime
src/gac/tools/build_bevformer.py     权重检查及 BEVFormer 子图/HXP 构建入口
src/gac/cli.py                       doctor / compile / inspect / run
examples/                            MLP、MSDA 示例与冻结合同
tests/                               编译器、Runtime、HXP、BEVFormer 与物理 CUDA 测试
docs/                                架构、设计、后端约定、路线图和验收证据
```

更完整的实现映射见[架构说明](docs/architecture.md)，BEVFormer 图与状态语义见[编译器设计](docs/bevformer_compiler_design.md)，后端扩展约定见[后端编写指南](docs/backend_authoring.md)，后续计划见[路线图](docs/roadmap.md)。

## 已知边界

- 当前支持声明限定为固定 Profile 的 BEVFormer-tiny **raw step**。CPU `decode_nms_free` 是公开的图外后处理边界，不包含在上述 GPU/wall 数字中。
- 正确性输入是冻结的合成、已归一化并 padding 的六相机 tensor 与固定几何输入；尚未完成真实 nuScenes 数据集的 mAP/NDS 验收。
- 完整 BEVFormer 只在 RTX 3080 SM86 上验收；桌面 RTX 3070 的延迟、峰值显存和稳定性仍待实测。
- 只覆盖首条路径所需的静态 Shape 和连续布局；主要计算仍为 FP32，尚未启用 Tensor Core、TF32、FP16 主计算或离线算法调优。
- 当前使用单 CUDA stream，调用同步返回；设备侧跨帧状态驻留、多 stream、I/O 预绑定、CUDA Graph、批量 dispatch 和 DLPack 仍待实现。
- `713,565,708` bytes 只是已计划 Tensor 与 cuDNN workspace 的预算/准入值，不能当作 `nvidia-smi` 进程峰值。
- 当前 HXP 与本版 codegen 绑定，尚无预编译 cubin、多 target variant、签名认证或通用异构 Host Execution Plan。
- 通用后端注册、代价驱动划分、CPU/GPU 自动回退、跨设备 alloc/copy/event/free 计划和 NPU 后端尚未实现。
- Session 建立时选择 cuDNN、CuPy/cuBLAS 或 generated CUDA；vendor 调用执行中失败会报错，不会逐节点自动换实现。

## 验收证据

最终数字与历史阶段日志的边界见[优化验收证据索引](docs/validation/sm86/bevformer/optimization-2026-10-10/README.md)。主要原始文件：

- [最终 cold/warm 基准与精度](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-benchmark-final.json)
- [fresh-process 首次 cold run](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-first-frame-final.json)
- [最终 Profile 摘要](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final-summary.json)与[逐 launch Profile](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final.json)
- [最终强制 CUDA 测试日志](docs/validation/sm86/bevformer/optimization-2026-10-10/unittest-full-final.txt)
- [最终 HXP 构建日志](docs/validation/sm86/bevformer/optimization-2026-10-10/build-production-step-final.txt)、[大小](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.size.txt)与[SHA-256](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.sha256.txt)
- [128 MiB cuDNN workspace 实验](docs/validation/sm86/bevformer/optimization-2026-10-10/cudnn-explicit-workspace-benchmark.json)
- [完整远程验收说明](docs/bevformer_sm86_validation_2026-10-10.md)

## 参考项目与许可证

本项目的架构研究参考 IREE、Apache TVM、ONNX Runtime、TileLang 和 OpenXLA/XLA，但当前实现为本原型独立编写，没有从这些项目复制源码。未来若引入外部源码、头文件、生成物或二进制依赖，需要保留适用的版权、SPDX、`NOTICE` 和第三方许可证，并单独核对 CUDA、cuDNN、厂商 SDK 与模型权重的分发条款。
