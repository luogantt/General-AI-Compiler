# 原生 MLIR CUDA PTX 后端

统一代码已合入 `main`，通过 `--target=cuda` 选择本后端。
前端、MLIR 图优化、算子语义和静态 Buffer
规划与 Vortex 共用，CUDA 使用独立的线程映射、指针 ABI 和 Runtime。
本路径不生成 CUDA C++ Kernel，也不调用 Python 算子或 cuBLAS 代替当前设备计算。

```text
ONNX ModelProto
    ↓ OnnxImporter.cpp：C++ protobuf → 注册的 gac Operation
MLIR ModuleOp / func.func / SSA / Tensor / 权重
    ↓ GraphPasses.cpp：形状、常量折叠、合法化、CSE、融合、布局检查
优化后的 GAC 图
    ↓ DeviceLowering.cpp：算子循环 + Dispatch + 静态全局 Buffer 规划
SCF / Arith / Math / LLVM / NVVM
    ↓ CudaLowering.cpp：线程/Block ID、Kernel 标记、libdevice 调用
MLIR LLVM Dialect
    ↓ mlir-translate → llvm-link(libdevice) → opt → llc(NVPTX)
PTX + plan.json + tensor_*.bin
    ↓ cuda-run.cpp：cuModuleLoadDataEx → cuLaunchKernel
CUDA Driver JIT → SASS → NVIDIA GPU
```

产物是本项目的 manifest 和 PTX，不是 IREE VMFB。Runtime 使用 NVIDIA Driver API，
不解释 PTX。Python 测试工具只负责生成夹具、文件组织和结果对比。

## 模块与交接格式

| 模块 | 代码 | 输出或契约 |
|---|---|---|
| C++ 导入、图优化 | [OnnxImporter.cpp](lib/OnnxImporter.cpp)、[GraphPasses.cpp](lib/GraphPasses.cpp) | 真正的 MLIR Module 和 Operation |
| 算子与内存共享层 | [DeviceLowering.cpp](lib/DeviceLowering.cpp) | SCF/Arith 操作、张量地址规划、dispatch 依赖 |
| CUDA 目标层 | [CudaLowering.cpp](lib/CudaLowering.cpp) | NVVM 操作、`nvvm.kernel`、libdevice 数学函数 |
| CUDA 计算模板 | [CudaMatmul.cpp](lib/CudaMatmul.cpp)、[CudaReduction.cpp](lib/CudaReduction.cpp) | 共享内存分块、线程寄存器块、CTA 协作归约 |
| 编译流水线 | [gac-compile.cpp](tools/gac-compile.cpp)、[compile-cuda.sh](tools/compile-cuda.sh) | 分阶段 MLIR、LLVM IR、PTX、manifest、工具哈希 |
| manifest 校验 | [dispatch-plan.h](runtime/dispatch-plan.h) | 类型、大小、存活区间和地址重叠检查 |
| 设备 Runtime | [cuda-run.cpp](runtime/cuda-run.cpp) | 显存分配、上传、提交、等待、下载和测量 |
| 数值验收 | [run_cuda_tests.py](tests/run_cuda_tests.py) | 真实设备输出与独立参考对比 |

Kernel 接收各输入的设备指针，最后一个参数是输出指针。指针为 64 位；当前元素索引
为 32 位，单张量字节数不得超过 2,147,483,647。CUDA 总分配不受 Vortex 的 RV32
总地址预算限制。标量模板的 `block_size` 默认 256，
`grid_size=ceil(输出元素数/256)`；优化模板按输出 tile 或归约行生成 launch，
Runtime 严格校验 manifest 中的形状、线程数和 shared memory 契约。

## 构建和执行

要求 LLVM/MLIR **20.1.8**，构建 LLVM 时启用 `NVPTX` 目标；另需 ONNX protobuf、
CUDA 官方头文件和 `libdevice.10.bc`。执行机器只需兼容的 NVIDIA Driver，
不需要安装完整 LLVM 或 PyTorch。

```bash
cmake -S native -B build/native-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DONNX_PROTO_DIR=/path/to/onnx \
  -DGAC_CUDA_INCLUDE_DIR=/path/to/cuda/include
cmake --build build/native-cuda --parallel 2

export GAC_NATIVE_BIN="$PWD/build/native-cuda"
export CUDA_LIBDEVICE=/path/to/cuda/nvvm/libdevice/libdevice.10.bc
export GAC_CUDA_ARCH=sm_86
export GAC_CONSTANT_FOLD_PROFILE=bevformer-base
# 可选离线汇编检查；它不能代替 GPU 数值测试。
export CUDA_PTXAS=/path/to/cuda/bin/ptxas
bash native/tools/compile-cuda.sh model.onnx build/model-cuda

build/native-cuda/gac-cuda-run \
  build/model-cuda/plan.json build/model-cuda/kernel.ptx \
  build/model-cuda build/model-cuda/outputs \
  --warmup 1 --repeat 5
```

编译输出目录必须不存在或为空，以免将旧 PTX/cubin 与新 manifest 混用。
先按 manifest 的张量编号准备 `input_<编号>.bin`，其 dtype、shape 和连续布局必须
符合声明。外置权重由编译器生成。Runtime 输出 `output_<公开输出编号>.bin`。
完整测试命令、错误处理和对比容差见 [CUDA 验证](tests/README.cuda.md)。

## 选择优化模板

以下选项由 C++ MLIR Lowering 实现；Python 不参与设备算子计算。默认参数仍保留
标量基线，便于同输入对比。可以在上述编译命令前显式设置：

```bash
export GAC_CUDA_MATMUL_SCHEDULE=auto
export GAC_CUDA_REDUCTION_SCHEDULE=cooperative
export GAC_CUDA_CONV_CHANNEL_TILE=8
bash native/tools/compile-cuda.sh model.onnx build/model-cuda-optimized
```

| 选项 | 行为与约束 |
|---|---|
| `GAC_CUDA_MATMUL_SCHEDULE=scalar` | 每线程一个输出，串行 K 归约；默认值 |
| `=shared-tiled` | A/B 协作搬到静态 shared memory，每线程持有多个 FP32 累加器；支持 MatMul、Bias、Bias+ReLU |
| `=auto` | 合法 FP32 MatMul 的 `M≥128,N≥64,K≥64` 选 `64×64×16`、每线程 `4×4`；其他形状保留 scalar |
| `GAC_CUDA_TILE_M/N/K` | 显式 shared-tiled 的块大小，默认 `32/32/16` |
| `GAC_CUDA_THREAD_TILE_M/N` | 显式 shared-tiled 的每线程输出块，默认 `2/2` |
| `GAC_CUDA_REDUCTION_SCHEDULE=cooperative` | FP32 Softmax/LayerNorm 每行由一个 CTA 协作归约；支持非末轴 Softmax 和 LayerNorm 后缀归约 |
| `GAC_CUDA_CONV_CHANNEL_TILE=1/2/4/8` | FP32 Conv/DCNv2 同一空间位置复用采样值计算多个输出通道，默认 1；块不能跨 group，组内通道不整除时保留标量 |

对应编译器参数为 `--cuda-matmul-schedule`、`--cuda-tile-m/n/k`、
`--cuda-thread-tile-m/n`、`--cuda-reduction-schedule`、`--cuda-conv-channel-tile`。
`auto` 是基于已测形状的启发式，不是 autotuning，也不保证对每种 GPU/形状最快。
它选择固定参数，显式 tile 参数用于 `shared-tiled`。
不符合优化模板契约但可由原路径计算的算子，会在 manifest 记录标量选择原因。
`32×32×16 / 2×2` MatMul、协作归约、Conv 通道块 4 的组合，以及上面的
`auto + cooperative + 通道块8` 组合，都已做完整 base GPU 数值验证。
后者本轮四次普通设备序列均约 6.43 秒；测量范围与温度影响见性能记录。

MatMul 保持每个输出的 K 累加顺序；尾块只计算有效 K，避免补零改变 NaN/Inf 语义。
CTA barrier 位于线程边界判断外，确保所有线程共同到达。LayerNorm 使用两遍中心化
方差，协作归约改变加法顺序，因此仍须与独立参考做精度对齐。

## Buffer 和显存

输入、外置常量独立分配；中间激活与输出复用按 SSA 生命周期规划的 arena。
同一条 Stream 保证前一个 dispatch 完成后，后一个 dispatch 才能覆盖已失效的值。
这里的全局 arena 与 Kernel 内部 shared memory 分开管理。优化 MatMul 为 A/B 各
申请一块静态 shared memory；归约模板申请一块 scratch，并在复用前同步。
manifest 的 `shared_memory_bytes=0` 表示**动态** shared memory 为零，不能据此说
没有使用 shared memory；静态预算位于 `cuda_schedule.static_shared_memory_bytes`，
Runtime 同时查询 PTX JIT 后的实际资源。当前 CUDA 模板尚未实现异步双缓冲。

本次固定输入 BEVFormer-base 的静态预算如下：

| 项目 | 字节数 |
|---|---:|
| 图像输入 `[1,6,3,928,1600]` FP32 | 106,905,600 |
| 外置常量 | 368,518,572 |
| 复用后的全局 arena | 1,554,944,000 |
| Runtime 显式分配合计 | 2,030,368,172 |
| 另外嵌入 PTX 的索引表 | 2,073,600 |

逻辑下限约 **2.03 GB / 1.89 GiB**，还需加驱动上下文、机器码、分配粒度及可能的
寄存器溢出空间。复用前所有中间值大小的累计和不是同时存活的显存需求。
`runtime-report.json` 的 `device_memory_snapshots` 是几个时点的全设备占用，
可能包含其他进程，不能称为本进程的精确峰值。

RTX 3070 Ti Laptop 8 GB 的首次完整执行中，约一秒间隔的 `nvidia-smi` 采样记录到
该进程最高 **2,160 MiB（2.11 GiB）**，完整执行无显存不足错误。这是观测到的最大值，
不保证捕获短暂峰值；8 GB 已足以运行本次固定模型配置。

## 性能与验收边界

已有 FP32 标量与上述优化模板；尚未在这条原生整网路径实现 CUDA Tensor Core、
异步双缓冲或实测 autotuning。编译成功或显存足够不代表达到 30 FPS。

CUDA Event 测量设备 dispatch 序列，排除上下文创建、PTX JIT、权重/输入上传和
输出下载。`--trace-dispatches` 会逐 Kernel 同步，仅用于定位问题。
`--profile-dispatches` 在普通计时之后另跑带逐 dispatch Event 的测量，输出算子
耗时分布、节点与资源信息；详细方法与限制见 [Profiling](README.cuda-profiling.md)。
整网模型执行、三个输出的数值验收、连续帧吞吐和实际场景精度分别记录。
当前模型契约仍是固定尺寸、固定标定、batch=1 的 cold frame；未据此验收 warm/history、
动态尺寸或完整数据集精度。

最初的标量编译、GPU 数值对比和身份摘要见
[62 个算子用例](../docs/validation/cuda/native-mlir-2026-10-17/README.md)和
[完整 BEVFormer-base](../docs/validation/cuda/native-mlir-bevformer-base-2026-10-17/README.md)。
完整模型三个输出与 PyTorch 对齐，最大绝对误差约 `6.18e-5`（`rtol=1e-3, atol=1e-4`）。
首次未预热设备执行约 **17.05 秒**；初始化/JIT 约 17.36 秒另外计量，不能混为推理时间。
随后预热 1 次、测量 3 次，设备执行分别为 `17.4682 / 17.8840 / 18.7906` 秒，
平均 **18.0476 秒（约 0.055 FPS）**。最终输出 SHA-256 与首轮通过数值比对的输出相同。
该结果验证了功能，但明显未达到 30 FPS。后续优化的完整模型计时、温度/频率波动、
精度和小算子证据单独记录在 [原生性能优化记录](../docs/validation/cuda/native-mlir-performance-2026-10-17/README.md)，
不能把上述历史标量数字直接当作任意新测量的同条件基线。
