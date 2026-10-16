# 原生 MLIR CUDA PTX 后端

分支：`lg/bevformer-cuda-ptx-full-ops`。前端、MLIR 图优化、算子语义和静态 Buffer
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
| 编译流水线 | [gac-compile.cpp](tools/gac-compile.cpp)、[compile-cuda.sh](tools/compile-cuda.sh) | 分阶段 MLIR、LLVM IR、PTX、manifest、工具哈希 |
| manifest 校验 | [dispatch-plan.h](runtime/dispatch-plan.h) | 类型、大小、存活区间和地址重叠检查 |
| 设备 Runtime | [cuda-run.cpp](runtime/cuda-run.cpp) | 显存分配、上传、提交、等待、下载和测量 |
| 数值验收 | [run_cuda_tests.py](tests/run_cuda_tests.py) | 真实设备输出与独立参考对比 |

Kernel 接收各输入的设备指针，最后一个参数是输出指针。指针为 64 位；当前元素索引
为 32 位，单张量字节数不得超过 2,147,483,647。CUDA 总分配不受 Vortex 的 RV32
总地址预算限制。`block_size` 默认 256，`grid_size=ceil(输出元素数/256)`。

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

## Buffer 和显存

输入、外置常量独立分配；中间激活与输出复用按 SSA 生命周期规划的 arena。
同一条 Stream 保证前一个 dispatch 完成后，后一个 dispatch 才能覆盖已失效的值。
这不是 shared memory 双缓冲；当前 `shared_memory_bytes=0`。

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

当前为 FP32 正确性基线：每线程负责一个输出，归约串行；未实现 Tensor Core、
shared-memory tiling、双缓冲或自动调优。因此，编译成功或显存足够不代表达到
30 FPS。全局 arena 复用已经实现，不能把它称为 shared-memory 流水优化。

CUDA Event 测量设备 dispatch 序列，排除上下文创建、PTX JIT、权重/输入上传和
输出下载。`--trace-dispatches` 会逐 Kernel 同步，仅用于定位问题。
整网模型执行、三个输出的数值验收、连续帧吞吐和实际场景精度分别记录。
当前模型契约仍是固定尺寸、固定标定、batch=1 的 cold frame；未据此验收 warm/history、
动态尺寸或完整数据集精度。

本次编译、GPU 数值对比和身份摘要见
[62 个算子用例](../docs/validation/cuda/native-mlir-2026-10-17/README.md)和
[完整 BEVFormer-base](../docs/validation/cuda/native-mlir-bevformer-base-2026-10-17/README.md)。
完整模型三个输出与 PyTorch 对齐，最大绝对误差约 `6.18e-5`（`rtol=1e-3, atol=1e-4`）。
首次未预热设备执行约 **17.05 秒**；初始化/JIT 约 17.36 秒另外计量，不能混为推理时间。
随后预热 1 次、测量 3 次，设备执行分别为 `17.4682 / 17.8840 / 18.7906` 秒，
平均 **18.0476 秒（约 0.055 FPS）**。最终输出 SHA-256 与首轮通过数值比对的输出相同。
该结果验证了功能，但明显未达到 30 FPS。
