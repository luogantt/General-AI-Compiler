# General AI Compiler（GAC）

以 **BEVFormer** 为验证模型的 AI 编译器与推理 Runtime。原生核心使用 **C++ / MLIR**，把 ONNX 编译到 **CUDA PTX** 或 **Vortex**；另提供可独立运行的 **TensorRT 推理项目**。

**TensorRT 最新结果：完整 BEVFormer-base 固定图达到 309.42 ms/帧（约 3.23 FPS）。** RTX 3070 Ti Laptop 8 GB 上，第三轮优化 DCNv2 后，同轮整网对照延时再降低 **5.07%**。前两轮 MSDA 分别降低 18.05%、2.60%；各轮比例不能相加，详见[优化结果](#优化结果)。

## 做了什么

| 部分 | 已实现内容 |
|---|---|
| 模型导入 | C++ 读取 ONNX，直接创建注册的 `gac` MLIR Operations，保留类型、形状、属性和权重 |
| 图优化与 Lowering | 形状推导、常量折叠、CSE、条件融合、合法化；规则算子走 Linalg，不规则算子走专用 Lowering |
| 设备编译与 Runtime | Dispatch 划分、静态 Buffer 复用、设备代码生成；Runtime 负责分配、上传、启动、同步和下载 |
| CUDA / Vortex | 共用前端和语义层，分别生成 PTX 或 VOLT 的 ELF/vxbin；已有矩阵分块、归约及 Vortex LMEM/DXA/TCU 实验模板 |
| Triton / Vortex（本分支） | C++ 将支持的 Linalg 逐元素子图送入 Triton；与原生 Vortex 内核链接到同一 vxbin，共用 Buffer 与 Runtime。[使用与边界](native/README.triton-vortex.md) |
| TensorRT | 完整 Base 固定 ONNX 的插件适配、构建、推理、混合 INT8/FP16、逐层 profiling，以及自定义 MSDA / DCNv2 Kernel 优化 |

**验证范围：** 原生 CUDA 已完成固定输入的 BEVFormer-base 整网执行和数值对齐；Vortex 已有完整模型编译产物和小模型 SimX 验证，完整模型的 Vortex 执行与 FPS 尚未验收。当前 Base 整网记录对应固定形状、固定几何、无历史帧到原始检测头输出，尚无 nuScenes mAP/NDS 验收。

## 编译链

```text
ONNX
  ↓ C++ Importer
GAC MLIR → 图优化 / 合法化
  ↓
Linalg（规则算子）＋ GAC 专用 Lowering（采样等不规则算子）
  ↓
循环、线程映射、Dispatch、Buffer 规划
  ├─ CUDA Lowering → NVVM / LLVM → NVPTX → PTX → CUDA Driver → NVIDIA GPU
  └─ Vortex Lowering → LLVM → VOLT → ELF / vxbin → Vortex Runtime
```

TensorRT 是独立的对照和部署路径：`ONNX → TensorRT engine → TensorRT Runtime`。本次约 18% 的收益来自其中的 **C++/CUDA 自定义插件**，没有计入原生 MLIR → PTX 路径。默认 Linalg 路线与旧专用模板的选择方式见[原生架构](native/README.md)。

## 如何启动推理

以下命令使用 Linux / WSL Bash；从仓库根目录开始。完整 BEVFormer 的 ONNX、权重、输入和参考输出需单独准备，源码仓库不包含这些大模型资产。输出目录使用新路径。

### 原生编译器：ONNX → MLIR → PTX → GPU

需要 LLVM/MLIR **20.1.8**（启用 NVPTX）、CMake/Ninja、C++17、protobuf/protoc、ONNX 的 `onnx.proto`、CUDA 头文件和 libdevice；执行机器需要 NVIDIA 驱动。[构建细节](native/README.cuda.md#构建和执行)与[Base 模型导出](native/README.bevformer-base.md#4-导出与编译)单独说明。

```bash
# 1. 按实际环境修改路径，构建编译器和 Runtime。
export LLVM_PATH=/path/to/llvm-20.1.8
cmake -S native -B build/native-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DONNX_PROTO_DIR=/path/to/onnx \
  -DGAC_CUDA_INCLUDE_DIR=/usr/local/cuda/include
cmake --build build/native-cuda --parallel 2

# 2. 从原始 GAC/ONNX 导出图生成 PTX。
export GAC_NATIVE_BIN="$PWD/build/native-cuda"
export CUDA_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
export GAC_CUDA_ARCH=sm_86
export GAC_CONSTANT_FOLD_PROFILE=bevformer-base
export GAC_REGULAR_LOWERING=linalg

# 这里使用原始 GAC/ONNX 导出图，不使用 TensorRT 插件改写后的 adapted.onnx。
bash native/tools/compile-cuda.sh /path/to/bevformer-base.onnx build/bevformer-cuda

# 3. 运行编译产物，并与同一输入的独立参考输出比较。
python3 native/tests/run_cuda_tests.py \
  --artifact build/bevformer-cuda \
  --inputs /path/to/inputs.npz --reference /path/to/reference.npz \
  --runner build/native-cuda/gac-cuda-run \
  --output-dir build/bevformer-cuda-run \
  --warmup 1 --repeat 5 --run-timeout 7200 --rtol 1e-3 --atol 1e-4
```

Python 工具需要 NumPy，负责输入文件转换和结果比较；模型计算由 **C++ Runtime 启动 PTX** 完成。编译目录包含 `kernel.ptx`、`plan.json`、权重和各阶段 MLIR；运行目录保存输出与 `cuda-validation.json`。直接调用 Runtime 和 Vortex 启动方式分别见 [CUDA](native/README.cuda.md) / [Vortex](native/README.md#6-从-onnx-到-vxbin-与执行)。

### TensorRT：构建 engine 并推理

独立项目位于 [standalone/bevformer-tensorrt](standalone/bevformer-tensorrt/README.md)。已测环境为 Python 3.11、TensorRT **10.13.3.9**、PyTorch **2.7.0+cu126**。

```bash
cd standalone/bevformer-tensorrt
bash scripts/setup.sh --python python3.11 --install-torch
source .venv/bin/activate
python -m bevformer_trt check --report results/environment.json
```

将配套模型资产放入 `models/bevformer-base/`；[资产契约](standalone/bevformer-tensorrt/docs/model-and-data.md)说明 ONNX、校准范围和六相机输入的格式。以下启动**官方插件的混合精度基线**：

```bash
export CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --engine results/mixed/base.engine --precision mixed-int8 \
  --calibrated-ranges models/bevformer-base/ranges.json \
  --workspace-mib 1024 --tactic-dram-mib 4096 \
  --builder-optimization-level 3 --max-aux-streams 0 \
  --report results/mixed/build.json

python -m bevformer_trt run \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output results/mixed/run --warmup 5 --repeat 20
```

结果为 `outputs.npz`（BEV 特征、分类 logits、3D 框编码）和 `report.json`（延时、环境与检查结果）。未传 `--reference` 时不作数值对照。配套合成输入用于回归，不用于重新校准或评估检测精度；FP32 启动命令见[独立项目 README](standalone/bevformer-tensorrt/README.md)。

**启用优化后的 MSDA：** 按[复现流程](standalone/bevformer-tensorrt/docs/msda-optimization.md)编译插件、显式改写六个 Spatial MSDA、重新绑定原校准范围并构建 engine。构建和运行都要传 `--plugin-library /path/to/libgac_msda_plugin.so`；上面的默认命令不会自动启用它。

**再启用 DCNv2 优化：** 在上述模型基础上，[编译 DCN 插件](standalone/bevformer-tensorrt/native/dcn/README.md)，构建时传 `--dcn-config` 精确节点配置，构建及运行均加载两份插件库。[完整命令和本轮验证](standalone/bevformer-tensorrt/docs/validation-dcn-round3-3070.md#代码与复现)

## 优化结果

2026-10-21，**RTX 3070 Ti Laptop 8 GB，BEVFormer-base，混合 INT8/FP16**。首轮对照：

| 测试范围 | NVIDIA 官方实现 | 自定义 MSDA 实现 | 收益 |
|---|---:|---:|---:|
| 单个 Spatial MSDA，相同输入 | 38.95 ms | 16.91 ms | 加速 **2.30 倍** |
| 完整固定图，引擎每帧中位延时 | 413.77 ms | 339.09 ms | 延时降低 **18.05%** |
| 完整固定图，延时倒数 | 2.42 FPS | 2.95 FPS | 吞吐换算约 **1.22 倍** |

优化保持采样公式和模型权重不变：**每线程 4 通道、half2、坐标/地址复用、固定形状专用化**；实测选择 128 线程/块，关闭未获益的预取。只替换六个 Encoder Spatial MSDA，其余节点保留。

完整引擎按“官方—优化—优化—官方”测试，每轮预热 5 次、测量 20 次，均从 55°C 开始；复用 timing cache 并核对其他层无 tactic/格式变化。**同一真实六相机输入的三个输出逐位一致**。计时排除预处理、传输、加载和框解码；单算子与整网分别计时，不能混算。

这证明本轮优化没有改变已测输入的官方混合引擎输出；不代表混合量化对 FP32 或整个数据集的精度已经达标。测试图仍使用固定合成几何和无历史帧，未达到 30 FPS。

原始数据、首次失败对照及最终受控结果见[MSDA 验证报告](standalone/bevformer-tensorrt/docs/validation-msda-3070.md)。原生 MLIR 的编译/JIT/推理数据单独记录在[CUDA 路径对照](docs/validation/cuda/linalg-comparison-2026-10-21/README.md)，不与 TensorRT 数字混用。

**第二轮：** 增加有溢出保护的 32 位索引与邻点地址复用，修复 Kernel helper 重构导致的性能回归。实际旧库与新库对照各 60 帧：**335.60 → 326.88 ms，延时再降 2.60%**；三个输出仍逐位一致。采用每线程 4 通道、128 线程块、`specialize=3`、`prefetch=0`。预取候选反而变慢，完整 CUDA Graph 捕获失败，均未计入收益。两轮百分比不能相加。[第二轮实测及失败记录](standalone/bevformer-tensorrt/docs/validation-msda-round2-3070.md) · [启用配置](standalone/bevformer-tensorrt/docs/msda-optimization.md#7-继续优化与已经优化的配置比较)

**第三轮（2026-10-22）：** DCNv2 采样阶段每线程复用 4 个通道的坐标、邻点和插值系数，保留 FP16 运算及 cuBLAS HGEMM。替换 156 个满足条件的节点，整网同轮对照 **325.97 → 309.42 ms，延时降低 5.07%，约 3.23 FPS**。三个输出与基线、官方插件参考逐位一致；320 项 GPU 候选检查和双上下文检查通过。[第三轮报告、原始数据与复现](standalone/bevformer-tensorrt/docs/validation-dcn-round3-3070.md)

## 代码与文档

| 入口 | 内容 |
|---|---|
| [native/](native/README.md) | C++ ONNX 导入、MLIR 方言/Pass、设备 Lowering、CUDA/Vortex Runtime |
| [native/templates/](native/templates/README.md) | CUDA/Vortex 计算模板及选择约束 |
| [standalone/bevformer-tensorrt/](standalone/bevformer-tensorrt/README.md) | 独立 TensorRT 构建、推理、校准和 MSDA 优化 |
| [src/gac/](src/gac) | Python 模型适配、工具和历史 Graph/HXP 路径 |
| [docs/](docs/index.html) | 架构图、原理说明、验证报告 |

- 理解模型：[BEVFormer 架构图](docs/bevformer-model-architecture.html)；理解编译器：[MLIR 源码架构图](docs/mlir-source-architecture.html)、[算子融合](docs/compiler-fusion.md)。HTML 下载后用浏览器打开。
- 历史实验：[训练记录](docs/training/bevformer-tiny/README.md)、[早期 FFN / HXP / UniAD 与版本记录](docs/history/README-before-2026-10-21.md)。各阶段的支持范围和性能分别查看。
