# 原生 MLIR 编译器

本目录是 **ONNX → MLIR Operation → MLIR Pass → 设备代码** 的原生编译路径，
包含 Vortex 和 CUDA PTX 后端。CUDA 的构建、ABI 与执行说明见
[CUDA 后端](README.cuda.md)。
编译器、图优化、设备 Lowering、Buffer 规划与 Runtime 使用 C++；不调用原有
Python `Graph`、Python 图 Pass 或 Python MLIR 文本发射器。

当前统一分支：**`lg/bevformer-full-ops`**，同时维护 CUDA PTX 和 Vortex 后端。
`DeviceOptions::target` 选择设备；`native` 目标只用于显式的 CPU 数值验证。
Vortex 路径已经用小模型完成真实
`ONNX → MLIR → VOLT → ELF → vxbin → SimX` 数值闭环。完整 BEVFormer-tiny cold
ONNX 已通过严格导入、MLIR 优化和设备 Lowering，生成 **1,548 个 dispatch** 的
主机验证产物及完整 Vortex ELF/vxbin，全部设备入口已核对。完整主机机器码已执行
1,548 个 dispatch，三个输出与 PyTorch 参考数值对齐；完整 SimX 执行仍待验收。
**严格编译通过与完整模型数值对齐是两项验收，不能互相替代。** 当前进度见
[整网验证记录](../docs/validation/vortex/native-mlir-bevformer-full-2026-10-16/README.md)。

官方 R101-DCNv2 / 200×200 BEV / 四级 FPN 的新增适配与独立验收范围见
[BEVFormer-base 说明](README.bevformer-base.md)。Base 新增真实 DCNv2、静态 nearest
Resize 和保留高层语义的流式 MSDeformAttn Lowering；其完整模型验收与上述 tiny
记录分开，不沿用 tiny 的性能数字。完整 base 已生成 **3,224 个 dispatch** 的
native 主机共享库与 Vortex ELF/vxbin，全部设备入口和产物审核通过；完整 CPU
机器码已执行全部 3,224 个 dispatch，三个最终输出与 PyTorch 参考对齐，见
[最终验证记录](../docs/validation/vortex/native-mlir-bevformer-base-2026-10-16/README.md)。
尚无 base 全图 Vortex 执行或硬件 FPS 结论。

CUDA 后端已使用同一 C++ 前端和共享算子 Lowering，把完整 base 的 3,224 个
dispatch 编译成 PTX，并在 RTX 3070 Ti Laptop 8 GB 上执行。固定 cold 输入的
三个输出与 PyTorch 参考对齐；预热后平均设备延时约 18.05 秒，显存采样约 2.11 GiB，
当前功能通过、性能尚未达到 30 FPS。范围与证据见
[CUDA 完整 base 验证](../docs/validation/cuda/native-mlir-bevformer-base-2026-10-17/README.md)。

统一分支合并时重新构建了同一个编译器，并用 MatMul、Conv、Softmax 分别通过
3 项 CPU 数值检查、3 项 Vortex RV32 对象检查和 3 项 CUDA PTX/ptxas 检查。
可执行源码与合并前 CUDA 分支一致，重建的编译器 SHA-256 也一致；本轮未重复
整网、GPU 或 SimX 执行，不能据此新增性能结论。命令、产物身份与范围见
[合并验证报告](../docs/validation/native-mlir-unified-2026-10-17.json)。

## 1. 实际架构与代码

```text
ONNX ModelProto                         protobuf C++ 解析
        ↓
ONNX Importer                           OpBuilder + 名字 → Value 映射
        ↓
MLIR ModuleOp / func.func                Tensor 类型、SSA、DenseElementsAttr 权重
├── gac.conv2d / gac.relu / gac.matmul
├── gac.softmax / gac.grid_sample / ...
├── gac.tile / gac.scatter_nd / gac.nonzero
├── gac.deform_conv2d / gac.resize / gac.ms_deform_attn
├── gac.split                           真正多结果 SSA Operation
└── gac.unsupported                     仅 inspection；不能进入执行编译
        ↓
MLIR PassManager
├── gac-infer-shapes                    固定点形状推导 + 有预算的纯常量求值
├── gac-legalize-tensor-ops              静态 Split → 多个 Slice Operation
├── gac-infer-shapes                    合法化后再次传播类型与常量
├── gac-normalize-provenance            清理内部显示名称，保留公开输出名称与诊断位置
├── canonicalize / cse                  上游 MLIR Pass
├── gac-fuse                            条件满足时融合，保留数值语义
└── gac-propagate-layout                 验证/传播可证明的轴布局
        ↓
DeviceLowering.cpp                     共享算子循环、SSA 依赖与静态 Buffer 规划
        ↓
按 --target 选择线程映射、ABI、设备数学和地址约束
        ├── vortex：VortexLowering.cpp / Vortex Dialect
        │      → LLVM Dialect → LLVM IR → VOLT → ELF → vxbin
        │      → vortex-run.cpp → Vortex Runtime → SimX / 设备
        │
        └── cuda：CudaLowering.cpp / NVVM
               → LLVM Dialect / NVVM → LLVM IR + libdevice → LLVM NVPTX → PTX
               → cuda-run.cpp → CUDA Driver JIT → SASS → NVIDIA GPU
```

| 层 | 源码 | 实际职责 |
|---|---|---|
| 构建与库依赖 | [CMakeLists.txt](CMakeLists.txt) | 链接真实 MLIR C++ 库、protobuf；锁定 LLVM 20.1.8 |
| 模型导入 | [OnnxImporter.cpp](lib/OnnxImporter.cpp) | 读 ModelProto，创建 ModuleOp/func/Operation，连接 SSA，读取权重 |
| 算子语义与类型 | [Dialect.h](include/gac/Dialect.h)、[Dialect.cpp](lib/Dialect.cpp) | 注册 `gac` 方言与操作类、输入约束、类型和形状规则 |
| 图优化 | [GraphPasses.cpp](lib/GraphPasses.cpp) | PassWrapper、IRRewriter、PatternRewriter，实际修改 MLIR 图 |
| 纯常量求值 | [ConstantFolding.cpp](lib/ConstantFolding.cpp) | APInt/APFloat 保留原始精度，对常量几何和 shape 子图求值 |
| 算子合法化 | [TensorLegalization.cpp](lib/TensorLegalization.cpp) | 将静态多输出 Split 重写为多个 Slice；保留来源与输出名 |
| 设备映射与 Buffer | [DeviceLowering.cpp](lib/DeviceLowering.cpp) | 共享静态资源检查、dispatch、按生命周期回收及合并/拆分空闲区间 |
| Kernel Lowering | [DeviceLowering.cpp](lib/DeviceLowering.cpp) | 共享算子语义，真正创建 SCF/Arith/Math/LLVM 操作；不拼接 Kernel 文本 |
| Vortex 方言与 ABI | [VortexLowering.cpp](lib/VortexLowering.cpp) | thread/block CSR、uniform、RV32 参数槽、入口 annotation |
| CUDA ABI 与数学库 | [CudaLowering.cpp](lib/CudaLowering.cpp) | NVVM 线程映射、64 位指针、设备 libdevice 数学函数 |
| 编译入口 | [gac-compile.cpp](tools/gac-compile.cpp) | 原生 ONNX 导入和 PassManager 流水线，可逐层输出 IR |
| MLIR 优化入口 | [gac-opt.cpp](tools/gac-opt.cpp) | 带 `gac`/`vortex` 注册的 `MlirOptMain` |
| 机器码构建 | [compile-vortex.sh](tools/compile-vortex.sh)、[Vortex.mk](tools/Vortex.mk) | 调用同版本 MLIR/VOLT，复用 Vortex startup/linker/封装工具 |
| 执行 | [vortex-run.cpp](runtime/vortex-run.cpp) | 读取 SSA 规划产生的 manifest；上传、提交、等待、写回，无 Host 算子回退 |
| CUDA 机器码与执行 | [compile-cuda.sh](tools/compile-cuda.sh)、[cuda-run.cpp](runtime/cuda-run.cpp) | LLVM NVPTX 生成 PTX，CUDA Driver JIT 和 Stream 顺序执行 |
| Runtime 共用契约 | [dispatch-plan.h](runtime/dispatch-plan.h) | manifest 类型、张量大小、生命周期与 Buffer 重叠校验 |

### 同一个编译器选择后端

两种目标共用 `gac-compile` / `gac-opt`，不必切 Git 分支。以下命令分别生成
LLVM 层 MLIR、执行计划和常量；它们还不是最终可执行机器码：

```bash
mkdir -p build/model-vortex build/model-cuda
gac-compile model.onnx --target=vortex --emit=llvm \
  --manifest build/model-vortex/plan.json --constants-dir build/model-vortex \
  -o build/model-vortex/kernel.mlir
gac-compile model.onnx --target=cuda --emit=llvm \
  --manifest build/model-cuda/plan.json --constants-dir build/model-cuda \
  -o build/model-cuda/kernel.mlir
```

对于完整 BEVFormer-base，另加 `--constant-fold-profile=bevformer-base`。
要从 ONNX 一次生成机器码，使用 `compile-vortex.sh` 或 `compile-cuda.sh`；
两者均可指向同一个 `GAC_NATIVE_BIN`，使用各自独立、全新的输出目录。
Vortex 环境设置见下方“构建与执行”，CUDA 见 [CUDA 使用说明](README.cuda.md)。

| 应共用的内容 | 必须由目标决定的内容 |
|---|---|
| ONNX 导入、GAC 算子定义、SSA、形状和常量推导 | Vortex RV32 / CUDA 64 位指针及地址预算 |
| 图优化与满足语义条件的融合规则 | CSR / NVVM 线程编号、Kernel ABI 与调用约定 |
| 当前标量算子语义、全局 Buffer 生命周期算法 | LMEM / CUDA shared memory 容量与布局、硬件矩阵指令 |
| manifest 通用校验与数值测试方法 | VOLT / NVPTX 机器码生成、设备加载与同步 API |

LMEM、DXA 双缓冲、TCU 与 CUDA Tensor Core 是后续各目标的优化工作；
当前共享标量路径没有因此获得这些功能。硬件限制、Tiling / Schedule 参数和
Runtime 同步方式应保持各自的实现，不能把 Vortex 的参数直接作为 CUDA 配置。

`gac-opt` 必须用于包含 `gac.*` 或 `vortex.*` 的 IR，因为普通上游 `mlir-opt`
没有注册本项目的方言。降成标准 LLVM Dialect 后，可以使用同版本的上游工具。
Operation 类采用 MLIR 正式支持的手写 C++ 注册方式；并非带陌生名字的未注册操作。

## 2. 数据交接与 Buffer

导入器内部的名字表只负责把 ONNX 的 Tensor 名字映射为 `mlir::Value`。交给
下一阶段的是 `OwningOpRef<ModuleOp>`，不是 Python dict，也不是 NumPy 计算结果。
权重是实际 `DenseElementsAttr`；中间激活只有类型/SSA 关系，编译时不假装已经执行。

完整 cold 模型里，固定标定参数、采样几何和 shape 运算可形成纯常量子图。
编译器仅在所需操作数确为常量时折叠：例如常量可见性掩码的 NonZero 可得到静态
索引，后续 Gather 仍然读取运行时图像产生的激活。输入依赖不会因输出 shape 已知
而被替换成参考数据；动态 NonZero 或动态 Split sizes 也不会被假装成静态量。

图优化 Pass 直接操作这个 Module。设备 Lowering 根据优化后的 SSA use-def 关系
建立 dispatch 依赖，并按每个 Value 的最后一次消费规划全局 arena。

- 输入和常量独立分配；中间值使用 64 字节对齐的 arena。
- 输入的最后一次读取发生在 dispatch `i` 时，不能在同一个 dispatch `i` 把它的
  地址直接分给输出；允许复用必须满足 `last_use < new_producer`。
- 图输出的存活期延长到所有 dispatch 结束，避免最终读回前被覆盖。
- Vortex Runtime 按规划顺序提交，并显式依赖上一个事件；CUDA 使用同一个有序
  Stream，保护 arena 地址复用。
- Vortex 设备入口的参数是输入地址、输出地址组成的 uint64 数组；Host Runtime 在
  提交前校验地址范围，设备端解包并转换为 RV32 指针。各 Kernel 的 shape 已静态特化。
- CUDA 设备入口直接接收 64 位指针参数，依次为输入、输出；不会使用 RV32 参数解包。

当前原生 Kernel 是标量正确性基线：每个 SIMT 线程负责一个输出，归约在寄存器内
循环。**LMEM=0、无 DXA、无双缓冲、无 TCU**。这里已有全局 Buffer，不能说没有
Buffer；但不能把全局 arena 复用当成 LMEM 双缓冲。

## 3. 已实现的优化与明确边界

`gac-fuse` 使用 MLIR PatternRewriter，已实现：

1. Conv（包含 ONNX Conv 内置的可选 bias）→ ReLU。
2. Add → ReLU。
3. MatMul → Add → ReLU，其中 Add 不扩大 MatMul 的结果域。

以上仅在中间结果单消费者且 FP32 时融合，避免删除仍被别的节点使用的结果，或
跳过 FP16 中间舍入。Add+ReLU 可以出现在残差路径，**不等于实现了完整的
Residual+LayerNorm 融合**。AttentionFusion 当前未实现，不注册一个空 Pass 冒充完成。

`gac-infer-shapes` 将形状推导与有预算的常量求值运行到固定点，覆盖整数 shape
运算及当前模型需要的常量几何、索引和布局组合。常量求值使用 APInt/APFloat：
FP64 操作按 FP64 语义计算，只有显式 Cast 才改变精度，不先缩成 FP32 再算。
这对相机投影和可见性判断尤其重要；一次错误舍入可能改变后续 Gather 的索引。
整数溢出、零除、非法索引会诊断；元素数、字节数和求值工作量均有上限。
不能折叠的输入依赖计算继续保留在图中执行。

`gac.split` 在导入时保留真实的多个结果。形状已确定且 split sizes 为常量时，
`gac-legalize-tensor-ops` 用 PatternRewriter 创建对应的 Slice 操作与边界常量，
使每个输出独立进入现有 dispatch ABI。它不依靠 Python 拆图，也不因 ONNX
声明了某个输出尺寸，就删除仍然依赖运行时的 split sizes。

设备端 ScatterND 当前要求 **constant indices、reduction=none、索引不重复**。
编译器把索引转成只读的“目标 slice → update slice”表，放入生成的 LLVM/ELF。
执行时每个输出线程选择读取原 data 或 updates，避免多个线程同时 scatter 写入。
data/updates 可以是运行时张量；预计算地址表不等于预计算激活结果。负索引按
ONNX 规则处理，越界与重复索引均拒绝。

LayoutPropagation 验证 NCHW/OIHW 并传播轴顺序，尚不是带性能搜索的自动
NCHW/NHWC 布局选择器。

`gac-normalize-provenance` 在 CSE 前移除内部算子的 `gac.source_name` 和
`gac.output_names`，避免不同 ONNX 显示名称阻止相同 SSA 计算被合并。`loc` 保留来源，
直接返回的公开输出保留名称，`source_op/domain/opset/input_positions` 等语义信息不变。

图中的 `TOSA / Linalg / SCF / Vector` 是可选择的表示层，不能机械地要求所有
算子经过全部层。**本轮实现选择 GAC → SCF 标量循环路线**；尚未实现自动
Linalg tiling、Vector 向量化或 TOSA 全覆盖转换，也不宣称已经完成这些性能优化。

设备端当前要求正的静态 shape、受支持的原生类型及属性。动态 shape、整数 shape
输入未静态求值、未支持的 GridSample 模式等会明确报错。LayerNorm 当前仅允许
FP32/stash_type=1。`gac.ms_deform_attn` 已实现本项目导出契约对应的静态 FP32
流式 Lowering，范围见 [base 适配说明](README.bevformer-base.md)。

## 4. 构建

需要与 VOLT 一致的 LLVM/MLIR **20.1.8 开发树**，以及 CMake、C++17 编译器、
protobuf 开发库与 protoc。ONNX 只需官方 `onnx.proto`；可以来自安装好的 ONNX
包，也可以来自官方 ONNX 源码。核心编译器不链接 Python 或 PyTorch。

```bash
cmake -S native -B build/native -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DONNX_PROTO_DIR=/path/to/onnx
cmake --build build/native --parallel 2
```

需要 C++ Vortex Runtime 时，配置时增加：

```bash
-DVORTEX_BUILD=/path/to/vortex/build \
-DVORTEX_SOURCE=/path/to/vortex/source
```

本次实际验证环境为 WSL Ubuntu 24.04，LLVM 开发树
`/home/thomas/gac-volt-llvm20/build-release`，原生编译器输出目录
`/home/thomas/gac-native-build`。源码仍在当前 Windows 仓库，WSL 直接构建这份源码。

## 5. 查看真正的 MLIR 与 Pass

```bash
# 原始模型导入，尚未跑图优化
gac-compile model.onnx --emit=import -o imported.mlir

# 运行 shape/constant folding、legalization、CSE、fusion、layout
gac-compile model.onnx --emit=optimized -o optimized.mlir

# 对可审查的 IR 单独运行原生 Pass
gac-opt imported.mlir \
  --pass-pipeline='builtin.module(gac-infer-shapes,gac-legalize-tensor-ops,gac-infer-shapes,gac-normalize-provenance,canonicalize,cse,gac-fuse,gac-normalize-provenance,canonicalize,cse,gac-infer-shapes,gac-propagate-layout)' \
  -o optimized.mlir

# 包含 SCF 和 Vortex 操作的设备 IR
gac-compile model.onnx --emit=device --target=vortex -o device.mlir

# 标准 LLVM Dialect + 设备 Buffer/dispatch manifest + 外置权重
gac-compile model.onnx --emit=llvm --target=vortex \
  --manifest output/plan.json --constants-dir output \
  --save-stages output -o output/kernel.mlir
```

标准 ONNX 导入目前针对 opset 13～18；旧 Softmax schema 等不能按新语义猜测。
未知 domain/operator 默认报错，`--allow-unsupported --emit=import` 只用于检查，
不会产生可执行文件。完整 BEVFormer 当前可用严格模式导入：

```bash
gac-compile build/bevformer-onnx-inspect/bevformer-tiny-cold.onnx \
  --emit=import -o build/bevformer-import.mlir
```

该真实模型包含 9,129 个节点、454 个 initializer，原生导入后共有 9,583 个 GAC
操作并通过 MLIR verifier。此前阻塞的 ScatterND 127、Split 55、NonZero 18、Tile 8
个节点现已进入注册的原生 Operation；经过纯常量折叠、合法化和优化后，严格
Lowering 生成 1,548 个运行时 dispatch。没有使用 `--allow-unsupported` 放行执行。

这个结论针对当前导出的 **固定尺寸、固定标定参数、cold 帧** 模型。图像输入仍是
运行时数据，形状为 FP32 `[1,6,3,480,800]`。修改被固化的相机标定需要重新导出/
编译，或另行实现相应的动态形状及索引契约；当前结果不包含 warm 时序路径验收。

`--save-stages` 在一次编译中保存 `01-import.mlir`、`02-optimized.mlir`、
`03-device.mlir`、`04-llvm.mlir`，最终 LLVM Dialect 也写入 `-o` 指定文件。这些是同一编译过程的
分层快照，可用于核查每一步 Operation 的变化。最新整网执行状态与历史失败记录
的区别见[整网验证记录](../docs/validation/vortex/native-mlir-bevformer-full-2026-10-16/README.md)。

## 6. 从 ONNX 到 vxbin 与执行

```bash
export GAC_NATIVE_BIN=/path/to/gac/native/build
export LLVM_PATH=/path/to/pinned/volt-llvm20/build
export VORTEX_BUILD=/path/to/configured/vortex/build
bash native/tools/compile-vortex.sh model.onnx build/native-model

VORTEX_DRIVER=simx LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime" \
  "$GAC_NATIVE_BIN/gac-vortex-run" \
  build/native-model/plan.json build/native-model/kernel.vxbin \
  build/native-model build/native-model/outputs
```

把每个输入按 `plan.json` 中 Tensor 的全局序号写为 `input_<序号>.bin`，按原生 dtype
与连续行主序存储。权重由编译器写为 `tensor_<序号>.bin`。Runtime 输出为
`output_<输出序号>.bin`，报告保存名称、shape 和 dtype 的对应关系。

`src/gac/tools/build_vortex_graph.py` 的 ONNX 分支现在默认调用 `gac-compile`，它只做
CLI 启动与文件组织。可通过 `--native-compiler` 或 `GAC_NATIVE_COMPILER` 指定可执行
文件；找不到编译器会报错，不回退 Python Graph。旧 Graph 入口只在显式传入
`--legacy-python-graph` 时用于对照。JSON、HXP、专用模型构图保留原有接口；新原生
CUDA 路径通过 `gac-compile --target=cuda` 和 `compile-cuda.sh` 使用同一 C++ 前端。

Python 仍可用于测试夹具和参考值；上游 Vortex 的配置脚本与 `vxbin.py` 负责硬件
配置和 ELF 封装。它们不执行本项目的图优化或算子 Lowering。

## 7. 验证与后续范围

数值测试使用 `native/tests/generate_models.py` 生成 ONNX 与参考值，然后由
`native/tests/run_tests.py` 调用原生编译器、MLIR translator 和 clang，执行真正的
本机机器码。`--target=native` 仅是显式的正确性测试目标，不是设备失败时的回退。

`native/tests/run_vortex_tests.py` 验证 RISC-V 对象、入口符号，并可构建 ELF/vxbin，
通过 C++ Runtime 在 SimX 执行。初始 MatMul/Bias/ReLU、Conv/Bias/ReLU、Softmax
三个设备案例已通过；新增 Tile、三结果 Split、ScatterND 整片/单元素负索引更新
四例也通过完整 SimX 数值验证，所有输出最大绝对误差为 0。证据见
[初始原生 MLIR 验证记录](../docs/validation/vortex/native-mlir-2026-10-16/README.md)
和[新增算子 SimX 结果](../docs/validation/vortex/native-mlir-bevformer-full-2026-10-16/full-ops-simx-results.json)。

当前主机回归为 **31 项既有测试 + 6 项几何常量测试 + 15 项新增算子测试**。
测试脚本只生成 fixture、驱动编译器和比较数值，实际运算来自 MLIR 生成的机器码。
整网执行工具、断点和逐层对齐方法见
[tests/README.full_model.md](tests/README.full_model.md)。

完整 Vortex ELF/vxbin 已生成并核对 1,548 个入口；完整主机生成代码的整网数值
已对齐，但完整 Vortex 执行仍待验收，不能把已有的小算子 SimX 结果扩大成
完整模型在设备上通过。
性能优化仍包括更大的融合范围、Linalg/Vector 计算模板、LMEM/DXA 双缓冲与 TCU、
成本模型/autotuning。静态编译成功不能证明这些性能能力已实现，也不能据此给出
物理设备的整网 FPS。
