# GAC MLIR 编译器架构

本文说明 GAC 如何从现有 Python Canonical Graph 逐步演进为基于 MLIR 的 AI 编译器。首个设备目标是 NVIDIA RTX 3070，对应 `cuda-sm86`。当前已经有标准 tensor/linalg IR emitter、严格限定 BEVFormer FFN pattern 的 `mlir-cuda-sm86` runtime-call 混合切片、同一生产形状 FFN 的 `mlir-ptx-sm86` 全 PTX 切片、两次 cuBLAS MatMul 配合两个 MLIR→PTX 融合 Kernel 的 `mlir-cublas-ptx-sm86` 生产候选，以及单缓冲三 dispatch 的 `mlir-tiled-ffn-sm86` 完整 Session。

本文严格区分已经落地的能力和后续路线图。M0/M1 包含 emitter、诊断和外部 verifier 接口；M2a 把两次 MatMul 显式绑定到 CuPy/cuBLAS，并把 `Bias+ReLU` 与 `Bias+Residual+LayerNorm` 绑定到 NVRTC Kernel；M2b 从严格匹配的 Canonical Graph 直接物化三个 MLIR `gpu.func`，再由官方 LLVM 24 `mlir-opt` 完成 GPU Dialect→NVVM→LLVM NVPTX→PTX 7.1；M2c 保留两个 cuBLAS MatMul，把两个融合阶段改成 MLIR PTX，并提供无权重 artifact、严格 Runtime 和 autotune adapter。M2d 已把显式 shared-memory tile 接成单缓冲完整 FFN，并实现真实输入 FP16/TF32 MMA 与 `cp.async` 双缓冲 MatMul 固定尺寸功能候选；后两类尚未接入生产 FFN。这些固定 FFN 路径不是任意 tensor/linalg 图的通用自动 lowering，也不是完整 BEVFormer 网络后端。某次性能结论必须来自保存了设备、范围、协议和原始样本的真机报告。

## 1. 目标与边界

MLIR 在项目中的职责是承载逐层 lowering、做合法性验证并建立可扩展的 Pass 管线。它不替代模型前端、HXP 包格式、设备 Runtime，也不等同于 CUDA Kernel 实现。

```text
JSON / 内存中的 Canonical Graph
          │
          ├── Tensor-level MLIR emitter             当前 M0/M1
          │     func + tensor + linalg + arith + math
          │       └── mlir-opt 解析、验证、规范化
          │
          └── 严格的生产形状 FFN pattern matcher
                │
                ├── mlir-cuda-sm86                  当前 M2a
                │     func + memref + 4 个 runtime calls
                │     ├── 2× CuPy/cuBLAS MatMul
                │     └── 2× NVRTC 融合 Kernel
                │
                ├── mlir-ptx-sm86                   当前 M2b
                      显式 GPU Dialect：3× gpu.func
                        │
                        ▼
                      SCF/Arith/MemRef → NVVM
                        │
                        ▼
                      LLVM NVPTX → textual PTX 7.1 / sm_86
                        │
                        ▼
                      MlirPtxSession 加载 PTX
                        │
                        ▼
                      CUDA Driver JIT → SASS

                └── mlir-cublas-ptx-sm86            当前 M2c
                      2× CuPy/cuBLAS MatMul
                      2× gpu.func 融合 Kernel
                         → NVVM → PTX 7.1 → SASS

mlir-tiled-ffn-sm86 / 功能 MMA 与双缓冲 MatMul / autotune 当前 M2d
  单缓冲完整 FFN；真实输入固定尺寸功能候选；三 Session 选择

通用 tensor/linalg → bufferization → GPU/NVGPU → NVVM → PTX
                                                    路线图 M3
HXP v2 混合 Execution Plan                          路线图 M4/M5
```

六个名称描述不同维度：

- `cuda-sm86` 是设备 target，约束 NVIDIA Ampere Compute Capability 8.6 的指令集、地址空间、资源上限和产物兼容性；
- `mlir-cuda-sm86` 是窄范围的 compiler backend/runtime 组合，只接受一个固定 FFN pattern，并通过显式 runtime call 使用 cuBLAS 和 NVRTC；
- `mlir-ptx-sm86` 接受同一个固定生产形状 FFN，生成三个 `gpu.func`，由官方 MLIR GPU lowering 和 LLVM NVPTX 后端生成 textual PTX，Runtime 加载该 PTX，并由 CUDA Driver JIT 为 SASS；
- `mlir-cublas-ptx-sm86` 接受同一个固定生产形状 FFN，执行两次 CuPy/cuBLAS MatMul，并加载由官方 MLIR/NVPTX 生成的两个融合 PTX entry；不使用 NVRTC，也没有其他后端 fallback；
- `mlir-tiled-ffn-sm86` 接受同一个固定生产形状 FFN，用两个单缓冲 FP32 tiled MatMul Kernel 和一个最终融合 PTX Kernel 执行三个 dispatch；它已有完整 Session 与无权重 artifact，并进入三候选 autotune；
- 通用 `mlir-nvvm` 仍是目标 compiler backend，要求任意已支持 tensor/linalg 区域经过通用 bufferization、tiling、GPU/NVGPU 和 NVVM lowering。M2b/M2c 的 pattern-specialized 路径与 M2d 的显式 schedule 都没有证明这项通用能力。

这种拆分允许以后让同一个设备 target 使用不同编译后端，也允许同一个 MLIR 前端接 CPU、其他 GPU 或 NPU 后端。

## 2. 当前实现状态

| 里程碑 | 当前能力 | 可验证证据 | 明确未包含 |
| --- | --- | --- | --- |
| M0：MLIR 接入 | 从 GAC 静态 Tensor 图生成确定性的 textual MLIR；模块使用已注册的 `builtin`、`func`、`tensor`、`linalg`、`arith` 和 `math` 方言 | 单元测试覆盖文本生成、诊断和工具调用封装；产物可交给本地 LLVM 源码构建出的 `mlir-opt` 解析和验证，成功结果须由实际命令记录 | Python MLIR binding、二进制 bytecode、GPU 代码生成 |
| M1：首个算子切片 | 手工 JSON 图中的 BEVFormer-like tiny FFN/MLP smoke；覆盖 MatMul、Bias Add、GELU/ReLU、`AddReLU` 和末轴 LayerNorm，包含静态 shape 和广播约束 | 单元测试检查确定性、类型/shape 拒绝路径和 MLIR 文本；可选集成测试在配置实际 `mlir-opt` 后运行 | 官方 checkpoint importer、真实 BEVFormer 子图导出、完整 Encoder/Decoder、MSDA、GridSample、Conv 等全图算子覆盖 |
| M2a：SM86 FFN runtime-call 切片 | 严格匹配 `MatMul → AddReLU → MatMul → Add → Add(query residual) → LayerNorm`；发出两次 CuPy/cuBLAS MatMul 和两个 NVRTC 融合 Kernel 的显式 call plan | runtime-call MLIR ABI manifest 已由 LLVM 24 `mlir-opt --verify-each` 验证；独立的 Python call records 已在 RTX 3080 SM86 上完成输出正确性测试 | GPU/NVGPU/NVVM/PTX、HXP v1 打包、任意 MLP、完整 BEVFormer |
| M2b：生产形状 FFN MLIR→PTX 切片 | 只接受 `[1,2500,256] → 512 → 256` 的 FP32 FFN；从严格 pattern 生成三个 `gpu.func`，由官方 MLIR GPU→NVVM pipeline 与 LLVM NVPTX 生成 textual PTX 7.1 | 保存实际 GPU MLIR、含唯一 `gpu.binary` 的 lowered MLIR、PTX、SHA-256 和工具链 revision；RTX 3080 上最大绝对误差 `7.153e-7` | 通用 tensor/linalg→GPU 自动 lowering、Tensor Core 调度、HXP v1 打包、任意 MLP、完整 BEVFormer |
| M2c：cuBLAS+MLIR PTX 生产候选 | 同一生产 FFN 的固定四调用计划：cuBLAS FC1、PTX `Bias+ReLU`、cuBLAS FC2、PTX `Bias+Residual+LayerNorm`；无 NVRTC | GPU MLIR→PTX 编译、entry/指针 ABI、call plan、Runtime 和无权重目录均有测试；RTX 3070 Ti Laptop 两轮为 `222.131 / 222.694 μs/FFN`，autotuner 均选择 M2c | cuBLASLt、HXP v1、完整 BEVFormer、桌面 RTX 3070 性能 |
| M2d：Kernel 调度实验 | 单缓冲 16×16×16 tiled 完整 FFN；真实输入 FP16/TF32 NVGPU MMA 与 `16×16×64` 双 stage `cp.async` MatMul；正确性优先的 autotuner | tiled Session 已与 M2a/M2c 同轮做三候选预验收；TF32/FP16 MMA 最大绝对误差为 0、分别为 `3.879600 / 4.012688 μs`；双缓冲 MatMul 逐元素 exact、为 `2.652160 μs` | 固定 commit 的 tiled 正式报告、双缓冲生产 FFN、生产 shape MMA/Tensor Core FC1/FC2、cuBLASLt |

M0/M1 的价值不是“把 CUDA 字符串换成另一种字符串”。实际执行 `mlir-opt` 时，它会解析 IR 和方言，并检查 SSA 使用、类型、区域、block 参数和算子约束；只有该命令成功，特定产物才算通过 verifier。通用 tensor/linalg 产物仍停在 tensor-level MLIR。M2a 的执行来自显式 CuPy/cuBLAS 与 NVRTC runtime binding。M2b 真正生成并加载三个 PTX Kernel；M2c 真正加载两个 MLIR PTX 融合 Kernel，并用结构化 call record 调度两个 cuBLAS MatMul；M2d 的 tiled Session 真正加载三个 PTX entry，MMA/异步候选也真正加载固定尺寸 PTX 并运行真实输入。它们都先对固定 FFN 或固定微内核做 pattern-specialized GPU IR 物化，不等于任意 tensor/linalg 图已经具备通用自动 lowering。

BEVFormer 当前完整生产路径仍由已有 CUDA/HXP 后端承担。MLIR 样例首先覆盖 FFN 与 `AddReLU`，原因是这部分语义清楚、静态 shape 稳定，适合先检查 IR 边界、Pass 基础设施和 runtime-call ABI，再处理 BEV 专用算子。CLI 示例仍是手工编写的 tiny JSON 图；生产尺寸 benchmark 使用相同构图器生成一个 FFN 片段。两者都不来自官方 checkpoint importer，也没有证明真实 BEVFormer 模块或完整图已经接入 MLIR。

## 3. 分层设计

### 3.1 前端与 Canonical Graph

现有 CUDA/BEVFormer 路径的模型导入层负责权重映射、输入合同和 cold/warm 状态语义。Canonical Graph 保存 TensorSpec、节点属性、常量和来源标识，是框架无关的模型语义层。当前 MLIR M0/M1 直接接收 JSON 或内存中的 Canonical Graph；官方 checkpoint 到 MLIR 的 importer 和真实 BEVFormer 子图导出尚未实现。

在进入 MLIR 前，编译器必须完成：

- shape、dtype 与广播合法性检查；
- 静态 Profile 选择；完整 BEVFormer 未来需要复用现有固定 Profile 合同；
- 常量和输入的明确区分；
- 从原模型节点到 MLIR operation 的可追溯标识；
- 不支持算子的显式诊断，禁止静默丢节点或改用错误语义。

### 3.2 Tensor-level MLIR

当前 emitter 将一个 GAC Graph 映射为 `module` 和单个 `func.func`。函数参数和返回值使用 ranked tensor type；权重成为常量；矩阵乘使用 `linalg.matmul`；带广播的 Bias Add、激活和融合后的 `AddReLU` 使用 `linalg.generic`、`arith` 与 `math` 标量算子表达。

M0/M1 为了让测试产物完全自包含，把小型 smoke 权重写成 `dense` 常量。完整 BEVFormer 不能用巨大的文本常量承载全部 checkpoint；进入生产包前需要资源化 ElementsAttr 或外部权重 ABI，并继续由 HXP 做摘要、边界和生命周期校验。

这一层保留 Tensor 语义，不提前决定显存 offset、CUDA block 大小或具体库调用。这样规范化、CSE、融合、算子分解和数值检查可以与最终设备代码生成解耦。

### 3.3 `mlir-cuda-sm86` FFN runtime-call 切片

当前可执行切片由 [`src/gac/backends/mlir_cuda.py`](../src/gac/backends/mlir_cuda.py) 与 [`src/gac/mlir_runtime.py`](../src/gac/mlir_runtime.py) 实现。它先调用 tensor-level `compile_mlir(..., fuse=True)`，再检查融合后的图是否**完整且恰好**为：

```text
MatMul → AddReLU → MatMul → Add(fc2 bias) → Add(query residual) → LayerNorm
```

匹配器还要求一个输入、一个输出、静态连续 FP32 Tensor、常驻的两组权重/偏置和 LayerNorm scale/bias、残差必须是原始 query、输出 shape 必须等于 query，且通道数不超过 4096。任何多余节点、缺失节点、顺序变化、非常量参数或 shape 不匹配都会在编译期报错；不尝试拆分子图，也不会转到 NumPy 或旧 generated-CUDA 后端。

匹配成功后，后端生成标准 `func`/`memref` MLIR。四个 private function symbol 定义 runtime ABI，入口函数按固定顺序描述调用关系：

| 顺序 | Runtime symbol | 实现 | 融合边界 |
| ---: | --- | --- | --- |
| 1 | `gac_cublas_fc1_matmul_f32` | `cupy.matmul(..., out=...)` / cuBLAS | FC1 MatMul |
| 2 | `gac_bias_relu_inplace_f32` | CuPy `RawModule` / NVRTC Kernel | FC1 bias + ReLU，原地写 hidden buffer |
| 3 | `gac_cublas_fc2_matmul_f32` | `cupy.matmul(..., out=...)` / cuBLAS | FC2 MatMul |
| 4 | `gac_bias_residual_layer_norm_f32` | CuPy `RawModule` / NVRTC Kernel | FC2 bias + query residual + LayerNorm |

这份 MLIR source 是可审阅、可交给 `mlir-opt` 验证的 ABI manifest，并不由 Runtime 解析或执行。编译器同步生成结构化 Python call records；它们才是实际 dispatch 的唯一来源。`MlirCudaSession` 校验 call record 的 symbol、输入输出、grid/block 与 alias 合同后才运行，不会按 MLIR 文本或 graph op 名称推测调用。

末尾 Kernel 每行启动一个 256-thread block，用 block reduction 计算均值与方差。所有调用使用同一条 CuPy non-blocking stream。Session 建立时把常量驻留到设备，分配输入、输出和两个 FP32 中间 workspace，并通过 NVRTC 编译融合 Kernel。执行失败会使 Session 失败，不会逐调用替换实现。

命令行入口直接读取 JSON graph：

```bash
python -m gac.cli run examples/bevformer_tiny_ffn.json \
  --backend mlir-cuda-sm86 \
  --verify \
  --report build/bevformer_tiny_ffn.mlir-cuda.json
```

运行设备的 Compute Capability 必须恰好为 8.6，并要求 CuPy、CUDA Runtime、NVRTC 和 cuBLAS 可用。HXP v1 没有保存该 runtime-call plan，因而 `run model.hxp --backend mlir-cuda-sm86` 会被明确拒绝。当前 `--trace`/`--profile` 也只服务旧 `cuda-sm86` Session；FFN 切片的 device-resident 测量使用 [`tools/benchmark_mlir_cuda_ffn.py`](../tools/benchmark_mlir_cuda_ffn.py) 在整个四调用计划外放置一对 CUDA Event。

真机基准只覆盖生产尺寸 `[1,2500,256] → 512 → 256` 的单个 FFN 片段。报告必须分别保存 device-resident 四调用 CUDA Event、同步 wall、公共 `run` 的 H2D+计算 GPU 时间、含输出 D2H 的 wall 时间、NumPy 正确性、执行顺序，以及 MLIR manifest、call plan 和 Kernel source 的 hash。比较时在同一进程共驻留两个 Session，每个 repeat 后交换 MLIR/CUDA 的先后位置，并用两种首轮顺序独立重跑，避免把库初始化、热态或动态频率偏差误写成编译器收益。任何具体数字都必须直接来自最终原始 JSON，并且只属于报告中的设备和 FFN 片段，不是完整 BEVFormer、RTX 3070 或通用 NVVM lowering 的结果。证据边界和两份原始报告见[验证说明](validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)。

### 3.4 `mlir-ptx-sm86` 生产形状 FFN 全 PTX 切片

全 PTX 切片由 [`src/gac/backends/mlir_ptx.py`](../src/gac/backends/mlir_ptx.py) 与 [`src/gac/mlir_ptx_runtime.py`](../src/gac/mlir_ptx_runtime.py) 实现。它从 Canonical Graph 分支复用 M2a 的严格 FFN pattern 检查，但进一步把支持范围固定为生产尺寸 `[1,2500,256] → 512 → 256`、FP32、连续布局、一个 query 输入和一个输出。Tensor-level MLIR 仅作为独立的调试/审阅输出，不是 GPU emitter 的输入，也不进入部署 PTX artifact。匹配成功后，编译器直接物化三个 MLIR GPU Kernel：

| PTX entry | 计算 |
| --- | --- |
| `gac_mlir_ptx_fc1_bias_relu_f32` | FC1 MatMul + bias + ReLU |
| `gac_mlir_ptx_fc2_f32` | FC2 MatMul |
| `gac_mlir_ptx_bias_residual_layer_norm_f32` | FC2 bias + query residual + LayerNorm |

GPU module 使用已注册的 `gpu`、`scf`、`arith`、`memref` 和 `nvvm` 方言。官方 LLVM 24 `mlir-opt` 执行 `gpu-lower-to-nvvm-pipeline`，参数固定为 bare-pointer Kernel ABI、`nvptx64-nvidia-cuda`、`sm_86`、`+ptx71`、ISA 输出和优化等级 3。lowered module 必须只含一个 `gpu.binary` 和一个 GPU object，不得残留 `gpu.module` 或 `gpu.func`；抽取出的文本必须恰好包含 `.target sm_86`、`.address_size 64` 和上表三个 entry。

编译产物同时保存输入 GPU MLIR 的 SHA-256、lowered binary MLIR、PTX、实际命令、`mlir-opt --version`、LLVM source revision、目标和 entry 集合。部署机器不需要安装 MLIR：`assemble_mlir_ptx_program` 用确定性 GPU module 的 hash 绑定离线产物，重新从 binary MLIR 抽取并比对 PTX；`MlirPtxSession` 在 CUDA 初始化前复核图、launch ABI、target 和完整性摘要，再由 CuPy `RawModule(path=...)` 加载 textual PTX。CUDA Driver 在模块加载时仍会把 PTX JIT 为 SASS；该路径避免的是部署机上的 MLIR lowering、CUDA C 和 NVRTC 编译，不是完全取消运行时代码生成。执行路径不调用 cuBLAS，也没有 CPU 或旧 CUDA 后端回退。

这个切片的“全流程”指其三个设备 Kernel 均由 MLIR GPU Dialect 经 NVVM/LLVM NVPTX 生成到 PTX；它不表示 tensor/linalg IR 已由通用 Pass 自动变成任意 GPU 程序。当前编译器是在验证固定 Canonical Graph 后，按受控模板显式生成 GPU Dialect，完整 CNN、Attention、MSDA、Decoder、Detection Head 和跨帧状态不在该切片内。

RTX 3080 SM86 上的同进程、双顺序、配对交替测试结果如下。每轮 20 个 repeat，每个 repeat 50 次 FFN：

| 首次测量次序 | 全 PTX device-resident | cuBLAS hybrid device-resident | 全 PTX 慢于 hybrid |
| --- | ---: | ---: | ---: |
| PTX first | 1.102 ms | 0.209 ms | 5.265× |
| Hybrid first | 1.092 ms | 0.214 ms | 5.150× |

全 PTX 最大绝对误差为 `7.153e-7`。这些数字只属于该 FFN 编译单元和 RTX 3080；它们不是完整 BEVFormer 延时，也不能替代 RTX 3070 实测。原始 JSON、GPU MLIR、binary MLIR、PTX 与摘要见 [M2b 验证目录](validation/sm86/mlir-ptx-2026-10-11/README.md)。

MLIR→PTX 本身不会自动提速。离线 textual PTX 可以移除部署机上的 MLIR 编译依赖，并避免 CUDA C/NVRTC 编译；CUDA Driver 仍会在加载 PTX 时 JIT 为 SASS，因此模块加载时间必须与稳态 Kernel 延时分开记录。它不会自动减少稳态 Kernel 的计算量。首版两个 MatMul 都是“一线程计算一个输出，并串行遍历 K”的正确性调度，所以比 cuBLAS hybrid 慢 5.150～5.265 倍。本轮性能工作把成熟 library-call 路径与生成 Kernel 研究分开推进。

### 3.5 `mlir-cublas-ptx-sm86` 四调用生产候选

[`src/gac/backends/mlir_cublas_ptx.py`](../src/gac/backends/mlir_cublas_ptx.py) 复用同一个严格 matcher，只接受静态连续 FP32 `[1,2500,256] → 512 → 256`。其 call record 顺序是：

| 顺序 | 实现 | 融合边界 |
| ---: | --- | --- |
| 1 | CuPy/cuBLAS | FC1 MatMul |
| 2 | MLIR→NVVM→PTX | Bias + ReLU，原地写 hidden buffer |
| 3 | CuPy/cuBLAS | FC2 MatMul |
| 4 | MLIR→NVVM→PTX | Bias + query residual + LayerNorm |

两个融合 `gpu.func` 经官方 LLVM 24 pipeline 生成 PTX 7.1。[`MlirCublasPtxSession`](../src/gac/mlir_cublas_ptx_runtime.py) 在访问 CUDA 前验证整个 program、PTX entry、裸指针参数数量、`.maxntid 256,1,1` 和四调用实现顺序，再用 `RawModule(path=...)` 加载 PTX。Runtime 不编译 CUDA C/NVRTC，也不回退到 NumPy、旧 CUDA 或全 PTX。MatMul 是 `cupy.matmul(..., out=...)` 的 cuBLAS 路径；当前没有直接调用 cuBLASLt，也没有把 epilogue 下推到 cuBLASLt。

[`mlir_cublas_ptx_artifact.py`](../src/gac/mlir_cublas_ptx_artifact.py) 可把 `fusions.gpu.mlir`、`fusions.binary.mlir`、`fusions.ptx` 与 manifest 原子发布为无权重目录。加载器限制成员集合、常规文件类型与大小，拒绝符号链接、重复 JSON key、摘要和目标不匹配，再用调用者提供的独立已验证 graph 重建完整计划。HXP v1 仍不承载这类 artifact。

RTX 3070 Ti Laptop（SM86、8 GB）上的两轮正式结果如下；每轮 warmup 50，40 个 repeat，每个 repeat 100 次完整四调用 FFN：

| 首次测量次序 | M2c cuBLAS+PTX | M2a cuBLAS+NVRTC | M2c 位置平衡加速 |
| --- | ---: | ---: | ---: |
| PTX first | 222.131 μs | 227.190 μs | 1.02414× |
| NVRTC first | 222.694 μs | 228.204 μs | 1.02677× |

两轮 `allclose=True`，M2c 最大绝对误差为 `7.152557e-7`，autotuner 都选择 `mlir-cublas-ptx-sm86`。M2c 公共 `run` 的 H2D+计算 GPU 中位数为 `565.072 / 564.096 μs`，含 D2H 与同步的 wall 中位数为 `1240.842 / 1203.280 μs`。这只证明一个生产形状 FFN 在这台 Laptop GPU 上略快于语义相同的 NVRTC 融合路径。RTX 3080 上的 `0.209～0.214 ms/FFN` 是旧 hybrid 的跨设备历史基线，不能与这里的绝对延时直接横比。

### 3.6 tiled、NVGPU/MMA、异步双 stage 与 autotuning

[`mlir_tiled_matmul.py`](../src/gac/mlir_tiled_matmul.py) 直接为生产 FC1 `[2500,256]@[256,512]` 和 FC2 `[2500,512]@[512,256]` 物化显式 GPU Dialect。每个 block 计算一个 output tile，线程用带边界检查的 strided loop 协作填充 A/B 两个 workgroup-memory tile，经过 barrier 后累加。16×16×16 候选每 block 为 256 线程、静态 shared memory 为 2048 bytes；LLVM 24 实际 PTX 检查到两个 shared 声明、shared load/store、两次 barrier 与向量 shared load。FC1 在同一 Kernel 合并 bias+ReLU。[`mlir_tiled_ffn.py`](../src/gac/mlir_tiled_ffn.py)、[`mlir_tiled_ffn_runtime.py`](../src/gac/mlir_tiled_ffn_runtime.py) 与 [`mlir_tiled_ffn_artifact.py`](../src/gac/mlir_tiled_ffn_artifact.py) 已把 FC1、FC2 和最终 Bias+Residual+LayerNorm 接成三个 PTX dispatch 的完整 `mlir-tiled-ffn-sm86` Session。它仍是单缓冲且一线程负责一个标量输出。

同一模块提供真实输入的 FP16 与 TF32 `nvgpu.mma.sync` 功能候选。TF32 计算 `A[16,16] f32 × B[16,8] f32 → C[16,8] f32`，用 4 条 `mma.sync.aligned.m16n8k4` 覆盖 K=16，最终 PTX 审计为 12 次 global load 和 4 次 store；FP16 计算 `A[16,32] f16 × B[32,8] f16 → C[16,8] f32`，用 2 条 `mma.sync.aligned.m16n8k16` 覆盖 K=32，最终 PTX 有 24 次 global load 和 4 次 store。RTX 3070 Ti Laptop 上，两者 `max_abs_error=0`，每次 launch 的 CUDA Event 中位数分别为 `3.879600 μs` 和 `4.012688 μs`（warmup 50、30 个样本、每样本 1000 次 launch）。它们仍是固定尺寸功能候选，没有生产 shape 的边界处理、占用率验收，也未接入 FC1/FC2 或完整 FFN。

[`mlir_async_tiled_matmul.py`](../src/gac/mlir_async_tiled_matmul.py) 把异步双 stage 从调度 smoke 升级为真实输入 FP32 `A[16,64] × B[64,16] → C[16,16]` MatMul 功能候选。最终 PTX 有 8 条 `cp.async`、4 条 `cp.async.commit_group`，四次 wait 的深度为 `[1,1,1,0]`，并包含 4096 bytes shared memory 与 7 个 barrier。RTX 3070 Ti Laptop 上 256 个输出逐元素 exact，`max_abs_error=0`，每次 launch 的 CUDA Event 中位数为 `2.652160 μs`（warmup 20、30 个样本、每样本 200 次 launch）。此前 128-byte、2-copy 的 schedule probe 继续作为较小的编译器调度证据；新的功能候选仍未接入生产 FC1/FC2 或上述完整 FFN，因此不能把其微基准解释为 FFN 加速。

[`mlir_ptx_autotune.py`](../src/gac/mlir_ptx_autotune.py) 用 `(backend, tile, block, precision, fusion)` 描述候选。每个候选先过数值正确性门，再采集指定重复次数的样本，并以中位数最小者获胜；完全相同的中位数用候选 SHA-256 确定性破同分。缓存绑定 graph/source SHA-256、设备描述、候选空间和测量配置，以原子 replace 发布，并拒绝损坏、重复 key 或上下文变化。

[`mlir_cublas_ptx_benchmark.py`](../src/gac/mlir_cublas_ptx_benchmark.py) 提供真实 CUDA adapter：三个常驻 Session 用 CUDA Event 测完整 FFN，`mlir-cublas-ptx-sm86`、`mlir-cuda-sm86` 和 `mlir-tiled-ffn-sm86` 都先过正确性门再进入选择。RTX 3070 Ti Laptop 的预验收采用 warmup 50、100 次/repeat、40 个 repeat 和 30 个 public samples，device-resident 中位数分别为 `0.221634 / 0.227610 / 0.891556 ms/FFN`；三者均 eligible，tiled 最大绝对误差为 `7.152557e-7`，最终选择 M2c。该轮还未固定最终 Git commit，数字只能标为预验收，不能替代正式报告。固定尺寸 MMA 与双缓冲 MatMul 没有完整 FFN Session，仍不参与这组三候选 autotune。双缓冲 tiled FFN、生产 Tensor Core FC1/FC2、cuBLASLt 和更多 tile 候选尚未完成。

复现编译证据：

```bash
PYTHONPATH=src python tools/probe_mlir_tiled_matmul.py \
  --output-dir build/mlir-tiled-probe \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  --which both --tile-m 16 --tile-n 16 --tile-k 16 \
  --probe-functional-mma

PYTHONPATH=src python tools/probe_mlir_async_tiled_matmul.py \
  --output-dir build/mlir-async-tiled-matmul \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
```

### 3.7 Pass 管线

Pass 管线按“每一阶段都有合法 IR 集合”的方式设计：

1. **Import cleanup**：canonicalize、CSE、常量规范化和死代码清理；
2. **Domain decomposition**：把组合算子分解或融合为后端支持的模式；
3. **Bufferization**：One-Shot Bufferize，把 Tensor SSA 值转换为显式 memref，并建立 alias/lifetime；
4. **Loop/GPU lowering**：将 `linalg` 降为并行循环，完成 tiling、fusion、vectorization 和 GPU mapping；
5. **NVVM lowering**：把 GPU/NVGPU 与向量操作降到 NVVM/LLVM 方言；
6. **Translation**：导出 LLVM IR，经过 NVPTX 目标生成 PTX，再由 CUDA 工具链生成可加载产物。

通用 tensor/linalg lowering 当前只落地第 1 阶段及 M1 所需的 tensor-level 表达。M2a FFN 切片通过严格 pattern matcher 直接物化 runtime-call ABI。M2b 与 M2c 也使用严格 matcher，跳过尚未实现的通用 One-Shot Bufferize 与 linalg tiling，直接生成显式 GPU Dialect；此后第 5、6 阶段由官方 `gpu-lower-to-nvvm-pipeline` 和 LLVM NVPTX 后端真实完成。M2d tiled 候选显式写出 workgroup tile 和 GPU mapping，仍不是从任意 `linalg.matmul` 自动推导。因而这些路径证明了受控 GPU module 到 PTX 的编译、加载或指令生成能力，尚未证明第 2 至第 4 阶段对任意 tensor/linalg 图通用。每个通用阶段仍应在边界运行 verifier，并检查上一层不允许残留的方言，避免把 lowering 缺口推迟到 Runtime 才暴露。

### 3.8 BEV 专用方言

完整 BEVFormer 有两个不能简单伪装成普通逐元素算子的核心操作：

- `bev.ms_deformable_attention`：表达多尺度特征、sampling location、attention weight 和 level metadata 的联合语义；
- `bev.grid_sample_2d`：表达坐标归一化、插值、padding 和边界行为。

它们计划进入独立 `bev` 方言，并定义 operation verifier、shape inference、canonicalization 和 lowering interface。早期可降低到 `linalg`/`scf` 的正确性实现；性能路径再匹配为专用 CUDA Kernel。方言本身只表达语义，不能把某个 CUDA 实现细节写进高层 op。

上述自定义方言尚未实现。当前 M1 不会用未注册 operation 假装完成完整 BEVFormer，也不会用 `allow-unregistered-dialect` 绕过验收。

### 3.9 Runtime 与包格式

编译器产物最终仍由 GAC Runtime 执行。规划中的 HXP v2 会保存多种 artifact、target requirements 和显式 Host Execution Plan，例如：

```text
load constant
allocate device buffers
launch MLIR-generated cubin
call cuBLAS / cuDNN
launch existing BEV CUDA kernel
record dependency/event
publish outputs and next_bev
```

这是一条混合执行路径：MLIR 负责适合通用 lowering 和融合的区域，vendor library 与已验证的专用 Kernel 继续执行更合适的区域。跨区域 tensor ABI、stream 顺序、workspace 和状态提交必须由计划明确描述。

HXP v2 混合执行尚未实现。现有 HXP v1 仍是完整 CUDA 路径的有效包格式和校验边界，但它只描述旧 CUDA backend 的源码、launch metadata 和静态内存计划，不能序列化 M2a 的 runtime-call plan、M2b 的全 PTX artifact、M2c 的 cuBLAS+PTX plan 或 M2d 的 tiled FFN plan。M2a 因此由 JSON graph 直接编译并在 `MlirCudaSession` 中运行；M2b 可离线编译并由 `MlirPtxSession` 加载；M2c 与 tiled FFN 的独立无权重目录分别由 `mlir_cublas_ptx_artifact.py` 和 `mlir_tiled_ffn_artifact.py` 写入和校验，但仍需调用者提供 graph 常量，尚未封装进 HXP。

## 4. 为什么保留 cuBLAS、cuDNN、Runtime 与 HXP v1

引入 MLIR 不等于重写所有已验证组件。

- **cuBLAS/cuDNN**：矩阵乘和卷积的高性能实现包含架构相关算法选择、workspace 管理和多年调优。M2a 与 M2c 都把两个 MatMul 显式降低为 CuPy/cuBLAS 调用；后续通用路径仍应通过 library-call lowering 或 HXP 调度复用 vendor library。当前没有直接使用 cuBLASLt。
- **GAC Runtime**：M2a 的 `MlirCudaSession` 负责输入绑定、驻留常量、stream、NVRTC module 和错误传播；M2b 的 `MlirPtxSession` 校验并加载离线 PTX；M2c 的 `MlirCublasPtxSession` 组合 cuBLAS 与两个已编译 PTX entry；M2d 的 `MlirTiledFfnSession` 校验并执行三个 PTX dispatch。MLIR/NVVM 生成的设备程序不负责完整 BEVFormer 跨帧 `prev_bev` 提交/回滚和全图 profile；这些职责继续属于 Runtime。
- **HXP v1**：它已经定义常量、图、CUDA artifact、摘要、目标检查和安全加载合同。保留 v1 可以让 MLIR 研发期间的生产 CUDA 路径继续可运行，并为 v2 迁移提供可比较基线。
- **现有 CUDA BEV Kernel**：MSDA、GridSample、运动补偿等路径已有正确性证据。MLIR 版本必须先达到数值和设备验收门槛，才能逐块替换。

这样的迁移方式把编译器结构变化与模型可用性分开：一个区域完成验证后才进入混合计划，失败时编译阶段明确拒绝，不在运行中静默改变算法。

## 5. 本地 LLVM/MLIR 工具链

本次使用的源码不是系统中未知版本的预编译工具，而是项目目录旁的本地 LLVM checkout：

| 项目 | 值 |
| --- | --- |
| 源码目录 | `C:/Users/Thomas/Desktop/universal_ai_compile/llvm-project` |
| LLVM/MLIR 版本 | `24.0.0git` |
| Git revision | `765392a6b4498f1dc57233f03a31b87462fdef01` |
| 构建环境 | WSL Ubuntu 24.04、CMake 3.28、GCC 13 |
| LLVM targets | `X86;NVPTX` |
| 首批工具 | `mlir-opt`、`mlir-translate`、`llc` |

Windows 工作树里的部分脚本带 CRLF，因此本机验证使用同一 revision 的 LF 源码副本。副本不改变 Git 内容，也不构成新的源码版本。仓库提供了可复现脚本 [`tools/build_mlir_wsl.sh`](../tools/build_mlir_wsl.sh)，在 WSL 中运行：

```bash
cd /mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler
bash tools/build_mlir_wsl.sh
```

脚本构建成功后会提供 MLIR 解析、Pass 和 NVPTX 相关工具。工具存在本身不证明某个 GAC 图已经完成 lowering；M2b 还必须保存实际 GPU MLIR、lowered binary MLIR、PTX、命令和真机执行证据。脚本用 Git revision 隔离受管源码和 build cache，并写入 `.gac-build-revision` marker。编译报告记录实际 executable、`mlir-opt --version` banner、source revision（受管 marker 存在时）、命令参数、Pass pipeline、超时和源码/产物 SHA-256，避免只保存一个不可追溯的 `.mlir` 文件。

### 5.1 本次真实验证证据

本次不是只做字符串快照测试。使用上表 revision 构建的 `mlir-opt 24.0.0git` 后，进行了以下真实调用：

- 完整 `examples/bevformer_tiny_ffn.json` 经 `mlir-compile` 生成 MLIR，再执行 `--verify-each --canonicalize --cse`，CLI 报告为 `verified: true`；
- 在 WSL 内运行后端集成测试，完整 tiny FFN 以及 rank-3 MatMul、singleton broadcast、rank-1 LayerNorm、GELU、`MatMulBiasGELU` 和 scalar Add 均通过真实 parser/verifier；
- `mlir-cuda-sm86` 生成的 `func`/`memref` runtime-call ABI manifest 已由同一 LLVM 24 `mlir-opt --verify-each` 解析和验证；
- RTX 3080 SM86 真机测试验证四条 call record 的实际执行与 NumPy 参考结果；生产尺寸 FFN benchmark 工具记录 warm-up、样本分布、device-resident CUDA Event、公共 `run` 边界和同轮 CUDA codegen + cuBLAS 基线，比较报告保存 artifact hash，并在同进程内完成位置平衡的配对交替测量及两种首轮顺序重跑；
- [`examples/mlir_sm86_add_relu.mlir`](../examples/mlir_sm86_add_relu.mlir) 作为独立的手写 GPU Dialect smoke，执行 `gpu-lower-to-nvvm-pipeline`，目标 `cubin-chip=sm_86`、`cubin-format=isa`，成功生成一个 `gpu.binary`；内嵌 PTX 包含 `.target sm_86` 与 `add_relu_broadcast` entry，生成结果再次执行 `--verify-each` 通过；
- M2b 的生产形状 FFN 从 Canonical Graph 生成三个 `gpu.func`，由同一官方 pipeline 生成 textual PTX 7.1；实际产物包含唯一 `gpu.binary`、`.target sm_86`、64 位地址和三个精确 entry，产物与工具链 revision 均已保存；
- RTX 3080 上，全 PTX FFN 最大绝对误差为 `7.153e-7`；两种首次测量顺序下 device-resident 中位数为 `1.102 ms` 和 `1.092 ms`，对应 cuBLAS hybrid 为 `0.209 ms` 和 `0.214 ms`，即全 PTX 慢 `5.265×` 和 `5.150×`。
- RTX 3070 Ti Laptop 上，M2c 在 PTX-first / NVRTC-first 两轮的 device-resident 中位数为 `222.131 / 222.694 μs`，同轮 M2a 为 `227.190 / 228.204 μs`，位置平衡加速 `1.02414× / 1.02677×`；两轮 autotuner 均选择 M2c；
- M2d 的 16×16×16 FC1/FC2 tiled module 已由同一 LLVM 24 revision 生成 PTX 7.1；指令审计分别确认两个 shared 声明、shared load/store、两次 barrier 和四处向量 shared load，且没有 `mma.sync`；单缓冲 tiled Kernel 已组成完整三 dispatch FFN Session；
- 三候选预验收中，tiled、M2c、M2a 都通过正确性门并进入计时，device-resident 中位数为 `0.891556 / 0.221634 / 0.227610 ms/FFN`，选择 M2c；该轮等待固定 Git commit 后正式重跑；
- TF32 与 FP16 的真实输入 NVGPU MMA 候选分别生成 4 条 `m16n8k4` 和 2 条 `m16n8k16` `mma.sync`，RTX 3070 Ti Laptop 上最大绝对误差均为 0，单次 launch 中位数为 `3.879600 / 4.012688 μs`；
- 真实输入 `16×16×64` 异步双 stage MatMul 的 PTX 含 8 条 `cp.async`、4 个 commit、`wait_group [1,1,1,0]` 和 4096 bytes shared memory，真机逐元素 exact，单次 launch 中位数为 `2.652160 μs`。MMA 与双缓冲结果均是固定尺寸功能候选，不是生产 FFN 性能。

手写 AddReLU smoke 只证明 stock MLIR 24 与 LLVM NVPTX 工具链可用。M2b 进一步证明 GAC 的固定生产形状 FFN 能生成、校验并执行真实 PTX；由于它通过 matcher 直接物化 GPU Dialect，任意 tensor/linalg 图的自动 GPU lowering 仍属于 M3。

## 6. RTX 3070 / SM86 适配原则

RTX 3070 的首个 target 固定为 `cuda-sm86`。MLIR M0/M1 验证设备无关的 tensor IR；M2a、M2b、M2c 与 M2d Runtime 都要求 Compute Capability 恰好等于 8.6，PTX 产物还固定 `.target sm_86`、PTX 7.1 和 64 位地址。后续通用 GPU lowering 必须把以下内容纳入 target profile：

- Compute Capability 8.6 与 NVPTX feature；
- 支持的 dtype、向量宽度、对齐和连续布局；
- block/thread mapping、shared memory 和 register pressure 限制；
- PTX 版本、CUDA Driver 最低要求和 artifact compatibility；
- FP32 基线，以及以后启用 TF32/FP16/Tensor Core 时独立的精度合同；
- 静态 workspace、常量驻留和跨 Kernel lifetime。

RTX 3080、RTX 3070 Ti Laptop 与桌面 RTX 3070 都属于 SM86，可以验证同一指令目标的兼容性；它们的核心数量、显存、带宽和功耗不同。RTX 3080 历史数据和 RTX 3070 Ti Laptop 的单 FFN 数据都不能作为桌面 RTX 3070 的完整模型延迟或峰值显存结论。

## 7. 验收门槛

每个里程碑必须有可重复的证据，不能用“生成了文件”代替编译器验收。

### M0/M1：当前门槛

- 同一输入重复生成字节一致的 MLIR；
- `mlir-opt` 能解析并验证所有已声明支持的示例；
- MLIR 中只出现阶段允许的已注册方言；
- static shape、dtype、MatMul 维度和广播不合法时在编译阶段失败；
- FFN 与 `AddReLU` 的结构测试覆盖普通路径和拒绝路径；
- 不把 textual MLIR 验证结果描述为 GPU 执行或性能提升。

### M2a：当前 FFN runtime-call 切片门槛

- 只接受完整且严格的 BEVFormer FFN pattern，其他图在编译阶段明确失败；
- runtime-call MLIR manifest 只使用当前允许的 `func`/`memref`，并通过实际 `mlir-opt --verify-each`；
- Python call records 与四调用 ABI 一致性在 Session 建立时复核，Runtime 不解析或执行 MLIR source；
- 两次 MatMul 明确调用 CuPy/cuBLAS，两个融合 Kernel 明确由 NVRTC 编译，不允许隐式 fallback；
- 真机报告分别记录 setup、首次调用、device-resident CUDA Event、公共 `run` GPU/wall 和正确性，不外推完整模型或其他 GPU。

### M2b：当前生产形状 FFN MLIR→PTX 切片门槛

- 只接受静态连续 FP32 `[1,2500,256] → 512 → 256` 的完整 FFN pattern，其他 shape 或多余节点在编译阶段失败；
- GPU module 必须使用已注册方言，并由实际 LLVM 24 `gpu-lower-to-nvvm-pipeline` 生成唯一 `gpu.binary/#gpu.object`；
- PTX 必须恰好为 `sm_86`、PTX 7.1、64 位地址和三个预期 entry，artifact 保存工具链 revision、命令、source hash 与完整性摘要；
- Runtime 在 CUDA 初始化前验证图、GPU MLIR、binary MLIR/PTX 一致性和 launch ABI，不允许 cuBLAS、NVRTC、CPU 或旧 CUDA fallback；
- RTX 3080 真机正确性和配对交替基准已保存；结果不得外推为完整 BEVFormer 或 RTX 3070 性能。

### M2c：cuBLAS+MLIR PTX 生产候选门槛

- 只接受与 M2b 相同的固定生产 FFN，并固定为 cuBLAS、PTX、cuBLAS、PTX 四调用顺序；
- 两个融合 `gpu.func` 必须经实际官方 pipeline 生成 `sm_86`/PTX 7.1，Runtime 在 CUDA 初始化前复核 entry、指针 ABI、block、call record 和整体摘要；
- 无权重 artifact 目录采用原子发布，加载时拒绝额外/缺失成员、非常规文件、符号链接、重复 JSON key、大小或摘要不匹配；
- 运行路径不得调用 NVRTC，也不得回退；MatMul 必须明确记录为 CuPy/cuBLAS，不能描述为 cuBLASLt；
- 已在 RTX 3070 Ti Laptop 上与 M2a 同进程共驻留、交换次序，用 CUDA Event 测完整四调用，并另报公共 `run` 的传输/wall 边界；结果只能归属单 FFN 和该设备，不能外推完整 BEVFormer 或桌面 RTX 3070。

### M2d：调度实验与 autotune 门槛

- tiled Kernel 必须在最终 PTX 中保留 shared load/store 和 barrier；是否向量化、是否使用 Tensor Core 都由 PTX 指令证据决定；
- FP16/TF32 MMA 固定尺寸功能候选必须加载真实输入，并保存 `mma.sync`、global load/store、真机正确性和 CUDA Event 证据；要升级为生产候选，还必须覆盖真实 FC1/FC2 shape、边界处理、占用率和完整 FFN 性能；
- 异步双 stage MatMul 功能候选必须在 PTX 中保留预期的 `cp.async`/commit group、wait 深度与两阶段 shared-memory 消费顺序，并保存真实输入正确性和 CUDA Event 证据；升级为生产候选仍需接入真实 FC1/FC2 与完整 FFN；
- autotune 候选先过正确性门才允许计时，缓存绑定图、源码、设备、候选空间和测量配置；
- 单缓冲 tiled 完整 FFN 必须与 M2a/M2c 使用同一正确性门和计时协议；正式性能结论需绑定固定 Git commit，预验收数字不得冒充正式报告；
- 固定尺寸 MMA 与双缓冲 MatMul 没有完整 FFN runner，不能填入 FFN 延时；双缓冲 tiled FFN 集成、cuBLASLt 和生产 Tensor Core 路径完成前继续标为未实现。

### M3：通用 Bufferization 与 GPU/NVVM 门槛

- tensor-to-memref 后无未预期 tensor op，alias、in-place 决策和 buffer lifetime 可检查；
- linalg tiling、fusion、vectorization 与 GPU mapping 不依赖固定 FFN 模板；
- GPU/NVGPU lowering 后不残留非法高层方言，LLVM NVPTX 可生成目标 PTX；
- artifact 明确记录 `cuda-sm86`、toolchain revision、ABI 和 hash；
- 性能使用 warm-up、GPU event、wall time 和峰值显存的独立记录，不外推其他 GPU 数据。

### M4：BEVFormer/HXP v2 门槛

- `bev.ms_deformable_attention` 和 `bev.grid_sample_2d` 有 verifier、shape inference、参考实现与边界用例；
- cold/warm raw outputs 及 `next_bev` 与官方 PyTorch/MMCV golden 对齐；
- 混合 plan 明确 MLIR artifact、cuBLAS/cuDNN 和已有 Kernel 的 stream 依赖与 tensor ABI；
- 执行失败不提交错误的下一帧状态；
- 完整图在 RTX 3070 上报告真实延迟、显存和精度，才能宣称 3070 适配完成。

## 8. 路线图

| 阶段 | 主要工作 | 产物 |
| --- | --- | --- |
| M0/M1（emitter 与验证接口已实现） | textual MLIR、`mlir-opt` 调用封装、手工 tiny FFN/MLP JSON 与 `AddReLU` smoke | `.mlir`、诊断，以及实际 verifier 成功后生成的验证报告 |
| M2a（严格 FFN runtime-call 切片已实现） | pattern matcher、四调用 ABI manifest/call records、CuPy/cuBLAS、NVRTC 融合 Kernel、SM86 Session | 可审阅/可验证的 MLIR ABI manifest、内存中 call plan、FFN 真机报告 |
| M2b（生产形状 FFN MLIR→PTX 已实现） | 三个 `gpu.func`、官方 GPU→NVVM→LLVM NVPTX pipeline、离线 artifact 校验、PTX Session | GPU MLIR、binary MLIR、SM86 PTX、SHA-256、RTX 3080 正确性与性能原始报告 |
| M2c（cuBLAS+MLIR PTX 生产候选已实测） | 两次 CuPy/cuBLAS、两个 MLIR PTX 融合 Kernel、无权重 artifact、严格 Session | 四调用 program/目录、GPU MLIR、binary MLIR、PTX；RTX 3070 Ti Laptop 双顺序配对与 autotune 报告 |
| M2d（调度/选择基础设施已实现） | 单缓冲 tiled 三 dispatch 完整 FFN、真实输入 FP16/TF32 MMA、真实 `cp.async` 双缓冲 MatMul、正确性优先 autotuner、三个真实 FFN Session adapter | tiled 无权重 artifact 与三候选预验收；MMA/异步功能报告；固定 commit 正式重跑、双缓冲生产 FFN 和生产 Tensor Core 待完成 |
| M3 | 通用 One-Shot Bufferize、memref lifetime、linalg tiling/fusion/vectorization、GPU/NVGPU/NVVM | 不依赖固定 FFN 模板的 bufferized MLIR、PTX/cubin 与内存报告 |
| M4 | `bev` 方言、MSDA/GridSample lowering、库调用选择 | BEVFormer 编译区域与混合 artifact |
| M5 | HXP v2 Host Execution Plan、多 artifact 校验、Runtime 混合调度 | 可部署的完整模型包 |

路线图不是完成声明。M2b 只证明固定生产形状 FFN 的 GPU Dialect 到 PTX 闭环；M2c 只证明固定 cuBLAS+PTX 四调用计划；M2d 的 MMA 和双缓冲 MatMul 仍是固定尺寸功能候选，单缓冲 tiled FFN 也未证明通用 lowering。它们不能替代仍未完成的通用 tensor/linalg lowering、自定义 `bev` 方言和 HXP v2。

## 9. 面试表述参考

可以这样简要介绍：

> 我把现有编译器设计成渐进式 MLIR 架构。M0/M1 能把 JSON/Canonical Graph 中的 tiny FFN smoke 生成标准 tensor/linalg IR，并接入本地 LLVM 24 verifier。M2a 针对严格 FFN pattern，用两次 cuBLAS MatMul 与两个 NVRTC 融合 Kernel 执行；M2b 对生产尺寸 `[1,2500,256] → 512 → 256` 生成三个 `gpu.func`，由官方 pipeline 降到 NVVM 和 PTX 7.1，Runtime 校验后加载。RTX 3080 历史结果显示全 PTX 为 `1.092～1.102 ms`，cuBLAS hybrid 为 `0.209～0.214 ms`，说明换成 MLIR 并不会自动提速，关键是调度与库调用。M2c 因此保留两次 cuBLAS，把 Bias+ReLU 和 Bias+Residual+LayerNorm 改为 MLIR PTX，形成无 NVRTC 的四调用生产候选；RTX 3070 Ti Laptop 两轮实测为 `222.131 / 222.694 μs`，比同轮 M2a 快 `1.02414× / 1.02677×`，autotuner 都选择 M2c。M2d 已把单缓冲 shared-memory tile 接成完整三 dispatch FFN，并与 M2a/M2c 做了三候选预验收；真实输入 FP16/TF32 `mma.sync` 与 `cp.async` 双缓冲 MatMul 也已完成固定尺寸正确性和微基准。生产 Tensor Core FFN、双缓冲生产 FFN、cuBLASLt 和固定 commit 的 tiled 正式重跑仍未完成。所有数字只覆盖一个 FFN 编译单元或固定尺寸微内核及对应设备，不是完整 BEVFormer；HXP v2 最终再把 MLIR artifact、vendor 调用和现有 BEV Kernel 编成显式混合计划。

被追问“是否已经用了 MLIR”时，准确回答是：已经有标准 tensor/linalg emitter，也已经有固定生产形状 FFN 的真实 GPU Dialect→NVVM→PTX 编译与执行闭环；新四调用路径的两个融合 Kernel 同样来自 MLIR PTX，而 MatMul 显式交给 cuBLAS。当前没有完成任意 tensor/linalg 图的通用自动 lowering、checkpoint 到 MLIR 的 importer、HXP v2、生产 Tensor Core Kernel 或完整 BEVFormer 网络的 MLIR 执行。

相关背景可继续阅读[总架构说明](architecture.md)、[BEVFormer 编译器设计](bevformer_compiler_design.md)和[SM86 真机验收记录](bevformer_sm86_validation_2026-10-10.md)。
