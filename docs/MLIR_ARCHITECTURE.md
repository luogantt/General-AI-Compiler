# GAC MLIR 编译器架构

本文说明 GAC 如何从现有 Python Canonical Graph 逐步演进为基于 MLIR 的 AI 编译器。首个设备目标是 NVIDIA RTX 3070，对应 `cuda-sm86`。当前已经有标准 tensor/linalg IR emitter、严格限定 BEVFormer FFN pattern 的 `mlir-cuda-sm86` runtime-call 混合切片，以及同一生产形状 FFN 的 `mlir-ptx-sm86` 全 PTX 切片。

本文严格区分已经落地的能力和后续路线图。M0/M1 包含 emitter、诊断和外部 verifier 接口；M2a 把两次 MatMul 显式绑定到 CuPy/cuBLAS，并把 `Bias+ReLU` 与 `Bias+Residual+LayerNorm` 绑定到 NVRTC Kernel；M2b 从严格匹配的 Canonical Graph 直接物化三个 MLIR `gpu.func`，再由官方 LLVM 24 `mlir-opt` 完成 GPU Dialect→NVVM→LLVM NVPTX→PTX 7.1。M2b 是这个固定 FFN 编译单元的真实 MLIR 到 PTX 路径，不是任意 tensor/linalg 图的通用自动 lowering，也不是完整 BEVFormer 网络后端。某次 textual MLIR 产物只有在对应 `mlir-opt` 调用成功后才能称为 verified；某次性能结论必须来自保存了设备、范围、协议和原始样本的真机报告。

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
                └── mlir-ptx-sm86                   当前 M2b
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

通用 tensor/linalg → bufferization → GPU/NVGPU → NVVM → PTX
                                                    路线图 M3
HXP v2 混合 Execution Plan                          路线图 M4/M5
```

四个名称描述不同维度：

- `cuda-sm86` 是设备 target，约束 NVIDIA Ampere Compute Capability 8.6 的指令集、地址空间、资源上限和产物兼容性；
- `mlir-cuda-sm86` 是窄范围的 compiler backend/runtime 组合，只接受一个固定 FFN pattern，并通过显式 runtime call 使用 cuBLAS 和 NVRTC；
- `mlir-ptx-sm86` 接受同一个固定生产形状 FFN，生成三个 `gpu.func`，由官方 MLIR GPU lowering 和 LLVM NVPTX 后端生成 textual PTX，Runtime 加载该 PTX，并由 CUDA Driver JIT 为 SASS；
- 通用 `mlir-nvvm` 仍是目标 compiler backend，要求任意已支持 tensor/linalg 区域经过通用 bufferization、tiling、GPU/NVGPU 和 NVVM lowering。M2b 的 pattern-specialized 路径没有证明这项通用能力。

这种拆分允许以后让同一个设备 target 使用不同编译后端，也允许同一个 MLIR 前端接 CPU、其他 GPU 或 NPU 后端。

## 2. 当前实现状态

| 里程碑 | 当前能力 | 可验证证据 | 明确未包含 |
| --- | --- | --- | --- |
| M0：MLIR 接入 | 从 GAC 静态 Tensor 图生成确定性的 textual MLIR；模块使用已注册的 `builtin`、`func`、`tensor`、`linalg`、`arith` 和 `math` 方言 | 单元测试覆盖文本生成、诊断和工具调用封装；产物可交给本地 LLVM 源码构建出的 `mlir-opt` 解析和验证，成功结果须由实际命令记录 | Python MLIR binding、二进制 bytecode、GPU 代码生成 |
| M1：首个算子切片 | 手工 JSON 图中的 BEVFormer-like tiny FFN/MLP smoke；覆盖 MatMul、Bias Add、GELU/ReLU、`AddReLU` 和末轴 LayerNorm，包含静态 shape 和广播约束 | 单元测试检查确定性、类型/shape 拒绝路径和 MLIR 文本；可选集成测试在配置实际 `mlir-opt` 后运行 | 官方 checkpoint importer、真实 BEVFormer 子图导出、完整 Encoder/Decoder、MSDA、GridSample、Conv 等全图算子覆盖 |
| M2a：SM86 FFN runtime-call 切片 | 严格匹配 `MatMul → AddReLU → MatMul → Add → Add(query residual) → LayerNorm`；发出两次 CuPy/cuBLAS MatMul 和两个 NVRTC 融合 Kernel 的显式 call plan | runtime-call MLIR ABI manifest 已由 LLVM 24 `mlir-opt --verify-each` 验证；独立的 Python call records 已在 RTX 3080 SM86 上完成输出正确性测试 | GPU/NVGPU/NVVM/PTX、HXP v1 打包、任意 MLP、完整 BEVFormer |
| M2b：生产形状 FFN MLIR→PTX 切片 | 只接受 `[1,2500,256] → 512 → 256` 的 FP32 FFN；从严格 pattern 生成三个 `gpu.func`，由官方 MLIR GPU→NVVM pipeline 与 LLVM NVPTX 生成 textual PTX 7.1 | 保存实际 GPU MLIR、含唯一 `gpu.binary` 的 lowered MLIR、PTX、SHA-256 和工具链 revision；RTX 3080 上最大绝对误差 `7.153e-7` | 通用 tensor/linalg→GPU 自动 lowering、Tensor Core 调度、HXP v1 打包、任意 MLP、完整 BEVFormer |

M0/M1 的价值不是“把 CUDA 字符串换成另一种字符串”。实际执行 `mlir-opt` 时，它会解析 IR 和方言，并检查 SSA 使用、类型、区域、block 参数和算子约束；只有该命令成功，特定产物才算通过 verifier。通用 tensor/linalg 产物仍停在 tensor-level MLIR。M2a 的执行来自显式 CuPy/cuBLAS 与 NVRTC runtime binding。M2b 则真正生成并加载 PTX，但编译器先对固定 FFN 做 pattern-specialized GPU IR 物化；它不等于任意 tensor/linalg 图已经具备通用自动 lowering。

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

MLIR→PTX 本身不会自动提速。离线 textual PTX 可以移除部署机上的 MLIR 编译依赖，并避免 CUDA C/NVRTC 编译；CUDA Driver 仍会在加载 PTX 时 JIT 为 SASS，因此模块加载时间必须与稳态 Kernel 延时分开记录。它不会自动减少稳态 Kernel 的计算量。当前两个 MatMul 都是“一线程计算一个输出，并串行遍历 K”的正确性调度，没有 shared-memory tiling、向量化、Tensor Core、流水化或 autotuning，因此比 cuBLAS hybrid 慢 5.150～5.265 倍。下一步性能工作是实现 shared-memory tiled MatMul，加入向量化和 NVGPU/MMA Tensor Core 路径，继续融合 epilogue/LayerNorm，并以 autotuning 选择 tile、block 和精度；生产模式也应保留 cuBLAS/cuBLASLt library-call lowering 作为成熟的 MatMul 候选。

### 3.5 Pass 管线

Pass 管线按“每一阶段都有合法 IR 集合”的方式设计：

1. **Import cleanup**：canonicalize、CSE、常量规范化和死代码清理；
2. **Domain decomposition**：把组合算子分解或融合为后端支持的模式；
3. **Bufferization**：One-Shot Bufferize，把 Tensor SSA 值转换为显式 memref，并建立 alias/lifetime；
4. **Loop/GPU lowering**：将 `linalg` 降为并行循环，完成 tiling、fusion、vectorization 和 GPU mapping；
5. **NVVM lowering**：把 GPU/NVGPU 与向量操作降到 NVVM/LLVM 方言；
6. **Translation**：导出 LLVM IR，经过 NVPTX 目标生成 PTX，再由 CUDA 工具链生成可加载产物。

通用 tensor/linalg lowering 当前只落地第 1 阶段及 M1 所需的 tensor-level 表达。M2a FFN 切片通过严格 pattern matcher 直接物化 runtime-call ABI。M2b 也使用严格 matcher，跳过尚未实现的通用 One-Shot Bufferize 与 linalg tiling，直接生成显式 GPU Dialect；此后第 5、6 阶段由官方 `gpu-lower-to-nvvm-pipeline` 和 LLVM NVPTX 后端真实完成。因而 M2b 证明了受控 FFN GPU module 到 PTX 的编译、加载和执行闭环，尚未证明第 2 至第 4 阶段对任意 tensor/linalg 图通用。每个通用阶段仍应在边界运行 verifier，并检查上一层不允许残留的方言，避免把 lowering 缺口推迟到 Runtime 才暴露。

### 3.6 BEV 专用方言

完整 BEVFormer 有两个不能简单伪装成普通逐元素算子的核心操作：

- `bev.ms_deformable_attention`：表达多尺度特征、sampling location、attention weight 和 level metadata 的联合语义；
- `bev.grid_sample_2d`：表达坐标归一化、插值、padding 和边界行为。

它们计划进入独立 `bev` 方言，并定义 operation verifier、shape inference、canonicalization 和 lowering interface。早期可降低到 `linalg`/`scf` 的正确性实现；性能路径再匹配为专用 CUDA Kernel。方言本身只表达语义，不能把某个 CUDA 实现细节写进高层 op。

上述自定义方言尚未实现。当前 M1 不会用未注册 operation 假装完成完整 BEVFormer，也不会用 `allow-unregistered-dialect` 绕过验收。

### 3.7 Runtime 与包格式

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

HXP v2 混合执行尚未实现。现有 HXP v1 仍是完整 CUDA 路径的有效包格式和校验边界，但它只描述旧 CUDA backend 的源码、launch metadata 和静态内存计划，不能序列化 M2a 的 runtime-call plan 或 M2b 的 PTX artifact。M2a 因此由 JSON graph 直接编译并在 `MlirCudaSession` 中运行；M2b 可离线编译并由 `MlirPtxSession` 加载经过 hash 与 ABI 校验的 PTX，但还没有被封装进 HXP。

## 4. 为什么保留 cuBLAS、cuDNN、Runtime 与 HXP v1

引入 MLIR 不等于重写所有已验证组件。

- **cuBLAS/cuDNN**：矩阵乘和卷积的高性能实现包含架构相关算法选择、workspace 管理和多年调优。M2a 已把两个 MatMul 显式降低为 CuPy/cuBLAS 调用；后续通用路径仍应通过 library-call lowering 或 HXP 调度复用 vendor library。
- **GAC Runtime**：M2a 的 `MlirCudaSession` 负责输入绑定、驻留常量、stream、NVRTC module 和错误传播；M2b 的 `MlirPtxSession` 校验并加载离线 PTX，持有常量和 workspace。MLIR/NVVM 生成的设备程序不负责完整 BEVFormer 跨帧 `prev_bev` 提交/回滚和全图 profile；这些职责继续属于 Runtime。
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

手写 AddReLU smoke 只证明 stock MLIR 24 与 LLVM NVPTX 工具链可用。M2b 进一步证明 GAC 的固定生产形状 FFN 能生成、校验并执行真实 PTX；由于它通过 matcher 直接物化 GPU Dialect，任意 tensor/linalg 图的自动 GPU lowering 仍属于 M3。

## 6. RTX 3070 / SM86 适配原则

RTX 3070 的首个 target 固定为 `cuda-sm86`。MLIR M0/M1 验证设备无关的 tensor IR；M2a 与 M2b Runtime 都要求 Compute Capability 恰好等于 8.6，M2b 产物还固定 `.target sm_86`、PTX 7.1 和 64 位地址。后续通用 GPU lowering 必须把以下内容纳入 target profile：

- Compute Capability 8.6 与 NVPTX feature；
- 支持的 dtype、向量宽度、对齐和连续布局；
- block/thread mapping、shared memory 和 register pressure 限制；
- PTX 版本、CUDA Driver 最低要求和 artifact compatibility；
- FP32 基线，以及以后启用 TF32/FP16/Tensor Core 时独立的精度合同；
- 静态 workspace、常量驻留和跨 Kernel lifetime。

RTX 3080 与 RTX 3070 都属于 SM86，可以验证同一指令目标的兼容性；它们的核心数量、显存和带宽不同，因此 RTX 3080 数据不能作为 RTX 3070 的延迟或峰值显存结论。MLIR 路径目前没有 RTX 3070 性能结果。

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
| M3 | 通用 One-Shot Bufferize、memref lifetime、linalg tiling/fusion/vectorization、GPU/NVGPU/NVVM | 不依赖固定 FFN 模板的 bufferized MLIR、PTX/cubin 与内存报告 |
| M4 | `bev` 方言、MSDA/GridSample lowering、库调用选择 | BEVFormer 编译区域与混合 artifact |
| M5 | HXP v2 Host Execution Plan、多 artifact 校验、Runtime 混合调度 | 可部署的完整模型包 |

路线图不是完成声明。M2b 只证明固定生产形状 FFN 的 GPU Dialect 到 PTX 闭环，不能替代仍未完成的通用 tensor/linalg lowering、自定义 `bev` 方言和 HXP v2。

## 9. 面试表述参考

可以这样简要介绍：

> 我把现有编译器设计成渐进式 MLIR 架构。M0/M1 能把 JSON/Canonical Graph 中的 tiny FFN smoke 生成 `func/tensor/linalg/arith/math` 标准 IR，并接入本地 LLVM 24 `mlir-opt` verifier。M2a 针对一个严格 FFN pattern，生成 runtime-call ABI manifest 和 call records，实际执行使用两次 cuBLAS MatMul 与两个 NVRTC 融合 Kernel。M2b 从 Canonical Graph 的另一分支对生产尺寸 `[1,2500,256] → 512 → 256` 的同一 FFN 生成三个 `gpu.func`，再由官方 MLIR GPU pipeline 降到 NVVM，并由 LLVM NVPTX 生成 textual PTX 7.1；Runtime 校验后加载 PTX，CUDA Driver 再 JIT 为 SASS。设备执行不调用 cuBLAS，也不经过 CUDA C/NVRTC。它在 RTX 3080 上达到 `7.153e-7` 最大绝对误差，但当前全 PTX 为 `1.092～1.102 ms`，cuBLAS hybrid 为 `0.209～0.214 ms`，所以当前慢 `5.150～5.265×`。原因是 MatMul 还是每线程一个输出、串行 K 的正确性调度；下一步是 shared-memory tiling、vectorization、NVGPU/MMA Tensor Core、融合和 autotuning。这个结果只覆盖一个 FFN 编译单元，不是完整 Encoder/Decoder 或完整 BEVFormer。HXP v1 继续承载现有 CUDA 生产路径；最终计划用 HXP v2 把 MLIR artifact、vendor 调用和现有 BEV Kernel 编成显式混合执行计划。

被追问“是否已经用了 MLIR”时，准确回答是：已经有标准 tensor/linalg emitter，也已经有一个固定生产形状 FFN 的真实 GPU Dialect→NVVM→PTX 编译与执行闭环。编译器会调用官方 `mlir-opt`，保存 binary MLIR、PTX、工具链 revision 和摘要，GPU Runtime 加载 textual PTX，CUDA Driver 再 JIT 为 SASS。当前没有完成任意 tensor/linalg 图的通用自动 lowering、checkpoint 到 MLIR 的 importer、HXP v2 或完整 BEVFormer 网络的 MLIR 执行。

相关背景可继续阅读[总架构说明](architecture.md)、[BEVFormer 编译器设计](bevformer_compiler_design.md)和[SM86 真机验收记录](bevformer_sm86_validation_2026-10-10.md)。
