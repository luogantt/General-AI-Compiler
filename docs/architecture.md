# 架构设计、实现状态与参考系统映射

## 1. 范围

GAC 的目标是一个完整的编译—推理闭环：输入模型经过统一图表示、后端能力查询、设备划分和执行计划生成，被封装成 `.hxp`，再由 Runtime 加载执行。当前已实现前端、图优化、CUDA SM 8.6 源码生成、单 GPU Session、版本化 HXP 原型、静态 arena 规划及 CUDA Session 对 arena offset 的消费；MLIR 侧已实现 tensor/linalg emitter 和严格限定一个 BEVFormer FFN pattern 的 SM86 runtime-call 切片。通用异构划分、通用 MLIR GPU/NVVM lowering 与通用 Host Execution Plan 仍是设计目标。

### 1.1 源码核对（2026-10-10）

| 模块 | 当前证据 | 状态 |
| --- | --- | --- |
| JSON 导入与确定性常量 | `src/gac/frontends/json_frontend.py` | 已实现 |
| Graph / Node / TensorSpec、图校验、JSON 序列化 | `src/gac/ir.py` | 已实现 |
| Shape/DType 推导与 NumPy 参考求值 | `src/gac/ops.py` | 已实现基础 Tensor（含 `Greater`/`Less`）、MLP、CNN 与 MSDA 所需静态算子 |
| 常量折叠、DCE、MLP 融合、Conv+BN folding、Add+ReLU 融合 | `src/gac/passes.py` | 已实现；生产 step 安全融合 39 组 Add+ReLU |
| 主机事件、逐 launch Profiler 与 Chrome Trace | `src/gac/profiler.py`、`runtime.py` | 可选 CUDA Event 计时；逐 launch 记录实际 implementation，并同时按 op / implementation 聚合 JSON；默认执行不创建逐 launch Event |
| CUDA SM 8.6 代码生成与静态能力校验 | `src/gac/backends/cuda.py` | 已实现基础 Tensor（含 `Greater`/`Less`）、MHA/CNN 所需算子与 MSDA；generated Conv/MatMul 保留为正确性路径 |
| CUDA Session、设备探测、Vendor Library 与 CLI | `src/gac/runtime.py`、`cli.py` | 已实现单卡执行、FP32 cuDNN Conv 和 FP32 CuPy/cuBLAS MatMul，不含自动回退 |
| MLIR tensor emitter 与 verifier 接口 | `src/gac/backends/mlir.py`、`src/gac/mlir_toolchain.py` | 已实现标准 `func/tensor/linalg/arith/math` textual MLIR；只有实际 `mlir-opt` 成功的产物才标记 verified |
| MLIR SM86 FFN runtime-call 切片 | `src/gac/backends/mlir_cuda.py`、`src/gac/mlir_runtime.py` | 严格匹配固定 FP32 FFN；MLIR source 是可审阅/可验证的 ABI manifest，实际由经校验的 Python call records 调度 2×CuPy/cuBLAS MatMul 与 2×NVRTC 融合 Kernel |
| HXP 包与静态 arena | `src/gac/package.py`、`memory.py`、`runtime.py` | 包内计划已实现并校验；CUDA Session 一次性分配内部 arena；内部 Reshape 和满足连续条件的 Slice 可复用同一存储 view |
| 图组合、MHA 与 BEVFormer 合同 | `src/gac/compose.py`、`transformer.py`、`bevformer_*.py`、`frontends/bevformer_*.py` | Graph Composer、MHA、ResNet50-FPN、feature-prep、learned encoder inputs、完整三层 Encoder、完整六层 Decoder、六层 raw detection projection、cold/history image-to-BEV、运动补偿、cold/Decoder/统一 cold-warm step pipeline、官方权重映射及导出/状态合同已实现；统一 step 已通过官方 checkpoint 的 SM86 真机数值验收 |
| BEVFormer 状态 Runtime | `src/gac/bevformer_runtime.py` | scene/CAN-bus delta、cold/warm flag、状态提交与失败回滚已实现；当前 `prev_bev` 由主机 NumPy 状态持有 |
| BEVFormer 主机后处理 | `src/gac/bevformer_postprocess.py` | 显式 NumPy CPU NMS-free Top-K、box denormalization、score/post-center-range 过滤已实现；尚未降低为 CUDA/HXP 图 |
| 通用能力注册、代价模型、设备划分 | 无对应模块 | 待实现 |
| 通用 Execution Plan | 当前只有 CUDA launch metadata 与静态内存计划 | 待实现 |
| 自动化测试 | `tests/` | 最终版本 RTX 3080 强制 CUDA 结果为 `Ran 287 tests in 18.881s ... OK (skipped=1)` |

历史上已在 RTX 3070 Ti Laptop GPU 完成 MLP 与独立 MSDA 的真实 CUDA 验证，证据见 [融合运行报告](validation/sm86/gpu-fused.json) 与 [BEVFormer SM86 验证目录](validation/sm86/bevformer/)。2026-10-10 又在 Ubuntu 22.04、RTX 3080（同为 SM 8.6）上，使用官方 `bevformer_tiny_epoch_24.pth` 完成统一 cold/warm HXP 的真机数值验收。该结果验证 SM86 兼容性和固定 Profile 的 raw step；桌面 RTX 3070 的显存、吞吐和功耗仍须在该型号上单独测量。

设计区分两类程序：

1. **Device Program**：当前 CUDA 后端生成源码，NVRTC 在 Session 初始化时编译为设备代码；HXP 的 `KernelLaunch` 同时显式保存 `implementation_candidates`，使 cuDNN、CuPy/cuBLAS 与 generated CUDA 都是可检查的执行候选。
2. **Host Execution Plan**：描述分配、复制、调用和输出。当前仅有 CUDA 启动元数据及固定 Session 执行顺序，通用计划仍待实现。

`mlir-cuda-sm86` 是单独的窄切片：编译器同时产生 runtime-call MLIR ABI manifest 和内存中的 Python call records。Runtime 不解析或执行 MLIR source，dispatch 只由 call records 驱动；融合 Kernel 在 Session 建立时由 NVRTC 编译。HXP v1 不承载这组 call records，因此该后端当前只能从 JSON graph 直接运行。

`sim-gpu` 将只验证控制面和异构边界。真实设备后端必须提供独立的编译产物、内存空间、队列和事件语义。

## 2. 端到端数据流

```text
Model JSON
  │
  ├─ inputs / deterministic constants / nodes / outputs
  ▼
Frontend Adapter
  │  schema validation, constant materialization
  ▼
Canonical Graph
  │  shape + dtype propagation, operator validation
  ▼
Backend Registry
  │  support constraints + estimated compute/launch/transfer cost
  ▼
Partitioner
  │  node placement + explicit cross-device boundaries + fallback
  ▼
Execution Plan Builder
  │  ordered host operations + stable provenance identifiers
  ▼
HXP Package
  │  manifest + plan + constants + target metadata
  ▼
Runtime Session
  │  load -> bind input -> execute -> collect output -> trace
  ▼
CUDA SM86 / explicit NumPy reference
```

编译器不把“某后端声称支持节点”直接等同于“应该把节点放到该后端”。划分还要比较启动开销、相邻节点切换产生的传输成本以及回退可用性。这是后续加入更多设备后仍保持不变的结构。

当前 BEVFormer 的已实现统一 cold/warm step 有一条更具体的数据流：

```text
images[f16] + lidar2img
  -> ResNet50 + one-level FPN
  -> feature-prep (layout + camera/level embeddings)
  -> learned BEV query/position + CAN-bus conditioning
prev_bev[f16] + can_bus_delta
  -> rotate previous BEV + shift previous reference points
has_prev_bev
  -> warm: every layer shares [motion-compensated prev_bev, initial query]
  -> cold: every layer builds [current layer query, current layer query]
  -> three-layer Encoder
  -> encoded BEV
  -> next_bev[f16]
  -> learned object-query init + six-layer Decoder
  -> six raw class/box projection branches
  -> fixed-shape raw tensors
  -> explicit CPU NMS-free Top-K + box denormalization
```

其中 cold/history image-to-BEV、rotate/shift、`has_prev_bev` 选择、三层 Encoder、六层 Decoder 与 raw head 已在同一个五输入三输出静态图中贯通，并由 `step-pipeline` 打包成 HXP。cold 不能复用 warm 的固定 temporal queue：它必须在每一层用该层 current query 动态构造两份 value；warm 则在各层共享由 motion-compensated history 和进入 Encoder 前的 initial query 组成的 queue。`BevFormerRuntime` 在 Session 成功并校验三个输出后提交 `next_bev`；失败不修改旧状态。冷启动只清零 `can_bus_delta[:3]` 和 `can_bus_delta[-1]`，四元数、绝对 heading 与其他 CAN-bus 特征保留。

生产 checkpoint HXP 的融合前图有 876 个节点；39 组安全 Add+ReLU 融合后为 837 个执行节点，内部 Reshape/连续 Slice 零拷贝后实际设备 dispatch 为 703。内部 arena 保持 `552,960,000` bytes，Tensor 计划总量为 `705,177,100` bytes；cuDNN 的 8 MiB（`8,388,608` bytes）workspace budget 单独管理，`713,565,708` bytes 是 Tensor + cuDNN workspace 的计划预算/准入值，不是严格设备峰值上界。本轮再加入 FP32 cuDNN Conv 和 FP32 CuPy/cuBLAS MatMul 后，相对官方 PyTorch/MMCV golden，cold 的 `cls_logits` / `bbox_regression` / `next_bev` 最大绝对误差为 `0.003891945` / `0.004856288` / `0.001953125`，warm 为 `0.000953674` / `0.000725389` / `0.001953125`。全生成 cold/warm 中位数分别为 `425.325073 / 426.032532 ms GPU` 与 `436.881591 / 437.718436 ms wall`；库加速 cold/warm 分别为 `57.624464 / 56.823263 ms GPU` 与 `69.403093 / 68.747440 ms wall`，对应 cold `7.38098× / 6.29484×`、warm `7.49750× / 6.36705×` 的 GPU/wall 加速。fresh-process 中 Session 构建完成后的首个 cold `run` 为 `277.443573 ms GPU / 289.668472 ms wall`，Session 构建时间另计；同进程基准的首次 cold 不代表 fresh-process。带逐 launch Event 的 Profile 会扰动执行，不能与普通基准直接比较。最后的 NMS-free Top-K 与 box denormalization 仍是显式 CPU 主机 API；真实 nuScenes 数据集与桌面 RTX 3070 仍待验收。

## 3. 设计分层

### 3.1 Frontend Adapter

Frontend 读取 `inputs`、`constants`、`nodes` 和 `outputs`。常量可以用 `random_normal`、`zeros` 或 `ones` 描述；`random_normal` 的 `seed`、`mean` 和 `std` 使权重生成可重复。

Frontend 的边界是“模型语义”，不包含 CUDA block、GPU shared memory 或 NPU DMA 描述。未来的 ONNX、`torch.export` 和 StableHLO 导入器都应落到相同的 Canonical Graph，而不是让每个硬件后端分别解析框架格式。

### 3.2 Canonical Graph 与公共分析

Canonical Graph 保存张量、节点、属性和来源标识。公共分析负责：

- 检查节点输入是否已定义；
- 推导演示算子的输出 Shape 和 DType；
- 验证广播、矩阵维度和 LayerNorm 轴；
- 保留从模型节点到计划步骤和 Trace 事件的关联。

现有 IR 与推导代码能够表达部分动态维度；首条运行期验收路径仍限定静态 Shape 和少量演示算子。动态 Shape Guard、量化、控制流、别名与副作用分析是后续工作。

### 3.3 Backend Registry 与能力查询

拟新增的后端注册表把目标名称映射到 Python 后端实现。能力查询至少需要回答某个节点是否可执行；代价接口为划分器提供计算和启动估计。真实后端还应逐步补齐：

- 支持的 DType、Shape 范围和动态维度；
- 输入输出 Layout 与对齐约束；
- 临时工作区和峰值内存；
- 编译成本、预计延迟和可靠性等级；
- 精度误差与确定性能力；
- 设备间传输路径和成本。

首版 Python 协议定位为进程内原型接口，不承诺二进制兼容。

### 3.4 Heterogeneous Partitioner

划分器在可用后端之间选择节点位置，并保留 CPU 回退。它的目标函数可概括为：

```text
total_cost = compute + launch + cross_device_transfer + fallback_penalty
```

首版拟使用确定、可解释的启发式估计，方便测试和调试。真实系统应以离线 Benchmark 表和设备 Profile 校准估计，并逐步纳入内存、精度、Deadline、驻留状态和 KV Cache 位置等硬约束。

### 3.5 Execution Plan

Execution Plan 是编译器与 Runtime 之间最重要的契约。它描述主机需要执行的动作，而不是再次携带一份待解释的高层图。跨设备边界必须在计划中可见，这样 Runtime 才能统一处理数据移动、同步、错误和 Trace。

计划应保持以下不变量：

- 每个计算步骤绑定一个后端和原始节点标识；
- 跨设备消费之前存在显式传输语义；
- 常量和输入只在第一次需要时进入目标内存空间；
- 输出的拥有者和生命周期明确；
- 不支持的设备路径能够回退，不能静默跳过计算；
- 执行顺序和 Trace 顺序可以稳定复现。

首版执行器采用同步执行，但计划从第一版就显式记录分配、复制、计算、释放和依赖，避免之后引入异步时重新定义数据所有权。

### 3.6 HXP 模型包

`.hxp` 的首个离线原型已经实现。当前包包含：

- 带格式版本、目标、条目大小和 SHA-256 的 canonical manifest；
- 不内嵌常量字节的 Canonical Graph；
- 每个常量独立保存的无 pickle `.npy` 数据；
- CUDA C 源码与 launch metadata；
- 区分外部 I/O、常驻常量和复用 arena 的静态内存计划。

加载器会拒绝未知主版本、重复/越界/额外条目、摘要不符、不安全路径、object dtype 和不匹配的确定性内存计划；执行前还会从规范图重新生成 CUDA，并核对源码与 launch metadata。当前 HXP 不含 MLIR runtime-call plan、通用 Host Execution Plan、CPU 回退、多 target variant、GPU cubin 或签名认证。后续真实后端可加入 CPU 本地代码、GPU cubin/hsaco、NPU blob 或厂商缓存，并继续校验目标兼容性、Shape variant 和 artifact hash，不能仅凭文件扩展名信任内容。

### 3.7 Runtime Session

Runtime Session 负责加载模型包、创建后端实例、绑定输入、执行计划和收集输出。设备后端负责具体算子或 artifact 的执行；Session 负责跨设备的全局顺序与错误传播。

建议的长期对象模型是：

```text
Runtime
  └─ Device
      └─ MemorySpace
          └─ Buffer / TensorView

Module / Executable
  └─ Session / Context
      └─ Invocation
          └─ Event / Trace
```

首版拟把这些概念映射为 Python 对象和 NumPy 数组，验证计划语义；真正的设备内存隔离和异步执行需要后续驱动实现。

## 4. 与参考项目的对应关系

| GAC 关注点 | 主要参考概念 | 采用的思想 | 没有照搬的部分 |
| --- | --- | --- | --- |
| 分层编译流水线 | IREE Input/Flow/Stream/HAL/VM | 图语义、设备调度、硬件接口、主机控制逐层下降 | 未复制 IREE Dialect、VM 字节码或 Runtime 源码 |
| 轻量调用与模块化产物 | TVM Relax/TIR、PackedFunc、Module | 高层控制与低层 Kernel 分开，模块可包含多种目标产物 | 当前没有采用 TVM FFI，也没有内嵌 TVM Runtime |
| Session、回退和设备提供器 | ONNX Runtime InferenceSession、IOBinding、Execution Provider | 后端按能力声明、CPU fallback、上层 Session 管理 | 能力接口将扩展为成本与约束，不只做支持/不支持 |
| 可编程 Kernel | TileLang target backend / execution backend | 编译后端与 Build/Load/Launch 执行路径正交 | 当前未集成 TileLang DSL 或其生成代码 |
| 稳定模型语义与设备 API | StableHLO、XLA HLO、PJRT | 版本化输入边界，Device/MemorySpace/Buffer/Event 对象 | 当前没有链接 XLA，也没有宣称 PJRT 兼容 |

这种映射是架构研究，不代表代码派生关系。仓库没有复制上述项目的源码；未来若真正链接、vendoring 或分发任何上游组件，必须按对应许可证和第三方 Notice 单独处理。

## 5. 关键设计选择

### 编译后端与 Runtime 驱动分离

编译后端回答“能否执行、怎样降低、生成什么 artifact”；Runtime 驱动回答“有哪些设备、怎样分配/复制、怎样加载/提交、事件何时完成”。一个编译目标可能对应多个执行环境，两者不应被同一个巨型插件接口绑死。

### 稳定边界使用序列化数据与 C ABI

Python 对象适合 v0.1 原型，但不能作为跨编译器版本的插件 ABI。未来二进制插件边界不传递 MLIR C++ 对象、Python 对象、STL 容器或异常；IR、计划和诊断使用版本化字节串，生命周期通过不透明句柄管理。

### Trace 是一等产物

模型节点、规范化节点、分区、计划步骤和运行时事件应共享稳定来源标识。这样性能回归能从设备事件回溯到编译决策和原始模型节点，而不是依赖易变的日志字符串。

## 6. 当前限制

当前 CUDA 后端与 Session 尚未实现通用执行计划、多设备划分和回退。HXP 与静态 arena 计划已实现；Session 会调用同一规划器、一次性分配内部 arena，并按 allocation offset 建立中间值 view。内部 Reshape 和满足连续区间约束的 Slice 已能共享源 Tensor 存储，其他布局变换仍会物化。IR 可以表示部分动态维度，但 CUDA 后端只接受静态 Shape。FP32 Conv/MatMul 已分别接入 cuDNN 和 CuPy/cuBLAS；`KernelLaunch` 明确记录 `implementation_candidates`，Runtime 的最终选择写入 `session.info`、`kernel_info` 和 profile，因此 vendor 调用不是隐藏替换。MLIR 的通用路径仍停在 tensor/linalg；可执行 `mlir-cuda-sm86` 只支持当前固定 FFN pattern，严格要求 Compute Capability 8.6、CuPy、cuBLAS 与 NVRTC，且 HXP v1 不支持该计划。Tensor Core/混合精度主计算、多流调度、DLPack、AOT CPU 机器码、NPU 和量化仍未实现。BEVFormer 的 feature/input preparation、三层 Encoder、六层 Decoder、六层 raw detection projection、cold/history image-to-BEV、历史 BEV rotate/shift、统一 cold/warm step 及 Runtime 状态提交/失败回滚已经实现；公开 image/state ABI 使用 FP16，内部主要计算仍为 FP32。官方 checkpoint 的 cold/warm raw step 已在 RTX 3080 上完成物理 SM86 数值与首轮性能验收。NMS-free Top-K/box denormalization 以显式 CPU API 实现；真实 nuScenes 指标及桌面 RTX 3070 的设备专属显存/性能数据仍未完成。

## 7. 从参考源码落实到接口

以下路径均对应本地已展开的参考树；ZIP 作为归档，不重复解压或混入 GAC。参考快照没有统一的固定提交标识，因此本文描述的是所检查文件中的设计，不将其等同于某个发布版本。

| 参考入口 | 核对到的机制 | GAC 的设计决定 |
| --- | --- | --- |
| [IREE 编译器分层](../../iree-main/compiler/src/iree/compiler/README.md)、[HAL device API](../../iree-main/runtime/src/iree/hal/device.h) | Flow/Stream/HAL/VM 分工；队列分配、复制、执行与释放接口 | 将计算程序与主机调度分开，Buffer 生命周期显式进入计划 |
| [TVM Relax VM](../../tvm-main/docs/arch/relax_vm.rst)、[Runtime](../../tvm-main/docs/arch/runtime.rst) | 高层函数编为控制字节码，低层函数编为目标 Kernel，再链接进可执行模块 | HXP 同时携带 Host Plan、常量和目标 artifact；Kernel 生成器可替换 |
| [ONNX Runtime Execution Provider](../../onnxruntime-main/include/onnxruntime/core/framework/execution_provider.h) | `GetCapability` 查询节点/子图能力，`Compile` 接收融合子图 | 支持查询应返回候选子图、限制与拒绝原因，不能只有算子名称集合 |
| [TileLang JITKernel](../../tilelang-main/tilelang/jit/kernel.py)、[驱动延迟加载](../../tilelang-main/docs/runtime_internals/stubs.md) | `target` 与 `execution_backend` 为不同配置；执行依赖可以延迟加载 | TileLang 作为可选 Kernel provider；编译请求与驱动初始化分离 |
| [PJRT C API](../../xla-main/xla/pjrt/c/pjrt_c_api.h) | Device/Memory/Buffer/Event/Executable 对象，带大小和版本的 C 接口结构 | 借鉴资源所有权和接口演进方式；首版 Python 协议不声称兼容 PJRT |

外部一手资料交叉核对：[IREE Dialects](https://iree.dev/reference/mlir-dialects/index.html)、[ONNX Runtime EP](https://onnxruntime.ai/docs/execution-providers/)、[PJRT 对象模型](https://openxla.org/xla/pjrt/cpp_api_overview)。上述映射是本项目的设计选择，不代表将五套完整编译器同时作为依赖引入。

## 8. 首版数据契约

以下结构同时包含已落地的 HXP/静态内存子集和仍待实现的通用异构接口。第一条可验收路径限定为静态 Shape、连续布局、FP32、纯函数图、同步执行；现有参考算子支持更宽的类型，不代表后端已经承诺这些类型。

### 8.1 编译接口与驱动接口

| 对象 | 最小字段或操作 | 必须明确的边界 |
| --- | --- | --- |
| `TargetProfile` | target_id、profile_version、dtype/layout/shape 约束、memory_spaces | 编译可读取静态配置，无须打开设备 |
| `ExecutionCandidate` | node_ids、target_id、输入输出约束、workspace_bytes、implementation_id | 一个候选可覆盖子图；被拒绝时保留原因 |
| `CostEstimate` | compute_us、launch_us、transfer_us、source | 标注 synthetic/measured；估计值与实测 Trace 分开 |
| `CompiledArtifact` | artifact_id、format、entry_points、target、digest、requirements | 描述数据和调用约定，不保存 Python 对象或驱动指针 |
| `CompilerBackend` | query_candidates、estimate、compile | 输出 artifact 与 Tensor ABI；不分配运行期设备内存 |
| `RuntimeDriver` | enumerate、allocate、copy、load、dispatch、release | 检查运行设备是否满足 artifact requirements；后续扩展 Event |

首版 CPU 与模拟设备可以复用 `evaluate_node`，但仍须经过能力校验和 artifact 入口；不能让 Session 根据算子名自行选择实现。未知目标和未支持组合必须产生可定位诊断。

### 8.2 Execution Plan 与校验器

计划包含 `schema_version`、目标槽位、Buffer 表、artifact 引用、步骤表、输入输出绑定和来源映射。图中的 value id 与物理 buffer id 分离：同一个逻辑值可以在多个内存空间拥有副本。

最小步骤为 `alloc`、`copy`、`dispatch`、`free`、`return`；每个步骤有稳定的 `step_id` 和前驱依赖。同步执行器按拓扑序执行，真实异步驱动再把依赖映射为事件。若使用零拷贝输入绑定，必须单独声明外部所有权。

校验器在写包前、加载后分别检查：

1. 步骤依赖无环，目标、Buffer、artifact 和入口引用存在。
2. 每次读都有已完成的写；分配本身不等于数据已经初始化。
3. `dispatch` 的输入处于要求的内存空间、布局、Shape 和 DType；跨空间读必须有复制或显式共享协议。
4. Buffer 容量满足张量大小及对齐；释放后不可访问，外部输入不得由 Session 释放。
5. `free` 依赖所有读写完成，包括异步复制；输出所有权转交调用者前不得释放。
6. 每个 dispatch 可回溯到 candidate、partition 和原始模型节点。

示例模型的一种**候选**方案为 GPU 执行融合层、CPU 执行 LayerNorm；这不意味着该方案一定比全 CPU 或全 GPU 快：

```text
Session 初始化：W、bias -> GPU 常量缓冲区（驻留）
每次 Invocation：
  alloc GPU input / activation
  copy host input -> GPU input
  dispatch GPU MatMulBiasGELU -> GPU activation
  alloc host activation
  copy GPU activation -> host activation
  dispatch CPU LayerNorm -> host output
  free 中间缓冲区（最后一次使用完成后）
  return host output（转移所有权）
```

模拟设备应实际复制 NumPy 数组并记录逻辑所有者，避免主机共享引用掩盖缺失的 copy；其耗时仍是主机执行耗时。

### 8.3 融合与划分的关系

公共 `default_pass_manager()` 会直接生成 `MatMulBiasGELU`。当前 CUDA 编译入口先验证原始算子，再检查优化结果；当融合后的 bias 形状不满足后端约束时，改用未融合图，并记录拒绝原因。也支持 `--no-fuse`。通用的融合/划分联合候选搜索仍未实现。

落地时保留语义规范化后的图，枚举融合与未融合候选，再按能力和总成本选取；首版可先提供关闭融合的选项及可验证的分解规则。`source_nodes` 只记录来源，不能替代原始节点属性和数值语义。动态维度、精度策略或多消费者不满足规则时不得强行融合。

### 8.4 代价模型与回退

同步首版比较：

```text
cost = compute + launch + actual_copies + layout_conversion
transfer_us = fixed_latency_us + bytes / bandwidth_bytes_per_us
```

按实际计划中每次复制收费：已驻留常量和同一设备可复用的副本不能按每条消费边重复计费。分别报告 Session 首次加载成本、冷启动和稳态单次执行成本，不能把三种时间混在一起。

实现可从全 CPU 可行解出发，对连续子图做有界候选搜索，包含分支边界、图输入上传与输出回传成本；每次接受变更都重新校验整体计划和峰值存活内存。它是启发式，不承诺全局最优。没有收益的方案应保留全 CPU；覆盖跨设备路径的测试使用显式放置约束，不伪造加速收益。

引入异步后，目标应转为依赖图和队列约束下的完成时间，简单求和不能表示计算与传输重叠。精度和内存上限先作为合法性条件，不能仅通过低惩罚权重允许违反。

回退分三种处理：编译时不支持则选择 CPU 候选；加载时设备不可用则选择包内预编译的 CPU 计划；执行中失败则默认返回错误。只有纯函数 Invocation、输入仍可用且无已提交外部副作用时，才允许显式配置的整次 CPU 重放，并在 Trace 标记失败与重放。不得把失败 dispatch 静默改名为 CPU 执行。

### 8.5 HXP 与 Session

当前原型使用 ZIP 容器便于检查，结构如下；原始愿景中的 FlatBuffers/字节码可以后续引入：

```text
model.hxp
  manifest.json          格式版本、I/O、目标、条目摘要、launch metadata、静态内存计划
  graph.json             不含常量字节的 Canonical Graph
  program.cu             已生成的 CUDA C 源码
  constants/0000.npy     仅数值数据，以索引关联逻辑名称
```

加载器已经拒绝未知主版本、重复或越界条目、摘要不符和目标/内存计划不匹配的包；NumPy 加载禁止 pickle/object dtype。摘要用于完整性检查，不等于来源认证。当前执行入口会重新生成 CUDA 并与包内源码及 launch metadata 核对；尚未保存 GPU cubin，也没有 `plans/cpu.json` 或跨设备步骤表。

当前 CLI 已能校验 `.hxp`、重建 CUDA Program，并交给既有 `CudaSession` 或 NumPy 参考路径执行。`CudaSession` 已按生命周期规划一次性分配内部 arena，并按 allocation offset 建立中间值 view。长期 Session 仍需解析通用执行计划、选择 target artifact、维持只读权重驻留和创建独立 Invocation；Buffer 复用继续以显式存活区间为依据，不能依靠 Python 垃圾回收推断设备完成状态。

## 9. 代码落点与验收边界

保留现有 `ir.py`、`ops.py`、`passes.py`、`profiler.py` 和 JSON Frontend。按以下依赖顺序增量实现，避免为了目录齐全而先创建空壳：

| 模块或计划模块 | 职责 | 当前状态/首个可验收结果 |
| --- | --- | --- |
| `backends/` | 协议、注册表、CPU 与模拟驱动 | 支持/拒绝原因明确，输出与参考语义一致 |
| `partition.py` | 候选、代价、放置报告 | 高搬运成本时保留 CPU，未知组合拒绝 |
| `plan.py` | Buffer/步骤定义、生成与校验 | 分支图、跨设备图均不读未初始化或已释放数据 |
| `compiler.py` | 编排公共 Pass、候选选择和产物构建 | 保留原图语义，支持关闭融合排查问题 |
| `package.py` | 版本化读写、条目校验 | 已实现；包往返及版本、摘要、路径、元数据错误拒绝 |
| `memory.py` | 存活期、存储类别、静态 arena 规划 | 已实现；Tensor 计划为 `705,177,100` bytes，cuDNN 8 MiB 单列后计划预算/准入值为 `713,565,708` bytes |
| `bevformer_*.py` | 固定 Profile 的 Encoder/Decoder/raw head、motion、cold/warm step、状态 Runtime 与显式 CPU 后处理 | 五输入三输出统一 step 与 Runtime 提交/回滚已实现；官方 checkpoint 的 cold/warm raw step 已通过 RTX 3080 SM86 数值验收 |
| `runtime.py` | Session、Invocation、执行计划、Trace | 重复执行不污染常量或共享临时 Buffer |
| `cli.py`、`tests/` | compile/inspect/run、契约和端到端测试 | README 三条命令在新进程中贯通 |

对应实施批次见 [路线图](roadmap.md)。目前优先落地了一条直接 CUDA C/NVRTC 路径，具体实现如下；上表中的通用模块化方案仍是后续目标。

## 10. RTX 3070 系列的首个实现

目标为 `cuda-sm86`，运行时校验设备计算能力恰好为 8.6。RTX 3070 的计算能力见 [NVIDIA 官方表](https://developer.nvidia.com/cuda/gpus)；历史切片验证设备是 RTX 3070 Ti Laptop GPU，本轮官方 checkpoint 全图验证设备是 RTX 3080，两者与 RTX 3070 都是 SM 8.6。相同计算能力不代表相同吞吐、显存带宽或功耗，不能把这些测试数字直接视为桌面 RTX 3070 性能。

编译入口 `compile_cuda(graph)` 不导入 CuPy、不访问设备。它检查静态 Shape、受支持 DType、连续布局及算子属性，执行公共 Pass，输出 `CudaProgram`：优化图、原始参考图、CUDA C，以及每个 `KernelLaunch` 的输入输出、grid、block、来源标识与 `implementation_candidates`。统一 step 的公开 image/state 边界允许 FP16，`has_prev_bev` 为 int32；其余核心计算仍以 FP32 为基线。`.cu` 和 `.plan.json` 是调试中间产物；CLI 也可把规范图、权重、源码、launch metadata 与静态内存计划写成可校验的 `.hxp`。

当前 Kernel 策略：

| 算子 | 映射策略 |
| --- | --- |
| 二维/批量 MatMul、MatMulBiasGELU | generated 路径使用 16×16 shared-memory tile；受支持的生产 FP32 MatMul 可派发到 CuPy/cuBLAS，融合 epilogue 仍由 generated 路径处理 |
| Add | 静态广播索引生成；每线程一个输出元素 |
| ReLU / GELU / Identity | 每线程一个元素；GELU 为 tanh 近似 |
| LayerNorm | 每行 256 线程；先均值、再中心化方差两遍 reduction；支持最后一维 scale/bias |
| Softmax | 每行 256 线程；先最大值、再指数和 reduction |
| Reshape / Slice / Transpose / Concat | 内部 Reshape 与连续 Slice 优先成为共享存储 view；不能证明连续安全时才生成复制 Kernel；Transpose/Concat 仍物化 |
| Cast / Where / Sin / Cos / GridSample2D | FP16/FP32 边界、cold/warm 选择以及 BEV rotate/shift 所需的静态逐元素或最近邻采样 Kernel |
| Conv2D / BatchNorm / MaxPool2D | Conv2D 的生产 FP32 路径使用 cuDNN；generated direct Conv 保留为可检查的正确性实现；其余按静态 Kernel 执行 |
| MSDeformableAttention | FP32 双线性采样与多 level/head/point 加权归约；独立切片有历史 SM86 验证 |

这是一套以 FP32 计算为主的实现；FP16 当前只用于统一 step 的图像/状态存储边界，没有 Tensor Core/TF32 或混合精度自动调优。generated Kernel 提供可检查的基线，cuDNN/cuBLAS 提供成熟密集算子的生产执行候选；后续引入快速数学与混合精度必须重新验证误差。

`CudaSession` 通过 [CuPy RawModule](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html) 调用 NVRTC，持有独立 stream 和只读驻留权重，并在 Session 建立时依据 HXP 候选为符合合同的 Conv/MatMul 选择 cuDNN/cuBLAS 或 generated CUDA。最终选择写入 `session.info`、`kernel_info` 和 profile；vendor 调用执行中失败不会逐节点自动切换实现。当前 HXP 与本版 codegen 绑定，优化前的 schema 1 包需重建。Session 初始化时根据图生成内存计划，为外部输入/输出与常量建立独立存储，并为内部值一次性分配 arena；每个内部 Tensor 通过 allocation offset 建立 view，互不重叠的生命周期可以复用同一片地址，零拷贝 Reshape/连续 Slice 则以不同 shape/offset 复用源存储。每次 run 检查输入、复制到设备、按 launch metadata 启动所选实现、取回主机输出。整张图都在单个 GPU 上执行；`numpy-cpu` 是显式的独立参考路径。生产尺寸 step HXP 已在 RTX 3080 上按 `705,177,100` bytes 的 Tensor 计划完成加载与 cold/warm 执行；cuDNN 另设 8 MiB workspace budget，合计计划预算/准入值为 `713,565,708` bytes，不含 module、context、Event 和 vendor 临时分配。桌面 RTX 3070 的实际峰值和性能仍需在该设备上单独复验。

计时遵循 [CUDA Event 与预热原则](https://docs.cupy.dev/en/stable/user_guide/performance.html)：Session 初始化、fresh-process 首次 `run` 和稳态端到端时间分别报告。`gpu_stream_ms` 含输入上传及设备队列内分配/提交间隙，不含输出下载；端到端壁钟包含输出下载和同步。`run(profile=True)` / CLI `--profile` 为每个 launch 记录 CUDA Event 和实际 `implementation`，并输出按 op 与 implementation 聚合的 count/total/mean/min/max/share；默认关闭，避免 Event 本身扰动普通执行。最终受扰动 Profile 为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 launch 的 Kernel Event 求和为 `63.4391 ms`：cuDNN 55 次 / `36.6409 ms`、CuPy/cuBLAS 118 次 / `8.0060 ms`、generated CUDA 530 次 / `18.7923 ms`。Trace 的 host 时间条与 GPU Event 耗时分别标注，未构造伪造的统一硬件时钟。cuDNN/cuBLAS 的第一次调用还包含库初始化和算法缓存，不能与预热后的 steady invocation 混为同一分布。

编译器自身和 CUDA Kernel 均在本项目中编写。CuPy 仅作为设备接入层，NVRTC 是 NVIDIA 提供的设备编译器；未来可以把该设备接入层替换为独立 C/C++ Runtime Driver。
