# 面向异构边缘推理的 AI 编译器—Runtime 协同架构设计

> 本文保留总体愿景。当前源码状态、接口落地方案与参考源码入口见 [架构说明](docs/architecture.md)，验收顺序以 [开发路线图](docs/roadmap.md) 为准。本文第 10 节的原始版本设想不代表已实现进度：目前已有 CUDA SM 8.6 单卡原型并完成 RTX 3070 Ti Laptop 验证；LLVM CPU、RISC-V 与 AIE 后端仍未实现。

## 1. 项目定位

本项目不是单纯实现“模型到机器码”的编译器，也不是只做一个模型推理引擎，而是设计一套：

> **面向异构边缘设备的 AI 编译器与推理 Runtime：统一模型语义、图优化、设备划分和执行接口；允许 CPU、GPU、NPU 后端以插件形式接入，并保留各自的硬件专用优化。**

更准确的英文定位可以是：

> **A compiler–runtime co-design prototype for heterogeneous edge inference**

中文可以称为：

> **面向异构边缘推理的编译器—Runtime 协同原型**

项目的重点不是重新制造一个比 IREE、TVM、OpenXLA 更庞大的“万能编译器”，而是解决现有系统中仍然存在的几个关键结构性问题：

1. **语义可移植相对容易，性能可移植很难。**
2. **编译器插件和 Runtime 插件之间缺少完整、统一的契约。**
3. **跨设备划分不能只看算子是否支持，还必须考虑数据搬运、Layout 转换、启动开销、内存预算和实时性。**
4. **编译器和推理 Runtime 往往割裂，难以统一规划计算、内存、同步和回退。**
5. **模型节点、编译 IR、设备 Kernel 和运行时性能之间缺乏清晰的跨层追踪。**

---

## 2. 总体架构

系统由编译期和运行期两部分组成。

```text
                         编译期

PyTorch / ONNX / StableHLO
             ↓
       Frontend Importer
             ↓
     Canonical Model IR
             ↓
Shape / Type / Fusion / Quantization
             ↓
       Partition IR
             ↓
       Common Tensor IR
             ↓
  Target Mapping / Legalization
             ↓
┌────────────┼────────────┬────────────┐
│ CPU 后端   │ GPU 后端   │ NPU 后端   │
│ LLVM       │ Triton等   │ MLIR-AIE等 │
└────────────┴────────────┴────────────┘
             ↓
设备 Kernel + Host Execution Plan
             ↓
        统一可执行模型包

================================================

                         运行期

加载模型包、权重和执行计划
             ↓
设备选择、Shape Variant 选择
             ↓
内存分配、Copy、Dispatch、Sync
             ↓
CPU / GPU / NPU 异构执行
             ↓
模型输出、性能统计和回退
```

核心设计思想是：

> **编译器不仅编译“设备上的计算程序”，还要编译“模型怎样被执行”的主机控制程序。**

因此，最终编译产物应同时包含：

1. **Device Program**
   - CPU 本地代码；
   - GPU Kernel；
   - NPU 可执行程序；
   - 厂商二进制 Blob。

2. **Host Execution Plan**
   - Buffer 分配；
   - 数据搬运；
   - Kernel 启动；
   - Event 和同步；
   - 设备间切换；
   - 回退路径；
   - 动态 Shape 变体选择。

---

## 3. 六级编译架构

## 3.1 Frontend Adapter

第一层负责从不同模型框架导入程序。

首期可支持：

```text
torch.export / FX
ONNX
StableHLO
```

这一层的职责包括：

- 导入模型；
- 规范化算子；
- 提取权重和常量；
- 解析 Shape、dtype 和量化参数；
- 将框架特有语义转换为公共语义；
- 保留源模型节点和后续 IR 之间的映射关系。

不建议让每个硬件后端直接读取 ONNX 或 PyTorch Graph，否则每个后端都需要重复实现：

- Broadcasting；
- Opset；
- 动态 Shape；
- 属性解析；
- 控制流；
- 量化语义；
- 框架特有算子。

第一版建议优先实现：

```text
torch.export → Canonical Model IR
```

然后再加入 ONNX 前端，用来验证 Frontend Adapter 是否真正可扩展。

---

## 3.2 Canonical Model IR

这一层回答：

> **模型要计算什么？**

例如：

```text
%0 = model.matmul %input, %weight
%1 = model.add %0, %bias
%2 = model.gelu %1
return %2
```

该层应表达：

- Tensor Shape；
- dtype；
- Broadcasting 语义；
- 动态维度；
- 量化参数；
- 权重和常量；
- 控制流；
- 副作用；
- 可变状态；
- 设备无关的 Layout 约束。

该层不应该出现：

```text
CUDA Warp
Shared Memory
AIE Compute Tile
DMA Buffer Descriptor
RISC-V Register
```

这些属于目标硬件映射层。

不建议从零定义上百个 AI 算子。可以采用：

```text
StableHLO 或上游 MLIR Dialect
+
少量自定义 Heterogeneous Dialect
```

自定义扩展可以集中在：

```text
hetero.state
hetero.kv_cache
hetero.opaque_call
hetero.device_constraint
hetero.accuracy_constraint
hetero.deadline
```

---

## 3.3 Graph Optimization 与分析层

这一层完成设备无关或相对设备无关的公共优化。

主要 Pass 包括：

```text
Shape Inference
Type Inference
Constant Folding
Dead Code Elimination
Common Subexpression Elimination
Operator Decomposition
Operator Fusion
Quantization Propagation
Layout Constraint Propagation
Alias Analysis
Side-effect Analysis
Weight Prepacking Preparation
```

例如：

```text
MatMul
  ↓
Add
  ↓
GELU
```

可以识别为：

```text
FusedMatMulBiasGELU
```

但公共层不必立刻决定最终是否一定生成一个融合 Kernel。

更合理的做法是：

- 公共层保留融合机会和等价语义；
- 后端根据硬件能力、寄存器压力、本地内存和性能模型决定是否真正融合。

---

## 3.4 Partition IR

这是整个系统中最值得形成差异化的层。

它回答：

> **模型的每一部分应该在哪个设备上执行？**

示例：

```text
%0 = hetero.partition @cuda {
    %a = matmul ...
    %b = gelu %a
    return %b
}

%1 = hetero.transfer %0
    from @cuda to @cpu

%2 = hetero.partition @cpu {
    %c = custom_op %1
    return %c
}
```

### 3.4.1 后端不能只返回“支持或不支持”

每个后端应返回更完整的信息：

```text
是否支持该算子
支持哪些 dtype
支持哪些 Shape
支持哪些动态维度
要求什么 Layout
需要多少本地内存
估计运行时间
估计启动开销
估计峰值内存
输入输出转换成本
编译成本
可靠性等级
精度误差范围
```

### 3.4.2 全局 Cost Model

设备划分不能只做贪心匹配，而应考虑：

\[
J =
\sum_v C_{\text{compute}}(v,d_v)
+
\sum_{(u,v)} C_{\text{transfer}}(d_u,d_v)
+
C_{\text{layout}}
+
C_{\text{launch}}
+
\lambda_m C_{\text{memory}}
+
\lambda_p C_{\text{p99 latency}}
\]

其中：

- \(C_{\text{compute}}\)：算子在目标设备上的计算成本；
- \(C_{\text{transfer}}\)：跨设备数据搬运成本；
- \(C_{\text{layout}}\)：Layout 转换成本；
- \(C_{\text{launch}}\)：Kernel 或子图启动成本；
- \(C_{\text{memory}}\)：峰值内存和内存预算；
- \(C_{\text{p99 latency}}\)：实时性和尾延迟目标。

约束条件包括：

```text
目标设备必须支持该算子
设备内存不能超限
精度误差不能超限
实时任务必须满足 Deadline
某些可变状态必须驻留在同一设备
KV Cache 不应频繁跨设备搬运
```

第一版不需要复杂的机器学习 Cost Model。

可以先使用：

- 静态硬件参数；
- 实测 Benchmark 表；
- 线性搬运模型；
- 简单动态规划或图划分算法。

---

## 3.5 Common Tensor IR

这一层回答：

> **张量计算怎样转换成迭代空间、Reduction、Tile 和 Buffer？**

建议尽量复用已有 MLIR Dialect：

```text
Linalg
Tensor
Arith
SCF
Affine
Vector
MemRef
```

公共 Tensor IR 适合表达：

```text
Iteration Space
Reduction Axis
Tile Candidate
Data Dependency
Buffer Lifetime
Layout Constraint
Vectorization Opportunity
Memory Reuse
```

它不应该直接绑定某个具体厂商。

例如，矩阵乘法可在该层表达为：

```text
M / N / K 迭代空间
K Reduction
A、B、C 的索引映射
Tile 候选
Buffer 约束
```

---

## 3.6 Target Mapping IR

低层不能强行统一。

正确设计应该是：

```text
                    Common Tensor IR
                            │
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
     CPU Mapping        GPU Mapping       Spatial NPU Mapping
```

### CPU Mapping IR

表达：

```text
Loop Nest
Cache Blocking
SIMD / RVV / AVX
Thread Parallelism
NUMA
Stack / Heap Buffer
```

### GPU Mapping IR

表达：

```text
Grid
Block
Warp
Thread
Shared Memory
Tensor Core
Async Copy
Register Tile
```

### Spatial NPU Mapping IR

表达：

```text
Compute Tile
Mem Tile
Local Memory
ObjectFifo
DMA
Stream Route
Lock
Pipeline
```

原则是：

> **上层统一计算语义，下层尊重硬件差异。**

不应设计一个同时直接表达 Warp、AIE Tile、RISC-V 寄存器的万能低层 IR。

---

## 4. Backend Package 设计

一个后端不能只是：

```python
compile(model)
```

建议将每个硬件后端定义为完整 Backend Package。

```text
backend_cuda/
├── target_profile.json
├── compiler_plugin.so
├── runtime_plugin.so
├── profiler_plugin.so
├── kernels/
└── conformance_tests/
```

一个完整 Backend 应由以下八部分组成：

```text
Backend =
Capability
+ Legalization
+ Cost Model
+ Target Lowering
+ Artifact Compiler
+ Runtime Driver
+ Profiler
+ Conformance Tests
```

---

## 4.1 Target Profile

描述目标硬件和软件栈能力：

```text
Target ID
硬件版本
驱动版本
固件版本
支持的算子
支持的 dtype
支持的 Shape
动态 Shape 能力
内存空间
对齐要求
向量宽度
矩阵指令形状
异步搬运能力
支持的 Layout
稀疏能力
最大本地内存
```

示例：

```json
{
  "target": "cuda-sm86",
  "dtypes": ["fp32", "fp16", "int8"],
  "dynamic_shape": "limited",
  "async_copy": true,
  "local_memory_bytes": 65536,
  "tensor_core_shapes": ["16x8x16"],
  "runtime": "cuda"
}
```

---

## 4.2 Compiler Plugin

建议接口：

```cpp
class CompilerBackend {
public:
    SupportResult analyzeSupport(
        Operation* op,
        const ShapeEnvironment& shapes);

    std::vector<ExecutionCandidate> enumerateCandidates(
        PartitionOp partition,
        const CompileObjective& objective);

    CostEstimate estimate(
        const ExecutionCandidate& candidate);

    TargetModule lower(
        PartitionOp partition,
        const ExecutionCandidate& candidate);

    CompiledArtifact compile(
        TargetModule module);
};
```

作用包括：

- 查询支持范围；
- 枚举不同实现方案；
- 估计成本；
- 完成目标 Legalization；
- Lower 到目标 IR；
- 生成二进制或厂商 Artifact。

---

## 4.3 Runtime Plugin

建议接口：

```cpp
class RuntimeBackend {
public:
    DeviceHandle openDevice(const DeviceOptions&);

    BufferHandle allocate(
        size_t bytes,
        MemoryType memoryType);

    BufferHandle importBuffer(
        const ExternalBuffer&);

    Event copy(
        const BufferHandle& src,
        const BufferHandle& dst);

    ExecutableHandle load(
        const CompiledArtifact& artifact);

    Event dispatch(
        const ExecutableHandle& executable,
        Span<TensorView> tensors);

    void wait(const Event& event);
};
```

Runtime Plugin 统一：

```text
allocate
free
copy
import/export buffer
load executable
dispatch
wait
synchronize
query device
```

底层可以分别对接：

```text
CUDA
ROCm
XRT
CANN
OpenCL
CPU Thread Runtime
```

---

## 4.4 Profiler Plugin

Profiler 返回：

```text
Kernel 延迟
Copy 延迟
Queue 等待时间
峰值内存
硬件计数器
功耗或能耗
错误信息
回退信息
```

这些信息既用于用户分析，也应反馈给 Cost Model。

---

## 4.5 Conformance Tests

每个后端必须通过统一测试：

```text
算子语义一致性
dtype 一致性
Shape 边界
动态 Shape
Layout 转换
异常输入
Runtime Buffer 生命周期
同步正确性
多次执行稳定性
Fallback 正确性
```

这样后端接入就不只是“能编译”，而是拥有可验证的契约。

---

## 5. 三类后端接入方式

不同厂商的开放程度不一样，因此系统必须支持三种接入深度。

## 5.1 Native Codegen Backend

编译器直接控制目标代码生成。

例如：

```text
RISC-V：
MLIR → LLVM → RISC-V ELF

AMD AIE：
MLIR-AIE → Peano → AIE2P ELF

开放 GPU：
LLVM / ISA Backend
```

这种后端可以深入到：

- 指令选择；
- 寄存器分配；
- 指令调度；
- 机器码生成。

---

## 5.2 Vendor Compiler Backend

系统编译到厂商公开边界，然后调用闭源或半闭源编译器。

```text
Common Tensor IR
        ↓
Vendor Graph IR / Kernel DSL
        ↓
厂商 Compiler
        ↓
Vendor Binary Blob
```

即使最终机器码由厂商编译器生成，该系统仍然可以负责：

- 模型导入；
- 图优化；
- 子图划分；
- 量化；
- Layout；
- 调度方案；
- Runtime 执行计划。

---

## 5.3 Library Delegate Backend

不生成 Kernel，而是直接调用成熟库：

```text
cuBLAS
cuDNN
oneDNN
厂商算子库
```

成熟系统应允许以下三种方式同时存在：

```text
自己生成 Kernel
调用厂商 Compiler
调用成熟算子库
```

然后由 Cost Model 选择最合适的执行候选。

---

## 6. Execution Plan IR

Execution Plan IR 是编译器和推理 Runtime 之间最重要的桥梁。

建议定义小型指令集合：

```text
exec.alloc
exec.free
exec.copy
exec.dispatch
exec.wait
exec.barrier
exec.if_shape
exec.call_host
exec.select_variant
exec.return
```

示例：

```text
%buf0 = exec.alloc @cuda 8MB

%e0 = exec.copy %input -> %buf0

%e1 = exec.dispatch @cuda.kernel0(%buf0)
    after %e0

%e2 = exec.copy %result -> @cpu
    after %e1

exec.wait %e2
exec.return %result
```

Partition IR 回答：

> **计算放在哪个设备？**

Execution Plan IR 回答：

> **什么时候分配、搬运、执行和同步？**

它应支持：

```text
异步执行
多 Stream / Queue
Event 依赖
内存复用
设备回退
Shape 分支
执行变体选择
CPU Host Call
```

---

## 7. 统一可执行模型包

最终产物不应只是一堆 `.so`、`.cubin` 或 `.xclbin`。

建议生成：

```text
model.hxp
├── manifest.fb
├── execution_plan.bytecode
├── constants.bin
├── weights.index
├── cpu/
│   └── model.so
├── cuda/
│   ├── kernel0.cubin
│   └── kernel1.cubin
├── aie/
│   └── model.xclbin
├── variants/
│   └── shape_guards
└── debug/
    └── provenance_map
```

其中：

### manifest

包含：

```text
模型版本
输入输出定义
设备要求
驱动和固件要求
精度约束
内存预算
可用执行变体
```

### execution_plan

包含：

```text
Host 执行控制程序
Buffer 分配
Copy
Dispatch
Event 和同步
Fallback
```

### constants 与 weights

包含：

```text
公共权重
常量
各后端预打包的权重
```

### target artifacts

包含：

```text
CPU Shared Library
CUDA Cubin / PTX
AIE xclbin
Vendor Binary Blob
```

### provenance_map

建立完整映射：

```text
原始 PyTorch / ONNX Node
        ↓
Canonical Model IR
        ↓
Fusion / Partition
        ↓
Target IR
        ↓
Kernel
        ↓
Runtime Dispatch
        ↓
Profiler Event
```

它用于回答：

```text
某个 Attention 为什么耗时 1.7 ms？
它经过了哪些 Pass？
融合成了哪个 Kernel？
运行时对应哪一次 Dispatch？
性能瓶颈出现在哪一层？
```

---

## 8. 推理 Runtime 架构

推理 Runtime 至少分成四层。

## 8.1 Device 与 Tensor 层

统一 Tensor 描述：

```text
data pointer
shape
stride
dtype
layout
device
memory type
ownership
```

支持的内存类型：

```text
Host Memory
Pinned Memory
Device Memory
External Buffer
DLPack
Zero-copy Buffer
```

核心对象可以定义为：

```cpp
struct TensorView {
    void* data;
    Shape shape;
    Strides strides;
    DType dtype;
    Layout layout;
    Device device;
    MemoryType memoryType;
};
```

---

## 8.2 Executable Loader

负责：

```text
加载模型包
检查版本和兼容性
加载后端 Artifact
检查设备能力
加载权重
建立常量索引
建立 Shape Variant 表
预创建必要的执行对象
```

---

## 8.3 Plan Executor

负责：

```text
执行 alloc / copy / dispatch / wait
管理 Stream、Queue 和 Event
管理 Buffer Arena
复用中间内存
重叠数据搬运和计算
传播错误
执行设备回退
收集性能信息
```

---

## 8.4 Inference Session

面向用户提供：

```cpp
Session session = Runtime::load("model.hxp");

session.bindInput("x", tensor);
session.run();

Tensor y = session.output("y");
```

Session 管理：

```text
模型状态
权重驻留
动态输入 Shape
缓存
运行统计
并发执行
错误和回退
```

未来如果扩展大模型 Serving，再增加：

```text
请求队列
Continuous Batching
KV Cache
Prefix Cache
流式输出
Prefill / Decode 调度
```

第一版不应立刻实现完整 Serving，而应先完成：

```text
单模型
单 Session
多后端
端到端推理闭环
```

---

## 9. 重点解决的五个问题

| 现有困难 | 本架构的设计响应 |
|---|---|
| 接一个新 NPU 需要修改很多层 | 版本化 Backend Package：Capability、Compiler、Runtime、Profiler |
| 图划分只判断支持性，不判断全局收益 | 计算、搬运、Layout、内存和实时约束共同驱动的 Cost Model |
| Compiler 和 Runtime 割裂 | Execution Plan IR + 统一模型包 |
| 同一个 Schedule 不能适应所有硬件 | Common Tensor IR + Target Mapping IR + Schedule Recipe |
| 跨层问题难定位 | Model Op → IR → Kernel → Runtime Timeline 的 Provenance |

最适合作为项目核心亮点的是：

> **Compiler–Runtime 协同与异构 Execution Plan。**

它能连接：

```text
模型图
编译优化
设备划分
内存计划
异步执行
真实推理性能
```

---

## 10. 开发版本路线

## v0.1：完整编译—推理闭环

支持算子：

```text
MatMul
Add
ReLU
GELU
LayerNorm
Softmax
```

限制：

```text
静态 Shape
FP32 / FP16
连续 Tensor
固定 Batch
单设备执行
```

后端：

```text
Reference Backend
→ NumPy 或朴素 CPU，用于正确性

LLVM CPU Backend
→ 本机 x86 执行

NVIDIA Backend
→ Triton 或 TileLang，在 RTX 3070 上执行
```

Runtime：

```text
Model Package
Execution Plan
Buffer Arena
CPU / CUDA Runtime Plugin
同步和异步 Dispatch
Profiler
```

验收标准：

```text
模型能够导入
公共 IR 可打印
Pass 有测试
两个后端结果一致
模型包可以加载
Runtime 可以执行
Profiler 能关联回模型节点
```

---

## v0.2：真正的异构推理

增加：

```text
CPU + GPU 子图划分
跨设备 Copy
Layout 转换
内存复用
失败回退
Shape Guard
多个编译变体
简单 Cost Model
```

验收：

```text
同一模型可部分在 CPU、部分在 GPU 执行
系统可以解释划分原因
跨设备搬运成本被纳入选择
后端不支持时可正确回退
```

---

## v0.3：验证跨 ISA 能力

增加：

```text
RISC-V LLVM Backend
```

目标链：

```text
Tensor
→ Loop
→ LLVM IR
→ RISC-V ISA
→ ELF
→ QEMU / Spike / Linux
```

目的不是追求最高性能，而是验证：

```text
公共 IR 不依赖 CUDA
编译链可以真正落到另一种 ISA
Runtime 接口具有可迁移性
```

---

## v0.4：空间型 NPU 后端

增加：

```text
AMD AIE2P Backend
```

链路：

```text
Common Tensor IR
       ↓
Spatial NPU Mapping IR
       ↓
MLIR-AIE / IRON
       ↓
Tile / ObjectFifo / DMA / Route
       ↓
Peano / LLVM-AIE
       ↓
xclbin + instruction stream
       ↓
XRT
```

目的：

```text
验证公共 Tensor IR 能支持空间型 NPU
验证 Backend Package 能适配不同执行模型
验证 Execution Plan 能驱动 XRT Runtime
```

---

## 11. 与现有系统的关系

本架构不应否定现有系统，而应吸收它们最成熟的设计。

```text
StableHLO
→ 公共高层模型语义

MLIR
→ Dialect、Pass、逐级 Lowering

TVM
→ Graph IR 与 Tensor IR 分层

ONNX Runtime
→ Capability、子图下放、Fallback

IREE
→ Execution Plan、HAL、AOT Runtime

PJRT
→ 统一设备和 Executable 接口
```

本项目的重点不是“发明第一个通用 AI 编译器”，而是：

> **将多后端接入、设备划分、执行计划、Runtime 和跨层观测定义成一套清晰、规模可控、能够真实运行的架构原型。**

---

## 12. 职业能力定位

该项目应形成“T 型能力”。

### 横向能力

```text
PyTorch / ONNX
→ Model IR
→ Graph Optimization
→ Tensor IR
→ Partition
→ Runtime
→ CPU / GPU / NPU
```

### 纵向能力

选择一个后端做到足够深，建议优先选择 NVIDIA GPU：

```text
Tensor IR
→ Tiling / Layout
→ Triton / TileLang
→ Warp / Shared Memory
→ PTX
→ CUDA Runtime
→ RTX 3070 真机 Profiling
```

因此职业定位可以是：

> **异构 AI 编译器与推理系统工程师**

纵向专长为：

> **GPU Kernel 编译、CodeGen 与推理性能优化**

横向具备：

> **模型、IR、Pass、Runtime 和多硬件后端的完整系统视野**

---

## 13. 最终设计原则

### 原则一：上层统一语义，下层尊重差异

```text
统一：
模型语义
Shape
dtype
量化
依赖关系
公共优化

分叉：
Layout
Schedule
Thread / Tile Mapping
Memory Hierarchy
Machine Code
Runtime Driver
```

### 原则二：编译器和 Runtime 共同设计

不能只生成 Kernel，还要生成：

```text
内存计划
数据搬运
执行顺序
同步关系
Fallback
Shape Variant
```

### 原则三：后端是完整契约，不是一个函数

```text
Capability
Legalization
Cost Model
Lowering
Compiler
Runtime
Profiler
Tests
```

### 原则四：先做完整闭环，再扩大算子数量

第一版最重要的是：

```text
一个模型
两个后端
一个异构 Execution Plan
一个可运行的推理 Runtime
一套可解释的性能数据
```

不是一开始支持几十个模型和全部算子。

### 原则五：架构广度必须有一个深后端支撑

否则容易变成：

```text
架构图很完整
但每个后端只是调用现成 API
```

项目必须在至少一个后端上做到：

```text
真实代码修改
正确性测试
性能 Profiling
收益解释
退化边界
端到端 Benchmark
```

---

## 14. 一句话总结

> **设计一套面向异构边缘推理的 AI 编译器—Runtime 协同架构：用公共 Model IR 和 Tensor IR 统一模型语义，用 Cost-aware Partition 完成设备划分，用可插拔 Backend Package 对接 CPU、GPU 和 NPU，用 Execution Plan IR 编译主机执行逻辑，并通过统一 Runtime 完成模型加载、内存管理、异步调度、回退和性能观测。**
