# AI 编译器中的 Tiling Shape 推导与选择

Tiling 是把一个大算子拆成多个较小计算块，使数据能够在寄存器、片上 SRAM、Cache 和外部
内存之间分层复用。Tiling Shape 决定每次处理多少数据，也决定线程组织、片上存储占用、
指令选择、流水深度和最终性能。

Tiling Shape 通常没有唯一的解析解。编译器需要先根据算子语义推导合法的 Tile 关系，再根据
目标硬件建立约束，生成候选配置，最后使用性能模型和目标设备实测选出结果。

完整流程是：

```text
Tensor Shape 与算子语义
          ↓
推导输入、输出和 Reduction Tile 的关系
          ↓
建立线程、寄存器、SRAM、指令和布局约束
          ↓
生成多级 Tiling 候选
          ↓
静态合法性过滤
          ↓
Cost Model 排序
          ↓
正确性验证与 Autotune
          ↓
最终 Schedule 与 Codegen
```

## 1. Tiling Shape 表示什么

### 1.1 Tensor Shape 与 Tile Shape

Tensor Shape 描述完整张量：

```text
A[M, K]
B[K, N]
C[M, N]
```

Tile Shape 描述一次局部计算覆盖的区域。矩阵乘通常写为：

```text
BM：一次计算 C 的多少行
BN：一次计算 C 的多少列
BK：一次处理归约维 K 的多少元素
```

对应关系是：

```text
A Tile = BM × BK
B Tile = BK × BN
C Tile = BM × BN
```

### 1.2 Tiling 通常是多级的

高性能后端不会只有一个 Tile。常见层次为：

```text
完整 Tensor
    ↓
Cluster / Core Tile
    ↓
CTA / Workgroup Tile
    ↓
Warp / Subgroup Tile
    ↓
Vector / MMA Instruction Tile
    ↓
Thread Register Fragment
```

例如一个配置可能是：

```text
CTA Tile         = 128 × 128 × 32
Warp Tile        = 64 × 64 × 32
MMA Tile         = 16 × 8 × 16
Pipeline Stages  = 3
```

每一级 Tile 都要能够被下一级执行单元合理覆盖。

### 1.3 逻辑 Shape 与物理 Storage Shape

逻辑 Tile 描述算法范围：

```text
A logical tile = 32 × 32
```

为了地址对齐或避免 SRAM Bank Conflict，物理分配可能是：

```text
A storage tile = 32 × 33
```

因此编译器必须分别记录：

- 逻辑计算区域；
- 物理存储布局；
- 有效元素范围；
- padding 和 alignment。

在这一阶段，DDR/Global Memory 中通常仍保存完整 Tensor。Tiling 首先产生的是地址范围和访问计划；
只有执行 DMA、load 或 copy 时，对应 Tile 的数据才真正进入片上 Buffer。

## 2. Tiling Shape 求解的两个阶段

### 2.1 阶段一：合法性推导

合法性推导回答：

> 如果输出计算这一块，至少需要读取哪些输入区域？Reduction 维需要怎样遍历？

它主要由以下信息决定：

- Tensor Shape；
- 算子 indexing relation；
- Broadcast 规则；
- Reduction 维；
- stride、dilation、padding；
- producer/consumer 数据依赖；
- Layout 与 stride。

这一步决定输入 Tile 和输出 Tile 之间的数学关系。

### 2.2 阶段二：性能选择

性能选择回答：

> 在所有合法 Tile 中，哪个配置在目标设备上最快？

它主要考虑：

- 片上存储容量；
- 寄存器压力；
- 线程、Warp、Core 和 Cluster 数量；
- Vector、SIMD、MMA 或专用矩阵指令粒度；
- DRAM 带宽和计算吞吐；
- 合并访存与 Bank Conflict；
- Pipeline Stage 和异步搬运；
- Tail、Padding 和 Mask 成本；
- 融合算子的额外工作集；
- Occupancy 与数据复用之间的平衡。

这一步一般是约束搜索问题，不会仅靠 Tensor Shape 得到唯一答案。

### 2.3 统一的问题定义

可以用一个与具体硬件无关的形式描述 Tiling。设算子的迭代域为 `D`，第 `j` 个 Operand 的
索引映射为 `f_j`。选定迭代 Tile `T ⊆ D` 后，这个 Tile 所需的数据区域为：

```text
F_j(T) = { f_j(i) | i ∈ T }
```

`F_j(T)` 是 Operand 的逻辑 footprint。编译器再根据 Layout、对齐、padding 和硬件存储规则，
把逻辑 footprint 变成物理 Buffer。最终求解的结果不是单独的几个尺寸，而是一份完整计划：

```text
TilePlan = {
  levels,              // Core / CTA / Warp / Instruction / Thread Tile
  operand_layouts,     // 每个 Operand 的物理布局
  thread_mapping,      // 迭代点到线程或计算单元的映射
  vector_width,
  instruction_shape,
  num_stages,
  buffer_offsets,
  copy_schedule,
  boundary_mode,
  runtime_guards,
  estimated_resources
}
```

求解目标可以写成：在所有满足语义、资源和指令约束的 `TilePlan` 中，选择预测或实测延迟最小的
方案。这个定义可以统一描述 Elementwise、Reduction、GEMM、Conv 和 Attention。

## 3. 不同算子的 Tile 传播规则

### 3.1 Elementwise

设：

```text
C[i, j] = A[i, j] + B[i, j]
```

若：

```text
C Tile = TM × TN
```

则：

```text
A Tile = TM × TN
B Tile = TM × TN
```

Elementwise 没有 Reduction 维，通常不需要复杂的数据复用。选择重点是连续访存、Vector Width、
每线程处理元素数和线程块大小。

### 3.2 Broadcast

设：

```text
C[i, j] = A[i, j] + B[j]
```

若输出 Tile 为：

```text
C Tile = TM × TN
```

则输入区域为：

```text
A Tile = TM × TN
B Tile = TN
```

`B` 在 M 轴广播。编译器应识别这种复用，避免重复搬运同一段 `B`。

### 3.3 Transpose

设：

```text
B[j, i] = A[i, j]
```

如果：

```text
B Tile = TN × TM
```

对应输入区域为：

```text
A Tile = TM × TN
```

逻辑上只是维度置换，但物理上可能产生跨步访问。后端需要决定：

- 在加载到 SRAM 时完成转置；
- 使用向量化转置；
- 改变 consumer 的 indexing map；
- 或物化一个独立 Transpose Kernel。

### 3.4 Reduction

设：

```text
Y[i] = Σ X[i, k]
```

Tile 通常表示为：

```text
Output Tile    = TM
Reduction Tile = TK
Input Tile     = TM × TK
```

为了完成一个输出 Tile，必须遍历完整 K，或者先生成多个 partial reduction，再执行合并。需要同时
决定：

- 一个输出由线程、Warp 还是 CTA 负责；
- 使用 Warp Shuffle 还是 SRAM Tree Reduction；
- accumulator dtype；
- 是否融合 Softmax、LayerNorm 等后续操作。

#### Softmax 与 LayerNorm

Softmax 通常至少包含 `max`、`exp`、`sum` 和归一化。若归约轴不能由一个执行单元一次覆盖，
编译器需要保存并合并各 Tile 的局部最大值和局部和；为了数值稳定，还要在最大值变化时重标定
已有结果。LayerNorm 需要计算均值与方差，可采用两遍归约或 Welford 形式，并通常使用 FP32
accumulator。它们的 Tile 选择必须同时决定：

- 归约轴由一个线程、Warp、CTA 还是多个 CTA 完成；
- partial result 存在寄存器、SRAM 还是全局内存；
- 跨 Tile 合并方式和同步边界；
- 尾部 mask 是否参与 `max`、`sum`、均值和方差；
- 中间精度和允许的数值误差。

### 3.5 GEMM

矩阵乘：

```text
C[M, N] = A[M, K] × B[K, N]
```

标量关系为：

```text
C[i, j] += A[i, k] × B[k, j]
```

选择：

```text
C Tile = BM × BN
K Chunk = BK
```

则：

```text
A Tile = BM × BK
B Tile = BK × BN
```

在三维循环空间 `(i, j, k)` 中，典型 indexing maps 是：

```text
A: (i, j, k) -> (i, k)
B: (i, j, k) -> (k, j)
C: (i, j, k) -> (i, j)
```

所以 `i→BM`、`j→BN`、`k→BK` 是由算子语义决定的。

### 3.6 Convolution

对于二维卷积，输出空间 Tile 为 `THo × TWo` 时，所需输入空间范围为：

```text
input_tile_h =
    (THo - 1) × stride_h
    + dilation_h × (kernel_h - 1)
    + 1

input_tile_w =
    (TWo - 1) × stride_w
    + dilation_w × (kernel_w - 1)
    + 1
```

还需要结合 padding 计算边界处的有效范围。Conv 的候选维度一般包括：

```text
TN、THo、TWo、TCin、TCout、TKh、TKw
```

如果将 Conv 映射为 implicit GEMM，Tile 还会转换为相应的 M/N/K 分块。

### 3.7 Attention

Attention 包含多个相互约束的 Tile：

```text
Q Tile       = BQ × D
K/V Tile     = BK × D
Score Tile   = BQ × BK
Output Tile  = BQ × D
```

这是单 Batch、单 Head 且 Head Dimension `D` 不再分块时的简化形式。完整计划还可能对 Batch、
Head 和 `D` 继续分块，并为 QK、Softmax 和 PV 三个阶段选择不同的物理 Layout。

如果采用分块在线 Softmax，还必须在多个 K/V Tile 之间维护：

- 当前行最大值；
- 指数和；
- 重标定后的输出 accumulator。

因此 Attention Tile 不仅受矩阵乘约束，还受 Softmax 状态、数值稳定性和 SRAM 生命周期约束。

### 3.8 不规则访问算子

Gather、Scatter、GridSample、稀疏算子和数据相关路由通常不能只依赖 affine indexing map 推导。
编译器可能需要：

- 索引范围分析；
- 专用计算模板；
- 运行时边界检查；
- 稀疏格式元数据；
- 将该算子设置为子图切分边界。

## 4. Tiling Shape 的约束条件

### 4.1 语义、Shape 与边界约束

Tile 计划首先必须保持原算子的语义：

- 所有迭代点被覆盖一次，不能遗漏或重复写入；
- 读取发生在对应数据产生之后；
- 并行写入不能产生未定义的数据竞争；
- Reduction 必须覆盖完整归约域，并使用合法的合并顺序；
- 原地写和别名不能破坏仍然存活的输入；
- 浮点重排和低精度累加必须满足精度契约。

存在 loop-carried dependence 时，Tiling 和重排还必须满足 dependence distance。跨 CTA 的
Reduction 不能直接假定共享寄存器或 SRAM，通常需要 partial-result Buffer、原子操作或第二个
Reduction Kernel。

Tile 不必整除 Tensor Shape，但必须正确定义尾部行为：

```text
tile_count = ceil_div(full_extent, tile_extent)
```

边界 Tile 可以使用：

- masked load/store；
- zero fill；
- padding；
- 独立 tail kernel；
- 主区域使用高性能模板，尾部使用通用模板。

动态 Shape 可以使用 Shape Bucket、运行时特化或保守 fallback。

### 4.2 片上存储容量

以 GEMM 为例，A/B 的元素字节数分别为 `E_A`、`E_B`，Pipeline Stage 数为 `S`，只缓存 A/B 时：

```text
tile_storage_bytes =
    S × (E_A × BM × BK + E_B × BK × BN)
```

考虑实际分配粒度后，更精确的形式为：

```text
tile_storage_bytes =
    Σ(stage=0..S-1) [allocated(A_stage) + allocated(B_stage)]
    + epilogue_scratch
    + shared_metadata
```

其中 `allocated()` 已包含该 Stage 的物理 stride、Layout/Bank padding 和地址对齐。真实分配还可能
包括：

```text
+ quantization parameters
+ optional C Tile
+ 与 Tile 共用同一存储池的其他临时值
```

硬约束首先包括单 CTA 分配上限：

```text
per_cta_storage <= max_storage_per_cta
```

同时至少要能在一个 Core 上驻留一个 CTA：

```text
align_up(per_cta_storage, allocation_granularity)
    <= available_storage_per_core
```

如果性能策略希望同时驻留 `R` 个 CTA：

```text
R × align_up(per_cta_storage, allocation_granularity)
    <= available_storage_per_core
```

`R=1` 也可能是最优配置。片上存储是否接近满载，是 Occupancy 与数据复用之间的性能权衡，
不是固定规则。

若 C partial sum 始终保存在寄存器中，不应把 C Tile 重复计入 SRAM；只有显式暂存、跨执行单元
归约或发生 spill 时，才按实际生命周期计入相应存储层。

### 4.3 寄存器约束

如果每线程保存 `RM × RN` 个 FP32 累加器，最低 accumulator 存储近似为：

```text
accumulator_bytes_per_thread ≈ RM × RN × 4
```

还要加上：

- A/B fragment；
- 地址和索引；
- Pipeline 临时值；
- Epilogue 临时值；
- Predicate 和循环状态。

寄存器使用过高会降低活跃 Warp/CTA 数，严重时发生 spill。CUDA 的 “local memory” spill 通常
位于设备显存，不能与 NPU/Vortex 的片上 Local Memory 混为一谈。

### 4.4 线程与执行层次约束

线程组织必须满足硬件上限：

```text
threads_per_cta <= hardware_max_threads_per_cta
warps_per_cta   <= hardware_max_warps_per_cta
```

总 CTA 数是否足以覆盖所有 Core/SM 属于性能条件：CTA 太少时设备利用率会下降，但只要计算覆盖和
边界处理正确，并不因此成为非法 Kernel。

简单的一线程一输出 GEMM 中：

```text
threads_per_cta = BM × BN
```

Tensor Core 或向量化实现通常让一个线程持有多个输出，此时 `threads_per_cta` 不再等于
`BM×BN`，而由 Warp Tile 和 Thread Fragment 决定。

### 4.5 指令粒度约束

若硬件矩阵指令粒度为：

```text
MMA_M × MMA_N × MMA_K
```

通常希望：

```text
BM 是 MMA_M 的整数倍
BN 是 MMA_N 的整数倍
BK 是 MMA_K 的整数倍
```

这是高效映射条件，不是数学正确性的必要条件。非整除部分可以使用 mask、padding 或尾部路径。

向量加载还要求：

- 地址满足对齐；
- 连续维长度适配 Vector Width；
- 线程访问形成合并事务；
- 边界处能够安全降级。

### 4.6 Layout 与合并访存约束

Tile Shape 必须和 Layout、线程映射一起设计。理想的全局内存访问类似：

```text
lane0 → address 0
lane1 → address 1
lane2 → address 2
...
```

如果相邻 lane 访问间隔很大，会产生多个内存事务。常见解决方法包括：

- 调整线程到数据的映射；
- 向量化加载；
- 在搬入 SRAM 时转置；
- 使用 layout transform 或 swizzle；
- 选择不同的 BM/BN 方向。

### 4.7 SRAM Bank Conflict 约束

Bank Conflict 由以下因素共同决定：

- Bank 数量；
- Bank 宽度；
- 元素字节数；
- 线程访问步长；
- Storage Shape。

解决方法包括：

- 添加 padding；
- 改变物理 Layout；
- XOR swizzle；
- 改变 Warp Tile 或 fragment 排列；
- 使用硬件提供的专用矩阵布局。

逻辑 `32×32` Tile 是否需要物理 `32×33`，必须根据具体访问模式判断。

### 4.8 Pipeline 与异步搬运约束

双 Buffer 的目标是重叠：

```text
计算当前 Tile
      与
搬运下一个 Tile
```

Stage 数增加会提高片上存储占用，也会增加同步和状态管理。一个有效流水需要满足：

- 硬件支持异步搬运或独立 DMA；
- 搬运与计算确实能够并行；
- Buffer 复用前存在正确的完成等待；
- 所有参与线程遵守 Barrier 语义；
- Stage 周期能够覆盖搬运延迟。

因此 `BM/BN/BK/num_stages` 应联合选择。

这里的 Buffer 是编译器为物理 SRAM 划分的存储区域及其复用计划。`num_stages=2` 时，通常由
`buffer[0]` 保存当前计算 Tile，`buffer[1]` 接收下一 Tile；下一轮交换角色。SRAM 单元属于硬件，
双 Buffer 的地址、生命周期、DMA、等待和切换顺序由编译器 Schedule 明确生成。

### 4.9 Occupancy 约束

Occupancy 受以下资源共同限制：

```text
threads_per_cta
registers_per_thread
registers_per_cta
shared_memory_per_cta
hardware_cta_limit
```

至少能驻留一个 CTA 是硬约束。更高 Occupancy 通常有利于隐藏延迟，但不是越高越好。低 Occupancy
的大 Tile 可能拥有更好的数据复用和 ILP，因此 Occupancy 一般应进入性能模型，而不是被设成统一的
硬阈值。

### 4.10 融合与生命周期约束

算子融合会改变 Tile 工作集。例如：

```text
MatMul + Bias + ReLU
```

通常可以让 MatMul accumulator 直接完成 Bias 和 ReLU，不需要写回再读取。

但：

```text
MatMul + Residual + LayerNorm
```

还需要 Residual Tile、均值、方差和归约临时值。融合后是否仍能放入寄存器和片上存储，需要重新
计算 Buffer 生命周期，不能沿用未融合 MatMul 的 Tile 预算。

### 4.11 数值精度约束

Tile 和 Reduction 顺序可能改变浮点累加顺序。编译器需要明确：

- 输入 dtype；
- accumulator dtype；
- rounding mode；
- 是否允许 TF32；
- 量化 scale/zero-point 粒度；
- 误差门限；
- NaN/Inf 行为；
- 确定性要求。

候选即使更快，只要超过精度门限，也不能被选择。

## 5. 性能模型

### 5.1 GEMM 计算量与数据量

一个 GEMM K Tile 的计算量为：

```text
FLOPs = 2 × BM × BN × BK
```

只考虑 A/B 搬入时，通用数据量为：

```text
Bytes_AB = E_A × BM × BK + E_B × BK × BN
```

若 A/B 的元素大小同为 `E` 字节，可以写成：

```text
Bytes_AB = E × (BM × BK + BK × BN)
```

理想 Tile Arithmetic Intensity 为：

```text
AI_tile ≈
    2 × BM × BN × BK
    -------------------------------
    E × (BM × BK + BK × BN)
```

约去 BK 后：

```text
AI_tile ≈ 2 × BM × BN / (E × (BM + BN))
```

所以在这个理想模型里，增大 `BK` 不会自动提高 Arithmetic Intensity。BK 主要影响：

- DMA 和向量指令粒度；
- 循环与 Barrier 次数；
- Pipeline overlap；
- K Tail；
- 片上存储容量。

完整模型还要加入 C 写回、Bias、Residual、量化参数、Cache 行为和边界开销。

### 5.2 Roofline 式估算

可以估算：

```text
compute_time ≈ FLOPs / effective_compute_throughput
memory_time  ≈ Bytes / effective_memory_bandwidth
```

没有有效重叠时：

```text
latency ≈ compute_time + memory_time + synchronization_overhead
```

搬运和计算充分重叠时：

```text
latency ≈ max(compute_time, memory_time)
          + startup
          + drain
          + synchronization_overhead
```

这只是排序模型。有效吞吐会受到 Occupancy、Bank Conflict、指令混合和硬件调度影响。

### 5.3 Tail 成本

对维度 `D` 和 Tile `TD`：

```text
padded_extent = ceil_div(D, TD) × TD
waste_ratio   = (padded_extent - D) / padded_extent
```

Tail 比例可以进入 Cost Model，但不能为了完全整除而选择不符合硬件粒度的奇怪 Tile。

## 6. Tiling Shape 的求解流程

### 6.1 建立统一问题描述

编译器先收集：

```text
算子类型与 indexing relation
Tensor Shape / dynamic bounds
dtype / accumulator dtype
layout、stride、alignment
融合区域
目标设备属性
```

### 6.2 推导合法 Tile 关系

通过算子接口或 IR 分析得到：

```text
output tile
    ↓ backward slice
required input tiles
    ↓
reduction tiles
    ↓
halo / padding / mask requirements
```

这一阶段只确定关系，不急于决定具体数值。

### 6.3 生成多级候选

一个候选配置可以表示为：

```text
Candidate {
  block_tile:        (BM, BN, BK),
  warp_tile:         (WM, WN, WK),
  instruction_tile:  (IM, IN, IK),
  thread_tile:       (RM, RN),
  num_warps,
  num_stages,
  vector_width,
  input_layout,
  storage_layout,
  padding,
  fusion_strategy,
}
```

候选来源可以是：

- 规则枚举；
- 架构模板；
- 历史数据库；
- 学习型 Cost Model；
- 相邻 Shape 的配置迁移；
- 用户或后端提供的固定 Schedule。

### 6.4 执行硬约束过滤

```python
legal = []

for candidate in candidates:
    if candidate.threads_per_cta > device.max_threads_per_cta:
        continue

    if candidate.storage_bytes > device.max_storage_per_cta:
        continue

    if candidate.registers_per_thread > device.max_registers_per_thread:
        continue

    if not device.can_reside_one_cta(candidate):
        continue

    if not candidate.satisfies_instruction_contract:
        continue

    if not candidate.has_safe_boundary_behavior:
        continue

    legal.append(candidate)
```

硬过滤只删除不可执行或不正确的配置。目标 Occupancy、预计复用率等性能偏好应保留到排序阶段。

### 6.5 Cost Model 排序

Cost Model 可以包含：

```text
DRAM bytes
Cache / SRAM bytes
FLOPs 与指令数量
预计活跃 CTA / Warp
寄存器压力
Bank Conflict 风险
Barrier 数量
Pipeline overlap
Tail waste
融合收益
```

通常只保留得分最高的一小部分候选进入编译和实测，降低 Autotune 成本。

### 6.6 正确性门

每个候选必须验证：

- 输出 Shape 和 dtype；
- 与参考实现的误差；
- NaN/Inf；
- 尾部 Tile；
- 动态 Shape 边界；
- 多 Stage Buffer 复用；
- Barrier 和异步搬运顺序；
- 越界访问；
- 确定性要求。

不正确的候选不能参加性能排名。

### 6.7 目标设备 Autotune

```text
正确候选
    ↓
编译
    ↓
Warmup
    ↓
多轮设备计时
    ↓
去除异常样本或使用中位数
    ↓
选出最快配置
```

Autotune Key 至少应包含：

```text
device architecture
operator / fused region
M/N/K 或 Shape Bucket
dtype / accumulator dtype
layout / stride
alignment
compiler revision
candidate-space version
```

缓存不能只按算子名保存，否则不同 Shape 或设备可能错误复用同一个 Tile。

## 7. 动态 Shape 的处理

动态 Shape 常见有四种策略。

### 7.1 Shape Bucket

```text
M ∈ [1, 64]      → config A
M ∈ [65, 256]    → config B
M ∈ [257, 1024]  → config C
```

每个 Bucket 编译一个版本，Runtime 根据实际 Shape 选择。

Bucket 上界满足资源约束并不等于每个实际 Shape 都能无条件执行。Kernel 仍要按实际 extent 生成
mask 或尾部路径，并用 runtime guard 检查 Shape 关系、stride、alignment 和 dtype；不满足 guard
时进入 fallback。

### 7.2 运行时特化

第一次遇到新 Shape 时编译和调优，之后缓存。适合 JIT，但必须控制编译延迟和缓存规模。缓存键
需要区分 Shape Bucket 或实际特化 Shape，并包含 Layout、stride、alignment 和目标设备。

### 7.3 参数化 Kernel

一个 Kernel 接受动态 M/N/K，并通过 mask 处理边界。代码数量少，但可能不如专用版本快。

### 7.4 主路径与 Fallback

常见 Shape 使用特化高性能 Kernel，其余 Shape 使用通用 Kernel 或库调用。

## 8. 一个 GEMM 标量映射示例

考虑：

```text
A[2500, 256] × B[256, 512] = C[2500, 512]
dtype = FP32
num_stages = 2
```

以下示例只用于演示约束推导，不代表现代 MMA 后端的主路径。假设采用“一线程计算一个 C 元素”
的简单映射，A/B 使用双 Buffer，C 留在寄存器：

```text
threads_per_cta = BM × BN
lmem_bytes       = 2 × BK × (BM + BN) × 4
cta_count        = ceil_div(2500, BM) × ceil_div(512, BN)
k_iterations     = ceil_div(256, BK)
```

候选结构量如下：

| Tile `BM×BN×BK` | 线程/CTA | 双 Buffer | CTA 数 | K 迭代 |
|---|---:|---:|---:|---:|
| `2×2×16` | 4 | 512 B | 320,000 | 16 |
| `8×8×16` | 64 | 2,048 B | 20,032 | 16 |
| `16×16×16` | 256 | 4,096 B | 5,024 | 16 |
| `16×16×32` | 256 | 8,192 B | 5,024 | 8 |

这个表不能直接给出最快答案，因为还缺少：

- 硬件最大线程数；
- 寄存器数量；
- 片上存储分配粒度；
- CTA/Warp 调度能力；
- DMA 和 Barrier 成本；
- 实际计算吞吐；
- Tail 和访存效率。

正确做法是先用硬件约束过滤，再通过 Cost Model 和设备实测比较。

## 9. Tiling 在编译器流水线中的位置

Tiling 通常位于算子合法化和目标 Codegen 之间：

```text
模型导入
   ↓
高层 Graph / Tensor IR
   ↓
Shape 与 Layout 推导
   ↓
算子融合和子图切分
   ↓
Tile 传播与 Tile 候选选择
   ↓
Loop / Thread / Warp / Instruction Mapping
   ↓
Bufferization 与片上存储规划
   ↓
异步搬运与 Pipeline Schedule
   ↓
目标指令 Lowering
   ↓
机器码与 Runtime Launch Plan
```

实际实现中，Tile 选择和 Buffer 规划通常需要迭代：一个候选 Tile 生成具体物理 Layout 后，可能
发现 SRAM、Bank 或寄存器约束不满足，需要返回候选搜索阶段。

## 10. 编译器应保存哪些 Tiling 信息

一个可复现的编译产物应保存完整 `TilePlan`，至少记录：

```text
logical tile shape
physical storage shape
thread / warp / core mapping
vector or MMA shape
num_stages
buffer bytes and alignment
boundary strategy
fusion strategy
estimated resource usage
measured latency samples
correctness result
target device identity
compiler and schedule version
```

如果只保存 `BM/BN/BK`，通常不足以重建真实 Schedule。

## 11. 设计检查项

### 11.1 Tile 不是 Tensor Shape 的固定函数

同一个 Tensor Shape 在不同 GPU、NPU、dtype、Layout 和融合策略下可能使用完全不同的 Tile。

### 11.2 同时检查全部资源

即使 SRAM 足够，也可能因为寄存器、线程数、Bank Conflict 或指令粒度而不可行或很慢。

### 11.3 联合评估 Tile 大小和并行度

大 Tile 提高复用，也可能降低 Occupancy、增加 Tail、增加同步并造成 spill。

### 11.4 把 Occupancy 作为性能变量

高 Occupancy 有助于隐藏延迟，但低 Occupancy 的大 Tile 可能拥有更好的复用和 ILP。

### 11.5 区分 MMA 高效条件与正确性条件

对齐决定主路径效率；尾部可以使用 mask、padding 或其他指令处理。

### 11.6 搜索完整 Schedule

完整搜索还应包括 Warp Tile、Instruction Tile、Thread Tile、Stage、Layout、Vector Width 和融合策略。

### 11.7 正确性验证先于性能排名

错误的边界处理、Barrier 或双 Buffer 复用可能产生偶发错误，不能进入性能排名。

## 12. 可直接使用的求解原则

面对一个新算子，可以按以下顺序处理：

1. 写出标量索引关系。
2. 区分平行维和 Reduction 维。
3. 从输出 Tile 反推最小输入区域。
4. 明确逻辑 Tile 与物理 Storage Shape。
5. 查询目标设备的线程、寄存器、片上存储和指令能力。
6. 建立 CTA、Warp、Instruction、Thread 多级 Tile。
7. 枚举 Stage、Layout、Vector Width 和融合策略。
8. 过滤硬件非法或边界不安全的候选。
9. 用 Cost Model 选出少量候选。
10. 先验证正确性，再在目标设备上 Autotune。
11. 按设备、Shape、dtype、Layout 和编译器版本缓存结果。
12. 为未覆盖的动态 Shape 保留正确的 Fallback。

## 13. 总结

Tiling Shape 的求解本质上是一个带约束的优化问题：

```text
maximize  Performance(schedule)

subject to
  数学索引关系正确
  边界访问安全
  线程数不超过硬件上限
  寄存器不超过硬件上限
  片上存储不超过硬件上限
  指令和对齐契约成立
  同步与 Buffer 生命周期正确
  数值误差满足要求
```

其中：

```text
算子语义决定 Tile 之间的关系；
硬件约束决定哪些候选可以执行；
性能模型决定哪些候选值得测试；
目标设备 Autotune 决定最终配置。
```

这四层共同构成 AI 编译器中的 Tiling Shape 推导与选择。

## 附录：本仓库中的实现参照

本文方法不依赖特定项目。本仓库中可以参考以下小型实现：

- [`TiledMatmulConfig`](../src/gac/targets/cuda/tiled_matmul.py#L59-L117)：CUDA shared-memory Tile、线程和容量约束；
- [`bevformer_tiled_candidates()`](../src/gac/targets/cuda/tiled_matmul.py#L338-L347)：静态 GEMM 候选集合；
- [`VortexBufferPlan`](../src/gac/targets/vortex/buffered_mlir.py#L19-L66)：Vortex FP32 双 Stage LMEM 计划；
- [`vortex.py`](../src/gac/targets/vortex/ffn.py#L1456-L1471)：Vortex Runtime occupancy 准入检查。

这些代码是具体后端示例，不代表本文中的完整通用 Tile 求解器已经全部实现。
