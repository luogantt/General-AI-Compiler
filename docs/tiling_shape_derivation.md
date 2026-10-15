# AI 编译器中的 Tiling Shape 推导：法则、约束与批判性分析

本文分析一份关于 “Tiling Shape 怎么确定” 的回答，并给出更严格、可落地到编译器实现的版本。
结论先说：原回答的主线是正确的，已经覆盖了算子索引关系、片上存储、寄存器、线程组织、
MMA 粒度、访存、流水和 Autotune。但其中若干表述把“数学合法性”“硬件合法性”和
“性能偏好”混在了一起，也没有明确区分逻辑 Tile、物理存储 Tile 和多级执行 Tile。

本文同时说明当前 GAC 的 CUDA 与 Vortex 路径实际实现到了哪一步，避免把通用设计方法误写成
已经完成的工程能力。

## 1. 对原回答的总体评价

原回答最重要的判断是正确的：

> Tile Shape 不是由 Tensor Shape 通过一个固定公式唯一算出来的。
> 编译器先推导合法的数据区域关系，再结合硬件约束生成候选，最后通过性能模型或实测选择。

可以把问题拆成两层：

```text
第一层：合法性推导
Tensor Shape + Indexing Relation + Data Dependency
                         ↓
输入 Tile、输出 Tile、Reduction Tile 之间的关系

第二层：性能选择
合法 Tile 关系 + 硬件资源 + Schedule 参数
                         ↓
候选过滤 → Cost Model → Autotune → 最终配置
```

原回答中各观点的准确性如下。

| 观点 | 判断 | 更严格的表述 |
|---|---|---|
| Elementwise、Broadcast、Transpose 可以反推输入 Tile | 正确 | 对规则、静态的索引关系成立 |
| GEMM 使用 `BM/BN/BK` 描述分块 | 正确 | 它们分别对应输出 M、输出 N 和归约 K |
| Tile 必须满足 SRAM、寄存器和线程限制 | 正确 | 性能选择还应权衡期望驻留 CTA 数，而不只是绝对容量 |
| Tile 应匹配 Warp、向量或 MMA 粒度 | 基本正确 | 这是性能约束，不是数学正确性的必要条件 |
| Layout、合并访存和 Bank Conflict 必须一起考虑 | 正确 | 逻辑 Tile Shape 和物理 Storage Shape 可能不同 |
| Tile 越大，数据复用越高 | 有条件正确 | 同时会增加寄存器、SRAM、尾部浪费并降低 Occupancy |
| 最终通常需要 Cost Model 或 Autotune | 正确 | 实测前必须先经过数值正确性门 |
| Indexing Map 可以解决所有 Tile 传播 | 不完整 | Gather、Scatter、GridSample 等数据相关访问需要专门分析 |

## 2. 首先区分五种容易混淆的 Shape

讨论 Tile 时不能只说一个“分块大小”。成熟后端通常同时存在以下层次：

```text
Tensor Shape
    ↓
CTA / Workgroup Tile
    ↓
Warp / Subgroup Tile
    ↓
Instruction Tile（MMA / Vector）
    ↓
Thread Register Tile / Fragment
```

还要区分：

1. **逻辑 Tile Shape**：算法上需要计算的数据区域，例如 `C[BM, BN]`。
2. **物理 Storage Shape**：在 SRAM/LMEM 中实际分配的布局，可能因为 padding 变成
   `A[BM, BK + 1]`。
3. **Launch Shape**：CTA 数、每 CTA 线程数、Warp 数。
4. **Pipeline Shape**：同一个逻辑 Tile 有多少个并行 Stage。
5. **Instruction Shape**：一条 MMA 或 Vector 指令原生处理的形状。

所以，`BM=128, BN=128, BK=32` 只描述了一层逻辑分块，不能单独定义完整 Kernel。

## 3. 合法性推导：算子索引关系决定 Tile 如何传播

### 3.1 Elementwise

```text
C[i, j] = A[i, j] + B[i, j]
```

若输出 Tile 为：

```text
C Tile = [TM, TN]
```

则输入区域为：

```text
A Tile = [TM, TN]
B Tile = [TM, TN]
```

这里没有归约维，Tile 传播完全由一一对应的索引关系决定。

### 3.2 Broadcast

```text
C[i, j] = A[i, j] + B[j]
```

如果：

```text
C Tile = [TM, TN]
```

那么：

```text
A Tile = [TM, TN]
B Tile = [TN]
```

`B` 在 M 轴广播，所以不需要加载 `TM` 份独立数据。

### 3.3 Transpose

```text
B[j, i] = A[i, j]
```

当：

```text
B Tile = [TN, TM]
```

对应的输入区域是：

```text
A Tile = [TM, TN]
```

这是维度置换，但物理访存是否连续还取决于源 Layout 和后端是否物化 Transpose。

### 3.4 Reduction

```text
Y[i] = Σ X[i, k]
```

输出平行维与归约维要分开描述：

```text
Y Tile = [TM]
X Tile = [TM, TK]
```

`TK` 只是一次处理的 Reduction Chunk。为了得到最终 `Y Tile`，Kernel 必须遍历全部 K，
或者产生 partial reduction，再由后续步骤合并。

### 3.5 不规则访问

Affine indexing map 很适合 Elementwise、Transpose、GEMM 和规则 Conv，但不能完整描述：

- Gather / Scatter；
- GridSample；
- 稀疏索引；
- 数据相关的路由或 Top-K；
- 间接地址访问。

这些算子需要范围分析、索引上界、专用模板，或者直接成为子图切分边界。

## 4. GEMM 的严格 Tile 关系

设：

```text
C[M, N] = A[M, K] × B[K, N]
```

标量关系为：

```text
C[i, j] += A[i, k] × B[k, j]
```

选择输出 Tile：

```text
C Tile = BM × BN
```

一次 K Chunk 对应：

```text
A Tile = BM × BK
B Tile = BK × BN
```

图示如下：

```text
                       N
             ┌─────────────────┐
             │   C Tile        │
          M  │   BM × BN       │
             └─────────────────┘

             A Tile                  B Tile
             BM × BK                 BK × BN
        ┌──────────────┐        ┌──────────────┐
     BM │              │     BK │              │
        └──────────────┘        └──────────────┘
                BK                    BN

             C_tile += A_tile × B_tile
```

在 MLIR `linalg.matmul` 的三维循环空间 `(i, j, k)` 中，典型 indexing maps 是：

```text
A: (i, j, k) -> (i, k)
B: (i, j, k) -> (k, j)
C: (i, j, k) -> (i, j)
```

因此 `i→BM`、`j→BN`、`k→BK` 是由算子语义决定的；具体数值则不是。

## 5. 硬件约束：哪些候选是可执行的

### 5.1 SRAM / Shared Memory / LMEM

若 A、B 都缓存在片上存储，数据类型字节数为 `E`，Pipeline Stage 数为 `S`，基础容量为：

```text
tile_storage_bytes =
    S × E × (BM × BK + BK × BN)
```

真实分配通常还要加上：

```text
+ bank-conflict padding
+ alignment padding
+ epilogue temporary
+ optional C tile storage
+ 与 Tile 共用同一 LMEM 池的其他后端临时空间
```

如果 C partial sum 始终保存在寄存器中，就不应重复计入 LMEM。若 C 也要暂存在片上存储，则加：

```text
BM × BN × accumulator_bytes
```

容量硬约束首先要求至少能够驻留一个 CTA。如果性能策略希望每个 Core 同时驻留 `R` 个 CTA，
还应检查按硬件分配粒度向上取整后的用量：

```text
R × align_up(per_cta_lmem, allocation_granularity)
    <= available_lmem_per_core
```

`R=1` 完全合法，而且当大 Tile 带来的复用或 ILP 足够高时可能最快。是否为更多 CTA 保留空间是
Occupancy 与复用之间的性能权衡，不是“SRAM 一定不能占满”的硬规则；未分配空间本身也不是
额外的双 Buffer。

### 5.2 线程与 Warp 映射

简单的一线程一输出实现中：

```text
threads_per_cta = BM × BN
```

Tensor Core Kernel 通常不是这种映射。一个 Warp 会持有多个 MMA fragment，因此应进一步指定：

```text
CTA Tile
  ├── Warp Tile
  │     └── MMA Tile
  └── 每线程 Register Fragment
```

“4 个 Warp 计算 128×128”只能作为上层示意；还必须说明 4 个 Warp 怎样排列，以及每个
Warp Tile 如何拆成多条 MMA 指令。

### 5.3 寄存器压力

若每个线程持有 `RM × RN` 个 FP32 accumulator，最低累加寄存器字节数近似为：

```text
register_accumulator_bytes_per_thread ≈ RM × RN × 4
```

真实寄存器还包括地址、循环变量、加载 fragment 和 epilogue 临时值。Tile 增大后可能出现：

```text
Register/thread 上升
        ↓
Active Warp/CTA 下降
        ↓
延迟隐藏能力下降
        ↓
严重时发生 spill
```

在 CUDA 术语中，spill 到 “local memory” 往往实际落在设备显存中；它不是 Vortex 的片上
LMEM，二者不能混用。

### 5.4 MMA / Vector 粒度

如果硬件指令粒度为 `MMA_M × MMA_N × MMA_K`，通常希望：

```text
BM 是 MMA_M 的整数倍
BN 是 MMA_N 的整数倍
BK 是 MMA_K 的整数倍
```

这是高效映射的偏好，不是数学正确性的必要条件。非整除 Shape 可以通过以下方式处理：

- padding；
- predication / mask；
- 独立 tail kernel；
- 主区使用 MMA，边界使用标量或 Vector 路径。

因此应说“优先对齐指令粒度”，不应说“不整除就不能计算”。

### 5.5 合并访存与物理布局

同样的逻辑 Tile，可以有完全不同的访存效率：

```text
良好：lane0→A[0], lane1→A[1], lane2→A[2] ...
较差：lane0→A[0], lane1→A[128], lane2→A[256] ...
```

所以搜索变量不应只有 `BM/BN/BK`，还应包含：

- layout；
- vector width；
- global-to-shared copy mapping；
- transpose/swizzle；
- 每线程加载元素数。

### 5.6 Bank Conflict 与 Padding

逻辑 Tile 和物理存储 Tile 可以不同。例如逻辑上是：

```text
A Tile = 32 × 32
```

为了打破特定列访问产生的 Bank Conflict，物理 SRAM 中可能分配：

```text
A Storage = 32 × 33
```

是否需要 `+1` 取决于 Bank 数量、Bank 宽度、元素宽度和实际访问模式，不能把它当成固定规则。

### 5.7 Pipeline Stage

`num_stages` 与 Tile Shape 是联合变量：

```text
BM / BN / BK / num_stages
```

双 Buffer 可以让：

```text
计算 Stage 0
      与
预取 Stage 1
```

发生重叠，但会把 A/B Tile 的片上存储翻倍。更多 Stage 只有在搬运延迟、计算时间和异步拷贝
能力匹配时才有价值。

### 5.8 Tail 和边界

Tile 不整除 Tensor Shape 时，可以采用 guarded load、zero fill、masked store 或 tail kernel。

例如：

```text
M = 1000, BM = 128
ceil(1000 / 128) = 8
最后一个 Tile 有 104 行有效、24 行无效
```

不能只为完全整除而选择 `BM=125`。`128` 可能更符合线程或 MMA 粒度，即使存在少量 Tail。

## 6. 性能模型：合法不等于快

一个 GEMM K Tile 的计算量为：

```text
FLOPs = 2 × BM × BN × BK
```

如果只计算 A/B 从下一级存储搬入 Tile 的字节数：

```text
Bytes_AB = E × (BM × BK + BK × BN)
```

粗略 Arithmetic Intensity 为：

```text
AI_tile ≈
    2 × BM × BN × BK
    -------------------------------
    E × (BM × BK + BK × BN)
```

在这个只计算 A/B 搬运的理想模型中，`BK` 可以约掉：

```text
AI_tile ≈ 2 × BM × BN / (E × (BM + BN))
```

因此增大 `BK` 不会自动提高这个理想算术强度。`BK` 的实际价值主要体现在 DMA/指令粒度、
Barrier 与循环开销、Pipeline overlap、K Tail 和片上存储占用。

这个估计尚未包括：

- C 的最终写回或读改写；
- Bias、Residual、Scale、Zero Point；
- Cache 命中；
- 重复加载；
- 边界 mask；
- Barrier 与地址计算。

所以它适合做候选排序特征，不能代替设备实测。

## 7. Elementwise、Reduction 和 Conv 不能套用同一规则

### 7.1 Elementwise

Elementwise 通常没有强数据复用，搜索重点是：

```text
threads_per_cta
elements_per_thread
vector_width
memory coalescing
occupancy
```

因此“Tile=256/512/1024”表述不够精确。它可能表示线程数，也可能表示每 CTA 处理的元素数。
例如：

```text
256 threads × 4 elements/thread = 1024 elements/CTA
```

### 7.2 Reduction

Reduction 需要同时选择输出 Tile 和 Reduction Chunk：

```text
TM × TK
```

还要决定：

- Warp shuffle 还是 Shared Memory tree reduction；
- 一个输出由一个线程、一个 Warp 还是一个 CTA 完成；
- 是否生成 partial reduction；
- accumulator dtype；
- 是否融合后续 normalization。

### 7.3 Convolution

Conv 输入 Tile 由输出索引关系和感受野共同决定。一般二维情况中：

```text
input_tile_h =
    (output_tile_h - 1) × stride_h
    + dilation_h × (kernel_h - 1)
    + 1

input_tile_w =
    (output_tile_w - 1) × stride_w
    + dilation_w × (kernel_w - 1)
    + 1
```

边界处还要结合 padding 裁剪有效输入区域。原回答中的：

```text
input_tile_h = output_tile_h + kernel_h - 1
```

只适用于 `stride=1`、`dilation=1` 的特殊情况。

## 8. 编译器中的推荐选择流程

一个实用的 GEMM Tile 选择器可以按以下顺序工作。

### 8.1 提取问题参数

```text
M, N, K
input dtype
accumulator dtype
layout / stride
epilogue
dynamic-shape profile
target device
```

### 8.2 生成结构化候选

候选不仅包含三维 Tile：

```text
Candidate = {
    block_tile:       (BM, BN, BK),
    warp_tile:        (WM, WN),
    instruction_tile: (IM, IN, IK),
    thread_tile:      (RM, RN),
    num_warps,
    num_stages,
    vector_width,
    layout,
    padding,
}
```

### 8.3 静态过滤

```python
for candidate in candidates:
    if candidate.threads_per_cta > device.max_threads_per_cta:
        continue

    if candidate.lmem_bytes > per_cta_lmem_budget:
        continue

    if candidate.registers_per_thread > device.hard_register_limit:
        continue

    if not candidate.satisfies_instruction_contract:
        continue

    if not device.can_reside_one_cta(candidate):
        continue

    legal.append(candidate)
```

### 8.4 Cost Model 排序

Cost Model 可以综合：

```text
预计 DRAM 字节数
预计 LMEM 复用率
预计指令数量
预计 occupancy
Barrier 数量
Tail 比例
Bank Conflict 风险
Pipeline overlap
```

期望 Occupancy 和为了达到该 Occupancy 设置的寄存器软预算属于排序因素，通常不应作为通用
硬门槛。低 Occupancy 的大 Tile 仍可能因为复用率或指令级并行度更高而成为最快候选。

### 8.5 真实设备 Autotune

最终选择应遵循：

```text
编译候选
   ↓
数值正确性门
   ↓
Warmup
   ↓
多轮设备计时
   ↓
使用中位数或稳健统计量比较
   ↓
按设备 + Shape + dtype + layout + epilogue 缓存结果
```

正确性必须先于性能。发生错误、NaN、越界或超过误差门限的候选不能进入排名。

## 9. 当前 GAC/Vortex 实现的真实边界

### 9.1 Vortex Direct-MLIR

当前 Vortex 双 Buffer 配置由
[`VortexBufferPlan`](../src/gac/backends/vortex_buffered_mlir.py#L19-L66) 表示：

```python
tile_m = 2
tile_n = 2
tile_k = 16
```

当前 Kernel 采用一线程一输出：

```text
threads_per_cta = BM × BN
local_row = thread_id / BN
local_col = thread_id % BN
```

FP32 双 Stage 的精确 LMEM 公式是：

```text
lmem_bytes = 2 × BK × (BM + BN) × 4
```

C partial sum 保留在每个输出线程的一个 FP32 SSA 累加值中。Kernel 中的 M/N/K 常量、
线程映射和 Stage 切换见
[`vortex_buffered_mlir.py`](../src/gac/backends/vortex_buffered_mlir.py#L122-L225)。

命令行默认调度仍是 `scalar`。只有显式选择
`--matmul-schedule dxa-double-buffer` 时，才会构造 `VortexBufferPlan`；此时
`--tile-m/--tile-n/--tile-k` 的默认值是 `2/2/16`，也可以手动覆盖。参数定义见
[`build_bevformer_vortex.py`](../src/gac/tools/build_bevformer_vortex.py#L80-L88)。Host 在启动前调用
`vx_check_occupancy(device, threads, lmem_bytes)`，见
[`vortex.py`](../src/gac/backends/vortex.py#L1456-L1471)。

因此当前状态是：

```text
手工 Tile
   ↓
编译期结构检查
   ↓
Runtime occupancy 检查
   ↓
执行
```

它还不是自动 Tile 推导或 Vortex Tile Autotuner。双缓冲模式下的默认 `2×2×16` 是保守的功能
配置，用于先验证 DXA、LMEM、双 Stage、Barrier、尾部和跨 K Tile 累加，不代表性能最优值。

已有人工实验也说明 Tile 必须按 Shape 选择：SimX 上合成生产宽度 FFN 的 `4×4×128` 相对
scalar 总周期减少约 `17.5%`，而小 Shape 的双缓冲版本反而更慢。该结果来自软件模拟器，
不能换算为硬件毫秒，也不构成通用最优 Tile。完整边界见
[DXA/LMEM 双缓冲验证](validation/vortex/buffered-mlir-2026-10-14/README.md)。

### 9.2 CUDA MLIR 实验路径

CUDA tiled MatMul 使用
[`TiledMatmulConfig`](../src/gac/mlir_tiled_matmul.py#L59-L117)，当前一线程一输出版本要求：

```text
block_x = tile_n
block_y = tile_m
threads = tile_m × tile_n <= 1024
shared_memory <= 48 KiB
```

仓库中定义了以下保守候选：

```text
(16, 16, 8)
(16, 16, 16)
(16, 16, 32)
(8, 32, 16)
(32, 8, 16)
```

见 [`bevformer_tiled_candidates()`](../src/gac/mlir_tiled_matmul.py#L338-L347)。这些候选提供了搜索空间，
但不等于当前 Vortex 路径已经自动选择了它们。

## 10. BEVFormer FFN 的具体计算

BEVFormer FC1 为：

```text
A[2500, 256] × B[256, 512] = C[2500, 512]
```

以下比较使用当前 Vortex 的“一线程一输出 + FP32 双 Stage”公式，只计算结构量，不代表实测性能：

| Tile `BM×BN×BK` | 线程/CTA | 双 Stage LMEM | CTA 数 | K Tile 数 |
|---|---:|---:|---:|---:|
| `2×2×16` | 4 | 512 B | 320,000 | 16 |
| `8×8×16` | 64 | 2,048 B | 20,032 | 16 |
| `16×16×16` | 256 | 4,096 B | 5,024 | 16 |

计算过程：

```text
CTA 数 = ceil(M / BM) × ceil(N / BN)
K Tile 数 = ceil(K / BK)
```

这个表只能说明不同 Tile 的并行粒度与资源需求。`16×16×16` 是否更快，还取决于：

- Vortex 每 CTA/每 Core 的线程能力；
- LMEM 容量和可同时驻留 CTA 数；
- DXA 每次搬运效率；
- Warp 数、Barrier 成本和调度；
- 标量 FMA 或 TCU 吞吐；
- 实际 RTL/FPGA 配置。

最终仍必须在目标 Vortex 配置上通过 correctness gate 和 cycle measurement 选择。

## 11. 对原回答最关键的六项修正

1. **MMA 对齐是性能偏好，不是数学合法性的硬条件。**
2. **SRAM 用量必须包含 Stage、padding 和同池临时值；目标 Occupancy 属于性能权衡。**
3. **Arithmetic Intensity 必须包含 dtype 字节数，完整模型还要考虑 C 和 epilogue 流量。**
4. **Warp Tile 必须继续拆解到 MMA fragment，不能停在“4 个 Warp 分一块 C”。**
5. **CUDA local memory 与 Vortex LMEM 是不同概念。**
6. **Affine indexing map 不能覆盖所有数据相关索引算子。**

此外，“Shape inference”和“Tile selection”应始终分开：

```text
Shape inference：求每个 Tensor 的逻辑 shape/dtype
Tile propagation：求算子各输入输出 Tile 的合法对应关系
Tile selection：从合法候选中选择目标硬件上的执行配置
```

## 12. 面试回答模板

### 30 秒版本

> Tiling Shape 的确定分成合法性推导和性能选择两个阶段。首先根据算子的 indexing relation、
> broadcast 和 reduction 关系，推导输出 Tile 对应的输入区域；然后结合片上存储、寄存器、
> 线程和 Warp 组织、MMA 或向量指令粒度、访存合并、Bank Conflict、流水 Stage 和 Tail 开销
> 生成候选。编译器过滤资源非法的候选，再通过 Cost Model 或目标设备 Autotune 选择最终配置。
> 因此 Tensor Shape 不会唯一决定 Tile Shape。

### 结合当前项目的版本

> 在我的 Vortex Direct-MLIR 路径中，GEMM 使用 `BM/BN/BK`，A Tile 是 `BM×BK`，B Tile 是
> `BK×BN`，C Tile 是 `BM×BN`。当前一线程负责一个 C 元素，FP32 双 Buffer 的 LMEM 用量是
> `2×BK×(BM+BN)×4` 字节，C 累加值留在寄存器。现阶段 Tile 由命令行传入，并由 Runtime
> 检查线程和 LMEM occupancy；启用双缓冲时默认 `2×2×16`，它只是功能基线。下一步是
> 建立候选生成、正确性门、性能模型和 Vortex 真机 Autotune。

## 13. 最终结论

原回答的主框架值得保留：

```text
Indexing Relation
      ↓
合法 Tile 传播
      ↓
硬件资源约束
      ↓
多级 Schedule 候选
      ↓
Cost Model / Autotune
      ↓
最终 Tile Shape
```

真正需要修正的是边界：哪些是 correctness 规则，哪些是 legality 规则，哪些只是 performance
heuristic；哪些是通用编译器方法，哪些已经在当前项目中实现。把这两组边界讲清楚，才能把
“懂 Tiling 概念”提升为“能够设计和实现后端 Tile 选择器”。
