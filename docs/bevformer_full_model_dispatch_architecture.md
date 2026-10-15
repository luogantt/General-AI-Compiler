# BEVFormer 全模型异构编译与 Dispatch 架构

本文说明 GAC 如何把一个完整、静态 Profile 的 BEVFormer 图，变成可审查的异构
`Dispatch DAG`、全局 Buffer 计划和 Runtime 调度计划。它描述的是
`lg/bevformer-vortex-direct-mlir` 分支当前的编译器边界，也明确区分“已经完成的规划与
代码生成能力”和“尚未接通的物理异构执行”。

## 1. 先说明 FFN 子图是什么

FFN 是 Transformer 中的 **Feed-Forward Network（前馈网络）**。BEVFormer 的每个
Encoder/Decoder Transformer layer 在 Attention 之后都有一段 FFN。当前静态 FP32
模式识别的计算链是：

```text
query [B, Q, C]
   │
   ├──────────────────────────────────────────── residual ────────┐
   │                                                              │
   ▼                                                              │
MatMul(C→H) → Bias → ReLU → MatMul(H→C) → Bias → Add(residual) → LayerNorm
                                                                  │
                                                                  ▼
                                                           output [B, Q, C]
```

这里的“子图”是完整计算图中一组边界闭合的节点：它只有一个动态输入 `query`、一个
动态输出 `output`，六个权重/偏置常量由该区域内部使用，而且 FC1、激活、FC2 等中间值
不能被区域外节点旁路消费。融合前它通常有 7 个节点；`Bias + ReLU` 融合后有 6 个节点。

冻结的 BEVFormer-tiny Profile 包含 3 个 Encoder layer 和 6 个 Decoder layer，因此
完整静态图中预期有 **9 个严格 FFN 岛**。只有通过 shape、dtype、布局、残差连接、常量
形状和边界闭合检查的岛，才会成为当前 `vortex-v3` Direct-MLIR 候选；名字看起来像 FFN
但不满足契约的节点会留在 `cuda-sm86`，不会猜测或强制迁移。

## 2. 总体编译链

```text
BEVFormer config + checkpoint + 固定输入 Profile
                        │
                        ▼
             完整 Canonical Graph
             （全模型 MLIR 尚未实现）
       （CNN、Geometry、Attention、FFN、Detection、State）
                        │
                        ▼
                  算子支持分析
       （语义分类 + 严格 FFN pattern/ABI 检查）
                        │
                        ▼
                   自动子图切分
        ┌───────────┬────────────┬────────────┬───────────┐
        │ CNN 区域  │ Attention  │ FFN 区域   │ Utility   │
        │ cuda-sm86 │ cuda-sm86  │ vortex-v3* │ cuda-sm86 │
        └───────────┴────────────┴────────────┴───────────┘
                        │
                        ▼
       各区域 Tiling / Schedule / Buffer / Lowering
                        │
                        ▼
                 完整 Dispatch DAG
          （真实 SSA 数据边、依赖和拓扑顺序）
                        │
                        ▼
       权威全图 MemoryPlan + 折叠后的 global-boundary MemoryPlan
                        │
                        ▼
       RuntimeStep：setup phase / invoke phase
       （逻辑 transfer 与 dispatch；尚无物理异构执行）

* 当前只有严格静态 FP32 FFN 是 Vortex 候选。
```

这条链始终保留一份完整 Canonical Graph 作为语义真相；当前没有把整张图转换成一个
统一 MLIR module。子图不是重新导入的小模型，
而是从全图按原始 SSA value name 提取出来的编译单元。这样 shape、来源节点、跨区依赖和
Buffer 生命周期都能追溯回全图。

现有完整 Graph 可以直接进入总控 API：

```python
from gac.bevformer_compiler import (
    lower_bevformer_dispatch_sources,
    plan_bevformer_graph,
)

plan = plan_bevformer_graph(complete_bevformer_graph)
plan_json = plan.to_json()
artifacts = lower_bevformer_dispatch_sources(plan, output_directory="build/dispatches")
```

如果调用方持有已经按完整 streaming-step 合同整理好的 Encoder/Decoder 权重，也可以使用
`build_bevformer_whole_model_plan(...)` 同时完成完整建图和规划。`plan_json` 是离线执行计划；
`artifacts` 是逐 Dispatch 的 CUDA C 或 Tensor/Linalg MLIR 源码清单，不是已经装载到统一
CUDA/Vortex Session 的可执行文件。

已经通过完整 step builder 生成 HXP 时，可直接运行：

```bash
gac-bevformer-plan complete-step.hxp \
  --output build/complete-step-plan.json \
  --lowered-source-dir build/dispatches
```

命令会先验证 HXP 的完整性和全模型 metadata，再原子写入计划文件。`--cuda-only` 可保留相同
语义边界但把所有 Dispatch 放回 CUDA；它主要用于对照和调试。

## 3. 各层职责和代码映射

| 阶段 | 输入 | 核心工作 | 输出 | 代码 |
| --- | --- | --- | --- | --- |
| 全模型建图 | 配置、权重、静态 Profile | 组合 Backbone、Encoder、Decoder、检测与状态图 | 一个经过验证的 `Graph` | [`bevformer_step_pipeline.py`](../src/gac/bevformer_step_pipeline.py)、[`compose.py`](../src/gac/compose.py) |
| 支持分析 | 完整 `Graph` | 节点语义分类；发现并严格验证 FFN 岛 | 每节点 target/kind/reason 和 FFN region | [`bevformer_partition.py`](../src/gac/bevformer_partition.py) |
| 通用切分 | 每节点 `NodePlacement` | 按 target 和 semantic group 提取子图；建立边界 | `Dispatch`、`DispatchEdge`、`DispatchDAG` | [`partition.py`](../src/gac/partition.py) |
| 全局内存 | 完整源图 | 基于定义/最后使用做静态 arena 分配和安全复用 | 权威 `MemoryPlan` | [`memory.py`](../src/gac/memory.py) |
| Runtime 规划 | DAG + 全图 MemoryPlan | 投影各 Dispatch binding；构造折叠的 global-boundary MemoryPlan；显式生成逻辑 transfer | setup/invoke `RuntimeStep`、Buffer lifetime | [`dispatch_runtime.py`](../src/gac/dispatch_runtime.py) |
| BEVFormer 总控 | 完整源图 | 串起分析、切分、策略、MemoryPlan 和 Runtime 计划 | `BevFormerWholeModelPlan` | [`bevformer_compiler.py`](../src/gac/bevformer_compiler.py) |
| 离线计划工具 | 完整 streaming-step HXP | 校验包、生成计划 JSON、可选逐 Dispatch 导出源码 | 原子写入的计划和 `.cu`/`.mlir` 清单 | [`plan_bevformer.py`](../src/gac/tools/plan_bevformer.py) |
| Vortex FFN 后端 | 一个规范化 FFN Graph | Tensor/Linalg MLIR → LLVM Dialect → VOLT → ELF/vxbin | 后端可生成三个静态 FFN Kernel artifact；全模型 planner 本身只发出离线计划，按需停在 Tensor/Linalg MLIR 源码 | [`vortex.py`](../src/gac/backends/vortex.py)、[`vortex_device_mlir.py`](../src/gac/backends/vortex_device_mlir.py) |

### 3.1 支持分析不是只看算子名称

CNN、Attention 和 utility 节点可以先按 operation、节点路径和 provenance 做语义分类。
Vortex FFN 则必须做整段 pattern 检查：

1. 两个 `MatMul` 的 shape、FP32 dtype 和 contiguous layout 合法；
2. `fc1_weight/fc1_bias/fc2_weight/fc2_bias/norm_weight/norm_bias` 都是常量；
3. 激活必须是当前支持的 ReLU 路径；
4. residual 必须来自原始 `query`；
5. 最终必须是当前契约的 LayerNorm；
6. 中间结果没有图输出或区域外消费者；
7. 融合前 7 节点和融合后 6 节点都能规范化为同一个 FFN ABI。

因此，分析结果同时记录“支持”以及“不支持的具体原因”。失败的 FFN 候选回到 CUDA
完整后端，编译器不会生成一个可能算错的 Vortex dispatch。

### 3.2 切分同时看 target 和 semantic group

`cuda-sm86` 能执行 CNN、Attention 和 utility，但这些区域仍保留不同的 semantic group。
这样做的原因是它们的 Tiling、融合、workspace 和性能模型不同，不能仅因为目标设备相同
就合并为一个巨大的 CUDA 编译单元。

每个通过检查的 FFN 使用唯一的 `ffn:<region-id>` group，所以相邻 FFN 也不会被错误
合并。通用切分器随后保证：

- 全图每个节点恰好属于一个 Dispatch；
- 节点顺序和运算语义不变；
- 外部产生的值成为子图输入；
- 被区域外消费的值成为子图输出；
- 常量复制到对应子图，不被误当作普通跨区数据边；
- `DispatchEdge` 只由真实 SSA producer/consumer 关系生成。

## 4. 各类子图如何 Tiling、Schedule 和 Lowering

| 区域 | 主要分块轴 | 本地 Buffer | 当前 Lowering |
| --- | --- | --- | --- |
| CNN | `N/Cout/H/W/Cin` | cuDNN workspace，或 CUDA shared memory/register tile | `cuda-sm86`：cuDNN 候选或 generated CUDA/NVRTC |
| Attention | query/head/level/sampling-point/channel | Q/K/V、Softmax/MSDA reduction 的 CUDA 局部存储 | `cuda-sm86`：cuBLAS MatMul 候选和 generated CUDA |
| FFN | `M/N/K` | Vortex LMEM 中 A/B tile 的两组 stage；C partial sum 留在寄存器 | 严格候选：Tensor/Linalg MLIR → LLVM Dialect → VOLT/vxbin |
| Utility | 输出元素域、reduction 轴或 layout 轴 | proven view 可零拷贝；其余使用全局 arena 和 Kernel 局部存储 | `cuda-sm86` generated CUDA |

FFN 的“双 Buffer”是一个 Dispatch 内部的局部调度策略。Stage 0 被计算单元读取时，DXA
可以把下一个 K tile 搬到 Stage 1，之后交换角色。它不会占用全局 MemoryPlan 中预留的
“空白 Buffer”；LMEM/shared memory/register 的大小和复用由后端 lowering 决定。

```text
Global/DDR A、B
      │ DXA/DMA，按 M/N/K tile
      ▼
LMEM Stage 0 ── compute ──┐
LMEM Stage 1 ◀─ prefetch  │  交替使用
                          ▼
                    C 累加寄存器
                          │ K 维全部完成
                          ▼
                     Global/DDR C
```

## 5. Dispatch DAG 如何形成

Dispatch DAG 不是按 `CNN → Attention → FFN` 写死的一条链。BEVFormer 有残差、状态、
多分支和汇合，所以依赖必须从全图 SSA 边计算。例如：

```text
dispatch_0000  CNN(cuda) ─────┐
                              ▼
dispatch_0001  Geometry(cuda) → dispatch_0002 Attention(cuda)
                                      │
                                      ▼
                            dispatch_0003 FFN(vortex)
                                      │
                                      ▼
                            dispatch_0004 Utility(cuda)
```

每个 `DispatchEdge` 保存 value name、TensorSpec、producer、consumers，以及它是否为公共
输入或输出。拓扑 schedule 只能在依赖满足后提交 Dispatch；序列化后重新加载也必须再次
验证节点覆盖、边界、依赖和顺序。

## 6. 全局 Buffer 规划

当前设计有两份互补的 `MemoryPlan`，其中只有完整源图计划是 allocation 语义权威：

1. **全图 `MemoryPlan`**：直接对完整 Canonical Graph 运行一次 `plan_memory`。每个
   input、constant、中间值和 output 都有确定的 storage、size、alignment、offset、
   first/last use。各 Dispatch 只是引用这些 allocation，不各算一套互相冲突的地址。
2. **折叠后的 global-boundary `MemoryPlan`**：把每个 Dispatch 折叠为一个节点，只保留
   公共输入输出、常量和跨 Dispatch value，并按 `RuntimeStep` 坐标记录这些 Buffer 的
   lifetime 与所在 target。它是独立、可校验的第二份计划，不会替代源图 allocation。
   Dispatch 私有值可以进一步进入 register、CUDA shared memory 或 Vortex LMEM。

arena 复用的条件是两个逻辑值的生命周期不重叠且 alignment/size 合法。它不等于两个
设备已经共享同一个物理地址；CUDA pointer、Vortex device address 和 host buffer 的映射
仍由物理 Runtime 负责。

## 7. Runtime 调度和跨设备 Adapter

Runtime 计划的 `RuntimeStep` 有 `transfer` 和 `dispatch` 两种 kind，并分为两个 phase：

- `setup`：只包含常量从 host 到目标设备的逻辑 `transfer`，可由物理 Runtime 跨调用缓存；
- `invoke`：包含输入搬运、跨 target 搬运、`dispatch` 提交，以及最终输出回到 host。

这里的 `transfer` 是执行计划中的逻辑要求，不代表 DMA、host staging、事件同步或共享地址
已经实现。`output` 也不是第三种 `RuntimeStep` kind；它表现为 `invoke` phase 中回到 host 的
最后一次 `transfer`。

一个 `cuda-sm86 → vortex-v3` 的边界在计划里必须显示为：

```text
CUDA producer dispatch
        │ value = encoder.layer0.ffn.query
        ▼
explicit transfer(cuda-sm86 → vortex-v3)
        │
        ▼
Vortex FFN dispatch
```

参考 orchestrator 要求为每一种跨 target 方向注册
`(producer_target, consumer_target) -> TransferAdapter`。Adapter 可以执行真实 DMA、
host staging 或转换 opaque device handle；缺少 Adapter 时执行会明确失败，不能把 CUDA
指针当作 Vortex 地址直接传入。

## 8. 当前已经完成与尚未完成

已完成的编译器结构：

- 完整 Canonical Graph 保持单一语义来源；
- 支持分析、严格 FFN pattern 检查和确定性语义切分；
- CNN/Attention/FFN/utility Dispatch DAG；
- 一份权威全图静态 `MemoryPlan`，以及一份折叠后的 global-boundary `MemoryPlan`；
- `setup`/`invoke` phase 的 `RuntimeStep`、逻辑 transfer 和全局 Buffer lifetime；
- 严格 FFN 子图可规范化到现有 Vortex Direct-MLIR ABI；
- coverage 中分别记录可用的后端流水线和本次 planner 实际停留阶段；生成计划不会被表述为已经产出 `vxbin`；
- CUDA 保留完整静态图的后端覆盖。

当前物理边界：

- 冻结 Profile 中预期的 9 个严格 FFN 岛才是 `vortex-v3` 候选；CNN、Attention、
  geometry、detection、state 和其他 utility 仍在 `cuda-sm86`；
- 完整全模型 MLIR 尚未实现；当前只把严格 FFN 规范化为 MLIR，其他 Dispatch 走 CUDA
  lowering；
- Planner 和 reference orchestrator 已表达跨 target transfer，但 CUDA 与 Vortex 的真实
  统一 Session、DMA/host-staging 实现、事件同步和错误回滚尚未集成；
- 尚未证明 CUDA 与 Vortex 可共享物理内存，因此不能宣称 zero-copy；
- 全模型计划不等于“完整 BEVFormer 已在 Vortex 上运行”，单个 FFN 的 SimX 或硬件数字
  也不能当作完整模型延迟。

后续完成物理闭环的顺序应是：先实现 CUDA↔Vortex Buffer Adapter 和事件依赖，再把 9 个
FFN artifact 与 CUDA 子图一起装载进统一 Session，做逐 Dispatch 数值对齐，最后测量完整
cold/warm step 的端到端延迟和峰值显存。CNN/Attention 的 Vortex lowering 可以在这个
运行闭环稳定后逐类替换，Dispatch DAG 和全局 value identity 不需要重写。

## 9. 验收条件

重构不能只看“能生成 JSON”。至少应检查：

1. 所有源节点恰好被覆盖一次，顺序和节点语义未变；
2. 冻结完整图识别 3 个 Encoder FFN 和 6 个 Decoder FFN，并拒绝破坏 residual、shape、
   dtype 或边界闭合的伪 FFN；
3. 每个 Vortex Dispatch 能抽取成现有 backend 接受的规范 FFN Graph；
4. DAG 每条边都对应一个真实 SSA value，拓扑依赖无环且稳定；
5. 全图 MemoryPlan 覆盖所有值，跨区 value 在各 Dispatch 中引用同一逻辑 allocation；
6. 不提供 CUDA↔Vortex Adapter 时，reference runtime 必须失败；提供 Adapter 后才可执行；
7. 最终物理 Runtime 需要逐层/逐 Dispatch 与 CUDA reference 对齐，再报告端到端性能。
