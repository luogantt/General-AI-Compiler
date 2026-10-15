# BEVFormer 整图 Vortex 后端：V6 十层实现

分支：`lg/bevformer-vortex-full-ops`。

本实现按照 [V6 架构图](vortex-matmul-end-to-end-v6.html) 的十个模块，把现有完整
BEVFormer streaming Graph 编译到**单一 Vortex 设备**。网络计算包含 ResNet/FPN、
运动补偿、时序/空间 Attention、Encoder、Decoder、原始检测头和下一帧 BEV 状态。
算子先由软件 Kernel 实现，不要求硬件为每种算子提供专用指令。

当前实现覆盖项目规范 IR 的 33 类算子。完整生产 Profile 与真实 checkpoint 已完成
MLIR Lowering、VOLT 指令选择、ELF 链接和 `.vxbin` 封装；数值验证使用完整拓扑的
缩小尺寸合成模型。**生产尺寸真实输入、nuScenes 精度和 FPGA 延时尚未验证**。
网络输出为 `cls_logits`、`bbox_regression`、`next_bev`；现有 NMS-free TopK/筛选和
三维框包装属于 Host 后处理，没有被算作已实现的 Vortex 算子。

## 1. 十个模块的代码与交付物

| V6 模块 | 实际动作 | 代码入口 | Artifact |
|---|---|---|---|
| 1 模型导入 | ONNX、torch.export、Graph JSON；完整 BEVFormer `.gacweights` 权重绑定 | [`build_vortex_graph.main`](../src/gac/tools/build_vortex_graph.py)、[`load_bevformer_step_graph`](../src/gac/frontends/bevformer_graph.py) | `01-import.json`、外置权重 `.bin` |
| 2 IR 转换 | 使用带名字、Shape、DType、属性和来源的 Graph IR；推导所有值类型 | [`Graph`](../src/gac/ir.py)、[`ShapeInferencePass`](../src/gac/passes.py) | `02-graph.json` |
| 3 图优化 Pass | 常量折叠、死代码消除、重新推导 Shape | [`compile_vortex_graph`](../src/gac/backends/vortex_graph.py) | `03-passes.json` |
| 4 合法化与 Lowering | 检查算子、静态 Shape、精度和 RV32 字节/索引范围；拒绝不支持的节点 | `_validate_static_specs`、`_validate_boolean_operations`、`_kernel_body` | `04-capabilities.json` |
| 5 子图切分 | 当前每个节点形成一个可执行子图；建立数据依赖，目标全部为 `vortex-v3` | `compile_vortex_graph` 的 dispatch 构造 | `05-dispatch-dag.json` |
| 6 计算模板 | 选择逐元素、布局、GEMM、卷积、归约和多层采样模板 | `_template`、[`emit_scalar_op`](../src/gac/backends/vortex_scalar_ops.py)、[`emit_spatial_op`](../src/gac/backends/vortex_spatial_ops.py) | `06-templates.json` |
| 7 Tiling | 基线按线性输出元素分工，每线程一个输出；归约轴在寄存器中循环 | [`ScalarBuilder`](../src/gac/backends/vortex_scalar_ir.py) 与各算子模板 | `07-tiling.json` |
| 8 Schedule | 推导全图活跃区间，规划全局 arena/视图；按拓扑串联事件，保护复用地址 | [`plan_memory`](../src/gac/memory.py)、dispatch 的 `depends_on/wait_for_dispatch` | `08-schedule.json` |
| 9 Codegen | SCF/Arith/Math/LLVM 混合设备 IR 降为纯 LLVM Dialect，再经 LLVM IR/VOLT 生成机器码 | `_kernel_body`、`_entry`、`LOWERING_PASSES` | `kernel.mlir` → `.ll` → `.elf` → `.vxbin` |
| 10 Runtime | 一个设备、一个队列、常驻权重、全局 arena、原生 DType 输入输出、事件等待 | [`emit_graph_host`](../src/gac/backends/vortex_graph_runtime.py) | `main.cpp`、`10-runtime.json` |

第 2 层的模型语义 IR 当前仍是项目的 Python Graph 对象；可执行 MLIR 在第 9 层由
已选模板生成。它没有伪装成通用 Linalg-to-Vortex 自动编译器，也没有绕道生成设备
C++。`main.cpp` 只承担 Host 调度，设备 Kernel 的源头是 `kernel.mlir`。

```text
模型 + 权重
    ↓ 导入 / Shape / DType
Canonical Graph
    ↓ 常量折叠 / 死代码消除
合法化后的整图
    ↓ 单设备子图 / 模板 / Tiling / Buffer / Schedule
SCF + Arith + Math + LLVM Dialect
    ↓ mlir-opt（同一棵 VOLT LLVM 20.1.8）
纯 LLVM Dialect
    ↓ mlir-translate
LLVM IR
    ↓ VOLT RISC-V/Vortex 后端 + ld.lld
ELF → vxbin.py → kernel.vxbin
    ↓ vortex2 Host Runtime
SimX / RTL / FPGA
```

## 2. 计算模板与算子语义

| 模板 | 算子 | 关键处理 |
|---|---|---|
| SIMT 逐元素 | Add/Sub/Mul/Div、比较、Where、激活、Clamp、Cast | 多维广播；int32 保持整数运算；Where 条件是 int32 |
| 布局复制 | Reshape/Transpose/Concat/Expand/Slice/Identity | 显式行主序索引、负步长 Slice；Concat 只读取被选择的输入 |
| 输出元素 GEMM | MatMul、MatMulBiasGELU | batch 广播、向量输入、串行 K 轴累加 |
| 直接卷积 | Conv2D | NCHW/OIHW、groups、stride、dilation、四边 padding、可选 Bias |
| 归约 | ReduceSum/Mean、LayerNorm、Softmax、MaxPool2D | 任意支持轴；Softmax 先减最大值；LayerNorm 两遍归约 |
| 推理归一化 | BatchNorm | 按通道读取冻结的均值、方差、scale、bias |
| 采样 | GridSample2D | bilinear/nearest、零 padding、align_corners、最近邻 ties-to-even |
| 多层可变形采样 | MSDeformableAttention | batch/query/head/level/point/channel 显式索引、双线性插值和加权累加 |

FP16 张量保留 16 位存储，浮点累加和主要数学运算使用 FP32；写出时按输出 DType
舍入。bool 当前只用于布局/复制，布尔算术会明确拒绝。PTQ、INT8 Tensor Core、
完整混合精度搜索属于后续优化，不计入本次覆盖。

所有有 padding 的卷积、池化和采样都在**读内存前**判断边界。不能先越界 load，
再用 select 把数值置零。VOLT 的部分浮点舍入伪指令不支持分歧执行，采样 floor
改为范围保护后的截断与修正；没有关闭分支发散处理来绕过错误。

## 3. Buffer 的物理位置与复用规则

```text
Device DDR / Global Memory
├── 原生 DType 输入 Buffer
├── 常驻权重 Buffer
├── 中间值 arena
│   ├── 不重叠活跃区间复用相同 offset
│   └── 已证明连续的 Reshape/Slice 共享视图
└── 最终输出 Buffer
          ↓ 普通 global load（经硬件 Cache）
      线程寄存器：输入标量、索引、FP32 累加值
          ↓ 普通 global store
      下一个 Kernel 的输入 / 最终输出
```

当前通用软件模板 **LMEM=0、DXA=false、TCU=false**。这条执行路径依靠 global
Buffer 与寄存器完成计算，不需要先放入 LMEM 才能正确执行。把值放到 LMEM 的目的
是跨线程复用和减少 DDR 访问；它是后续计算模板的选择。

`plan_memory` 使用包含消费节点的活跃区间：某节点读输入的同时写输出时，两者不能
随意复用同一块内存。连续视图则延长底层所有者的活跃期。Runtime 把 arena offset
转换成真实设备地址，并让每次 dispatch 等待上一次完成；即使两个节点没有张量
依赖，也可能通过 arena 复用产生执行顺序约束。

现有固定 FFN 的 [DXA/LMEM 双缓冲实现](designs/vortex_buffered_mlir.md) 保留为单独
路径。本次通用后端尚未把它自动套到全图 MatMul；因此不会把全局 Buffer 复用称为
已经实现 LMEM ping-pong。

## 4. ABI、Codegen 和 Runtime

每个 Kernel 的参数是一个 `uint64_t` 地址数组，顺序为各输入地址、一个输出地址。
设备入口从 uniform `a0` 参数缓冲解包，并转换为 RV32 指针。维度和算子属性已经
静态特化到代码中。CTA 的 thread/block/block-dimension 来自 Vortex CSR。

模板函数标记为 internal/alwaysinline，使其在 VOLT 分歧处理前进入 Kernel。
生成的 LLVM annotation、`.vx_entry.*` 和 retention globals 复用现有直接 MLIR ABI。
Makefile 锁定 LLVM/VOLT 20.1.8 与 revision，并保留运行时使用的 x31。

`verify_artifact.py` 校验源文件、权重、可选输入与 golden 的大小及 SHA-256；它用于
防止构建产物错配。`verify_outputs.py` 独立比较设备写回值，Runtime 不读取 golden。
没有隐式 CPU/CUDA 算子回退。动态 Shape、零尺寸、未支持类型或算子会在编译时报错。

## 5. 使用与复现

```bash
# 完整 BEVFormer-tiny：真实 data-only checkpoint → 整图 → Vortex artifact
PYTHONPATH=src python3 tools/build_vortex_graph.py \
  --bevformer-weights build/checkpoints/bevformer_tiny.gacweights \
  --mlir-opt "$LLVM_PATH/bin/mlir-opt" \
  --output build/bevformer-vortex-full

# 通用模型也可走同一个入口；输入 NPZ 使用模型的完整张量名称
PYTHONPATH=src python3 tools/build_vortex_graph.py \
  --model model.onnx --inputs inputs.npz --golden \
  --mlir-opt "$LLVM_PATH/bin/mlir-opt" --output build/model-vortex

# 从已经配置的 Vortex build 目录执行，保持 CONFIGS 与驱动一致
make -C /absolute/artifact ROOT_DIR="$VORTEX_BUILD" LLVM_PATH="$LLVM_PATH" all
VORTEX_DRIVER=simx LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime" \
  /absolute/artifact/gac_bevformer_full_vortex \
  --data-dir /absolute/artifact --kernel /absolute/artifact/kernel.vxbin
python3 /absolute/artifact/verify_outputs.py --data-dir /absolute/artifact
```

未提供 `--inputs` 时只生成模型，不生成虚构输入。`--golden` 是显式的离线 NumPy
参考计算，生产尺寸可能很慢。`.pth` 先用已有 checkpoint 导出工具转 `.gacweights`；
`.pt2` 需要显式 `--trusted`。原始 ONNX 中不受支持的算子不会因为 Vortex 硬件可编程
就自动变成已实现算子。

```bash
# 缩小尺寸、保留完整网络拓扑；附加四个独立输出覆盖其余规范算子
PYTHONPATH=src python3 tools/validate_vortex_full_graph.py \
  --llvm-path "$LLVM_PATH" --vortex-build "$VORTEX_BUILD" \
  --all-ops --output /new/validation/directory

# 真实 MLIR→本机机器码的算子数值回归；不是设备推理回退
PYTHONPATH=src GAC_VORTEX_LLVM_BIN="$LLVM_PATH/bin" \
  python3 -m unittest tests.test_vortex_scalar_ops tests.test_vortex_spatial_ops \
  tests.test_vortex_graph tests.test_vortex_graph_runtime
```

详细实测边界、误差与日志见 [2026-10-15 验证记录](validation/vortex/full-graph-2026-10-15/README.md)。

## 6. 后续性能优化的入口

下一阶段在模板选择和 Schedule 层替换实现：GEMM/Conv 加入 LMEM tile 和搬运，
再加入双 Stage 及 arrive/wait；归约改为 warp/CTA 合作；最后在真实能力检查通过后
选择 TCU、融合和调优。每替换一个模板，都沿用现有 Tensor ABI、Buffer 规划和
数值验证。合并 Kernel 或改变并行 Schedule 后必须重新计算活跃区间，不能直接
沿用当前串行事件保证下的 arena offset。
