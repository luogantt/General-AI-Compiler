# 计算模板库

`GACComputeTemplates` 是可单独链接的 C++ 静态库。它接收已经验证的 MLIR Operation，
检查模板适用条件，生成执行计划，再按计划创建真实的 MLIR 操作和 Runtime 启动信息。
它不依赖 BEVFormer 的模型名称，也不负责导入 ONNX 或执行推理。

```text
ONNX → GAC MLIR → 图优化/融合 → 合法化/设备编译
                                  │
                    默认 linalg   │   显式 legacy
                         ↓        │        ↓
                Linalg 结构化 IR  │    模板 Registry
                + 同坐标融合      │    select：检查类型/shape/资源
                         ↓        │        ↓
                Linalg → SCF      │    checked Plan
                                  │     ├─ emit → SCF/Arith/NVVM/Vortex
                                  │     └─ annotate → launch/Buffer 参数
                                  └───────────────┬───────────────
                                                  ↓
                                            LLVM → PTX / VOLT
                                                  ↓
                                            Runtime 按 manifest 执行
```

当前六个优化模板来自已有设备路径，需要显式 `--regular-lowering=legacy`。
模板整理不会让默认 Linalg 自动获得 shared-memory tiling 或 Tensor Core。
Linalg、tensor.pad 和标量基线仍由设备编译层生成，并在 manifest 中分别标识。

## 目录与依赖

```text
native/
├── include/gac/DeviceOptions.h         设备配置，独立于编译 Pass
├── include/gac/templates/
│   ├── Registry.h                     统一 Selection / select / emit / annotate
│   ├── CudaMatmul.h                    具体模板的检查后计划
│   ├── CudaReduction.h
│   ├── CudaConvolution.h
│   ├── VortexMatmul.h
│   ├── VortexTcu.h
│   └── ScalarBuilder.h                共用数值/坐标构建器
├── templates/
│   ├── CMakeLists.txt                  GACComputeTemplates
│   ├── Registry.cpp                   实际注册表与 manifest 适配
│   ├── cuda/{Matmul,Reduction,Convolution}.cpp
│   └── vortex/{Matmul,Tcu}.cpp
└── lib/
    ├── DeviceLowering.cpp              分配、DAG、Kernel ABI、调用统一模板接口
    ├── fusion/GraphFusionPatterns.cpp  数学语义层融合
    ├── LinalgDevice.cpp               Linalg 同坐标生产者融合与循环映射
    ├── CudaLowering.cpp               底层线程 ID / Kernel ABI
    ├── VortexLowering.cpp             底层 CSR / LMEM / DXA / ABI
    └── {Cuda,Vortex}Pass.cpp           高层 Pass 包装，与硬件辅助函数分离

GACCompiler → GACComputeTemplates → GACTargetSupport → MLIR/LLVM
```

库之间按箭头单向依赖。`gac-template-list` 只链接模板库，输出真实注册表的 JSON，
不需要 ONNX 模型，也不链接 ONNX 导入器。正常构建后执行：

```bash
cmake --build <build-dir> --target gac-compile gac-template-list
<build-dir>/templates/gac-template-list
```

## 已注册模板

| 稳定 ID | 算子范围 | 数据与 Buffer | 选择参数 |
|---|---|---|---|
| `cuda.matmul.shared` | MatMul、MatMul+Bias、MatMul+Bias+ReLU | FP32；A/B 存 static shared memory；每线程寄存器小块 | `--cuda-matmul-schedule=shared-tiled` 或 `auto` |
| `cuda.reduction.row` | Softmax、LayerNorm | FP32；一 CTA 协作一行；shared scratch | `--cuda-reduction-schedule=cooperative` |
| `cuda.convolution.channels` | Conv、Conv+ReLU、DCNv2 | FP32；同一输入采样复用到多个输出通道；累加保存在寄存器 | `--cuda-conv-channel-tile=2/4/8` |
| `vortex.matmul.lmem` | MatMul 及其 Bias/ReLU 变体 | FP32；单级 LMEM A/B tile；寄存器累加 | `--vortex-matmul-schedule=lmem-tiled` |
| `vortex.matmul.dxa` | 右矩阵为共享 rank-2 的 MatMul 及变体 | FP32；两级 LMEM Buffer + DXA 搬运 | `--vortex-matmul-schedule=dxa-double-buffer` |
| `vortex.matmul.tcu` | MatMul 及其 Bias/ReLU 变体 | TF32×3、FP32 累加；寄存器 WMMA fragment；不使用 LMEM/DXA | `--vortex-matmul-schedule=tcu` |

`auto` 是已有形状启发式，不是额外模板，也不是实测 autotuning。
最终适用性由 planner 检查静态形状、数据类型、广播、tile、线程数和内存上限。
所有模板都不会扩大当前算子的支持范围：FlashAttention、稀疏卷积不会因为注册表存在而自动实现。

## 一次编译中如何使用

```cpp
templates::Selection selection;
if (failed(templates::select(op, options, selection)))
  return failure();                         // 非法配置：明确失败

if (selection.hasTemplate()) {
  // collective 模板全 CTA 进入；边界保护由模板内部实现。
  if (failed(templates::emit(builder, op, pointers, selection)))
    return failure();
} else {
  // DeviceLowering 在同一目标上使用 Linalg / pad / scalar 基线。
}
templates::annotate(dispatch, selection, "gac.scalar");
```

三个结果必须区分：不匹配此算子；匹配但资源/精度不支持并记录回退原因；配置非法而编译失败。
已知模板使用类型化 Plan，不通过 Python 字典或字符串拼接来实现算术。

每个 dispatch 新增 `compute_template`：稳定 ID、选择策略、是否需要线程集体执行，
以及优化模板的算术精度和存储方式。原有 `cuda_schedule`、`vortex` 字段保留，兼容 Runtime。

**Buffer 的数量、大小和使用方式由计划决定。** CUDA 的 `shared_memory_bytes` 是
启动时动态共享内存，现有模板仍为 0；实际静态大小记在
`cuda_schedule.static_shared_memory_bytes`。Vortex 的 `lmem_bytes` 是每 CTA 的空间，
`stages` 对 LMEM/DXA/TCU 分别为 1/2/0。TCU 不能因为有矩阵单元就记为双缓冲。
图级全局激活内存与生命周期复用仍属于 `DeviceLowering.cpp`，不是每个模板私自分配。

## 新增模板的步骤

1. 明确算子语义：轴、布局、广播、精度、边界和输入输出指针顺序。
2. 在对应目标目录实现 planner 和 emitter；planner 不修改 IR，先检查资源合法性。
3. 增加类型化 Plan，在 `Registry.cpp` 注册算子范围、选择条件和稳定 ID。
4. 在统一 `emit` / `annotate` 中接入该 Plan，由同一计划产生机器代码和启动信息。
5. 添加 CMake 源文件，并验证正确性、尾块、资源回退、同步安全和 launch 信息。
6. 独立测量硬件收益。注册成功、编译成功都不能代表性能更快。

纯数学融合规则放在 [GraphFusionPatterns.cpp](../lib/fusion/GraphFusionPatterns.cpp)，
不混进模板选择器。融合范围与本轮验证见 [融合说明](../../docs/compiler-fusion.md)。
本轮构建、IR 对比与数值验收见 [验证记录](../../docs/validation/templates/2026-10-21/README.md)。
