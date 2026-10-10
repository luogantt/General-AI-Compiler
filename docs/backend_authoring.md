# 后端编写指南

> 状态（2026-10-10）：当前已有 `backends/cuda.py` 的专用 SM 8.6 编译入口、`runtime.py` 的 CUDA Session、版本化 HXP、静态 arena 规划，以及 CUDA Session 的一次性 arena 分配和 allocation offset view。内部 Reshape/连续 Slice 已支持零拷贝，生产 FP32 Conv/MatMul 已分别接入 cuDNN 与 CuPy/cuBLAS，逐 launch CUDA Event Profiler 可输出按 op 聚合 JSON。MHA/CNN/ResNet50-FPN、BEVFormer feature/input preparation、三层 Encoder、六层 Decoder、六层 raw detection projection、cold/history image-to-BEV、历史 BEV rotate/shift、统一 cold/warm 五输入三输出 step、`step-pipeline` HXP 和 Runtime 状态提交/失败回滚均已实现。官方 `bevformer_tiny_epoch_24.pth` 的统一 step HXP 已在 Ubuntu 22.04、RTX 3080（SM 8.6）上完成 cold/warm 数值验收；融合前 876 节点经 39 组 Add+ReLU 融合后为 837 个执行节点，零拷贝后实际 703 次设备 dispatch，最终强制 CUDA 测试为 `Ran 287 tests in 18.881s ... OK (skipped=1)`。全生成 cold/warm 中位数分别为 `425.325073 / 426.032532 ms GPU` 与 `436.881591 / 437.718436 ms wall`；库加速 cold/warm 分别为 `57.624464 / 56.823263 ms GPU` 与 `69.403093 / 68.747440 ms wall`。warm 每层共享 `[motion prev, initial query]`，cold 每层动态构造 `[current query, current query]`。`bevformer_postprocess.py` 明确保留 CPU NMS-free Top-K/box denormalization 边界；桌面 RTX 3070 和真实 nuScenes 数据仍待验证。通用后端协议、注册表和 `sim-gpu` 尚未实现。本文描述通用后端的目标约定；当前具体能力见 README 与架构说明第 10 节。

当前 HXP 的 `KernelLaunch` 已显式保存 `implementation_candidates`（`cudnn`、`cupy-cublas`、`generated-cuda`）；Session 建立时选定实现并写入 `session.info`、`kernel_info` 和 profile，vendor 调用执行中失败不会逐节点自动切换实现。当前 HXP 与本版 codegen 绑定，优化前的 schema 1 包需重建。Profile JSON 逐 launch 记录实际 implementation，并同时按 op 与 implementation 聚合。生产 Tensor 计划为 `705,177,100` bytes；cuDNN 另有 8 MiB workspace budget，`713,565,708` bytes 是计划预算/准入值，不含 module、context、Event 和 vendor 临时分配。

## 1. 首版接口定位

v0.1 的后端是与编译器同进程运行的 Python 实现。它让新后端可以快速参与能力查询、代价估算和节点执行，适合研究算法与编写 Conformance Test。

它**不是稳定插件 ABI**：类名、方法参数和 Python 数据结构在原型阶段可以随包格式演进。不要把当前协议用于独立发布的闭源驱动，也不要假设不同 GAC 版本之间能够直接加载同一个 Python 后端。

后端作者应先理解两个彼此独立的角色：

| 角色 | 编译期职责 | 运行期职责 |
| --- | --- | --- |
| Compiler Backend | 能力与约束、代价估计、IR Legalization、代码生成、artifact 元数据 | 无 |
| Runtime Driver | 无 | 设备枚举、内存、复制、队列、artifact 加载、dispatch、event、profiling |

NumPy 原型可以用同一个 Python 实现承载两种职责，但接口仍应分开。这样编译器无须加载设备驱动，Runtime 也无须依赖 MLIR/LLVM。

## 2. 首版 Python 后端需要表达什么

一个原型后端至少要提供下列语义；协议定义尚待加入 `src/gac`，以下名称是提案：

1. 唯一、稳定的目标名称，例如 `numpy-cpu` 或 `sim-gpu`。
2. 对规范化节点的能力查询。未知算子必须明确返回不支持，不能默认接受。
3. 可解释的节点代价估计，供划分器比较计算与启动成本。
4. 节点执行入口，接收已经物化的输入 Tensor 和节点属性，返回与声明 Shape/DType 一致的输出。
5. 必要时提供跨设备传输成本或传输实现。

推荐把能力描述为数据，而不是在划分器中写后端名称判断：

```python
capability = {
    "supported": True,
    "dtypes": ["float32"],
    "max_rank": 4,
    "dynamic_shapes": False,
    "required_layout": "contiguous",
    "workspace_bytes": 0,
    "estimated_compute_us": 12.5,
    "estimated_launch_us": 3.0,
}
```

初版协议可以只使用其中一部分；额外字段展示长期方向。划分决策所需的信息应由后端返回，而不是让核心编译器了解每个设备的私有规则。

## 3. 正确性约定

后端实现必须满足以下约定：

- 只执行自己明确声明支持的算子、DType、Shape 和属性组合；
- 不修改只读输入或共享常量，除非算子语义明确允许原地更新；
- 输出数量、Shape 和 DType 与编译期推导一致；
- 对非法属性、内存不足和设备错误返回明确错误，不生成“看似成功”的空输出；
- 相同输入、常量和确定性配置产生可比较结果；
- `sim-gpu` 一类模拟实现必须在名称和文档中明确其模拟性质；
- Runtime 不可用时允许编译器拒绝目标或选择显式 fallback，不能悄悄把真实目标改成 CPU。
- 有意留在主机的阶段必须像 `bevformer_postprocess.py` 一样在 ABI、Trace 和文档中显式命名；显式 CPU Top-K/box decode 不能被包装成 CUDA 图成功或 GPU 端到端结果。

新增算子时，优先把参考语义放入公共测试，再分别验证各后端结果。浮点比较应声明容差，不能用放宽容差掩盖错误的 Shape、广播或归一化轴。

## 4. 代价与划分

能力查询和代价查询必须分开：支持某节点并不意味着把它放到该设备更快。至少考虑：

```text
node cost       = compute latency + launch latency
edge cost       = transfer latency + layout conversion
partition cost  = node costs + cut-edge costs + memory pressure
```

开发阶段可以返回静态估计，但应注明估计来源。真实后端应使用版本化设备 Profile 或离线 Benchmark 表，并把以下信息纳入 cache key：设备型号、驱动/固件版本、artifact 版本、Shape、DType、Layout 和关键编译选项。

如果后端要求整段子图编译，而不是逐节点执行，它应返回候选子图及边界约束。核心划分器决定是否接受候选，后端不能在内部偷偷吞掉未声明节点。

## 5. Artifact 与包集成

当前 `.hxp` 已保存目标、CUDA 源码、带 `implementation_candidates` 的 launch metadata、规范图、常量摘要和静态内存计划；加载时会校验并重建 CUDA Program。候选是包内显式契约，vendor library 不是 Runtime 的隐式替换。未来原生后端生成的二进制 artifact 还应附带：

- backend ID 和 artifact format/version；
- 目标 triple、ISA/compute capability、必需特性；
- 入口点、参数 ABI、workspace 和 alignment；
- 编译器版本、构建选项和内容 hash；
- 可选的兼容性字符串与重新编译提示；
- 来源节点和 Kernel 的 provenance 映射。

模型包可以同时保存多个 Shape/设备 variant。Runtime 只能加载通过兼容性检查的 variant；找不到匹配项时应回退到包中明确存在的兼容实现，或报告可操作错误。

## 6. 未来稳定 C ABI 方向

长期插件边界将借鉴 PJRT C API 和 ONNX Runtime Plugin EP 的可演进模式，但不会复制其源码。方向如下：

```c
typedef struct gac_api_version_t {
  uint32_t major;
  uint32_t minor;
} gac_api_version_t;

typedef struct gac_driver_api_t {
  size_t struct_size;
  gac_api_version_t api_version;
  const void* extension_chain;

  gac_status_t* (*enumerate_devices)(...);
  gac_status_t* (*create_device)(...);
  gac_status_t* (*allocate)(...);
  gac_status_t* (*copy)(...);
  gac_status_t* (*load_executable)(...);
  gac_status_t* (*dispatch)(...);
  gac_status_t* (*event_wait)(...);
} gac_driver_api_t;
```

这只是方向示意，不是已发布头文件。稳定 ABI 需要遵循：

- 所有结构体以 `struct_size` 开头，只在末尾追加字段；
- major 表示破坏性变化，minor 表示向后兼容扩展；
- 可选能力通过 capability bit 或 extension chain 协商；
- 对象使用不透明句柄和明确的 create/retain/release 所有权；
- 错误使用稳定状态对象，不跨边界抛 C++ 异常；
- 不传递 STL、`absl::Status`、MLIR C++ 类型或 Python 对象；
- Tensor 描述使用固定宽度整数、显式 Shape/Stride/DType/MemorySpace；
- 编译请求、Execution Plan 与诊断使用带 schema version 的序列化字节；
- 插件只导出少量固定 C 符号，其余符号隐藏；
- loader 在调用函数表前校验 ABI 版本、结构大小、目标和 artifact 兼容性。

Compiler Backend 也会使用独立函数表，提供 capability query、cost query、compile 和 artifact metadata；它不会获得 Runtime Driver 的私有对象。二者可以由同一厂商实现，但生命周期和版本必须独立。

## 7. 后端验收清单

合并一个后端前，应至少验证：

- 注册、重复注册和未知目标错误；
- 每个支持算子的正例与不支持组合的反例；
- Shape/DType/属性边界；
- 与 `numpy-cpu` 参考结果的数值对比；
- 设备切换产生的传输步骤和成本；
- 常量不被意外修改；
- 模型包保存、重新加载和版本拒绝；
- 确定性 Trace 中的 backend/node/provenance 字段；
- 后端失败时的显式错误或预先编译好的 fallback；
- 资源释放、重复 Session 和并发调用行为。

在真实异步设备加入后，还必须加入事件顺序、跨队列等待、输入生命周期、超时、设备丢失和内存耗尽测试。

## 8. 许可证与分发

独立编写后端不自动继承参考项目代码的许可证，但链接或复制第三方组件会产生相应义务。提交前需要列出所有源码、头文件、静态/动态库、生成代码和 SDK 的来源，并确认：

- 是否需要保留版权、SPDX、LICENSE 或 NOTICE；
- 是否允许重新分发预编译二进制；
- CUDA、cuDNN、ROCm 或 NPU SDK 是否要求用户自行安装；
- Kernel 模板、模型权重和调优数据库是否有独立条款。

当前 GAC 原型没有复制 IREE、TVM、ONNX Runtime、TileLang 或 XLA 源码。若后端将来引入这些组件，必须在变更中单独记录和审查。
