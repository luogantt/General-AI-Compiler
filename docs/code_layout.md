# 源码分层与编译流程

交互式入口：[架构图目录](index.html) → [V8：分层与十阶段源码](vortex-matmul-end-to-end-v8.html)。
V8 对应本次重构；V6 的旧路径、行号和代码摘录是固定历史快照。

本项目采用“通用图语义、通用编译分析、模型构图、目标实现、执行与验证”分层。
ONNX 与 `torch.export` 先导入 Python 表达的 Canonical Graph；目标后端再生成 MLIR。
因此，模型导入器不直接生成机器码，也不负责分配设备 Buffer。

```text
ONNX / torch.export / Canonical JSON
                  │
       frontends/ 通用模型导入
                  │
           core/ Graph 与算子语义
                  │
       compiler/ 通用 Pass、分区、内存分析
                  │
       选择一个目标并调用其编译入口
          ┌───────┴────────┐
    targets/vortex/   targets/cuda/
          │                │
    MLIR → LLVM IR       CUDA / PTX
          │
      VOLT → 机器码
          │
    Vortex host Runtime
```

`models/` 是另一条模型来源：把已经理解的 BEVFormer/UniAD 结构和权重构造成同一种 Graph。
它与 ONNX/`torch.export` 导入的结果汇合，不要求通用编译器识别某个模型名称。
Vortex 全图编译选择一个 Vortex 设备执行全部已合法化算子；这里的多个目标目录代表可选择的后端。

## 各目录负责什么

| 目录 | 职责 | 主要入口 |
|---|---|---|
| `src/gac/core/` | Graph、Node、TensorSpec、算子语义、异常 | `ir.py`、`ops.py` |
| `src/gac/compiler/` | 通用阶段契约、Shape 推导、图 Pass、分区、Buffer 生命周期分析 | `pipeline.py`、`passes.py`、`partition.py`、`memory.py` |
| `src/gac/frontends/` | 通用 JSON、ONNX、`torch.export` 导入 | `model_frontend.py` |
| `src/gac/models/common/` | 可复用 Transformer/FFN 构图 | `transformer.py`、`ffn.py` |
| `src/gac/models/bevformer/` | BEVFormer 构图、模型约束、历史状态协议和专用权重导入 | `step_pipeline.py`、`frontends/graph.py` |
| `src/gac/models/uniad/` | UniAD 的 FFN 边界、权重和数据包协议 | `contract.py`、`checkpoint.py` |
| `src/gac/targets/vortex/` | Vortex 合法化、计算模板、MLIR 生成、设备 ABI、Host 源码生成 | `compiler.py` |
| `src/gac/targets/cuda/` | CUDA/PTX 和既有 MLIR CUDA 编译路径 | `compiler.py`、`ptx.py`、`tiled_ffn.py` |
| `src/gac/targets/mlir.py` | 通用 Graph 到 Tensor/Linalg MLIR 的已有导出路径 | `compile_mlir` |
| `src/gac/toolchains/` | 外部 MLIR/LLVM 工具调用、版本与产物检查 | `mlir.py` |
| `src/gac/execution/` | Python 设备 Session、Dispatch 执行和性能采集 | `cuda.py`、`dispatch.py` |
| `src/gac/artifacts/` | 编译产物、权重包和已有 CUDA 产物的保存/读取 | `hxp.py`、`weights.py` |
| `src/gac/validation/` | NumPy 参考执行、golden 数据与误差比较 | `reference.py`、`golden.py` |
| `src/gac/workflows/` | 需要同时了解模型、编译器和目标的上层编排 | `bevformer.py`、`bevformer_partition.py` |
| `src/gac/tuning/` | 既有 CUDA 配置搜索和基准工具 | `cuda_ptx.py`、`cuda_cublas_ptx.py` |
| `src/gac/tools/` | 命令行参数、文件输入和具体入口选择 | `build_vortex_graph.py`、`cli.py` |

模型专用导入放在 `models/bevformer/frontends/`，通用框架导入放在顶层 `frontends/`。
历史状态包装器 `models/bevformer/runtime.py` 通过 Session 协议调用执行器，自身不导入 CUDA 或 Vortex 后端。
`workflows/bevformer.py` 保留已有的模型专用分析/异构规划 API；当前单 Vortex 全图入口是 `targets/vortex/compiler.py`。

## 十个阶段在哪里

共同阶段顺序由 `compiler/pipeline.py` 的 `COMPILER_STAGES`、`run_pipeline()` 定义。
这个模块只定义阶段契约及执行顺序，不导入 BEVFormer、CUDA 或 Vortex。
目标实现提供每个阶段的实际处理函数。

| 阶段 | 输入与结果 | Vortex 对应实现 |
|---|---|---|
| 1. 模型导入 | 文件/权重 → Graph 与公开输入输出 | `tools/build_vortex_graph.py` 调用 `frontends/` 或 `models/bevformer/frontends/` |
| 2. IR 转换 | 确定 Node、TensorSpec、坐标轴及 dtype | `core/ir.py`、`compiler/passes.py` |
| 3. 图优化 | 常量折叠、死代码消除、Shape 推导 | `compiler/passes.py`，由 `targets/vortex/compiler.py` 调用 |
| 4. 合法化 | 检查算子、属性、静态 Shape、精度及 RV32 索引边界 | `targets/vortex/compiler.py` |
| 5. 子图切分 | 合法节点组织为可执行区域和依赖 DAG | `targets/vortex/compiler.py`；通用分区工具在 `compiler/partition.py` |
| 6. 计算模板 | 选择逐元素、归约、卷积、采样等实现 | `targets/vortex/scalar_ops.py`、`spatial_ops.py` |
| 7. Tiling | 决定每线程输出和归约范围 | `targets/vortex/compiler.py` 的当前标量基线方案 |
| 8. Schedule | Dispatch 顺序及全局 Buffer 生命周期 | `compiler/memory.py`、`targets/vortex/compiler.py` |
| 9. Codegen | SCF/Arith/Math/LLVM → LLVM IR → VOLT | `targets/vortex/scalar_ir.py`、`device_mlir.py`、`toolchains/mlir.py` |
| 10. Runtime | 分配设备内存、上传数据、按依赖发射并读回输出 | `targets/vortex/host_runtime.py` 生成的 Host 程序 |

阶段报告保存在 `01-import.json` 至 `10-runtime.json` 等产物中。
报告中的模板、Tile 和 Buffer 字段应描述实际执行方案；不能仅修改元数据来宣称增加了 LMEM、双缓冲或 TCU 优化。

## Buffer 分为编译计划和实际存储

```text
compiler/memory.py
    张量大小 + 生产/最后消费时刻
                  ↓
    中间张量地址区间 + 可复用 arena 偏移
                  ↓
targets/vortex/host_runtime.py
    生成 Host 内存分配、地址绑定及 Dispatch 参数
                  ↓
Vortex Runtime
    真实设备 Buffer / 设备地址
```

`TensorSpec` 只保存形状、类型和布局；权重数值保存在 `Graph.constants`。
内存规划返回的是布局计划，Host 调用设备 Runtime 后才获得实际设备内存。

当前 Vortex 全图标量基线的中间张量放在设备全局内存 arena 中，按生命周期复用；
一个线程负责一个输出元素，归约在 FP32 寄存器中串行完成。
这条基线尚未把所有算子改成 LMEM 双缓冲、DXA 搬运或 TCU 模板。
已有的优化 FFN 实现保存在 `targets/vortex/ffn.py`、`buffered_mlir.py` 等模块，
不能据此把全图标量路径描述为已经启用这些优化。

## 新代码如何调用

通用模型编译使用 Canonical Graph 边界：

```python
from gac.frontends.model_frontend import load_model
from gac.targets.vortex.compiler import compile_vortex_graph

graph = load_model("model.onnx")
program = compile_vortex_graph(graph, mlir_opt="/path/to/mlir-opt")
```

BEVFormer 专用权重入口构图后也交给同一个目标编译器：

```python
from gac.models.bevformer.frontends.graph import load_bevformer_step_graph
from gac.targets.vortex.compiler import compile_vortex_graph

graph = load_bevformer_step_graph("model.gacweights")
program = compile_vortex_graph(graph)
```

`compile_vortex_graph()` 生成编译计划与源码；传入 `mlir_opt` 时会验证并执行 MLIR Lowering。
最终 VOLT 编译、链接和设备运行仍由生成产物的构建/执行流程完成。
Native 验证入口 `targets/vortex/validation.py` 把相同算子体编译为 Host 机器码，
用于检查数值正确性；它不能证明 SIMT、设备 ABI 或真实硬件延时。

## 依赖规则与兼容路径

允许的核心依赖方向是：

```text
core ← compiler ← frontends
  ↑        ↑
  └── models / targets

tools / workflows → 选择模型、目标和执行器
```

- `core/` 不导入任何上层模块。
- `compiler/` 只依赖 `core/` 和本层；通用编译分析不选择模型或设备。
- 通用 `frontends/` 不导入具体模型、目标或执行器。
- `models/` 不导入 `targets/`、`execution/` 或上层工作流。
- `models/common/` 不导入 BEVFormer 或 UniAD 模型模块。
- 具体模型包不互相导入，公共部分经 `models/common/` 复用。
- `targets/` 接收 Graph，不导入具体模型构图模块。
- `validation/` 只依赖 `core/` 和本层，NumPy 参考执行不加载设备执行器。
- 项目内部使用新的规范路径；旧路径仅用于兼容外部调用者。

例如，旧的 `gac.ir` 对应 `gac.core.ir`，旧的 `gac.backends.vortex_graph`
对应 `gac.targets.vortex.compiler`。统一映射在 `gac._compat.LEGACY_MODULE_ALIASES`；
兼容层按需加载同一个模块对象，不保留第二份实现。
单独 `import gac` 不会加载模型、目标或 Torch/CuPy/ONNX 等可选依赖。

`tests/test_architecture_layers.py` 用 AST 检查实际 import 边界，包括函数内导入、
`TYPE_CHECKING` 导入及字符串常量形式的动态导入，并在独立 Python 进程验证根包的延迟加载。
执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_architecture_layers.py -v
```

增加模型应主要扩展 `models/`；增加硬件应扩展 `targets/`。
涉及模型和目标共同决策的功能放在 `workflows/`，避免把具体模型策略写回通用编译层。

## 模型适配接口与复用

`models/api.py` 定义 `ModelAdapter` 协议和延迟加载注册表。每个适配器声明 `name`、`scope`，
并提供两个入口：`load(path, **options)` 校验文件后构图，`build(data, **options)` 从模型自己的
内存数据契约构图。两者都返回 `core.ir.Graph`，不选择设备、不生成机器码、不运行推理。

| 注册名称 | `load` 输入 | `build` 输入 | 当前覆盖范围 |
|---|---|---|---|
| `bevformer-step` | 数据格式 `.gacweights` | `artifacts.weights.LoadedWeights` | Backbone/FPN、运动补偿、完整 Encoder/Decoder、原始检测头和下一帧 BEV；Top-K/框解码为显式 Host 后处理 |
| `uniad-ffn` | 已有 UniAD FFN bundle | `models.uniad.artifact.UniADFFNBundle` | Stage-2 BEV Encoder 中一层 FFN，包含 Residual/LayerNorm；不是完整 UniAD 网络 |

```python
from gac.models.api import get_model_adapter, list_model_adapters, load_model_graph
from gac.targets.vortex.compiler import compile_vortex_graph

print(list_model_adapters())  # ('bevformer-step', 'uniad-ffn')

graph = load_model_graph("bevformer-step", "model.gacweights")
program = compile_vortex_graph(graph)

# 另一份独立编译任务：使用同一目标后端编译 UniAD 的现有 FFN 组件。
ffn_graph = get_model_adapter("uniad-ffn").load("uniad_encoder_ffn.bundle")
ffn_program = compile_vortex_graph(ffn_graph)
```

同样的选择已接到命令行；以下命令各自生成一份单设备编译产物：

```bash
PYTHONPATH=src python -m gac.tools.build_vortex_graph \
  --model model.gacweights --adapter bevformer-step --output build/bevformer-vortex
PYTHONPATH=src python -m gac.tools.build_vortex_graph \
  --model uniad_encoder_ffn.bundle --adapter uniad-ffn --output build/uniad-ffn-vortex
```

注册表查询不会导入模型实现或 NumPy；选取某个适配器也不会初始化 CUDA、Vortex 或设备 Runtime。
每个适配器只接受其明确声明的参数，错误参数会直接报错。
`load_model_graph()` 还会检查返回值确实是 Graph，并执行结构校验。

`models/common/ffn.py` 的 `TransformerFFNConfig` 和 `build_transformer_ffn_graph()` 提供共享的
MatMul → Bias → ReLU → MatMul → Bias → Residual → 可选 LayerNorm 构图。
公共配置显式接受 `tokens`、`embed_dim`、`hidden_dim`，不绑定某个模型的 BEV 网格或固定倍率。
BEVFormer 包装器负责其 `hidden_dim = 2 * embed_dim` 约束，并保留原节点来源和元数据；
UniAD 适配器使用自身 bundle 契约和节点来源，直接调用公共 FFN 构图器。

后续增加 VLA 或其他模型时，先实现真实的数据契约与 `ModelAdapter`，再注册工厂：

```python
from gac.models.api import register_model_adapter

register_model_adapter("my-model", MyModelAdapter)
```

`MyModelAdapter` 是新模型实际实现的类，构造函数不应初始化设备。注册时不会调用工厂，
只有 `get_model_adapter("my-model")` 才创建实例。新 Graph 继续经过同一套算子合法化、
Buffer 规划和目标 Lowering；当前没有注册未实现的 VLA 占位适配器。

## 本次重构验证（2026-10-15）

- Windows 全量回归共 684 项：657 项通过，27 项按可选依赖或设备环境跳过。
- 81 个旧模块路径均可导入，与规范路径共享同一个模块对象；旧 CLI 和 wheel 包内导入也已检查。
- 重构前后，固定小图和完整小尺寸 BEVFormer 的设备/Native MLIR、Dispatch、内存计划及 Pass 报告一致。
- 真实 LLVM 20 工具链执行了数值回归；完整小尺寸拓扑加算子探针覆盖 33 类算子、811 次 Dispatch，
  Native 与 Vortex SimX 的冷帧、热帧和历史状态探针均通过。未测量芯片延时或生产尺寸模型推理。
- 新 UniAD 适配器通过独立 NumPy 参考计算；其 Vortex 编译测试显式禁止导入 CUDA 和 BEVFormer 模型代码。
- 产物文本统一写入 LF，避免 Windows 换行转换破坏设备 MLIR 与 manifest 的 SHA-256 绑定。

回归命令：

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
PYTHONPATH=src python tools/validate_vortex_full_graph.py \
  --llvm-path "$LLVM_PATH" --vortex-build "$VORTEX_BUILD" \
  --all-ops --output /new/validation/directory
```
