# General AI Compiler（GAC）

Vortex 原生编译器存档：`bevformer-base-mlir-vortex`，分支
`lg/bevformer-vortex-full-ops`。源码恢复来源与验证范围见
[Vortex 版本存档记录](docs/validation/vortex/native-mlir-snapshot-2026-10-17/README.md)。

**新的 ONNX → Vortex 主入口：[原生 C++ MLIR 编译器](native/README.md)。**
它直接把 ONNX ModelProto 导入注册的 `gac` Operations，使用真实 MLIR Pass 做
shape 推导、常量折叠、CSE、条件融合和布局验证，再由 C++ Lowering/Buffer 规划生成
Vortex 设备代码；小模型已完成 `VOLT → ELF → vxbin → C++ Runtime → SimX` 数值闭环。
完整 BEVFormer-tiny 固定 cold ONNX 已通过严格原生编译，生成 **1,548 个 dispatch**；
图像激活仍由生成的 Kernel 在运行时计算，常量几何在 MLIR Pass 中求值。完整
Vortex ELF/vxbin 已生成并核对全部 1,548 个入口；主机生成的机器码已完整执行
整网，三个输出与 PyTorch 参考对齐。完整 SimX 执行仍待验收，尚无物理设备 FPS，详见
[整网验证记录](docs/validation/vortex/native-mlir-bevformer-full-2026-10-16/README.md)。
官方 R101-DCNv2 / 200×200 / 四级 FPN 的适配单独记录在
[BEVFormer-base 原生 MLIR 说明](native/README.bevformer-base.md)：完整 base 已生成
**3,224 个 dispatch** 的 native 主机共享库和 Vortex ELF/vxbin，并核对全部设备
入口。base 完整 CPU 数值验证已通过：3,224 个 dispatch 全部执行，三个输出与
PyTorch 参考对齐，见[最终验证记录](docs/validation/vortex/native-mlir-bevformer-base-2026-10-16/README.md)。
尚无整网 Vortex 执行或 30 FPS 结论。
旧 Python Graph 路径作为兼容实现保留；以下历史能力
与新原生路径的支持范围分别记录，不能混算。

`lg/bevformer-vortex-full-ops` 分支新增[按照 V6 十层架构实现的整图 Vortex 后端](docs/bevformer_vortex_full_graph.md)：现有 BEVFormer streaming Graph 的 33 类规范算子均可生成独立软件 MLIR Kernel，统一在一台 Vortex 上执行。包含全局 Buffer 复用、事件调度和原生 DType Runtime；完整拓扑的小尺寸模型已通过真实 MLIR 编译及 SimX 数值验证。真实 checkpoint 的生产 Profile 已编译为 `.vxbin`，生产尺寸输入的精度与速度仍待验证。

> 面向边缘推理的 AI 编译器与 Runtime 原型；当前生产闭环是固定 Profile 的 BEVFormer-tiny raw step 和 CUDA SM86，同时提供 tensor-level MLIR emitter、cuBLAS 混合 FFN、全 PTX FFN，以及两次 cuBLAS MatMul 配合两个 MLIR→PTX 融合 Kernel 的 `mlir-cublas-ptx-sm86` 生产候选。

GAC 的 CUDA 路径把模型配置、权重和静态计算图转换为可校验的 HXP 包，在 Runtime 中完成设备检查、显存规划、实现选择、NVRTC 编译和 CUDA 执行。逐元素、布局、Attention 与 BEV 专用算子由项目生成 CUDA；生产尺寸 FP32 Conv 和 MatMul 可分别选择 cuDNN 与 CuPy/cuBLAS。MLIR 侧有五条边界不同的路径：`mlir-linalg` 生成标准 Tensor/Linalg MLIR；`mlir-cuda-sm86` 用两次 cuBLAS MatMul 和两个 NVRTC 融合 Kernel 执行固定 FFN；`mlir-ptx-sm86` 把同一生产形状 FFN 的三个 `gpu.func` 经官方 LLVM 24 `mlir-opt` 降到 SM86 PTX；`mlir-cublas-ptx-sm86` 保留两次 CuPy/cuBLAS MatMul，并把两个融合阶段改为 MLIR PTX；`mlir-tiled-ffn-sm86` 用两个单缓冲 tiled MatMul Kernel 和一个最终融合 PTX Kernel 执行三个 dispatch。

```text
JSON graph / ONNX / torch.export / BEVFormer config + checkpoint
                    │
                    ▼
      Frontend → Canonical Graph
                    │  shape / dtype / layout validation
                    ▼
   graph passes → fusion → static memory plan
                    │
                    ▼
 Generated CUDA + vendor candidates + metadata
                    │
                    ▼
        versioned, validated HXP package
                    │
                    ▼
 CudaSession → NVRTC / cuDNN / CuPy-cuBLAS → raw outputs
                    └────────────────────────→ trace / profile
```

当前 GPU 路径不会在 CUDA 缺失时静默改用 CPU。HXP 中的 `implementation_candidates` 供 Session 建立时选择；vendor 调用执行中失败不会逐节点自动切回 generated CUDA。NumPy 是显式选择的参考后端。

## 通用模型导入

GAC 现在提供统一、延迟加载的模型入口：`.json` 直接读取 Canonical Graph，`.onnx` 通过 ONNX checker、Shape Inference 和算子合法化进入同一张图；内存中的 `torch.export.ExportedProgram` 通过 FX/Graph Signature 区分用户输入、参数、Buffer 和常量，再把 ATen 节点合法化。ONNX 和 PyTorch 都是可选依赖，普通 `import gac`、JSON 编译和 CLI 帮助不会加载它们。

ONNX 导入覆盖当前 Canonical Op 能忠实表达的 MatMul/Gemm、逐元素计算、Conv2D、推理 BatchNorm、MaxPool2D、LayerNorm、Softmax、Reshape、Transpose、Concat、Expand、Slice、Reduce、GridSample 等路径。结构参数必须是编译期常量；LayerNorm 暂时严格限定为 FP32，避免在 IR 尚未记录 accumulator/stash precision 时改变 FP16 数值语义。未知算子在严格模式下一次性报告，也可以通过 `--allow-unsupported` 保留为带来源域的节点，用于支持度分析。导入成功表示模型语义已经进入 Canonical Graph；能否编译还取决于所选后端的算子、DType、Shape 和 Layout 能力。

```powershell
# ONNX → 可审查的 Canonical Graph JSON
python -m pip install -e ".[onnx]"
gac import model.onnx -o build/model.gac.json --report build/model.import.json

# 支持的 ONNX 图也可以直接进入现有编译入口
gac compile model.onnx -o build/model.hxp

# 未支持算子保留为 onnx domain 节点，只用于覆盖率/切分分析
gac import model.onnx -o build/model.analysis.json --allow-unsupported
```

`.pt2` 是序列化的 `torch.export` 产物，文件加载属于可信模型边界。CLI 要求显式 `--trusted-model`，并可在导入 PyTorch 前验证 SHA-256：

```powershell
python -m pip install -e ".[torch-export]"
gac import model.pt2 -o build/model.gac.json `
  --trusted-model `
  --sha256 <64位摘要> `
  --report build/model.import.json
```

Python 中已经持有 `ExportedProgram` 时无需再次反序列化：

```python
from gac.frontends.torch_export_frontend import import_exported_program

graph = import_exported_program(exported_program)
```

普通 `.pth`/`state_dict` 只有权重，不包含足以恢复任意模型的完整计算图，因此不作为通用模型输入。当前旧版 PyTorch/MMCV BEVFormer 也尚未直接导出为 ONNX/`torch.export`：它仍需要状态输入规范化、MMCV Deformable Attention 自定义算子注册和真实产物逐层对齐，所以 BEVFormer 的专用静态构图路径继续保留。

`mlir-cuda-sm86`、`mlir-ptx-sm86`、`mlir-cublas-ptx-sm86` 和 `mlir-tiled-ffn-sm86` 都不会静默回退到 NumPy 或旧 CUDA 图后端。它们从导入后的 Canonical Graph 建立计划；HXP v1 不保存这类产物，因此不能把 `.hxp` 作为输入。模型来源可以是 JSON 或已完全合法化的 ONNX，但后端仍只接受各自明确支持的 FFN pattern。

## 当前已完成

截至 2026-10-15，GAC 已跑通单 GPU 的模型导入、图优化、静态内存规划、CUDA 代码生成、vendor library 选择、HXP 打包和真机执行闭环。MLIR M0/M1 已完成 tensor IR emitter 和外部 verifier；生产形状 BEVFormer FFN 已完成首版 GPU Dialect→NVVM→PTX 的真机正确性与性能基线。本轮又完成 `mlir-cublas-ptx-sm86`、单缓冲三 dispatch 的 `mlir-tiled-ffn-sm86`、真实输入 FP16/TF32 MMA、真实 `cp.async` 双缓冲 MatMul 功能候选，以及三候选 autotuning；三个完整 FFN Session 已在固定提交 `5794d2c` 上完成正式双顺序报告。完整 BEVFormer 仍未全部进入 MLIR 路径。

| 层次 | 已完成能力 |
| --- | --- |
| 编译器核心 | JSON/ONNX/`torch.export` 统一导入、Canonical Graph、Shape/DType 推导、静态连续布局检查、常量折叠、死代码消除、MatMul/Add/GELU 融合、安全 Add+ReLU 融合 |
| MLIR M0/M1 | 确定性 textual MLIR、标准注册方言、外部 `mlir-opt` verifier 接口；覆盖手工构造的 BEVFormer-like tiny FFN JSON smoke 所需的 MatMul、广播 Add、ReLU、LayerNorm，以及 GELU、Identity 和融合 AddReLU/MatMulBiasGELU |
| MLIR CUDA FFN 切片 | `mlir-cuda-sm86` 严格匹配一个静态 FP32 BEVFormer FFN：两次 CuPy/cuBLAS MatMul、`Bias+ReLU` 融合 Kernel、`Bias+Residual+LayerNorm` 融合 Kernel；生成显式 runtime-call plan 并在 Compute Capability 8.6 上执行 |
| MLIR PTX FFN 切片 | `mlir-ptx-sm86` 严格匹配生产形状 `[1,2500,256]→512→256` FP32 FFN；生成三个 GPU Dialect kernel，调用官方 `mlir-opt` 降到 NVVM/PTX 7.1，校验产物后由 Runtime 直接加载 PTX；没有 cuBLAS/NVRTC fallback |
| MLIR cuBLAS+PTX 生产候选 | `mlir-cublas-ptx-sm86` 使用固定四调用计划：cuBLAS FC1、MLIR PTX `Bias+ReLU`、cuBLAS FC2、MLIR PTX `Bias+Residual+LayerNorm`；不提交 CUDA C 给 NVRTC；RTX 3070 Ti Laptop 正式 device-resident 中位数为 `221.915998～223.343678 μs/FFN` |
| MLIR tiled FFN 功能候选 | `mlir-tiled-ffn-sm86` 用单缓冲 16×16×16 FP32 tiled FC1+Bias+ReLU、tiled FC2 和 MLIR PTX 最终融合组成三个 dispatch；已有完整 Session、无权重 artifact 和三候选 autotune；RTX 3070 Ti Laptop 正式结果为 `0.892597122～0.893539200 ms/FFN` |
| MLIR Kernel 优化基础设施 | FP16 `16×8×32` 与 TF32 `16×8×16` 真实输入 MMA 已在 RTX 3070 Ti Laptop 功能运行；独立 `16×16×64` FP32 `cp.async` 双缓冲 MatMul 候选也已 exact 运行；两者均未接入生产 FFN。autotuner 已实现正确性门、median 选择和绑定设备/图/源码/候选空间的原子缓存 |
| CUDA SM86 后端 | 基础 Tensor/MLP、MHA、CNN、MSDeformableAttention、BEV geometry、Spatial/Temporal Attention、运动补偿及 step 边界所需 Cast/Where；不支持的类型、属性和布局会明确拒绝 |
| HXP 与 Runtime | 确定性打包、摘要校验、安全加载、源码与 launch metadata 复核、静态 arena、驻留权重、单 stream 执行、状态提交/回滚、Trace 和可选逐 launch CUDA Event Profile |
| BEVFormer-tiny | 官方 checkpoint 严格映射；ResNet-50 + 单层 FPN；feature/input preparation；三层 Encoder；六层 Decoder；六层 raw detection projection；cold/history image-to-BEV；统一 cold/warm 五输入三输出 step |
| 时序状态 | 历史 BEV rotate/shift、`has_prev_bev` 选择、scene/CAN-bus delta 管理；只有完整调用和三个输出校验成功后才提交 `next_bev` |
| 主机后处理 | 显式 CPU `decode_nms_free`、Top-K 300、box denormalization 与范围过滤；该阶段位于 HXP/CUDA 图外 |

冻结的 BEVFormer-tiny Profile 为 6 相机、`480×800` 图像、`50×50` BEV、256 embedding。统一 step 输入为 `images`、`lidar2img`、`can_bus_delta`、`prev_bev`、`has_prev_bev`，输出最后一层 `cls_logits`、raw `bbox_regression` 和 FP16 `next_bev`。公开图像与状态存储边界使用 FP16，Transformer/MSDA 等主要计算仍以 FP32 为基线。

## BEVFormer 全模型 Dispatch 规划

完整 Canonical Graph（包括从完整 HXP 恢复的图）现在可以先做算子支持分析，再按 CNN、Attention、FFN 和 utility 语义区域自动切分，生成带真实 SSA 数据依赖的 Dispatch DAG。Runtime 规划同时保存两份互补的静态内存结果：完整源图的 `MemoryPlan` 是所有 value allocation 的语义权威；按 Dispatch 折叠后的 global-boundary `MemoryPlan` 只保留公共 ABI、常量和跨 Dispatch value，并给出 Runtime schedule 坐标下的 Buffer lifetime。冻结 Profile 中预期的 3 个 Encoder FFN 与 6 个 Decoder FFN，只有在满足严格静态 FP32 pattern 和边界契约时才是 `vortex-v3` Direct-MLIR 候选；CNN、Attention、geometry、detection、state 和其他 utility 仍使用 `cuda-sm86`。

[`bevformer_compiler.py`](src/gac/workflows/bevformer.py) 是这条全模型规划链的总控入口。当前 Runtime 计划已经生成 `setup`/`invoke` 两个 phase 的 `RuntimeStep`：常量上传属于 `setup`，输入、跨 target value、最终输出的逻辑搬运及 Dispatch 提交属于 `invoke`。这些 transfer 只描述必须发生的数据移动，并要求跨 target Runtime Adapter；物理 CUDA↔Vortex 统一 Session、DMA/host staging、事件同步和 zero-copy 尚未集成，因此这份离线计划不代表完整 BEVFormer 已经在 Vortex 上执行。分层设计、全局 Buffer 语义、Dispatch DAG 和验收条件见 [BEVFormer 全模型异构编译与 Dispatch 架构](docs/bevformer_full_model_dispatch_architecture.md)。

对已经生成并校验过的完整 streaming-step HXP，可以执行 `gac-bevformer-plan step.hxp -o plan.json --lowered-source-dir build/dispatches`，生成确定性的异构计划 JSON，并按 Dispatch 输出可审查的 CUDA C 或 Tensor/Linalg MLIR 源码。这个命令只做离线规划和源码生成，不会启动物理 CUDA/Vortex Runtime。

## Vortex 3.0 FFN 垂直切片

`vortex-v3` 后端已接通一条独立于 CUDA/PTX 的 BEVFormer FFN 路径。两种设备 lowering 共用 Canonical Graph → 标准 Tensor/Linalg MLIR → 第一次外部 `mlir-opt --verify-each --canonicalize --cse` → `optimized.mlir` 严格解析 → 确定性 lowering 到标准 `func.call` Runtime ABI → 第二次 `mlir-opt` → `vortex.mlir` 严格解析。`cpp-template` compatibility 模式随后生成固定 Vortex C++ Kernel；`direct-mlir` 模式则直接生成 LLVM Dialect MLIR，经同一棵 VOLT LLVM 20.1.8 工具树生成机器码。`vortex.mlir` 使用静态 FP32 memref，保存三次调用的顺序与类型、launch size、LayerNorm epsilon 和六个外部权重 SHA-256；两种设备实现都由这份已经验证的 call plan 驱动生成。

两种模式当前都只覆盖 `MatMul → Bias+ReLU → MatMul → Bias+Residual → LayerNorm` FFN，不包含 ResNet/FPN、MSDeformableAttention、Encoder/Decoder、检测头或完整时序 step。上游 Vortex 3.0 没有 Vortex MLIR dialect 或通用 MLIR-to-Vortex kernel translator；这里的 direct lowering 是 GAC 针对这三个静态 FFN Kernel 实现的受限 LLVM Dialect 发射器，并保留 VOLT 做最终指令选择、寄存器分配和 ELF 生成。artifact manifest 采用精确的 17 个 payload 文件白名单；`make` 的 host、VOLT、`.vxbin` 以及所有 `run-*` 目标都会先执行 `verify_artifact.py`，同时拒绝非 RV32 build。代码边界、artifact 文件、本地生成和远端部署命令见 [BEVFormer FFN 到 Vortex 3.0 后端](docs/bevformer_vortex_backend.md)。

compatibility C++ bridge 已用官方 BEVFormer-tiny checkpoint 第 0 层真实权重跑通一行生产宽度 `[1,1,256] → 512 → 256`，SimX 对冻结 GAC/NumPy golden 的最大绝对误差为 `1.90734863e-6`；完整 2500-row 真实权重 artifact 已成功编译为 `.vxbin`。验证记录见 [Vortex BEVFormer FFN validation](docs/validation/vortex/bevformer-ffn-2026-10-13/README.md)。

`lg/bevformer-vortex-direct-mlir` 分支加入 `--device-lowering direct-mlir`：三个设备 Kernel 由 [vortex_device_mlir.py](src/gac/targets/vortex/device_mlir.py) 直接生成 LLVM Dialect MLIR，携带 RV32 参数 ABI、CTA CSR、`vortex.kernel` annotation、`llvm.used` 和 `llvm.compiler.used`。artifact 的 Makefile 从同一个固定 VOLT LLVM 20.1.8 树依次调用 `mlir-opt`、`mlir-translate`、VOLT clang 和 `ld.lld`，最后由 `vxbin.py` 生成 `.vxbin`；设备目录中不再生成 `kernel.cpp`。这条路径现已实际完成 `MLIR → LLVM IR → VOLT → ELF → vxbin → SimX`：合成 `[1,2,4] → 8 → 4` smoke 的最大绝对误差为 `1.1920929e-7`。10 次 SimX 的 runtime event-profile 均值为 `4.447282 ms`、端到端均值为 `4.602381 ms`；这些是软件模拟器的小尺寸结果，不能作为 Vortex 硬件或完整 BEVFormer 的推理速度。统一工具链、生成命令、ABI 和验证边界见 [BEVFormer FFN 到 Vortex 3.0 后端](docs/bevformer_vortex_backend.md)、[Vortex Direct-MLIR ABI](docs/designs/vortex_direct_mlir_abi.md) 与 [direct-MLIR 验证记录](docs/validation/vortex/direct-mlir-2026-10-13/README.md)。这一步先建立标量机器码基线；分块、LMEM 复用、融合、双缓冲和 TCU 映射属于后续性能 lowering。

如果要从输入模型开始逐层理解每一份 MLIR、Vortex Kernel ABI、VOLT 链接与 Runtime
调度，请阅读 [BEVFormer Direct-MLIR → Vortex 编译器分层架构](docs/bevformer_vortex_direct_mlir_architecture.md)。

Direct MLIR 现支持 `--matmul-schedule dxa-double-buffer`：FC1/FC2 按 M/N/K 分块，DXA 将 A/B Tile 搬到 LMEM 的两组 Stage，每线程在寄存器中累加，完成后写回 Global Memory。`--tile-m`、`--tile-n`、`--tile-k` 明确指定分块；Host 检查 occupancy、使用物理地址描述符并设置 `lmem_size`。默认仍保留 scalar 对照。分层实现与代码映射见[双缓冲设计](docs/designs/vortex_buffered_mlir.md)和[可点击源码图](docs/bevformer-vortex-buffered-code-map.html)。

关于 Tile 如何从算子索引关系、边界条件、片上存储、寄存器、线程组织、指令粒度和 Autotune 中
推导与选择，见 [AI 编译器中的 Tiling Shape 推导与选择](docs/tiling_shape_derivation.md)。

2026-10-14 的 SimX 验证覆盖尾部 Tile、多 warp 和 Stage 复用。在合成生产宽度 `[1,8,256] → 512 → 256` 上，`4×4×128` Tile 的总周期比同输入 scalar 少 **17.5%**，最大绝对误差 `1.43051147e-6`；小尺寸双缓冲也可能更慢。这是一个静态 FFN 的模拟器结果，不代表完整 BEVFormer 或 FPGA/ASIC 延迟，见[原始对照记录](docs/validation/vortex/buffered-mlir-2026-10-14/README.md)。

Vortex 的交互式 HTML 说明统一放在本仓库 `docs/`。
先打开[架构图目录](docs/index.html)，再进入 **V8** 查看本次源码分层后的真实代码映射；
旧图保留为历史版本，不能把固定 FFN 的 LMEM/DXA 优化视为全图默认实现。

| 版本 | 内容 |
| --- | --- |
| [V1：矩阵数据流](docs/vortex-matmul-dataflow.html) | 数据搬运与硬件数据流 |
| [V2：矩阵计算](docs/vortex-matmul-compute-v2.html) | 分块计算与执行过程 |
| [V3：端到端搬运](docs/vortex-matmul-end-to-end-v3.html) | DDR → LMEM Buffer → 累加寄存器 → DDR |
| [V4：代码映射](docs/vortex-matmul-code-map-v4.html) | 数据流步骤对应编译器与 RTL 源码 |
| [V5：编译器全流程](docs/vortex-matmul-end-to-end-v5.html) | 模型导入、优化、计算模板、后端调度与精度 |
| [V6：逐模块源码](docs/vortex-matmul-end-to-end-v6.html) | 固定 FFN 历史快照：模型导入到 Runtime 的源码摘录、解释与链接 |
| [V7：BEVFormer 全模型异构设计](docs/bevformer-compiler-architecture.html) | 历史设计：模型导入、Canonical Graph、自动切分、后端 Lowering、全局 Buffer 与 Runtime 边界 |
| [V8：重构后的分层与十阶段源码](docs/vortex-matmul-end-to-end-v8.html) | 模型适配器、实际十阶段函数、全图 Buffer、MLIR/VOLT/Runtime；源码摘录按当前规范路径自动生成 |

下载仓库后用浏览器打开 HTML 即可交互。V8 的 GAC 源码链接使用仓库内相对路径，
并附带固定源码提交的 GitHub 链接。历史 V4/V6 的摘录、旧目录与 SHA 保持原版本；
其中 Vortex/LLVM 链接仍指向页面标注的独立 checkout。

V8 从真实函数提取代码、行号和文件 SHA-256，更新源码后运行
`python tools/build_architecture_docs.py`；加 `--check` 可验证页面是否需要重新生成。

## MLIR / `mlir-opt` 文件在哪里？

**`mlir-opt` 是 LLVM/MLIR 编译出来的命令行程序。** 新的 [native/](native/README.md) 核心直接链接 MLIR C++ 库，`gac-opt` 注册本项目方言，`gac-compile` 在进程内运行 PassManager。下述历史 Python 路径仍通过子进程调用上游 `mlir-opt`：CUDA 路径把 GPU Dialect 降到 NVVM/SM86 PTX；旧 Vortex direct 路径用固定的 VOLT LLVM 20.1.8 验证 LLVM Dialect。新旧路径均可用同版本 `mlir-translate` 生成 LLVM IR。编译器可执行文件由本机构建，不提交到 Git；部署已编译的设备产物不需要重新运行这些编译 Pass。

GitHub 上完整的 BEVFormer MLIR/PTX 实现位于 [`lg/bevformer-mlir-ptx-opt`](https://github.com/luogantt/General-AI-Compiler/tree/lg/bevformer-mlir-ptx-opt) 分支；默认 `main` 当前没有下面这些工具链、lowering 和验证产物，因此直接浏览 `main` 会找不到。查看代码时请先切换到该分支。

- [`tools/build_mlir_wsl.sh`](tools/build_mlir_wsl.sh) 从项目外部的 `llvm-project` checkout 构建 `mlir-opt`、`mlir-translate` 和 `llc`。默认稳定路径是 WSL 中的 `$HOME/.cache/gac-mlir/current-build/bin/mlir-opt`。
- 本次 BEVFormer/UniAD 验证实际使用的可执行文件是 WSL 中的 `/home/thomas/gac-mlir-build-lf/bin/mlir-opt`；它是本机编译产物，不在 Git 中。对应的外部 LLVM 源码入口是 `C:\Users\Thomas\Desktop\universal_ai_compile\llvm-project\mlir\tools\mlir-opt\mlir-opt.cpp`，同样位于 General-AI-Compiler 仓库之外。
- [`src/gac/toolchains/mlir.py`](src/gac/toolchains/mlir.py) 负责查找 `mlir-opt`、构造无 shell 的命令参数、执行官方 GPU→NVVM pipeline、提取 PTX，并记录 LLVM revision、版本、命令和摘要。
- [`src/gac/targets/cuda/ptx.py`](src/gac/targets/cuda/ptx.py) 是 BEVFormer 固定 FFN 的全 PTX lowering；[`src/gac/targets/cuda/cublas_ptx.py`](src/gac/targets/cuda/cublas_ptx.py) 保留两个 cuBLAS MatMul，并把两个融合 Kernel 由 MLIR 降到 PTX。
- Git 中可直接查看的 MLIR 示例是 [`examples/mlir_sm86_add_relu.mlir`](examples/mlir_sm86_add_relu.mlir) 和 [`examples/mlir_sm86_layernorm_fusion.mlir`](examples/mlir_sm86_layernorm_fusion.mlir)。保存的真实 lowering 样例位于 [`docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.gpu.mlir`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.gpu.mlir)、[`production-ffn.binary.mlir`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.binary.mlir) 和 [`production-ffn.ptx`](docs/validation/sm86/mlir-ptx-2026-10-11/production-ffn.ptx)。
- 普通编译生成的 `.mlir`、`.ptx`、manifest 和报告写入 `build/`；该目录不进入 Git。新 ONNX→Vortex 核心的实现位于 `native/` C++ 源码；历史路径的 emitter、lowering、artifact 校验和 Runtime 位于 `src/gac/`。

在 Windows PowerShell 中构建并取得实际路径：

```powershell
wsl -d Ubuntu-24.04 -- bash tools/build_mlir_wsl.sh
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
```

## MLIR 编译路径（M0/M1）

MLIR 后端复用 Canonical Graph 数据结构和图 Pass；JSON 和已合法化的 ONNX 都进入这层。`examples/bevformer_tiny_ffn.json` 是一个手工构造的小尺寸 BEVFormer-like FFN smoke，用于检查 FFN 结构、广播和 LayerNorm lowering。它不读取官方 BEVFormer 代码或 checkpoint，也不是完整 BEVFormer importer。当前链路如下：

```text
JSON graph / hand-written BEVFormer-like tiny FFN smoke
                    │
                    ▼
             Canonical Graph
                    │  shape / dtype / broadcast validation
                    ▼
 func + tensor + linalg + arith + math
                    │
                    ▼
     mlir-opt --verify-each --canonicalize --cse
                    │
                    ▼
 textual .mlir（外部 verifier 成功后才标记 verified）
```

通用 `mlir-linalg` 产物把 `compiler_backend` 记录为已经实际使用的后端，把 `cuda-sm86` 记录为 `target_intent`，并把 `mlir-nvvm` 记录为计划的 codegen。它仍停在 tensor 层，`executable` 为 false。新增 `mlir-ptx-sm86` 则是一个独立、严格限定生产 FFN pattern 的可执行路径；它不是通用 Tensor/Linalg 自动 GPU 映射，也没有覆盖完整 BEVFormer。

Windows + WSL 的可复现工具链构建脚本是 [`tools/build_mlir_wsl.sh`](tools/build_mlir_wsl.sh)。它从项目旁的本地 `llvm-project` checkout 创建 LF 源码快照，并构建 `mlir-opt`、`mlir-translate` 和 `llc`：

```bash
wsl -d Ubuntu-24.04 -- bash tools/build_mlir_wsl.sh
```

从 Windows 调用 WSL 内构建出的 `mlir-opt` 验证上述 tiny FFN smoke：

```powershell
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
python -m gac.cli mlir-compile examples/bevformer_tiny_ffn.json `
  -o build/bevformer_tiny_ffn.mlir `
  --wsl `
  --wsl-distribution Ubuntu-24.04 `
  --mlir-opt $mlirOpt `
  --report build/bevformer_tiny_ffn.mlir.json
```

CLI 默认必须调用 `mlir-opt`；只有显式指定 `--emit-only` 才会跳过外部验证，并在报告中记录 `verified: false`。验证失败时不会写出目标 `.mlir` 文件。若命令在 WSL 内执行，可省略 `--wsl` 和 `--wsl-distribution`，把 `--mlir-opt` 直接指向 `$HOME/.cache/gac-mlir/current-build/bin/mlir-opt`。

本次已用本地 LLVM revision `765392a6b4498f1dc57233f03a31b87462fdef01` 构建的 `mlir-opt 24.0.0git` 真实执行上述命令，完整 tiny FFN 返回 `verified: true`。报告记录实际 executable、版本 banner、命令参数、Pass、超时和源码/产物摘要；使用仓库构建脚本生成的工具时，还会从受管 build marker 记录 LLVM source revision。

[`examples/mlir_sm86_add_relu.mlir`](examples/mlir_sm86_add_relu.mlir) 是独立的手写 GPU Dialect 工具链 smoke。以下命令已真实生成带 `#nvvm.target<chip = "sm_86">` 的 `gpu.binary`，其中内嵌 PTX 包含 `.target sm_86` 和 `add_relu_broadcast` entry：

```bash
"$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  examples/mlir_sm86_add_relu.mlir \
  "--gpu-lower-to-nvvm-pipeline=kernel-bare-ptr-calling-convention=1 cubin-chip=sm_86 cubin-format=isa allow-pattern-rollback=0" \
  -o build/mlir_sm86_add_relu.binary.mlir
```

这个 smoke 验证本地 stock MLIR/LLVM NVPTX 工具链能从手写 `gpu.module` 生成 SM86 PTX。`mlir-ptx-sm86` 已把 GAC Canonical Graph 中严格匹配的生产 FFN 连接到同一官方 GPU/NVVM 管线；通用图仍停在 tensor/linalg。

分层设计、里程碑和验收门槛见 [MLIR 编译器架构](docs/MLIR_ARCHITECTURE.md)。

## `mlir-cuda-sm86` 可执行 FFN 切片

`mlir-cuda-sm86` 是首个 MLIR-derived 可执行 runtime-call 切片，不是通用 MLIR GPU 后端。编译器先复用 tensor-level MLIR 的规范图和融合 Pass，然后只接受融合后的下列完整顺序：

```text
MatMul → AddReLU → MatMul → Add(bias) → Add(query residual) → LayerNorm
   │         │          │                      │
   ▼         ▼          ▼                      ▼
CuPy/     NVRTC      CuPy/          NVRTC Bias+Residual+
cuBLAS  Bias+ReLU    cuBLAS              LayerNorm
```

两次 MatMul 都是显式 `cupy.matmul(..., out=...)`，由 CuPy 调用 cuBLAS。第一个生成 Kernel 把 FC1 bias 与 ReLU 原地融合；第二个生成 Kernel 把 FC2 bias、原始 query residual 和末轴 LayerNorm 融合，并为每一行使用一个 CUDA block 做归约。MLIR source 记录相同的四调用 ABI，供审阅和 `mlir-opt` 验证；执行顺序来自编译器同步生成并由 Session 复核的四条 Python call record。Runtime 不解析 MLIR source，也不会按 op 名称猜测实现。

当前匹配器要求一个输入、一个输出、静态连续 FP32 Tensor、常驻的两组权重/偏置和 LayerNorm scale/bias、query identity residual，以及通道数不超过 4096。任何额外、缺失、重排或不匹配的节点都会在编译阶段被拒绝；它没有覆盖 GELU FFN、任意 MLP、完整 Encoder/Decoder、MSDA、CNN 或状态执行。

安装 CUDA 依赖后，可直接从 JSON graph 运行 tiny FFN smoke：

```bash
python -m pip install -e '.[cuda12]'
python -m gac.cli run examples/bevformer_tiny_ffn.json \
  --backend mlir-cuda-sm86 \
  --verify \
  --warmup 3 \
  --repeat 10 \
  --report build/bevformer_tiny_ffn.mlir-cuda.json
```

该路径要求 NVIDIA Compute Capability **恰好为 8.6**，并要求 CuPy、CUDA Runtime、NVRTC 和 cuBLAS 可用。两个融合 Kernel 在 Session 建立时通过 CuPy `RawModule`/NVRTC 编译；设备缺失、架构不匹配、cuBLAS/NVRTC 失败或 call plan 不匹配都会直接报错。`--trace` 和 `--profile` 仍只适用于 `cuda-sm86`；要测量这个 FFN 切片，请使用专用工具：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cuda_ffn.py \
  --warmup 50 \
  --iterations 100 \
  --repeats 20 \
  --public-samples 50 \
  --output build/mlir-cuda-cublas-ffn.json
```

基准工具只测一个生产尺寸 FFN 片段：`[1,2500,256] → 512 → 256 + ReLU + query residual + LayerNorm`，并把 device-resident 四调用 CUDA Event、公共 `run` 的 H2D+计算 GPU 时间、含 D2H/同步的 wall 时间、正确性、调用顺序和 artifact hash 分开记录。同进程共驻留的两条路径在每个 repeat 后交换先后位置，并分别用 MLIR-first 与 CUDA-first 作为首轮顺序重跑，以抵消初始化、热态和动态频率偏差。只有最终 JSON 中保存的真实设备样本可以作为数字来源，且不能外推为完整 BEVFormer、RTX 3070 或通用 MLIR/NVVM lowering；证据边界和两份原始报告见[真机验证说明](docs/validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)。

RTX 3080 配对交替实测中，同一进程内的两条路径按 repeat 轮换先后位置。以 MLIR-first / CUDA-first 作为首轮顺序的两份独立报告中，这个四调用切片的 device-resident 中位数分别为 **202.906 μs** 和 **209.848 μs**；同轮 CUDA codegen + cuBLAS 对照为 **220.313 μs** 和 **226.811 μs**。按两个测量位置平衡后的加速比分别为 **1.0858×** 和 **1.0816×**，两轮最大绝对误差均为 `9.5367431640625e-7`。这些数字只代表一个 FFN 子图。

仓库还提供一个独立的 [MLIR GPU LayerNorm 融合参考](examples/mlir_sm86_layernorm_fusion.mlir)，它真实经过 GPU Dialect → NVVM → SM86 PTX：block reduction LayerNorm 为 **10.271 μs**，相对 **174.121 μs** 的串行 MLIR 实现快 **16.95×**；Bias + Residual + LayerNorm 融合为 **13.604 μs**，相对 **28.344 μs** 的三个未融合 dispatch 快 **2.08×**。该 PTX 微基准尚未接入 `MlirCudaSession`，产品路径仍执行语义等价的 NVRTC Kernel；完整边界见[微基准说明](docs/validation/sm86/mlir-layernorm-fusion-2026-10-10/README.md)。

HXP v1 只携带旧 CUDA 后端的源码、launch metadata 和静态内存计划，不包含上述 MLIR runtime-call plan。因此下面的命令会被明确拒绝：

```bash
python -m gac.cli run model.hxp --backend mlir-cuda-sm86
```

`mlir-cuda-sm86` 仍是混合 runtime-call 路径，不能作为 MLIR→PTX 完成声明。

## `mlir-ptx-sm86` 官方 MLIR→PTX FFN 路径

`mlir-ptx-sm86` 复用相同的严格 FFN pattern matcher，但三个计算阶段都由 MLIR
GPU Dialect 表达并经官方 `mlir-opt` 降到 NVVM 和 PTX。当前 pattern matcher 与
GPU 调度器是 GAC 的 Python 编译 Pass：它直接根据 Canonical Graph 的类型和拓扑
合同物化显式 `gpu.func`；Tensor/Linalg 分支只复用相同合同，不是 GPU emitter 的
输入，也并非让 stock `mlir-opt` 自动猜测 Tensor/Linalg 的 GPU 调度。

```text
Canonical Graph
  ├→ Tensor/Linalg MLIR（独立审阅/验证产物）
  └→ strict FFN matcher + GAC GPU schedule
       → 3× gpu.func → NVVM → SM86 textual PTX 7.1
       → CuPy RawModule(path=PTX) → CUDA Driver JIT to SASS
       → 3× PTX kernel
```

Runtime 在访问 CUDA 前重建规范图和 launch plan，检查 GPU MLIR/source hash、唯一
`gpu.binary`、`.target sm_86`、`.address_size 64`、精确 entry 集合和整体 artifact
摘要。执行阶段没有 cuBLAS、NVRTC 或 CPU fallback。离线编译和 GPU 主机加载使用
同一 `MlirPtxArtifact`，GPU 主机不需要安装 LLVM/MLIR。部署目录只保存 `gpu.mlir`、
`binary.mlir`、`kernels.ptx` 和 `manifest.json`；tensor-level MLIR 会内联权重，因此不进入
部署目录。PTX 是中间设备代码，CUDA Driver 在模块加载时仍会 JIT 为 SASS。

用模型 JSON 离线编译一个可校验的生产尺寸 FFN 产物。输出目录必须尚不存在或为空，
报告文件必须位于该目录外：

```powershell
$env:PYTHONPATH = "src"
python -m gac.cli mlir-ptx-compile model.json `
  -o build/mlir-ptx-artifact `
  --mlir-opt /home/thomas/gac-mlir-build-lf/bin/mlir-opt `
  --wsl --wsl-distribution Ubuntu-24.04 `
  --report build/mlir-ptx-compile.json
```

在 GPU 主机上用同一生产 FFN JSON 提供权重，并加载离线目录：

```bash
PYTHONPATH=src python -m gac.cli run model.json \
  --backend mlir-ptx-sm86 \
  --mlir-ptx-artifact build/mlir-ptx-artifact
```

`manifest.json` 的摘要用于发现损坏和模型/launch-plan 不匹配，不是发布者签名；只加载由
可信离线编译环境生成或由外部可信摘要确认的 PTX 目录。

RTX 3080 配对交替实测中，全 PTX 路径为 **1.092～1.102 ms/FFN**，cuBLAS
混合路径为 **0.209～0.214 ms/FFN**；当前全 PTX 正确性基线慢 **5.150～5.265×**，
最大绝对误差为 `7.153e-7`。原因是首版两个 MatMul 采用每线程串行 K 循环，尚未
使用 shared-memory tiling、向量化或 Tensor Core。MLIR→PTX 本身不自动提高速度；
性能来自 scheduling、融合和目标专用 lowering。完整编译产物、原始报告和复现命令见
[MLIR→PTX 真机验证](docs/validation/sm86/mlir-ptx-2026-10-11/README.md)。

## `mlir-cublas-ptx-sm86` 四调用生产候选

`mlir-cublas-ptx-sm86` 面向同一个固定 FP32 生产 FFN，并复用严格 pattern matcher。它把成熟矩阵乘和项目可控融合边界组合成以下执行计划：

```text
FC1 MatMul                 CuPy/cuBLAS
Bias + ReLU               MLIR GPU Dialect → NVVM → PTX
FC2 MatMul                 CuPy/cuBLAS
Bias + residual + LayerNorm  MLIR GPU Dialect → NVVM → PTX
```

两个融合 entry 使用官方 LLVM 24 pipeline 生成 PTX 7.1；`MlirCublasPtxSession` 在 CUDA 初始化前重建并校验规范计划，再用 `RawModule(path=PTX)` 加载。`mlir_cublas_ptx_artifact.py` 可把 GPU MLIR、binary MLIR、PTX 和 manifest 原子发布为无权重目录；加载时用另行校验的 graph 常量重建完整四调用计划。这个 Runtime 不向 NVRTC 提交 CUDA C，也没有 NumPy、旧 CUDA 或全 PTX fallback。MatMul 当前是 `cupy.matmul(..., out=...)` 的 cuBLAS 路径，尚未接入 cuBLASLt epilogue。

在已有生产尺寸 JSON graph 上可直接编译并运行：

```bash
PYTHONPATH=src python -m gac.cli run build/bevformer-production-ffn.json \
  --backend mlir-cublas-ptx-sm86 \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  --verify --warmup 50 --repeat 100 \
  --report build/mlir-cublas-ptx-sm86.json
```

输入 JSON 必须恰好满足 `[1,2500,256] → 512 → 256`、FP32、连续布局及 query residual 合同；小尺寸 `examples/bevformer_tiny_ffn.json` 会被明确拒绝。

RTX 3070 Ti Laptop（SM86、8 GB、Driver 580.178.04、CUDA Runtime 12.9、CuPy 13.6.0）上，两轮各采用 warmup 50、40 个 repeat、每个 repeat 100 次完整 FFN，并在每轮内交换 M2c/M2a 测量顺序。PTX-first 报告中，M2c、M2a 与 tiled 的 device-resident 中位数分别为 **221.915998 / 227.223997 / 892.597122 μs**，M2c 位置平衡加速为 **1.0248387378×**；NVRTC-first 报告中分别为 **223.343678 / 229.038086 / 893.539200 μs**，加速为 **1.0274777097×**。两轮三个候选均为 `eligible`，autotuner 都选择 `mlir-cublas-ptx-sm86`；M2c 与 tiled 均为 `allclose=True`，最大绝对误差都为 `7.152557e-7`。

M2c 公共 `run` 的 H2D+计算 GPU 中位数为 **584.239990 / 575.888008 μs**，含 D2H 与同步的 wall 中位数为 **1233.796007 / 1192.150492 μs**（PTX-first / NVRTC-first）。正式报告绑定 clean Git 提交 `5794d2c`，并保存了已与本地逐一匹配的 80 个源码 SHA-256。这些数字只属于单个生产形状 FFN，不是完整 BEVFormer。上一版 RTX 3080 cuBLAS hybrid 的 `0.209～0.214 ms/FFN` 是另一台设备上的历史基线，不能与这台 Laptop GPU 的绝对延时直接横比。

专用工具会构造确定性的生产图、支持无权重 artifact，并保存逐次原始样本。两次运行只改变首轮顺序：

```bash
PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --mlir-opt /path/to/mlir-opt \
  --emit-artifact-dir build/mlir-cublas-ptx-artifact \
  --compile-only --output build/mlir-cublas-ptx-compile.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order ptx-first --warmup 50 --iterations 100 --repeats 40 \
  --public-samples 30 --autotune \
  --autotune-cache build/mlir-cublas-ptx-autotune-ptx-first.json \
  --output build/mlir-cublas-ptx-ptx-first.json

PYTHONPATH=src python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --artifact-dir build/mlir-cublas-ptx-artifact \
  --benchmark-order nvrtc-first --warmup 50 --iterations 100 --repeats 40 \
  --public-samples 30 --autotune \
  --autotune-cache build/mlir-cublas-ptx-autotune-nvrtc-first.json \
  --output build/mlir-cublas-ptx-nvrtc-first.json
```

## tiled、NVGPU/MMA、异步双 stage 与 autotuning

[`src/gac/targets/cuda/tiled_ffn.py`](src/gac/targets/cuda/tiled_ffn.py) 已把单缓冲 16×16×16 FP32 shared-memory tile 接成完整生产形状 FFN：FC1+Bias+ReLU、FC2、Bias+Residual+LayerNorm 三次 PTX dispatch，执行不经过 NVRTC。[`MlirTiledFfnSession`](src/gac/execution/mlir_tiled_ffn.py) 与无权重 artifact 已接到 benchmark/autotune。RTX 3070 Ti Laptop 上使用 warmup 50、每个 repeat 100 次、40 个 repeat、public samples 30 的固定提交 `5794d2c` 正式结果如下：

| 完整 FFN 候选 | PTX-first | NVRTC-first | 正确性/选择状态 |
| --- | ---: | ---: | --- |
| `mlir-tiled-ffn-sm86` | 0.892597122 ms | 0.893539200 ms | `allclose=True`，max abs `7.152557e-7`，eligible |
| `mlir-cublas-ptx-sm86` | 0.221915998 ms | 0.223343678 ms | `allclose=True`，max abs `7.152557e-7`，eligible，autotuner 选择 |
| `mlir-cuda-sm86` | 0.227223997 ms | 0.229038086 ms | eligible |

[`src/gac/targets/cuda/tiled_matmul.py`](src/gac/targets/cuda/tiled_matmul.py) 还提供真实输入 Tensor Core 功能候选。TF32 路径计算 `A[16,16] × B[16,8] → C[16,8]`，PTX 含 4 条 `mma.sync.m16n8k4`、12 次 global load 和 4 次 store；FP16 路径计算 `A[16,32] × B[32,8] → C_f32[16,8]`，PTX 含 2 条 `mma.sync.m16n8k16`、24 次 global load 和 4 次 store。两者在 RTX 3070 Ti Laptop 上均为 `max_abs_error=0`，每次 launch 的 CUDA Event 中位数分别为 **3.8796 μs** 和 **4.0127 μs**。它们仍是固定尺寸功能候选，没有接入生产 FC1/FC2 或完整 FFN。

[`src/gac/targets/cuda/async_tiled_matmul.py`](src/gac/targets/cuda/async_tiled_matmul.py) 实现了真实输入 FP32 `16×16×64` 双缓冲 MatMul。LLVM 24→SM86 PTX 7.1 包含 8 条 `cp.async`、4 个 commit、`wait_group [1,1,1,0]` 和 4096 B shared memory；RTX 3070 Ti Laptop 上 256 个输出逐元素 exact，`max_abs_error=0`，每次 launch 中位数 **2.652160 μs**。这个固定尺寸候选尚未接入生产 FFN。

功能候选可以独立复现：

```bash
PYTHONPATH=src python tools/probe_mlir_tiled_matmul.py \
  --output-dir build/mlir-tiled-probe \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt" \
  --which fc1 --tile-m 16 --tile-n 16 --tile-k 16 \
  --probe-functional-mma

PYTHONPATH=src python tools/probe_mlir_async_tiled_matmul.py \
  --output-dir build/mlir-async-tiled-matmul \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
```

[`src/gac/tuning/cuda_ptx.py`](src/gac/tuning/cuda_ptx.py) 定义候选的 backend、`(M,N,K)` tile、CUDA block、FP32/TF32/FP16 精度和融合策略。它先对每个候选做输出正确性检查，再计时并按重复样本的中位数选择赢家；缓存绑定 graph/source SHA-256、设备、候选空间和计时配置，并用原子替换发布。[`src/gac/tuning/cuda_cublas_ptx.py`](src/gac/tuning/cuda_cublas_ptx.py) 现在可以让 `mlir-cublas-ptx-sm86`、`mlir-cuda-sm86` 和一个 `mlir-tiled-ffn-sm86` Session 同轮进入 CUDA Event 计时。真实输入 MMA 与 `cp.async` MatMul 仍未接到完整 FFN autotune。生产 Tensor Core FFN、双缓冲生产 FFN、cuBLASLt 和更多 tile 候选仍未完成。

本轮优化边界、真机结果和未接线候选见 [cuBLAS+PTX 优化验证说明](docs/validation/sm86/mlir-cublas-ptx-2026-10-11/README.md)。

## RTX 3070 / SM86 适配边界

首个目标是桌面 RTX 3070 对应的 `cuda-sm86`。最终全图验收实际使用 **RTX 3080 10 GB**；本轮 cuBLAS+MLIR PTX 单 FFN 使用 **RTX 3070 Ti Laptop 8 GB**。这些 GPU 与桌面 RTX 3070 的 Compute Capability 都是 8.6，因此结果验证了相同 SM86 编译目标下对应切片的代码生成、加载和 Runtime 契约。

这不等于桌面 RTX 3070 性能验收。不同卡的计算资源、带宽、显存容量和功耗不同，本文所有 BEVFormer 全图耗时仍属于 RTX 3080；RTX 3070 Ti Laptop 新数据只覆盖一个 FFN。桌面 RTX 3070 的全图延迟、实际峰值显存和稳定性尚未实测。

## RTX 3080 最终验收

验收环境为 Ubuntu 22.04、RTX 3080、CUDA SM86；GAC 使用 Python 3.11.17、CuPy 13.6.0、CUDA Runtime/NVRTC 12.9。官方 `bevformer_tiny_epoch_24.pth` 在独立 PyTorch/MMCV 参考环境中生成 golden。

下表是固定 Profile raw step 的稳态中位数：全生成路径各测 3 次，优化路径各测 10 次。GPU stream 与调用 wall time 均不包含 Session 构建、完整数据预处理或 CPU NMS-free 后处理。

| 路径 | 全生成 GPU / wall | 零拷贝 + FP32 cuDNN/cuBLAS GPU / wall | GPU / wall 加速比 |
| --- | ---: | ---: | ---: |
| cold | 425.325073 / 436.881591 ms | 57.624464 / 69.403093 ms | 7.38098× / 6.29484× |
| warm | 426.032532 / 437.718436 ms | 56.823263 / 68.747440 ms | 7.49750× / 6.36705× |

输出相对官方 PyTorch/MMCV golden 的最大绝对误差为：

| 路径 | `cls_logits` | `bbox_regression` | `next_bev` |
| --- | ---: | ---: | ---: |
| cold | 0.003891945 | 0.004856288 | 0.001953125 |
| warm | 0.000953674 | 0.000725389 | 0.001953125 |

其他最终验收数据：

- 独立进程完成 Session 构建后的首个 cold `run`：`277.443573 ms GPU / 289.668472 ms wall`。同一进程基准里的 `116.635651 ms` 发生在 Session 已初始化后，不是 fresh-process 首次 `run`。
- 带逐 launch CUDA Event 的独立 Profile 会扰动执行：该轮为 `76.05085 ms GPU / 95.8321 ms wall`，703 个 Kernel Event 求和 `63.4391 ms`；其中 cuDNN 55 次 / `36.6409 ms`、CuPy/cuBLAS 118 次 / `8.0060 ms`、generated CUDA 530 次 / `18.7923 ms`。这些值用于归因，不与上表普通基准直接比较。
- 强制 CUDA 的完整测试：`Ran 287 tests in 18.881s`，`OK (skipped=1)`。
- 最终 HXP：`137,896,143` bytes，SHA-256 `0afa4892ac5cdadd0916c5aa17abc3e153b6a03c71eb1d0adfdc9723fe2bb7f4`。
- 图优化规模：融合前 876 节点，39 组 Add+ReLU 融合后 837 个执行节点，零拷贝后 703 次设备 dispatch；内部 arena `552,960,000` bytes。
- Tensor 计划为 `705,177,100` bytes；加上 cuDNN 8 MiB 后，`713,565,708` bytes 是 Tensor + workspace 的计划预算/准入值，并非设备进程峰值，不包含 CUDA module/context、Event 和 vendor 临时分配。

显式 128 MiB cuDNN workspace 实验把 55 个 Conv 的 p50 从 `35.030 ms` 降到 `25.781 ms`，但完整 cold 图交错测量 p50 从 `57.924 ms` 变为 `58.231 ms`，并比当前 8 MiB budget 多占 120 MiB，因此本轮没有集成。下一步优先推进 CUDA Graph、批量 dispatch 和更多融合；当前 `cupyx.cudnn` 与 capture 不兼容，cuDNN 图捕获仍需要稳定的低层 API。

## 快速运行

离线编译与 NumPy 参考路径需要 Python 3.11+。Linux 或 WSL2 上的 CUDA 路径还需要 NVIDIA 驱动和项目的 `cuda12` 可选依赖。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[cuda12]'

.venv/bin/python -m gac.cli doctor
.venv/bin/python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
.venv/bin/python -m gac.cli inspect build/mlp-sm86.hxp
.venv/bin/python -m gac.cli run build/mlp-sm86.hxp \
  --verify \
  --trace build/trace.json \
  --profile build/profile.json \
  --report build/report.json
```

`--profile` 会额外执行一次带逐 launch Event 的调用；默认运行路径不创建这些 Event。`compile` 也可输出 `.cu` 和 `.plan.json` 调试产物。当前 HXP 保存规范图、常量、CUDA 源码、launch metadata 和静态内存计划，不是预编译 cubin；它与当前 codegen 绑定，优化前的 schema 1 包需要重建。

没有 NVIDIA GPU 时，可显式使用 NumPy 参考后端检查语义：

```bash
python -m pip install -e .
python -m gac.cli compile examples/heterogeneous_mlp.json -o build/mlp-sm86.hxp
python -m gac.cli inspect build/mlp-sm86.hxp
python -m gac.cli run build/mlp-sm86.hxp --backend numpy-cpu --verify
```

Windows 将解释器路径换成 `.\.venv\Scripts\python.exe`。原生 Windows 的离线编译与 CPU 路径已检查，CUDA 路径尚未实测。

BEVFormer 的合同检查、权重检查和子图/HXP 构建由 `gac-bevformer` 提供：

```bash
gac-bevformer --help
gac-bevformer step-pipeline --help
```

完整的官方 checkpoint 环境、golden 生成和远程复现步骤见 [BEVFormer 参考环境](docs/bevformer_reference_environment.md)与[远程验收记录](docs/bevformer_sm86_validation_2026-10-10.md)。

## UniAD 2.0 adapter

GAC 现在包含一个自包含的 UniAD 2.0 `stage2_e2e` BEV encoder FFN adapter。它从可信 checkpoint 精确映射第 0～5 层的两组 Linear、bias 和 post-FFN LayerNorm 参数，通过带 SHA-256 的 data-only bundle 跨越官方 UniAD Python 3.9/CUDA 11.8 环境与 GAC Python 3.11/CUDA 12 环境。bundle 加载禁用 pickle，并严格检查成员、摘要、shape、dtype、连续布局和有限值。

**这里编译的是一个 UniAD `stage2` BEV encoder layer 的 FFN，不是完整 UniAD。** 静态 FP32 Profile 是 `[1,40000,256] → 512 → 256`。attention、tracking、mapping、motion、occupancy、planning 和其余层仍在这个编译单元之外，因此单个 FFN 的延迟不能换算成完整 UniAD 帧率。

这个 FFN 目前有四条可核验路径：

- `cuda-hxp`：六 dispatch 基线，两个 MatMul 在 Runtime 中选择 CuPy/cuBLAS；
- `mlir-cuda-sm86`：旧四调用路径，两个 cuBLAS MatMul 加两个运行时 NVRTC 融合 Kernel；
- `mlir-cublas-ptx-sm86`：两个 cuBLAS MatMul 加两个由 `mlir-opt` 离线生成的融合 PTX Kernel，部署执行不需要 `mlir-opt` 或 NVRTC；
- `mlir-tiled-ffn-sm86`：FC1+Bias+ReLU、FC2、最终 Bias+Residual+LayerNorm 三个离线 PTX dispatch，部署执行同样不需要 `mlir-opt` 或 NVRTC。

`--backend both` 为兼容旧流程，只生成 HXP 和 `mlir-cuda-sm86`；`--backend all` 才会额外生成两种离线 PTX artifact。没有 checkpoint 时可生成生产 shape 的确定性 synthetic bundle。它只用于编译、正确性和性能链路验证，不包含真实模型权重，也不能用于完整 UniAD 精度或端到端结论：

```bash
python tools/uniad_make_synthetic_bundle.py build/uniad/layer0.uniadffn
```

在 Windows PowerShell 中调用 WSL 的 `mlir-opt`，一次生成全部四种产物：

```powershell
$mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
python tools/uniad_compile_ffn.py `
  build/uniad/layer0.uniadffn build/uniad/layer0-compiled `
  --backend all --tile 16,16,16 `
  --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt
```

若命令本身在 WSL/Linux 内运行，直接传 Linux 路径：

```bash
python3 tools/uniad_compile_ffn.py \
  build/uniad/layer0.uniadffn build/uniad/layer0-compiled \
  --backend all --tile 16,16,16 \
  --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
```

`build/uniad/layer0-compiled/mlir-cublas-ptx-sm86/` 和 `mlir-tiled-ffn-sm86/` 是无内联权重、带完整性 manifest 的离线代码目录。部署机加载这些目录时不运行 `mlir-opt`，也不把 CUDA C 交给 NVRTC；CUDA Driver 仍会在加载 textual PTX 时完成 PTX→SASS JIT，cuBLAS+PTX 路径还需要 CuPy/cuBLAS。

如果只需要两种新 PTX artifact，也可以让正式 benchmark 工具执行纯离线编译；`--compile-only` 不访问 CUDA：

```powershell
python tools/benchmark_mlir_cublas_ptx_ffn.py `
  --uniad-bundle build/uniad/layer0.uniadffn --seed 20261011 `
  --emit-artifact-dir build/uniad/layer0-compiled/mlir-cublas-ptx-sm86 `
  --emit-tiled-artifact-dir build/uniad/layer0-compiled/mlir-tiled-ffn-sm86 `
  --compile-only --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt `
  --output build/uniad/layer0-compile.json
```

官方 UniAD 环境中的 checkpoint 导出使用同一 checkout wrapper，不需要把 Python 3.11 的完整 GAC 环境安装进去：

```bash
python tools/uniad_export_ffn.py \
  /path/to/UniAD-2.0/ckpts/uniad_base_e2e.pth \
  build/uniad/layer0.uniadffn --layer 0 --trusted-checkpoint \
  --with-synthetic-golden
```

导出器优先使用 Torch 的 `weights_only=True` 数据加载路径。`--trusted-checkpoint` 只允许不支持该参数的旧版 Torch 回退到 pickle 反序列化，因此只应对来源可信的本地 checkpoint 使用。

在 SM86 GPU 上，使用同一个 UniAD bundle 和已编译 artifact 做三候选配对测量与 autotune。`--repeats` 必须为偶数；resident GPU 时间与包含 H2D、D2H 和同步的 public 时间会分开报告：

```bash
python tools/benchmark_mlir_cublas_ptx_ffn.py \
  --uniad-bundle build/uniad/layer0.uniadffn \
  --seed 20261011 \
  --artifact-dir build/uniad/layer0-compiled/mlir-cublas-ptx-sm86 \
  --tiled-artifact-dir build/uniad/layer0-compiled/mlir-tiled-ffn-sm86 \
  --device 0 --warmup 10 --iterations 30 --repeats 6 \
  --public-samples 3 --benchmark-order ptx-first \
  --autotune --autotune-cache build/uniad/layer0-autotune.json \
  --output build/uniad/layer0-benchmark.json
```

正式复现还要启动第二个独立进程，把 `--benchmark-order` 改为 `nvrtc-first`，并为该进程使用不同的 `--autotune-cache` 和 `--output` 文件。

`tools/uniad_benchmark_ffn.py`（安装入口 `gac-uniad-benchmark-ffn`）保留为旧 `cuda-hxp` 与 `mlir-cuda-sm86`/NVRTC 的基线对比；上面的 `benchmark_mlir_cublas_ptx_ffn.py --uniad-bundle` 才测量新的离线 PTX 和 tiled 候选。

最终 v3 在 RTX 3070 Ti Laptop（SM86）上使用真实 `uniad_base_e2e` checkpoint 的 layer-0 权重、seed `20261011` 的确定性 synthetic query 和 GAC NumPy reference。两个独立进程分别改变首次后端顺序，正式协议为 warmup 10、6 个 repeat、每个 repeat 30 次 FFN，以及 3 个 public `run()` 样本：

| 初始顺序 | cuBLAS+MLIR PTX resident | cuBLAS+NVRTC resident | tiled PTX resident | cuBLAS+MLIR PTX public wall | cuBLAS+NVRTC public wall | resident 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTX-first | 3.050785 ms | 3.150221 ms | 13.932249 ms | 20.710336 ms | 20.564843 ms | 1.035303× |
| NVRTC-first | 3.076282 ms | 3.182030 ms | 13.936361 ms | 21.537857 ms | 21.664099 ms | 1.032900× |

两份报告的 resident primary estimator 几何平均为 **1.034101×，即约 3.41% 优势**。public wall 包含参数检查、H2D、计算、D2H 和同步，两种顺序下 PTX 与 NVRTC 相差都不到 `0.15 ms`，方向还会随初始顺序变化，因此只能表述为基本持平。三个候选均通过 `rtol=1e-4`、`atol=1e-5` 的 `allclose`；两次 autotune 都选择 `mlir-cublas-ptx-sm86`。完整原始样本、源码摘要和 artifact manifest 见 [UniAD 单层 FFN 最终 MLIR/PTX 真机证据](docs/validation/sm86/uniad/mlir-ptx-2026-10-11/README.md)。

这只是一个 `[1,40000,256] → 512 → 256` FFN。bundle 不含 expected sample，真实的是 checkpoint 权重；输入仍是 synthetic query，正确性参考由 NumPy 计算。结果不代表完整 UniAD 的真实数据集精度、端到端延迟或 FPS。

迁移前历史基线是在 RTX 3080 上用合成权重和合成输入测得：旧四调用 MLIR/NVRTC 为 `1.635～1.645 ms`，同轮 CUDA/HXP + cuBLAS 为 `2.074～2.075 ms`。完整 UniAD native PyTorch 模型调用的历史均值为 `822.885 ms`。设备、权重和实现边界均不同，不能把这些历史数字与 v3 直接比较，也不能解释为 MLIR 整网加速。完整 native PyTorch 基线通过必选的 `--uniad-root` 连接外部 UniAD checkout；详细环境与边界见 [UniAD 编译器说明](docs/uniad_compiler.md)。

统一测试入口：

```bash
python -m unittest discover -s tests -v
GAC_REQUIRE_CUDA=1 python -m unittest discover -s tests -v
```

`GAC_REQUIRE_CUDA=1` 用于真机验收；CUDA 不可用时会失败，不会把 GPU 测试静默算作通过。

## 项目结构

```text
src/gac/
├── core/                 Graph、TensorSpec、算子语义与异常
├── compiler/             十阶段契约、图 Pass、分区、Buffer 规划
├── frontends/            通用 JSON / ONNX / torch.export 导入
├── models/
│   ├── api.py            延迟加载的模型适配器注册表
│   ├── common/           可复用 Transformer / FFN 构图
│   ├── bevformer/        BEVFormer 构图、状态与专用权重导入
│   └── uniad/            当前 UniAD 单层 FFN 适配与数据包
├── targets/
│   ├── vortex/           十阶段实现、MLIR、设备 ABI、Host 生成
│   ├── cuda/             CUDA / MLIR / PTX 编译与计算模板
│   └── mlir.py           通用 Tensor / Linalg MLIR 导出
├── toolchains/           外部 MLIR / LLVM 工具调用与版本检查
├── execution/            设备 Session、Dispatch 与性能采集
├── artifacts/            HXP、权重和离线 PTX 产物
├── validation/           独立 NumPy 参考执行与精度验证
├── tuning/               CUDA 候选搜索与基准
├── workflows/            同时了解模型与目标的上层编排
├── tools/                命令行入口
└── _compat.py            旧导入路径的集中兼容映射

examples/                 模型示例与冻结合同
tools/                    构建、验证和基准脚本
tests/                    编译、模型、Runtime 与架构边界测试
docs/                     设计说明与验收证据
```

分层职责、十阶段代码位置与新增模型方法见[源码分层说明](docs/code_layout.md)。
新代码使用规范导入，例如 `gac.core.ir`、`gac.models.bevformer.ffn` 和
`gac.targets.vortex.compiler`；原来的 `gac.ir`、`gac.bevformer_ffn`、
`gac.backends.vortex_graph` 等名称仍按需映射到同一实现。
未来 VLA 模型通过 `ModelAdapter → Graph` 接入；这不代表其全部算子或完整推理已经实现。

更完整的实现映射见[架构说明](docs/architecture.md)，MLIR 分层与里程碑见 [MLIR 编译器架构](docs/MLIR_ARCHITECTURE.md)，BEVFormer 图与状态语义见[编译器设计](docs/bevformer_compiler_design.md)，后端扩展约定见[后端编写指南](docs/backend_authoring.md)，后续计划见[路线图](docs/roadmap.md)。

## 已知边界

- 当前支持声明限定为固定 Profile 的 BEVFormer-tiny **raw step**。CPU `decode_nms_free` 是公开的图外后处理边界，不包含在上述 GPU/wall 数字中。
- 正确性输入是冻结的合成、已归一化并 padding 的六相机 tensor 与固定几何输入；尚未完成真实 nuScenes 数据集的 mAP/NDS 验收。
- 完整 BEVFormer 只在 RTX 3080 SM86 上验收；桌面 RTX 3070 的延迟、峰值显存和稳定性仍待实测。
- 通用 `mlir-linalg` 路径仍只生成 tensor-level textual MLIR，并提供外部 verifier 接口。`mlir-ptx-sm86` 已完成一个生产形状 FFN 的 GPU/NVVM/PTX 闭环；`mlir-cublas-ptx-sm86` 只完成固定四调用生产候选。两者都不是通用 bufferization/GPU mapping，也没有完整 BEVFormer 或 RTX 3070 性能数据。
- MLIR 的通用 smoke 仍包含手写 tiny FFN JSON；此外，完整 HXP 已可经过严格 FFN 识别导出逐子图 Tensor/Linalg MLIR。该导出尚不等于完整模型统一 MLIR lowering 或物理异构执行。
- 完整 BEVFormer 尚未由 MLIR lowering 覆盖；MSDeformableAttention、GridSample、CNN 和状态执行仍使用现有 CUDA/HXP 路径。
- Vortex 3.0 路径当前只覆盖一个静态 FP32 BEVFormer FFN；SimX 用于 ISA/ABI/数值 smoke，其耗时不是 FPGA/ASIC 延时，也不能与 RTX 3070 数据直接比较。
- UniAD 编译覆盖当前只包含一个 stage2 BEV encoder FFN；tracking、mapping、motion、occupancy、planning 和 attention 尚未接入 HXP/MLIR。完整 UniAD 报告是外部 checkout 的 native PyTorch 基线，不是编译后整网数据。
- HXP v1 不携带 MLIR runtime-call plan 或 PTX artifact；`mlir-ptx-sm86` 可通过独立的已校验目录和 `--mlir-ptx-artifact` 加载，尚未并入 HXP。
- M0/M1 把小型 smoke 权重写成内联 `dense` 常量；完整模型需要资源化常量或外部权重 ABI，不能直接沿用这种文本承载方式。
- 只覆盖首条路径所需的静态 Shape 和连续布局；主要计算仍为 FP32。FP16/TF32 `mma.sync` 与双缓冲 `cp.async` 已有真实输入固定尺寸功能候选，但尚未扩展为生产 FC1/FC2。单缓冲 tiled 完整 FFN 已在固定提交 `5794d2c` 上完成三候选正式 autotune；生产 Tensor Core FFN、双缓冲生产 FFN 和 cuBLASLt 仍未完成。
- 当前使用单 CUDA stream，调用同步返回；设备侧跨帧状态驻留、多 stream、I/O 预绑定、CUDA Graph、批量 dispatch 和 DLPack 仍待实现。
- `713,565,708` bytes 只是已计划 Tensor 与 cuDNN workspace 的预算/准入值，不能当作 `nvidia-smi` 进程峰值。
- 当前 HXP 与本版 codegen 绑定，尚无预编译 cubin、多 target variant、签名认证或通用异构 Host Execution Plan。
- 已实现 BEVFormer 专用语义切分、逻辑跨设备 transfer 与全局 Buffer lifetime 计划；物理 CUDA/Vortex 统一 Session、设备 alloc/free、DMA/copy、event 同步、代价驱动通用划分和 NPU 后端尚未实现。
- Session 建立时选择 cuDNN、CuPy/cuBLAS 或 generated CUDA；vendor 调用执行中失败会报错，不会逐节点自动换实现。

## 验收证据

最终数字与历史阶段日志的边界见[优化验收证据索引](docs/validation/sm86/bevformer/optimization-2026-10-10/README.md)。主要原始文件：

- [最终 cold/warm 基准与精度](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-benchmark-final.json)
- [fresh-process 首次 cold run](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-first-frame-final.json)
- [最终 Profile 摘要](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final-summary.json)与[逐 launch Profile](docs/validation/sm86/bevformer/optimization-2026-10-10/gac-production-profile-final.json)
- [最终强制 CUDA 测试日志](docs/validation/sm86/bevformer/optimization-2026-10-10/unittest-full-final.txt)
- [最终 HXP 构建日志](docs/validation/sm86/bevformer/optimization-2026-10-10/build-production-step-final.txt)、[大小](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.size.txt)与[SHA-256](docs/validation/sm86/bevformer/optimization-2026-10-10/bevformer_tiny_step_sm86-final.sha256.txt)
- [128 MiB cuDNN workspace 实验](docs/validation/sm86/bevformer/optimization-2026-10-10/cudnn-explicit-workspace-benchmark.json)
- [`mlir-cuda-sm86` 生产尺寸 FFN 真机验证与配对交替原始报告](docs/validation/sm86/mlir-cuda-cublas-2026-10-10/README.md)
- [`mlir-ptx-sm86` 官方 MLIR→NVVM→PTX 产物与真机配对报告](docs/validation/sm86/mlir-ptx-2026-10-11/README.md)
- [`mlir-cublas-ptx-sm86` 真机结果及 tiled/MMA/异步双 stage/autotune 的实现边界](docs/validation/sm86/mlir-cublas-ptx-2026-10-11/README.md)
- [UniAD 单层 FFN 最终 MLIR/PTX 真机证据](docs/validation/sm86/uniad/mlir-ptx-2026-10-11/README.md)
- [UniAD 单层 FFN 迁移前真机历史证据](docs/validation/sm86/uniad/ffn-2026-10-11/README.md)
- [UniAD native PyTorch 端到端历史基线](docs/validation/sm86/uniad/e2e-native-nuscenes-mini-2026-10-11/README.md)
- [完整远程验收说明](docs/bevformer_sm86_validation_2026-10-10.md)

## 参考项目与许可证

本项目的架构研究参考 IREE、Apache TVM、ONNX Runtime、TileLang 和 OpenXLA/XLA，但当前实现为本原型独立编写，没有从这些项目复制源码。未来若引入外部源码、头文件、生成物或二进制依赖，需要保留适用的版权、SPDX、`NOTICE` 和第三方许可证，并单独核对 CUDA、cuDNN、厂商 SDK 与模型权重的分发条款。
