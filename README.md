# General AI Compiler（GAC）

以 **BEVFormer** 为验证模型的 AI 编译器与推理 Runtime。原生核心使用 **C++ / MLIR**，把 ONNX 编译到 **CUDA PTX** 或 **Vortex**；另提供可独立运行的 **TensorRT 推理项目**。

**TensorRT 最新结果：完整 BEVFormer-base 固定图达到 309.42 ms/帧（约 3.23 FPS）。** RTX 3070 Ti Laptop 8 GB 上，第三轮优化 DCNv2 后，同轮整网对照延时再降低 **5.07%**。前两轮 MSDA 分别降低 18.05%、2.60%；各轮比例不能相加，详见[优化结果](#优化结果)。

**GalaxeaVLA 当前已验证结果：G0.5 LIBERO 整模型在单张 Ascend 910B2 上达到 p50 155.57 ms、p95 156.79 ms。** 保持双相机、真实样本、BF16 和 10 步 FM，组合使用 GE/ACLGraph、Triton GDN、fused AE attention/Norm、FP32 matrix RoPE、functional MRoPE、固定几何 Vision 图、按请求索引的时间条件预计算及未消费 codec 消除。与同为 **native 随机数计时**的 eager 1011.55 ms 相比加速 **6.50 倍**。A/B/C/D/E/F 共 12 次同设备核验，最坏相对 L2 误差 **0.2511%**、最大绝对误差 **0.004586**；不代表闭环任务准确率。用户 RTX 3090 的 120.48 ms 目标尚未达到。[原始数据、误差验证与复现](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/README.md)

## 做了什么

| 部分 | 已实现内容 |
|---|---|
| 模型导入 | C++ 读取 ONNX，直接创建注册的 `gac` MLIR Operations，保留类型、形状、属性和权重 |
| 图优化与 Lowering | 形状推导、常量折叠、CSE、条件融合、合法化；规则算子走 Linalg，不规则算子走专用 Lowering |
| 设备编译与 Runtime | Dispatch 划分、静态 Buffer 复用、设备代码生成；Runtime 负责分配、上传、启动、同步和下载 |
| CUDA / Vortex | 共用前端和语义层，分别生成 PTX 或 VOLT 的 ELF/vxbin；已有矩阵分块、归约及 Vortex LMEM/DXA/TCU 实验模板 |
| Triton / Vortex（适配基线） | C++ 将支持的 Linalg 子图送入 Triton；与原生 Vortex 内核链接到同一 vxbin，共用 Buffer 与 Runtime。[使用与边界](native/README.triton-vortex.md) |
| GalaxeaVLA / Ascend 910B | 官方 G0.5 整模已运行：LIBERO 真实样本的当前优化组合达到 155.57 ms（native），完成 A/B/C/D/E/F 12 次同设备检查，包含两相机分别变化。163.91/172.50/200.70 ms 对照、各 shared-noise 版本、native 219.18 ms 对照及退化实验分别保留。GAC 的 MLP 切片已贯通 ONNX → TTIR → `npubin`。整模 GAC 编译与通用 NPU Runtime 尚未实现。[范围与实测](docs/galaxeavla-ascend910.md)；[源码补丁及运行环境](integrations/galaxeavla/README.md) |
| TensorRT | 完整 Base 固定 ONNX 的插件适配、构建、推理、混合 INT8/FP16、逐层 profiling，以及自定义 MSDA / DCNv2 Kernel 优化 |

**验证范围：** 原生 CUDA 已完成固定输入的 BEVFormer-base 整网执行和数值对齐；Vortex 已有完整模型编译产物和小模型 SimX 验证，完整模型的 Vortex 执行与 FPS 尚未验收。GalaxeaVLA/Ascend 已完成官方 checkpoint 整模测速、LIBERO 同设备 eager/TorchAir/`npugraph_ex` 输出对照与变化输入的图重放验证，以及独立的 GAC MLP 切片编译执行验证；尚无整模 GAC 编译、跨 CUDA 数值对齐或机器人任务精度验收。当前 Base 整网记录对应固定形状、固定几何、无历史帧到原始检测头输出，尚无 nuScenes mAP/NDS 验收。

## 编译链

```text
ONNX
  ↓ C++ Importer
GAC MLIR → 图优化 / 合法化
  ↓
Linalg（规则算子）＋ GAC 专用 Lowering（采样等不规则算子）
  ↓
循环、线程映射、Dispatch、Buffer 规划
  ├─ CUDA Lowering → NVVM / LLVM → NVPTX → PTX → CUDA Driver → NVIDIA GPU
  ├─ Vortex Lowering → LLVM → VOLT → ELF / vxbin → Vortex Runtime
  └─ Ascend phase 1 → standard TTIR → Triton-Ascend → npubin 工件（无 GAC NPU Runtime）
```

TensorRT 是独立的对照和部署路径：`ONNX → TensorRT engine → TensorRT Runtime`。本次约 18% 的收益来自其中的 **C++/CUDA 自定义插件**，没有计入原生 MLIR → PTX 路径。默认 Linalg 路线与旧专用模板的选择方式见[原生架构](native/README.md)。

## GalaxeaVLA 整模推理（Ascend 910B2）

**LIBERO 120 ms 对标优化，2026-11-16：** 使用与用户 RTX 3090 测试相同的官方 `g05-libero` checkpoint 和 episode 0/frame 0，双相机 256×256、159 tokens、batch 1、32×20 动作、10 步 FM。BF16 常驻权重保留官方 FP32 区域；编译变体及 GDN/未消费 codec 优化分别记录，权重、输入与采样步数不变。三角求解变体保留 FP32 状态但改变浮点累加顺序，不宣称逐元素相同。

| LIBERO 配置 | 噪声计时 | Prefill p50 | FM10 p50 | 整模型 p50 | 整模型 p95 |
|---|---|---:|---:|---:|---:|
| 910B2 BF16 eager | native | 265.29 ms | 746.76 ms | 1011.55 ms | 1022.70 ms |
| 910B2 GE AE / ACLGraph VLM + triangular + 跳过未消费 codec | native | 119.03 ms | 100.10 ms | 219.18 ms | 221.87 ms |
| 上一行 + exact 时间条件预计算，退化实验 | native | 117.20 ms | 111.30 ms | 228.46 ms | 229.25 ms |
| 910B2 GE AE / ACLGraph VLM + Triton GDN + fused AE + 跳过未消费 codec | native | 103.04 ms | 97.67 ms | 200.70 ms | 203.10 ms |
| 上一行 + request 时间预计算 + 固定几何 Vision 图 | native | 82.72 ms | 89.74 ms | 172.50 ms | 173.58 ms |
| 172.50 ms 配置 + GE single_stream，未提速 | native | 83.41 ms | 89.63 ms | 173.09 ms | 173.58 ms |
| 172.50 ms 配置 + AE FP32 matrix RoPE | native | 82.64 ms | 81.20 ms | 163.91 ms | 164.62 ms |
| 上一行 + fused AE Norm + functional MRoPE | native | 83.36 ms | 72.18 ms | **155.57 ms** | **156.79 ms** |
| 910B2 BF16 eager，早期基线 | shared | 266.32 ms | 747.07 ms | 1013.87 ms | 1026.25 ms |
| 910B2 BF16 + TorchAir AE | shared | 257.39 ms | 108.03 ms | 365.39 ms | 369.58 ms |
| 910B2 BF16 + `npugraph_ex` frozen AE + VLM，原公式 | shared | 155.04 ms | 142.82 ms | 297.89 ms | 300.62 ms |
| 910B2 BF16 + TorchAir frozen AE + VLM，兼容分解 | shared | 1490.62 ms | 100.54 ms | 1590.90 ms | 1597.47 ms |
| 910B2 BF16 + frozen graph AE/VLM + triangular + 跳过未消费 codec | shared | 118.12 ms | 146.38 ms | 264.46 ms | 265.40 ms |
| 910B2 BF16 + frozen GE AE / ACLGraph VLM + triangular + 跳过未消费 codec | shared | 117.61 ms | 108.01 ms | 225.58 ms | 226.40 ms |
| RTX 3090 BF16 + Inductor AE，用户已有实测 | 原设备 randn | 75.42 ms | 45.03 ms | **120.48 ms** | 125.33 ms |

各轮 Ascend 均预热 5 次、正式测量 20 次，覆盖设备输入 → Vision/VLM prefill → 完整 FM10 → 归一化动作。`native` 计入原设备 `torch.randn`，`shared` 计入固定噪声 `clone`；**6.50 倍仅用 native 基线计算**。数值验证和归档 `action.npy` 始终使用单独共享噪声。当前最快组合的 A/B/A/B 加 C/D/E/F 各两次，共 **12 次检查**：最坏 relative-L2 **0.2511%**、max-abs **0.004586**，均出现在独立修改外部相机的 E；动作及 gripper 符号一致率均 100%，padding 为 0，C/D/E/F tokens/mask 与各自 eager 参考一致。首次含编译请求 **153.65 秒**单列；最快单次 155.13 ms 不是 p50。

Vision 图只缓存固定双相机几何，逐请求核对网格、读取当前 patches 和学习位置嵌入；每层仍执行两次独立 256-token SDPA。时间条件保持原 FP32 10 步递推和 49 个 AdaLN 投影，只在原 `FMHelper.infer` 请求内使用计数索引，取消每步 D2H 查表。当前报告记录 39 次请求完整消费 10 步，无失败；未跨请求缓存图像、状态、KV 或模型结果。此前同时加入这两项得到 172.50 ms；再启用 AE matrix RoPE 为 163.91 ms，加入 fused Norm 与 functional MRoPE 后完整实测为 155.57 ms。单流候选 173.09 ms 未提速。matrix RoPE 每请求读取当前 cos/sin；functional MRoPE 只以固定索引选择当前频率，Norm 保持 FP32 数学及原输出 dtype。

混合后端由同一 TorchAir 包提供：AE 使用 GE，VLM 使用 `mode=npugraph_ex` 的 ACLGraph 路径，避免独立后端包的重复注册冲突。最新组合将 GDN 的 31 次逐行递推放入 FP32 Triton kernel，并启用 NPU fused AE attention；两者均需数值检查，不宣称位级一致。所有整模型行均独立实测，未拼接阶段耗时。172.50 ms 对照的 [Level1 trace 汇总](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/profile-172ms-level1-summary.json)和[最新组合的 Level1 汇总](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/profile-155ms-level1-summary.json)均来自计时后诊断，不能替代正式计时；163.91 ms 报告未采集 trace。独立 [RoPE 微实验](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/ae-rotary-npu.json)、[Norm GE 微实验](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/ae-norm-ge.json)及 [functional MRoPE 的 24 次零误差检查](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/functional-mrope-ge-acl.json)与整模型结果分别归档。本轮相关本地测试、可视化 builder 测试及 overlay 检查均通过。

[原 CSV 的精确算子统计](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/experiments/kernel-type-comparison-172-vs-155.json)显示：172.50→155.57 ms 两个组合的 trace 中，ScatterElements **40→0**、StridedSliceD **2120→180**、Square **960→0**，新路径使用 490 次 RmsNorm 和 480 次 RotaryPositionEmbedding。仍有 **4 个 AI_CPU** 任务；算子 duration 合计不是 wall-clock 延迟，也不能把各项变化相加当成加速收益。

原公式 frozen graph 的 **297.89 ms** 结果仍保留：同设备 A/B/A/B 有效 7 维逐元素误差为 0，首次含编译请求 153.82 秒。TorchAir AE 的 365.39 ms 及 0.1337% 最坏相对 L2 误差也保留。当前仍未达到 120 ms，且所有这些检查均不代表跨 CUDA 或闭环精度验证。[完整证据及命令](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/README.md)

整 VLM 的 TorchAir GE 路径曾分别遇到 Softplus、eye converter 未实现；加入局部分解后整模型通过同设备 A/B/A/B 检查（最坏相对 L2 0.1956%、最大绝对误差 0.003065），但总延迟退化到 **1590.90 ms**，保留实验记录而不选为最佳。GDN triangular 的独立 helper 微实验（eager 9.72 → 4.92 ms、graph 4.29 → 3.14 ms）与整模型结果分开记录，不能据此拆分当前组合中各项优化的收益。[失败摘要与实验边界](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/README.md)

[LIBERO 优化交互图](docs/visualizations/g05-ascend-optimization.html)可切换 3090、910B2 eager 和编译结果，查看 Prefill/FM10 的实测分布、误差与冷编译证据。新的 [LIBERO 算量原始记录](docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16/libero-bf16-ge-frozen-compatible/workload.json)来自计时后恢复 eager 的独立请求，已覆盖算术为 **1.064344764416 TFLOP/请求**，仅计 Linear/Conv/AdaLN；它是原 eager 路径的已覆盖算量下界，遗漏 attention/GDN 等运算，不能据此宣称总 FLOPs、硬件利用率或所需峰值算力。该数含原路径 codec 的 1.2108 GFLOP 和逐请求时间条件投影；当前删除或预计算这些工作的新变体须重新计数，不能粗减后称为完整实测算量。它属于真实 LIBERO 输入，不沿用下述 R1 Lite 计算量。

峰值 BF16 吞吐不能直接换算这里 batch 1、159-token 请求的时延。上述 trace 中的 slice、transpose、cast、vector 小算子及 AI_CPU 工作说明，短序列还受访存、执行调度和算子实现影响。图编译可减少启动与中间张量访存，但不会让整模型权重全部常驻片上 SRAM；是否提速仍以同工作负载的完整测量为准。

**R1 Lite 完整双头基线：** 以下保留三相机、CoT 与离散动作分支的独立记录，不与 LIBERO 表计算加速倍数。

2026-11-16，**单张 910B2 64 GB，CANN 9.0.0、PyTorch 2.7.1、torch_npu 2.7.1.post4**。加载官方 `OpenGalaxea/G05` 的 `g05-base`，使用 BF16 autocast 并保留 FP32 精度区域。完整路径为三相机预处理 → Vision/VLM prefill → CoT AR → **10 步 FM** → 离散动作 AR → ActionCodec → 动作后处理。

输入为固定合成 R1 Lite 观测，batch 1、三路 720×1280 图像（模型处理为 256×256）和 14 维原始状态。一次推理返回 **32 步**双臂和夹爪动作。计时同步 NPU，预热 5 次、正式测量 30 次：

| 范围 | p50 | p95 |
|---|---:|---:|
| 完整请求，包含预处理、H2D/D2H 和后处理 | **3036.79 ms** | **3130.30 ms** |
| `policy.predict_action` 整模调用 | 2994.28 ms | 3087.88 ms |
| CPU 原始观测整理与预处理 | 38.68 ms | 39.35 ms |

完整请求 p50 对应 **0.329 次推理/秒**；同轮模型加载 81.95 秒、首次请求 4.381 秒，正式测量峰值 allocated **15.93 GiB**、reserved **17.03 GiB**。冷启动、预热和测量阶段的动作输出均有限，形状符合预期。

该容器可见 256 个逻辑 CPU，但配额为 32 核。默认 PyTorch 256 线程时整条请求 p50 为 5401.08 ms；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 后得到上表结果。两轮原始数据均保留，这项调整只改变 CPU 并行设置，整模仍为 **TorchNPU eager**，没有使用整模 GAC/Triton 编译。

这组输入的 CoT 阶段返回 20 个原始 token，后续 action AR 返回 1 个终止 token，完整序列解析出 18 个动作 token；实际输入和生成长度会影响延迟。数据用于合成输入吞吐评估，不包含网络服务或机器人执行，也不代表任务准确率。

[完整结果、分阶段耗时与原始 JSON](docs/validation/ascend/galaxeavla-g05-full-910b2-2026-11-16/README.md) · [国内 PyPI 环境安装、官方权重准备与运行命令](integrations/galaxeavla/runtime-setup.md) · [固定版本的 Ascend 源码补丁](integrations/galaxeavla/README.md)

**交互式算力与时延图：** [离线 HTML 网络图](docs/visualizations/g05-ascend-compute-latency.html)（下载后浏览器打开），支持 137 个子图、节点计算量与时延、目标请求频率的算力预算。对相同输入另做一次算子诊断，实际前缀为 **234 token**，已统计的 Linear、普通 Conv 和 AdaLN 条件投影合计 **1.50515 TFLOP/请求**。它不是完整模型总 FLOPs 或芯片峰值算力；未覆盖 attention/GDN 功能运算、ConvTranspose 等。页面的延迟仍来自上述独立的 30 次正式测量，内部未计时模块不按计算量分摊时间。[统计口径与数据](docs/validation/ascend/galaxeavla-g05-full-910b2-2026-11-16/README.md#counted-arithmetic-and-interactive-view)

## 如何启动推理

以下命令使用 Linux / WSL Bash；从仓库根目录开始。完整 BEVFormer 的 ONNX、权重、输入和参考输出需单独准备，源码仓库不包含这些大模型资产。输出目录使用新路径。

### 原生编译器：ONNX → MLIR → PTX → GPU

需要 LLVM/MLIR **20.1.8**（启用 NVPTX）、CMake/Ninja、C++17、protobuf/protoc、ONNX 的 `onnx.proto`、CUDA 头文件和 libdevice；执行机器需要 NVIDIA 驱动。[构建细节](native/README.cuda.md#构建和执行)与[Base 模型导出](native/README.bevformer-base.md#4-导出与编译)单独说明。

```bash
# 1. 按实际环境修改路径，构建编译器和 Runtime。
export LLVM_PATH=/path/to/llvm-20.1.8
cmake -S native -B build/native-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PATH/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PATH/lib/cmake/llvm" \
  -DONNX_PROTO_DIR=/path/to/onnx \
  -DGAC_CUDA_INCLUDE_DIR=/usr/local/cuda/include
cmake --build build/native-cuda --parallel 2

# 2. 从原始 GAC/ONNX 导出图生成 PTX。
export GAC_NATIVE_BIN="$PWD/build/native-cuda"
export CUDA_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
export GAC_CUDA_ARCH=sm_86
export GAC_CONSTANT_FOLD_PROFILE=bevformer-base
export GAC_REGULAR_LOWERING=linalg

# 这里使用原始 GAC/ONNX 导出图，不使用 TensorRT 插件改写后的 adapted.onnx。
bash native/tools/compile-cuda.sh /path/to/bevformer-base.onnx build/bevformer-cuda

# 3. 运行编译产物，并与同一输入的独立参考输出比较。
python3 native/tests/run_cuda_tests.py \
  --artifact build/bevformer-cuda \
  --inputs /path/to/inputs.npz --reference /path/to/reference.npz \
  --runner build/native-cuda/gac-cuda-run \
  --output-dir build/bevformer-cuda-run \
  --warmup 1 --repeat 5 --run-timeout 7200 --rtol 1e-3 --atol 1e-4
```

Python 工具需要 NumPy，负责输入文件转换和结果比较；模型计算由 **C++ Runtime 启动 PTX** 完成。编译目录包含 `kernel.ptx`、`plan.json`、权重和各阶段 MLIR；运行目录保存输出与 `cuda-validation.json`。直接调用 Runtime 和 Vortex 启动方式分别见 [CUDA](native/README.cuda.md) / [Vortex](native/README.md#6-从-onnx-到-vxbin-与执行)。

### TensorRT：构建 engine 并推理

独立项目位于 [standalone/bevformer-tensorrt](standalone/bevformer-tensorrt/README.md)。已测环境为 Python 3.11、TensorRT **10.13.3.9**、PyTorch **2.7.0+cu126**。

```bash
cd standalone/bevformer-tensorrt
bash scripts/setup.sh --python python3.11 --install-torch
source .venv/bin/activate
python -m bevformer_trt check --report results/environment.json
```

将配套模型资产放入 `models/bevformer-base/`；[资产契约](standalone/bevformer-tensorrt/docs/model-and-data.md)说明 ONNX、校准范围和六相机输入的格式。以下启动**官方插件的混合精度基线**：

```bash
export CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0
python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --engine results/mixed/base.engine --precision mixed-int8 \
  --calibrated-ranges models/bevformer-base/ranges.json \
  --workspace-mib 1024 --tactic-dram-mib 4096 \
  --builder-optimization-level 3 --max-aux-streams 0 \
  --report results/mixed/build.json

python -m bevformer_trt run \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output results/mixed/run --warmup 5 --repeat 20
```

结果为 `outputs.npz`（BEV 特征、分类 logits、3D 框编码）和 `report.json`（延时、环境与检查结果）。未传 `--reference` 时不作数值对照。配套合成输入用于回归，不用于重新校准或评估检测精度；FP32 启动命令见[独立项目 README](standalone/bevformer-tensorrt/README.md)。

**启用优化后的 MSDA：** 按[复现流程](standalone/bevformer-tensorrt/docs/msda-optimization.md)编译插件、显式改写六个 Spatial MSDA、重新绑定原校准范围并构建 engine。构建和运行都要传 `--plugin-library /path/to/libgac_msda_plugin.so`；上面的默认命令不会自动启用它。

**再启用 DCNv2 优化：** 在上述模型基础上，[编译 DCN 插件](standalone/bevformer-tensorrt/native/dcn/README.md)，构建时传 `--dcn-config` 精确节点配置，构建及运行均加载两份插件库。[完整命令和本轮验证](standalone/bevformer-tensorrt/docs/validation-dcn-round3-3070.md#代码与复现)

## 优化结果

2026-10-21，**RTX 3070 Ti Laptop 8 GB，BEVFormer-base，混合 INT8/FP16**。首轮对照：

| 测试范围 | NVIDIA 官方实现 | 自定义 MSDA 实现 | 收益 |
|---|---:|---:|---:|
| 单个 Spatial MSDA，相同输入 | 38.95 ms | 16.91 ms | 加速 **2.30 倍** |
| 完整固定图，引擎每帧中位延时 | 413.77 ms | 339.09 ms | 延时降低 **18.05%** |
| 完整固定图，延时倒数 | 2.42 FPS | 2.95 FPS | 吞吐换算约 **1.22 倍** |

优化保持采样公式和模型权重不变：**每线程 4 通道、half2、坐标/地址复用、固定形状专用化**；实测选择 128 线程/块，关闭未获益的预取。只替换六个 Encoder Spatial MSDA，其余节点保留。

完整引擎按“官方—优化—优化—官方”测试，每轮预热 5 次、测量 20 次，均从 55°C 开始；复用 timing cache 并核对其他层无 tactic/格式变化。**同一真实六相机输入的三个输出逐位一致**。计时排除预处理、传输、加载和框解码；单算子与整网分别计时，不能混算。

这证明本轮优化没有改变已测输入的官方混合引擎输出；不代表混合量化对 FP32 或整个数据集的精度已经达标。测试图仍使用固定合成几何和无历史帧，未达到 30 FPS。

原始数据、首次失败对照及最终受控结果见[MSDA 验证报告](standalone/bevformer-tensorrt/docs/validation-msda-3070.md)。原生 MLIR 的编译/JIT/推理数据单独记录在[CUDA 路径对照](docs/validation/cuda/linalg-comparison-2026-10-21/README.md)，不与 TensorRT 数字混用。

**第二轮：** 增加有溢出保护的 32 位索引与邻点地址复用，修复 Kernel helper 重构导致的性能回归。实际旧库与新库对照各 60 帧：**335.60 → 326.88 ms，延时再降 2.60%**；三个输出仍逐位一致。采用每线程 4 通道、128 线程块、`specialize=3`、`prefetch=0`。预取候选反而变慢，完整 CUDA Graph 捕获失败，均未计入收益。两轮百分比不能相加。[第二轮实测及失败记录](standalone/bevformer-tensorrt/docs/validation-msda-round2-3070.md) · [启用配置](standalone/bevformer-tensorrt/docs/msda-optimization.md#7-继续优化与已经优化的配置比较)

**第三轮（2026-10-22）：** DCNv2 采样阶段每线程复用 4 个通道的坐标、邻点和插值系数，保留 FP16 运算及 cuBLAS HGEMM。替换 156 个满足条件的节点，整网同轮对照 **325.97 → 309.42 ms，延时降低 5.07%，约 3.23 FPS**。三个输出与基线、官方插件参考逐位一致；320 项 GPU 候选检查和双上下文检查通过。[第三轮报告、原始数据与复现](standalone/bevformer-tensorrt/docs/validation-dcn-round3-3070.md)

## 代码与文档

| 入口 | 内容 |
|---|---|
| [native/](native/README.md) | C++ ONNX 导入、MLIR 方言/Pass、设备 Lowering、CUDA/Vortex Runtime |
| [native/templates/](native/templates/README.md) | CUDA/Vortex 计算模板及选择约束 |
| [standalone/bevformer-tensorrt/](standalone/bevformer-tensorrt/README.md) | 独立 TensorRT 构建、推理、校准和 MSDA 优化 |
| [src/gac/](src/gac) | Python 模型适配、工具和历史 Graph/HXP 路径 |
| [docs/](docs/index.html) | 架构图、原理说明、验证报告 |

- 理解模型：[BEVFormer 架构图](docs/bevformer-model-architecture.html)；理解编译器：[MLIR 源码架构图](docs/mlir-source-architecture.html)、[算子融合](docs/compiler-fusion.md)。HTML 下载后用浏览器打开。
- 历史实验：[训练记录](docs/training/bevformer-tiny/README.md)、[早期 FFN / HXP / UniAD 与版本记录](docs/history/README-before-2026-10-21.md)。各阶段的支持范围和性能分别查看。
