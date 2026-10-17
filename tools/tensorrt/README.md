# BEVFormer TensorRT 实验入口

分支：`lg/bevformer-tensorrt`，基于原生 CUDA/Vortex 优化提交 `baac4de`。
本分支先建立完整 BEVFormer 的 TensorRT 对照路线，用于比较数值与性能。
截至 2026-10-17，已固定两份上游参考源码，并在 `robot@192.168.3.33` 安装
TensorRT 10.13.3.9，通过真实 GPU 加法网络测试。当前直接适配本项目已经导出的
Base cold ONNX，使用 TensorRT 内置 DCNv2/MSDA v2 插件，保留相同输入和参考输出。
完整模型构建、数值与延时的结果以验证报告为准；环境测试不代表模型测试通过。

本地执行入口及精度、计时范围见 [README.runtime.md](README.runtime.md)；入口/head FP16、
backbone 普通卷积 INT8 的规则和校准方式见 [MIXED_PRECISION.md](MIXED_PRECISION.md)。
混合精度代码、完整模型逐层计划、校准缓存和小 CNN 的 GPU 集成测试已完成；
真实 nuScenes mini 的 32 帧六相机输入已完成旧完整图的校准并写出缓存，但随后 engine
构建因 Conv 常量权重的 Identity 别名失败。GPU 对照已复现问题，适配器已修复这些引用。
修复后的 backbone 已完成 32 帧真实校准并构建 INT8 引擎；完整 Base 构建遇到显存限制，尚无整网精度或 FPS 结果。
小网络的结果不能替代整网结果。

## 当前测试环境

测试目录为 `/home/robot/project/gac-tensorrt-20261017`。
`venv` 使用 Python3.11，复用已有 PyTorch2.7.0+cu126 的只读包依赖，TensorRT
安装在新 venv 中，没有改动原 Isaac 环境或系统驱动。GPU 为 RTX3070 Ti Laptop 8GB，
驱动 580.178.04。独立环境检查入口：

```sh
python tools/tensorrt/check_environment.py --report build/tensorrt/environment.json
```

它真正构建、加载并执行一个 FP32 加法 engine，核对输出，同时检查两个插件版本。
只有 import 成功或 `nvidia-smi` 正常，不算完成这项测试。
这个环境使用 TensorRT 自带 CUDA 插件，暂不需要另装 CUDA 编译器或编译 DerryHub 插件。

## 选用的开源实现

| 来源 | 已核实的内容 | 本项目用途 |
|---|---|---|
| [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) | 完整 Tiny/Small/Base、导出器、自定义 CUDA 插件、TensorRT 构建和 nuScenes 评估 | 完整数据集部署与性能参考 |
| [NVIDIA/DL4AGX：bevformer-int8-eq](https://github.com/NVIDIA/DL4AGX/tree/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/bevformer-int8-eq) | 在 DerryHub 基础上提供 TRT10 补丁、ModelOpt 显式量化和容器；其公布实验为 Tiny | 新版工具链与后续 INT8 的参考 |

这是同一实现的延伸关系。NVIDIA 示例中的 Tiny 结果不能当作 Base 验证。
两份源码及许可证保留在独立 checkout 中：DerryHub 顶层文件名是 `LICENCE`，
NVIDIA 示例是 `LICENSE`，均为 Apache-2.0；TensorRT SDK 等依赖遵循各自许可。

## 获取和核对源码

从 GAC 仓库根目录执行，Python 仅负责 Git 下载与版本核对：

```sh
python tools/tensorrt/fetch_sources.py --fetch
python tools/tensorrt/fetch_sources.py --check
```

精确版本记录在 [sources.json](sources.json)。默认下载到 Git 已忽略的目录：

```text
build/external/BEVFormer_tensorrt/   # 303d3140c14016047c07f9db73312af364f0dd7c
build/external/DL4AGX/              # 9f7b29104c253d5bc68334e7b83b3eecb72d4572
```

脚本核对 commit、工作区与入口文件，并运行 `git apply --check`。
遇到已有改动会报错；可用 `--cache <新目录>` 保留已修改的 checkout。
下载器不会安装 CUDA、修改驱动、应用补丁或执行上游安装脚本。
这里只固定这两个 Git 仓库；容器镜像和传递依赖尚未形成完整锁定环境。
Windows checkout 用 Windows Git/Python 检查；WSL/Linux 建议通过 `--cache`
在各自文件系统新建 checkout，避免跨系统 Git 行尾配置造成误报修改。

## 实际执行链与代码位置

```text
现有 Base cold ONNX + 固定输入 + PyTorch 参考输出
    ↓ adapt_onnx.py：自定义算子 ABI、shape 与 Conv 常量权重引用适配
ONNX（标准算子 + TensorRT 插件节点）
    ↓ run_engine.py build：Parser + 已注册的 C++/CUDA 插件
TensorRT Builder：融合、精度与 tactic 选择
    ↓
TensorRT engine
    ↓ run_engine.py run：TensorRT Runtime + CUDA
GPU 推理 → 三路 raw-head 数值检查、延时
```

这个实验使用 TensorRT 自身的编译器与 Runtime。现有 `native/` 的 C++ MLIR
编译链保留独立，未增加 `gac-compile --target=tensorrt`，也未宣称 TensorRT
执行了我们的 MLIR Pass。此 cold 测试不测 nuScenes 指标、历史帧或框后处理。

以下路径相对 `build/external/BEVFormer_tensorrt/`：

| 阶段 | 入口 |
|---|---|
| Base、half2 插件模型配置 | `configs/bevformer/plugin/bevformer_base_trt_p2.py` |
| checkpoint → ONNX | `tools/pth2onnx.py` |
| 自定义算子导出 | `det2trt/models/` |
| 插件 C++ 接口与 CUDA 实现 | `TensorRT/` |
| ONNX → engine | `tools/bevformer/onnx2trt.py` |
| 数据输入、跨帧状态、评估 | `tools/bevformer/evaluate_trt.py` |
| TensorRT 执行和计时 | `det2trt/utils/tensorrt.py` |

## Base FP16 的准确入口

先按上游文档准备隔离的 Linux 环境、插件、checkpoint、nuScenes/CAN bus 和
预处理后的 annotation 文件。**上游构建脚本即使只选择 FP16，也会读取数据集配置**；
下面不是无需数据的“一键跑通”命令。

进入上游仓库，将 `bevformer_r101_dcn_24ep.pth` 放入 `checkpoints/pytorch/`，
以下是未经 TRT10 补丁修改的原始 TRT8 流程：

```sh
cd build/external/BEVFormer_tensorrt
bash samples/bevformer/plugin/base/pth2onnx_2.sh -d 0
bash samples/bevformer/plugin/base/onnx2trt_fp16_2.sh -d 0
bash samples/bevformer/plugin/base/trt_evaluate_fp16_2.sh -d 0
```

产物依次为 `checkpoints/onnx/bevformer_r101_dcn_24ep_cp2.onnx` 和
`checkpoints/tensorrt/bevformer_r101_dcn_24ep_cp2_fp16.trt`。
`plugin/base/` 和 `_2` 对应 half2 路线，不能误用普通 `base/` 的 MMDeploy 配置。

环境必须区分：上游 README 的原测速环境是 CUDA11.6 / TensorRT8.5.1.7 / PyTorch1.12.1；
该 commit 的 Dockerfile 实际默认 CUDA11.8、Ubuntu20.04、TensorRT8.5.3.1。
它还写死了本机代理地址，直接照搬构建可能失败。保留来源不变，在专用环境中修订并记录差异。

插件构建入口为 `TensorRT/CMakeLists.txt`，生成 `TensorRT/lib/libtensorrt_ops.so`；
上游还依赖 MMCV、MMDetection、MMDeploy 和 `third_party/bev_mmdet3d` 扩展。
构建时会探测 GPU 架构，需要可见 GPU。原始 builder 默认 workspace 参数虽写作 `1`，
实现实际分配上限为 `1 << 32` 字节，即 4 GiB；它不是 engine 总显存用量。
8GB 上能否构建 Base，仍需单独测量，不能用推理显存数字替代构建峰值。

新版 TRT10 路线应从 NVIDIA 示例的 Dockerfile 与补丁入手，其当前配置为
`nvcr.io/nvidia/tensorrt:26.01-py3`、PyTorch2.9.0、ModelOpt0.41.0，涉及 CUDA13。
需要另行核对 GPU 驱动与容器兼容性。**不能把 TRT8 的构建命令和插件二进制直接混入 TRT10。**
补丁适用性检查通过，只证明源码上下文匹配，不代表 Base 插件已经编译或推理成功。

## 为什么不能直接拿现有 ONNX 文件跑

当前 GAC Base ONNX 的契约见
[export-reference-contract.json](../../docs/validation/cuda/native-mlir-bevformer-base-2026-10-17/export-reference-contract.json)：
它是固定合成标定的 cold raw-head 图，仅输入 `images`，含 `gac::MSDeformAttn`
及 MMCV DCNv2 节点。TensorRT 不会自动识别自定义的 `gac` 算子。

上游 Base 的主要输入为：

| 输入 | shape |
|---|---|
| `image` | `[1,6,3,928,1600]` |
| `prev_bev` | `[40000,1,256]` |
| `use_prev_bev` | `[1]` |
| `can_bus` | `[18]` |
| `lidar2img` | `[1,6,4,4]` |

这些接口张量使用 FP32，内部算子可以采用 FP16。上游输出为 `bev_embed`、
`outputs_classes`、`outputs_coords`；后两者首维为 decoder 层数，不能解释为每相机输出。

本次采用另一条可比较的入口：保留现有 ONNX，在 `adapt_onnx.py` 中转换插件 ABI。
156 个 DCNv2 节点转换属性名；18 个 MSDA 节点新增 INT32 spatial shapes、
level start offsets，并将插件四维输出 reshape 回原三维输出。另将 390 个标准 Conv 的
780 条常量 weight/bias Identity 别名改为直接 initializer 引用，保留权重数值、激活边、
节点名称、输出和模型 IO；动态权重不改写。它没有用 Python 执行神经网络，也没有用
预计算激活替代算子。
适配器拒绝未知自定义算子、动态 spatial shapes 和未经验证的非对称 DCN 参数。

DerryHub 插件的输入语义不同，不能仅靠改名替换本项目 MSDA。本次使用 NVIDIA
TensorRT 自带插件，其源码契约见
[MSDA](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.cpp)
和 [DCNv2](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/modulatedDeformConvPlugin/modulatedDeformConvPlugin.cpp)。

## 真实 mini 校准输入

官方 nuScenes mini 解压后，用 [prepare_nuscenes_calibration.py](prepare_nuscenes_calibration.py)
按当前 Base 契约准备输入；它只需要 NumPy、OpenCV，不执行模型或访问 GPU：

```sh
python tools/tensorrt/prepare_nuscenes_calibration.py \
  --data-root /home/robot/datasets/nuscenes \
  --output build/trt/mini-train-calibration --max-samples 32 \
  --export-report build/bevformer-base-onnx/export-report.json
```

目录需包含 `v1.0-mini/` JSON 表和 `samples/` 图像。脚本从八个 `mini_train` 场景均衡
选择 32 帧六相机图像，保留两个 `mini_val` 场景用于后续验证；输出 FP32
`[1,6,3,928,1600]` 的 `.npy` 与记录来源、哈希和预处理的 `calibration.json`。
使用 BGR 均值归一化后底部补零，不缩放或增强图像。
本次已生成 32 帧（192 张相机图像），每个 mini_train 场景 4 帧。来源与处理证据见
[准备摘要](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/preparation-summary.json)、
[校准 manifest](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/calibration.json) 和
[数据集记录](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/dataset.json)。

真实图像不会替换当前 ONNX 中已经固定的 synthetic all-visible 投影和零 CAN-bus。
因此这只能为当前固定几何的 cold 图准备校准数据，不能据此宣称真实几何、时序推理
或 nuScenes mAP/NDS 已验证。旧完整图已完成 32 帧校准并写出缓存，随后构建失败；见
[校准摘要](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/full-calibration-summary.json)。
失败 Conv 的原始 shape、权重与真实校准范围用于 GPU 对照：保留 Identity 别名时失败，
直接 initializer 时成功，见 [复现结果](../../docs/validation/tensorrt/bevformer-base-2026-10-17/nuscenes-mini/conv-weight-alias-probe.json)。
适配器现已规范化这些常量引用；新模型 SHA 已变化，旧缓存不能重标或直接复用。命令和范围见
[MIXED_PRECISION.md](MIXED_PRECISION.md)。

另有 [backbone_calibration.py](backbone_calibration.py) 和
[calibration_ranges.py](calibration_ranges.py) 作为低显存备用：提取保留原 462 个待量化
卷积的六路 backbone，再将它自己的校准缓存转换成独立范围产物。结构提取已通过 CPU
测试；小 CNN 的首次校准、缓存复用和显式范围构建均已通过 GPU 测试，缓存与范围路径
输出最大绝对差为 0，见 [范围接口结果](../../docs/validation/tensorrt/bevformer-base-2026-10-17/mixed-precision-ranges-interface/summary.json)。
修复后的 Base backbone 已完成真实 32 帧校准及构建；完整图仍未生成可执行引擎，不能报告整网 FPS。
子图缓存不能冒充整网缓存，备用方案也不能保证完整 engine 适合 8GB 显存。

## 性能验收边界

完整 Base 的 FP32/FP16 与混合 INT8 数值对照仍需验证。目前已有 87 项 CPU
测试（含 19 项适配器、8 项 nuScenes 预处理、20 项子图提取、18 项范围转换测试）和三层小 CNN 的
GPU 测试，确认了 8 批校准、缓存复用、显式范围构建，以及实际 engine 中的
FP16 / INT8 / FP16 计算格式。完整模型的逐层约束表可审查，但不是实际 engine 精度报告。

验收应分别记录 engine 构建时间、预热、样本数、GPU 温度/频率、设备执行延时和
包含搬运/预后处理的整链延时，cold 与连续帧分别报告。当前工具记录的显存数据是
初始化/分配后的快照和 context 需求，不是峰值；整网峰值仍需实测。
早期整网 OOM 日志来自会初始化 PyTorch CUDA 的旧 builder；现已改用 CUDA Driver
直接初始化，尚无证据说明这一改动解决了完整 Base 的显存不足。

上游 RTX3090、batch1 的 Base half2 FP16 为 6.6 FPS，混合 FP16/INT8 为 8.6 FPS；
这些是公开参考，**不是本项目在 RTX3070 Ti Laptop 上的测量结果**。
其计时包含提交与同步，排除 H2D/D2H 和预后处理。

环境与测试证据保存在
[`docs/validation/tensorrt/bevformer-base-2026-10-17/`](../../docs/validation/tensorrt/bevformer-base-2026-10-17/)。
测试机目前为 `192.168.3.33`；旧地址 `.24` 的连接失败不代表当前环境状态。
