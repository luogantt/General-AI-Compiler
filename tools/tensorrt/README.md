# BEVFormer TensorRT 实验入口

分支：`lg/bevformer-tensorrt`，基于原生 CUDA/Vortex 优化提交 `baac4de`。
本分支先建立完整 BEVFormer 的 TensorRT 对照路线，用于比较数值与性能。
截至 2026-10-17，已取得并固定两份上游源码、核对入口和接口，检查 TRT10 补丁能否应用。
**尚未构建 TensorRT engine，也没有本机 TensorRT 延时结果。**

## 选用的开源实现

| 来源 | 已核实的内容 | 本项目用途 |
|---|---|---|
| [DerryHub/BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt) | 完整 Tiny/Small/Base、导出器、自定义 CUDA 插件、TensorRT 构建和 nuScenes 评估 | Base FP16 对照的主要实现 |
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
官方模型配置 + checkpoint + 图像/历史 BEV/标定
    ↓ 上游 PyTorch 模型和 TensorRT 专用导出包装
ONNX（包含插件节点）
    ↓ TensorRT Parser + 已注册的 C++/CUDA 插件
TensorRT Builder：融合、精度与 tactic 选择
    ↓
TensorRT engine
    ↓ TensorRT Runtime + CUDA
GPU 推理 → 数值检查、nuScenes 指标与延时
```

这个实验使用 TensorRT 自身的编译器与 Runtime。现有 `native/` 的 C++ MLIR
编译链保留独立，未增加 `gac-compile --target=tensorrt`，也未宣称 TensorRT
执行了我们的 MLIR Pass。后续统一输入和报告格式后，才能进行公平比较。

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

因此先用上游完整 Base 和插件重新导出，确认基本正确性，再对齐我们的 cold 输入、
标定、checkpoint、相机分批策略和三路 raw-head 输出。不能仅改 ONNX 算子名称，
也不能把历史状态置零就假定两条计算图数值一致。

## 性能验收边界

先建立 Base FP32/FP16 的数值对照，再考虑有真实校准数据的 INT8。
分别记录 engine 构建时间、显存峰值、预热、样本数、GPU 温度/频率、
设备执行延时和包含搬运/预后处理的整链延时。cold 与连续帧应分别报告。

上游 RTX3090、batch1 的 Base half2 FP16 为 6.6 FPS，混合 FP16/INT8 为 8.6 FPS；
这些是公开参考，**不是本项目在 RTX3070 Ti Laptop 上的测量结果**。
其计时包含提交与同步，排除 H2D/D2H 和预后处理。

本轮本机检查范围：两份源码版本与入口、TRT10 patch applicability。
本轮 SSH 连接 `192.168.3.24` 超时，未安装远端依赖、未编译插件、未运行 engine。
后续完成这些步骤和数值验证后，才会新增本机 benchmark 记录。
