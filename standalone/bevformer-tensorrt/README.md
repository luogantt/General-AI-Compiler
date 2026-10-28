# BEVFormer TensorRT

独立的 BEVFormer-base 推理项目：从 ONNX 构建 TensorRT engine，执行六相机推理、混合精度校准和性能分析。无需安装 GAC、MLIR、Vortex、MMCV 或 MMDetection。

**最新结果（2026-10-22）：RTX 3070 Ti Laptop 8 GB，完整固定图混合 INT8/FP16 推理约 309.42 ms/帧（3.23 FPS）。** 第三轮 DCNv2 优化的同轮对照为 325.97 → 309.42 ms，延时降低 **5.07%**；三个输出与基线及官方插件参考逐位一致。[第三轮报告与复现](docs/validation-dcn-round3-3070.md)

前两轮 MSDA 优化分别降低 18.05%、2.60%；各轮比例不相加。[首轮](docs/validation-msda-3070.md) · [第二轮](docs/validation-msda-round2-3070.md)

## 我们做了什么

- 打通 ResNet-101＋DCNv2 → FPN → BEV Encoder → 检测 Decoder → 检测头的完整固定图推理。
- 提供 FP32 和混合 INT8/FP16 构建、输入输出检查、真实图像校准、层级耗时报告。
- 实现可选的 TensorRT MSDA 插件：每线程处理四个相邻通道，复用坐标、插值系数和地址，用 `half2` 成对计算；实测选择线程块和形状专用参数。保持模型权重、采样点数和数学计算不变。
- 第二轮增加安全 32 位索引；当前六个 Spatial MSDA 的实测配置为 `channels_per_thread=4, block_threads=128, specialize=3, prefetch=0`。[配置与复测](docs/msda-optimization.md#7-继续优化与已经优化的配置比较)
- 第三轮增加可选 DCNv2 插件：3×3 采样每线程处理四个通道，复用 offset/mask 和插值计算，矩阵乘仍用 cuBLAS。当前 156 个静态 FP16 节点配置为 `block_threads=128, specialize=4`。[构建](native/dcn/README.md) · [启用命令](docs/validation-dcn-round3-3070.md#代码与复现)

```text
BEVFormer ONNX（包含权重）
        ↓ adapt：适配 TensorRT 插件接口
已适配 ONNX + 校准范围
        ↓ build：在目标 GPU 构建
TensorRT engine
        ↓ run：输入预处理后的六相机张量
BEV 特征 + 分类 logits + 3D 框编码 + 测速报告
```

当前图固定形状、相机投影和 CAN-bus 元数据，无历史帧；输出为检测头张量。真实视频时序输入、框解码和 nuScenes mAP/NDS 评估尚未包含。

## 启动推理

以下 Bash 命令从本目录执行。目标平台为 **Linux x86_64＋NVIDIA GPU**；已验证 Python 3.11、TensorRT **10.13.3.9**、PyTorch **2.7.0+cu126**、驱动 580.178.04。

### 1. 安装环境

```bash
bash scripts/setup.sh --python python3.11 --install-torch
source .venv/bin/activate
python -m bevformer_trt check --report results/environment.json
```

普通 Python 依赖可加 `--mirror tsinghua` 或 `--mirror ustc`。已有匹配的 CUDA Torch 环境时，激活后执行：

```bash
python -m pip install -r requirements.txt -r requirements-tensorrt.txt
python -m pip install --no-deps -e .
```

默认使用 NVIDIA 自带插件，无需 `nvcc`；可选优化插件需要 CUDA Toolkit 和 TensorRT 头文件。`check` 只验证环境、插件和小网络。

### 2. 准备模型资产

将配套模型包解压到本目录；源码仓库不包含大模型权重和数据。

```text
models/bevformer-base/
├── adapted.onnx          # 已适配的完整 ONNX，包含权重
├── ranges.json           # 与该模型绑定的真实图像校准范围
├── images.npy            # 合成输入 [1,6,3,928,1600]
├── reference.npz         # 同一合成输入的 PyTorch 参考输出
├── export-report.json
├── fixed-metadata.json
└── manifest.json         # 来源、大小和 SHA-256
```

`images.npy` 用于数值回归，不能用来重新校准或评估检测精度。真实输入需符合 [模型与数据契约](docs/model-and-data.md)。更换形状、权重或时序输入，需要重新导出和验证。

### 3. 构建并运行混合精度引擎

```bash
export CUDA_MODULE_LOADING=LAZY NVIDIA_TF32_OVERRIDE=0

python -m bevformer_trt build \
  --onnx models/bevformer-base/adapted.onnx \
  --engine results/mixed/base.engine --precision mixed-int8 \
  --calibrated-ranges models/bevformer-base/ranges.json \
  --workspace-mib 1024 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0 \
  --report results/mixed/build.json

python -m bevformer_trt run \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output results/mixed/run --warmup 5 --repeat 20
```

这条基础命令仍使用 NVIDIA 官方 MSDA。**要复现 18.05% 的优化，需要另行编译自定义插件、改写六个节点，并在构建和运行时传入 `--plugin-library`**，完整命令见 [MSDA 优化流程](docs/msda-optimization.md)。

混合精度让 backbone 普通卷积使用 INT8，入口、DCNv2 offset/mask、FPN 和检测头等保留浮点；公开 I/O 为 FP32。FP32 基线将构建参数换成 `--precision fp32 --plugins-fp32 --workspace-mib 4096`，移除 `--calibrated-ranges`，并使用新的输出路径。运行时可加 `--reference models/bevformer-base/reference.npz` 验证同一输入。[精度与校准规则](docs/precision-and-calibration.md)

Engine 应在目标 GPU 和软件环境构建。工作区参数不是进程总显存上限；所有输出使用新路径，程序拒绝覆盖已有结果。

### 4. 查看输出

`results/mixed/run/outputs.npz` 保存张量，`report.json` 保存环境、文件哈希和逐次延时；`build.layers.json` 保存实际层、布局和 tactic。

| 输出 | 形状 | 含义 |
|---|---|---|
| `bev_embed` | `[40000,1,256]` | 200×200 BEV 特征 |
| `all_cls_scores` | `[6,1,900,10]` | 六层 Decoder 的分类 logits |
| `all_bbox_preds` | `[6,1,900,10]` | 六层 Decoder 的框编码 |

未传 `--reference` 时数值验证状态为 `not_checked`，执行成功不代表检测精度达标。

## 性能与验收

**2026-10-21，RTX 3070 Ti Laptop 8 GB，TensorRT 10.13.3.9，同一真实六相机输入。**

| 测试范围 | NVIDIA 官方插件 | 自定义 MSDA 插件 | 收益 |
|---|---:|---:|---|
| 单个 Spatial MSDA | 38.9508 ms | 16.9093 ms | 逐对加速比中位数 **2.30 倍** |
| 完整固定图，混合 INT8/FP16 | 413.774 ms | 339.086 ms | 延时下降 **18.05%** |
| 完整图延时倒数 | 2.42 FPS | 2.95 FPS | 加速约 1.22 倍 |

整网按官方—优化—优化—官方顺序测量，每轮预热 5 次、测量 20 次，每侧合计 40 次样本。复用 timing cache 后检查其他层的 tactic 和格式无变化；三个整网输出逐位一致。单算子使用 CUDA Graph replay，整网没有；两种计时不可直接相加。

计时不含预处理、H2D/D2H、加载、构建和框解码。GPU 未锁频，温度和频率会影响绝对延时。结果仅覆盖所测真实输入，**没有完成整个数据集的 mAP/NDS 验收，也不表示混合精度对 FP32 的精度差异已解决**。[完整验证与证据](docs/validation-msda-3070.md)

历史 FP32/混合精度数据见 [2026-10-20 独立项目验证](docs/validation-3070.md) 和 [RTX 3090 历史记录](docs/historical-results.md)。不同设备、输入及运行条件的结果分开记录。

## 代码与工具

| 入口 | 作用 |
|---|---|
| [cli.py](src/bevformer_trt/cli.py) / [engine.py](src/bevformer_trt/engine.py) | 命令入口、构建、加载和推理 |
| [msda.py](src/bevformer_trt/msda.py) | 节点改写、校准范围派生与结构验证 |
| [native/msda](native/msda/README.md) | C++ TensorRT 插件和 CUDA Kernel |
| [capture_msda.py](scripts/capture_msda.py) / [benchmark_msda.py](scripts/benchmark_msda.py) | 捕获算子输入、参数搜索、GPU 数值和耗时对比 |

```bash
python -m bevformer_trt --help
python -m unittest discover -s tests -v
python -m bevformer_trt profile \
  --engine results/mixed/base.engine \
  --inputs models/bevformer-base/images.npy \
  --output-dir results/profile --mode layers --warmup 5 --repeat 20
```

CPU 测试检查工具逻辑；GPU 对照验证 Kernel 数值和性能。Profiler 有额外开销，层级报告不能直接代替正常推理延时。
