# UniAD 2.0 FFN 编译器原型

## 当前状态与范围

- 开发分支：`codex/uniad-cuda-mlir`
- 当前编译单元：UniAD 2.0 `stage2_e2e` 的一个 BEV encoder FFN。
- 静态输入：`float32 [1, 40000, 256]`，其中 `40000 = 200 × 200` 个 BEV token。
- 中间宽度：`512`；输出：`float32 [1, 40000, 256]`。
- 推理语义：`MatMul(256→512) + Bias + ReLU → MatMul(512→256) + Bias + Residual + LayerNorm`。配置中的 dropout 在推理模式下为恒等操作。
- `--layer` 可选择 encoder 的第 0～5 层；一次导出和编译只处理其中一层。

这个原型用于验证 UniAD 子图的权重导出、编译、封装、正确性和 GPU 性能测量。它**不是完整 UniAD 编译器**：BEV attention、deformable attention、tracking、mapping、motion、occupancy 和 planning 等模块仍由原有 PyTorch/MMCV 流程执行，因此 FFN 编译结果不能作为完整 UniAD 的端到端延迟、吞吐量或精度数据。本文同时记录了一组完整 UniAD 的 native PyTorch 端到端基线，用于后续整网接入时比较；该基线没有执行任何已编译子图。

## 两阶段实现

### 第一阶段：CUDA/HXP + cuBLAS

导出器从可信的 UniAD checkpoint 中提取指定 encoder 层的两组全连接权重、bias 和 LayerNorm 参数，转成固定布局的 NumPy 数据，并生成带 SHA-256 校验的数据包。CUDA backend 将同一 FFN 图编译成 HXP 包；运行时把两个 MatMul 明确交给 CuPy/cuBLAS，其余算子使用生成的 CUDA 调度。编译报告同时记录输入契约、后端选择、dispatch 数量和产物摘要，便于复现与审计。

这一阶段是后续优化的正确性与性能基线。基准程序会检查运行时实际选择了 `cupy-cublas`，不会把 CUDA MatMul fallback 的结果当作 cuBLAS 基线。

### 第二阶段：MLIR + cuBLAS + 融合核

MLIR backend 针对同一 FFN 生成可审查的 tensor MLIR、运行时 ABI MLIR、调用清单和 CUDA kernel 源码。当前可执行路径是经过完整性校验的 MLIR runtime-call plan，共四次 GPU 调用：

1. `gac_cublas_fc1_matmul_f32`：第一个 MatMul，调用 CuPy/cuBLAS。
2. `gac_bias_relu_inplace_f32`：融合 `Bias + ReLU`，由 NVRTC 编译 CUDA kernel。
3. `gac_cublas_fc2_matmul_f32`：第二个 MatMul，调用 CuPy/cuBLAS。
4. `gac_bias_residual_layer_norm_f32`：融合 `Bias + Residual + LayerNorm`，每行由一个 CUDA block 完成归约。

这将原始的七算子语义压缩成四次调用，并减少中间结果写回与 kernel launch。这里的 MLIR 文件描述并校验 ABI 和调用关系，实际 dispatch 由校验后的 call records 驱动；当前融合 LayerNorm kernel 由 NVRTC 编译，尚未把整段 MLIR 直接 lowering 为 NVVM/PTX。

## 两个运行环境

UniAD 与编译器依赖不同，建议严格分开环境，通过数据包交换权重和可选的 golden sample。

| 环境 | 用途 | 主要版本 |
| --- | --- | --- |
| UniAD 环境 | 读取 checkpoint、提取权重、可选生成 Torch eager golden | Python 3.9、PyTorch 2.0.1+cu118、CUDA 11.8 运行栈 |
| GAC 编译环境 | 生成 HXP/MLIR 产物并在 GPU 上执行基准 | Python 3.11、CUDA 12、CuPy/cuBLAS/NVRTC |

UniAD adapter 现在直接随 GAC 发布在 `src/gac/uniad`，不再依赖 UniAD 仓库中的 `projects/uniad_compiler`。GAC 的 Python 版本要求是 3.11+，因此不要把完整 GAC wheel 安装进官方 UniAD 环境；在 UniAD Python 3.9 环境中直接运行本仓库的 checkout wrapper 即可。该 wrapper 只载入轻量的 `gac.uniad` 数据合同与 NumPy，不会载入 CUDA 12 Runtime。

## 使用方法

以下命令均从 General-AI-Compiler 仓库根目录执行。生成物放在已被 `.gitignore` 排除的 `build/uniad/` 下；`/path/to/UniAD-2.0` 表示独立的外部 UniAD checkout。

没有 checkpoint 时，可以先生成与生产形状完全相同的确定性合成 bundle，仅用于编译和性能链路验证：

```bash
python tools/uniad_make_synthetic_bundle.py build/uniad/synthetic-layer0.uniadffn --seed 20261011
```

1. 在 UniAD Python 3.9 环境导出第 0 层 FFN。导出器优先使用 `weights_only=True`；`--trusted-checkpoint` 只在旧版 Torch 不支持该安全加载参数时允许 pickle 回退，因此只应对来源可信的本地 checkpoint 使用。`--with-synthetic-golden` 会额外生成确定性的 `[1,40000,256]` 输入及 Torch eager 输出，未压缩数据约 82 MiB；不需要该数据时可省略此参数。

   ```bash
   python tools/uniad_export_ffn.py /path/to/UniAD-2.0/ckpts/uniad_base_e2e.pth build/uniad/layer0.uniadffn --layer 0 --trusted-checkpoint --with-synthetic-golden
   ```

2. 切换到 GAC Python 3.11 / CUDA 12 环境，同时生成 CUDA/HXP 和 MLIR CUDA 产物。

   ```bash
   python tools/uniad_compile_ffn.py build/uniad/layer0.uniadffn build/uniad/layer0-compiled --backend both
   ```

3. 在 SM86 GPU 上运行配对基准。它先校验两条路径与 golden（若数据包未带 golden，则使用 GAC NumPy reference）的输出一致，再交替两种后端的测量顺序，以减少先后顺序造成的偏差。

   ```bash
   python tools/uniad_benchmark_ffn.py build/uniad/layer0.uniadffn --device 0 --warmup 10 --iterations 20 --repeats 10 --output build/uniad/layer0-benchmark.json
   ```

基准报告区分两种时间边界：resident GPU 时间保持输入、权重和 workspace 常驻显存；public `run()` 时间包含参数检查、H2D、计算、D2H 和同步。正确性阈值为 `rtol=1e-4`、`atol=1e-5`。`--repeats` 必须是偶数，才能保证两种测量顺序样本数相同。

完整 UniAD native PyTorch 基线需要真实 checkpoint 和数据集，并在 UniAD Python 3.9 环境中运行。下面的命令保留时序状态，先 warmup 10 帧，再测量 50 帧；把示例路径替换成实际数据位置。

```bash
python -B tools/uniad_benchmark_e2e.py \
  --uniad-root /path/to/UniAD-2.0 \
  /path/to/UniAD-2.0/projects/configs/stage2_e2e/base_e2e.py \
  /path/to/UniAD-2.0/ckpts/uniad_base_e2e.pth \
  --output build/uniad/uniad-e2e-rtx3080-50frames.json \
  --device 0 --warmup 10 --measure 50 --workers 0 --log-interval 5 \
  --cfg-options \
  data.test.data_root=/path/to/nuscenes \
  data.test.ann_file=/path/to/nuscenes_infos_temporal_val.pkl \
  model.motion_head.anchor_info_path=/path/to/motion_anchor_infos_mode6.pkl
```

这项基准使用 `strict=True` 加载 checkpoint，并要求 tracking、mapping、motion、occupancy、planning 五个 head 全部存在。`--uniad-root` 必须指向含 `projects/mmdet3d_plugin` 的外部 checkout；工具把这个根目录加入 `sys.path`，并相对它解析 config plugin 和数据路径。模型计时从 dataloader 生成 CPU batch 后开始，包含 `MMDataParallel` scatter/H2D、完整 `forward_test`、Python 主机工作和 CUDA 执行，并在计时前后同步 CUDA。阻塞的 `next(data_loader)` 等待单独记录；报告还给出未重叠的 `next() + model` 串行观测时间，并分别保存 GAC 与 UniAD checkout 的 Git provenance。

## 性能数据边界

### 完整 UniAD native PyTorch 基线

在 RTX 3080 / SM86 上使用真实 `uniad_base_e2e.pth` 和 nuScenes `v1.0-mini` 完成了完整 UniAD 2.0 `stage2_e2e` forward 基线。协议为 batch 1、`workers=0`、按数据集顺序执行，warmup 10 帧后测量 50 帧；测量跨 2 个场景，包含 1 个首帧和 49 个稳态帧。五个任务 head 全部执行，checkpoint 严格加载。

| 计时项 | 均值 | 中位数 | P90 | P99 | 由均值换算的吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整模型调用，50 帧 | 822.885 ms | 806.619 ms | 902.287 ms | 982.392 ms | 1.215 FPS |
| dataloader 等待，50 帧 | 2914.904 ms | 2990.312 ms | 3616.270 ms | 4350.067 ms | — |
| 串行 input pipeline + 模型，50 帧 | 3737.789 ms | 3784.887 ms | 4376.574 ms | 5303.153 ms | 0.268 FPS |

模型调用的 49 个稳态帧均值为 `824.581 ms`、中位数为 `807.731 ms`；唯一首帧为 `739.767 ms`，单一样本不足以代表通常的首帧开销。测量期 PyTorch allocator 峰值为 `2.405 GiB` allocated、`3.340 GiB` reserved。

这是一组 native PyTorch 同步 wall-time 基线，JSON 中的 `compiled_subgraphs` 为空。当前 MLIR FFN 尚未接入整网，因而这里没有 MLIR 端到端加速比。数据来自 nuScenes mini 中配置后长度为 81 的 test split，本次只测量其中 50 帧，不是完整 nuScenes validation，也不包含精度评测。完整逐帧数据、日志、输入/源码 SHA-256 和复核说明见 [`validation/sm86/uniad/e2e-native-nuscenes-mini-2026-10-11`](validation/sm86/uniad/e2e-native-nuscenes-mini-2026-10-11/README.md)。

### BEV encoder 单层 FFN 编译基准

已经在 RTX 3080 / SM86 上使用上述 seed `20261011` 的 bundle 完成**生产形状、合成权重和合成输入**的真机验证。两个独立进程分别改变首次后端顺序，得到下列 resident GPU 中位数：

| 首次顺序 | MLIR（4 calls） | CUDA/HXP + cuBLAS（6 dispatches） | MLIR 加速比 |
| --- | ---: | ---: | ---: |
| MLIR first | 1635.330 µs | 2074.368 µs | 1.26440× |
| CUDA first | 1645.388 µs | 2075.340 µs | 1.26061× |

两条路径的最大绝对误差都是 `9.5367431640625e-7`。这是单层 FFN 的设备常驻计算时间；包含约 41 MB 输入 H2D 和约 41 MB 输出 D2H 的 public `run()` wall time约为 34 ms，不能和 resident 数字混用。原始 JSON、逐 repeat 顺序、源文件摘要和协议说明见 [`validation/sm86/uniad/ffn-2026-10-11`](validation/sm86/uniad/ffn-2026-10-11/README.md)。

上表仍使用生产形状的合成权重和合成输入，只代表单层 FFN。真实 checkpoint 已用于前述 native PyTorch 整网基线，但尚未用于 FFN 编译路径的真实权重 golden，也尚未完成完整 UniAD 精度评测或 MLIR 整网接入。FFN 数字不能换算成完整 UniAD 帧率；两组结果都不是 RTX 3070 的设备实测。

本节两组 JSON 是迁移前在 UniAD adapter 仓库中生成的历史证据，内含当时 UniAD 与固定 GAC `c494d9e` 的源码哈希。迁移时保持原始 JSON 和 `SHA256SUMS` 不变，不能把其中的 Git/source provenance 解释为当前 GAC 分支的新测量。迁移后的 CPU 编译 smoke 已复现相同的生产形状 bundle SHA-256 `551788d2...`、HXP SHA-256 `cb4bae73...` 和 MLIR program integrity SHA-256 `9dbe3a20...`；新的真机性能结论需要在当前 GAC commit 上重新运行并生成新报告。

## 下一步优化

- 使用 cuBLASLt epilogue 融合第一层 MatMul、bias 和 ReLU，把当前四调用路径进一步压缩为三调用路径。
- 将融合 LayerNorm 从 NVRTC CUDA 源码迁移到 MLIR GPU/NVVM lowering，并产出可缓存、可验证的 PTX。
- 通过 DLPack 接入 PyTorch tensor，并在 PyTorch current stream 上执行，避免不必要的 H2D/D2H 拷贝、额外 stream 和同步。
- 在单层 FFN 结果稳定后逐步覆盖 encoder 多层及 attention 子图，最后再讨论整网端到端性能。
