# UniAD 2.0 FFN 编译器原型

## 当前状态与范围

- 开发分支：`lg/uniad-mlir-ptx-opt`
- 当前编译单元：UniAD 2.0 `stage2_e2e` 的一个 BEV encoder FFN。
- 静态输入：`float32 [1, 40000, 256]`，其中 `40000 = 200 × 200` 个 BEV token。
- 中间宽度：`512`；输出：`float32 [1, 40000, 256]`。
- 推理语义：`MatMul(256→512) + Bias + ReLU → MatMul(512→256) + Bias + Residual + LayerNorm`。配置中的 dropout 在推理模式下为恒等操作。
- `--layer` 可选择 encoder 的第 0～5 层；一次导出和编译只处理其中一层。

这个原型用于验证 UniAD 子图的权重导出、编译、封装、正确性和 GPU 性能测量。它**不是完整 UniAD 编译器**：BEV attention、deformable attention、tracking、mapping、motion、occupancy 和 planning 等模块仍由原有 PyTorch/MMCV 流程执行，因此 FFN 编译结果不能作为完整 UniAD 的端到端延迟、吞吐量或精度数据。本文同时记录了一组完整 UniAD 的 native PyTorch 端到端基线，用于后续整网接入时比较；该基线没有执行任何已编译子图。

## MLIR / `mlir-opt` 文件在哪里？

**`mlir-opt` 不在 General-AI-Compiler 仓库中，也不是 GAC 直接调用的动态库。** 它是项目外部 LLVM/MLIR 源码构建得到的命令行程序。GAC 通过 [`mlir_toolchain.py`](../src/gac/mlir_toolchain.py) 启动该程序，将 GPU Dialect MLIR 离线降低到 NVVM 和 SM86 PTX。二进制与构建目录依赖平台和 LLVM revision，所以不提交到 Git。

GitHub 上完整的 BEVFormer MLIR/PTX 实现位于 [`lg/bevformer-mlir-ptx-opt`](https://github.com/luogantt/General-AI-Compiler/tree/lg/bevformer-mlir-ptx-opt) 分支；默认 `main` 当前不含这些文件。当前机器用于正式验证的二进制是 `/home/thomas/gac-mlir-build-lf/bin/mlir-opt`，其外部 LLVM 源码入口是 `C:\Users\Thomas\Desktop\universal_ai_compile\llvm-project\mlir\tools\mlir-opt\mlir-opt.cpp`。

- [`tools/build_mlir_wsl.sh`](../tools/build_mlir_wsl.sh) 从本地 `llvm-project` checkout 构建 `mlir-opt`、`mlir-translate` 和 `llc`。默认稳定路径为 WSL 的 `$HOME/.cache/gac-mlir/current-build/bin/mlir-opt`。
- [`src/gac/backends/mlir_ptx.py`](../src/gac/backends/mlir_ptx.py) 实现 BEVFormer 固定 FFN 的全 PTX lowering；[`src/gac/backends/mlir_cublas_ptx.py`](../src/gac/backends/mlir_cublas_ptx.py) 实现 BEVFormer/UniAD 固定 FFN 的 cuBLAS + MLIR PTX lowering。
- Git 跟踪的输入示例是 [`mlir_sm86_add_relu.mlir`](../examples/mlir_sm86_add_relu.mlir) 和 [`mlir_sm86_layernorm_fusion.mlir`](../examples/mlir_sm86_layernorm_fusion.mlir)。保存的实际 lowering 样例是 [`production-ffn.gpu.mlir`](validation/sm86/mlir-ptx-2026-10-11/production-ffn.gpu.mlir)、[`production-ffn.binary.mlir`](validation/sm86/mlir-ptx-2026-10-11/production-ffn.binary.mlir) 和 [`production-ffn.ptx`](validation/sm86/mlir-ptx-2026-10-11/production-ffn.ptx)。
- UniAD 编译生成的 `.mlir`、`.ptx` 和 manifest 默认位于 `build/uniad/`，该目录被 Git 忽略。离线 artifact 部署后不需要 `mlir-opt` 或 NVRTC；CUDA Driver 仍会在模块加载时把 textual PTX JIT 为设备 SASS。

## 两阶段实现

### 第一阶段：CUDA/HXP + cuBLAS

导出器从可信的 UniAD checkpoint 中提取指定 encoder 层的两组全连接权重、bias 和 LayerNorm 参数，转成固定布局的 NumPy 数据，并生成带 SHA-256 校验的数据包。CUDA backend 将同一 FFN 图编译成 HXP 包；运行时把两个 MatMul 明确交给 CuPy/cuBLAS，其余算子使用生成的 CUDA 调度。编译报告同时记录输入契约、后端选择、dispatch 数量和产物摘要，便于复现与审计。

这一阶段是后续优化的正确性与性能基线。基准程序会检查运行时实际选择了 `cupy-cublas`，不会把 CUDA MatMul fallback 的结果当作 cuBLAS 基线。

### 第二阶段：MLIR 离线 PTX 与候选选择

同一个 `[1,40000,256] → 512 → 256` FFN 现在支持三条 MLIR 路径：

| backend | MatMul | 融合阶段 | dispatch | 部署时要求 |
| --- | --- | --- | ---: | --- |
| `mlir-cuda-sm86` | 2× CuPy/cuBLAS | 2× NVRTC CUDA Kernel | 4 | CUDA Driver、CuPy/cuBLAS、NVRTC |
| `mlir-cublas-ptx-sm86` | 2× CuPy/cuBLAS | 2× `mlir-opt` 离线生成的 PTX Kernel | 4 | CUDA Driver、CuPy/cuBLAS；不需要 `mlir-opt`/NVRTC |
| `mlir-tiled-ffn-sm86` | 2× 单缓冲 shared-memory tiled PTX，其中 FC1 融合 Bias+ReLU | 1× Bias+Residual+LayerNorm PTX | 3 | CUDA Driver、CuPy PTX loader；不需要 `mlir-opt`/NVRTC |

`mlir-cublas-ptx-sm86` 的四次调用依次是 cuBLAS FC1、PTX `Bias+ReLU`、cuBLAS FC2、PTX `Bias+Residual+LayerNorm`。`mlir-tiled-ffn-sm86` 把前两次调用融合为 tiled FC1+Bias+ReLU，并用另一个 tiled PTX Kernel 执行 FC2。两种路径都保存 GPU MLIR、lowered binary MLIR、textual PTX、调用计划、工具链 provenance 和 SHA-256 manifest；Runtime 在 CUDA 初始化前重建规范计划并验证 entry、指针 ABI、shape、target 与摘要。

`mlir-tiled-ffn-sm86` 是可执行 autotune 候选，不保证比 cuBLAS 更快。autotuner 会先检查每条真实 Session 的输出，再以 CUDA Event resident 中位数选择候选；不能执行的 tile 或仅有 metadata 的候选不会进入计时。

## 导出、编译和部署环境

UniAD 与编译器依赖不同，建议严格分开环境，通过数据包交换权重和可选的 expected sample。

| 环境 | 用途 | 主要要求 |
| --- | --- | --- |
| UniAD 导出环境 | 读取 checkpoint、提取权重、可选用 Torch eager 生成 expected sample | Python 3.9、PyTorch 2.0.1+cu118、CUDA 11.8 运行栈 |
| GAC 离线编译环境 | 从 bundle 生成 HXP、GPU MLIR、binary MLIR 和 PTX artifact | Python 3.11+；外部 LLVM/MLIR `mlir-opt`；编译步骤本身不要求 GPU |
| GAC SM86 部署/基准环境 | 校验 artifact 并运行 FFN | NVIDIA SM86、CUDA Driver、Python 3.11+、CuPy；cuBLAS+PTX 路径需要 cuBLAS，离线 PTX 路径不需要 `mlir-opt` 或 NVRTC |

UniAD adapter 现在直接随 GAC 发布在 `src/gac/uniad`，不再依赖 UniAD 仓库中的 `projects/uniad_compiler`。GAC 的 Python 版本要求是 3.11+，因此不要把完整 GAC wheel 安装进官方 UniAD 环境；在 UniAD Python 3.9 环境中直接运行本仓库的 checkout wrapper 即可。该 wrapper 只载入轻量的 `gac.uniad` 数据合同与 NumPy，不会载入 CUDA 12 Runtime。

## 使用方法

以下命令均从 General-AI-Compiler 仓库根目录执行。生成物放在已被 `.gitignore` 排除的 `build/uniad/` 下；`/path/to/UniAD-2.0` 表示独立的外部 UniAD checkout。

没有 checkpoint 时，可以先生成与生产形状完全相同的确定性合成 bundle，仅用于编译和性能链路验证：

```bash
python tools/uniad_make_synthetic_bundle.py build/uniad/synthetic-layer0.uniadffn --seed 20261011
```

1. 在 UniAD Python 3.9 环境导出第 0 层 FFN。导出器优先使用 `weights_only=True`；`--trusted-checkpoint` 只在旧版 Torch 不支持该安全加载参数时允许 pickle 回退，因此只应对来源可信的本地 checkpoint 使用。`--with-synthetic-golden` 是兼容保留的参数名；它会额外生成确定性的 `[1,40000,256]` 输入及 Torch eager expected output，未压缩数据约 82 MiB；不需要该数据时可省略此参数。

   ```bash
   python tools/uniad_export_ffn.py /path/to/UniAD-2.0/ckpts/uniad_base_e2e.pth build/uniad/layer0.uniadffn --layer 0 --trusted-checkpoint --with-synthetic-golden
   ```

2. 构建一次外部 MLIR 工具。下面是 Windows PowerShell + WSL 方式；脚本不会修改外部 `llvm-project` checkout，产物位于 WSL 用户缓存中。

   ```powershell
   wsl -d Ubuntu-24.04 -- bash tools/build_mlir_wsl.sh
   $mlirOpt = (wsl -d Ubuntu-24.04 -- bash -lc 'realpath "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"').Trim()
   ```

3. 在 Windows 侧的 GAC Python 3.11+ 环境生成 HXP、旧 MLIR/NVRTC call plan、cuBLAS+PTX 和 tiled PTX 四种产物。

   ```powershell
   python tools/uniad_compile_ffn.py `
     build/uniad/layer0.uniadffn build/uniad/layer0-compiled `
     --backend all --tile 16,16,16 `
     --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt
   ```

   命令直接在 WSL/Linux 中运行时，不传 `--wsl`，把 `--mlir-opt` 指向 Linux 可执行文件：

   ```bash
   python3 tools/uniad_compile_ffn.py \
     build/uniad/layer0.uniadffn build/uniad/layer0-compiled \
     --backend all --tile 16,16,16 \
     --mlir-opt "$HOME/.cache/gac-mlir/current-build/bin/mlir-opt"
   ```

   `--backend both` 继续表示旧 `cuda-hxp + mlir-cuda-sm86` 两条路径；`--backend all` 才包含 `mlir-cublas-ptx-sm86` 和 `mlir-tiled-ffn-sm86`。编译完成后的两个 PTX artifact 目录可复制到部署机；加载它们不需要 `mlir-opt` 或 NVRTC。

   如果只生成两种新 PTX artifact，正式 benchmark 工具也提供不访问 CUDA 的编译模式：

   ```powershell
   python tools/benchmark_mlir_cublas_ptx_ffn.py `
     --uniad-bundle build/uniad/layer0.uniadffn --seed 20261011 `
     --emit-artifact-dir build/uniad/layer0-compiled/mlir-cublas-ptx-sm86 `
     --emit-tiled-artifact-dir build/uniad/layer0-compiled/mlir-tiled-ffn-sm86 `
     --compile-only --wsl --wsl-distribution Ubuntu-24.04 --mlir-opt $mlirOpt `
     --output build/uniad/layer0-compile.json
   ```

4. 在 SM86 GPU 上用已编译 artifact 运行三候选配对基准和 autotune。若 bundle 含 expected sample，工具使用 bundle 提供的 expected output；官方 exporter 可用 Torch eager 生成该样本。bundle 不含 expected sample 时，工具使用同一图的 GAC NumPy reference。每个 repeat 会交换 PTX/NVRTC 的测量顺序。

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

   正式对比应再启动一个独立进程，把 `--benchmark-order` 改成 `nvrtc-first`，并使用新的 cache 和 output 文件。`--repeats` 必须为偶数。报告区分 resident GPU 时间与包含校验、H2D、计算、D2H 和同步的 public `run()` 时间；正确性阈值为 `rtol=1e-4`、`atol=1e-5`。

旧 `tools/uniad_benchmark_ffn.py` / `gac-uniad-benchmark-ffn` 仍只比较 `cuda-hxp` 与 `mlir-cuda-sm86`/NVRTC，用于历史基线，不会测量新离线 PTX artifact。

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

最终 v3 报告在 RTX 3070 Ti Laptop（SM86）上使用真实 `uniad_base_e2e` checkpoint 的 layer-0 权重。bundle 不含 expected sample；输入是 seed `20261011` 的确定性 synthetic query，正确性基准是 GAC NumPy reference。两个独立进程分别以 PTX-first 和 NVRTC-first 启动，均执行 warmup 10、6 个 repeat、每个 repeat 30 次 FFN，并采集 3 个 public `run()` 样本。

| 初始顺序 | cuBLAS+MLIR PTX resident | cuBLAS+NVRTC resident | tiled PTX resident | cuBLAS+MLIR PTX public wall | cuBLAS+NVRTC public wall | resident 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PTX-first | 3.050785 ms | 3.150221 ms | 13.932249 ms | 20.710336 ms | 20.564843 ms | 1.035303× |
| NVRTC-first | 3.076282 ms | 3.182030 ms | 13.936361 ms | 21.537857 ms | 21.664099 ms | 1.032900× |

两个独立进程的 resident primary estimator 取几何平均为 **1.034101×，即约 3.41% 优势**。public wall 包含参数检查、H2D、计算、D2H 和同步；两种初始顺序下 PTX 与 NVRTC 相差都不到 `0.15 ms`，且领先方向不同，因此 public wall 基本持平。三个候选均通过 `rtol=1e-4`、`atol=1e-5` 的 `allclose`：cuBLAS+PTX 和 NVRTC 的最大绝对误差均为 `3.337860107421875e-6`，tiled PTX 为 `3.0994415283203125e-6`。两次 autotune 都选择 `mlir-cublas-ptx-sm86`。

这些数据只代表一个 FP32 `[1,40000,256] → 512 → 256` encoder layer-0 FFN。真实的是 checkpoint 权重，query 仍为 synthetic，NumPy 一致性也不等于真实 nuScenes 数据集精度。结果不包含 attention、tracking、mapping、motion、occupancy、planning、数据加载或整网 host 工作，不能换算成完整 UniAD 的端到端延迟或 FPS。完整逐 repeat 样本、源码 SHA-256、checkpoint provenance 和 artifact manifest 见 [`validation/sm86/uniad/mlir-ptx-2026-10-11`](validation/sm86/uniad/mlir-ptx-2026-10-11/README.md)。

#### 迁移前历史基线

已经在 RTX 3080 / SM86 上使用上述 seed `20261011` 的 bundle 完成**生产形状、合成权重和合成输入**的真机验证。两个独立进程分别改变首次后端顺序，得到下列 resident GPU 中位数：

| 首次顺序 | MLIR（4 calls） | CUDA/HXP + cuBLAS（6 dispatches） | MLIR 加速比 |
| --- | ---: | ---: | ---: |
| MLIR first | 1635.330 µs | 2074.368 µs | 1.26440× |
| CUDA first | 1645.388 µs | 2075.340 µs | 1.26061× |

两条路径的最大绝对误差都是 `9.5367431640625e-7`。这是单层 FFN 的设备常驻计算时间；包含约 41 MB 输入 H2D 和约 41 MB 输出 D2H 的 public `run()` wall time约为 34 ms，不能和 resident 数字混用。原始 JSON、逐 repeat 顺序、源文件摘要和协议说明见 [`validation/sm86/uniad/ffn-2026-10-11`](validation/sm86/uniad/ffn-2026-10-11/README.md)。

上表使用生产形状的合成权重和合成输入，只代表迁移前的单层 FFN。它采用 RTX 3080、旧 GAC `c494d9e` 和旧 HXP/NVRTC 路径，与上面的 RTX 3070 Ti Laptop v3 在设备、权重、源码和后端上都不同，不能直接比较，也不能换算成完整 UniAD 帧率。

这两组 JSON 是迁移前在 UniAD adapter 仓库中生成的历史证据，内含当时 UniAD 与固定 GAC `c494d9e` 的源码哈希。迁移时保持原始 JSON 和 `SHA256SUMS` 不变，不能把其中的 Git/source provenance 解释为当前分支的 v3 测量。

## 下一步优化

- 使用 cuBLASLt epilogue 融合第一层 MatMul、bias 和 ReLU，把当前四调用路径进一步压缩为三调用路径。
- 扩展生产尺寸 Tensor Core 与双缓冲 tiled MatMul，并继续让正确性门和 autotune 决定实际候选，而不是按 backend 名称预设赢家。
- 通过 DLPack 接入 PyTorch tensor，并在 PyTorch current stream 上执行，避免不必要的 H2D/D2H 拷贝、额外 stream 和同步。
- 在单层 FFN 结果稳定后逐步覆盖 encoder 多层及 attention 子图，最后再讨论整网端到端性能。
