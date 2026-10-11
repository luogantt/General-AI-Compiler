# UniAD layer-0 FFN：SM86 MLIR/PTX 真机验证（2026-10-11）

本目录保存最终 **v3** 验证证据。被测单元是 UniAD 2.0 `stage2_e2e` 中一个 BEV encoder layer-0 FFN：FP32 `[1,40000,256] -> 512 -> 256`，语义为两次 MatMul、Bias、ReLU、Residual 和 LayerNorm。权重来自真实 `uniad_base_e2e` checkpoint；query 是 seed `20261011` 生成的确定性 synthetic input。导出的 bundle 不含 expected sample，正确性基准是 GAC NumPy reference。

这是单个 encoder layer 的 FFN 编译子图，不是完整 UniAD。结果不包含 attention、tracking、mapping、motion、occupancy、planning、数据加载或整网 host 工作，不能换算成完整 UniAD 的端到端延迟、FPS 或模型精度。

## 结论

- `mlir-cublas-ptx-sm86` 在两个独立进程中都通过 `rtol=1e-4`、`atol=1e-5` 正确性门。
- 两份最终报告的 resident primary speedup 分别为 `1.035303285575289x`（PTX-first）和 `1.0328999749855314x`（NVRTC-first）。几何平均为 `1.0341009321014822x`，即 **resident CUDA Event 延迟约快 3.41%**。
- 两个顺序的 public wall 中位数再取中位数后，cuBLAS+PTX 为 `21.12409649998881 ms`，NVRTC 为 `21.114470961038023 ms`，差异约 `0.046%`。因此 **public wall 基本持平**，不能据此宣称端到端加速。
- `mlir-tiled-ffn-sm86` 数值正确，但当前单缓冲 `16x16x16` FP32 tiled MatMul 明显更慢。两个顺序下 resident 中位数分别是 cuBLAS+PTX 的 `4.566775159582117x` 和 `4.530261672626344x`。
- 两次 autotune 都执行三个候选的正确性检查和计时，并选择 `mlir-cublas-ptx-sm86`。

## 权重与源码 provenance

真实 checkpoint 与 bundle 本身没有提交到 Git；本目录只保存摘要和来源记录：

| 项目 | 值 |
| --- | --- |
| checkpoint 来源 | local UniAD 2.0 checkpoint，layer `0` |
| checkpoint 文件名 | `uniad_base_e2e.proxy2.pth` |
| checkpoint SHA-256 | `0ad0c2f5dc9788a41c313305779ea49346aeb742d1f6bb5ad25c46f9beffc990` |
| checkpoint 大小 | `996345538` bytes |
| bundle SHA-256 | `a27bc607c4c7b03e8b95d3b60c42b44b74c200110ad745cf37e4f3e7382e1b41` |
| bundle expected sample | `null`；报告使用 synthetic query 与 NumPy reference |
| 权重转换 | Torch Tensor -> Python list -> NumPy；provenance 记录为精确保留 FP32 值 |
| Git branch / commit | `codex/uniad-mlir-ptx-opt` / `f77fed791cef4602ccff68181f32c7af831a8d3e` |
| Git 状态 | `dirty: true`；以 source snapshot 的逐文件 SHA-256 为实际被测源码依据 |
| source snapshot | `80` 个源码文件；规范化 source map SHA-256 `8a0a4f2f62a36b9bf7edcbfc2ae4c9638ff09deb1c415b10899031b8b6a21ed1` |
| 远端冻结 archive | SHA-256 `e7bc37a72ec3e138978feb3550ed23d93c09c7e3cdd0eeb29bb5f25de70353b0` |

完整字段见 [checkpoint-provenance.json](checkpoint-provenance.json) 和 [source-snapshot.json](source-snapshot.json)。两份 benchmark 报告内的 `source_sha256` 均与这 80 个文件逐项一致；复验结束时，它们也与当前本地源码逐项一致。archive、bundle、checkpoint 和权重均未复制到本目录。

## 设备、软件与编译器

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop GPU |
| Compute Capability | `8.6` / `cuda-sm86` |
| SM 数量 | `46` |
| 显存 | `8220901376` bytes |
| CUDA Driver / Runtime 原始版本值 | `13000` / `12090` |
| CuPy | `13.6.0` |
| Python / NumPy | `3.12.3` / `2.5.3` |
| OS | `Linux-7.0.0-34-generic-x86_64-with-glibc2.39` |
| 外部 `mlir-opt` | `/home/thomas/gac-mlir-build-lf/bin/mlir-opt`，通过 WSL `Ubuntu-24.04` 调用 |
| LLVM/MLIR | `LLVM version 24.0.0git`，`Optimized build` |
| LLVM source revision | `765392a6b4498f1dc57233f03a31b87462fdef01` |
| lowering target | `sm_86`、PTX `7.1`、`opt-level=3`、`--verify-each` |

`mlir-opt` 是外部 LLVM 构建产物，没有提交到本验证目录。两个 artifact manifest 保存实际命令、版本、LLVM revision、entry 和输入源码摘要。

## 测量协议

两个独立进程分别以 `ptx-first` 和 `nvrtc-first` 启动。每个进程均执行：

- warmup `10` 次；
- resident 测量 `6` 个 repeat，每个 repeat `30` 次 FFN；
- 每个 repeat 后反转 PTX/NVRTC 顺序，两个 Session 同时驻留；
- 输入、权重和 workspace 驻留 GPU，使用 CUDA Event 测量 pipeline；
- public `run()` 测量 `3` 个 sample，边界包含参数校验、H2D、计算、D2H 和同步；
- 正确性阈值 `rtol=0.0001`、`atol=0.00001`；
- 开启 autotune，三个可执行候选全部先验证正确性再计时。

首个 public 调用包含顺序相关的 CUDA 初始化成本。PTX-first 报告中 cuBLAS+PTX 首次 wall time 为 `136485.71603698656 us`；NVRTC-first 报告中 NVRTC 首次 wall time 为 `137179.67801494524 us`。主比较使用 warmup 后、顺序平衡的 resident CUDA Event 数据。

## Resident 与 public 结果

`resident GPU` 是 device-resident pipeline CUDA Event 中位数；`resident wall` 是同一 resident repeat 的同步 wall 中位数；`public GPU` 和 `public wall` 是三个 public `run()` 样本的中位数。

### 初始顺序：PTX-first

| backend | resident GPU median (us) | resident wall median (us) | public GPU median (us) | public wall median (us) | FFN/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlir-cublas-ptx-sm86` | `3050.7850646972656` | `3051.1836166666662` | `7697.535991668701` | `20710.33598622307` | `327.7844813034155` |
| `mlir-cuda-sm86` / NVRTC | `3150.2213795979815` | `3150.482916666667` | `7677.599906921387` | `20564.842969179153` | `317.43800815916495` |
| `mlir-tiled-ffn-sm86` | `13932.249450683594` | `13932.111766666667` | `18049.24774169922` | `31132.16598285362` | `71.77591842148178` |

### 初始顺序：NVRTC-first

| backend | resident GPU median (us) | resident wall median (us) | public GPU median (us) | public wall median (us) | FFN/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlir-cublas-ptx-sm86` | `3076.281611124675` | `3076.5662` | `7808.800220489502` | `21537.857013754547` | `325.06776895318256` |
| `mlir-cuda-sm86` / NVRTC | `3182.0303599039717` | `3182.3211833333335` | `7943.424224853516` | `21664.098952896893` | `314.26475768451763` |
| `mlir-tiled-ffn-sm86` | `13936.360677083334` | `13936.424183333333` | `18039.487838745117` | `31452.402006834745` | `71.75474452554744` |

两种初始顺序的 resident GPU 中位数再取中位数后，cuBLAS+PTX、NVRTC 和 tiled 分别为 `3.0635333379109704 ms`、`3.1661258697509767 ms` 和 `13.934305063883464 ms`。完整样本、P10/P90、min/max、首次调用和每个 repeat 的顺序见 [benchmark-ptx-first-rtx3070ti-laptop.json](benchmark-ptx-first-rtx3070ti-laptop.json) 与 [benchmark-nvrtc-first-rtx3070ti-laptop.json](benchmark-nvrtc-first-rtx3070ti-laptop.json)。

## 正确性

两个进程的正确性字段相同：

| backend | allclose | max abs error | max relative error（`1e-3` floor） |
| --- | --- | ---: | ---: |
| `mlir-cublas-ptx-sm86` | `true` | `3.337860107421875e-06` | `0.000521540641784668` |
| `mlir-cuda-sm86` / NVRTC | `true` | `3.337860107421875e-06` | `0.0005218898877501488` |
| `mlir-tiled-ffn-sm86` | `true` | `3.0994415283203125e-06` | `0.000455416738986969` |

这证明三个实现对当前真实权重与 synthetic query 符合 NumPy reference 阈值；它不构成真实数据集上的模型精度验证。

## Autotune

| 初始顺序 | cuBLAS+PTX median (ms) | NVRTC median (ms) | tiled median (ms) | 选择结果 |
| --- | ---: | ---: | ---: | --- |
| PTX-first | `3.0751574198404947` | `3.1610458374023436` | `14.007452901204427` | `mlir-cublas-ptx-sm86` |
| NVRTC-first | `3.1066965738932293` | `3.208857599894206` | `13.992652893066406` | `mlir-cublas-ptx-sm86` |

三个候选状态均为 `eligible`，两次运行均为 `cache_hit: false`。tiled 候选的 autotune 中位数分别是 cuBLAS+PTX 的 `4.555036048180902x` 和 `4.504029460312304x`。另有两个只有 metadata、没有完整 FFN Runtime Session 的候选被明确跳过：FP32 `64x64x32` tiled policy 与 FP16 NVGPU/MMA `128x64x32` policy；它们没有被执行或计时。

## Artifact 与 schema

[compile-report.json](compile-report.json) 是最终 v3 `--backend all` 编译报告。本目录保存两个离线 PTX artifact 的 manifest，但不复制 PTX/MLIR payload 或模型权重，因此 manifest 是审计证据，不是可直接部署的完整 artifact 目录。

| artifact | program schema | on-disk artifact schema | runtime ABI | 调用结构 | 部署边界 |
| --- | ---: | ---: | ---: | --- | --- |
| `mlir-cublas-ptx-sm86` | `2` | `1` | `2` | 2 次 CuPy/cuBLAS MatMul + 2 次 MLIR->PTX fusion，共 4 dispatch | fusion device code 与校验后的四调用 ABI；权重位于单独校验的 graph constants |
| `mlir-tiled-ffn-sm86` | `2` | `2` | `2` | 2 次 tiled MatMul + 1 次最终 fusion，共 3 个 PTX dispatch | 三个校验后的 PTX dispatch；权重位于单独校验的 graph constants |

最终 compile report 的计时边界如下：

| artifact | complete compile (ms) | 本次 invocation / incremental (ms) | dependency (ms) | base program reused |
| --- | ---: | ---: | ---: | --- |
| `mlir-cublas-ptx-sm86` | `3730.5903` | 不适用 | 不适用 | 不适用 |
| `mlir-tiled-ffn-sm86` | `10846.4942` | `7115.9039` | `3730.5903` | `true` |

cuBLAS+PTX 的 `3730.5903 ms` 是完整 standalone artifact 编译时间。tiled 的 `10846.4942 ms` 是包含所复用 fusion 依赖的完整 standalone artifact 时间；`7115.9039 ms` 只表示本次 invocation 新增的 incremental 工作，不能当作 tiled 独立完整编译时间。

tiled schema 2 manifest 的精确 profile 为 `query_shape=[1,40000,256]`、`rows=40000`、`channels=256`、`hidden_channels=512`。两个 manifest 都声明 `nvrtc_required: false`。cuBLAS+PTX 仍要求 CUDA Driver、cuBLAS 和 CuPy；tiled artifact 是 FP32、单缓冲 shared-memory `16x16x16` tile，`async_double_buffered: false`。

原始 manifest 见 [artifact-manifest-mlir-cublas-ptx-sm86.json](artifact-manifest-mlir-cublas-ptx-sm86.json) 和 [artifact-manifest-mlir-tiled-ffn-sm86.json](artifact-manifest-mlir-tiled-ffn-sm86.json)。

## 本地测试

[pytest-local.txt](pytest-local.txt) 保存最终本地测试日志：`501 passed, 28 skipped in 61.43s`。该日志证明同一工作树的本地测试状态；GPU 性能与正确性数字仍以两份 RTX 3070 Ti 远端 benchmark JSON 为准。

## 目录文件与完整性

- `benchmark-ptx-first-rtx3070ti-laptop.json`：最终 v3 PTX-first 独立进程报告；
- `benchmark-nvrtc-first-rtx3070ti-laptop.json`：最终 v3 NVRTC-first 独立进程报告；
- `artifact-manifest-mlir-cublas-ptx-sm86.json`：cuBLAS+PTX artifact manifest；
- `artifact-manifest-mlir-tiled-ffn-sm86.json`：tiled FFN schema 2 artifact manifest；
- `checkpoint-provenance.json`：checkpoint 与未提交 bundle 的摘要；
- `source-snapshot.json`：实际远程测试源码的逐文件 SHA-256；
- `compile-report.json`：最终 v3 `--backend all` 编译报告；
- `pytest-local.txt`：本地测试原始日志；
- `SHA256SUMS`：上述七份原始 JSON 与 pytest 日志的 SHA-256。

在 Linux/WSL 中验证：

```bash
sha256sum -c SHA256SUMS
```

`README.md` 和 `SHA256SUMS` 不相互纳入摘要，避免说明文字与摘要文件形成循环依赖。
