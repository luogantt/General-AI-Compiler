# CUDA 算子验证与局部性能证据

这些记录验证 C++ MLIR → LLVM/NVVM → PTX → CUDA Driver 的算子实现。GPU 为 **RTX 3070 Ti Laptop，计算能力 8.6**。NumPy、PyTorch 和 torchvision 只生成或核对测试参考值，不参与被测算子的执行。

**下表全部是算子或小子图结果，不能换算为完整 BEVFormer 的 FPS。** 整网结果见本目录的 `README.md`。这里仅保存精简 JSON、产物哈希和验证结论，不包含输入、权重、PTX、大型输出或完整运行日志。

| 证据 | 范围 | 结果 |
|---|---|---|
| [matmul-small.json](matmul-small.json) | 9 个案例 × scalar / shared-tiled，共 18 次 GPU 执行 | 全部通过；覆盖尾块、广播、批次、Reshape、FFN 链及非有限数 |
| [matmul-shapes.json](matmul-shapes.json) | 5 个模型相关形状 × scalar / tile32 / tile64，共 15 次 GPU 执行 | 全部通过独立 NumPy 全元素比较；优化输出与 scalar 逐字节相同 |
| [matmul-auto-selection.json](matmul-auto-selection.json) | 8 个 ONNX → LLVM 编译选择测试 | 全部通过；这是编译选择测试，不是 GPU 性能测试 |
| [convolution-kernels.json](convolution-kernels.json) | 14 个 Conv/DCNv2 案例 × channel4 / channel8，共 28 次 GPU 执行 | 全部通过；包含有意回退到 scalar 的情况 |
| [reduction-kernels.json](reduction-kernels.json) | 29 个 Softmax/LayerNorm GPU 案例 | 全部通过最终专用比较器；另一个 FP16 LayerNorm 案例被合法化阶段拒绝，不计为数值通过 |
| [reduction-paired-performance.json](reduction-paired-performance.json) | 6 个归约案例的 scalar / cooperative 配对实验 | 保存各模式总中位数、每轮中位数及范围；不能代表整网收益 |

## 数值比较与非有限数

普通 FP32 案例使用 `atol=rtol=2e-5`。MatMul 小案例最大有限绝对误差为 `1.1920929e-6`；较大形状为 `4.5299530e-6`；Conv/DCNv2 为 `7.7486038e-7`；归约案例为 `7.1525574e-7`。

MatMul 的 `nonfinite`、归约的 `softmax_nonfinite` 和 `layer_norm_nonfinite` 是刻意构造的特殊案例：必须分别匹配 **NaN、正无穷、负无穷的位置掩码**，其余有限元素仍按正常容差比较。不能将任意意外 NaN 当成通过。归约最初使用的通用比较器要求所有输出有限，因而拒绝了这两个特殊案例；原始报告仍保留在工作目录。这里收录的是使用显式 `expected_nonfinite` 元数据重新检查后的最终结果。

Conv 的 `conv_padding_nonfinite_weights` 有另一层含义：边界以外的卷积步被跳过，不能替换为 `0 × Inf/NaN`。该案例采用“只有中心权重有效”的解析参考值，验证既有 scalar 语义的一致性；它不是 PyTorch 在非有限权重下的通用一致性证明。其预期输出是有限数。其余普通 Conv/DCNv2 参考来自独立的 PyTorch/torchvision CPU 实现，版本记录在 JSON 中。

## MatMul 测量

每个形状和模式独立启动进程，预热 1 次、测量 5 次，表中为 CUDA event 区间平均值。该区间包含案例的全部 dispatch 和可能的主机提交间隙，排除 JIT、文件读写与数据上传下载。这组实验没有采用长时间交替配对，因此微秒级差异需要谨慎解读。

`tile32` 为输出块 `32×32×16`、每线程 `2×2`；`tile64` 为 `64×64×16`、每线程 `4×4`。二者均使用 256 线程、FP32 计算，保持 K 轴累加顺序，未启用 Tensor Core 或低精度替代。

| M × K × N | scalar / ms | tile32 / ms | tile64 / ms |
|---|---:|---:|---:|
| 1024 × 256 × 256 | 0.166656 | 0.092275 | 0.069005 |
| 40000 × 256 × 256 | 7.332474 | 2.916506 | 1.504410 |
| 40000 × 256 × 512 | 15.075546 | 5.800499 | 2.956230 |
| 40000 × 512 × 256 | 15.643757 | 4.912896 | 2.525709 |
| 900 × 256 × 256 | 0.125107 | 0.078080 | 0.071245 |

分块并非对所有形状有效。例如小 FFN 案例 `[8,256] × [256,512] + Bias → ReLU`，scalar 为 `0.013523 ms`，tile32 为 `0.030272 ms`，出现退化。因此保留 scalar 候选是必要的。

`--cuda-matmul-schedule=auto` 是**形状启发式**：`M≥128、N≥64、K≥64` 时选择 tile64，其余保留 scalar 并记录原因。阈值是基于本轮观测提出的策略，不表示阈值附近所有形状都已经测量。它不是 autotuner，不会现场搜索，也不保证不同 GPU 或每个形状上都最快。默认 CLI 策略仍为 scalar；显式 `shared-tiled` 可以单独配置 tile。

8 个编译选择测试区分“高维 A 与二维 B 可展平的投影”和“独立 batch 的小矩阵”，并覆盖广播、短 K Attention、尾块与显式覆盖，避免仅用总元素数量误判可复用的 M。

## Conv/DCNv2 与归约

Conv/DCNv2 的 channel4 运行中，11 个案例使用优化模板、3 个回退；channel8 中，9 个使用优化模板、5 个回退。FP16、每组输出通道数不能整除 tile 等情况会明确保留 scalar。**28 次通过不等于 28 次都执行了优化模板。** 本轮没有独立的 scalar GPU 配对基线，不能从这组记录单独声称 Conv/DCNv2 的加速倍数。

归约配对实验采用 4 轮交替模式顺序，每轮每模式预热 5 次、测量 30 次，即每案例每模式 120 个样本。下表是中位数，完整精度与每轮中位数见 JSON。

| 案例 | scalar / ms | cooperative / ms | 比值 |
|---|---:|---:|---:|
| Softmax width 17 | 0.004096 | 0.004096 | 1.00× |
| Softmax width 900 | 0.056320 | 0.006144 | 9.17× |
| LayerNorm width 17 | 0.004352 | 0.004096 | 1.06× |
| LayerNorm width 257 | 0.010944 | 0.005120 | 2.14× |
| Softmax `[4096,32]` | 0.020480 | 0.013312 | 1.54× |
| LayerNorm `[4096,256]` | 0.552960 | 0.067584 | 8.18× |

归约重新安排了加法次序，因此要求误差容限内一致，而不要求与 scalar 逐位相同。配对记录中的 `output_hashes_identical` 表示同一模式重复执行的输出一致，不应解释成 scalar 和 cooperative 的输出逐字节相同。

## 产物身份与复现入口

JSON 保留编译器、运行器、manifest、PTX、参考值及输出的可用 SHA-256；原始完整报告另有哈希。MatMul/归约最初验证用编译器 SHA-256 为 `91641cd43f6efd860d280d992acf95425e21601de731627503e8dd89e46d4414`；auto 8 项测试用 clean build 后的 `4c2538adf43e5fd8b6520c6b1ef2f4340189571c5fadcd34b19b9c933bdd85c1`。不同阶段的证据保留各自身份，不假定所有报告来自同一个最终二进制。

相关可复现入口：

- MatMul 基础案例：[run_vortex_matmul_tests.py](../../../../native/tests/run_vortex_matmul_tests.py) 的 `fixtures()`；测试数据可复用，CUDA 执行没有使用 Vortex runtime。
- 自动选择：[test_cuda_matmul_selection.py](../../../../native/tests/test_cuda_matmul_selection.py)。
- Conv/DCNv2：[generate_cuda_convolution_models.py](../../../../native/tests/generate_cuda_convolution_models.py)。
- Softmax/LayerNorm：[generate_cuda_reduction_models.py](../../../../native/tests/generate_cuda_reduction_models.py)、[check_cuda_reduction_results.py](../../../../native/tests/check_cuda_reduction_results.py)。
- CUDA 编译和执行：[compile-cuda.sh](../../../../native/tools/compile-cuda.sh)、[run_cuda_tests.py](../../../../native/tests/run_cuda_tests.py)。

这些是正确性和模板选择的阶段证据。最终整网性能、端到端口径和精度验收应使用本目录根说明中的完整模型结果。
