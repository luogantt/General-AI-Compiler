# 计算模板库与融合验证 · 2026-10-21

本轮将已有 CUDA/Vortex 优化模板整理为 `GACComputeTemplates`，将目标硬件辅助函数与
编译 Pass 分离，并新增受限 `Conv → 通道 Bias Add → 可选 ReLU` 图融合。
使用真实 LLVM/MLIR 20.1.8 构建；Python 仅生成参考数据和驱动测试。

| 检查 | 结果 | 原始报告 |
|---|---:|---|
| 原生算子/图优化回归 | 31/31 | [native-regressions.json](native-regressions.json) |
| 默认 Linalg、同坐标融合、非法 IR 与禁止绕过 Linalg | 28/28 | [linalg-regressions.json](linalg-regressions.json) |
| 模板选择、资源、回退与新旧代码生成对比 | 27/27 | [registry-contracts.json](registry-contracts.json) |
| CUDA MatMul 形状启发式选择 | 8/8 | [cuda-selection.json](cuda-selection.json) |
| 抽取后的 DCNv2 共用数值实现与拒绝条件 | 7/7 | [dcn-regressions.json](dcn-regressions.json) |
| 新融合契约 | 13/13 组，52 条执行路径 | [fusion-contracts.json](fusion-contracts.json) |

共 114 个检查/场景；其中 13 个融合场景各执行
`legacy / Linalg × --no-optimize / 优化开启` 四种组合。
拒绝非法输入、选择模板和机器码数值执行属于不同检查，不能都称为 GPU 测试。

模板注册表见 [catalog.json](catalog.json)，由新构建的 `gac-template-list` 从实际 C++
注册表导出。该工具只链接模板库及其底层依赖，独立链接成功。

## 代码生成保持一致

模板回归包含六类优化模板，以及 FP16、输出通道不整除、LMEM 容量、DXA 批量矩阵、
TCU lane 数等回退条件。12 个案例分别生成 device 和 LLVM IR，24 次与重构前冻结
编译器的结果比较：**IR 文本完全一致，Runtime 启动字段完全一致**。
新 manifest 的 `compute_template` 是附加信息，原 Runtime ABI 字段保留。

最终编译器 SHA-256：

```text
4fbfe9ad1acd72974483a34be059db44d489b4699ef7379da9d7a9168460e71a
```

对比用的重构前编译器 SHA-256：

```text
4a2856cfb21ad455a876aca6693954c32c5293502987af38d499611e71bad4b3
```

## 融合实际改变了什么

| 小模型 | legacy dispatch | Linalg dispatch |
|---|---:|---:|
| Conv＋运行时通道 Bias＋ReLU | 3 → 1 | 3 → 1 |
| Conv＋运行时通道 Bias | 2 → 1 | 2 → 1 |
| 含 padding 的分组 Conv 链 | 3 → 1 | 2 → 2 |
| FP16 舍入边界 | 3 → 3 | 3 → 3 |

有限值逐元素比较、NaN/正负 Inf 分布、公开输出保留、全局 arena 生命周期检查通过。
参考来自独立 ONNX 实现；DCNv2 用已有 torchvision 参考数据。执行对象为编译器生成的
CPU 机器码，不使用 Python 作为被测算子的实现。

本轮没有执行完整 BEVFormer、GPU 或 SimX，也没有新增性能结论。
模板工程化本身不应改变既有模板的速度；新增融合的硬件收益需要独立实测。

## 复现入口

```bash
# 与 native/README.md 相同的 LLVM/MLIR 20.1.8、Protobuf、ONNX 构建环境
cmake --build <build-dir> --target gac-compile gac-opt gac-template-list
<build-dir>/templates/gac-template-list

python native/tests/test_template_registry.py \
  --compiler <build-dir>/gac-compile \
  --reference-compiler <frozen-old-build>/gac-compile \
  --output <fresh-output-directory>

python native/tests/run_fusion_contract_tests.py \
  --compiler <build-dir>/gac-compile \
  --llvm-bin <llvm-build>/bin \
  --fixtures <fusion-fixtures-directory>
```

其余检查分别使用 `run_tests.py`、`run_linalg_tests.py`、`test_cuda_matmul_selection.py`
和 `run_base_ops_tests.py`。DCNv2 参考从既有 `build/native-current-regressions/base` 中
复制七个 DCNv2 案例的 ONNX、inputs、expected 和 cases 元数据到独立目录后执行，
未修改原参考文件。

设计入口：[模板库说明](../../../../native/templates/README.md) ·
[融合规则与边界](../../../compiler-fusion.md)。
