# BEVFormer-base：完整模型编译与 CPU 整网数值验证

日期：2026-10-16。

**完整官方 BEVFormer-base 已从原始 ONNX 成功编译生成 native 主机共享库，以及 Vortex 的 ELF / vxbin 机器码；生成的 CPU 机器码已执行全部 3,224 个 dispatch，三路最终输出与 PyTorch 参考比对通过。** ONNX 含 22,412 个节点；manifest 的生命周期、依赖与存活 Buffer 重叠审计通过。Vortex plan、ELF 和 vxbin 的 3,224 个 Kernel 入口逐一核对，ELF 无未定义符号，常量文件长度与产物 hash 审计通过，见 [完整 Vortex 产物审计](full-base-vortex/artifact-audit.json)。**尚无全图 Vortex 执行或 30 FPS 结论。**

最终证据：[数值结果摘要](full-base-native-host/final-summary.json)、[完整执行报告](full-base-native-host/full-model-report.json)、[进程计时与退出状态](full-base-native-host/execution-time.txt)。报告中的 `completed_dispatches=3224`、`full_model_executed=true`、`passed=true`；校验阈值为 `rtol=1e-3, atol=1e-4`，三个输出均为有限值且超差元素为 0。完整报告归档 SHA-256 为 `ba16db3982035121aa203412a2855033a1aaafd9433f126ca15e5743defcfae1e`，已与 WSL 原文件核对一致。

| 输出 | Shape | 最大绝对误差 |
|---|---|---:|
| `bev_embed` | `[40000, 1, 256]` | `2.30074e-5` |
| `all_cls_scores` | `[6, 1, 900, 10]` | `1.97887e-5` |
| `all_bbox_preds` | `[6, 1, 900, 10]` | `6.53714e-5` |

本轮在本机 WSL Ubuntu-24.04 上运行生成的 CPU Kernel，整个验证进程耗时 **2:56:48**，退出码 0；其中 Kernel 墙钟时间合计 `10556.22 s`。验证含检查点写盘与结果比较，不能用这次总耗时报告 Vortex 设备延时。覆盖范围是固定导出配置与输入下的 FP32 cold-frame 完整网络及三路原始输出，不包含 warm/history 连续帧、检测后处理或数据集精度评测，也不代表所有输入形状均已验证。

北京时间 **2026-10-16 20:25** 的 [运行中快照](full-base-native-execution-progress.json)、[执行输入/共享库 hash](full-base-native-execution-source-hashes.txt) 和 [进程、输出目录及恢复记录](full-base-native-host/running-status.json) 作为历史证据保留；其中“运行中”和恢复命令已被本次最终报告取代，不应重新启动已完成的任务。

当前资源审计和 plan 摘要见 [full-base-native-resource-audit.json](full-base-native-resource-audit.json) 与 [full-base-native-plan-summary.json](full-base-native-plan-summary.json)：

| 存储项 | 字节数 |
|---|---:|
| 输入 | 106,905,600 |
| 常量 Tensor | 368,518,572 |
| 复用后的 arena | 1,554,944,000 |
| 机器码内嵌常量表预算 | 2,073,600 |
| 最低聚合存储 | **2,032,441,772** |

合计约 `2.03 GB`，不含指令代码、栈、runtime 元数据和驱动分配；不能把该数值直接当作实际板卡显存占用。原 ONNX 至 native LLVM 编译耗时 `174.26 s`、峰值 RSS `1,298,172 KiB`、swap 计数 0，见 [编译计时](full-base-native-compile-time.log)。这是编译时间，不是模型推理延时。

完整 Vortex 编译与链接耗时 `279.46 s`，峰值 RSS `1,300,316 KiB`，无 swap，退出码 0，见 [Vortex 编译计时](full-base-vortex/compile-time.txt)。ELF 为 `1,285,396 B`，vxbin 为 `1,085,390 B`；[编译前](full-base-vortex/compiler-before.sha256) 与 [编译后](full-base-vortex/compiler-after.sha256) 的编译器 hash 相同。产物生成与符号审计证明代码链已贯通，不代表所有 Kernel 已在完整模型设备执行中验证数值。

优化前的历史预算诊断见 [编译日志](full-base-rv32-budget-before-matmul-bias.log) 和 [存活内存明细](full-base-memory-before-matmul-bias.log)。该轮规划 3,460 个 dispatch，聚合 Buffer 需求 `2,857,401,828 B` 超过 signed RV32 的 `2,147,483,647 B` 预算；arena 为 `2,292,224,000 B`，峰值同时存活 arena 为 `1,935,257,600 B`。这些指标口径不同，且均为旧轮次数字。峰值中 MatMul 与紧随其后的 Add 输出各占 `491,520,000 B`，后续优化针对这种中间结果的物化成本，保留预算检查。

## 验证方法

被测路径为 `ONNX → C++ Importer → MLIR Operation / Pass → LLVM IR → 机器码`。

- Native 测试用 LLVM 工具生成主机共享库，通过 `ctypes` 调用其中的机器码函数。它验证 lowering 数值，不是 Vortex 设备推理。
- SimX 测试经过 VOLT、ELF、vxbin，并由 C++ Vortex runtime 在模拟器执行。模拟器的主机墙钟时间不能当成实际硬件推理延迟。
- Python 只生成独立参考、调用编译工具、装载测试数据及比较结果，不替代被测算子。DCNv2 参考使用真实 `torchvision.ops.deform_conv2d` CPU 算子；Resize 使用 PyTorch interpolate；MSDA 使用 PyTorch grid_sample 与加权归约。

## 已通过的检查

| 检查 | 结果 | 证据 |
|---|---:|---|
| 既有 native 回归：融合、广播、布局、精度、Reshape 等 | 31/31 | [native-regression-results.json](native-regression-results.json) |
| 既有常量几何折叠回归 | 6/6 | [geometry-regression-results.json](geometry-regression-results.json) |
| 既有 Tile / Split / ScatterND / NonZero 合同与数值 | 15/15 | [full-ops-regression-results.json](full-ops-regression-results.json) |
| 新增 DCNv2 / Resize / MSDA 与内存分支图 | 21/21 | [base-ops-native-results.json](base-ops-native-results.json) |
| 自定义算子导入合同、预算和常量视图 | 11/11 | [base-contract-geometry-results.json](base-contract-geometry-results.json) |
| 真实 torch.onnx.export 的 DCN、1 层和 4 层 MSDA | 3/3 | [actual-export-contract-results.json](actual-export-contract-results.json) |
| DCN、Resize、MSDA 三个 SimX 代表例 | 3/3 | [base-ops-simx-results.json](base-ops-simx-results.json) |
| DCN 坐标范围 lowering 修改后的 SimX 复测 | 1/1，重复 DCN 代表例 | [base-dcn-final-simx-results.json](base-dcn-final-simx-results.json) |
| 连续 view 物理根复用、形状与生命周期 | 12/12 | [view-alias-native-results.json](view-alias-native-results.json) |
| Shape 条件控制的 If 精确模式及拒绝检查 | 21/21 | [shape-if-native-results.json](shape-if-native-results.json) |
| View 晚消费分支、不同逻辑 shape 的 Softmax，真实 SimX | 2/2 | [view-alias-simx-results.json](view-alias-simx-results.json) |
| View 改动后的内存 fork 图复测 | 1/1，重复内存图 | [memory-fork-after-views-results.json](memory-fork-after-views-results.json) |
| View / Range 改动后的四组完整回归重跑 | 73/73 | [current-regressions-results.json](current-regressions-results.json) |
| 精确整数 Range、预算、动态与浮点边界 | 24/24；13 项常量字节精确 | [range-native-results.json](range-native-results.json)、[range-exact-results.json](range-exact-results.json) |
| MatMul + Bias 融合、广播与舍入保护 | 17/17 | [matmul-bias-native-results.json](matmul-bias-native-results.json) |
| MatMul + Bias 广播代表例，真实 SimX | 2/2 | [matmul-bias-simx-results.json](matmul-bias-simx-results.json) |
| 静态 ScatterND 语义化简与保护条件 | 16/16 | [scatter-legalization-native-results.json](scatter-legalization-native-results.json) |
| 三阶段 ScatterND 广播链化简，真实 SimX | 1/1，输出误差为零 | [scatter-legalization-simx-results.json](scatter-legalization-simx-results.json) |
| 内存融合最终构建的四组完整回归重跑 | 73/73 | [memory-fusion-regressions-results.json](memory-fusion-regressions-results.json) |
| 最终构建 MatMul + Bias，包括向量形状 | 23/23 | [matmul-bias-rank-one-native-results.json](matmul-bias-rank-one-native-results.json) |
| MSDA 坐标按需融合与保护条件 | 16/16 | [msda-coordinate-native-results.json](msda-coordinate-native-results.json) |
| MSDA 坐标融合代表例，真实 SimX | 2/2 | [msda-coordinate-simx-results.json](msda-coordinate-simx-results.json) |

前四组是 `71ef6dab89a9` 构建上的 **73 项历史回归**，原报告保持不变。后续在 `7b2819fcbe64` 上重新完整跑过同四组 73 项，起止编译器与 optimizer hash 均一致；新报告分别为 [native 31](current-native-regression-results.json)、[geometry 6](current-geometry-regression-results.json)、[full-ops 15](current-full-ops-regression-results.json)、[base 21](current-base-regression-results.json)。其中包含预期拒绝的非法输入，不代表 73 个完整模型或 73 种算子全部支持；重复测试不增加独立算子数量。

加入 MatMul Bias、坐标融合和 ScatterND 证明后，又在 `050f89d4f3ff` 上完整重跑四组，起止 compiler / optimizer hash 同样一致：[native 31](memory-fusion-native-regression-results.json)、[geometry 6](memory-fusion-geometry-regression-results.json)、[full-ops 15](memory-fusion-full-ops-regression-results.json)、[base 21](memory-fusion-base-regression-results.json)。以上三轮的报告分别保存。

新增 21 项中，14 项是数值正例，7 项是预期拒绝。DCN 覆盖普通分组、deform group、stride、padding、dilation、可选 bias、部分零 mask、边界和零 offset 等价卷积；Resize 覆盖 sizes、非整数 scales 和真实零长度可选输入；MSDA 覆盖多层、多 head、多 batch、像素中心、边界与零权重。DCN、MSDA 的 native 最大绝对误差分别为 `2.98e-7`、`1.49e-8`，Resize 结果完全一致。真实导出合同三例最大绝对误差为 `4.77e-7`。

原 73 项中成功生成的正例 manifest 经过 `run_full_model.load_plan()` 检查，包括定义顺序、依赖、arena 边界和存活 Buffer 重叠。新增 view native、view SimX 和内存 fork 也经过该检查。八节点 fork 图同时保留早期输出和最终输出；初始 arena 为 `1152 B`，后续分配器构建降至 `1024 B`，两个输出仍正确。测试检查预算与数值，未把特定分配偏移写死为答案。

## 新增 View 与 If 验证

12 项 view 测试的参考来自 ONNX Runtime，覆盖 `0/-1` Reshape 链、Squeeze/Unsqueeze、input/constant/arena 根、分支延迟消费、不同逻辑 shape 的 Softmax 和公开返回值。内部 view 不再发出复制 dispatch；公开返回的 view 保留名字、shape 和必要复制。非 identity Transpose 与 FP16 Cast 往返仍执行真实变换。大常量根的特定测试移除了两个复制，元数据记录消除的逻辑复制量为 `33,570,816 B`；这不是整网带宽或吞吐率测量。

两项 SimX 测试进一步验证物理根 ID、producer 和依赖在 C++ runtime 中有效：晚消费分支执行五个 dispatch，最大误差 `2.38e-7`；同根不同 shape 的 Softmax 执行两个 dispatch，最大误差 `2.98e-8`。两例各消除两个 view 复制。

21 项 If 检查仅覆盖导出器产生的精确 shape-guarded Squeeze/Identity 模式，以及不匹配模式的拒绝；不代表任意动态控制流已支持。原始报告没有记录 binary hash，构建记录仅关联到 `99de4ef2e559` 的执行窗口，不能作为该轮报告自带的 hash 证明。

## Range 合同与精确数值

24 项 Range 测试包括 13 项常量数值正例、2 项编译期拒绝、9 项保留 IR 后设备拒绝。正例覆盖 i16/i32/i64、正负步长、跨零、非整除长度、空范围消费、超过 `2**53` 和 INT64 边界；i16 结果经 Cast 到现有设备 ABI 支持的 i64。测试读取 manifest 实际输出常量文件并逐字节比较，避免浮点容差隐藏整数误差。运行时参数、超过折叠预算的范围及尚未实现的浮点 Range 不会伪装成常量。见 [合同案例](range-contract-cases.json) 与 [起止 binary hash](range-compiler-provenance.json)。

[ONNX Range-11 官方文档](https://onnx.ai/onnx/operators/onnx__Range.html#range-11) 同时列出整数长度公式和含浮点转换的 Function Body。极值 `Range(INT64_MIN, INT64_MAX, INT64_MAX)` 的精确整数公式产生三项，ONNX 1.21 ReferenceEvaluator 此次产生两项；该案例明确验证本实现的精确整数合同，不宣称与 ReferenceEvaluator 或 ORT 一致。普通正例经过 ReferenceEvaluator 对照，完整 base 使用的常规几何范围不涉及此极值差异。

## 常量折叠资源检查

独立合同工具验证了逻辑 `4,194,320` 元素的几何常量路径：最终为零 dispatch，输出 `1,048,580` 个常量元素，FP64 抵消测试的结果精确为 `1.0`。全组 11 项测试的主机最大 RSS 为 `73,560 KiB`，墙钟时间 `3.42 s`，swap 计数为零。见 [base-contract-geometry-time.log](base-contract-geometry-time.log)。这是该小规模测试进程的资源数据，不能外推为完整 base 的峰值内存或推理时间。

## 静态 ScatterND 与 MatMul + Bias

ScatterND 化简由 C++ MLIR TensorLegalization Pass 完成。完整且有序的静态索引覆盖直接替换为 updates 的 Identity/Reshape；同一 updates 的单使用链仅在逐索引证明无重复、全覆盖并严格符合广播坐标映射后替换为 Expand。证明限制目标前缀最多 `1,048,576` 个 slice、最多 `8,388,608` 个索引坐标及 64 个链节点，乘积先检查溢出；不按模型名字或固定相机数量匹配。

16 项检查覆盖形状变化、合法负索引、乱序映射、局部更新、不同 updates、中间结果公开返回、重复索引与 reduction 保护。缺洞或跨阶段重叠的链保留原操作；单个 ScatterND 内含重复索引仍由后端明确拒绝。三阶段零填充测试经生成的 CPU 机器码和真实 VOLT/SimX 均得到正确输出；manifest 中原零 Buffer 消失，只剩一个 Expand dispatch，见 [Buffer 检查](scatter-buffer-elimination-results.json)。该小图 arena 为 `128 B`；完整模型的内存收益另行测量，不从小图推断。

MatMul + Bias 的 17 项检查覆盖多种广播、加法操作数顺序、FP16 舍入边界、NaN/Inf、完整归约后加 bias，以及 MatMul 输出仍需公开或另有消费者时不融合。两个实际 SimX 广播案例均只剩一个融合 dispatch，最大绝对误差分别为 `5.96e-8` 和 `1.19e-7`。这些结果属于 `df0f1cdc73c4` 构建；Scatter 检查属于后续 `050f89d4f3ff` 构建，两组不混用版本证据。

## 构建版本与证据边界

- 初始 73 项回归和 DCN SimX 复测：`gac-compile` SHA-256 前缀 `71ef6dab89a9`。
- 最初三项 SimX 和三项真实导出合同：前一个稳定构建，SHA-256 前缀 `f86f75610389`。
- 12 项 view native 和内存 fork 复测：`99de4ef2e559`，起止 hash 相同。
- 两项 view SimX：`91891c9873d9`，起止 hash 相同。该构建加入了 `01-shaped.mlir` 快照。
- 新一轮完整 73 项与 24 项 Range：`7b2819fcbe64`，各自 wrapper 记录的起止 hash 相同；Range 另检查 13 项常量输出字节。
- MatMul + Bias 的 17 项 native 与两项 SimX：`df0f1cdc73c4`，各自起止 hash 相同。
- ScatterND 的 16 项 native 与一项 SimX：`050f89d4f3ff`，各自起止 hash 相同，另检查原零 Buffer 确实消失。
- 同一 `050f89d4f3ff` 构建还通过新一轮 73 项回归、23 项 MatMul Bias、16 项 MSDA 坐标融合及两项坐标融合 SimX；报告均记录起止 hash。
- 21 项 If 的 hash 未捕获；构建窗口关联见上一节。独立 11 项合同工具也没有单独捕获 hash，其执行窗口关联到初始 `71ef6dab89a9` 构建。
- 不同构建先后加入常量视图、折叠预算、坐标检查、内部 view 物理根和调试快照。每组报告保留自己的证据范围，不能把历史通过自动迁移到当前代码。

完整工具 hash、测试源码 hash 和各组证据范围见 [provenance.json](provenance.json)。[export-reference-contracts.json](export-reference-contracts.json) 保留导出端参考实现与实际 ONNX 合同的比较。归档只保存小型文本报告，不包含权重、ONNX 模型或机器码二进制；完整测试产物留在 `build/`。

## 复现入口

参考模型生成器：[`native/tests/generate_base_ops_models.py`](../../../../native/tests/generate_base_ops_models.py)。使用已有 PyTorch/Torchvision 环境生成 fixture；执行器仅需 NumPy。

```sh
python native/tests/generate_base_ops_models.py build/native-base-ops-tests
python native/tests/run_base_ops_tests.py \
  --compiler /path/to/gac-compile \
  --llvm-bin /path/to/llvm/bin \
  --fixtures build/native-base-ops-tests
```

SimX 使用同一执行器的 `--simx-case`、`--compile-script`、`--runner` 和 `--vortex-build` 参数。请在测试窗口保持编译器二进制稳定。

Range 的复现入口是 [`generate_range_models.py`](../../../../native/tests/generate_range_models.py) 与 [`run_range_tests.py`](../../../../native/tests/run_range_tests.py)。后者接受 `--compiler`、`--optimizer`、`--mlir-translate`、`--clang`、`--fixtures`，先执行真实 native 编译链，再核验常量输出字节并记录起止编译器 hash。
