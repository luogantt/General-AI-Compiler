# 完整 BEVFormer-tiny：原生生成代码数值验证

本次完整执行了导出的 cold-frame BEVFormer-tiny，输入保持 `float32[1,6,3,480,800]`，从 Backbone、FPN、Encoder、Decoder 到三个原始检测输出，共 **1,548 个 dispatch**。测试程序依次调用 C++ 编译器经过 MLIR/LLVM 生成的 CPU 共享库函数，没有使用 Python、NumPy、BLAS 或 ONNX Runtime 替代任何算子。

这是 **native CPU 目标的整网正确性验证**。Vortex 机器码生成和设备执行有独立证据，不能用本报告替代；这里没有物理 GPU 延时或 FPS。

## 结果

采用现有 PyTorch 输出，阈值 `rtol=1e-3, atol=1e-4`。所有输出有限，全部元素通过比较。

| 输出 | Shape | 最大绝对误差 |
|---|---|---:|
| `bev_embed` | `[2500,1,256]` | `3.4570694e-6` |
| `all_cls_scores` | `[6,1,900,10]` | `1.2874603e-5` |
| `all_bbox_preds` | `[6,1,900,10]` | `4.9591064e-5` |

独立 ONNX Runtime 参考仅用于事后检查。Backbone 最后输出、FPN lateral Conv、FPN 输出 Conv 的最大误差分别为 `4.3511391e-6`、`1.0430813e-6`、`1.0013580e-5`，也全部通过。

输入是现有固定 seeded fixture，标定/CAN-bus 元数据固定。这证明当前导出 cold 模型的这次数值执行，不代表 warm/history 路径、真实数据集精度或模型后处理已经评估。

## 时间与性能瓶颈

生成的 Kernel 主机调用时间合计 **435.561 秒**。dispatch 循环 wall time 为 **572.246 秒**，包括每 50 个 dispatch 保存一次 714 MB arena 断点、日志和选定中间结果。该 wall time 不包括编译、输入/权重/共享库加载及最终输出文件保存。它是一次带诊断开销的正确性运行，不是反复预热后的性能基准。

| 部分 | Kernel 主机调用时间 |
|---|---:|
| Backbone | 313.375 s |
| FPN | 4.156 s |
| Encoder | 9.913 s |
| Decoder | 105.922 s |
| 检测头与其他 | 2.195 s |

目前 Conv 仍采用标量归约，两个 Conv 模板累计约 **316.956 秒**。另一个明显瓶颈是 Softmax：18 次调用累计 **98.435 秒**，其中 Decoder 的六个 `[8,900,900]` Softmax 每个约 15–17 秒。当前模板为每个输出重复求同一行的最大值和指数和，重复了本可共用的归约工作。

后续应使用按行的 Softmax 计算模板：每行只求一次 max 和 sum，再归一化该行输出；Vortex 可进一步将归约和线程映射结合。实现时保持稳定 Softmax 语义，明确浮点累加顺序，并重新做数值验证。本报告记录的是优化前实测，不把理论收益当成已实现的加速。

## Buffer 与容量

输入 27,648,000 B，外部常量 159,899,584 B，复用 arena 714,316,800 B，模块内嵌常量 3,333,600 B，最低数据存储合计 **905,197,984 B**。该数值未计机器指令、栈和 Runtime 元数据。

所有逻辑 arena tensor 的字节数累计为 7,896,629,120 B；依据 SSA 活跃区间复用后只分配 714,316,800 B。测试驱动额外检查读前定义、依赖和活跃地址重叠。

## 证据

- [紧凑完整报告](full-native-validation.json)：输入、参考、MLIR/LLVM/共享库/manifest、工具与脚本哈希、误差、容量及计时口径。
- [中间输出对齐](host-native-intermediate-comparison.json)。
- [按算子和阶段的时间统计](host-native-profile-summary.json)。
- 1,548 条完整 dispatch 记录保留在 `build/native-bevformer-complete/host-results/full-model-report.json`，权重不复制进文档。

产生本次 artifact 时没有记录当时 `gac-compile` 可执行文件的精确哈希；报告明确保留这个缺口，没有把后续重建的编译器哈希冒认为旧 artifact 的生成工具。实际执行的 `.so`、MLIR、LLVM IR、manifest、输入和权重集合哈希均有记录。
