# 完整 BEVFormer 原生 MLIR 编译与推理验证

本记录对应 `lg/bevformer-vortex-full-ops` 的 C++ 原生编译路径。
模型经过 **ONNX → 注册的 MLIR Operation → MLIR Pass → SCF/Vortex → LLVM**，
没有经过 Python Graph 优化，也没有生成设备 C++ Kernel。

## 当前验收状态

| 项目 | 状态与范围 |
|---|---|
| 源模型 | 完整 `bevformer-tiny-cold.onnx`，9,129 个节点、454 个 initializer |
| 严格原生导入 | 通过，生成 9,583 个 GAC Operation；不使用 inspection 放行未知算子 |
| MLIR 优化、合法化与设备 Lowering | 通过，生成 1,548 个 dispatch |
| 完整主机机器码产物 | 已生成；使用同一套 C++ Lowering 的显式 native 验证目标 |
| 完整主机数值执行 | 通过，1,548 个 dispatch 全部执行，三个输出与 PyTorch 参考对齐 |
| 完整 Vortex ELF/vxbin 构建 | 通过，readelf 核对全部 1,548 个 `__vx_kentry_*` 入口 |
| 新增算子 SimX | Tile、三结果 Split、两种负索引 ScatterND 四例通过，输出最大绝对误差均为 0 |
| 完整 Vortex/SimX 数值执行 | 300 秒验证窗口内完成第 0 个 Reshape，进入第 1 个 Conv；超时结束，尚无完整数值结果 |
| 物理设备整网延时/FPS | 尚无测量，不能由主机或仿真时间换算 |

原始图包含 Backbone、FPN、Encoder、Decoder 和检测输出；本次没有缩小输入来
替代生产尺寸模型。输入为 FP32 `images[1,6,3,480,800]`，输出为 `bev_embed`、
`all_cls_scores`、`all_bbox_preds`。这些是原始网络输出，未包含检测后处理验收。

完整主机验证采用 `rtol=1e-3`、`atol=1e-4`，三个输出均为有限值，超出容差的
元素数均为 0：

| 输出 | Shape | 最大绝对误差 |
|---|---|---:|
| `bev_embed` | `[2500,1,256]` | `3.4570694e-6` |
| `all_cls_scores` | `[6,1,900,10]` | `1.2874603e-5` |
| `all_bbox_preds` | `[6,1,900,10]` | `4.9591064e-5` |

本次 CPU 生成 Kernel 的主机 wall time 合计 **435.561 秒**；包含测试驱动、断点
等开销的整次执行为 **572.246 秒**。这些数值只描述当前标量代码在 CPU 验证环境
的执行成本，**不是 Vortex/GPU 延时或吞吐率**。完整结果摘要见
[full-native-validation.json](full-native-validation.json)。572.246 秒是测试驱动的
dispatch 循环（含断点开销），不含编译、初始化和最终写盘，不能当成应用端到端延时。

完整 Vortex 试运行记录见 [full-simx-attempt.json](full-simx-attempt.json)。
分层 IR、完整 ELF/vxbin 和编译器来源摘要见
[full-vortex-validation.json](full-vortex-validation.json)。
`--trace-dispatches` 等待设备事件完成后才写 `end`，因此日志中的第 0 项完成是真实
设备模拟执行。第 1 项卷积约有 54.19 亿 MAC；当前一核、四 warp、每 warp 四线程的
SimX 标量路径未在 300 秒内完成。Grid 大小在 ABI 的 uint32 范围内，日志和源码核查
未发现死锁证据；驱动等待超过 10 秒的告警不能直接判断设备卡死。

## 分层实现与数据边界

1. [OnnxImporter.cpp](../../../../native/lib/OnnxImporter.cpp) 用 protobuf 读取
   ModelProto，通过 OpBuilder 创建 ModuleOp、func.func、注册的 `gac.*` 操作。
   名字表连接 MLIR Value；权重保存为 DenseElementsAttr；激活只记录类型和 SSA 关系。
2. [Dialect.cpp](../../../../native/lib/Dialect.cpp) 校验输入类型、shape 和属性，
   包括 Tile、ScatterND、NonZero、真正多结果的 Split。运行时 NonZero 的输出长度
   保持动态，不按某次样本假设固定。
3. [GraphPasses.cpp](../../../../native/lib/GraphPasses.cpp) 运行真实 MLIR Pass。
   [ConstantFolding.cpp](../../../../native/lib/ConstantFolding.cpp) 只计算纯常量
   子图；APInt/APFloat 保留输入精度，FP64 几何计算不会提前缩成 FP32。显式 Cast
   才改变精度，整数溢出、零除及非法索引会被拒绝。
4. [TensorLegalization.cpp](../../../../native/lib/TensorLegalization.cpp) 将静态
   Split 重写成多个 Slice，并保留来源位置与各输出名称。输入依赖的 split sizes
   不会因为 value_info 声明了输出尺寸就被消除。
5. [VortexLowering.cpp](../../../../native/lib/VortexLowering.cpp) 为剩余图计算
   构造 SCF/Arith/Math/Vortex 操作，生成 dispatch DAG 和全局 Buffer 复用计划。
   ScatterND 的常量索引变成 ELF 中的只读逆索引表；运行时 Kernel 仍从 data 或
   updates 读取真实激活并写入输出。当前仅支持无重复索引、reduction=none 的模板。
6. [compile-vortex.sh](../../../../native/tools/compile-vortex.sh) 调用统一版本的
   MLIR translator、VOLT 和 Vortex 链接/封装工具，产出 ELF/vxbin。
   [vortex-run.cpp](../../../../native/runtime/vortex-run.cpp) 负责 Buffer 上传、
   Kernel 提交、事件依赖和结果读回。

当前 ONNX 固化了 cold 帧标定与部分几何参数，因此对应的可见性判断和索引可在
编译期计算。**图像数据不是常量**：Backbone、Attention 等激活运算仍由生成的
机器码执行。若更换被固化的标定参数，需要重新导出/编译，或扩展动态索引契约。
当前验收范围不包括 warm 时序输入、任意相机标定、任意动态 shape 或整个数据集精度。

## Buffer 与性能口径

本次 native manifest 有 2,106 个 tensor，规划的全局 arena 为 **714,316,800 B**，
内嵌索引常量为 **3,333,600 B**。arena 不包括独立输入、外部常量、机器指令、栈和
Runtime 元数据；设备容量应按完整资源预算核对。

arena 根据 SSA 最后一次消费复用，要求 `last_use < new_producer`，并将图输出保留
到全部 dispatch 结束。Runtime 的顺序事件依赖保证旧数据读取完成后才能覆盖地址。
这属于全局 Buffer 生命周期规划，**没有 LMEM 分块、DXA 搬运、双缓冲或 TCU**。
当前 Kernel 是一个线程负责一个输出、归约循环保留在寄存器里的标量正确性基线。

主机验证时间、SimX 主机运行时间和物理设备推理延时分别记录。保存中间值、断点、
编译时间也不能计为设备 Kernel 延时。验证驱动说明见
[全模型执行工具](../../../../native/tests/README.full_model.md)。

## 回归与复现

本轮主机回归全部通过：**31 项既有测试 + 6 项常量几何测试 + 15 项新增算子测试**。
执行过程是 C++ 导入、MLIR Pass、LLVM 翻译、clang 生成共享库，再调用真实机器码
与独立参考比较。Python 只承担测试夹具、命令组织与比较，不实现算子回退。

新增算子覆盖了 Tile，opset 17 等分/显式 Split，opset 18 不等分 Split，常量多结果
Split，以及负索引 ScatterND 和 NonZero。错误 sizes、重复索引、错误 updates shape、
unsigned 索引、运行时动态 NonZero/Split 的设备部署均有明确拒绝测试。

本目录保存的回归证据：

- [native-regression-results.json](native-regression-results.json)：31 项既有主机测试。
- [geometry-regression-results.json](geometry-regression-results.json)：6 项几何常量测试。
- [full-ops-regression-results.json](full-ops-regression-results.json)：15 项新增算子测试。
- [full-ops-simx-results.json](full-ops-simx-results.json)：4 项新增 Vortex/SimX 数值结果。

本地完整产物及详细结果文件：

- `build/native-reshape-tests/results.json`：31 项既有主机测试。
- `build/native-geometry-tests/results.json`：6 项几何常量测试。
- `build/native-full-ops-tests/results.json`：15 项新增算子测试。
- `build/native-full-ops-tests/vortex-object-results.json`：4 项新增 SimX 结果。
- `build/native-bevformer-complete/plan.json`：完整 native dispatch 与 Buffer 计划。
- `build/native-bevformer-complete/host-results/full-model-report.json`：完整主机执行、
  逐 dispatch 时间、artifact 摘要与三个输出的数值误差。
- `build/native-bevformer-vortex-complete/`：完整 Vortex 分层 IR 和机器码产物目录。
- `build/native-bevformer-vortex-complete/full-simx-attempt.json`：ELF/vxbin/manifest
  摘要、入口数及完整 SimX 试运行状态，`full_model_executed` 尚为 false。

一次编译保存中间阶段：

```bash
gac-compile build/bevformer-onnx-inspect/bevformer-tiny-cold.onnx \
  --emit=llvm --target=native \
  --save-stages build/native-bevformer-complete \
  --manifest build/native-bevformer-complete/plan.json \
  --constants-dir build/native-bevformer-complete/constants \
  -o build/native-bevformer-complete/kernel.mlir
```

`01-import.mlir`、`02-optimized.mlir`、`03-device.mlir`、`04-llvm.mlir` 是同一编译
过程的快照；最终 LLVM Dialect 也写入 `-o` 指定文件。机器码执行及逐层比较命令见
上述验证工具文档。

## 本目录的历史报告

[native-full-model-report.json](native-full-model-report.json)、
[unsupported-nodes.json](unsupported-nodes.json)、
[backend-gap-audit.json](backend-gap-audit.json) 和当时的 `strict-vortex-compile.log`
保存的是**本轮补齐算子之前的失败证据**。其中“208 个未支持节点”“严格编译首先
停在 Tile”“设备 Lowering 停在 Split”属于历史状态，已由本轮原生实现解决。
文件保留用于追踪问题，不代表当前编译器仍有同一阻塞。

同样，解决这些编译阻塞不自动等于整网数值通过；应以本页的分项验收状态和最终
执行报告为准。
