# BEVFormer-base：原生 MLIR / Vortex 适配

本适配使用官方 `bevformer_base.py` 和完整检测模型权重，保留 ResNet-101
（Caffe style）、DCNv2、四层 FPN、200×200 BEV、六层 Encoder、六层 Decoder
和 900 个检测 Query。六相机输入的默认 padded shape 为 `[1,6,3,928,1600]`。
这是 base 的独立 Profile，不能套用 tiny 的算量、延时或内存结论。

当前验证边界是 **固定标定、cold frame、FP32、三个 raw head 输出**。
输入 fixture 是已进入网络的张量；图像读取、官方 BGR 预处理、历史 BEV 更新、
box decode 和 nuScenes 指标不是该 fixture 的验证范围。30 FPS / 33.3 ms 是
目标，不能由编译成功、CPU 验证或 SimX 的主机耗时推导出已经达标。

当前完整官方 base ONNX 已通过原生 C++ 导入、图优化与 LLVM Dialect 编译，
生成 native 主机共享库及 Vortex ELF / vxbin，两者包含 3,224 个 dispatch。
全部 Vortex Kernel 入口、常量文件与生命周期 / Buffer 重叠审计通过。
最低聚合存储为 `2,032,441,772 B`，其中 arena 为 `1,554,944,000 B`，不含代码、
栈和 runtime 开销。完整 CPU 机器码已执行全部 3,224 个 dispatch，三个最终
输出与 PyTorch 参考在 `rtol=1e-3, atol=1e-4` 下对齐；尚无全图 Vortex 推理成绩。
详细证据见
[base 验证记录](../docs/validation/vortex/native-mlir-bevformer-base-2026-10-16/README.md)。

## 1. 编译链与实现位置

```text
官方模型 + checkpoint
  │ PyTorch CPU 执行与 ONNX 导出（仅模型导出/独立参考）
  ▼
ONNX opset 17
  ├─ 标准 Conv / Resize / MatMul / ...
  ├─ mmcv::MMCVModulatedDeformConv2d
  └─ gac::MSDeformAttn
  │ native/lib/OnnxImporter.cpp：protobuf、OpBuilder、SSA
  ▼
注册的 gac.conv2d / gac.resize / gac.deform_conv2d / gac.ms_deform_attn
  │ native/lib/GraphPasses.cpp + ConstantFolding.cpp：真实 MLIR Pass
  ▼
优化后的 ModuleOp
  │ native/lib/VortexLowering.cpp：算子模板、Buffer 生命周期、设备循环
  ▼
SCF / Arith / Math / LLVM / Vortex Operations
  ▼
LLVM Dialect → LLVM IR → VOLT → ELF / vxbin
  ▼
native/runtime/vortex-run.cpp → Vortex Queue / Event / Buffer
```

Python 不参与上述 ONNX 之后的图优化、设备 Kernel 实现或 Buffer 分配。
CPU 参考中的 torchvision/GridSample 不会进入 Vortex Runtime；目标设备所有
算术来自 C++ MLIR Lowering 生成的机器码。

## 2. 新增算子的精确契约

| ONNX 节点 | 注册的 MLIR Operation | 当前设备语义 |
|---|---|---|
| `mmcv::MMCVModulatedDeformConv2d`，domain opset 1 | `gac.deform_conv2d` | FP32 NCHW；真实 offset、mask、groups、deform_groups、stride、padding、dilation；可选 bias |
| `Resize` | `gac.resize` | 静态 sizes/scales，nearest + asymmetric + floor；不支持的插值/坐标规则明确报错 |
| `gac::MSDeformAttn`，domain opset 1 | `gac.ms_deform_attn` | FP32，多 batch/head/level/point，bilinear + zeros + align_corners=false |

DCNv2 输入顺序是 `X, offset, mask, weight[, bias]`。每个 deform group 的 offset
按采样点交错存储 `dy0, dx0, dy1, dx1, ...`，mask 已由原模型的 Sigmoid 产生。
Lowering 逐输出通道、输入通道和卷积点完成双线性采样、乘 mask、乘 weight、累加，
不会用普通 Conv 替换变形卷积。

MSDeformAttn 的四输入为：

```text
value              [B, S, heads, channels]
spatial_shapes     [levels, 2]                    INT64；Lowering 前必须成为常量
sampling_locations [B, Q, heads, levels, points, 2]
attention_weights  [B, Q, heads, levels, points]
output             [B, Q, heads × channels]
```

`S = sum(height[level] × width[level])`。归一化位置 `(u,v)` 在每层对应
`x=u×width−0.5, y=v×height−0.5`，对四个邻近像素进行带边界保护的采样，再乘
attention weight，并沿 level/point 累加。输入中需要运行时计算的 offset、mask、
采样位置和 attention weight 都保留为运行时张量。

自定义 ONNX 节点保留高层算子语义。普通 ONNX Runtime 无对应 custom-op 实现时
不能执行这种 ONNX；不能把 ONNX checker 通过说成 ORT 已验证全模型。

## 3. 为什么必须改变 Attention 的 Buffer 方案

在固定 all-visible 标定下，base 的旧 CPU 展开形式可能构造：

```text
6 cameras × 8 heads × 32 channels × 40,000 queries × 4 levels × 8 points
= 1,966,080,000 个 FP32 元素 = 7,864,320,000 bytes
```

这只是一个采样中间张量，尚未计乘权重和其他激活。原生模板保留高层 MSDA，
每个输出元素只持有一个累加器，按 level/point 读取所需像素并累加，不分配上述
采样 stack。导出端 CPU 参考也按 Query 分块、逐层采样，以独立验证同一数学语义。
CPU 参考的 chunk 参数不等于 Vortex 的 tile 配置。

设备端的内存优化由真实 C++ MLIR Pass 与 Lowering 实现：

- MatMul + Bias：在完整归约后应用 bias，省去独立 MatMul 中间张量；保留
  公开返回值、额外消费者和低精度舍入边界所需的未融合路径。
- MSDA 坐标融合：证明 Div / view / Add / view 链可内联后，在采样时按需计算
  坐标，避免把整份中间坐标张量写入再读回 DDR。
- 静态 ScatterND：逐常量索引证明完整有序覆盖或同 updates 广播链后，替换
  为 Identity / Reshape / Expand，删除不再被读取的初始零张量和中间复制。
- 内部连续 view：逻辑 shape 保留，物理 Buffer 复用其根张量，生命周期覆盖
  所有别名消费者；公开输出仍保留必要的 ABI 名称与物化。

前三级优化主要减少 DDR 中间张量与存活时间。实现分别在
[GraphPasses.cpp](lib/GraphPasses.cpp)、[TensorLegalization.cpp](lib/TensorLegalization.cpp)
与 [VortexLowering.cpp](lib/VortexLowering.cpp)，验证记录见
[base 算子与内存回归](../docs/validation/vortex/native-mlir-bevformer-base-2026-10-16/README.md)。
这些变化仍使用标量 SIMT 模板，当前 LMEM 使用量为 0，没有启用 DXA、TCU
或 LMEM 双缓冲；内存下降不能证明达到 30 FPS。验证边界仍是固定标定、cold
frame 与三个 raw head 输出。

全局 Buffer 规划采用有序 dispatch 的最后使用位置，释放死亡区间，合并相邻
空闲区间，从最小足够区间分配并拆分剩余空间。输入不能在同一次 dispatch 内被
输出覆盖，公开输出保留到最终读回。Runtime 再检查活跃区间是否重叠。

现阶段仍是标量 SIMT 正确性模板：**没有宣称 LMEM 双缓冲、DXA 或 TCU 已启用**。
保留 signed RV32 的约 2 GiB 聚合预算检查；真正超过预算时必须继续做 tiling、
融合或调度优化，不能直接删除检查。

## 4. 导出与编译

依赖版本与环境构建见 [原生编译器说明](README.md)。以下路径按本地安装调整。
官方模型权重来源为 [BEVFormer 官方模型表](https://github.com/fundamentalvision/BEVFormer)，
文件 `bevformer_r101_dcn_24ep.pth`，本次下载 SHA-256：

```text
5fda3ed3e0d2d1dc09baca337aa1ed64aece057fed7d8238525c5ed342649177
```

```bash
python tools/export_bevformer_onnx.py \
  --source-root /path/to/BEVFormer-master \
  --config /path/to/BEVFormer-master/projects/configs/bevformer/bevformer_base.py \
  --checkpoint build/checkpoints/bevformer_r101_dcn_24ep.pth \
  --trusted-checkpoint --camera-chunk 1 --reference-mode trace --threads 4 \
  --output-dir build/bevformer-base-onnx

gac-compile build/bevformer-base-onnx/bevformer-base-cold.onnx \
  --constant-fold-profile=bevformer-base --target=vortex --emit=llvm \
  --save-stages build/bevformer-base-native \
  --manifest build/bevformer-base-native/plan.json \
  --constants-dir build/bevformer-base-native \
  -o build/bevformer-base-native/kernel.mlir

# 一次完成原生编译、VOLT 链接及 vxbin 封装
GAC_CONSTANT_FOLD_PROFILE=bevformer-base \
GAC_NATIVE_BIN=/path/to/native-build LLVM_PATH=/path/to/volt-llvm20 \
VORTEX_BUILD=/path/to/vortex-build32 \
bash native/tools/compile-vortex.sh \
  build/bevformer-base-onnx/bevformer-base-cold.onnx build/bevformer-base-vortex
```

本次完整尺寸导出已在本机 WSL 完成：ONNX 有 22,412 个顶层节点、777 个
initializer、156 个 DCNv2 和 18 个 MSDeformAttn。`onnx.checker` 完整检查和
严格 shape inference 通过；`images.npy`、`pytorch-outputs.npz` 保存同一次实际
前向的输入和三个输出。产物与导出源码的 SHA-256、环境和历史失败记录见
[导出证据](../docs/validation/vortex/native-mlir-bevformer-base-2026-10-16/export-attempt.json)。
这些是模型导出与 CPU 参考结果，不能代替原生机器码整网数值验收。

导出实现位于 [export_bevformer_onnx.py](../tools/export_bevformer_onnx.py) 和
[bevformer_export_ops.py](../tools/bevformer_export_ops.py)。为控制主机内存，采用：

- `camera-chunk=1`：按原顺序逐相机执行原始 R101/DCNv2/FPN，再拼接特征。
  六相机、完整输入分辨率和模型权重均保留；BatchNorm 使用 eval 固定统计量。
- `reference-mode=trace`：从同一次 ONNX tracing 中捕获真实 CPU 前向结果，
  避免额外运行两遍整网。未宣称完整模型改写前后两次独立前向已对齐；算子替代
  和小尺寸原始 backbone/FPN 相机分批分别有独立数值对照。
- MSDA 自定义 `autograd.Function` 的 forward 仍执行真实双线性采样与累加。
  forward 内暂时关闭 PyTorch tracing，`finally` 恢复原状态，避免记录随后
  会被 symbolic 丢弃的庞大 CPU 参考实现；外层 symbolic 保留高层 ONNX 算子。
  这里依赖 PyTorch 2.0.1 的私有 tracing API，因此版本固定并做契约回归。

[导出契约测试](../tools/bevformer_export_contract_tests.py) 检查真实 custom-op
导出保留、单层/四层 MSDA 与原 MMCV 参考数值、DCNv2 offset 规则、tracing 的
真实输出与 eager 一致，以及参考执行异常时 tracing 状态仍被恢复。原始模型
小尺寸两相机 batch/逐相机比较最大绝对误差为 `5.34e-5`（容差 `1e-4`）。

成功导出的峰值 RSS 是 6,066,820 KiB，整次前向、导出与检查用时 542.15 秒；
**该时间不是 Vortex 推理延时，也不是纯 CPU forward 延时**。导出器仅捕获
固定 cold-frame raw head 路径；动态标定、历史帧和数据集精度需要另外验证。

`bevformer-base` 常量求值 Profile 显式扩大固定几何的有限预算；它只读取纯常量
SSA 运算，不能把图像输入或激活当作常量，也不改变设备内存上限。默认 Profile
继续服务原 tiny 路径。

## 5. 验证方式

[generate_base_ops_models.py](tests/generate_base_ops_models.py) 使用独立 PyTorch /
torchvision 实现产生数值 fixture；[run_base_ops_tests.py](tests/run_base_ops_tests.py)
调用真实 C++ 编译器、mlir-translate、Clang 生成主机机器码再执行比较，可选继续
经 VOLT 和 C++ Runtime 在 SimX 验证。覆盖 DCNv2 的分组/采样边界、MSDA 的多层
采样、Resize 的空可选参数、错误契约拒绝，以及有分支和保留输出的 Buffer 复用。

完整尺寸导出、严格 C++ 编译、Vortex ELF / vxbin 产物审核和完整 CPU 机器码
三输出数值对齐已完成。剩余验收为 Vortex 整网执行和目标硬件性能，以及
warm/history 连续帧、检测后处理和数据集精度；这些项目不能用 CPU 验证、
小算子测试或编译成功替代。
