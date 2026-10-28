# Triton / Vortex LayerNorm 验证记录

本目录保存 GAC 原生 C++ MLIR 前端把完整 ONNX `LayerNormalization`
子图原子地交给 Triton/Vortex 后端的验证证据。验证目标是正确性、ABI
一致性和失败边界；这些结果不是硬件性能数据。

## 结论

2026-10-22 的完整集成检查为 **81/81 通过**，其中 33 项直接覆盖
LayerNorm 的选择、执行、完整 native 回退和非法 Manifest 拒绝。

```text
ONNX LayerNormalization
        ↓ C++ ONNX importer / legalize-to-Linalg
完整、闭合的 LayerNorm Linalg 子图
        ↓ 原子匹配
一个 TTIR LayerNorm kernel
        ↓ Triton/Vortex C++ lowering
VOLT → Vortex vxbin
        ↓ GAC Runtime
SimX 设备执行
```

首版 Triton 合同严格限制为：

- FP32，静态 `rank >= 2`；
- 最后一轴归一化，`1 <= width <= 1024`；
- 同时具有 Scale 和 Bias；
- ONNX `stash_type=1`、单输出；
- `epsilon` 是有限正 FP32；
- `X`、Scale、Bias 和 Y 的物理 Buffer 不别名，且 View 根的物理 shape
  与逻辑参数 shape 完全相同；
- `rows * width <= INT32_MAX`。

任何条件不满足时，完整 LayerNorm 统计链保留为 **3 个 native Vortex
dispatch**；不会把内部 mean 或 reduction 单独选成 Triton，也不会退到
CPU/CUDA。

## 原子子图与 TTIR

匹配器要求以下五个内部 Linalg operation 与最终 root 完整闭合：

```text
zero-init → sum-reduction → mean
zero-init → squared-deviation-reduction
mean + variance + X + Scale + Bias → LayerNorm root
```

合法路径把五个内部 operation 原子移除，Manifest 中精确记录
`linalg_fused_producers == 5`，最终只生成一个 Triton dispatch。TTIR 执行：

```text
masked load X (tail other = 0)
  → add reduce 得到 sum
  → mean = sum / N
  → centered = X - mean
  → select(mask, centered, 0)       # 平方前把尾 lane 清零
  → square
  → add reduce 得到 sumsq
  → variance = sumsq / N
  → sqrt(variance + epsilon)
  → centered / stddev
  → * Scale + Bias
  → masked store Y
```

参数 ABI 为：

```text
X pointer / Scale pointer / Bias pointer / Y pointer / N:i32 / epsilon:f32
```

Manifest 名称为 `vortex-u64-pointers-i32-n-f32-epsilon`。Runtime 为每个
参数保留 8 字节槽；`N` 和 `epsilon` 使用各自槽的低 32 位。`1e-5` 在
本次证据中的原始 FP32 位模式是 `0x3727c5ac`（十进制 `925353388`）。

## 数值与边界结果

| 用例 | 路径 | 验证 | 最大绝对误差 |
|---|---:|---|---:|
| BEV hidden width 256，shape `[2,256]` | 1 Triton / 0 native | SimX | `4.76837158203125e-7` |
| 非二次幂尾块，shape `[2,3,19]` | 1 Triton / 0 native | SimX | `1.1920928955078125e-7` |
| width 1，epsilon `1e-5` | 1 Triton / 0 native | SimX | `0` |
| width 1024 | 1 Triton / 0 native | 完整编译和选择 | 未执行 |
| rank 1、非末轴、缺 Bias、别名、View shape 不同、非法 epsilon | 0 Triton / 3 native | 小形状均完成 SimX；超宽及非有限 epsilon 为编译/选择验证 | 见 `validation.json` |
| 非末轴 LayerNorm + 独立 Add | 1 Triton sibling / 3 native LayerNorm | SimX | LayerNorm `3.5762786865234375e-7`；Add `0` |

`selector-contract.log` 还覆盖了 ONNX 无法直接表达的 internal SSA fanout：
测试把 mean 暴露为第二个函数输出，再调用完整 DeviceLowering，精确得到
`0 Triton + 3 native + 3 fused`。这证明 rejection 不会发生半融合。

15 组 Manifest 变异覆盖 grid、rows、`rows*width` 溢出、width、N、tile、
epsilon、ABI、输入重复、输出别名以及 X/Scale/Y shape。Python 构建驱动
和 C++ Runtime 均在打开设备前拒绝这些错误。

## 复现命令

先构建编译器、Runtime 和 device-planning 合同测试：

```bash
cmake --build "$GAC_NATIVE_BUILD" \
  --target gac-compile gac-vortex-run \
           gac-triton-layernorm-selector-contract -j4
```

确认使用本次 LayerNorm 后端，而不是旧的已安装二进制，然后运行集成：

```bash
export TRITON_VORTEX_OPT=/path/to/triton/third_party/vortex/build/triton-vortex-opt
export PYTHONPATH=/path/to/python/site-packages
export TRITON_CACHE_DIR=/tmp/gac-triton-layernorm-cache
export VORTEX_DRIVER=simx
export LD_LIBRARY_PATH="$VORTEX_BUILD/sw/runtime:${LD_LIBRARY_PATH:-}"

python native/tests/run_triton_vortex_tests.py \
  --compiler "$GAC_NATIVE_BUILD/gac-compile" \
  --runtime "$GAC_NATIVE_BUILD/gac-vortex-run" \
  --output /tmp/gac-triton-layernorm-validation
```

完整运行使用的精确命令、源文件和二进制 SHA-256 在
[`evidence.json`](evidence.json) 中。

## 证据文件

- [`validation.json`](validation.json)：81 项逐项结果。
- [`width-256`](width-256/)：BEV 关键宽度的 Linalg、原始/解析后 TTIR、
  plan 和 Runtime 报告。
- [`tail-19`](tail-19/)：非二次幂尾 lane 数值证据。
- [`width-1`](width-1/)：`epsilon=1e-5` 位级传递证据。
- [`width-1024`](width-1024/)：合同上界的完整编译证据。
- [`native-fallbacks`](native-fallbacks/)：每类不支持条件的 0 Triton / 3
  native plan。
- [`mixed-sibling`](mixed-sibling/)：LayerNorm 回退不误阻止独立 Triton
  pointwise 的同图证据。
- [`rejections`](rejections/)：15 组非法 Manifest 与 Runtime 拒绝日志。
- [`selector-contract.log`](selector-contract.log)：internal fanout 的完整
  DeviceLowering 合同结果。

width 1024 在 GAC 集成中只做完整编译/选择验证，因为小型 SimX 上执行该
边界不是有意义的性能或回归负载。当前 LayerNorm 后端是串行正确性模板；
本目录不声称性能提升，也不代表完整 BEVFormer 已全部走 Triton。
