# BEVFormer Base：TensorRT 静态内存分析

分析对象为 `build/tensorrt/base-cold-plugins.onnx`，由固定 cold Base 导出图适配得到。以下为单个 Encoder spatial-attention 的逻辑张量大小：输入 shape 来自本地 ONNX shape inference，MSDA 输出 shape 按 TensorRT v2 插件 ABI 推导。FP32 每元素 4 字节，MiB 按 `2^20` 字节换算；六个 Encoder 层存在同样形状，但不能因此把表中大小乘六当作峰值。

| 对象 | Shape / 字节计算 | 字节数 | MiB |
|---|---|---:|---:|
| MSDA value | `[6,30825,8,32] × 4` | 189,388,800 | 180.62 |
| Sampling locations | `[6,40000,8,4,8,2] × 4` | 491,520,000 | 468.75 |
| Attention weights | `[6,40000,8,4,8] × 4` | 245,760,000 | 234.38 |
| MSDA output | `[6,40000,8,32] × 4` | 245,760,000 | 234.38 |
| 上述四项合计 | 不含残差、其他活跃张量与分配器开销 | 1,172,428,800 | 1,118.12 |
| DCN stage 3 单次 workspace | `256 × 3 × 3 × 58 × 100 × 4` | 53,452,800 | 50.98 |
| DCN stage 4 单次 workspace | `512 × 3 × 3 × 29 × 50 × 4` | 26,726,400 | 25.49 |

最大已知单张量对应 `encoder/layers.0/attentions.1/deformable_attention/sampling_offsets/MatMul` 等节点，输出 `[6,40000,512]`，后续变为表中的六维 locations。NVIDIA TensorRT **release/10.13** 源码规定 [MSDA 插件 workspace 为 0](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.cpp#L231)，[输出为 `[N,Lq,M,D]`](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableAttnPlugin.cpp#L113)；[DCN workspace](https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/modulatedDeformConvPlugin/modulatedDeformConvPlugin.cpp#L169)为 `Cin × Kw × Kh × Hout × Wout × sizeof(dtype)`，向 16 字节对齐。本图的 138 个 stage 3 DCN 与 18 个 stage 4 DCN 分别对应表中两种大小。**这些是静态容量计算，不是实测显存峰值。** 构建时失败的 `1,789,657,600` 字节申请不能据此定位到某一层，也不等于单个 DCN 或 MSDA 插件的 workspace；它可能属于合并的 activation/scratch 分配，必须结合分配器日志确认。

下一步可先限制辅助 stream 与 tactic workspace，但这不会消除上述 Attention 输入输出。更直接的方案是将 **sampling-offset/weight 投影 → 归一化 → MSDA → 输出**一起按 camera 或 query 分块；仅切 MSDA 本体，仍会提前生成完整 locations。当前 backbone 已逐 camera 执行，spatial-attention 仍处理六个 camera。若每次处理一个 camera，上述四项逻辑容量由 1,172,428,800 字节降为 195,404,800 字节（186.35 MiB）；再将 query 分为最多 4096 个一组，保留单 camera 完整 value 时，四项合计为 48,342,016 字节（46.10 MiB）。这只是指定分块方式的容量估算，仍需保证编译器实际复用缓冲区，并保留跨 camera 累加、可见性掩码、归一化及尾块语义，经过精度验证后再评估实际效果。
