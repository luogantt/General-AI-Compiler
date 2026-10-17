# TensorRT 逐层性能报告

**本次为无历史帧、合成投影条件下的静态引擎 profiling；不能作为完整真实时序性能或 nuScenes mAP 验收。**

打开 [index.html](index.html) 可离线搜索、排序、切换引擎并查看层详情。

## 公开归档说明

公开 SQLite 移除了全部环境元数据记录，并将 StringIds 中出现的凭据值替换为占位符；保留字符串 ID，其他采集数据（包括计时表）未改变。原始 .nsys-rep 二进制仅在本地保留，不纳入 Git 或下载 ZIP；profile-data.zip 是净化后的公开归档。

净化范围与计时表核对见 [publication-sanitization.json](publication-sanitization.json)。
[下载净化后的完整数据包](profile-data.zip)。

## 统计口径

层占比 = 该回调 mean_ms / 全部 IProfiler 回调 mean_ms 之和；不是 baseline 全图耗时占比，也不是 ONNX 原始节点占比。

每个 ordinal 回调仅计一次。分类依据唯一精确匹配的 inspector LayerType / PluginType；名称、Metadata 中的原始节点只供追溯，不用于推断或拆分耗时。TrainStation、kgen 和已融合执行层均保留原始回调整体；不将其内部节点额外加入统计。不据邻近层推断 TrainStation 的语义。回调统计不能证明底层 GPU 时间区间互斥，不能当作无重叠的 GPU 时间分解。

baseline、profiled 全图与逐层采样来自独立顺序执行。差值包含 profiling 扰动和运行间波动；加载、输入拷贝、warmup 不在计时内。CUDA event 也可能包含主机提交造成的 GPU 空闲。

Mixed 与 FP32 是两套已构建引擎，执行层数、融合方式和 tactic 均可能不同；此对比不是仅改变 precision 的单变量实验。层/类别耗时差不能直接解释为某个精度选项的因果收益，也不能仅凭这些计时断言计算或显存带宽瓶颈。

输出有限只检查各测量 pass 最后一次输出；本报告没有数值参考对比或 mAP 评估。

## 全图测量

| 引擎 | baseline mean / median / p95 (ms) | profiled mean / median / p95 (ms) | mean 扰动 | 回调数 | 层 mean 之和 (ms) | 唯一匹配 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Mixed | 209.418 / 204.224 / 232.416 | 217.215 / 213.511 / 230.218 | 7.797 ms (3.72%) | 3819 | 211.475 | 3819 |
| FP32 | 379.488 / 375.967 / 398.980 | 386.694 / 385.120 / 397.908 | 7.206 ms (1.90%) | 3260 | 380.366 | 3260 |

## Mixed

warmup=10，repeat=30；状态 `completed`。

Engine SHA256: `f89f51852eb5e7f3862bb213a2b7008742847c16710403e150491430337938a2`

Input SHA256: `b473352f8c2e4c88002a3d3fcf7323226d0c6cdf53b9ec66dac433a0d73c5968`

### 按可信执行层类别分组

| 类别 | 回调数 | mean 之和 (ms) | 层占比 |
| --- | ---: | ---: | ---: |
| MSDA | 18 | 74.188 | 35.08% |
| Convolution | 667 | 37.380 | 17.68% |
| kgen（代码生成/融合层） | 741 | 29.505 | 13.95% |
| DCN | 156 | 28.635 | 13.54% |
| Reformat | 1112 | 18.348 | 8.68% |
| MatrixMultiply | 153 | 11.231 | 5.31% |
| TrainStation（整体回调） | 8 | 3.492 | 1.65% |
| 其他已知类型 · PointWiseV2 | 316 | 2.695 | 1.27% |
| 其他已知类型 · DeviceToShapeHost | 36 | 2.549 | 1.21% |
| 其他已知类型 · Scale | 156 | 1.898 | 0.90% |
| 其他已知类型 · NonZero | 36 | 0.500 | 0.24% |
| 其他已知类型 · CaskPooling | 6 | 0.464 | 0.22% |
| 其他已知类型 · Shuffle | 4 | 0.380 | 0.18% |
| 其他已知类型 · Resize | 12 | 0.179 | 0.08% |
| 其他已知类型 · shape_call | 12 | 0.029 | 0.01% |
| 其他已知类型 · Constant | 200 | 0.000 | 0.00% |
| 其他已知类型 · NoOp | 186 | 0.000 | 0.00% |

Convolution / MatrixMultiply 类别可能包含 bias、激活或其他融合内容；数字代表整条回调，不能解释为单一原始算子的独立耗时。

### 最慢 20 条回调

| 排名 | ordinal（0 起） | mean (ms) | 层占比 | 类别 | 完整回调名 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 3104 | 11.728 | 5.55% | MSDA | /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MSDeformAttn |
| 2 | 3350 | 11.693 | 5.53% | MSDA | /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/MSDeformAttn |
| 3 | 3268 | 11.659 | 5.51% | MSDA | /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/MSDeformAttn |
| 4 | 3186 | 11.631 | 5.50% | MSDA | /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MSDeformAttn |
| 5 | 3432 | 11.624 | 5.50% | MSDA | /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/MSDeformAttn |
| 6 | 3022 | 11.377 | 5.38% | MSDA | /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MSDeformAttn |
| 7 | 2971 | 1.355 | 0.64% | Convolution | /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/value_proj/Add |
| 8 | 3183 | 1.034 | 0.49% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3000_21 |
| 9 | 3265 | 1.032 | 0.49% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3017_21 |
| 10 | 3347 | 1.031 | 0.49% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3034_21 |
| 11 | 3429 | 1.031 | 0.49% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3051_21 |
| 12 | 3101 | 1.020 | 0.48% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2983_21 |
| 13 | 3019 | 1.013 | 0.48% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2966_21 |
| 14 | 3155 | 0.792 | 0.37% | 其他已知类型 · DeviceToShapeHost | /pts_bbox_head/transformer/encoder/layers.2/attentions.1/NonZero_1[size][DevicetoShapeHostCopy] |
| 15 | 3318 | 0.735 | 0.35% | MSDA | /pts_bbox_head/transformer/encoder/layers.4/attentions.0/MSDeformAttn |
| 16 | 3151 | 0.734 | 0.35% | MSDA | /pts_bbox_head/transformer/encoder/layers.2/attentions.0/MSDeformAttn |
| 17 | 3400 | 0.733 | 0.35% | MSDA | /pts_bbox_head/transformer/encoder/layers.5/attentions.0/MSDeformAttn |
| 18 | 3236 | 0.732 | 0.35% | MSDA | /pts_bbox_head/transformer/encoder/layers.3/attentions.0/MSDeformAttn |
| 19 | 3072 | 0.721 | 0.34% | MSDA | /pts_bbox_head/transformer/encoder/layers.1/attentions.0/MSDeformAttn |
| 20 | 242 | 0.718 | 0.34% | MSDA | /pts_bbox_head/transformer/encoder/layers.0/attentions.0/MSDeformAttn |

### Nsight kernel Top 10

Nsight 为单独 trace pass。Total 为捕获范围内累计 kernel 时间；每帧累计 = Total / trace.frames，不等于单帧延迟，重叠 kernel 的时间也会累加。Instances 是所有捕获帧的调用总数。Kernel 占比分母为该 trace 全部 kernel 累计时间。NVTX push/pop 是 CPU range，不是 GPU 层耗时。

捕获帧数：5；累计 kernel：991.033 ms；每帧累计：198.207 ms。

| Kernel | Total (ms) | 每帧累计 (ms) | Instances（全部帧） | Kernel 占比 |
| --- | ---: | ---: | ---: | ---: |
| void ms_deformable_im2col_gpu_kernel<__half>(int, const T1 *, const int *, const int *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *) | 375.588 | 75.118 | 90 | 37.90% |
| void modulatedDeformableIm2colGpuKernel<__half>(int, const T1 *, const T1 *, const T1 *, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, T1 *) | 81.655 | 16.331 | 780 | 8.24% |
| sm86_xmma_fprop_implicit_gemm_interleaved_indexed_i8f16_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize64x128x64_stage4_warpsize2x2x1_g1_tensor16x8x32_execute_kernel_trt | 59.852 | 11.970 | 1710 | 6.04% |
| ampere_h1688gemm_128x128_ldg8_stages_32x1_nn | 54.206 | 10.841 | 690 | 5.47% |
| generatedNativePointwise | 53.837 | 10.767 | 2990 | 5.43% |
| void CUTENSOR_NAMESPACE::permutationKernelPLC3<CUTENSOR_NAMESPACE::VectorWrite2DTensorView<(unsigned char)0, (unsigned char)1, (bool)0, (unsigned int)8, float, signed char, CUTENSOR_NAMESPACE::GeneralUnarySmall<float>>, CUTENSOR_NAMESPACE::VectorRead2DTensorView<(unsigned char)0, (unsigned char)1, (bool)0, (unsigned int)8, float, __half, CUTENSOR_NAMESPACE::GeneralUnarySmall<float>>, CUTENSOR_NAMESPACE::ThreadLevelElementwise<CUTENSOR_NAMESPACE::ElementwiseConfig2DCommonCase<CUTENSOR_NAMESPACE::GeneralUnarySmall<float>, CUTENSOR_NAMESPACE::GeneralBinary<float>, (int)2, (int)32, (int)64, (int)128, (signed char)8, (bool)1, (bool)0, (bool)1, (bool)0, (bool)0>, float>, CUTENSOR_NAMESPACE::ElementwiseRuntimePLC3<__half, float, signed char, float>::Params>(T4) | 40.024 | 8.005 | 1410 | 4.04% |
| sm80_xmma_gemm_f16f16_f16f16_f16_tn_n_tilesize256x128x32_stage3_warpsize4x2x1_tensor16x8x16_execute_kernel_trt | 31.259 | 6.252 | 30 | 3.15% |
| sm80_xmma_fprop_implicit_gemm_interleaved_i8f16_i8i32_f32_nchw_vect_c_32kcrs_vect_c_32_nchw_vect_c_32_tilesize64x64x64_stage6_warpsize2x2x1_g1_tensor16x8x32_t1r1s1_execute_kernel_trt | 19.944 | 3.989 | 240 | 2.01% |
| __myl_ReshReshMulReshAddReshCast_0xc0e6c6bed6e631626b4ccd587de5b41d | 18.761 | 3.752 | 30 | 1.89% |
| sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize128x32x32_stage4_warpsize4x1x1_g1_tensor16x8x16_t1r3s3_execute_kernel_trt | 13.292 | 2.658 | 690 | 1.34% |

### 采集数据（公开版）

- [report.json](mixed-layers/report.json)
- [layers.json](mixed-layers/layers.json)
- [layers.raw.json](mixed-layers/layers.raw.json)
- [layers.csv](mixed-layers/layers.csv)
- [engine.layers.json](mixed-layers/engine.layers.json)
- [mixed-stats_cuda_gpu_kern_sum.csv](mixed-stats_cuda_gpu_kern_sum.csv)
- [mixed-stats_cuda_api_sum.csv](mixed-stats_cuda_api_sum.csv)
- [mixed-stats_nvtx_pushpop_sum.csv](mixed-stats_nvtx_pushpop_sum.csv)
- [report.json](mixed-trace/report.json)
- [nsys-summary.json](nsys-summary.json)
- [mixed-nsys-summary-kernel-ranked.csv](mixed-nsys-summary-kernel-ranked.csv)
- [mixed-nsys-summary-kernel-calls.csv](mixed-nsys-summary-kernel-calls.csv)
- [mixed-nsys-summary-frames.csv](mixed-nsys-summary-frames.csv)
- [mixed-nsys.sqlite](mixed-nsys.sqlite)
- [publication-sanitization.json](publication-sanitization.json)

## FP32

warmup=10，repeat=30；状态 `completed`。

Engine SHA256: `171b028d9583f77ca42e123017d3ba4feba4fdbbf1b17dc0eb5b35f0cd63e9d0`

Input SHA256: `b473352f8c2e4c88002a3d3fcf7323226d0c6cdf53b9ec66dac433a0d73c5968`

### 按可信执行层类别分组

| 类别 | 回调数 | mean 之和 (ms) | 层占比 |
| --- | ---: | ---: | ---: |
| Convolution | 666 | 107.068 | 28.15% |
| MSDA | 18 | 78.708 | 20.69% |
| DCN | 156 | 78.242 | 20.57% |
| kgen（代码生成/融合层） | 819 | 52.523 | 13.81% |
| MatrixMultiply | 164 | 47.660 | 12.53% |
| Reformat | 468 | 6.717 | 1.77% |
| TrainStation（整体回调） | 8 | 3.385 | 0.89% |
| 其他已知类型 · Scale | 156 | 2.953 | 0.78% |
| 其他已知类型 · PointWiseV2 | 156 | 0.879 | 0.23% |
| 其他已知类型 · CaskPooling | 6 | 0.861 | 0.23% |
| 其他已知类型 · NonZero | 36 | 0.514 | 0.14% |
| 其他已知类型 · DeviceToShapeHost | 36 | 0.449 | 0.12% |
| 其他已知类型 · Resize | 12 | 0.375 | 0.10% |
| 其他已知类型 · shape_call | 12 | 0.030 | 0.01% |
| 其他已知类型 · Constant | 192 | 0.000 | 0.00% |
| 其他已知类型 · NoOp | 355 | 0.000 | 0.00% |

Convolution / MatrixMultiply 类别可能包含 bias、激活或其他融合内容；数字代表整条回调，不能解释为单一原始算子的独立耗时。

### 最慢 20 条回调

| 排名 | ordinal（0 起） | mean (ms) | 层占比 | 类别 | 完整回调名 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2509 | 12.427 | 3.27% | MSDA | /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MSDeformAttn |
| 2 | 2420 | 12.386 | 3.26% | MSDA | /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MSDeformAttn |
| 3 | 2687 | 12.312 | 3.24% | MSDA | /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/MSDeformAttn |
| 4 | 2776 | 12.295 | 3.23% | MSDA | /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/MSDeformAttn |
| 5 | 2598 | 12.254 | 3.22% | MSDA | /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MSDeformAttn |
| 6 | 2865 | 12.245 | 3.22% | MSDA | /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/MSDeformAttn |
| 7 | 2417 | 3.996 | 1.05% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2357_28 |
| 8 | 2506 | 3.965 | 1.04% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2374_28 |
| 9 | 2595 | 3.948 | 1.04% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2391_28 |
| 10 | 2773 | 3.923 | 1.03% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2425_28 |
| 11 | 2684 | 3.923 | 1.03% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2408_28 |
| 12 | 2862 | 3.923 | 1.03% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2442_28 |
| 13 | 2508 | 1.178 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2374_30 |
| 14 | 2419 | 1.178 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2357_30 |
| 15 | 2597 | 1.178 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2391_30 |
| 16 | 2686 | 1.178 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2408_30 |
| 17 | 2864 | 1.177 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2442_30 |
| 18 | 2775 | 1.177 | 0.31% | kgen（代码生成/融合层） | __myl_ReshReshMulReshAddReshCast_myl2425_30 |
| 19 | 2479 | 1.117 | 0.29% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/value_proj/MatMul_myl2374_1 |
| 20 | 2568 | 1.105 | 0.29% | MatrixMultiply | /pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/value_proj/MatMul_myl2391_1 |

### Nsight kernel Top 10

Nsight 为单独 trace pass。Total 为捕获范围内累计 kernel 时间；每帧累计 = Total / trace.frames，不等于单帧延迟，重叠 kernel 的时间也会累加。Instances 是所有捕获帧的调用总数。Kernel 占比分母为该 trace 全部 kernel 累计时间。NVTX push/pop 是 CPU range，不是 GPU 层耗时。

捕获帧数：5；累计 kernel：1827.099 ms；每帧累计：365.420 ms。

| Kernel | Total (ms) | 每帧累计 (ms) | Instances（全部帧） | Kernel 占比 |
| --- | ---: | ---: | ---: | ---: |
| void ms_deformable_im2col_gpu_kernel<float>(int, const T1 *, const int *, const int *, const T1 *, const T1 *, int, int, int, int, int, int, int, T1 *) | 390.007 | 78.001 | 90 | 21.35% |
| ampere_sgemm_128x64_nn | 254.514 | 50.903 | 690 | 13.93% |
| sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x256x8_stage3_warpsize2x4x1_ffma_aligna4_alignc4_execute_kernel_trt | 194.850 | 38.970 | 240 | 10.66% |
| sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x64x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel_trt | 148.932 | 29.786 | 925 | 8.15% |
| sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x64x8_stage3_warpsize2x2x1_ffma_aligna4_alignc4_execute_kernel_trt | 125.103 | 25.021 | 720 | 6.85% |
| sm80_xmma_fprop_wngd_f32f32_f32_f32_nchwkcrs_nchw_tilesize8x16x64x8_warpsize8x1x1_wngd2x2_execute_kernel_trt | 83.117 | 16.623 | 300 | 4.55% |
| void modulatedDeformableIm2colGpuKernel<float>(int, const T1 *, const T1 *, const T1 *, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, T1 *) | 82.686 | 16.537 | 780 | 4.53% |
| trt_ampere_scudnn_128x64_relu_interior_nn_v1 | 55.322 | 11.064 | 270 | 3.03% |
| void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params) | 47.098 | 9.420 | 90 | 2.58% |
| trt_ampere_scudnn_128x64_relu_medium_nn_v1 | 41.007 | 8.201 | 150 | 2.24% |

### 采集数据（公开版）

- [report.json](fp32-layers/report.json)
- [layers.json](fp32-layers/layers.json)
- [layers.raw.json](fp32-layers/layers.raw.json)
- [layers.csv](fp32-layers/layers.csv)
- [engine.layers.json](fp32-layers/engine.layers.json)
- [fp32-stats_cuda_gpu_kern_sum.csv](fp32-stats_cuda_gpu_kern_sum.csv)
- [fp32-stats_cuda_api_sum.csv](fp32-stats_cuda_api_sum.csv)
- [fp32-stats_nvtx_pushpop_sum.csv](fp32-stats_nvtx_pushpop_sum.csv)
- [report.json](fp32-trace/report.json)
- [nsys-summary.json](nsys-summary.json)
- [fp32-nsys-summary-kernel-ranked.csv](fp32-nsys-summary-kernel-ranked.csv)
- [fp32-nsys-summary-kernel-calls.csv](fp32-nsys-summary-kernel-calls.csv)
- [fp32-nsys-summary-frames.csv](fp32-nsys-summary-frames.csv)
- [fp32-nsys.sqlite](fp32-nsys.sqlite)
- [publication-sanitization.json](publication-sanitization.json)

## 完整导出与复现

- [overview.json](overview.json)：全图统计、匹配覆盖、类别和前 20 层。
- [layers-ranked.csv](layers-ranked.csv)：全部回调，按引擎和耗时排名；保留 ordinal、全部原始统计及匹配元数据（UTF-8 BOM）。
- [operators.csv](operators.csv)：执行层类别汇总，非原始 ONNX 算子独立计时（UTF-8 BOM）。

```powershell
python tools/tensorrt/render_profile_report.py --root "C:/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler/docs/validation/tensorrt/bevformer-base-2026-10-17/profiling"
```

脚本只需 Python 标准库，无需 TensorRT / CUDA；只读取 report、layers、inspector 和 trace 数据，不修改采集数据或执行净化。
HTML 内嵌全部回调和匹配元数据，不使用 fetch、CDN 或外部 JavaScript。
