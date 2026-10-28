# TensorRT 层性能摘要

每个 ordinal 回调仅计一次。类别取自唯一精确匹配的 Inspector LayerType / PluginType；重名、未匹配或缺少类型的回调标为 unknown。融合层保持整体，不拆成内部 ONNX 算子。

回调数：3816；回调 mean 之和：334.953 ms。

**下面的占比是回调 mean 之和的占比，不能当作无重叠的 GPU 时间分解，也不是 baseline 整网时间占比。** Profiling 可能扰动运行；全图计时排除加载、输入拷贝和预热。

| 测量 | CUDA Event mean (ms) |
| --- | ---: |
| 未附加 profiler 的整网 | 333.107 |
| 附加 profiler 的整网 | 337.387 |

## 按执行层类别

| 类别 | 回调数 | mean 合计 (ms) | 回调占比 |
| --- | ---: | ---: | ---: |
| PluginV3/GACMSDeformAttn | 6 | 76.841 | 22.94% |
| CaskConvolution | 649 | 64.889 | 19.37% |
| kgen | 1017 | 61.878 | 18.47% |
| PluginV3/ModulatedDeformConv2d | 156 | 57.089 | 17.04% |
| Reformat | 836 | 27.243 | 8.13% |
| gemm | 152 | 23.781 | 7.10% |
| PluginV3/MultiscaleDeformableAttnPlugin_TRT | 12 | 10.845 | 3.24% |
| PointWiseV2 | 316 | 3.879 | 1.16% |
| Scale | 156 | 2.889 | 0.86% |
| CaskGemmConvolution | 18 | 1.425 | 0.43% |
| TrainStation | 8 | 1.260 | 0.38% |
| CaskPooling | 6 | 0.907 | 0.27% |
| Shuffle | 4 | 0.798 | 0.24% |
| NonZero | 36 | 0.589 | 0.18% |
| DeviceToShapeHost | 36 | 0.316 | 0.09% |
| Resize | 12 | 0.303 | 0.09% |
| shape_call | 12 | 0.019 | 0.01% |
| Constant | 200 | 0.000 | 0.00% |
| NoOp | 184 | 0.000 | 0.00% |

## 最慢 20 条回调

| ordinal | 类别 | mean (ms) | 名称 |
| ---: | --- | ---: | --- |
| 3102 | PluginV3/GACMSDeformAttn | 13.325 | /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/MSDeformAttn |
| 3184 | PluginV3/GACMSDeformAttn | 13.175 | /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/MSDeformAttn |
| 3266 | PluginV3/GACMSDeformAttn | 13.116 | /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/MSDeformAttn |
| 3348 | PluginV3/GACMSDeformAttn | 12.916 | /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/MSDeformAttn |
| 3430 | PluginV3/GACMSDeformAttn | 12.861 | /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/MSDeformAttn |
| 3020 | PluginV3/GACMSDeformAttn | 11.448 | /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/MSDeformAttn |
| 2970 | CaskConvolution | 2.553 | /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.0/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.1/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.2/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.3/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.4/attentions.1/deformable_attention/value_proj/Add \|\| /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/value_proj/MatMul + /pts_bbox_head/transformer/encoder/layers.5/attentions.1/deformable_attention/value_proj/Add |
| 3181 | gemm | 2.448 | /pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_2/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2998_21 |
| 3263 | gemm | 2.427 | /pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_3/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3015_21 |
| 3099 | gemm | 2.426 | /pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_1/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2981_21 |
| 3427 | gemm | 2.376 | /pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_5/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3049_21 |
| 3345 | gemm | 2.372 | /pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_4/attentions_1/deformable_attention/sampling_offsets/MatMul_myl3032_21 |
| 3017 | gemm | 2.072 | /pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/attention_weights/MatMul+/pts_bbox_head/transformer/encoder/layers_0/attentions_1/deformable_attention/sampling_offsets/MatMul_myl2964_21 |
| 3152 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.831 | /pts_bbox_head/transformer/encoder/layers.2/attentions.0/MSDeformAttn |
| 3234 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.808 | /pts_bbox_head/transformer/encoder/layers.3/attentions.0/MSDeformAttn |
| 3316 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.798 | /pts_bbox_head/transformer/encoder/layers.4/attentions.0/MSDeformAttn |
| 3398 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.787 | /pts_bbox_head/transformer/encoder/layers.5/attentions.0/MSDeformAttn |
| 3070 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.747 | /pts_bbox_head/transformer/encoder/layers.1/attentions.0/MSDeformAttn |
| 242 | PluginV3/MultiscaleDeformableAttnPlugin_TRT | 1.698 | /pts_bbox_head/transformer/encoder/layers.0/attentions.0/MSDeformAttn |
| 3183 | kgen | 1.588 | __myl_ReshReshMulReshAddReshCast_myl2998_23 |

## 追溯

匹配状态：`{"matched": 3816}`。

[完整摘要与全部 ordinal](summary.json) · [Top 20 CSV](top.csv)

Engine SHA-256：`9fb58a02198c63e017ecc09bbee40b23fd99cfa214df4f0391dc9b5fe9955c98`

输入 SHA-256：`b473352f8c2e4c88002a3d3fcf7323226d0c6cdf53b9ec66dac433a0d73c5968`

本摘要只分析计时，不新增数值正确性、完整时序或 mAP/NDS 验证。
