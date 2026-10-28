#loc = loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":0:0)
#loc1 = loc("Scale")
#loc2 = loc("Bias")
#loc3 = loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":1:0)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#loc4 = loc("LayerNormalization#0"(#loc3))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<2x3x19xf32> loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":0:0)) -> tensor<2x3x19xf32> attributes {gac.input_names = ["X"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {value = dense<[0.979446947, -0.651603103, -0.642044961, 0.320986778, -0.44301039, 0.950704514, -0.202663615, -0.963335216, -0.0691407919, 0.365349412, -0.480517119, 0.367128342, 0.710562348, 0.269450575, 0.976848125, -0.130192608, 0.302919269, -0.128498822, 0.77930522]> : tensor<19xf32>} : () -> tensor<19xf32> loc(#loc1)
    %1 = "gac.constant"() {value = dense<[-0.146774337, 0.789747357, 0.957665085, -0.598656714, -0.100228608, 0.3307437, 0.560321152, 0.299635798, -0.135275856, 0.749515533, 0.919398069, -0.278547883, -0.227929249, -0.0678850114, -0.862052321, 0.561740637, 0.82482934, 0.316964924, 0.931097388]> : tensor<19xf32>} : () -> tensor<19xf32> loc(#loc2)
    %2 = tensor.empty() : tensor<2x3xf32> loc(#loc4)
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<2x3xf32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("LayerNormalization#0"(#loc3))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc4)
      linalg.yield %cst : f32 loc(#loc4)
    } -> tensor<2x3xf32> loc(#loc4)
    %4 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0 : tensor<2x3x19xf32>) outs(%3 : tensor<2x3xf32>) attrs =  {gac.linalg_role = "sum-reduction"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":0:0), %out: f32 loc("LayerNormalization#0"(#loc3))):
      %12 = arith.addf %out, %in : f32 loc(#loc4)
      linalg.yield %12 : f32 loc(#loc4)
    } -> tensor<2x3xf32> loc(#loc4)
    %5 = tensor.empty() : tensor<2x3xf32> loc(#loc4)
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<2x3xf32>) outs(%5 : tensor<2x3xf32>) attrs =  {gac.linalg_role = "mean"} {
    ^bb0(%in: f32 loc("LayerNormalization#0"(#loc3)), %out: f32 loc("LayerNormalization#0"(#loc3))):
      %cst = arith.constant 1.900000e+01 : f32 loc(#loc4)
      %12 = arith.divf %in, %cst : f32 loc(#loc4)
      linalg.yield %12 : f32 loc(#loc4)
    } -> tensor<2x3xf32> loc(#loc4)
    %7 = tensor.empty() : tensor<2x3xf32> loc(#loc4)
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%7 : tensor<2x3xf32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("LayerNormalization#0"(#loc3))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc4)
      linalg.yield %cst : f32 loc(#loc4)
    } -> tensor<2x3xf32> loc(#loc4)
    %9 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %6 : tensor<2x3x19xf32>, tensor<2x3xf32>) outs(%8 : tensor<2x3xf32>) attrs =  {gac.linalg_role = "squared-deviation-reduction"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":0:0), %in_0: f32 loc("LayerNormalization#0"(#loc3)), %out: f32 loc("LayerNormalization#0"(#loc3))):
      %12 = arith.subf %in, %in_0 : f32 loc(#loc4)
      %13 = arith.mulf %12, %12 : f32 loc(#loc4)
      %14 = arith.addf %out, %13 : f32 loc(#loc4)
      linalg.yield %14 : f32 loc(#loc4)
    } -> tensor<2x3xf32> loc(#loc4)
    %10 = tensor.empty() : tensor<2x3x19xf32> loc(#loc4)
    %11 = linalg.generic {indexing_maps = [#map1, #map2, #map2, #map3, #map3, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %6, %9, %0, %1 : tensor<2x3x19xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<19xf32>, tensor<19xf32>) outs(%10 : tensor<2x3x19xf32>) attrs =  {gac.linalg_role = "layer-normalization", gac.lowered_from = "gac.layer_norm", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "LayerNormalization#0", gac.source_op = "LayerNormalization"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-layernorm-gac-final7-20261022/triton_layer_norm_tail_19/model.onnx":0:0), %in_0: f32 loc("LayerNormalization#0"(#loc3)), %in_1: f32 loc("LayerNormalization#0"(#loc3)), %in_2: f32 loc("Scale"), %in_3: f32 loc("Bias"), %out: f32 loc("LayerNormalization#0"(#loc3))):
      %cst = arith.constant 1.900000e+01 : f32 loc(#loc4)
      %12 = arith.divf %in_1, %cst : f32 loc(#loc4)
      %cst_4 = arith.constant 9.765625E-4 : f32 loc(#loc4)
      %13 = arith.addf %12, %cst_4 : f32 loc(#loc4)
      %14 = math.sqrt %13 : f32 loc(#loc4)
      %15 = arith.subf %in, %in_0 : f32 loc(#loc4)
      %16 = arith.divf %15, %14 : f32 loc(#loc4)
      %17 = arith.mulf %16, %in_2 : f32 loc(#loc4)
      %18 = arith.addf %17, %in_3 : f32 loc(#loc4)
      linalg.yield %18 : f32 loc(#loc4)
    } -> tensor<2x3x19xf32> loc(#loc4)
    return %11 : tensor<2x3x19xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)

