#loc = loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/native_reduce_sum_too_wide/model.onnx":0:0)
#loc2 = loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/native_reduce_sum_too_wide/model.onnx":1:0)
#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#loc3 = loc("ReduceSum#0"(#loc2))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-reduction-integration-20261022-final2/native_reduce_sum_too_wide/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<1025xf32> loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/native_reduce_sum_too_wide/model.onnx":0:0)) -> tensor<f32> attributes {gac.input_names = ["A"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64> loc(#loc1)
    %1 = tensor.empty() : tensor<f32> loc(#loc3)
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = []} outs(%1 : tensor<f32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("ReduceSum#0"(#loc2))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc3)
      linalg.yield %cst : f32 loc(#loc3)
    } -> tensor<f32> loc(#loc3)
    %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction"]} ins(%arg0 : tensor<1025xf32>) outs(%2 : tensor<f32>) attrs =  {gac.linalg_role = "sum-reduction", gac.lowered_from = "gac.reduce_sum", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "ReduceSum#0", gac.source_op = "ReduceSum"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/native_reduce_sum_too_wide/model.onnx":0:0), %out: f32 loc("ReduceSum#0"(#loc2))):
      %4 = arith.addf %out, %in : f32 loc(#loc3)
      linalg.yield %4 : f32 loc(#loc3)
    } -> tensor<f32> loc(#loc3)
    return %3 : tensor<f32> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("axes")

