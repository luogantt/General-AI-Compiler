#loc = loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/triton_reduce_sum_scalar/model.onnx":0:0)
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-reduction-integration-20261022-final2/triton_reduce_sum_scalar/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<37xf32> loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/triton_reduce_sum_scalar/model.onnx":0:0)) -> tensor<f32> attributes {gac.input_names = ["A"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {gac.output_names = ["axes"], gac.source_name = "axes", value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64> loc(#loc1)
    %1 = "gac.reduce_sum"(%arg0, %0) {gac.input_positions = array<i64: 0, 1>, gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "ReduceSum#0", gac.source_op = "ReduceSum", keepdims = 0 : i64} : (tensor<37xf32>, tensor<1xi64>) -> tensor<f32> loc(#loc3)
    return %1 : tensor<f32> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("axes")
#loc2 = loc("/home/thomas/gac-triton-reduction-integration-20261022-final2/triton_reduce_sum_scalar/model.onnx":1:0)
#loc3 = loc("ReduceSum#0"(#loc2))

