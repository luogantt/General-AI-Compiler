#loc = loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":0:0)
#loc2 = loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":1:0)
#loc3 = loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":2:0)
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#loc4 = loc("Reshape#0"(#loc2))
#loc5 = loc("MatMul#1"(#loc3))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<34x32xf32> loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":0:0), %arg1: tensor<2x32x19xf32> loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":0:0)) -> tensor<2x17x19xf32> attributes {gac.input_names = ["ARaw", "B"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {value = dense<[2, 17, 32]> : tensor<3xi64>} : () -> tensor<3xi64> loc(#loc1)
    %1 = "gac.reshape"(%arg0, %0) {gac.input_positions = array<i64: 0, 1>, gac.opset = 18 : i64, gac.source_domain = "", gac.source_op = "Reshape"} : (tensor<34x32xf32>, tensor<3xi64>) -> tensor<2x17x32xf32> loc(#loc4)
    %2 = tensor.empty() : tensor<2x17x19xf32> loc(#loc5)
    %3 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%2 : tensor<2x17x19xf32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("MatMul#1"(#loc3))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc5)
      linalg.yield %cst : f32 loc(#loc5)
    } -> tensor<2x17x19xf32> loc(#loc5)
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<2x17x32xf32>, tensor<2x32x19xf32>) outs(%3 : tensor<2x17x19xf32>) attrs =  {gac.linalg_role = "matmul", gac.lowered_from = "gac.matmul", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "MatMul#1", gac.source_op = "MatMul"} {
    ^bb0(%in: f32 loc("Reshape#0"(#loc2)), %in_0: f32 loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_input_reshape/model.onnx":0:0), %out: f32 loc("MatMul#1"(#loc3))):
      %5 = arith.mulf %in, %in_0 : f32 loc(#loc5)
      %6 = arith.addf %out, %5 : f32 loc(#loc5)
      linalg.yield %6 : f32 loc(#loc5)
    } -> tensor<2x17x19xf32> loc(#loc5)
    return %4 : tensor<2x17x19xf32> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("AShape")

