#loc = loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":0:0)
#loc1 = loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":1:0)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#loc2 = loc("MatMul#0"(#loc1))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<7x15xf32> loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":0:0), %arg1: tensor<15x11xf32> loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":0:0)) -> tensor<7x11xf32> attributes {gac.input_names = ["A", "B"], gac.output_names = ["Y"]} {
    %0 = tensor.empty() : tensor<7x11xf32> loc(#loc2)
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<7x11xf32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("MatMul#0"(#loc1))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)
      linalg.yield %cst : f32 loc(#loc2)
    } -> tensor<7x11xf32> loc(#loc2)
    %2 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<7x15xf32>, tensor<15x11xf32>) outs(%1 : tensor<7x11xf32>) attrs =  {gac.linalg_role = "matmul", gac.lowered_from = "gac.matmul", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "MatMul#0", gac.source_op = "MatMul"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":0:0), %in_0: f32 loc("/home/thomas/gac-triton-batched-matmul-final5-20261022-2230/native_matmul_unaligned_k/model.onnx":0:0), %out: f32 loc("MatMul#0"(#loc1))):
      %3 = arith.mulf %in, %in_0 : f32 loc(#loc2)
      %4 = arith.addf %out, %3 : f32 loc(#loc2)
      linalg.yield %4 : f32 loc(#loc2)
    } -> tensor<7x11xf32> loc(#loc2)
    return %2 : tensor<7x11xf32> loc(#loc2)
  } loc(#loc)
} loc(#loc)

