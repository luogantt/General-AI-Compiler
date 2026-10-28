#loc = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0)
#loc2 = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":2:0)
#map = affine_map<(d0, d1) -> (d0, d1)>
#loc4 = loc("Add#1"(#loc2))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<2x19xf32> loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0), %arg1: tensor<2x19xf32> loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0), %arg2: tensor<2x19xf32> loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0)) -> (tensor<2x19xf32>, tensor<2x19xf32>) attributes {gac.input_names = ["X", "A", "B"], gac.output_names = ["Y", "Z"]} {
    %0 = tensor.empty() : tensor<2x19xf32> loc(#loc3)
    %1 = linalg.softmax {gac.linalg_role = "softmax", gac.lowered_from = "gac.softmax", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "Softmax#0", gac.source_op = "Softmax"} dimension(0) ins(%arg0 : tensor<2x19xf32>) outs(%0 : tensor<2x19xf32>) -> tensor<2x19xf32> loc(#loc3)
    %2 = tensor.empty() : tensor<2x19xf32> loc(#loc4)
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2 : tensor<2x19xf32>, tensor<2x19xf32>) outs(%2 : tensor<2x19xf32>) attrs =  {gac.linalg_role = "elementwise", gac.lowered_from = "gac.add", gac.opset = 18 : i64, gac.output_names = ["Z"], gac.source_domain = "", gac.source_name = "Add#1", gac.source_op = "Add"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0), %in_0: f32 loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":0:0), %out: f32 loc("Add#1"(#loc2))):
      %4 = arith.addf %in, %in_0 : f32 loc(#loc4)
      linalg.yield %4 : f32 loc(#loc4)
    } -> tensor<2x19xf32> loc(#loc4)
    return %1, %3 : tensor<2x19xf32>, tensor<2x19xf32> loc(#loc4)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/mixed_softmax_axis0_sibling_add/model.onnx":1:0)
#loc3 = loc("Softmax#0"(#loc1))

