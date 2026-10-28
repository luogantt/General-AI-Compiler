#loc = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/triton_softmax_rank3_bev_width_256/model.onnx":0:0)
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-rowsoftmax-final5-20261022/triton_softmax_rank3_bev_width_256/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<1x2x256xf32> loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/triton_softmax_rank3_bev_width_256/model.onnx":0:0)) -> tensor<1x2x256xf32> attributes {gac.input_names = ["A"], gac.output_names = ["Y"]} {
    %0 = tensor.empty() : tensor<1x2x256xf32> loc(#loc2)
    %1 = linalg.softmax {gac.linalg_role = "softmax", gac.lowered_from = "gac.softmax", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "Softmax#0", gac.source_op = "Softmax"} dimension(2) ins(%arg0 : tensor<1x2x256xf32>) outs(%0 : tensor<1x2x256xf32>) -> tensor<1x2x256xf32> loc(#loc2)
    return %1 : tensor<1x2x256xf32> loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/triton_softmax_rank3_bev_width_256/model.onnx":1:0)
#loc2 = loc("Softmax#0"(#loc1))

