#loc = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/native_softmax_input_reshape/model.onnx":0:0)
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-rowsoftmax-final5-20261022/native_softmax_input_reshape/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<57xf32> loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/native_softmax_input_reshape/model.onnx":0:0)) -> tensor<3x19xf32> attributes {gac.input_names = ["XRaw"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {value = dense<[3, 19]> : tensor<2xi64>} : () -> tensor<2xi64> loc(#loc1)
    %1 = "gac.reshape"(%arg0, %0) {gac.input_positions = array<i64: 0, 1>, gac.opset = 18 : i64, gac.source_domain = "", gac.source_op = "Reshape"} : (tensor<57xf32>, tensor<2xi64>) -> tensor<3x19xf32> loc(#loc4)
    %2 = tensor.empty() : tensor<3x19xf32> loc(#loc5)
    %3 = linalg.softmax {gac.linalg_role = "softmax", gac.lowered_from = "gac.softmax", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "Softmax#1", gac.source_op = "Softmax"} dimension(1) ins(%1 : tensor<3x19xf32>) outs(%2 : tensor<3x19xf32>) -> tensor<3x19xf32> loc(#loc5)
    return %3 : tensor<3x19xf32> loc(#loc5)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("Shape")
#loc2 = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/native_softmax_input_reshape/model.onnx":1:0)
#loc3 = loc("/home/thomas/gac-triton-rowsoftmax-final5-20261022/native_softmax_input_reshape/model.onnx":2:0)
#loc4 = loc("Reshape#0"(#loc2))
#loc5 = loc("Softmax#1"(#loc3))

