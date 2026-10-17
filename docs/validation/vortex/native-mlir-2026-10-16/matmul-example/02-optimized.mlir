module attributes {gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler/build/native-tests/matmul_bias_relu/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x4xf32> attributes {gac.input_names = ["X"], gac.output_names = ["Y"]} {
    %0 = "gac.constant"() {value = dense<[[-0.527820587, 0.665015637, 0.841673076, 0.362871081], [0.356437922, 0.269289017, 0.190943137, -0.458101749], [-0.614149809, -1.65041721, -0.296622962, -2.547710e+00]]> : tensor<3x4xf32>} : () -> tensor<3x4xf32>
    %1 = "gac.constant"() {value = dense<[-0.0110531338, -0.457764357, -0.0702236667, 0.357426912]> : tensor<4xf32>} : () -> tensor<4xf32>
    %2 = "gac.matmul_bias_relu"(%arg0, %0, %1) {gac.fusion = "matmul_bias_relu", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "activation", gac.source_op = "Add"} : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %2 : tensor<2x4xf32>
  }
}

