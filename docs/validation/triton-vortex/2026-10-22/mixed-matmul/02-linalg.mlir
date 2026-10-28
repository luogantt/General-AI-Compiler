#loc = loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0)
#loc1 = loc("W")
#loc2 = loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":1:0)
#loc3 = loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":2:0)
#loc4 = loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":3:0)
#loc5 = loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":4:0)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#loc6 = loc("Add#0"(#loc2))
#loc7 = loc("MatMul#1"(#loc3))
#loc8 = loc("Mul#2"(#loc4))
#loc9 = loc("Relu#3"(#loc5))
module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx", gac.unsupported_count = 0 : i64} {
  func.func @main(%arg0: tensor<5x7xf32> loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0), %arg1: tensor<5x7xf32> loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0), %arg2: tensor<5x9xf32> loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0)) -> (tensor<5x9xf32>, tensor<5x9xf32>) attributes {gac.input_names = ["A", "B", "R"], gac.output_names = ["M", "Y"]} {
    %0 = "gac.constant"() {value = dense<[[-0.404498518, 0.980502963, 0.786487519, -0.24218227, 0.567006946, -0.771218121, 0.353679776, 0.287669957, -0.149192944], [-0.799691081, 0.704636275, 0.629756927, 0.209801495, 8.405850e-01, 0.825490653, -0.130218595, -0.0637045875, 0.727607667], [0.755777418, -0.855200827, -0.177072152, 0.121165872, 0.612740814, 0.253560543, -0.157253042, 0.383763701, 0.27935344], [0.816882133, -0.461465389, -0.783412933, 0.873150229, 0.734314262, 0.917481243, -0.210362241, 0.207989678, 0.951431214], [-0.957109272, -0.518442094, -0.12847589, -0.536143303, 0.508776605, 0.785758554, 0.369369656, -0.969564676, 0.459076732], [0.261030316, -0.702131271, 0.97437632, -0.363266617, 0.940688073, 0.372276604, -0.294697464, -0.765571356, 0.94944179], [0.757992506, -0.0734351799, 0.812159121, -0.413905501, -0.255053282, 0.974800467, -0.523445964, 0.235247597, 0.214676514]]> : tensor<7x9xf32>} : () -> tensor<7x9xf32> loc(#loc1)
    %1 = tensor.empty() : tensor<5x7xf32> loc(#loc6)
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<5x7xf32>, tensor<5x7xf32>) outs(%1 : tensor<5x7xf32>) attrs =  {gac.linalg_role = "elementwise", gac.lowered_from = "gac.add", gac.opset = 18 : i64, gac.source_domain = "", gac.source_op = "Add"} {
    ^bb0(%in: f32 loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0), %in_0: f32 loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0), %out: f32 loc("Add#0"(#loc2))):
      %10 = arith.addf %in, %in_0 : f32 loc(#loc6)
      linalg.yield %10 : f32 loc(#loc6)
    } -> tensor<5x7xf32> loc(#loc6)
    %3 = tensor.empty() : tensor<5x9xf32> loc(#loc7)
    %4 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%3 : tensor<5x9xf32>) attrs =  {gac.linalg_role = "initialize"} {
    ^bb0(%out: f32 loc("MatMul#1"(#loc3))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc7)
      linalg.yield %cst : f32 loc(#loc7)
    } -> tensor<5x9xf32> loc(#loc7)
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2, %0 : tensor<5x7xf32>, tensor<7x9xf32>) outs(%4 : tensor<5x9xf32>) attrs =  {gac.linalg_role = "matmul", gac.lowered_from = "gac.matmul", gac.opset = 18 : i64, gac.output_names = ["M"], gac.source_domain = "", gac.source_name = "MatMul#1", gac.source_op = "MatMul"} {
    ^bb0(%in: f32 loc("Add#0"(#loc2)), %in_0: f32 loc("W"), %out: f32 loc("MatMul#1"(#loc3))):
      %10 = arith.mulf %in, %in_0 : f32 loc(#loc7)
      %11 = arith.addf %out, %10 : f32 loc(#loc7)
      linalg.yield %11 : f32 loc(#loc7)
    } -> tensor<5x9xf32> loc(#loc7)
    %6 = tensor.empty() : tensor<5x9xf32> loc(#loc8)
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %arg2 : tensor<5x9xf32>, tensor<5x9xf32>) outs(%6 : tensor<5x9xf32>) attrs =  {gac.linalg_role = "elementwise", gac.lowered_from = "gac.mul", gac.opset = 18 : i64, gac.source_domain = "", gac.source_op = "Mul"} {
    ^bb0(%in: f32 loc("MatMul#1"(#loc3)), %in_0: f32 loc("/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx":0:0), %out: f32 loc("Mul#2"(#loc4))):
      %10 = arith.mulf %in, %in_0 : f32 loc(#loc8)
      linalg.yield %10 : f32 loc(#loc8)
    } -> tensor<5x9xf32> loc(#loc8)
    %8 = tensor.empty() : tensor<5x9xf32> loc(#loc9)
    %9 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<5x9xf32>) outs(%8 : tensor<5x9xf32>) attrs =  {gac.linalg_role = "elementwise", gac.lowered_from = "gac.relu", gac.opset = 18 : i64, gac.output_names = ["Y"], gac.source_domain = "", gac.source_name = "Relu#3", gac.source_op = "Relu"} {
    ^bb0(%in: f32 loc("Mul#2"(#loc4)), %out: f32 loc("Relu#3"(#loc5))):
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc9)
      %10 = arith.cmpf olt, %in, %cst : f32 loc(#loc9)
      %11 = arith.select %10, %cst, %in : f32 loc(#loc9)
      linalg.yield %11 : f32 loc(#loc9)
    } -> tensor<5x9xf32> loc(#loc9)
    return %5, %9 : tensor<5x9xf32>, tensor<5x9xf32> loc(#loc9)
  } loc(#loc)
} loc(#loc)

