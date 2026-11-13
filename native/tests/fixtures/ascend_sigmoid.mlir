module {
  func.func @main(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> attributes {gac.input_names = ["X"], gac.output_names = ["Y"]} {
    %0 = "gac.sigmoid"(%arg0) {gac.output_names = ["Y"], gac.source_name = "Sigmoid#0"} : (tensor<1x16xf32>) -> tensor<1x16xf32>
    return %0 : tensor<1x16xf32>
  }
}
