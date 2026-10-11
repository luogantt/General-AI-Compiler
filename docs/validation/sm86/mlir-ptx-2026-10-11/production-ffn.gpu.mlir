// GAC mlir-ptx-sm86: production BEVFormer FFN device module.
module attributes {gpu.container_module} {
  gpu.module @gac_bevformer_ffn_sm86 {
    gpu.func @gac_mlir_ptx_fc1_bias_relu_f32(
        %query: memref<2500x256xf32, #gpu.address_space<global>>,
        %weight: memref<256x512xf32, #gpu.address_space<global>>,
        %bias: memref<512xf32, #gpu.address_space<global>>,
        %output: memref<2500x512xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index
      %count = arith.constant 1280000 : index
      %in_bounds = arith.cmpi ult, %linear, %count : index
      scf.if %in_bounds {
        %width = arith.constant 512 : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index
        %lower = arith.constant 0 : index
        %upper = arith.constant 256 : index
        %step = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %k = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {
          %lhs = memref.load %query[%row, %k]
            : memref<2500x256xf32, #gpu.address_space<global>>
          %rhs = memref.load %weight[%k, %column]
            : memref<256x512xf32, #gpu.address_space<global>>
          %product = arith.mulf %lhs, %rhs : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }
        %bias_value = memref.load %bias[%column]
          : memref<512xf32, #gpu.address_space<global>>
        %biased = arith.addf %sum, %bias_value : f32
        %relu = arith.maximumf %biased, %zero : f32
        memref.store %relu, %output[%row, %column]
          : memref<2500x512xf32, #gpu.address_space<global>>
      }
      gpu.return
    }

    gpu.func @gac_mlir_ptx_fc2_f32(
        %input: memref<2500x512xf32, #gpu.address_space<global>>,
        %weight: memref<512x256xf32, #gpu.address_space<global>>,
        %output: memref<2500x256xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index
      %count = arith.constant 640000 : index
      %in_bounds = arith.cmpi ult, %linear, %count : index
      scf.if %in_bounds {
        %width = arith.constant 256 : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index
        %lower = arith.constant 0 : index
        %upper = arith.constant 512 : index
        %step = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %k = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {
          %lhs = memref.load %input[%row, %k]
            : memref<2500x512xf32, #gpu.address_space<global>>
          %rhs = memref.load %weight[%k, %column]
            : memref<512x256xf32, #gpu.address_space<global>>
          %product = arith.mulf %lhs, %rhs : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }
        memref.store %sum, %output[%row, %column]
          : memref<2500x256xf32, #gpu.address_space<global>>
      }
      gpu.return
    }

    gpu.func @gac_mlir_ptx_bias_residual_layer_norm_f32(
        %projected: memref<2500x256xf32, #gpu.address_space<global>>,
        %bias: memref<256xf32, #gpu.address_space<global>>,
        %residual: memref<2500x256xf32, #gpu.address_space<global>>,
        %gamma: memref<256xf32, #gpu.address_space<global>>,
        %beta: memref<256xf32, #gpu.address_space<global>>,
        %output: memref<2500x256xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %row = gpu.block_id x
      %column = gpu.thread_id x
      %projected_value = memref.load %projected[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      %bias_value = memref.load %bias[%column]
        : memref<256xf32, #gpu.address_space<global>>
      %residual_value = memref.load %residual[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      %biased = arith.addf %projected_value, %bias_value : f32
      %value = arith.addf %biased, %residual_value : f32
      %sum = gpu.all_reduce add %value uniform {} : (f32) -> (f32)
      %width = arith.constant 256.0 : f32
      %mean = arith.divf %sum, %width : f32
      %centered = arith.subf %value, %mean : f32
      %square = arith.mulf %centered, %centered : f32
      %square_sum = gpu.all_reduce add %square uniform {} : (f32) -> (f32)
      %variance = arith.divf %square_sum, %width : f32
      %epsilon = arith.constant 9.99999975e-06 : f32
      %adjusted = arith.addf %variance, %epsilon : f32
      %inv_stddev = nvvm.rsqrt %adjusted : f32
      %normalized = arith.mulf %centered, %inv_stddev : f32
      %scale = memref.load %gamma[%column]
        : memref<256xf32, #gpu.address_space<global>>
      %shift = memref.load %beta[%column]
        : memref<256xf32, #gpu.address_space<global>>
      %scaled = arith.mulf %normalized, %scale : f32
      %result = arith.addf %scaled, %shift : f32
      memref.store %result, %output[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      gpu.return
    }
  }
}
