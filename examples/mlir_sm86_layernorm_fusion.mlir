// Standalone MLIR -> NVVM/PTX reference for BEVFormer FFN post-normalization.
//
// This module is intentionally independent of GAC graph lowering and runtime.
// It demonstrates and measures a real GPU-dialect LayerNorm implementation;
// MlirCudaSession currently executes an equivalent NVRTC CUDA C fusion kernel.
// Launch contracts are recorded on each kernel below.
module attributes {gpu.container_module} {
  gpu.module @gac_mlir_layernorm {
    // Serial baseline matching the first device-level MLIR benchmark. Each
    // thread handles a complete row. Launch grid=(10,1,1), block=(256,1,1).
    gpu.func @layer_norm_serial(
        %input: memref<2500x256xf32, #gpu.address_space<global>>,
        %gamma: memref<256xf32, #gpu.address_space<global>>,
        %beta: memref<256xf32, #gpu.address_space<global>>,
        %output: memref<2500x256xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %row = arith.addi %base, %thread : index
      %row_count = arith.constant 2500 : index
      %in_bounds = arith.cmpi ult, %row, %row_count : index
      scf.if %in_bounds {
        %lower = arith.constant 0 : index
        %upper = arith.constant 256 : index
        %step = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %column = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {
          %value = memref.load %input[%row, %column]
            : memref<2500x256xf32, #gpu.address_space<global>>
          %next = arith.addf %acc, %value : f32
          scf.yield %next : f32
        }
        %inv_width = arith.constant 0.00390625 : f32
        %mean = arith.mulf %sum, %inv_width : f32
        %square_sum = scf.for %column = %lower to %upper step %step
            iter_args(%acc = %zero) -> (f32) {
          %value = memref.load %input[%row, %column]
            : memref<2500x256xf32, #gpu.address_space<global>>
          %centered = arith.subf %value, %mean : f32
          %square = arith.mulf %centered, %centered : f32
          %next = arith.addf %acc, %square : f32
          scf.yield %next : f32
        }
        %variance = arith.mulf %square_sum, %inv_width : f32
        %epsilon = arith.constant 0.00001 : f32
        %adjusted = arith.addf %variance, %epsilon : f32
        %inv_stddev = nvvm.rsqrt %adjusted : f32
        scf.for %column = %lower to %upper step %step {
          %value = memref.load %input[%row, %column]
            : memref<2500x256xf32, #gpu.address_space<global>>
          %scale = memref.load %gamma[%column]
            : memref<256xf32, #gpu.address_space<global>>
          %shift = memref.load %beta[%column]
            : memref<256xf32, #gpu.address_space<global>>
          %centered = arith.subf %value, %mean : f32
          %normalized = arith.mulf %centered, %inv_stddev : f32
          %scaled = arith.mulf %normalized, %scale : f32
          %result = arith.addf %scaled, %shift : f32
          memref.store %result, %output[%row, %column]
            : memref<2500x256xf32, #gpu.address_space<global>>
        }
      }
      gpu.return
    }

    // One block owns one row and one thread owns one column. gpu.all_reduce
    // lowers to warp shuffles plus a block reduction on NVPTX. Launch
    // grid=(2500,1,1), block=(256,1,1).
    gpu.func @layer_norm_block(
        %input: memref<2500x256xf32, #gpu.address_space<global>>,
        %gamma: memref<256xf32, #gpu.address_space<global>>,
        %beta: memref<256xf32, #gpu.address_space<global>>,
        %output: memref<2500x256xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %row = gpu.block_id x
      %column = gpu.thread_id x
      %value = memref.load %input[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      %sum = gpu.all_reduce add %value uniform {} : (f32) -> (f32)
      %width = arith.constant 256.0 : f32
      %mean = arith.divf %sum, %width : f32
      %centered = arith.subf %value, %mean : f32
      %square = arith.mulf %centered, %centered : f32
      %square_sum = gpu.all_reduce add %square uniform {} : (f32) -> (f32)
      %variance = arith.divf %square_sum, %width : f32
      %epsilon = arith.constant 0.00001 : f32
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

    // Fuses the FFN second projection epilogue with normalization. Launch
    // grid=(2500,1,1), block=(256,1,1):
    // output = LN(matmul_output + bias + residual, gamma, beta).
    // cuBLAS owns the GEMM. This kernel replaces Bias Add, Residual Add and LN.
    gpu.func @residual_bias_layer_norm_block(
        %matmul_output: memref<2500x256xf32, #gpu.address_space<global>>,
        %residual: memref<2500x256xf32, #gpu.address_space<global>>,
        %bias: memref<256xf32, #gpu.address_space<global>>,
        %gamma: memref<256xf32, #gpu.address_space<global>>,
        %beta: memref<256xf32, #gpu.address_space<global>>,
        %output: memref<2500x256xf32, #gpu.address_space<global>>)
        kernel attributes {known_block_size = array<i32: 256, 1, 1>} {
      %row = gpu.block_id x
      %column = gpu.thread_id x
      %projected = memref.load %matmul_output[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      %skip = memref.load %residual[%row, %column]
        : memref<2500x256xf32, #gpu.address_space<global>>
      %bias_value = memref.load %bias[%column]
        : memref<256xf32, #gpu.address_space<global>>
      %biased = arith.addf %projected, %bias_value : f32
      %value = arith.addf %biased, %skip : f32
      %sum = gpu.all_reduce add %value uniform {} : (f32) -> (f32)
      %width = arith.constant 256.0 : f32
      %mean = arith.divf %sum, %width : f32
      %centered = arith.subf %value, %mean : f32
      %square = arith.mulf %centered, %centered : f32
      %square_sum = gpu.all_reduce add %square uniform {} : (f32) -> (f32)
      %variance = arith.divf %square_sum, %width : f32
      %epsilon = arith.constant 0.00001 : f32
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
