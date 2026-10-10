// Hand-written MLIR/LLVM NVPTX toolchain smoke fixture.
//
// This module is intentionally independent of GAC graph lowering.  It checks
// that an upstream MLIR build with the NVPTX target can lower GPU-dialect IR
// into an embedded PTX assembly object for NVIDIA SM 8.6.
module attributes {gpu.container_module} {
  gpu.module @gac_kernels {
    gpu.func @add_relu_broadcast(
        %x: memref<64x256xf32, #gpu.address_space<global>>,
        %bias: memref<256xf32, #gpu.address_space<global>>,
        %out: memref<64x256xf32, #gpu.address_space<global>>)
        kernel attributes {
          known_block_size = array<i32: 256, 1, 1>
        } {
      %block = gpu.block_id x
      %block_size = gpu.block_dim x
      %thread = gpu.thread_id x
      %base = arith.muli %block, %block_size : index
      %linear = arith.addi %base, %thread : index

      %element_count = arith.constant 16384 : index
      %in_bounds = arith.cmpi ult, %linear, %element_count : index
      scf.if %in_bounds {
        %width = arith.constant 256 : index
        %row = arith.divui %linear, %width : index
        %column = arith.remui %linear, %width : index

        %input = memref.load %x[%row, %column]
          : memref<64x256xf32, #gpu.address_space<global>>
        %bias_value = memref.load %bias[%column]
          : memref<256xf32, #gpu.address_space<global>>
        %sum = arith.addf %input, %bias_value : f32
        %zero = arith.constant 0.0 : f32
        %relu = arith.maximumf %sum, %zero : f32
        memref.store %relu, %out[%row, %column]
          : memref<64x256xf32, #gpu.address_space<global>>
      }
      gpu.return
    }
  }
}
