module attributes {gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/mnt/c/Users/Thomas/Desktop/universal_ai_compile/General-AI-Compiler/build/native-tests/matmul_bias_relu/model.onnx", gac.unsupported_count = 0 : i64, llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-unknown-elf"} {
  llvm.mlir.global private constant @__gac_annotation("vortex.kernel\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @__gac_source("gac.mlir\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %1 = llvm.mlir.addressof @gac_op_0 : !llvm.ptr
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<1 x ptr> 
    llvm.return %2 : !llvm.array<1 x ptr>
  }
  llvm.mlir.global appending @llvm.compiler.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %1 = llvm.mlir.addressof @gac_op_0 : !llvm.ptr
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<1 x ptr> 
    llvm.return %2 : !llvm.array<1 x ptr>
  }
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.undef : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %1 = llvm.mlir.addressof @__gac_annotation : !llvm.ptr
    %2 = llvm.mlir.addressof @__gac_source : !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %6 = llvm.mlir.addressof @gac_op_0 : !llvm.ptr
    %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %8 = llvm.insertvalue %1, %7[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %9 = llvm.insertvalue %2, %8[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %10 = llvm.insertvalue %3, %9[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %11 = llvm.insertvalue %4, %10[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %12 = llvm.insertvalue %11, %0[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> 
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @gac_op_0(%arg0: !llvm.ptr) attributes {passthrough = ["vortex-kernel"], section = ".vx_entry.gac_op_0"} {
    %0 = "vortex.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %c0_i32 = arith.constant 0 : i32
    %1 = llvm.getelementptr %0[%c0_i32] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %2 = llvm.load %1 : !llvm.ptr -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = llvm.inttoptr %3 : i32 to !llvm.ptr
    %c8_i32 = arith.constant 8 : i32
    %5 = llvm.getelementptr %0[%c8_i32] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %6 = llvm.load %5 : !llvm.ptr -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = llvm.inttoptr %7 : i32 to !llvm.ptr
    %c16_i32 = arith.constant 16 : i32
    %9 = llvm.getelementptr %0[%c16_i32] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %10 = llvm.load %9 : !llvm.ptr -> i64
    %11 = arith.trunci %10 : i64 to i32
    %12 = llvm.inttoptr %11 : i32 to !llvm.ptr
    %c24_i32 = arith.constant 24 : i32
    %13 = llvm.getelementptr %0[%c24_i32] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %14 = llvm.load %13 : !llvm.ptr -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = llvm.inttoptr %15 : i32 to !llvm.ptr
    %17 = "vortex.thread_id"() : () -> i32
    %18 = "vortex.block_id"() : () -> i32
    %19 = "vortex.block_dim"() : () -> i32
    %20 = arith.muli %18, %19 : i32
    %21 = arith.addi %20, %17 : i32
    %c8_i32_0 = arith.constant 8 : i32
    %22 = arith.cmpi slt, %21, %c8_i32_0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %23 = arith.cmpi sge, %21, %c0_i32_1 : i32
    %24 = arith.andi %23, %22 : i1
    scf.if %24 {
      %c2_i32 = arith.constant 2 : i32
      %c4_i32 = arith.constant 4 : i32
      %25 = arith.divsi %21, %c4_i32 : i32
      %26 = arith.remsi %25, %c2_i32 : i32
      %c4_i32_2 = arith.constant 4 : i32
      %c1_i32 = arith.constant 1 : i32
      %27 = arith.divsi %21, %c1_i32 : i32
      %28 = arith.remsi %27, %c4_i32_2 : i32
      %cst = arith.constant 0.000000e+00 : f32
      %c0_i32_3 = arith.constant 0 : i32
      %c3_i32 = arith.constant 3 : i32
      %c1_i32_4 = arith.constant 1 : i32
      %29 = scf.for %arg1 = %c0_i32_3 to %c3_i32 step %c1_i32_4 iter_args(%arg2 = %cst) -> (f32)  : i32 {
        %c0_i32_13 = arith.constant 0 : i32
        %c3_i32_14 = arith.constant 3 : i32
        %42 = arith.muli %c0_i32_13, %c3_i32_14 : i32
        %43 = arith.addi %42, %arg1 : i32
        %c4_i32_15 = arith.constant 4 : i32
        %44 = arith.muli %43, %c4_i32_15 : i32
        %45 = arith.addi %44, %28 : i32
        %46 = llvm.getelementptr %8[%45] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %47 = llvm.load %46 : !llvm.ptr -> f32
        %c0_i32_16 = arith.constant 0 : i32
        %c2_i32_17 = arith.constant 2 : i32
        %48 = arith.muli %c0_i32_16, %c2_i32_17 : i32
        %49 = arith.addi %48, %26 : i32
        %c3_i32_18 = arith.constant 3 : i32
        %50 = arith.muli %49, %c3_i32_18 : i32
        %51 = arith.addi %50, %arg1 : i32
        %52 = llvm.getelementptr %4[%51] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %53 = llvm.load %52 : !llvm.ptr -> f32
        %54 = arith.mulf %53, %47 : f32
        %55 = arith.addf %arg2, %54 : f32
        scf.yield %55 : f32
      }
      %c2_i32_5 = arith.constant 2 : i32
      %c4_i32_6 = arith.constant 4 : i32
      %30 = arith.divsi %21, %c4_i32_6 : i32
      %31 = arith.remsi %30, %c2_i32_5 : i32
      %c4_i32_7 = arith.constant 4 : i32
      %c1_i32_8 = arith.constant 1 : i32
      %32 = arith.divsi %21, %c1_i32_8 : i32
      %33 = arith.remsi %32, %c4_i32_7 : i32
      %c0_i32_9 = arith.constant 0 : i32
      %c4_i32_10 = arith.constant 4 : i32
      %34 = arith.muli %c0_i32_9, %c4_i32_10 : i32
      %35 = arith.addi %34, %33 : i32
      %36 = llvm.getelementptr %12[%35] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      %37 = llvm.load %36 : !llvm.ptr -> f32
      %38 = arith.addf %29, %37 : f32
      %cst_11 = arith.constant 0.000000e+00 : f32
      %cst_12 = arith.constant 0.000000e+00 : f32
      %39 = arith.cmpf olt, %38, %cst_12 : f32
      %40 = arith.select %39, %cst_11, %38 : f32
      %41 = llvm.getelementptr %16[%21] : (!llvm.ptr, i32) -> !llvm.ptr, f32
      llvm.store %40, %41 : f32, !llvm.ptr
    }
    llvm.return
  }
}

