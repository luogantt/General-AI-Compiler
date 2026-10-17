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
    %0 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %3 = llvm.load %2 : !llvm.ptr -> i64
    %4 = llvm.trunc %3 : i64 to i32
    %5 = llvm.inttoptr %4 : i32 to !llvm.ptr
    %6 = llvm.mlir.constant(8 : i32) : i32
    %7 = llvm.getelementptr %0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %8 = llvm.load %7 : !llvm.ptr -> i64
    %9 = llvm.trunc %8 : i64 to i32
    %10 = llvm.inttoptr %9 : i32 to !llvm.ptr
    %11 = llvm.mlir.constant(16 : i32) : i32
    %12 = llvm.getelementptr %0[%11] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %13 = llvm.load %12 : !llvm.ptr -> i64
    %14 = llvm.trunc %13 : i64 to i32
    %15 = llvm.inttoptr %14 : i32 to !llvm.ptr
    %16 = llvm.mlir.constant(24 : i32) : i32
    %17 = llvm.getelementptr %0[%16] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %18 = llvm.load %17 : !llvm.ptr -> i64
    %19 = llvm.trunc %18 : i64 to i32
    %20 = llvm.inttoptr %19 : i32 to !llvm.ptr
    %21 = llvm.mlir.constant(3283 : i32) : i32
    %22 = llvm.inline_asm "csrr $0, $1", "=r,i" %21 : (i32) -> i32
    %23 = llvm.mlir.constant(3286 : i32) : i32
    %24 = llvm.inline_asm "csrr $0, $1", "=r,i" %23 : (i32) -> i32
    %25 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%24) : (i32) -> i32
    %26 = llvm.mlir.constant(3289 : i32) : i32
    %27 = llvm.inline_asm "csrr $0, $1", "=r,i" %26 : (i32) -> i32
    %28 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%27) : (i32) -> i32
    %29 = llvm.mul %25, %28 : i32
    %30 = llvm.add %29, %22 : i32
    %31 = llvm.mlir.constant(8 : i32) : i32
    %32 = llvm.icmp "slt" %30, %31 : i32
    %33 = llvm.mlir.constant(0 : i32) : i32
    %34 = llvm.icmp "sge" %30, %33 : i32
    %35 = llvm.and %34, %32 : i1
    llvm.cond_br %35, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %36 = llvm.mlir.constant(2 : i32) : i32
    %37 = llvm.mlir.constant(4 : i32) : i32
    %38 = llvm.sdiv %30, %37 : i32
    %39 = llvm.srem %38, %36 : i32
    %40 = llvm.mlir.constant(4 : i32) : i32
    %41 = llvm.mlir.constant(1 : i32) : i32
    %42 = llvm.srem %30, %40 : i32
    %43 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %44 = llvm.mlir.constant(0 : i32) : i32
    %45 = llvm.mlir.constant(3 : i32) : i32
    %46 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb2(%44, %43 : i32, f32)
  ^bb2(%47: i32, %48: f32):  // 2 preds: ^bb1, ^bb3
    %49 = llvm.icmp "slt" %47, %45 : i32
    llvm.cond_br %49, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %50 = llvm.mlir.constant(0 : i32) : i32
    %51 = llvm.mlir.constant(3 : i32) : i32
    %52 = llvm.mlir.constant(0 : i32) : i32
    %53 = llvm.add %52, %47 : i32
    %54 = llvm.mlir.constant(4 : i32) : i32
    %55 = llvm.mul %53, %54 : i32
    %56 = llvm.add %55, %42 : i32
    %57 = llvm.getelementptr %10[%56] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %58 = llvm.load %57 : !llvm.ptr -> f32
    %59 = llvm.mlir.constant(0 : i32) : i32
    %60 = llvm.mlir.constant(2 : i32) : i32
    %61 = llvm.mlir.constant(0 : i32) : i32
    %62 = llvm.add %61, %39 : i32
    %63 = llvm.mlir.constant(3 : i32) : i32
    %64 = llvm.mul %62, %63 : i32
    %65 = llvm.add %64, %47 : i32
    %66 = llvm.getelementptr %5[%65] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %67 = llvm.load %66 : !llvm.ptr -> f32
    %68 = llvm.fmul %67, %58 : f32
    %69 = llvm.fadd %48, %68 : f32
    %70 = llvm.add %47, %46 : i32
    llvm.br ^bb2(%70, %69 : i32, f32)
  ^bb4:  // pred: ^bb2
    %71 = llvm.mlir.constant(2 : i32) : i32
    %72 = llvm.mlir.constant(4 : i32) : i32
    %73 = llvm.sdiv %30, %72 : i32
    %74 = llvm.srem %73, %71 : i32
    %75 = llvm.mlir.constant(4 : i32) : i32
    %76 = llvm.mlir.constant(1 : i32) : i32
    %77 = llvm.srem %30, %75 : i32
    %78 = llvm.mlir.constant(0 : i32) : i32
    %79 = llvm.mlir.constant(4 : i32) : i32
    %80 = llvm.mlir.constant(0 : i32) : i32
    %81 = llvm.add %80, %77 : i32
    %82 = llvm.getelementptr %15[%81] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %83 = llvm.load %82 : !llvm.ptr -> f32
    %84 = llvm.fadd %48, %83 : f32
    %85 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %86 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %87 = llvm.fcmp "olt" %84, %86 : f32
    %88 = llvm.select %87, %85, %84 : i1, f32
    %89 = llvm.getelementptr %20[%30] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %88, %89 : f32, !llvm.ptr
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
}

