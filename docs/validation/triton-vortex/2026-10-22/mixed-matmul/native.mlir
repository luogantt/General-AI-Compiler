module attributes {gac.constant_fold_profile = "default", gac.inspection_only = false, gac.onnx_ir_version = 9 : i64, gac.source_format = "onnx", gac.source_path = "/home/thomas/gac-triton-integration-final-20261022/mixed_matmul/model.onnx", gac.unsupported_count = 0 : i64, llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-unknown-elf"} {
  llvm.mlir.global private constant @__gac_annotation("vortex.kernel\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @__gac_source("gac.mlir\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %1 = llvm.mlir.addressof @gac_op_1 : !llvm.ptr
    %2 = llvm.insertvalue %1, %0[0] : !llvm.array<1 x ptr> 
    llvm.return %2 : !llvm.array<1 x ptr>
  }
  llvm.mlir.global appending @llvm.compiler.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
    %0 = llvm.mlir.undef : !llvm.array<1 x ptr>
    %1 = llvm.mlir.addressof @gac_op_1 : !llvm.ptr
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
    %6 = llvm.mlir.addressof @gac_op_1 : !llvm.ptr
    %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %8 = llvm.insertvalue %1, %7[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %9 = llvm.insertvalue %2, %8[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %10 = llvm.insertvalue %3, %9[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %11 = llvm.insertvalue %4, %10[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %12 = llvm.insertvalue %11, %0[0] : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>> 
    llvm.return %12 : !llvm.array<1 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.func @gac_op_1(%arg0: !llvm.ptr) attributes {passthrough = ["vortex-kernel"], section = ".vx_entry.gac_op_1"} {
    %0 = llvm.mlir.constant(7 : index) : i32
    %1 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(1 : index) : i32
    %3 = llvm.mlir.constant(0 : index) : i32
    %4 = llvm.mlir.constant(5 : index) : i32
    %5 = llvm.mlir.constant(9 : index) : i32
    %6 = llvm.mlir.constant(9 : index) : i32
    %7 = llvm.mlir.constant(1 : index) : i32
    %8 = llvm.mlir.constant(7 : index) : i32
    %9 = llvm.mlir.constant(5 : index) : i32
    %10 = llvm.mlir.constant(0 : index) : i32
    %11 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)>
    %12 = llvm.mlir.constant(45 : i32) : i32
    %13 = llvm.mlir.constant(3289 : i32) : i32
    %14 = llvm.mlir.constant(3286 : i32) : i32
    %15 = llvm.mlir.constant(3283 : i32) : i32
    %16 = llvm.mlir.constant(0 : i32) : i32
    %17 = llvm.mlir.constant(1 : index) : i32
    %18 = llvm.alloca %17 x f32 : (i32) -> !llvm.ptr
    %19 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i32)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i32)> 
    %22 = llvm.mlir.constant(0 : index) : i32
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i32)> 
    %24 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.trunc %25 : i64 to i32
    %27 = llvm.inttoptr %26 : i32 to !llvm.ptr
    %28 = llvm.getelementptr %24[8] : (!llvm.ptr) -> !llvm.ptr, i8
    %29 = llvm.load %28 : !llvm.ptr -> i64
    %30 = llvm.trunc %29 : i64 to i32
    %31 = llvm.inttoptr %30 : i32 to !llvm.ptr
    %32 = llvm.getelementptr %24[16] : (!llvm.ptr) -> !llvm.ptr, i8
    %33 = llvm.load %32 : !llvm.ptr -> i64
    %34 = llvm.trunc %33 : i64 to i32
    %35 = llvm.inttoptr %34 : i32 to !llvm.ptr
    %36 = llvm.inline_asm "csrr $0, $1", "=r,i" %15 : (i32) -> i32
    %37 = llvm.inline_asm "csrr $0, $1", "=r,i" %14 : (i32) -> i32
    %38 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%37) : (i32) -> i32
    %39 = llvm.inline_asm "csrr $0, $1", "=r,i" %13 : (i32) -> i32
    %40 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%39) : (i32) -> i32
    %41 = llvm.mul %38, %40 : i32
    %42 = llvm.add %41, %36 : i32
    %43 = llvm.icmp "slt" %42, %12 : i32
    %44 = llvm.icmp "sge" %42, %16 : i32
    %45 = llvm.and %44, %43 : i1
    llvm.cond_br %45, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    %46 = llvm.insertvalue %27, %11[0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %47 = llvm.insertvalue %27, %46[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %48 = llvm.insertvalue %10, %47[2] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %49 = llvm.insertvalue %9, %48[3, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %50 = llvm.insertvalue %8, %49[4, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %51 = llvm.insertvalue %8, %50[3, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %52 = llvm.insertvalue %7, %51[4, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %53 = llvm.insertvalue %31, %11[0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %54 = llvm.insertvalue %31, %53[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %55 = llvm.insertvalue %10, %54[2] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %56 = llvm.insertvalue %8, %55[3, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %57 = llvm.insertvalue %6, %56[4, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %58 = llvm.insertvalue %6, %57[3, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %59 = llvm.insertvalue %7, %58[4, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %60 = llvm.insertvalue %35, %11[0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %61 = llvm.insertvalue %35, %60[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %62 = llvm.insertvalue %10, %61[2] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %63 = llvm.insertvalue %9, %62[3, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %64 = llvm.insertvalue %6, %63[4, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %65 = llvm.insertvalue %6, %64[3, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %66 = llvm.insertvalue %7, %65[4, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %67 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)>
    %68 = llvm.extractvalue %23[0] : !llvm.struct<(ptr, ptr, i32)> 
    %69 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i32)> 
    %70 = llvm.insertvalue %68, %67[0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %71 = llvm.insertvalue %69, %70[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %72 = llvm.mlir.constant(0 : index) : i32
    %73 = llvm.insertvalue %72, %71[2] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %74 = llvm.mlir.constant(5 : index) : i32
    %75 = llvm.insertvalue %74, %73[3, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %76 = llvm.mlir.constant(0 : index) : i32
    %77 = llvm.insertvalue %76, %75[4, 0] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %78 = llvm.mlir.constant(9 : index) : i32
    %79 = llvm.insertvalue %78, %77[3, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %80 = llvm.mlir.constant(0 : index) : i32
    %81 = llvm.insertvalue %80, %79[4, 1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %82 = llvm.urem %42, %5 : i32
    %83 = llvm.udiv %42, %5 : i32
    %84 = llvm.urem %83, %4 : i32
    %85 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %86 = llvm.mlir.constant(0 : index) : i32
    %87 = llvm.mul %84, %86 : i32
    %88 = llvm.mlir.constant(0 : index) : i32
    %89 = llvm.mul %82, %88 : i32
    %90 = llvm.add %87, %89 : i32
    %91 = llvm.getelementptr %85[%90] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %1, %91 : f32, !llvm.ptr
    %92 = llvm.urem %42, %5 : i32
    %93 = llvm.udiv %42, %5 : i32
    %94 = llvm.urem %93, %4 : i32
    %95 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %96 = llvm.mlir.constant(0 : index) : i32
    %97 = llvm.mul %94, %96 : i32
    %98 = llvm.mlir.constant(0 : index) : i32
    %99 = llvm.mul %92, %98 : i32
    %100 = llvm.add %97, %99 : i32
    %101 = llvm.getelementptr %95[%100] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %102 = llvm.load %101 : !llvm.ptr -> f32
    %103 = llvm.extractvalue %66[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %104 = llvm.mlir.constant(9 : index) : i32
    %105 = llvm.mul %94, %104 : i32
    %106 = llvm.add %105, %92 : i32
    %107 = llvm.getelementptr %103[%106] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %102, %107 : f32, !llvm.ptr
    llvm.br ^bb2(%3 : i32)
  ^bb2(%108: i32):  // 2 preds: ^bb1, ^bb3
    %109 = llvm.icmp "slt" %108, %0 : i32
    llvm.cond_br %109, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %110 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %111 = llvm.mlir.constant(7 : index) : i32
    %112 = llvm.mul %94, %111 : i32
    %113 = llvm.add %112, %108 : i32
    %114 = llvm.getelementptr %110[%113] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %115 = llvm.load %114 : !llvm.ptr -> f32
    %116 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %117 = llvm.mlir.constant(9 : index) : i32
    %118 = llvm.mul %108, %117 : i32
    %119 = llvm.add %118, %92 : i32
    %120 = llvm.getelementptr %116[%119] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %121 = llvm.load %120 : !llvm.ptr -> f32
    %122 = llvm.extractvalue %66[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %123 = llvm.mlir.constant(9 : index) : i32
    %124 = llvm.mul %94, %123 : i32
    %125 = llvm.add %124, %92 : i32
    %126 = llvm.getelementptr %122[%125] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %127 = llvm.load %126 : !llvm.ptr -> f32
    %128 = llvm.fmul %115, %121 : f32
    %129 = llvm.fadd %127, %128 : f32
    %130 = llvm.extractvalue %66[1] : !llvm.struct<(ptr, ptr, i32, array<2 x i32>, array<2 x i32>)> 
    %131 = llvm.mlir.constant(9 : index) : i32
    %132 = llvm.mul %94, %131 : i32
    %133 = llvm.add %132, %92 : i32
    %134 = llvm.getelementptr %130[%133] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %129, %134 : f32, !llvm.ptr
    %135 = llvm.add %108, %2 : i32
    llvm.br ^bb2(%135 : i32)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return
  }
}

