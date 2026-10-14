module attributes {llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128", llvm.target_triple = "riscv32-unknown-unknown-elf"} {
  llvm.mlir.global private unnamed_addr constant @__gac_vortex_kernel_annotation("vortex.kernel\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @__gac_vortex_source_file("gac.mlir\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global appending @llvm.global.annotations() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> {
    %0 = llvm.mlir.addressof @bevformer_layer_norm : !llvm.ptr
    %1 = llvm.mlir.addressof @bevformer_fc2_bias_residual : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr, i32, ptr)>
    %4 = llvm.mlir.addressof @__gac_vortex_kernel_annotation : !llvm.ptr
    %5 = llvm.mlir.addressof @__gac_vortex_source_file : !llvm.ptr
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.zero : !llvm.ptr
    %8 = llvm.mlir.addressof @bevformer_fc1_bias_relu : !llvm.ptr
    %9 = llvm.insertvalue %8, %3[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %10 = llvm.insertvalue %4, %9[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %11 = llvm.insertvalue %5, %10[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %12 = llvm.insertvalue %6, %11[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %13 = llvm.insertvalue %7, %12[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %14 = llvm.insertvalue %13, %2[0] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> 
    %15 = llvm.insertvalue %1, %3[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %16 = llvm.insertvalue %4, %15[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %17 = llvm.insertvalue %5, %16[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %18 = llvm.insertvalue %6, %17[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %19 = llvm.insertvalue %7, %18[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %20 = llvm.insertvalue %19, %14[1] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> 
    %21 = llvm.insertvalue %0, %3[0] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %22 = llvm.insertvalue %4, %21[1] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %23 = llvm.insertvalue %5, %22[2] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %24 = llvm.insertvalue %6, %23[3] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %25 = llvm.insertvalue %7, %24[4] : !llvm.struct<(ptr, ptr, ptr, i32, ptr)> 
    %26 = llvm.insertvalue %25, %20[2] : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>> 
    llvm.return %26 : !llvm.array<3 x struct<(ptr, ptr, ptr, i32, ptr)>>
  }
  llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.addressof @bevformer_layer_norm : !llvm.ptr
    %1 = llvm.mlir.addressof @bevformer_fc2_bias_residual : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @bevformer_fc1_bias_relu : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global appending @llvm.compiler.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<3 x ptr> {
    %0 = llvm.mlir.addressof @bevformer_layer_norm : !llvm.ptr
    %1 = llvm.mlir.addressof @bevformer_fc2_bias_residual : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.array<3 x ptr>
    %3 = llvm.mlir.addressof @bevformer_fc1_bias_relu : !llvm.ptr
    %4 = llvm.insertvalue %3, %2[0] : !llvm.array<3 x ptr> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.array<3 x ptr> 
    %6 = llvm.insertvalue %0, %5[2] : !llvm.array<3 x ptr> 
    llvm.return %6 : !llvm.array<3 x ptr>
  }
  llvm.mlir.global private constant @__gac_plan_rows(2 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_plan_channels(4 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_plan_hidden(8 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_plan_epsilon(9.99999974E-6 : f32) {addr_space = 0 : i32, section = "llvm.metadata"} : f32
  llvm.mlir.global private unnamed_addr constant @__gac_optimized_mlir_sha256("c7e19ce709c23edc6ffff1a8f3334e09db667657f6867828e9c79160ed216165\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global private unnamed_addr constant @__gac_call_plan_mlir_sha256("677ca5e82c8a00dac181f2899c4dff13dcb51966865360e583d27152a11ae26f\00") {addr_space = 0 : i32, section = "llvm.metadata"}
  llvm.mlir.global private constant @__gac_buffer_tile_m(2 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_buffer_tile_n(2 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_buffer_tile_k(16 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_buffer_stages(2 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.mlir.global private constant @__gac_buffer_lmem_bytes(512 : i32) {addr_space = 0 : i32, section = "llvm.metadata"} : i32
  llvm.func internal @__gac_dxa_expect_two(%arg0: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
    %0 = llvm.mlir.constant(11 : i32) : i32
    %1 = llvm.mlir.constant(-2147483646 : i32) : i32
    %2 = llvm.inline_asm has_side_effects ".insn r $1, 6, 0, $0, $2, $3", "=r,i,r,r,~{memory}" %0, %arg0, %1 : (i32, i32, i32) -> i32
    llvm.return
  }
  llvm.func internal @__gac_dxa_issue_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
    %0 = llvm.mlir.constant(11 : i32) : i32
    %1 = llvm.mlir.constant(43 : i32) : i32
    %2 = llvm.mlir.constant(4 : i32) : i32
    %3 = llvm.shl %arg1, %2 : i32
    %4 = llvm.or %3, %arg2 : i32
    %5 = llvm.inline_asm has_side_effects ".insn r4 $1, 0, 0, $0, $3, $4, $5\0A\09.insn r $6, 0, 3, x0, $0, x0", "={x31},i,0,r,r,r,i,~{memory}" %1, %arg0, %4, %arg3, %arg4, %0 : (i32, i32, i32, i32, i32, i32) -> i32
    llvm.return
  }
  llvm.func internal @__gac_cta_barrier(%arg0: i32, %arg1: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
    %0 = llvm.mlir.constant(11 : i32) : i32
    llvm.inline_asm has_side_effects ".insn r $0, 4, 0, x0, $1, $2", "i,r,r,~{memory}" %0, %arg0, %arg1 : (i32, i32, i32) -> ()
    llvm.return
  }
  llvm.func @bevformer_fc1_bias_relu(%arg0: !llvm.ptr) attributes {passthrough = ["vortex-kernel", "convergent"], section = ".vx_entry.bevformer_fc1_bias_relu"} {
    %0 = llvm.mlir.constant(64 : i32) : i32
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.mlir.constant(16 : i32) : i32
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.mlir.constant(0 : i32) : i32
    %8 = llvm.mlir.constant(3281 : i32) : i32
    %9 = llvm.mlir.constant(3282 : i32) : i32
    %10 = llvm.mlir.constant(3280 : i32) : i32
    %11 = llvm.mlir.constant(3295 : i32) : i32
    %12 = llvm.mlir.constant(3286 : i32) : i32
    %13 = llvm.mlir.constant(3283 : i32) : i32
    %14 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %15 = llvm.getelementptr %14[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
    %16 = llvm.load %15 : !llvm.ptr -> i32
    %17 = llvm.getelementptr %14[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
    %18 = llvm.load %17 : !llvm.ptr -> i32
    %19 = llvm.getelementptr %14[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
    %20 = llvm.load %19 : !llvm.ptr -> i32
    %21 = llvm.getelementptr %14[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
    %22 = llvm.load %21 : !llvm.ptr -> i64
    %23 = llvm.getelementptr %14[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
    %24 = llvm.load %23 : !llvm.ptr -> i64
    %25 = llvm.trunc %22 : i64 to i32
    %26 = llvm.inttoptr %25 : i32 to !llvm.ptr
    %27 = llvm.trunc %24 : i64 to i32
    %28 = llvm.inttoptr %27 : i32 to !llvm.ptr
    %29 = llvm.inline_asm "csrr $0, $1", "=r,i" %13 : (i32) -> i32
    %30 = llvm.inline_asm "csrr $0, $1", "=r,i" %12 : (i32) -> i32
    %31 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%30) : (i32) -> i32
    %32 = llvm.inline_asm "csrr $0, $1", "=r,i" %11 : (i32) -> i32
    %33 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%32) : (i32) -> i32
    %34 = llvm.inline_asm "csrr $0, $1", "=r,i" %10 : (i32) -> i32
    %35 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%34) : (i32) -> i32
    %36 = llvm.inline_asm "csrr $0, $1", "=r,i" %9 : (i32) -> i32
    %37 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%36) : (i32) -> i32
    %38 = llvm.inline_asm "csrr $0, $1", "=r,i" %8 : (i32) -> i32
    %39 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%38) : (i32) -> i32
    %40 = llvm.inttoptr %33 : i32 to !llvm.ptr
    %41 = llvm.sub %20, %6 : i32
    %42 = llvm.udiv %41, %3 : i32
    %43 = llvm.add %42, %6 : i32
    %44 = llvm.udiv %31, %43 : i32
    %45 = llvm.urem %31, %43 : i32
    %46 = llvm.mul %44, %3 : i32
    %47 = llvm.mul %45, %3 : i32
    %48 = llvm.udiv %29, %3 : i32
    %49 = llvm.urem %29, %3 : i32
    %50 = llvm.add %46, %48 : i32
    %51 = llvm.add %47, %49 : i32
    %52 = llvm.icmp "ult" %50, %16 : i32
    %53 = llvm.icmp "ult" %51, %20 : i32
    %54 = llvm.and %52, %53 : i1
    %55 = llvm.icmp "eq" %39, %7 : i32
    llvm.cond_br %54, ^bb1, ^bb2(%4 : f32)
  ^bb1:  // pred: ^bb0
    %56 = llvm.getelementptr %26[%51] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %57 = llvm.load %56 : !llvm.ptr -> f32
    llvm.br ^bb2(%57 : f32)
  ^bb2(%58: f32):  // 2 preds: ^bb0, ^bb1
    llvm.cond_br %55, ^bb3, ^bb4(%7, %7, %58 : i32, i32, f32)
  ^bb3:  // pred: ^bb2
    llvm.call @__gac_dxa_expect_two(%35) {convergent} : (i32) -> ()
    %59 = llvm.mul %7, %0 : i32
    %60 = llvm.getelementptr %40[%59] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %61 = llvm.add %59, %1 : i32
    %62 = llvm.getelementptr %40[%61] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %63 = llvm.ptrtoint %60 : !llvm.ptr to i32
    %64 = llvm.ptrtoint %62 : !llvm.ptr to i32
    llvm.call @__gac_dxa_issue_2d(%63, %35, %7, %7, %46) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.call @__gac_dxa_issue_2d(%64, %35, %6, %47, %7) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.br ^bb4(%7, %7, %58 : i32, i32, f32)
  ^bb4(%65: i32, %66: i32, %67: f32):  // 3 preds: ^bb2, ^bb3, ^bb9
    %68 = llvm.add %65, %2 : i32
    %69 = llvm.icmp "ult" %68, %18 : i32
    %70 = llvm.xor %66, %6 : i32
    %71 = llvm.shl %66, %5 : i32
    %72 = llvm.or %71, %35 : i32
    %73 = llvm.shl %70, %5 : i32
    %74 = llvm.or %73, %35 : i32
    %75 = llvm.and %69, %55 : i1
    llvm.cond_br %75, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @__gac_dxa_expect_two(%74) {convergent} : (i32) -> ()
    %76 = llvm.mul %70, %0 : i32
    %77 = llvm.getelementptr %40[%76] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %78 = llvm.add %76, %1 : i32
    %79 = llvm.getelementptr %40[%78] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %80 = llvm.ptrtoint %77 : !llvm.ptr to i32
    %81 = llvm.ptrtoint %79 : !llvm.ptr to i32
    llvm.call @__gac_dxa_issue_2d(%80, %74, %7, %68, %46) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.call @__gac_dxa_issue_2d(%81, %74, %6, %47, %68) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    llvm.call @__gac_cta_barrier(%72, %37) {convergent} : (i32, i32) -> ()
    %82 = llvm.mul %66, %0 : i32
    %83 = llvm.getelementptr %40[%82] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %84 = llvm.add %82, %1 : i32
    %85 = llvm.getelementptr %40[%84] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %86 = llvm.sub %18, %65 : i32
    %87 = llvm.icmp "ult" %86, %2 : i32
    %88 = llvm.select %87, %86, %2 : i1, i32
    llvm.br ^bb7(%7, %67 : i32, f32)
  ^bb7(%89: i32, %90: f32):  // 2 preds: ^bb6, ^bb8
    %91 = llvm.icmp "ult" %89, %88 : i32
    llvm.cond_br %91, ^bb8, ^bb9(%90 : f32)
  ^bb8:  // pred: ^bb7
    %92 = llvm.mul %48, %2 : i32
    %93 = llvm.add %92, %89 : i32
    %94 = llvm.mul %89, %3 : i32
    %95 = llvm.add %94, %49 : i32
    %96 = llvm.getelementptr %83[%93] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %97 = llvm.getelementptr %85[%95] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %98 = llvm.load %96 : !llvm.ptr -> f32
    %99 = llvm.load %97 : !llvm.ptr -> f32
    %100 = llvm.fmul %98, %99 : f32
    %101 = llvm.fadd %90, %100 : f32
    %102 = llvm.add %89, %6 : i32
    llvm.br ^bb7(%102, %101 : i32, f32)
  ^bb9(%103: f32):  // pred: ^bb7
    llvm.call @__gac_cta_barrier(%72, %37) {convergent} : (i32, i32) -> ()
    llvm.cond_br %69, ^bb4(%68, %70, %103 : i32, i32, f32), ^bb10(%103 : f32)
  ^bb10(%104: f32):  // pred: ^bb9
    llvm.cond_br %54, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %105 = llvm.mul %50, %20 : i32
    %106 = llvm.add %105, %51 : i32
    %107 = llvm.fcmp "ogt" %104, %4 : f32
    %108 = llvm.select %107, %104, %4 : i1, f32
    %109 = llvm.getelementptr %28[%106] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %108, %109 : f32, !llvm.ptr
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    llvm.return
  }
  llvm.func @bevformer_fc2_bias_residual(%arg0: !llvm.ptr) attributes {passthrough = ["vortex-kernel", "convergent"], section = ".vx_entry.bevformer_fc2_bias_residual"} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.constant(64 : i32) : i32
    %2 = llvm.mlir.constant(32 : i32) : i32
    %3 = llvm.mlir.constant(16 : i32) : i32
    %4 = llvm.mlir.constant(2 : i32) : i32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(8 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(0 : i32) : i32
    %9 = llvm.mlir.constant(3281 : i32) : i32
    %10 = llvm.mlir.constant(3282 : i32) : i32
    %11 = llvm.mlir.constant(3280 : i32) : i32
    %12 = llvm.mlir.constant(3295 : i32) : i32
    %13 = llvm.mlir.constant(3286 : i32) : i32
    %14 = llvm.mlir.constant(3283 : i32) : i32
    %15 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %16 = llvm.getelementptr %15[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %17 = llvm.load %16 : !llvm.ptr -> i32
    %18 = llvm.getelementptr %15[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %19 = llvm.load %18 : !llvm.ptr -> i32
    %20 = llvm.getelementptr %15[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %21 = llvm.load %20 : !llvm.ptr -> i32
    %22 = llvm.getelementptr %15[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %23 = llvm.load %22 : !llvm.ptr -> i64
    %24 = llvm.getelementptr %15[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %25 = llvm.load %24 : !llvm.ptr -> i64
    %26 = llvm.getelementptr %15[0, 8] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>
    %27 = llvm.load %26 : !llvm.ptr -> i64
    %28 = llvm.trunc %23 : i64 to i32
    %29 = llvm.inttoptr %28 : i32 to !llvm.ptr
    %30 = llvm.trunc %25 : i64 to i32
    %31 = llvm.inttoptr %30 : i32 to !llvm.ptr
    %32 = llvm.trunc %27 : i64 to i32
    %33 = llvm.inttoptr %32 : i32 to !llvm.ptr
    %34 = llvm.inline_asm "csrr $0, $1", "=r,i" %14 : (i32) -> i32
    %35 = llvm.inline_asm "csrr $0, $1", "=r,i" %13 : (i32) -> i32
    %36 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%35) : (i32) -> i32
    %37 = llvm.inline_asm "csrr $0, $1", "=r,i" %12 : (i32) -> i32
    %38 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%37) : (i32) -> i32
    %39 = llvm.inline_asm "csrr $0, $1", "=r,i" %11 : (i32) -> i32
    %40 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%39) : (i32) -> i32
    %41 = llvm.inline_asm "csrr $0, $1", "=r,i" %10 : (i32) -> i32
    %42 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%41) : (i32) -> i32
    %43 = llvm.inline_asm "csrr $0, $1", "=r,i" %9 : (i32) -> i32
    %44 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%43) : (i32) -> i32
    %45 = llvm.inttoptr %38 : i32 to !llvm.ptr
    %46 = llvm.sub %19, %7 : i32
    %47 = llvm.udiv %46, %4 : i32
    %48 = llvm.add %47, %7 : i32
    %49 = llvm.udiv %36, %48 : i32
    %50 = llvm.urem %36, %48 : i32
    %51 = llvm.mul %49, %4 : i32
    %52 = llvm.mul %50, %4 : i32
    %53 = llvm.udiv %34, %4 : i32
    %54 = llvm.urem %34, %4 : i32
    %55 = llvm.add %51, %53 : i32
    %56 = llvm.add %52, %54 : i32
    %57 = llvm.icmp "ult" %55, %17 : i32
    %58 = llvm.icmp "ult" %56, %19 : i32
    %59 = llvm.and %57, %58 : i1
    %60 = llvm.icmp "eq" %44, %8 : i32
    llvm.cond_br %59, ^bb1, ^bb2(%5 : f32)
  ^bb1:  // pred: ^bb0
    %61 = llvm.getelementptr %31[%56] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %62 = llvm.load %61 : !llvm.ptr -> f32
    llvm.br ^bb2(%62 : f32)
  ^bb2(%63: f32):  // 2 preds: ^bb0, ^bb1
    llvm.cond_br %60, ^bb3, ^bb4(%8, %8, %63 : i32, i32, f32)
  ^bb3:  // pred: ^bb2
    llvm.call @__gac_dxa_expect_two(%40) {convergent} : (i32) -> ()
    %64 = llvm.mul %8, %1 : i32
    %65 = llvm.getelementptr %45[%64] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %66 = llvm.add %64, %2 : i32
    %67 = llvm.getelementptr %45[%66] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %68 = llvm.ptrtoint %65 : !llvm.ptr to i32
    %69 = llvm.ptrtoint %67 : !llvm.ptr to i32
    llvm.call @__gac_dxa_issue_2d(%68, %40, %4, %8, %51) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.call @__gac_dxa_issue_2d(%69, %40, %0, %52, %8) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.br ^bb4(%8, %8, %63 : i32, i32, f32)
  ^bb4(%70: i32, %71: i32, %72: f32):  // 3 preds: ^bb2, ^bb3, ^bb9
    %73 = llvm.add %70, %3 : i32
    %74 = llvm.icmp "ult" %73, %21 : i32
    %75 = llvm.xor %71, %7 : i32
    %76 = llvm.shl %71, %6 : i32
    %77 = llvm.or %76, %40 : i32
    %78 = llvm.shl %75, %6 : i32
    %79 = llvm.or %78, %40 : i32
    %80 = llvm.and %74, %60 : i1
    llvm.cond_br %80, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.call @__gac_dxa_expect_two(%79) {convergent} : (i32) -> ()
    %81 = llvm.mul %75, %1 : i32
    %82 = llvm.getelementptr %45[%81] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %83 = llvm.add %81, %2 : i32
    %84 = llvm.getelementptr %45[%83] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %85 = llvm.ptrtoint %82 : !llvm.ptr to i32
    %86 = llvm.ptrtoint %84 : !llvm.ptr to i32
    llvm.call @__gac_dxa_issue_2d(%85, %79, %4, %73, %51) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.call @__gac_dxa_issue_2d(%86, %79, %0, %52, %73) {convergent} : (i32, i32, i32, i32, i32) -> ()
    llvm.br ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    llvm.call @__gac_cta_barrier(%77, %42) {convergent} : (i32, i32) -> ()
    %87 = llvm.mul %71, %1 : i32
    %88 = llvm.getelementptr %45[%87] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %89 = llvm.add %87, %2 : i32
    %90 = llvm.getelementptr %45[%89] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %91 = llvm.sub %21, %70 : i32
    %92 = llvm.icmp "ult" %91, %3 : i32
    %93 = llvm.select %92, %91, %3 : i1, i32
    llvm.br ^bb7(%8, %72 : i32, f32)
  ^bb7(%94: i32, %95: f32):  // 2 preds: ^bb6, ^bb8
    %96 = llvm.icmp "ult" %94, %93 : i32
    llvm.cond_br %96, ^bb8, ^bb9(%95 : f32)
  ^bb8:  // pred: ^bb7
    %97 = llvm.mul %53, %3 : i32
    %98 = llvm.add %97, %94 : i32
    %99 = llvm.mul %94, %4 : i32
    %100 = llvm.add %99, %54 : i32
    %101 = llvm.getelementptr %88[%98] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %102 = llvm.getelementptr %90[%100] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %103 = llvm.load %101 : !llvm.ptr -> f32
    %104 = llvm.load %102 : !llvm.ptr -> f32
    %105 = llvm.fmul %103, %104 : f32
    %106 = llvm.fadd %95, %105 : f32
    %107 = llvm.add %94, %7 : i32
    llvm.br ^bb7(%107, %106 : i32, f32)
  ^bb9(%108: f32):  // pred: ^bb7
    llvm.call @__gac_cta_barrier(%77, %42) {convergent} : (i32, i32) -> ()
    llvm.cond_br %74, ^bb4(%73, %75, %108 : i32, i32, f32), ^bb10(%108 : f32)
  ^bb10(%109: f32):  // pred: ^bb9
    llvm.cond_br %59, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %110 = llvm.mul %55, %19 : i32
    %111 = llvm.add %110, %56 : i32
    %112 = llvm.getelementptr %29[%111] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %113 = llvm.load %112 : !llvm.ptr -> f32
    %114 = llvm.fadd %109, %113 : f32
    %115 = llvm.getelementptr %33[%111] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    llvm.store %114, %115 : f32, !llvm.ptr
    llvm.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    llvm.return
  }
  llvm.func @bevformer_layer_norm(%arg0: !llvm.ptr) attributes {passthrough = ["vortex-kernel"], section = ".vx_entry.bevformer_layer_norm"} {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(3289 : i32) : i32
    %5 = llvm.mlir.constant(3286 : i32) : i32
    %6 = llvm.mlir.constant(3283 : i32) : i32
    %7 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg0) : (!llvm.ptr) -> !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %9 = llvm.getelementptr %7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %10 = llvm.getelementptr %7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %11 = llvm.getelementptr %7[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %12 = llvm.getelementptr %7[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %13 = llvm.getelementptr %7[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %14 = llvm.getelementptr %7[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>
    %15 = llvm.load %8 : !llvm.ptr -> i32
    %16 = llvm.load %9 : !llvm.ptr -> i32
    %17 = llvm.load %10 : !llvm.ptr -> f32
    %18 = llvm.load %11 : !llvm.ptr -> i64
    %19 = llvm.load %12 : !llvm.ptr -> i64
    %20 = llvm.load %13 : !llvm.ptr -> i64
    %21 = llvm.load %14 : !llvm.ptr -> i64
    %22 = llvm.inline_asm "csrr $0, $1", "=r,i" %6 : (i32) -> i32
    %23 = llvm.inline_asm "csrr $0, $1", "=r,i" %5 : (i32) -> i32
    %24 = llvm.inline_asm "csrr $0, $1", "=r,i" %4 : (i32) -> i32
    %25 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%23) : (i32) -> i32
    %26 = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%24) : (i32) -> i32
    %27 = llvm.mul %25, %26 : i32
    %28 = llvm.add %27, %22 : i32
    %29 = llvm.icmp "ult" %28, %15 : i32
    llvm.cond_br %29, ^bb1, ^bb10
  ^bb1:  // pred: ^bb0
    %30 = llvm.trunc %18 : i64 to i32
    %31 = llvm.trunc %19 : i64 to i32
    %32 = llvm.trunc %20 : i64 to i32
    %33 = llvm.trunc %21 : i64 to i32
    %34 = llvm.inttoptr %30 : i32 to !llvm.ptr
    %35 = llvm.inttoptr %31 : i32 to !llvm.ptr
    %36 = llvm.inttoptr %32 : i32 to !llvm.ptr
    %37 = llvm.inttoptr %33 : i32 to !llvm.ptr
    %38 = llvm.mul %28, %16 : i32
    llvm.br ^bb2(%3, %2 : i32, f32)
  ^bb2(%39: i32, %40: f32):  // 2 preds: ^bb1, ^bb3
    %41 = llvm.icmp "ult" %39, %16 : i32
    llvm.cond_br %41, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %42 = llvm.add %38, %39 : i32
    %43 = llvm.getelementptr %34[%42] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %44 = llvm.load %43 : !llvm.ptr -> f32
    %45 = llvm.fadd %40, %44 : f32
    %46 = llvm.add %39, %1 : i32
    llvm.br ^bb2(%46, %45 : i32, f32)
  ^bb4:  // pred: ^bb2
    %47 = llvm.uitofp %16 : i32 to f32
    %48 = llvm.fdiv %40, %47 : f32
    llvm.br ^bb5(%3, %2 : i32, f32)
  ^bb5(%49: i32, %50: f32):  // 2 preds: ^bb4, ^bb6
    %51 = llvm.icmp "ult" %49, %16 : i32
    llvm.cond_br %51, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %52 = llvm.add %38, %49 : i32
    %53 = llvm.getelementptr %34[%52] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %54 = llvm.load %53 : !llvm.ptr -> f32
    %55 = llvm.fsub %54, %48 : f32
    %56 = llvm.fmul %55, %55 : f32
    %57 = llvm.fadd %50, %56 : f32
    %58 = llvm.add %49, %1 : i32
    llvm.br ^bb5(%58, %57 : i32, f32)
  ^bb7:  // pred: ^bb5
    %59 = llvm.fdiv %50, %47 : f32
    %60 = llvm.fadd %59, %17 : f32
    %61 = llvm.call_intrinsic "llvm.sqrt"(%60) : (f32) -> f32
    %62 = llvm.fdiv %0, %61 : f32
    llvm.br ^bb8(%3 : i32)
  ^bb8(%63: i32):  // 2 preds: ^bb7, ^bb9
    %64 = llvm.icmp "ult" %63, %16 : i32
    llvm.cond_br %64, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %65 = llvm.add %38, %63 : i32
    %66 = llvm.getelementptr %34[%65] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %67 = llvm.getelementptr %35[%63] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %68 = llvm.getelementptr %36[%63] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %69 = llvm.getelementptr %37[%65] : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %70 = llvm.load %66 : !llvm.ptr -> f32
    %71 = llvm.load %67 : !llvm.ptr -> f32
    %72 = llvm.load %68 : !llvm.ptr -> f32
    %73 = llvm.fsub %70, %48 : f32
    %74 = llvm.fmul %73, %62 : f32
    %75 = llvm.fmul %74, %71 : f32
    %76 = llvm.fadd %75, %72 : f32
    llvm.store %76, %69 : f32, !llvm.ptr
    %77 = llvm.add %63, %1 : i32
    llvm.br ^bb8(%77 : i32)
  ^bb10:  // 2 preds: ^bb0, ^bb8
    llvm.return
  }
}

