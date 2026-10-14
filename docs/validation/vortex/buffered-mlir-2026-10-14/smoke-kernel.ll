; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@__gac_vortex_kernel_annotation = private unnamed_addr constant [14 x i8] c"vortex.kernel\00", section "llvm.metadata"
@__gac_vortex_source_file = private unnamed_addr constant [9 x i8] c"gac.mlir\00", section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @bevformer_fc1_bias_relu, ptr @__gac_vortex_kernel_annotation, ptr @__gac_vortex_source_file, i32 1, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @bevformer_fc2_bias_residual, ptr @__gac_vortex_kernel_annotation, ptr @__gac_vortex_source_file, i32 1, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @bevformer_layer_norm, ptr @__gac_vortex_kernel_annotation, ptr @__gac_vortex_source_file, i32 1, ptr null }], section "llvm.metadata"
@llvm.used = appending global [3 x ptr] [ptr @bevformer_fc1_bias_relu, ptr @bevformer_fc2_bias_residual, ptr @bevformer_layer_norm], section "llvm.metadata"
@llvm.compiler.used = appending global [3 x ptr] [ptr @bevformer_fc1_bias_relu, ptr @bevformer_fc2_bias_residual, ptr @bevformer_layer_norm], section "llvm.metadata"
@__gac_plan_rows = private constant i32 2, section "llvm.metadata"
@__gac_plan_channels = private constant i32 4, section "llvm.metadata"
@__gac_plan_hidden = private constant i32 8, section "llvm.metadata"
@__gac_plan_epsilon = private constant float 0x3EE4F8B580000000, section "llvm.metadata"
@__gac_optimized_mlir_sha256 = private unnamed_addr constant [65 x i8] c"c7e19ce709c23edc6ffff1a8f3334e09db667657f6867828e9c79160ed216165\00", section "llvm.metadata"
@__gac_call_plan_mlir_sha256 = private unnamed_addr constant [65 x i8] c"677ca5e82c8a00dac181f2899c4dff13dcb51966865360e583d27152a11ae26f\00", section "llvm.metadata"
@__gac_buffer_tile_m = private constant i32 2, section "llvm.metadata"
@__gac_buffer_tile_n = private constant i32 2, section "llvm.metadata"
@__gac_buffer_tile_k = private constant i32 16, section "llvm.metadata"
@__gac_buffer_stages = private constant i32 2, section "llvm.metadata"
@__gac_buffer_lmem_bytes = private constant i32 512, section "llvm.metadata"

; Function Attrs: convergent noinline optnone
define internal void @__gac_dxa_expect_two(i32 %0) #0 {
  %2 = call i32 asm sideeffect ".insn r $1, 6, 0, $0, $2, $3", "=r,i,r,r,~{memory}"(i32 11, i32 %0, i32 -2147483646)
  ret void
}

; Function Attrs: convergent noinline optnone
define internal void @__gac_dxa_issue_2d(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4) #0 {
  %6 = shl i32 %1, 4
  %7 = or i32 %6, %2
  %8 = call i32 asm sideeffect ".insn r4 $1, 0, 0, $0, $3, $4, $5\0A\09.insn r $6, 0, 3, x0, $0, x0", "={x31},i,0,r,r,r,i,~{memory}"(i32 43, i32 %0, i32 %7, i32 %3, i32 %4, i32 11)
  ret void
}

; Function Attrs: convergent noinline optnone
define internal void @__gac_cta_barrier(i32 %0, i32 %1) #0 {
  call void asm sideeffect ".insn r $0, 4, 0, x0, $1, $2", "i,r,r,~{memory}"(i32 11, i32 %0, i32 %1)
  ret void
}

; Function Attrs: convergent
define void @bevformer_fc1_bias_relu(ptr %0) #1 section ".vx_entry.bevformer_fc1_bias_relu" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  %7 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 2
  %8 = load i32, ptr %7, align 4
  %9 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 6
  %10 = load i64, ptr %9, align 8
  %11 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 7
  %12 = load i64, ptr %11, align 8
  %13 = trunc i64 %10 to i32
  %14 = inttoptr i32 %13 to ptr
  %15 = trunc i64 %12 to i32
  %16 = inttoptr i32 %15 to ptr
  %17 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %18 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %19 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %18)
  %20 = call i32 asm "csrr $0, $1", "=r,i"(i32 3295)
  %21 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %20)
  %22 = call i32 asm "csrr $0, $1", "=r,i"(i32 3280)
  %23 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %22)
  %24 = call i32 asm "csrr $0, $1", "=r,i"(i32 3282)
  %25 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %24)
  %26 = call i32 asm "csrr $0, $1", "=r,i"(i32 3281)
  %27 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %26)
  %28 = inttoptr i32 %21 to ptr
  %29 = sub i32 %8, 1
  %30 = udiv i32 %29, 2
  %31 = add i32 %30, 1
  %32 = udiv i32 %19, %31
  %33 = urem i32 %19, %31
  %34 = mul i32 %32, 2
  %35 = mul i32 %33, 2
  %36 = udiv i32 %17, 2
  %37 = urem i32 %17, 2
  %38 = add i32 %34, %36
  %39 = add i32 %35, %37
  %40 = icmp ult i32 %38, %4
  %41 = icmp ult i32 %39, %8
  %42 = and i1 %40, %41
  %43 = icmp eq i32 %27, 0
  br i1 %42, label %44, label %47

44:                                               ; preds = %1
  %45 = getelementptr float, ptr %14, i32 %39
  %46 = load float, ptr %45, align 4
  br label %47

47:                                               ; preds = %44, %1
  %48 = phi float [ %46, %44 ], [ 0.000000e+00, %1 ]
  br i1 %43, label %49, label %54

49:                                               ; preds = %47
  call void @__gac_dxa_expect_two(i32 %23) #5
  %50 = getelementptr float, ptr %28, i32 0
  %51 = getelementptr float, ptr %28, i32 32
  %52 = ptrtoint ptr %50 to i32
  %53 = ptrtoint ptr %51 to i32
  call void @__gac_dxa_issue_2d(i32 %52, i32 %23, i32 0, i32 0, i32 %34) #5
  call void @__gac_dxa_issue_2d(i32 %53, i32 %23, i32 1, i32 %35, i32 0) #5
  br label %54

54:                                               ; preds = %97, %49, %47
  %55 = phi i32 [ %58, %97 ], [ 0, %49 ], [ 0, %47 ]
  %56 = phi i32 [ %60, %97 ], [ 0, %49 ], [ 0, %47 ]
  %57 = phi float [ %98, %97 ], [ %48, %49 ], [ %48, %47 ]
  %58 = add i32 %55, 16
  %59 = icmp ult i32 %58, %6
  %60 = xor i32 %56, 1
  %61 = shl i32 %56, 8
  %62 = or i32 %61, %23
  %63 = shl i32 %60, 8
  %64 = or i32 %63, %23
  %65 = and i1 %59, %43
  br i1 %65, label %66, label %73

66:                                               ; preds = %54
  call void @__gac_dxa_expect_two(i32 %64) #5
  %67 = mul i32 %60, 64
  %68 = getelementptr float, ptr %28, i32 %67
  %69 = add i32 %67, 32
  %70 = getelementptr float, ptr %28, i32 %69
  %71 = ptrtoint ptr %68 to i32
  %72 = ptrtoint ptr %70 to i32
  call void @__gac_dxa_issue_2d(i32 %71, i32 %64, i32 0, i32 %58, i32 %34) #5
  call void @__gac_dxa_issue_2d(i32 %72, i32 %64, i32 1, i32 %35, i32 %58) #5
  br label %73

73:                                               ; preds = %66, %54
  call void @__gac_cta_barrier(i32 %62, i32 %25) #5
  %74 = mul i32 %56, 64
  %75 = getelementptr float, ptr %28, i32 %74
  %76 = add i32 %74, 32
  %77 = getelementptr float, ptr %28, i32 %76
  %78 = sub i32 %6, %55
  %79 = icmp ult i32 %78, 16
  %80 = select i1 %79, i32 %78, i32 16
  br label %81

81:                                               ; preds = %85, %73
  %82 = phi i32 [ %96, %85 ], [ 0, %73 ]
  %83 = phi float [ %95, %85 ], [ %57, %73 ]
  %84 = icmp ult i32 %82, %80
  br i1 %84, label %85, label %97

85:                                               ; preds = %81
  %86 = mul i32 %36, 16
  %87 = add i32 %86, %82
  %88 = mul i32 %82, 2
  %89 = add i32 %88, %37
  %90 = getelementptr float, ptr %75, i32 %87
  %91 = getelementptr float, ptr %77, i32 %89
  %92 = load float, ptr %90, align 4
  %93 = load float, ptr %91, align 4
  %94 = fmul float %92, %93
  %95 = fadd float %83, %94
  %96 = add i32 %82, 1
  br label %81

97:                                               ; preds = %81
  %98 = phi float [ %83, %81 ]
  call void @__gac_cta_barrier(i32 %62, i32 %25) #5
  br i1 %59, label %54, label %99

99:                                               ; preds = %97
  %100 = phi float [ %98, %97 ]
  br i1 %42, label %101, label %107

101:                                              ; preds = %99
  %102 = mul i32 %38, %8
  %103 = add i32 %102, %39
  %104 = fcmp ogt float %100, 0.000000e+00
  %105 = select i1 %104, float %100, float 0.000000e+00
  %106 = getelementptr float, ptr %16, i32 %103
  store float %105, ptr %106, align 4
  br label %107

107:                                              ; preds = %101, %99
  ret void
}

; Function Attrs: convergent
define void @bevformer_fc2_bias_residual(ptr %0) #1 section ".vx_entry.bevformer_fc2_bias_residual" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 0
  %4 = load i32, ptr %3, align 4
  %5 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 1
  %6 = load i32, ptr %5, align 4
  %7 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 2
  %8 = load i32, ptr %7, align 4
  %9 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 4
  %10 = load i64, ptr %9, align 8
  %11 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 7
  %12 = load i64, ptr %11, align 8
  %13 = getelementptr { i32, i32, i32, i32, i64, i64, i64, i64, i64 }, ptr %2, i32 0, i32 8
  %14 = load i64, ptr %13, align 8
  %15 = trunc i64 %10 to i32
  %16 = inttoptr i32 %15 to ptr
  %17 = trunc i64 %12 to i32
  %18 = inttoptr i32 %17 to ptr
  %19 = trunc i64 %14 to i32
  %20 = inttoptr i32 %19 to ptr
  %21 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %22 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %23 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %22)
  %24 = call i32 asm "csrr $0, $1", "=r,i"(i32 3295)
  %25 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %24)
  %26 = call i32 asm "csrr $0, $1", "=r,i"(i32 3280)
  %27 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %26)
  %28 = call i32 asm "csrr $0, $1", "=r,i"(i32 3282)
  %29 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %28)
  %30 = call i32 asm "csrr $0, $1", "=r,i"(i32 3281)
  %31 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %30)
  %32 = inttoptr i32 %25 to ptr
  %33 = sub i32 %6, 1
  %34 = udiv i32 %33, 2
  %35 = add i32 %34, 1
  %36 = udiv i32 %23, %35
  %37 = urem i32 %23, %35
  %38 = mul i32 %36, 2
  %39 = mul i32 %37, 2
  %40 = udiv i32 %21, 2
  %41 = urem i32 %21, 2
  %42 = add i32 %38, %40
  %43 = add i32 %39, %41
  %44 = icmp ult i32 %42, %4
  %45 = icmp ult i32 %43, %6
  %46 = and i1 %44, %45
  %47 = icmp eq i32 %31, 0
  br i1 %46, label %48, label %51

48:                                               ; preds = %1
  %49 = getelementptr float, ptr %18, i32 %43
  %50 = load float, ptr %49, align 4
  br label %51

51:                                               ; preds = %48, %1
  %52 = phi float [ %50, %48 ], [ 0.000000e+00, %1 ]
  br i1 %47, label %53, label %58

53:                                               ; preds = %51
  call void @__gac_dxa_expect_two(i32 %27) #5
  %54 = getelementptr float, ptr %32, i32 0
  %55 = getelementptr float, ptr %32, i32 32
  %56 = ptrtoint ptr %54 to i32
  %57 = ptrtoint ptr %55 to i32
  call void @__gac_dxa_issue_2d(i32 %56, i32 %27, i32 2, i32 0, i32 %38) #5
  call void @__gac_dxa_issue_2d(i32 %57, i32 %27, i32 3, i32 %39, i32 0) #5
  br label %58

58:                                               ; preds = %101, %53, %51
  %59 = phi i32 [ %62, %101 ], [ 0, %53 ], [ 0, %51 ]
  %60 = phi i32 [ %64, %101 ], [ 0, %53 ], [ 0, %51 ]
  %61 = phi float [ %102, %101 ], [ %52, %53 ], [ %52, %51 ]
  %62 = add i32 %59, 16
  %63 = icmp ult i32 %62, %8
  %64 = xor i32 %60, 1
  %65 = shl i32 %60, 8
  %66 = or i32 %65, %27
  %67 = shl i32 %64, 8
  %68 = or i32 %67, %27
  %69 = and i1 %63, %47
  br i1 %69, label %70, label %77

70:                                               ; preds = %58
  call void @__gac_dxa_expect_two(i32 %68) #5
  %71 = mul i32 %64, 64
  %72 = getelementptr float, ptr %32, i32 %71
  %73 = add i32 %71, 32
  %74 = getelementptr float, ptr %32, i32 %73
  %75 = ptrtoint ptr %72 to i32
  %76 = ptrtoint ptr %74 to i32
  call void @__gac_dxa_issue_2d(i32 %75, i32 %68, i32 2, i32 %62, i32 %38) #5
  call void @__gac_dxa_issue_2d(i32 %76, i32 %68, i32 3, i32 %39, i32 %62) #5
  br label %77

77:                                               ; preds = %70, %58
  call void @__gac_cta_barrier(i32 %66, i32 %29) #5
  %78 = mul i32 %60, 64
  %79 = getelementptr float, ptr %32, i32 %78
  %80 = add i32 %78, 32
  %81 = getelementptr float, ptr %32, i32 %80
  %82 = sub i32 %8, %59
  %83 = icmp ult i32 %82, 16
  %84 = select i1 %83, i32 %82, i32 16
  br label %85

85:                                               ; preds = %89, %77
  %86 = phi i32 [ %100, %89 ], [ 0, %77 ]
  %87 = phi float [ %99, %89 ], [ %61, %77 ]
  %88 = icmp ult i32 %86, %84
  br i1 %88, label %89, label %101

89:                                               ; preds = %85
  %90 = mul i32 %40, 16
  %91 = add i32 %90, %86
  %92 = mul i32 %86, 2
  %93 = add i32 %92, %41
  %94 = getelementptr float, ptr %79, i32 %91
  %95 = getelementptr float, ptr %81, i32 %93
  %96 = load float, ptr %94, align 4
  %97 = load float, ptr %95, align 4
  %98 = fmul float %96, %97
  %99 = fadd float %87, %98
  %100 = add i32 %86, 1
  br label %85

101:                                              ; preds = %85
  %102 = phi float [ %87, %85 ]
  call void @__gac_cta_barrier(i32 %66, i32 %29) #5
  br i1 %63, label %58, label %103

103:                                              ; preds = %101
  %104 = phi float [ %102, %101 ]
  br i1 %46, label %105, label %112

105:                                              ; preds = %103
  %106 = mul i32 %42, %6
  %107 = add i32 %106, %43
  %108 = getelementptr float, ptr %16, i32 %107
  %109 = load float, ptr %108, align 4
  %110 = fadd float %104, %109
  %111 = getelementptr float, ptr %20, i32 %107
  store float %110, ptr %111, align 4
  br label %112

112:                                              ; preds = %105, %103
  ret void
}

define void @bevformer_layer_norm(ptr %0) #2 section ".vx_entry.bevformer_layer_norm" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 0
  %4 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 1
  %5 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 2
  %6 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 4
  %7 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 5
  %8 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 6
  %9 = getelementptr { i32, i32, float, i32, i64, i64, i64, i64 }, ptr %2, i32 0, i32 7
  %10 = load i32, ptr %3, align 4
  %11 = load i32, ptr %4, align 4
  %12 = load float, ptr %5, align 4
  %13 = load i64, ptr %6, align 8
  %14 = load i64, ptr %7, align 8
  %15 = load i64, ptr %8, align 8
  %16 = load i64, ptr %9, align 8
  %17 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %18 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3289)
  %20 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %18)
  %21 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %19)
  %22 = mul i32 %20, %21
  %23 = add i32 %22, %17
  %24 = icmp ult i32 %23, %10
  br i1 %24, label %25, label %82

25:                                               ; preds = %1
  %26 = trunc i64 %13 to i32
  %27 = trunc i64 %14 to i32
  %28 = trunc i64 %15 to i32
  %29 = trunc i64 %16 to i32
  %30 = inttoptr i32 %26 to ptr
  %31 = inttoptr i32 %27 to ptr
  %32 = inttoptr i32 %28 to ptr
  %33 = inttoptr i32 %29 to ptr
  %34 = mul i32 %23, %11
  br label %35

35:                                               ; preds = %39, %25
  %36 = phi i32 [ %44, %39 ], [ 0, %25 ]
  %37 = phi float [ %43, %39 ], [ 0.000000e+00, %25 ]
  %38 = icmp ult i32 %36, %11
  br i1 %38, label %39, label %45

39:                                               ; preds = %35
  %40 = add i32 %34, %36
  %41 = getelementptr float, ptr %30, i32 %40
  %42 = load float, ptr %41, align 4
  %43 = fadd float %37, %42
  %44 = add i32 %36, 1
  br label %35

45:                                               ; preds = %35
  %46 = uitofp i32 %11 to float
  %47 = fdiv float %37, %46
  br label %48

48:                                               ; preds = %52, %45
  %49 = phi i32 [ %59, %52 ], [ 0, %45 ]
  %50 = phi float [ %58, %52 ], [ 0.000000e+00, %45 ]
  %51 = icmp ult i32 %49, %11
  br i1 %51, label %52, label %60

52:                                               ; preds = %48
  %53 = add i32 %34, %49
  %54 = getelementptr float, ptr %30, i32 %53
  %55 = load float, ptr %54, align 4
  %56 = fsub float %55, %47
  %57 = fmul float %56, %56
  %58 = fadd float %50, %57
  %59 = add i32 %49, 1
  br label %48

60:                                               ; preds = %48
  %61 = fdiv float %50, %46
  %62 = fadd float %61, %12
  %63 = call float @llvm.sqrt.f32(float %62)
  %64 = fdiv float 1.000000e+00, %63
  br label %65

65:                                               ; preds = %68, %60
  %66 = phi i32 [ %81, %68 ], [ 0, %60 ]
  %67 = icmp ult i32 %66, %11
  br i1 %67, label %68, label %82

68:                                               ; preds = %65
  %69 = add i32 %34, %66
  %70 = getelementptr float, ptr %30, i32 %69
  %71 = getelementptr float, ptr %31, i32 %66
  %72 = getelementptr float, ptr %32, i32 %66
  %73 = getelementptr float, ptr %33, i32 %69
  %74 = load float, ptr %70, align 4
  %75 = load float, ptr %71, align 4
  %76 = load float, ptr %72, align 4
  %77 = fsub float %74, %47
  %78 = fmul float %77, %64
  %79 = fmul float %78, %75
  %80 = fadd float %79, %76
  store float %80, ptr %73, align 4
  %81 = add i32 %66, 1
  br label %65

82:                                               ; preds = %65, %1
  ret void
}

; Function Attrs: nounwind memory(none)
declare ptr @llvm.riscv.vx.uniform.p0.p0(ptr) #3

; Function Attrs: nounwind memory(none)
declare i32 @llvm.riscv.vx.uniform.i32.i32(i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #4

attributes #0 = { convergent noinline optnone }
attributes #1 = { convergent "vortex-kernel" }
attributes #2 = { "vortex-kernel" }
attributes #3 = { nounwind memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
