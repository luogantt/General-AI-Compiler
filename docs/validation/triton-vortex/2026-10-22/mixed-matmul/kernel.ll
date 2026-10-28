; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@llvm.used = appending global [3 x ptr] [ptr @gac_op_1, ptr @gac_op_0, ptr @gac_op_2], section "llvm.metadata"
@llvm.compiler.used = appending global [3 x ptr] [ptr @gac_op_1, ptr @gac_op_0, ptr @gac_op_2], section "llvm.metadata"
@llvm.global.annotations = appending global [3 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @gac_op_1, ptr @__gac_annotation, ptr @__gac_source, i32 1, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @gac_op_0, ptr @__tv_annotation, ptr @__tv_source, i32 1, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @gac_op_2, ptr @__tv_annotation.7, ptr @__tv_source.8, i32 1, ptr null }], section "llvm.metadata"
@__gac_annotation = private constant [14 x i8] c"vortex.kernel\00", section "llvm.metadata"
@__gac_source = private constant [9 x i8] c"gac.mlir\00", section "llvm.metadata"
@__tv_annotation = private constant [14 x i8] c"vortex.kernel\00"
@__tv_source = private constant [7 x i8] c"triton\00"
@__tv_annotation.7 = private constant [14 x i8] c"vortex.kernel\00"
@__tv_source.8 = private constant [7 x i8] c"triton\00"

define void @gac_op_1(ptr %0) #0 section ".vx_entry.gac_op_1" {
  %2 = alloca float, align 4
  %3 = insertvalue { ptr, ptr, i32 } undef, ptr %2, 0
  %4 = insertvalue { ptr, ptr, i32 } %3, ptr %2, 1
  %5 = insertvalue { ptr, ptr, i32 } %4, i32 0, 2
  %6 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %7 = load i64, ptr %6, align 8
  %8 = trunc i64 %7 to i32
  %9 = inttoptr i32 %8 to ptr
  %10 = getelementptr i8, ptr %6, i32 8
  %11 = load i64, ptr %10, align 8
  %12 = trunc i64 %11 to i32
  %13 = inttoptr i32 %12 to ptr
  %14 = getelementptr i8, ptr %6, i32 16
  %15 = load i64, ptr %14, align 8
  %16 = trunc i64 %15 to i32
  %17 = inttoptr i32 %16 to ptr
  %18 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %20 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %19)
  %21 = call i32 asm "csrr $0, $1", "=r,i"(i32 3289)
  %22 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %21)
  %23 = mul i32 %20, %22
  %24 = add i32 %23, %18
  %25 = icmp slt i32 %24, 45
  %26 = icmp sge i32 %24, 0
  %27 = and i1 %26, %25
  br i1 %27, label %28, label %107

28:                                               ; preds = %1
  %29 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } undef, ptr %9, 0
  %30 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %29, ptr %9, 1
  %31 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %30, i32 0, 2
  %32 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %31, i32 5, 3, 0
  %33 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %32, i32 7, 4, 0
  %34 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %33, i32 7, 3, 1
  %35 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %34, i32 1, 4, 1
  %36 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } undef, ptr %13, 0
  %37 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %36, ptr %13, 1
  %38 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %37, i32 0, 2
  %39 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %38, i32 7, 3, 0
  %40 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %39, i32 9, 4, 0
  %41 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %40, i32 9, 3, 1
  %42 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %41, i32 1, 4, 1
  %43 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } undef, ptr %17, 0
  %44 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %43, ptr %17, 1
  %45 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %44, i32 0, 2
  %46 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %45, i32 5, 3, 0
  %47 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %46, i32 9, 4, 0
  %48 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %47, i32 9, 3, 1
  %49 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %48, i32 1, 4, 1
  %50 = extractvalue { ptr, ptr, i32 } %5, 0
  %51 = extractvalue { ptr, ptr, i32 } %5, 1
  %52 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } undef, ptr %50, 0
  %53 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %52, ptr %51, 1
  %54 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %53, i32 0, 2
  %55 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %54, i32 5, 3, 0
  %56 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %55, i32 0, 4, 0
  %57 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %56, i32 9, 3, 1
  %58 = insertvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %57, i32 0, 4, 1
  %59 = urem i32 %24, 9
  %60 = udiv i32 %24, 9
  %61 = urem i32 %60, 5
  %62 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %58, 1
  %63 = mul i32 %61, 0
  %64 = mul i32 %59, 0
  %65 = add i32 %63, %64
  %66 = getelementptr float, ptr %62, i32 %65
  store float 0.000000e+00, ptr %66, align 4
  %67 = urem i32 %24, 9
  %68 = udiv i32 %24, 9
  %69 = urem i32 %68, 5
  %70 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %58, 1
  %71 = mul i32 %69, 0
  %72 = mul i32 %67, 0
  %73 = add i32 %71, %72
  %74 = getelementptr float, ptr %70, i32 %73
  %75 = load float, ptr %74, align 4
  %76 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %49, 1
  %77 = mul i32 %69, 9
  %78 = add i32 %77, %67
  %79 = getelementptr float, ptr %76, i32 %78
  store float %75, ptr %79, align 4
  br label %80

80:                                               ; preds = %83, %28
  %81 = phi i32 [ %105, %83 ], [ 0, %28 ]
  %82 = icmp slt i32 %81, 7
  br i1 %82, label %83, label %106

83:                                               ; preds = %80
  %84 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %35, 1
  %85 = mul i32 %69, 7
  %86 = add i32 %85, %81
  %87 = getelementptr float, ptr %84, i32 %86
  %88 = load float, ptr %87, align 4
  %89 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %42, 1
  %90 = mul i32 %81, 9
  %91 = add i32 %90, %67
  %92 = getelementptr float, ptr %89, i32 %91
  %93 = load float, ptr %92, align 4
  %94 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %49, 1
  %95 = mul i32 %69, 9
  %96 = add i32 %95, %67
  %97 = getelementptr float, ptr %94, i32 %96
  %98 = load float, ptr %97, align 4
  %99 = fmul float %88, %93
  %100 = fadd float %98, %99
  %101 = extractvalue { ptr, ptr, i32, [2 x i32], [2 x i32] } %49, 1
  %102 = mul i32 %69, 9
  %103 = add i32 %102, %67
  %104 = getelementptr float, ptr %101, i32 %103
  store float %100, ptr %104, align 4
  %105 = add i32 %81, 1
  br label %80

106:                                              ; preds = %80
  br label %107

107:                                              ; preds = %106, %1
  ret void
}

; Function Attrs: nounwind memory(none)
declare ptr @llvm.riscv.vx.uniform.p0.p0(ptr) #1

; Function Attrs: nounwind memory(none)
declare i32 @llvm.riscv.vx.uniform.i32.i32(i32) #1

define void @gac_op_0(ptr %0) #0 section ".vx_entry.gac_op_0" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr i8, ptr %2, i32 0
  %4 = load i64, ptr %3, align 8
  %5 = trunc i64 %4 to i32
  %6 = inttoptr i32 %5 to ptr
  %7 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %6)
  %8 = getelementptr i8, ptr %2, i32 8
  %9 = load i64, ptr %8, align 8
  %10 = trunc i64 %9 to i32
  %11 = inttoptr i32 %10 to ptr
  %12 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %11)
  %13 = getelementptr i8, ptr %2, i32 16
  %14 = load i64, ptr %13, align 8
  %15 = trunc i64 %14 to i32
  %16 = inttoptr i32 %15 to ptr
  %17 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %16)
  %18 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3289)
  %20 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %19)
  br label %21

21:                                               ; preds = %47, %1
  %22 = phi i32 [ %48, %47 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 128
  br i1 %23, label %24, label %49

24:                                               ; preds = %21
  %25 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %26 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %25)
  %27 = mul i32 %26, 128
  %28 = add i32 %27, %22
  %29 = icmp slt i32 %28, 35
  %30 = getelementptr float, ptr %7, i32 %28
  br i1 %29, label %31, label %33

31:                                               ; preds = %24
  %32 = load float, ptr %30, align 4
  br label %34

33:                                               ; preds = %24
  br label %34

34:                                               ; preds = %33, %31
  %35 = phi float [ 0.000000e+00, %33 ], [ %32, %31 ]
  br label %36

36:                                               ; preds = %34
  %37 = getelementptr float, ptr %12, i32 %28
  br i1 %29, label %38, label %40

38:                                               ; preds = %36
  %39 = load float, ptr %37, align 4
  br label %41

40:                                               ; preds = %36
  br label %41

41:                                               ; preds = %40, %38
  %42 = phi float [ 0.000000e+00, %40 ], [ %39, %38 ]
  br label %43

43:                                               ; preds = %41
  %44 = fadd float %35, %42
  %45 = getelementptr float, ptr %17, i32 %28
  br i1 %29, label %46, label %47

46:                                               ; preds = %43
  store float %44, ptr %45, align 4
  br label %47

47:                                               ; preds = %46, %43
  %48 = add i32 %22, %20
  br label %21

49:                                               ; preds = %21
  ret void
}

define void @gac_op_2(ptr %0) #0 section ".vx_entry.gac_op_2" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr i8, ptr %2, i32 0
  %4 = load i64, ptr %3, align 8
  %5 = trunc i64 %4 to i32
  %6 = inttoptr i32 %5 to ptr
  %7 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %6)
  %8 = getelementptr i8, ptr %2, i32 8
  %9 = load i64, ptr %8, align 8
  %10 = trunc i64 %9 to i32
  %11 = inttoptr i32 %10 to ptr
  %12 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %11)
  %13 = getelementptr i8, ptr %2, i32 16
  %14 = load i64, ptr %13, align 8
  %15 = trunc i64 %14 to i32
  %16 = inttoptr i32 %15 to ptr
  %17 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %16)
  %18 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3289)
  %20 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %19)
  br label %21

21:                                               ; preds = %49, %1
  %22 = phi i32 [ %50, %49 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 128
  br i1 %23, label %24, label %51

24:                                               ; preds = %21
  %25 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %26 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %25)
  %27 = mul i32 %26, 128
  %28 = add i32 %27, %22
  %29 = icmp slt i32 %28, 45
  %30 = getelementptr float, ptr %7, i32 %28
  br i1 %29, label %31, label %33

31:                                               ; preds = %24
  %32 = load float, ptr %30, align 4
  br label %34

33:                                               ; preds = %24
  br label %34

34:                                               ; preds = %33, %31
  %35 = phi float [ 0.000000e+00, %33 ], [ %32, %31 ]
  br label %36

36:                                               ; preds = %34
  %37 = getelementptr float, ptr %12, i32 %28
  br i1 %29, label %38, label %40

38:                                               ; preds = %36
  %39 = load float, ptr %37, align 4
  br label %41

40:                                               ; preds = %36
  br label %41

41:                                               ; preds = %40, %38
  %42 = phi float [ 0.000000e+00, %40 ], [ %39, %38 ]
  br label %43

43:                                               ; preds = %41
  %44 = fmul float %35, %42
  %45 = fcmp olt float %44, 0.000000e+00
  %46 = select i1 %45, float 0.000000e+00, float %44
  %47 = getelementptr float, ptr %17, i32 %28
  br i1 %29, label %48, label %49

48:                                               ; preds = %43
  store float %46, ptr %47, align 4
  br label %49

49:                                               ; preds = %48, %43
  %50 = add i32 %22, %20
  br label %21

51:                                               ; preds = %21
  ret void
}

attributes #0 = { "vortex-kernel" }
attributes #1 = { nounwind memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
