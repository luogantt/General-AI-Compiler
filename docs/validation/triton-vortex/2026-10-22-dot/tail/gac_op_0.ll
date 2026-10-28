; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@__tv_annotation = private constant [14 x i8] c"vortex.kernel\00"
@__tv_source = private constant [7 x i8] c"triton\00"
@llvm.used = appending global [1 x ptr] [ptr @gac_op_0], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @gac_op_0], section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @gac_op_0, ptr @__tv_annotation, ptr @__tv_source, i32 1, ptr null }], section "llvm.metadata"

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

21:                                               ; preds = %103, %1
  %22 = phi i32 [ %104, %103 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 256
  br i1 %23, label %24, label %105

24:                                               ; preds = %21
  %25 = udiv i32 %22, 16
  %26 = urem i32 %22, 16
  %27 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %28 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %27)
  %29 = udiv i32 %28, 2
  %30 = mul i32 %29, 16
  %31 = add i32 %25, %30
  %32 = mul i32 %31, 19
  %33 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %34 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %33)
  %35 = urem i32 %34, 2
  %36 = mul i32 %35, 16
  %37 = add i32 %26, %36
  %38 = add i32 %32, %37
  %39 = getelementptr float, ptr %17, i32 %38
  br label %40

40:                                               ; preds = %84, %24
  %41 = phi i32 [ %87, %84 ], [ 0, %24 ]
  %42 = phi float [ %86, %84 ], [ 0.000000e+00, %24 ]
  %43 = icmp slt i32 %41, 32
  br i1 %43, label %44, label %88

44:                                               ; preds = %40
  %45 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %46 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %45)
  %47 = udiv i32 %46, 2
  %48 = mul i32 %47, 16
  %49 = add i32 %25, %48
  %50 = mul i32 %49, 32
  %51 = add i32 %50, %41
  %52 = getelementptr float, ptr %7, i32 %51
  %53 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %54 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %53)
  %55 = udiv i32 %54, 2
  %56 = mul i32 %55, 16
  %57 = add i32 %25, %56
  %58 = icmp slt i32 %57, 17
  br i1 %58, label %59, label %61

59:                                               ; preds = %44
  %60 = load float, ptr %52, align 4
  br label %62

61:                                               ; preds = %44
  br label %62

62:                                               ; preds = %59, %61
  %63 = phi float [ 0.000000e+00, %61 ], [ %60, %59 ]
  br label %64

64:                                               ; preds = %62
  %65 = mul i32 %41, 19
  %66 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %67 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %66)
  %68 = urem i32 %67, 2
  %69 = mul i32 %68, 16
  %70 = add i32 %26, %69
  %71 = add i32 %65, %70
  %72 = getelementptr float, ptr %12, i32 %71
  %73 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %74 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %73)
  %75 = urem i32 %74, 2
  %76 = mul i32 %75, 16
  %77 = add i32 %26, %76
  %78 = icmp slt i32 %77, 19
  br i1 %78, label %79, label %81

79:                                               ; preds = %64
  %80 = load float, ptr %72, align 4
  br label %82

81:                                               ; preds = %64
  br label %82

82:                                               ; preds = %79, %81
  %83 = phi float [ 0.000000e+00, %81 ], [ %80, %79 ]
  br label %84

84:                                               ; preds = %82
  %85 = fmul float %63, %83
  %86 = fadd float %42, %85
  %87 = add i32 %41, 1
  br label %40

88:                                               ; preds = %40
  %89 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %90 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %89)
  %91 = udiv i32 %90, 2
  %92 = mul i32 %91, 16
  %93 = add i32 %25, %92
  %94 = icmp slt i32 %93, 17
  %95 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %96 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %95)
  %97 = urem i32 %96, 2
  %98 = mul i32 %97, 16
  %99 = add i32 %26, %98
  %100 = icmp slt i32 %99, 19
  %101 = and i1 %94, %100
  br i1 %101, label %102, label %103

102:                                              ; preds = %88
  store float %42, ptr %39, align 4
  br label %103

103:                                              ; preds = %102, %88
  %104 = add i32 %22, %20
  br label %21

105:                                              ; preds = %21
  ret void
}

; Function Attrs: nounwind memory(none)
declare ptr @llvm.riscv.vx.uniform.p0.p0(ptr) #1

; Function Attrs: nounwind memory(none)
declare i32 @llvm.riscv.vx.uniform.i32.i32(i32) #1

attributes #0 = { "vortex-kernel" }
attributes #1 = { nounwind memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
