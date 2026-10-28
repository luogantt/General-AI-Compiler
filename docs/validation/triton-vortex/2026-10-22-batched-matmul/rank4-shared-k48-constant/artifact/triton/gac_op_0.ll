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

21:                                               ; preds = %99, %1
  %22 = phi i32 [ %100, %99 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 256
  br i1 %23, label %24, label %101

24:                                               ; preds = %21
  %25 = udiv i32 %22, 16
  %26 = urem i32 %22, 16
  %27 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %28 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %27)
  %29 = add i32 %25, 0
  %30 = mul i32 %29, 7
  %31 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %32 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %31)
  %33 = add i32 %26, 0
  %34 = add i32 %30, %33
  %35 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %36 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %35)
  %37 = mul i32 %36, 35
  %38 = add i32 %34, %37
  %39 = getelementptr float, ptr %17, i32 %38
  br label %40

40:                                               ; preds = %84, %24
  %41 = phi i32 [ %87, %84 ], [ 0, %24 ]
  %42 = phi float [ %86, %84 ], [ 0.000000e+00, %24 ]
  %43 = icmp slt i32 %41, 64
  br i1 %43, label %44, label %88

44:                                               ; preds = %40
  %45 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %46 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %45)
  %47 = add i32 %25, 0
  %48 = mul i32 %47, 48
  %49 = add i32 %48, %41
  %50 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %51 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %50)
  %52 = mul i32 %51, 240
  %53 = add i32 %49, %52
  %54 = getelementptr float, ptr %7, i32 %53
  %55 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %56 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %55)
  %57 = add i32 %25, 0
  %58 = icmp slt i32 %57, 5
  %59 = icmp slt i32 %41, 48
  %60 = and i1 %58, %59
  br i1 %60, label %61, label %63

61:                                               ; preds = %44
  %62 = load float, ptr %54, align 4
  br label %64

63:                                               ; preds = %44
  br label %64

64:                                               ; preds = %61, %63
  %65 = phi float [ 0.000000e+00, %63 ], [ %62, %61 ]
  br label %66

66:                                               ; preds = %64
  %67 = mul i32 %41, 7
  %68 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %69 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %68)
  %70 = add i32 %26, 0
  %71 = add i32 %67, %70
  %72 = getelementptr float, ptr %12, i32 %71
  %73 = icmp slt i32 %41, 48
  %74 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %75 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %74)
  %76 = add i32 %26, 0
  %77 = icmp slt i32 %76, 7
  %78 = and i1 %73, %77
  br i1 %78, label %79, label %81

79:                                               ; preds = %66
  %80 = load float, ptr %72, align 4
  br label %82

81:                                               ; preds = %66
  br label %82

82:                                               ; preds = %79, %81
  %83 = phi float [ 0.000000e+00, %81 ], [ %80, %79 ]
  br label %84

84:                                               ; preds = %82
  %85 = fmul float %65, %83
  %86 = fadd float %42, %85
  %87 = add i32 %41, 1
  br label %40

88:                                               ; preds = %40
  %89 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %90 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %89)
  %91 = add i32 %25, 0
  %92 = icmp slt i32 %91, 5
  %93 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %94 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %93)
  %95 = add i32 %26, 0
  %96 = icmp slt i32 %95, 7
  %97 = and i1 %92, %96
  br i1 %97, label %98, label %99

98:                                               ; preds = %88
  store float %42, ptr %39, align 4
  br label %99

99:                                               ; preds = %98, %88
  %100 = add i32 %22, %20
  br label %21

101:                                              ; preds = %21
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
