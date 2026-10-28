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

21:                                               ; preds = %91, %1
  %22 = phi i32 [ %92, %91 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 256
  br i1 %23, label %24, label %93

24:                                               ; preds = %21
  %25 = udiv i32 %22, 16
  %26 = urem i32 %22, 16
  %27 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %28 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %27)
  %29 = mul i32 %28, 16
  %30 = add i32 %25, %29
  %31 = mul i32 %30, 16
  %32 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %33 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %32)
  %34 = add i32 %26, 0
  %35 = add i32 %31, %34
  %36 = getelementptr float, ptr %17, i32 %35
  br label %37

37:                                               ; preds = %75, %24
  %38 = phi i32 [ %78, %75 ], [ 0, %24 ]
  %39 = phi float [ %77, %75 ], [ 0.000000e+00, %24 ]
  %40 = icmp slt i32 %38, 16
  br i1 %40, label %41, label %79

41:                                               ; preds = %37
  %42 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %43 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %42)
  %44 = mul i32 %43, 16
  %45 = add i32 %25, %44
  %46 = mul i32 %45, 16
  %47 = add i32 %46, %38
  %48 = getelementptr float, ptr %7, i32 %47
  %49 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %50 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %49)
  %51 = mul i32 %50, 16
  %52 = add i32 %25, %51
  %53 = icmp slt i32 %52, 16
  br i1 %53, label %54, label %56

54:                                               ; preds = %41
  %55 = load float, ptr %48, align 4
  br label %57

56:                                               ; preds = %41
  br label %57

57:                                               ; preds = %54, %56
  %58 = phi float [ 0.000000e+00, %56 ], [ %55, %54 ]
  br label %59

59:                                               ; preds = %57
  %60 = mul i32 %38, 16
  %61 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %62 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %61)
  %63 = add i32 %26, 0
  %64 = add i32 %60, %63
  %65 = getelementptr float, ptr %12, i32 %64
  %66 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %67 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %66)
  %68 = add i32 %26, 0
  %69 = icmp slt i32 %68, 16
  br i1 %69, label %70, label %72

70:                                               ; preds = %59
  %71 = load float, ptr %65, align 4
  br label %73

72:                                               ; preds = %59
  br label %73

73:                                               ; preds = %70, %72
  %74 = phi float [ 0.000000e+00, %72 ], [ %71, %70 ]
  br label %75

75:                                               ; preds = %73
  %76 = fmul float %58, %74
  %77 = fadd float %39, %76
  %78 = add i32 %38, 1
  br label %37

79:                                               ; preds = %37
  %80 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %81 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %80)
  %82 = mul i32 %81, 16
  %83 = add i32 %25, %82
  %84 = icmp slt i32 %83, 16
  %85 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %86 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %85)
  %87 = add i32 %26, 0
  %88 = icmp slt i32 %87, 16
  %89 = and i1 %84, %88
  br i1 %89, label %90, label %91

90:                                               ; preds = %79
  store float %39, ptr %36, align 4
  br label %91

91:                                               ; preds = %90, %79
  %92 = add i32 %22, %20
  br label %21

93:                                               ; preds = %21
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
