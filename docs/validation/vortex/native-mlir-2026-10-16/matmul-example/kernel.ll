; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@__gac_annotation = private constant [14 x i8] c"vortex.kernel\00", section "llvm.metadata"
@__gac_source = private constant [9 x i8] c"gac.mlir\00", section "llvm.metadata"
@llvm.used = appending global [1 x ptr] [ptr @gac_op_0], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @gac_op_0], section "llvm.metadata"
@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @gac_op_0, ptr @__gac_annotation, ptr @__gac_source, i32 1, ptr null }], section "llvm.metadata"

define void @gac_op_0(ptr %0) #0 section ".vx_entry.gac_op_0" {
  %2 = call ptr @llvm.riscv.vx.uniform.p0.p0(ptr %0)
  %3 = getelementptr i8, ptr %2, i32 0
  %4 = load i64, ptr %3, align 8
  %5 = trunc i64 %4 to i32
  %6 = inttoptr i32 %5 to ptr
  %7 = getelementptr i8, ptr %2, i32 8
  %8 = load i64, ptr %7, align 8
  %9 = trunc i64 %8 to i32
  %10 = inttoptr i32 %9 to ptr
  %11 = getelementptr i8, ptr %2, i32 16
  %12 = load i64, ptr %11, align 8
  %13 = trunc i64 %12 to i32
  %14 = inttoptr i32 %13 to ptr
  %15 = getelementptr i8, ptr %2, i32 24
  %16 = load i64, ptr %15, align 8
  %17 = trunc i64 %16 to i32
  %18 = inttoptr i32 %17 to ptr
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %20 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %21 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %20)
  %22 = call i32 asm "csrr $0, $1", "=r,i"(i32 3289)
  %23 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %22)
  %24 = mul i32 %21, %23
  %25 = add i32 %24, %19
  %26 = icmp slt i32 %25, 8
  %27 = icmp sge i32 %25, 0
  %28 = and i1 %27, %26
  br i1 %28, label %29, label %62

29:                                               ; preds = %1
  %30 = sdiv i32 %25, 4
  %31 = srem i32 %30, 2
  %32 = srem i32 %25, 4
  br label %33

33:                                               ; preds = %37, %29
  %34 = phi i32 [ %50, %37 ], [ 0, %29 ]
  %35 = phi float [ %49, %37 ], [ 0.000000e+00, %29 ]
  %36 = icmp slt i32 %34, 3
  br i1 %36, label %37, label %51

37:                                               ; preds = %33
  %38 = add i32 0, %34
  %39 = mul i32 %38, 4
  %40 = add i32 %39, %32
  %41 = getelementptr float, ptr %10, i32 %40
  %42 = load float, ptr %41, align 4
  %43 = add i32 0, %31
  %44 = mul i32 %43, 3
  %45 = add i32 %44, %34
  %46 = getelementptr float, ptr %6, i32 %45
  %47 = load float, ptr %46, align 4
  %48 = fmul float %47, %42
  %49 = fadd float %35, %48
  %50 = add i32 %34, 1
  br label %33

51:                                               ; preds = %33
  %52 = sdiv i32 %25, 4
  %53 = srem i32 %52, 2
  %54 = srem i32 %25, 4
  %55 = add i32 0, %54
  %56 = getelementptr float, ptr %14, i32 %55
  %57 = load float, ptr %56, align 4
  %58 = fadd float %35, %57
  %59 = fcmp olt float %58, 0.000000e+00
  %60 = select i1 %59, float 0.000000e+00, float %58
  %61 = getelementptr float, ptr %18, i32 %25
  store float %60, ptr %61, align 4
  br label %62

62:                                               ; preds = %51, %1
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
