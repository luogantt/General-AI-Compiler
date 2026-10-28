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
  %13 = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %14 = icmp eq i32 %13, 0
  %15 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %16 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %15)
  %17 = icmp eq i32 %16, 0
  %18 = and i1 %14, %17
  %19 = call i32 asm "csrr $0, $1", "=r,i"(i32 3287)
  %20 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %19)
  %21 = icmp eq i32 %20, 0
  %22 = and i1 %18, %21
  %23 = call i32 asm "csrr $0, $1", "=r,i"(i32 3288)
  %24 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %23)
  %25 = icmp eq i32 %24, 0
  %26 = and i1 %22, %25
  br i1 %26, label %27, label %44

27:                                               ; preds = %1
  br label %28

28:                                               ; preds = %40, %27
  %29 = phi i32 [ %42, %40 ], [ 0, %27 ]
  %30 = phi float [ %41, %40 ], [ 0.000000e+00, %27 ]
  %31 = icmp slt i32 %29, 64
  br i1 %31, label %32, label %43

32:                                               ; preds = %28
  %33 = getelementptr float, ptr %7, i32 %29
  %34 = icmp slt i32 %29, 37
  br i1 %34, label %35, label %37

35:                                               ; preds = %32
  %36 = load float, ptr %33, align 4
  br label %38

37:                                               ; preds = %32
  br label %38

38:                                               ; preds = %35, %37
  %39 = phi float [ 0.000000e+00, %37 ], [ %36, %35 ]
  br label %40

40:                                               ; preds = %38
  %41 = fadd float %30, %39
  %42 = add i32 %29, 1
  br label %28

43:                                               ; preds = %28
  store float %30, ptr %12, align 4
  br label %44

44:                                               ; preds = %43, %1
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
