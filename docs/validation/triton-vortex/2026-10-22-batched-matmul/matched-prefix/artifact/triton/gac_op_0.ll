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

21:                                               ; preds = %126, %1
  %22 = phi i32 [ %127, %126 ], [ %18, %1 ]
  %23 = icmp slt i32 %22, 256
  br i1 %23, label %24, label %128

24:                                               ; preds = %21
  %25 = udiv i32 %22, 16
  %26 = urem i32 %22, 16
  %27 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %28 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %27)
  %29 = urem i32 %28, 4
  %30 = udiv i32 %29, 2
  %31 = mul i32 %30, 16
  %32 = add i32 %25, %31
  %33 = mul i32 %32, 19
  %34 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %35 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %34)
  %36 = urem i32 %35, 4
  %37 = urem i32 %36, 2
  %38 = mul i32 %37, 16
  %39 = add i32 %26, %38
  %40 = add i32 %33, %39
  %41 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %42 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %41)
  %43 = udiv i32 %42, 4
  %44 = mul i32 %43, 323
  %45 = add i32 %40, %44
  %46 = getelementptr float, ptr %17, i32 %45
  br label %47

47:                                               ; preds = %105, %24
  %48 = phi i32 [ %108, %105 ], [ 0, %24 ]
  %49 = phi float [ %107, %105 ], [ 0.000000e+00, %24 ]
  %50 = icmp slt i32 %48, 32
  br i1 %50, label %51, label %109

51:                                               ; preds = %47
  %52 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %53 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %52)
  %54 = urem i32 %53, 4
  %55 = udiv i32 %54, 2
  %56 = mul i32 %55, 16
  %57 = add i32 %25, %56
  %58 = mul i32 %57, 32
  %59 = add i32 %58, %48
  %60 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %61 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %60)
  %62 = udiv i32 %61, 4
  %63 = mul i32 %62, 544
  %64 = add i32 %59, %63
  %65 = getelementptr float, ptr %7, i32 %64
  %66 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %67 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %66)
  %68 = urem i32 %67, 4
  %69 = udiv i32 %68, 2
  %70 = mul i32 %69, 16
  %71 = add i32 %25, %70
  %72 = icmp slt i32 %71, 17
  br i1 %72, label %73, label %75

73:                                               ; preds = %51
  %74 = load float, ptr %65, align 4
  br label %76

75:                                               ; preds = %51
  br label %76

76:                                               ; preds = %73, %75
  %77 = phi float [ 0.000000e+00, %75 ], [ %74, %73 ]
  br label %78

78:                                               ; preds = %76
  %79 = mul i32 %48, 19
  %80 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %81 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %80)
  %82 = urem i32 %81, 4
  %83 = urem i32 %82, 2
  %84 = mul i32 %83, 16
  %85 = add i32 %26, %84
  %86 = add i32 %79, %85
  %87 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %88 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %87)
  %89 = udiv i32 %88, 4
  %90 = mul i32 %89, 608
  %91 = add i32 %86, %90
  %92 = getelementptr float, ptr %12, i32 %91
  %93 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %94 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %93)
  %95 = urem i32 %94, 4
  %96 = urem i32 %95, 2
  %97 = mul i32 %96, 16
  %98 = add i32 %26, %97
  %99 = icmp slt i32 %98, 19
  br i1 %99, label %100, label %102

100:                                              ; preds = %78
  %101 = load float, ptr %92, align 4
  br label %103

102:                                              ; preds = %78
  br label %103

103:                                              ; preds = %100, %102
  %104 = phi float [ 0.000000e+00, %102 ], [ %101, %100 ]
  br label %105

105:                                              ; preds = %103
  %106 = fmul float %77, %104
  %107 = fadd float %49, %106
  %108 = add i32 %48, 1
  br label %47

109:                                              ; preds = %47
  %110 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %111 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %110)
  %112 = urem i32 %111, 4
  %113 = udiv i32 %112, 2
  %114 = mul i32 %113, 16
  %115 = add i32 %25, %114
  %116 = icmp slt i32 %115, 17
  %117 = call i32 asm "csrr $0, $1", "=r,i"(i32 3286)
  %118 = call i32 @llvm.riscv.vx.uniform.i32.i32(i32 %117)
  %119 = urem i32 %118, 4
  %120 = urem i32 %119, 2
  %121 = mul i32 %120, 16
  %122 = add i32 %26, %121
  %123 = icmp slt i32 %122, 19
  %124 = and i1 %116, %123
  br i1 %124, label %125, label %126

125:                                              ; preds = %109
  store float %49, ptr %46, align 4
  br label %126

126:                                              ; preds = %125, %109
  %127 = add i32 %22, %20
  br label %21

128:                                              ; preds = %21
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
