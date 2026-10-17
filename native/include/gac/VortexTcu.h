#ifndef GAC_VORTEX_TCU_H
#define GAC_VORTEX_TCU_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace gac {
struct DeviceOptions;
struct VortexMatmulPlan;

// Four-lane WMMA with FP32 input/output tensors and TF32x3 arithmetic:
// A ~= Ahi + Alo, B ~= Bhi + Blo, C += Ahi*Bhi + Ahi*Blo + Alo*Bhi.
// Each component is truncated toward zero to Vortex's packed 19-bit TF32;
// accumulation is FP32. Omitting Alo*Blo makes this an explicit approximate
// mode, NOT native FP32 multiplication. After all collective WMMA calls, any
// non-finite accumulator is recomputed from the original operands by device
// scalar FP32 instructions. This repairs exceptional split/accumulation cases;
// finite approximate results are not guaranteed bit-identical to scalar FP32.
// The default scalar schedule is unchanged.
// Register fragments are used, not LMEM/WGMMA or an emulated FPU reduction.
bool configureVortexTcuMatmul(const DeviceOptions &options,
                             VortexMatmulPlan &plan,
                             std::string &fallbackReason);
mlir::LogicalResult emitVortexTcuMatmul(
    mlir::OpBuilder &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> pointers, const VortexMatmulPlan &plan);
} // namespace gac
#endif
