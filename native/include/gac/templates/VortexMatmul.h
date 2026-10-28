#ifndef GAC_VORTEX_MATMUL_H
#define GAC_VORTEX_MATMUL_H

#include "gac/DeviceOptions.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace gac {
// A checked compute template, shared by code generation and launch metadata.
// LMEM holds A/B tiles; each output's running sum stays in a register.
struct VortexMatmulPlan {
  std::string schedule;
  int64_t m = 0, n = 0, k = 0, batches = 1;
  int64_t tileM = 0, tileN = 0, tileK = 0;
  int64_t threadTileM = 1, threadTileN = 1;
  int64_t blockSize = 0, gridSize = 0, lmemBytes = 0, stages = 1, lanes = 4;
  llvm::SmallVector<int64_t> aShape, bShape, outShape, biasShape;
  bool flattenedProjection = false, biasFirst = false, relu = false;
  bool tcuCombined = false;
};

// Unsupported contracts return success with no plan and an explicit reason;
// malformed target options are diagnosed as failure. Scalar remains valid.
mlir::LogicalResult planVortexMatmul(
    mlir::Operation *op, const DeviceOptions &options,
    std::optional<VortexMatmulPlan> &plan, std::string &fallbackReason);

// Pointers are inputs in operation order, followed by output. The caller must
// not add a per-output bounds guard: all CTA lanes participate in barriers.
// This emits the body only; the caller appends the LLVM return.
mlir::LogicalResult emitVortexMatmul(
    mlir::OpBuilder &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> pointers, const VortexMatmulPlan &plan);
} // namespace gac
#endif
