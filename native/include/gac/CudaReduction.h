#ifndef GAC_CUDA_REDUCTION_H
#define GAC_CUDA_REDUCTION_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>

namespace gac {
// One CTA cooperates on one reduction row. Scratch is a statically allocated
// shared-memory array, distinct from the global activation-buffer arena.
struct CudaReductionPlan {
  std::string schedule = "row-reduction";
  bool layerNorm = false;
  int64_t axis = 0, width = 0, rows = 0, stride = 1;
  int64_t blockSize = 0, gridSize = 0, sharedMemoryBytes = 0;
  double epsilon = 1e-5;
  llvm::SmallVector<int64_t> inputShape, scaleShape, biasShape;
};

// Called only when the cooperative schedule is selected. Unsupported types or
// contracts leave plan empty and give an explicit scalar-fallback reason.
mlir::LogicalResult planCudaReduction(
    mlir::Operation *op, std::optional<CudaReductionPlan> &plan,
    std::string &fallbackReason);

// Inputs in Operation order, followed by the output pointer. Do not wrap the
// body in a per-element bounds guard: every lane must execute every barrier.
// The caller appends the LLVM return. All computation is built as MLIR ops.
mlir::LogicalResult emitCudaReduction(
    mlir::OpBuilder &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> pointers, const CudaReductionPlan &plan);

llvm::json::Object cudaReductionMetadata(const CudaReductionPlan &plan);
} // namespace gac
#endif
