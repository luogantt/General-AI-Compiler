#ifndef GAC_CUDA_MATMUL_H
#define GAC_CUDA_MATMUL_H

#include "gac/DeviceOptions.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace gac {
// One CTA owns a BM x BN output tile; each thread carries RM x RN FP32
// accumulators. A/B are cooperatively staged in static CUDA shared memory.
struct CudaMatmulPlan {
  std::string schedule = "shared-tiled";
  int64_t m = 0, n = 0, k = 0, batches = 1;
  int64_t tileM = 0, tileN = 0, tileK = 0;
  int64_t threadTileM = 0, threadTileN = 0;
  int64_t blockSize = 0, gridSize = 0, sharedMemoryBytes = 0;
  llvm::SmallVector<int64_t> aShape, bShape, outShape, biasShape;
  bool flattenedProjection = false, biasFirst = false, relu = false;
};
mlir::LogicalResult planCudaMatmul(
    mlir::Operation *op, const DeviceOptions &options,
    std::optional<CudaMatmulPlan> &plan, std::string &fallbackReason);
// Every CTA thread enters this body, including M/N tail threads. Only loads
// and final stores are predicated; neither CTA barrier is inside that guard.
mlir::LogicalResult emitCudaMatmul(
    mlir::OpBuilder &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> pointers, const CudaMatmulPlan &plan);
} // namespace gac
#endif
