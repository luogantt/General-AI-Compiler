#ifndef GAC_TEMPLATES_CUDA_CONVOLUTION_H
#define GAC_TEMPLATES_CUDA_CONVOLUTION_H

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <optional>
#include <string>

namespace gac::templates {

// One thread computes consecutive output channels at the same (N, H, W).
// Inputs and weights stay in global memory; the accumulators are SSA values
// held in registers. This template does not allocate shared-memory buffers.
struct CudaConvolutionPlan {
  std::string schedule = "channel-tiled";
  std::string kind;
  int64_t channelTile = 1;
  int64_t groups = 1;
  int64_t channelsPerGroup = 0;
  int64_t logicalThreads = 0;
  int64_t blockSize = 256;
  int64_t gridSize = 0;
  int64_t sharedMemoryBytes = 0;
  llvm::SmallVector<llvm::SmallVector<int64_t>> inputShapes;
  llvm::SmallVector<int64_t> outputShape;
};

// Success with no plan means the scalar fallback remains applicable. A reason
// is provided for a requested convolution template that cannot be selected;
// unrelated operators and channelTile=1 leave the reason empty.
mlir::LogicalResult planCudaConvolution(
    mlir::Operation *op, int64_t requestedChannelTile,
    std::optional<CudaConvolutionPlan> &plan, std::string &fallbackReason);

// pointers are in operation-input order followed by the output pointer.
mlir::LogicalResult emitCudaConvolution(
    mlir::OpBuilder &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> pointers, const CudaConvolutionPlan &plan);

llvm::json::Object cudaConvolutionMetadata(const CudaConvolutionPlan &plan);

namespace detail {
struct Scalar;

// Shared numerical implementation for this template and every target's
// channelTile=1 scalar fallback. Keep the reduction, masking and rounding
// order identical across both execution strategies.
llvm::SmallVector<mlir::Value> emitConvolutionOutputs(
    Scalar &builder, mlir::Operation *op,
    llvm::ArrayRef<mlir::Value> inputPointers,
    llvm::ArrayRef<mlir::Value> outputCoordinates, int64_t channelTile);
} // namespace detail
} // namespace gac::templates
#endif
