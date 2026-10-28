#ifndef GAC_TRITON_BRIDGE_H
#define GAC_TRITON_BRIDGE_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir {
class Operation;
}

namespace gac {

// One Triton program computes one 16x16 output tile and one tt.dot reduces a
// power-of-two tile whose lanes beyond the static K are masked to zero.
// Batched MatMul flattens the output
// batch prefix into `batches`; A must carry that exact prefix, while B either
// carries the same prefix or is one rank-2 matrix shared by every batch.
// Singleton per-axis broadcasting and a shared rank-2 A are not supported.
struct TritonDotPlan {
  int64_t m = 0, n = 0, k = 0;
  int64_t batches = 1;
  int64_t blockM = 16, blockN = 16, blockK = 0;
  int64_t gridSize = 0;
  bool aBatched = false, bBatched = false;
  std::string batchMode;
  llvm::SmallVector<int64_t> aShape, bShape, cShape;
};

// A single Triton program reduces one positive, statically known FP32 vector.
// tile is the next power of two so tt.make_range can describe the logical
// tensor; elements in [width, tile) are masked and contribute exact zero.
struct TritonReductionPlan {
  int64_t width = 0;
  int64_t tile = 0;
  int64_t gridSize = 1;
};

// One Triton program computes a numerically stable Softmax over one logical
// row of a positive, statically known FP32 tensor. N remains a runtime i32
// kernel argument; program_id(0) selects the row and [width, tile) is masked.
struct TritonSoftmaxPlan {
  int64_t rows = 0;
  int64_t width = 0;
  int64_t tile = 0;
  int64_t gridSize = 0;
};

// One Triton program normalizes one logical row.  LayerNorm is represented by
// several Linalg operations, so selection also records the exact internal SSA
// subgraph that must disappear from the dispatch and Buffer plan atomically.
// input/scale/bias are the only input buffers exposed by the fused kernel.
struct TritonLayerNormPlan {
  int64_t rows = 0;
  int64_t width = 0;
  int64_t tile = 0;
  int64_t gridSize = 0;
  uint32_t epsilonF32Bits = 0;
  mlir::Value input;
  mlir::Value scale;
  mlir::Value bias;
  llvm::SmallVector<mlir::Operation *, 5> internalOperations;
};

// Match only the complete canonical ONNX LayerNormalization decomposition:
// zero+sum -> mean -> zero+squared-deviation-sum -> affine normalization.
// Every role, indexing map, scalar payload and SSA use must agree; partial
// matches stay on the native Vortex Linalg path.
bool selectTritonLayerNorm(mlir::Operation *op, TritonLayerNormPlan &plan,
                           std::string &reason);

// Emit one row-wise TTIR kernel with two real add reductions. Arguments are
// X, Scale, Bias, Y, runtime i32 N and runtime f32 epsilon, in that order.
mlir::LogicalResult emitTritonLayerNorm(
    mlir::Operation *op, llvm::StringRef kernelName,
    const TritonLayerNormPlan &plan, llvm::StringRef outputFile);

// Select only linalg.softmax over the last axis of a static FP32 tensor.
// Wider, non-last-axis and non-FP32 forms retain the native Vortex path.
bool selectTritonSoftmax(mlir::Operation *op, TritonSoftmaxPlan &plan,
                         std::string &reason);

// Emit the standard Triton 3.8 stable-Softmax graph: masked load with -Inf,
// maxnumf reduction, exp(x-max), addf reduction, division and masked store.
// Arguments are input pointer, output pointer and runtime i32 N.
mlir::LogicalResult emitTritonSoftmax(
    mlir::Operation *op, llvm::StringRef kernelName,
    const TritonSoftmaxPlan &plan, llvm::StringRef outputFile);

// Select only the canonical Linalg sum produced by ReduceSum over the sole
// axis of a rank-one tensor. Rank-zero and one-element tensor results share the
// same one-pointer output ABI. Every unsupported reduction stays native.
bool selectTritonReduction(mlir::Operation *op,
                           TritonReductionPlan &plan,
                           std::string &reason);

// Emit standard Triton 3.8 TTIR with a real tt.reduce region containing
// arith.addf + tt.reduce.return. Arguments are input then scalar output.
mlir::LogicalResult emitTritonReduction(
    mlir::Operation *op, llvm::StringRef kernelName,
    const TritonReductionPlan &plan, llvm::StringRef outputFile);

// Select canonical row-major FP32 Linalg MatMul. Rank-2 retains the legacy
// batch=1 contract. For rank>=3, A must have C's exact batch prefix; B may
// either have that same prefix or be one rank-2 matrix shared by every batch.
// A rejected candidate stays native.
bool selectTritonDot(mlir::Operation *op, TritonDotPlan &plan,
                     std::string &reason);

// Emit standard TTIR containing one real tt.dot. Kernel arguments are A, B,
// then output, matching the shared manifest/runtime pointer ABI.
mlir::LogicalResult emitTritonDot(mlir::Operation *op,
                                  llvm::StringRef kernelName,
                                  const TritonDotPlan &plan,
                                  llvm::StringRef outputFile);

// The first bridge supports only same-shape pointwise Linalg payloads. This
// includes their already-selected, single-use fused producers. A rejected
// candidate remains on the native device path; reason explains that decision.
bool selectTritonPointwise(mlir::Operation *op, std::string &reason);

// Emit one standard TTIR kernel, with deduplicated leaf-input pointers followed
// by the output pointer. Shapes are static, so no extra scalar argument is
// used. One Triton program computes block consecutive elements, with tail
// masking.
mlir::LogicalResult emitTritonPointwise(mlir::Operation *op,
                                        llvm::StringRef kernelName,
                                        int64_t block,
                                        llvm::StringRef outputFile);

} // namespace gac
#endif
