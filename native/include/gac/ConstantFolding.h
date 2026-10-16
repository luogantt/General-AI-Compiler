#ifndef GAC_CONSTANT_FOLDING_H
#define GAC_CONSTANT_FOLDING_H

#include "mlir/IR/PatternMatch.h"
#include <cstdint>

namespace gac {
// Constant evaluation is compiler work, never model execution. These bounds
// prevent expanding learned weights or a constant neural network indefinitely.
struct ConstantFoldBudget {
  int64_t bytes = 256LL << 20;
  int64_t scalarOperations = 64LL << 20;
  int64_t maxElements = 4LL << 20;
  int64_t maxMatmulMultiplyAdds = 4LL << 20;
  int64_t maxMatmulReduction = 16;

  // Explicit opt-in for the official base model's 200x200 fixed camera
  // geometry. These remain per-pass finite limits, not an unrestricted
  // constant-network evaluator. The default profile is unchanged.
  static ConstantFoldBudget bevformerBase() {
    ConstantFoldBudget budget;
    budget.bytes = 2LL << 30;
    budget.scalarOperations = 512LL << 20;
    budget.maxElements = 16LL << 20;
    budget.maxMatmulMultiplyAdds = 32LL << 20;
    return budget;
  }
};

mlir::FailureOr<bool> foldConstantOperation(mlir::Operation *op,
                                            mlir::IRRewriter &rewriter,
                                            ConstantFoldBudget &budget);
} // namespace gac
#endif
