#ifndef GAC_FUSION_H
#define GAC_FUSION_H

#include "mlir/IR/PatternMatch.h"

namespace gac {
/// Add target-independent graph fusions. Each pattern proves its own tensor
/// domain, use-count and floating-point boundary constraints. Device template
/// selection happens later and must not weaken these semantic guarantees.
void populateGraphFusionPatterns(mlir::RewritePatternSet &patterns);
} // namespace gac

#endif
