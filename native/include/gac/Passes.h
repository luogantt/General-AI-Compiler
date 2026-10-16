#ifndef GAC_PASSES_H
#define GAC_PASSES_H

#include "gac/ConstantFolding.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace gac {
std::unique_ptr<mlir::Pass>
createShapeInferencePass(ConstantFoldBudget budget = {});
std::unique_ptr<mlir::Pass> createNormalizeProvenancePass();
std::unique_ptr<mlir::Pass> createGraphFusionPass();
std::unique_ptr<mlir::Pass> createLayoutPropagationPass();
std::unique_ptr<mlir::Pass> createLegalizeTensorOpsPass();
void registerTensorLegalizationPasses();
void registerGacPasses();
} // namespace gac
#endif
