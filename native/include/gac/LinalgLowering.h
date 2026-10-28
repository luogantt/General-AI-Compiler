#ifndef GAC_LINALG_LOWERING_H
#define GAC_LINALG_LOWERING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace gac {
/// Replace regular graph operators with structured tensor computations. The
/// resulting Linalg operations, including their indexing maps and regions, are
/// the executable source of truth for downstream scheduling and code generation.
/// Irregular operators keep their GAC semantics and dedicated device lowering.
mlir::LogicalResult lowerRegularToLinalg(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass> createLowerRegularToLinalgPass();
void registerLinalgPasses();
void registerLinalgDialects(mlir::DialectRegistry &registry);
} // namespace gac
#endif
