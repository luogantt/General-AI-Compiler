#ifndef GAC_VORTEX_LOWERING_H
#define GAC_VORTEX_LOWERING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <string>

namespace gac {
struct DeviceOptions {
  // native is an explicit numerical test target, never an execution fallback.
  bool native = false;
  std::string constantsDirectory;
};
void registerVortexDialect(mlir::DialectRegistry &registry);
void registerVortexPasses();
std::unique_ptr<mlir::Pass> createLowerToVortexPass();
std::unique_ptr<mlir::Pass>
createLowerToVortexPass(DeviceOptions options, llvm::json::Object *manifest);
std::unique_ptr<mlir::Pass> createVortexToLLVMPass();
// Input is the optimized tensor ModuleOp. Output contains genuine registered
// SCF/Arith/Math/LLVM/Vortex operations, plus an SSA-derived dispatch plan.
mlir::LogicalResult lowerToDevice(mlir::ModuleOp module,
                                  const DeviceOptions &options,
                                  llvm::json::Object &manifest);
void addLLVMConversionPasses(mlir::PassManager &pm);
} // namespace gac
#endif
