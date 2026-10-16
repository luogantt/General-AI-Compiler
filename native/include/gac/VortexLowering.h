#ifndef GAC_VORTEX_LOWERING_H
#define GAC_VORTEX_LOWERING_H

#include "gac/DeviceLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

namespace gac {
void registerVortexDialect(mlir::DialectRegistry &registry);
void registerVortexPasses();
std::unique_ptr<mlir::Pass> createLowerToVortexPass();
std::unique_ptr<mlir::Pass>
createLowerToVortexPass(DeviceOptions options, llvm::json::Object *manifest);
std::unique_ptr<mlir::Pass> createVortexToLLVMPass();
void configureVortexModule(mlir::ModuleOp module,
                           llvm::ArrayRef<std::string> kernelNames);
void configureVortexKernel(mlir::LLVM::LLVMFuncOp kernel);
llvm::SmallVector<mlir::Value>
unpackVortexKernelPointers(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value blob, unsigned count);
mlir::Value createVortexGlobalThreadId(mlir::OpBuilder &builder,
                                      mlir::Location location);
} // namespace gac
#endif
