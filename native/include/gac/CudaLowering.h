#ifndef GAC_CUDA_LOWERING_H
#define GAC_CUDA_LOWERING_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

namespace gac {
void registerCudaDialect(mlir::DialectRegistry &registry);
void registerCudaPasses();
void configureCudaModule(mlir::ModuleOp module);
void configureCudaKernel(mlir::LLVM::LLVMFuncOp kernel);
mlir::Value createCudaGlobalThreadId(mlir::OpBuilder &builder,
                                     mlir::Location location);
// CUDA uses device libdevice calls, never host libm or a host fallback.
std::unique_ptr<mlir::Pass> createCudaMathToLibdevicePass();
} // namespace gac
#endif
