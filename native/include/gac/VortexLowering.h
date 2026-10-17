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
mlir::Value createVortexLocalThreadId(mlir::OpBuilder &, mlir::Location);
mlir::Value createVortexBlockId(mlir::OpBuilder &, mlir::Location);
mlir::Value createVortexCTAId(mlir::OpBuilder &, mlir::Location);
mlir::Value createVortexWarpRank(mlir::OpBuilder &, mlir::Location);
mlir::Value createVortexWarpCount(mlir::OpBuilder &, mlir::Location);
mlir::Value createVortexLMEMAddress(mlir::OpBuilder &, mlir::Location);
// Collective calls carry convergent semantics and a memory clobber. The
// noinline wrappers also prevent LLVM from speculating their inline assembly.
// Pass the already attached module explicitly: SCF builder callbacks may be
// constructing a region whose parent operation has not been inserted yet.
void createVortexBarrier(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                         mlir::Value barrier, mlir::Value warps);
void createVortexDxaExpectTwo(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                              mlir::Value barrier);
void createVortexDxaIssue2D(mlir::OpBuilder &, mlir::Location, mlir::ModuleOp,
                           mlir::Value destination, mlir::Value barrier,
                           mlir::Value descriptor, mlir::Value coord0,
                           mlir::Value coord1);
} // namespace gac
#endif
