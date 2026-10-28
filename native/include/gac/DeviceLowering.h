#ifndef GAC_DEVICE_LOWERING_H
#define GAC_DEVICE_LOWERING_H

#include "gac/DeviceOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <memory>
#include <string>

namespace gac {
// All targets share tensor semantics, static allocation and dispatch ordering.
// Only the kernel ABI, thread IDs, address width and device math differ.
void registerDeviceDialects(mlir::DialectRegistry &registry);
void registerDevicePasses();
mlir::LogicalResult validateDeviceGraph(mlir::ModuleOp module);
std::unique_ptr<mlir::Pass>
createLowerToDevicePass(DeviceOptions options, llvm::json::Object *manifest);
mlir::LogicalResult lowerToDevice(mlir::ModuleOp module,
                                 const DeviceOptions &options,
                                 llvm::json::Object &manifest);
void addLLVMConversionPasses(mlir::PassManager &pm);
} // namespace gac
#endif
