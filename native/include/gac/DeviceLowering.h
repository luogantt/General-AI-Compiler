#ifndef GAC_DEVICE_LOWERING_H
#define GAC_DEVICE_LOWERING_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <memory>
#include <string>

namespace gac {
enum class DeviceTarget { Vortex, Native, CUDA };

struct DeviceOptions {
  // Native is an explicit numerical test target, never an execution fallback.
  DeviceTarget target = DeviceTarget::Vortex;
  std::string constantsDirectory;
  // Explicit target schedules keep the scalar baseline reproducible. Shapes
  // outside a template's contract retain scalar code with a manifest reason.
  std::string vortexMatmulSchedule = "scalar";
  int64_t tileM = 4, tileN = 4, tileK = 16;
  // LMEM/DXA output elements per thread; 1x1 preserves the original template.
  int64_t vortexThreadTileM = 1, vortexThreadTileN = 1;
  bool vortexTcuCombined = false;
  int64_t vortexLanes = 4;
  int64_t lmemBudget = 16384;
  std::string cudaMatmulSchedule = "scalar";
  int64_t cudaTileM = 32, cudaTileN = 32, cudaTileK = 16;
  int64_t cudaThreadTileM = 2, cudaThreadTileN = 2;
  std::string cudaReductionSchedule = "scalar";
  int64_t cudaConvChannelTile = 1;
};

// All targets share tensor semantics, static allocation and dispatch ordering.
// Only the kernel ABI, thread IDs, address width and device math differ.
void registerDeviceDialects(mlir::DialectRegistry &registry);
void registerDevicePasses();
std::unique_ptr<mlir::Pass>
createLowerToDevicePass(DeviceOptions options, llvm::json::Object *manifest);
mlir::LogicalResult lowerToDevice(mlir::ModuleOp module,
                                 const DeviceOptions &options,
                                 llvm::json::Object &manifest);
void addLLVMConversionPasses(mlir::PassManager &pm);
} // namespace gac
#endif
