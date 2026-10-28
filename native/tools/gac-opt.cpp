#include "gac/Dialect.h"
#include "gac/CudaLowering.h"
#include "gac/DeviceLowering.h"
#include "gac/Passes.h"
#include "gac/LinalgLowering.h"
#include "gac/VortexLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  gac::registerGacDialects(registry);
  gac::registerDeviceDialects(registry);
  gac::registerLinalgDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect, mlir::LLVM::LLVMDialect>();
  mlir::registerTransformsPasses();
  gac::registerGacPasses();
  gac::registerTensorLegalizationPasses();
  gac::registerVortexPasses();
  gac::registerDevicePasses();
  gac::registerCudaPasses();
  gac::registerLinalgPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "GAC native MLIR optimizer\n", registry));
}
