#include "gac/CudaLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
namespace gac {
void registerCudaDialect(DialectRegistry &registry) {
  registry.insert<NVVM::NVVMDialect>();
}

void configureCudaModule(ModuleOp module) {
  OpBuilder b(module.getContext());
  module->setAttr("llvm.target_triple",
                  b.getStringAttr("nvptx64-nvidia-cuda"));
  module->setAttr("llvm.data_layout", b.getStringAttr(
      "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"));
}

void configureCudaKernel(LLVM::LLVMFuncOp kernel) {
  // The upstream NVVM translation emits LLVM PTX_Kernel calling convention.
  // Parameters are ordinary 64-bit device pointers in inputs/output order.
  kernel->setAttr(NVVM::NVVMDialect::getKernelFuncAttrName(),
                  UnitAttr::get(kernel.getContext()));
}

Value createCudaGlobalThreadId(OpBuilder &b, Location loc) {
  Value tid = b.create<NVVM::ThreadIdXOp>(loc, b.getI32Type());
  Value block = b.create<NVVM::BlockIdXOp>(loc, b.getI32Type());
  Value size = b.create<NVVM::BlockDimXOp>(loc, b.getI32Type());
  Value base = b.create<arith::MulIOp>(loc, block, size);
  return b.create<arith::AddIOp>(loc, base, tid);
}

} // namespace gac
