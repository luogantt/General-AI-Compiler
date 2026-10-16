#include "gac/CudaLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/StringSwitch.h"

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

namespace {
struct LowerToCuda : PassWrapper<LowerToCuda, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToCuda)
  StringRef getArgument() const final { return "gac-lower-to-cuda"; }
  StringRef getDescription() const final {
    return "Lower shared GAC tensor semantics with CUDA NVVM and pointer ABI";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    createLowerToDevicePass({}, nullptr)->getDependentDialects(r);
  }
  void runOnOperation() override {
    llvm::json::Object plan;
    if (failed(lowerToDevice(getOperation(), {DeviceTarget::CUDA, ""}, plan)))
      signalPassFailure();
  }
};

struct CudaMathToLibdevice
    : PassWrapper<CudaMathToLibdevice, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaMathToLibdevice)
  StringRef getArgument() const final { return "gac-cuda-math-to-libdevice"; }
  StringRef getDescription() const final {
    return "Lower FP32 transcendental math to NVIDIA device-library calls";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<LLVM::LLVMDialect, math::MathDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> work;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "math")
        work.push_back(op);
    });
    for (Operation *op : work) {
      StringRef symbol = llvm::StringSwitch<StringRef>(
          op->getName().getStringRef())
          .Case("math.exp", "__nv_expf")
          .Case("math.log", "__nv_logf")
          .Case("math.sin", "__nv_sinf")
          .Case("math.cos", "__nv_cosf")
          .Case("math.sqrt", "__nv_sqrtf")
          .Case("math.powf", "__nv_powf")
          .Default("");
      if (symbol.empty() || op->getNumResults() != 1 ||
          !op->getResult(0).getType().isF32() ||
          !llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return type.isF32(); })) {
        op->emitError("CUDA device math requires an explicitly supported "
                      "FP32 libdevice lowering");
        return signalPassFailure();
      }
      OpBuilder b(op);
      auto functionType = LLVM::LLVMFunctionType::get(
          b.getF32Type(), SmallVector<Type>(op->getOperandTypes()));
      auto declaration = module.lookupSymbol<LLVM::LLVMFuncOp>(symbol);
      if (declaration && declaration.getFunctionType() != functionType) {
        op->emitError("conflicting libdevice declaration: ") << symbol;
        return signalPassFailure();
      }
      if (!declaration) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(module.getBody());
        declaration = b.create<LLVM::LLVMFuncOp>(op->getLoc(), symbol,
                                                functionType);
      }
      auto call = b.create<LLVM::CallOp>(op->getLoc(), declaration,
                                        op->getOperands());
      op->getResult(0).replaceAllUsesWith(call.getResult());
      op->erase();
    }
  }
};
} // namespace
std::unique_ptr<Pass> createCudaMathToLibdevicePass() {
  return std::make_unique<CudaMathToLibdevice>();
}
void registerCudaPasses() {
  PassRegistration<LowerToCuda>();
  PassRegistration<CudaMathToLibdevice>();
}
} // namespace gac
