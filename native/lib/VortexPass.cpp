// Compiler pass wrappers are separate from the target support library.
// Hardware ABI/CSR helpers can be used by compute templates without a
// reverse dependency on the graph compiler and its lowerToDevice pipeline.
#include "gac/VortexLowering.h"
#include "gac/DeviceLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
namespace gac {
namespace {
struct VortexToLLVM : PassWrapper<VortexToLLVM, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VortexToLLVM)
  StringRef getArgument() const final { return "gac-vortex-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower registered Vortex launch operations to LLVM intrinsics and "
           "CSRs";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<LLVM::LLVMDialect, arith::ArithDialect>();
    registerVortexDialect(r);
  }
  void runOnOperation() override {
    SmallVector<Operation *> work;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "vortex")
        work.push_back(op);
    });
    for (Operation *op : work) {
      OpBuilder b(op);
      Value result;
      StringRef name = op->getName().getStringRef();
      if (name == "vortex.uniform")
        result =
            b.create<LLVM::CallIntrinsicOp>(
                 op->getLoc(), op->getResult(0).getType(),
                 b.getStringAttr("llvm.riscv.vx.uniform"), op->getOperands())
                .getResult(0);
      else {
        int64_t csr = name == "vortex.thread_id"   ? 0xCD3
                      : name == "vortex.block_id"  ? 0xCD6
                      : name == "vortex.block_dim" ? 0xCD9
                      : name == "vortex.cta_id" ? 0xCD0
                      : name == "vortex.warp_rank" ? 0xCD1
                      : name == "vortex.warp_count" ? 0xCD2
                      : name == "vortex.lmem_address" ? 0xCDF
                                                   : 0;
        if (!csr) {
          op->emitError("unimplemented Vortex operation");
          signalPassFailure();
          return;
        }
        Value number = b.create<arith::ConstantIntOp>(op->getLoc(), csr, 32);
        OperationState state(op->getLoc(), "llvm.inline_asm");
        state.addTypes(b.getI32Type());
        state.addOperands(number);
        state.addAttribute("asm_string", b.getStringAttr("csrr $0, $1"));
        state.addAttribute("constraints", b.getStringAttr("=r,i"));
        result = b.create(state)->getResult(0);
        if (name != "vortex.thread_id")
          result =
              b.create<LLVM::CallIntrinsicOp>(
                   op->getLoc(), b.getI32Type(),
                   b.getStringAttr("llvm.riscv.vx.uniform"), ValueRange{result})
                  .getResult(0);
      }
      op->getResult(0).replaceAllUsesWith(result);
      op->erase();
    }
  }
};
} // namespace
namespace {
struct LowerToVortex : PassWrapper<LowerToVortex, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToVortex)
  StringRef getArgument() const final { return "gac-lower-to-vortex"; }
  StringRef getDescription() const final {
    return "Lower the shared GAC device semantics with the Vortex launch ABI";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    createLowerToDevicePass({}, nullptr)->getDependentDialects(r);
  }
  void runOnOperation() override {
    llvm::json::Object plan;
    if (failed(lowerToDevice(getOperation(), {}, plan)))
      signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> createLowerToVortexPass() {
  return std::make_unique<LowerToVortex>();
}
std::unique_ptr<Pass> createLowerToVortexPass(DeviceOptions options,
                                            llvm::json::Object *manifest) {
  return createLowerToDevicePass(std::move(options), manifest);
}
std::unique_ptr<Pass> createVortexToLLVMPass() {
  return std::make_unique<VortexToLLVM>();
}
void registerVortexPasses() {
  PassRegistration<LowerToVortex>();
  PassRegistration<VortexToLLVM>();
}
} // namespace gac
