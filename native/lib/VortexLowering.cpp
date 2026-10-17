#include "gac/VortexLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
namespace gac {
namespace {
// Only target-specific operations belong to this dialect. Tensor algorithms
// remain ordinary Arith/Math/SCF operations, reusable by another target.
template <typename T>
class TargetValueOp
    : public Op<T, OpTrait::VariadicOperands, OpTrait::OneResult,
                OpTrait::ZeroRegions, MemoryEffectOpInterface::Trait> {
public:
  using Op<T, OpTrait::VariadicOperands, OpTrait::OneResult,
           OpTrait::ZeroRegions, MemoryEffectOpInterface::Trait>::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &) {}
  LogicalResult verify() {
    Operation *op = this->getOperation();
    bool uniform = op->getName().getStringRef() == "vortex.uniform";
    if (uniform) {
      if (op->getNumOperands() != 1 ||
          op->getOperand(0).getType() != op->getResult(0).getType())
        return op->emitOpError(
            "requires one operand and a result of the same type");
    } else if (op->getNumOperands() ||
               !op->getResult(0).getType().isInteger(32))
      return op->emitOpError("requires no operands and one i32 result");
    return success();
  }
};
#define GAC_TARGET_OP(CLASS, NAME)                                             \
  class CLASS : public TargetValueOp<CLASS> {                                  \
  public:                                                                      \
    using TargetValueOp<CLASS>::TargetValueOp;                                 \
    static StringRef getOperationName() { return NAME; }                       \
  };
GAC_TARGET_OP(ThreadIdOp, "vortex.thread_id")
GAC_TARGET_OP(BlockIdOp, "vortex.block_id")
GAC_TARGET_OP(BlockDimOp, "vortex.block_dim")
GAC_TARGET_OP(CTAIdOp, "vortex.cta_id")
GAC_TARGET_OP(WarpRankOp, "vortex.warp_rank")
GAC_TARGET_OP(WarpCountOp, "vortex.warp_count")
GAC_TARGET_OP(LMEMAddressOp, "vortex.lmem_address")
GAC_TARGET_OP(UniformOp, "vortex.uniform")
#undef GAC_TARGET_OP
class VortexDialect : public Dialect {
public:
  explicit VortexDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<VortexDialect>()) {
    addOperations<ThreadIdOp, BlockIdOp, BlockDimOp, CTAIdOp, WarpRankOp,
                  WarpCountOp, LMEMAddressOp, UniformOp>();
  }
  static StringRef getDialectNamespace() { return "vortex"; }
};

static Value targetValue(OpBuilder &b, Location loc, StringRef name, Type type,
                         ValueRange inputs = {}) {
  OperationState state(loc, name);
  state.addTypes(type);
  state.addOperands(inputs);
  return b.create(state)->getResult(0);
}

// Globals mirror Clang's kernel ABI metadata. The VOLT backend consumes the
// annotation to create __vx_kentry_*; the attribute alone is not sufficient.
static void kernelAnnotations(ModuleOp module, ArrayRef<std::string> names) {
  if (names.empty())
    return;
  OpBuilder b(module.getContext());
  b.setInsertionPointToStart(module.getBody());
  Location loc = module.getLoc();
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  auto text = [&](StringRef name, StringRef value) {
    auto g = b.create<LLVM::GlobalOp>(
        loc, LLVM::LLVMArrayType::get(b.getI8Type(), value.size()), true,
        LLVM::Linkage::Private, name, b.getStringAttr(value));
    g->setAttr("section", b.getStringAttr("llvm.metadata"));
  };
  text("__gac_annotation", StringRef("vortex.kernel\0", 14));
  text("__gac_source", StringRef("gac.mlir\0", 9));
  auto retained = [&](StringRef name) {
    auto type = LLVM::LLVMArrayType::get(ptr, names.size());
    auto g = b.create<LLVM::GlobalOp>(
        loc, type, false, LLVM::Linkage::Appending, name, Attribute());
    g->setAttr("section", b.getStringAttr("llvm.metadata"));
    OpBuilder::InsertionGuard guard(b);
    Block *block = new Block();
    g.getInitializerRegion().push_back(block);
    b.setInsertionPointToStart(block);
    Value list = b.create<LLVM::UndefOp>(loc, type);
    for (auto [idx, n] : llvm::enumerate(names)) {
      Value address = b.create<LLVM::AddressOfOp>(loc, ptr, n);
      list = b.create<LLVM::InsertValueOp>(loc, list, address,
                                           ArrayRef<int64_t>{int64_t(idx)});
    }
    b.create<LLVM::ReturnOp>(loc, ValueRange{list});
  };
  retained("llvm.used");
  retained("llvm.compiler.used");
  auto rec = LLVM::LLVMStructType::getLiteral(
      b.getContext(), {ptr, ptr, ptr, b.getI32Type(), ptr});
  auto array = LLVM::LLVMArrayType::get(rec, names.size());
  auto global =
      b.create<LLVM::GlobalOp>(loc, array, false, LLVM::Linkage::Appending,
                               "llvm.global.annotations", Attribute());
  global->setAttr("section", b.getStringAttr("llvm.metadata"));
  Block *body = new Block();
  global.getInitializerRegion().push_back(body);
  b.setInsertionPointToStart(body);
  Value list = b.create<LLVM::UndefOp>(loc, array),
        annotation = b.create<LLVM::AddressOfOp>(loc, ptr, "__gac_annotation"),
        source = b.create<LLVM::AddressOfOp>(loc, ptr, "__gac_source");
  Value one = b.create<LLVM::ConstantOp>(loc, b.getI32Type(),
                                         b.getI32IntegerAttr(1)),
        null = b.create<LLVM::ZeroOp>(loc, ptr);
  for (auto [idx, name] : llvm::enumerate(names)) {
    Value record = b.create<LLVM::UndefOp>(loc, rec),
          address = b.create<LLVM::AddressOfOp>(loc, ptr, name);
    SmallVector<Value> fields = {address, annotation, source, one, null};
    for (auto [field, v] : llvm::enumerate(fields))
      record = b.create<LLVM::InsertValueOp>(loc, record, v,
                                             ArrayRef<int64_t>{int64_t(field)});
    list = b.create<LLVM::InsertValueOp>(loc, list, record,
                                         ArrayRef<int64_t>{int64_t(idx)});
  }
  b.create<LLVM::ReturnOp>(loc, ValueRange{list});
}

} // namespace

void registerVortexDialect(DialectRegistry &registry) {
  registry.insert<VortexDialect>();
}

void configureVortexModule(ModuleOp module, ArrayRef<std::string> names) {
  OpBuilder b(module.getContext());
  module->setAttr("llvm.target_triple",
                  b.getStringAttr("riscv32-unknown-unknown-elf"));
  module->setAttr("llvm.data_layout",
                  b.getStringAttr("e-m:e-p:32:32-i64:64-n32-S128"));
  kernelAnnotations(module, names);
}

void configureVortexKernel(LLVM::LLVMFuncOp kernel) {
  OpBuilder b(kernel.getContext());
  kernel->setAttr("passthrough",
                  b.getArrayAttr({b.getStringAttr("vortex-kernel")}));
  kernel->setAttr("section",
                  b.getStringAttr(".vx_entry." + kernel.getName()));
}

SmallVector<Value> unpackVortexKernelPointers(OpBuilder &b, Location loc,
                                             Value argument, unsigned count) {
  Type ptr = LLVM::LLVMPointerType::get(b.getContext());
  Value blob = targetValue(b, loc, "vortex.uniform", ptr, argument);
  SmallVector<Value> pointers;
  for (unsigned p = 0; p < count; ++p) {
    Value offset = b.create<arith::ConstantIntOp>(loc, p * 8, 32);
    Value address = b.create<LLVM::GEPOp>(loc, ptr, b.getI8Type(), blob,
                                         ValueRange{offset});
    Value raw = b.create<LLVM::LoadOp>(loc, b.getI64Type(), address);
    Value narrow = b.create<arith::TruncIOp>(loc, b.getI32Type(), raw);
    pointers.push_back(b.create<LLVM::IntToPtrOp>(loc, ptr, narrow));
  }
  return pointers;
}

Value createVortexGlobalThreadId(OpBuilder &b, Location loc) {
  Value tid = targetValue(b, loc, "vortex.thread_id", b.getI32Type());
  Value block = targetValue(b, loc, "vortex.block_id", b.getI32Type());
  Value size = targetValue(b, loc, "vortex.block_dim", b.getI32Type());
  Value base = b.create<arith::MulIOp>(loc, block, size);
  return b.create<arith::AddIOp>(loc, base, tid);
}

#define GAC_CSR_HELPER(FUNCTION, NAME)                                        \
  Value FUNCTION(OpBuilder &b, Location loc) {                               \
    return targetValue(b, loc, NAME, b.getI32Type());                         \
  }
GAC_CSR_HELPER(createVortexLocalThreadId, "vortex.thread_id")
GAC_CSR_HELPER(createVortexBlockId, "vortex.block_id")
GAC_CSR_HELPER(createVortexCTAId, "vortex.cta_id")
GAC_CSR_HELPER(createVortexWarpRank, "vortex.warp_rank")
GAC_CSR_HELPER(createVortexWarpCount, "vortex.warp_count")
GAC_CSR_HELPER(createVortexLMEMAddress, "vortex.lmem_address")
#undef GAC_CSR_HELPER

namespace {
enum class Collective { Barrier, ExpectTwo, Issue2D };
// See vx_barrier.h/vx_dxa.h. Keep WGATHER and its DXA consumer in one asm
// expression: x31 holds four lanes, which must not be copied to another reg.
static void collectiveCall(OpBuilder &b, Location loc, ModuleOp module, Collective kind,
                           ValueRange arguments) {
  StringRef name = kind == Collective::Barrier ? "__gac_cta_barrier"
                   : kind == Collective::ExpectTwo ? "__gac_dxa_expect_two"
                                                    : "__gac_dxa_issue_2d";
  auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!fn) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(b.getContext()),
        SmallVector<Type>(arguments.size(), b.getI32Type()));
    fn = b.create<LLVM::LLVMFuncOp>(loc, name, type, LLVM::Linkage::Internal);
    fn->setAttr("convergent", b.getUnitAttr());
    fn->setAttr("passthrough", b.getArrayAttr(
        {b.getStringAttr("convergent"), b.getStringAttr("noinline"),
         b.getStringAttr("optnone")}));
    Block *entry = fn.addEntryBlock(b);
    b.setInsertionPointToStart(entry);
    auto c = [&](int32_t n) -> Value {
      return b.create<LLVM::ConstantOp>(loc, b.getI32Type(),
                                         b.getI32IntegerAttr(n));
    };
    OperationState asmState(loc, "llvm.inline_asm");
    asmState.addAttribute("has_side_effects", b.getUnitAttr());
    if (kind == Collective::Barrier) {
      asmState.addOperands({c(11), entry->getArgument(0), entry->getArgument(1)});
      asmState.addAttribute("asm_string",
                            b.getStringAttr(".insn r $0, 4, 0, x0, $1, $2"));
      asmState.addAttribute("constraints", b.getStringAttr("i,r,r,~{memory}"));
    } else if (kind == Collective::ExpectTwo) {
      asmState.addTypes(b.getI32Type());
      asmState.addOperands({c(11), entry->getArgument(0), c(int32_t(0x80000002u))});
      asmState.addAttribute("asm_string",
                            b.getStringAttr(".insn r $1, 6, 0, $0, $2, $3"));
      asmState.addAttribute("constraints", b.getStringAttr("=r,i,r,r,~{memory}"));
    } else {
      Value shifted = b.create<LLVM::ShlOp>(loc, entry->getArgument(1), c(4));
      Value metadata = b.create<LLVM::OrOp>(loc, shifted, entry->getArgument(2));
      asmState.addTypes(b.getI32Type());
      asmState.addOperands({c(43), entry->getArgument(0), metadata,
                            entry->getArgument(3), entry->getArgument(4), c(11)});
      asmState.addAttribute("asm_string", b.getStringAttr(
          ".insn r4 $1, 0, 0, $0, $3, $4, $5\n\t.insn r $6, 0, 3, x0, $0, x0"));
      asmState.addAttribute("constraints",
                            b.getStringAttr("={x31},i,0,r,r,r,i,~{memory}"));
    }
    b.create(asmState);
    b.create<LLVM::ReturnOp>(loc, ValueRange{});
  }
  auto call = b.create<LLVM::CallOp>(loc, fn, arguments);
  call->setAttr("convergent", b.getUnitAttr());
}
} // namespace

void createVortexBarrier(OpBuilder &b, Location loc, ModuleOp module,
                         Value barrier, Value warps) {
  collectiveCall(b, loc, module, Collective::Barrier, {barrier, warps});
}
void createVortexDxaExpectTwo(OpBuilder &b, Location loc, ModuleOp module,
                              Value barrier) {
  collectiveCall(b, loc, module, Collective::ExpectTwo, {barrier});
}
void createVortexDxaIssue2D(OpBuilder &b, Location loc, ModuleOp module, Value destination,
                           Value barrier, Value descriptor, Value coord0,
                           Value coord1) {
  collectiveCall(b, loc, module, Collective::Issue2D,
                  {destination, barrier, descriptor, coord0, coord1});
}

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
