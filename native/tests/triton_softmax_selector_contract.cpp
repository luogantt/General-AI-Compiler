#include "gac/TritonBridge.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

int fail(StringRef message) {
  llvm::errs() << message << "\n";
  return 1;
}

struct Fixture {
  OwningOpRef<ModuleOp> module;
  linalg::SoftmaxOp softmax;
};

Fixture makeSoftmax(MLIRContext &context, ArrayRef<int64_t> shape,
                    int64_t axis) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  auto type = RankedTensorType::get(shape, builder.getF32Type());
  builder.setInsertionPointToStart(module->getBody());
  auto function = builder.create<func::FuncOp>(
      loc, "softmax", builder.getFunctionType({type}, {type}));
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value empty = builder.create<tensor::EmptyOp>(
      loc, type.getShape(), type.getElementType());
  auto softmax = builder.create<linalg::SoftmaxOp>(
      loc, TypeRange{type}, entry->getArgument(0), empty, axis);
  softmax->setAttr("gac.linalg_role", builder.getStringAttr("softmax"));
  builder.create<func::ReturnOp>(loc, softmax->getResult(0));
  return {std::move(module), softmax};
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, linalg::LinalgDialect,
                  tensor::TensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto expectSelected = [&](ArrayRef<int64_t> shape, int64_t axis,
                            int64_t rows, int64_t width) -> bool {
    Fixture fixture = makeSoftmax(context, shape, axis);
    if (failed(verify(*fixture.module))) {
      llvm::errs() << "generated canonical Softmax module did not verify\n";
      return false;
    }
    gac::TritonSoftmaxPlan plan;
    std::string reason;
    if (!gac::selectTritonSoftmax(fixture.softmax, plan, reason)) {
      llvm::errs() << "canonical Softmax was rejected: " << reason << "\n";
      return false;
    }
    if (plan.rows != rows || plan.width != width ||
        plan.gridSize != rows) {
      llvm::errs() << "selected row-wise Softmax dimensions disagree\n";
      return false;
    }
    return true;
  };
  auto expectRejected = [&](ArrayRef<int64_t> shape, int64_t axis,
                            StringRef expected) -> bool {
    Fixture fixture = makeSoftmax(context, shape, axis);
    gac::TritonSoftmaxPlan plan;
    std::string reason;
    if (gac::selectTritonSoftmax(fixture.softmax, plan, reason)) {
      llvm::errs() << "unsupported Softmax was selected\n";
      return false;
    }
    if (!StringRef(reason).contains(expected)) {
      llvm::errs() << "unexpected rejection reason: " << reason << "\n";
      return false;
    }
    return true;
  };

  if (!expectSelected({19}, 0, 1, 19) ||
      !expectSelected({2, 3, 19}, 2, 6, 19) ||
      !expectSelected({1, 1024}, 1, 1, 1024) ||
      !expectRejected({2, 19}, 0, "last logical axis") ||
      !expectRejected({1, 1025}, 1, "width <= 1024") ||
      !expectRejected({INT32_MAX, 2}, 1, "flattened element count") ||
      !expectRejected({46341, 46341, 1}, 2, "row count"))
    return fail("row-wise Softmax selector contract failed");

  llvm::outs() << "passed: rank1 and row-wise selection; last-axis, width, "
                  "row-product and total-element boundaries\n";
  return 0;
}
