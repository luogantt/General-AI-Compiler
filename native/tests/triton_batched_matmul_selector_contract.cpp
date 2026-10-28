#include "gac/TritonBridge.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  linalg::GenericOp matmul;
};

Fixture makeMatmul(MLIRContext &context, ArrayRef<int64_t> aShape,
                   ArrayRef<int64_t> bShape, ArrayRef<int64_t> cShape,
                   bool aliasInputs = false) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  auto f32 = builder.getF32Type();
  auto aType = RankedTensorType::get(aShape, f32);
  auto bType = RankedTensorType::get(bShape, f32);
  auto cType = RankedTensorType::get(cShape, f32);
  SmallVector<Type> inputs{aType};
  if (!aliasInputs)
    inputs.push_back(bType);
  builder.setInsertionPointToStart(module->getBody());
  auto function = builder.create<func::FuncOp>(
      loc, "matmul", builder.getFunctionType(inputs, {cType}));
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value a = entry->getArgument(0);
  Value b = aliasInputs ? a : entry->getArgument(1);

  Value empty = builder.create<tensor::EmptyOp>(
      loc, cType.getShape(), cType.getElementType());
  SmallVector<AffineExpr> cMap;
  SmallVector<utils::IteratorType> cIterators(
      cType.getRank(), utils::IteratorType::parallel);
  for (unsigned axis = 0; axis < unsigned(cType.getRank()); ++axis)
    cMap.push_back(getAffineDimExpr(axis, &context));
  auto initializer = builder.create<linalg::GenericOp>(
      loc, TypeRange{cType}, ValueRange{}, ValueRange{empty},
      ArrayRef<AffineMap>{AffineMap::get(cType.getRank(), 0, cMap, &context)},
      cIterators,
      [&](OpBuilder &nested, Location at, ValueRange) {
        Value zero = nested.create<arith::ConstantFloatOp>(
            at, APFloat(0.0f), f32);
        nested.create<linalg::YieldOp>(at, zero);
      });
  initializer->setAttr("gac.linalg_role", builder.getStringAttr("initialize"));

  unsigned rank = cType.getRank(), loops = rank + 1;
  auto dim = [&](unsigned axis) { return getAffineDimExpr(axis, &context); };
  auto batchMap = [&](RankedTensorType type, SmallVectorImpl<AffineExpr> &map) {
    for (int64_t axis = 0; axis + 2 < type.getRank(); ++axis)
      map.push_back(type.getDimSize(axis) == 1
                        ? getAffineConstantExpr(0, &context)
                        : dim((rank - 2) - (type.getRank() - 2) + axis));
  };
  SmallVector<AffineExpr> aMap, bMap;
  batchMap(aType, aMap);
  batchMap(bType, bMap);
  aMap.push_back(dim(rank - 2));
  aMap.push_back(dim(rank));
  bMap.push_back(dim(rank));
  bMap.push_back(dim(rank - 1));
  SmallVector<AffineMap> maps{
      AffineMap::get(loops, 0, aMap, &context),
      AffineMap::get(loops, 0, bMap, &context),
      AffineMap::get(loops, 0, cMap, &context)};
  SmallVector<utils::IteratorType> iterators(
      rank, utils::IteratorType::parallel);
  iterators.push_back(utils::IteratorType::reduction);
  auto matmul = builder.create<linalg::GenericOp>(
      loc, TypeRange{cType}, ValueRange{a, b},
      ValueRange{initializer.getResult(0)}, maps, iterators,
      [&](OpBuilder &nested, Location at, ValueRange arguments) {
        Value product = nested.create<arith::MulFOp>(
            at, arguments[0], arguments[1]);
        Value sum = nested.create<arith::AddFOp>(
            at, arguments[2], product);
        nested.create<linalg::YieldOp>(at, sum);
      });
  matmul->setAttr("gac.linalg_role", builder.getStringAttr("matmul"));
  builder.create<func::ReturnOp>(loc, matmul.getResult(0));
  return {std::move(module), matmul};
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect,
                  linalg::LinalgDialect, tensor::TensorDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto expectSelected = [&](ArrayRef<int64_t> aShape,
                            ArrayRef<int64_t> bShape,
                            ArrayRef<int64_t> cShape, int64_t batches,
                            StringRef mode) {
    Fixture fixture = makeMatmul(context, aShape, bShape, cShape);
    if (failed(verify(*fixture.module))) {
      llvm::errs() << "generated canonical MatMul module did not verify\n";
      return false;
    }
    gac::TritonDotPlan plan;
    std::string reason;
    if (!gac::selectTritonDot(fixture.matmul, plan, reason)) {
      llvm::errs() << "canonical MatMul was rejected: " << reason << "\n";
      return false;
    }
    int64_t blockK = 16;
    while (blockK < aShape.back())
      blockK *= 2;
    if (plan.batches != batches || plan.batchMode != mode ||
        plan.blockK != blockK ||
        plan.aShape != aShape || plan.bShape != bShape ||
        plan.cShape != cShape) {
      llvm::errs() << "selected batched MatMul plan disagrees\n";
      return false;
    }
    return true;
  };
  auto expectRejected = [&](ArrayRef<int64_t> aShape,
                            ArrayRef<int64_t> bShape,
                            ArrayRef<int64_t> cShape, StringRef expected,
                            bool aliasInputs = false) {
    Fixture fixture =
        makeMatmul(context, aShape, bShape, cShape, aliasInputs);
    gac::TritonDotPlan plan;
    std::string reason;
    if (gac::selectTritonDot(fixture.matmul, plan, reason)) {
      llvm::errs() << "unsupported MatMul was selected\n";
      return false;
    }
    if (!StringRef(reason).contains(expected)) {
      llvm::errs() << "unexpected rejection reason: " << reason << "\n";
      return false;
    }
    return true;
  };

  if (!expectSelected({17, 32}, {32, 19}, {17, 19}, 1, "rank2") ||
      !expectSelected({2, 17, 32}, {2, 32, 19}, {2, 17, 19}, 2,
                      "matched-prefix") ||
      !expectSelected({3, 17, 32}, {32, 19}, {3, 17, 19}, 3,
                      "shared-rhs") ||
      !expectSelected({1, 2, 5, 16}, {1, 2, 16, 7}, {1, 2, 5, 7}, 2,
                      "matched-prefix") ||
      !expectSelected({2, 2, 5, 48}, {48, 7}, {2, 2, 5, 7}, 4,
                      "shared-rhs") ||
      !expectRejected({17, 32}, {2, 32, 19}, {2, 17, 19},
                      "full-rank A") ||
      !expectRejected({2, 17, 32}, {1, 32, 19}, {2, 17, 19},
                      "B batch prefix") ||
      !expectRejected({2, 17, 32}, {3, 32, 19}, {2, 17, 19},
                      "B batch prefix") ||
      !expectRejected({2, 17, 15}, {2, 15, 19}, {2, 17, 19},
                      "K divisible by 16") ||
      !expectRejected({2, 16, 16}, {2, 16, 16}, {2, 16, 16},
                      "distinct A and B", true) ||
      !expectRejected({46341, 46341, 1, 16}, {16, 1},
                      {46341, 46341, 1, 1}, "batch count"))
    return fail("batched MatMul selector contract failed");

  llvm::outs() << "passed: rank2, matched-prefix, shared-RHS and unit-prefix "
                  "selection; broadcast, K, alias and i32 boundaries\n";
  return 0;
}
