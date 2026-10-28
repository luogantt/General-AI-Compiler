#include "gac/ConstantFolding.h"
#include "gac/Dialect.h"
#include "gac/Fusion.h"
#include "gac/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>

using namespace mlir;
namespace gac {
namespace {

// Shape reads tensor metadata, not activation values. It is the only fold
// allowed when an operand is a function argument; numerical folding lives in
// ConstantFolding.cpp and requires genuine constant SSA operands.
FailureOr<bool> foldStaticShape(Operation *op, IRRewriter &rewriter,
                                ConstantFoldBudget &budget) {
  if (op->getName().getStringRef() != "gac.shape" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !input.hasStaticShape() || !output || !output.hasStaticShape())
    return false;
  int64_t rank = input.getRank();
  auto bound = [&](StringRef key, int64_t fallback) {
    auto attr = op->getAttrOfType<IntegerAttr>(key);
    int64_t value = attr ? attr.getInt() : fallback;
    return std::clamp<int64_t>(value < 0 ? value + rank : value, 0, rank);
  };
  int64_t start = bound("start", 0), end = bound("end", rank);
  int64_t count = std::max<int64_t>(0, end - start);
  if (count > budget.bytes / 8 || count > budget.scalarOperations)
    return false;
  SmallVector<APInt> data;
  for (int64_t axis = start; axis < end; ++axis)
    data.emplace_back(64, input.getDimSize(axis));
  auto value = DenseIntElementsAttr::get(output, data);
  rewriter.setInsertionPoint(op);
  OperationState state(op->getLoc(), "gac.constant");
  state.addTypes(output);
  state.addAttribute("value", value);
  for (StringRef key : {"gac.source_name", "gac.output_names"})
    if (Attribute attr = op->getAttr(key))
      state.addAttribute(key, attr);
  rewriter.replaceOp(op, rewriter.create(state)->getResults());
  budget.bytes -= count * 8;
  budget.scalarOperations -= count;
  return true;
}

struct ShapeInferencePass
    : PassWrapper<ShapeInferencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  ShapeInferencePass() = default;
  explicit ShapeInferencePass(ConstantFoldBudget budget)
      : initialBudget(budget) {}
  ConstantFoldBudget initialBudget;
  StringRef getArgument() const final { return "gac-infer-shapes"; }
  StringRef getDescription() const final {
    return "Infer GAC tensor shapes and fold bounded constant geometry "
           "programs to "
           "a fixed point";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerGacDialects(registry);
    registry.insert<func::FuncDialect>();
  }
  void runOnOperation() override {
    ConstantFoldBudget foldBudget = initialBudget;
    if (foldBudget.bytes < 0 || foldBudget.scalarOperations < 0 ||
        foldBudget.maxElements < 0 || foldBudget.maxMatmulMultiplyAdds < 0 ||
        foldBudget.maxMatmulReduction < 0 ||
        foldBudget.maxMatmulReduction > INT64_MAX / 2) {
      getOperation().emitError("invalid constant folding resource limits");
      signalPassFailure();
      return;
    }
    bool changed = true;
    while (changed) {
      changed = false;
      WalkResult result = getOperation().walk([&](Operation *operation) {
        if (operation->getName().getDialectNamespace() != "gac")
          return WalkResult::advance();
        SmallVector<Type> oldTypes(operation->getResultTypes());
        if (failed(inferOperation(operation)))
          return WalkResult::interrupt();
        changed |= !llvm::equal(oldTypes, operation->getResultTypes());
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        signalPassFailure();
        return;
      }
      SmallVector<Operation *> candidates;
      getOperation().walk([&](Operation *op) {
        if (op->getName().getDialectNamespace() == "gac")
          candidates.push_back(op);
      });
      IRRewriter rewriter(&getContext());
      for (Operation *op : candidates) {
        FailureOr<bool> folded =
            foldConstantOperation(op, rewriter, foldBudget);
        if (succeeded(folded) && !*folded)
          folded = foldStaticShape(op, rewriter, foldBudget);
        if (failed(folded)) {
          signalPassFailure();
          return;
        }
        changed |= *folded;
      }
    }
    // Function results are the public ABI of the imported graph. Refinement of
    // an SSA value also refines an unranked/dynamic function result, but never
    // changes a known static declaration without diagnosing the contradiction.
    WalkResult result = getOperation().walk([&](func::FuncOp function) {
      if (function.isExternal())
        return WalkResult::advance();
      auto ret =
          dyn_cast<func::ReturnOp>(function.getBody().back().getTerminator());
      if (!ret)
        return WalkResult::advance();
      if (ret.getNumOperands() != function.getNumResults()) {
        function.emitError("return/result count mismatch");
        return WalkResult::interrupt();
      }
      SmallVector<Type> results;
      for (auto [declared, value] :
           llvm::zip(function.getResultTypes(), ret.getOperands())) {
        auto oldTensor = dyn_cast<TensorType>(declared);
        auto tensor = dyn_cast<TensorType>(value.getType());
        if (oldTensor && tensor &&
            oldTensor.getElementType() != tensor.getElementType()) {
          function.emitError(
              "inferred function result element type differs from declaration");
          return WalkResult::interrupt();
        }
        auto oldRanked = dyn_cast<RankedTensorType>(declared);
        auto current = dyn_cast<RankedTensorType>(value.getType());
        if (oldRanked && current) {
          if (oldRanked.getRank() != current.getRank()) {
            function.emitError(
                "inferred function result rank differs from declaration");
            return WalkResult::interrupt();
          }
          for (int64_t axis = 0; axis < oldRanked.getRank(); ++axis)
            if (!oldRanked.isDynamicDim(axis) && !current.isDynamicDim(axis) &&
                oldRanked.getDimSize(axis) != current.getDimSize(axis)) {
              function.emitError(
                  "inferred function result shape differs from declaration");
              return WalkResult::interrupt();
            }
        }
        results.push_back(value.getType());
      }
      function.setType(FunctionType::get(function.getContext(),
                                         function.getArgumentTypes(), results));
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

/// Upstream CSE compares operation attributes. ONNX node/tensor display names
/// therefore prevent equal computations from merging if kept as attributes.
/// Locations retain diagnostic provenance; only internal display names are
/// removed. Result names on public outputs stay intact because the runtime
/// manifest currently uses those names as part of its public output contract.
struct NormalizeProvenancePass
    : PassWrapper<NormalizeProvenancePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeProvenancePass)
  StringRef getArgument() const final { return "gac-normalize-provenance"; }
  StringRef getDescription() const final {
    return "Remove internal display-name attributes that obstruct semantic CSE";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (op->getName().getDialectNamespace() != "gac" ||
          isa<UnsupportedOp>(op))
        return;
      for (Value result : op->getResults())
        for (Operation *user : result.getUsers())
          if (isa<func::ReturnOp>(user))
            return;
      op->removeAttr("gac.source_name");
      op->removeAttr("gac.output_names");
      // Preserve source schema/domain/opset, optional input positions and every
      // other attribute conservatively: these can affect semantic legality.
    });
  }
};

struct GraphFusionPass : PassWrapper<GraphFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GraphFusionPass)
  StringRef getArgument() const final { return "gac-fuse"; }
  StringRef getDescription() const final {
    return "Fuse proven single-use FP32 bias/activation and MSDA coordinate chains";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerGacDialects(registry);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGraphFusionPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

struct LayoutPropagationPass
    : PassWrapper<LayoutPropagationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPropagationPass)
  StringRef getArgument() const final { return "gac-propagate-layout"; }
  StringRef getDescription() const final {
    return "Validate Conv NCHW/OIHW contracts and propagate proven axis "
           "layouts";
  }
  void runOnOperation() override {
    Builder builder(&getContext());
    auto layoutOf = [](Value value) -> StringAttr {
      Operation *definition = value.getDefiningOp();
      return definition ? definition->getAttrOfType<StringAttr>("gac.layout")
                        : StringAttr();
    };
    WalkResult result = getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (!name.starts_with("gac.") || op->getNumResults() != 1)
        return WalkResult::advance();
      auto setLayout = [&](StringRef layout) -> LogicalResult {
        if (auto existing = op->getAttrOfType<StringAttr>("gac.layout"))
          if (existing.getValue() != layout)
            return op->emitOpError(
                "declared layout disagrees with proven axis order");
        op->setAttr("gac.layout", builder.getStringAttr(layout));
        return success();
      };
      if (name == "gac.conv2d" || name == "gac.conv2d_relu") {
        for (StringRef key : {"layout", "data_layout", "gac.input_layout"})
          if (auto attr = op->getAttrOfType<StringAttr>(key))
            if (attr.getValue() != "NCHW") {
              op->emitOpError("Conv requires NCHW; layout changes need an "
                              "explicit transpose");
              return WalkResult::interrupt();
            }
        if (auto source = layoutOf(op->getOperand(0)))
          if (source.getValue() != "NCHW") {
            op->emitOpError("Conv input producer is not NCHW");
            return WalkResult::interrupt();
          }
        if (auto weights = op->getAttrOfType<StringAttr>("gac.weight_layout"))
          if (weights.getValue() != "OIHW") {
            op->emitOpError("Conv weights must be OIHW");
            return WalkResult::interrupt();
          }
        op->setAttr("gac.input_layout", builder.getStringAttr("NCHW"));
        op->setAttr("gac.weight_layout", builder.getStringAttr("OIHW"));
        if (failed(setLayout("NCHW")))
          return WalkResult::interrupt();
      } else if (name == "gac.transpose") {
        auto input = layoutOf(op->getOperand(0));
        auto permutation = getI64ArrayAttr(op, "perm");
        if (input && input.getValue() == "NCHW" &&
            permutation == SmallVector<int64_t>({0, 2, 3, 1})) {
          if (failed(setLayout("NHWC")))
            return WalkResult::interrupt();
        } else if (input && input.getValue() == "NHWC" &&
                   permutation == SmallVector<int64_t>({0, 3, 1, 2})) {
          if (failed(setLayout("NCHW")))
            return WalkResult::interrupt();
        }
      } else {
        // Propagate only across operators preserving all logical axes. Reshape,
        // Gather and reductions require a separate axis analysis and are
        // skipped.
        bool preserve = name == "gac.relu" || name == "gac.identity" ||
                        name == "gac.cast" || name == "gac.softmax" ||
                        name == "gac.batch_norm" || name == "gac.max_pool2d";
        if (preserve && op->getNumOperands())
          if (auto source = layoutOf(op->getOperand(0)))
            if (failed(setLayout(source.getValue())))
              return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createShapeInferencePass(ConstantFoldBudget budget) {
  return std::make_unique<ShapeInferencePass>(budget);
}
std::unique_ptr<Pass> createNormalizeProvenancePass() {
  return std::make_unique<NormalizeProvenancePass>();
}
std::unique_ptr<Pass> createGraphFusionPass() {
  return std::make_unique<GraphFusionPass>();
}
std::unique_ptr<Pass> createLayoutPropagationPass() {
  return std::make_unique<LayoutPropagationPass>();
}
void registerGacPasses() {
  PassRegistration<ShapeInferencePass>();
  PassRegistration<NormalizeProvenancePass>();
  PassRegistration<GraphFusionPass>();
  PassRegistration<LayoutPropagationPass>();
}
} // namespace gac
