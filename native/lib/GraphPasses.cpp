#include "gac/ConstantFolding.h"
#include "gac/Dialect.h"
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

/// Fusion changes graph operations through PatternRewriter. The single-use
/// checks avoid duplicating reductions or erasing graph outputs used elsewhere.
/// No numeric reassociation is performed: multiply/add/activation order stays.
// A view changes only the logical coordinates of a dense FP32 tensor. This
// deliberately excludes Transpose, Cast and any layout encoding.
std::optional<int64_t> denseFP32Elements(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      !type.getElementType().isF32())
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension <= 0 || elements > INT64_MAX / dimension)
      return std::nullopt;
    elements *= dimension;
  }
  return elements;
}

Value peelCoordinateViews(Value value, SmallVectorImpl<Operation *> &views) {
  while (Operation *op = value.getDefiningOp()) {
    StringRef name = op->getName().getStringRef();
    if (name != "gac.identity" && name != "gac.reshape" &&
        name != "gac.squeeze" && name != "gac.unsqueeze")
      break;
    if (!value.hasOneUse() || op->getNumResults() != 1 || !op->getNumOperands())
      return {};
    auto before = denseFP32Elements(op->getOperand(0));
    auto after = denseFP32Elements(value);
    if (!before || !after || *before != *after)
      return {};
    for (Value parameter : op->getOperands().drop_front())
      if (!getConstantIntegers(parameter))
        return {};
    views.push_back(op);
    value = op->getOperand(0);
  }
  return value;
}

/// Preserve Div -> logical reshape -> Add -> logical reshape exactly, but
/// evaluate the two scalar operations when MSDeformAttn consumes a location.
/// Neither activation values nor a model-specific visibility mask are known
/// or assumed here; the normalizer/reference may themselves be runtime inputs.
class MSDeformAttnCoordinateFusion : public RewritePattern {
public:
  explicit MSDeformAttnCoordinateFusion(MLIRContext *context)
      : RewritePattern("gac.ms_deform_attn", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 4 || op->getNumResults() != 1)
      return failure();
    Value locations = op->getOperand(2);
    if (!denseFP32Elements(locations))
      return failure();
    SmallVector<Operation *> outerViews;
    Value sum = peelCoordinateViews(locations, outerViews);
    Operation *add = sum ? sum.getDefiningOp() : nullptr;
    if (!add || add->getName().getStringRef() != "gac.add" ||
        add->getNumOperands() != 2 || !sum.hasOneUse())
      return failure();
    for (unsigned side = 0; side < 2; ++side) {
      SmallVector<Operation *> innerViews;
      Value quotient = peelCoordinateViews(add->getOperand(side), innerViews);
      Operation *div = quotient ? quotient.getDefiningOp() : nullptr;
      if (!div || div->getName().getStringRef() != "gac.div" ||
          div->getNumOperands() != 2 || !quotient.hasOneUse() ||
          quotient.getType() != locations.getType() ||
          div->getOperand(0).getType() != locations.getType() ||
          add->getOperand(side).getType() != sum.getType())
        continue;
      Value normalizer = div->getOperand(1),
            reference = add->getOperand(1 - side);
      if (!denseFP32Elements(div->getOperand(0)) ||
          !denseFP32Elements(normalizer) || !denseFP32Elements(reference))
        continue;
      // The numerator is always Div operand 0. Never commute division, merge
      // broadcasts across the reshape, or remove an FP16/Cast round.
      OperationState state(op->getLoc(),
                           MSDeformAttnOffsetsOp::getOperationName());
      state.addOperands({op->getOperand(0), op->getOperand(1),
                         div->getOperand(0), op->getOperand(3), normalizer,
                         reference});
      state.addTypes(op->getResultTypes());
      state.addAttributes(op->getAttrs());
      state.attributes.erase("gac.input_positions");
      state.attributes.set(
          "coordinate_shape",
          rewriter.getDenseI64ArrayAttr(
              cast<RankedTensorType>(sum.getType()).getShape()));
      state.attributes.set("reference_first",
                           rewriter.getI64IntegerAttr(side == 1));
      state.attributes.set(
          "gac.fusion", rewriter.getStringAttr("ms_deform_attn_coordinates"));
      rewriter.replaceOp(op, rewriter.create(state)->getResults());
      for (Operation *view : outerViews)
        rewriter.eraseOp(view);
      rewriter.eraseOp(add);
      for (Operation *view : innerViews)
        rewriter.eraseOp(view);
      rewriter.eraseOp(div);
      return success();
    }
    return failure();
  }
};

class MatMulBiasFusion : public RewritePattern {
public:
  explicit MatMulBiasFusion(MLIRContext *context)
      : RewritePattern("gac.add", 1, context) {}
  LogicalResult matchAndRewrite(Operation *add,
                                PatternRewriter &rewriter) const override {
    if (add->getNumOperands() != 2 || add->getNumResults() != 1)
      return failure();
    auto output = dyn_cast<RankedTensorType>(add->getResult(0).getType());
    if (!output || !output.hasStaticShape() || !output.getElementType().isF32())
      return failure();
    for (unsigned side = 0; side < 2; ++side) {
      Operation *matmul = add->getOperand(side).getDefiningOp();
      if (!matmul || matmul->getName().getStringRef() != "gac.matmul" ||
          matmul->getNumOperands() != 2 || matmul->getNumResults() != 1 ||
          !matmul->getResult(0).hasOneUse() ||
          matmul->getResult(0).getType() != output)
        continue;
      Value bias = add->getOperand(1 - side);
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || !biasType.hasStaticShape() ||
          !biasType.getElementType().isF32())
        continue;
      // The verified Add has the exact MatMul result type. Therefore its bias
      // broadcasts into that domain instead of introducing larger batch/M/N
      // dimensions. FP32 avoids removing an intervening low-precision round.
      OperationState state(add->getLoc(), MatMulBiasOp::getOperationName());
      state.addOperands(matmul->getOperands());
      state.addOperands(bias);
      state.addTypes(output);
      state.addAttributes(add->getAttrs());
      state.attributes.erase("gac.input_positions");
      state.attributes.set("bias_first", rewriter.getI64IntegerAttr(side == 1));
      state.attributes.set("gac.fusion", rewriter.getStringAttr("matmul_bias"));
      rewriter.replaceOp(add, rewriter.create(state)->getResults());
      rewriter.eraseOp(matmul);
      return success();
    }
    return failure();
  }
};

class ReluFusion : public RewritePattern {
public:
  explicit ReluFusion(MLIRContext *context)
      : RewritePattern("gac.relu", 1, context) {}
  LogicalResult matchAndRewrite(Operation *relu,
                                PatternRewriter &rewriter) const override {
    if (relu->getNumOperands() != 1 || relu->getNumResults() != 1)
      return failure();
    Operation *producer = relu->getOperand(0).getDefiningOp();
    if (!producer || producer->getNumResults() != 1 ||
        !producer->getResult(0).hasOneUse())
      return failure();
    auto resultType = dyn_cast<TensorType>(producer->getResult(0).getType());
    // Current fused kernels accumulate in FP32. Removing an FP16/BF16 store
    // removes a rounding boundary between MatMul/Conv, bias and activation.
    // Until the lowering models those rounds explicitly, fuse FP32 only.
    if (!resultType || !resultType.getElementType().isF32())
      return failure();
    StringRef name = producer->getName().getStringRef();
    StringRef fused;
    SmallVector<Value> inputs(producer->getOperands());
    Operation *matmul = nullptr;
    bool biasFirst = false;
    if (name == "gac.conv2d") {
      // ONNX Conv already carries an optional channel bias as its third input.
      // This covers Conv+Bias+ReLU without inventing general broadcast rules.
      fused = "gac.conv2d_relu";
    } else if (name == "gac.matmul_bias") {
      // Either greedy rewrite order reaches the same MatMul+Bias+ReLU op.
      fused = "gac.matmul_bias_relu";
    } else if (name == "gac.add") {
      fused = "gac.add_relu";
      for (unsigned side = 0; side < 2; ++side) {
        Operation *candidate = producer->getOperand(side).getDefiningOp();
        if (!candidate || candidate->getName().getStringRef() != "gac.matmul" ||
            !candidate->getResult(0).hasOneUse())
          continue;
        auto shape =
            dyn_cast<RankedTensorType>(candidate->getResult(0).getType());
        // The fused template requires the Add to preserve the MatMul domain.
        // A broadcast that expands M/N or adds batch axes is intentionally left
        // as AddReLU until a template explicitly implements that larger domain.
        if (!shape || !shape.hasStaticShape() ||
            candidate->getResult(0).getType() !=
                producer->getResult(0).getType())
          continue;
        matmul = candidate;
        biasFirst = side == 1;
        inputs.assign(candidate->getOperands().begin(),
                      candidate->getOperands().end());
        inputs.push_back(producer->getOperand(1 - side));
        fused = "gac.matmul_bias_relu";
        break;
      }
    } else
      return failure();

    OperationState state(relu->getLoc(), fused);
    state.addOperands(inputs);
    state.addTypes(relu->getResultTypes());
    // Conv's stride/padding/group attributes remain on its fused operation.
    state.addAttributes(producer->getAttrs());
    // The result tensor name belongs to the ReLU output, not its input
    // producer.
    for (StringRef key : {"gac.output_names", "gac.source_name"})
      if (Attribute attr = relu->getAttr(key))
        state.attributes.set(key, attr);
    // Input positions were source-ONNX bookkeeping; the fused operation uses a
    // new canonical operand list and cannot retain the producer's old mapping.
    state.attributes.erase("gac.input_positions");
    if (matmul)
      state.attributes.set("bias_first", rewriter.getI64IntegerAttr(biasFirst));
    state.attributes.set("gac.fusion",
                         rewriter.getStringAttr(fused.drop_front(4)));
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(relu, replacement->getResults());
    rewriter.eraseOp(producer);
    if (matmul)
      rewriter.eraseOp(matmul);
    return success();
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
    return "Fuse single-use FP32 MatMul+Bias and Conv/Add/MatMul+Bias+ReLU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerGacDialects(registry);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulBiasFusion, ReluFusion, MSDeformAttnCoordinateFusion>(
        &getContext());
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
