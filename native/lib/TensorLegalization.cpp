#include "gac/Dialect.h"
#include "gac/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"
#include <limits>
#include <optional>

using namespace mlir;

namespace gac {
namespace {

// Proof work is bounded independently of the number of tensor elements. Each
// ScatterND tuple selects an entire trailing slice; never enumerate channels.
constexpr int64_t maxScatterProofSlices = 1LL << 20;
constexpr int64_t maxScatterProofCoordinates = 8LL << 20;
constexpr unsigned maxScatterProofStages = 64;

std::optional<int64_t> positiveProduct(ArrayRef<int64_t> shape,
                                     int64_t limit) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    if (dim <= 0 || size > limit / dim)
      return std::nullopt;
    size *= dim;
  }
  return size;
}

struct ScatterProof {
  Operation *op;
  RankedTensorType data, updates;
  DenseIntElementsAttr indices;
  int64_t depth, tuples, targetSlices;
};

std::optional<ScatterProof> scatterProof(Operation *op) {
  if (!op || op->getName().getStringRef() != "gac.scatter_nd" ||
      op->getNumOperands() != 3 || op->getNumResults() != 1)
    return std::nullopt;
  if (op->hasAttr("gac.input_positions")) {
    auto positions = getI64ArrayAttr(op, "gac.input_positions");
    if (positions.size() != 3 || positions[0] != 0 || positions[1] != 1 ||
        positions[2] != 2)
      return std::nullopt;
  }
  for (NamedAttribute attr : op->getAttrs())
    if (!attr.getName().getValue().starts_with("gac.") &&
        attr.getName().getValue() != "reduction")
      return std::nullopt;
  if (Attribute attr = op->getAttr("reduction")) {
    auto reduction = dyn_cast<StringAttr>(attr);
    if (!reduction || reduction.getValue() != "none")
      return std::nullopt;
  }
  auto data = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto updates = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
  auto indexType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  Operation *constant = op->getOperand(1).getDefiningOp();
  auto indices = constant && constant->getName().getStringRef() == "gac.constant"
      ? constant->getAttrOfType<DenseIntElementsAttr>("value")
      : DenseIntElementsAttr();
  if (!data || !updates || !indexType || !indices ||
      !data.hasStaticShape() || !updates.hasStaticShape() ||
      !indexType.hasStaticShape() || data.getRank() < 1 ||
      indexType.getRank() < 1 || indices.getType() != indexType ||
      !indexType.getElementType().isSignlessInteger(64) ||
      data.getElementType() != updates.getElementType() ||
      op->getResult(0).getType() != data)
    return std::nullopt;
  int64_t depth = indexType.getShape().back();
  if (depth < 1 || depth > data.getRank() ||
      !positiveProduct(data.getShape(), std::numeric_limits<int64_t>::max()) ||
      !positiveProduct(updates.getShape(), std::numeric_limits<int64_t>::max()) ||
      !positiveProduct(indexType.getShape(), maxScatterProofCoordinates))
    return std::nullopt;
  SmallVector<int64_t> expected(indexType.getShape().drop_back());
  expected.append(data.getShape().begin() + depth, data.getShape().end());
  if (ArrayRef<int64_t>(expected) != updates.getShape())
    return std::nullopt;
  auto tuples = positiveProduct(indexType.getShape().drop_back(), maxScatterProofSlices);
  auto slices = positiveProduct(data.getShape().take_front(depth), maxScatterProofSlices);
  if (!tuples || !slices)
    return std::nullopt;
  return ScatterProof{op, data, updates, indices, depth, *tuples, *slices};
}

// Return the flattened target prefix and, when requested, its source prefix
// under ONNX Expand broadcasting. Reject negative indices outside the extent
// before normalization, and reject positive indices beyond it.
bool scatterAddress(const ScatterProof &proof, int64_t ordinal,
                    int64_t &target, int64_t *broadcastSource = nullptr) {
  auto values = proof.indices.getValues<APInt>();
  target = 0;
  if (broadcastSource)
    *broadcastSource = 0;
  for (int64_t axis = 0; axis < proof.depth; ++axis) {
    int64_t extent = proof.data.getDimSize(axis);
    int64_t index = values[ordinal * proof.depth + axis].getSExtValue();
    if (index < -extent || index >= extent)
      return false;
    if (index < 0)
      index += extent;
    target = target * extent + index; // Bounded targetSlices checked above.
    if (broadcastSource) {
      int64_t sourceExtent = proof.updates.getDimSize(axis);
      *broadcastSource = *broadcastSource * sourceExtent +
                         (sourceExtent == 1 ? 0 : index);
    }
  }
  return true;
}

void replaceScatterView(Operation *op, Value updates, StringRef name,
                        StringRef proofName, PatternRewriter &rewriter) {
  auto output = cast<RankedTensorType>(op->getResult(0).getType());
  OperationState replacement(op->getLoc(), name);
  replacement.addTypes(output);
  replacement.addOperands(updates);
  bool shapeOperand = name != IdentityOp::getOperationName();
  if (shapeOperand) {
    SmallVector<int64_t> shape(output.getShape());
    auto type = RankedTensorType::get({static_cast<int64_t>(shape.size())},
                                     rewriter.getI64Type());
    OperationState constant(op->getLoc(), ConstantOp::getOperationName());
    constant.addTypes(type);
    constant.addAttribute("value", DenseIntElementsAttr::get(type, shape));
    replacement.addOperands(rewriter.create(constant)->getResult(0));
    if (name == ReshapeOp::getOperationName())
      replacement.addAttribute("allowzero", rewriter.getI64IntegerAttr(0));
  }
  for (NamedAttribute attr : op->getAttrs())
    if (attr.getName().getValue().starts_with("gac.") &&
        attr.getName().getValue() != "gac.input_positions" &&
        attr.getName().getValue() != "gac.legalized_from")
      replacement.addAttribute(attr.getName(), attr.getValue());
  replacement.addAttribute("gac.input_positions", shapeOperand
      ? rewriter.getDenseI64ArrayAttr({0, 1})
      : rewriter.getDenseI64ArrayAttr({0}));
  replacement.addAttribute("gac.legalized_from", rewriter.getStringAttr(proofName));
  rewriter.replaceOp(op, rewriter.create(replacement)->getResults());
}

/// A complete ordered overwrite never reads data, independent of its value.
/// The proof checks every real constant index rather than matching a model.
struct FullOverwriteScatterPattern : RewritePattern {
  explicit FullOverwriteScatterPattern(MLIRContext *context)
      : RewritePattern("gac.scatter_nd", 2, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto proof = scatterProof(op);
    if (!proof || proof->tuples != proof->targetSlices)
      return failure();
    for (int64_t i = 0; i < proof->tuples; ++i) {
      int64_t address;
      if (!scatterAddress(*proof, i, address) || address != i)
        return failure();
    }
    replaceScatterView(op, op->getOperand(2),
        proof->updates == proof->data ? IdentityOp::getOperationName()
                                    : ReshapeOp::getOperationName(),
        "gac.scatter_nd.full_ordered_overwrite", rewriter);
    return success();
  }
};

/// A single-use chain can fill separate slices with the same update tensor.
/// Replace it only if a bounded proof shows exact Expand broadcasting over the
/// entire target, with neither missing nor multiply-written slices.
struct BroadcastScatterChainPattern : RewritePattern {
  explicit BroadcastScatterChainPattern(MLIRContext *context)
      : RewritePattern("gac.scatter_nd", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto last = scatterProof(op);
    if (!last || last->data.getRank() != last->updates.getRank())
      return failure();
    for (int64_t axis = 0; axis < last->data.getRank(); ++axis)
      if (last->updates.getDimSize(axis) != 1 &&
          last->updates.getDimSize(axis) != last->data.getDimSize(axis))
        return failure();
    Value updates = op->getOperand(2);
    llvm::SmallBitVector covered(last->targetSlices);
    int64_t count = 0, coordinates = 0;
    Operation *current = op;
    for (unsigned stage = 0; stage < maxScatterProofStages; ++stage) {
      auto proof = scatterProof(current);
      if (!proof || proof->data != last->data || proof->updates != last->updates ||
          proof->depth != last->depth || current->getOperand(2) != updates ||
          proof->tuples > (maxScatterProofCoordinates - coordinates) / proof->depth)
        return failure();
      coordinates += proof->tuples * proof->depth;
      for (int64_t i = 0; i < proof->tuples; ++i) {
        int64_t target, source;
        if (!scatterAddress(*proof, i, target, &source) || source != i ||
            covered.test(target))
          return failure();
        covered.set(target);
        ++count;
      }
      if (count == last->targetSlices) {
        replaceScatterView(op, updates, ExpandOp::getOperationName(),
                           "gac.scatter_nd.broadcast_chain", rewriter);
        return success();
      }
      Value data = current->getOperand(0);
      // Returning an intermediate result or consuming it on another branch
      // prevents removal of that stage. Never erase such observable values.
      if (!data.hasOneUse())
        return failure();
      current = data.getDefiningOp();
    }
    return failure();
  }
};

/// Import has proved the exact ONNX shape guard and both branch semantics.
/// Now choose the view from the inferred input extent, never activation data.
struct StaticSqueezeIfUnitDimPattern : RewritePattern {
  explicit StaticSqueezeIfUnitDimPattern(MLIRContext *context)
      : RewritePattern("gac.squeeze_if_unit_dim", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto attribute = op->getAttrOfType<IntegerAttr>("axis");
    if (!input || !input.hasStaticShape() || !output ||
        !output.hasStaticShape() || !attribute)
      return rewriter.notifyMatchFailure(op, "shape guard is not statically resolved");
    int64_t axis = attribute.getInt();
    if (axis < 0)
      axis += input.getRank();
    if (axis < 0 || axis >= input.getRank())
      return failure();
    bool squeeze = input.getDimSize(axis) == 1;
    OperationState replacement(op->getLoc(),
        squeeze ? SqueezeOp::getOperationName() : IdentityOp::getOperationName());
    replacement.addTypes(output);
    replacement.addOperands(op->getOperand(0));
    if (squeeze) {
      // Keep an explicit one-element axes tensor even for rank-one -> scalar.
      // A Reshape shape tensor would have zero elements in that legal case.
      auto type = RankedTensorType::get({1}, rewriter.getI64Type());
      OperationState constant(op->getLoc(), ConstantOp::getOperationName());
      constant.addTypes(type);
      constant.addAttribute("value", DenseIntElementsAttr::get(type, {axis}));
      replacement.addOperands(rewriter.create(constant)->getResult(0));
    }
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("gac.") &&
          attr.getName().getValue() != "gac.input_positions")
        replacement.addAttribute(attr.getName(), attr.getValue());
    replacement.addAttribute("gac.input_positions", squeeze
        ? rewriter.getDenseI64ArrayAttr({0, 1})
        : rewriter.getDenseI64ArrayAttr({0}));
    replacement.addAttribute("gac.legalized_from",
                            rewriter.getStringAttr("gac.squeeze_if_unit_dim"));
    rewriter.replaceOp(op, rewriter.create(replacement)->getResults());
    return success();
  }
};

/// This is a semantic legalization, not target code generation. The importer
/// retains every Split result as a separate SSA value. Once the split offsets
/// are compile-time known, each result is exactly a Slice of the same input.
/// The source location and each original ONNX output name survive the rewrite.
struct StaticSplitPattern : RewritePattern {
  explicit StaticSplitPattern(MLIRContext *context)
      : RewritePattern("gac.split", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumOperands() > 2)
      return failure();
    auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!input)
      return rewriter.notifyMatchFailure(op, "Split input rank is unresolved");
    // A declared output shape is not evidence that a runtime split parameter
    // is constant. Never specialize away a data-dependent sizes operand.
    if (op->getNumOperands() == 2 &&
        !getConstantIntegers(op->getOperand(1)))
      return rewriter.notifyMatchFailure(op, "Split sizes are not constant");
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    int64_t axis = axisAttr ? axisAttr.getInt() : 0;
    if (axis < 0)
      axis += input.getRank();
    if (axis < 0 || axis >= input.getRank() || input.isDynamicDim(axis))
      return rewriter.notifyMatchFailure(op, "Split axis extent is unresolved");
    SmallVector<int64_t> sizes;
    int64_t total = 0;
    for (Type type : op->getResultTypes()) {
      auto result = dyn_cast<RankedTensorType>(type);
      if (!result || result.getRank() != input.getRank() ||
          result.isDynamicDim(axis))
        return rewriter.notifyMatchFailure(op, "Split result extents are unresolved");
      int64_t size = result.getDimSize(axis);
      if (size > input.getDimSize(axis) - total)
        return rewriter.notifyMatchFailure(op, "Split result sizes are inconsistent");
      sizes.push_back(size);
      total += size;
    }
    if (total != input.getDimSize(axis))
      return rewriter.notifyMatchFailure(op, "Split result sizes do not cover input");

    auto constant = [&](int64_t value) -> Value {
      auto type = RankedTensorType::get({1}, rewriter.getI64Type());
      OperationState state(op->getLoc(), ConstantOp::getOperationName());
      state.addTypes(type);
      state.addAttribute("value", DenseIntElementsAttr::get(type, {value}));
      return rewriter.create(state)->getResult(0);
    };
    Value axisValue = constant(axis), stepValue = constant(1);
    auto names = op->getAttrOfType<ArrayAttr>("gac.output_names");
    SmallVector<Value> replacements;
    int64_t start = 0;
    for (auto [index, size] : llvm::enumerate(sizes)) {
      OperationState slice(op->getLoc(), SliceOp::getOperationName());
      slice.addTypes(op->getResult(index).getType());
      slice.addOperands({op->getOperand(0), constant(start),
                         constant(start + size), axisValue, stepValue});
      for (NamedAttribute attr : op->getAttrs()) {
        StringRef key = attr.getName().getValue();
        if (key == "gac.source_name" || key == "gac.source_domain" ||
            key == "gac.source_op" || key == "gac.opset")
          slice.addAttribute(attr.getName(), attr.getValue());
      }
      slice.addAttribute("gac.input_positions",
                         rewriter.getDenseI64ArrayAttr({0, 1, 2, 3, 4}));
      if (names && index < names.size())
        slice.addAttribute("gac.output_names",
                           rewriter.getArrayAttr({names[index]}));
      slice.addAttribute("gac.legalized_from",
                         rewriter.getStringAttr("gac.split"));
      replacements.push_back(rewriter.create(slice)->getResult(0));
      start += size;
    }
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

struct LegalizeTensorOpsPass
    : PassWrapper<LegalizeTensorOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeTensorOpsPass)
  StringRef getArgument() const final { return "gac-legalize-tensor-ops"; }
  StringRef getDescription() const final {
    return "Legalize statically shaped GAC tensor operations without changing their semantics";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerGacDialects(registry);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StaticSplitPattern, StaticSqueezeIfUnitDimPattern,
                 FullOverwriteScatterPattern, BroadcastScatterChainPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    bool unresolved = false;
    getOperation().walk([&](SqueezeIfUnitDimOp op) {
      op.emitOpError("shape-guarded If requires a static input shape before device legalization");
      unresolved = true;
    });
    if (unresolved)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createLegalizeTensorOpsPass() {
  return std::make_unique<LegalizeTensorOpsPass>();
}

void registerTensorLegalizationPasses() {
  PassRegistration<LegalizeTensorOpsPass>();
}
} // namespace gac
