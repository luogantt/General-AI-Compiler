#include "gac/Fusion.h"
#include "gac/Dialect.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
namespace gac {
namespace {

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

/// Absorb an explicit NCHW channel-bias Add into Conv's existing epilogue.
/// This is deliberately not general Conv+Add fusion: spatial residuals, a
/// second bias, reversed Add operands and low-precision boundaries stay intact.
class ConvChannelBiasFusion : public RewritePattern {
public:
  ConvChannelBiasFusion(MLIRContext *context, StringRef root)
      : RewritePattern(root, 1, context) {}

  LogicalResult matchAndRewrite(Operation *add,
                                PatternRewriter &rewriter) const override {
    if (add->getNumOperands() != 2 || add->getNumResults() != 1)
      return failure();
    Operation *conv = add->getOperand(0).getDefiningOp();
    if (!conv || conv->getName().getStringRef() != "gac.conv2d" ||
        conv->getNumOperands() != 2 || conv->getNumResults() != 1 ||
        !conv->getResult(0).hasOneUse() ||
        conv->getResult(0).getType() != add->getResult(0).getType())
      return failure();
    auto output = dyn_cast<RankedTensorType>(add->getResult(0).getType());
    auto bias = dyn_cast<RankedTensorType>(add->getOperand(1).getType());
    if (!output || !output.hasStaticShape() || output.getRank() != 4 ||
        output.getEncoding() || !output.getElementType().isF32() ||
        !bias || !bias.hasStaticShape() || bias.getEncoding() ||
        !bias.getElementType().isF32())
      return failure();
    int64_t channels = output.getDimSize(1);
    if (channels <= 0)
      return failure();
    // Right-aligned ONNX broadcasting: [C,1,1] or [1,C,1,1] names the
    // channel axis. A plain [C] instead names width, even when C == W.
    if (bias.getRank() == 3) {
      if (bias.getShape() != ArrayRef<int64_t>({channels, 1, 1}))
        return failure();
    } else if (bias.getRank() == 4) {
      if (bias.getShape() != ArrayRef<int64_t>({1, channels, 1, 1}))
        return failure();
    } else {
      return failure();
    }

    // Conv's canonical bias ABI is [C]. This dense logical view is an alias;
    // it does not copy data or evaluate the convolution during compilation.
    auto shapeType = RankedTensorType::get({1}, rewriter.getI64Type());
    OperationState shape(add->getLoc(), ConstantOp::getOperationName());
    shape.addTypes(shapeType);
    shape.addAttribute("value", DenseIntElementsAttr::get(
        shapeType, ArrayRef<int64_t>({channels})));
    Value shapeValue = rewriter.create(shape)->getResult(0);
    OperationState view(add->getLoc(), ReshapeOp::getOperationName());
    view.addOperands({add->getOperand(1), shapeValue});
    view.addTypes(RankedTensorType::get({channels}, output.getElementType()));
    Value channelBias = rewriter.create(view)->getResult(0);

    bool relu = add->getName().getStringRef() == "gac.add_relu";
    OperationState state(add->getLoc(),
                         relu ? Conv2DReluOp::getOperationName()
                              : Conv2DOp::getOperationName());
    state.addOperands(conv->getOperands());
    state.addOperands(channelBias);
    state.addTypes(add->getResultTypes());
    state.addAttributes(conv->getAttrs());
    state.attributes.erase("gac.input_positions");
    for (StringRef key : {"gac.output_names", "gac.source_name"}) {
      state.attributes.erase(key);
      if (Attribute attr = add->getAttr(key))
        state.attributes.set(key, attr);
    }
    state.attributes.set("gac.fusion", rewriter.getStringAttr(
        relu ? "conv_channel_bias_relu" : "conv_channel_bias"));
    rewriter.replaceOp(add, rewriter.create(state)->getResults());
    rewriter.eraseOp(conv);
    return success();
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

} // namespace

void populateGraphFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvChannelBiasFusion>(patterns.getContext(), "gac.add");
  patterns.add<ConvChannelBiasFusion>(patterns.getContext(), "gac.add_relu");
  patterns.add<MatMulBiasFusion, ReluFusion, MSDeformAttnCoordinateFusion>(
      patterns.getContext());
}
} // namespace gac
