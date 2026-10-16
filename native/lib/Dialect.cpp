#include "gac/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

using namespace mlir;
namespace gac {

GacDialect::GacDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<GacDialect>()) {
  addOperations<ConstantOp, MatMulOp, GemmOp, AddOp, SubOp, MulOp, DivOp, PowOp,
                GreaterOp, LessOp, EqualOp, AndOp, MaxOp, WhereOp, ReluOp,
                GeluOp, SigmoidOp, LogOp, SinOp, CosOp, ExpOp, SqrtOp, ClampOp,
                IdentityOp, CastOp, ReshapeOp, TransposeOp, ConcatOp, ExpandOp,
                SliceOp, ReduceSumOp, ReduceMeanOp, SoftmaxOp, LayerNormOp,
                Conv2DOp, DeformConv2DOp, BatchNormOp, MaxPool2DOp,
                GridSampleOp, MSDeformAttnOp, MSDeformAttnOffsetsOp, ShapeOp,
                GatherOp, UnsqueezeOp, SqueezeOp, SqueezeIfUnitDimOp, RangeOp,
                ConstantOfShapeOp, ResizeOp, AddReluOp, Conv2DReluOp,
                MatMulBiasOp, MatMulBiasReluOp, MatMulBiasGeluOp, TileOp,
                ScatterNDOp, NonZeroOp, SplitOp, UnsupportedOp>();
}

void registerGacDialects(DialectRegistry &registry) {
  registry.insert<GacDialect>();
}

SmallVector<int64_t> getI64ArrayAttr(Operation *op, StringRef name) {
  if (auto values = op->getAttrOfType<DenseI64ArrayAttr>(name))
    return SmallVector<int64_t>(values.asArrayRef());
  if (auto values = op->getAttrOfType<ArrayAttr>(name)) {
    SmallVector<int64_t> result;
    for (Attribute value : values) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return {};
      result.push_back(integer.getInt());
    }
    return result;
  }
  return {};
}

std::optional<SmallVector<int64_t>> getConstantIntegers(Value value) {
  if (!value)
    return std::nullopt;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "gac.constant")
    return std::nullopt;
  auto attr = definition->getAttrOfType<DenseIntElementsAttr>("value");
  if (!attr)
    return std::nullopt;
  SmallVector<int64_t> result;
  for (const APInt &integer : attr.getValues<APInt>()) {
    if (integer.getBitWidth() > 64)
      return std::nullopt;
    result.push_back(integer.getSExtValue());
  }
  return result;
}

namespace {
int64_t integer(Operation *op, StringRef name, int64_t fallback) {
  auto value = op->getAttrOfType<IntegerAttr>(name);
  return value ? value.getInt() : fallback;
}

// ONNX omits optional inputs with empty names. The importer keeps their
// original positions so Resize's sizes cannot accidentally become its roi, for
// example.
Value operand(Operation *op, unsigned position) {
  auto positions = getI64ArrayAttr(op, "gac.input_positions");
  if (positions.empty())
    return position < op->getNumOperands() ? op->getOperand(position) : Value();
  for (auto [index, original] : llvm::enumerate(positions))
    if (original == position && index < op->getNumOperands())
      return op->getOperand(index);
  return {};
}

RankedTensorType ranked(Value value) {
  return value ? dyn_cast<RankedTensorType>(value.getType())
               : RankedTensorType();
}

std::optional<SmallVector<int64_t>>
constantParameter(Operation *op, unsigned index, StringRef attrName) {
  if (Value value = operand(op, index))
    return getConstantIntegers(value);
  if (op->hasAttr(attrName))
    return getI64ArrayAttr(op, attrName);
  return std::nullopt;
}

bool compatibleDim(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

/// NumPy/ONNX trailing-axis broadcasting. An unknown dimension joined with a
/// known non-one dimension is known; two incompatible known dimensions fail.
LogicalResult broadcast(Operation *op, ArrayRef<int64_t> lhs,
                        ArrayRef<int64_t> rhs, SmallVectorImpl<int64_t> &out) {
  out.assign(std::max(lhs.size(), rhs.size()), 1);
  for (unsigned n = 0; n < out.size(); ++n) {
    int64_t a = n < lhs.size() ? lhs[lhs.size() - n - 1] : 1;
    int64_t b = n < rhs.size() ? rhs[rhs.size() - n - 1] : 1;
    int64_t dim;
    if (a == 1)
      dim = b;
    else if (b == 1 || a == b)
      dim = a;
    else if (ShapedType::isDynamic(a))
      dim = b;
    else if (ShapedType::isDynamic(b))
      dim = a;
    else
      return op->emitOpError("incompatible broadcast dimensions ")
             << a << " and " << b;
    out[out.size() - n - 1] = dim;
  }
  return success();
}

LogicalResult assignType(Operation *op, TensorType inferred, bool update,
                         unsigned resultIndex = 0) {
  auto result = op->getResult(resultIndex);
  auto current = dyn_cast<TensorType>(result.getType());
  if (!current)
    return op->emitOpError("requires a tensor result");
  if (current.getElementType() != inferred.getElementType())
    return op->emitOpError(
        "result element type disagrees with operator semantics");
  auto oldRanked = dyn_cast<RankedTensorType>(current);
  auto newRanked = dyn_cast<RankedTensorType>(inferred);
  if (oldRanked && newRanked) {
    if (oldRanked.getRank() != newRanked.getRank())
      return op->emitOpError(
          "declared result rank disagrees with inferred rank");
    SmallVector<int64_t> merged(oldRanked.getShape());
    for (int64_t index = 0; index < oldRanked.getRank(); ++index) {
      if (!compatibleDim(merged[index], newRanked.getDimSize(index)))
        return op->emitOpError("declared result dimension ")
               << index << " disagrees with inferred shape " << newRanked;
      if (ShapedType::isDynamic(merged[index]))
        merged[index] = newRanked.getDimSize(index);
    }
    if (update)
      result.setType(RankedTensorType::get(merged, current.getElementType(),
                                           oldRanked.getEncoding()));
  } else if (!oldRanked && update)
    result.setType(inferred);
  return success();
}

Type onnxElementType(MLIRContext *context, int64_t code) {
  switch (code) {
  case 1:
    return Float32Type::get(context);
  case 2:
    return IntegerType::get(context, 8, IntegerType::Unsigned);
  case 3:
    return IntegerType::get(context, 8);
  case 4:
    return IntegerType::get(context, 16, IntegerType::Unsigned);
  case 5:
    return IntegerType::get(context, 16);
  case 6:
    return IntegerType::get(context, 32);
  case 12:
    return IntegerType::get(context, 32, IntegerType::Unsigned);
  case 7:
    return IntegerType::get(context, 64);
  case 13:
    return IntegerType::get(context, 64, IntegerType::Unsigned);
  case 9:
    return IntegerType::get(context, 1);
  case 10:
    return Float16Type::get(context);
  case 11:
    return Float64Type::get(context);
  case 16:
    return BFloat16Type::get(context);
  default:
    return {};
  }
}

LogicalResult normalizedAxis(Operation *op, int64_t &axis, int64_t rank) {
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return op->emitOpError("axis is outside tensor rank");
  return success();
}

LogicalResult matmulShape(Operation *op, RankedTensorType lhs,
                          RankedTensorType rhs,
                          SmallVectorImpl<int64_t> &shape) {
  if (!lhs.getRank() || !rhs.getRank())
    return op->emitOpError("MatMul does not accept scalar operands");
  int64_t kLeft = lhs.getShape().back();
  int64_t kRight = rhs.getDimSize(rhs.getRank() == 1 ? 0 : rhs.getRank() - 2);
  if (!compatibleDim(kLeft, kRight))
    return op->emitOpError("MatMul reduction dimensions differ");
  ArrayRef<int64_t> leftBatch =
      lhs.getShape().take_front(std::max<int64_t>(0, lhs.getRank() - 2));
  ArrayRef<int64_t> rightBatch =
      rhs.getShape().take_front(std::max<int64_t>(0, rhs.getRank() - 2));
  if (failed(broadcast(op, leftBatch, rightBatch, shape)))
    return failure();
  if (lhs.getRank() != 1)
    shape.push_back(lhs.getDimSize(lhs.getRank() - 2));
  if (rhs.getRank() != 1)
    shape.push_back(rhs.getShape().back());
  return success();
}

// Custom ONNX contracts deliberately have one spelling and one layout. Do not
// infer a different DCN/MSDA schema from operand ranks or silently ignore a
// misspelled attribute: such an import can otherwise compile the wrong model.
LogicalResult customInputs(Operation *op, unsigned min, unsigned max) {
  if (op->getNumOperands() < min || op->getNumOperands() > max)
    return op->emitOpError("invalid custom operator operand count");
  for (unsigned i = 0; i < op->getNumOperands(); ++i)
    if (operand(op, i) != op->getOperand(i))
      return op->emitOpError("custom operator inputs cannot contain holes");
  return success();
}

LogicalResult intParameter(Operation *op, StringRef key, int64_t fallback,
                           int64_t &out) {
  out = fallback;
  if (Attribute attr = op->getAttr(key)) {
    auto value = dyn_cast<IntegerAttr>(attr);
    if (!value || value.getValue().getBitWidth() > 64)
      return op->emitOpError("requires an integer attribute: ") << key;
    out = value.getInt();
  }
  return success();
}

LogicalResult pairParameter(Operation *op, StringRef key, int64_t fallback,
                            SmallVector<int64_t> &out) {
  out = op->hasAttr(key) ? getI64ArrayAttr(op, key)
                         : SmallVector<int64_t>{fallback, fallback};
  if (out.size() != 2)
    return op->emitOpError("requires a two-integer attribute: ") << key;
  return success();
}

LogicalResult exactString(Operation *op, StringRef key, StringRef expected) {
  if (Attribute attr = op->getAttr(key)) {
    auto value = dyn_cast<StringAttr>(attr);
    if (!value || value.getValue() != expected)
      return op->emitOpError("requires ") << key << " = '" << expected << "'";
  }
  return success();
}

// Multiplication of nonnegative shape components, with no signed overflow.
std::optional<int64_t> shapeProduct(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > INT64_MAX / b))
    return std::nullopt;
  return a * b;
}

LogicalResult inferDeformConv(Operation *op, bool update) {
  if (failed(customInputs(op, 4, 5)))
    return failure();
  for (Value value : op->getOperands())
    if (!cast<TensorType>(value.getType()).getElementType().isF32())
      return op->emitOpError("DeformConv2D currently requires FP32 tensors");
  SmallVector<int64_t> stride, padding, dilation;
  int64_t groups, deformGroups;
  if (failed(pairParameter(op, "stride", 1, stride)) ||
      failed(pairParameter(op, "padding", 0, padding)) ||
      failed(pairParameter(op, "dilation", 1, dilation)) ||
      failed(intParameter(op, "groups", 1, groups)) ||
      failed(intParameter(op, "deform_groups", 1, deformGroups)))
    return failure();
  if (groups <= 0 || deformGroups <= 0 ||
      llvm::any_of(stride, [](int64_t v) { return v <= 0; }) ||
      llvm::any_of(dilation, [](int64_t v) { return v <= 0; }) ||
      llvm::any_of(padding, [](int64_t v) { return v < 0; }))
    return op->emitOpError("DeformConv2D requires positive "
                           "stride/dilation/groups and nonnegative padding");
  auto x = ranked(operand(op, 0)), offset = ranked(operand(op, 1)),
       mask = ranked(operand(op, 2)), weight = ranked(operand(op, 3));
  for (unsigned i = 0; i < 4; ++i)
    if (auto type = ranked(operand(op, i)); type && type.getRank() != 4)
      return op->emitOpError("DeformConv2D expects X/offset/mask NCHW and W "
                             "OIHW rank-four tensors");
  auto bias = ranked(operand(op, 4));
  if (bias && bias.getRank() != 1)
    return op->emitOpError("DeformConv2D bias must be [output_channels]");
  if (!x || !weight)
    return success();
  int64_t cin = x.getDimSize(1), cout = weight.getDimSize(0);
  if ((!ShapedType::isDynamic(cin) &&
       (cin <= 0 || cin % groups || cin % deformGroups)) ||
      (!ShapedType::isDynamic(cout) && (cout <= 0 || cout % groups)))
    return op->emitOpError(
        "DeformConv2D channels must be positive and divisible by their groups");
  if (!weight.isDynamicDim(1)) {
    auto grouped = shapeProduct(weight.getDimSize(1), groups);
    if (!grouped || !compatibleDim(cin, *grouped))
      return op->emitOpError(
          "DeformConv2D input channels disagree with grouped weights");
  }
  if (bias && !compatibleDim(bias.getDimSize(0), cout))
    return op->emitOpError("DeformConv2D bias must be [output_channels]");
  SmallVector<int64_t> shape{x.getDimSize(0), cout, ShapedType::kDynamic,
                             ShapedType::kDynamic};
  for (unsigned d = 0; d < 2; ++d) {
    int64_t kernel = weight.getDimSize(d + 2), size = x.getDimSize(d + 2);
    if (!ShapedType::isDynamic(kernel) && kernel <= 0)
      return op->emitOpError("DeformConv2D kernel extents must be positive");
    if (ShapedType::isDynamic(kernel) || ShapedType::isDynamic(size))
      continue;
    auto span = shapeProduct(dilation[d], kernel - 1);
    auto pad = shapeProduct(2, padding[d]);
    if (!span || *span == INT64_MAX || !pad || size > INT64_MAX - *pad)
      return op->emitOpError("DeformConv2D spatial extent overflows int64");
    int64_t padded = size + *pad, effectiveKernel = *span + 1;
    if (padded < effectiveKernel)
      return op->emitOpError("DeformConv2D kernel exceeds padded input");
    shape[d + 2] = (padded - effectiveKernel) / stride[d] + 1;
  }
  int64_t offsetChannels = ShapedType::kDynamic;
  int64_t maskChannels = ShapedType::kDynamic;
  if (!weight.isDynamicDim(2) && !weight.isDynamicDim(3)) {
    auto area = shapeProduct(weight.getDimSize(2), weight.getDimSize(3));
    auto count = area ? shapeProduct(*area, deformGroups) : std::nullopt;
    auto doubled = count ? shapeProduct(*count, 2) : std::nullopt;
    if (!doubled)
      return op->emitOpError(
          "DeformConv2D offset channel count overflows int64");
    maskChannels = *count;
    offsetChannels = *doubled;
  }
  for (auto [tensor, channels] :
       {std::pair{offset, offsetChannels}, std::pair{mask, maskChannels}}) {
    if (!tensor)
      continue;
    if (!compatibleDim(tensor.getDimSize(1), channels))
      return op->emitOpError("DeformConv2D offset/mask channels must be "
                             "2*deform_groups*kH*kW / deform_groups*kH*kW");
    for (unsigned d : {0u, 2u, 3u})
      if (!compatibleDim(tensor.getDimSize(d), shape[d]))
        return op->emitOpError(
            "DeformConv2D offset/mask batch or spatial dimensions disagree");
  }
  return assignType(op, RankedTensorType::get(shape, x.getElementType()),
                    update);
}

LogicalResult inferMSDeformAttn(Operation *op, bool update) {
  bool offsets = op->getName().getStringRef() == "gac.ms_deform_attn_offsets";
  unsigned count = offsets ? 6 : 4;
  if (failed(customInputs(op, count, count)))
    return failure();
  int64_t version, align, step;
  if (failed(intParameter(op, "version", 1, version)) ||
      failed(intParameter(op, "align_corners", 0, align)) ||
      failed(intParameter(op, "im2col_step", 64, step)) ||
      failed(exactString(op, "mode", "bilinear")) ||
      failed(exactString(op, "padding_mode", "zeros")))
    return failure();
  if (version != 1 || align != 0 || step <= 0)
    return op->emitOpError("MSDeformAttn requires version=1, align_corners=0 "
                           "and positive im2col_step");
  for (unsigned i : {0u, 2u, 3u})
    if (!cast<TensorType>(operand(op, i).getType()).getElementType().isF32())
      return op->emitOpError(
          "MSDeformAttn value/locations/weights require FP32");
  if (offsets)
    for (unsigned i : {4u, 5u})
      if (!cast<TensorType>(operand(op, i).getType()).getElementType().isF32())
        return op->emitOpError(
            "MSDeformAttn normalizer/reference require FP32");
  if (!cast<TensorType>(operand(op, 1).getType())
           .getElementType()
           .isSignlessInteger(64))
    return op->emitOpError("MSDeformAttn spatial_shapes requires i64");
  auto value = ranked(operand(op, 0)), spatial = ranked(operand(op, 1)),
       locations = ranked(operand(op, 2)), weights = ranked(operand(op, 3));
  for (auto [type, rank] : {std::pair{value, 4}, std::pair{spatial, 2},
                            std::pair{locations, 6}, std::pair{weights, 5}})
    if (type && type.getRank() != rank)
      return op->emitOpError(
          "MSDeformAttn expects value[B,S,H,D], shapes[L,2], "
          "locations[B,Q,H,L,P,2], weights[B,Q,H,L,P]");
  if (offsets) {
    // coordinate_shape is the logical shape at the original Add, possibly
    // [B,Q,H,L,P/Z,Z,2]. Both surrounding reshapes preserve flat element order.
    // Div broadcasts in locations shape; Add broadcasts in coordinate_shape.
    auto coordinate = op->getAttrOfType<DenseI64ArrayAttr>("coordinate_shape");
    auto normalizer = ranked(operand(op, 4)),
         reference = ranked(operand(op, 5));
    int64_t referenceFirst;
    if (failed(intParameter(op, "reference_first", 0, referenceFirst)))
      return failure();
    if (referenceFirst != 0 && referenceFirst != 1)
      return op->emitOpError("reference_first must be zero or one");
    if (!coordinate || !locations || !locations.hasStaticShape() ||
        !normalizer || !normalizer.hasStaticShape() || !reference ||
        !reference.hasStaticShape())
      return op->emitOpError("fused coordinate operands require static shapes "
                             "and a coordinate_shape attribute");
    if (locations.getEncoding() || normalizer.getEncoding() ||
        reference.getEncoding())
      return op->emitOpError(
          "fused coordinate operands require dense row-major "
          "layout without an encoding");
    auto countElements =
        [&](ArrayRef<int64_t> dimensions) -> std::optional<int64_t> {
      int64_t count = 1;
      for (int64_t dimension : dimensions) {
        auto next = shapeProduct(count, dimension);
        if (dimension <= 0 || !next)
          return std::nullopt;
        count = *next;
      }
      return count;
    };
    auto coordinates = countElements(coordinate.asArrayRef());
    auto elements = countElements(locations.getShape());
    if (!coordinates || !elements || *coordinates != *elements)
      return op->emitOpError("coordinate_shape must preserve the offsets' "
                             "positive, nonoverflowing element count");
    SmallVector<int64_t> merged;
    if (failed(
            broadcast(op, locations.getShape(), normalizer.getShape(), merged)))
      return failure();
    if (!llvm::equal(merged, locations.getShape()))
      return op->emitOpError("normalizer must broadcast into offsets shape");
    if (failed(broadcast(op, coordinate.asArrayRef(), reference.getShape(),
                         merged)))
      return failure();
    if (!llvm::equal(merged, coordinate.asArrayRef()))
      return op->emitOpError("reference must broadcast into coordinate_shape");
  }
  auto sizes = getConstantIntegers(operand(op, 1));
  if (spatial &&
      (!compatibleDim(spatial.getDimSize(1), 2) || spatial.getDimSize(0) == 0))
    return op->emitOpError(
        "MSDeformAttn spatial_shapes must be nonempty i64[L,2]");
  // Shape/Gather/Concat may not have folded at import verification yet. The
  // graph IR can express this chain; device legalization requires constants.
  if (sizes) {
    if (sizes->empty() || sizes->size() % 2)
      return op->emitOpError("MSDeformAttn spatial_shapes needs pairs of H,W");
    int64_t sequence = 0;
    for (size_t i = 0; i < sizes->size(); i += 2) {
      auto area = shapeProduct((*sizes)[i], (*sizes)[i + 1]);
      if (!area || (*sizes)[i] <= 0 || (*sizes)[i + 1] <= 0 ||
          sequence > INT64_MAX - *area)
        return op->emitOpError("MSDeformAttn spatial sizes must be positive "
                               "with nonoverflowing total area");
      sequence += *area;
    }
    if (value && !compatibleDim(value.getDimSize(1), sequence))
      return op->emitOpError(
          "MSDeformAttn value sequence length must equal sum(H_l*W_l)");
  }
  if (locations) {
    if (!compatibleDim(locations.getDimSize(5), 2) ||
        (spatial &&
         !compatibleDim(locations.getDimSize(3), spatial.getDimSize(0))))
      return op->emitOpError("MSDeformAttn locations require final dimension 2 "
                             "and matching levels");
    if (value &&
        (!compatibleDim(value.getDimSize(0), locations.getDimSize(0)) ||
         !compatibleDim(value.getDimSize(2), locations.getDimSize(2))))
      return op->emitOpError("MSDeformAttn batch/head dimensions disagree");
    if (weights)
      for (unsigned d = 0; d < 5; ++d)
        if (!compatibleDim(weights.getDimSize(d), locations.getDimSize(d)))
          return op->emitOpError(
              "MSDeformAttn weight/sampling dimensions disagree");
  }
  if (!value || !locations)
    return success();
  int64_t heads = value.getDimSize(2), channels = value.getDimSize(3);
  int64_t outputChannels = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(heads) && !ShapedType::isDynamic(channels)) {
    auto count = shapeProduct(heads, channels);
    if (!count || heads <= 0 || channels <= 0)
      return op->emitOpError("MSDeformAttn head/channel count must be positive "
                             "and nonoverflowing");
    outputChannels = *count;
  }
  return assignType(
      op,
      RankedTensorType::get(
          {value.getDimSize(0), locations.getDimSize(1), outputChannels},
          value.getElementType()),
      update);
}
} // namespace

LogicalResult inferOperation(Operation *op, bool update) {
  StringRef name = op->getName().getStringRef();
  if (!name.starts_with("gac.") || name == "gac.unsupported")
    return success();
  if (name == "gac.split" && op->getNumResults() == 0)
    return op->emitOpError("Split requires at least one tensor result");
  if (op->getNumResults() != 1 && name != "gac.split")
    return op->emitOpError("requires one tensor result");
  if (name == "gac.constant") {
    auto attr = op->getAttrOfType<DenseElementsAttr>("value");
    if (!attr)
      return op->emitOpError("requires a DenseElementsAttr named 'value'");
    if (op->getNumOperands())
      return op->emitOpError("constant cannot have operands");
    return assignType(op, cast<TensorType>(attr.getType()), update);
  }
  if (op->getNumOperands() == 0)
    return op->emitOpError("requires input tensor operands");
  for (Value input : op->getOperands())
    if (!isa<TensorType>(input.getType()))
      return op->emitOpError("requires tensor operands");
  if (name == "gac.deform_conv2d")
    return inferDeformConv(op, update);
  if (name == "gac.ms_deform_attn" || name == "gac.ms_deform_attn_offsets")
    return inferMSDeformAttn(op, update);
  auto first = cast<TensorType>(op->getOperand(0).getType());
  Type element = first.getElementType();
  auto a = ranked(operand(op, 0));
  auto b = ranked(operand(op, 1));
  SmallVector<int64_t> shape;
  auto finish = [&]() {
    return assignType(op, RankedTensorType::get(shape, element), update);
  };
  auto require = [&](unsigned min, unsigned max) -> LogicalResult {
    if (op->getNumOperands() < min || op->getNumOperands() > max)
      return op->emitOpError("invalid operand count; expected ")
             << min << ".." << max;
    return success();
  };

  if (name == "gac.squeeze_if_unit_dim") {
    if (failed(require(1, 1)))
      return failure();
    auto attribute = op->getAttrOfType<IntegerAttr>("axis");
    if (!attribute)
      return op->emitOpError("shape-guarded Squeeze requires integer axis");
    if (!a)
      return success();
    int64_t axis = attribute.getInt();
    if (failed(normalizedAxis(op, axis, a.getRank())))
      return failure();
    // An unknown extent can produce either rank. Do not infer a rank from the
    // ONNX branch declaration or speculate that the Squeeze branch is taken.
    if (a.isDynamicDim(axis))
      return success();
    shape.assign(a.getShape().begin(), a.getShape().end());
    if (a.getDimSize(axis) == 1)
      shape.erase(shape.begin() + axis);
    return finish();
  }

  if (name == "gac.split") {
    if (failed(require(1, 2)))
      return failure();
    for (Value result : op->getResults()) {
      auto type = dyn_cast<TensorType>(result.getType());
      if (!type || type.getElementType() != element)
        return op->emitOpError(
            "Split results must preserve the input element type");
    }
    if (b && (b.getRank() != 1 || !b.getElementType().isSignlessInteger(64)))
      return op->emitOpError("Split sizes must be a rank-one i64 tensor");
    int64_t outputs = op->getNumResults();
    int64_t opset = integer(op, "gac.opset", 17);
    bool hasNumOutputs = op->hasAttr("num_outputs");
    if (hasNumOutputs &&
        (opset < 18 || integer(op, "num_outputs", 0) != outputs))
      return op->emitOpError(
          "Split num_outputs must match result count and requires opset 18");
    if (hasNumOutputs && operand(op, 1))
      return op->emitOpError("Split cannot specify both sizes and num_outputs");
    if (opset >= 18 && !hasNumOutputs && !operand(op, 1))
      return op->emitOpError("opset-18 Split requires sizes or num_outputs");
    if (!a)
      return success();
    int64_t axis = integer(op, "axis", 0);
    if (failed(normalizedAxis(op, axis, a.getRank())))
      return failure();
    auto sizes = getConstantIntegers(operand(op, 1));
    if (b && !b.isDynamicDim(0) && b.getDimSize(0) != outputs)
      return op->emitOpError("Split sizes length must equal result count");
    int64_t extent = a.getDimSize(axis);
    if (sizes) {
      if (int64_t(sizes->size()) != outputs)
        return op->emitOpError("Split sizes length must equal result count");
      int64_t total = 0;
      for (int64_t size : *sizes) {
        if (size < 0 || size > std::numeric_limits<int64_t>::max() - total)
          return op->emitOpError(
              "Split sizes must be nonnegative and not overflow");
        total += size;
      }
      if (!ShapedType::isDynamic(extent) && total != extent)
        return op->emitOpError("Split sizes must sum to the input axis extent");
    } else if (!operand(op, 1) && !ShapedType::isDynamic(extent)) {
      if (!hasNumOutputs) {
        if (extent % outputs)
          return op->emitOpError(
              "equal Split requires a divisible axis extent");
        sizes = SmallVector<int64_t>(outputs, extent / outputs);
      } else {
        int64_t chunk = extent / outputs + (extent % outputs != 0);
        // ONNX requires the final chunk to be the smaller one, rather than
        // silently inventing extra empty results for an invalid partition.
        if (chunk && outputs - 1 > extent / chunk)
          return op->emitOpError(
              "Split num_outputs produces an invalid final chunk");
        sizes = SmallVector<int64_t>(outputs, chunk);
        sizes->back() = extent - chunk * (outputs - 1);
      }
    }
    for (int64_t index = 0; index < outputs; ++index) {
      shape.assign(a.getShape().begin(), a.getShape().end());
      shape[axis] = sizes ? (*sizes)[index] : ShapedType::kDynamic;
      if (failed(assignType(op, RankedTensorType::get(shape, element), update,
                            index)))
        return failure();
    }
    return success();
  }

  if (name == "gac.nonzero") {
    if (failed(require(1, 1)))
      return failure();
    element = IntegerType::get(op->getContext(), 64);
    int64_t count = ShapedType::kDynamic;
    Operation *definition = operand(op, 0).getDefiningOp();
    if (definition && definition->getName().getStringRef() == "gac.constant") {
      if (auto value =
              definition->getAttrOfType<DenseIntElementsAttr>("value")) {
        count = 0;
        if (value.isSplat())
          count = value.getSplatValue<APInt>().isZero()
                      ? 0
                      : value.getNumElements();
        else
          for (const APInt &item : value.getValues<APInt>())
            count += !item.isZero();
      } else if (auto value =
                     definition->getAttrOfType<DenseFPElementsAttr>("value")) {
        count = 0;
        if (value.isSplat())
          count = value.getSplatValue<APFloat>().isZero()
                      ? 0
                      : value.getNumElements();
        else
          for (const APFloat &item : value.getValues<APFloat>())
            count += !item.isZero();
      }
    }
    // A runtime input only determines the first dimension. The number of
    // nonzero elements is data dependent, and must remain dynamic until a
    // legitimate constant fold or a dynamic-shape runtime resolves it.
    shape = {a ? a.getRank() : ShapedType::kDynamic, count};
    return finish();
  }

  if (name == "gac.tile") {
    if (failed(require(2, 2)))
      return failure();
    if (b && (b.getRank() != 1 || !b.getElementType().isSignlessInteger(64)))
      return op->emitOpError("Tile repeats must be a rank-one i64 tensor");
    if (!a)
      return success();
    if (b && !b.isDynamicDim(0) && b.getDimSize(0) != a.getRank())
      return op->emitOpError("Tile repeats length must equal input rank");
    auto repeats = getConstantIntegers(operand(op, 1));
    shape.assign(a.getRank(), ShapedType::kDynamic);
    if (repeats) {
      if (int64_t(repeats->size()) != a.getRank())
        return op->emitOpError("Tile repeats length must equal input rank");
      for (int64_t axis = 0; axis < a.getRank(); ++axis) {
        int64_t repeat = (*repeats)[axis], extent = a.getDimSize(axis);
        if (repeat < 0)
          return op->emitOpError("Tile repeats must be nonnegative");
        if (repeat == 0 || extent == 0)
          shape[axis] = 0;
        else if (!ShapedType::isDynamic(extent)) {
          if (extent > std::numeric_limits<int64_t>::max() / repeat)
            return op->emitOpError("Tile result dimension overflows i64");
          shape[axis] = extent * repeat;
        }
      }
    }
    return finish();
  }

  if (name == "gac.scatter_nd") {
    if (failed(require(3, 3)))
      return failure();
    auto updates = ranked(operand(op, 2));
    if (cast<TensorType>(operand(op, 2).getType()).getElementType() != element)
      return op->emitOpError("ScatterND data and updates element types differ");
    if (b && (b.getRank() < 1 || !b.getElementType().isSignlessInteger(64)))
      return op->emitOpError(
          "ScatterND indices require rank >= 1 and i64 elements");
    StringRef reduction = "none";
    if (auto attr = op->getAttrOfType<StringAttr>("reduction"))
      reduction = attr.getValue();
    int64_t opset = integer(op, "gac.opset", 17);
    if (reduction != "none" && reduction != "add" && reduction != "mul" &&
        reduction != "min" && reduction != "max")
      return op->emitOpError("unknown ScatterND reduction");
    if ((reduction != "none" && opset < 16) ||
        ((reduction == "min" || reduction == "max") && opset < 18))
      return op->emitOpError(
          "ScatterND reduction is unavailable in this opset");
    if (!a)
      return success();
    if (a.getRank() < 1)
      return op->emitOpError("ScatterND data rank must be at least one");
    if (b && !b.isDynamicDim(b.getRank() - 1)) {
      int64_t depth = b.getShape().back();
      if (depth > a.getRank())
        return op->emitOpError("ScatterND index depth exceeds data rank");
      SmallVector<int64_t> expected(b.getShape().drop_back());
      expected.append(a.getShape().begin() + depth, a.getShape().end());
      if (updates) {
        if (size_t(updates.getRank()) != expected.size())
          return op->emitOpError(
              "ScatterND updates rank disagrees with index slices");
        for (auto [actual, wanted] : llvm::zip(updates.getShape(), expected))
          if (!compatibleDim(actual, wanted))
            return op->emitOpError(
                "ScatterND updates shape disagrees with index slices");
      }
    }
    shape.assign(a.getShape().begin(), a.getShape().end());
    return finish();
  }

  bool binary =
      llvm::StringSwitch<bool>(name)
          .Cases("gac.add", "gac.sub", "gac.mul", "gac.div", true)
          .Cases("gac.pow", "gac.greater", "gac.less", "gac.add_relu", true)
          .Cases("gac.equal", "gac.and", "gac.max", true)
          .Default(false);
  if (binary || name == "gac.where") {
    if (failed(
            require(name == "gac.where" ? 3 : 2, name == "gac.where" ? 3 : 2)))
      return failure();
    if (name == "gac.where") {
      if (!first.getElementType().isInteger(1))
        return op->emitOpError("Where condition must be boolean");
      a = ranked(operand(op, 1));
      b = ranked(operand(op, 2));
      element = cast<TensorType>(operand(op, 1).getType()).getElementType();
    }
    if (!a || !b)
      return success();
    if (a.getElementType() != b.getElementType())
      return op->emitOpError("binary input element types differ");
    if (failed(broadcast(op, a.getShape(), b.getShape(), shape)))
      return failure();
    if (name == "gac.where") {
      auto condition = ranked(operand(op, 0));
      if (!condition)
        return success();
      SmallVector<int64_t> merged;
      if (failed(broadcast(op, shape, condition.getShape(), merged)))
        return failure();
      shape = std::move(merged);
    }
    if (name == "gac.and" && !element.isInteger(1))
      return op->emitOpError("And inputs must be boolean");
    if (name == "gac.greater" || name == "gac.less" || name == "gac.equal")
      element = IntegerType::get(op->getContext(), 1);
    return finish();
  }
  if (name == "gac.matmul" || name == "gac.matmul_bias" ||
      name == "gac.matmul_bias_relu" || name == "gac.matmul_bias_gelu") {
    bool fused = name != "gac.matmul";
    if (failed(require(fused ? 3 : 2, fused ? 3 : 2)))
      return failure();
    if (name == "gac.matmul_bias")
      for (Value input : op->getOperands())
        if (!cast<TensorType>(input.getType()).getElementType().isF32())
          return op->emitOpError("MatMulBias currently requires FP32 operands");
    if (Attribute order = op->getAttr("bias_first")) {
      auto value = dyn_cast<IntegerAttr>(order);
      if (!fused || !value || (value.getInt() != 0 && value.getInt() != 1))
        return op->emitOpError(
            "bias_first must be integer zero or one on a fused MatMul");
    }
    if (!a || !b)
      return success();
    if (a.getElementType() != b.getElementType())
      return op->emitOpError("MatMul element types differ");
    if (failed(matmulShape(op, a, b, shape)))
      return failure();
    if (fused) {
      auto bias = ranked(operand(op, 2));
      if (!bias)
        return success();
      if (bias.getElementType() != element)
        return op->emitOpError("bias element type differs");
      SmallVector<int64_t> merged;
      if (failed(broadcast(op, shape, bias.getShape(), merged)))
        return failure();
      if (merged.size() != shape.size())
        return op->emitOpError("bias must not expand the MatMul output domain");
      for (size_t axis = 0; axis < shape.size(); ++axis)
        if (!ShapedType::isDynamic(shape[axis]) && merged[axis] != shape[axis])
          return op->emitOpError(
              "bias must not expand the MatMul output domain");
      shape = std::move(merged);
    }
    return finish();
  }
  if (name == "gac.gemm") {
    if (failed(require(2, 3)))
      return failure();
    if (!a || !b)
      return success();
    if (a.getRank() != 2 || b.getRank() != 2)
      return op->emitOpError("Gemm requires rank-two inputs");
    bool ta = integer(op, "transA", 0), tb = integer(op, "transB", 0);
    if (!compatibleDim(a.getDimSize(ta ? 0 : 1), b.getDimSize(tb ? 1 : 0)))
      return op->emitOpError("Gemm reduction dimensions differ");
    shape = {a.getDimSize(ta ? 1 : 0), b.getDimSize(tb ? 0 : 1)};
    if (auto bias = ranked(operand(op, 2))) {
      SmallVector<int64_t> merged;
      if (failed(broadcast(op, shape, bias.getShape(), merged)))
        return failure();
      if (merged != shape)
        return op->emitOpError("Gemm bias must broadcast into M x N");
    }
    return finish();
  }
  if (name == "gac.shape") {
    if (failed(require(1, 1)))
      return failure();
    element = IntegerType::get(op->getContext(), 64);
    if (!a) {
      shape = {ShapedType::kDynamic};
      return finish();
    }
    int64_t rank = a.getRank();
    auto clamp = [rank](int64_t i) {
      return std::clamp<int64_t>(i < 0 ? i + rank : i, 0, rank);
    };
    shape = {std::max<int64_t>(0, clamp(integer(op, "end", rank)) -
                                      clamp(integer(op, "start", 0)))};
    return finish();
  }
  if (name == "gac.constant_of_shape") {
    if (failed(require(1, 1)))
      return failure();
    auto requested = getConstantIntegers(operand(op, 0));
    if (!requested)
      return success();
    for (int64_t dim : *requested)
      if (dim < 0)
        return op->emitOpError("negative ConstantOfShape dimension");
    shape = *requested;
    auto value = op->getAttrOfType<DenseElementsAttr>("value");
    if (value && value.getNumElements() != 1)
      return op->emitOpError(
          "ConstantOfShape value must contain exactly one element");
    element =
        value ? value.getElementType() : Float32Type::get(op->getContext());
    return finish();
  }
  if (name == "gac.range") {
    if (failed(require(3, 3)))
      return failure();
    auto integerType = dyn_cast<IntegerType>(element);
    if (!(element.isF32() || element.isF64() ||
          (integerType && !integerType.isUnsigned() &&
           (integerType.getWidth() == 16 || integerType.getWidth() == 32 ||
            integerType.getWidth() == 64))))
      return op->emitOpError("Range requires FP32/FP64 or signed i16/i32/i64");
    for (Value input : op->getOperands()) {
      auto type = ranked(input);
      if (cast<TensorType>(input.getType()).getElementType() != element ||
          (type && type.getRank() != 0))
        return op->emitOpError(
            "Range inputs must be scalars with the same element type");
    }
    shape = {ShapedType::kDynamic};
    auto start = getConstantIntegers(operand(op, 0));
    auto limit = getConstantIntegers(operand(op, 1));
    auto delta = getConstantIntegers(operand(op, 2));
    if (start && limit && delta && start->size() == 1 && limit->size() == 1 &&
        delta->size() == 1) {
      if ((*delta)[0] == 0)
        return op->emitOpError("Range delta must not be zero");
      // A signed i64 interval can span 2^64-1. Compute its exact ceiling in
      // wider integers, never host floating point (whose precision varies by
      // platform). Handle the empty direction before division/conversion.
      APInt begin = APInt(64, (*start)[0], true).sext(128);
      APInt end = APInt(64, (*limit)[0], true).sext(128);
      APInt step = APInt(64, (*delta)[0], true).sext(128);
      bool ascending = !step.isNegative();
      APInt size(128, 0);
      if (ascending ? begin.slt(end) : begin.sgt(end)) {
        APInt distance = ascending ? end - begin : begin - end;
        APInt stride = ascending ? step : -step;
        size = (distance + stride - 1).udiv(stride);
      }
      if (size.getActiveBits() > 63)
        return op->emitOpError("Range length overflows i64");
      shape[0] = size.getZExtValue();
    }
    return finish();
  }
  if (!a)
    return success();
  shape.assign(a.getShape().begin(), a.getShape().end());

  if (name == "gac.cast") {
    if (failed(require(1, 1)))
      return failure();
    element = onnxElementType(op->getContext(), integer(op, "to", -1));
    if (!element)
      return op->emitOpError("unsupported ONNX Cast destination type");
    return finish();
  }
  if (name == "gac.transpose") {
    if (failed(require(1, 1)))
      return failure();
    auto permutation = getI64ArrayAttr(op, "perm");
    if (!op->hasAttr("perm")) {
      permutation.resize(a.getRank());
      std::iota(permutation.rbegin(), permutation.rend(), 0);
    }
    if (permutation.size() != shape.size())
      return op->emitOpError("Transpose permutation rank mismatch");
    llvm::SmallSet<int64_t, 8> seen;
    for (auto [index, axis] : llvm::enumerate(permutation)) {
      if (axis < 0 || axis >= a.getRank() || !seen.insert(axis).second)
        return op->emitOpError(
            "Transpose permutation must contain each axis once");
      shape[index] = a.getDimSize(axis);
    }
    return finish();
  }
  if (name == "gac.reshape") {
    if (failed(require(1, 2)))
      return failure();
    auto requested = constantParameter(op, 1, "shape");
    if (!requested)
      return success();
    shape = *requested;
    int64_t allowZero = integer(op, "allowzero", 0);
    if (allowZero != 0 && allowZero != 1)
      return op->emitOpError("Reshape allowzero must be 0 or 1");
    int64_t inferredIndex = -1;
    bool hasLiteralZero = false;
    for (auto [index, dimension] : llvm::enumerate(shape)) {
      if (dimension == 0 && !allowZero) {
        if (index >= static_cast<size_t>(a.getRank()))
          return op->emitOpError("Reshape zero axis exceeds input rank");
        shape[index] = a.getDimSize(index);
      } else if (dimension == -1) {
        if (inferredIndex >= 0)
          return op->emitOpError("Reshape has multiple inferred dimensions");
        inferredIndex = index;
        // ONNX uses -1 for inference; MLIR's dynamic dimension is a distinct
        // sentinel (INT64_MIN in LLVM 20). Never multiply or store ONNX -1 as
        // a RankedTensorType extent.
        shape[index] = ShapedType::kDynamic;
      } else if (dimension < -1)
        return op->emitOpError("invalid Reshape dimension");
      else if (dimension == 0)
        hasLiteralZero = true;
    }
    if (hasLiteralZero && inferredIndex >= 0)
      return op->emitOpError("Reshape allowzero=1 cannot combine zero and -1");

    // Count only known nonnegative extents. Zero makes a product known even if
    // another axis is dynamic; otherwise unknown axes keep it unresolved. A
    // separate optional avoids confusing an unknown product with ONNX's -1.
    auto checkedProduct =
        [&](ArrayRef<int64_t> dimensions,
            std::optional<int64_t> &product) -> LogicalResult {
      if (llvm::is_contained(dimensions, int64_t(0))) {
        product = 0;
        return success();
      }
      if (llvm::any_of(dimensions, ShapedType::isDynamic)) {
        product.reset();
        return success();
      }
      int64_t count = 1;
      for (int64_t dimension : dimensions) {
        if (dimension < 0 ||
            count > std::numeric_limits<int64_t>::max() / dimension)
          return op->emitOpError("Reshape dimension product overflows i64");
        count *= dimension;
      }
      product = count;
      return success();
    };
    SmallVector<int64_t> specifiedDimensions;
    for (auto [index, dimension] : llvm::enumerate(shape))
      if (static_cast<int64_t>(index) != inferredIndex)
        specifiedDimensions.push_back(dimension);
    std::optional<int64_t> inputElements, specifiedElements;
    if (failed(checkedProduct(a.getShape(), inputElements)) ||
        failed(checkedProduct(specifiedDimensions, specifiedElements)))
      return failure();

    if (inferredIndex >= 0) {
      if (specifiedElements && *specifiedElements == 0)
        return op->emitOpError(
            "Reshape cannot infer a unique dimension from a zero product");
      if (inputElements && specifiedElements) {
        if (*inputElements % *specifiedElements)
          return op->emitOpError(
              "Reshape cannot infer a unique integral dimension");
        shape[inferredIndex] = *inputElements / *specifiedElements;
      }
    } else if (inputElements && specifiedElements &&
               *inputElements != *specifiedElements)
      return op->emitOpError("Reshape changes element count");
    return finish();
  }
  if (name == "gac.expand") {
    if (failed(require(1, 2)))
      return failure();
    auto requested = constantParameter(op, 1, "shape");
    if (!requested)
      return success();
    if (llvm::any_of(*requested, [](int64_t d) { return d < 0; }))
      return op->emitOpError("negative Expand shape");
    if (failed(broadcast(op, a.getShape(), *requested, shape)))
      return failure();
    return finish();
  }
  if (name == "gac.concat") {
    int64_t axis = integer(op, "axis", 0);
    if (failed(normalizedAxis(op, axis, a.getRank())))
      return failure();
    for (Value input : op->getOperands().drop_front()) {
      auto type = ranked(input);
      if (!type)
        return success();
      if (type.getRank() != a.getRank())
        return op->emitOpError("Concat ranks differ");
      for (int64_t d = 0; d < a.getRank(); ++d)
        if (d != axis && !compatibleDim(shape[d], type.getDimSize(d)))
          return op->emitOpError("Concat non-axis dimensions differ");
      shape[axis] =
          ShapedType::isDynamic(shape[axis]) || type.isDynamicDim(axis)
              ? ShapedType::kDynamic
              : shape[axis] + type.getDimSize(axis);
    }
    return finish();
  }
  if (name == "gac.gather") {
    if (failed(require(2, 2)))
      return failure();
    if (!b)
      return success();
    int64_t axis = integer(op, "axis", 0);
    if (failed(normalizedAxis(op, axis, a.getRank())))
      return failure();
    shape.assign(a.getShape().begin(), a.getShape().begin() + axis);
    shape.append(b.getShape().begin(), b.getShape().end());
    shape.append(a.getShape().begin() + axis + 1, a.getShape().end());
    return finish();
  }
  if (name == "gac.unsqueeze" || name == "gac.squeeze") {
    if (failed(require(1, 2)))
      return failure();
    auto axes = constantParameter(op, 1, "axes");
    if (operand(op, 1) && !axes)
      return success();
    if (name == "gac.unsqueeze" && !axes)
      return op->emitOpError("Unsqueeze requires axes");
    llvm::SmallSet<int64_t, 8> selected;
    int64_t rank = a.getRank() + (name == "gac.unsqueeze" ? axes->size() : 0);
    if (axes)
      for (int64_t axis : *axes) {
        if (failed(normalizedAxis(op, axis, rank)))
          return failure();
        if (!selected.insert(axis).second)
          return op->emitOpError("duplicate squeeze/unsqueeze axis");
      }
    shape.clear();
    if (name == "gac.unsqueeze") {
      int64_t input = 0;
      for (int64_t i = 0; i < rank; ++i)
        shape.push_back(selected.count(i) ? 1 : a.getDimSize(input++));
    } else {
      if (!axes && !a.hasStaticShape())
        return success();
      for (int64_t i = 0; i < rank; ++i) {
        bool erase = axes ? selected.count(i) : a.getDimSize(i) == 1;
        if (erase && !compatibleDim(a.getDimSize(i), 1))
          return op->emitOpError("Squeeze axis must have size one");
        if (!erase)
          shape.push_back(a.getDimSize(i));
      }
    }
    return finish();
  }
  if (name == "gac.reduce_sum" || name == "gac.reduce_mean") {
    if (failed(require(1, 2)))
      return failure();
    auto axes = constantParameter(op, 1, "axes");
    if (operand(op, 1) && !axes)
      return success();
    if (!axes || axes->empty()) {
      if (integer(op, "noop_with_empty_axes", 0))
        return finish();
      axes = SmallVector<int64_t>(a.getRank());
      std::iota(axes->begin(), axes->end(), 0);
    }
    llvm::SmallSet<int64_t, 8> reduced;
    for (int64_t axis : *axes) {
      if (failed(normalizedAxis(op, axis, a.getRank())))
        return failure();
      if (!reduced.insert(axis).second)
        return op->emitOpError("duplicate reduction axis");
    }
    shape.clear();
    for (int64_t i = 0; i < a.getRank(); ++i)
      if (!reduced.count(i))
        shape.push_back(a.getDimSize(i));
      else if (integer(op, "keepdims", 1))
        shape.push_back(1);
    return finish();
  }
  if (name == "gac.conv2d" || name == "gac.conv2d_relu" ||
      name == "gac.max_pool2d") {
    bool pool = name == "gac.max_pool2d";
    if (failed(require(pool ? 1 : 2, pool ? 1 : 3)))
      return failure();
    if (a.getRank() != 4 || (!pool && b && b.getRank() != 4))
      return op->emitOpError(
          "requires NCHW input and OIHW convolution weights");
    if (!pool && !b)
      return success();
    auto kernel = getI64ArrayAttr(op, "kernel_shape");
    if (!pool && kernel.empty())
      kernel = {b.getDimSize(2), b.getDimSize(3)};
    auto strides = getI64ArrayAttr(op, "strides");
    if (strides.empty())
      strides = {1, 1};
    auto dilations = getI64ArrayAttr(op, "dilations");
    if (dilations.empty())
      dilations = {1, 1};
    auto pads = getI64ArrayAttr(op, "pads");
    if (pads.empty())
      pads = {0, 0, 0, 0};
    if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2 ||
        pads.size() != 4)
      return op->emitOpError("invalid 2D kernel/stride/dilation/padding rank");
    StringRef autoPad = "NOTSET";
    if (auto value = op->getAttrOfType<StringAttr>("auto_pad"))
      autoPad = value.getValue();
    if (autoPad != "NOTSET" && autoPad != "VALID" && autoPad != "SAME_UPPER" &&
        autoPad != "SAME_LOWER")
      return op->emitOpError("unknown auto_pad mode");
    if (!pool) {
      int64_t group = integer(op, "group", 1);
      if (group <= 0)
        return op->emitOpError("group must be positive");
      if (!a.isDynamicDim(1) && !b.isDynamicDim(1) &&
          a.getDimSize(1) != b.getDimSize(1) * group)
        return op->emitOpError(
            "input channels disagree with weight channels times group");
      if (!b.isDynamicDim(0) && b.getDimSize(0) % group)
        return op->emitOpError("output channels must divide into groups");
      if (auto bias = ranked(operand(op, 2)))
        if (bias.getRank() != 1 ||
            !compatibleDim(bias.getDimSize(0), b.getDimSize(0)))
          return op->emitOpError("Conv bias must be [output_channels]");
      shape[1] = b.getDimSize(0);
      for (unsigned i = 0; i < 2; ++i)
        if (!compatibleDim(kernel[i], b.getDimSize(i + 2)))
          return op->emitOpError(
              "kernel_shape differs from convolution weight shape");
    }
    for (unsigned i = 0; i < 2; ++i) {
      if (strides[i] <= 0 || dilations[i] <= 0 ||
          (!ShapedType::isDynamic(kernel[i]) && kernel[i] <= 0))
        return op->emitOpError("kernel, stride and dilation must be positive");
      if (a.isDynamicDim(i + 2) || ShapedType::isDynamic(kernel[i])) {
        shape[i + 2] = ShapedType::kDynamic;
        continue;
      }
      int64_t size = a.getDimSize(i + 2), stride = strides[i];
      if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER")
        shape[i + 2] = (size + stride - 1) / stride;
      else {
        int64_t before = autoPad == "VALID" ? 0 : pads[i],
                after = autoPad == "VALID" ? 0 : pads[i + 2];
        int64_t numerator =
            size + before + after - dilations[i] * (kernel[i] - 1) - 1;
        double extent = static_cast<double>(numerator) / stride;
        shape[i + 2] = static_cast<int64_t>(pool && integer(op, "ceil_mode", 0)
                                                ? std::ceil(extent)
                                                : std::floor(extent)) +
                       1;
        if (pool && integer(op, "ceil_mode", 0) &&
            (shape[i + 2] - 1) * stride >= size + before)
          --shape[i + 2];
        if (shape[i + 2] < 0)
          return op->emitOpError("kernel produces a negative spatial extent");
      }
    }
    return finish();
  }
  if (name == "gac.grid_sample") {
    if (failed(require(2, 2)))
      return failure();
    if (!b)
      return success();
    if (a.getRank() != 4 || b.getRank() != 4 ||
        !compatibleDim(b.getDimSize(3), 2) ||
        !compatibleDim(a.getDimSize(0), b.getDimSize(0)))
      return op->emitOpError(
          "2D GridSample expects X[N,C,H,W] and grid[N,Ho,Wo,2]");
    shape = {a.getDimSize(0), a.getDimSize(1), b.getDimSize(1),
             b.getDimSize(2)};
    return finish();
  }
  if (name == "gac.resize") {
    if (!operand(op, 0) || op->getNumOperands() > 4)
      return op->emitOpError("Resize requires X and optional roi/scales/sizes");
    for (unsigned index : {1u, 2u, 3u}) {
      if (auto input = operand(op, index)) {
        auto type = cast<TensorType>(input.getType());
        if ((index == 3 && !type.getElementType().isSignlessInteger(64)) ||
            (index == 2 && !type.getElementType().isF32()) ||
            (index == 1 && !isa<FloatType>(type.getElementType())))
          return op->emitOpError(
              "Resize sizes must be i64, scales f32, and roi floating point");
        if (auto rank = dyn_cast<RankedTensorType>(type);
            rank && rank.getRank() != 1)
          return op->emitOpError("Resize roi/scales/sizes must have rank one");
      }
    }
    auto sizes = getConstantIntegers(operand(op, 3));
    if (sizes && sizes->empty())
      sizes.reset();
    auto axes = getI64ArrayAttr(op, "axes");
    if (op->hasAttr("axes") && axes.empty())
      return op->emitOpError("Resize axes must be a nonempty integer array");
    if (axes.empty()) {
      axes.resize(a.getRank());
      std::iota(axes.begin(), axes.end(), 0);
    }
    llvm::SmallSet<int64_t, 8> seenAxes;
    for (int64_t &axis : axes)
      if (failed(normalizedAxis(op, axis, a.getRank())))
        return failure();
      else if (!seenAxes.insert(axis).second)
        return op->emitOpError("Resize axes must be unique");
    Value scalesValue = operand(op, 2);
    auto scalesOp = scalesValue ? scalesValue.getDefiningOp() : nullptr;
    auto scales = scalesOp
                      ? scalesOp->getAttrOfType<DenseFPElementsAttr>("value")
                      : DenseFPElementsAttr();
    if (sizes && scales && scales.getNumElements())
      return op->emitOpError(
          "Resize cannot supply both nonempty scales and sizes");
    if (sizes) {
      if (sizes->size() != axes.size())
        return op->emitOpError("Resize sizes/axes rank mismatch");
      for (auto [index, axis] : llvm::enumerate(axes)) {
        if ((*sizes)[index] < 0)
          return op->emitOpError("negative Resize extent");
        shape[axis] = (*sizes)[index];
      }
      return finish();
    }
    if (!scales)
      return success();
    if (static_cast<size_t>(scales.getNumElements()) != axes.size())
      return op->emitOpError("Resize scales/axes rank mismatch");
    for (auto [index, value] : llvm::enumerate(scales.getValues<APFloat>())) {
      double scale = value.convertToDouble();
      if (!(scale > 0) || !std::isfinite(scale))
        return op->emitOpError("Resize scale must be finite and positive");
      int64_t axis = axes[index];
      if (!a.isDynamicDim(axis)) {
        double extent = std::floor(a.getDimSize(axis) * scale);
        if (!std::isfinite(extent) || extent >= std::ldexp(1.0, 63))
          return op->emitOpError("Resize extent overflows int64");
        shape[axis] = static_cast<int64_t>(extent);
      }
    }
    return finish();
  }
  if (name == "gac.slice") {
    auto starts = constantParameter(op, 1, "starts"),
         ends = constantParameter(op, 2, "ends");
    if (!starts || !ends)
      return success();
    auto axes = constantParameter(op, 3, "axes"),
         steps = constantParameter(op, 4, "steps");
    if ((operand(op, 3) && !axes) || (operand(op, 4) && !steps))
      return success();
    if (!axes) {
      axes = SmallVector<int64_t>(starts->size());
      std::iota(axes->begin(), axes->end(), 0);
    }
    if (!steps)
      steps = SmallVector<int64_t>(starts->size(), 1);
    if (starts->size() != ends->size() || starts->size() != axes->size() ||
        starts->size() != steps->size())
      return op->emitOpError("Slice parameter lengths differ");
    llvm::SmallSet<int64_t, 8> seen;
    for (unsigned i = 0; i < starts->size(); ++i) {
      int64_t axis = (*axes)[i], step = (*steps)[i];
      if (failed(normalizedAxis(op, axis, a.getRank())))
        return failure();
      if (!step || !seen.insert(axis).second)
        return op->emitOpError("Slice needs nonzero steps and unique axes");
      int64_t size = a.getDimSize(axis);
      if (ShapedType::isDynamic(size))
        continue;
      auto bound = [size, step](int64_t v, bool end) {
        if (step < 0 && end && v == std::numeric_limits<int64_t>::min())
          return int64_t(-1);
        if (v < 0)
          v = v < -size ? -1 : v + size;
        return std::clamp<int64_t>(v, step < 0 ? -1 : 0,
                                   step < 0 ? size - 1 : size);
      };
      int64_t start = bound((*starts)[i], false), end = bound((*ends)[i], true);
      long double length =
          std::ceil((static_cast<long double>(end) - start) / step);
      shape[axis] = std::max<int64_t>(0, static_cast<int64_t>(length));
    }
    return finish();
  }
  if (name == "gac.softmax" || name == "gac.layer_norm") {
    if (failed(require(1, name == "gac.layer_norm" ? 3 : 1)))
      return failure();
    int64_t axis = integer(op, "axis", -1);
    if (failed(normalizedAxis(op, axis, a.getRank())))
      return failure();
    if (name == "gac.layer_norm") {
      for (Value input : op->getOperands().drop_front()) {
        auto parameter = ranked(input);
        if (!parameter)
          continue;
        if (parameter.getElementType() != element)
          return op->emitOpError("LayerNorm Scale/B element type must match X");
        SmallVector<int64_t> merged;
        if (failed(broadcast(op, a.getShape(), parameter.getShape(), merged)))
          return failure();
        if (merged.size() != shape.size())
          return op->emitOpError(
              "LayerNorm Scale/B cannot add dimensions to X");
        for (auto [expected, actual] : llvm::zip(shape, merged))
          if (!compatibleDim(expected, actual))
            return op->emitOpError(
                "LayerNorm Scale/B must broadcast into X without expanding it");
      }
    }
    return finish();
  }
  if (name == "gac.batch_norm") {
    if (failed(require(5, 5)))
      return failure();
    if (a.getRank() < 2)
      return op->emitOpError("BatchNorm requires a channel axis");
    for (Value parameter : op->getOperands().drop_front())
      if (auto type = ranked(parameter))
        if (type.getRank() != 1 ||
            !compatibleDim(type.getDimSize(0), a.getDimSize(1)))
          return op->emitOpError(
              "BatchNorm parameters must match the channel axis");
    return finish();
  }
  if (name == "gac.clamp") {
    if (failed(require(1, 3)))
      return failure();
    for (Value bound : op->getOperands().drop_front())
      if (auto type = ranked(bound))
        if (type.getRank() != 0 || type.getElementType() != element)
          return op->emitOpError(
              "Clip min/max must be scalars with the data element type");
    return finish();
  }
  // Remaining registered operations are shape-preserving unary tensor math.
  if (failed(require(1, 1)))
    return failure();
  return finish();
}
} // namespace gac
