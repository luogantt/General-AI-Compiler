#include "gac/LinalgConvolution.h"
#include "gac/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include <limits>

using namespace mlir;

namespace gac {
namespace {

static Value extendToF32(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isF32())
    return value;
  return b.create<arith::ExtFOp>(loc, b.getF32Type(), value);
}

static Value zeroAccumulator(OpBuilder &b, Location loc,
                             RankedTensorType type) {
  Value empty = b.create<tensor::EmptyOp>(loc, type.getShape(),
                                         type.getElementType());
  SmallVector<utils::IteratorType> parallel(type.getRank(),
                                           utils::IteratorType::parallel);
  auto map = b.getMultiDimIdentityMap(type.getRank());
  return b.create<linalg::GenericOp>(
      loc, TypeRange{type}, ValueRange{}, ValueRange{empty},
      ArrayRef<AffineMap>{map}, parallel,
      [](OpBuilder &nested, Location at, ValueRange) {
        Value zero = nested.create<arith::ConstantFloatOp>(
            at, APFloat(0.0f), nested.getF32Type());
        nested.create<linalg::YieldOp>(at, zero);
      }).getResult(0);
}

/// Padding is a tensor operation, not a change to the convolution indexing
/// contract. The zero is local to the pad region (no scalar graph ABI input).
static Value padInput(OpBuilder &b, Location loc, Value input,
                      ArrayRef<int64_t> pads) {
  if (llvm::all_of(pads, [](int64_t amount) { return amount == 0; }))
    return input;
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape(inputType.getShape());
  shape[2] += pads[0] + pads[2];
  shape[3] += pads[1] + pads[3];
  auto type = RankedTensorType::get(shape, inputType.getElementType());
  SmallVector<int64_t> low{0, 0, pads[0], pads[1]};
  SmallVector<int64_t> high{0, 0, pads[2], pads[3]};
  auto pad = b.create<tensor::PadOp>(loc, type, input, low, high,
                                    ValueRange{}, ValueRange{});
  {
    OpBuilder::InsertionGuard guard(b);
    SmallVector<Type> argTypes(4, b.getIndexType());
    SmallVector<Location> argLocations(4, loc);
    b.createBlock(&pad.getRegion(), pad.getRegion().end(), argTypes,
                  argLocations);
    Value zero = b.create<arith::ConstantOp>(
        loc, inputType.getElementType(), b.getZeroAttr(inputType.getElementType()));
    b.create<tensor::YieldOp>(loc, zero);
  }
  return pad.getResult();
}

} // namespace

FailureOr<Value> lowerConvolutionToLinalg(OpBuilder &b, Operation *op) {
  auto fail = [&](StringRef reason) -> FailureOr<Value> {
    op->emitOpError() << "Linalg convolution: " << reason;
    return failure();
  };
  if (op->getNumOperands() < 2 || op->getNumOperands() > 3 ||
      op->getNumResults() != 1)
    return fail("requires input, weights, optional bias and one result");
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto weightType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto validType = [](RankedTensorType type) {
    return type && type.hasStaticShape() && type.getRank() == 4 &&
           (type.getElementType().isF16() || type.getElementType().isF32()) &&
           llvm::all_of(type.getShape(), [](int64_t d) {
             return d > 0 && d <= std::numeric_limits<int32_t>::max();
           });
  };
  if (!validType(inputType) || !validType(weightType) || !validType(resultType))
    return fail("requires positive static rank-4 FP16/FP32 tensors");
  if (inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != resultType.getElementType())
    return fail("input, weights and result element types must agree");
  if (auto mode = op->getAttrOfType<StringAttr>("auto_pad"))
    if (mode.getValue() != "NOTSET")
      return fail("requires explicit pads (auto_pad=NOTSET)");

  auto strides = getI64ArrayAttr(op, "strides");
  auto dilations = getI64ArrayAttr(op, "dilations");
  auto pads = getI64ArrayAttr(op, "pads");
  if (strides.empty()) strides = {1, 1};
  if (dilations.empty()) dilations = {1, 1};
  if (pads.empty()) pads = {0, 0, 0, 0};
  if (strides.size() != 2 || dilations.size() != 2 || pads.size() != 4 ||
      llvm::any_of(strides, [](int64_t n) { return n <= 0; }) ||
      llvm::any_of(dilations, [](int64_t n) { return n <= 0; }) ||
      llvm::any_of(pads, [](int64_t n) { return n < 0; }))
    return fail("requires two positive strides/dilations and four nonnegative pads");
  int64_t groups = 1;
  if (auto group = op->getAttrOfType<IntegerAttr>("group"))
    groups = group.getInt();
  auto x = inputType.getShape(), w = weightType.getShape();
  auto y = resultType.getShape();
  if (groups <= 0 || x[1] % groups || w[0] % groups ||
      x[1] / groups != w[1] || y[0] != x[0] || y[1] != w[0])
    return fail("input/output channels or batch disagree with grouped weights");
  for (unsigned axis = 0; axis < 2; ++axis) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (pads[axis] > limit || pads[axis + 2] > limit ||
        x[axis + 2] + pads[axis] + pads[axis + 2] > limit ||
        strides[axis] > limit || dilations[axis] > limit ||
        (w[axis + 2] - 1) > limit / dilations[axis])
      return fail("padded extent or kernel exceeds the 32-bit device index range");
    int64_t numerator = x[axis + 2] + pads[axis] + pads[axis + 2] -
                        dilations[axis] * (w[axis + 2] - 1) - 1;
    if (numerator < 0 || numerator / strides[axis] + 1 != y[axis + 2])
      return fail("result spatial shape disagrees with stride/dilation/padding");
  }
  bool hasBias = op->getNumOperands() == 3;
  if (hasBias) {
    auto bias = dyn_cast<RankedTensorType>(op->getOperand(2).getType());
    if (!bias || !bias.hasStaticShape() || bias.getRank() != 1 ||
        bias.getDimSize(0) != y[1] ||
        bias.getElementType() != inputType.getElementType())
      return fail("bias must have shape [Cout] and the input element type");
  }

  Location loc = op->getLoc();
  Value input = padInput(b, loc, op->getOperand(0), pads);
  auto accumulatorType = RankedTensorType::get(y, b.getF32Type());
  Value init = zeroAccumulator(b, loc, accumulatorType);

  // Four output axes identify one independent accumulator. The remaining
  // three axes are reductions; group selection is part of the input map.
  // This information is explicit to Linalg tiling/fusion, rather than hidden
  // inside a hand-written device loop.
  SmallVector<AffineExpr> d;
  for (unsigned i = 0; i < 7; ++i)
    d.push_back(b.getAffineDimExpr(i));
  AffineExpr channel = d[4];
  if (groups != 1)
    channel = d[1].floorDiv(w[0] / groups) * w[1] + d[4];
  auto inputMap = AffineMap::get(7, 0,
      {d[0], channel, d[2] * strides[0] + d[5] * dilations[0],
       d[3] * strides[1] + d[6] * dilations[1]}, b.getContext());
  auto weightMap = AffineMap::get(7, 0, {d[1], d[4], d[5], d[6]}, b.getContext());
  auto outputMap = AffineMap::get(7, 0, {d[0], d[1], d[2], d[3]}, b.getContext());
  SmallVector<utils::IteratorType> iterators(4, utils::IteratorType::parallel);
  iterators.append(3, utils::IteratorType::reduction);
  bool padded = llvm::any_of(pads, [](int64_t n) { return n != 0; });
  Value sum = b.create<linalg::GenericOp>(
      loc, TypeRange{accumulatorType},
      ValueRange{input, op->getOperand(1)}, ValueRange{init},
      ArrayRef<AffineMap>{inputMap, weightMap, outputMap}, iterators,
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value a = extendToF32(nested, at, args[0]);
        Value weight = extendToF32(nested, at, args[1]);
        Value product = nested.create<arith::MulFOp>(at, a, weight);
        Value next = nested.create<arith::AddFOp>(at, args[2], product);
        if (padded) {
          // Existing GAC semantics skip padded terms, including 0 * Inf/NaN.
          // The padded input guarantees bounds safety; this predicate also
          // preserves those non-finite edge cases without fast-math changes.
          auto index = [&](unsigned axis) -> Value {
            return nested.create<linalg::IndexOp>(at, axis);
          };
          auto constant = [&](int64_t value) -> Value {
            return nested.create<arith::ConstantIndexOp>(at, value);
          };
          Value valid;
          for (unsigned axis = 0; axis < 2; ++axis) {
            Value start = nested.create<arith::MulIOp>(
                at, index(2 + axis), constant(strides[axis]));
            Value kernel = nested.create<arith::MulIOp>(
                at, index(5 + axis), constant(dilations[axis]));
            Value coordinate = nested.create<arith::AddIOp>(at, start, kernel);
            Value low = nested.create<arith::CmpIOp>(
                at, arith::CmpIPredicate::sge, coordinate, constant(pads[axis]));
            Value high = nested.create<arith::CmpIOp>(
                at, arith::CmpIPredicate::slt, coordinate,
                constant(pads[axis] + x[2 + axis]));
            Value inside = nested.create<arith::AndIOp>(at, low, high);
            valid = valid ? Value(nested.create<arith::AndIOp>(at, valid, inside))
                          : inside;
          }
          next = nested.create<arith::SelectOp>(at, valid, next, args[2]);
        }
        nested.create<linalg::YieldOp>(at, next);
      }).getResult(0);

  bool relu = op->getName().getStringRef() == "gac.conv2d_relu";
  if (!hasBias && !relu && resultType == accumulatorType)
    return sum;

  // Bias and activation are an elementwise epilogue after the full reduction.
  // FP16 stores round once here; the accumulator and bias arithmetic stay FP32.
  Value empty = b.create<tensor::EmptyOp>(loc, y, resultType.getElementType());
  SmallVector<Value> inputs{sum};
  auto identity = b.getMultiDimIdentityMap(4);
  SmallVector<AffineMap> maps{identity};
  if (hasBias) {
    inputs.push_back(op->getOperand(2));
    maps.push_back(AffineMap::get(4, 0, {b.getAffineDimExpr(1)}, b.getContext()));
  }
  maps.push_back(identity);
  SmallVector<utils::IteratorType> parallel(4, utils::IteratorType::parallel);
  return b.create<linalg::GenericOp>(
      loc, TypeRange{resultType}, inputs, ValueRange{empty}, maps, parallel,
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value value = args[0];
        if (hasBias)
          value = nested.create<arith::AddFOp>(
              at, value, extendToF32(nested, at, args[1]));
        if (relu) {
          Value zero = nested.create<arith::ConstantFloatOp>(
              at, APFloat(0.0f), nested.getF32Type());
          Value negative = nested.create<arith::CmpFOp>(
              at, arith::CmpFPredicate::OLT, value, zero);
          value = nested.create<arith::SelectOp>(at, negative, zero, value);
        }
        if (resultType.getElementType().isF16())
          value = nested.create<arith::TruncFOp>(at, resultType.getElementType(), value);
        nested.create<linalg::YieldOp>(at, value);
      }).getResult(0);
}

} // namespace gac
