#include "gac/ConstantFolding.h"
#include "gac/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <memory>
#include <numeric>

using namespace mlir;
namespace gac {
namespace {
constexpr auto roundNearest = APFloat::rmNearestTiesToEven;

Value operand(Operation *op, unsigned position) {
  auto positions = getI64ArrayAttr(op, "gac.input_positions");
  if (positions.empty())
    return position < op->getNumOperands() ? op->getOperand(position) : Value();
  for (auto [index, source] : llvm::enumerate(positions))
    if (source == position && index < op->getNumOperands())
      return op->getOperand(index);
  return {};
}

DenseElementsAttr constant(Value value) {
  auto *op = value ? value.getDefiningOp() : nullptr;
  return op && op->getName().getStringRef() == "gac.constant"
             ? op->getAttrOfType<DenseElementsAttr>("value")
             : DenseElementsAttr();
}

int64_t integerAttr(Operation *op, StringRef name, int64_t fallback) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  return attr ? attr.getInt() : fallback;
}

std::optional<int64_t> elements(ArrayRef<int64_t> shape, int64_t maxElements) {
  int64_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0 || (dim && count > maxElements / dim))
      return std::nullopt;
    count *= dim;
  }
  return count;
}

SmallVector<int64_t> coordinates(int64_t flat, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result(shape.size());
  for (size_t axis = shape.size(); axis; --axis) {
    result[axis - 1] = flat % shape[axis - 1];
    flat /= shape[axis - 1];
  }
  return result;
}

int64_t linear(ArrayRef<int64_t> coords, ArrayRef<int64_t> shape) {
  int64_t index = 0;
  for (auto [coord, size] : llvm::zip(coords, shape))
    index = index * size + coord;
  return index;
}

int64_t broadcast(ArrayRef<int64_t> coords, ArrayRef<int64_t> shape) {
  int64_t index = 0, offset = coords.size() - shape.size();
  for (auto [axis, size] : llvm::enumerate(shape))
    index = index * size + (size == 1 ? 0 : coords[offset + axis]);
  return index;
}

// Scalars are carried as raw APInt bits. Float operations reconstruct APFloat
// with the tensor's exact semantics: f64 is never evaluated as f32 and casts
// occur only at an explicit ONNX Cast. Indexing/transpose copies bits
// unchanged.
struct Tensor {
  DenseElementsAttr value;
  RankedTensorType type;
  std::shared_ptr<Tensor> source;
  StringRef view;
  SmallVector<int64_t> permutation;
  Tensor() = default;
  explicit Tensor(DenseElementsAttr value)
      : value(value), type(value ? dyn_cast<RankedTensorType>(value.getType())
                                 : RankedTensorType()) {}
  explicit operator bool() const {
    return (value || source) && type &&
           (isa<IntegerType>(type.getElementType()) ||
            isa<FloatType>(type.getElementType()));
  }
  Type element() const { return type.getElementType(); }
  int64_t count() const { return type.getNumElements(); }
  bool isSplat() const { return source ? source->isSplat() : value.isSplat(); }
  int64_t readCost() const {
    return source ? source->readCost() + (view == "gac.cast") : 0;
  }
  APInt at(int64_t index) const {
    if (source) {
      if (view == "gac.cast") {
        APFloat number(cast<FloatType>(source->element()).getFloatSemantics(),
                       source->at(index));
        bool losesInfo;
        number.convert(cast<FloatType>(element()).getFloatSemantics(),
                       roundNearest, &losesInfo);
        return number.bitcastToAPInt();
      }
      if (view == "gac.expand" || view == "gac.tile" ||
          view == "gac.transpose") {
        auto coord = coordinates(index, type.getShape());
        if (view == "gac.expand")
          index = broadcast(coord, source->type.getShape());
        else {
          SmallVector<int64_t> src(source->type.getRank());
          for (size_t d = 0; d < src.size(); ++d)
            if (view == "gac.tile")
              src[d] = coord[d] % source->type.getDimSize(d);
            else
              src[permutation[d]] = coord[d];
          index = linear(src, source->type.getShape());
        }
      }
      return source->at(index);
    }
    if (isa<FloatType>(element()))
      return value.getValues<APFloat>()[index].bitcastToAPInt();
    return value.getValues<APInt>()[index];
  }
};

// A repeated 4x4 camera matrix need not become a 120 MiB DenseElementsAttr.
// Read-only views retain the exact index transform and explicit FP cast; only
// a consumer's bounded output is materialized. No activation-dependent value
// can enter this evaluator, and the recursion depth itself is bounded.
Tensor tensor(Value value, unsigned depth = 0) {
  if (auto attr = constant(value))
    return Tensor(attr);
  Operation *op = value ? value.getDefiningOp() : nullptr;
  if (!op || depth >= 16 || op->getNumResults() != 1 || !op->getNumOperands())
    return {};
  StringRef name = op->getName().getStringRef();
  bool layout = name == "gac.reshape" || name == "gac.squeeze" ||
                name == "gac.squeeze_if_unit_dim" || name == "gac.unsqueeze" ||
                name == "gac.identity" || name == "gac.expand" ||
                name == "gac.tile" || name == "gac.transpose";
  if (!layout && name != "gac.cast")
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || !elements(type.getShape(), INT64_MAX))
    return {};
  // Shape parameters still have to be real MLIR constants. This also prevents
  // treating a dynamically chosen reshape/repetition as a compile-time view.
  for (Value parameter : op->getOperands().drop_front())
    if (!constant(parameter))
      return {};
  Tensor source = tensor(op->getOperand(0), depth + 1);
  if (!source || (layout && source.element() != type.getElementType()))
    return {};
  if (name == "gac.cast" && (!isa<FloatType>(source.element()) ||
                             !isa<FloatType>(type.getElementType())))
    return {}; // Integer casts keep the checked conversion path below.
  if (failed(inferOperation(op, false)))
    return {};
  Tensor result;
  result.type = type;
  result.source = std::make_shared<Tensor>(std::move(source));
  result.view = name;
  if (name == "gac.transpose") {
    result.permutation = getI64ArrayAttr(op, "perm");
    if (!op->hasAttr("perm")) {
      result.permutation.resize(type.getRank());
      std::iota(result.permutation.rbegin(), result.permutation.rend(), 0);
    }
  }
  return result;
}

APFloat floating(const APInt &bits, Type type) {
  return APFloat(cast<FloatType>(type).getFloatSemantics(), bits);
}

bool nonzero(const APInt &bits, Type type) {
  return isa<FloatType>(type) ? !floating(bits, type).isZero() : !bits.isZero();
}

DenseElementsAttr dense(RankedTensorType type, ArrayRef<APInt> data) {
  if (auto fp = dyn_cast<FloatType>(type.getElementType())) {
    SmallVector<APFloat> values;
    values.reserve(data.size());
    for (const APInt &bits : data)
      values.emplace_back(fp.getFloatSemantics(), bits);
    return DenseElementsAttr::get(type, values);
  }
  return DenseElementsAttr::get(type, data);
}

FailureOr<APInt> convert(const APInt &bits, Type input, Type output) {
  if (input == output)
    return bits;
  if (output.isInteger(1))
    return APInt(1, nonzero(bits, input));
  auto srcInt = dyn_cast<IntegerType>(input);
  auto dstInt = dyn_cast<IntegerType>(output);
  if (srcInt && dstInt)
    return srcInt.isUnsigned() || srcInt.getWidth() == 1
               ? bits.zextOrTrunc(dstInt.getWidth())
               : bits.sextOrTrunc(dstInt.getWidth());
  if (srcInt) {
    APFloat value(cast<FloatType>(output).getFloatSemantics());
    value.convertFromAPInt(bits, !srcInt.isUnsigned() && srcInt.getWidth() != 1,
                           roundNearest);
    return value.bitcastToAPInt();
  }
  APFloat value = floating(bits, input);
  bool exact = false;
  if (dstInt) {
    llvm::APSInt integer(dstInt.getWidth(), dstInt.isUnsigned());
    auto status =
        value.convertToInteger(integer, APFloat::rmTowardZero, &exact);
    if (status & APFloat::opInvalidOp)
      return failure();
    return APInt(integer);
  }
  value.convert(cast<FloatType>(output).getFloatSemantics(), roundNearest,
                &exact);
  return value.bitcastToAPInt();
}

FailureOr<APInt> binary(StringRef operation, const APInt &x, const APInt &y,
                        Type type) {
  if (auto fp = dyn_cast<FloatType>(type)) {
    APFloat a = floating(x, type), b = floating(y, type);
    auto comparison = a.compare(b);
    if (operation == "gac.equal")
      return APInt(1, comparison == APFloat::cmpEqual);
    if (operation == "gac.less")
      return APInt(1, comparison == APFloat::cmpLessThan);
    if (operation == "gac.greater")
      return APInt(1, comparison == APFloat::cmpGreaterThan);
    if (operation == "gac.add")
      a.add(b, roundNearest);
    else if (operation == "gac.sub")
      a.subtract(b, roundNearest);
    else if (operation == "gac.mul")
      a.multiply(b, roundNearest);
    else if (operation == "gac.div")
      a.divide(b, roundNearest);
    else if (operation == "gac.max") {
      // Preserve NaNs rather than using host std::max's operand-order behavior.
      if (a.isNaN())
        return x;
      if (b.isNaN())
        return y;
      if (comparison == APFloat::cmpLessThan ||
          (a.isZero() && b.isZero() && a.isNegative()))
        return y;
    } else
      return failure();
    return a.bitcastToAPInt();
  }
  auto integer = cast<IntegerType>(type);
  bool isUnsigned = integer.isUnsigned() || integer.getWidth() == 1;
  if (operation == "gac.equal")
    return APInt(1, x == y);
  if (operation == "gac.less")
    return APInt(1, isUnsigned ? x.ult(y) : x.slt(y));
  if (operation == "gac.greater")
    return APInt(1, isUnsigned ? x.ugt(y) : x.sgt(y));
  if (operation == "gac.and")
    return x & y;
  if (operation == "gac.max")
    return (isUnsigned ? x.ugt(y) : x.sgt(y)) ? x : y;
  bool overflow = false;
  APInt value;
  if (operation == "gac.add")
    value = isUnsigned ? x.uadd_ov(y, overflow) : x.sadd_ov(y, overflow);
  else if (operation == "gac.sub")
    value = isUnsigned ? x.usub_ov(y, overflow) : x.ssub_ov(y, overflow);
  else if (operation == "gac.mul")
    value = isUnsigned ? x.umul_ov(y, overflow) : x.smul_ov(y, overflow);
  else if (operation == "gac.div") {
    if (y.isZero() || (!isUnsigned && x.isMinSignedValue() && y.isAllOnes()))
      return failure();
    value = isUnsigned ? x.udiv(y) : x.sdiv(y);
  } else
    return failure();
  return overflow ? FailureOr<APInt>(failure()) : FailureOr<APInt>(value);
}

FailureOr<bool> replace(Operation *op, IRRewriter &rewriter,
                        ArrayRef<DenseElementsAttr> values,
                        ConstantFoldBudget &budget, int64_t work) {
  int64_t bytes = 0;
  for (auto value : values)
    bytes += value.isSplat()
                 ? value.getElementType().getIntOrFloatBitWidth() / 8 + 1
                 : value.getRawData().size();
  if (bytes > budget.bytes || work > budget.scalarOperations)
    return false;
  budget.bytes -= bytes;
  budget.scalarOperations -= work;
  rewriter.setInsertionPoint(op);
  SmallVector<Value> replacements;
  auto names = op->getAttrOfType<ArrayAttr>("gac.output_names");
  for (auto [index, value] : llvm::enumerate(values)) {
    OperationState state(op->getLoc(), "gac.constant");
    state.addTypes(value.getType());
    state.addAttribute("value", value);
    if (auto attr = op->getAttr("gac.source_name"))
      state.addAttribute("gac.source_name", attr);
    if (names && index < names.size())
      state.addAttribute("gac.output_names",
                         rewriter.getArrayAttr({names[index]}));
    replacements.push_back(rewriter.create(state)->getResult(0));
  }
  rewriter.replaceOp(op, replacements);
  return true;
}

std::optional<SmallVector<int64_t>> parameter(Operation *op, unsigned position,
                                              StringRef attribute) {
  if (Value value = operand(op, position))
    return getConstantIntegers(value);
  if (op->hasAttr(attribute))
    return getI64ArrayAttr(op, attribute);
  return std::nullopt;
}
} // namespace

FailureOr<bool> foldConstantOperation(Operation *op, IRRewriter &rewriter,
                                      ConstantFoldBudget &budget) {
  StringRef name = op->getName().getStringRef();
  if (!name.starts_with("gac.") || name == "gac.constant" ||
      name == "gac.unsupported" || !op->getNumResults())
    return false;
  bool supported =
      llvm::StringSwitch<bool>(name)
          .Cases("gac.constant_of_shape", "gac.reshape", "gac.squeeze", true)
          .Cases("gac.unsqueeze", "gac.identity", "gac.cast", true)
          .Case("gac.squeeze_if_unit_dim", true)
          .Cases("gac.expand", "gac.tile", "gac.transpose", "gac.slice", true)
          .Cases("gac.concat", "gac.split", "gac.gather", "gac.nonzero", true)
          .Cases("gac.scatter_nd", "gac.reduce_sum", "gac.reduce_mean", true)
          .Cases("gac.matmul", "gac.where", "gac.clamp", true)
          .Cases("gac.add", "gac.sub", "gac.mul", "gac.div", true)
          .Cases("gac.equal", "gac.less", "gac.greater", true)
          .Cases("gac.and", "gac.max", true)
          .Case("gac.range", true)
          .Default(false);
  if (!supported)
    return false;
  // Check real MLIR constants, not names or the model's input values. No
  // activation dependent on a function argument can enter this evaluator.
  SmallVector<Tensor> inputs;
  for (Value value : op->getOperands()) {
    Tensor input = tensor(value);
    if (!input)
      return false;
    inputs.push_back(input);
  }
  if (inputs.empty())
    return false;
  SmallVector<RankedTensorType> types;
  int64_t total = 0;
  for (Type type : op->getResultTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() ||
        (!isa<IntegerType>(tensor.getElementType()) &&
         !isa<FloatType>(tensor.getElementType())))
      return false;
    auto count = elements(tensor.getShape(), budget.maxElements);
    if (!count || total > budget.maxElements - *count)
      return false;
    total += *count;
    types.push_back(tensor);
  }
  // An APFloat reduction/MatMul is deliberately not fused: each source multiply
  // and add rounds in the source element type. Never silently use host FMA.
  int64_t costPerElement = 1;
  for (const Tensor &input : inputs)
    costPerElement += input.readCost();
  if (total > budget.scalarOperations / costPerElement ||
      total > budget.bytes / 8)
    return false;
  int64_t work = total * costPerElement;
  // Keep large constant-only layout/cast nodes as views until a real consumer
  // folds. CSE/DCE subsequently removes the now-unused view chain. This cap
  // is independent of the opt-in base output budget and matches the previous
  // default materialization bound.
  if (total > (4LL << 20) && tensor(op->getResult(0)))
    return false;
  Tensor first = inputs.front();
  RankedTensorType out = types.front();
  Type element = out.getElementType();
  SmallVector<DenseElementsAttr> results;
  SmallVector<APInt> data;
  if (name != "gac.constant_of_shape")
    data.reserve(total);
  auto emitError = [&]() -> FailureOr<bool> {
    op->emitOpError("invalid constant arithmetic/index or integer overflow");
    return failure();
  };
  auto output = [&]() -> FailureOr<bool> {
    results.push_back(dense(out, data));
    return replace(op, rewriter, results, budget, work);
  };
  auto firstShape = first.type.getShape();

  if (name == "gac.range") {
    auto integerType = dyn_cast<IntegerType>(element);
    // Floating Range needs its own length/rounding contract. Do not evaluate
    // it through double or silently claim a runtime template exists.
    if (!integerType || integerType.isUnsigned() || inputs.size() != 3)
      return false;
    for (const Tensor &input : inputs)
      if (input.type.getRank() != 0 || input.element() != element)
        return emitError();
    unsigned width = integerType.getWidth();
    if (width != 16 && width != 32 && width != 64)
      return false;
    APInt value = first.at(0).sext(128);
    APInt limit = inputs[1].at(0).sext(128);
    APInt step = inputs[2].at(0).sext(128);
    if (step.isZero())
      return emitError();
    auto inRange = [&](const APInt &current) {
      return step.isNegative() ? current.sgt(limit) : current.slt(limit);
    };
    // Shape inference already established the exact count. These checks also
    // guard standalone folding from inconsistent types. Widened arithmetic
    // permits a last step beyond i64 without wrapping an emitted element.
    for (int64_t i = 0; i < total; ++i) {
      if (!inRange(value) || !value.isSignedIntN(width))
        return emitError();
      data.push_back(value.trunc(width));
      value += step;
    }
    if (inRange(value))
      return emitError();
    return output();
  }
  if (name == "gac.constant_of_shape") {
    Tensor fill(op->getAttrOfType<DenseElementsAttr>("value"));
    APInt value(element.getIntOrFloatBitWidth(), 0);
    if (fill) {
      if (fill.value.getNumElements() != 1 || fill.element() != element)
        return false;
      value = fill.at(0);
    }
    if (total)
      data.push_back(value); // DenseElementsAttr stores a splat, not N zeros.
    work = 1;
    return output();
  }
  if (name == "gac.reshape" || name == "gac.squeeze" ||
      name == "gac.squeeze_if_unit_dim" || name == "gac.unsqueeze" ||
      name == "gac.identity") {
    if (first.count() != total || first.element() != element)
      return false;
    if (first.value) {
      results.push_back(first.value.reshape(out));
      return replace(op, rewriter, results, budget, 1);
    }
    for (int64_t i = 0; i < total; ++i)
      data.push_back(first.at(i));
    return output();
  }
  if (name == "gac.cast") {
    for (int64_t i = 0; i < total; ++i) {
      auto converted = convert(first.at(i), first.element(), element);
      if (failed(converted))
        return emitError();
      data.push_back(*converted);
    }
    return output();
  }
  if (name == "gac.expand" || name == "gac.tile" || name == "gac.transpose" ||
      name == "gac.slice") {
    auto perm = getI64ArrayAttr(op, "perm");
    SmallVector<int64_t> start(first.type.getRank(), 0),
        steps(first.type.getRank(), 1);
    if (name == "gac.transpose" && !op->hasAttr("perm")) {
      perm.resize(first.type.getRank());
      std::iota(perm.rbegin(), perm.rend(), 0);
    }
    if (name == "gac.slice") {
      auto begin = parameter(op, 1, "starts"), axes = parameter(op, 3, "axes"),
           stride = parameter(op, 4, "steps");
      if (!begin)
        return false;
      for (unsigned i = 0; i < begin->size(); ++i) {
        int64_t axis = axes ? (*axes)[i] : i;
        if (axis < 0)
          axis += first.type.getRank();
        if (axis < 0 || axis >= first.type.getRank())
          return emitError();
        int64_t size = firstShape[axis], value = (*begin)[i];
        steps[axis] = stride ? (*stride)[i] : 1;
        if (!steps[axis])
          return emitError();
        if (value < 0)
          value = value < -size ? -1 : value + size;
        start[axis] = std::clamp<int64_t>(value, steps[axis] < 0 ? -1 : 0,
                                          steps[axis] < 0 ? size - 1 : size);
      }
    }
    if (first.isSplat() && total) {
      data.push_back(first.at(0));
      work = 1;
      return output();
    }
    for (int64_t i = 0; i < total; ++i) {
      auto coord = coordinates(i, out.getShape());
      SmallVector<int64_t> src(firstShape.size());
      int64_t index = 0;
      if (name == "gac.expand")
        index = broadcast(coord, firstShape);
      else {
        for (unsigned axis = 0; axis < src.size(); ++axis) {
          if (name == "gac.transpose")
            src[perm[axis]] = coord[axis];
          else if (name == "gac.tile")
            src[axis] = coord[axis] % firstShape[axis];
          else
            src[axis] = start[axis] + coord[axis] * steps[axis];
        }
        index = linear(src, firstShape);
      }
      if (index < 0 || index >= first.count())
        return emitError();
      data.push_back(first.at(index));
    }
    return output();
  }
  if (name == "gac.concat") {
    int64_t axis = integerAttr(op, "axis", 0);
    if (axis < 0)
      axis += out.getRank();
    int64_t groups = 1;
    for (int64_t i = 0; i < axis; ++i)
      groups *= out.getDimSize(i);
    for (int64_t group = 0; group < groups; ++group)
      for (const Tensor &input : inputs) {
        int64_t chunk = groups ? input.count() / groups : 0;
        for (int64_t i = 0; i < chunk; ++i)
          data.push_back(input.at(group * chunk + i));
      }
    return output();
  }
  if (name == "gac.split") {
    int64_t axis = integerAttr(op, "axis", 0), offset = 0;
    if (axis < 0)
      axis += first.type.getRank();
    for (RankedTensorType type : types) {
      data.clear();
      for (int64_t i = 0; i < type.getNumElements(); ++i) {
        auto coord = coordinates(i, type.getShape());
        coord[axis] += offset;
        data.push_back(first.at(linear(coord, firstShape)));
      }
      results.push_back(dense(type, data));
      offset += type.getDimSize(axis);
    }
    return replace(op, rewriter, results, budget, work);
  }
  if (name == "gac.gather") {
    Tensor indexes = tensor(operand(op, 1));
    if (!indexes || !isa<IntegerType>(indexes.element()))
      return false;
    int64_t axis = integerAttr(op, "axis", 0);
    if (axis < 0)
      axis += first.type.getRank();
    int64_t inner = 1, outer = 1;
    for (int64_t i = 0; i < axis; ++i)
      outer *= firstShape[i];
    for (int64_t i = axis + 1; i < first.type.getRank(); ++i)
      inner *= firstShape[i];
    for (int64_t group = 0; group < outer; ++group)
      for (int64_t i = 0; i < indexes.count(); ++i) {
        int64_t index = indexes.at(i).getSExtValue();
        if (index < 0)
          index += firstShape[axis];
        if (index < 0 || index >= firstShape[axis])
          return emitError();
        for (int64_t j = 0; j < inner; ++j)
          data.push_back(
              first.at((group * firstShape[axis] + index) * inner + j));
      }
    return output();
  }
  if (name == "gac.nonzero") {
    if (first.count() > budget.maxElements ||
        first.count() > budget.scalarOperations / (1 + first.readCost()))
      return false;
    SmallVector<int64_t> selected;
    for (int64_t i = 0; i < first.count(); ++i)
      if (nonzero(first.at(i), first.element()))
        selected.push_back(i);
    if (out.getRank() != 2 || out.getDimSize(0) != first.type.getRank() ||
        out.getDimSize(1) != static_cast<int64_t>(selected.size()))
      return false;
    for (int64_t axis = 0; axis < first.type.getRank(); ++axis)
      for (int64_t index : selected)
        data.emplace_back(64, coordinates(index, firstShape)[axis]);
    work = first.count() * (1 + first.readCost());
    return output();
  }
  if (name == "gac.scatter_nd") {
    Tensor indexes = tensor(operand(op, 1)), updates = tensor(operand(op, 2));
    auto reduction = op->getAttrOfType<StringAttr>("reduction");
    if (!indexes || !updates || (reduction && reduction.getValue() != "none"))
      return false;
    int64_t depth = indexes.type.getShape().back(), chunk = 1;
    if (depth < 1 || depth > first.type.getRank())
      return false;
    for (int64_t axis = depth; axis < first.type.getRank(); ++axis)
      chunk *= firstShape[axis];
    for (int64_t i = 0; i < total; ++i)
      data.push_back(first.at(i));
    llvm::SmallSet<int64_t, 32> destinations;
    for (int64_t tuple = 0; tuple < indexes.count() / depth; ++tuple) {
      int64_t dest = 0;
      for (int64_t axis = 0; axis < depth; ++axis) {
        int64_t index = indexes.at(tuple * depth + axis).getSExtValue();
        if (index < 0)
          index += firstShape[axis];
        if (index < 0 || index >= firstShape[axis])
          return emitError();
        dest = dest * firstShape[axis] + index;
      }
      // ONNX reduction=none requires distinct indices; avoid choosing a
      // sequential winner for a model whose parallel behavior is undefined.
      if (!destinations.insert(dest).second)
        return emitError();
      for (int64_t i = 0; i < chunk; ++i)
        data[dest * chunk + i] = updates.at(tuple * chunk + i);
    }
    return output();
  }
  if (name == "gac.reduce_sum" || name == "gac.reduce_mean") {
    // Half-precision reductions may have a wider accumulator contract. Do not
    // replace them until that contract is represented explicitly in the IR.
    if (isa<FloatType>(element) && !element.isF32() && !element.isF64())
      return false;
    auto axes = parameter(op, 1, "axes");
    if (!axes || axes->empty()) {
      if (integerAttr(op, "noop_with_empty_axes", 0)) {
        if (first.value) {
          results.push_back(first.value.reshape(out));
          return replace(op, rewriter, results, budget, work);
        }
        for (int64_t i = 0; i < total; ++i)
          data.push_back(first.at(i));
        return output();
      }
      axes = SmallVector<int64_t>(first.type.getRank());
      std::iota(axes->begin(), axes->end(), 0);
    }
    SmallVector<bool> reduced(first.type.getRank(), false);
    for (int64_t axis : *axes) {
      if (axis < 0)
        axis += first.type.getRank();
      if (axis < 0 || axis >= first.type.getRank())
        return emitError();
      reduced[axis] = true;
    }
    int64_t count = first.count();
    if (count > budget.maxElements ||
        count > budget.scalarOperations / (1 + first.readCost()))
      return false;
    data.assign(total, APInt(element.getIntOrFloatBitWidth(), 0));
    SmallVector<int64_t> counts(total, 0);
    bool keep = integerAttr(op, "keepdims", 1);
    for (int64_t i = 0; i < count; ++i) {
      auto coords = coordinates(i, firstShape);
      SmallVector<int64_t> dest;
      for (unsigned axis = 0; axis < coords.size(); ++axis)
        if (!reduced[axis])
          dest.push_back(coords[axis]);
        else if (keep)
          dest.push_back(0);
      int64_t index = linear(dest, out.getShape());
      auto sum = binary("gac.add", data[index], first.at(i), element);
      if (failed(sum))
        return emitError();
      data[index] = *sum;
      ++counts[index];
    }
    if (name == "gac.reduce_mean")
      for (int64_t i = 0; i < total; ++i) {
        auto denominator =
            convert(APInt(64, counts[i]), rewriter.getI64Type(), element);
        if (failed(denominator))
          return emitError();
        auto mean = binary("gac.div", data[i], *denominator, element);
        if (failed(mean))
          return emitError();
        data[i] = *mean;
      }
    work = count * (1 + first.readCost());
    return output();
  }
  if (name == "gac.matmul") {
    if (isa<FloatType>(element) && !element.isF32() && !element.isF64())
      return false;
    Tensor right = tensor(operand(op, 1));
    if (!right || first.type.getRank() < 2 || right.type.getRank() < 2)
      return false;
    int64_t k = firstShape.back();
    int64_t perMultiplyAdd = 2 + first.readCost() + right.readCost();
    // Fixed geometry uses small 4x4 transforms. Leave expensive constant
    // neural network MatMuls to device execution, even when inputs are static.
    if (k > budget.maxMatmulReduction ||
        (k && total > budget.maxMatmulMultiplyAdds / k) ||
        (k && total > budget.scalarOperations / perMultiplyAdd / k))
      return false;
    auto bShape = right.type.getShape();
    for (int64_t i = 0; i < total; ++i) {
      auto coords = coordinates(i, out.getShape());
      auto batch = ArrayRef<int64_t>(coords).drop_back(2);
      int64_t aBatch = broadcast(batch, firstShape.drop_back(2));
      int64_t bBatch = broadcast(batch, bShape.drop_back(2));
      int64_t m = coords[coords.size() - 2], n = coords.back();
      APInt sum(element.getIntOrFloatBitWidth(), 0);
      for (int64_t j = 0; j < k; ++j) {
        APInt a =
            first.at((aBatch * firstShape[firstShape.size() - 2] + m) * k + j);
        APInt b = right.at((bBatch * k + j) * bShape.back() + n);
        auto product = binary("gac.mul", a, b, element);
        if (failed(product))
          return emitError();
        auto next = binary("gac.add", sum, *product, element);
        if (failed(next))
          return emitError();
        sum = *next;
      }
      data.push_back(sum);
    }
    work = total * perMultiplyAdd * k;
    return output();
  }
  if (name == "gac.where") {
    Tensor yes = tensor(operand(op, 1)), no = tensor(operand(op, 2));
    if (!yes || !no)
      return false;
    for (int64_t i = 0; i < total; ++i) {
      auto coords = coordinates(i, out.getShape());
      bool condition =
          nonzero(first.at(broadcast(coords, firstShape)), first.element());
      const Tensor &selected = condition ? yes : no;
      data.push_back(selected.at(broadcast(coords, selected.type.getShape())));
    }
    return output();
  }
  if (name == "gac.add" || name == "gac.sub" || name == "gac.mul" ||
      name == "gac.div" || name == "gac.equal" || name == "gac.less" ||
      name == "gac.greater" || name == "gac.and" || name == "gac.max") {
    if (inputs.size() < 2)
      return false;
    for (int64_t i = 0; i < total; ++i) {
      auto coords = coordinates(i, out.getShape());
      APInt value = first.at(broadcast(coords, firstShape));
      for (const Tensor &other : ArrayRef<Tensor>(inputs).drop_front()) {
        auto result = binary(name, value,
                             other.at(broadcast(coords, other.type.getShape())),
                             first.element());
        if (failed(result))
          return emitError();
        value = *result;
      }
      data.push_back(value);
    }
    return output();
  }
  if (name == "gac.clamp") {
    Tensor min = tensor(operand(op, 1)), max = tensor(operand(op, 2));
    for (int64_t i = 0; i < total; ++i) {
      APInt value = first.at(i);
      if (min) {
        auto lt = binary("gac.less", value, min.at(0), element);
        if (failed(lt))
          return emitError();
        if (!lt->isZero())
          value = min.at(0);
      }
      if (max) {
        auto gt = binary("gac.greater", value, max.at(0), element);
        if (failed(gt))
          return emitError();
        if (!gt->isZero())
          value = max.at(0);
      }
      data.push_back(value);
    }
    return output();
  }
  return false;
}
} // namespace gac
