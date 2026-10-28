#include "gac/TritonBridge.h"
#include "gac/LinalgDevice.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ToolOutputFile.h"
#include <functional>
#include <limits>

using namespace mlir;

namespace gac {
namespace {

bool elementType(Type type) {
  return type.isF32() || type.isSignlessInteger(32);
}
bool payloadType(Type type) {
  return elementType(type) || type.isSignlessInteger(1);
}

// DeviceLowering elides dense Identity/Reshape/Squeeze/Unsqueeze nodes by
// assigning their result the source tensor's physical buffer ID.  LayerNorm's
// pointer ABI deliberately requires X, Scale and Bias to be three distinct
// buffers, so selection must reason about the same aliasing rule before a
// manifest is emitted.  Rejecting a view here is conservative when it is also
// a public graph output (and therefore would not be elided), but preserves the
// essential contract: a selected kernel can always pass all later validators.
Value denseViewAliasRoot(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    StringRef name = def->getName().getStringRef();
    if (name != "gac.identity" && name != "gac.reshape" &&
        name != "gac.squeeze" && name != "gac.unsqueeze")
      break;
    if (def->getNumResults() != 1 || def->getNumOperands() == 0 ||
        def->getResult(0) != value)
      break;
    auto source = dyn_cast<RankedTensorType>(def->getOperand(0).getType());
    auto result = dyn_cast<RankedTensorType>(value.getType());
    if (!source || !result || source.getEncoding() || result.getEncoding() ||
        source.getElementType() != result.getElementType() ||
        source.getNumElements() != result.getNumElements())
      break;
    value = def->getOperand(0);
  }
  return value;
}

bool arithmetic(StringRef name) {
  // Integer division/remainder are intentionally absent: a masked tail filled
  // with zero must not introduce division-by-zero in inactive tensor elements.
  return llvm::StringSwitch<bool>(name)
      .Cases("arith.constant", "arith.addf", "arith.subf", "arith.mulf", true)
      .Cases("arith.divf", "arith.negf", "arith.maximumf", "arith.minimumf",
             true)
      .Cases("arith.maxnumf", "arith.minnumf", "arith.cmpf", true)
      .Cases("arith.addi", "arith.subi", "arith.muli", "arith.cmpi", true)
      .Cases("arith.andi", "arith.ori", "arith.xori", "arith.select", true)
      .Cases("arith.maxsi", "arith.maxui", "arith.minsi", "arith.minui", true)
      .Default(false);
}

bool checkCandidate(Operation *op, RankedTensorType expected,
                    llvm::DenseSet<Operation *> &checked, std::string &reason) {
  auto reject = [&](StringRef message) {
    reason = message.str();
    return false;
  };
  auto current = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!current || !current.hasPureTensorSemantics() ||
      current.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      op->getNumRegions() != 1 || !llvm::hasSingleElement(op->getRegion(0)))
    return reject("requires a single-result tensor Linalg operation");
  if (!checked.insert(op).second)
    return true;
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!output || !output.hasStaticShape() || output.getEncoding() ||
      !elementType(output.getElementType()) ||
      output.getElementType() != expected.getElementType() ||
      llvm::any_of(output.getShape(), [](int64_t size) { return size <= 0; }) ||
      output.getShape() != expected.getShape())
    return reject(
        "requires positive static, unencoded FP32/i32 tensors of one shape");
  int64_t elements = 1;
  for (int64_t size : output.getShape()) {
    if (size > INT32_MAX / elements)
      return reject("flattened element count exceeds signed 32-bit indexing");
    elements *= size;
  }
  if (current.payloadUsesValueFromOperand(current.getDpsInitOperand(0)))
    return reject("destination initializer is read; this bridge does not lower "
                  "reductions");
  if (llvm::any_of(current.getIteratorTypesArray(),
                   [](utils::IteratorType type) {
                     return type != utils::IteratorType::parallel;
                   }))
    return reject("reduction or nonparallel iterator requires native lowering");
  for (OpOperand &operand : op->getOpOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.get().getType());
    if (!type || type.getShape() != output.getShape() || type.getEncoding() ||
        type.getElementType() != output.getElementType())
      return reject("broadcast, layout conversion or unsupported tensor type");
    AffineMap map = current.getMatchingIndexingMap(&operand);
    if (map.getNumDims() != unsigned(output.getRank()) || map.getNumSymbols() ||
        map.getNumResults() != unsigned(output.getRank()))
      return reject("requires identity indexing maps for every operand");
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto dimension = dyn_cast<AffineDimExpr>(expression);
          dimension && dimension.getPosition() == axis)
        continue;
      // Linalg's broadcast map canonicalizes a size-one axis to zero even
      // when every operand has exactly the output shape. It is byte-identical
      // to the corresponding identity axis, not an actual broadcast.
      auto constant = dyn_cast<AffineConstantExpr>(expression);
      if (!constant || constant.getValue() != 0 || output.getDimSize(axis) != 1)
        return reject("requires identity indexing maps for every operand");
    }
  }
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  if (!captures.empty())
    return reject("Linalg payload captures external SSA values");
  Block &body = op->getRegion(0).front();
  if (body.getNumArguments() != op->getNumOperands())
    return reject("Linalg payload arguments disagree with its operand ABI");
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0).getType() != output.getElementType())
    return reject("Linalg payload must yield one scalar output element");
  for (Operation &nested : body.without_terminator()) {
    if (!arithmetic(nested.getName().getStringRef()) ||
        nested.getNumRegions() || nested.getNumResults() != 1 ||
        llvm::any_of(nested.getOperandTypes(),
                     [](Type t) { return !payloadType(t); }) ||
        !payloadType(nested.getResult(0).getType()))
      return reject("payload contains arithmetic outside the supported "
                    "Triton/Vortex subset");
    if (auto constant = dyn_cast<arith::ConstantOp>(nested))
      if (!isa<IntegerAttr, FloatAttr>(constant.getValue()))
        return reject("payload constants must be scalar integer or "
                      "floating-point values");
  }
  // Fusion was selected by LinalgDevice. Do not silently drop an unsupported
  // producer from this dispatch, or its tensor would lack allocated storage.
  for (OpOperand *operand : current.getDpsInputOperands())
    if (Operation *producer = operand->get().getDefiningOp();
        isFusedLinalgProducer(producer) &&
        !checkCandidate(producer, expected, checked, reason))
      return false;
  return true;
}

// TTIR is a textual interchange boundary between GAC's LLVM20 and Triton's
// independently built MLIR. This small emitter dialect is private to a separate
// context; it never enables unregistered operations in the GAC graph context.
struct PointerStorage : TypeStorage {
  using KeyTy = Type;
  explicit PointerStorage(Type element) : element(element) {}
  bool operator==(const KeyTy &key) const { return element == key; }
  static PointerStorage *construct(TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<PointerStorage>()) PointerStorage(key);
  }
  Type element;
};
class Pointer : public Type::TypeBase<Pointer, Type, PointerStorage> {
public:
  using Base::Base;
  static constexpr llvm::StringLiteral name = "tt.ptr";
  static Pointer get(Type element) {
    return Base::get(element.getContext(), element);
  }
  Type getElementType() const { return getImpl()->element; }
};
class Return : public Op<Return, OpTrait::ZeroOperands, OpTrait::ZeroResults,
                         OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tt.return"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
};
class ReduceReturn
    : public Op<ReduceReturn, OpTrait::OneOperand, OpTrait::ZeroResults,
                OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "tt.reduce.return"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
};
class InterchangeDialect : public Dialect {
public:
  static StringRef getDialectNamespace() { return "tt"; }
  explicit InterchangeDialect(MLIRContext *context)
      : Dialect("tt", context, TypeID::get<InterchangeDialect>()) {
    addTypes<Pointer>();
    addOperations<Return, ReduceReturn>();
    allowUnknownOperations();
  }
  void printType(Type type, DialectAsmPrinter &printer) const override {
    printer << "ptr<" << cast<Pointer>(type).getElementType() << ">";
  }
};

Type localType(Type type, OpBuilder &builder) {
  return type.isF32()
             ? Type(builder.getF32Type())
             : Type(builder.getIntegerType(type.getIntOrFloatBitWidth()));
}
Attribute localAttribute(Attribute attribute, MLIRContext *context) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return parseAttribute(text, context);
}
Operation *operation(OpBuilder &builder, Location location, StringRef name,
                     ValueRange operands, TypeRange results,
                     ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attributes);
  return builder.create(state);
}

bool dimensions(AffineMap map, unsigned loops, ArrayRef<unsigned> expected) {
  if (map.getNumDims() != loops || map.getNumSymbols() ||
      map.getNumResults() != expected.size())
    return false;
  for (auto [expression, position] : llvm::zip(map.getResults(), expected)) {
    auto dimension = dyn_cast<AffineDimExpr>(expression);
    if (!dimension || dimension.getPosition() != position)
      return false;
  }
  return true;
}

bool dimensions(AffineMap map, ArrayRef<unsigned> expected) {
  return dimensions(map, 3, expected);
}

// ONNX MatMul batch axes that are statically one are canonicalized by the
// Linalg importer to affine constant zero. Accept that representation only on
// an exact full-rank operand prefix whose physical extent is one. This keeps
// exact-prefix batching valid without admitting singleton broadcast.
bool dimensionsModuloUnitPrefix(AffineMap map, unsigned loops,
                                ArrayRef<unsigned> expected,
                                RankedTensorType operandType,
                                unsigned prefixAxes) {
  if (map.getNumDims() != loops || map.getNumSymbols() ||
      map.getNumResults() != expected.size() ||
      operandType.getRank() != int64_t(expected.size()))
    return false;
  for (auto [axis, pair] : llvm::enumerate(
           llvm::zip(map.getResults(), expected))) {
    AffineExpr expression = std::get<0>(pair);
    unsigned position = std::get<1>(pair);
    if (auto dimension = dyn_cast<AffineDimExpr>(expression);
        dimension && dimension.getPosition() == position)
      continue;
    auto zero = dyn_cast<AffineConstantExpr>(expression);
    if (!zero || zero.getValue() != 0 || axis >= prefixAxes ||
        operandType.getDimSize(axis) != 1)
      return false;
  }
  return true;
}

bool zeroInitializer(Value value) {
  auto op = dyn_cast_or_null<linalg::LinalgOp>(value.getDefiningOp());
  auto role =
      op ? op->getAttrOfType<StringAttr>("gac.linalg_role") : StringAttr();
  if (!op || !role || role.getValue() != "initialize" || op.getNumDpsInputs() ||
      op.getNumDpsInits() != 1 || !llvm::hasSingleElement(op->getRegion(0)))
    return false;
  Block &body = op->getRegion(0).front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return false;
  auto constant = yield.getOperand(0).getDefiningOp<arith::ConstantOp>();
  auto valueAttr =
      constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  return valueAttr && valueAttr.getValue().isZero();
}

bool canonicalF32MatmulPayload(linalg::LinalgOp op) {
  Block &body = op->getRegion(0).front();
  if (body.getNumArguments() != 3)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield && yield.getNumOperands() == 1
                 ? yield.getOperand(0).getDefiningOp<arith::AddFOp>()
                 : arith::AddFOp();
  if (!add)
    return false;
  Value product = add.getLhs() == body.getArgument(2)   ? add.getRhs()
                  : add.getRhs() == body.getArgument(2) ? add.getLhs()
                                                        : Value();
  auto multiply =
      product ? product.getDefiningOp<arith::MulFOp>() : arith::MulFOp();
  return multiply && ((multiply.getLhs() == body.getArgument(0) &&
                       multiply.getRhs() == body.getArgument(1)) ||
                      (multiply.getLhs() == body.getArgument(1) &&
                       multiply.getRhs() == body.getArgument(0)));
}

bool canonicalF32SumPayload(linalg::LinalgOp op) {
  Block &body = op->getRegion(0).front();
  if (body.getNumArguments() != 2 ||
      !llvm::hasSingleElement(body.without_terminator()))
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield && yield.getNumOperands() == 1
                 ? yield.getOperand(0).getDefiningOp<arith::AddFOp>()
                 : arith::AddFOp();
  return add && yield.getOperand(0) == add.getResult() &&
         ((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getLhs() == body.getArgument(1) &&
           add.getRhs() == body.getArgument(0)));
}

bool reductionMaps(linalg::LinalgOp op, bool scalarOutput) {
  AffineMap input = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
  AffineMap output = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  unsigned loops = scalarOutput ? 1 : 2;
  unsigned reductionAxis = scalarOutput ? 0 : 1;
  if (input.getNumDims() != loops || input.getNumSymbols() ||
      input.getNumResults() != 1 || output.getNumDims() != loops ||
      output.getNumSymbols() ||
      output.getNumResults() != unsigned(scalarOutput ? 0 : 1))
    return false;
  auto source = dyn_cast<AffineDimExpr>(input.getResult(0));
  if (!source || source.getPosition() != reductionAxis)
    return false;
  if (!scalarOutput) {
    auto destination = dyn_cast<AffineDimExpr>(output.getResult(0));
    if (!destination || destination.getPosition() != 0)
      return false;
  }
  return true;
}

StringRef linalgRole(Operation *op) {
  if (!op)
    return {};
  auto role = op->getAttrOfType<StringAttr>("gac.linalg_role");
  return role ? role.getValue() : StringRef();
}

bool exactMap(AffineMap actual, unsigned dimensions,
              ArrayRef<AffineExpr> results) {
  return actual == AffineMap::get(dimensions, 0, results, actual.getContext());
}

// Linalg canonicalization may replace an operand index with constant zero
// when that operand dimension is statically one.  This is semantically the
// same map, but only for that exact unit dimension; accepting it here avoids
// rejecting rows=1/width=1024 while preserving every non-unit axis contract.
bool exactMapModuloUnitDims(AffineMap actual, unsigned dimensions,
                            ArrayRef<AffineExpr> expected,
                            RankedTensorType operandType) {
  if (actual.getNumDims() != dimensions || actual.getNumSymbols() != 0 ||
      actual.getNumResults() != expected.size() ||
      operandType.getRank() != int64_t(expected.size()))
    return false;
  for (auto [axis, pair] : llvm::enumerate(
           llvm::zip(actual.getResults(), expected))) {
    AffineExpr observed = std::get<0>(pair);
    AffineExpr wanted = std::get<1>(pair);
    if (observed == wanted)
      continue;
    auto zero = dyn_cast<AffineConstantExpr>(observed);
    if (!zero || zero.getValue() != 0 || operandType.getDimSize(axis) != 1)
      return false;
  }
  return true;
}

bool exactIterators(linalg::LinalgOp op, unsigned rank,
                    bool reduceLast = false) {
  auto iterators = op.getIteratorTypesArray();
  if (iterators.size() != rank)
    return false;
  for (auto [axis, iterator] : llvm::enumerate(iterators)) {
    auto expected = reduceLast && axis + 1 == rank
                        ? utils::IteratorType::reduction
                        : utils::IteratorType::parallel;
    if (iterator != expected)
      return false;
  }
  return true;
}

bool directEmptyDestination(linalg::LinalgOp op) {
  return op.getNumDpsInits() == 1 &&
         op.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>();
}

bool exactUses(Value value,
               ArrayRef<std::pair<Operation *, unsigned>> expected) {
  SmallVector<bool> found(expected.size(), false);
  unsigned count = 0;
  for (OpOperand &use : value.getUses()) {
    ++count;
    bool matched = false;
    for (auto [index, item] : llvm::enumerate(expected)) {
      if (!found[index] && use.getOwner() == item.first &&
          use.getOperandNumber() == item.second) {
        found[index] = true;
        matched = true;
        break;
      }
    }
    if (!matched)
      return false;
  }
  return count == expected.size() && llvm::all_of(found, [](bool value) {
           return value;
         });
}

FloatAttr scalarF32Constant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto attr = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  return attr && attr.getType().isF32() ? attr : FloatAttr();
}

bool f32ConstantEquals(Value value, int64_t expected) {
  FloatAttr attr = scalarF32Constant(value);
  return attr && attr.getValue().convertToDouble() == double(expected);
}

bool noExternalCaptures(Operation *op) {
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  return captures.empty();
}

bool canonicalLayerNormInitializer(Operation *candidate,
                                   RankedTensorType stats) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(candidate);
  if (!generic || linalgRole(candidate) != "initialize" ||
      !generic.hasPureTensorSemantics() || generic.getNumDpsInputs() != 0 ||
      generic.getNumDpsInits() != 1 || candidate->getNumResults() != 1 ||
      candidate->getResult(0).getType() != stats ||
      !directEmptyDestination(generic) ||
      !exactIterators(generic, stats.getRank()) ||
      !exactMapModuloUnitDims(
          generic.getMatchingIndexingMap(generic.getDpsInitOperand(0)),
          stats.getRank(),
          AffineMap::getMultiDimIdentityMap(stats.getRank(),
                                            candidate->getContext())
              .getResults(),
          stats) ||
      !noExternalCaptures(candidate) ||
      !llvm::hasSingleElement(candidate->getRegion(0)))
    return false;
  Block &body = candidate->getRegion(0).front();
  if (body.getNumArguments() != 1 || !body.getArgument(0).use_empty() ||
      llvm::range_size(body.without_terminator()) != 1)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  FloatAttr zero = yield && yield.getNumOperands() == 1
                       ? scalarF32Constant(yield.getOperand(0))
                       : FloatAttr();
  return zero && zero.getValue().isZero();
}

bool canonicalLayerNormSum(linalg::GenericOp sum, RankedTensorType input,
                           RankedTensorType stats) {
  if (!sum || linalgRole(sum) != "sum-reduction" ||
      !sum.hasPureTensorSemantics() || sum.getNumDpsInputs() != 1 ||
      sum.getNumDpsInits() != 1 || sum->getNumResults() != 1 ||
      sum.getDpsInputOperand(0)->get().getType() != input ||
      sum->getResult(0).getType() != stats ||
      !exactIterators(sum, input.getRank(), true) ||
      !noExternalCaptures(sum))
    return false;
  SmallVector<AffineExpr> identity, prefix;
  for (unsigned axis = 0; axis < unsigned(input.getRank()); ++axis)
    identity.push_back(getAffineDimExpr(axis, sum->getContext()));
  for (unsigned axis = 0; axis < unsigned(stats.getRank()); ++axis)
    prefix.push_back(getAffineDimExpr(axis, sum->getContext()));
  return exactMapModuloUnitDims(
             sum.getMatchingIndexingMap(sum.getDpsInputOperand(0)),
             input.getRank(), identity, input) &&
         exactMapModuloUnitDims(
             sum.getMatchingIndexingMap(sum.getDpsInitOperand(0)),
             input.getRank(), prefix, stats) &&
         canonicalF32SumPayload(sum);
}

bool canonicalLayerNormMean(linalg::GenericOp mean, RankedTensorType stats,
                            int64_t width) {
  if (!mean || linalgRole(mean) != "mean" ||
      !mean.hasPureTensorSemantics() || mean.getNumDpsInputs() != 1 ||
      mean.getNumDpsInits() != 1 || mean->getNumResults() != 1 ||
      mean.getDpsInputOperand(0)->get().getType() != stats ||
      mean->getResult(0).getType() != stats ||
      !directEmptyDestination(mean) ||
      !exactIterators(mean, stats.getRank()) || !noExternalCaptures(mean))
    return false;
  auto identity = AffineMap::getMultiDimIdentityMap(stats.getRank(),
                                                    mean->getContext())
                      .getResults();
  if (!exactMapModuloUnitDims(
          mean.getMatchingIndexingMap(mean.getDpsInputOperand(0)),
          stats.getRank(), identity, stats) ||
      !exactMapModuloUnitDims(
          mean.getMatchingIndexingMap(mean.getDpsInitOperand(0)),
          stats.getRank(), identity, stats))
    return false;
  Block &body = mean->getRegion(0).front();
  if (body.getNumArguments() != 2 || !body.getArgument(1).use_empty() ||
      llvm::range_size(body.without_terminator()) != 2)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto divide = yield && yield.getNumOperands() == 1
                    ? yield.getOperand(0).getDefiningOp<arith::DivFOp>()
                    : arith::DivFOp();
  return divide && divide.getLhs() == body.getArgument(0) &&
         f32ConstantEquals(divide.getRhs(), width);
}

bool canonicalLayerNormDeviation(linalg::GenericOp deviation,
                                 RankedTensorType input,
                                 RankedTensorType stats) {
  if (!deviation || linalgRole(deviation) !=
                        "squared-deviation-reduction" ||
      !deviation.hasPureTensorSemantics() ||
      deviation.getNumDpsInputs() != 2 || deviation.getNumDpsInits() != 1 ||
      deviation->getNumResults() != 1 ||
      deviation.getDpsInputOperand(0)->get().getType() != input ||
      deviation.getDpsInputOperand(1)->get().getType() != stats ||
      deviation->getResult(0).getType() != stats ||
      !exactIterators(deviation, input.getRank(), true) ||
      !noExternalCaptures(deviation))
    return false;
  SmallVector<AffineExpr> identity, prefix;
  for (unsigned axis = 0; axis < unsigned(input.getRank()); ++axis)
    identity.push_back(getAffineDimExpr(axis, deviation->getContext()));
  for (unsigned axis = 0; axis < unsigned(stats.getRank()); ++axis)
    prefix.push_back(getAffineDimExpr(axis, deviation->getContext()));
  if (!exactMapModuloUnitDims(
          deviation.getMatchingIndexingMap(deviation.getDpsInputOperand(0)),
          input.getRank(), identity, input) ||
      !exactMapModuloUnitDims(
          deviation.getMatchingIndexingMap(deviation.getDpsInputOperand(1)),
          input.getRank(), prefix, stats) ||
      !exactMapModuloUnitDims(
          deviation.getMatchingIndexingMap(deviation.getDpsInitOperand(0)),
          input.getRank(), prefix, stats))
    return false;
  Block &body = deviation->getRegion(0).front();
  if (body.getNumArguments() != 3 ||
      llvm::range_size(body.without_terminator()) != 3)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto add = yield && yield.getNumOperands() == 1
                 ? yield.getOperand(0).getDefiningOp<arith::AddFOp>()
                 : arith::AddFOp();
  Value square = add && add.getLhs() == body.getArgument(2) ? add.getRhs()
                 : add && add.getRhs() == body.getArgument(2) ? add.getLhs()
                                                                  : Value();
  auto multiply = square ? square.getDefiningOp<arith::MulFOp>()
                         : arith::MulFOp();
  if (!multiply || multiply.getLhs() != multiply.getRhs())
    return false;
  auto subtract = multiply.getLhs().getDefiningOp<arith::SubFOp>();
  return subtract && subtract.getLhs() == body.getArgument(0) &&
         subtract.getRhs() == body.getArgument(1);
}

bool canonicalLayerNormFinal(linalg::GenericOp root, int64_t width,
                             uint32_t &epsilonBits) {
  Block &body = root->getRegion(0).front();
  if (body.getNumArguments() != 6 || !body.getArgument(5).use_empty() ||
      llvm::range_size(body.without_terminator()) != 9)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto affine = yield && yield.getNumOperands() == 1
                    ? yield.getOperand(0).getDefiningOp<arith::AddFOp>()
                    : arith::AddFOp();
  if (!affine || affine.getRhs() != body.getArgument(4))
    return false;
  auto scale = affine.getLhs().getDefiningOp<arith::MulFOp>();
  if (!scale || scale.getRhs() != body.getArgument(3))
    return false;
  auto normalize = scale.getLhs().getDefiningOp<arith::DivFOp>();
  if (!normalize)
    return false;
  auto centered = normalize.getLhs().getDefiningOp<arith::SubFOp>();
  auto squareRoot = normalize.getRhs().getDefiningOp<math::SqrtOp>();
  if (!centered || centered.getLhs() != body.getArgument(0) ||
      centered.getRhs() != body.getArgument(1) || !squareRoot)
    return false;
  auto addEpsilon = squareRoot.getOperand().getDefiningOp<arith::AddFOp>();
  if (!addEpsilon)
    return false;
  auto variance = addEpsilon.getLhs().getDefiningOp<arith::DivFOp>();
  FloatAttr epsilon = scalarF32Constant(addEpsilon.getRhs());
  if (!variance || variance.getLhs() != body.getArgument(2) || !epsilon ||
      !f32ConstantEquals(variance.getRhs(), width))
    return false;
  const APFloat &epsilonValue = epsilon.getValue();
  if (!epsilonValue.isFinite() || epsilonValue.isNegative() ||
      epsilonValue.isZero())
    return false;
  epsilonBits = uint32_t(
      epsilonValue.bitcastToAPInt().getZExtValue());
  return true;
}

} // namespace

bool selectTritonLayerNorm(Operation *op, TritonLayerNormPlan &plan,
                           std::string &reason) {
  plan = TritonLayerNormPlan{};
  reason.clear();
  auto reject = [&](StringRef message) {
    reason = message.str();
    return false;
  };
  auto root = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!root || linalgRole(op) != "layer-normalization")
    return reject("requires the canonical Linalg LayerNormalization root");
  auto loweredFrom = op->getAttrOfType<StringAttr>("gac.lowered_from");
  auto sourceOp = op->getAttrOfType<StringAttr>("gac.source_op");
  auto opset = op->getAttrOfType<IntegerAttr>("gac.opset");
  auto outputNames = op->getAttrOfType<ArrayAttr>("gac.output_names");
  if (!loweredFrom || loweredFrom.getValue() != "gac.layer_norm" ||
      !sourceOp || sourceOp.getValue() != "LayerNormalization" || !opset ||
      opset.getInt() < 17 || !outputNames || outputNames.size() != 1)
    return reject("requires a single-output ONNX LayerNormalization from opset >= 17");
  if (!root.hasPureTensorSemantics() || root.getNumDpsInputs() != 5 ||
      root.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      !llvm::hasSingleElement(op->getRegion(0)) ||
      !directEmptyDestination(root) || !noExternalCaptures(op))
    return reject("requires five inputs, one write-only destination and one result");

  auto input = dyn_cast<RankedTensorType>(
      root.getDpsInputOperand(0)->get().getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto scale = dyn_cast<RankedTensorType>(
      root.getDpsInputOperand(3)->get().getType());
  auto bias = dyn_cast<RankedTensorType>(
      root.getDpsInputOperand(4)->get().getType());
  if (!input || !output || input != output || input.getRank() < 2 ||
      !input.hasStaticShape() || input.getEncoding() ||
      !input.getElementType().isF32() ||
      llvm::any_of(input.getShape(), [](int64_t size) { return size <= 0; }))
    return reject("requires an equal positive static rank>=2 unencoded FP32 input/output");
  int64_t width = input.getShape().back();
  if (width > 1024)
    return reject("the first Triton LayerNorm contract supports last-axis width <= 1024");
  if (!scale || !bias || scale != bias || scale.getRank() != 1 ||
      !scale.hasStaticShape() || scale.getEncoding() ||
      !scale.getElementType().isF32() || scale.getDimSize(0) != width)
    return reject("requires FP32 Scale and Bias vectors matching the last axis");
  Value inputValue = root.getDpsInputOperand(0)->get();
  Value scaleValue = root.getDpsInputOperand(3)->get();
  Value biasValue = root.getDpsInputOperand(4)->get();
  Value inputBuffer = denseViewAliasRoot(inputValue);
  Value scaleBuffer = denseViewAliasRoot(scaleValue);
  Value biasBuffer = denseViewAliasRoot(biasValue);
  if (inputBuffer.getType() != inputValue.getType() ||
      scaleBuffer.getType() != scaleValue.getType() ||
      biasBuffer.getType() != biasValue.getType())
    return reject("requires view roots to preserve the LayerNorm pointer shapes");
  if (inputBuffer == scaleBuffer || inputBuffer == biasBuffer ||
      scaleBuffer == biasBuffer)
    return reject("requires distinct X, Scale and Bias physical buffers");
  int64_t rows = 1;
  for (int64_t size : input.getShape().drop_back()) {
    if (size > INT32_MAX / rows)
      return reject("LayerNorm row count exceeds signed 32-bit indexing");
    rows *= size;
  }
  if (width > INT32_MAX / rows)
    return reject("LayerNorm element count exceeds signed 32-bit indexing");
  auto stats = RankedTensorType::get(input.getShape().drop_back(),
                                      input.getElementType());
  if (root.getDpsInputOperand(1)->get().getType() != stats ||
      root.getDpsInputOperand(2)->get().getType() != stats ||
      !exactIterators(root, input.getRank()))
    return reject("requires last-axis normalization with canonical FP32 statistics");

  SmallVector<AffineExpr> identity, prefix;
  for (unsigned axis = 0; axis < unsigned(input.getRank()); ++axis)
    identity.push_back(getAffineDimExpr(axis, op->getContext()));
  for (unsigned axis = 0; axis + 1 < unsigned(input.getRank()); ++axis)
    prefix.push_back(getAffineDimExpr(axis, op->getContext()));
  AffineExpr channel = width == 1
                           ? AffineExpr(getAffineConstantExpr(0, op->getContext()))
                           : AffineExpr(getAffineDimExpr(input.getRank() - 1,
                                                        op->getContext()));
  if (!exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInputOperand(0)),
          input.getRank(), identity, input) ||
      !exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInputOperand(1)),
          input.getRank(), prefix, stats) ||
      !exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInputOperand(2)),
          input.getRank(), prefix, stats) ||
      !exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInputOperand(3)),
          input.getRank(), ArrayRef<AffineExpr>{channel}, scale) ||
      !exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInputOperand(4)),
          input.getRank(), ArrayRef<AffineExpr>{channel}, bias) ||
      !exactMapModuloUnitDims(
          root.getMatchingIndexingMap(root.getDpsInitOperand(0)),
          input.getRank(), identity, output))
    return reject("LayerNorm indexing maps do not describe the canonical last-axis form");

  auto mean = root.getDpsInputOperand(1)->get().getDefiningOp<linalg::GenericOp>();
  auto deviation =
      root.getDpsInputOperand(2)->get().getDefiningOp<linalg::GenericOp>();
  auto total = mean ? mean.getDpsInputOperand(0)->get()
                          .getDefiningOp<linalg::GenericOp>()
                    : linalg::GenericOp();
  Operation *firstInit = total
                             ? total.getDpsInitOperand(0)->get().getDefiningOp()
                             : nullptr;
  Operation *secondInit = deviation
                              ? deviation.getDpsInitOperand(0)->get().getDefiningOp()
                              : nullptr;
  if (!canonicalLayerNormInitializer(firstInit, stats) ||
      !canonicalLayerNormSum(total, input, stats) ||
      !canonicalLayerNormMean(mean, stats, width) ||
      !canonicalLayerNormInitializer(secondInit, stats) ||
      !canonicalLayerNormDeviation(deviation, input, stats))
    return reject("LayerNorm statistics subgraph is incomplete or noncanonical");
  if (total.getDpsInputOperand(0)->get() !=
          root.getDpsInputOperand(0)->get() ||
      deviation.getDpsInputOperand(0)->get() !=
          root.getDpsInputOperand(0)->get() ||
      deviation.getDpsInputOperand(1)->get() != mean->getResult(0))
    return reject("LayerNorm statistics do not share the root input and mean");

  if (!exactUses(firstInit->getResult(0),
                 {{total.getOperation(), 1}}) ||
      !exactUses(total->getResult(0), {{mean.getOperation(), 0}}) ||
      !exactUses(mean->getResult(0),
                 {{deviation.getOperation(), 1}, {op, 1}}) ||
      !exactUses(secondInit->getResult(0),
                 {{deviation.getOperation(), 2}}) ||
      !exactUses(deviation->getResult(0), {{op, 2}}))
    return reject("LayerNorm internal values have missing or external uses");

  uint32_t epsilonBits = 0;
  if (!canonicalLayerNormFinal(root, width, epsilonBits))
    return reject("LayerNorm affine payload does not match the canonical formula");
  int64_t tile = std::max<int64_t>(16, llvm::PowerOf2Ceil(uint64_t(width)));
  if (tile < width || tile > 1024)
    return reject("Triton LayerNorm tile is outside [16,1024]");
  plan.rows = rows;
  plan.width = width;
  plan.tile = tile;
  plan.gridSize = rows;
  plan.epsilonF32Bits = epsilonBits;
  plan.input = inputValue;
  plan.scale = scaleValue;
  plan.bias = biasValue;
  plan.internalOperations = {firstInit, total.getOperation(),
                             mean.getOperation(), secondInit,
                             deviation.getOperation()};
  return true;
}

LogicalResult emitTritonLayerNorm(Operation *op, StringRef kernelName,
                                  const TritonLayerNormPlan &plan,
                                  StringRef outputFile) {
  TritonLayerNormPlan checked;
  std::string reason;
  if (!selectTritonLayerNorm(op, checked, reason))
    return op->emitOpError("cannot emit Triton LayerNorm kernel: ") << reason;
  if (checked.rows != plan.rows || checked.width != plan.width ||
      checked.tile != plan.tile || checked.gridSize != plan.gridSize ||
      checked.epsilonF32Bits != plan.epsilonF32Bits ||
      checked.input != plan.input || checked.scale != plan.scale ||
      checked.bias != plan.bias ||
      checked.internalOperations != plan.internalOperations)
    return op->emitOpError("Triton LayerNorm plan changed after selection");

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, math::MathDialect,
                  InterchangeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Type f32 = builder.getF32Type(), i32 = builder.getI32Type();
  Type pointer = Pointer::get(f32);
  builder.setInsertionPointToEnd(module->getBody());
  OperationState function(loc, "tt.func");
  function.addAttribute("sym_name", builder.getStringAttr(kernelName));
  function.addAttribute("sym_visibility", builder.getStringAttr("public"));
  function.addAttribute(
      "function_type",
      TypeAttr::get(builder.getFunctionType(
          {pointer, pointer, pointer, pointer, i32, f32}, {})));
  function.addAttribute("noinline", builder.getBoolAttr(false));
  function.addRegion();
  Operation *kernel = builder.create(function);
  Block *entry = new Block;
  kernel->getRegion(0).push_back(entry);
  entry->addArgument(pointer, loc); // X
  entry->addArgument(pointer, loc); // Scale
  entry->addArgument(pointer, loc); // Bias
  entry->addArgument(pointer, loc); // Y
  entry->addArgument(i32, loc);     // N
  entry->addArgument(f32, loc);     // epsilon
  builder.setInsertionPointToStart(entry);

  auto vector = [&](Type element) {
    return RankedTensorType::get({plan.tile}, element);
  };
  auto splat = [&](Value value) {
    return operation(builder, loc, "tt.splat", value, vector(value.getType()))
        ->getResult(0);
  };
  auto addReduction = [&](Value values) {
    OperationState state(loc, "tt.reduce");
    state.addOperands(values);
    state.addTypes(f32);
    state.addAttribute("axis", builder.getI32IntegerAttr(0));
    state.addRegion();
    Operation *result = builder.create(state);
    Block *body = new Block;
    result->getRegion(0).push_back(body);
    body->addArgument(f32, loc);
    body->addArgument(f32, loc);
    builder.setInsertionPointToStart(body);
    Value combined = builder.create<arith::AddFOp>(
        loc, body->getArgument(0), body->getArgument(1));
    operation(builder, loc, "tt.reduce.return", combined, {});
    builder.setInsertionPointAfter(result);
    return result->getResult(0);
  };
  auto maskedLoad = [&](BlockArgument base, Value indices, Value mask,
                        Value other) {
    Type pointerVector = vector(pointer);
    Value pointers =
        operation(builder, loc, "tt.splat", base, pointerVector)->getResult(0);
    pointers = operation(builder, loc, "tt.addptr",
                         ValueRange{pointers, indices}, pointerVector)
                   ->getResult(0);
    return operation(
               builder, loc, "tt.load", ValueRange{pointers, mask, other},
               vector(f32),
               {builder.getNamedAttr(
                    "operandSegmentSizes",
                    builder.getDenseI32ArrayAttr({1, 1, 1})),
                builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
                builder.getNamedAttr("evict", builder.getI32IntegerAttr(1)),
                builder.getNamedAttr("isVolatile",
                                     builder.getBoolAttr(false))})
        ->getResult(0);
  };

  Value zero = builder.create<arith::ConstantOp>(
      loc, vector(f32),
      DenseElementsAttr::get(vector(f32), builder.getF32FloatAttr(0)));
  Value offsets =
      operation(builder, loc, "tt.make_range", {}, vector(i32),
                {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
                 builder.getNamedAttr("end",
                                      builder.getI32IntegerAttr(plan.tile))})
          ->getResult(0);
  Value width = splat(entry->getArgument(4));
  Value mask = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, offsets, width);
  Value row =
      operation(builder, loc, "tt.get_program_id", {}, i32,
                {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))})
          ->getResult(0);
  Value rowBase = builder.create<arith::MulIOp>(
      loc, row, entry->getArgument(4));
  Value indices = builder.create<arith::AddIOp>(loc, offsets, splat(rowBase));
  Value values = maskedLoad(entry->getArgument(0), indices, mask, zero);

  Value count = builder.create<arith::SIToFPOp>(
      loc, f32, entry->getArgument(4));
  Value mean = builder.create<arith::DivFOp>(loc, addReduction(values), count);
  Value centeredCandidate =
      builder.create<arith::SubFOp>(loc, values, splat(mean));
  // Inactive lanes loaded zero above, but zero-minus-mean is not zero.  Mask
  // the centered value itself before squaring so a tail never contributes to
  // the variance reduction.
  Value centered = builder.create<arith::SelectOp>(
      loc, mask, centeredCandidate, zero);
  Value squared = builder.create<arith::MulFOp>(loc, centered, centered);
  Value variance =
      builder.create<arith::DivFOp>(loc, addReduction(squared), count);
  Value adjusted = builder.create<arith::AddFOp>(
      loc, variance, entry->getArgument(5));
  Value denominator = builder.create<math::SqrtOp>(loc, adjusted);
  Value normalized = builder.create<arith::DivFOp>(
      loc, centered, splat(denominator));
  Value scales = maskedLoad(entry->getArgument(1), offsets, mask, zero);
  Value biases = maskedLoad(entry->getArgument(2), offsets, mask, zero);
  Value affine = builder.create<arith::AddFOp>(
      loc, builder.create<arith::MulFOp>(loc, normalized, scales), biases);

  Type pointerVector = vector(pointer);
  Value outputPointers =
      operation(builder, loc, "tt.splat", entry->getArgument(3), pointerVector)
          ->getResult(0);
  outputPointers = operation(builder, loc, "tt.addptr",
                             ValueRange{outputPointers, indices}, pointerVector)
                       ->getResult(0);
  operation(builder, loc, "tt.store",
            ValueRange{outputPointers, affine, mask}, {},
            {builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1))});
  operation(builder, loc, "tt.return", {}, {});

  if (failed(verify(*module)))
    return op->emitOpError(
        "generated TTIR LayerNorm fails structural verification");
  std::error_code error;
  llvm::ToolOutputFile file(outputFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return op->emitOpError("cannot write TTIR LayerNorm: ")
           << error.message();
  module->print(file.os(), OpPrintingFlags().printGenericOpForm());
  file.os() << '\n';
  file.keep();
  return success();
}

bool selectTritonSoftmax(Operation *op, TritonSoftmaxPlan &plan,
                         std::string &reason) {
  plan = TritonSoftmaxPlan{};
  reason.clear();
  auto reject = [&](StringRef message) {
    reason = message.str();
    return false;
  };
  auto softmax = dyn_cast_or_null<linalg::SoftmaxOp>(op);
  auto role = softmax ? op->getAttrOfType<StringAttr>("gac.linalg_role")
                      : StringAttr();
  if (!softmax || !role || role.getValue() != "softmax")
    return reject("requires a canonical linalg.softmax root");
  if (!softmax.hasPureTensorSemantics() || softmax.getNumDpsInputs() != 1 ||
      softmax.getNumDpsInits() != 1 || op->getNumResults() != 1)
    return reject("requires one input, one write-only destination and one result");
  auto input = dyn_cast<RankedTensorType>(softmax.getInput().getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!input || !output || input != output || input.getRank() < 1 ||
      !input.hasStaticShape() || input.getEncoding() ||
      !input.getElementType().isF32() ||
      llvm::any_of(input.getShape(), [](int64_t size) { return size <= 0; }))
    return reject("requires equal positive static rank>=1 unencoded FP32 input "
                  "and output tensors");
  if (softmax.getDimension() != input.getRank() - 1)
    return reject("Triton Softmax requires the last logical axis");
  if (!softmax.getOutput().getDefiningOp<tensor::EmptyOp>())
    return reject("requires the canonical write-only tensor.empty destination");
  SmallVector<Value> leaves = getLinalgDeviceInputs(op);
  if (leaves.size() != 1 || leaves.front() != softmax.getInput())
    return reject("Triton Softmax ABI accepts one unfused input");

  // DeviceLowering elides byte-identical dense views.  Do not select a
  // logical Softmax shape whose manifest would later name a differently
  // shaped physical allocation: the driver/runtime validate the actual
  // pointer ABI rather than reconstructing view metadata.
  Value inputBuffer = denseViewAliasRoot(softmax.getInput());
  if (inputBuffer.getType() != softmax.getInput().getType())
    return reject("Softmax input dense view changes the physical buffer shape");
  if (inputBuffer == op->getResult(0))
    return reject("Softmax input and output must use distinct physical buffers");

  constexpr int64_t maxTile = 1024;
  constexpr int64_t maxElements = std::numeric_limits<int32_t>::max();
  int64_t width = input.getDimSize(input.getRank() - 1);
  if (width > maxTile)
    return reject("Triton Softmax supports last-axis width <= 1024");
  int64_t rows = 1;
  for (int64_t size : input.getShape().drop_back()) {
    if (size > maxElements / rows)
      return reject("Softmax row count exceeds signed 32-bit indexing");
    rows *= size;
  }
  if (rows > maxElements / width)
    return reject("Softmax flattened element count exceeds signed 32-bit indexing");
  int64_t tile = std::max<int64_t>(16, llvm::PowerOf2Ceil(uint64_t(width)));
  if (tile < width || tile > maxTile)
    return reject("Triton Softmax tile is outside [16,1024]");
  plan.rows = rows;
  plan.width = width;
  plan.tile = tile;
  plan.gridSize = rows;
  return true;
}

LogicalResult emitTritonSoftmax(Operation *op, StringRef kernelName,
                                const TritonSoftmaxPlan &plan,
                                StringRef outputFile) {
  TritonSoftmaxPlan checked;
  std::string reason;
  if (!selectTritonSoftmax(op, checked, reason))
    return op->emitOpError("cannot emit Triton Softmax kernel: ") << reason;
  if (checked.rows != plan.rows || checked.width != plan.width ||
      checked.tile != plan.tile ||
      checked.gridSize != plan.gridSize)
    return op->emitOpError("Triton Softmax plan changed after selection");

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, math::MathDialect,
                  InterchangeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Type f32 = builder.getF32Type(), i32 = builder.getI32Type();
  Type pointer = Pointer::get(f32);
  builder.setInsertionPointToEnd(module->getBody());
  OperationState function(loc, "tt.func");
  function.addAttribute("sym_name", builder.getStringAttr(kernelName));
  function.addAttribute("sym_visibility", builder.getStringAttr("public"));
  function.addAttribute(
      "function_type",
      TypeAttr::get(builder.getFunctionType({pointer, pointer, i32}, {})));
  function.addAttribute("noinline", builder.getBoolAttr(false));
  function.addRegion();
  Operation *kernel = builder.create(function);
  Block *entry = new Block;
  kernel->getRegion(0).push_back(entry);
  entry->addArgument(pointer, loc);
  entry->addArgument(pointer, loc);
  entry->addArgument(i32, loc);
  builder.setInsertionPointToStart(entry);

  auto vector = [&](Type element) {
    return RankedTensorType::get({plan.tile}, element);
  };
  auto splat = [&](Value value) {
    return operation(builder, loc, "tt.splat", value, vector(value.getType()))
        ->getResult(0);
  };
  auto reduce = [&](Value values, StringRef payload) {
    OperationState state(loc, "tt.reduce");
    state.addOperands(values);
    state.addTypes(f32);
    state.addAttribute("axis", builder.getI32IntegerAttr(0));
    state.addRegion();
    Operation *result = builder.create(state);
    Block *body = new Block;
    result->getRegion(0).push_back(body);
    body->addArgument(f32, loc);
    body->addArgument(f32, loc);
    builder.setInsertionPointToStart(body);
    Value combined = payload == "maxnumf"
                         ? Value(builder.create<arith::MaxNumFOp>(
                               loc, body->getArgument(0), body->getArgument(1)))
                         : Value(builder.create<arith::AddFOp>(
                               loc, body->getArgument(0), body->getArgument(1)));
    operation(builder, loc, "tt.reduce.return", combined, {});
    builder.setInsertionPointAfter(result);
    return result->getResult(0);
  };

  Value negativeInfinity = builder.create<arith::ConstantOp>(
      loc, vector(f32),
      DenseElementsAttr::get(
          vector(f32),
          builder.getF32FloatAttr(-std::numeric_limits<float>::infinity())));
  Value offsets =
      operation(builder, loc, "tt.make_range", {}, vector(i32),
                {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
                 builder.getNamedAttr("end",
                                      builder.getI32IntegerAttr(plan.tile))})
          ->getResult(0);
  Value widths = splat(entry->getArgument(2));
  Value mask = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, offsets, widths);
  Value row =
      operation(builder, loc, "tt.get_program_id", {}, i32,
                {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))})
          ->getResult(0);
  Value rowBase = builder.create<arith::MulIOp>(
      loc, row, entry->getArgument(2));
  Value indices = builder.create<arith::AddIOp>(loc, splat(rowBase), offsets);
  auto pointerVector = vector(pointer);
  Value inputPointers =
      operation(builder, loc, "tt.splat", entry->getArgument(0), pointerVector)
          ->getResult(0);
  inputPointers = operation(builder, loc, "tt.addptr",
                            ValueRange{inputPointers, indices}, pointerVector)
                      ->getResult(0);
  Value values =
      operation(
          builder, loc, "tt.load",
          ValueRange{inputPointers, mask, negativeInfinity}, vector(f32),
          {builder.getNamedAttr("operandSegmentSizes",
                                builder.getDenseI32ArrayAttr({1, 1, 1})),
           builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
           builder.getNamedAttr("evict", builder.getI32IntegerAttr(1)),
           builder.getNamedAttr("isVolatile", builder.getBoolAttr(false))})
          ->getResult(0);
  Value maximum = reduce(values, "maxnumf");
  Value centered =
      builder.create<arith::SubFOp>(loc, values, splat(maximum));
  Value numerators = builder.create<math::ExpOp>(loc, centered);
  Value denominator = reduce(numerators, "addf");
  Value normalized =
      builder.create<arith::DivFOp>(loc, numerators, splat(denominator));
  Value outputPointers =
      operation(builder, loc, "tt.splat", entry->getArgument(1), pointerVector)
          ->getResult(0);
  outputPointers = operation(builder, loc, "tt.addptr",
                             ValueRange{outputPointers, indices}, pointerVector)
                       ->getResult(0);
  operation(builder, loc, "tt.store",
            ValueRange{outputPointers, normalized, mask}, {},
            {builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1))});
  operation(builder, loc, "tt.return", {}, {});

  if (failed(verify(*module)))
    return op->emitOpError(
        "generated TTIR Softmax fails structural verification");
  std::error_code error;
  llvm::ToolOutputFile file(outputFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return op->emitOpError("cannot write TTIR Softmax: ") << error.message();
  module->print(file.os(), OpPrintingFlags().printGenericOpForm());
  file.os() << '\n';
  file.keep();
  return success();
}

bool selectTritonReduction(Operation *op, TritonReductionPlan &plan,
                           std::string &reason) {
  plan = TritonReductionPlan{};
  reason.clear();
  auto reject = [&](StringRef message) {
    reason = message.str();
    return false;
  };
  auto reduction = dyn_cast_or_null<linalg::LinalgOp>(op);
  auto role = reduction
                  ? op->getAttrOfType<StringAttr>("gac.linalg_role")
                  : StringAttr();
  if (!reduction || !role || role.getValue() != "sum-reduction")
    return reject("requires a canonical Linalg sum-reduction root");
  if (!reduction.hasPureTensorSemantics() ||
      reduction.getNumDpsInputs() != 1 ||
      reduction.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      !llvm::hasSingleElement(op->getRegion(0)))
    return reject("requires one input, one initializer and one result");
  auto input = dyn_cast<RankedTensorType>(
      reduction.getDpsInputOperand(0)->get().getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  bool scalarOutput = output && output.getRank() == 0;
  bool singletonOutput = output && output.getRank() == 1 &&
                         output.getDimSize(0) == 1;
  if (!input || !output || input.getRank() != 1 ||
      !input.hasStaticShape() || !output.hasStaticShape() ||
      input.getEncoding() || output.getEncoding() ||
      !input.getElementType().isF32() || !output.getElementType().isF32() ||
      input.getDimSize(0) <= 0 || (!scalarOutput && !singletonOutput))
    return reject("requires a positive static rank-one FP32 input and scalar "
                  "or one-element FP32 output");
  auto iterators = reduction.getIteratorTypesArray();
  if ((scalarOutput &&
       (iterators.size() != 1 ||
        iterators[0] != utils::IteratorType::reduction)) ||
      (singletonOutput &&
       (iterators.size() != 2 ||
        iterators[0] != utils::IteratorType::parallel ||
        iterators[1] != utils::IteratorType::reduction)) ||
      !reductionMaps(reduction, scalarOutput))
    return reject("requires a full-axis rank-one reduction with canonical "
                  "Linalg indexing maps");
  if (!zeroInitializer(reduction.getDpsInitOperand(0)->get()) ||
      !canonicalF32SumPayload(reduction))
    return reject("requires an FP32 zero-initialized add reduction");
  SmallVector<Value> leaves = getLinalgDeviceInputs(op);
  if (leaves.size() != 1 ||
      leaves[0] != reduction.getDpsInputOperand(0)->get())
    return reject("the first tt.reduce ABI accepts one unfused input");
  int64_t width = input.getDimSize(0);
  // This first backend expands tt.reduce to a scalar FPU loop. Bound the TTIR
  // tensor and its generated code before calculating the next power of two.
  constexpr int64_t maxTile = 1024;
  if (width > maxTile)
    return reject("the first tt.reduce contract supports width <= 1024");
  int64_t tile = std::max<int64_t>(16, llvm::PowerOf2Ceil(uint64_t(width)));
  if (tile < width || tile > maxTile)
    return reject("Triton reduction tile is outside [16,1024]");
  plan.width = width;
  plan.tile = tile;
  plan.gridSize = 1;
  return true;
}

LogicalResult emitTritonReduction(Operation *op, StringRef kernelName,
                                  const TritonReductionPlan &plan,
                                  StringRef outputFile) {
  TritonReductionPlan checked;
  std::string reason;
  if (!selectTritonReduction(op, checked, reason))
    return op->emitOpError("cannot emit Triton reduction kernel: ") << reason;
  if (checked.width != plan.width || checked.tile != plan.tile ||
      checked.gridSize != plan.gridSize)
    return op->emitOpError("Triton reduction plan changed after selection");

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, InterchangeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Type f32 = builder.getF32Type(), i32 = builder.getI32Type();
  Type pointer = Pointer::get(f32);
  builder.setInsertionPointToEnd(module->getBody());
  OperationState function(loc, "tt.func");
  function.addAttribute("sym_name", builder.getStringAttr(kernelName));
  function.addAttribute("sym_visibility", builder.getStringAttr("public"));
  function.addAttribute(
      "function_type",
      TypeAttr::get(builder.getFunctionType({pointer, pointer}, {})));
  function.addAttribute("noinline", builder.getBoolAttr(false));
  function.addRegion();
  Operation *kernel = builder.create(function);
  Block *entry = new Block;
  kernel->getRegion(0).push_back(entry);
  entry->addArgument(pointer, loc);
  entry->addArgument(pointer, loc);
  builder.setInsertionPointToStart(entry);

  auto vector = [&](Type element) {
    return RankedTensorType::get({plan.tile}, element);
  };
  Value zero = builder.create<arith::ConstantOp>(
      loc, vector(f32), DenseElementsAttr::get(vector(f32),
                                               builder.getF32FloatAttr(0)));
  Value offsets =
      operation(builder, loc, "tt.make_range", {}, vector(i32),
                {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
                 builder.getNamedAttr("end",
                                      builder.getI32IntegerAttr(plan.tile))})
          ->getResult(0);
  Value width = builder.create<arith::ConstantIntOp>(loc, plan.width, 32);
  Value widths =
      operation(builder, loc, "tt.splat", width, vector(i32))->getResult(0);
  Value mask = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, offsets, widths);
  auto pointerVector = vector(pointer);
  Value pointers = operation(builder, loc, "tt.splat", entry->getArgument(0),
                             pointerVector)
                       ->getResult(0);
  pointers = operation(builder, loc, "tt.addptr",
                       ValueRange{pointers, offsets}, pointerVector)
                 ->getResult(0);
  Value values =
      operation(
          builder, loc, "tt.load", ValueRange{pointers, mask, zero},
          vector(f32),
          {builder.getNamedAttr("operandSegmentSizes",
                                builder.getDenseI32ArrayAttr({1, 1, 1})),
           builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
           builder.getNamedAttr("evict", builder.getI32IntegerAttr(1)),
           builder.getNamedAttr("isVolatile", builder.getBoolAttr(false))})
          ->getResult(0);

  OperationState reductionState(loc, "tt.reduce");
  reductionState.addOperands(values);
  reductionState.addTypes(f32);
  reductionState.addAttribute("axis", builder.getI32IntegerAttr(0));
  reductionState.addRegion();
  Operation *reduce = builder.create(reductionState);
  Block *body = new Block;
  reduce->getRegion(0).push_back(body);
  body->addArgument(f32, loc);
  body->addArgument(f32, loc);
  builder.setInsertionPointToStart(body);
  Value sum = builder.create<arith::AddFOp>(loc, body->getArgument(0),
                                            body->getArgument(1));
  operation(builder, loc, "tt.reduce.return", sum, {});
  builder.setInsertionPointAfter(reduce);
  operation(builder, loc, "tt.store",
            ValueRange{entry->getArgument(1), reduce->getResult(0)}, {},
            {builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1))});
  operation(builder, loc, "tt.return", {}, {});

  if (failed(verify(*module)))
    return op->emitOpError(
        "generated TTIR reduction fails structural verification");
  std::error_code error;
  llvm::ToolOutputFile file(outputFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return op->emitOpError("cannot write TTIR reduction: ")
           << error.message();
  module->print(file.os(), OpPrintingFlags().printGenericOpForm());
  file.os() << '\n';
  file.keep();
  return success();
}

bool selectTritonDot(Operation *op, TritonDotPlan &plan, std::string &reason) {
  plan = TritonDotPlan{};
  reason.clear();
  auto reject = [&](StringRef message) {
    reason = message.str();
    return false;
  };
  auto matmul = dyn_cast_or_null<linalg::LinalgOp>(op);
  auto role =
      matmul ? op->getAttrOfType<StringAttr>("gac.linalg_role") : StringAttr();
  if (!matmul || !role || role.getValue() != "matmul")
    return reject("requires a canonical Linalg MatMul root");
  if (!matmul.hasPureTensorSemantics() || matmul.getNumDpsInputs() != 2 ||
      matmul.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      !llvm::hasSingleElement(op->getRegion(0)))
    return reject("requires two matrix inputs, one initializer and one result");
  auto a =
      dyn_cast<RankedTensorType>(matmul.getDpsInputOperand(0)->get().getType());
  auto b =
      dyn_cast<RankedTensorType>(matmul.getDpsInputOperand(1)->get().getType());
  auto c = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!a || !b || !c || c.getRank() < 2 || !a.hasStaticShape() ||
      !b.hasStaticShape() || !c.hasStaticShape() || a.getEncoding() ||
      b.getEncoding() || c.getEncoding() || !a.getElementType().isF32() ||
      !b.getElementType().isF32() || !c.getElementType().isF32() ||
      llvm::any_of(a.getShape(), [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(b.getShape(), [](int64_t size) { return size <= 0; }) ||
      llvm::any_of(c.getShape(), [](int64_t size) { return size <= 0; }))
    return reject("requires positive static, unencoded FP32 matrices");

  int64_t rank = c.getRank();
  bool legacy = rank == 2;
  bool aBatched = !legacy && a.getRank() == rank;
  bool bBatched = !legacy && b.getRank() == rank;
  if ((legacy && (a.getRank() != 2 || b.getRank() != 2)) ||
      (!legacy && (!aBatched || (b.getRank() != 2 && !bBatched))))
    return reject("batched MatMul requires full-rank A and full-rank or shared rank-two B");
  ArrayRef<int64_t> batchShape = c.getShape().drop_back(2);
  if (aBatched && a.getShape().drop_back(2) != batchShape)
    return reject("A batch prefix must exactly match C; singleton broadcast is unsupported");
  if (bBatched && b.getShape().drop_back(2) != batchShape)
    return reject("B batch prefix must exactly match C; singleton broadcast is unsupported");

  int64_t m = c.getDimSize(rank - 2), n = c.getDimSize(rank - 1);
  int64_t k = a.getDimSize(a.getRank() - 1);
  if (a.getDimSize(a.getRank() - 2) != m ||
      b.getDimSize(b.getRank() - 2) != k ||
      b.getDimSize(b.getRank() - 1) != n)
    return reject("matrix shapes must end in A[...,M,K] x B[...,K,N] -> C[...,M,N]");
  if (k < 16 || k % 16 || k > 65536)
    return reject(
        "the tt.dot contract requires 16 <= K <= 65536 and K divisible by 16");

  Value aValue = matmul.getDpsInputOperand(0)->get();
  Value bValue = matmul.getDpsInputOperand(1)->get();
  Value aRoot = denseViewAliasRoot(aValue);
  Value bRoot = denseViewAliasRoot(bValue);
  if (aRoot.getType() != aValue.getType() || bRoot.getType() != bValue.getType())
    return reject("shape-changing dense views require native MatMul lowering");
  if (aRoot == bRoot)
    return reject("Triton MatMul requires distinct A and B physical buffers");

  auto iterators = matmul.getIteratorTypesArray();
  if (iterators.size() != unsigned(rank + 1) ||
      llvm::any_of(
          ArrayRef<utils::IteratorType>(iterators).drop_back(),
          [](utils::IteratorType type) {
        return type != utils::IteratorType::parallel;
      }) || iterators.back() != utils::IteratorType::reduction)
    return reject("requires batch/output parallel axes followed by one reduction axis");
  SmallVector<unsigned> aMap, bMap, cMap;
  if (aBatched)
    for (unsigned axis = 0; axis + 2 < unsigned(rank); ++axis)
      aMap.push_back(axis);
  aMap.push_back(unsigned(rank - 2));
  aMap.push_back(unsigned(rank));
  if (bBatched)
    for (unsigned axis = 0; axis + 2 < unsigned(rank); ++axis)
      bMap.push_back(axis);
  bMap.push_back(unsigned(rank));
  bMap.push_back(unsigned(rank - 1));
  for (unsigned axis = 0; axis < unsigned(rank); ++axis)
    cMap.push_back(axis);
  if (!dimensionsModuloUnitPrefix(
          matmul.getMatchingIndexingMap(matmul.getDpsInputOperand(0)),
          rank + 1, aMap, a, aBatched ? unsigned(rank - 2) : 0) ||
      !dimensionsModuloUnitPrefix(
          matmul.getMatchingIndexingMap(matmul.getDpsInputOperand(1)),
          rank + 1, bMap, b, bBatched ? unsigned(rank - 2) : 0) ||
      !dimensions(matmul.getMatchingIndexingMap(matmul.getDpsInitOperand(0)),
                  rank + 1, cMap))
    return reject("requires canonical row-major batched MatMul indexing maps");
  if (!zeroInitializer(matmul.getDpsInitOperand(0)->get()) ||
      !canonicalF32MatmulPayload(matmul))
    return reject("requires an FP32 zero-initialized multiply-add reduction");
  SmallVector<Value> leaves = getLinalgDeviceInputs(op);
  if (leaves.size() != 2 || leaves[0] != matmul.getDpsInputOperand(0)->get() ||
      leaves[1] != matmul.getDpsInputOperand(1)->get())
    return reject("the first tt.dot ABI accepts only A and B before output");
  constexpr int64_t limit = std::numeric_limits<int32_t>::max();
  auto multiply = [&](int64_t &value, int64_t factor) {
    if (factor <= 0 || value > limit / factor)
      return false;
    value *= factor;
    return true;
  };
  int64_t batches = 1;
  for (int64_t extent : batchShape)
    if (!multiply(batches, extent))
      return reject("flattened batch count exceeds signed 32-bit limits");
  int64_t logicalA = batches;
  int64_t logicalB = bBatched ? batches : 1;
  int64_t logicalC = batches;
  if (!multiply(logicalA, m) || !multiply(logicalA, k) ||
      !multiply(logicalB, k) || !multiply(logicalB, n) ||
      !multiply(logicalC, m) || !multiply(logicalC, n))
    return reject("batched matrix indexing exceeds signed 32-bit limits");
  int64_t tilesM = (m + 15) / 16, tilesN = (n + 15) / 16;
  int64_t grid = batches;
  if (!multiply(grid, tilesM) || !multiply(grid, tilesN))
    return reject("matrix tile grid exceeds signed 32-bit limits");
  plan.m = m;
  plan.n = n;
  plan.k = k;
  plan.batches = batches;
  plan.blockK = int64_t(llvm::PowerOf2Ceil(uint64_t(k)));
  plan.gridSize = grid;
  plan.aBatched = aBatched;
  plan.bBatched = bBatched;
  plan.batchMode = legacy ? "rank2" : bBatched ? "matched-prefix" : "shared-rhs";
  plan.aShape.assign(a.getShape().begin(), a.getShape().end());
  plan.bShape.assign(b.getShape().begin(), b.getShape().end());
  plan.cShape.assign(c.getShape().begin(), c.getShape().end());
  return true;
}

LogicalResult emitTritonDot(Operation *op, StringRef kernelName,
                            const TritonDotPlan &plan, StringRef outputFile) {
  TritonDotPlan checked;
  std::string reason;
  if (!selectTritonDot(op, checked, reason))
    return op->emitOpError("cannot emit Triton dot kernel: ") << reason;
  if (checked.m != plan.m || checked.n != plan.n || checked.k != plan.k ||
      checked.batches != plan.batches ||
      checked.blockM != plan.blockM || checked.blockN != plan.blockN ||
      checked.blockK != plan.blockK || checked.gridSize != plan.gridSize ||
      checked.aBatched != plan.aBatched ||
      checked.bBatched != plan.bBatched ||
      checked.batchMode != plan.batchMode || checked.aShape != plan.aShape ||
      checked.bShape != plan.bShape || checked.cShape != plan.cShape)
    return op->emitOpError("Triton dot plan changed after selection");

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, InterchangeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Type f32 = builder.getF32Type(), i32 = builder.getI32Type();
  Type pointer = Pointer::get(f32);
  SmallVector<Type> arguments{pointer, pointer, pointer};
  builder.setInsertionPointToEnd(module->getBody());
  OperationState function(loc, "tt.func");
  function.addAttribute("sym_name", builder.getStringAttr(kernelName));
  function.addAttribute("sym_visibility", builder.getStringAttr("public"));
  function.addAttribute("function_type",
                        TypeAttr::get(builder.getFunctionType(arguments, {})));
  function.addAttribute("noinline", builder.getBoolAttr(false));
  function.addRegion();
  Operation *kernel = builder.create(function);
  Block *entry = new Block;
  kernel->getRegion(0).push_back(entry);
  for (Type type : arguments)
    entry->addArgument(type, loc);
  builder.setInsertionPointToStart(entry);

  auto ranked = [&](ArrayRef<int64_t> shape, Type element) {
    return RankedTensorType::get(shape, element);
  };
  auto splat = [&](Value value, ArrayRef<int64_t> shape) {
    return operation(builder, loc, "tt.splat", value,
                     ranked(shape, value.getType()))
        ->getResult(0);
  };
  auto range = [&](int64_t width) {
    return operation(
               builder, loc, "tt.make_range", {}, ranked({width}, i32),
               {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
                builder.getNamedAttr("end", builder.getI32IntegerAttr(width))})
        ->getResult(0);
  };
  auto expand = [&](Value value, int64_t axis, ArrayRef<int64_t> shape) {
    return operation(
               builder, loc, "tt.expand_dims", value,
               ranked(shape,
                      cast<RankedTensorType>(value.getType()).getElementType()),
               {builder.getNamedAttr("axis", builder.getI32IntegerAttr(axis))})
        ->getResult(0);
  };
  auto broadcast = [&](Value value, ArrayRef<int64_t> shape) {
    return operation(
               builder, loc, "tt.broadcast", value,
               ranked(shape,
                      cast<RankedTensorType>(value.getType()).getElementType()))
        ->getResult(0);
  };
  auto matrix = [&](Value vector, int64_t axis, ArrayRef<int64_t> expanded,
                    ArrayRef<int64_t> shape) {
    return broadcast(expand(vector, axis, expanded), shape);
  };
  auto load = [&](BlockArgument base, Value offsets, Value mask,
                  ArrayRef<int64_t> shape) {
    auto pointerTile = ranked(shape, pointer);
    Value pointers =
        operation(builder, loc, "tt.splat", base, pointerTile)->getResult(0);
    pointers = operation(builder, loc, "tt.addptr",
                         ValueRange{pointers, offsets}, pointerTile)
                   ->getResult(0);
    auto valueTile = ranked(shape, f32);
    Value zero = builder.create<arith::ConstantOp>(
        loc, valueTile,
        DenseElementsAttr::get(valueTile, builder.getF32FloatAttr(0)));
    return operation(
               builder, loc, "tt.load", ValueRange{pointers, mask, zero},
               valueTile,
               {builder.getNamedAttr("operandSegmentSizes",
                                     builder.getDenseI32ArrayAttr({1, 1, 1})),
                builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
                builder.getNamedAttr("evict", builder.getI32IntegerAttr(1)),
                builder.getNamedAttr("isVolatile", builder.getBoolAttr(false))})
        ->getResult(0);
  };

  Value program =
      operation(builder, loc, "tt.get_program_id", {}, i32,
                {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))})
          ->getResult(0);
  bool batched = plan.batchMode != "rank2";
  int64_t tilesM = (plan.m + plan.blockM - 1) / plan.blockM;
  int64_t tilesN = (plan.n + plan.blockN - 1) / plan.blockN;
  Value tileColumns = builder.create<arith::ConstantIntOp>(loc, tilesN, 32);
  Value batchId;
  Value tileProgram = program;
  if (batched) {
    Value tilesPerBatch = builder.create<arith::ConstantIntOp>(
        loc, tilesM * tilesN, 32);
    batchId = builder.create<arith::DivUIOp>(loc, program, tilesPerBatch);
    tileProgram = builder.create<arith::RemUIOp>(loc, program, tilesPerBatch);
  }
  Value pidM = builder.create<arith::DivUIOp>(loc, tileProgram, tileColumns);
  Value pidN = builder.create<arith::RemUIOp>(loc, tileProgram, tileColumns);
  Value rowBase = builder.create<arith::MulIOp>(
      loc, pidM, builder.create<arith::ConstantIntOp>(loc, plan.blockM, 32));
  Value colBase = builder.create<arith::MulIOp>(
      loc, pidN, builder.create<arith::ConstantIntOp>(loc, plan.blockN, 32));
  Value rows = builder.create<arith::AddIOp>(loc, range(plan.blockM),
                                             splat(rowBase, {plan.blockM}));
  Value cols = builder.create<arith::AddIOp>(loc, range(plan.blockN),
                                             splat(colBase, {plan.blockN}));
  Value reduction = range(plan.blockK);
  Value reductionValid;
  if (plan.blockK != plan.k)
    reductionValid = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, reduction,
        splat(builder.create<arith::ConstantIntOp>(loc, plan.k, 32),
              {plan.blockK}));
  Value rowValid = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, rows,
      splat(builder.create<arith::ConstantIntOp>(loc, plan.m, 32),
            {plan.blockM}));
  Value colValid = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, cols,
      splat(builder.create<arith::ConstantIntOp>(loc, plan.n, 32),
            {plan.blockN}));

  SmallVector<int64_t> aShape{plan.blockM, plan.blockK};
  Value aRows = matrix(rows, 1, {plan.blockM, 1}, aShape);
  Value aK = matrix(reduction, 0, {1, plan.blockK}, aShape);
  Value aOffsets = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc, aRows,
          splat(builder.create<arith::ConstantIntOp>(loc, plan.k, 32), aShape)),
      aK);
  if (plan.aBatched) {
    Value aBatchBase = builder.create<arith::MulIOp>(
        loc, batchId,
        builder.create<arith::ConstantIntOp>(loc, plan.m * plan.k, 32));
    aOffsets = builder.create<arith::AddIOp>(
        loc, aOffsets, splat(aBatchBase, aShape));
  }
  Value aMask = matrix(rowValid, 1, {plan.blockM, 1}, aShape);
  if (reductionValid)
    aMask = builder.create<arith::AndIOp>(
        loc, aMask,
        matrix(reductionValid, 0, {1, plan.blockK}, aShape));
  Value aTile = load(entry->getArgument(0), aOffsets, aMask, aShape);

  SmallVector<int64_t> bShape{plan.blockK, plan.blockN};
  Value bK = matrix(reduction, 1, {plan.blockK, 1}, bShape);
  Value bCols = matrix(cols, 0, {1, plan.blockN}, bShape);
  Value bOffsets = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc, bK,
          splat(builder.create<arith::ConstantIntOp>(loc, plan.n, 32), bShape)),
      bCols);
  if (plan.bBatched) {
    Value bBatchBase = builder.create<arith::MulIOp>(
        loc, batchId,
        builder.create<arith::ConstantIntOp>(loc, plan.k * plan.n, 32));
    bOffsets = builder.create<arith::AddIOp>(
        loc, bOffsets, splat(bBatchBase, bShape));
  }
  Value bMask = matrix(colValid, 0, {1, plan.blockN}, bShape);
  if (reductionValid)
    bMask = builder.create<arith::AndIOp>(
        loc, matrix(reductionValid, 1, {plan.blockK, 1}, bShape), bMask);
  Value bTile = load(entry->getArgument(1), bOffsets, bMask, bShape);

  SmallVector<int64_t> cShape{plan.blockM, plan.blockN};
  auto cType = ranked(cShape, f32);
  Value zero = builder.create<arith::ConstantOp>(
      loc, cType, DenseElementsAttr::get(cType, builder.getF32FloatAttr(0)));
  // No precision attribute is needed: standard tt.dot defaults to IEEE, which
  // is also the only precision accepted by the initial Vortex backend.
  Value result =
      operation(builder, loc, "tt.dot", ValueRange{aTile, bTile, zero}, cType)
          ->getResult(0);
  Value cRows = matrix(rows, 1, {plan.blockM, 1}, cShape);
  Value cCols = matrix(cols, 0, {1, plan.blockN}, cShape);
  Value cOffsets = builder.create<arith::AddIOp>(
      loc,
      builder.create<arith::MulIOp>(
          loc, cRows,
          splat(builder.create<arith::ConstantIntOp>(loc, plan.n, 32), cShape)),
      cCols);
  if (batched) {
    Value cBatchBase = builder.create<arith::MulIOp>(
        loc, batchId,
        builder.create<arith::ConstantIntOp>(loc, plan.m * plan.n, 32));
    cOffsets = builder.create<arith::AddIOp>(
        loc, cOffsets, splat(cBatchBase, cShape));
  }
  Value cMask = builder.create<arith::AndIOp>(
      loc, matrix(rowValid, 1, {plan.blockM, 1}, cShape),
      matrix(colValid, 0, {1, plan.blockN}, cShape));
  auto pointerTile = ranked(cShape, pointer);
  Value outputPointers =
      operation(builder, loc, "tt.splat", entry->getArgument(2), pointerTile)
          ->getResult(0);
  outputPointers = operation(builder, loc, "tt.addptr",
                             ValueRange{outputPointers, cOffsets}, pointerTile)
                       ->getResult(0);
  operation(builder, loc, "tt.store", ValueRange{outputPointers, result, cMask},
            {},
            {builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1))});
  operation(builder, loc, "tt.return", {}, {});

  if (failed(verify(*module)))
    return op->emitOpError("generated TTIR dot fails structural verification");
  std::error_code error;
  llvm::ToolOutputFile file(outputFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return op->emitOpError("cannot write TTIR dot: ") << error.message();
  module->print(file.os(), OpPrintingFlags().printGenericOpForm());
  file.os() << '\n';
  file.keep();
  return success();
}

bool selectTritonPointwise(Operation *op, std::string &reason) {
  reason.clear();
  if (!op || op->getNumResults() != 1) {
    reason = "requires one result";
    return false;
  }
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!output) {
    reason = "requires a ranked tensor result";
    return false;
  }
  llvm::DenseSet<Operation *> checked;
  return checkCandidate(op, output, checked, reason);
}

LogicalResult emitTritonPointwise(Operation *op, StringRef kernelName,
                                  int64_t block, StringRef outputFile) {
  std::string reason;
  if (!selectTritonPointwise(op, reason))
    return op->emitOpError("cannot emit Triton pointwise kernel: ") << reason;
  if (block < 16 || block > 1024 || !llvm::isPowerOf2_64(block))
    return op->emitOpError("Triton block must be a power of two in [16,1024]");

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, InterchangeDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  SmallVector<Value> leaves = getLinalgDeviceInputs(op);
  auto output = cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<Type> arguments;
  for (Value leaf : leaves)
    arguments.push_back(Pointer::get(localType(
        cast<RankedTensorType>(leaf.getType()).getElementType(), builder)));
  arguments.push_back(
      Pointer::get(localType(output.getElementType(), builder)));
  builder.setInsertionPointToEnd(module->getBody());
  OperationState function(loc, "tt.func");
  function.addAttribute("sym_name", builder.getStringAttr(kernelName));
  function.addAttribute("sym_visibility", builder.getStringAttr("public"));
  function.addAttribute("function_type",
                        TypeAttr::get(builder.getFunctionType(arguments, {})));
  function.addAttribute("noinline", builder.getBoolAttr(false));
  function.addRegion();
  Operation *kernel = builder.create(function);
  Block *entry = new Block;
  kernel->getRegion(0).push_back(entry);
  for (Type type : arguments)
    entry->addArgument(type, loc);
  builder.setInsertionPointToStart(entry);
  auto tile = [&](Type type) { return RankedTensorType::get({block}, type); };
  auto splat = [&](Value value) {
    return operation(builder, loc, "tt.splat", value, tile(value.getType()))
        ->getResult(0);
  };
  Value program =
      operation(builder, loc, "tt.get_program_id", {}, builder.getI32Type(),
                {builder.getNamedAttr("axis", builder.getI32IntegerAttr(0))})
          ->getResult(0);
  Value width = builder.create<arith::ConstantIntOp>(loc, block, 32);
  Value base = builder.create<arith::MulIOp>(loc, program, width);
  Value range =
      operation(builder, loc, "tt.make_range", {}, tile(builder.getI32Type()),
                {builder.getNamedAttr("start", builder.getI32IntegerAttr(0)),
                 builder.getNamedAttr("end", builder.getI32IntegerAttr(block))})
          ->getResult(0);
  Value indices = builder.create<arith::AddIOp>(loc, splat(base), range);
  Value size =
      builder.create<arith::ConstantIntOp>(loc, output.getNumElements(), 32);
  Value mask = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                             indices, splat(size));
  auto address = [&](BlockArgument argument) {
    Value pointers = splat(argument);
    return operation(builder, loc, "tt.addptr", ValueRange{pointers, indices},
                     pointers.getType())
        ->getResult(0);
  };
  llvm::DenseMap<Value, Value> values;
  for (auto [index, leaf] : llvm::enumerate(leaves)) {
    Type type = localType(
        cast<RankedTensorType>(leaf.getType()).getElementType(), builder);
    Attribute zero = type.isF32() ? Attribute(builder.getF32FloatAttr(0))
                                  : Attribute(builder.getI32IntegerAttr(0));
    Value other = builder.create<arith::ConstantOp>(
        loc, tile(type), DenseElementsAttr::get(tile(type), zero));
    // Generic-form TTIR carries defaults explicitly for Triton 3.8. The actual
    // Triton frontend verifier is the authority before backend code generation.
    values[leaf] =
        operation(
            builder, loc, "tt.load",
            ValueRange{address(entry->getArgument(index)), mask, other},
            tile(type),
            {builder.getNamedAttr("operandSegmentSizes",
                                  builder.getDenseI32ArrayAttr({1, 1, 1})),
             builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("isVolatile", builder.getBoolAttr(false))})
            ->getResult(0);
  }
  std::function<FailureOr<Value>(Operation *)> emit;
  emit = [&](Operation *current) -> FailureOr<Value> {
    auto linalg = cast<linalg::LinalgOp>(current);
    Block &body = current->getRegion(0).front();
    llvm::DenseMap<Value, Value> payload;
    for (auto [index, operand] :
         llvm::enumerate(linalg.getDpsInputOperands())) {
      Value input = operand->get();
      auto found = values.find(input);
      if (found == values.end()) {
        auto producer = emit(input.getDefiningOp());
        if (failed(producer))
          return failure();
        values[input] = *producer;
      }
      payload[body.getArgument(index)] = values.lookup(input);
    }
    for (Operation &scalar : body.without_terminator()) {
      SmallVector<Value> operands;
      for (Value operand : scalar.getOperands()) {
        auto found = payload.find(operand);
        if (found == payload.end())
          return current->emitOpError("unmapped Triton payload operand"),
                 failure();
        operands.push_back(found->second);
      }
      Type result = localType(scalar.getResult(0).getType(), builder);
      SmallVector<NamedAttribute> attributes;
      for (NamedAttribute attr : scalar.getAttrs()) {
        Attribute value = localAttribute(attr.getValue(), &context);
        if (!value)
          return current->emitOpError(
                     "cannot transfer arithmetic attribute to TTIR"),
                 failure();
        if (isa<arith::ConstantOp>(scalar) && attr.getName() == "value")
          value = DenseElementsAttr::get(tile(result), value);
        attributes.push_back(
            builder.getNamedAttr(attr.getName().strref(), value));
      }
      payload[scalar.getResult(0)] =
          operation(builder, loc, scalar.getName().getStringRef(), operands,
                    tile(result), attributes)
              ->getResult(0);
    }
    Value yielded = body.getTerminator()->getOperand(0);
    auto found = payload.find(yielded);
    if (found == payload.end())
      return current->emitOpError("unmapped Triton payload result"), failure();
    return found->second;
  };
  auto result = emit(op);
  if (failed(result))
    return failure();
  operation(builder, loc, "tt.store",
            ValueRange{address(entry->getArguments().back()), *result, mask},
            {},
            {builder.getNamedAttr("cache", builder.getI32IntegerAttr(1)),
             builder.getNamedAttr("evict", builder.getI32IntegerAttr(1))});
  operation(builder, loc, "tt.return", {}, {});
  if (failed(verify(*module)))
    return op->emitOpError(
        "generated TTIR fails structural/arithmetic verification");
  std::error_code error;
  llvm::ToolOutputFile file(outputFile, error, llvm::sys::fs::OF_Text);
  if (error)
    return op->emitOpError("cannot write TTIR: ") << error.message();
  module->print(file.os(), OpPrintingFlags().printGenericOpForm());
  file.os() << '\n';
  file.keep();
  return success();
}

} // namespace gac
