#include "gac/LinalgDevice.h"

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

using namespace mlir;

namespace gac {
namespace {

bool readsInitializer(linalg::LinalgOp op) {
  return op.payloadUsesValueFromOperand(op.getDpsInitOperand(0));
}

bool purePayload(linalg::LinalgOp op) {
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  if (!captures.empty())
    return false;
  WalkResult result = op->getRegion(0).walk([](Operation *nested) {
    return isMemoryEffectFree(nested) ? WalkResult::advance()
                                      : WalkResult::interrupt();
  });
  return !result.wasInterrupted();
}

// Descriptors describe already-planned device storage. They allocate no memory
// and preserve the runtime's pointer ABI and global arena lifetime analysis.
Value asMemRef(OpBuilder &builder, Location loc, Value pointer,
               RankedTensorType tensorType) {
  auto memrefType = MemRefType::get(tensorType.getShape(),
                                    tensorType.getElementType());
  LowerToLLVMOptions options(builder.getContext());
  options.overrideIndexBitwidth(32);
  LLVMTypeConverter converter(builder.getContext(), options);
  Value descriptor = MemRefDescriptor::fromStaticShape(
      builder, loc, converter, memrefType, pointer);
  return builder.create<UnrealizedConversionCastOp>(
      loc, TypeRange{memrefType}, ValueRange{descriptor}).getResult(0);
}

SmallVector<Value> outputCoordinates(OpBuilder &builder, Location loc,
                                     Value tid, ArrayRef<int64_t> shape) {
  Value flat = tid;
  if (!isa<IndexType>(flat.getType()))
    flat = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), flat);
  SmallVector<Value> coordinates(shape.size());
  for (int64_t axis = int64_t(shape.size()) - 1; axis >= 0; --axis) {
    Value extent = builder.create<arith::ConstantIndexOp>(loc, shape[axis]);
    coordinates[axis] = builder.create<arith::RemUIOp>(loc, flat, extent);
    flat = builder.create<arith::DivUIOp>(loc, flat, extent);
  }
  return coordinates;
}

} // namespace

bool isFusedLinalgProducer(Operation *op) {
  auto producer = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!producer || !producer.hasPureTensorSemantics() ||
      producer.getNumDpsInits() != 1 || op->getNumResults() != 1 ||
      !op->getResult(0).hasOneUse())
    return false;
  // Aggregate named operations have no scalar payload region.  They remain
  // explicit roots so their complete numerical contract is not lost.
  if (isa<linalg::SoftmaxOp>(op))
    return false;
  if (auto role = op->getAttrOfType<StringAttr>("gac.linalg_role");
      role && role.getValue() == "layer-normalization")
    return false;
  OpOperand &use = *op->getResult(0).getUses().begin();
  auto consumer = dyn_cast<linalg::LinalgOp>(use.getOwner());
  if (!consumer || !consumer.hasPureTensorSemantics() ||
      consumer.getNumDpsInits() != 1 || consumer->getNumResults() != 1 ||
      isa<linalg::SoftmaxOp>(consumer.getOperation()) ||
      !purePayload(producer) || !purePayload(consumer))
    return false;
  if (auto role = consumer->getAttrOfType<StringAttr>("gac.linalg_role");
      role && role.getValue() == "layer-normalization")
    return false;
  auto source = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto destination = dyn_cast<RankedTensorType>(consumer->getResult(0).getType());
  if (!source || !destination || !source.hasStaticShape() ||
      source.getShape() != destination.getShape())
    return false;

  // An initializer such as zeroTensor has no input dependence and can be
  // evaluated directly for the accumulator element, including reductions.
  if (consumer.isDpsInit(&use))
    return producer.getNumDpsInputs() == 0 && !readsInitializer(producer);

  // A reduction consumer may visit many producer elements. This first schedule
  // fuses only elementwise consumers whose operand map is exactly their output
  // map. Broadcasts, transposes, fanout, and cross-element reuse remain separate.
  if (llvm::any_of(consumer.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type != utils::IteratorType::parallel;
      }))
    return false;
  if (consumer.getMatchingIndexingMap(&use) ==
      consumer.getMatchingIndexingMap(consumer.getDpsInitOperand(0)))
    return true;

  // Canonicalization is allowed to rewrite an indexing map for a unit
  // dimension from `(d0) -> (d0)` to `(d0) -> (0)`.  The LayerNorm sum and
  // mean tensors contain exactly one element, so both maps address the same
  // scalar.  Keep this exception deliberately narrow: it is only the
  // canonical sum-reduction -> mean edge, and both logical tensors must be
  // statically single-element.  This preserves the three-dispatch native
  // fallback when the aggregate Triton LayerNorm contract is rejected.
  auto producerRole =
      op->getAttrOfType<StringAttr>("gac.linalg_role");
  auto consumerRole =
      consumer->getAttrOfType<StringAttr>("gac.linalg_role");
  return producerRole && consumerRole &&
         producerRole.getValue() == "sum-reduction" &&
         consumerRole.getValue() == "mean" &&
         source.getNumElements() == 1 && destination.getNumElements() == 1;
}

SmallVector<Value> getLinalgDeviceInputs(Operation *op) {
  if (auto softmax = dyn_cast<linalg::SoftmaxOp>(op))
    return SmallVector<Value>{softmax.getInput()};
  SmallVector<Value> inputs;
  llvm::DenseSet<Value> seen;
  std::function<void(linalg::LinalgOp)> visitOperation;
  auto visitValue = [&](Value value) {
    if (isFusedLinalgProducer(value.getDefiningOp())) {
      visitOperation(cast<linalg::LinalgOp>(value.getDefiningOp()));
    } else if (seen.insert(value).second) {
      inputs.push_back(value);
    }
  };
  visitOperation = [&](linalg::LinalgOp current) {
    for (OpOperand *operand : current.getDpsInputOperands())
      visitValue(operand->get());
    if (readsInitializer(current))
      visitValue(current.getDpsInitOperand(0)->get());
  };
  visitOperation(cast<linalg::LinalgOp>(op));
  return inputs;
}

LogicalResult validateLinalgDeviceOp(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    if (!type || !type.hasStaticShape() ||
        llvm::any_of(type.getShape(), [](int64_t d) { return d <= 0; }))
      return op->emitOpError("device Linalg operands must be nonempty, "
                             "statically shaped tensors");
    if (type.getEncoding())
      return op->emitOpError(
          "device Linalg ABI requires unencoded row-major tensors");
  }

  if (auto softmax = dyn_cast<linalg::SoftmaxOp>(op)) {
    auto inputType = dyn_cast<RankedTensorType>(softmax.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(softmax.getOutput().getType());
    auto resultType = dyn_cast<RankedTensorType>(softmax->getResult(0).getType());
    if (!softmax.hasPureTensorSemantics() || softmax->getNumResults() != 1 ||
        !inputType || inputType != outputType || outputType != resultType ||
        !isa<FloatType>(inputType.getElementType()) ||
        softmax.getDimension() < 0 ||
        softmax.getDimension() >= inputType.getRank())
      return op->emitOpError(
          "device linalg.softmax requires equal floating-point input/output "
          "tensors and an in-range reduction dimension");
    return success();
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasPureTensorSemantics() ||
      linalgOp.getNumDpsInits() != 1 || op->getNumResults() != 1)
    return op->emitOpError("device Linalg lowering requires one tensor result "
                           "and one destination initializer");

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType || resultType != linalgOp.getDpsInitOperand(0)->get().getType())
    return op->emitOpError("Linalg result and destination types must agree");

  auto iterators = linalgOp.getIteratorTypesArray();
  auto outputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  if (outputMap.getNumSymbols() ||
      outputMap.getNumResults() != unsigned(resultType.getRank()))
    return op->emitOpError("unsupported Linalg output indexing map");

  // Each parallel dimension must name exactly one output axis. Reduction axes
  // must not occur in the output address. Size-one keepdims axes may be zero.
  SmallVector<bool> mapped(iterators.size(), false);
  for (auto [axis, expr] : llvm::enumerate(outputMap.getResults())) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      if (constant.getValue() == 0 && resultType.getDimSize(axis) == 1)
        continue;
    }
    auto dim = dyn_cast<AffineDimExpr>(expr);
    if (!dim || dim.getPosition() >= iterators.size() ||
        iterators[dim.getPosition()] != utils::IteratorType::parallel ||
        mapped[dim.getPosition()])
      return op->emitOpError("device schedule requires a projected-permutation "
                             "output map over parallel loops");
    mapped[dim.getPosition()] = true;
  }
  for (auto [axis, iterator] : llvm::enumerate(iterators)) {
    if (iterator == utils::IteratorType::parallel && !mapped[axis])
      return op->emitOpError("a parallel loop is absent from the output map");
    if (iterator != utils::IteratorType::parallel &&
        iterator != utils::IteratorType::reduction)
      return op->emitOpError("unsupported Linalg iterator type");
  }
  if (!linalgOp.getShapesToLoopsMap())
    return op->emitOpError("Linalg loop extents cannot be derived from operands");
  if (llvm::any_of(linalgOp.getStaticLoopRanges(),
                   [](int64_t size) { return size <= 0; }))
    return op->emitOpError("device Linalg loops require positive static extents");

  // Capturing a host graph SSA value would leave an invalid use after graph
  // erasure. Constants belong inside the Linalg payload; tensor values belong
  // in its explicit operands so the runtime can plan their dependencies.
  llvm::SetVector<Value> captures;
  getUsedValuesDefinedAbove(op->getRegions(), captures);
  if (!captures.empty())
    return op->emitOpError("Linalg payload must not capture external SSA values");
  return success();
}

namespace {

// The shape remains available to upstream Linalg loop construction, while all
// accesses address one private scalar. Eligibility proves that this invocation
// needs only the same output coordinate from this producer. The scalar's actual
// element type preserves FP16 rounding at an intermediate operation boundary.
FailureOr<Value> scalarTile(OpBuilder &builder, Location loc,
                            RankedTensorType type) {
  Operation *parent = builder.getInsertionBlock()->getParentOp();
  while (parent && !isa<LLVM::LLVMFuncOp>(parent))
    parent = parent->getParentOp();
  if (!parent)
    return emitError(loc, "Linalg scalar tiles require an LLVM kernel scope"), failure();
  auto kernel = cast<LLVM::LLVMFuncOp>(parent);
  Value scalar;
  {
    // One slot per fused producer and kernel. In the native target the outer
    // loop evaluates many output elements; allocating inside that loop would
    // grow the stack until the function returns. GPU threads own private slots.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&kernel.getBody().front());
    scalar = builder.create<memref::AllocaOp>(
        loc, MemRefType::get({}, type.getElementType()));
  }
  if (type.getRank() == 0)
    return scalar;
  SmallVector<int64_t> strides(type.getRank(), 0);
  auto layout = StridedLayoutAttr::get(builder.getContext(), 0, strides);
  auto tileType = MemRefType::get(type.getShape(), type.getElementType(), layout);
  return Value(builder.create<memref::ReinterpretCastOp>(
      loc, tileType, scalar, int64_t(0), type.getShape(), strides));
}

LogicalResult emitBufferizedElement(OpBuilder &builder, Operation *op,
                                     ArrayRef<Value> inputs, Value initializer,
                                     Value output, Value tid) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Location loc = op->getLoc();
  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
  SmallVector<Value> coordinates =
      outputCoordinates(builder, loc, tid, outputType.getShape());
  SmallVector<Value> operands(inputs.begin(), inputs.end());
  operands.push_back(output);

  // Tensor semantics require the old destination value for reductions. The
  // output has separate storage, so copy just this thread's element. A fused
  // initializer is a private scalar, not a full zero-filled device tensor.
  if (initializer) {
    Value initial = builder.create<memref::LoadOp>(loc, initializer, coordinates);
    builder.create<memref::StoreOp>(loc, initial, output, coordinates);
  }

  OperationState state(loc, op->getName());
  state.addOperands(operands);
  state.addAttributes(op->getAttrs());
  state.addRegion();
  Operation *bufferOperation = builder.create(state);
  IRMapping mapping;
  op->getRegion(0).cloneInto(&bufferOperation->getRegion(0), mapping);
  auto bufferOp = cast<linalg::LinalgOp>(bufferOperation);

  // Use upstream lowering for indexing maps, payload arithmetic, loads/stores,
  // and reduction loops. Linalg remains the authoritative computation here.
  IRRewriter rewriter(builder.getContext());
  rewriter.setInsertionPoint(bufferOperation);
  auto loops = linalg::linalgOpToLoops(rewriter, bufferOp);
  if (failed(loops))
    return op->emitOpError("upstream Linalg-to-SCF lowering failed");

  auto iterators = linalgOp.getIteratorTypesArray();
  if (loops->size() != iterators.size())
    return op->emitOpError("Linalg loop lowering returned an unexpected loop nest");
  SmallVector<Value> parallelCoordinates(iterators.size());
  auto outputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  for (auto [axis, expr] : llvm::enumerate(outputMap.getResults()))
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      parallelCoordinates[dim.getPosition()] = coordinates[axis];

  // Do not execute the entire matrix in every thread. Replace only parallel
  // loops with this thread's output coordinates; reduction loops remain intact.
  for (int64_t axis = int64_t(loops->size()) - 1; axis >= 0; --axis) {
    if (iterators[axis] != utils::IteratorType::parallel)
      continue;
    auto loop = cast<scf::ForOp>((*loops)[axis]);
    rewriter.eraseOp(loop.getBody()->getTerminator());
    rewriter.inlineBlockBefore(loop.getBody(), loop,
                               ValueRange{parallelCoordinates[axis]});
    rewriter.eraseOp(loop);
  }
  rewriter.eraseOp(bufferOperation);
  return success();
}

} // namespace

LogicalResult emitLinalgDeviceElement(OpBuilder &builder, Operation *op,
                                      ArrayRef<Value> pointers, Value tid) {
  if (failed(validateLinalgDeviceOp(op)))
    return failure();
  SmallVector<Value> leaves = getLinalgDeviceInputs(op);
  if (pointers.size() != leaves.size() + 1)
    return op->emitOpError("Linalg dispatch pointer count does not match inputs");
  llvm::DenseMap<Value, Value> buffers;
  for (auto [index, value] : llvm::enumerate(leaves))
    buffers[value] = asMemRef(builder, op->getLoc(), pointers[index],
                              cast<RankedTensorType>(value.getType()));

  std::function<LogicalResult(Operation *, Value)> emitOperation;
  std::function<FailureOr<Value>(Value)> materialize = [&](Value value) -> FailureOr<Value> {
    if (auto found = buffers.find(value); found != buffers.end())
      return found->second;
    Operation *producer = value.getDefiningOp();
    if (!isFusedLinalgProducer(producer))
      return op->emitOpError("Linalg fusion reached a value absent from the dispatch ABI"), failure();
    auto tile = scalarTile(builder, producer->getLoc(),
                            cast<RankedTensorType>(value.getType()));
    if (failed(tile) || failed(emitOperation(producer, *tile)))
      return failure();
    buffers[value] = *tile;
    return *tile;
  };
  emitOperation = [&](Operation *current, Value output) -> LogicalResult {
    if (failed(validateLinalgDeviceOp(current)))
      return failure();
    auto linalgOp = cast<linalg::LinalgOp>(current);
    SmallVector<Value> inputs;
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      auto input = materialize(operand->get());
      if (failed(input))
        return failure();
      inputs.push_back(*input);
    }
    Value initializer;
    if (readsInitializer(linalgOp)) {
      auto init = materialize(linalgOp.getDpsInitOperand(0)->get());
      if (failed(init))
        return failure();
      initializer = *init;
    }
    return emitBufferizedElement(builder, current, inputs, initializer, output, tid);
  };

  Value output = asMemRef(builder, op->getLoc(), pointers.back(),
                          cast<RankedTensorType>(op->getResult(0).getType()));
  return emitOperation(op, output);
}

} // namespace gac
