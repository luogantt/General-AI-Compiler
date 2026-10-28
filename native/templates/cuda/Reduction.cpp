#include "gac/templates/CudaReduction.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cmath>
#include <functional>
#include <limits>

using namespace mlir;
namespace gac {
namespace {
bool checkedProduct(ArrayRef<int64_t> shape, int64_t &size) {
  size = 1;
  // DeviceLowering currently uses signed 32-bit element/byte indices.
  constexpr int64_t maxElements = std::numeric_limits<int32_t>::max() / 4;
  for (int64_t dim : shape) {
    if (dim <= 0 || size > maxElements / dim)
      return false;
    size *= dim;
  }
  return true;
}
bool broadcastsTo(ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
  if (from.size() > to.size())
    return false;
  for (size_t i = 0; i < from.size(); ++i)
    if (from[i] != 1 && from[i] != to[to.size() - from.size() + i])
      return false;
  return true;
}

// These helpers construct typed MLIR SSA, never source strings. They are
// intentionally local to this compute template rather than a second graph IR.
struct RowBuilder {
  OpBuilder &b;
  Location loc;
  Value i(int64_t value) {
    return b.create<arith::ConstantIntOp>(loc, value, 32);
  }
  Value f(double value) {
    return b.create<arith::ConstantFloatOp>(loc, APFloat(float(value)),
                                           b.getF32Type());
  }
  Value addi(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value muli(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value divi(Value a, Value c) { return b.create<arith::DivUIOp>(loc, a, c); }
  Value remi(Value a, Value c) { return b.create<arith::RemUIOp>(loc, a, c); }
  Value add(Value a, Value c) { return b.create<arith::AddFOp>(loc, a, c); }
  Value sub(Value a, Value c) { return b.create<arith::SubFOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulFOp>(loc, a, c); }
  Value div(Value a, Value c) { return b.create<arith::DivFOp>(loc, a, c); }
  Value lt(Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, c);
  }
  Value address(Value base, Value index) {
    return b.create<LLVM::GEPOp>(loc, base.getType(), b.getF32Type(), base,
                                ValueRange{index});
  }
  Value load(Value base, Value index) {
    return b.create<LLVM::LoadOp>(loc, b.getF32Type(), address(base, index));
  }
  void store(Value value, Value base, Value index) {
    b.create<LLVM::StoreOp>(loc, value, address(base, index));
  }
  void barrier() { b.create<NVVM::Barrier0Op>(loc); }
  void when(Value condition, const std::function<void()> &body) {
    auto branch = b.create<scf::IfOp>(loc, condition, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    body();
  }
  Value laneReduce(Value lane, int64_t width, int64_t blockSize, Value initial,
                   const std::function<Value(Value, Value)> &body) {
    auto loop = b.create<scf::ForOp>(
        loc, lane, i(width), i(blockSize), ValueRange{initial},
        [&](OpBuilder &, Location, Value k, ValueRange carry) {
          b.create<scf::YieldOp>(loc, body(k, carry[0]));
        });
    return loop.getResult(0);
  }
  void laneLoop(Value lane, int64_t width, int64_t blockSize,
                const std::function<void(Value)> &body) {
    b.create<scf::ForOp>(
        loc, lane, i(width), i(blockSize), ValueRange{},
        [&](OpBuilder &, Location, Value k, ValueRange) {
          body(k);
          b.create<scf::YieldOp>(loc);
        });
  }
  Value blockReduce(Value scratch, Value lane, int64_t blockSize,
                    Value partial, bool maximum = false) {
    store(partial, scratch, lane);
    barrier();
    for (int64_t offset = blockSize / 2; offset; offset /= 2) {
      when(lt(lane, i(offset)), [&] {
        Value a = load(scratch, lane);
        Value c = load(scratch, addi(lane, i(offset)));
        // maximumf propagates NaNs, matching the original scalar lowering.
        Value combined = maximum ? Value(b.create<arith::MaximumFOp>(loc, a, c))
                                 : add(a, c);
        store(combined, scratch, lane);
      });
      barrier();
    }
    Value result = load(scratch, i(0));
    // Every lane must have read the result before the next reduction writes
    // this same shared array. The next store's barrier alone is insufficient.
    barrier();
    return result;
  }
  Value broadcastIndex(Value index, ArrayRef<int64_t> from,
                       ArrayRef<int64_t> to) {
    Value result = i(0);
    int64_t inputStride = 1, outputStride = 1;
    for (int64_t axis = int64_t(to.size()) - 1; axis >= 0; --axis) {
      int64_t source = axis - int64_t(to.size() - from.size());
      if (source >= 0) {
        if (from[source] != 1) {
          Value coordinate = remi(divi(index, i(outputStride)), i(to[axis]));
          result = addi(result, muli(coordinate, i(inputStride)));
        }
        inputStride *= from[source];
      }
      outputStride *= to[axis];
    }
    return result;
  }
};
} // namespace

LogicalResult planCudaReduction(Operation *op,
                                std::optional<CudaReductionPlan> &plan,
                                std::string &fallbackReason) {
  plan.reset();
  fallbackReason.clear();
  auto fallback = [&](StringRef reason) {
    fallbackReason = reason.str();
    return success();
  };
  StringRef name = op->getName().getStringRef();
  if (name != "gac.softmax" && name != "gac.layer_norm")
    return fallback("operation has no cooperative CUDA reduction template");
  bool norm = name == "gac.layer_norm";
  if (op->getNumResults() != 1 || op->getNumOperands() < 1 ||
      op->getNumOperands() > (norm ? 3u : 1u))
    return fallback("reduction template requires one result and input/scale/bias operands");
  for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    int64_t elements;
    if (!tensor || !tensor.getElementType().isF32() || tensor.getEncoding() ||
        !checkedProduct(tensor.getShape(), elements))
      return fallback("reduction template requires static contiguous FP32 tensors");
  }
  auto shape = [](Value v) { return cast<RankedTensorType>(v.getType()).getShape(); };
  CudaReductionPlan p;
  p.layerNorm = norm;
  p.inputShape = SmallVector<int64_t>(shape(op->getOperand(0)));
  if (p.inputShape.empty() || shape(op->getResult(0)) != ArrayRef<int64_t>(p.inputShape))
    return fallback("reduction input and output must have matching non-scalar shapes");
  p.axis = -1;
  if (auto axis = op->getAttrOfType<IntegerAttr>("axis"))
    p.axis = axis.getInt();
  if (p.axis < 0)
    p.axis += p.inputShape.size();
  if (p.axis < 0 || p.axis >= int64_t(p.inputShape.size()))
    return op->emitError("CUDA reduction axis is out of range");
  int64_t elements;
  checkedProduct(p.inputShape, elements);
  if (norm) {
    checkedProduct(ArrayRef<int64_t>(p.inputShape).drop_front(p.axis), p.width);
    if (auto epsilon = op->getAttrOfType<FloatAttr>("epsilon"))
      p.epsilon = epsilon.getValueAsDouble();
    if (!std::isfinite(p.epsilon) || p.epsilon < 0)
      return fallback("LayerNorm template requires a finite nonnegative epsilon");
    if (op->getNumOperands() > 1) {
      p.scaleShape = SmallVector<int64_t>(shape(op->getOperand(1)));
      if (!broadcastsTo(p.scaleShape, p.inputShape))
        return fallback("LayerNorm scale must broadcast to the output");
    }
    if (op->getNumOperands() > 2) {
      p.biasShape = SmallVector<int64_t>(shape(op->getOperand(2)));
      if (!broadcastsTo(p.biasShape, p.inputShape))
        return fallback("LayerNorm bias must broadcast to the output");
    }
  } else {
    p.width = p.inputShape[p.axis];
    checkedProduct(ArrayRef<int64_t>(p.inputShape).drop_front(p.axis + 1), p.stride);
  }
  p.rows = elements / p.width;
  p.gridSize = p.rows;
  p.blockSize = p.width <= 32 ? 32 : p.width <= 128 ? 128 : 256;
  p.sharedMemoryBytes = p.blockSize * 4;
  plan = std::move(p);
  return success();
}

LogicalResult emitCudaReduction(OpBuilder &b, Operation *op,
                                ArrayRef<Value> pointers,
                                const CudaReductionPlan &p) {
  if (pointers.size() != op->getNumOperands() + 1)
    return op->emitError("CUDA reduction pointer ABI mismatch");
  // Capture enclosing objects before constructing nested, initially detached
  // SCF regions. Looking up a module from inside their callback is unsafe.
  auto module = op->getParentOfType<ModuleOp>();
  auto kernel = dyn_cast<LLVM::LLVMFuncOp>(b.getInsertionBlock()->getParentOp());
  if (!module || !kernel)
    return op->emitError("CUDA reduction requires an attached module and kernel");
  Location loc = op->getLoc();
  std::string scratchName = kernel.getSymName().str() + "_reduce_scratch";
  auto scratchType = LLVM::LLVMArrayType::get(b.getF32Type(), p.blockSize);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    auto scratch = b.create<LLVM::GlobalOp>(
        loc, scratchType, false, LLVM::Linkage::Internal, scratchName,
        Attribute(), 16, 3);
    Block *initializer = new Block();
    scratch.getInitializerRegion().push_back(initializer);
    b.setInsertionPointToStart(initializer);
    Value undef = b.create<LLVM::UndefOp>(loc, scratchType);
    b.create<LLVM::ReturnOp>(loc, ValueRange{undef});
  }
  auto sharedPointer = LLVM::LLVMPointerType::get(b.getContext(), 3);
  Value scratch = b.create<LLVM::AddressOfOp>(loc, sharedPointer, scratchName);
  RowBuilder s{b, loc};
  Value lane = b.create<NVVM::ThreadIdXOp>(loc, b.getI32Type());
  Value row = b.create<NVVM::BlockIdXOp>(loc, b.getI32Type());
  // For non-last-axis Softmax, each row is strided; this is a logical address
  // mapping, not a transpose or an additional materialized tensor.
  Value base = s.addi(s.muli(s.divi(row, s.i(p.stride)),
                             s.i(p.width * p.stride)),
                      s.remi(row, s.i(p.stride)));
  auto index = [&](Value k) { return s.addi(base, s.muli(k, s.i(p.stride))); };
  auto read = [&](Value k) { return s.load(pointers[0], index(k)); };
  Value first, second;
  if (p.layerNorm) {
    Value partial = s.laneReduce(lane, p.width, p.blockSize, s.f(0),
        [&](Value k, Value acc) { return s.add(acc, read(k)); });
    first = s.div(s.blockReduce(scratch, lane, p.blockSize, partial), s.f(p.width));
    // Two-pass centered variance avoids E[x*x] - E[x]*E[x] cancellation.
    partial = s.laneReduce(lane, p.width, p.blockSize, s.f(0),
        [&](Value k, Value acc) {
          Value delta = s.sub(read(k), first);
          return s.add(acc, s.mul(delta, delta));
        });
    Value variance = s.div(s.blockReduce(scratch, lane, p.blockSize, partial),
                           s.f(p.width));
    second = b.create<math::SqrtOp>(loc, s.add(variance, s.f(p.epsilon)));
  } else {
    Value partial = s.laneReduce(lane, p.width, p.blockSize, s.f(-INFINITY),
        [&](Value k, Value acc) {
          return Value(b.create<arith::MaximumFOp>(loc, acc, read(k)));
        });
    first = s.blockReduce(scratch, lane, p.blockSize, partial, true);
    partial = s.laneReduce(lane, p.width, p.blockSize, s.f(0),
        [&](Value k, Value acc) {
          return s.add(acc, b.create<math::ExpOp>(loc, s.sub(read(k), first)));
        });
    second = s.blockReduce(scratch, lane, p.blockSize, partial);
  }
  s.laneLoop(lane, p.width, p.blockSize, [&](Value k) {
    Value offset = index(k);
    Value value = s.sub(read(k), first);
    if (!p.layerNorm)
      value = b.create<math::ExpOp>(loc, value);
    value = s.div(value, second);
    if (p.layerNorm && op->getNumOperands() > 1)
      value = s.mul(value, s.load(pointers[1],
          s.broadcastIndex(offset, p.scaleShape, p.inputShape)));
    if (p.layerNorm && op->getNumOperands() > 2)
      value = s.add(value, s.load(pointers[2],
          s.broadcastIndex(offset, p.biasShape, p.inputShape)));
    s.store(value, pointers.back(), offset);
  });
  return success();
}

llvm::json::Object cudaReductionMetadata(const CudaReductionPlan &p) {
  auto shape = [](ArrayRef<int64_t> dimensions) {
    llvm::json::Array result;
    for (int64_t dim : dimensions)
      result.push_back(dim);
    return result;
  };
  return llvm::json::Object{{"schedule", p.schedule},
          {"kind", p.layerNorm ? "layer_norm" : "softmax"},
          {"axis", p.axis}, {"width", p.width}, {"rows", p.rows},
          {"stride", p.stride}, {"block_size", p.blockSize},
          {"grid_size", p.gridSize},
          {"static_shared_memory_bytes", p.sharedMemoryBytes},
          {"input_shape", shape(p.inputShape)},
          {"scale_shape", shape(p.scaleShape)},
          {"bias_shape", shape(p.biasShape)}};
}
} // namespace gac
