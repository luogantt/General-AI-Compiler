#include "gac/templates/CudaMatmul.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;
namespace gac {
namespace {
constexpr int64_t maxIndex = std::numeric_limits<int32_t>::max();
static int64_t ceilDiv(int64_t a, int64_t b) { return (a - 1) / b + 1; }
static bool product(ArrayRef<int64_t> shape, int64_t &result) {
  result = 1;
  for (int64_t d : shape) {
    if (d <= 0 || result > maxIndex / d) return false;
    result *= d;
  }
  return true;
}
static bool broadcastsTo(ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
  if (from.size() > to.size()) return false;
  for (size_t i = 0; i < from.size(); ++i)
    if (from[i] != 1 && from[i] != to[to.size() - from.size() + i]) return false;
  return true;
}
struct TileBuilder {
  OpBuilder &b;
  Location loc;
  Value i(int64_t v) { return b.create<arith::ConstantIntOp>(loc, v, 32); }
  Value zero() { return b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type()); }
  Value add(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value div(Value a, Value c) { return b.create<arith::DivUIOp>(loc, a, c); }
  Value rem(Value a, Value c) { return b.create<arith::RemUIOp>(loc, a, c); }
  Value lt(Value a, Value c) { return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, c); }
  Value both(Value a, Value c) { return b.create<arith::AndIOp>(loc, a, c); }
  Value pointer(Value base, Value offset) {
    return b.create<LLVM::GEPOp>(loc, base.getType(), b.getF32Type(), base, ValueRange{offset});
  }
  Value load(Value base, Value offset) {
    return b.create<LLVM::LoadOp>(loc, b.getF32Type(), pointer(base, offset));
  }
  void store(Value value, Value base, Value offset) {
    b.create<LLVM::StoreOp>(loc, value, pointer(base, offset));
  }
  Value guardedLoad(Value cond, Value base, Value offset) {
    auto branch = b.create<scf::IfOp>(loc, TypeRange{b.getF32Type()}, cond, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.thenBlock());
      b.create<scf::YieldOp>(loc, load(base, offset));
    }
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(branch.elseBlock());
      b.create<scf::YieldOp>(loc, zero());
    }
    return branch.getResult(0);
  }
  void when(Value cond, const std::function<void(TileBuilder &)> &body) {
    auto branch = b.create<scf::IfOp>(loc, cond, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    body(*this);
  }
  void loop(Value start, Value stop, Value step,
            const std::function<void(TileBuilder &, Value)> &body) {
    b.create<scf::ForOp>(loc, start, stop, step, ValueRange{},
        [&](OpBuilder &nested, Location l, Value iv, ValueRange) {
          TileBuilder s{nested, l}; body(s, iv); nested.create<scf::YieldOp>(l);
        });
  }
  Value broadcastIndex(Value flat, ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
    Value result = i(0);
    int64_t sourceStride = 1, targetStride = 1;
    for (int64_t axis = int64_t(to.size()) - 1; axis >= 0; --axis) {
      int64_t source = axis - int64_t(to.size() - from.size());
      if (source >= 0) {
        if (from[source] != 1)
          result = add(result, mul(rem(div(flat, i(targetStride)), i(to[axis])), i(sourceStride)));
        sourceStride *= from[source];
      }
      targetStride *= to[axis];
    }
    return result;
  }
};
} // namespace

LogicalResult planCudaMatmul(Operation *op, const DeviceOptions &options,
                            std::optional<CudaMatmulPlan> &result,
                            std::string &fallbackReason) {
  result.reset(); fallbackReason.clear();
  auto fallback = [&](StringRef reason) { fallbackReason = reason.str(); return success(); };
  StringRef kind = op->getName().getStringRef();
  if (kind != "gac.matmul" && kind != "gac.matmul_bias" && kind != "gac.matmul_bias_relu")
    return success();
  if (options.cudaMatmulSchedule == "scalar") return success();
  if (options.cudaMatmulSchedule != "shared-tiled" && options.cudaMatmulSchedule != "auto")
    return op->emitError("unknown CUDA MatMul schedule");
  bool bias = kind != "gac.matmul";
  if (op->getNumOperands() != (bias ? 3u : 2u) || op->getNumResults() != 1)
    return fallback("CUDA tiled MatMul expects two operands and optional bias");
  for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    int64_t elements;
    if (!tensor || !tensor.getElementType().isF32() || tensor.getEncoding() ||
        !product(tensor.getShape(), elements) || elements > maxIndex / 4)
      return fallback("CUDA tiled MatMul requires static contiguous FP32 tensors with signed 32-bit byte sizes");
  }
  CudaMatmulPlan p;
  auto shape = [](Value v) { return cast<RankedTensorType>(v.getType()).getShape(); };
  p.aShape = SmallVector<int64_t>(shape(op->getOperand(0)));
  p.bShape = SmallVector<int64_t>(shape(op->getOperand(1)));
  p.outShape = SmallVector<int64_t>(shape(op->getResult(0)));
  if (p.aShape.size() < 2 || p.bShape.size() < 2 || p.outShape.size() < 2)
    return fallback("rank-one CUDA MatMul retains scalar lowering");
  p.k = p.aShape.back(); p.n = p.bShape.back();
  if (p.bShape[p.bShape.size() - 2] != p.k)
    return fallback("matrix reduction dimensions disagree");
  p.flattenedProjection = p.bShape.size() == 2;
  if (p.flattenedProjection) {
    product(ArrayRef<int64_t>(p.aShape).drop_back(), p.m);
    SmallVector<int64_t> expected = p.aShape; expected.back() = p.n;
    if (expected != p.outShape) return fallback("linear projection output shape mismatch");
  } else {
    p.m = p.outShape[p.outShape.size() - 2];
    auto outBatch = ArrayRef<int64_t>(p.outShape).drop_back(2);
    if (p.aShape[p.aShape.size() - 2] != p.m || p.outShape.back() != p.n ||
        !broadcastsTo(ArrayRef<int64_t>(p.aShape).drop_back(2), outBatch) ||
        !broadcastsTo(ArrayRef<int64_t>(p.bShape).drop_back(2), outBatch))
      return fallback("batched matrix broadcasting mismatch");
    product(outBatch, p.batches);
  }
  if (bias) {
    p.biasShape = SmallVector<int64_t>(shape(op->getOperand(2)));
    if (!broadcastsTo(p.biasShape, p.outShape)) return fallback("bias broadcasting mismatch");
    if (auto value = op->getAttrOfType<IntegerAttr>("bias_first")) p.biasFirst = value.getInt() != 0;
  }
  p.relu = kind == "gac.matmul_bias_relu";
  p.tileM = options.cudaTileM; p.tileN = options.cudaTileN; p.tileK = options.cudaTileK;
  p.threadTileM = options.cudaThreadTileM; p.threadTileN = options.cudaThreadTileN;
  if (options.cudaMatmulSchedule == "auto") {
    // A conservative shape heuristic informed by scalar/tile32/tile64 GPU
    // measurements. This is not autotuning and does not promise the fastest
    // schedule for every shape/device. Explicit shared-tiled remains available
    // for experiments. Small M or short reductions avoid staging/barrier cost.
    if (p.m < 128 || p.n < 64 || p.k < 64)
      return fallback("CUDA auto heuristic keeps scalar when M<128, N<64 or K<64; no autotuning performed");
    p.tileM = p.tileN = 64;
    p.tileK = 16;
    p.threadTileM = p.threadTileN = 4;
  }
  if (p.tileM <= 0 || p.tileN <= 0 || p.tileK <= 0 || p.tileM > 128 ||
      p.tileN > 128 || p.tileK > 128 || p.threadTileM <= 0 ||
      p.threadTileN <= 0 || p.threadTileM > 8 || p.threadTileN > 8 ||
      p.tileM % p.threadTileM || p.tileN % p.threadTileN)
    return op->emitError("invalid CUDA output/reduction or per-thread tile");
  p.blockSize = (p.tileM / p.threadTileM) * (p.tileN / p.threadTileN);
  p.sharedMemoryBytes = 4 * p.tileK * (p.tileM + p.tileN);
  if (p.blockSize > 1024 || p.blockSize < 32 || p.blockSize % 32 ||
      p.sharedMemoryBytes > 49152)
    return op->emitError("CUDA tiled MatMul exceeds 1024 threads/48 KiB static shared memory or requires incomplete warps");
  p.gridSize = p.batches * ceilDiv(p.m, p.tileM) * ceilDiv(p.n, p.tileN);
  if (p.gridSize > maxIndex) return fallback("CUDA tile grid exceeds signed 32-bit indexing");
  result = p;
  return success();
}

LogicalResult emitCudaMatmul(OpBuilder &b, Operation *op,
                            ArrayRef<Value> pointers, const CudaMatmulPlan &p) {
  if (pointers.size() != op->getNumOperands() + 1)
    return op->emitError("CUDA MatMul template pointer count mismatch");
  TileBuilder s{b, op->getLoc()};
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto kernel = cast<LLVM::LLVMFuncOp>(b.getInsertionBlock()->getParentOp());
  std::string sharedName = (kernel.getName() + "_tiles").str();
  int64_t aElements = p.tileM * p.tileK;
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(module.getBody());
    b.create<LLVM::GlobalOp>(s.loc,
        LLVM::LLVMArrayType::get(b.getF32Type(), p.sharedMemoryBytes / 4),
        false, LLVM::Linkage::Internal, sharedName, Attribute(),
        /*alignment=*/16, /*addrSpace=*/3);
  }
  Value shared = b.create<LLVM::AddressOfOp>(
      s.loc, LLVM::LLVMPointerType::get(b.getContext(), 3), sharedName);
  Value tid = b.create<NVVM::ThreadIdXOp>(s.loc, b.getI32Type());
  Value block = b.create<NVVM::BlockIdXOp>(s.loc, b.getI32Type());
  int64_t tilesN = ceilDiv(p.n, p.tileN), tilesM = ceilDiv(p.m, p.tileM);
  int64_t threadRows = p.tileM / p.threadTileM, threadCols = p.tileN / p.threadTileN;
  Value batch = s.div(block, s.i(tilesM * tilesN));
  Value rowBase = s.mul(s.rem(s.div(block, s.i(tilesN)), s.i(tilesM)), s.i(p.tileM));
  Value colBase = s.mul(s.rem(block, s.i(tilesN)), s.i(p.tileN));
  Value threadRow = s.div(tid, s.i(threadCols)), threadCol = s.rem(tid, s.i(threadCols));
  Value aBase = s.i(0), bBase = s.i(0);
  if (!p.flattenedProjection) {
    auto outBatch = ArrayRef<int64_t>(p.outShape).drop_back(2);
    aBase = s.mul(s.broadcastIndex(batch, ArrayRef<int64_t>(p.aShape).drop_back(2), outBatch), s.i(p.m * p.k));
    bBase = s.mul(s.broadcastIndex(batch, ArrayRef<int64_t>(p.bShape).drop_back(2), outBatch), s.i(p.k * p.n));
  }
  SmallVector<Value> zeros(p.threadTileM * p.threadTileN, s.zero());
  auto outer = b.create<scf::ForOp>(s.loc, s.i(0), s.i(ceilDiv(p.k, p.tileK)), s.i(1), zeros,
      [&](OpBuilder &nested, Location loc, Value tile, ValueRange carried) {
        TileBuilder t{nested, loc};
        Value kBase = t.mul(tile, t.i(p.tileK));
        t.loop(tid, t.i(aElements), t.i(p.blockSize), [&](TileBuilder &u, Value index) {
          Value row = u.add(rowBase, u.div(index, u.i(p.tileK)));
          Value k = u.add(kBase, u.rem(index, u.i(p.tileK)));
          Value valid = u.both(u.lt(row, u.i(p.m)), u.lt(k, u.i(p.k)));
          Value source = u.add(aBase, u.add(u.mul(row, u.i(p.k)), k));
          u.store(u.guardedLoad(valid, pointers[0], source), shared, index);
        });
        t.loop(tid, t.i(p.tileK * p.tileN), t.i(p.blockSize), [&](TileBuilder &u, Value index) {
          Value k = u.add(kBase, u.div(index, u.i(p.tileN)));
          Value col = u.add(colBase, u.rem(index, u.i(p.tileN)));
          Value valid = u.both(u.lt(k, u.i(p.k)), u.lt(col, u.i(p.n)));
          Value source = u.add(bBase, u.add(u.mul(k, u.i(p.n)), col));
          u.store(u.guardedLoad(valid, pointers[1], source), shared, u.add(u.i(aElements), index));
        });
        nested.create<NVVM::Barrier0Op>(loc);
        Value remaining = nested.create<arith::SubIOp>(loc, t.i(p.k), kBase);
        Value validK = nested.create<arith::MinUIOp>(loc, remaining, t.i(p.tileK));
        // Each K step loads RM A and RN B values and reuses them for RM*RN
        // multiply-adds. Accumulators are SSA loop carries (physical registers).
        auto inner = nested.create<scf::ForOp>(loc, t.i(0), validK, t.i(1), carried,
            [&](OpBuilder &ib, Location il, Value k, ValueRange accumulators) {
              TileBuilder u{ib, il};
              SmallVector<Value> a, bv, next;
              for (int64_t r = 0; r < p.threadTileM; ++r)
                a.push_back(u.load(shared, u.add(u.mul(u.add(threadRow, u.i(r * threadRows)), u.i(p.tileK)), k)));
              for (int64_t c = 0; c < p.threadTileN; ++c)
                bv.push_back(u.load(shared, u.add(u.i(aElements), u.add(u.mul(k, u.i(p.tileN)), u.add(threadCol, u.i(c * threadCols))))));
              for (int64_t r = 0; r < p.threadTileM; ++r)
                for (int64_t c = 0; c < p.threadTileN; ++c) {
                  Value product = ib.create<arith::MulFOp>(il, a[r], bv[c]);
                  next.push_back(ib.create<arith::AddFOp>(il, accumulators[r * p.threadTileN + c], product));
                }
              ib.create<scf::YieldOp>(il, next);
            });
        // Consumers finish before producers overwrite the same shared tile.
        nested.create<NVVM::Barrier0Op>(loc);
        nested.create<scf::YieldOp>(loc, inner.getResults());
      });
  for (int64_t r = 0; r < p.threadTileM; ++r)
    for (int64_t c = 0; c < p.threadTileN; ++c) {
      Value row = s.add(rowBase, s.add(threadRow, s.i(r * threadRows)));
      Value col = s.add(colBase, s.add(threadCol, s.i(c * threadCols)));
      s.when(s.both(s.lt(row, s.i(p.m)), s.lt(col, s.i(p.n))), [&](TileBuilder &t) {
        Value index = t.add(t.mul(batch, t.i(p.m * p.n)), t.add(t.mul(row, t.i(p.n)), col));
        Value result = outer.getResult(r * p.threadTileN + c);
        if (op->getNumOperands() == 3) {
          Value bias = t.load(pointers[2], t.broadcastIndex(index, p.biasShape, p.outShape));
          result = p.biasFirst ? t.b.create<arith::AddFOp>(t.loc, bias, result)
                               : t.b.create<arith::AddFOp>(t.loc, result, bias);
        }
        if (p.relu) {
          Value negative = t.b.create<arith::CmpFOp>(t.loc, arith::CmpFPredicate::OLT, result, t.zero());
          result = t.b.create<arith::SelectOp>(t.loc, negative, t.zero(), result);
        }
        t.store(result, pointers.back(), index);
      });
    }
  return success();
}
} // namespace gac
