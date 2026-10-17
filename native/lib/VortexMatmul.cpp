#include "gac/VortexMatmul.h"
#include "gac/VortexLowering.h"
#include "gac/VortexTcu.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <functional>
#include <limits>

using namespace mlir;
namespace gac {
namespace {
constexpr int64_t maxIndex = std::numeric_limits<int32_t>::max();
static bool checkedProduct(ArrayRef<int64_t> shape, int64_t &result) {
  result = 1;
  for (int64_t d : shape) {
    if (d <= 0 || result > maxIndex / d)
      return false;
    result *= d;
  }
  return true;
}
static int64_t ceilDiv(int64_t a, int64_t b) { return (a - 1) / b + 1; }
static bool broadcastsTo(ArrayRef<int64_t> from, ArrayRef<int64_t> to) {
  if (from.size() > to.size())
    return false;
  for (size_t i = 0; i < from.size(); ++i)
    if (from[i] != 1 && from[i] != to[to.size() - from.size() + i])
      return false;
  return true;
}

// Small SSA construction helpers, not a string IR generator. The mathematical
// loop order is identical to scalar MatMul; only operand storage is different.
struct TileBuilder {
  OpBuilder &b;
  Location loc;
  Value i(int64_t v) { return b.create<arith::ConstantIntOp>(loc, v, 32); }
  Value zero() {
    return b.create<arith::ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
  }
  Value add(Value a, Value c) { return b.create<arith::AddIOp>(loc, a, c); }
  Value mul(Value a, Value c) { return b.create<arith::MulIOp>(loc, a, c); }
  Value div(Value a, Value c) { return b.create<arith::DivUIOp>(loc, a, c); }
  Value rem(Value a, Value c) { return b.create<arith::RemUIOp>(loc, a, c); }
  Value lt(Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, a, c);
  }
  Value eq(Value a, Value c) {
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, a, c);
  }
  Value both(Value a, Value c) { return b.create<arith::AndIOp>(loc, a, c); }
  Value pointer(Value base, Value offset) {
    return b.create<LLVM::GEPOp>(loc, base.getType(), b.getF32Type(), base,
                                 ValueRange{offset});
  }
  Value load(Value base, Value offset) {
    return b.create<LLVM::LoadOp>(loc, b.getF32Type(), pointer(base, offset));
  }
  void store(Value value, Value base, Value offset) {
    b.create<LLVM::StoreOp>(loc, value, pointer(base, offset));
  }
  void when(Value cond, const std::function<void(TileBuilder &)> &body) {
    auto branch = b.create<scf::IfOp>(loc, cond, false);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(branch.thenBlock());
    body(*this);
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
  void loop(Value start, Value stop, Value step,
            const std::function<void(TileBuilder &, Value)> &body) {
    b.create<scf::ForOp>(loc, start, stop, step, ValueRange{},
        [&](OpBuilder &nested, Location l, Value iv, ValueRange) {
          TileBuilder s{nested, l};
          body(s, iv);
          nested.create<scf::YieldOp>(l);
        });
  }
  SmallVector<Value> reduce(Value stop, ValueRange initial,
      const std::function<SmallVector<Value>(TileBuilder &, Value, ValueRange)> &body) {
    auto loop = b.create<scf::ForOp>(loc, i(0), stop, i(1), initial,
        [&](OpBuilder &nested, Location l, Value iv, ValueRange carry) {
          TileBuilder s{nested, l};
          SmallVector<Value> next = body(s, iv, carry);
          nested.create<scf::YieldOp>(l, next);
        });
    return SmallVector<Value>(loop.getResults());
  }
  // A full linear output coordinate also handles scalar/channel/full-tensor
  // bias correctly after flattening a higher-rank linear projection.
  Value broadcastIndex(Value flat, ArrayRef<int64_t> from,
                       ArrayRef<int64_t> to) {
    Value result = i(0);
    int64_t inputStride = 1, outputStride = 1;
    for (int64_t axis = int64_t(to.size()) - 1; axis >= 0; --axis) {
      int64_t source = axis - int64_t(to.size() - from.size());
      if (source >= 0) {
        if (from[source] != 1) {
          Value coord = rem(div(flat, i(outputStride)), i(to[axis]));
          result = add(result, mul(coord, i(inputStride)));
        }
        inputStride *= from[source];
      }
      outputStride *= to[axis];
    }
    return result;
  }
};
} // namespace

LogicalResult planVortexMatmul(Operation *op, const DeviceOptions &options,
                              std::optional<VortexMatmulPlan> &result,
                              std::string &fallbackReason) {
  result.reset();
  fallbackReason.clear();
  auto fallback = [&](StringRef why) {
    fallbackReason = why.str();
    return success();
  };
  StringRef kind = op->getName().getStringRef();
  if (kind != "gac.matmul" && kind != "gac.matmul_bias" &&
      kind != "gac.matmul_bias_relu")
    return fallback("operation has no Vortex MatMul template");
  if (options.vortexMatmulSchedule == "scalar")
    return success();
  if (options.vortexMatmulSchedule != "lmem-tiled" &&
      options.vortexMatmulSchedule != "dxa-double-buffer" &&
      options.vortexMatmulSchedule != "tcu")
    return op->emitError("unknown Vortex MatMul schedule");
  if (options.tileM <= 0 || options.tileN <= 0 || options.tileK <= 0 ||
      options.tileM > 65535 || options.tileN > 65535 || options.tileK > 65535 ||
      options.vortexLanes <= 0 || options.vortexLanes > 32 ||
      options.lmemBudget <= 0)
    return op->emitError("invalid Vortex tile, lanes or LMEM budget");
  if (options.vortexThreadTileM <= 0 || options.vortexThreadTileM > 4 ||
      options.vortexThreadTileN <= 0 || options.vortexThreadTileN > 4)
    return op->emitError("Vortex register tile dimensions must be in 1..4");
  if (options.vortexMatmulSchedule == "tcu" &&
      (options.vortexThreadTileM != 1 || options.vortexThreadTileN != 1))
    return op->emitError("TCU uses a fixed fragment layout; LMEM register tile options do not apply");
  bool bias = kind != "gac.matmul";
  if (op->getNumOperands() != (bias ? 3u : 2u) || op->getNumResults() != 1)
    return fallback("template requires two matrices and an optional bias");
  for (Type type : llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    int64_t elements;
    if (!tensor || !tensor.getElementType().isF32() || tensor.getEncoding() ||
        !checkedProduct(tensor.getShape(), elements) || elements > maxIndex / 4)
      return fallback("template requires static contiguous FP32 tensors within RV32 limits");
  }
  VortexMatmulPlan p;
  p.schedule = options.vortexMatmulSchedule;
  p.lanes = options.vortexLanes;
  auto shape = [](Value v) { return cast<RankedTensorType>(v.getType()).getShape(); };
  p.aShape = SmallVector<int64_t>(shape(op->getOperand(0)));
  p.bShape = SmallVector<int64_t>(shape(op->getOperand(1)));
  p.outShape = SmallVector<int64_t>(shape(op->getResult(0)));
  if (p.aShape.size() < 2 || p.bShape.size() < 2 || p.outShape.size() < 2)
    return fallback("rank-one MatMul retains scalar lowering");
  p.k = p.aShape.back();
  p.n = p.bShape.back();
  if (p.bShape[p.bShape.size() - 2] != p.k)
    return fallback("matrix reduction dimensions disagree");
  p.flattenedProjection = p.bShape.size() == 2;
  if (p.flattenedProjection) {
    checkedProduct(ArrayRef<int64_t>(p.aShape).drop_back(), p.m);
    SmallVector<int64_t> expected = p.aShape;
    expected.back() = p.n;
    if (expected != p.outShape)
      return fallback("linear projection output shape mismatch");
  } else {
    p.m = p.outShape[p.outShape.size() - 2];
    auto outBatch = ArrayRef<int64_t>(p.outShape).drop_back(2);
    if (p.aShape[p.aShape.size() - 2] != p.m || p.outShape.back() != p.n ||
        !broadcastsTo(ArrayRef<int64_t>(p.aShape).drop_back(2), outBatch) ||
        !broadcastsTo(ArrayRef<int64_t>(p.bShape).drop_back(2), outBatch))
      return fallback("batched matrix broadcasting mismatch");
    checkedProduct(outBatch, p.batches);
  }
  if (bias) {
    p.biasShape = SmallVector<int64_t>(shape(op->getOperand(2)));
    if (!broadcastsTo(p.biasShape, p.outShape))
      return fallback("bias does not broadcast to the matrix result");
    if (auto a = op->getAttrOfType<IntegerAttr>("bias_first"))
      p.biasFirst = a.getInt() != 0;
  }
  p.relu = kind == "gac.matmul_bias_relu";
  if (p.schedule == "tcu") {
    if (!configureVortexTcuMatmul(options, p, fallbackReason))
      return success();
    result = p;
    return success();
  }
  p.tileM = options.tileM;
  p.tileN = options.tileN;
  p.tileK = options.tileK;
  p.threadTileM = options.vortexThreadTileM;
  p.threadTileN = options.vortexThreadTileN;
  if (p.tileM % p.threadTileM || p.tileN % p.threadTileN)
    return op->emitError("Vortex block tile must be divisible by its register tile");
  if (p.tileM > maxIndex / p.tileN)
    return fallback("tile thread count exceeds RV32 indexing");
  p.blockSize = (p.tileM / p.threadTileM) * (p.tileN / p.threadTileN);
  if (p.blockSize % p.lanes)
    return fallback("collective template requires complete hardware warps");
  p.stages = p.schedule == "dxa-double-buffer" ? 2 : 1;
  p.lmemBytes = p.stages * 4 * p.tileK * (p.tileM + p.tileN);
  if (p.lmemBytes > options.lmemBudget || p.lmemBytes > maxIndex)
    return fallback("tile operands exceed the per-CTA LMEM budget");
  p.gridSize = p.batches * ceilDiv(p.m, p.tileM) * ceilDiv(p.n, p.tileN);
  if (p.gridSize > maxIndex || p.blockSize > maxIndex / p.gridSize)
    return fallback("tile launch exceeds RV32 thread indexing");
  if (p.schedule == "dxa-double-buffer") {
    if (!p.flattenedProjection)
      return fallback("DXA 2D descriptors require one shared rank-2 right matrix");
    if (p.lanes != 4)
      return fallback("DXA WGATHER template is validated for four-lane warps");
    if (std::max({p.m, p.n, p.k}) > 65535)
      return fallback("DXA coordinates exceed the validated 16-bit descriptor range");
  }
  result = p;
  return success();
}

LogicalResult emitVortexMatmul(OpBuilder &builder, Operation *op,
                              ArrayRef<Value> pointers,
                              const VortexMatmulPlan &p) {
  if (p.schedule == "tcu")
    return emitVortexTcuMatmul(builder, op, pointers, p);
  if (pointers.size() != op->getNumOperands() + 1)
    return op->emitError("MatMul template pointer count mismatch");
  TileBuilder s{builder, op->getLoc()};
  ModuleOp module = op->getParentOfType<ModuleOp>();
  bool dxa = p.schedule == "dxa-double-buffer";
  Value tid = createVortexLocalThreadId(s.b, s.loc);
  Value block = createVortexBlockId(s.b, s.loc);
  Value cta = createVortexCTAId(s.b, s.loc);
  Value warps = createVortexWarpCount(s.b, s.loc);
  Value localAddress = createVortexLMEMAddress(s.b, s.loc);
  Value lmem = s.b.create<LLVM::IntToPtrOp>(
      s.loc, LLVM::LLVMPointerType::get(s.b.getContext()), localAddress);
  int64_t tilesN = ceilDiv(p.n, p.tileN), tilesM = ceilDiv(p.m, p.tileM);
  int64_t aElements = p.tileM * p.tileK;
  int64_t stageElements = aElements + p.tileK * p.tileN;
  Value batch = s.div(block, s.i(tilesM * tilesN));
  Value rowBase = s.mul(s.rem(s.div(block, s.i(tilesN)), s.i(tilesM)), s.i(p.tileM));
  Value colBase = s.mul(s.rem(block, s.i(tilesN)), s.i(p.tileN));
  int64_t threadRows = p.tileM / p.threadTileM;
  int64_t threadCols = p.tileN / p.threadTileN;
  Value localRow = s.div(tid, s.i(threadCols));
  Value localCol = s.rem(tid, s.i(threadCols));
  Value aBase = s.i(0), bBase = s.i(0);
  if (!p.flattenedProjection) {
    auto outBatch = ArrayRef<int64_t>(p.outShape).drop_back(2);
    aBase = s.mul(s.broadcastIndex(batch, ArrayRef<int64_t>(p.aShape).drop_back(2),
                                   outBatch), s.i(p.m * p.k));
    bBase = s.mul(s.broadcastIndex(batch, ArrayRef<int64_t>(p.bShape).drop_back(2),
                                   outBatch), s.i(p.k * p.n));
  }
  Value loader;
  if (dxa)
    loader = s.eq(createVortexWarpRank(s.b, s.loc), s.i(0));
  auto stageBarrier = [&](TileBuilder &t, Value stage) {
    return t.add(cta, t.mul(stage, t.i(256)));
  };
  auto issue = [&](TileBuilder &t, Value stage, Value kBase) {
    t.when(loader, [&](TileBuilder &u) {
      Value bar = stageBarrier(u, stage);
      createVortexDxaExpectTwo(u.b, u.loc, module, bar);
      Value base = u.add(localAddress, u.mul(stage, u.i(stageElements * 4)));
      createVortexDxaIssue2D(u.b, u.loc, module, base, bar, u.i(0), kBase, rowBase);
      createVortexDxaIssue2D(u.b, u.loc, module, u.add(base, u.i(aElements * 4)),
                             bar, u.i(1), colBase, kBase);
    });
  };
  if (dxa)
    issue(s, s.i(0), s.i(0));

  // Each lane keeps RM*RN SSA accumulators in registers. Interleaving its
  // output cells preserves contiguous B accesses across adjacent lanes.
  // Every accumulator's K order stays 0..K-1, including partial final tiles.
  SmallVector<Value> initial(p.threadTileM * p.threadTileN, s.zero());
  SmallVector<Value> total = s.reduce(s.i(ceilDiv(p.k, p.tileK)), initial,
      [&](TileBuilder &t, Value tile, ValueRange carried) {
        Value kBase = t.mul(tile, t.i(p.tileK));
        Value stage = dxa ? t.rem(tile, t.i(2)) : t.i(0);
        Value offset = t.mul(stage, t.i(stageElements));
        Value bar = stageBarrier(t, stage);
        if (dxa) {
          Value next = t.add(tile, t.i(1));
          t.when(t.lt(next, t.i(ceilDiv(p.k, p.tileK))), [&](TileBuilder &u) {
            issue(u, u.rem(next, u.i(2)), u.mul(next, u.i(p.tileK)));
          });
        } else {
          // Cooperative global loads materialize both operand tiles in LMEM.
          // Boundary elements are zero-filled; the reduction below still skips
          // padded K to avoid changing NaN/Inf or signed-zero behavior.
          t.loop(tid, t.i(aElements), t.i(p.blockSize),
              [&](TileBuilder &u, Value index) {
                Value r = u.add(rowBase, u.div(index, u.i(p.tileK)));
                Value k = u.add(kBase, u.rem(index, u.i(p.tileK)));
                Value valid = u.both(u.lt(r, u.i(p.m)), u.lt(k, u.i(p.k)));
                Value src = u.add(aBase, u.add(u.mul(r, u.i(p.k)), k));
                u.store(u.guardedLoad(valid, pointers[0], src), lmem, index);
              });
          t.loop(tid, t.i(p.tileK * p.tileN), t.i(p.blockSize),
              [&](TileBuilder &u, Value index) {
                Value k = u.add(kBase, u.div(index, u.i(p.tileN)));
                Value c = u.add(colBase, u.rem(index, u.i(p.tileN)));
                Value valid = u.both(u.lt(k, u.i(p.k)), u.lt(c, u.i(p.n)));
                Value src = u.add(bBase, u.add(u.mul(k, u.i(p.n)), c));
                u.store(u.guardedLoad(valid, pointers[1], src), lmem,
                        u.add(u.i(aElements), index));
              });
        }
        // DXA completion and CTA arrivals must both finish before LMEM reads.
        createVortexBarrier(t.b, t.loc, module, bar, warps);
        Value remaining = t.b.create<arith::SubIOp>(t.loc, t.i(p.k), kBase);
        Value validK = t.b.create<arith::MinUIOp>(t.loc, remaining, t.i(p.tileK));
        SmallVector<Value> sum = t.reduce(validK, carried,
            [&](TileBuilder &u, Value k, ValueRange acc) {
              SmallVector<Value> aValues, bValues, next;
              // Load each A/B fragment only once per K. A value participates
              // in RN products and a B value in RM products in this thread.
              for (int64_t r = 0; r < p.threadTileM; ++r) {
                Value row = u.add(localRow, u.i(r * threadRows));
                Value ai = u.add(offset, u.add(u.mul(row, u.i(p.tileK)), k));
                aValues.push_back(u.load(lmem, ai));
              }
              for (int64_t c = 0; c < p.threadTileN; ++c) {
                Value col = u.add(localCol, u.i(c * threadCols));
                Value bi = u.add(offset, u.add(u.i(aElements),
                                               u.add(u.mul(k, u.i(p.tileN)), col)));
                bValues.push_back(u.load(lmem, bi));
              }
              for (int64_t r = 0; r < p.threadTileM; ++r)
                for (int64_t c = 0; c < p.threadTileN; ++c) {
                  Value product = u.b.create<arith::MulFOp>(u.loc, aValues[r], bValues[c]);
                  next.push_back(u.b.create<arith::AddFOp>(
                      u.loc, acc[r * p.threadTileN + c], product));
                }
              return next;
            });
        // No writer may reuse this stage while another warp still reads it.
        createVortexBarrier(t.b, t.loc, module, bar, warps);
        return sum;
      });
  for (int64_t r = 0; r < p.threadTileM; ++r)
    for (int64_t c = 0; c < p.threadTileN; ++c) {
      Value row = s.add(rowBase, s.add(localRow, s.i(r * threadRows)));
      Value col = s.add(colBase, s.add(localCol, s.i(c * threadCols)));
      Value validOutput = s.both(s.lt(row, s.i(p.m)), s.lt(col, s.i(p.n)));
      s.when(validOutput, [&](TileBuilder &t) {
        Value flat = t.add(t.mul(batch, t.i(p.m * p.n)),
                           t.add(t.mul(row, t.i(p.n)), col));
        Value result = total[r * p.threadTileN + c];
        if (op->getNumOperands() == 3) {
          Value bias = t.load(pointers[2], t.broadcastIndex(flat, p.biasShape, p.outShape));
          result = p.biasFirst ? t.b.create<arith::AddFOp>(t.loc, bias, result)
                               : t.b.create<arith::AddFOp>(t.loc, result, bias);
        }
        if (p.relu) {
          Value negative = t.b.create<arith::CmpFOp>(
              t.loc, arith::CmpFPredicate::OLT, result, t.zero());
          result = t.b.create<arith::SelectOp>(t.loc, negative, t.zero(), result);
        }
        t.store(result, pointers.back(), flat);
      });
    }
  return success();
}
} // namespace gac
