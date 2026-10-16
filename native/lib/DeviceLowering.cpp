#include "gac/DeviceLowering.h"
#include "gac/CudaLowering.h"
#include "gac/VortexLowering.h"
#include "gac/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

using namespace mlir;
namespace gac {
namespace {
static int64_t product(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t(1),
                         std::multiplies<int64_t>());
}
static int64_t integer(Operation *op, StringRef name, int64_t fallback) {
  if (auto a = op->getAttrOfType<IntegerAttr>(name))
    return a.getInt();
  return fallback;
}
static double floating(Operation *op, StringRef name, double fallback) {
  if (auto a = op->getAttrOfType<FloatAttr>(name))
    return a.getValueAsDouble();
  return fallback;
}
static std::string stringAttr(Operation *op, StringRef name,
                              StringRef fallback = "") {
  if (auto a = op->getAttrOfType<StringAttr>(name))
    return a.str();
  return fallback.str();
}
static SmallVector<int64_t> ints(Operation *op, StringRef name,
                                 ArrayRef<int64_t> fallback = {}) {
  if (auto a = op->getAttrOfType<DenseI64ArrayAttr>(name))
    return SmallVector<int64_t>(a.asArrayRef());
  if (auto a = op->getAttrOfType<ArrayAttr>(name)) {
    SmallVector<int64_t> values;
    for (Attribute v : a)
      values.push_back(cast<IntegerAttr>(v).getInt());
    return values;
  }
  return SmallVector<int64_t>(fallback);
}
static SmallVector<int64_t> constantInts(Value value) {
  auto *op = value.getDefiningOp();
  auto attr = op ? op->getAttrOfType<DenseIntElementsAttr>("value")
                 : DenseIntElementsAttr();
  SmallVector<int64_t> result;
  if (attr)
    for (auto n : attr.getValues<APInt>())
      result.push_back(n.getSExtValue());
  return result;
}
static RankedTensorType tensor(Value v) {
  return cast<RankedTensorType>(v.getType());
}
// ONNX optional operands retain their original positions after import.
static Value operandAt(Operation *op, int64_t position) {
  auto positions = ints(op, "gac.input_positions");
  if (positions.empty())
    return position < op->getNumOperands() ? op->getOperand(position) : Value();
  for (auto [index, original] : llvm::enumerate(positions))
    if (original == position)
      return op->getOperand(index);
  return {};
}
static StringRef opKind(Operation *op) {
  return op->getName().getStringRef().drop_front(4);
}

// The builder owns no graph or string IR. Every helper creates an actual MLIR
// operation with typed SSA operands; standard MLIR passes lower these further.
struct Scalar {
  OpBuilder &b;
  Location loc;
  // A read-only device global generated from constant ScatterND indices.
  // It selects update slices in O(1), without racy scatter stores.
  Value scatterLookup;
  Value i(int64_t n) { return b.create<arith::ConstantIntOp>(loc, n, 32); }
  Value f(double n) {
    return b.create<arith::ConstantFloatOp>(loc, APFloat(float(n)),
                                            b.getF32Type());
  }
  Value zero(Type t) {
    return b.create<arith::ConstantOp>(loc, t, b.getZeroAttr(t));
  }
  Value cvt(Value v, Type to) {
    Type from = v.getType();
    if (from == to)
      return v;
    if (to.isInteger(1))
      return cmp("ne", v, zero(from));
    if (isa<FloatType>(from) && isa<FloatType>(to)) {
      if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth())
        return b.create<arith::ExtFOp>(loc, to, v);
      return b.create<arith::TruncFOp>(loc, to, v);
    }
    if (isa<FloatType>(from))
      return b.create<arith::FPToSIOp>(loc, to, v);
    if (isa<FloatType>(to)) {
      if (from.isInteger(1))
        return b.create<arith::UIToFPOp>(loc, to, v);
      return b.create<arith::SIToFPOp>(loc, to, v);
    }
    if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth()) {
      if (from.isInteger(1))
        return b.create<arith::ExtUIOp>(loc, to, v);
      return b.create<arith::ExtSIOp>(loc, to, v);
    }
    return b.create<arith::TruncIOp>(loc, to, v);
  }
  Value binary(StringRef name, Value a, Value c) {
    bool fp = isa<FloatType>(a.getType());
    if (name == "add")
      return fp ? Value(b.create<arith::AddFOp>(loc, a, c))
                : Value(b.create<arith::AddIOp>(loc, a, c));
    if (name == "sub")
      return fp ? Value(b.create<arith::SubFOp>(loc, a, c))
                : Value(b.create<arith::SubIOp>(loc, a, c));
    if (name == "mul")
      return fp ? Value(b.create<arith::MulFOp>(loc, a, c))
                : Value(b.create<arith::MulIOp>(loc, a, c));
    if (name == "div")
      return fp ? Value(b.create<arith::DivFOp>(loc, a, c))
                : Value(b.create<arith::DivSIOp>(loc, a, c));
    if (name == "and")
      return b.create<arith::AndIOp>(loc, a, c);
    return b.create<arith::RemSIOp>(loc, a, c);
  }
  Value add(Value a, Value c) { return binary("add", a, c); }
  Value sub(Value a, Value c) { return binary("sub", a, c); }
  Value mul(Value a, Value c) { return binary("mul", a, c); }
  Value div(Value a, Value c) { return binary("div", a, c); }
  Value rem(Value a, Value c) { return binary("rem", a, c); }
  Value cmp(StringRef p, Value a, Value c) {
    if (isa<FloatType>(a.getType())) {
      auto pred = p == "lt"   ? arith::CmpFPredicate::OLT
                  : p == "le" ? arith::CmpFPredicate::OLE
                  : p == "gt" ? arith::CmpFPredicate::OGT
                  : p == "ge" ? arith::CmpFPredicate::OGE
                  : p == "eq" ? arith::CmpFPredicate::OEQ
                              : arith::CmpFPredicate::UNE;
      return b.create<arith::CmpFOp>(loc, pred, a, c);
    }
    auto pred = p == "lt"   ? arith::CmpIPredicate::slt
                : p == "le" ? arith::CmpIPredicate::sle
                : p == "gt" ? arith::CmpIPredicate::sgt
                : p == "ge" ? arith::CmpIPredicate::sge
                : p == "eq" ? arith::CmpIPredicate::eq
                            : arith::CmpIPredicate::ne;
    return b.create<arith::CmpIOp>(loc, pred, a, c);
  }
  Value select(Value p, Value a, Value c) {
    return b.create<arith::SelectOp>(loc, p, a, c);
  }
  Value maximum(Value a, Value c) {
    if (isa<FloatType>(a.getType()))
      return b.create<arith::MaximumFOp>(loc, a, c);
    return select(cmp("gt", a, c), a, c);
  }
  Value relu(Value v) {
    return select(cmp("lt", v, zero(v.getType())), zero(v.getType()), v);
  }
  Value address(Value ptr, Value index, Type element) {
    return b.create<LLVM::GEPOp>(loc, ptr.getType(), element, ptr,
                                 ValueRange{index});
  }
  Value read(Value ptr, Value index, Type element) {
    Type storage = element.isInteger(1) ? Type(b.getI8Type()) : element;
    Value v =
        b.create<LLVM::LoadOp>(loc, storage, address(ptr, index, storage));
    if (element.isInteger(1))
      return cmp("ne", v, zero(storage));
    return isa<FloatType>(element) ? cvt(v, b.getF32Type()) : v;
  }
  void write(Value v, Value ptr, Value index, Type element) {
    Type storage = element.isInteger(1) ? Type(b.getI8Type()) : element;
    b.create<LLVM::StoreOp>(loc, cvt(v, storage), address(ptr, index, storage));
  }
  SmallVector<Value> coords(Value index, ArrayRef<int64_t> shape) {
    SmallVector<Value> result;
    int64_t stride = product(shape);
    for (int64_t n : shape) {
      stride /= n;
      result.push_back(rem(div(index, i(stride)), i(n)));
    }
    return result;
  }
  Value linear(ArrayRef<Value> coords, ArrayRef<int64_t> shape) {
    Value offset = i(0);
    for (size_t d = 0; d < shape.size(); ++d)
      offset = add(mul(offset, i(shape[d])), coords[d]);
    return offset;
  }
  Value broadcast(Value index, ArrayRef<int64_t> out, ArrayRef<int64_t> in) {
    auto c = coords(index, out);
    SmallVector<Value> mapped;
    for (size_t d = 0; d < in.size(); ++d)
      mapped.push_back(in[d] == 1 ? i(0) : c[out.size() - in.size() + d]);
    return linear(mapped, in);
  }
  Value loop(int64_t count, Value initial,
             std::function<Value(Value, Value)> body) {
    Value begin = i(0), end = i(count), step = i(1);
    auto forOp = b.create<scf::ForOp>(
        loc, begin, end, step, ValueRange{initial},
        [&](OpBuilder &, Location, Value iv, ValueRange carry) {
          Value next = body(iv, carry[0]);
          b.create<scf::YieldOp>(loc, next);
        });
    return forOp.getResult(0);
  }
  Value branch(Value cond, Type type, std::function<Value()> yes,
               std::function<Value()> no) {
    auto op = b.create<scf::IfOp>(loc, TypeRange{type}, cond, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(op.thenBlock());
      b.create<scf::YieldOp>(loc, yes());
    }
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(op.elseBlock());
      b.create<scf::YieldOp>(loc, no());
    }
    return op.getResult(0);
  }
  // Absolute pixel coordinates; zero padding. Guard before FP-to-int and
  // before every load, including fractional coordinates outside the image.
  // The callback supplies the layout (NCHW for DCN, BSHD for attention).
  Value bilinear(Value x, Value y, int64_t height, int64_t width,
                 std::function<Value(Value, Value)> load) {
    Value valid = binary(
        "and", binary("and", cmp("gt", x, f(-1)), cmp("lt", x, f(width))),
        binary("and", cmp("gt", y, f(-1)), cmp("lt", y, f(height))));
    return branch(
        valid, b.getF32Type(),
        [&] {
          auto floor = [&](Value v) {
            Value t = cvt(v, b.getI32Type());
            return select(cmp("lt", v, cvt(t, b.getF32Type())), sub(t, i(1)),
                          t);
          };
          Value x0 = floor(x), y0 = floor(y);
          Value dx = sub(x, cvt(x0, b.getF32Type()));
          Value dy = sub(y, cvt(y0, b.getF32Type()));
          auto sample = [&](Value xx, Value yy) {
            Value inside = binary(
                "and",
                binary("and", cmp("ge", xx, i(0)), cmp("lt", xx, i(width))),
                binary("and", cmp("ge", yy, i(0)), cmp("lt", yy, i(height))));
            return branch(
                inside, b.getF32Type(), [&] { return load(xx, yy); },
                [&] { return f(0); });
          };
          Value a = mul(sample(x0, y0), mul(sub(f(1), dx), sub(f(1), dy)));
          Value c = mul(sample(add(x0, i(1)), y0), mul(dx, sub(f(1), dy)));
          Value d = mul(sample(x0, add(y0, i(1))), mul(sub(f(1), dx), dy));
          Value e = mul(sample(add(x0, i(1)), add(y0, i(1))), mul(dx, dy));
          return add(add(a, c), add(d, e));
        },
        [&] { return f(0); });
  }
};

static Value emitElement(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid);
static Value emitSpatial(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid);

static Value emitMatmul(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                        Value tid) {
  auto left = tensor(op->getOperand(0)), right = tensor(op->getOperand(1));
  auto out = tensor(op->getResult(0));
  auto ls = left.getShape(), rs = right.getShape(), os = out.getShape();
  auto c = s.coords(tid, os);
  bool lv = ls.size() == 1, rv = rs.size() == 1;
  unsigned batchRank = os.size() - unsigned(!lv) - unsigned(!rv);
  Value row = lv ? s.i(0) : c[batchRank], col = rv ? s.i(0) : c.back();
  auto batch = [&](ArrayRef<int64_t> shape) {
    SmallVector<Value> result;
    for (size_t d = 0; d + 2 < shape.size(); ++d)
      result.push_back(shape[d] == 1 ? s.i(0)
                                     : c[batchRank - (shape.size() - 2) + d]);
    return result;
  };
  auto lc = batch(ls), rc = batch(rs);
  Type compute = isa<FloatType>(left.getElementType()) ? Type(s.b.getF32Type())
                                                       : left.getElementType();
  Value v = s.loop(ls.back(), s.zero(compute), [&](Value k, Value acc) {
    SmallVector<Value> l = lc, r = rc;
    if (!lv)
      l.push_back(row);
    l.push_back(k);
    r.push_back(k);
    if (!rv)
      r.push_back(col);
    return s.add(
        acc, s.mul(s.read(ptrs[0], s.linear(l, ls), left.getElementType()),
                   s.read(ptrs[1], s.linear(r, rs), right.getElementType())));
  });
  if (opKind(op) == "matmul_bias" || opKind(op) == "matmul_bias_relu") {
    auto bias = tensor(op->getOperand(2));
    Value value = s.read(ptrs[2], s.broadcast(tid, os, bias.getShape()),
                         bias.getElementType());
    // Keep both the original Add operand order and the FP32 reduction's
    // rounding boundary. This is a separate fadd after the MatMul loop.
    v = integer(op, "bias_first", 0) ? s.add(value, v) : s.add(v, value);
    if (opKind(op) == "matmul_bias_relu")
      v = s.relu(v);
  }
  return v;
}

static Value emitElement(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid) {
  StringRef kind = opKind(op);
  auto out = tensor(op->getResult(0));
  auto os = out.getShape();
  auto readAt = [&](unsigned p, Value idx) {
    return s.read(ptrs[p], idx, tensor(op->getOperand(p)).getElementType());
  };
  auto read = [&](unsigned p) {
    return readAt(p,
                  s.broadcast(tid, os, tensor(op->getOperand(p)).getShape()));
  };
  if (kind == "identity" || kind == "reshape" || kind == "squeeze" ||
      kind == "unsqueeze" || kind == "cast")
    return s.cvt(readAt(0, tid), isa<FloatType>(out.getElementType())
                                     ? Type(s.b.getF32Type())
                                     : out.getElementType());
  if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" ||
      kind == "and" || kind == "add_relu") {
    Value v = s.binary(kind == "add_relu" ? "add" : kind, read(0), read(1));
    return kind == "add_relu" ? s.relu(v) : v;
  }
  if (kind == "equal" || kind == "greater" || kind == "less")
    return s.cmp(kind == "equal"     ? "eq"
                 : kind == "greater" ? "gt"
                                     : "lt",
                 read(0), read(1));
  if (kind == "where")
    return s.select(read(0), read(1), read(2));
  if (kind == "max") {
    Value v = read(0);
    for (unsigned p = 1; p < ptrs.size(); ++p)
      v = s.maximum(v, read(p));
    return v;
  }
  if (kind == "relu")
    return s.relu(read(0));
  if (kind == "exp")
    return s.b.create<math::ExpOp>(s.loc, read(0));
  if (kind == "log")
    return s.b.create<math::LogOp>(s.loc, read(0));
  if (kind == "sin")
    return s.b.create<math::SinOp>(s.loc, read(0));
  if (kind == "cos")
    return s.b.create<math::CosOp>(s.loc, read(0));
  if (kind == "sqrt")
    return s.b.create<math::SqrtOp>(s.loc, read(0));
  if (kind == "pow")
    return s.b.create<math::PowFOp>(s.loc, read(0), read(1));
  if (kind == "sigmoid") {
    Value x = read(0), positive = s.cmp("ge", x, s.f(0));
    Value abs = s.select(positive, x, s.sub(s.f(0), x));
    Value e = s.b.create<math::ExpOp>(s.loc, s.sub(s.f(0), abs));
    Value denominator = s.add(s.f(1), e);
    return s.select(positive, s.div(s.f(1), denominator),
                    s.div(e, denominator));
  }
  if (kind == "clamp" || kind == "clip") {
    Value x = read(0);
    // Optional-input positions are checked before this emitter runs.
    if (ptrs.size() > 1) {
      Value lo = read(1);
      x = s.select(s.cmp("lt", x, lo), lo, x);
    }
    if (ptrs.size() > 2) {
      Value hi = read(2);
      x = s.select(s.cmp("gt", x, hi), hi, x);
    }
    return x;
  }
  if (kind == "matmul" || kind == "matmul_bias" || kind == "matmul_bias_relu")
    return emitMatmul(s, op, ptrs, tid);
  if (kind == "expand")
    return read(0);
  if (kind == "tile") {
    auto shape = tensor(op->getOperand(0)).getShape();
    auto c = s.coords(tid, os);
    for (size_t d = 0; d < shape.size(); ++d)
      c[d] = s.rem(c[d], s.i(shape[d]));
    return readAt(0, s.linear(c, shape));
  }
  if (kind == "constant_of_shape") {
    Type element = out.getElementType();
    auto fill = op->getAttrOfType<DenseElementsAttr>("value");
    Value v = fill ? Value(s.b.create<arith::ConstantOp>(
                         s.loc, element,
                         cast<TypedAttr>(*fill.getValues<Attribute>().begin())))
                   : s.zero(element);
    return isa<FloatType>(element) ? s.cvt(v, s.b.getF32Type()) : v;
  }
  if (kind == "shape") {
    auto shape = tensor(op->getOperand(0)).getShape();
    int64_t rank = shape.size(), begin = integer(op, "start", 0);
    begin = std::clamp<int64_t>(begin < 0 ? begin + rank : begin, 0, rank);
    Value result = s.i(0);
    for (int64_t i = 0; i < out.getNumElements(); ++i)
      result =
          s.select(s.cmp("eq", tid, s.i(i)), s.i(shape[begin + i]), result);
    return s.cvt(result, out.getElementType());
  }
  if (kind == "scatter_nd") {
    auto ds = tensor(op->getOperand(0)).getShape();
    auto is = tensor(op->getOperand(1)).getShape();
    int64_t slice = product(ds.drop_front(is.back()));
    Value update =
        s.read(s.scatterLookup, s.div(tid, s.i(slice)), s.b.getI32Type());
    Type compute = isa<FloatType>(out.getElementType()) ? Type(s.b.getF32Type())
                                                        : out.getElementType();
    return s.branch(
        s.cmp("ge", update, s.i(0)), compute,
        [&] {
          return readAt(
              2, s.add(s.mul(update, s.i(slice)), s.rem(tid, s.i(slice))));
        },
        [&] { return readAt(0, tid); });
  }
  if (kind == "transpose") {
    auto shape = tensor(op->getOperand(0)).getShape();
    auto perm = ints(op, "perm");
    if (perm.empty())
      for (int64_t d = shape.size(); d > 0; --d)
        perm.push_back(d - 1);
    auto c = s.coords(tid, os);
    SmallVector<Value> src(c.size());
    for (size_t d = 0; d < c.size(); ++d)
      src[perm[d]] = c[d];
    return readAt(0, s.linear(src, shape));
  }
  if (kind == "concat") {
    int64_t axis = integer(op, "axis", 0);
    if (axis < 0)
      axis += os.size();
    auto c = s.coords(tid, os);
    std::function<Value(unsigned, int64_t)> pick = [&](unsigned p,
                                                       int64_t begin) -> Value {
      auto shape = tensor(op->getOperand(p)).getShape();
      auto selected = [&] {
        auto src = c;
        src[axis] = s.sub(c[axis], s.i(begin));
        return readAt(p, s.linear(src, shape));
      };
      if (p + 1 == ptrs.size())
        return selected();
      int64_t end = begin + shape[axis];
      Type t = isa<FloatType>(out.getElementType()) ? Type(s.b.getF32Type())
                                                    : out.getElementType();
      return s.branch(s.cmp("lt", c[axis], s.i(end)), t, selected,
                      [&] { return pick(p + 1, end); });
    };
    return pick(0, 0);
  }
  if (kind == "softmax") {
    auto shape = tensor(op->getOperand(0)).getShape();
    int64_t axis = integer(op, "axis", -1);
    if (axis < 0)
      axis += shape.size();
    int64_t n = shape[axis], stride = product(shape.drop_front(axis + 1));
    Value base =
        s.sub(tid, s.mul(s.rem(s.div(tid, s.i(stride)), s.i(n)), s.i(stride)));
    auto at = [&](Value k) {
      return readAt(0, s.add(base, s.mul(k, s.i(stride))));
    };
    Value max = s.loop(n, s.f(-INFINITY),
                       [&](Value k, Value a) { return s.maximum(a, at(k)); });
    Value sum = s.loop(n, s.f(0), [&](Value k, Value a) {
      return s.add(a, s.b.create<math::ExpOp>(s.loc, s.sub(at(k), max)));
    });
    return s.div(s.b.create<math::ExpOp>(s.loc, s.sub(readAt(0, tid), max)),
                 sum);
  }
  if (kind == "layer_norm") {
    auto shape = tensor(op->getOperand(0)).getShape();
    int64_t axis = integer(op, "axis", -1);
    if (axis < 0)
      axis += shape.size();
    int64_t n = product(shape.drop_front(axis));
    Value base = s.mul(s.div(tid, s.i(n)), s.i(n));
    Value mean = s.div(s.loop(n, s.f(0),
                              [&](Value k, Value a) {
                                return s.add(a, readAt(0, s.add(base, k)));
                              }),
                       s.f(n));
    Value var = s.div(s.loop(n, s.f(0),
                             [&](Value k, Value a) {
                               Value d = s.sub(readAt(0, s.add(base, k)), mean);
                               return s.add(a, s.mul(d, d));
                             }),
                      s.f(n));
    Value v = s.div(s.sub(readAt(0, tid), mean),
                    s.b.create<math::SqrtOp>(
                        s.loc, s.add(var, s.f(floating(op, "epsilon", 1e-5)))));
    if (ptrs.size() > 1)
      v = s.mul(v, read(1));
    if (ptrs.size() > 2)
      v = s.add(v, read(2));
    return v;
  }
  if (kind == "reduce_sum" || kind == "reduce_mean") {
    auto shape = tensor(op->getOperand(0)).getShape();
    auto axes = ints(op, "axes");
    if (op->getNumOperands() > 1)
      axes = constantInts(op->getOperand(1));
    if (axes.empty() && integer(op, "noop_with_empty_axes", 0))
      return readAt(0, tid);
    if (axes.empty())
      for (int64_t d = 0; d < int64_t(shape.size()); ++d)
        axes.push_back(d);
    for (auto &a : axes)
      if (a < 0)
        a += shape.size();
    llvm::sort(axes);
    auto c = s.coords(tid, os);
    SmallVector<Value> src;
    SmallVector<int64_t> reductionShape;
    bool keep = integer(op, "keepdims", 1);
    unsigned axis = 0;
    for (int64_t d = 0; d < int64_t(shape.size()); ++d) {
      if (llvm::is_contained(axes, d)) {
        src.push_back(s.i(0));
        reductionShape.push_back(shape[d]);
        if (keep)
          ++axis;
      } else
        src.push_back(c[axis++]);
    }
    Type compute = isa<FloatType>(out.getElementType()) ? Type(s.b.getF32Type())
                                                        : out.getElementType();
    int64_t n = product(reductionShape);
    Value sum = s.loop(n, s.zero(compute), [&](Value k, Value a) {
      auto rc = s.coords(k, reductionShape), all = src;
      for (size_t d = 0; d < axes.size(); ++d)
        all[axes[d]] = rc[d];
      return s.add(a, readAt(0, s.linear(all, shape)));
    });
    return kind == "reduce_mean"
               ? s.div(sum, isa<FloatType>(compute) ? s.f(n)
                                                    : s.cvt(s.i(n), compute))
               : sum;
  }
  if (kind == "gather") {
    auto ds = tensor(op->getOperand(0)).getShape(),
         is = tensor(op->getOperand(1)).getShape();
    int64_t axis = integer(op, "axis", 0);
    if (axis < 0)
      axis += ds.size();
    auto c = s.coords(tid, os);
    SmallVector<Value> indices(c.begin() + axis, c.begin() + axis + is.size());
    Value pick = s.cvt(readAt(1, s.linear(indices, is)), s.b.getI32Type());
    pick =
        s.select(s.cmp("lt", pick, s.i(0)), s.add(pick, s.i(ds[axis])), pick);
    SmallVector<Value> dc;
    for (size_t d = 0; d < ds.size(); ++d)
      dc.push_back(d < size_t(axis)    ? c[d]
                   : d == size_t(axis) ? pick
                                       : c[d - 1 + is.size()]);
    // The verifier requires constant indices and checks their range, so the
    // generated load is valid without a data-dependent out-of-bounds fallback.
    return readAt(0, s.linear(dc, ds));
  }
  return emitSpatial(s, op, ptrs, tid);
}

// Spatial implementations and target/runtime boundaries follow below.
static Value emitSpatial(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid) {
  StringRef kind = opKind(op);
  auto os = tensor(op->getResult(0)).getShape();
  auto read = [&](unsigned p, Value idx) {
    return s.read(ptrs[p], idx, tensor(op->getOperand(p)).getElementType());
  };
  if (kind == "deform_conv2d") {
    auto xs = tensor(op->getOperand(0)).getShape();
    auto offsets = tensor(op->getOperand(1)).getShape();
    auto masks = tensor(op->getOperand(2)).getShape();
    auto ws = tensor(op->getOperand(3)).getShape();
    auto strides = ints(op, "stride", {1, 1});
    auto pads = ints(op, "padding", {0, 0});
    auto dilation = ints(op, "dilation", {1, 1});
    int64_t groups = integer(op, "groups", 1);
    int64_t deformGroups = integer(op, "deform_groups", 1);
    int64_t points = ws[2] * ws[3];
    auto c = s.coords(tid, os);
    // Each output owns its accumulator. Sample directly from the feature map;
    // no im2col activation or host implementation is materialized.
    Value acc = s.loop(ws[1] * points, s.f(0), [&](Value k, Value sum) {
      auto r = s.coords(k, {ws[1], ws[2], ws[3]});
      Value channel =
          s.add(s.mul(s.div(c[1], s.i(os[1] / groups)), s.i(ws[1])), r[0]);
      Value deformGroup = s.div(channel, s.i(xs[1] / deformGroups));
      Value point = s.add(s.mul(r[1], s.i(ws[3])), r[2]);
      Value maskChannel = s.add(s.mul(deformGroup, s.i(points)), point);
      Value offsetChannel = s.mul(maskChannel, s.i(2));
      Value dy = read(1, s.linear({c[0], offsetChannel, c[2], c[3]}, offsets));
      Value dx =
          read(1, s.linear({c[0], s.add(offsetChannel, s.i(1)), c[2], c[3]},
                           offsets));
      Value y = s.add(s.cvt(s.sub(s.add(s.mul(c[2], s.i(strides[0])),
                                        s.mul(r[1], s.i(dilation[0]))),
                                  s.i(pads[0])),
                            s.b.getF32Type()),
                      dy);
      Value x = s.add(s.cvt(s.sub(s.add(s.mul(c[3], s.i(strides[1])),
                                        s.mul(r[2], s.i(dilation[1]))),
                                  s.i(pads[1])),
                            s.b.getF32Type()),
                      dx);
      Value sample = s.bilinear(x, y, xs[2], xs[3], [&](Value xx, Value yy) {
        return read(0, s.linear({c[0], channel, yy, xx}, xs));
      });
      Value mask = read(2, s.linear({c[0], maskChannel, c[2], c[3]}, masks));
      Value weight = read(3, s.linear({c[1], r[0], r[1], r[2]}, ws));
      return s.add(sum, s.mul(s.mul(sample, mask), weight));
    });
    return ptrs.size() == 5 ? s.add(acc, read(4, c[1])) : acc;
  }
  if (kind == "ms_deform_attn" || kind == "ms_deform_attn_offsets") {
    bool fusedCoordinates = kind == "ms_deform_attn_offsets";
    auto vs = tensor(op->getOperand(0)).getShape();
    auto spatial = constantInts(op->getOperand(1));
    auto ls = tensor(op->getOperand(2)).getShape();
    auto as = tensor(op->getOperand(3)).getShape();
    auto c = s.coords(tid, os);
    auto coordinateShape = ints(op, "coordinate_shape");
    auto coordinate = [&](Value index) {
      Value result = read(2, index);
      if (!fusedCoordinates)
        return result;
      auto normalizer = tensor(op->getOperand(4)).getShape();
      auto reference = tensor(op->getOperand(5)).getShape();
      result = s.div(result, read(4, s.broadcast(index, ls, normalizer)));
      Value base = read(5, s.broadcast(index, coordinateShape, reference));
      // Separate FP32 fdiv/fadd preserve the original materialized rounding
      // points. Only the flat index is shared across the two logical shapes.
      return integer(op, "reference_first", 0) ? s.add(base, result)
                                               : s.add(result, base);
    };
    Value head = s.div(c[2], s.i(vs[3])), channel = s.rem(c[2], s.i(vs[3]));
    Value sum = s.f(0);
    int64_t start = 0;
    // Static levels, a point loop and one register accumulator per output.
    // This streams L*P samples: it never allocates [B,H,D,Q,L,P].
    for (int64_t level = 0; level < ls[3]; ++level) {
      int64_t height = spatial[2 * level], width = spatial[2 * level + 1];
      sum = s.loop(ls[4], sum, [&](Value point, Value acc) {
        Value x = coordinate(
            s.linear({c[0], c[1], head, s.i(level), point, s.i(0)}, ls));
        Value y = coordinate(
            s.linear({c[0], c[1], head, s.i(level), point, s.i(1)}, ls));
        x = s.sub(s.mul(x, s.f(width)), s.f(0.5));
        y = s.sub(s.mul(y, s.f(height)), s.f(0.5));
        Value sample = s.bilinear(x, y, height, width, [&](Value xx, Value yy) {
          Value spatialIndex =
              s.add(s.i(start), s.add(s.mul(yy, s.i(width)), xx));
          return read(0, s.linear({c[0], spatialIndex, head, channel}, vs));
        });
        Value weight =
            read(3, s.linear({c[0], c[1], head, s.i(level), point}, as));
        return s.add(acc, s.mul(sample, weight));
      });
      start += height * width;
    }
    return sum;
  }
  if (kind == "resize") {
    auto shape = tensor(op->getOperand(0)).getShape();
    auto c = s.coords(tid, os);
    auto axes = ints(op, "axes");
    if (axes.empty())
      for (int64_t d = 0; d < int64_t(shape.size()); ++d)
        axes.push_back(d);
    auto scaleValue = operandAt(op, 2);
    auto scaleOp = scaleValue ? scaleValue.getDefiningOp() : nullptr;
    auto scales = scaleOp ? scaleOp->getAttrOfType<DenseFPElementsAttr>("value")
                          : DenseFPElementsAttr();
    SmallVector<double> scale;
    if (scales)
      for (APFloat value : scales.getValues<APFloat>())
        scale.push_back(value.convertToDouble());
    for (auto [index, axis] : llvm::enumerate(axes)) {
      if (axis < 0)
        axis += shape.size();
      // asymmetric + floor: source=floor(destination/scale). For sizes,
      // use exact integer division so integer pixel boundaries stay exact.
      Value source =
          scale.empty()
              ? s.cvt(s.div(s.mul(s.cvt(c[axis], s.b.getI64Type()),
                                  s.cvt(s.i(shape[axis]), s.b.getI64Type())),
                            s.cvt(s.i(os[axis]), s.b.getI64Type())),
                      s.b.getI32Type())
              : s.cvt(
                    s.div(s.cvt(c[axis], s.b.getF32Type()), s.f(scale[index])),
                    s.b.getI32Type());
      c[axis] = s.select(s.cmp("lt", source, s.i(shape[axis])), source,
                         s.i(shape[axis] - 1));
    }
    return read(0, s.linear(c, shape));
  }
  if (kind == "gemm") {
    auto as = tensor(op->getOperand(0)).getShape(),
         bs = tensor(op->getOperand(1)).getShape();
    auto c = s.coords(tid, os);
    bool ta = integer(op, "transA", 0), tb = integer(op, "transB", 0);
    Value v = s.loop(as[ta ? 0 : 1], s.f(0), [&](Value k, Value a) {
      SmallVector<Value> ac =
          ta ? SmallVector<Value>{k, c[0]} : SmallVector<Value>{c[0], k};
      SmallVector<Value> bc =
          tb ? SmallVector<Value>{c[1], k} : SmallVector<Value>{k, c[1]};
      return s.add(a,
                   s.mul(read(0, s.linear(ac, as)), read(1, s.linear(bc, bs))));
    });
    v = s.mul(v, s.f(floating(op, "alpha", 1)));
    if (ptrs.size() > 2)
      v = s.add(
          v, s.mul(s.f(floating(op, "beta", 1)),
                   read(2, s.broadcast(tid, os,
                                       tensor(op->getOperand(2)).getShape()))));
    return v;
  }
  if (kind == "batch_norm") {
    auto shape = tensor(op->getOperand(0)).getShape();
    Value channel = s.coords(tid, shape)[1];
    Value variance =
        s.add(read(4, channel), s.f(floating(op, "epsilon", 1e-5)));
    return s.add(s.mul(s.div(s.sub(read(0, tid), read(3, channel)),
                             s.b.create<math::SqrtOp>(s.loc, variance)),
                       read(1, channel)),
                 read(2, channel));
  }
  if (kind == "conv2d" || kind == "conv2d_relu" || kind == "max_pool2d") {
    bool conv = kind != "max_pool2d";
    auto xs = tensor(op->getOperand(0)).getShape();
    auto ws = conv ? tensor(op->getOperand(1)).getShape() : ArrayRef<int64_t>();
    auto kernel =
        conv ? SmallVector<int64_t>{ws[2], ws[3]} : ints(op, "kernel_shape");
    auto strides = ints(op, "strides", {1, 1}),
         dilation = ints(op, "dilations", {1, 1}),
         pads = ints(op, "pads", {0, 0, 0, 0});
    auto c = s.coords(tid, os);
    int64_t groups = integer(op, "group", 1);
    int64_t cin = conv ? ws[1] : 1, count = cin * kernel[0] * kernel[1];
    Value accum =
        s.loop(count, s.f(conv ? 0 : -INFINITY), [&](Value k, Value a) {
          auto rc = s.coords(k, {cin, kernel[0], kernel[1]});
          Value channel =
              conv ? s.add(s.mul(s.div(c[1], s.i(os[1] / groups)), s.i(cin)),
                           rc[0])
                   : c[1];
          Value y = s.sub(s.add(s.mul(c[2], s.i(strides[0])),
                                s.mul(rc[1], s.i(dilation[0]))),
                          s.i(pads[0]));
          Value x = s.sub(s.add(s.mul(c[3], s.i(strides[1])),
                                s.mul(rc[2], s.i(dilation[1]))),
                          s.i(pads[1]));
          Value valid = s.binary("and",
                                 s.binary("and", s.cmp("ge", y, s.i(0)),
                                          s.cmp("lt", y, s.i(xs[2]))),
                                 s.binary("and", s.cmp("ge", x, s.i(0)),
                                          s.cmp("lt", x, s.i(xs[3]))));
          // Bounds guard is control flow, not a select after an invalid load.
          return s.branch(
              valid, s.b.getF32Type(),
              [&] {
                Value value = read(0, s.linear({c[0], channel, y, x}, xs));
                if (conv)
                  return s.add(
                      a,
                      s.mul(value, read(1, s.linear({c[1], rc[0], rc[1], rc[2]},
                                                    ws))));
                return s.maximum(a, value);
              },
              [&] { return a; });
        });
    if (conv && ptrs.size() > 2)
      accum = s.add(accum, read(2, c[1]));
    return kind == "conv2d_relu" ? s.relu(accum) : accum;
  }
  if (kind == "slice") {
    auto shape = tensor(op->getOperand(0)).getShape();
    auto c = s.coords(tid, os);
    auto starts = constantInts(op->getOperand(1)),
         ends = constantInts(op->getOperand(2));
    auto axes = ptrs.size() > 3 ? constantInts(op->getOperand(3))
                                : SmallVector<int64_t>();
    auto steps = ptrs.size() > 4 ? constantInts(op->getOperand(4))
                                 : SmallVector<int64_t>();
    if (axes.empty())
      for (size_t d = 0; d < starts.size(); ++d)
        axes.push_back(d);
    if (steps.empty())
      steps.assign(starts.size(), 1);
    for (size_t d = 0; d < starts.size(); ++d) {
      int64_t axis = axes[d];
      if (axis < 0)
        axis += shape.size();
      int64_t n = shape[axis], start = starts[d];
      if (start < 0)
        start += n;
      start =
          std::max<int64_t>(steps[d] > 0 ? 0 : -1,
                            std::min<int64_t>(start, steps[d] > 0 ? n : n - 1));
      c[axis] = s.add(s.i(start), s.mul(c[axis], s.i(steps[d])));
    }
    return read(0, s.linear(c, shape));
  }
  if (kind == "grid_sample") {
    auto shape = tensor(op->getOperand(0)).getShape(),
         gs = tensor(op->getOperand(1)).getShape();
    auto c = s.coords(tid, os);
    bool align = integer(op, "align_corners", 0);
    Value gx = read(1, s.linear({c[0], c[2], c[3], s.i(0)}, gs));
    Value gy = read(1, s.linear({c[0], c[2], c[3], s.i(1)}, gs));
    // ONNX: a NaN grid coordinate is interpreted as -1.
    gx = s.select(s.cmp("ne", gx, gx), s.f(-1), gx);
    gy = s.select(s.cmp("ne", gy, gy), s.f(-1), gy);
    auto unnormalize = [&](Value v, int64_t size) {
      return align ? s.mul(s.add(v, s.f(1)), s.f((size - 1) * 0.5))
                   : s.mul(s.sub(s.mul(s.add(v, s.f(1)), s.f(size)), s.f(1)),
                           s.f(0.5));
    };
    Value x = unnormalize(gx, shape[3]), y = unnormalize(gy, shape[2]);
    return s.bilinear(x, y, shape[2], shape[3], [&](Value xx, Value yy) {
      return read(0, s.linear({c[0], c[1], yy, xx}, shape));
    });
  }
  llvm_unreachable(
      "preflight must reject operations without a native lowering");
}

static LogicalResult preflight(Operation *op) {
  static const llvm::StringSet<> supported = {"matmul",
                                              "matmul_bias",
                                              "matmul_bias_relu",
                                              "add",
                                              "add_relu",
                                              "sub",
                                              "mul",
                                              "div",
                                              "and",
                                              "max",
                                              "equal",
                                              "greater",
                                              "less",
                                              "where",
                                              "relu",
                                              "exp",
                                              "log",
                                              "sin",
                                              "cos",
                                              "sqrt",
                                              "pow",
                                              "sigmoid",
                                              "clamp",
                                              "identity",
                                              "reshape",
                                              "squeeze",
                                              "unsqueeze",
                                              "cast",
                                              "expand",
                                              "transpose",
                                              "concat",
                                              "softmax",
                                              "layer_norm",
                                              "reduce_sum",
                                              "reduce_mean",
                                              "gather",
                                              "gemm",
                                              "batch_norm",
                                              "conv2d",
                                              "conv2d_relu",
                                              "max_pool2d",
                                              "slice",
                                              "grid_sample",
                                              "tile",
                                              "constant_of_shape",
                                              "shape",
                                              "scatter_nd",
                                              "deform_conv2d",
                                              "ms_deform_attn",
                                              "ms_deform_attn_offsets",
                                              "resize"};
  StringRef k = opKind(op);
  if (!supported.contains(k))
    return op->emitError("native Vortex lowering is not implemented for ")
           << op->getName();
  auto positions = ints(op, "gac.input_positions");
  for (size_t p = 0; p < positions.size(); ++p)
    if (positions[p] != int64_t(p) && k != "resize")
      return op->emitError(
          "native lowering does not yet support omitted middle operands");
  auto fail = [&](StringRef reason) { return op->emitError(reason); };
  if (k == "deform_conv2d" || k == "ms_deform_attn" ||
      k == "ms_deform_attn_offsets") {
    if (!tensor(op->getOperand(0)).getElementType().isF32())
      return fail("DCNv2/MSDeformAttn device templates currently require FP32");
    if ((k == "ms_deform_attn" || k == "ms_deform_attn_offsets") &&
        !getConstantIntegers(op->getOperand(1)))
      return fail("MSDeformAttn spatial_shapes must be compile-time constants");
  }
  if (k == "deform_conv2d") {
    auto stride = ints(op, "stride", {1, 1});
    auto dilation = ints(op, "dilation", {1, 1});
    auto padding = ints(op, "padding", {0, 0});
    auto ws = tensor(op->getOperand(3)).getShape();
    auto os = tensor(op->getResult(0)).getShape();
    for (unsigned d = 0; d < 2; ++d) {
      if (stride[d] > INT32_MAX || dilation[d] > INT32_MAX ||
          padding[d] > INT32_MAX)
        return fail(
            "DCNv2 spatial parameters exceed signed RV32 coordinate range");
      if ((os[d + 2] - 1) * stride[d] + (ws[d + 2] - 1) * dilation[d] >
          INT32_MAX)
        return fail(
            "DCNv2 intermediate integer coordinates exceed signed RV32 range");
    }
  }
  if (k == "resize") {
    if (stringAttr(op, "mode", "nearest") != "nearest" ||
        stringAttr(op, "coordinate_transformation_mode", "half_pixel") !=
            "asymmetric" ||
        stringAttr(op, "nearest_mode", "round_prefer_floor") != "floor" ||
        integer(op, "antialias", 0) || integer(op, "exclude_outside", 0) ||
        stringAttr(op, "keep_aspect_ratio_policy", "stretch") != "stretch")
      return fail("native Resize currently requires nearest/asymmetric/floor, "
                  "no antialias, stretch sizes");
    auto sizes = operandAt(op, 3), scales = operandAt(op, 2);
    auto scaleOp = scales ? scales.getDefiningOp() : nullptr;
    auto dense = scaleOp ? scaleOp->getAttrOfType<DenseFPElementsAttr>("value")
                         : DenseFPElementsAttr();
    if (sizes && scales && dense && dense.getNumElements())
      return fail("Resize must specify sizes or scales, not both");
    if (sizes ? !getConstantIntegers(sizes) : !dense || !dense.getNumElements())
      return fail("Resize sizes/scales must be compile-time constants");
    if (Value roi = operandAt(op, 1))
      if (tensor(roi).getNumElements())
        return fail("native asymmetric Resize requires empty/omitted ROI");
  }
  static const llvm::StringSet<> booleanOps = {
      "identity", "reshape",   "squeeze", "unsqueeze", "transpose",
      "concat",   "expand",    "slice",   "gather",    "cast",
      "and",      "equal",     "where",   "tile",      "constant_of_shape",
      "shape",    "scatter_nd"};
  if (!booleanOps.contains(k))
    for (Value v : op->getOperands())
      if (tensor(v).getElementType().isInteger(1))
        return fail(
            "Boolean tensors are not legal numeric arithmetic operands");
  if (k == "reshape" || k == "expand" || k == "unsqueeze" || k == "tile") {
    if (op->getNumOperands() < 2 || !getConstantIntegers(op->getOperand(1)))
      return fail("shape/axes inputs must be compile-time integer constants; a "
                  "declared output shape alone is insufficient");
  }
  if (k == "constant_of_shape" && !getConstantIntegers(op->getOperand(0)))
    return fail("ConstantOfShape dimensions must be compile-time constants");
  if (k == "scatter_nd") {
    if (stringAttr(op, "reduction", "none") != "none")
      return fail(
          "ScatterND device template currently requires reduction=none");
    if (!getConstantIntegers(op->getOperand(1)))
      return fail("ScatterND indices must be compile-time constants for the "
                  "static slice map");
  }
  if (k == "squeeze" && op->getNumOperands() > 1 &&
      !getConstantIntegers(op->getOperand(1)))
    return fail("Squeeze axes must be compile-time integer constants");
  if (k == "slice" && op->getNumOperands() < 3)
    return fail("native Slice requires explicit starts/ends operands");
  if (k == "conv2d" || k == "conv2d_relu" || k == "max_pool2d") {
    if (stringAttr(op, "auto_pad", "NOTSET") != "NOTSET")
      return fail("native spatial lowering requires explicit padding "
                  "(auto_pad=NOTSET)");
    if (k == "max_pool2d" && integer(op, "ceil_mode", 0))
      return fail("native pooling ceil_mode is not implemented");
  }
  if (k == "batch_norm" && integer(op, "training_mode", 0))
    return fail("BatchNorm training outputs are not supported");
  if (k == "layer_norm" &&
      (!tensor(op->getOperand(0)).getElementType().isF32() ||
       integer(op, "stash_type", 1) != 1))
    return fail("native LayerNorm currently requires FP32 with stash_type=1 to "
                "preserve ONNX rounding semantics");
  if (k == "grid_sample" &&
      (stringAttr(op, "mode", "bilinear") != "bilinear" ||
       stringAttr(op, "padding_mode", "zeros") != "zeros"))
    return fail("native GridSample currently supports bilinear, zeros only");
  if (k == "slice")
    for (unsigned p = 1; p < op->getNumOperands(); ++p)
      if (!getConstantIntegers(op->getOperand(p)))
        return fail("Slice parameters must be compile-time integer constants");
  if (k == "gather") {
    auto indices = getConstantIntegers(op->getOperand(1));
    if (!indices)
      return fail(
          "native Gather currently requires constant, bounds-checked indices");
    auto shape = tensor(op->getOperand(0)).getShape();
    int64_t axis = integer(op, "axis", 0);
    if (axis < 0)
      axis += shape.size();
    for (int64_t i : *indices)
      if (i < -shape[axis] || i >= shape[axis])
        return fail("Gather index is outside its axis");
  }
  if (k == "reduce_sum" || k == "reduce_mean")
    if (op->getNumOperands() > 1 && !getConstantIntegers(op->getOperand(1)))
      return fail("reduction axes must be compile-time constants");
  // Reduction and nonlinear templates deliberately accumulate FP16/FP32 in
  // FP32. Integer versions need separate legality and overflow contracts.
  if (k == "softmax" || k == "layer_norm" || k == "conv2d" ||
      k == "conv2d_relu" || k == "max_pool2d" || k == "grid_sample" ||
      k == "gemm" || k == "batch_norm" || k == "exp" || k == "log" ||
      k == "sin" || k == "cos" || k == "sqrt" || k == "pow" || k == "sigmoid")
    if (!isa<FloatType>(tensor(op->getOperand(0)).getElementType()))
      return fail("this native template requires floating-point data");
  return success();
}

struct TensorAllocation {
  Value value;
  std::string name, storage, file;
  int64_t bytes = 0, offset = 0, last = 0;
};
static std::string dtype(Type t) {
  if (t.isF32())
    return "float32";
  if (t.isF16())
    return "float16";
  if (auto integerType = dyn_cast<IntegerType>(t))
    if (integerType.isUnsigned())
      return "";
  if (t.isInteger(1))
    return "bool";
  if (t.isInteger(32))
    return "int32";
  if (t.isInteger(64))
    return "int64";
  return "";
}
static LogicalResult tensorSize(Value v, int64_t &bytes) {
  auto t = dyn_cast<RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape() || dtype(t.getElementType()).empty())
    return failure();
  int64_t n = 1;
  for (int64_t d : t.getShape()) {
    if (d <= 0 || d > INT32_MAX || n > INT32_MAX / d)
      return failure();
    n *= d;
  }
  int64_t width =
      std::max<unsigned>(8, t.getElementType().getIntOrFloatBitWidth()) / 8;
  if (n > INT32_MAX / width)
    return failure();
  bytes = n * width;
  return success();
}
static std::string namedResult(Operation *op) {
  if (auto names = op->getAttrOfType<ArrayAttr>("gac.output_names"))
    if (!names.empty())
      return cast<StringAttr>(names[0]).str();
  return {};
}

// Convert a constant ScatterND index tensor to destination-slice ->
// update-slice addressing. Data/updates remain runtime inputs. The table is
// immutable and embedded in the generated device ELF; no host operator
// evaluates activations.
static FailureOr<SmallVector<int32_t>> scatterSliceMap(Operation *op) {
  auto ds = tensor(op->getOperand(0)).getShape();
  auto is = tensor(op->getOperand(1)).getShape();
  if (is.empty() || is.back() < 1 || is.back() > int64_t(ds.size())) {
    op->emitError("ScatterND index depth must be within data rank");
    return failure();
  }
  int64_t depth = is.back(), prefixes = product(ds.take_front(depth));
  constexpr int64_t maxTableEntries = 1 << 24;
  if (prefixes > maxTableEntries) {
    op->emitError("ScatterND static index table exceeds 64 MiB target budget");
    return failure();
  }
  auto indices = constantInts(op->getOperand(1));
  int64_t updates = product(is.drop_back());
  if (int64_t(indices.size()) != updates * depth) {
    op->emitError(
        "ScatterND constant index count disagrees with its tensor type");
    return failure();
  }
  SmallVector<int32_t> table(prefixes, -1);
  for (int64_t i = 0; i < updates; ++i) {
    int64_t flat = 0;
    for (int64_t d = 0; d < depth; ++d) {
      int64_t index = indices[i * depth + d];
      if (index < 0)
        index += ds[d];
      if (index < 0 || index >= ds[d]) {
        op->emitError("ScatterND index is outside its data dimension");
        return failure();
      }
      flat = flat * ds[d] + index;
    }
    if (table[flat] != -1) {
      op->emitError(
          "duplicate ScatterND indices with reduction=none are ambiguous");
      return failure();
    }
    table[flat] = i;
  }
  return table;
}

static std::string sourceNode(Operation *op) {
  if (auto name = op->getAttrOfType<StringAttr>("gac.source_name"))
    return name.str();
  if (auto name = dyn_cast<NameLoc>(op->getLoc()))
    return name.getName().str();
  return {};
}
static LogicalResult writeConstant(TensorAllocation &a, StringRef dir) {
  if (dir.empty())
    return success();
  if (auto error = llvm::sys::fs::create_directories(dir))
    return a.value.getDefiningOp()->emitError(error.message());
  llvm::SmallString<256> path(dir);
  llvm::sys::path::append(path, a.file);
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
  if (error)
    return a.value.getDefiningOp()->emitError(error.message());
  auto dense =
      a.value.getDefiningOp()->getAttrOfType<DenseElementsAttr>("value");
  unsigned width =
      std::max<unsigned>(
          8, tensor(a.value).getElementType().getIntOrFloatBitWidth()) /
      8;
  auto write = [&](APInt bits) {
    uint64_t n = bits.getZExtValue();
    for (unsigned i = 0; i < width; ++i)
      stream << char((n >> (8 * i)) & 255);
  };
  if (isa<FloatType>(dense.getElementType()))
    for (APFloat v : dense.getValues<APFloat>())
      write(v.bitcastToAPInt());
  else
    for (APInt v : dense.getValues<APInt>())
      write(v);
  stream.flush();
  if (stream.has_error())
    return a.value.getDefiningOp()->emitError("writing constant failed");
  return success();
}

} // namespace

LogicalResult lowerToDevice(ModuleOp module, const DeviceOptions &options,
                            llvm::json::Object &manifest) {
  const bool native = options.target == DeviceTarget::Native;
  const bool cuda = options.target == DeviceTarget::CUDA;
  const bool vortex = options.target == DeviceTarget::Vortex;
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return module.emitError(
        "native device lowering requires exactly one graph function");
  auto func = functions.front();
  if (!llvm::hasSingleElement(func.getBody()))
    return func.emitError("graph function must have one block");
  auto ret = dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
  if (!ret)
    return func.emitError("graph must terminate with func.return");
  SmallVector<Operation *> graphNodes, constants;
  for (Operation &op : func.getBody().front()) {
    if (&op == ret.getOperation())
      continue;
    if (op.getName().getStringRef() == "gac.constant")
      constants.push_back(&op);
    else {
      if (op.getName().getDialectNamespace() != "gac" ||
          op.getNumResults() != 1)
        return op.emitError(
            "expected a supported, single-output GAC tensor operation");
      graphNodes.push_back(&op);
    }
  }
  SmallVector<TensorAllocation> allocations;
  llvm::DenseMap<Value, unsigned> ids;
  llvm::StringSet<> usedNames, reservedNames;
  // Debug names may disappear during CSE. Reserve every remaining explicit
  // name before generating internal IDs, especially the public input/output
  // ABI: an anonymous temporary must never rename a later public output.
  auto reserve = [&](ArrayAttr names) {
    if (names)
      for (Attribute name : names)
        reservedNames.insert(cast<StringAttr>(name).getValue());
  };
  reserve(func->getAttrOfType<ArrayAttr>("gac.input_names"));
  reserve(func->getAttrOfType<ArrayAttr>("gac.output_names"));
  for (Operation &op : func.getBody().front())
    reserve(op.getAttrOfType<ArrayAttr>("gac.output_names"));
  auto validateTensor = [&](Value v, StringRef name,
                            int64_t &bytes) -> LogicalResult {
    if (failed(tensorSize(v, bytes)))
      return emitError(v.getLoc(), "device requires positive static "
                                   "FP16/FP32/int32/int64/bool tensors "
                                   "within the scalar template's signed "
                                   "32-bit per-tensor indexing/byte limit: ")
             << name << " : " << v.getType();
    return success();
  };
  auto add = [&](Value v, std::string name,
                 StringRef storage) -> LogicalResult {
    int64_t bytes;
    if (failed(validateTensor(v, name, bytes)))
      return failure();
    if (name.empty()) {
      std::string base = "value_" + std::to_string(allocations.size());
      name = base;
      unsigned suffix = 0;
      while (reservedNames.contains(name) || usedNames.contains(name))
        name = base + "_" + std::to_string(++suffix);
    }
    if (!usedNames.insert(name).second)
      return func.emitError("duplicate tensor name in device plan: ") << name;
    ids[v] = allocations.size();
    allocations.push_back({v, name, storage.str(), "", bytes, 0, 0});
    return success();
  };
  auto inputNames = func->getAttrOfType<ArrayAttr>("gac.input_names");
  for (auto [i, arg] : llvm::enumerate(func.getArguments()))
    if (failed(add(arg,
                   inputNames && i < inputNames.size()
                       ? cast<StringAttr>(inputNames[i]).str()
                       : "input_" + std::to_string(i),
                   "input")))
      return failure();
  for (Operation *op : constants)
    if (failed(add(op->getResult(0), namedResult(op), "constant")))
      return failure();
  // Logical tensor views retain their SSA types for indexing and verification.
  // Only the device buffer map is shared: a consumer of reshape(X) receives
  // X's pointer while its kernel still uses the reshape's logical dimensions.
  // There is no runtime alias object and no overlapping allocation to exempt
  // from lifetime checks. Every use/producer dependency below uses this same
  // physical ID, including nested views and branches with later consumers.
  SmallVector<Operation *> nodes;
  int64_t elidedViewCount = 0, elidedViewCopyBytes = 0;
  for (Operation *op : graphNodes) {
    for (Value input : op->getOperands())
      if (!ids.count(input))
        return op->emitError("operand is not part of the graph SSA plan");
    Value output = op->getResult(0);
    int64_t bytes;
    // Elision must not bypass static-size, dtype, optional-input or constant
    // shape checks. In particular, a declared output shape is not sufficient
    // proof that a runtime Reshape parameter has that value.
    if (failed(validateTensor(output, namedResult(op), bytes)) ||
        failed(preflight(op)))
      return failure();
    StringRef kind = opKind(op);
    bool isView = kind == "identity" || kind == "reshape" ||
                  kind == "squeeze" || kind == "unsqueeze";
    bool publicOutput = llvm::is_contained(ret.getOperands(), output);
    if (isView && !publicOutput && op->getNumOperands()) {
      Value source = op->getOperand(0);
      auto inputType = tensor(source), outputType = tensor(output);
      // The existing device ABI is dense, contiguous row-major storage.
      // Encoded layouts, transposes and dtype-changing casts are deliberately
      // excluded; equal element counts alone do not prove those are views.
      if (!inputType.getEncoding() && !outputType.getEncoding() &&
          inputType.getElementType() == outputType.getElementType() &&
          inputType.getNumElements() == outputType.getNumElements()) {
        ids[output] = ids[source];
        ++elidedViewCount;
        elidedViewCopyBytes += bytes;
        continue;
      }
    }
    if (failed(add(output, namedResult(op), "arena")))
      return failure();
    nodes.push_back(op);
  }
  llvm::DenseMap<Operation *, SmallVector<int32_t>> scatterTables;
  int64_t embeddedBytes = 0;
  for (Operation *op : nodes)
    if (opKind(op) == "scatter_nd") {
      auto table = scatterSliceMap(op);
      if (failed(table))
        return failure();
      embeddedBytes += table->size() * sizeof(int32_t);
      scatterTables[op] = std::move(*table);
    }
  for (auto [idx, op] : llvm::enumerate(nodes))
    for (Value v : op->getOperands()) {
      if (!ids.count(v))
        return op->emitError("operand is not part of the graph SSA plan");
      allocations[ids[v]].last =
          std::max<int64_t>(allocations[ids[v]].last, idx);
    }
  for (Value v : ret.getOperands())
    allocations[ids[v]].last = nodes.size();
  // Inclusive last-use: an output cannot reuse an input's bytes during the
  // same dispatch. Reclaim, coalesce and split dead intervals before choosing
  // the smallest sufficient interval. A large early backbone activation must
  // not permanently reserve a slot subsequently occupied by a tiny tensor.
  struct Slot {
    int64_t offset, bytes, last;
  };
  SmallVector<Slot> active, free;
  int64_t arena = 0, peakLive = 0, peakDispatch = 0;
  for (auto [idx, op] : llvm::enumerate(nodes)) {
    for (auto it = active.begin(); it != active.end();) {
      if (it->last < int64_t(idx)) {
        free.push_back(*it);
        it = active.erase(it);
      } else
        ++it;
    }
    llvm::sort(
        free, [](const Slot &a, const Slot &b) { return a.offset < b.offset; });
    for (size_t i = 1; i < free.size();) {
      if (free[i - 1].offset + free[i - 1].bytes == free[i].offset) {
        free[i - 1].bytes += free[i].bytes;
        free.erase(free.begin() + i);
      } else
        ++i;
    }
    auto &a = allocations[ids[op->getResult(0)]];
    a.last = std::max<int64_t>(a.last, idx);
    int64_t aligned = (a.bytes + 63) & ~int64_t(63);
    auto found = free.end();
    for (auto it = free.begin(); it != free.end(); ++it)
      if (it->bytes >= aligned &&
          (found == free.end() || it->bytes < found->bytes))
        found = it;
    if (found != free.end()) {
      a.offset = found->offset;
      found->offset += aligned;
      found->bytes -= aligned;
      if (!found->bytes)
        free.erase(found);
    } else if (!free.empty() &&
               free.back().offset + free.back().bytes == arena) {
      // Extend the free tail rather than leaving it as a hole and appending
      // an entire allocation. No active tensor can occupy this interval.
      a.offset = free.back().offset;
      free.pop_back();
      arena = a.offset + aligned;
    } else {
      a.offset = arena;
      arena += aligned;
    }
    active.push_back({a.offset, aligned, a.last});
    int64_t live = 0;
    for (const auto &slot : active)
      live += slot.bytes;
    if (live > peakLive) {
      peakLive = live;
      peakDispatch = idx;
    }
  }
  int64_t resident = arena + embeddedBytes;
  for (const auto &a : allocations)
    if (a.storage != "arena")
      resident += a.bytes;
  // Vortex's pointer ABI is RV32. CUDA uses independent 64-bit allocations;
  // its aggregate address space must not inherit this Vortex-only budget.
  // Individual tensor sizes remain bounded by the checked scalar template.
  if (!cuda && resident > INT32_MAX) {
    auto diagnostic = emitError(
        func.getLoc(), "aggregate buffers exceed signed RV32 budget: ");
    diagnostic << resident << " bytes (arena " << arena
               << ", embedded constants " << embeddedBytes
               << ", peak simultaneously live arena " << peakLive
               << " at dispatch " << peakDispatch
               << "); tiling/fusion or memory scheduling is required";
    SmallVector<std::pair<Operation *, const TensorAllocation *>> live;
    for (auto [idx, op] : llvm::enumerate(nodes)) {
      const auto &allocation = allocations[ids[op->getResult(0)]];
      if (int64_t(idx) <= peakDispatch && allocation.last >= peakDispatch)
        live.push_back({op, &allocation});
    }
    llvm::sort(live, [](const auto &left, const auto &right) {
      return left.second->bytes > right.second->bytes;
    });
    for (size_t index = 0; index < std::min<size_t>(8, live.size()); ++index) {
      auto [op, allocation] = live[index];
      diagnostic << "\n  live " << allocation->bytes
                 << " bytes: " << op->getName() << " " << sourceNode(op)
                 << " : " << allocation->value.getType();
    }
    return failure();
  }
  for (auto [idx, a] : llvm::enumerate(allocations))
    if (a.storage == "constant") {
      a.file = "tensor_" + std::to_string(idx) + ".bin";
      if (failed(writeConstant(a, options.constantsDirectory)))
        return failure();
    }
  OpBuilder b(module.getContext());
  Location loc = module.getLoc();
  Type ptr = LLVM::LLVMPointerType::get(module.getContext());
  llvm::json::Array dispatches;
  SmallVector<std::string> kernelNames;
  llvm::DenseMap<unsigned, int64_t> producer;
  for (auto [idx, op] : llvm::enumerate(nodes)) {
    std::string name = "gac_op_" + std::to_string(idx);
    kernelNames.push_back(name);
    b.setInsertionPointToEnd(module.getBody());
    std::string tableName;
    if (auto found = scatterTables.find(op); found != scatterTables.end()) {
      tableName = name + "_scatter_slices";
      auto values = ArrayRef<int32_t>(found->second);
      auto tensorType =
          RankedTensorType::get({int64_t(values.size())}, b.getI32Type());
      b.create<LLVM::GlobalOp>(
          op->getLoc(), LLVM::LLVMArrayType::get(b.getI32Type(), values.size()),
          true, LLVM::Linkage::Internal, tableName,
          DenseIntElementsAttr::get(tensorType, values));
    }
    SmallVector<Type> arguments(vortex ? 1 : op->getNumOperands() + 1,
                                ptr);
    auto kernel = b.create<LLVM::LLVMFuncOp>(
        op->getLoc(), name,
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(b.getContext()),
                                    arguments));
    if (vortex)
      configureVortexKernel(kernel);
    else if (cuda)
      configureCudaKernel(kernel);
    Block *entry = kernel.addEntryBlock(b);
    b.setInsertionPointToStart(entry);
    Scalar s{b, op->getLoc()};
    if (!tableName.empty())
      s.scatterLookup = b.create<LLVM::AddressOfOp>(s.loc, ptr, tableName);
    SmallVector<Value> pointers;
    if (!vortex)
      pointers.append(entry->args_begin(), entry->args_end());
    else
      pointers = unpackVortexKernelPointers(b, s.loc, entry->getArgument(0),
                                             op->getNumOperands() + 1);
    int64_t elements = tensor(op->getResult(0)).getNumElements();
    auto compute = [&](Value tid) {
      Value result =
          emitElement(s, op, ArrayRef<Value>(pointers).drop_back(), tid);
      s.write(result, pointers.back(), tid,
              tensor(op->getResult(0)).getElementType());
    };
    if (native) {
      Value begin = s.i(0), end = s.i(elements), step = s.i(1);
      b.create<scf::ForOp>(s.loc, begin, end, step, ValueRange{},
                           [&](OpBuilder &, Location, Value iv, ValueRange) {
                             compute(iv);
                             b.create<scf::YieldOp>(s.loc);
                           });
    } else {
      Value tid = cuda ? createCudaGlobalThreadId(b, s.loc)
                       : createVortexGlobalThreadId(b, s.loc);
      Value valid = s.binary("and", s.cmp("ge", tid, s.i(0)),
                             s.cmp("lt", tid, s.i(elements)));
      b.create<scf::IfOp>(s.loc, valid, [&](OpBuilder &, Location) {
        compute(tid);
        b.create<scf::YieldOp>(s.loc);
      });
    }
    b.create<LLVM::ReturnOp>(s.loc, ValueRange{});
    llvm::json::Array inputs, dependencies;
    SmallVector<int64_t> deps;
    for (Value v : op->getOperands()) {
      unsigned physicalId = ids[v];
      inputs.push_back(allocations[physicalId].name);
      if (producer.count(physicalId) &&
          !llvm::is_contained(deps, producer[physicalId]))
        deps.push_back(producer[physicalId]);
    }
    llvm::sort(deps);
    for (auto d : deps)
      dependencies.push_back(d);
    llvm::json::Object dispatch{{"kernel", name},
                           {"op", op->getName().getStringRef().str()},
                           {"source_node", sourceNode(op)},
                           {"inputs", std::move(inputs)},
                           {"output", allocations[ids[op->getResult(0)]].name},
                           {"elements", elements},
                           {"depends_on", std::move(dependencies)},
                           {"wait_for_dispatch", int64_t(idx) - 1}};
    if (cuda) {
      // This baseline chooses a fixed one-dimensional launch. This is not
      // a tuned GEMM tile or a claim of shared-memory/tensor-core execution.
      constexpr int64_t blockSize = 256;
      dispatch["block_size"] = blockSize;
      dispatch["grid_size"] = (elements + blockSize - 1) / blockSize;
      dispatch["shared_memory_bytes"] = 0;
    }
    dispatches.push_back(std::move(dispatch));
    producer[ids[op->getResult(0)]] = idx;
  }
  llvm::json::Array tensors, inputs, outputs;
  for (const auto &a : allocations) {
    llvm::json::Array shape;
    for (int64_t n : tensor(a.value).getShape())
      shape.push_back(n);
    llvm::json::Object row{{"name", a.name},
                           {"dtype", dtype(tensor(a.value).getElementType())},
                           {"shape", std::move(shape)},
                           {"bytes", a.bytes},
                           {"storage", a.storage},
                           {"offset", a.offset}};
    if (!a.file.empty())
      row["file"] = a.file;
    tensors.push_back(std::move(row));
    if (a.storage == "input")
      inputs.push_back(a.name);
  }
  for (Value v : ret.getOperands())
    outputs.push_back(allocations[ids[v]].name);
  manifest = llvm::json::Object{
      {"schema_version", 1},
      {"compiler_backend", "native-mlir"},
      {"target", native ? "native" : cuda ? "cuda" : "vortex"},
      {"inputs", std::move(inputs)},
      {"outputs", std::move(outputs)},
      {"tensors", std::move(tensors)},
      {"dispatches", std::move(dispatches)},
      {"arena_bytes", arena},
      {"embedded_constant_bytes", embeddedBytes},
      {"schedule", "serialized dispatches; SSA last-use buffer reuse"},
      {"view_elision",
       llvm::json::Object{
           {"dispatches_removed", elidedViewCount},
           // Cumulative logical copy volume, not a claim about peak memory.
           {"logical_copy_bytes_removed", elidedViewCopyBytes},
           {"public_output_views_materialized", true}}},
      {"lmem_bytes", 0},
      {"double_buffer", false},
      {"template", "one output per SIMT thread; serial reductions"}};
  func.erase();
  if (vortex)
    configureVortexModule(module, kernelNames);
  else if (cuda) {
    configureCudaModule(module);
    manifest["cuda"] = llvm::json::Object{
        {"kernel_abi", "direct-pointers"},
        {"pointer_bits", 64},
        {"index_bits", 32},
        {"max_tensor_bytes", int64_t(INT32_MAX)},
        {"math_library", "libdevice"}};
  }
  return verify(module);
}

namespace {
struct LowerToDevice : PassWrapper<LowerToDevice, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToDevice)
  DeviceOptions options;
  llvm::json::Object *manifest = nullptr;
  LowerToDevice() = default;
  LowerToDevice(DeviceOptions options, llvm::json::Object *manifest)
      : options(std::move(options)), manifest(manifest) {}
  StringRef getArgument() const final { return "gac-lower-to-device"; }
  StringRef getDescription() const final {
    return "Lower static GAC graph operations to SIMT scalar kernels and "
           "the selected device launch ABI";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<LLVM::LLVMDialect, arith::ArithDialect, math::MathDialect,
             scf::SCFDialect>();
    registerDeviceDialects(r);
  }
  void runOnOperation() override {
    llvm::json::Object plan;
    if (failed(lowerToDevice(getOperation(), options, plan)))
      signalPassFailure();
    else if (manifest)
      *manifest = std::move(plan);
  }
};
} // namespace
std::unique_ptr<Pass> createLowerToDevicePass(DeviceOptions options,
                                               llvm::json::Object *manifest) {
  return std::make_unique<LowerToDevice>(std::move(options), manifest);
}
void registerDeviceDialects(DialectRegistry &registry) {
  registerVortexDialect(registry);
  registerCudaDialect(registry);
}
void registerDevicePasses() { PassRegistration<LowerToDevice>(); }
void addLLVMConversionPasses(PassManager &pm) {
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}
} // namespace gac
