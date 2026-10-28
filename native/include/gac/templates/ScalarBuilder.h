#ifndef GAC_TEMPLATES_SCALAR_BUILDER_H
#define GAC_TEMPLATES_SCALAR_BUILDER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include <functional>
#include <numeric>

namespace gac::templates::detail {
using namespace mlir;

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
    int64_t stride = std::accumulate(shape.begin(), shape.end(), int64_t(1),
                                     std::multiplies<int64_t>());
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

} // namespace gac::templates::detail
#endif
