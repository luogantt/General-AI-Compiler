#include "gac/DeviceLowering.h"
#include "gac/CudaLowering.h"
#include "gac/VortexLowering.h"
#include "gac/templates/Registry.h"
#include "gac/templates/ScalarBuilder.h"
#include "gac/Dialect.h"
#include "gac/LinalgLowering.h"
#include "gac/LinalgDevice.h"
#include "gac/TritonBridge.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

static bool isLinalgDeviceOperation(Operation *op) {
  return isa<linalg::LinalgOp, linalg::SoftmaxOp>(op);
}

// A destination tensor whose payload is never read (e.g. tensor.empty in a
// pointwise op) is a shape carrier, not a runtime input or an allocation.
static SmallVector<Value> deviceInputs(Operation *op) {
  if (isLinalgDeviceOperation(op))
    return getLinalgDeviceInputs(op);
  return SmallVector<Value>(op->getOperands());
}

static LogicalResult validatePad(tensor::PadOp pad) {
  if (!pad.getLow().empty() || !pad.getHigh().empty() ||
      !pad.getConstantPaddingValue() ||
      !pad.getConstantPaddingValue().getDefiningOp<arith::ConstantOp>())
    return pad.emitError("device tensor.pad requires static padding and a literal padding value");
  return success();
}

// ============================================================================
// Dedicated Lowering：不规则 gac.* 算子的专用发射
//
// 这里不经过 Linalg，而是直接用 ScalarBuilder 创建 typed SCF/Arith/Math/
// LLVM Operation。共同约定：
//   - 一个 dispatch 一个 Kernel，线程 tid 负责一个输出元素；
//   - 输入/输出都是已分配的全局 Buffer 指针；
//   - 归约和空间采样在寄存器内串行累加，不分配设备端中间张量；
//   - preflight() 是入口白名单；没有实现的算子不会被静默放过。
// ============================================================================
using templates::detail::Scalar;
using templates::detail::emitConvolutionOutputs;

static Value emitElement(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid);
static Value emitSpatial(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid);

// linalg.softmax 是带聚合语义的命名算子，而不是标量 payload 的 linalg.generic。
// 这里保留显式的 native 正确性路径：每个线程负责一个输出元素，归约轴串行遍历。
// Triton 路径只有在更严格的静态“最后一轴 row-wise”合同被证明后才替换整段计算。
static void emitLinalgSoftmaxElement(Scalar &s, linalg::SoftmaxOp softmax,
                                    ArrayRef<Value> ptrs, Value tid) {
  auto input = tensor(softmax.getInput());
  auto output = tensor(softmax->getResult(0));
  ArrayRef<int64_t> shape = input.getShape();
  int64_t axis = softmax.getDimension();
  int64_t width = shape[axis];
  int64_t stride = product(shape.drop_front(axis + 1));
  Value base = s.sub(
      tid, s.mul(s.rem(s.div(tid, s.i(stride)), s.i(width)), s.i(stride)));
  auto at = [&](Value k) {
    return s.read(ptrs.front(), s.add(base, s.mul(k, s.i(stride))),
                  input.getElementType());
  };
  Value maximum = s.loop(width, s.f(-INFINITY),
                         [&](Value k, Value value) {
                           return s.maximum(value, at(k));
                         });
  Value denominator = s.loop(width, s.f(0), [&](Value k, Value value) {
    return s.add(value,
                 s.b.create<math::ExpOp>(s.loc, s.sub(at(k), maximum)));
  });
  Value numerator = s.b.create<math::ExpOp>(
      s.loc, s.sub(s.read(ptrs.front(), tid, input.getElementType()), maximum));
  s.write(s.div(numerator, denominator), ptrs.back(), tid,
          output.getElementType());
}

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
    // 保留原 Add 的操作数顺序和 FP32 归约的舍入边界：这是 MatMul 循环之后
    // 一次独立的 fadd，而不是把 bias 提前塞进归约累加器。
    v = integer(op, "bias_first", 0) ? s.add(value, v) : s.add(v, value);
    if (opKind(op) == "matmul_bias_relu")
      v = s.relu(v);
  }
  return v;
}

/// Dedicated Lowering 的逐元素入口：tid 是输出元素的扁平下标。
/// 对每个输出元素读取广播后的输入、执行标量运算并返回结果；
/// 写回由调用方统一完成。大多数算子在这里结束，空间/采样算子转交 emitSpatial。
static Value emitElement(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid) {
  StringRef kind = opKind(op);
  auto out = tensor(op->getResult(0));
  auto os = out.getShape();
  // readAt：按扁平索引读取第 p 个输入；read：先按输出 shape 广播再读取。
  auto readAt = [&](unsigned p, Value idx) {
    return s.read(ptrs[p], idx, tensor(op->getOperand(p)).getElementType());
  };
  auto read = [&](unsigned p) {
    return readAt(p,
                  s.broadcast(tid, os, tensor(op->getOperand(p)).getShape()));
  };
  // view/cast：设备端只需逐元素拷贝或转换；非公开 view 通常已在规划期 elide。
  if (kind == "identity" || kind == "reshape" || kind == "squeeze" ||
      kind == "unsqueeze" || kind == "cast")
    return s.cvt(readAt(0, tid), isa<FloatType>(out.getElementType())
                                     ? Type(s.b.getF32Type())
                                     : out.getElementType());
  // 二元算术：两个输入都按输出 shape 广播；add_relu 在同一 region 内完成加法和 ReLU。
  if (kind == "add" || kind == "sub" || kind == "mul" || kind == "div" ||
      kind == "and" || kind == "add_relu") {
    Value v = s.binary(kind == "add_relu" ? "add" : kind, read(0), read(1));
    return kind == "add_relu" ? s.relu(v) : v;
  }
  // 比较类输出 i1；Where 按条件选择；Max 支持 N 个输入逐元素取最大。
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
  // 单输入激活/数学函数；sigmoid 用数值稳定形式，clamp 的 min/max 是可选输入。
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
    // 可选输入的位置在进入发射器之前已经由 preflight 检查过。
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
  // legacy 路径的标量 MatMul：每个线程算一个输出元素，K 维在寄存器内串行归约。
  if (kind == "matmul" || kind == "matmul_bias" || kind == "matmul_bias_relu")
    return emitMatmul(s, op, ptrs, tid);
  // Expand 只需广播读取；Tile 把输出坐标对输入维取模；Shape 返回静态维度。
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
  // ScatterND：preflight/合法化已证明索引为常量且不重复；
  // scatterLookup 是只读表，选择读 updates 还是原 data，避免并发写。
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
  // Transpose 通过 perm 重排坐标；Concat 按 axis 区间选择输入，边界用 branch 保护。
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
  // legacy 归约/归一化：每个线程负责一个输出，归约轴在寄存器内串行；
  // 默认 Linalg 路径不会走到这里，除非显式选择 legacy。
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
  // Gather：索引必须是常量且已做越界检查；负索引在这里归一化。
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
    // verifier 要求索引为常量并已检查范围，因此生成的 load 不需要
    // 依赖运行时的越界回退分支。
    return readAt(0, s.linear(dc, ds));
  }
  // 到这里仍未匹配的算子必须由 emitSpatial 处理；preflight 已保证不会出现
  // “无人实现”的算子在运行时才失败。
  return emitSpatial(s, op, ptrs, tid);
}

// 空间实现以及目标/运行时边界在下面。
/// 空间/采样算子的专用发射入口。
/// 这些算子通常无法用逐元素广播表达，需要显式循环：
///   - DCNv2/MSDeformAttn 按 level/point 流式采样；
///   - Resize/GridSample 做坐标变换 + 双线性插值；
///   - MaxPool/BatchNorm/Slice 在输出坐标上回溯输入。
/// 所有实现都只写最终输出，不物化大中间张量。
static Value emitSpatial(Scalar &s, Operation *op, ArrayRef<Value> ptrs,
                         Value tid) {
  StringRef kind = opKind(op);
  auto os = tensor(op->getResult(0)).getShape();
  auto read = [&](unsigned p, Value idx) {
    return s.read(ptrs[p], idx, tensor(op->getOperand(p)).getElementType());
  };
  // 卷积/DCNv2 与模板共用 emitConvolutionOutputs；这里每个线程只取一个输出通道。
  if (kind == "deform_conv2d" || kind == "conv2d" || kind == "conv2d_relu")
    return emitConvolutionOutputs(s, op, ptrs, s.coords(tid, os), 1).front();
  // MSDeformAttn：value[B,S,H,D] + locations[B,Q,H,L,P,2] + weights[B,Q,H,L,P]。
  // 按 level → point 双层循环流式采样，每个输出元素在寄存器中累加。
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
      // 独立的 FP32 fdiv/fadd 保留原图中物化位置的舍入点；
      // 两个逻辑 shape 只共享扁平索引，不共享逻辑维度。
      return integer(op, "reference_first", 0) ? s.add(base, result)
                                               : s.add(result, base);
    };
    Value head = s.div(c[2], s.i(vs[3])), channel = s.rem(c[2], s.i(vs[3]));
    Value sum = s.f(0);
    int64_t start = 0;
    // 静态 level 循环 + 采样点循环 + 每个输出一个寄存器累加器。
    // 这里流式处理 L*P 个样本，绝不分配 [B,H,D,Q,L,P] 中间张量。
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
  // Resize：目前只支持 nearest/asymmetric/floor；sizes 用精确整数除法，
  // scales 用浮点除法，越界坐标 clamp 到输入边界。
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
      // asymmetric + floor：source = floor(destination / scale)。
      // 对 sizes 使用精确整数除法，保证整数像素边界不产生浮点误差。
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
  // legacy Gemm：rank-2，K 维循环 + alpha/beta/bias；默认 Linalg 路径不会走到这里。
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
  // BatchNorm 推理：y = (x - mean) / sqrt(var + eps) * scale + bias。
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
  // MaxPool：在 kernel 窗口内取最大；越界用控制流跳过，避免非法 load。
  if (kind == "max_pool2d") {
    auto xs = tensor(op->getOperand(0)).getShape();
    auto kernel = ints(op, "kernel_shape");
    auto strides = ints(op, "strides", {1, 1}),
         dilation = ints(op, "dilations", {1, 1}),
         pads = ints(op, "pads", {0, 0, 0, 0});
    auto c = s.coords(tid, os);
    int64_t count = kernel[0] * kernel[1];
    Value accum =
        s.loop(count, s.f(-INFINITY), [&](Value k, Value a) {
          auto rc = s.coords(k, {1, kernel[0], kernel[1]});
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
          // 越界保护是控制流，而不是先做一次非法 load 再用 select 修正。
          return s.branch(
              valid, s.b.getF32Type(),
              [&] {
                Value value = read(0, s.linear({c[0], c[1], y, x}, xs));
                return s.maximum(a, value);
              },
              [&] { return a; });
        });
    return accum;
  }
  // Slice：starts/ends/axes/steps 都是常量；负索引和 clamp 按 ONNX 规则处理。
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
  // GridSample：grid 坐标归一化到 [-1,1]，NaN 按 ONNX 规则视为 -1，
  // 然后做双线性采样；越界由 bilinear 的边界处理覆盖。
  if (kind == "grid_sample") {
    auto shape = tensor(op->getOperand(0)).getShape(),
         gs = tensor(op->getOperand(1)).getShape();
    auto c = s.coords(tid, os);
    bool align = integer(op, "align_corners", 0);
    Value gx = read(1, s.linear({c[0], c[2], c[3], s.i(0)}, gs));
    Value gy = read(1, s.linear({c[0], c[2], c[3], s.i(1)}, gs));
    // ONNX 规定：grid 坐标为 NaN 时按 -1 处理。
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

/// Dedicated Lowering 的入口白名单与逐算子约束。
/// 在建立 dispatch 前拒绝不支持的形式；不会把未知算子留给运行时，
/// 也不会在失败时回退到 CPU/CUDA 或另一个实现。
static LogicalResult preflight(Operation *op) {
  // 当前可由专用发射器执行的 gac.* 算子白名单。新增算子必须同时扩展这里
  // 和 emitElement/emitSpatial，避免“能注册但不能执行”。
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
  // ONNX 可选输入可能留下空洞；除 Resize 外暂不支持中间操作数缺省。
  auto positions = ints(op, "gac.input_positions");
  for (size_t p = 0; p < positions.size(); ++p)
    if (positions[p] != int64_t(p) && k != "resize")
      return op->emitError(
          "native lowering does not yet support omitted middle operands");
  auto fail = [&](StringRef reason) { return op->emitError(reason); };
  // DCNv2/MSDeformAttn 的精度与静态形状契约。
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
  // Resize 只接受 nearest/asymmetric/floor，且 sizes/scales 必须编译期已知。
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
  // 布尔张量只允许出现在形状/索引类算子中，不能作为数值算术操作数。
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
  // shape/axes 必须是编译期整数常量；仅声明输出形状不足以证明运行时一致。
  if (k == "reshape" || k == "expand" || k == "unsqueeze" || k == "tile") {
    if (op->getNumOperands() < 2 || !getConstantIntegers(op->getOperand(1)))
      return fail("shape/axes inputs must be compile-time integer constants; a "
                  "declared output shape alone is insufficient");
  }
  if (k == "constant_of_shape" && !getConstantIntegers(op->getOperand(0)))
    return fail("ConstantOfShape dimensions must be compile-time constants");
  // ScatterND 设备端只支持 reduction=none，且索引必须可静态展开。
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
  // Conv/Pool 要求显式 padding；MaxPool 不支持 ceil_mode。
  if (k == "conv2d" || k == "conv2d_relu" || k == "max_pool2d") {
    if (stringAttr(op, "auto_pad", "NOTSET") != "NOTSET")
      return fail("native spatial lowering requires explicit padding "
                  "(auto_pad=NOTSET)");
    if (k == "max_pool2d" && integer(op, "ceil_mode", 0))
      return fail("native pooling ceil_mode is not implemented");
  }
  // 推理/精度模式限制：BatchNorm 不支持 training 输出；LayerNorm 固定
  // FP32 + stash_type=1；GridSample 只支持 bilinear + zeros。
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
  // Gather 索引必须常量且越界检查通过，生成的 load 才无需运行时保护。
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
  // 归约轴必须编译期已知，设备端才能把归约维展开成寄存器内循环。
  if (k == "reduce_sum" || k == "reduce_mean")
    if (op->getNumOperands() > 1 && !getConstantIntegers(op->getOperand(1)))
      return fail("reduction axes must be compile-time constants");
  // 归约/非线性模板刻意用 FP32 累加 FP16/FP32 输入；整数版本需要单独的
  // 合法性、溢出和舍入契约，不能直接复用这些模板。
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

LogicalResult validateDeviceGraph(ModuleOp module) {
  // Validate the original operators before decomposition, so moving a regular
  // operator to Linalg cannot accidentally discard ONNX/target legality checks.
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!llvm::hasSingleElement(function.getBody()))
      return function.emitError("device graph must have one block");
    for (Operation &op : function.getBody().front()) {
      if (op.getName().getDialectNamespace() != "gac" ||
          op.getName().getStringRef() == "gac.constant")
        continue;
      for (auto [position, value] : llvm::enumerate(op.getOperands())) {
        // Empty constant axes are valid ONNX metadata (reduce all axes, or a
        // no-op with noop_with_empty_axes). The Linalg rewrite consumes them;
        // they are not zero-byte runtime data buffers.
        if (position == 1 && (opKind(&op) == "reduce_sum" || opKind(&op) == "reduce_mean"))
          if (auto type = dyn_cast<RankedTensorType>(value.getType()))
            if (type.hasStaticShape() && type.getNumElements() == 0 &&
                getConstantIntegers(value))
              continue;
        int64_t bytes;
        if (failed(tensorSize(value, bytes)))
          return op.emitError("device requires positive static tensors within the signed 32-bit per-tensor byte limit");
      }
      for (Value value : op.getResults()) {
        int64_t bytes;
        if (failed(tensorSize(value, bytes)))
          return op.emitError("device requires positive static tensors within the signed 32-bit per-tensor byte limit");
      }
      if (failed(preflight(&op)))
        return failure();
    }
  }
  return success();
}

LogicalResult lowerToDevice(ModuleOp module, const DeviceOptions &options,
                            llvm::json::Object &manifest) {
  const bool native = options.target == DeviceTarget::Native;
  const bool cuda = options.target == DeviceTarget::CUDA;
  const bool vortex = options.target == DeviceTarget::Vortex;
  const bool tritonEnabled = !options.vortexTritonDirectory.empty();
  if (tritonEnabled) {
    auto powerOfTwo = [](int64_t n) { return n > 0 && !(n & (n - 1)); };
    if (!vortex || options.regularLowering != "linalg" ||
        !powerOfTwo(options.tritonBlock) || options.tritonBlock < 16 ||
        options.tritonBlock > 1024 || !powerOfTwo(options.tritonNumWarps) ||
        options.tritonNumWarps > 4 || !powerOfTwo(options.vortexLanes) ||
        options.vortexLanes > 32)
      return module.emitError("Triton requires Vortex/Linalg, power-of-two block "
                              "16..1024, warps 1..4 and lanes 1..32");
    if (auto error = llvm::sys::fs::create_directories(options.vortexTritonDirectory))
      return module.emitError("cannot create Triton artifact directory: ")
             << error.message();
  }
  if (options.regularLowering != "linalg" && options.regularLowering != "legacy")
    return module.emitError("regular lowering must be linalg or legacy");
  if (options.regularLowering == "linalg") {
    if (options.vortexMatmulSchedule != "scalar" ||
        options.cudaMatmulSchedule != "scalar" ||
        options.cudaReductionSchedule != "scalar" ||
        options.cudaConvChannelTile != 1)
      return module.emitError("device templates require explicit legacy lowering; they cannot bypass Linalg");
    if (failed(validateDeviceGraph(module)) || failed(lowerRegularToLinalg(module)))
      return failure();
  }
  if (options.cudaConvChannelTile != 1 && options.cudaConvChannelTile != 2 &&
      options.cudaConvChannelTile != 4 && options.cudaConvChannelTile != 8)
    return module.emitError("CUDA convolution channel tile must be 1, 2, 4, or 8");
  if (!cuda && options.cudaConvChannelTile != 1)
    return module.emitError("CUDA convolution channel tiling requires the CUDA target");
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
  // LayerNorm is a closed multi-operation Linalg graph.  Select every complete
  // instance before collecting graph roots, then hide all five internal
  // operations atomically.  A partial match remains entirely on the native
  // Vortex path; no intermediate buffer or dispatch is silently dropped.
  llvm::DenseMap<Operation *, TritonLayerNormPlan> tritonLayerNormPlans;
  llvm::DenseSet<Operation *> tritonLayerNormInternal;
  // Also remember the structural ownership of every LayerNorm decomposition,
  // even when the aggregate is ineligible (for example width > 1024).  Its
  // mean is an ordinary-looking pointwise Linalg op; allowing the generic
  // Triton selector to take only that node would create a mixed half-fusion
  // instead of the required complete native fallback.
  llvm::DenseSet<Operation *> tritonLayerNormOwned;
  if (tritonEnabled) {
    for (Operation &candidate : func.getBody().front()) {
      auto role = candidate.getAttrOfType<StringAttr>("gac.linalg_role");
      if (role && role.getValue() == "layer-normalization") {
        tritonLayerNormOwned.insert(&candidate);
        if (auto root = dyn_cast<linalg::GenericOp>(&candidate);
            root && root.getNumDpsInputs() >= 3) {
          auto mean = root.getDpsInputOperand(1)->get()
                          .getDefiningOp<linalg::GenericOp>();
          auto deviation = root.getDpsInputOperand(2)->get()
                               .getDefiningOp<linalg::GenericOp>();
          if (mean) {
            tritonLayerNormOwned.insert(mean);
            if (mean.getNumDpsInputs() >= 1) {
              if (auto total = mean.getDpsInputOperand(0)->get()
                                   .getDefiningOp<linalg::GenericOp>()) {
                tritonLayerNormOwned.insert(total);
                if (total.getNumDpsInits() >= 1)
                  if (Operation *init = total.getDpsInitOperand(0)->get()
                                            .getDefiningOp())
                    tritonLayerNormOwned.insert(init);
              }
            }
          }
          if (deviation) {
            tritonLayerNormOwned.insert(deviation);
            if (deviation.getNumDpsInits() >= 1)
              if (Operation *init = deviation.getDpsInitOperand(0)->get()
                                        .getDefiningOp())
                tritonLayerNormOwned.insert(init);
          }
        }
      }
      TritonLayerNormPlan plan;
      std::string reason;
      if (!selectTritonLayerNorm(&candidate, plan, reason))
        continue;
      if (tritonLayerNormInternal.contains(&candidate) ||
          tritonLayerNormPlans.count(&candidate))
        return candidate.emitError("overlapping Triton LayerNorm root");
      for (Operation *internal : plan.internalOperations) {
        if (tritonLayerNormPlans.count(internal) ||
            !tritonLayerNormInternal.insert(internal).second)
          return candidate.emitError(
              "overlapping Triton LayerNorm internal subgraph");
      }
      tritonLayerNormPlans.try_emplace(&candidate, std::move(plan));
    }
  }
  auto plannedDeviceInputs = [&](Operation *op) {
    if (auto found = tritonLayerNormPlans.find(op);
        found != tritonLayerNormPlans.end())
      return SmallVector<Value>{found->second.input, found->second.scale,
                                found->second.bias};
    return deviceInputs(op);
  };
  SmallVector<Operation *> graphNodes, constants;
  int64_t fusedLinalgProducers = 0;
  for (Operation &op : func.getBody().front()) {
    if (&op == ret.getOperation())
      continue;
    if (isa<tensor::EmptyOp>(op))
      continue;
    if (op.getName().getStringRef() == "gac.constant") {
      if (!op.use_empty())
        constants.push_back(&op);
    }
    else {
      if ((op.getName().getDialectNamespace() != "gac" &&
           !isLinalgDeviceOperation(&op) && !isa<tensor::PadOp>(op)) ||
          op.getNumResults() != 1)
        return op.emitError(
            "expected a supported single-output GAC, Linalg or tensor.pad operation");
      // Check every structured operation, including producers whose scalar
      // payload is emitted inside a consumer. Fusion must not hide invalid IR.
      if (isLinalgDeviceOperation(&op)) {
        if (failed(validateLinalgDeviceOp(&op)))
          return failure();
        if (tritonLayerNormInternal.contains(&op)) {
          ++fusedLinalgProducers;
          continue;
        }
        if (tritonLayerNormPlans.count(&op)) {
          graphNodes.push_back(&op);
          continue;
        }
        if (isFusedLinalgProducer(&op)) {
          ++fusedLinalgProducers;
          continue;
        }
      }
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
    for (Value input : plannedDeviceInputs(op))
      if (!ids.count(input))
        return op->emitError("operand is not part of the graph SSA plan");
    Value output = op->getResult(0);
    int64_t bytes;
    // Elision must not bypass static-size, dtype, optional-input or constant
    // shape checks. In particular, a declared output shape is not sufficient
    // proof that a runtime Reshape parameter has that value.
    if (failed(validateTensor(output, namedResult(op), bytes)))
      return failure();
    LogicalResult legal = isLinalgDeviceOperation(op)
                              ? validateLinalgDeviceOp(op)
                          : isa<tensor::PadOp>(op)
                              ? validatePad(cast<tensor::PadOp>(op))
                              : preflight(op);
    if (failed(legal))
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
    for (Value v : plannedDeviceInputs(op)) {
      if (!ids.count(v))
        return op->emitError("operand is not part of the graph SSA plan");
      allocations[ids[v]].last =
          std::max<int64_t>(allocations[ids[v]].last, idx);
    }
  for (Value v : ret.getOperands()) {
    if (!ids.count(v))
      return ret.emitError("graph output has no initialized device tensor; tensor.empty is only a write-only destination");
    allocations[ids[v]].last = nodes.size();
  }
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
  int64_t maxLmemBytes = 0, optimizedMatmuls = 0;
  int64_t cudaOptimizedMatmuls = 0, cudaOptimizedReductions = 0;
  int64_t cudaOptimizedConvolutions = 0;
  int64_t linalgDispatches = 0, tritonDispatches = 0;
  bool usesDoubleBuffer = false;
  SmallVector<std::string> kernelNames;
  llvm::DenseMap<unsigned, int64_t> producer;
  for (auto [idx, op] : llvm::enumerate(nodes)) {
    SmallVector<Value> graphInputs = plannedDeviceInputs(op);
    templates::Selection selection;
    if (failed(templates::select(op, options, selection)))
      return failure();
    std::string name = "gac_op_" + std::to_string(idx);
    TritonLayerNormPlan tritonLayerNormPlan;
    TritonSoftmaxPlan tritonSoftmaxPlan;
    TritonReductionPlan tritonReductionPlan;
    TritonDotPlan tritonDotPlan;
    std::string tritonLayerNormReason, tritonSoftmaxReason,
        tritonReductionReason, tritonDotReason, tritonPointwiseReason;
    bool layerNormMatches =
        tritonEnabled &&
        selectTritonLayerNorm(op, tritonLayerNormPlan,
                              tritonLayerNormReason);
    bool useTritonLayerNorm = tritonLayerNormPlans.count(op);
    if (layerNormMatches != useTritonLayerNorm)
      return op->emitError(
          "Triton LayerNorm selection changed during device planning");
    bool rejectedLayerNormMember =
        tritonLayerNormOwned.contains(op) && !useTritonLayerNorm;
    bool useTritonSoftmax =
        tritonEnabled && !useTritonLayerNorm && !rejectedLayerNormMember &&
        selectTritonSoftmax(op, tritonSoftmaxPlan, tritonSoftmaxReason);
    bool useTritonReduction =
        tritonEnabled && !useTritonLayerNorm && !rejectedLayerNormMember &&
        !useTritonSoftmax &&
        selectTritonReduction(op, tritonReductionPlan,
                              tritonReductionReason);
    bool useTritonDot =
        tritonEnabled && !useTritonLayerNorm && !rejectedLayerNormMember &&
        !useTritonSoftmax && !useTritonReduction &&
        selectTritonDot(op, tritonDotPlan, tritonDotReason);
    bool useTritonPointwise =
        tritonEnabled && !useTritonLayerNorm && !rejectedLayerNormMember &&
        !useTritonSoftmax && !useTritonReduction && !useTritonDot &&
        selectTritonPointwise(op, tritonPointwiseReason);
    bool useTriton = useTritonLayerNorm || useTritonSoftmax ||
                     useTritonReduction || useTritonDot ||
                     useTritonPointwise;
    int64_t elements = tensor(op->getResult(0)).getNumElements();
    if (useTriton) {
      llvm::SmallString<256> path(options.vortexTritonDirectory);
      llvm::sys::path::append(path, name + ".ttir");
      LogicalResult emitted = useTritonLayerNorm
                                  ? emitTritonLayerNorm(
                                        op, name, tritonLayerNormPlan, path)
                              : useTritonSoftmax
                                  ? emitTritonSoftmax(
                                        op, name, tritonSoftmaxPlan, path)
                              : useTritonReduction
                                  ? emitTritonReduction(
                                        op, name, tritonReductionPlan, path)
                              : useTritonDot
                                  ? emitTritonDot(op, name, tritonDotPlan, path)
                                  : emitTritonPointwise(
                                        op, name, options.tritonBlock, path);
      if (failed(emitted))
        return failure();
    } else {
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
      SmallVector<Type> arguments(vortex ? 1 : graphInputs.size() + 1,
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
                                               graphInputs.size() + 1);
      LogicalResult emitted = success();
      auto compute = [&](Value tid) {
        if (auto softmax = dyn_cast<linalg::SoftmaxOp>(op)) {
          emitLinalgSoftmaxElement(s, softmax, pointers, tid);
          return;
        }
        if (isa<linalg::LinalgOp>(op)) {
          emitted = emitLinalgDeviceElement(b, op, pointers, tid);
          return;
        }
        if (auto pad = dyn_cast<tensor::PadOp>(op)) {
          auto shape = tensor(pad.getSource()).getShape();
          auto coords = s.coords(tid, tensor(pad.getResult()).getShape());
          Value inside = b.create<arith::ConstantIntOp>(s.loc, 1, 1);
          for (auto [axis, extent] : llvm::enumerate(shape)) {
            coords[axis] = s.sub(coords[axis], s.i(pad.getStaticLow()[axis]));
            inside = s.binary("and", inside,
                s.binary("and", s.cmp("ge", coords[axis], s.i(0)),
                                   s.cmp("lt", coords[axis], s.i(extent))));
          }
          Type element = tensor(pad.getSource()).getElementType();
          Type computeType = isa<FloatType>(element) ? Type(b.getF32Type()) : element;
          Value result = s.branch(inside, computeType, [&] {
            return s.read(pointers[0], s.linear(coords, shape), element);
          }, [&] {
            auto constant = pad.getConstantPaddingValue().getDefiningOp<arith::ConstantOp>();
            return s.cvt(b.create<arith::ConstantOp>(s.loc, constant.getValue()), computeType);
          });
          s.write(result, pointers.back(), tid, element);
          return;
        }
        Value result =
            emitElement(s, op, ArrayRef<Value>(pointers).drop_back(), tid);
        s.write(result, pointers.back(), tid,
                tensor(op->getResult(0)).getElementType());
      };
      if (selection.hasTemplate()) {
        // Collective templates own their tail guards: every CTA lane must reach
        // each barrier. Keep the device ABI and add convergent only when needed.
        if (selection.descriptor->collective) {
          SmallVector<Attribute> attributes;
          if (vortex) attributes.push_back(b.getStringAttr("vortex-kernel"));
          attributes.push_back(b.getStringAttr("convergent"));
          kernel->setAttr("passthrough", b.getArrayAttr(attributes));
        }
        if (failed(templates::emit(b, op, pointers, selection)))
          return failure();
      } else if (native) {
        Value begin = s.i(0), end = s.i(elements), step = s.i(1);
        auto loop = b.create<scf::ForOp>(s.loc, begin, end, step);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(loop.getBody()->getTerminator());
        // Build the body after attaching the loop to its kernel: Linalg fusion
        // hoists fixed private buffers to that kernel's entry block.
        compute(loop.getInductionVar());
      } else {
        Value tid = cuda ? createCudaGlobalThreadId(b, s.loc)
                         : createVortexGlobalThreadId(b, s.loc);
        Value valid = s.binary("and", s.cmp("ge", tid, s.i(0)),
                               s.cmp("lt", tid, s.i(elements)));
        auto condition = b.create<scf::IfOp>(s.loc, valid, false);
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(condition.thenBlock()->getTerminator());
        compute(tid);
      }
      if (failed(emitted))
        return failure();
      b.create<LLVM::ReturnOp>(s.loc, ValueRange{});
    }
    llvm::json::Array inputs, dependencies;
    SmallVector<int64_t> deps;
    for (Value v : graphInputs) {
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
    // 记录该 dispatch 实际使用的 Lowering 路径。Linalg 载荷通过共享的
    // destination Buffer 映射和上游 Linalg-to-SCF 发射；tensor.pad 是 Conv
    // 合法化过程产生的内部算子，有自己的标量 Lowering；其余 gac.* 算子保留
    // 图语义，走本文件的专用发射器（emitElement/emitSpatial）。这些字符串是
    // manifest 契约的一部分，必须描述真正生成 Kernel 的路径，而不是计划中的优化。
    if (isLinalgDeviceOperation(op)) {
      ++linalgDispatches;
      // destination 是预先分配的全局 arena Buffer，设备端不会调用 malloc；
      // 只有单消费者融合的中间值放在线程私有标量槽位中，规范归约仍在单线程内串行完成。
      dispatch["lowering"] = "linalg-to-scf";
      dispatch["bufferization"] = "static global destinations; thread-local scalar fusion buffers; no device malloc";
      dispatch["schedule"] = "parallel output coordinates per thread; serial reduction loops";
    } else {
      // tensor.pad 没有对应的 gac.* 算子，它由 Conv 合法化创建；
      // 其余情况都是本文件处理的不规则 GAC 算子。
      dispatch["lowering"] = isa<tensor::PadOp>(op) ? "tensor-pad-to-scf" : "gac-specialized";
    }
    if (cuda) {
      // This baseline chooses a fixed one-dimensional launch. This is not
      // a tuned GEMM tile or a claim of shared-memory/tensor-core execution.
      constexpr int64_t blockSize = 256;
      dispatch["block_size"] = blockSize;
      dispatch["grid_size"] = (elements + blockSize - 1) / blockSize;
      dispatch["shared_memory_bytes"] = 0;
    }
    templates::annotate(dispatch, selection,
        isLinalgDeviceOperation(op) ? "linalg.output-thread"
        : isa<tensor::PadOp>(op) ? "tensor.pad.scalar" : "gac.scalar");
    if (useTriton) {
      ++tritonDispatches;
      dispatch["lowering"] = "linalg-to-triton";
      if (useTritonLayerNorm) {
        dispatch["compute_template"] = llvm::json::Object{
            {"id", "triton.vortex.layer-norm-fp32"},
            {"selected", true},
            {"reason", "validated closed ONNX LayerNormalization Linalg subgraph"}};
        dispatch["schedule"] =
            "one row per TTIR program; two masked axis-zero add reductions";
        dispatch["bufferization"] =
            "static X/Scale/Bias/Y buffers; mean and variance remain scalar SSA values";
        dispatch["triton"] = llvm::json::Object{
            {"kind", "layer_norm"},
            {"ttir_file", name + ".ttir"},
            {"rows", tritonLayerNormPlan.rows},
            {"width", tritonLayerNormPlan.width},
            {"n", tritonLayerNormPlan.width},
            {"tile", tritonLayerNormPlan.tile},
            {"epsilon_f32_bits", int64_t(tritonLayerNormPlan.epsilonF32Bits)},
            {"num_warps", options.tritonNumWarps},
            {"lanes", options.vortexLanes},
            {"block_size", options.tritonNumWarps * options.vortexLanes},
            {"grid_size", tritonLayerNormPlan.gridSize},
            {"abi", "vortex-u64-pointers-i32-n-f32-epsilon"}};
      } else if (useTritonSoftmax) {
        dispatch["compute_template"] = llvm::json::Object{
            {"id", "triton.vortex.softmax-fp32"},
            {"selected", true},
            {"reason", "validated static FP32 last-axis row-wise Softmax"}};
        dispatch["schedule"] =
            "one row per TTIR program; masked max and sum tt.reduce stages";
        dispatch["bufferization"] =
            "separate static FP32 input/output buffers; scalar reductions";
        dispatch["triton"] = llvm::json::Object{
            {"kind", "softmax"},
            {"ttir_file", name + ".ttir"},
            {"rows", tritonSoftmaxPlan.rows},
            {"width", tritonSoftmaxPlan.width},
            {"n", tritonSoftmaxPlan.width},
            {"tile", tritonSoftmaxPlan.tile},
            {"num_warps", options.tritonNumWarps},
            {"lanes", options.vortexLanes},
            {"block_size", options.tritonNumWarps * options.vortexLanes},
            {"grid_size", tritonSoftmaxPlan.gridSize},
            {"abi", "vortex-u64-pointers-i32-n"}};
      } else if (useTritonReduction) {
        dispatch["compute_template"] = llvm::json::Object{
            {"id", "triton.vortex.reduce-sum-fp32"},
            {"selected", true},
            {"reason", "validated static rank-one FP32 full-axis ReduceSum"}};
        dispatch["schedule"] =
            "one masked power-of-two TTIR tile; one axis-zero tt.reduce";
        dispatch["bufferization"] =
            "shared static global buffers; scalar reduction accumulator";
        dispatch["triton"] = llvm::json::Object{
            {"kind", "reduction"},
            {"ttir_file", name + ".ttir"},
            {"width", tritonReductionPlan.width},
            {"tile", tritonReductionPlan.tile},
            {"num_warps", options.tritonNumWarps},
            {"lanes", options.vortexLanes},
            {"block_size", options.tritonNumWarps * options.vortexLanes},
            {"grid_size", tritonReductionPlan.gridSize},
            {"abi", "vortex-u64-pointers"}};
      } else if (useTritonDot) {
        llvm::json::Array aShape, bShape, cShape;
        for (int64_t extent : tritonDotPlan.aShape)
          aShape.push_back(extent);
        for (int64_t extent : tritonDotPlan.bShape)
          bShape.push_back(extent);
        for (int64_t extent : tritonDotPlan.cShape)
          cShape.push_back(extent);
        dispatch["compute_template"] = llvm::json::Object{
            {"id", "triton.vortex.matmul-fp32"},
            {"selected", true},
            {"reason", "validated static FP32 rank-two or batched MatMul"}};
        dispatch["schedule"] =
            "flattened batch/tile program id; one 16x16 output tile and one masked power-of-two-K IEEE tt.dot";
        dispatch["bufferization"] =
            "shared static global buffers; dot accumulator in registers";
        dispatch["triton"] = llvm::json::Object{
            {"kind", "matmul"},
            {"ttir_file", name + ".ttir"},
            {"m", tritonDotPlan.m},
            {"n", tritonDotPlan.n},
            {"k", tritonDotPlan.k},
            {"batches", tritonDotPlan.batches},
            {"batch_mode", tritonDotPlan.batchMode},
            {"a_shape", std::move(aShape)},
            {"b_shape", std::move(bShape)},
            {"c_shape", std::move(cShape)},
            {"block_m", tritonDotPlan.blockM},
            {"block_n", tritonDotPlan.blockN},
            {"block_k", tritonDotPlan.blockK},
            {"num_warps", options.tritonNumWarps},
            {"lanes", options.vortexLanes},
            {"block_size", options.tritonNumWarps * options.vortexLanes},
            {"grid_size", tritonDotPlan.gridSize},
            {"abi", "vortex-u64-pointers"}};
      } else {
        dispatch["compute_template"] = llvm::json::Object{
            {"id", "triton.vortex.pointwise"},
            {"selected", true},
            {"reason", "validated same-shape pointwise Linalg subgraph"}};
        dispatch["schedule"] =
            "flat masked TTIR blocks; lane-strided output elements";
        dispatch["bufferization"] =
            "shared static global arena; fused intermediates in registers";
        dispatch["triton"] = llvm::json::Object{
            {"kind", "pointwise"},
            {"ttir_file", name + ".ttir"},
            {"block_elements", options.tritonBlock},
            {"num_warps", options.tritonNumWarps},
            {"lanes", options.vortexLanes},
            {"block_size", options.tritonNumWarps * options.vortexLanes},
            {"grid_size",
             (elements + options.tritonBlock - 1) / options.tritonBlock},
            {"abi", "vortex-u64-pointers"}};
      }
    } else if (tritonEnabled) {
      dispatch["triton_exclusion"] =
          "layer_norm: " + tritonLayerNormReason + "; softmax: " +
          tritonSoftmaxReason + "; reduction: " + tritonReductionReason +
          "; dot: " + tritonDotReason + "; pointwise: " +
          tritonPointwiseReason;
    }
    if (std::holds_alternative<CudaMatmulPlan>(selection.plan))
      ++cudaOptimizedMatmuls;
    if (std::holds_alternative<CudaReductionPlan>(selection.plan))
      ++cudaOptimizedReductions;
    if (std::holds_alternative<templates::CudaConvolutionPlan>(selection.plan))
      ++cudaOptimizedConvolutions;
    if (auto *p = std::get_if<VortexMatmulPlan>(&selection.plan)) {
      maxLmemBytes = std::max(maxLmemBytes, p->lmemBytes);
      usesDoubleBuffer |= p->stages == 2;
      ++optimizedMatmuls;
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
      {"lmem_bytes", maxLmemBytes},
      {"double_buffer", usesDoubleBuffer},
      {"template", "one output per SIMT thread; serial reductions"}};
  manifest["compute_template_library"] = "GACComputeTemplates/v1";
  manifest["regular_lowering"] = options.regularLowering;
  manifest["linalg_dispatches"] = linalgDispatches;
  manifest["linalg_fused_producers"] = fusedLinalgProducers;
  if (tritonEnabled) {
    manifest["triton_dispatches"] = tritonDispatches;
    manifest["kernel_backends"] = "native-vortex + triton-vortex; single linked device module";
    manifest["template"] = "per-dispatch native kernels or flat masked Triton tiles";
  }
  if (vortex && options.vortexMatmulSchedule != "scalar") {
    manifest["vortex_matmul_schedule"] = options.vortexMatmulSchedule;
    manifest["optimized_matmul_dispatches"] = optimizedMatmuls;
    manifest["template"] = "per-dispatch Vortex MatMul templates; scalar for other operations";
  }
  if (cuda && options.cudaMatmulSchedule != "scalar") {
    manifest["cuda_matmul_schedule"] = options.cudaMatmulSchedule;
    manifest["optimized_matmul_dispatches"] = cudaOptimizedMatmuls;
    manifest["template"] = "per-dispatch CUDA shared-memory/register MatMul; scalar for other operations";
  }
  if (cuda && options.cudaReductionSchedule != "scalar") {
    manifest["cuda_reduction_schedule"] = options.cudaReductionSchedule;
    manifest["optimized_reduction_dispatches"] = cudaOptimizedReductions;
  }
  if (cuda && options.cudaConvChannelTile > 1) {
    manifest["cuda_conv_channel_tile"] = options.cudaConvChannelTile;
    manifest["optimized_convolution_dispatches"] = cudaOptimizedConvolutions;
    manifest["template"] = "per-dispatch CUDA compute templates; scalar for unsupported shapes";
  }
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
  registerLinalgDialects(registry);
  registry.insert<memref::MemRefDialect, affine::AffineDialect>();
  registerVortexDialect(registry);
  registerCudaDialect(registry);
}
void registerDevicePasses() { PassRegistration<LowerToDevice>(); }
void addLLVMConversionPasses(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertMathToLLVMPass());
  FinalizeMemRefToLLVMConversionPassOptions memrefOptions;
  memrefOptions.indexBitwidth = 32;
  pm.addPass(createFinalizeMemRefToLLVMConversionPass(memrefOptions));
  ArithToLLVMConversionPassOptions arithOptions;
  arithOptions.indexBitwidth = 32;
  pm.addPass(createArithToLLVMConversionPass(arithOptions));
  ConvertControlFlowToLLVMPassOptions cfOptions;
  cfOptions.indexBitwidth = 32;
  pm.addPass(createConvertControlFlowToLLVMPass(cfOptions));
  pm.addPass(createReconcileUnrealizedCastsPass());
}
} // namespace gac
