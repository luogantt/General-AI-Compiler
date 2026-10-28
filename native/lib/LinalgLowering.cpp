#include "gac/LinalgLowering.h"
#include "gac/Dialect.h"
#include "gac/LinalgConvolution.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <limits>
#include <numeric>

using namespace mlir;
namespace gac {
namespace {
using Iterator = utils::IteratorType;
using Payload = llvm::function_ref<Value(OpBuilder &, Location, ValueRange)>;

static int64_t integer(Operation *op, StringRef key, int64_t fallback) {
  auto attr = op->getAttrOfType<IntegerAttr>(key);
  return attr ? attr.getInt() : fallback;
}
static double floating(Operation *op, StringRef key, double fallback) {
  auto attr = op->getAttrOfType<FloatAttr>(key);
  return attr ? attr.getValueAsDouble() : fallback;
}
static RankedTensorType tensor(Value value) {
  return cast<RankedTensorType>(value.getType());
}
static int64_t product(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t(1),
                         std::multiplies<int64_t>());
}
static Type accumulationType(Type element, OpBuilder &b) {
  return isa<FloatType>(element) ? Type(b.getF32Type()) : element;
}

// Scalar payloads are real MLIR regions. Constants live inside those regions,
// not as unrelated graph nodes that the device tensor allocator must interpret.
static Value number(OpBuilder &b, Location loc, Type type, double value) {
  Attribute attr = isa<FloatType>(type) ? Attribute(b.getFloatAttr(type, value))
                                      : Attribute(b.getIntegerAttr(type, int64_t(value)));
  return b.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
}
static Value castScalar(OpBuilder &b, Location loc, Value value, Type to) {
  Type from = value.getType();
  if (from == to)
    return value;
  if (isa<FloatType>(from) && isa<FloatType>(to)) {
    if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth())
      return b.create<arith::ExtFOp>(loc, to, value);
    return b.create<arith::TruncFOp>(loc, to, value);
  }
  if (isa<IntegerType>(from) && isa<IntegerType>(to)) {
    if (from.getIntOrFloatBitWidth() < to.getIntOrFloatBitWidth())
      return b.create<arith::ExtSIOp>(loc, to, value);
    return b.create<arith::TruncIOp>(loc, to, value);
  }
  llvm_unreachable("regular lowering validates numeric element types");
}
static Value binary(OpBuilder &b, Location loc, StringRef kind, Value a, Value c) {
  bool fp = isa<FloatType>(a.getType());
  if (kind == "add")
    return fp ? Value(b.create<arith::AddFOp>(loc, a, c))
              : Value(b.create<arith::AddIOp>(loc, a, c));
  if (kind == "sub")
    return fp ? Value(b.create<arith::SubFOp>(loc, a, c))
              : Value(b.create<arith::SubIOp>(loc, a, c));
  if (kind == "mul")
    return fp ? Value(b.create<arith::MulFOp>(loc, a, c))
              : Value(b.create<arith::MulIOp>(loc, a, c));
  return fp ? Value(b.create<arith::DivFOp>(loc, a, c))
            : Value(b.create<arith::DivSIOp>(loc, a, c));
}
static Value relu(OpBuilder &b, Location loc, Value value) {
  Value zero = number(b, loc, value.getType(), 0);
  Value negative = isa<FloatType>(value.getType())
      ? Value(b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, value, zero))
      : Value(b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, value, zero));
  return b.create<arith::SelectOp>(loc, negative, zero, value);
}
static AffineMap identity(unsigned rank, MLIRContext *context) {
  return AffineMap::getMultiDimIdentityMap(rank, context);
}
static AffineMap map(unsigned loops, ArrayRef<AffineExpr> expressions,
                     MLIRContext *context) {
  return AffineMap::get(loops, 0, expressions, context);
}
static AffineMap broadcastMap(RankedTensorType input, unsigned outputRank,
                              MLIRContext *context) {
  SmallVector<AffineExpr> expressions;
  for (int64_t d = 0; d < input.getRank(); ++d)
    expressions.push_back(input.getDimSize(d) == 1
        ? getAffineConstantExpr(0, context)
        : getAffineDimExpr(outputRank - input.getRank() + d, context));
  return map(outputRank, expressions, context);
}
static Value empty(OpBuilder &b, Location loc, RankedTensorType type) {
  return b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
}

/// Every node has one destination, one result and explicit iteration semantics.
/// The unused destination argument of a write-only payload is intentional:
/// bufferization must not read uninitialized tensor.empty storage.
static Value generic(OpBuilder &b, Location loc, ValueRange inputs, Value init,
                     ArrayRef<AffineMap> maps, ArrayRef<Iterator> iterators,
                     StringRef role, Payload payload) {
  auto op = b.create<linalg::GenericOp>(
      loc, TypeRange{init.getType()}, inputs, ValueRange{init}, maps, iterators,
      [&](OpBuilder &nested, Location at, ValueRange args) {
        nested.create<linalg::YieldOp>(at, payload(nested, at, args));
      });
  op->setAttr("gac.linalg_role", b.getStringAttr(role));
  return op.getResult(0);
}
static Value pointwise(OpBuilder &b, Location loc, ValueRange inputs,
                       RankedTensorType out, StringRef role, Payload payload) {
  SmallVector<AffineMap> maps;
  for (Value input : inputs)
    maps.push_back(broadcastMap(tensor(input), out.getRank(), b.getContext()));
  maps.push_back(identity(out.getRank(), b.getContext()));
  SmallVector<Iterator> iterators(out.getRank(), Iterator::parallel);
  return generic(b, loc, inputs, empty(b, loc, out), maps, iterators, role, payload);
}
static Value zeroTensor(OpBuilder &b, Location loc, RankedTensorType type) {
  return pointwise(b, loc, {}, type, "initialize",
      [&](OpBuilder &nested, Location at, ValueRange) {
        return number(nested, at, type.getElementType(), 0);
      });
}
static Value convertTensor(OpBuilder &b, Location loc, Value input,
                           RankedTensorType out) {
  if (input.getType() == out)
    return input;
  return pointwise(b, loc, ValueRange{input}, out, "convert",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        return castScalar(nested, at, args[0], out.getElementType());
      });
}

static FailureOr<Value> lowerPointwise(OpBuilder &b, Operation *op,
                                       StringRef kind) {
  auto out = tensor(op->getResult(0));
  Type compute = accumulationType(out.getElementType(), b);
  return pointwise(b, op->getLoc(), op->getOperands(), out, "elementwise",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value result = castScalar(nested, at, args[0], compute);
        if (kind != "relu") {
          Value rhs = castScalar(nested, at, args[1], compute);
          result = binary(nested, at, kind == "add_relu" ? "add" : kind,
                          result, rhs);
        }
        if (kind == "relu" || kind == "add_relu")
          result = relu(nested, at, result);
        return castScalar(nested, at, result, out.getElementType());
      });
}

// 把 MatMul/Gemm（以及带 bias 的融合形式）改写成 linalg.generic：
//   - 输出维度都是 parallel 迭代轴；
//   - 额外增加一个 K 轴作为唯一的 reduction 迭代轴；
//   - A/B/输出各有一张 AffineMap，批维通过共享 loop dim 或常量 0 表达广播。
// 这样后续可以直接使用 Linalg 的 tiling/fusion，而不是把矩阵乘展开成平铺标量循环。
static FailureOr<Value> lowerMatmul(OpBuilder &b, Operation *op,
                                    StringRef kind) {
  auto left = tensor(op->getOperand(0)), right = tensor(op->getOperand(1));
  auto out = tensor(op->getResult(0));
  auto *context = b.getContext();
  bool gemm = kind == "gemm";
  // ONNX MatMul 允许 rank-1 操作数：向量参与时对应输出维会被消去。
  bool lv = left.getRank() == 1, rv = right.getRank() == 1;
  // loops = 输出维 + 1 个 K 归约维；K 轴放在最后，编号为 rank。
  unsigned rank = out.getRank(), loops = rank + 1;
  auto dim = [&](unsigned d) { return getAffineDimExpr(d, context); };
  AffineExpr k = dim(rank);
  SmallVector<AffineExpr> lhs, rhs, dst;
  // 输出使用恒等映射：第 d 个输出维直接对应第 d 个 loop。
  for (unsigned d = 0; d < rank; ++d)
    dst.push_back(dim(d));
  // Gemm 只接受 rank-2；transA/transB 决定 K 轴出现在 A/B 的第一维还是第二维。
  if (gemm) {
    if (left.getRank() != 2 || right.getRank() != 2 || rank != 2)
      return op->emitOpError("Linalg Gemm requires rank-two A, B and Y"), failure();
    // A: [M,K] 或 [K,M]；B: [K,N] 或 [N,K]。
    lhs = integer(op, "transA", 0) ? SmallVector<AffineExpr>{k, dim(0)}
                                  : SmallVector<AffineExpr>{dim(0), k};
    rhs = integer(op, "transB", 0) ? SmallVector<AffineExpr>{dim(1), k}
                                  : SmallVector<AffineExpr>{k, dim(1)};
  } else {
    // 普通 MatMul：batches 是输出中属于 batch 广播的维数。
    // 例：A[M,K] × B[K,N] → rank=2, batches=0；
    //     A[B,M,K] × B[B,K,N] → rank=3, batches=1。
    unsigned batches = rank - unsigned(!lv) - unsigned(!rv);
    // 只映射 A/B 自己的 batch 维，并右对齐到输出的 batch 维：
    // 输入该轴长度为 1 时映射到常量 0（广播），否则共享对应输出 loop。
    auto batch = [&](RankedTensorType input, SmallVectorImpl<AffineExpr> &index) {
      for (int64_t d = 0; d + 2 < input.getRank(); ++d)
        index.push_back(input.getDimSize(d) == 1
            ? getAffineConstantExpr(0, context)
            : dim(batches - (input.getRank() - 2) + d));
    };
    batch(left, lhs);
    batch(right, rhs);
    // A 的矩阵维：非向量时取 M，最后是 K；向量的 A 只有 K。
    if (!lv)
      lhs.push_back(dim(batches));
    lhs.push_back(k);
    // B 的矩阵维：先 K，非向量时再取 N。
    rhs.push_back(k);
    if (!rv)
      rhs.push_back(dim(rank - 1));
  }
  // 累加类型：FP16 输入用 FP32 累加，避免逐次乘加在 half 上累积误差；
  // 最终通过 convertTensor 转回声明的输出元素类型。
  Type compute = accumulationType(out.getElementType(), b);
  auto accum = RankedTensorType::get(out.getShape(), compute);
  // 所有输出维都是 parallel，只有 K 是 reduction；后续 tiling 可以据此
  // 区分独立输出块和必须串行累加的归约轴。
  SmallVector<Iterator> iterators(rank, Iterator::parallel);
  iterators.push_back(Iterator::reduction);
  SmallVector<AffineMap> maps{map(loops, lhs, context), map(loops, rhs, context),
                              map(loops, dst, context)};
  // linalg.generic 的 payload：acc + cast(A) * cast(B)，初值为零张量。
  Value result = generic(b, op->getLoc(), op->getOperands().take_front(2),
      zeroTensor(b, op->getLoc(), accum), maps, iterators, "matmul",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value a = castScalar(nested, at, args[0], compute);
        Value c = castScalar(nested, at, args[1], compute);
        return binary(nested, at, "add", args[2], binary(nested, at, "mul", a, c));
      });

  // epilogue：Gemm 先乘 alpha；bias 存在时（Gemm 中先乘 beta）加到结果；
  // bias_first 只影响非 Gemm 的加法顺序，以保持原图的浮点舍入点；
  // matmul_bias_relu 最后再应用 ReLU。
  bool bias = op->getNumOperands() > 2;
  if (gemm || bias) {
    SmallVector<Value> inputs{result};
    if (bias)
      inputs.push_back(op->getOperand(2));
    result = pointwise(b, op->getLoc(), inputs, out, "matmul-epilogue",
        [&](OpBuilder &nested, Location at, ValueRange args) {
          Value value = args[0];
          if (gemm)
            value = binary(nested, at, "mul", value,
                           number(nested, at, compute, floating(op, "alpha", 1)));
          if (bias) {
            Value addend = castScalar(nested, at, args[1], compute);
            if (gemm)
              addend = binary(nested, at, "mul",
                  number(nested, at, compute, floating(op, "beta", 1)), addend);
            value = integer(op, "bias_first", 0) && !gemm
                ? binary(nested, at, "add", addend, value)
                : binary(nested, at, "add", value, addend);
          }
          if (kind == "matmul_bias_relu")
            value = relu(nested, at, value);
          return castScalar(nested, at, value, out.getElementType());
        });
  }
  // 最后把累加类型转换回输出声明的元素类型（例如 FP32 累加 → FP16 输出）。
  return convertTensor(b, op->getLoc(), result, out);
}

/// Loop order is [output dimensions, reduction dimensions]. Keeping these
/// axes explicit is what lets later Linalg tiling distinguish independent work
/// from accumulation; it is not recoverable from a mere scalar flat loop.
static Value sum(OpBuilder &b, Location loc, Value input,
                 ArrayRef<int64_t> axes, RankedTensorType out,
                 bool keepDimensions, Value mean = {}) {
  auto in = tensor(input);
  unsigned rank = out.getRank(), loops = rank + axes.size();
  auto *context = b.getContext();
  SmallVector<AffineExpr> source, destination;
  unsigned p = 0, r = rank;
  for (int64_t d = 0; d < in.getRank(); ++d) {
    if (llvm::is_contained(axes, d)) {
      source.push_back(getAffineDimExpr(r++, context));
      if (keepDimensions)
        ++p;
    } else {
      source.push_back(getAffineDimExpr(p++, context));
    }
  }
  for (unsigned d = 0; d < rank; ++d)
    destination.push_back(getAffineDimExpr(d, context));
  SmallVector<Value> inputs{input};
  SmallVector<AffineMap> maps{map(loops, source, context)};
  if (mean) {
    inputs.push_back(mean);
    maps.push_back(map(loops, destination, context));
  }
  maps.push_back(map(loops, destination, context));
  SmallVector<Iterator> iterators(rank, Iterator::parallel);
  iterators.append(axes.size(), Iterator::reduction);
  Type compute = out.getElementType();
  return generic(b, loc, inputs, zeroTensor(b, loc, out), maps, iterators,
      mean ? "squared-deviation-reduction" : "sum-reduction",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value value = castScalar(nested, at, args[0], compute);
        if (mean) {
          Value difference = binary(nested, at, "sub", value, args[1]);
          value = binary(nested, at, "mul", difference, difference);
        }
        return binary(nested, at, "add", args.back(), value);
      });
}
static Value divide(OpBuilder &b, Location loc, Value input, int64_t count,
                    RankedTensorType out) {
  Type compute = tensor(input).getElementType();
  return pointwise(b, loc, ValueRange{input}, out, "mean",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Value result = binary(nested, at, "div", args[0],
                              number(nested, at, compute, count));
        return castScalar(nested, at, result, out.getElementType());
      });
}

static FailureOr<Value> lowerReduce(OpBuilder &b, Operation *op, bool mean) {
  Value input = op->getOperand(0);
  auto in = tensor(input), out = tensor(op->getResult(0));
  auto axes = getI64ArrayAttr(op, "axes");
  if (op->getNumOperands() > 1) {
    auto constant = getConstantIntegers(op->getOperand(1));
    if (!constant)
      return op->emitOpError("Linalg reduction requires static axes"), failure();
    axes = *constant;
  }
  if (axes.empty() && integer(op, "noop_with_empty_axes", 0))
    return pointwise(b, op->getLoc(), ValueRange{input}, out, "identity",
        [](OpBuilder &, Location, ValueRange args) { return args[0]; });
  if (axes.empty())
    for (int64_t d = 0; d < in.getRank(); ++d)
      axes.push_back(d);
  for (auto &axis : axes)
    if (axis < 0)
      axis += in.getRank();
  llvm::sort(axes);
  Type compute = accumulationType(out.getElementType(), b);
  auto accum = RankedTensorType::get(out.getShape(), compute);
  Value result = sum(b, op->getLoc(), input, axes, accum,
                     integer(op, "keepdims", 1));
  int64_t count = 1;
  for (int64_t axis : axes)
    count *= in.getDimSize(axis);
  return mean ? divide(b, op->getLoc(), result, count, out)
              : convertTensor(b, op->getLoc(), result, out);
}

static FailureOr<Value> lowerSoftmax(OpBuilder &b, Operation *op) {
  Value input = op->getOperand(0);
  auto out = tensor(op->getResult(0));
  int64_t axis = integer(op, "axis", -1);
  if (axis < 0)
    axis += out.getRank();
  auto softmax = b.create<linalg::SoftmaxOp>(
      op->getLoc(), TypeRange{out}, input, empty(b, op->getLoc(), out), axis);
  softmax->setAttr("gac.linalg_role", b.getStringAttr("softmax"));
  return softmax->getResult(0);
}

static FailureOr<Value> lowerLayerNorm(OpBuilder &b, Operation *op) {
  Value input = op->getOperand(0);
  auto out = tensor(op->getResult(0));
  if (!out.getElementType().isF32() || integer(op, "stash_type", 1) != 1)
    return op->emitOpError("Linalg LayerNorm currently requires FP32 with stash_type=1"), failure();
  int64_t axis = integer(op, "axis", -1);
  if (axis < 0)
    axis += out.getRank();
  SmallVector<int64_t> axes;
  for (int64_t d = axis; d < out.getRank(); ++d)
    axes.push_back(d);
  auto stats = RankedTensorType::get(out.getShape().take_front(axis), b.getF32Type());
  int64_t count = product(out.getShape().drop_front(axis));
  Value total = sum(b, op->getLoc(), input, axes, stats, false);
  Value mean = divide(b, op->getLoc(), total, count, stats);
  // A second centered reduction preserves the existing stable variance
  // algorithm; E[x*x] - E[x]^2 would lose accuracy for nearly constant inputs.
  Value deviations = sum(b, op->getLoc(), input, axes, stats, false, mean);
  SmallVector<Value> inputs{input, mean, deviations};
  for (Value parameter : op->getOperands().drop_front())
    inputs.push_back(parameter);
  unsigned rank = out.getRank();
  SmallVector<AffineExpr> prefix;
  for (int64_t d = 0; d < axis; ++d)
    prefix.push_back(getAffineDimExpr(d, b.getContext()));
  SmallVector<AffineMap> maps{identity(rank, b.getContext()),
                              map(rank, prefix, b.getContext()),
                              map(rank, prefix, b.getContext())};
  for (Value parameter : op->getOperands().drop_front())
    maps.push_back(broadcastMap(tensor(parameter), rank, b.getContext()));
  maps.push_back(identity(rank, b.getContext()));
  SmallVector<Iterator> iterators(rank, Iterator::parallel);
  return generic(b, op->getLoc(), inputs, empty(b, op->getLoc(), out), maps,
      iterators, "layer-normalization",
      [&](OpBuilder &nested, Location at, ValueRange args) {
        Type fp = nested.getF32Type();
        Value x = castScalar(nested, at, args[0], fp);
        Value variance = binary(nested, at, "div", args[2], number(nested, at, fp, count));
        Value denominator = nested.create<math::SqrtOp>(at,
            binary(nested, at, "add", variance,
                   number(nested, at, fp, floating(op, "epsilon", 1e-5))));
        Value value = binary(nested, at, "div",
                             binary(nested, at, "sub", x, args[1]), denominator);
        if (op->getNumOperands() > 1)
          value = binary(nested, at, "mul", value, castScalar(nested, at, args[3], fp));
        if (op->getNumOperands() > 2)
          value = binary(nested, at, "add", value, castScalar(nested, at, args[4], fp));
        return castScalar(nested, at, value, out.getElementType());
      });
}

static bool isRegular(StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases("matmul", "gemm", "matmul_bias", "matmul_bias_relu", true)
      .Cases("add", "sub", "mul", "div", "relu", "add_relu", true)
      .Cases("reduce_sum", "reduce_mean", "softmax", "layer_norm", true)
      .Cases("conv2d", "conv2d_relu", true).Default(false);
}
static LogicalResult validate(Operation *op) {
  if (failed(inferOperation(op, false)))
    return failure();
  StringRef kind = op->getName().getStringRef().drop_front(4);
  bool reduction = kind == "reduce_sum" || kind == "reduce_mean";
  SmallVector<Type> types(op->getOperandTypes());
  types.append(op->getResultTypes().begin(), op->getResultTypes().end());
  Type dataElement = cast<ShapedType>(op->getResult(0).getType()).getElementType();
  for (auto [index, type] : llvm::enumerate(types)) {
    auto shaped = dyn_cast<RankedTensorType>(type);
    bool axesOperand = reduction && index == 1 && op->getNumOperands() > 1;
    if (!shaped || !shaped.hasStaticShape() ||
        llvm::any_of(shaped.getShape(), [&](int64_t size) {
          return size < 0 || (size == 0 && !axesOperand);
        }))
      return op->emitOpError("regular Linalg lowering requires positive static tensor shapes");
    int64_t elements = 1;
    for (int64_t extent : shaped.getShape()) {
      if (extent && elements > std::numeric_limits<int64_t>::max() / extent)
        return op->emitOpError("regular Linalg tensor element count overflows i64");
      elements *= extent;
    }
    Type element = shaped.getElementType();
    if (shaped.getEncoding())
      return op->emitOpError("regular Linalg lowering requires unencoded row-major tensors");
    if (!element.isF16() && !element.isF32() && !element.isInteger(32) &&
        !element.isInteger(64))
      return op->emitOpError("regular Linalg lowering supports FP16, FP32, i32 and i64 tensors");
    if (!axesOperand && element != dataElement)
      return op->emitOpError("regular Linalg data operands and result must have matching element types");
  }
  if (kind == "gemm" && !isa<FloatType>(dataElement))
    return op->emitOpError("Linalg Gemm requires floating-point tensors");
  if (kind == "softmax" && !isa<FloatType>(dataElement))
    return op->emitOpError("Linalg Softmax requires floating-point tensors");
  return success();
}

struct LowerRegularToLinalgPass
    : PassWrapper<LowerRegularToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRegularToLinalgPass)
  StringRef getArgument() const final { return "gac-regular-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower regular GAC tensor operators to executable structured Linalg IR";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registerLinalgDialects(registry);
  }
  void runOnOperation() override {
    if (failed(lowerRegularToLinalg(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult lowerRegularToLinalg(ModuleOp module) {
  SmallVector<Operation *> worklist;
  module.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() == "gac" &&
        isRegular(op->getName().getStringRef().drop_front(4)))
      worklist.push_back(op);
  });
  // Reject unsupported regular forms before changing any graph nodes. They must
  // not silently take the old scalar GAC path and pretend to use Linalg.
  for (Operation *op : worklist)
    if (failed(validate(op)))
      return failure();
  for (Operation *op : worklist) {
    OpBuilder b(op);
    StringRef kind = op->getName().getStringRef().drop_front(4);
    FailureOr<Value> replacement = failure();
    if (kind == "conv2d" || kind == "conv2d_relu")
      replacement = lowerConvolutionToLinalg(b, op);
    else if (kind == "matmul" || kind == "gemm" || kind == "matmul_bias" ||
             kind == "matmul_bias_relu")
      replacement = lowerMatmul(b, op, kind);
    else if (kind == "reduce_sum" || kind == "reduce_mean")
      replacement = lowerReduce(b, op, kind == "reduce_mean");
    else if (kind == "softmax")
      replacement = lowerSoftmax(b, op);
    else if (kind == "layer_norm")
      replacement = lowerLayerNorm(b, op);
    else
      replacement = lowerPointwise(b, op, kind);
    if (failed(replacement))
      return failure();
    Operation *resultOp = replacement->getDefiningOp();
    for (NamedAttribute attr : op->getAttrs())
      if (attr.getName().getValue().starts_with("gac.") &&
          attr.getName().getValue() != "gac.input_positions" &&
          attr.getName().getValue() != "gac.input_layout" &&
          attr.getName().getValue() != "gac.weight_layout")
        resultOp->setAttr(attr.getName(), attr.getValue());
    resultOp->setAttr("gac.lowered_from", b.getStringAttr(op->getName().getStringRef()));
    op->getResult(0).replaceAllUsesWith(*replacement);
    op->erase();
  }
  return success();
}
std::unique_ptr<Pass> createLowerRegularToLinalgPass() {
  return std::make_unique<LowerRegularToLinalgPass>();
}
void registerLinalgPasses() { PassRegistration<LowerRegularToLinalgPass>(); }
void registerLinalgDialects(DialectRegistry &registry) {
  registerGacDialects(registry);
  registry.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                  math::MathDialect, tensor::TensorDialect>();
  // Keep upstream tiling interfaces available while operators are structured;
  // selecting a hardware tile/schedule is a separate policy, not an import rule.
  linalg::registerTilingInterfaceExternalModels(registry);
}
} // namespace gac
