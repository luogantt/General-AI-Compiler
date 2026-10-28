#include "gac/templates/CudaConvolution.h"
#include "gac/templates/ScalarBuilder.h"
#include "gac/CudaLowering.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
namespace gac::templates {
namespace {
static int64_t integer(Operation *op, StringRef name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
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
static RankedTensorType tensor(Value value) {
  return cast<RankedTensorType>(value.getType());
}
static StringRef opKind(Operation *op) {
  return op->getName().getStringRef().drop_front(4);
}
} // namespace

namespace detail {
// Consecutive output channels in one convolution group share the same input
// sample at each reduction step. Keep a small vector of FP32 accumulators in
// SSA registers, sharing address calculation and (for DCNv2) bilinear sampling.
// Tile=1 is the scalar implementation used by every target. The CUDA planner
// only selects a larger tile when it cannot cross a convolution group.
SmallVector<Value> emitConvolutionOutputs(
    Scalar &s, Operation *op, ArrayRef<Value> ptrs, ArrayRef<Value> c,
    int64_t channelTile) {
  bool deform = opKind(op) == "deform_conv2d";
  unsigned weightOperand = deform ? 3 : 1;
  auto xs = tensor(op->getOperand(0)).getShape();
  auto ws = tensor(op->getOperand(weightOperand)).getShape();
  auto os = tensor(op->getResult(0)).getShape();
  auto strides = ints(op, deform ? "stride" : "strides", {1, 1});
  auto dilation = ints(op, deform ? "dilation" : "dilations", {1, 1});
  auto pads = ints(op, deform ? "padding" : "pads", {0, 0});
  int64_t groups = integer(op, deform ? "groups" : "group", 1);
  int64_t points = ws[2] * ws[3];
  auto read = [&](unsigned operand, Value index) {
    return s.read(ptrs[operand], index,
                  tensor(op->getOperand(operand)).getElementType());
  };
  SmallVector<Value> initial(channelTile, s.f(0));
  Value begin = s.i(0), end = s.i(ws[1] * points), step = s.i(1);
  auto reduction = s.b.create<scf::ForOp>(
      s.loc, begin, end, step, initial,
      [&](OpBuilder &, Location, Value k, ValueRange carry) {
        auto r = s.coords(k, {ws[1], ws[2], ws[3]});
        Value channel = s.add(
            s.mul(s.div(c[1], s.i(os[1] / groups)), s.i(ws[1])), r[0]);
        Value y = s.sub(s.add(s.mul(c[2], s.i(strides[0])),
                              s.mul(r[1], s.i(dilation[0]))), s.i(pads[0]));
        Value x = s.sub(s.add(s.mul(c[3], s.i(strides[1])),
                              s.mul(r[2], s.i(dilation[1]))), s.i(pads[1]));
        auto accumulate = [&](Value sample) {
          SmallVector<Value> next;
          for (int64_t tile = 0; tile < channelTile; ++tile) {
            Value outputChannel = s.add(c[1], s.i(tile));
            Value weight = read(weightOperand,
                s.linear({outputChannel, r[0], r[1], r[2]}, ws));
            next.push_back(s.add(carry[tile], s.mul(sample, weight)));
          }
          return next;
        };
        SmallVector<Value> next;
        if (deform) {
          auto offsets = tensor(op->getOperand(1)).getShape();
          auto masks = tensor(op->getOperand(2)).getShape();
          int64_t deformGroups = integer(op, "deform_groups", 1);
          Value deformGroup = s.div(channel, s.i(xs[1] / deformGroups));
          Value point = s.add(s.mul(r[1], s.i(ws[3])), r[2]);
          Value maskChannel = s.add(s.mul(deformGroup, s.i(points)), point);
          Value offsetChannel = s.mul(maskChannel, s.i(2));
          Value dy = read(1, s.linear({c[0], offsetChannel, c[2], c[3]}, offsets));
          Value dx = read(1, s.linear(
              {c[0], s.add(offsetChannel, s.i(1)), c[2], c[3]}, offsets));
          y = s.add(s.cvt(y, s.b.getF32Type()), dy);
          x = s.add(s.cvt(x, s.b.getF32Type()), dx);
          Value sample = s.bilinear(x, y, xs[2], xs[3], [&](Value xx, Value yy) {
            return read(0, s.linear({c[0], channel, yy, xx}, xs));
          });
          Value mask = read(2, s.linear({c[0], maskChannel, c[2], c[3]}, masks));
          // Keep sample*mask before multiplication by weight, including the
          // original FP32 rounding and nonfinite behavior.
          next = accumulate(s.mul(sample, mask));
        } else {
          Value valid = s.binary(
              "and", s.binary("and", s.cmp("ge", y, s.i(0)),
                               s.cmp("lt", y, s.i(xs[2]))),
              s.binary("and", s.cmp("ge", x, s.i(0)),
                       s.cmp("lt", x, s.i(xs[3]))));
          SmallVector<Type> types(channelTile, s.b.getF32Type());
          auto guarded = s.b.create<scf::IfOp>(s.loc, types, valid, true);
          {
            OpBuilder::InsertionGuard guard(s.b);
            s.b.setInsertionPointToStart(guarded.thenBlock());
            Value sample = read(0, s.linear({c[0], channel, y, x}, xs));
            s.b.create<scf::YieldOp>(s.loc, accumulate(sample));
          }
          {
            OpBuilder::InsertionGuard guard(s.b);
            s.b.setInsertionPointToStart(guarded.elseBlock());
            // Padding skips the multiply entirely. Multiplying a zero-filled
            // sample by an Inf/NaN weight would change the scalar semantics.
            s.b.create<scf::YieldOp>(s.loc, carry);
          }
          next.assign(guarded.getResults().begin(), guarded.getResults().end());
        }
        s.b.create<scf::YieldOp>(s.loc, next);
      });
  SmallVector<Value> result(reduction.getResults());
  for (int64_t tile = 0; tile < channelTile; ++tile) {
    if (ptrs.size() > weightOperand + 1)
      result[tile] = s.add(result[tile], read(weightOperand + 1, s.add(c[1], s.i(tile))));
    if (opKind(op) == "conv2d_relu")
      result[tile] = s.relu(result[tile]);
  }
  return result;
}
} // namespace detail

LogicalResult planCudaConvolution(
    Operation *op, int64_t requested, std::optional<CudaConvolutionPlan> &plan,
    std::string &reason) {
  plan.reset();
  reason.clear();
  if (requested != 1 && requested != 2 && requested != 4 && requested != 8)
    return op->emitError("CUDA convolution channel tile must be 1, 2, 4 or 8");
  StringRef kind = opKind(op);
  if (requested == 1 || (kind != "conv2d" && kind != "conv2d_relu" &&
                         kind != "deform_conv2d"))
    return success();

  // The graph verifier checks the convolution semantic contract. The planner
  // additionally checks the resource/layout requirements of this template.
  if (op->getNumResults() != 1 || op->getNumOperands() < 2) {
    reason = "output-channel register tiling requires a single convolution output";
    return success();
  }
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!output || !output.hasStaticShape() || output.getRank() != 4 ||
      llvm::any_of(output.getShape(), [](int64_t dim) { return dim <= 0; })) {
    reason = "output-channel register tiling requires a positive static NCHW output";
    return success();
  }
  if (!output.getElementType().isF32() ||
      llvm::any_of(op->getOperands(), [](Value input) {
        auto type = dyn_cast<RankedTensorType>(input.getType());
        return !type || !type.hasStaticShape() || !type.getElementType().isF32();
      })) {
    reason = "output-channel register tiling currently requires FP32 operands";
    return success();
  }
  int64_t groups = integer(op, kind == "deform_conv2d" ? "groups" : "group", 1);
  if (groups <= 0 || output.getDimSize(1) % groups) {
    reason = "output channels must be divisible by a positive convolution group count";
    return success();
  }
  int64_t channelsPerGroup = output.getDimSize(1) / groups;
  if (channelsPerGroup % requested) {
    reason = "output channels per group are not divisible by the requested register tile";
    return success();
  }
  if (output.getNumElements() > std::numeric_limits<int32_t>::max()) {
    reason = "output-channel register tiling requires signed 32-bit output indexing";
    return success();
  }

  CudaConvolutionPlan selected;
  selected.kind = kind.str();
  selected.channelTile = requested;
  selected.groups = groups;
  selected.channelsPerGroup = channelsPerGroup;
  selected.logicalThreads = output.getNumElements() / requested;
  selected.gridSize = (selected.logicalThreads + selected.blockSize - 1) /
                      selected.blockSize;
  for (Value input : op->getOperands())
    selected.inputShapes.emplace_back(tensor(input).getShape());
  selected.outputShape.assign(output.getShape().begin(), output.getShape().end());
  plan = std::move(selected);
  return success();
}

LogicalResult emitCudaConvolution(OpBuilder &builder, Operation *op,
                                  ArrayRef<Value> pointers,
                                  const CudaConvolutionPlan &plan) {
  detail::Scalar s{builder, op->getLoc(), {}};
  auto output = tensor(op->getResult(0));
  SmallVector<int64_t> workShape(plan.outputShape);
  workShape[1] /= plan.channelTile;
  Value tid = createCudaGlobalThreadId(s.b, s.loc);
  Value valid = s.binary("and", s.cmp("ge", tid, s.i(0)),
                         s.cmp("lt", tid, s.i(plan.logicalThreads)));
  s.b.create<scf::IfOp>(s.loc, valid, [&](OpBuilder &, Location) {
    auto coordinates = s.coords(tid, workShape);
    coordinates[1] = s.mul(coordinates[1], s.i(plan.channelTile));
    auto values = detail::emitConvolutionOutputs(
        s, op, pointers.drop_back(), coordinates, plan.channelTile);
    for (int64_t tile = 0; tile < plan.channelTile; ++tile) {
      auto destination = coordinates;
      destination[1] = s.add(coordinates[1], s.i(tile));
      s.write(values[tile], pointers.back(),
              s.linear(destination, output.getShape()), output.getElementType());
    }
    s.b.create<scf::YieldOp>(s.loc);
  });
  return success();
}

llvm::json::Object cudaConvolutionMetadata(const CudaConvolutionPlan &plan) {
  auto shape = [](ArrayRef<int64_t> dimensions) {
    llvm::json::Array result;
    for (int64_t dimension : dimensions)
      result.push_back(dimension);
    return result;
  };
  llvm::json::Array inputs;
  for (const auto &input : plan.inputShapes)
    inputs.push_back(shape(input));
  return llvm::json::Object{{"schedule", plan.schedule}, {"kind", plan.kind},
          {"channel_tile", plan.channelTile}, {"groups", plan.groups},
          {"channels_per_group", plan.channelsPerGroup},
          {"logical_threads", plan.logicalThreads},
          {"block_size", plan.blockSize}, {"grid_size", plan.gridSize},
          {"static_shared_memory_bytes", plan.sharedMemoryBytes},
          {"input_shapes", std::move(inputs)},
          {"output_shape", shape(plan.outputShape)}};
}
} // namespace gac::templates
