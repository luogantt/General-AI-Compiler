#ifndef GAC_DIALECT_H
#define GAC_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace gac {

/// Refine tensor result types without evaluating an operator. Known dimensions
/// are checked, never silently overwritten. Unranked tensors are legal at
/// import.
mlir::LogicalResult inferOperation(mlir::Operation *op,
                                   bool updateResult = true);
llvm::SmallVector<int64_t> getI64ArrayAttr(mlir::Operation *op,
                                           llvm::StringRef name);
std::optional<llvm::SmallVector<int64_t>>
getConstantIntegers(mlir::Value value);

/// Graph operations have tensor value semantics, not device-memory side
/// effects. This interface lets upstream CSE and canonicalization reason about
/// dead ops.
template <typename Concrete>
class TensorOp
    : public mlir::Op<Concrete, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                      mlir::MemoryEffectOpInterface::Trait> {
public:
  using Base = mlir::Op<Concrete, mlir::OpTrait::VariadicOperands,
                        mlir::OpTrait::OneResult, mlir::OpTrait::ZeroRegions,
                        mlir::MemoryEffectOpInterface::Trait>;
  using Base::Base;
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
  void
  getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &) {}
  mlir::LogicalResult verify() {
    return inferOperation(this->getOperation(), false);
  }
};

// Handwritten, registered MLIR operation classes. Generic assembly is
// deliberate: the IR remains parseable without maintaining a second custom
// parser/printer.
#define GAC_DECLARE_OP(CLASS, NAME)                                            \
  class CLASS : public TensorOp<CLASS> {                                       \
  public:                                                                      \
    using TensorOp::TensorOp;                                                  \
    static llvm::StringRef getOperationName() { return "gac." NAME; }          \
  };

GAC_DECLARE_OP(ConstantOp, "constant")
GAC_DECLARE_OP(MatMulOp, "matmul")
GAC_DECLARE_OP(GemmOp, "gemm")
GAC_DECLARE_OP(AddOp, "add")
GAC_DECLARE_OP(SubOp, "sub")
GAC_DECLARE_OP(MulOp, "mul")
GAC_DECLARE_OP(DivOp, "div")
GAC_DECLARE_OP(PowOp, "pow")
GAC_DECLARE_OP(GreaterOp, "greater")
GAC_DECLARE_OP(LessOp, "less")
GAC_DECLARE_OP(EqualOp, "equal")
GAC_DECLARE_OP(AndOp, "and")
GAC_DECLARE_OP(MaxOp, "max")
GAC_DECLARE_OP(WhereOp, "where")
GAC_DECLARE_OP(ReluOp, "relu")
GAC_DECLARE_OP(GeluOp, "gelu")
GAC_DECLARE_OP(SigmoidOp, "sigmoid")
GAC_DECLARE_OP(LogOp, "log")
GAC_DECLARE_OP(SinOp, "sin")
GAC_DECLARE_OP(CosOp, "cos")
GAC_DECLARE_OP(ExpOp, "exp")
GAC_DECLARE_OP(SqrtOp, "sqrt")
GAC_DECLARE_OP(ClampOp, "clamp")
GAC_DECLARE_OP(IdentityOp, "identity")
GAC_DECLARE_OP(CastOp, "cast")
GAC_DECLARE_OP(ReshapeOp, "reshape")
GAC_DECLARE_OP(TransposeOp, "transpose")
GAC_DECLARE_OP(ConcatOp, "concat")
GAC_DECLARE_OP(ExpandOp, "expand")
GAC_DECLARE_OP(SliceOp, "slice")
GAC_DECLARE_OP(ReduceSumOp, "reduce_sum")
GAC_DECLARE_OP(ReduceMeanOp, "reduce_mean")
GAC_DECLARE_OP(SoftmaxOp, "softmax")
GAC_DECLARE_OP(LayerNormOp, "layer_norm")
GAC_DECLARE_OP(Conv2DOp, "conv2d")
GAC_DECLARE_OP(DeformConv2DOp, "deform_conv2d")
GAC_DECLARE_OP(BatchNormOp, "batch_norm")
GAC_DECLARE_OP(MaxPool2DOp, "max_pool2d")
GAC_DECLARE_OP(GridSampleOp, "grid_sample")
GAC_DECLARE_OP(MSDeformAttnOp, "ms_deform_attn")
// Evaluate coordinate Div/Add at the sampled point; do not materialize them.
GAC_DECLARE_OP(MSDeformAttnOffsetsOp, "ms_deform_attn_offsets")
GAC_DECLARE_OP(ShapeOp, "shape")
GAC_DECLARE_OP(GatherOp, "gather")
GAC_DECLARE_OP(UnsqueezeOp, "unsqueeze")
GAC_DECLARE_OP(SqueezeOp, "squeeze")
// Shape-guarded ONNX If: remove one axis iff its extent equals one. The input
// data is never evaluated to select the branch; unresolved extents stay legal
// in graph IR and must resolve before static device legalization.
GAC_DECLARE_OP(SqueezeIfUnitDimOp, "squeeze_if_unit_dim")
GAC_DECLARE_OP(RangeOp, "range")
GAC_DECLARE_OP(ConstantOfShapeOp, "constant_of_shape")
GAC_DECLARE_OP(ResizeOp, "resize")
GAC_DECLARE_OP(TileOp, "tile")
GAC_DECLARE_OP(ScatterNDOp, "scatter_nd")
GAC_DECLARE_OP(NonZeroOp, "nonzero")
GAC_DECLARE_OP(AddReluOp, "add_relu")
GAC_DECLARE_OP(Conv2DReluOp, "conv2d_relu")
GAC_DECLARE_OP(MatMulBiasOp, "matmul_bias")
GAC_DECLARE_OP(MatMulBiasReluOp, "matmul_bias_relu")
GAC_DECLARE_OP(MatMulBiasGeluOp, "matmul_bias_gelu")
#undef GAC_DECLARE_OP

/// Split has one SSA result per ONNX output. Keeping the result tuple intact at
/// import allows shape inference and constant folding to see the true graph;
/// target legalization later replaces a static Split with independent slices.
class SplitOp : public mlir::Op<SplitOp, mlir::OpTrait::VariadicOperands,
                                mlir::OpTrait::VariadicResults,
                                mlir::OpTrait::ZeroRegions,
                                mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "gac.split"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
  void
  getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &) {}
  mlir::LogicalResult verify() { return inferOperation(getOperation(), false); }
};

/// Inspection-only escape hatch. Unknown semantics are NOT declared pure and
/// codegen must reject this operation rather than inventing a fallback kernel.
class UnsupportedOp
    : public mlir::Op<UnsupportedOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::VariadicResults,
                      mlir::OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "gac.unsupported"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
};

class GacDialect : public mlir::Dialect {
public:
  explicit GacDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "gac"; }
};

void registerGacDialects(mlir::DialectRegistry &registry);
} // namespace gac
#endif
