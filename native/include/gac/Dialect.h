#ifndef GAC_DIALECT_H
#define GAC_DIALECT_H

// GAC 高层图方言。
//
// 本文件只声明算子类和少量通用查询接口；每个算子的类型、形状、属性和语义
// 校验集中在 Dialect.cpp 的 inferOperation()/verify() 中。ONNX Importer 直接
// 创建这里注册的 Operation，后续图 Pass 和设备 Lowering 都基于这些 Operation
// 的 SSA 关系工作，而不是按 ONNX 名字重新解释语义。
//
// 所有算子使用 MLIR 的通用 assembly 形式，不维护第二套自定义 parser/printer，
// 因此 IR 始终可以被 gac-opt / mlir-opt 解析、打印和验证。

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace gac {

/// 推导/精化张量结果类型，但不执行算子本身。
/// 已知维度只做一致性检查，不会被静默覆盖；导入阶段允许 unranked tensor。
mlir::LogicalResult inferOperation(mlir::Operation *op,
                                   bool updateResult = true);
/// 读取整数数组属性；属性缺失时返回空数组。
llvm::SmallVector<int64_t> getI64ArrayAttr(mlir::Operation *op,
                                           llvm::StringRef name);
/// 如果 value 是常量整数张量，返回其元素；否则返回 nullopt。
std::optional<llvm::SmallVector<int64_t>>
getConstantIntegers(mlir::Value value);

/// 所有单结果、无 Region 的 GAC 图算子共用这个基类。
/// 这些算子具有 Tensor 值语义，不产生设备内存副作用，因此上游 CSE 和
/// canonicalize 可以安全地删除死代码、合并等价计算。verify() 直接复用
/// inferOperation()，保证“注册即校验”。
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

// 手写注册的 MLIR Operation 类。使用通用 assembly 是刻意选择：
// 不维护第二套自定义 parser/printer，IR 始终可被上游工具解析和打印。
#define GAC_DECLARE_OP(CLASS, NAME)                                            \
  class CLASS : public TensorOp<CLASS> {                                       \
  public:                                                                      \
    using TensorOp::TensorOp;                                                  \
    static llvm::StringRef getOperationName() { return "gac." NAME; }          \
  };

// ---- 常量与矩阵乘法 ----
GAC_DECLARE_OP(ConstantOp, "constant")
GAC_DECLARE_OP(MatMulOp, "matmul")
GAC_DECLARE_OP(GemmOp, "gemm")

// ---- 逐元素二元算术 ----
GAC_DECLARE_OP(AddOp, "add")
GAC_DECLARE_OP(SubOp, "sub")
GAC_DECLARE_OP(MulOp, "mul")
GAC_DECLARE_OP(DivOp, "div")
GAC_DECLARE_OP(PowOp, "pow")

// ---- 比较、逻辑与选择 ----
GAC_DECLARE_OP(GreaterOp, "greater")
GAC_DECLARE_OP(LessOp, "less")
GAC_DECLARE_OP(EqualOp, "equal")
GAC_DECLARE_OP(AndOp, "and")
GAC_DECLARE_OP(MaxOp, "max")
GAC_DECLARE_OP(WhereOp, "where")

// ---- 激活函数与逐元素数学 ----
GAC_DECLARE_OP(ReluOp, "relu")
GAC_DECLARE_OP(GeluOp, "gelu")
GAC_DECLARE_OP(SigmoidOp, "sigmoid")
GAC_DECLARE_OP(LogOp, "log")
GAC_DECLARE_OP(SinOp, "sin")
GAC_DECLARE_OP(CosOp, "cos")
GAC_DECLARE_OP(ExpOp, "exp")
GAC_DECLARE_OP(SqrtOp, "sqrt")
GAC_DECLARE_OP(ClampOp, "clamp")

// ---- 形状与布局变换（设备端通常退化为 view 或逐元素拷贝） ----
GAC_DECLARE_OP(IdentityOp, "identity")
GAC_DECLARE_OP(CastOp, "cast")
GAC_DECLARE_OP(ReshapeOp, "reshape")
GAC_DECLARE_OP(TransposeOp, "transpose")
GAC_DECLARE_OP(ConcatOp, "concat")
GAC_DECLARE_OP(ExpandOp, "expand")
GAC_DECLARE_OP(SliceOp, "slice")

// ---- 归约与归一化 ----
GAC_DECLARE_OP(ReduceSumOp, "reduce_sum")
GAC_DECLARE_OP(ReduceMeanOp, "reduce_mean")
GAC_DECLARE_OP(SoftmaxOp, "softmax")
GAC_DECLARE_OP(LayerNormOp, "layer_norm")

// ---- 卷积、池化与空间采样 ----
GAC_DECLARE_OP(Conv2DOp, "conv2d")
GAC_DECLARE_OP(DeformConv2DOp, "deform_conv2d")
GAC_DECLARE_OP(BatchNormOp, "batch_norm")
GAC_DECLARE_OP(MaxPool2DOp, "max_pool2d")
GAC_DECLARE_OP(GridSampleOp, "grid_sample")
GAC_DECLARE_OP(MSDeformAttnOp, "ms_deform_attn")
// 在采样点处求值坐标 Div/Add，不把中间坐标物化成完整张量。
GAC_DECLARE_OP(MSDeformAttnOffsetsOp, "ms_deform_attn_offsets")

// ---- Shape 子图（通常在导入后被常量折叠） ----
GAC_DECLARE_OP(ShapeOp, "shape")
GAC_DECLARE_OP(GatherOp, "gather")
GAC_DECLARE_OP(UnsqueezeOp, "unsqueeze")
GAC_DECLARE_OP(SqueezeOp, "squeeze")
// 带 shape 守卫的 ONNX If：仅当某个轴的长度等于 1 时删除该轴。
// 不会根据输入数据选择分支；未解析的维度可以留在图 IR 中，
// 但必须在静态设备合法化之前解析完成。
GAC_DECLARE_OP(SqueezeIfUnitDimOp, "squeeze_if_unit_dim")
GAC_DECLARE_OP(RangeOp, "range")
GAC_DECLARE_OP(ConstantOfShapeOp, "constant_of_shape")
GAC_DECLARE_OP(ResizeOp, "resize")
GAC_DECLARE_OP(TileOp, "tile")

// ---- 索引与写回 ----
GAC_DECLARE_OP(ScatterNDOp, "scatter_nd")
GAC_DECLARE_OP(NonZeroOp, "nonzero")

// ---- 图融合产生的复合算子 ----
GAC_DECLARE_OP(AddReluOp, "add_relu")
GAC_DECLARE_OP(Conv2DReluOp, "conv2d_relu")
GAC_DECLARE_OP(MatMulBiasOp, "matmul_bias")
GAC_DECLARE_OP(MatMulBiasReluOp, "matmul_bias_relu")
GAC_DECLARE_OP(MatMulBiasGeluOp, "matmul_bias_gelu")
#undef GAC_DECLARE_OP

/// Split 的每个 ONNX 输出对应一个 SSA result。导入时保留完整的多结果元组，
/// 让形状推导和常量折叠看到真实图结构；目标合法化阶段再把静态 Split
/// 替换成多个独立的 Slice。
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

/// 仅供 inspection 使用的逃生口。未知语义不会被声明为纯计算，
/// codegen 必须拒绝该 Operation，而不是发明一个回退 Kernel。
class UnsupportedOp
    : public mlir::Op<UnsupportedOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::VariadicResults,
                      mlir::OpTrait::ZeroRegions> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { return "gac.unsupported"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }
};

/// "gac" 方言。Operation 类的实际注册在 Dialect.cpp 中完成。
class GacDialect : public mlir::Dialect {
public:
  explicit GacDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "gac"; }
};

/// 把 gac 方言加入 MLIRContext 的 DialectRegistry。
void registerGacDialects(mlir::DialectRegistry &registry);
} // namespace gac
#endif
