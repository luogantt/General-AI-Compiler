#include "gac/OnnxImporter.h"
#include "gac/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "onnx.pb.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

// 本文件实现 GAC 的原生 C++ ONNX 前端。完整数据流如下：
//
//   ONNX 二进制文件
//       -> protobuf 解析得到 onnx::ModelProto / GraphProto / NodeProto
//       -> 创建 mlir::ModuleOp 和 func.func @main
//       -> 将输入、权重和每个 ONNX 节点转换为 MLIR SSA Value / gac.* Operation
//       -> func.return
//
// 这里没有执行模型，也没有先把模型转换成 Python dict。导入器的任务是把
// ONNX 的“按字符串名称连接的计算图”重建成 MLIR 的“由 SSA Value 直接连接的
// Operation 图”。后续图优化、Lowering 和 Codegen 都在这个 ModuleOp 上进行。
using namespace mlir;
namespace fs = std::filesystem;

namespace gac {
namespace {
// 导入失败时使用当前 ONNX 节点的 Location 报错，外层只负责统一捕获异常。
class Importer {
public:
  Importer(StringRef filename, MLIRContext &context, bool inspection)
      : filename(filename.str()), builder(&context), inspection(inspection),
        location(FileLineColLoc::get(&context, filename, 0, 0)) {}

  OwningOpRef<ModuleOp> run();

private:
  // ONNX 文件路径，也是 MLIR 诊断和 external_data 路径解析的基准。
  std::string filename;
  // 在同一个 MLIRContext 中创建类型、属性和 Operation。
  OpBuilder builder;
  // 是否允许用 gac.unsupported 保留尚未支持的节点，仅供导入检查。
  bool inspection;
  // 当前正在导入的 ONNX 节点位置，用于诊断回溯。
  Location location;

  // ONNX 通过 tensor 名称连接节点，MLIR 通过 SSA Value 连接 Operation。
  // 这个表就是两种表示之间的核心桥梁：
  //   "images" -> func.func 的 BlockArgument
  //   "conv_0" -> gac.conv2d 的 OpResult
  // Value 是 IR 中的“结果句柄”，这里不保存运行时张量内容。
  llvm::StringMap<Value> values;

  // 从 graph.input/value_info/output 收集到的静态类型声明。它优先用于确定
  // Operation 的结果类型；没有 value_info 时只能保留元素类型和未知形状。
  llvm::StringMap<Type> declaredTypes;

  // ONNX 每个 domain 都有独立的 opset 版本，例如标准域 "" -> 17。
  llvm::StringMap<int64_t> opsets;
  // inspection 模式下被保留下来的不支持节点数量。
  uint64_t unsupportedCount = 0;

  [[noreturn]] void fail(const Twine &message) {
    emitError(location) << message;
    throw std::runtime_error("ONNX import failed");
  }
  Type elementType(int onnxType);
  Type tensorType(const onnx::ValueInfoProto &info);
  DenseElementsAttr tensor(const onnx::TensorProto &proto);
  std::string externalBytes(const onnx::TensorProto &proto, uint64_t expected);
  NamedAttrList attributes(const onnx::NodeProto &node, std::string &reason);
  Type resultType(const onnx::NodeProto &node, StringRef output,
                  ArrayRef<Value> operands);
  void bind(StringRef name, Value value);
  void importNode(const onnx::NodeProto &node, size_t index);
  std::optional<std::pair<Value, int64_t>>
  matchShapeGuardIf(const onnx::NodeProto &node, std::string &reason);
  Operation *create(StringRef name, TypeRange results, ValueRange operands,
                    const NamedAttrList &attrs);
};

// 识别一个语义明确的 ONNX 控制流子图，而不是按模型名或节点名做特判：
// If(Equal(Gather(Shape(X), [axis]), [1]), Squeeze(X,[axis]), Identity(X)).
// 它表示“只有该轴长度为 1 时才 Squeeze，否则保持 X”。导入为一个注册的
// gac.squeeze_if_unit_dim，可将条件维度保留到 ShapeInference，而不会提前创建
// 一个在动态形状下可能非法、但运行时不会执行的 Squeeze 分支。
std::optional<std::pair<Value, int64_t>>
Importer::matchShapeGuardIf(const onnx::NodeProto &node, std::string &reason) {
  reason = "If is supported only as an exact shape-guarded Squeeze/Identity pattern";
  auto standard = [](const onnx::NodeProto &n, StringRef name) {
    return (n.domain().empty() || n.domain() == "ai.onnx") &&
           n.op_type() == name && n.overload().empty();
  };
  if (!standard(node, "If") || node.input_size() != 1 ||
      node.output_size() != 1 || node.output(0).empty() ||
      node.attribute_size() != 2)
    return std::nullopt;
  const onnx::GraphProto *thenGraph = nullptr, *elseGraph = nullptr;
  for (const auto &attribute : node.attribute()) {
    if (attribute.type() != onnx::AttributeProto::GRAPH ||
        !attribute.ref_attr_name().empty())
      return std::nullopt;
    if (attribute.name() == "then_branch" && !thenGraph)
      thenGraph = &attribute.g();
    else if (attribute.name() == "else_branch" && !elseGraph)
      elseGraph = &attribute.g();
    else
      return std::nullopt;
  }
  auto simpleGraph = [](const onnx::GraphProto *graph, int nodes) {
    return graph && graph->input_size() == 0 && graph->output_size() == 1 &&
           graph->initializer_size() == 0 && graph->sparse_initializer_size() == 0 &&
           graph->node_size() == nodes;
  };
  if (!simpleGraph(thenGraph, 2) || !simpleGraph(elseGraph, 1))
    return std::nullopt;
  const auto &constant = thenGraph->node(0);
  const auto &squeeze = thenGraph->node(1);
  const auto &identity = elseGraph->node(0);
  if (!standard(constant, "Constant") || constant.input_size() != 0 ||
      constant.output_size() != 1 || constant.output(0).empty() ||
      constant.attribute_size() != 1 ||
      constant.attribute(0).name() != "value" ||
      constant.attribute(0).type() != onnx::AttributeProto::TENSOR ||
      !constant.attribute(0).ref_attr_name().empty() ||
      !standard(squeeze, "Squeeze") || squeeze.input_size() != 2 ||
      squeeze.output_size() != 1 || squeeze.output(0).empty() ||
      squeeze.attribute_size() != 0 ||
      squeeze.input(1) != constant.output(0) ||
      squeeze.input(0) == constant.output(0) ||
      squeeze.output(0) == constant.output(0) ||
      squeeze.output(0) == squeeze.input(0) ||
      thenGraph->output(0).name() != squeeze.output(0) ||
      !standard(identity, "Identity") || identity.input_size() != 1 ||
      identity.output_size() != 1 || identity.output(0).empty() ||
      identity.attribute_size() != 0 ||
      identity.output(0) == identity.input(0) ||
      identity.input(0) != squeeze.input(0) ||
      elseGraph->output(0).name() != identity.output(0))
    return std::nullopt;
  const auto &axisTensor = constant.attribute(0).t();
  if (axisTensor.data_type() != onnx::TensorProto::INT64 ||
      axisTensor.dims_size() != 1 || axisTensor.dims(0) != 1 ||
      axisTensor.data_location() == onnx::TensorProto::EXTERNAL)
    return std::nullopt;
  DenseElementsAttr axisValue = tensor(axisTensor);
  int64_t axis = (*axisValue.getValues<APInt>().begin()).getSExtValue();
  auto found = values.find(squeeze.input(0));
  auto condition = values.find(node.input(0));
  if (found == values.end() || condition == values.end())
    return std::nullopt;
  Value input = found->second;
  auto inputType = dyn_cast<TensorType>(input.getType());
  if (!inputType)
    return std::nullopt;
  for (const auto *graph : {thenGraph, elseGraph}) {
    const auto &output = graph->output(0);
    if (!output.has_type() || !output.type().has_tensor_type() ||
        elementType(output.type().tensor_type().elem_type()) != inputType.getElementType())
      return std::nullopt;
  }

  auto exactOp = [&](Operation *op, StringRef name,
                     ArrayRef<StringRef> allowed) {
    if (!op || op->getName().getStringRef() != name)
      return false;
    auto domain = op->getAttrOfType<StringAttr>("gac.source_domain");
    auto version = op->getAttrOfType<IntegerAttr>("gac.opset");
    if (!domain || !domain.getValue().empty() || !version ||
        version.getInt() < 13 || version.getInt() > 18)
      return false;
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef key = attr.getName().getValue();
      if (!key.starts_with("gac.") && !llvm::is_contained(allowed, key))
        return false;
    }
    return true;
  };
  auto integerConstant = [&](Value value, int64_t expected) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    auto data = getConstantIntegers(value);
    return type && type.hasStaticShape() && type.getRank() <= 1 &&
           type.getNumElements() == 1 && type.getElementType().isSignlessInteger(64) &&
           data && data->size() == 1 && data->front() == expected;
  };
  auto equal = condition->second.getDefiningOp();
  if (!exactOp(equal, "gac.equal", {}) || equal->getNumOperands() != 2)
    return std::nullopt;
  auto conditionType = dyn_cast<TensorType>(condition->second.getType());
  if (!conditionType || !conditionType.getElementType().isInteger(1))
    return std::nullopt;
  Operation *gather = nullptr;
  for (int side = 0; side < 2; ++side)
    if (integerConstant(equal->getOperand(1 - side), 1))
      gather = equal->getOperand(side).getDefiningOp();
  if (!exactOp(gather, "gac.gather", {"axis"}) ||
      gather->getNumOperands() != 2 ||
      !integerConstant(gather->getOperand(1), axis))
    return std::nullopt;
  if (Attribute attr = gather->getAttr("axis")) {
    auto integer = dyn_cast<IntegerAttr>(attr);
    if (!integer || integer.getInt() != 0)
      return std::nullopt;
  }
  auto shape = gather->getOperand(0).getDefiningOp();
  if (!exactOp(shape, "gac.shape", {}) || shape->getNumOperands() != 1 ||
      shape->getOperand(0) != input)
    return std::nullopt;
  reason.clear();
  return std::make_pair(input, axis);
}

Type Importer::elementType(int type) {
  // ONNX TensorProto 的 dtype 枚举和 MLIR Type 是两套类型系统。这里仅完成
  // 标量元素类型转换，张量的 rank/shape 由 tensorType() 另外处理。
  auto *ctx = builder.getContext();
  switch (type) {
  case onnx::TensorProto::FLOAT:
    return builder.getF32Type();
  case onnx::TensorProto::DOUBLE:
    return builder.getF64Type();
  case onnx::TensorProto::FLOAT16:
    return builder.getF16Type();
  case onnx::TensorProto::BFLOAT16:
    return builder.getBF16Type();
  case onnx::TensorProto::BOOL:
    return builder.getI1Type();
  case onnx::TensorProto::INT8:
    return builder.getIntegerType(8);
  case onnx::TensorProto::INT16:
    return builder.getIntegerType(16);
  case onnx::TensorProto::INT32:
    return builder.getI32Type();
  case onnx::TensorProto::INT64:
    return builder.getI64Type();
  case onnx::TensorProto::UINT8:
    return IntegerType::get(ctx, 8, IntegerType::Unsigned);
  case onnx::TensorProto::UINT16:
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  case onnx::TensorProto::UINT32:
    return IntegerType::get(ctx, 32, IntegerType::Unsigned);
  case onnx::TensorProto::UINT64:
    return IntegerType::get(ctx, 64, IntegerType::Unsigned);
  default:
    return {};
  }
}

Type Importer::tensorType(const onnx::ValueInfoProto &info) {
  // ValueInfoProto 描述一个图输入、图输出或中间值的类型。
  // - 有完整 shape：生成 RankedTensorType。
  // - 无 shape：生成 UnrankedTensorType。
  // - 符号维度或未知维度：该维写成 ShapedType::kDynamic。
  if (!info.has_type() || !info.type().has_tensor_type())
    fail("value '" + info.name() + "' must have a tensor type");
  const auto &type = info.type().tensor_type();
  Type element = elementType(type.elem_type());
  if (!element)
    fail("unsupported ONNX element type on value '" + info.name() + "'");
  if (!type.has_shape())
    return UnrankedTensorType::get(element);
  SmallVector<int64_t> shape;
  for (const auto &dim : type.shape().dim()) {
    if (dim.has_dim_value() && dim.dim_value() < 0)
      fail("negative concrete tensor dimension on '" + info.name() + "'");
    shape.push_back(dim.has_dim_value() ? dim.dim_value()
                                        : ShapedType::kDynamic);
  }
  return RankedTensorType::get(shape, element);
}

std::string Importer::externalBytes(const onnx::TensorProto &proto,
                                    uint64_t expected) {
  // 大权重可能不嵌在 .onnx 文件中，而由 external_data 指向外部二进制文件。
  // 除了读取 offset/length，这里还限制路径必须留在模型目录内，防止 ../ 或
  // 符号链接逃逸。返回值仍是 ONNX 规定的小端原始字节串。
  llvm::StringMap<std::string> entries;
  for (const auto &entry : proto.external_data()) {
    if (!entries.try_emplace(entry.key(), entry.value()).second)
      fail("duplicate external_data key: " + entry.key());
  }
  auto found = entries.find("location");
  if (found == entries.end() || found->second.empty())
    fail("external tensor has no location: " + proto.name());
  // Treat both separators as separators on every host. Reject drive-qualified,
  // absolute, and parent-traversal paths before resolving symlinks.
  std::string relative = found->second;
  std::replace(relative.begin(), relative.end(), '\\', '/');
  fs::path relativePath(relative);
  if (relativePath.is_absolute() || relative.find(':') != std::string::npos)
    fail("external_data location must be a relative path");
  for (const auto &part : relativePath)
    if (part == "..")
      fail("external_data parent traversal is forbidden");
  std::error_code error;
  fs::path root =
      fs::canonical(fs::absolute(fs::path(filename)).parent_path(), error);
  if (error)
    fail("cannot resolve ONNX model directory");
  fs::path target = fs::canonical(root / relativePath, error);
  if (error)
    fail("external tensor file does not exist: " + relative);
  auto rootIt = root.begin(), targetIt = target.begin();
  for (; rootIt != root.end(); ++rootIt, ++targetIt)
    if (targetIt == target.end() || *rootIt != *targetIt)
      fail("external_data symlink escapes the model directory");
  if (!fs::is_regular_file(target, error) || error)
    fail("external_data location is not a regular file");
  auto integer = [&](StringRef key, uint64_t defaultValue) {
    auto item = entries.find(key);
    if (item == entries.end())
      return defaultValue;
    uint64_t value;
    if (StringRef(item->second).getAsInteger(10, value))
      fail("invalid unsigned external_data " + key);
    return value;
  };
  uint64_t offset = integer("offset", 0);
  uint64_t length = integer("length", expected);
  if (length != expected)
    fail("external tensor length does not match its shape and dtype");
  uint64_t size = fs::file_size(target, error);
  if (error || offset > size || length > size - offset)
    fail("external tensor byte range exceeds file size");
  if (offset > uint64_t(std::numeric_limits<std::streamoff>::max()) ||
      length > uint64_t(std::numeric_limits<std::streamsize>::max()))
    fail("external tensor byte range cannot be represented on this host");
  std::ifstream input(target, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset));
  std::string bytes(static_cast<size_t>(length), '\0');
  if (length && !input.read(bytes.data(), static_cast<std::streamsize>(length)))
    fail("failed to read external tensor data");
  return bytes;
}

DenseElementsAttr Importer::tensor(const onnx::TensorProto &proto) {
  // 将 ONNX TensorProto 转为 MLIR DenseElementsAttr。DenseElementsAttr 同时保存
  // RankedTensorType 和常量数据，随后成为 gac.constant 的 value 属性。
  // 权重可能来自 raw_data、external_data，也可能来自 protobuf repeated 字段。
  Type element = elementType(proto.data_type());
  if (!element)
    fail("unsupported initializer dtype for '" + proto.name() + "'");
  SmallVector<int64_t> shape;
  uint64_t count = 1;
  for (int64_t dim : proto.dims()) {
    if (dim < 0 || (dim != 0 && count > uint64_t(INT64_MAX) / uint64_t(dim)))
      fail("invalid or overflowing initializer dimensions");
    shape.push_back(dim);
    count *= dim;
  }
  auto type = RankedTensorType::get(shape, element);
  unsigned width = element.getIntOrFloatBitWidth();
  uint64_t byteWidth = (width + 7) / 8;
  if (count > std::numeric_limits<size_t>::max() / byteWidth)
    fail("initializer exceeds host address space");
  uint64_t expected = count * byteWidth;
  std::string raw;
  bool external = proto.data_location() == onnx::TensorProto::EXTERNAL;
  if (external) {
    if (!proto.raw_data().empty())
      fail("external initializer also contains raw_data");
    raw = externalBytes(proto, expected);
  } else if (!proto.external_data().empty()) {
    fail("external_data supplied without EXTERNAL data_location");
  } else if (!proto.raw_data().empty()) {
    raw = proto.raw_data();
  } else {
    // ONNX repeated fields are unpacked into exactly the same little-endian
    // representation as raw_data. Float16/BFloat16 store bit patterns in int32.
    raw.reserve(static_cast<size_t>(expected));
    auto append = [&](uint64_t value) {
      for (unsigned b = 0; b < byteWidth; ++b)
        raw.push_back(char((value >> (8 * b)) & 0xff));
    };
    switch (proto.data_type()) {
    case onnx::TensorProto::FLOAT:
      for (float value : proto.float_data()) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        append(bits);
      }
      break;
    case onnx::TensorProto::DOUBLE:
      for (double value : proto.double_data()) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        append(bits);
      }
      break;
    case onnx::TensorProto::INT64:
      for (int64_t value : proto.int64_data())
        append(uint64_t(value));
      break;
    case onnx::TensorProto::UINT32:
    case onnx::TensorProto::UINT64:
      for (uint64_t value : proto.uint64_data())
        append(value);
      break;
    default:
      for (int32_t value : proto.int32_data()) {
        if (proto.data_type() == onnx::TensorProto::BOOL && value != 0 &&
            value != 1)
          fail("BOOL initializer contains a value other than zero or one");
        append(uint64_t(value));
      }
      break;
    }
  }
  if (raw.size() != expected)
    fail("initializer byte count disagrees with shape: " + proto.name());
  if (width == 1) {
    SmallVector<APInt> bits;
    bits.reserve(static_cast<size_t>(count));
    for (unsigned char value : raw) {
      if (value > 1)
        fail("BOOL raw_data contains a non-boolean byte");
      bits.emplace_back(1, value);
    }
    return DenseElementsAttr::get(type, bits);
  }
  // MLIR's dense raw storage is native-endian, whereas ONNX is little-endian.
  if (llvm::endianness::native == llvm::endianness::big)
    for (size_t i = 0; i < raw.size(); i += byteWidth)
      std::reverse(raw.begin() + i, raw.begin() + i + byteWidth);
  return DenseElementsAttr::getFromRawBuffer(
      type, ArrayRef<char>(raw.data(), raw.size()));
}

Operation *Importer::create(StringRef name, TypeRange results,
                            ValueRange operands, const NamedAttrList &attrs) {
  // 这是 ONNX NodeProto 真正变成 MLIR Operation 的位置。
  // OperationState 描述一个操作的五个核心部分：名称、结果类型、输入 SSA Value、
  // 属性和 Location；OpBuilder::create() 才把它插入当前 func.func 的 Block。
  // 因此转换不是简单字符串改名，而是重新构造一个完整且可验证的 MLIR 节点。
  OperationState state(location, name);
  state.addTypes(results);
  state.addOperands(operands);
  state.addAttributes(attrs);
  return builder.create(state);
}

void Importer::bind(StringRef name, Value value) {
  // 注册生产者结果。后面的 ONNX 节点只需用输入名称查表，就能取得前面
  // Operation 的 SSA Value，从而在 MLIR 中建立 use-def 边。
  if (name.empty() || !values.try_emplace(name, value).second)
    fail("duplicate or empty ONNX SSA value name: " + name);
}

NamedAttrList Importer::attributes(const onnx::NodeProto &node,
                                   std::string &reason) {
  // 把 ONNX AttributeProto 转成 MLIR Attribute。这里处理标量、数组和稠密
  // Tensor 属性；Graph、SparseTensor、TypeProto 等尚未实现的属性不会被静默
  // 丢弃，而是通过 reason 进入严格报错或 inspection 占位路径。
  //
  //   ONNX FLOAT/INT/STRING -> MLIR FloatAttr/IntegerAttr/StringAttr
  //   ONNX TENSOR           -> MLIR DenseElementsAttr
  //   ONNX INTS             -> MLIR DenseI64ArrayAttr
  //   ONNX FLOATS/STRINGS   -> MLIR ArrayAttr
  //
  // Attribute 是编译期元数据，例如 Conv 的 strides/pads；它不是 SSA operand。
  NamedAttrList result;
  llvm::StringSet<> seen;
  for (const auto &attr : node.attribute()) {
    if (!seen.insert(attr.name()).second)
      fail("duplicate attribute: " + attr.name());
    if (!attr.ref_attr_name().empty()) {
      reason = "ONNX function attribute references require function expansion";
      continue;
    }
    Attribute value;
    switch (attr.type()) {
    case onnx::AttributeProto::FLOAT:
      value = builder.getF32FloatAttr(attr.f());
      break;
    case onnx::AttributeProto::INT:
      value = builder.getI64IntegerAttr(attr.i());
      break;
    case onnx::AttributeProto::STRING:
      value = builder.getStringAttr(attr.s());
      break;
    case onnx::AttributeProto::TENSOR:
      value = tensor(attr.t());
      break;
    case onnx::AttributeProto::INTS: {
      SmallVector<int64_t> entries(attr.ints().begin(), attr.ints().end());
      value = builder.getDenseI64ArrayAttr(entries);
      break;
    }
    case onnx::AttributeProto::FLOATS: {
      SmallVector<Attribute> entries;
      for (float item : attr.floats())
        entries.push_back(builder.getF32FloatAttr(item));
      value = builder.getArrayAttr(entries);
      break;
    }
    case onnx::AttributeProto::STRINGS: {
      SmallVector<Attribute> entries;
      for (const auto &item : attr.strings())
        entries.push_back(builder.getStringAttr(item));
      value = builder.getArrayAttr(entries);
      break;
    }
    default:
      reason = "graph/sparse/type/repeated-tensor attributes require "
               "additional import support";
      continue;
    }
    result.set(attr.name(), value);
  }
  return result;
}

static StringRef operationName(StringRef type) {
  // 标准 ONNX op_type 到 GAC 方言操作名的语义映射表。
  // 这一步只决定“要创建哪一种 Operation”；输入/输出、类型、属性、版本规则和
  // SSA 连接仍由 importNode() 完成，所以它不是把文本中的名字直接替换一下。
  // 例如 Conv -> gac.conv2d、Clip -> gac.clamp、MaxPool -> gac.max_pool2d。
  return llvm::StringSwitch<StringRef>(type)
      .Case("Constant", "gac.constant")
      .Case("Conv", "gac.conv2d")
      .Case("Relu", "gac.relu")
      .Case("MatMul", "gac.matmul")
      .Case("Gemm", "gac.gemm")
      .Case("Add", "gac.add")
      .Case("Sub", "gac.sub")
      .Case("Mul", "gac.mul")
      .Case("Div", "gac.div")
      .Case("Pow", "gac.pow")
      .Case("Greater", "gac.greater")
      .Case("Less", "gac.less")
      .Case("Where", "gac.where")
      .Case("Equal", "gac.equal")
      .Case("And", "gac.and")
      .Case("Max", "gac.max")
      .Case("Sigmoid", "gac.sigmoid")
      .Case("Log", "gac.log")
      .Case("Sin", "gac.sin")
      .Case("Cos", "gac.cos")
      .Case("Exp", "gac.exp")
      .Case("Sqrt", "gac.sqrt")
      .Case("Clip", "gac.clamp")
      .Case("Identity", "gac.identity")
      .Case("Softmax", "gac.softmax")
      .Case("Reshape", "gac.reshape")
      .Case("Transpose", "gac.transpose")
      .Case("Concat", "gac.concat")
      .Case("Slice", "gac.slice")
      .Case("Expand", "gac.expand")
      .Case("Cast", "gac.cast")
      .Case("ReduceSum", "gac.reduce_sum")
      .Case("ReduceMean", "gac.reduce_mean")
      .Case("LayerNormalization", "gac.layer_norm")
      .Case("BatchNormalization", "gac.batch_norm")
      .Case("MaxPool", "gac.max_pool2d")
      .Case("GridSample", "gac.grid_sample")
      .Case("Shape", "gac.shape")
      .Case("Gather", "gac.gather")
      .Case("Unsqueeze", "gac.unsqueeze")
      .Case("Squeeze", "gac.squeeze")
      .Case("Range", "gac.range")
      .Case("ConstantOfShape", "gac.constant_of_shape")
      .Case("Resize", "gac.resize")
      .Case("Tile", "gac.tile")
      .Case("ScatterND", "gac.scatter_nd")
      .Case("NonZero", "gac.nonzero")
      .Case("Split", "gac.split")
      .Default("");
}

static StringRef customOperationName(StringRef domain, StringRef type) {
  // 非标准 domain 必须有显式导入契约，避免把同名但语义不同的自定义算子
  // 错当成标准 ONNX 算子。目前登记 MMCV DCNv2 和 GAC MSDeformAttn。
  if (domain == "mmcv" && type == "MMCVModulatedDeformConv2d")
    return "gac.deform_conv2d";
  if (domain == "gac" && type == "MSDeformAttn")
    return "gac.ms_deform_attn";
  return {};
}

Type Importer::resultType(const onnx::NodeProto &node, StringRef output,
                          ArrayRef<Value> operands) {
  // 结果类型的优先级：
  //   1. ONNX value_info 中的完整声明；
  //   2. 按算子 schema 推断元素类型，并暂时使用 UnrankedTensorType；
  //   3. 无法确定时返回 NoneType，importNode() 会给出明确诊断。
  // 精确结果 shape 会由后续 GAC ShapeInference Pass 继续传播和细化。
  auto declared = declaredTypes.find(output);
  if (declared != declaredTypes.end())
    return declared->second;
  StringRef name = node.op_type();
  Type element;
  if (name == "Shape" || name == "NonZero" || name == "ArgMax" ||
      name == "ArgMin")
    element = builder.getI64Type();
  else if (name == "Greater" || name == "Less" || name == "Equal" ||
           name == "GreaterOrEqual" || name == "LessOrEqual" || name == "And" ||
           name == "Or" || name == "Not" || name == "Xor")
    element = builder.getI1Type();
  else if (name == "Cast") {
    for (const auto &attr : node.attribute())
      if (attr.name() == "to")
        element = elementType(attr.i());
  } else if (name == "ConstantOfShape") {
    element = builder.getF32Type();
    for (const auto &attr : node.attribute())
      if (attr.name() == "value" && attr.type() == onnx::AttributeProto::TENSOR)
        element = elementType(attr.t().data_type());
  } else if (!customOperationName(node.domain(), name).empty() ||
             !operationName(name).empty() || name == "Split" ||
             name == "ScatterND" || name == "Tile" || name == "Max" || name == "If") {
    // 这些算子保持数据输入的元素类型。Where 的第 0 个输入是 bool 条件，
    // 因此它的结果元素类型取自第 1 个输入；其余算子通常取第 0 个输入。
    size_t index = name == "Where" ? 1 : 0;
    if (operands.size() > index)
      if (auto shaped = dyn_cast<ShapedType>(operands[index].getType()))
        element = shaped.getElementType();
  }
  return element ? Type(UnrankedTensorType::get(element))
                 : Type(builder.getNoneType());
}

void Importer::importNode(const onnx::NodeProto &node, size_t index) {
  // 第 1 步：确定稳定的源节点名称和 MLIR Location，使后续错误能定位回 ONNX。
  std::string sourceName = node.name().empty()
                               ? node.op_type() + "#" + std::to_string(index)
                               : node.name();
  location = NameLoc::get(
      builder.getStringAttr(sourceName),
      FileLineColLoc::get(builder.getContext(), filename, index + 1, 0));
  std::string domain = node.domain() == "ai.onnx" ? "" : node.domain();
  auto version = opsets.find(domain);
  if (version == opsets.end())
    fail("node domain has no opset import: " + domain);

  // 第 2 步：把 NodeProto.input() 的字符串名称解析为 MLIR SSA operands。
  // ONNX 用空字符串表示可选输入缺席，因此不能简单按数组下标压入 operands；
  // inputPositions 保存压缩前的位置，供验证、Lowering 或 Runtime 恢复原语义。
  SmallVector<Value> operands;
  SmallVector<int64_t> inputPositions;
  for (int i = 0; i < node.input_size(); ++i) {
    if (node.input(i).empty())
      continue;
    auto found = values.find(node.input(i));
    if (found == values.end())
      fail("input '" + node.input(i) +
           "' has no producer; graph must be topologically ordered");
    // ONNX Resize accepts either an absent name or an empty optional tensor.
    // Normalize only these schema-defined placeholders, before the runtime's
    // positive-size buffer contract sees them. Keep original input positions.
    if (domain.empty() && node.op_type() == "Resize" && i >= 1 && i <= 3) {
      auto type = dyn_cast<RankedTensorType>(found->second.getType());
      if (type && type.hasStaticShape() && type.getNumElements() == 0)
        continue;
    }
    operands.push_back(found->second);
    inputPositions.push_back(i);
  }
  std::string reason;
  bool shapeGuardIf = domain.empty() && node.op_type() == "If";
  NamedAttrList attrs;

  // 第 3 步：解析属性。受支持的特殊 If 先折叠为一个高层 GAC 语义操作；
  // 普通节点则逐项转换 AttributeProto。
  if (shapeGuardIf) {
    if (auto matched = matchShapeGuardIf(node, reason)) {
      operands.assign({matched->first});
      inputPositions.assign({0});
      attrs.set("axis", builder.getI64IntegerAttr(matched->second));
      attrs.set("gac.if_condition_source", builder.getStringAttr(node.input(0)));
    }
  } else {
    attrs = attributes(node, reason);
  }
  StringRef target = domain.empty()
                         ? operationName(node.op_type())
                         : customOperationName(domain, node.op_type());
  if (shapeGuardIf)
    target = "gac.squeeze_if_unit_dim";

  // 第 4 步：验证导入契约。除了检查“有没有名字映射”，还检查 domain/opset、
  // 自定义属性、算子输入输出个数、推理模式以及当前后端支持的 rank 等条件。
  if (!domain.empty() && target.empty())
    reason = "custom ONNX domain has no registered import contract";
  else if (!domain.empty() && version->second != 1)
    reason = "registered custom ONNX operators require domain opset 1";
  else if (target.empty())
    reason = "ONNX operator has no registered GAC import mapping";
  else if (domain.empty() && (version->second < 13 || version->second > 18))
    reason = "importer supports standard ONNX opsets 13 through 18 only";
  if (!domain.empty() && !target.empty()) {
    for (const auto &attr : node.attribute()) {
      StringRef key = attr.name();
      bool known =
          target == "gac.deform_conv2d"
              ? (key == "stride" || key == "padding" || key == "dilation" ||
                 key == "groups" || key == "deform_groups")
              : (key == "version" || key == "mode" || key == "padding_mode" ||
                 key == "align_corners" || key == "im2col_step");
      if (!known)
        reason = "unrecognized custom operator attribute: " + attr.name();
    }
  }
  if (node.op_type() == "GridSample" && version->second < 16)
    reason = "GridSample requires ONNX opset 16 or later";
  if (node.op_type() == "LayerNormalization" && version->second < 17)
    reason = "LayerNormalization requires ONNX opset 17 or later";
  if (node.op_type() == "Softmax" && version->second < 13)
    reason =
        "pre-opset-13 Softmax flattening semantics need an explicit conversion";
  if (node.op_type() == "Max" && operands.size() != 2)
    reason = "registered Max currently supports exactly two operands";
  for (Value operand : operands)
    if (!isa<TensorType>(operand.getType()))
      reason = "input element type is unavailable from an unsupported producer";
  if (auto training = attrs.get("training_mode"))
    if (auto integer = dyn_cast<IntegerAttr>(training);
        integer && integer.getInt() != 0)
      reason = "training-mode operators are outside this inference importer";
  SmallVector<StringRef> outputs;
  SmallVector<Attribute> outputNames;
  for (const auto &output : node.output()) {
    if (output.empty())
      continue;
    outputs.push_back(output);
    outputNames.push_back(builder.getStringAttr(output));
  }
  if (outputs.size() != 1 && !target.empty() && target != "gac.split")
    reason = "registered GAC operator currently requires one nonempty result";
  if (target == "gac.split" &&
      (outputs.empty() || outputs.size() != size_t(node.output_size())))
    reason = "Split requires nonempty names for every variadic output";
  if ((node.op_type() == "Conv" || node.op_type() == "MaxPool") &&
      !operands.empty()) {
    auto rank = dyn_cast<RankedTensorType>(operands.front().getType());
    if (rank && rank.getRank() != 4)
      reason = "only 2D NCHW convolution/pooling is supported";
    if (auto kernel =
            dyn_cast_or_null<DenseI64ArrayAttr>(attrs.get("kernel_shape")))
      if (kernel.size() != 2)
        reason = "only 2D convolution/pooling is supported";
  }

  // 第 5 步：保留 ONNX 来源信息。这些 gac.source_* 属性不参与数学计算，
  // 但可用于 IR 检查、错误报告和把优化后的节点追溯到原模型。
  attrs.set("gac.source_name", builder.getStringAttr(sourceName));
  attrs.set("gac.source_domain", builder.getStringAttr(domain));
  attrs.set("gac.source_op", builder.getStringAttr(node.op_type()));
  attrs.set("gac.opset", builder.getI64IntegerAttr(version->second));
  attrs.set("gac.input_positions",
            builder.getDenseI64ArrayAttr(inputPositions));
  attrs.set("gac.output_names", builder.getArrayAttr(outputNames));

  // 第 6 步：构造结果类型。Constant 是特例：先把各种 ONNX 常量编码统一成
  // DenseElementsAttr，再直接使用常量自身的 RankedTensorType。
  SmallVector<Type> resultTypes;
  if (target == "gac.constant" && reason.empty()) {
    auto value = dyn_cast_or_null<DenseElementsAttr>(attrs.get("value"));
    // Scalar/vector Constant alternatives are normalized into a tensor
    // constant.
    if (!value) {
      onnx::TensorProto proto;
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "value_float") {
          proto.set_data_type(onnx::TensorProto::FLOAT);
          proto.add_float_data(attr.f());
        } else if (attr.name() == "value_int") {
          proto.set_data_type(onnx::TensorProto::INT64);
          proto.add_int64_data(attr.i());
        } else if (attr.name() == "value_floats") {
          proto.set_data_type(onnx::TensorProto::FLOAT);
          proto.add_dims(attr.floats_size());
          for (float item : attr.floats())
            proto.add_float_data(item);
        } else if (attr.name() == "value_ints") {
          proto.set_data_type(onnx::TensorProto::INT64);
          proto.add_dims(attr.ints_size());
          for (int64_t item : attr.ints())
            proto.add_int64_data(item);
        }
      }
      if (proto.data_type())
        value = tensor(proto);
    }
    if (value) {
      attrs.set("value", value);
      resultTypes.push_back(value.getType());
    } else
      reason = "Constant requires a supported numeric tensor or scalar value";
  }
  if (resultTypes.empty())
    for (StringRef output : outputs)
      resultTypes.push_back(resultType(node, output, operands));
  for (Type type : resultTypes)
    if (isa<NoneType>(type) && reason.empty())
      reason = "result element type is unknown; provide ONNX value_info or an "
               "explicit type rule";

  // strict 模式下任何语义缺口都会停止编译；inspection 模式则生成
  // gac.unsupported，并把原始 NodeProto 序列化保存，便于完整查看未知模型。
  // gac.unsupported 只用于检查，不能进入后端执行编译。
  if (!reason.empty()) {
    if (!inspection)
      fail(reason + ": " + node.op_type());
    target = "gac.unsupported";
    attrs.set("gac.unsupported_reason", builder.getStringAttr(reason));
    // Preserve all original fields (including nested graphs) for inspection.
    attrs.set("gac.onnx_node_proto",
              builder.getStringAttr(node.SerializeAsString()));
    ++unsupportedCount;
  }

  // 第 7 步：创建 Operation，并将每个 ONNX 输出名称绑定到对应 OpResult。
  // 至此生产者和后续消费者之间已经形成真正的 MLIR SSA use-def 链。
  // 例如 ONNX MatMul(A, B) -> C 会成为：
  //   values["A"], values["B"] -> %c = "gac.matmul"(...) -> values["C"] = %c
  Operation *op = create(target, resultTypes, operands, attrs);
  for (size_t i = 0; i < outputs.size(); ++i)
    bind(outputs[i], op->getResult(i));
}

OwningOpRef<ModuleOp> Importer::run() {
  // 1. 解析 protobuf。ModelProto 是整个 ONNX 文件的根对象，内部包含 GraphProto、
  // opset、元数据以及可选的本地函数定义。
  std::ifstream input(filename, std::ios::binary);
  if (!input)
    fail("cannot open ONNX model: " + filename);
  onnx::ModelProto model;
  if (!model.ParseFromIstream(&input) || !model.has_graph())
    fail("invalid ONNX ModelProto or missing graph");
  if (model.ir_version() <= 0)
    fail("ONNX model has no valid IR version");
  if (model.training_info_size())
    fail("training_info is not supported by the inference importer");
  if (model.functions_size() && !inspection)
    fail("local ONNX functions require expansion before strict import");

  // 2. 建立 domain -> opset 版本表。后续每个节点都必须在这里找到自己的版本。
  for (const auto &opset : model.opset_import()) {
    std::string domain = opset.domain() == "ai.onnx" ? "" : opset.domain();
    if (opset.version() <= 0 ||
        !opsets.try_emplace(domain, opset.version()).second)
      fail("invalid or duplicated ONNX opset domain: " + domain);
  }
  const auto &graph = model.graph();
  if (graph.sparse_initializer_size())
    fail("sparse initializers are not supported");

  // 3. 收集图输入、中间值和图输出的类型声明。ONNX 不要求每个中间值都有
  // value_info，所以缺失的 shape 会在导入后由类型规则和 ShapeInference 补全。
  auto recordType = [&](const onnx::ValueInfoProto &info) {
    Type type = tensorType(info);
    auto inserted = declaredTypes.try_emplace(info.name(), type);
    if (!inserted.second && inserted.first->second != type)
      fail("conflicting ONNX type declarations for '" + info.name() + "'");
  };
  for (const auto &info : graph.input())
    recordType(info);
  for (const auto &info : graph.value_info())
    recordType(info);
  for (const auto &info : graph.output())
    recordType(info);

  // 4. 区分运行时输入和 initializer 权重，并由 ONNX 图签名构造 @main 的 ABI。
  llvm::StringSet<> initializers;
  for (const auto &proto : graph.initializer())
    if (proto.name().empty() || !initializers.insert(proto.name()).second)
      fail("duplicate or empty initializer name");
  SmallVector<Type> inputs, results;
  SmallVector<Attribute> inputNames, outputNames;
  for (const auto &info : graph.input()) {
    // An initializer that is also a graph input is overridable in ONNX.
    // Treating it as immutable would silently change the model's public
    // function ABI.
    if (initializers.contains(info.name()))
      fail("overridable initializer/input requires an explicit "
           "specialization: " +
           info.name());
    inputs.push_back(declaredTypes.lookup(info.name()));
    inputNames.push_back(builder.getStringAttr(info.name()));
  }
  for (const auto &info : graph.output()) {
    results.push_back(declaredTypes.lookup(info.name()));
    outputNames.push_back(builder.getStringAttr(info.name()));
  }

  // 5. 创建顶层 ModuleOp 和 func.func @main：
  //
  //   module {
  //     func.func @main(%images: tensor<...>) -> tensor<...> { ... }
  //   }
  //
  // Module 属性记录模型来源；函数参数和返回类型构成可被后端调用的公开 ABI。
  auto module = OwningOpRef<ModuleOp>(ModuleOp::create(location));
  (*module)->setAttr("gac.source_format", builder.getStringAttr("onnx"));
  (*module)->setAttr("gac.onnx_ir_version",
                     builder.getI64IntegerAttr(model.ir_version()));
  (*module)->setAttr("gac.source_path", builder.getStringAttr(filename));
  if (model.functions_size()) {
    SmallVector<Attribute> functions;
    for (const auto &function : model.functions())
      functions.push_back(builder.getStringAttr(function.SerializeAsString()));
    (*module)->setAttr("gac.onnx_functions", builder.getArrayAttr(functions));
  }
  auto function = func::FuncOp::create(
      location, "main", builder.getFunctionType(inputs, results));
  function->setAttr("gac.input_names", builder.getArrayAttr(inputNames));
  function->setAttr("gac.output_names", builder.getArrayAttr(outputNames));
  module->push_back(function);
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  // 6. 函数实参本身就是 SSA Value。先把图输入名绑定到 BlockArgument，再把
  // initializer 物化成 gac.constant，使所有后续 Node 都能统一通过 values 查找。
  for (size_t i = 0; i < inputs.size(); ++i)
    bind(graph.input(i).name(), entry->getArgument(i));
  for (const auto &proto : graph.initializer()) {
    location = NameLoc::get(builder.getStringAttr(proto.name()));
    DenseElementsAttr value = tensor(proto);
    if (auto declared = declaredTypes.lookup(proto.name());
        declared && declared != value.getType())
      fail("initializer type disagrees with its ONNX value_info: " +
           proto.name());
    NamedAttrList attrs;
    attrs.set("value", value);
    attrs.set("gac.source_name", builder.getStringAttr(proto.name()));
    attrs.set("gac.output_names",
              builder.getArrayAttr({builder.getStringAttr(proto.name())}));
    bind(proto.name(),
         create("gac.constant", value.getType(), {}, attrs)->getResult(0));
  }

  // 7. ONNX GraphProto.node 应按拓扑顺序排列。逐节点导入后，每个输出立即进入
  // values 表，因此其消费者可以取得同一个 SSA Value，而无需复制中间张量。
  for (int i = 0; i < graph.node_size(); ++i)
    importNode(graph.node(i), i);

  // 8. 解析图输出名称，生成 func.return，并运行 MLIR verifier 检查结果数量、
  // 类型、Operation 约束及 SSA 结构。verifier 不执行模型，也不检查数值精度；
  // 只有结构验证通过的 ModuleOp 才交给后续 Pass。
  SmallVector<Value> returned;
  for (const auto &info : graph.output()) {
    auto found = values.find(info.name());
    if (found == values.end())
      fail("graph output has no producer: " + info.name());
    if (found->second.getType() != declaredTypes.lookup(info.name()))
      fail("graph output type disagrees with its producer: " + info.name());
    returned.push_back(found->second);
  }
  builder.create<func::ReturnOp>(location, returned);
  (*module)->setAttr("gac.unsupported_count",
                     builder.getI64IntegerAttr(unsupportedCount));
  (*module)->setAttr("gac.inspection_only", builder.getBoolAttr(inspection));
  if (failed(verify(*module)))
    fail("imported module failed registered MLIR verification");
  return module;
}
} // namespace

OwningOpRef<ModuleOp> importOnnx(StringRef path, MLIRContext &context,
                                 bool allowUnsupported) {
  // 公开入口必须先注册/加载导入结果会用到的方言。GacDialect 提供 gac.* 操作的
  // 定义与 verifier，FuncDialect 提供 func.func / func.return。
  context.getOrLoadDialect<GacDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  try {
    return Importer(path, context, allowUnsupported).run();
  } catch (const std::exception &error) {
    if (StringRef(error.what()) != "ONNX import failed")
      emitError(UnknownLoc::get(&context))
          << "ONNX import failed: " << error.what();
    return {};
  }
}
} // namespace gac
