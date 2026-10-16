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

using namespace mlir;
namespace fs = std::filesystem;

namespace gac {
namespace {
// Import errors carry ONNX node locations through the outer diagnostic handler.
class Importer {
public:
  Importer(StringRef filename, MLIRContext &context, bool inspection)
      : filename(filename.str()), builder(&context), inspection(inspection),
        location(FileLineColLoc::get(&context, filename, 0, 0)) {}

  OwningOpRef<ModuleOp> run();

private:
  std::string filename;
  OpBuilder builder;
  bool inspection;
  Location location;
  llvm::StringMap<Value> values;
  llvm::StringMap<Type> declaredTypes;
  llvm::StringMap<int64_t> opsets;
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

// This recognizes a semantic ONNX subgraph, not a model/node-name shortcut:
// If(Equal(Gather(Shape(X), [axis]), [1]), Squeeze(X,[axis]), Identity(X)).
// The two ranks need not agree. A registered GAC op retains that conditional
// rank until shape inference knows X; no invalid dead Squeeze is instantiated.
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
  OperationState state(location, name);
  state.addTypes(results);
  state.addOperands(operands);
  state.addAttributes(attrs);
  return builder.create(state);
}

void Importer::bind(StringRef name, Value value) {
  if (name.empty() || !values.try_emplace(name, value).second)
    fail("duplicate or empty ONNX SSA value name: " + name);
}

NamedAttrList Importer::attributes(const onnx::NodeProto &node,
                                   std::string &reason) {
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
  if (domain == "mmcv" && type == "MMCVModulatedDeformConv2d")
    return "gac.deform_conv2d";
  if (domain == "gac" && type == "MSDeformAttn")
    return "gac.ms_deform_attn";
  return {};
}

Type Importer::resultType(const onnx::NodeProto &node, StringRef output,
                          ArrayRef<Value> operands) {
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
    // These ONNX schemas preserve the data input's element type. Where's first
    // input is a boolean condition, so its data type comes from its second
    // input.
    size_t index = name == "Where" ? 1 : 0;
    if (operands.size() > index)
      if (auto shaped = dyn_cast<ShapedType>(operands[index].getType()))
        element = shaped.getElementType();
  }
  return element ? Type(UnrankedTensorType::get(element))
                 : Type(builder.getNoneType());
}

void Importer::importNode(const onnx::NodeProto &node, size_t index) {
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
  attrs.set("gac.source_name", builder.getStringAttr(sourceName));
  attrs.set("gac.source_domain", builder.getStringAttr(domain));
  attrs.set("gac.source_op", builder.getStringAttr(node.op_type()));
  attrs.set("gac.opset", builder.getI64IntegerAttr(version->second));
  attrs.set("gac.input_positions",
            builder.getDenseI64ArrayAttr(inputPositions));
  attrs.set("gac.output_names", builder.getArrayAttr(outputNames));
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
  Operation *op = create(target, resultTypes, operands, attrs);
  for (size_t i = 0; i < outputs.size(); ++i)
    bind(outputs[i], op->getResult(i));
}

OwningOpRef<ModuleOp> Importer::run() {
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
  for (const auto &opset : model.opset_import()) {
    std::string domain = opset.domain() == "ai.onnx" ? "" : opset.domain();
    if (opset.version() <= 0 ||
        !opsets.try_emplace(domain, opset.version()).second)
      fail("invalid or duplicated ONNX opset domain: " + domain);
  }
  const auto &graph = model.graph();
  if (graph.sparse_initializer_size())
    fail("sparse initializers are not supported");
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
  for (int i = 0; i < graph.node_size(); ++i)
    importNode(graph.node(i), i);
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
