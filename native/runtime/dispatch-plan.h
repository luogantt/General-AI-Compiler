// Native dispatch contract shared by device runtimes. Parsing also proves
// single-stream dependencies and inclusive arena-buffer lifetimes before any
// device code is loaded or launched.
#pragma once
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace gac::runtime {
namespace fs = std::filesystem;
namespace json = llvm::json;
[[noreturn]] inline void fail(const std::string &message) {
  throw std::runtime_error(message);
}
inline const json::Object &object(const json::Value &value) {
  auto *result = value.getAsObject();
  if (!result)
    fail("manifest entry must be an object");
  return *result;
}
inline const json::Array &array(const json::Object &value,
                                llvm::StringRef name) {
  auto *result = value.getArray(name);
  if (!result)
    fail("missing array: " + name.str());
  return *result;
}
inline std::string string(const json::Object &value, llvm::StringRef name) {
  auto result = value.getString(name);
  if (!result)
    fail("missing string: " + name.str());
  return result->str();
}
inline int64_t integer(const json::Object &value, llvm::StringRef name) {
  auto result = value.getInteger(name);
  if (!result)
    fail("missing integer: " + name.str());
  return *result;
}
inline uint64_t nonnegative(const json::Object &value, llvm::StringRef name) {
  int64_t result = integer(value, name);
  if (result < 0)
    fail("negative value for " + name.str());
  return uint64_t(result);
}

struct Tensor {
  std::string name, dtype, storage, file;
  uint64_t bytes, offset, elements;
  std::vector<int64_t> shape;
  size_t buffer = 0;
  int64_t producer = -1, lastUse = -1;
};
struct Dispatch {
  std::string kernel;
  std::vector<size_t> inputs;
  size_t output;
  uint32_t elements;
  uint32_t blockSize = 0, gridSize = 0;
  // Empty schedule preserves the original scalar global-memory contract.
  std::string vortexSchedule;
  uint32_t m = 0, n = 0, k = 0, batches = 0;
  uint32_t tileM = 0, tileN = 0, tileK = 0;
  uint32_t lmemBytes = 0, stages = 0, lanes = 0;
};
struct Plan {
  std::vector<Tensor> tensors;
  std::vector<Dispatch> dispatches;
  std::vector<size_t> inputs, outputs;
  uint64_t arenaBytes, embeddedConstantBytes;
  uint64_t lmemBytes = 0;
  bool doubleBuffer = false;
};

enum class Target { Vortex, CUDA };

inline Plan parsePlan(const fs::path &path, Target target = Target::Vortex) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer)
    fail("cannot read manifest: " + buffer.getError().message());
  auto parsed = json::parse((*buffer)->getBuffer());
  if (!parsed)
    fail("invalid manifest JSON: " + llvm::toString(parsed.takeError()));
  const auto &root = object(*parsed);
  bool cuda = target == Target::CUDA;
  if (integer(root, "schema_version") != 1 ||
      string(root, "target") != (cuda ? "cuda" : "vortex") ||
      string(root, "compiler_backend") != "native-mlir")
    fail("expected schema 1, native-mlir compiler, and " +
         std::string(cuda ? "cuda" : "vortex") + " target");
  uint64_t lmemBytes = nonnegative(root, "lmem_bytes");
  auto doubleBuffer = root.getBoolean("double_buffer");
  if (!doubleBuffer || lmemBytes > UINT32_MAX)
    fail("invalid root LMEM/buffer contract");
  if (cuda && (lmemBytes != 0 || *doubleBuffer))
    fail("CUDA runtime supports the scalar global-memory template only");
  if (cuda) {
    auto *abi = root.getObject("cuda");
    if (!abi || string(*abi, "kernel_abi") != "direct-pointers" ||
        integer(*abi, "pointer_bits") != 64 ||
        integer(*abi, "index_bits") != 32 ||
        string(*abi, "math_library") != "libdevice")
      fail("unsupported CUDA kernel ABI");
  }
  Plan plan;
  plan.lmemBytes = lmemBytes;
  plan.doubleBuffer = *doubleBuffer;
  plan.arenaBytes = nonnegative(root, "arena_bytes");
  plan.embeddedConstantBytes =
      root.get("embedded_constant_bytes")
          ? nonnegative(root, "embedded_constant_bytes")
          : 0;
  if (!cuda && (plan.arenaBytes > INT32_MAX ||
                plan.embeddedConstantBytes > INT32_MAX - plan.arenaBytes))
    fail("arena and embedded constants exceed compiler RV32 budget");
  if (plan.embeddedConstantBytes > uint64_t(INT64_MAX) - plan.arenaBytes)
    fail("resident allocation size overflows host limits");
  std::map<std::string, size_t> names;
  uint64_t residentBytes = plan.arenaBytes + plan.embeddedConstantBytes;
  for (const auto &item : array(root, "tensors")) {
    const auto &row = object(item);
    Tensor tensor;
    tensor.name = string(row, "name");
    tensor.dtype = string(row, "dtype");
    tensor.storage = string(row, "storage");
    tensor.bytes = nonnegative(row, "bytes");
    tensor.offset = nonnegative(row, "offset");
    if (tensor.name.empty() ||
        !names.emplace(tensor.name, plan.tensors.size()).second)
      fail("empty or duplicate tensor name");
    unsigned width = tensor.dtype == "float32" || tensor.dtype == "int32" ? 4
                     : tensor.dtype == "int64"                            ? 8
                     : tensor.dtype == "float16"                          ? 2
                     : tensor.dtype == "bool"                             ? 1
                                                                          : 0;
    if (!width)
      fail("unsupported tensor dtype: " + tensor.dtype);
    tensor.elements = 1;
    for (const auto &dim : array(row, "shape")) {
      auto size = dim.getAsInteger();
      if (!size || *size <= 0 || uint64_t(*size) > INT32_MAX / tensor.elements)
        fail("tensor requires positive static dimensions within signed 32-bit "
             "index range");
      tensor.shape.push_back(*size);
      tensor.elements *= uint64_t(*size);
    }
    if (tensor.elements > INT32_MAX / width ||
        tensor.elements * width != tensor.bytes)
      fail("tensor shape/dtype/byte count disagree: " + tensor.name);
    if (tensor.storage == "arena") {
      if (tensor.offset % 64 || tensor.offset > plan.arenaBytes ||
          tensor.bytes > plan.arenaBytes - tensor.offset)
        fail("tensor is outside its aligned arena allocation: " + tensor.name);
    } else {
      if (tensor.storage != "input" && tensor.storage != "constant")
        fail("unknown tensor storage: " + tensor.storage);
      if (tensor.offset)
        fail("standalone input/constant buffer must have offset zero");
      if (tensor.bytes > uint64_t(INT64_MAX) - residentBytes)
        fail("resident allocation size overflows host limits");
      residentBytes += tensor.bytes;
      if (!cuda && residentBytes > INT32_MAX)
        fail("resident buffers exceed compiler RV32 budget");
      if (tensor.storage == "constant") {
        tensor.file = string(row, "file");
        // Compiler-generated file identities, not tensor names, form paths.
        if (tensor.file !=
            "tensor_" + std::to_string(plan.tensors.size()) + ".bin")
          fail("constant file must match its compiler-assigned tensor index");
      }
    }
    plan.tensors.push_back(std::move(tensor));
  }
  auto tensorID = [&](const json::Value &value) -> size_t {
    auto name = value.getAsString();
    if (!name)
      fail("tensor reference must be a string");
    auto found = names.find(name->str());
    if (found == names.end())
      fail("unknown tensor reference: " + name->str());
    return found->second;
  };
  std::set<size_t> inputSet;
  for (const auto &input : array(root, "inputs")) {
    size_t id = tensorID(input);
    if (plan.tensors[id].storage != "input" || !inputSet.insert(id).second)
      fail("invalid or duplicate graph input");
    plan.inputs.push_back(id);
  }
  for (size_t id = 0; id < plan.tensors.size(); ++id)
    if (plan.tensors[id].storage == "input" && !inputSet.count(id))
      fail("input storage tensor is absent from graph input list");
  for (const auto &output : array(root, "outputs"))
    plan.outputs.push_back(tensorID(output));

  uint64_t maximumLmem = 0;
  bool usesDoubleBuffer = false;
  for (const auto &item : array(root, "dispatches")) {
    const auto &row = object(item);
    int64_t index = plan.dispatches.size();
    if (integer(row, "wait_for_dispatch") != index - 1)
      fail("arena reuse requires the compiler's serialized dispatch order");
    Dispatch dispatch;
    dispatch.kernel = string(row, "kernel");
    if (dispatch.kernel.empty())
      fail("empty kernel entry name");
    std::set<int64_t> dependencies;
    for (const auto &dependency : array(row, "depends_on")) {
      auto value = dependency.getAsInteger();
      if (!value || *value < 0 || *value >= index)
        fail("dispatch dependencies must reference preceding dispatches");
      dependencies.insert(*value);
    }
    for (const auto &input : array(row, "inputs")) {
      size_t id = tensorID(input);
      auto &tensor = plan.tensors[id];
      if (tensor.storage == "arena" && tensor.producer < 0)
        fail("dispatch reads an arena tensor before its producer");
      if (tensor.producer >= 0 && !dependencies.count(tensor.producer))
        fail("dispatch dependency list omits an input producer");
      tensor.lastUse = std::max(tensor.lastUse, index);
      dispatch.inputs.push_back(id);
    }
    auto *output = row.get("output");
    if (!output)
      fail("dispatch has no output");
    dispatch.output = tensorID(*output);
    auto &result = plan.tensors[dispatch.output];
    if (result.storage != "arena" || result.producer >= 0)
      fail("dispatch must define a previously undefined arena tensor");
    result.producer = result.lastUse = index;
    uint64_t elements = nonnegative(row, "elements");
    if (elements != result.elements || elements > INT32_MAX)
      fail("dispatch element count disagrees with its output");
    dispatch.elements = uint32_t(elements);
    if (cuda) {
      uint64_t block = nonnegative(row, "block_size");
      uint64_t grid = nonnegative(row, "grid_size");
      if (!block || block > 1024 || grid != (elements + block - 1) / block ||
          grid > INT32_MAX || nonnegative(row, "shared_memory_bytes") != 0)
        fail("invalid CUDA scalar kernel launch dimensions");
      dispatch.blockSize = uint32_t(block);
      dispatch.gridSize = uint32_t(grid);
      if (row.get("vortex"))
        fail("CUDA dispatch cannot carry a Vortex schedule");
    } else if (auto *vortex = row.get("vortex")) {
      const auto &schedule = object(*vortex);
      auto dimension = [&](llvm::StringRef name, bool allowZero = false) {
        uint64_t value = nonnegative(schedule, name);
        if ((!allowZero && !value) || value > INT32_MAX)
          fail("invalid Vortex schedule dimension: " + name.str());
        return uint32_t(value);
      };
      dispatch.vortexSchedule = string(schedule, "schedule");
      dispatch.m = dimension("m");
      dispatch.n = dimension("n");
      dispatch.k = dimension("k");
      dispatch.batches = dimension("batches");
      dispatch.tileM = dimension("tile_m");
      dispatch.tileN = dimension("tile_n");
      dispatch.tileK = dimension("tile_k");
      dispatch.blockSize = dimension("block_size");
      dispatch.gridSize = dimension("grid_size");
      dispatch.lmemBytes = dimension("lmem_bytes", true);
      dispatch.stages = dimension("stages", true);
      dispatch.lanes = dimension("lanes");
      bool dxa = dispatch.vortexSchedule == "dxa-double-buffer";
      bool tcu = dispatch.vortexSchedule == "tcu";
      if (!dxa && !tcu && dispatch.vortexSchedule != "lmem-tiled")
        fail("unknown Vortex compute schedule");
      uint64_t mn = uint64_t(dispatch.m) * dispatch.n;
      if (mn > INT32_MAX || dispatch.batches > INT32_MAX / mn ||
          mn * dispatch.batches != elements)
        fail("Vortex matrix dimensions disagree with output elements");
      uint64_t rows = (uint64_t(dispatch.m) + dispatch.tileM - 1) / dispatch.tileM;
      uint64_t cols = (uint64_t(dispatch.n) + dispatch.tileN - 1) / dispatch.tileN;
      if (rows * cols * dispatch.batches != dispatch.gridSize ||
          dispatch.blockSize % dispatch.lanes)
        fail("Vortex tile/grid/lane dimensions disagree");
      if (dispatch.inputs.size() < 2 || dispatch.inputs.size() > 3)
        fail("Vortex matrix template requires A, B, and optional bias");
      // Views such as Reshape share their producer's physical allocation.
      // Its storage shape may differ from the logical shape used by this
      // kernel. Prove byte identity before validating the logical MatMul ABI.
      auto logicalOperand = [&](size_t id, llvm::StringRef name) {
        Tensor operand = plan.tensors[id];
        operand.shape.clear();
        uint64_t elements = 1;
        for (const auto &dim : array(schedule, name)) {
          auto size = dim.getAsInteger();
          if (!size || *size <= 0 ||
              uint64_t(*size) > operand.elements / elements)
            fail("Vortex logical operand shape exceeds physical storage");
          elements *= uint64_t(*size);
          operand.shape.push_back(*size);
        }
        if (elements != operand.elements)
          fail("Vortex logical operand shape differs from physical byte count");
        return operand;
      };
      const auto a = logicalOperand(dispatch.inputs[0], "a_shape");
      const auto b = logicalOperand(dispatch.inputs[1], "b_shape");
      if (a.dtype != "float32" || b.dtype != "float32" ||
          result.dtype != "float32" || a.shape.size() < 2 ||
          b.shape.size() < 2 || result.shape.size() < 2 ||
          a.shape.back() != dispatch.k ||
          b.shape[b.shape.size() - 2] != dispatch.k ||
          b.shape.back() != dispatch.n || result.shape.back() != dispatch.n)
        fail("Vortex matrix template requires compatible FP32 matrix operands");
      if (dispatch.batches == 1 && b.shape.size() == 2) {
        if (a.elements != uint64_t(dispatch.m) * dispatch.k)
          fail("Vortex flattened projection shape disagrees with A");
        if (result.shape.size() != a.shape.size() ||
            !std::equal(a.shape.begin(), a.shape.end() - 1, result.shape.begin()))
          fail("Vortex flattened projection output shape disagrees with A");
      } else {
        if (a.shape[a.shape.size() - 2] != dispatch.m ||
            result.shape[result.shape.size() - 2] != dispatch.m)
          fail("Vortex batch matrix shape disagrees with M");
        auto checkBroadcast = [&](const Tensor &operand) {
          if (operand.shape.size() > result.shape.size())
            fail("Vortex operand batch rank exceeds output rank");
          size_t pad = result.shape.size() - operand.shape.size();
          for (size_t i = 0; i + 2 < operand.shape.size(); ++i)
            if (operand.shape[i] != 1 &&
                operand.shape[i] != result.shape[pad + i])
              fail("Vortex matrix batch dimensions cannot broadcast");
        };
        checkBroadcast(a);
        checkBroadcast(b);
        if (result.shape.size() != std::max(a.shape.size(), b.shape.size()))
          fail("Vortex matrix output batch rank disagrees with operands");
        for (size_t axis = 0; axis + 2 < result.shape.size(); ++axis) {
          auto dim = [&](const Tensor &operand) -> int64_t {
            size_t pad = result.shape.size() - operand.shape.size();
            return axis < pad ? 1 : operand.shape[axis - pad];
          };
          if (result.shape[axis] != std::max(dim(a), dim(b)))
            fail("Vortex output batch shape disagrees with broadcast operands");
        }
      }
      if (dispatch.inputs.size() == 3) {
        const auto bias = logicalOperand(dispatch.inputs[2], "bias_shape");
        if (bias.dtype != "float32" || bias.shape.size() > result.shape.size())
          fail("Vortex matrix bias must be broadcastable FP32");
        size_t pad = result.shape.size() - bias.shape.size();
        for (size_t i = 0; i < bias.shape.size(); ++i)
          if (bias.shape[i] != 1 && bias.shape[i] != result.shape[pad + i])
            fail("Vortex matrix bias shape cannot broadcast");
      } else if (!array(schedule, "bias_shape").empty()) {
        fail("Vortex matrix without a bias cannot carry a bias shape");
      }
      if (tcu) {
        if (string(schedule, "arithmetic") != "tf32x3-fp32-accumulate")
          fail("TCU schedule requires explicit TF32x3 arithmetic with FP32 accumulation");
        if (dispatch.lanes != 4 || dispatch.blockSize != 4 ||
            dispatch.tileM != 8 || dispatch.tileN != 4 || dispatch.tileK != 4 ||
            dispatch.lmemBytes || dispatch.stages)
          fail("unsupported Vortex FP32 WMMA geometry");
      } else {
        if (dispatch.tileM > 65535 || dispatch.tileN > 65535 ||
            dispatch.tileK > 65535)
          fail("Vortex LMEM tile dimensions exceed supported range");
        uint64_t tileElements = uint64_t(dispatch.tileM) * dispatch.tileN;
        uint64_t stageBytes = 4 * uint64_t(dispatch.tileK) *
                              (uint64_t(dispatch.tileM) + dispatch.tileN);
        if (tileElements != dispatch.blockSize ||
            dispatch.stages != (dxa ? 2u : 1u) ||
            stageBytes > INT32_MAX / dispatch.stages ||
            stageBytes * dispatch.stages != dispatch.lmemBytes)
          fail("Vortex LMEM allocation does not match tile/stage geometry");
        if (dxa && (dispatch.batches != 1 || b.shape.size() != 2 ||
                    dispatch.lanes != 4 || dispatch.tileM > 65535 ||
                    dispatch.tileN > 65535 || dispatch.tileK > 65535 ||
                    dispatch.m > 65535 || dispatch.n > 65535 || dispatch.k > 65535))
          fail("DXA requires a 2D matrix with 16-bit tile coordinates and four lanes");
      }
      maximumLmem = std::max(maximumLmem, uint64_t(dispatch.lmemBytes));
      usesDoubleBuffer |= dxa;
    }
    plan.dispatches.push_back(std::move(dispatch));
  }
  if (!cuda && (maximumLmem != plan.lmemBytes ||
                usesDoubleBuffer != plan.doubleBuffer))
    fail("root LMEM/buffer summary disagrees with dispatch schedules");
  for (size_t output : plan.outputs)
    plan.tensors[output].lastUse = plan.dispatches.size();
  for (const auto &tensor : plan.tensors)
    if (tensor.storage == "arena" && tensor.producer < 0)
      fail("arena tensor has no producing dispatch: " + tensor.name);
  // Inclusive last-use is essential: output writes cannot overwrite an input
  // that is read by the SAME kernel, even though the queue is serialized.
  std::vector<size_t> active;
  for (size_t i = 0; i < plan.dispatches.size(); ++i) {
    active.erase(std::remove_if(active.begin(), active.end(),
                                [&](size_t id) {
                                  return plan.tensors[id].lastUse < int64_t(i);
                                }),
                 active.end());
    size_t id = plan.dispatches[i].output;
    const auto &next = plan.tensors[id];
    for (size_t live : active) {
      const auto &other = plan.tensors[live];
      if (next.offset < other.offset + other.bytes &&
          other.offset < next.offset + next.bytes)
        fail("live arena tensors overlap: " + next.name + " and " + other.name);
    }
    active.push_back(id);
  }
  return plan;
}

} // namespace gac::runtime
