// Execute a native-MLIR dispatch manifest through Vortex's asynchronous C API.
// This host program allocates/copies/submits/waits; it never evaluates an AI
// op.
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <vortex2.h>

namespace {
namespace fs = std::filesystem;
namespace json = llvm::json;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;
constexpr uint64_t addressLimit = uint64_t{1} << 32;

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}
void check(vx_result_t result, const char *expression) {
  if (result != VX_SUCCESS)
    fail(std::string(expression) + ": " + vx_result_string(result));
}
#define CHECK(expression) check((expression), #expression)

const json::Object &object(const json::Value &value) {
  auto *result = value.getAsObject();
  if (!result)
    fail("manifest entry must be an object");
  return *result;
}
const json::Array &array(const json::Object &value, llvm::StringRef name) {
  auto *result = value.getArray(name);
  if (!result)
    fail("missing array: " + name.str());
  return *result;
}
std::string string(const json::Object &value, llvm::StringRef name) {
  auto result = value.getString(name);
  if (!result)
    fail("missing string: " + name.str());
  return result->str();
}
int64_t integer(const json::Object &value, llvm::StringRef name) {
  auto result = value.getInteger(name);
  if (!result)
    fail("missing integer: " + name.str());
  return *result;
}
uint64_t nonnegative(const json::Object &value, llvm::StringRef name) {
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
};
struct Plan {
  std::vector<Tensor> tensors;
  std::vector<Dispatch> dispatches;
  std::vector<size_t> inputs, outputs;
  uint64_t arenaBytes, embeddedConstantBytes;
};

Plan parsePlan(const fs::path &path) {
  auto buffer = llvm::MemoryBuffer::getFile(path.string());
  if (!buffer)
    fail("cannot read manifest: " + buffer.getError().message());
  auto parsed = json::parse((*buffer)->getBuffer());
  if (!parsed)
    fail("invalid manifest JSON: " + llvm::toString(parsed.takeError()));
  const auto &root = object(*parsed);
  if (integer(root, "schema_version") != 1 ||
      string(root, "target") != "vortex" ||
      string(root, "compiler_backend") != "native-mlir")
    fail("expected schema 1, native-mlir compiler, and vortex target");
  if (nonnegative(root, "lmem_bytes") != 0 ||
      root.getBoolean("double_buffer").value_or(true))
    fail("this runtime supports the scalar global-memory template only");
  Plan plan;
  plan.arenaBytes = nonnegative(root, "arena_bytes");
  plan.embeddedConstantBytes =
      root.get("embedded_constant_bytes")
          ? nonnegative(root, "embedded_constant_bytes")
          : 0;
  if (plan.arenaBytes > INT32_MAX ||
      plan.embeddedConstantBytes > INT32_MAX - plan.arenaBytes)
    fail("arena and embedded constants exceed compiler RV32 budget");
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
        fail("tensor requires positive static dimensions within signed RV32 "
             "range");
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
      residentBytes += tensor.bytes;
      if (residentBytes > INT32_MAX)
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
    plan.dispatches.push_back(std::move(dispatch));
  }
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

Bytes readBytes(const fs::path &path, uint64_t count) {
  if (count > std::numeric_limits<size_t>::max() || count > INT64_MAX)
    fail("file exceeds host size limits");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != std::streamoff(count))
    fail("missing file or wrong byte size: " + path.string());
  file.seekg(0);
  Bytes result(size_t(count), 0);
  if (count && !file.read(reinterpret_cast<char *>(result.data()),
                          std::streamsize(count)))
    fail("cannot read: " + path.string());
  return result;
}
void writeBytes(const fs::path &path, const Bytes &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file ||
      !file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()))
    fail("cannot write: " + path.string());
}

struct Device {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module = nullptr;
  std::vector<vx_buffer_h> buffers;
  std::vector<uint64_t> addresses;
  std::vector<vx_kernel_h> kernels;
  std::vector<vx_event_h> events;
  // Host argument blobs must remain valid throughout asynchronous submission.
  // Destruction drains the queue before these vectors or device buffers die.
  std::vector<std::vector<uint64_t>> arguments;
  uint32_t threads = 1;

  ~Device() {
    if (queue)
      vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
    for (auto event : events)
      if (event)
        vx_event_release(event);
    for (auto kernel : kernels)
      if (kernel)
        vx_kernel_release(kernel);
    if (module)
      vx_module_release(module);
    for (auto buffer : buffers)
      if (buffer)
        vx_buffer_release(buffer);
    if (queue)
      vx_queue_release(queue);
    if (device)
      vx_device_release(device);
  }
  size_t allocate(uint64_t bytes) {
    buffers.push_back(nullptr);
    CHECK(vx_buffer_create(device, std::max(uint64_t{1}, bytes),
                           VX_MEM_READ_WRITE, &buffers.back()));
    uint64_t address = 0;
    CHECK(vx_buffer_address(buffers.back(), &address));
    if (address >= addressLimit || bytes > addressLimit - address)
      fail("device allocation exceeds the RV32 address space");
    addresses.push_back(address);
    return buffers.size() - 1;
  }
  uint64_t address(const Tensor &tensor) const {
    uint64_t base = addresses.at(tensor.buffer);
    if (tensor.offset >= addressLimit - base ||
        tensor.bytes > addressLimit - base - tensor.offset)
      fail("tensor address exceeds RV32: " + tensor.name);
    return base + tensor.offset;
  }
  void initialize(Plan &plan, const std::string &kernelFile) {
    CHECK(vx_device_open(0, &device));
    uint64_t isa = 0, lanes = 0, warps = 0;
    CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa));
    if (VX_ISA_ARCH(isa) != 32)
      fail("native Vortex compiler artifact requires an RV32 device");
    CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &lanes));
    CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
    if (!lanes || !warps || lanes > UINT32_MAX / warps)
      fail("invalid device thread capacity");
    threads = uint32_t(std::min(uint64_t{32}, lanes * warps));
    vx_queue_info_t info = {};
    info.struct_size = sizeof(info);
    info.priority = VX_QUEUE_PRIORITY_NORMAL;
    CHECK(vx_queue_create(device, &info, &queue));
    size_t arena = plan.arenaBytes ? allocate(plan.arenaBytes) : 0;
    for (auto &tensor : plan.tensors) {
      tensor.buffer =
          tensor.storage == "arena" ? arena : allocate(tensor.bytes);
      address(tensor);
    }
    CHECK(vx_module_load_file(device, kernelFile.c_str(), &module));
    for (const auto &dispatch : plan.dispatches) {
      kernels.push_back(nullptr);
      CHECK(vx_module_get_kernel(module, dispatch.kernel.c_str(),
                                 &kernels.back()));
    }
  }
  void transfer(const Tensor &tensor, Bytes &bytes, bool upload,
                vx_event_h prior = nullptr) {
    if (bytes.size() != tensor.bytes)
      fail("host transfer byte count mismatch");
    events.push_back(nullptr);
    try {
      if (upload)
        CHECK(vx_enqueue_write(queue, buffers.at(tensor.buffer), tensor.offset,
                               bytes.data(), bytes.size(), 0, nullptr,
                               &events.back()));
      else
        CHECK(vx_enqueue_read(queue, bytes.data(), buffers.at(tensor.buffer),
                              tensor.offset, bytes.size(), prior ? 1 : 0,
                              prior ? &prior : nullptr, &events.back()));
      CHECK(vx_event_wait_value(events.back(), 1, VX_TIMEOUT_INFINITE));
    } catch (...) {
      // The caller's temporary host vector may be destroyed immediately after
      // this throws. Wait before allowing its memory to leave scope.
      vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
      throw;
    }
  }
  vx_event_h launch(const Plan &plan, std::ostream *trace = nullptr) {
    arguments.reserve(plan.dispatches.size());
    vx_event_h prior = nullptr;
    for (size_t i = 0; i < plan.dispatches.size(); ++i) {
      const auto &dispatch = plan.dispatches[i];
      auto traceEvent = [&](llvm::StringRef event) {
        if (!trace)
          return;
        json::Object row{{"index", int64_t(i)},
                         {"kernel", dispatch.kernel},
                         {"event", event.str()}};
        *trace << llvm::formatv("{0}\n", json::Value(std::move(row))).str();
        trace->flush();
        if (!*trace)
          fail("writing dispatch trace failed");
      };
      traceEvent("begin");
      arguments.emplace_back();
      for (size_t input : dispatch.inputs)
        arguments.back().push_back(address(plan.tensors[input]));
      arguments.back().push_back(address(plan.tensors[dispatch.output]));
      vx_launch_info_t launch = {};
      launch.struct_size = sizeof(launch);
      launch.kernel = kernels.at(i);
      launch.args_host = arguments.back().data();
      launch.args_size = arguments.back().size() * sizeof(uint64_t);
      launch.ndim = 1;
      launch.grid_dim[0] =
          uint32_t((uint64_t(dispatch.elements) + threads - 1) / threads);
      launch.block_dim[0] = threads;
      events.push_back(nullptr);
      CHECK(vx_enqueue_launch(queue, &launch, prior ? 1 : 0,
                              prior ? &prior : nullptr, &events.back()));
      prior = events.back();
      // Debug mode records actual completion, not merely successful enqueue.
      // Dependencies and arena lifetimes are identical to the normal path.
      if (trace) {
        CHECK(vx_event_wait_value(prior, 1, VX_TIMEOUT_INFINITE));
        traceEvent("end");
      }
    }
    if (prior)
      CHECK(vx_event_wait_value(prior, 1, VX_TIMEOUT_INFINITE));
    return prior;
  }
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}
} // namespace

int main(int argc, char **argv) {
  bool traceDispatches =
      argc == 6 && llvm::StringRef(argv[5]) == "--trace-dispatches";
  if (argc != 5 && !traceDispatches) {
    llvm::errs() << "usage: gac-vortex-run MANIFEST.json KERNEL.vxbin DATA_DIR "
                    "OUTPUT_DIR [--trace-dispatches]\n";
    return 1;
  }
  try {
    auto started = Clock::now();
    Plan plan = parsePlan(argv[1]);
    fs::path dataDirectory(argv[3]), outputDirectory(argv[4]);
    fs::create_directories(outputDirectory);
    Device device;
    device.initialize(plan, argv[2]);
    auto initialized = Clock::now();
    json::Array inputFiles, outputFiles;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
      const auto &tensor = plan.tensors[i];
      if (tensor.storage == "arena")
        continue;
      std::string file = tensor.storage == "constant"
                             ? tensor.file
                             : "input_" + std::to_string(i) + ".bin";
      Bytes bytes = readBytes(dataDirectory / file, tensor.bytes);
      device.transfer(tensor, bytes, true);
      if (tensor.storage == "input")
        inputFiles.push_back(json::Object{{"name", tensor.name},
                                          {"file", file},
                                          {"tensor_index", int64_t(i)}});
    }
    auto uploaded = Clock::now();
    std::ofstream trace;
    if (traceDispatches) {
      trace.open(outputDirectory / "dispatch-progress.jsonl");
      if (!trace)
        fail("cannot open dispatch trace");
    }
    vx_event_h done = device.launch(plan, traceDispatches ? &trace : nullptr);
    auto computed = Clock::now();
    for (size_t i = 0; i < plan.outputs.size(); ++i) {
      size_t id = plan.outputs[i];
      const auto &tensor = plan.tensors[id];
      Bytes bytes(size_t(tensor.bytes));
      device.transfer(tensor, bytes, false, done);
      std::string file = "output_" + std::to_string(i) + ".bin";
      writeBytes(outputDirectory / file, bytes);
      json::Array shape;
      for (int64_t dim : tensor.shape)
        shape.push_back(dim);
      outputFiles.push_back(json::Object{{"name", tensor.name},
                                         {"file", file},
                                         {"tensor_index", int64_t(id)},
                                         {"dtype", tensor.dtype},
                                         {"shape", std::move(shape)}});
    }
    CHECK(vx_queue_finish(device.queue, VX_TIMEOUT_INFINITE));
    auto finished = Clock::now();
    const char *backend = std::getenv("VORTEX_DRIVER");
    json::Object report{
        {"schema_version", 1},
        {"execution", "vortex-device"},
        {"driver", backend ? backend : "runtime-default"},
        {"host_operator_fallback", false},
        {"dispatches", int64_t(plan.dispatches.size())},
        {"arena_bytes", int64_t(plan.arenaBytes)},
        {"embedded_constant_bytes", int64_t(plan.embeddedConstantBytes)},
        {"trace_dispatches", traceDispatches},
        {"threads_per_block", int64_t(device.threads)},
        {"inputs", std::move(inputFiles)},
        {"outputs", std::move(outputFiles)},
        {"timing_kind", "host_wall_clock"},
        {"timing_note",
         "Includes host/runtime overhead; simulator wall time is not silicon "
         "inference latency. Upload/download times include file I/O."},
        {"initialization_ms", milliseconds(started, initialized)},
        {"upload_ms", milliseconds(initialized, uploaded)},
        {"dispatch_wall_ms", milliseconds(uploaded, computed)},
        {"download_ms", milliseconds(computed, finished)},
        {"total_host_wall_ms", milliseconds(started, finished)}};
    std::string text =
        llvm::formatv("{0:2}\n", json::Value(std::move(report))).str();
    std::ofstream file(outputDirectory / "runtime-report.json");
    if (!file || !(file << text))
      fail("cannot write runtime-report.json");
    llvm::outs() << text;
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "gac-vortex-run: " << error.what() << '\n';
    return 1;
  }
}
