#include "dispatch-plan.h"
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
#include <dxa.h>
#include <VX_config.h>
#include <VX_types.h>

// Vortex's public caps expose TCU presence, but not its enabled input formats.
// This is an explicit SDK/build-profile assertion, not a hardware format query.
// The linked driver/simulator/RTL must be built with the matching TF32 feature.
#ifndef GAC_VORTEX_TCU_TF32
#define GAC_VORTEX_TCU_TF32 0
#endif

namespace {
namespace fs = std::filesystem;
namespace json = llvm::json;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;
constexpr uint64_t addressLimit = uint64_t{1} << 32;

using namespace gac::runtime;
void check(vx_result_t result, const char *expression) {
  if (result != VX_SUCCESS)
    fail(std::string(expression) + ": " + vx_result_string(result));
}
#define CHECK(expression) check((expression), #expression)

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
  uint64_t isa = 0, lanes = 0, warps = 0, cores = 0, localMemory = 0;
  bool physicalBuffers = false;

  struct Counters { uint64_t cycles = 0, instructions = 0; };
  std::vector<Counters> counters() {
    std::vector<Counters> result(cores);
    for (uint32_t core = 0; core < cores; ++core) {
      CHECK(vx_device_mpm_query(device, 0, VX_CSR_MCYCLE, core,
                                &result[core].cycles));
      CHECK(vx_device_mpm_query(device, 0, VX_CSR_MINSTRET, core,
                                &result[core].instructions));
    }
    return result;
  }

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
                           VX_MEM_READ_WRITE | (physicalBuffers ? VX_MEM_PHYS : 0),
                           &buffers.back()));
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
    CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa));
    if (VX_ISA_ARCH(isa) != 32)
      fail("native Vortex compiler artifact requires an RV32 device");
    CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &lanes));
    CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps));
    CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &cores));
    CHECK(vx_device_query(device, VX_CAPS_LOCAL_MEM_SIZE, &localMemory));
    if (!lanes || !warps || lanes > UINT32_MAX / warps)
      fail("invalid device thread capacity");
    if (!cores || cores > UINT32_MAX)
      fail("invalid device core count");
    physicalBuffers = plan.doubleBuffer;
    for (const auto &dispatch : plan.dispatches) {
      if (dispatch.vortexSchedule.empty())
        continue;
      if (dispatch.lanes != lanes || dispatch.blockSize > lanes * warps)
        fail("Vortex schedule lane/block size is incompatible with this device");
      if (dispatch.lmemBytes > localMemory ||
          (dispatch.lmemBytes && !(isa & VX_ISA_EXT_LMEM)))
        fail("Vortex schedule LMEM requirement exceeds device capability");
      if (dispatch.vortexSchedule == "dxa-double-buffer" &&
          (!(isa & VX_ISA_EXT_DXA) || VX_CFG_NUM_BARRIERS < 2))
        fail("DXA double buffering requires DXA and two transaction barrier slots");
      if (dispatch.vortexSchedule == "tcu" && !(isa & VX_ISA_EXT_TCU))
        fail("TCU schedule requires the Vortex TCU extension");
      if (dispatch.vortexSchedule == "tcu" && !GAC_VORTEX_TCU_TF32)
        fail("TCU TF32x3 schedule requires a matching TF32-enabled Vortex build "
             "and runtime configured with GAC_VORTEX_TCU_TF32=ON");
      // The device dispatcher may reduce resident CTAs when LMEM is large;
      // do not reject a valid tile merely because it lowers occupancy.
      if (vx_check_occupancy(device, dispatch.blockSize, dispatch.lmemBytes))
        fail("Vortex tile cannot fit one CTA on this device");
    }
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
  void programDxa(const Plan &plan, const Dispatch &dispatch,
                  vx_event_h prior) {
    if (dispatch.vortexSchedule != "dxa-double-buffer")
      return;
    // Descriptors are device-wide mutable DCR state, not kernel parameters.
    // Wait before recycling slots 0/1. Vortex's launch completion includes
    // its post-kernel cache flush, so an arena tensor written by the previous
    // kernel is visible to DXA's physical-memory reader before the next copy.
    if (prior)
      CHECK(vx_event_wait_value(prior, 1, VX_TIMEOUT_INFINITE));
    auto descriptor = [&](uint32_t slot, size_t tensor, uint32_t columns,
                          uint32_t rows, uint32_t tileColumns,
                          uint32_t tileRows) {
      int result = vortex::dxa::program_2d(
          device, slot, address(plan.tensors[tensor]), columns, rows,
          columns * sizeof(float), tileColumns, tileRows, sizeof(float));
      if (result)
        fail("programming DXA physical 2D descriptor failed");
    };
    descriptor(0, dispatch.inputs[0], dispatch.k, dispatch.m,
               dispatch.tileK, dispatch.tileM);
    descriptor(1, dispatch.inputs[1], dispatch.n, dispatch.k,
               dispatch.tileN, dispatch.tileK);
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
      programDxa(plan, dispatch, prior);
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
      bool tiled = !dispatch.vortexSchedule.empty();
      launch.grid_dim[0] = tiled ? dispatch.gridSize :
          uint32_t((uint64_t(dispatch.elements) + threads - 1) / threads);
      launch.block_dim[0] = tiled ? dispatch.blockSize : threads;
      launch.lmem_size = dispatch.lmemBytes;
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
  bool traceDispatches = false, profileCounters = false;
  bool validArguments = argc >= 5;
  for (int i = 5; i < argc; ++i) {
    if (llvm::StringRef(argv[i]) == "--trace-dispatches" && !traceDispatches)
      traceDispatches = true;
    else if (llvm::StringRef(argv[i]) == "--profile-counters" && !profileCounters)
      profileCounters = true;
    else
      validArguments = false;
  }
  if (!validArguments) {
    llvm::errs() << "usage: gac-vortex-run MANIFEST.json KERNEL.vxbin DATA_DIR "
                    "OUTPUT_DIR [--trace-dispatches] [--profile-counters]\n";
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
    std::vector<Device::Counters> before;
    if (profileCounters)
      before = device.counters();
    vx_event_h done = device.launch(plan, traceDispatches ? &trace : nullptr);
    auto computed = Clock::now();
    json::Object counterReport{{"available", profileCounters}};
    if (profileCounters) {
      auto after = device.counters();
      json::Array perCore;
      uint64_t maximumCycles = 0, totalInstructions = 0;
      for (size_t core = 0; core < after.size(); ++core) {
        if (after[core].cycles < before[core].cycles ||
            after[core].instructions < before[core].instructions)
          fail("device MPM counters reset during execution");
        uint64_t cycles = after[core].cycles - before[core].cycles;
        uint64_t instructions =
            after[core].instructions - before[core].instructions;
        maximumCycles = std::max(maximumCycles, cycles);
        totalInstructions += instructions;
        perCore.push_back(json::Object{{"core", int64_t(core)},
                                       {"cycles", int64_t(cycles)},
                                       {"instret", int64_t(instructions)}});
      }
      counterReport["kind"] = "device_mpm";
      counterReport["cores"] = std::move(perCore);
      counterReport["max_core_cycles"] = int64_t(maximumCycles);
      counterReport["sum_core_instructions"] = int64_t(totalInstructions);
      counterReport["measurement_note"] =
          "Per-core cumulative MPM counter deltas around the complete dispatch "
          "sequence, including device-side launch/flush overhead. Snapshots are "
          "not simultaneous across cores. SimX cycles model this configuration; "
          "they are not measured silicon time or application FPS.";
    }
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
        {"profile_counters", profileCounters},
        {"device_counters", std::move(counterReport)},
        {"device_profile", json::Object{{"lanes", int64_t(device.lanes)},
                                          {"warps_per_core", int64_t(device.warps)},
                                          {"cores", int64_t(device.cores)},
                                          {"lmem_bytes_per_core", int64_t(device.localMemory)},
                                          {"isa_flags", int64_t(device.isa)},
                                          {"build_barrier_slots", VX_CFG_NUM_BARRIERS},
                                          {"build_tcu_tf32", bool(GAC_VORTEX_TCU_TF32)},
                                          {"build_profile_note",
                                           "Barrier count and TCU TF32 support are build-profile "
                                           "assertions; the public device caps do not query them. "
                                           "Use a driver/simulator/RTL built with the matching profile."}}},
        {"max_dispatch_lmem_bytes", int64_t(plan.lmemBytes)},
        {"dxa_physical_buffers", device.physicalBuffers},
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
