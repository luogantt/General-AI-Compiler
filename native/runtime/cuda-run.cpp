// Native MLIR kernels execute on the CUDA device. The host only validates the
// dispatch contract, manages buffers, and submits pointer arguments in order.
// Official cuda.h provides the ABI; the driver is loaded at runtime so this
// program can be built and diagnose missing GPUs on a compiler-only machine.
#include "dispatch-plan.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cuda.h>
#include <fstream>
#include <limits>
#include <numeric>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
using namespace gac::runtime;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

class DriverLibrary {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA("nvcuda.dll");
#else
  void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
public:
  DriverLibrary() {
    if (!handle)
      fail("CUDA driver library unavailable; install an NVIDIA driver on a "
           "machine with a supported GPU (no host operator fallback)");
  }
  ~DriverLibrary() {
#ifdef _WIN32
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
  }
  DriverLibrary(const DriverLibrary &) = delete;
  DriverLibrary &operator=(const DriverLibrary &) = delete;
  template <typename T> T symbol(const char *name) {
#ifdef _WIN32
    auto address = GetProcAddress(handle, name);
#else
    auto address = dlsym(handle, name);
#endif
    if (!address)
      fail(std::string("CUDA driver is missing symbol: ") + name);
    return reinterpret_cast<T>(address);
  }
};

// Expand cuda.h aliases (e.g. cuMemAlloc -> cuMemAlloc_v2) before dlsym.
#define CUDA_STRING_IMPL(name) #name
#define CUDA_STRING(name) CUDA_STRING_IMPL(name)
#define CUDA_API(field, name)                                                  \
  decltype(&name) field = library.symbol<decltype(&name)>(CUDA_STRING(name))
struct Driver {
  DriverLibrary library;
  CUDA_API(init, cuInit);
  CUDA_API(deviceCount, cuDeviceGetCount);
  CUDA_API(deviceGet, cuDeviceGet);
  CUDA_API(deviceName, cuDeviceGetName);
  CUDA_API(deviceAttribute, cuDeviceGetAttribute);
  CUDA_API(deviceMemory, cuDeviceTotalMem);
  CUDA_API(memoryInfo, cuMemGetInfo);
  CUDA_API(driverVersion, cuDriverGetVersion);
  CUDA_API(retainContext, cuDevicePrimaryCtxRetain);
  CUDA_API(releaseContext, cuDevicePrimaryCtxRelease);
  CUDA_API(setContext, cuCtxSetCurrent);
  CUDA_API(synchronizeContext, cuCtxSynchronize);
  CUDA_API(loadModule, cuModuleLoadDataEx);
  CUDA_API(unloadModule, cuModuleUnload);
  CUDA_API(getFunction, cuModuleGetFunction);
  CUDA_API(functionAttribute, cuFuncGetAttribute);
  CUDA_API(allocate, cuMemAlloc);
  CUDA_API(free, cuMemFree);
  CUDA_API(upload, cuMemcpyHtoD);
  CUDA_API(download, cuMemcpyDtoH);
  CUDA_API(createStream, cuStreamCreate);
  CUDA_API(destroyStream, cuStreamDestroy);
  CUDA_API(synchronize, cuStreamSynchronize);
  CUDA_API(launch, cuLaunchKernel);
  CUDA_API(createEvent, cuEventCreate);
  CUDA_API(destroyEvent, cuEventDestroy);
  CUDA_API(recordEvent, cuEventRecord);
  CUDA_API(waitEvent, cuEventSynchronize);
  CUDA_API(eventTime, cuEventElapsedTime);
  CUDA_API(errorName, cuGetErrorName);
  CUDA_API(errorString, cuGetErrorString);

  void check(CUresult result, const char *operation) const {
    if (result == CUDA_SUCCESS)
      return;
    const char *name = nullptr, *description = nullptr;
    errorName(result, &name);
    errorString(result, &description);
    fail(std::string(operation) + ": " + (name ? name : "CUDA_ERROR") + " (" +
         std::to_string(int(result)) +
         "): " + (description ? description : "no driver error description"));
  }
};
#undef CUDA_API
#define CUDA_CHECK(expression) api.check((expression), #expression)

Bytes readBytes(const fs::path &path, uint64_t count) {
  if (count > std::numeric_limits<size_t>::max() || count > INT64_MAX)
    fail("tensor file exceeds host size limits");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != std::streamoff(count))
    fail("missing tensor file or incorrect byte size: " + path.string());
  file.seekg(0);
  Bytes bytes(size_t(count), 0);
  if (count && !file.read(reinterpret_cast<char *>(bytes.data()), count))
    fail("cannot read tensor file: " + path.string());
  return bytes;
}

void writeBytes(const fs::path &path, const Bytes &bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file ||
      !file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()))
    fail("cannot write tensor file: " + path.string());
}

void writeReport(const fs::path &directory, json::Object report) {
  std::string text =
      llvm::formatv("{0:2}\n", json::Value(std::move(report))).str();
  std::ofstream file(directory / "runtime-report.json");
  if (!file || !(file << text))
    fail("cannot write runtime-report.json");
  llvm::outs() << text;
}

struct Options {
  int device = 0, repeat = 1, warmup = 0;
  bool trace = false, profile = false;
};

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 5; i < argc; ++i) {
    std::string flag(argv[i]);
    if (flag == "--trace-dispatches") {
      options.trace = true;
      continue;
    }
    if (flag == "--profile-dispatches") {
      options.profile = true;
      continue;
    }
    if ((flag != "--device" && flag != "--repeat" && flag != "--warmup") ||
        ++i >= argc)
      fail("unknown option or missing argument: " + flag);
    std::string text(argv[i]);
    if (text.empty() ||
        text.find_first_not_of("0123456789") != std::string::npos)
      fail("option requires a nonnegative integer: " + flag);
    uint64_t number = std::stoull(text);
    if (number > INT32_MAX)
      fail("option value is too large: " + flag);
    if (flag == "--device")
      options.device = int(number);
    if (flag == "--warmup")
      options.warmup = int(number);
    if (flag == "--repeat")
      options.repeat = int(number);
  }
  if (!options.repeat)
    fail("--repeat must be positive");
  if (options.trace && options.profile)
    fail("--trace-dispatches and --profile-dispatches are mutually exclusive: "
         "trace serializes the host after every kernel");
  return options;
}

struct KernelResources {
  int registers = 0, sharedBytes = 0, localBytes = 0, maxThreads = 0;
};

struct ProfileSample {
  CUevent begin = nullptr, end = nullptr;
  double sumMs = 0, minMs = std::numeric_limits<double>::infinity(), maxMs = 0;
};

struct Device {
  Driver api;
  CUdevice device = 0;
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUmodule module = nullptr;
  CUevent begin = nullptr, end = nullptr;
  std::vector<CUdeviceptr> allocations;
  std::vector<CUfunction> functions;
  std::vector<std::vector<CUdeviceptr>> arguments;
  // Pointer argument storage is fixed after initialize. Reuse the addresses
  // instead of allocating a parameter vector for every launch of every frame.
  std::vector<std::vector<void *>> parameters;
  std::vector<KernelResources> resources;
  std::vector<ProfileSample> profile;
  std::string name, jitLog;
  int major = 0, minor = 0, version = 0;
  size_t totalMemory = 0;
  size_t completedDispatches = 0;
  json::Array memorySnapshots;

  void snapshotMemory(llvm::StringRef stage) {
    size_t free = 0, total = 0;
    CUDA_CHECK(api.memoryInfo(&free, &total));
    memorySnapshots.push_back(
        json::Object{{"stage", stage.str()},
                     {"free_bytes", int64_t(free)},
                     {"total_bytes", int64_t(total)},
                     {"used_device_bytes", int64_t(total - free)}});
  }

  ~Device() {
    // Drain before freeing even after an asynchronous device error. Cleanup
    // must not hide the original failure with a second exception.
    if (context)
      api.setContext(context);
    if (stream)
      api.synchronize(stream);
    if (begin)
      api.destroyEvent(begin);
    if (end)
      api.destroyEvent(end);
    for (auto &sample : profile) {
      if (sample.begin)
        api.destroyEvent(sample.begin);
      if (sample.end)
        api.destroyEvent(sample.end);
    }
    if (module)
      api.unloadModule(module);
    for (auto allocation : allocations)
      api.free(allocation);
    if (stream)
      api.destroyStream(stream);
    if (context)
      api.releaseContext(device);
  }

  size_t allocate(uint64_t bytes) {
    if (bytes > std::numeric_limits<size_t>::max())
      fail("allocation exceeds host addressable size");
    CUdeviceptr pointer = 0;
    CUDA_CHECK(api.allocate(&pointer, size_t(std::max(bytes, uint64_t{1}))));
    allocations.push_back(pointer);
    return allocations.size() - 1;
  }

  CUdeviceptr address(const Tensor &tensor) const {
    CUdeviceptr base = allocations.at(tensor.buffer);
    if (tensor.offset > UINT64_MAX - base ||
        tensor.bytes > UINT64_MAX - base - tensor.offset)
      fail("tensor address overflow: " + tensor.name);
    return base + tensor.offset;
  }

  void initialize(Plan &plan, const fs::path &ptxPath, int ordinal) {
    CUDA_CHECK(api.init(0));
    int count = 0;
    CUDA_CHECK(api.deviceCount(&count));
    if (!count || ordinal >= count)
      fail("requested CUDA device is unavailable (device count=" +
           std::to_string(count) + ")");
    CUDA_CHECK(api.deviceGet(&device, ordinal));
    char deviceName[256] = {};
    CUDA_CHECK(api.deviceName(deviceName, sizeof(deviceName), device));
    name = deviceName;
    CUDA_CHECK(api.deviceAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CUDA_CHECK(api.deviceAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CUDA_CHECK(api.deviceMemory(&totalMemory, device));
    CUDA_CHECK(api.driverVersion(&version));
    CUDA_CHECK(api.retainContext(&context, device));
    CUDA_CHECK(api.setContext(context));
    snapshotMemory("context-created");
    CUDA_CHECK(api.createStream(&stream, CU_STREAM_NON_BLOCKING));
    CUDA_CHECK(api.createEvent(&begin, CU_EVENT_DEFAULT));
    CUDA_CHECK(api.createEvent(&end, CU_EVENT_DEFAULT));

    auto ptx = llvm::MemoryBuffer::getFile(ptxPath.string());
    if (!ptx)
      fail("cannot read PTX: " + ptx.getError().message());
    if (!(*ptx)->getBuffer().contains(".version") ||
        !(*ptx)->getBuffer().contains(".target"))
      fail("kernel file is not textual PTX");
    char information[8192] = {}, errors[8192] = {};
    CUjit_option options[] = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void *values[] = {
        information, reinterpret_cast<void *>(uintptr_t(sizeof(information))),
        errors, reinterpret_cast<void *>(uintptr_t(sizeof(errors)))};
    CUresult status =
        api.loadModule(&module, (*ptx)->getBufferStart(), 4, options, values);
    information[sizeof(information) - 1] = 0;
    errors[sizeof(errors) - 1] = 0;
    jitLog = information;
    if (status != CUDA_SUCCESS) {
      if (errors[0])
        llvm::errs() << "CUDA PTX JIT: " << errors << '\n';
      try {
        api.check(status, "cuModuleLoadDataEx");
      } catch (const std::exception &error) {
        fail(std::string(error.what()) + "\nJIT: " + errors);
      }
    }
    snapshotMemory("module-loaded");
    size_t arena = plan.arenaBytes ? allocate(plan.arenaBytes) : 0;
    for (auto &tensor : plan.tensors) {
      tensor.buffer =
          tensor.storage == "arena" ? arena : allocate(tensor.bytes);
      address(tensor);
    }
    snapshotMemory("buffers-allocated");
    int maxGrid = 0, maxBlock = 0;
    CUDA_CHECK(api.deviceAttribute(&maxGrid, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                   device));
    CUDA_CHECK(api.deviceAttribute(
        &maxBlock, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device));
    for (const auto &dispatch : plan.dispatches) {
      CUfunction function = nullptr;
      CUDA_CHECK(api.getFunction(&function, module, dispatch.kernel.c_str()));
      KernelResources resource;
      CUDA_CHECK(api.functionAttribute(&resource.maxThreads,
                                       CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                       function));
      CUDA_CHECK(api.functionAttribute(&resource.registers,
                                       CU_FUNC_ATTRIBUTE_NUM_REGS, function));
      CUDA_CHECK(api.functionAttribute(&resource.sharedBytes,
                                       CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                       function));
      CUDA_CHECK(api.functionAttribute(&resource.localBytes,
                                       CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                       function));
      if (dispatch.gridSize > uint32_t(maxGrid) ||
          dispatch.blockSize > uint32_t(maxBlock) ||
          dispatch.blockSize > uint32_t(resource.maxThreads))
        fail("launch dimensions exceed device/kernel limits: " +
             dispatch.kernel);
      if (uint64_t(resource.sharedBytes) < dispatch.staticSharedBytes)
        fail("PTX JIT shared-memory allocation is smaller than the dispatch "
             "contract: " + dispatch.kernel);
      resources.push_back(resource);
      functions.push_back(function);
      arguments.emplace_back();
      for (size_t input : dispatch.inputs)
        arguments.back().push_back(address(plan.tensors[input]));
      arguments.back().push_back(address(plan.tensors[dispatch.output]));
    }
    for (auto &pointers : arguments) {
      parameters.emplace_back();
      for (auto &pointer : pointers)
        parameters.back().push_back(&pointer);
    }
  }

  void initializeProfile() {
    profile.resize(functions.size());
    for (auto &sample : profile) {
      CUDA_CHECK(api.createEvent(&sample.begin, CU_EVENT_DEFAULT));
      CUDA_CHECK(api.createEvent(&sample.end, CU_EVENT_DEFAULT));
    }
  }

  // A single stream executes dispatches in manifest order. This order is the
  // proof used by static arena reuse, so no per-kernel wait is needed normally.
  double launch(const Plan &plan, int iteration, bool warmup,
                std::ostream *trace, bool instrument = false) {
    completedDispatches = 0;
    CUDA_CHECK(api.recordEvent(begin, stream));
    for (size_t i = 0; i < plan.dispatches.size(); ++i) {
      const auto &dispatch = plan.dispatches[i];
      auto emit = [&](const char *event) {
        if (!trace)
          return;
        json::Object row{{"iteration", iteration},
                         {"warmup", warmup},
                         {"index", int64_t(i)},
                         {"kernel", dispatch.kernel},
                         {"event", event}};
        *trace << llvm::formatv("{0}\n", json::Value(std::move(row))).str();
        trace->flush();
        if (!*trace)
          fail("cannot write dispatch trace");
      };
      emit("begin");
      if (instrument)
        CUDA_CHECK(api.recordEvent(profile[i].begin, stream));
      CUDA_CHECK(api.launch(functions[i], dispatch.gridSize, 1, 1,
                            dispatch.blockSize, 1, 1, 0, stream,
                            parameters[i].data(), nullptr));
      if (instrument)
        CUDA_CHECK(api.recordEvent(profile[i].end, stream));
      if (trace) {
        CUDA_CHECK(api.synchronize(stream));
        completedDispatches = i + 1;
        emit("end");
      }
    }
    CUDA_CHECK(api.recordEvent(end, stream));
    CUDA_CHECK(api.waitEvent(end));
    completedDispatches = plan.dispatches.size();
    // The final event proves all preceding events on this stream completed.
    // No per-dispatch host wait is introduced by event profiling.
    if (instrument)
      for (auto &sample : profile) {
        float duration = 0;
        CUDA_CHECK(api.eventTime(&duration, sample.begin, sample.end));
        sample.sumMs += duration;
        sample.minMs = std::min(sample.minMs, double(duration));
        sample.maxMs = std::max(sample.maxMs, double(duration));
      }
    float duration = 0;
    CUDA_CHECK(api.eventTime(&duration, begin, end));
    return duration;
  }

  json::Array resourceReport(const Plan &plan) const {
    json::Array rows;
    for (size_t i = 0; i < plan.dispatches.size(); ++i) {
      const auto &dispatch = plan.dispatches[i];
      const auto &resource = resources[i];
      rows.push_back(json::Object{
          {"index", int64_t(i)}, {"kernel", dispatch.kernel},
          {"op", dispatch.op}, {"source_node", dispatch.source},
          {"schedule", dispatch.cudaSchedule},
          {"grid", json::Array{dispatch.gridSize, 1, 1}},
          {"block", json::Array{dispatch.blockSize, 1, 1}},
          {"registers_per_thread", resource.registers},
          {"static_shared_bytes_per_block", resource.sharedBytes},
          {"dynamic_shared_bytes_per_block", 0},
          {"local_bytes_per_thread", resource.localBytes},
          {"max_threads_per_block", resource.maxThreads}});
    }
    return rows;
  }

  json::Object profileReport(const Plan &plan, int repeat,
                             json::Array sequenceSamples, double wallMs) const {
    struct Aggregate { int64_t count = 0; double sumMs = 0; };
    std::map<std::string, Aggregate> operators;
    json::Array rows, operatorRows;
    double totalMs = 0;
    for (const auto &sample : profile)
      totalMs += sample.sumMs;
    for (size_t i = 0; i < plan.dispatches.size(); ++i) {
      const auto &dispatch = plan.dispatches[i];
      const auto &sample = profile[i];
      auto &aggregate = operators[dispatch.op];
      ++aggregate.count;
      aggregate.sumMs += sample.sumMs;
      rows.push_back(json::Object{
          {"index", int64_t(i)}, {"kernel", dispatch.kernel},
          {"op", dispatch.op}, {"source_node", dispatch.source},
          {"invocations", repeat},
          {"mean_ms", sample.sumMs / repeat},
          {"min_ms", sample.minMs}, {"max_ms", sample.maxMs},
          {"share_of_profiled_dispatch_time",
           totalMs ? sample.sumMs / totalMs : 0}});
    }
    for (const auto &[op, aggregate] : operators)
      operatorRows.push_back(json::Object{
          {"op", op}, {"dispatch_count", aggregate.count},
          {"invocations", aggregate.count * repeat},
          {"mean_total_ms_per_sequence", aggregate.sumMs / repeat},
          {"mean_ms_per_invocation", aggregate.sumMs / (aggregate.count * repeat)},
          {"share_of_profiled_dispatch_time",
           totalMs ? aggregate.sumMs / totalMs : 0}});
    std::sort(operatorRows.begin(), operatorRows.end(),
              [](const json::Value &left, const json::Value &right) {
                return *left.getAsObject()->getNumber("mean_total_ms_per_sequence") >
                       *right.getAsObject()->getNumber("mean_total_ms_per_sequence");
              });
    return json::Object{
        {"measured_runs", repeat},
        {"timing_kind", "cuda-events-per-dispatch-separate-instrumented-runs"},
        {"timing_note",
         "Profiling runs execute after uninstrumented timing. Two timing events "
         "bracket each dispatch on the same stream; only the final sequence "
         "event is synchronized. Event and submission overhead can perturb "
         "short kernels. These are elapsed intervals, not hardware cycle, "
         "occupancy, bank-conflict or bandwidth counters. Use gpu_dispatch_ms "
         "for uninstrumented sequence comparisons."},
        {"sum_dispatch_mean_ms", totalMs / repeat},
        {"sequence_samples_ms", std::move(sequenceSamples)},
        {"host_wall_ms", wallMs}, {"by_op", std::move(operatorRows)},
        {"dispatches", std::move(rows)}};
  }
};
} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    llvm::errs()
        << "usage: gac-cuda-run MANIFEST.json KERNEL.ptx DATA_DIR OUTPUT_DIR "
           "[--device N] [--repeat N] [--warmup N] "
           "[--trace-dispatches | --profile-dispatches]\n";
    return 1;
  }
  fs::path outputDirectory(argv[4]);
  std::string stage = "arguments";
  size_t completedDispatches = 0;
  auto started = Clock::now();
  try {
    Options options = parseOptions(argc, argv);
    fs::create_directories(outputDirectory);
    stage = "manifest-validation";
    Plan plan = parsePlan(argv[1], Target::CUDA);
    uint64_t plannedBufferBytes = plan.arenaBytes;
    for (const auto &tensor : plan.tensors)
      if (tensor.storage != "arena")
        plannedBufferBytes += tensor.bytes;
    stage = "driver-initialization-and-ptx-jit";
    Device device;
    device.initialize(plan, argv[2], options.device);
    double initialization = elapsed(started);
    stage = "upload";
    json::Array inputFiles, outputFiles;
    double upload = 0, inputIO = 0, download = 0, outputIO = 0;
    for (size_t i = 0; i < plan.tensors.size(); ++i) {
      const auto &tensor = plan.tensors[i];
      if (tensor.storage == "arena")
        continue;
      std::string file = tensor.storage == "constant"
                             ? tensor.file
                             : "input_" + std::to_string(i) + ".bin";
      auto ioStart = Clock::now();
      Bytes bytes = readBytes(fs::path(argv[3]) / file, tensor.bytes);
      inputIO += elapsed(ioStart);
      auto transferStart = Clock::now();
      device.api.check(
          device.api.upload(device.address(tensor), bytes.data(), bytes.size()),
          "cuMemcpyHtoD");
      upload += elapsed(transferStart);
      if (tensor.storage == "input")
        inputFiles.push_back(json::Object{{"name", tensor.name},
                                          {"file", file},
                                          {"tensor_index", int64_t(i)}});
    }
    // Pageable host-to-device copies may return after staging but before DMA
    // has completed. Drain them once before submitting to our nonblocking
    // stream; freeing the host staging vectors is safe for synchronous copies.
    auto uploadDrainStarted = Clock::now();
    device.api.check(device.api.synchronizeContext(),
                     "cuCtxSynchronize(upload)");
    upload += elapsed(uploadDrainStarted);
    std::ofstream trace;
    if (options.trace) {
      trace.open(outputDirectory / "dispatch-progress.jsonl");
      if (!trace)
        fail("cannot open dispatch trace");
    }
    json::Array samples;
    json::Object dispatchProfile;
    double sum = 0;
    double dispatchWall = 0;
    try {
      stage = "warmup";
      for (int i = 0; i < options.warmup; ++i)
        device.launch(plan, i, true, options.trace ? &trace : nullptr);
      stage = "dispatch";
      auto dispatchStarted = Clock::now();
      for (int i = 0; i < options.repeat; ++i) {
        double duration =
            device.launch(plan, i, false, options.trace ? &trace : nullptr);
        sum += duration;
        samples.push_back(duration);
      }
      dispatchWall = elapsed(dispatchStarted);
      if (options.profile) {
        stage = "dispatch-profiling";
        // Allocate events only after the normal measurements. Their GPU memory
        // and command overhead must not affect the uninstrumented sample set.
        device.initializeProfile();
        json::Array profileSamples;
        auto profileStarted = Clock::now();
        for (int i = 0; i < options.repeat; ++i)
          profileSamples.push_back(device.launch(plan, i, false, nullptr, true));
        dispatchProfile = device.profileReport(plan, options.repeat,
                                              std::move(profileSamples),
                                              elapsed(profileStarted));
      }
    } catch (...) {
      completedDispatches = device.completedDispatches;
      throw;
    }
    completedDispatches = device.completedDispatches;
    device.snapshotMemory("dispatches-completed");
    stage = "download";
    for (size_t i = 0; i < plan.outputs.size(); ++i) {
      size_t id = plan.outputs[i];
      const auto &tensor = plan.tensors[id];
      Bytes bytes(size_t(tensor.bytes), 0);
      auto transferStarted = Clock::now();
      device.api.check(device.api.download(bytes.data(), device.address(tensor),
                                           bytes.size()),
                       "cuMemcpyDtoH");
      download += elapsed(transferStarted);
      std::string file = "output_" + std::to_string(i) + ".bin";
      auto ioStarted = Clock::now();
      writeBytes(outputDirectory / file, bytes);
      outputIO += elapsed(ioStarted);
      json::Array shape;
      for (auto dim : tensor.shape)
        shape.push_back(dim);
      outputFiles.push_back(json::Object{{"name", tensor.name},
                                         {"file", file},
                                         {"tensor_index", int64_t(id)},
                                         {"dtype", tensor.dtype},
                                         {"shape", std::move(shape)}});
    }
    stage = "report";
    writeReport(
        outputDirectory,
        json::Object{
            {"schema_version", 1},
            {"status", "completed"},
            {"execution", "cuda-device"},
            {"host_operator_fallback", false},
            {"device_ordinal", options.device},
            {"gpu_name", device.name},
            {"compute_capability",
             std::to_string(device.major) + "." + std::to_string(device.minor)},
            {"driver_version", device.version},
            {"device_memory_bytes", int64_t(device.totalMemory)},
            {"dispatches", int64_t(plan.dispatches.size())},
            {"completed_dispatches", int64_t(completedDispatches)},
            {"arena_bytes", int64_t(plan.arenaBytes)},
            {"embedded_constant_bytes", int64_t(plan.embeddedConstantBytes)},
            {"planned_buffer_bytes", int64_t(plannedBufferBytes)},
            {"static_allocated_tensor_bytes", int64_t(plannedBufferBytes)},
            {"memory_note",
             "Static device buffer allocations only; excludes CUDA context, "
             "PTX module/JIT, embedded constants and possible local-memory "
             "spills. This is not a measured GPU-memory peak."},
            {"device_memory_snapshots", std::move(device.memorySnapshots)},
            {"memory_snapshot_note",
             "cuMemGetInfo point samples of the whole device; may include "
             "other processes and driver allocations. These samples are not "
             "this process's measured peak."},
            {"warmup_runs", options.warmup},
            {"measured_runs", options.repeat},
            {"trace_dispatches", options.trace},
            {"profile_dispatches", options.profile},
            {"kernel_resources", device.resourceReport(plan)},
            {"kernel_resources_note",
             "CUDA Driver function attributes after PTX JIT. Local-memory size "
             "can include stack, spills and compiler-managed arrays; it does "
             "not measure executed spill traffic or achieved occupancy."},
            {"dispatch_profile", std::move(dispatchProfile)},
            {"inputs", std::move(inputFiles)},
            {"outputs", std::move(outputFiles)},
            {"scheduling", "single-cuda-stream-manifest-order"},
            {"jit_log", device.jitLog},
            {"timing_kind", "cuda-events-whole-dispatch-sequence"},
            {"timing_note",
             "GPU interval includes device work and any host submission gaps; "
             "excludes PTX JIT, transfers, file I/O and warmup. Trace mode "
             "synchronizes every dispatch and is not a throughput measurement. "
             "Transfers use host API wall time."},
            {"gpu_dispatch_ms", sum / options.repeat},
            {"gpu_dispatch_samples_ms", std::move(samples)},
            {"initialization_ms", initialization},
            {"upload_ms", upload},
            {"download_ms", download},
            {"input_file_io_ms", inputIO},
            {"output_file_io_ms", outputIO},
            {"dispatch_wall_ms", dispatchWall},
            {"total_host_wall_ms", elapsed(started)}});
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "gac-cuda-run [" << stage << "]: " << error.what() << '\n';
    try {
      fs::create_directories(outputDirectory);
      writeReport(
          outputDirectory,
          json::Object{{"schema_version", 1},
                       {"status", "failed"},
                       {"execution", "cuda-device"},
                       {"host_operator_fallback", false},
                       {"failure_stage", stage},
                       {"error", error.what()},
                       {"completed_dispatches", int64_t(completedDispatches)},
                       {"total_host_wall_ms", elapsed(started)}});
    } catch (
        ...) { /* Preserve the original error if its report cannot be saved. */
    }
    return 1;
  }
}
