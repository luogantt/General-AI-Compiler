"""Generate the single-device Vortex host for a complete static tensor graph.

All tensor data remains on the selected Vortex device between dispatches.  The
host owns external tensors and resident constants, while a single byte arena
implements the compiler's intermediate-buffer plan.  This is deliberately a
correctness-first, ordered schedule: even independent nodes depend on the prior
dispatch, because arena lifetimes were computed for that order.
"""

from __future__ import annotations

import json
import re
from typing import Any

from gac.core.errors import BackendError
from gac.core.ir import Graph
from gac.compiler.memory import MemoryPlan


def _cpp_string(value: str) -> str:
    # Fixed-width octal UTF-8 bytes also handle non-BMP Unicode and control
    # characters, which JSON's \u escapes cannot represent in C++ literals.
    escaped = []
    for byte in value.encode("utf-8"):
        if byte == 34:
            escaped.append(r'\"')
        elif byte == 92:
            escaped.append(r"\\")
        elif 32 <= byte < 127:
            escaped.append(chr(byte))
        else:
            escaped.append(f"\\{byte:03o}")
    return '"' + "".join(escaped) + '"'


def emit_graph_host(
    graph: Graph, memory_plan: MemoryPlan, dispatches: list[dict[str, Any]]
) -> str:
    """Emit a raw-byte host ABI, preserving every tensor's native dtype.

    The device argument blob is a packed array of little-endian uint64 tensor
    addresses, inputs first and output last.  Device kernels narrow each address
    to the RV32 pointer type.  Shape and operator attributes are specialized in
    the generated MLIR rather than passed through this blob.
    """
    graph.validate()
    memory_plan.validate()
    if memory_plan.graph_name != graph.name:
        raise BackendError("Vortex graph host received a different graph's memory plan")
    allocations = list(memory_plan.allocations)
    values = {item.name: index for index, item in enumerate(allocations)}
    expected = set(graph.inputs) | set(graph.constants)
    expected.update(name for node in graph.nodes for name in node.outputs)
    if set(values) != expected:
        raise BackendError("Vortex graph host memory plan does not cover every tensor")

    buffers: list[tuple[str, int, str]] = []
    arena_index: int | None = None
    if any(item.storage == "arena" for item in allocations):
        arena_index = len(buffers)
        buffers.append(("intermediate arena", memory_plan.arena_size, "VX_MEM_READ_WRITE"))
    buffer_rows: list[str] = []
    for item in allocations:
        if item.storage == "arena":
            index, offset = arena_index, item.offset
        else:
            index, offset = len(buffers), 0
            # Input tensors can be forwarded as outputs, and graph outputs can
            # be consumed by later nodes.  READ_WRITE is the conservative ABI.
            mode = "VX_MEM_READ" if item.resident else "VX_MEM_READ_WRITE"
            buffers.append((item.name, item.size_bytes, mode))
        buffer_rows.append(
            f"  {{{_cpp_string(item.name)}, {index}u, {offset or 0}ull, {item.size_bytes}ull}}"
        )

    kernel_names: list[str] = []
    dispatch_rows: list[str] = []
    for dispatch in dispatches:
        symbol = dispatch["kernel"]
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol):
            raise BackendError("invalid Vortex graph kernel symbol")
        if symbol not in kernel_names:
            kernel_names.append(symbol)
        names = list(dispatch["inputs"]) + [dispatch["output"]]
        if any(name not in values for name in names):
            raise BackendError(f"Vortex dispatch {symbol!r} references an unplanned tensor")
        elements = dispatch["elements"]
        if isinstance(elements, bool) or not isinstance(elements, int) or not 0 <= elements <= 0xFFFFFFFF:
            raise BackendError(f"Vortex dispatch {symbol!r} exceeds the RV32 launch index range")
        arguments = ", ".join(f"{values[name]}u" for name in names)
        dispatch_rows.append(
            f"  {{{kernel_names.index(symbol)}u, {elements}u, {{{arguments}}}}}"
        )

    buffer_specs = ",\n".join(
        f"  {{{_cpp_string(name)}, {size}ull, {mode}}}" for name, size, mode in buffers
    )
    constants = ",\n".join(
        f'  {{{values[name]}u, "constants/constant_{index:04d}.bin"}}'
        for index, name in enumerate(graph.constants)
    )
    inputs = ",\n".join(
        f'  {{{values[name]}u, "inputs/input_{index:04d}.bin"}}'
        for index, name in enumerate(graph.inputs)
    )
    outputs = ",\n".join(
        f'  {{{values[name]}u, "outputs/output_{index:04d}.bin"}}'
        for index, name in enumerate(graph.outputs)
    )
    substitutions = {
        "@BUFFER_SPECS@": buffer_specs,
        "@VALUES@": ",\n".join(buffer_rows),
        "@CONSTANTS@": constants,
        "@INPUTS@": inputs,
        "@OUTPUTS@": outputs,
        "@KERNELS@": ",\n".join(f"  {_cpp_string(name)}" for name in kernel_names),
        "@DISPATCHES@": ",\n".join(dispatch_rows),
        "@GRAPH_NAME_JSON@": _cpp_string(json.dumps(graph.name, ensure_ascii=True)),
        "@ARENA_BYTES@": str(memory_plan.arena_size),
    }
    source = _HOST_SOURCE
    for key, value in substitutions.items():
        source = source.replace(key, value)
    return source


_HOST_SOURCE = r'''#include <vortex2.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<uint8_t>;

void check(vx_result_t result, const char* expression) {
  if (result != VX_SUCCESS)
    throw std::runtime_error(std::string(expression) + ": " + vx_result_string(result));
}
#define CHECK(expression) check((expression), #expression)

struct BufferSpec { const char* name; uint64_t bytes; uint32_t flags; };
struct Value { const char* name; size_t buffer; uint64_t offset; uint64_t bytes; };
struct File { size_t value; const char* path; };
struct Dispatch { size_t kernel; uint32_t elements; std::vector<size_t> values; };

const std::vector<BufferSpec> kBuffers = {
@BUFFER_SPECS@
};
const std::vector<Value> kValues = {
@VALUES@
};
const std::vector<File> kConstants = {
@CONSTANTS@
};
const std::vector<File> kInputs = {
@INPUTS@
};
const std::vector<File> kOutputs = {
@OUTPUTS@
};
const std::vector<const char*> kKernelNames = {
@KERNELS@
};
const std::vector<Dispatch> kDispatches = {
@DISPATCHES@
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

uint32_t parse_u32(const char* text, const std::string& option) {
  const std::string value(text);
  if (value.empty()) throw std::runtime_error(option + " needs an unsigned integer");
  uint64_t result = 0;
  for (const char digit : value) {
    if (digit < '0' || digit > '9')
      throw std::runtime_error(option + " needs decimal digits only");
    result = result * 10 + static_cast<uint32_t>(digit - '0');
    if (result > std::numeric_limits<uint32_t>::max())
      throw std::runtime_error(option + " exceeds uint32");
  }
  return static_cast<uint32_t>(result);
}

std::filesystem::path data_path(const std::string& root, const char* path) {
  return std::filesystem::path(root) / path;
}

Bytes read_bytes(const std::filesystem::path& path, uint64_t bytes) {
  if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
      bytes > std::numeric_limits<size_t>::max())
    throw std::runtime_error("host file size is too large: " + path.string());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() != static_cast<std::streamsize>(bytes))
    throw std::runtime_error("missing file or incorrect byte size: " + path.string());
  stream.seekg(0);
  Bytes result(static_cast<size_t>(bytes));
  if (bytes && !stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("cannot read " + path.string());
  return result;
}

void write_bytes(const std::filesystem::path& path, const Bytes& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream || (bytes.size() && !stream.write(reinterpret_cast<const char*>(bytes.data()),
                                               static_cast<std::streamsize>(bytes.size()))))
    throw std::runtime_error("cannot write " + path.string());
}

// Resource ownership survives exceptions: wait before freeing memory that may
// still be accessed by a queued command, then release in dependency order.
struct Device {
  vx_device_h device = nullptr;
  vx_queue_h queue = nullptr;
  vx_module_h module = nullptr;
  std::vector<vx_buffer_h> buffers;
  std::vector<uint64_t> addresses;
  std::vector<vx_kernel_h> kernels;
  std::vector<vx_event_h> events;

  void release_events() noexcept {
    for (auto event : events) if (event) vx_event_release(event);
    events.clear();
  }
  ~Device() {
    if (queue) vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
    release_events();
    for (auto kernel : kernels) if (kernel) vx_kernel_release(kernel);
    if (module) vx_module_release(module);
    for (auto buffer : buffers) if (buffer) vx_buffer_release(buffer);
    if (queue) vx_queue_release(queue);
    if (device) vx_device_release(device);
  }

  uint64_t address(size_t index) const {
    const Value& value = kValues.at(index);
    const uint64_t base = addresses.at(value.buffer);
    if (base > std::numeric_limits<uint32_t>::max() ||
        value.offset > std::numeric_limits<uint32_t>::max() - base ||
        value.bytes > (uint64_t{1} << 32) - base - value.offset)
      throw std::runtime_error(std::string("tensor exceeds RV32 address space: ") + value.name);
    return base + value.offset;
  }

  void upload(size_t index, const Bytes& data) {
    const Value& value = kValues.at(index);
    if (data.size() != value.bytes) throw std::runtime_error("upload size mismatch");
    if (data.empty()) return;
    events.push_back(nullptr);
    try {
      CHECK(vx_enqueue_write(queue, buffers.at(value.buffer), value.offset, data.data(),
                             data.size(), 0, nullptr, &events.back()));
      CHECK(vx_event_wait_value(events.back(), 1, VX_TIMEOUT_INFINITE));
    } catch (...) {
      // A temporary weight staging vector may be destroyed by the caller as
      // soon as this throws.  Drain before it can leave scope.
      vx_queue_finish(queue, VX_TIMEOUT_INFINITE);
      throw;
    }
    release_events();
  }

  void initialize(const std::string& kernel) {
    CHECK(vx_device_open(0, &device));
    vx_queue_info_t info = {};
    info.struct_size = sizeof(info);
    info.priority = VX_QUEUE_PRIORITY_NORMAL;
    CHECK(vx_queue_create(device, &info, &queue));
    for (const auto& spec : kBuffers) {
      buffers.push_back(nullptr);
      CHECK(vx_buffer_create(device, std::max(uint64_t{1}, spec.bytes), spec.flags, &buffers.back()));
      addresses.push_back(0);
      CHECK(vx_buffer_address(buffers.back(), &addresses.back()));
    }
    for (size_t i = 0; i < kValues.size(); ++i) address(i);
    CHECK(vx_module_load_file(device, kernel.c_str(), &module));
    for (const char* name : kKernelNames) {
      kernels.push_back(nullptr);
      CHECK(vx_module_get_kernel(module, name, &kernels.back()));
    }
  }
};

struct Timing { double h2d = 0, dispatch = 0, d2h = 0, total = 0; };

Timing run(Device& device, uint32_t threads, const std::vector<Bytes>& inputs,
           std::vector<Bytes>& outputs) {
  Timing result;
  const auto begin = Clock::now();
  for (size_t i = 0; i < inputs.size(); ++i) device.upload(kInputs[i].value, inputs[i]);
  const auto uploaded = Clock::now();

  // The blobs must outlive asynchronous enqueueing.  They contain device
  // addresses only; host pointers and dtype conversions never enter the ABI.
  std::vector<std::vector<uint64_t>> blobs;
  blobs.reserve(kDispatches.size());
  struct DrainOnError {
    vx_queue_h queue;
    bool active = true;
    ~DrainOnError() { if (active) vx_queue_finish(queue, VX_TIMEOUT_INFINITE); }
  } pending{device.queue};
  device.events.reserve(kDispatches.size() + kOutputs.size());
  vx_event_h prior = nullptr;
  for (const auto& dispatch : kDispatches) {
    if (dispatch.elements == 0) continue;
    blobs.emplace_back();
    for (size_t index : dispatch.values) blobs.back().push_back(device.address(index));
    vx_launch_info_t info = {};
    info.struct_size = sizeof(info);
    info.kernel = device.kernels.at(dispatch.kernel);
    info.args_host = blobs.back().data();
    info.args_size = blobs.back().size() * sizeof(uint64_t);
    info.ndim = 1;
    info.grid_dim[0] = static_cast<uint32_t>((uint64_t{dispatch.elements} + threads - 1) / threads);
    info.block_dim[0] = threads;
    device.events.push_back(nullptr);
    CHECK(vx_enqueue_launch(device.queue, &info, prior ? 1 : 0,
                            prior ? &prior : nullptr, &device.events.back()));
    prior = device.events.back();
  }
  if (prior) CHECK(vx_event_wait_value(prior, 1, VX_TIMEOUT_INFINITE));
  const auto computed = Clock::now();

  for (size_t i = 0; i < kOutputs.size(); ++i) {
    const Value& value = kValues.at(kOutputs[i].value);
    outputs[i].resize(static_cast<size_t>(value.bytes));
    if (!value.bytes) continue;
    device.events.push_back(nullptr);
    CHECK(vx_enqueue_read(device.queue, outputs[i].data(), device.buffers.at(value.buffer),
                          value.offset, value.bytes, prior ? 1 : 0,
                          prior ? &prior : nullptr, &device.events.back()));
    CHECK(vx_event_wait_value(device.events.back(), 1, VX_TIMEOUT_INFINITE));
  }
  const auto end = Clock::now();
  device.release_events();
  pending.active = false;
  result.h2d = milliseconds(begin, uploaded);
  result.dispatch = milliseconds(uploaded, computed);
  result.d2h = milliseconds(computed, end);
  result.total = milliseconds(begin, end);
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data_dir = ".", kernel = "kernel.vxbin";
    uint32_t threads = 32, repeat = 1, warmup = 0;
    bool explicit_threads = false;
    for (int i = 1; i < argc; ++i) {
      const std::string option(argv[i]);
      if (i + 1 == argc) throw std::runtime_error("missing value for " + option);
      if (option == "--data-dir") data_dir = argv[++i];
      else if (option == "--kernel") kernel = argv[++i];
      else if (option == "--threads") {
        threads = parse_u32(argv[++i], option);
        explicit_threads = true;
      }
      else if (option == "--repeat") repeat = parse_u32(argv[++i], option);
      else if (option == "--warmup") warmup = parse_u32(argv[++i], option);
      else throw std::runtime_error("unknown option: " + option);
    }
    if (!threads || !repeat) throw std::runtime_error("threads and repeat must be positive");
    Device device;
    device.initialize(kernel);
    uint64_t warp_threads = 0, warps = 0;
    CHECK(vx_device_query(device.device, VX_CAPS_NUM_THREADS, &warp_threads));
    CHECK(vx_device_query(device.device, VX_CAPS_NUM_WARPS, &warps));
    if (!warp_threads || !warps || warp_threads > std::numeric_limits<uint32_t>::max() / warps)
      throw std::runtime_error("invalid device thread capacity");
    const uint64_t capacity = warp_threads * warps;
    if (!explicit_threads) threads = static_cast<uint32_t>(std::min(uint64_t{32}, capacity));
    if (threads > capacity)
      throw std::runtime_error("--threads exceeds the device's per-core CTA capacity");
    // Stage one constant at a time.  Completed uploads remain resident, and
    // their potentially large host byte vectors are immediately destroyed.
    for (const auto& file : kConstants)
      device.upload(file.value, read_bytes(data_path(data_dir, file.path), kValues[file.value].bytes));
    std::vector<Bytes> inputs, outputs(kOutputs.size());
    for (const auto& file : kInputs)
      inputs.push_back(read_bytes(data_path(data_dir, file.path), kValues[file.value].bytes));
    for (uint32_t i = 0; i < warmup; ++i) run(device, threads, inputs, outputs);
    Timing sum;
    for (uint32_t i = 0; i < repeat; ++i) {
      const Timing result = run(device, threads, inputs, outputs);
      sum.h2d += result.h2d;
      sum.dispatch += result.dispatch;
      sum.d2h += result.d2h;
      sum.total += result.total;
    }
    for (size_t i = 0; i < outputs.size(); ++i)
      write_bytes(data_path(data_dir, kOutputs[i].path), outputs[i]);
    std::printf("{\n"
                "  \"schema_version\": 1,\n"
                "  \"backend\": \"vortex-full-graph\",\n"
                "  \"graph\": %s,\n"
                "  \"timing_kind\": \"host_wall_including_driver_or_simulation\",\n"
                "  \"hardware_inference_latency_measured\": false,\n"
                "  \"device_count\": 1,\n"
                "  \"dispatch_count\": %zu,\n"
                "  \"arena_bytes\": @ARENA_BYTES@,\n"
                "  \"threads\": %u,\n"
                "  \"warmup\": %u,\n"
                "  \"repeat\": %u,\n"
                "  \"h2d_host_wall_mean_ms\": %.6f,\n"
                "  \"dispatch_host_wall_mean_ms\": %.6f,\n"
                "  \"d2h_host_wall_mean_ms\": %.6f,\n"
                "  \"end_to_end_host_wall_mean_ms\": %.6f,\n"
                "  \"numerical_verification\": \"run verify_outputs.py separately\"\n"
                "}\n", @GRAPH_NAME_JSON@, kDispatches.size(), threads, warmup, repeat,
                sum.h2d / repeat, sum.dispatch / repeat, sum.d2h / repeat, sum.total / repeat);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gac-vortex-full-graph: %s\n", error.what());
    return 2;
  }
}
'''


def emit_graph_makefile() -> str:
    """Use one revision-pinned MLIR/VOLT LLVM tree through machine code.

    Like existing artifacts, place this directory under ``build/tests/regression``
    or pass ``ROOT_DIR`` and ``VORTEX_HOME`` for another artifact location.
    """
    from gac.targets.vortex.ffn import _direct_makefile

    source = _direct_makefile()
    source = source.replace("PROJECT := gac_bevformer_vortex", "PROJECT := gac_bevformer_full_vortex")
    source = source.replace("VX_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/model_config.h", "VX_HDRS :=")
    source = source.replace("OPTS ?= --data-dir . --kernel kernel.vxbin --warmup 1 --repeat 3", "OPTS ?= --data-dir . --kernel kernel.vxbin --warmup 0 --repeat 1")
    source = source.replace("include ../common.mk", "include $(ROOT_DIR)/tests/regression/common.mk")
    source = source.replace("$(MLIR_OPT) --verify-each --canonicalize --cse $< -o $@", "$(MLIR_OPT) --verify-each --canonicalize --cse --convert-scf-to-cf --convert-math-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts $< -o $@")
    # std::filesystem handles creating the raw-output directory portably.
    source = source.replace("SRCS := $(SRC_DIR)/main.cpp", "SRCS := $(SRC_DIR)/main.cpp\nCXXFLAGS += -std=c++17")
    # Vortex's direct entry/startup convention uses x31 as its reserved scratch
    # register, matching the already validated direct-MLIR FFN artifacts.
    return source + "\nVX_CFLAGS += -ffixed-x31\n"


def emit_graph_verifier() -> str:
    """Emit a standalone, streaming source-artifact integrity verifier.

    This checks the exact emitted source-file contract and declared tensor
    files. Derived ELF/object/vxbin files and runtime outputs deliberately are
    not source inputs; the Makefile rebuilds them from the verified MLIR.
    """
    return r'''#!/usr/bin/env python3
"""Verify a GAC whole-graph Vortex artifact before building or executing it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

REQUIRED = {
    "01-import.json", "02-graph.json", "03-passes.json", "04-capabilities.json",
    "05-dispatch-dag.json", "06-templates.json", "07-tiling.json", "08-schedule.json",
    "10-runtime.json", "kernel.mlir", "main.cpp", "Makefile", "verify_artifact.py",
}
OPTIONAL = {"kernel.lowered.mlir", "kernel.native.mlir", "verify_outputs.py"}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant {value!r}")


def tensor_files(records, role):
    if not isinstance(records, list):
        raise ValueError(f"{role} tensor metadata must be an array")
    result = {}
    names = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"invalid {role} tensor record")
        name = record["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"duplicate or invalid {role} tensor name")
        names.add(name)
        filename = f"{role}/{role[:-1]}_{index:04d}.bin"
        if record["file"] != filename:
            raise ValueError(f"unexpected {role} tensor filename {record['file']!r}")
        shape = record["shape"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"invalid shape for {name!r}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        size = elements * {"float16": 2, "float32": 4, "int32": 4, "bool": 1}[record["dtype"]]
        if type(record["size_bytes"]) is not int or record["size_bytes"] != size:
            raise ValueError(f"tensor byte size disagrees with shape/dtype for {name!r}")
        result[filename] = size
    return result


def safe_file(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid artifact filename")
    relative = PurePosixPath(name)
    if relative.is_absolute() or str(relative) != name or any(p in {".", ".."} for p in relative.parts):
        raise ValueError(f"unsafe artifact path {name!r}")
    path = root
    for component in relative.parts:
        path = path / component
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {name!r}")
    if not path.is_file():
        raise ValueError(f"missing artifact file {name!r}")
    return path


def main():
    root = Path(__file__).resolve().parent
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"),
                              object_pairs_hook=reject_duplicates, parse_constant=reject_nonfinite)
        if manifest.get("schema_version") != 1 or manifest.get("compiler_backend") != "vortex-whole-graph":
            raise ValueError("unsupported Vortex graph manifest")
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("manifest files must be a non-empty object")
        tensors = manifest["tensor_files"]
        inputs = tensor_files(tensors["inputs"], "inputs")
        constants = tensor_files(tensors["constants"], "constants")
        outputs = tensor_files(tensors["outputs"], "outputs")
        goldens = {name.replace("outputs/output_", "goldens/golden_"): size
                   for name, size in outputs.items()}
        required = REQUIRED | set(constants)
        allowed = required | OPTIONAL | set(inputs) | set(goldens)
        if not required <= set(files) or not set(files) <= allowed:
            raise ValueError(f"artifact file set mismatch: missing={sorted(required - set(files))}, "
                             f"unexpected={sorted(set(files) - allowed)}")
        for group, label in ((inputs, "input"), (goldens, "golden")):
            present = set(files) & set(group)
            if present and present != set(group):
                raise ValueError(f"incomplete {label} tensor file set")
        expected_sizes = {**inputs, **constants, **goldens}
        for name, record in files.items():
            path = safe_file(root, name)
            if not isinstance(record, dict) or set(record) != {"size_bytes", "sha256"}:
                raise ValueError(f"invalid file metadata for {name!r}")
            size, digest = record["size_bytes"], record["sha256"]
            if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid size or SHA-256 for {name!r}")
            if path.stat().st_size != size or (name in expected_sizes and expected_sizes[name] != size):
                raise ValueError(f"byte size mismatch for {name!r}")
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
            if actual.hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch for {name!r}")
        if files["kernel.mlir"]["sha256"] != manifest["source_sha256"]:
            raise ValueError("device MLIR is not bound to manifest source_sha256")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"gac-vortex graph artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifact_verified": True, "file_count": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
