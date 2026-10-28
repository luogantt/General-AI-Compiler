// Native dispatch contract shared by device runtimes. Parsing also proves
// single-stream dependencies and inclusive arena-buffer lifetimes before any
// device code is loaded or launched.
#pragma once
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  std::string op, source;
  std::vector<size_t> inputs;
  size_t output;
  uint32_t elements;
  uint32_t blockSize = 0, gridSize = 0;
  // A Triton program covers a logical tile, potentially larger than its
  // physical CTA. Its grid is ceil(elements / block_elements), not / threads.
  bool triton = false;
  std::string tritonKind, tritonAbi;
  uint32_t tritonBlockElements = 0, tritonNumWarps = 0;
  uint32_t tritonReductionWidth = 0, tritonReductionTile = 0;
  uint32_t tritonSoftmaxRows = 0, tritonSoftmaxWidth = 0;
  uint32_t tritonSoftmaxN = 0;
  uint32_t tritonSoftmaxTile = 0;
  uint32_t tritonLayerNormRows = 0, tritonLayerNormWidth = 0;
  uint32_t tritonLayerNormN = 0, tritonLayerNormTile = 0;
  uint32_t tritonLayerNormEpsilonF32Bits = 0;
  std::string cudaSchedule;
  uint32_t staticSharedBytes = 0;
  // Empty schedule preserves the original scalar global-memory contract.
  std::string vortexSchedule;
  uint32_t m = 0, n = 0, k = 0, batches = 0;
  uint32_t tileM = 0, tileN = 0, tileK = 0;
  std::string tritonMatmulBatchMode;
  uint32_t vortexThreadTileM = 1, vortexThreadTileN = 1;
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

inline void validateTritonSchedule(const json::Object &schedule,
                                  Dispatch &dispatch, const Plan &plan) {
  auto powerOfTwo = [&](llvm::StringRef name, uint64_t minimum,
                        uint64_t maximum) {
    uint64_t value = nonnegative(schedule, name);
    if (value < minimum || value > maximum || (value & (value - 1)))
      fail("unsupported Triton Vortex dimension: " + name.str());
    return uint32_t(value);
  };
  dispatch.tritonKind = string(schedule, "kind");
  dispatch.tritonAbi = string(schedule, "abi");
  const std::string expectedAbi = dispatch.tritonKind == "softmax"
                                      ? "vortex-u64-pointers-i32-n"
                                  : dispatch.tritonKind == "layer_norm"
                                      ? "vortex-u64-pointers-i32-n-f32-epsilon"
                                      : "vortex-u64-pointers";
  if (dispatch.tritonAbi != expectedAbi)
    fail("unsupported Triton Vortex kernel ABI");
  dispatch.tritonNumWarps = powerOfTwo("num_warps", 1, 4);
  dispatch.lanes = powerOfTwo("lanes", 1, 32);
  uint64_t block = nonnegative(schedule, "block_size");
  uint64_t grid = nonnegative(schedule, "grid_size");
  if (!grid || grid > INT32_MAX ||
      block != uint64_t(dispatch.tritonNumWarps) * dispatch.lanes)
    fail("Triton Vortex physical launch geometry disagrees");
  dispatch.blockSize = uint32_t(block);
  dispatch.gridSize = uint32_t(grid);
  const Tensor &output = plan.tensors[dispatch.output];
  if (dispatch.tritonKind == "pointwise") {
    dispatch.tritonBlockElements = powerOfTwo("block_elements", 16, 1024);
    if (grid !=
        (uint64_t(dispatch.elements) + dispatch.tritonBlockElements - 1) /
            dispatch.tritonBlockElements)
      fail("Triton Vortex pointwise logical tile and grid disagree");
    if ((output.dtype != "float32" && output.dtype != "int32") ||
        output.bytes != uint64_t(dispatch.elements) * 4)
      fail("Triton Vortex requires FP32 or I32 elementwise output");
    for (size_t id : dispatch.inputs) {
      const Tensor &input = plan.tensors[id];
      // A Reshape alias names the original allocation. Its rank/dimensions may
      // differ, but flattening must preserve the complete byte-identical tensor.
      if (input.dtype != output.dtype || input.elements != output.elements ||
          input.bytes != output.bytes)
        fail("Triton Vortex inputs must match output dtype and element count");
    }
  } else if (dispatch.tritonKind == "softmax") {
    uint64_t rows = nonnegative(schedule, "rows");
    uint64_t width = nonnegative(schedule, "width");
    uint64_t n = nonnegative(schedule, "n");
    if (!rows || rows > INT32_MAX || !width || width > 1024 || n != width ||
        rows > uint64_t(INT32_MAX) / width)
      fail("invalid Triton Vortex Softmax rows/width/N contract");
    dispatch.tritonSoftmaxRows = uint32_t(rows);
    dispatch.tritonSoftmaxWidth = uint32_t(width);
    dispatch.tritonSoftmaxN = uint32_t(n);
    dispatch.tritonSoftmaxTile = powerOfTwo("tile", 16, 1024);
    uint64_t expectedTile = 16;
    while (expectedTile < width)
      expectedTile *= 2;
    if (dispatch.tritonSoftmaxTile != expectedTile || grid != rows)
      fail("Triton Vortex Softmax tile or grid disagrees");
    if (dispatch.inputs.size() != 1 ||
        dispatch.elements != rows * width ||
        dispatch.inputs[0] == dispatch.output)
      fail("Triton Vortex Softmax pointer ABI contract disagrees");
    const Tensor &input = plan.tensors[dispatch.inputs[0]];
    if (input.dtype != "float32" || input.shape.empty() ||
        input.shape.back() != int64_t(width) ||
        input.elements != rows * width || input.bytes != rows * width * 4)
      fail("Triton Vortex Softmax input must be static FP32 [...,width]");
    uint64_t prefixRows = 1;
    for (size_t axis = 0; axis + 1 < input.shape.size(); ++axis) {
      if (input.shape[axis] <= 0 ||
          uint64_t(input.shape[axis]) > uint64_t(INT32_MAX) / prefixRows)
        fail("Triton Vortex Softmax input has an invalid prefix shape");
      prefixRows *= uint64_t(input.shape[axis]);
    }
    if (prefixRows != rows || output.dtype != "float32" ||
        output.shape != input.shape || output.elements != rows * width ||
        output.bytes != rows * width * 4)
      fail("Triton Vortex Softmax output must exactly match input");
  } else if (dispatch.tritonKind == "layer_norm") {
    uint64_t rows = nonnegative(schedule, "rows");
    uint64_t width = nonnegative(schedule, "width");
    uint64_t n = nonnegative(schedule, "n");
    uint64_t epsilonBits = nonnegative(schedule, "epsilon_f32_bits");
    uint32_t epsilonRaw = uint32_t(epsilonBits);
    float epsilon = 0;
    static_assert(sizeof(epsilon) == sizeof(epsilonRaw));
    std::memcpy(&epsilon, &epsilonRaw, sizeof(epsilon));
    if (!rows || rows > INT32_MAX || !width || width > 1024 || n != width ||
        epsilonBits > UINT32_MAX || !std::isfinite(epsilon) || epsilon <= 0)
      fail("invalid Triton Vortex LayerNorm rows/width/N/epsilon contract");
    dispatch.tritonLayerNormRows = uint32_t(rows);
    dispatch.tritonLayerNormWidth = uint32_t(width);
    dispatch.tritonLayerNormN = uint32_t(n);
    dispatch.tritonLayerNormTile = powerOfTwo("tile", 16, 1024);
    dispatch.tritonLayerNormEpsilonF32Bits = uint32_t(epsilonBits);
    uint64_t expectedTile = 16;
    while (expectedTile < width)
      expectedTile *= 2;
    if (dispatch.tritonLayerNormTile != expectedTile || grid != rows)
      fail("Triton Vortex LayerNorm tile or grid disagrees");
    if (rows > uint64_t(INT32_MAX) / width ||
        dispatch.elements != rows * width || dispatch.inputs.size() != 3 ||
        dispatch.inputs[0] == dispatch.inputs[1] ||
        dispatch.inputs[0] == dispatch.inputs[2] ||
        dispatch.inputs[1] == dispatch.inputs[2] ||
        dispatch.output == dispatch.inputs[0] ||
        dispatch.output == dispatch.inputs[1] ||
        dispatch.output == dispatch.inputs[2])
      fail("Triton Vortex LayerNorm pointer ABI contract disagrees");
    const Tensor &input = plan.tensors[dispatch.inputs[0]];
    const Tensor &scale = plan.tensors[dispatch.inputs[1]];
    const Tensor &bias = plan.tensors[dispatch.inputs[2]];
    if (input.dtype != "float32" || input.shape.size() < 2 ||
        input.shape.back() != int64_t(width) ||
        input.elements != rows * width || input.bytes != rows * width * 4)
      fail("Triton Vortex LayerNorm X must be static FP32 [...,width]");
    uint64_t prefixRows = 1;
    for (size_t axis = 0; axis + 1 < input.shape.size(); ++axis) {
      if (input.shape[axis] <= 0 ||
          uint64_t(input.shape[axis]) > uint64_t(INT32_MAX) / prefixRows)
        fail("Triton Vortex LayerNorm X has an invalid prefix shape");
      prefixRows *= uint64_t(input.shape[axis]);
    }
    if (prefixRows != rows || output.dtype != "float32" ||
        output.shape != input.shape || output.elements != rows * width ||
        output.bytes != rows * width * 4)
      fail("Triton Vortex LayerNorm Y must exactly match X");
    auto vectorParameter = [&](const Tensor &tensor) {
      return tensor.dtype == "float32" && tensor.shape.size() == 1 &&
             tensor.shape[0] == int64_t(width) && tensor.elements == width &&
             tensor.bytes == width * 4;
    };
    if (!vectorParameter(scale) || !vectorParameter(bias))
      fail("Triton Vortex LayerNorm Scale/Bias must be FP32 vector[width]");
  } else if (dispatch.tritonKind == "reduction") {
    uint64_t width = nonnegative(schedule, "width");
    if (!width || width > 1024)
      fail("invalid Triton Vortex reduction width");
    dispatch.tritonReductionWidth = uint32_t(width);
    dispatch.tritonReductionTile = powerOfTwo("tile", 16, 1024);
    uint64_t expectedTile = 16;
    while (expectedTile < width)
      expectedTile *= 2;
    if (dispatch.tritonReductionTile != expectedTile || grid != 1)
      fail("Triton Vortex reduction tile or grid disagrees");
    if (dispatch.inputs.size() != 1 || dispatch.elements != 1 ||
        output.dtype != "float32" || output.elements != 1 ||
        output.bytes != 4 ||
        !(output.shape.empty() ||
          (output.shape.size() == 1 && output.shape[0] == 1)))
      fail("Triton Vortex reduction output must be one FP32 scalar");
    const Tensor &input = plan.tensors[dispatch.inputs[0]];
    if (input.dtype != "float32" || input.shape.size() != 1 ||
        input.shape[0] != int64_t(width) || input.elements != width ||
        input.bytes != width * 4)
      fail("Triton Vortex reduction input must be FP32 vector[width]");
  } else if (dispatch.tritonKind == "matmul") {
    auto dimension = [&](llvm::StringRef name) {
      uint64_t value = nonnegative(schedule, name);
      if (!value || value > INT32_MAX)
        fail("invalid Triton Vortex MatMul dimension: " + name.str());
      return uint32_t(value);
    };
    dispatch.m = dimension("m");
    dispatch.n = dimension("n");
    dispatch.k = dimension("k");
    dispatch.tileM = dimension("block_m");
    dispatch.tileN = dimension("block_n");
    dispatch.tileK = dimension("block_k");
    uint64_t expectedTileK = 16;
    while (expectedTileK < dispatch.k)
      expectedTileK *= 2;
    if (dispatch.tileM != 16 || dispatch.tileN != 16 ||
        dispatch.k < 16 || dispatch.k > 65536 || dispatch.k % 16 ||
        dispatch.tileK != expectedTileK)
      fail("unsupported Triton Vortex MatMul tile contract");
    if (dispatch.inputs.size() != 2 ||
        dispatch.inputs[0] == dispatch.inputs[1] ||
        dispatch.output == dispatch.inputs[0] ||
        dispatch.output == dispatch.inputs[1])
      fail("Triton Vortex MatMul pointer ABI contract disagrees");
    const Tensor &a = plan.tensors[dispatch.inputs[0]];
    const Tensor &b = plan.tensors[dispatch.inputs[1]];
    bool hasBatches = schedule.get("batches");
    bool hasMode = schedule.get("batch_mode");
    bool hasA = schedule.get("a_shape");
    bool hasB = schedule.get("b_shape");
    bool hasC = schedule.get("c_shape");
    unsigned batchFields = unsigned(hasBatches) + unsigned(hasMode) +
                           unsigned(hasA) + unsigned(hasB) + unsigned(hasC);
    if (batchFields != 0 && batchFields != 5)
      fail("incomplete Triton Vortex batched MatMul contract");
    auto logicalShape = [&](llvm::StringRef name, const Tensor &physical,
                            bool legacy) {
      if (legacy)
        return physical.shape;
      std::vector<int64_t> shape;
      uint64_t elements = 1;
      for (const auto &dim : array(schedule, name)) {
        auto value = dim.getAsInteger();
        if (!value || *value <= 0 || uint64_t(*value) > INT32_MAX / elements)
          fail("invalid Triton Vortex MatMul logical shape: " + name.str());
        elements *= uint64_t(*value);
        shape.push_back(*value);
      }
      if (shape.size() < 2 || shape != physical.shape ||
          elements != physical.elements || physical.bytes != elements * 4)
        fail("Triton Vortex MatMul logical/physical shape disagrees: " +
             name.str());
      return shape;
    };
    bool legacyManifest = batchFields == 0;
    dispatch.batches = legacyManifest ? 1 : dimension("batches");
    dispatch.tritonMatmulBatchMode =
        legacyManifest ? "rank2" : string(schedule, "batch_mode");
    auto aShape = logicalShape("a_shape", a, legacyManifest);
    auto bShape = logicalShape("b_shape", b, legacyManifest);
    auto cShape = logicalShape("c_shape", output, legacyManifest);
    auto endsWith = [](const std::vector<int64_t> &shape, uint32_t first,
                       uint32_t second) {
      return shape.size() >= 2 && shape[shape.size() - 2] == int64_t(first) &&
             shape.back() == int64_t(second);
    };
    bool validShapes = false;
    if (dispatch.tritonMatmulBatchMode == "rank2") {
      validShapes = dispatch.batches == 1 && aShape.size() == 2 &&
                    bShape.size() == 2 && cShape.size() == 2 &&
                    endsWith(aShape, dispatch.m, dispatch.k) &&
                    endsWith(bShape, dispatch.k, dispatch.n) &&
                    endsWith(cShape, dispatch.m, dispatch.n);
    } else if (dispatch.tritonMatmulBatchMode == "matched-prefix" ||
               dispatch.tritonMatmulBatchMode == "shared-rhs") {
      if (cShape.size() >= 3 && aShape.size() == cShape.size() &&
          endsWith(aShape, dispatch.m, dispatch.k) &&
          endsWith(cShape, dispatch.m, dispatch.n)) {
        std::vector<int64_t> prefix(cShape.begin(), cShape.end() - 2);
        uint64_t batches = 1;
        for (int64_t dim : prefix) {
          if (uint64_t(dim) > INT32_MAX / batches)
            fail("Triton Vortex MatMul batch prefix exceeds signed i32");
          batches *= uint64_t(dim);
        }
        bool aPrefix = std::equal(prefix.begin(), prefix.end(), aShape.begin());
        bool bMatches =
            dispatch.tritonMatmulBatchMode == "matched-prefix"
                ? bShape.size() == cShape.size() &&
                      std::equal(prefix.begin(), prefix.end(), bShape.begin()) &&
                      endsWith(bShape, dispatch.k, dispatch.n)
                : bShape.size() == 2 &&
                      endsWith(bShape, dispatch.k, dispatch.n);
        validShapes = aPrefix && bMatches && batches == dispatch.batches;
      }
    }
    auto checkedProduct = [&](std::initializer_list<uint64_t> factors,
                              llvm::StringRef label) {
      uint64_t value = 1;
      for (uint64_t factor : factors) {
        if (!factor || factor > uint64_t(INT32_MAX) / value)
          fail("Triton Vortex MatMul " + label.str() +
               " exceeds signed i32");
        value *= factor;
      }
      return value;
    };
    uint64_t expectedA =
        checkedProduct({dispatch.batches, dispatch.m, dispatch.k}, "A size");
    uint64_t expectedB = checkedProduct(
        {dispatch.tritonMatmulBatchMode == "matched-prefix"
             ? dispatch.batches
             : 1,
         dispatch.k, dispatch.n},
        "B size");
    uint64_t expectedC =
        checkedProduct({dispatch.batches, dispatch.m, dispatch.n}, "C size");
    if (!validShapes || a.dtype != "float32" || b.dtype != "float32" ||
        output.dtype != "float32" ||
        a.elements != expectedA || b.elements != expectedB ||
        output.elements != expectedC || a.bytes != expectedA * 4 ||
        b.bytes != expectedB * 4 || output.bytes != expectedC * 4 ||
        dispatch.elements != expectedC)
      fail("Triton Vortex MatMul tensor/batch contract disagrees");
    uint64_t tilesM =
        (uint64_t(dispatch.m) + dispatch.tileM - 1) / dispatch.tileM;
    uint64_t tilesN =
        (uint64_t(dispatch.n) + dispatch.tileN - 1) / dispatch.tileN;
    if (tilesM > INT32_MAX / tilesN ||
        tilesM * tilesN > uint64_t(INT32_MAX) / dispatch.batches ||
        grid != tilesM * tilesN * dispatch.batches)
      fail("Triton Vortex MatMul logical tile and grid disagree");
  } else {
    fail("unsupported Triton Vortex dispatch kind: " + dispatch.tritonKind);
  }
  dispatch.triton = true;
}

// Optimized launch geometry is part of the compiled kernel ABI. Validate it
// against logical operand shapes (views may share a differently shaped buffer),
// rather than accepting arbitrary block/grid numbers from the manifest.
inline void validateCudaSchedule(const json::Object &schedule,
                                 Dispatch &dispatch, const Plan &plan) {
  const Tensor &output = plan.tensors[dispatch.output];
  auto positive = [&](llvm::StringRef name) -> uint64_t {
    uint64_t value = nonnegative(schedule, name);
    if (!value || value > INT32_MAX)
      fail("invalid CUDA schedule dimension: " + name.str());
    return value;
  };
  auto logical = [&](size_t operand, llvm::StringRef name) {
    if (operand >= dispatch.inputs.size())
      fail("CUDA schedule has a missing input");
    Tensor value = plan.tensors[dispatch.inputs[operand]];
    value.shape.clear();
    uint64_t count = 1;
    for (const auto &dim : array(schedule, name)) {
      auto size = dim.getAsInteger();
      if (!size || *size <= 0 || uint64_t(*size) > value.elements / count)
        fail("CUDA logical shape exceeds physical storage");
      count *= uint64_t(*size);
      value.shape.push_back(*size);
    }
    if (count != value.elements || value.dtype != "float32")
      fail("CUDA template requires byte-identical FP32 logical operands");
    return value;
  };
  auto broadcasts = [&](const Tensor &value) {
    if (value.shape.size() > output.shape.size())
      fail("CUDA broadcast operand rank exceeds output rank");
    size_t pad = output.shape.size() - value.shape.size();
    for (size_t i = 0; i < value.shape.size(); ++i)
      if (value.shape[i] != 1 && value.shape[i] != output.shape[pad + i])
        fail("CUDA operand shape cannot broadcast to output");
  };
  dispatch.cudaSchedule = string(schedule, "schedule");
  uint64_t shared = nonnegative(schedule, "static_shared_memory_bytes");
  if (shared > 48 * 1024 || output.dtype != "float32" ||
      positive("block_size") != dispatch.blockSize ||
      positive("grid_size") != dispatch.gridSize)
    fail("CUDA schedule resource/launch contract disagrees");
  dispatch.staticSharedBytes = uint32_t(shared);
  if (dispatch.cudaSchedule == "shared-tiled") {
    uint64_t m = positive("m"), n = positive("n"), k = positive("k");
    uint64_t batches = positive("batches");
    uint64_t tm = positive("tile_m"), tn = positive("tile_n"), tk = positive("tile_k");
    uint64_t rm = positive("thread_tile_m"), rn = positive("thread_tile_n");
    // Bound dimensions before multiplying; these are supported template
    // resources, not an estimate of the hardware's theoretical maximum.
    if (tm > 128 || tn > 128 || tk > 128 || rm > 8 || rn > 8 ||
        dispatch.blockSize < 32 || dispatch.blockSize % 32 ||
        tm % rm || tn % rn || (tm / rm) * (tn / rn) != dispatch.blockSize ||
        4 * tk * (tm + tn) != shared ||
        m * n > INT32_MAX || batches > uint64_t(INT32_MAX) / (m * n) ||
        m * n * batches != output.elements ||
        ((m + tm - 1) / tm) * ((n + tn - 1) / tn) * batches != dispatch.gridSize)
      fail("CUDA matrix tile/storage geometry disagrees");
    bool bias = dispatch.op == "gac.matmul_bias" || dispatch.op == "gac.matmul_bias_relu";
    if ((!bias && dispatch.op != "gac.matmul") ||
        dispatch.inputs.size() != (bias ? 3u : 2u))
      fail("CUDA matrix template requires MatMul with optional fused bias");
    Tensor a = logical(0, "a_shape"), b = logical(1, "b_shape");
    if (a.shape.size() < 2 || b.shape.size() < 2 || output.shape.size() < 2 ||
        a.shape.back() != int64_t(k) || b.shape[b.shape.size()-2] != int64_t(k) ||
        b.shape.back() != int64_t(n) || output.shape.back() != int64_t(n))
      fail("CUDA matrix operand dimensions disagree");
    if (b.shape.size() == 2) {
      if (batches != 1 || a.elements != m * k || output.shape.size() != a.shape.size() ||
          !std::equal(a.shape.begin(), a.shape.end()-1, output.shape.begin()))
        fail("CUDA flattened matrix projection shape disagrees");
    } else {
      if (a.shape[a.shape.size()-2] != int64_t(m) ||
          output.shape[output.shape.size()-2] != int64_t(m) ||
          output.shape.size() != std::max(a.shape.size(), b.shape.size()))
        fail("CUDA batch matrix dimensions disagree");
      for (size_t axis = 0; axis + 2 < output.shape.size(); ++axis) {
        auto dim = [&](const Tensor &value) {
          size_t pad = output.shape.size() - value.shape.size();
          return axis < pad ? int64_t(1) : value.shape[axis-pad];
        };
        int64_t av = dim(a), bv = dim(b);
        if ((av != 1 && bv != 1 && av != bv) || output.shape[axis] != std::max(av,bv))
          fail("CUDA matrix batch broadcasting disagrees");
      }
    }
    if (bias)
      broadcasts(logical(2, "bias_shape"));
    else if (!array(schedule, "bias_shape").empty())
      fail("CUDA matrix without bias has a bias shape");
  } else if (dispatch.cudaSchedule == "channel-tiled") {
    std::string kind = string(schedule, "kind");
    bool deform = kind == "deform_conv2d";
    if ((!deform && kind != "conv2d" && kind != "conv2d_relu") ||
        dispatch.op != "gac." + kind || output.shape.size() != 4 || shared)
      fail("CUDA channel tile does not match convolution operator");
    uint64_t tile = positive("channel_tile"), groups = positive("groups");
    uint64_t channels = positive("channels_per_group");
    if ((tile != 2 && tile != 4 && tile != 8) || channels % tile ||
        groups * channels != uint64_t(output.shape[1]) || output.elements % tile ||
        positive("logical_threads") != output.elements / tile || dispatch.blockSize != 256 ||
        dispatch.gridSize != (output.elements / tile + 255) / 256)
      fail("CUDA channel tile/group/launch geometry disagrees");
    const auto &shapes = array(schedule, "input_shapes");
    if (shapes.size() != dispatch.inputs.size() ||
        dispatch.inputs.size() < (deform ? 4u : 2u) ||
        dispatch.inputs.size() > (deform ? 5u : 3u))
      fail("CUDA convolution operand count disagrees");
    std::vector<std::vector<int64_t>> logicalShapes;
    for (size_t i = 0; i < shapes.size(); ++i) {
      auto *shape = shapes[i].getAsArray();
      const Tensor &tensor = plan.tensors[dispatch.inputs[i]];
      if (!shape || tensor.dtype != "float32")
        fail("CUDA convolution requires FP32 logical operand shapes");
      uint64_t elements = 1;
      std::vector<int64_t> dims;
      for (const auto &dim : *shape) {
        auto value = dim.getAsInteger();
        if (!value || *value <= 0 || uint64_t(*value) > tensor.elements / elements)
          fail("CUDA convolution logical shape exceeds storage");
        elements *= uint64_t(*value);
        dims.push_back(*value);
      }
      if (elements != tensor.elements)
        fail("CUDA convolution logical shape differs from storage");
      logicalShapes.push_back(std::move(dims));
    }
    const auto &x = logicalShapes[0], &weight = logicalShapes[deform ? 3 : 1];
    if (x.size() != 4 || weight.size() != 4 || x[0] != output.shape[0] ||
        weight[0] != output.shape[1] || uint64_t(weight[1]) * groups != uint64_t(x[1]))
      fail("CUDA convolution channel/group dimensions disagree");
    const auto &outShape = array(schedule, "output_shape");
    if (outShape.size() != output.shape.size())
      fail("CUDA convolution output rank disagrees");
    for (size_t i = 0; i < outShape.size(); ++i)
      if (outShape[i].getAsInteger() != output.shape[i])
        fail("CUDA convolution output shape disagrees");
    if (dispatch.inputs.size() == (deform ? 5u : 3u) &&
        logicalShapes.back() != std::vector<int64_t>{output.shape[1]})
      fail("CUDA convolution bias shape disagrees");
    if (deform) {
      const auto &offset = logicalShapes[1], &mask = logicalShapes[2];
      if (offset.size() != 4 || mask.size() != 4 ||
          offset[0] != x[0] || mask[0] != x[0] || offset[1] != 2 * mask[1] ||
          offset[2] != output.shape[2] || offset[3] != output.shape[3] ||
          mask[2] != output.shape[2] || mask[3] != output.shape[3] ||
          mask[1] % (weight[2] * weight[3]) ||
          x[1] % (mask[1] / (weight[2] * weight[3])))
        fail("CUDA deformable convolution sampling shapes disagree");
    }
  } else if (dispatch.cudaSchedule == "row-reduction") {
    std::string kind = string(schedule, "kind");
    bool layerNorm = kind == "layer_norm";
    if ((!layerNorm && kind != "softmax") || dispatch.op != "gac." + kind ||
        dispatch.inputs.empty() || dispatch.inputs.size() > (layerNorm ? 3u : 1u))
      fail("CUDA reduction template does not match its operator");
    Tensor input = logical(0, "input_shape");
    int64_t axis = integer(schedule, "axis");
    if (input.shape != output.shape || axis < 0 || axis >= int64_t(input.shape.size()))
      fail("CUDA reduction axis or shape disagrees");
    uint64_t width = layerNorm ? 1 : uint64_t(input.shape[axis]), stride = 1;
    for (size_t i = size_t(axis) + (layerNorm ? 0 : 1); i < input.shape.size(); ++i)
      (layerNorm ? width : stride) *= uint64_t(input.shape[i]);
    if (positive("width") != width || positive("stride") != stride ||
        positive("rows") != input.elements / width ||
        dispatch.gridSize != input.elements / width || shared != 4 * dispatch.blockSize ||
        (dispatch.blockSize != 32 && dispatch.blockSize != 128 && dispatch.blockSize != 256))
      fail("CUDA reduction row/launch geometry disagrees");
    for (size_t index = 1; index <= 2; ++index) {
      llvm::StringRef field = index == 1 ? "scale_shape" : "bias_shape";
      if (index < dispatch.inputs.size())
        broadcasts(logical(index, field));
      else if (!array(schedule, field).empty())
        fail("CUDA reduction has a shape for an absent affine input");
    }
  } else {
    fail("unknown CUDA compute schedule");
  }
}

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
    fail("CUDA kernels cannot use a Vortex LMEM/buffer contract");
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
    if (auto op = row.getString("op"))
      dispatch.op = op->str();
    if (auto source = row.getString("source_node"))
      dispatch.source = source->str();
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
    auto lowering = row.getString("lowering");
    if (auto *triton = row.get("triton")) {
      auto regular = root.getString("regular_lowering");
      if (cuda || !regular || *regular != "linalg" || !lowering ||
          *lowering != "linalg-to-triton" || root.get("cuda") ||
          row.get("vortex") || row.get("vortex_schedule") ||
          row.get("cuda_schedule"))
        fail("Triton dispatch requires Vortex Linalg lowering without another schedule");
      validateTritonSchedule(object(*triton), dispatch, plan);
    } else if (lowering && *lowering == "linalg-to-triton") {
      fail("Triton dispatch is missing its launch contract");
    } else if (cuda) {
      uint64_t block = nonnegative(row, "block_size");
      uint64_t grid = nonnegative(row, "grid_size");
      if (!block || block > 1024 || !grid || grid > INT32_MAX ||
          nonnegative(row, "shared_memory_bytes") != 0)
        fail("invalid CUDA kernel launch dimensions or dynamic shared allocation");
      dispatch.blockSize = uint32_t(block);
      dispatch.gridSize = uint32_t(grid);
      if (auto *schedule = row.get("cuda_schedule"))
        validateCudaSchedule(object(*schedule), dispatch, plan);
      else if (grid != (elements + block - 1) / block)
        fail("invalid CUDA scalar kernel launch dimensions");
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
      // Older manifests are the one-output-per-thread template.
      dispatch.vortexThreadTileM = schedule.get("thread_tile_m")
          ? dimension("thread_tile_m") : 1;
      dispatch.vortexThreadTileN = schedule.get("thread_tile_n")
          ? dimension("thread_tile_n") : 1;
      if (dispatch.vortexThreadTileM > 4 || dispatch.vortexThreadTileN > 4)
        fail("Vortex register tile dimensions exceed supported range");
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
            dispatch.lmemBytes || dispatch.stages ||
            dispatch.vortexThreadTileM != 1 || dispatch.vortexThreadTileN != 1)
          fail("unsupported Vortex FP32 WMMA geometry");
        if (schedule.get("tcu_call_mode")) {
          std::string mode = string(schedule, "tcu_call_mode");
          if (mode != "separate" && mode != "combined")
            fail("unsupported Vortex TCU call mode");
        }
      } else {
        if (dispatch.tileM > 65535 || dispatch.tileN > 65535 ||
            dispatch.tileK > 65535)
          fail("Vortex LMEM tile dimensions exceed supported range");
        if (dispatch.tileM % dispatch.vortexThreadTileM ||
            dispatch.tileN % dispatch.vortexThreadTileN)
          fail("Vortex block tile must be divisible by the register tile");
        uint64_t tileThreads = uint64_t(dispatch.tileM / dispatch.vortexThreadTileM) *
                               (dispatch.tileN / dispatch.vortexThreadTileN);
        uint64_t stageBytes = 4 * uint64_t(dispatch.tileK) *
                              (uint64_t(dispatch.tileM) + dispatch.tileN);
        if (tileThreads != dispatch.blockSize ||
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
