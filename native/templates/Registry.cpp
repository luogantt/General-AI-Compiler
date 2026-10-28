#include "gac/templates/Registry.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace gac::templates {
namespace {
using Planner = LogicalResult (*)(Operation *, const DeviceOptions &, Selection &);
struct Entry {
  Descriptor descriptor;
  bool (*enabled)(const DeviceOptions &);
  Planner planner;
};

LogicalResult cudaMatmul(Operation *op, const DeviceOptions &options, Selection &s) {
  std::optional<CudaMatmulPlan> p;
  if (failed(planCudaMatmul(op, options, p, s.matmulFallback))) return failure();
  if (p) s.plan = *p;
  return success();
}
LogicalResult cudaReduction(Operation *op, const DeviceOptions &, Selection &s) {
  std::optional<CudaReductionPlan> p;
  if (failed(planCudaReduction(op, p, s.reductionFallback))) return failure();
  if (p) s.plan = *p;
  return success();
}
LogicalResult cudaConvolution(Operation *op, const DeviceOptions &options, Selection &s) {
  std::optional<CudaConvolutionPlan> p;
  if (failed(planCudaConvolution(op, options.cudaConvChannelTile, p,
                                 s.convolutionFallback))) return failure();
  if (p) s.plan = *p;
  return success();
}
LogicalResult vortexMatmul(Operation *op, const DeviceOptions &options, Selection &s) {
  std::optional<VortexMatmulPlan> p;
  if (failed(planVortexMatmul(op, options, p, s.matmulFallback))) return failure();
  if (p) s.plan = *p;
  return success();
}

const StringRef matmulOps[] = {"gac.matmul", "gac.matmul_bias", "gac.matmul_bias_relu"};
const StringRef reductionOps[] = {"gac.softmax", "gac.layer_norm"};
const StringRef convolutionOps[] = {"gac.conv2d", "gac.conv2d_relu", "gac.deform_conv2d"};
// Deliberately compile-time registration: a new algorithm supplies a checked
// plan + emitter, then registers its contract here. No model names are used.
const Entry entries[] = {
  {{"cuda.matmul.shared", DeviceTarget::CUDA, matmulOps, "fp32",
    "static shared A/B tiles; per-thread register accumulators", true},
   [](const DeviceOptions &o) { return o.cudaMatmulSchedule != "scalar"; }, cudaMatmul},
  {{"cuda.reduction.row", DeviceTarget::CUDA, reductionOps, "fp32",
    "static shared reduction scratch; one CTA per row", true},
   [](const DeviceOptions &o) { return o.cudaReductionSchedule != "scalar"; }, cudaReduction},
  {{"cuda.convolution.channels", DeviceTarget::CUDA, convolutionOps, "fp32",
    "register channel tile; no shared staging", false},
   [](const DeviceOptions &o) { return o.cudaConvChannelTile != 1; }, cudaConvolution},
  {{"vortex.matmul.lmem", DeviceTarget::Vortex, matmulOps, "fp32",
    "one LMEM A/B stage; register accumulators", true},
   [](const DeviceOptions &o) { return o.vortexMatmulSchedule == "lmem-tiled"; }, vortexMatmul},
  {{"vortex.matmul.dxa", DeviceTarget::Vortex, matmulOps, "fp32",
    "two LMEM A/B stages; DXA copy; register accumulators", true},
   [](const DeviceOptions &o) { return o.vortexMatmulSchedule == "dxa-double-buffer"; }, vortexMatmul},
  {{"vortex.matmul.tcu", DeviceTarget::Vortex, matmulOps, "tf32x3-fp32-accumulate",
    "register WMMA fragments; no LMEM/DXA staging", true},
   [](const DeviceOptions &o) { return o.vortexMatmulSchedule == "tcu"; }, vortexMatmul},
};

llvm::json::Array shape(ArrayRef<int64_t> dimensions) {
  llvm::json::Array values;
  for (int64_t dim : dimensions) values.push_back(dim);
  return values;
}

llvm::json::Object matmulMetadata(const CudaMatmulPlan &p) {
  return llvm::json::Object{{"schedule", p.schedule}, {"m", p.m}, {"n", p.n}, {"k", p.k},
          {"batches", p.batches}, {"tile_m", p.tileM}, {"tile_n", p.tileN},
          {"tile_k", p.tileK}, {"thread_tile_m", p.threadTileM},
          {"thread_tile_n", p.threadTileN}, {"block_size", p.blockSize},
          {"grid_size", p.gridSize}, {"static_shared_memory_bytes", p.sharedMemoryBytes},
          {"a_shape", shape(p.aShape)}, {"b_shape", shape(p.bShape)},
          {"bias_shape", shape(p.biasShape)}};
}
llvm::json::Object matmulMetadata(const VortexMatmulPlan &p) {
  llvm::json::Object object{
      {"schedule", p.schedule}, {"m", p.m}, {"n", p.n}, {"k", p.k},
      {"batches", p.batches}, {"tile_m", p.tileM}, {"tile_n", p.tileN},
      {"tile_k", p.tileK}, {"block_size", p.blockSize},
      {"grid_size", p.gridSize}, {"lmem_bytes", p.lmemBytes},
      {"thread_tile_m", p.threadTileM}, {"thread_tile_n", p.threadTileN},
      {"stages", p.stages}, {"lanes", p.lanes},
      {"a_shape", shape(p.aShape)}, {"b_shape", shape(p.bShape)},
      {"bias_shape", shape(p.biasShape)}};
  if (p.schedule == "tcu") {
    object["arithmetic"] = "tf32x3-fp32-accumulate";
    object["tcu_call_mode"] = p.tcuCombined ? "combined" : "separate";
  }
  return object;
}
} // namespace

llvm::json::Object catalog() {
  llvm::json::Array rows;
  for (const Entry &entry : entries) {
    const auto &d = entry.descriptor;
    llvm::json::Array operations;
    for (StringRef op : d.operations) operations.push_back(op.str());
    rows.push_back(llvm::json::Object{
        {"id", d.id.str()}, {"target", d.target == DeviceTarget::CUDA ? "cuda" : "vortex"},
        {"operations", std::move(operations)}, {"regular_lowering", "legacy"},
        {"arithmetic", d.arithmetic.str()}, {"storage", d.storage.str()},
        {"collective", d.collective}});
  }
  return llvm::json::Object{{"schema_version", 1}, {"library", "GACComputeTemplates"},
          {"selection_policy", "explicit options + checked contracts; not autotuning"},
          {"templates", std::move(rows)}};
}

LogicalResult select(Operation *op, const DeviceOptions &options, Selection &s) {
  s = Selection{};
  // Preserve the Linalg route: target templates must never silently turn a
  // structured computation back into an unrelated, hand-selected algorithm.
  if (options.regularLowering != "legacy") return success();
  for (const Entry &entry : entries) {
    if (entry.descriptor.target != options.target || !entry.enabled(options) ||
        !llvm::is_contained(entry.descriptor.operations, op->getName().getStringRef()))
      continue;
    if (failed(entry.planner(op, options, s))) return failure();
    if (!std::holds_alternative<std::monostate>(s.plan)) {
      s.descriptor = &entry.descriptor;
      return success();
    }
  }
  return success();
}

LogicalResult emit(OpBuilder &builder, Operation *op, ArrayRef<Value> pointers,
                   const Selection &s) {
  if (!s.hasTemplate()) return op->emitError("no checked compute template selected");
  if (auto *p = std::get_if<CudaMatmulPlan>(&s.plan))
    return emitCudaMatmul(builder, op, pointers, *p);
  if (auto *p = std::get_if<CudaReductionPlan>(&s.plan))
    return emitCudaReduction(builder, op, pointers, *p);
  if (auto *p = std::get_if<CudaConvolutionPlan>(&s.plan))
    return emitCudaConvolution(builder, op, pointers, *p);
  if (auto *p = std::get_if<VortexMatmulPlan>(&s.plan))
    return emitVortexMatmul(builder, op, pointers, *p);
  return op->emitError("compute template has no typed plan");
}

void annotate(llvm::json::Object &dispatch, const Selection &s, StringRef baselineId) {
  llvm::json::Object description{{"id", s.hasTemplate() ? s.descriptor->id.str() : baselineId.str()},
      {"selection_policy", "explicit-options-and-contracts"},
      {"collective", s.hasTemplate() && s.descriptor->collective}};
  if (s.hasTemplate()) {
    description["arithmetic"] = s.descriptor->arithmetic.str();
    description["storage"] = s.descriptor->storage.str();
  }
  dispatch["compute_template"] = std::move(description);
  auto cuda = [&](int64_t block, int64_t grid, llvm::json::Object metadata) {
    dispatch["block_size"] = block;
    dispatch["grid_size"] = grid;
    // shared_memory_bytes is the dynamic launch allocation. Templates above
    // use static globals; recording the static size here would double-allocate.
    dispatch["shared_memory_bytes"] = 0;
    dispatch["cuda_schedule"] = std::move(metadata);
  };
  if (auto *p = std::get_if<CudaMatmulPlan>(&s.plan))
    cuda(p->blockSize, p->gridSize, matmulMetadata(*p));
  if (auto *p = std::get_if<CudaReductionPlan>(&s.plan))
    cuda(p->blockSize, p->gridSize, cudaReductionMetadata(*p));
  if (auto *p = std::get_if<CudaConvolutionPlan>(&s.plan))
    cuda(p->blockSize, p->gridSize, cudaConvolutionMetadata(*p));
  if (auto *p = std::get_if<VortexMatmulPlan>(&s.plan))
    dispatch["vortex"] = matmulMetadata(*p);
  if (!s.matmulFallback.empty()) {
    dispatch["matmul_schedule"] = "scalar";
    dispatch["fallback_reason"] = s.matmulFallback;
  }
  if (!s.reductionFallback.empty()) {
    dispatch["reduction_schedule"] = "scalar";
    dispatch["reduction_fallback_reason"] = s.reductionFallback;
  }
  if (!s.convolutionFallback.empty()) {
    dispatch["convolution_schedule"] = "scalar";
    dispatch["convolution_fallback_reason"] = s.convolutionFallback;
  }
}
} // namespace gac::templates
