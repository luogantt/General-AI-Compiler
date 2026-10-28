#ifndef GAC_TEMPLATES_REGISTRY_H
#define GAC_TEMPLATES_REGISTRY_H

#include "gac/DeviceOptions.h"
#include "gac/templates/CudaConvolution.h"
#include "gac/templates/CudaMatmul.h"
#include "gac/templates/CudaReduction.h"
#include "gac/templates/VortexMatmul.h"
#include "llvm/Support/JSON.h"
#include <variant>

namespace gac::templates {
// A template is a checked algorithm/schedule, not generated C++ text. Its
// emitter builds typed MLIR operations. Stable IDs identify it in artifacts.
struct Descriptor {
  llvm::StringRef id;
  DeviceTarget target;
  llvm::ArrayRef<llvm::StringRef> operations;
  llvm::StringRef arithmetic;
  llvm::StringRef storage;
  bool collective;
};

using Plan = std::variant<std::monostate, CudaMatmulPlan, CudaReductionPlan,
                          CudaConvolutionPlan, VortexMatmulPlan>;
struct Selection {
  // Null means the caller uses its Linalg/pad/scalar baseline, on the SAME
  // device. It never requests a host or vendor-library execution fallback.
  const Descriptor *descriptor = nullptr;
  Plan plan;
  std::string matmulFallback, reductionFallback, convolutionFallback;
  bool hasTemplate() const { return descriptor != nullptr; }
};

// Catalog and selection share the same compiled registry. Selection does not
// mutate IR. Unsupported contracts record why; invalid options are errors.
llvm::json::Object catalog();
mlir::LogicalResult select(mlir::Operation *op, const DeviceOptions &options,
                           Selection &selection);
// Call without an outer per-element bounds guard. Collective emitters own
// tail predicates and barriers; output pointer follows all input pointers.
mlir::LogicalResult emit(mlir::OpBuilder &builder, mlir::Operation *op,
                         llvm::ArrayRef<mlir::Value> pointers,
                         const Selection &selection);
// Both Runtime launch metadata and inspection metadata derive from the same
// checked plan used by emit(). Existing manifest ABI keys are preserved.
void annotate(llvm::json::Object &dispatch, const Selection &selection,
              llvm::StringRef baselineId);
} // namespace gac::templates
#endif
