#ifndef GAC_ONNX_IMPORTER_H
#define GAC_ONNX_IMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace gac {
/// Parse ONNX protobuf directly into registered MLIR operations and SSA values.
/// No Python Graph, textual MLIR generation, or device execution is involved.
/// Strict mode rejects unsupported semantics. Inspection mode keeps such nodes
/// as gac.unsupported, with their original protobuf and a diagnostic reason.
/// Failure emits an MLIR diagnostic and returns an empty owning reference.
mlir::OwningOpRef<mlir::ModuleOp> importOnnx(llvm::StringRef path,
                                             mlir::MLIRContext &context,
                                             bool allowUnsupported = false);
} // namespace gac
#endif
