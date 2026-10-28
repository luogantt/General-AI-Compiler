#ifndef GAC_LINALG_DEVICE_H
#define GAC_LINALG_DEVICE_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace gac {

// A single-use producer can be evaluated for the consumer's current output
// element when its indexing proves that no other producer element is needed.
bool isFusedLinalgProducer(mlir::Operation *op);

// Tensor operands required by a dispatch. An unused DPS initializer (typically
// tensor.empty) has no value to transfer and is deliberately absent from its ABI.
// Eligible producers are recursively expanded to their deduplicated leaf inputs.
llvm::SmallVector<mlir::Value> getLinalgDeviceInputs(mlir::Operation *op);

// The initial device schedule assigns one output element to each thread and
// keeps reduction dimensions sequential inside that thread.
mlir::LogicalResult validateLinalgDeviceOp(mlir::Operation *op);

// Pointers are getLinalgDeviceInputs(op), followed by the allocated output.
// This performs destination-style bufferization at the existing dispatch ABI,
// then uses upstream Linalg-to-SCF lowering; it does not reimplement the payload.
mlir::LogicalResult
emitLinalgDeviceElement(mlir::OpBuilder &builder, mlir::Operation *op,
                        llvm::ArrayRef<mlir::Value> pointers, mlir::Value tid);

} // namespace gac
#endif
