#ifndef GAC_LINALG_CONVOLUTION_H
#define GAC_LINALG_CONVOLUTION_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace gac {

/// Convert a static NCHW gac.conv2d / gac.conv2d_relu to tensor Linalg.
/// The returned tensor is computed by the emitted operations; no GAC
/// convolution remains hidden behind a Linalg annotation or external call.
mlir::FailureOr<mlir::Value>
lowerConvolutionToLinalg(mlir::OpBuilder &builder, mlir::Operation *op);

} // namespace gac
#endif
