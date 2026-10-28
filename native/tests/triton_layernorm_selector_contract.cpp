#include "gac/DeviceLowering.h"
#include "gac/DeviceOptions.h"
#include "gac/LinalgLowering.h"
#include "gac/TritonBridge.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

int fail(StringRef message) {
  llvm::errs() << message << "\n";
  return 1;
}

} // namespace

// Directly exercise the selector on the real Linalg stage emitted by
// gac-compile. ONNX exposes only the final LayerNormalization outputs, so an
// external use of an internal statistics tensor cannot be authored through a
// valid single-output ONNX node. This test inserts that use at the MLIR layer,
// where the atomic-subgraph contract is defined.
int main(int argc, char **argv) {
  if (argc != 3)
    return fail("usage: gac-triton-layernorm-selector-contract "
                "02-linalg.mlir triton-output-directory");

  DialectRegistry registry;
  gac::registerDeviceDialects(registry);
  gac::createLowerToDevicePass({}, nullptr)->getDependentDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module || failed(verify(*module)))
    return fail("cannot parse and verify the LayerNorm Linalg stage");

  linalg::GenericOp root;
  module->walk([&](linalg::GenericOp candidate) {
    auto role = candidate->getAttrOfType<StringAttr>("gac.linalg_role");
    if (role && role.getValue() == "layer-normalization")
      root = candidate;
  });
  if (!root)
    return fail("LayerNorm aggregate root is missing");

  gac::TritonLayerNormPlan plan;
  std::string reason;
  if (!gac::selectTritonLayerNorm(root, plan, reason))
    return fail("canonical LayerNorm was rejected: " + reason);

  Value mean = root.getDpsInputOperand(1)->get();
  OpBuilder builder(&context);
  builder.setInsertionPointAfter(mean.getDefiningOp());
  auto fanout = builder.create<UnrealizedConversionCastOp>(
      builder.getUnknownLoc(), TypeRange{mean.getType()}, ValueRange{mean});
  if (gac::selectTritonLayerNorm(root, plan, reason) ||
      !StringRef(reason).contains("internal values"))
    return fail("internal statistics fanout did not reject the whole subgraph");
  fanout.erase();

  OpOperand *biasOperand = root.getDpsInputOperand(4);
  Value originalBias = biasOperand->get();
  biasOperand->set(root.getDpsInputOperand(3)->get());
  if (gac::selectTritonLayerNorm(root, plan, reason) ||
      !StringRef(reason).contains("distinct X, Scale and Bias"))
    return fail("duplicate Scale/Bias SSA did not reject the whole subgraph");
  biasOperand->set(originalBias);

  if (!gac::selectTritonLayerNorm(root, plan, reason))
    return fail("restored canonical LayerNorm was rejected: " + reason);

  // The selector-level fanout check above is necessary but not sufficient:
  // device planning must also keep every operation in the rejected aggregate
  // on the native path.  Expose the mean as a second public result to create a
  // real third use without introducing an unsupported test-only operation.
  auto function = root->getParentOfType<func::FuncOp>();
  auto returnOp = dyn_cast<func::ReturnOp>(function.getBody().front().back());
  if (!returnOp)
    return fail("expected func.return in LayerNorm test module");
  returnOp->insertOperands(returnOp.getNumOperands(), ValueRange{mean});
  SmallVector<Type> resultTypes(function.getResultTypes());
  resultTypes.push_back(mean.getType());
  function.setType(FunctionType::get(&context, function.getArgumentTypes(),
                                     resultTypes));
  auto outputNames = function->getAttrOfType<ArrayAttr>("gac.output_names");
  SmallVector<Attribute> names(outputNames.begin(), outputNames.end());
  names.push_back(builder.getStringAttr("MeanDebug"));
  function->setAttr("gac.output_names", builder.getArrayAttr(names));
  if (failed(verify(*module)))
    return fail("internal-fanout module became invalid");
  if (gac::selectTritonLayerNorm(root, plan, reason) ||
      !StringRef(reason).contains("internal values"))
    return fail("public mean fanout did not reject the whole subgraph");

  gac::DeviceOptions options;
  options.target = gac::DeviceTarget::Vortex;
  options.regularLowering = "linalg";
  options.vortexTritonDirectory = argv[2];
  llvm::json::Object manifest;
  if (failed(gac::lowerToDevice(*module, options, manifest)))
    return fail("device planning failed for the rejected fanout graph");
  auto tritonDispatches = manifest.getInteger("triton_dispatches");
  auto linalgDispatches = manifest.getInteger("linalg_dispatches");
  auto fusedProducers = manifest.getInteger("linalg_fused_producers");
  if (!tritonDispatches || *tritonDispatches != 0 || !linalgDispatches ||
      *linalgDispatches != 3 || !fusedProducers || *fusedProducers != 3)
    return fail("device planning half-fused the rejected LayerNorm subgraph");

  llvm::outs() << "passed: canonical selection; selector fanout rejection; "
                  "duplicate-affine rejection; device fanout 0 Triton + "
                  "3 native + 3 fused\n";
  return 0;
}
