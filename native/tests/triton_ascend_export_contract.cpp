#include "gac/DeviceLowering.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;

namespace {

int fail(StringRef message) {
  llvm::errs() << message << '\n';
  return 1;
}

OwningOpRef<ModuleOp> loadModule(StringRef path, MLIRContext &context) {
  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(path, &context);
  if (!module || failed(verify(*module)))
    return {};
  return module;
}

llvm::SmallString<256> childPath(StringRef parent, StringRef child) {
  llvm::SmallString<256> path(parent);
  llvm::sys::path::append(path, child);
  return path;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5)
    return fail("usage: gac-triton-ascend-export-contract "
                "ELIGIBLE-02-linalg.mlir REJECTED-02-linalg.mlir "
                "SIGMOID-GAC.mlir OUTPUT-DIR");
  if (llvm::sys::fs::exists(argv[4]))
    return fail("Ascend export contract output directory must not exist");

  DialectRegistry registry;
  gac::registerDeviceDialects(registry);
  gac::createLowerToDevicePass({}, nullptr)->getDependentDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // The export-only target is never allowed to fall through to native device
  // lowering when its artifact destination is absent.
  OwningOpRef<ModuleOp> missingDirectory = loadModule(argv[1], context);
  if (!missingDirectory)
    return fail("cannot parse eligible Ascend TTIR contract fixture");
  gac::DeviceOptions missingOptions;
  missingOptions.target = gac::DeviceTarget::Ascend;
  missingOptions.regularLowering = "linalg";
  llvm::json::Object ignoredManifest;
  if (succeeded(
          gac::lowerToDevice(*missingDirectory, missingOptions, ignoredManifest)))
    return fail("Ascend export without a Triton artifact directory succeeded");

  // The public C++ API must enforce the same constants-output contract as the
  // CLI.  Otherwise a graph with embedded weights could appear to export
  // successfully while silently omitting those weight files.
  OwningOpRef<ModuleOp> missingConstants = loadModule(argv[1], context);
  if (!missingConstants)
    return fail("cannot reload missing-constants Ascend contract fixture");
  llvm::SmallString<256> missingConstantsDirectory =
      childPath(argv[4], "missing-constants");
  missingOptions.tritonDirectory = missingConstantsDirectory.str().str();
  if (succeeded(
          gac::lowerToDevice(*missingConstants, missingOptions, ignoredManifest)))
    return fail("Ascend export without a constants directory succeeded");
  if (llvm::sys::fs::exists(missingConstantsDirectory))
    return fail("rejected missing-constants export created partial artifacts");

  llvm::SmallString<256> acceptedDirectory = childPath(argv[4], "accepted");
  OwningOpRef<ModuleOp> accepted = loadModule(argv[1], context);
  if (!accepted)
    return fail("cannot reload eligible Ascend TTIR contract fixture");
  gac::DeviceOptions options;
  options.target = gac::DeviceTarget::Ascend;
  options.regularLowering = "linalg";
  options.tritonDirectory = acceptedDirectory.str().str();
  options.constantsDirectory = acceptedDirectory.str().str();
  llvm::json::Object manifest;
  if (failed(gac::lowerToDevice(*accepted, options, manifest)))
    return fail("eligible Ascend TTIR-only export failed");

  auto target = manifest.getString("target");
  auto execution = manifest.getString("execution");
  auto artifactFormat = manifest.getString("artifact_format");
  auto runtime = manifest.getBoolean("runtime_available");
  auto tritonDispatches = manifest.getInteger("triton_dispatches");
  auto *dispatches = manifest.getArray("dispatches");
  if (!target || *target != "ascend" || !execution ||
      *execution != "artifact-only" || !artifactFormat ||
      *artifactFormat != "standard-ttir" || !runtime || *runtime ||
      !tritonDispatches || *tritonDispatches != 1 || !dispatches ||
      dispatches->size() != 1 || manifest.get("lmem_bytes") ||
      manifest.get("double_buffer"))
    return fail("Ascend root manifest is not an explicit TTIR-only contract");
  auto *dispatch = (*dispatches)[0].getAsObject();
  auto lowering = dispatch ? dispatch->getString("lowering") : std::nullopt;
  auto *triton = dispatch ? dispatch->getObject("triton") : nullptr;
  auto kind = triton ? triton->getString("kind") : std::nullopt;
  auto ttirFile = triton ? triton->getString("ttir_file") : std::nullopt;
  auto perDispatchFormat =
      triton ? triton->getString("artifact_format") : std::nullopt;
  auto signature =
      triton ? triton->getString("ttir_signature") : std::nullopt;
  if (!lowering || *lowering != "linalg-to-triton" || !kind ||
      *kind != "softmax" || !ttirFile || *ttirFile != "gac_op_0.ttir" ||
      !perDispatchFormat || *perDispatchFormat != "standard-ttir" ||
      !signature || signature->empty())
    return fail("Ascend dispatch manifest is missing its standard TTIR contract");
  if (triton->get("abi") || triton->get("lanes") ||
      triton->get("block_size") || triton->get("num_warps") ||
      dispatch->get("vortex") || dispatch->get("vortex_schedule"))
    return fail("Ascend TTIR-only dispatch leaked a Vortex launch or ABI field");

  llvm::SmallString<256> ttirPath =
      childPath(acceptedDirectory, *ttirFile);
  auto ttir = llvm::MemoryBuffer::getFile(ttirPath.str());
  if (!ttir)
    return fail("Ascend standard TTIR artifact was not written");
  StringRef text = (*ttir)->getBuffer();
  for (StringRef token : {"\"tt.func\"", "\"tt.get_program_id\"",
                          "\"tt.reduce\"", "\"tt.store\""})
    if (!text.contains(token))
      return fail("Ascend export TTIR is missing a required standard operation");
  if (text.contains("vortex") || text.contains("npubin"))
    return fail("Ascend standard TTIR artifact claims a backend binary path");

  bool hasDeviceFunction = false;
  accepted->walk([&](Operation *op) {
    hasDeviceFunction |= isa<func::FuncOp, LLVM::LLVMFuncOp>(op);
  });
  if (hasDeviceFunction)
    return fail("Ascend TTIR-only export emitted an LLVM/native device kernel");

  // Exercise the real GAC-to-Linalg route used by the GalaxeaVLA SwiGLU
  // graph.  The generated pointwise TTIR must retain the numerically stable
  // scalar sigmoid expansion, including standard math.exp, and no fallback.
  llvm::SmallString<256> sigmoidDirectory = childPath(argv[4], "sigmoid");
  OwningOpRef<ModuleOp> sigmoid = loadModule(argv[3], context);
  if (!sigmoid)
    return fail("cannot parse the GAC Sigmoid contract fixture");
  options.tritonDirectory = sigmoidDirectory.str().str();
  options.constantsDirectory = sigmoidDirectory.str().str();
  llvm::json::Object sigmoidManifest;
  if (failed(gac::lowerToDevice(*sigmoid, options, sigmoidManifest)))
    return fail("FP32 GAC Sigmoid did not lower to Ascend pointwise TTIR");
  auto sigmoidDispatches = sigmoidManifest.getInteger("triton_dispatches");
  auto *sigmoidRows = sigmoidManifest.getArray("dispatches");
  auto *sigmoidDispatch =
      sigmoidRows && sigmoidRows->size() == 1
          ? (*sigmoidRows)[0].getAsObject()
          : nullptr;
  auto *sigmoidContract =
      sigmoidDispatch ? sigmoidDispatch->getObject("triton") : nullptr;
  auto sigmoidKind =
      sigmoidContract ? sigmoidContract->getString("kind") : std::nullopt;
  if (!sigmoidDispatches || *sigmoidDispatches != 1 || !sigmoidDispatch ||
      !sigmoidContract || !sigmoidKind || *sigmoidKind != "pointwise" ||
      sigmoidDispatch->get("triton_exclusion"))
    return fail("Sigmoid dispatch did not remain on the all-Triton path");
  llvm::SmallString<256> sigmoidTtir =
      childPath(sigmoidDirectory, "gac_op_0.ttir");
  auto sigmoidBuffer = llvm::MemoryBuffer::getFile(sigmoidTtir.str());
  if (!sigmoidBuffer)
    return fail("Sigmoid standard TTIR artifact was not written");
  StringRef sigmoidText = (*sigmoidBuffer)->getBuffer();
  for (StringRef token : {"\"math.exp\"", "\"arith.cmpf\"",
                          "\"arith.select\"", "\"arith.negf\""})
    if (!sigmoidText.contains(token))
      return fail("Sigmoid TTIR is missing its stable scalar expansion");
  if (sigmoidText.contains("gac.sigmoid") ||
      sigmoidText.contains("vortex") || sigmoidText.contains("npubin"))
    return fail("Sigmoid TTIR leaked an unlowered op or backend binary path");
  bool sigmoidHasDeviceFunction = false;
  sigmoid->walk([&](Operation *op) {
    sigmoidHasDeviceFunction |= isa<func::FuncOp, LLVM::LLVMFuncOp>(op);
  });
  if (sigmoidHasDeviceFunction)
    return fail("Sigmoid TTIR export emitted an LLVM/native device kernel");

  llvm::SmallString<256> rejectedDirectory = childPath(argv[4], "rejected");
  OwningOpRef<ModuleOp> rejected = loadModule(argv[2], context);
  if (!rejected)
    return fail("cannot parse rejected Ascend TTIR contract fixture");
  options.tritonDirectory = rejectedDirectory.str().str();
  options.constantsDirectory = rejectedDirectory.str().str();
  llvm::json::Object rejectedManifest;
  if (succeeded(gac::lowerToDevice(*rejected, options, rejectedManifest)))
    return fail("unsupported Ascend dispatch silently used a native fallback");
  for (StringRef filename : {"gac_op_0.ttir", "gac_op_1.ttir"})
    if (llvm::sys::fs::exists(
            childPath(rejectedDirectory, filename).str()))
      return fail("failed Ascend preflight left a partial TTIR artifact");

  llvm::outs() << "passed: Ascend standard-TTIR-only manifest/artifact; "
                  "stable FP32 Sigmoid math.exp; no Vortex ABI, LLVM kernel, "
                  "native fallback, or partial rejected artifact\n";
  return 0;
}
