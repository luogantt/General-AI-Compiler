#include "gac/ConstantFolding.h"
#include "gac/CudaLowering.h"
#include "gac/DeviceLowering.h"
#include "gac/Dialect.h"
#include "gac/OnnxImporter.h"
#include "gac/Passes.h"
#include "gac/VortexLowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::Required,
                                        llvm::cl::desc("input.onnx"));
static llvm::cl::opt<std::string> output("o", llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    emit("emit", llvm::cl::init("optimized"),
         llvm::cl::desc("import | optimized | device | llvm"));
static llvm::cl::opt<std::string> target(
    "target", llvm::cl::init("vortex"),
    llvm::cl::desc("vortex | cuda | native (explicit host verification)"));
static llvm::cl::opt<std::string> manifestPath("manifest");
static llvm::cl::opt<std::string> constantsDir("constants-dir");
static llvm::cl::opt<std::string> stagesDir(
    "save-stages",
    llvm::cl::desc("Directory for verified per-layer MLIR snapshots"));
static llvm::cl::opt<bool> allowUnsupported("allow-unsupported",
                                            llvm::cl::init(false));
static llvm::cl::opt<bool> noOptimize("no-optimize", llvm::cl::init(false));
static llvm::cl::opt<std::string> vortexMatmulSchedule(
    "vortex-matmul-schedule", llvm::cl::init("scalar"),
    llvm::cl::desc("Vortex MatMul template: scalar | lmem-tiled | dxa-double-buffer | tcu"));
static llvm::cl::opt<int64_t> tileM("tile-m", llvm::cl::init(4));
static llvm::cl::opt<int64_t> tileN("tile-n", llvm::cl::init(4));
static llvm::cl::opt<int64_t> tileK("tile-k", llvm::cl::init(16));
static llvm::cl::opt<int64_t> vortexLanes("vortex-lanes", llvm::cl::init(4));
static llvm::cl::opt<int64_t> lmemBudget("lmem-budget", llvm::cl::init(16384),
    llvm::cl::desc("Maximum LMEM bytes per tiled Vortex CTA"));
static llvm::cl::opt<std::string> foldProfile(
    "constant-fold-profile", llvm::cl::init("default"),
    llvm::cl::desc(
        "Bounded constant geometry evaluation: default | bevformer-base"));

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ONNX to native GAC MLIR compiler\n");
  if (emit != "import" && emit != "optimized" && emit != "device" &&
      emit != "llvm") {
    llvm::errs() << "unknown --emit value\n";
    return 1;
  }
  if (target != "native" && target != "vortex" && target != "cuda") {
    llvm::errs() << "unknown --target value\n";
    return 1;
  }
  if (vortexMatmulSchedule != "scalar" && vortexMatmulSchedule != "lmem-tiled" &&
      vortexMatmulSchedule != "dxa-double-buffer" && vortexMatmulSchedule != "tcu") {
    llvm::errs() << "unknown --vortex-matmul-schedule\n";
    return 1;
  }
  if (vortexMatmulSchedule != "scalar" && target != "vortex") {
    llvm::errs() << "Vortex MatMul schedules require --target=vortex\n";
    return 1;
  }
  if (tileM <= 0 || tileN <= 0 || tileK <= 0 || vortexLanes <= 0 ||
      lmemBudget <= 0 || lmemBudget > INT32_MAX) {
    llvm::errs() << "tile dimensions, Vortex lanes and LMEM budget must be positive; "
                   "LMEM budget must fit signed 32-bit addressing\n";
    return 1;
  }
  if (foldProfile != "default" && foldProfile != "bevformer-base") {
    llvm::errs() << "unknown --constant-fold-profile\n";
    return 1;
  }
  gac::ConstantFoldBudget foldBudget =
      foldProfile == "bevformer-base" ? gac::ConstantFoldBudget::bevformerBase()
                                      : gac::ConstantFoldBudget();
  if (allowUnsupported && emit != "import") {
    llvm::errs() << "--allow-unsupported is inspection-only (--emit=import)\n";
    return 1;
  }
  DialectRegistry registry;
  gac::registerGacDialects(registry);
  gac::registerDeviceDialects(registry);
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  scf::SCFDialect, cf::ControlFlowDialect, LLVM::LLVMDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = gac::importOnnx(input, context, allowUnsupported);
  if (!module || failed(verify(*module)))
    return 1;
  (*module)->setAttr("gac.constant_fold_profile",
                     StringAttr::get(&context, foldProfile));
  auto snapshot = [&](llvm::StringRef name) -> bool {
    if (stagesDir.empty())
      return true;
    if (auto error = llvm::sys::fs::create_directories(stagesDir)) {
      llvm::errs() << error.message() << '\n';
      return false;
    }
    llvm::SmallString<256> path(stagesDir);
    llvm::sys::path::append(path, name);
    std::error_code error;
    llvm::ToolOutputFile file(path, error, llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << error.message() << '\n';
      return false;
    }
    module->print(file.os(), OpPrintingFlags().enableDebugInfo());
    file.os() << '\n';
    file.os().flush();
    if (file.os().has_error())
      return false;
    file.keep();
    return true;
  };
  if (!snapshot("01-import.mlir"))
    return 1;
  if (emit != "import") {
    // Preserve the shape/constant boundary even if later legalization rejects
    // an unresolved dimension. This is the same first pass and budget used by
    // the compilation pipeline, not a separate inspection approximation.
    PassManager shapes(&context);
    shapes.enableVerifier(true);
    shapes.addPass(gac::createShapeInferencePass(foldBudget));
    if (failed(shapes.run(*module)) || !snapshot("01-shaped.mlir"))
      return 1;
    PassManager pm(&context);
    pm.enableVerifier(true);
    pm.addPass(gac::createLegalizeTensorOpsPass());
    pm.addPass(gac::createShapeInferencePass(foldBudget));
    if (!noOptimize) {
      pm.addPass(gac::createNormalizeProvenancePass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      // CSE exposes equal update tensors in statically indexed ScatterND
      // chains. Prove full overwrite/broadcast only after these SSA values
      // are shared, then remove the now-dead data buffers before fusion.
      pm.addPass(gac::createLegalizeTensorOpsPass());
      pm.addPass(gac::createShapeInferencePass(foldBudget));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      pm.addPass(gac::createGraphFusionPass());
      pm.addPass(gac::createNormalizeProvenancePass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }
    pm.addPass(gac::createShapeInferencePass(foldBudget));
    pm.addPass(gac::createLayoutPropagationPass());
    if (failed(pm.run(*module)))
      return 1;
    if (!snapshot("02-optimized.mlir"))
      return 1;
  }
  llvm::json::Object plan;
  if (emit == "device" || emit == "llvm") {
    gac::DeviceOptions options{
        target == "native" ? gac::DeviceTarget::Native
        : target == "cuda" ? gac::DeviceTarget::CUDA
                           : gac::DeviceTarget::Vortex,
        constantsDir};
    options.vortexMatmulSchedule = vortexMatmulSchedule;
    options.tileM = tileM;
    options.tileN = tileN;
    options.tileK = tileK;
    options.vortexLanes = vortexLanes;
    options.lmemBudget = lmemBudget;
    PassManager lowering(&context);
    lowering.enableVerifier(true);
    lowering.addPass(gac::createLowerToDevicePass(options, &plan));
    if (failed(lowering.run(*module)))
      return 1;
    if (!snapshot("03-device.mlir"))
      return 1;
    if (emit == "llvm") {
      PassManager pm(&context);
      pm.enableVerifier(true);
      if (target == "vortex")
        pm.addPass(gac::createVortexToLLVMPass());
      else if (target == "cuda")
        pm.addPass(gac::createCudaMathToLibdevicePass());
      gac::addLLVMConversionPasses(pm);
      if (failed(pm.run(*module)))
        return 1;
      if (!snapshot("04-llvm.mlir"))
        return 1;
    }
  }
  if (failed(verify(*module)))
    return 1;
  std::error_code error;
  llvm::ToolOutputFile file(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << error.message() << '\n';
    return 1;
  }
  module->print(file.os());
  file.os() << '\n';
  if (!manifestPath.empty()) {
    llvm::ToolOutputFile manifest(manifestPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      llvm::errs() << error.message() << '\n';
      return 1;
    }
    manifest.os() << llvm::formatv("{0:2}\n",
                                   llvm::json::Value(std::move(plan)));
    manifest.keep();
  }
  file.keep();
  return 0;
}
