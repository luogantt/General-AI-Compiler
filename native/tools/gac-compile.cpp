#include "gac/ConstantFolding.h"
#include "gac/CudaLowering.h"
#include "gac/DeviceLowering.h"
#include "gac/Dialect.h"
#include "gac/OnnxImporter.h"
#include "gac/LinalgLowering.h"
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

// gac-compile 是原生 C++ 编译流水线的命令行入口。它本身不实现 Conv、
// MatMul 等算子的数学循环，而是负责把各个前端、Pass 和后端 Lowering 串起来：
//
//   ONNX
//     -> gac.* 高层 MLIR
//     -> 合法化、形状推导、图优化
//     -> （默认）Linalg 结构化 IR
//     -> 目标相关的 Device/Vortex/CUDA IR
//     -> LLVM Dialect MLIR
//
// 最后一步仍然输出 MLIR 文本。LLVM Dialect -> 真正 LLVM IR 的转换由构建脚本
// 随后调用 `mlir-translate --mlir-to-llvmir` 完成，不在这个 main() 中执行。
using namespace mlir;

// ------------------------------ 输入与输出 ------------------------------

// 位置参数：待导入的 ONNX 文件。
static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::Required,
                                        llvm::cl::desc("input.onnx"));
// `-o` 指定最终 MLIR 文本；默认 "-" 表示写到标准输出。
static llvm::cl::opt<std::string> output("o", llvm::cl::init("-"));

// 选择流水线在哪一层停止。每一种值都输出“执行完该阶段后的 ModuleOp”：
//   import    : ONNX 刚转换成 gac.* Operation；
//   optimized : 完成合法化、形状推导、图优化和布局传播；
//   linalg    : 规则算子继续转换成 Linalg；
//   device    : 已生成目标设备 Kernel/ABI，但仍可能含 SCF/Arith 等方言；
//   llvm      : 已全部转换为可交给 mlir-translate 的 LLVM Dialect。
static llvm::cl::opt<std::string>
    emit("emit", llvm::cl::init("optimized"),
         llvm::cl::desc("import | optimized | linalg | device | llvm"));

// 规则张量算子的两条 Lowering 路径：
//   linalg : 先保留为结构化循环，便于后续 Tiling/Fusion/Vectorization；
//   legacy : 直接选择已有的设备专用模板，作为明确的性能/正确性基线。
static llvm::cl::opt<std::string> regularLowering(
    "regular-lowering", llvm::cl::init("linalg"),
    llvm::cl::desc("linalg (structured regular ops) | legacy (device templates)"));

// 目标决定 Kernel ABI、线程索引、数学库以及最终所需的设备属性。
// native 只用于主机数值验证，不会在设备不支持时被自动选作回退后端。
static llvm::cl::opt<std::string> target(
    "target", llvm::cl::init("vortex"),
    llvm::cl::desc("vortex | cuda | native (explicit host verification)"));

// 可选辅助产物：设备执行计划、外置常量目录以及每阶段 MLIR 快照。
static llvm::cl::opt<std::string> manifestPath("manifest");
static llvm::cl::opt<std::string> constantsDir("constants-dir");
static llvm::cl::opt<std::string> stagesDir(
    "save-stages",
    llvm::cl::desc("Directory for verified per-layer MLIR snapshots"));

// -------------------------- 可选 Triton/Vortex 桥接 -----------------------

// 当指定目录时，符合条件的 Linalg pointwise dispatch 会另外导出 TTIR。
// GAC 的内存规划和依赖 DAG 仍然是主计划；Triton 只承接这些候选 Kernel 的实现。
static llvm::cl::opt<std::string> vortexTritonDir(
    "vortex-triton-dir",
    llvm::cl::desc("Emit eligible Linalg pointwise dispatches as TTIR in this directory"));
static llvm::cl::opt<int64_t> tritonBlock("triton-block", llvm::cl::init(128));
static llvm::cl::opt<int64_t> tritonNumWarps("triton-num-warps", llvm::cl::init(1));

// `allow-unsupported` 只允许前端保留未知 ONNX 节点以便检查，不能生成可执行代码。
// `no-optimize` 跳过可选的规范化、CSE 和融合，但仍执行正确性必需的合法化与推导。
static llvm::cl::opt<bool> allowUnsupported("allow-unsupported",
                                            llvm::cl::init(false));
static llvm::cl::opt<bool> noOptimize("no-optimize", llvm::cl::init(false));

// ----------------------------- Vortex 调度参数 ---------------------------

// 只有 legacy 路径使用这些固定 MatMul 模板；linalg 路径不会暗中绕回旧模板。
static llvm::cl::opt<std::string> vortexMatmulSchedule(
    "vortex-matmul-schedule", llvm::cl::init("scalar"),
    llvm::cl::desc("Vortex MatMul template: scalar | lmem-tiled | dxa-double-buffer | tcu"));
// BM/BN/BK：一个工作块负责的输出行、输出列以及归约轴分块大小。
static llvm::cl::opt<int64_t> tileM("tile-m", llvm::cl::init(4));
static llvm::cl::opt<int64_t> tileN("tile-n", llvm::cl::init(4));
static llvm::cl::opt<int64_t> tileK("tile-k", llvm::cl::init(16));
// 寄存器分块：每个 Vortex 线程负责多少行、多少列输出元素。
static llvm::cl::opt<int64_t> vortexThreadTileM("vortex-thread-tile-m", llvm::cl::init(1));
static llvm::cl::opt<int64_t> vortexThreadTileN("vortex-thread-tile-n", llvm::cl::init(1));
// TCU 路径是否把三条 TF32 WMMA 保留在同一个 convergent helper 中。
static llvm::cl::opt<bool> vortexTcuCombined("vortex-tcu-combined", llvm::cl::init(false),
    llvm::cl::desc("Keep the three TF32 WMMA instructions in one convergent helper"));
// 目标硬件 lane 数和单个 CTA 可使用的 LMEM 上限，用于模板合法性检查。
static llvm::cl::opt<int64_t> vortexLanes("vortex-lanes", llvm::cl::init(4));
static llvm::cl::opt<int64_t> lmemBudget("lmem-budget", llvm::cl::init(16384),
    llvm::cl::desc("Maximum LMEM bytes per tiled Vortex CTA"));

// 常量折叠采用预算而不是无限制执行，避免 BEVFormer 大型几何常量在编译期
// 消耗失控；bevformer-base 配置使用针对该模型验证过的预算。
static llvm::cl::opt<std::string> foldProfile(
    "constant-fold-profile", llvm::cl::init("default"),
    llvm::cl::desc(
        "Bounded constant geometry evaluation: default | bevformer-base"));

// ------------------------------ CUDA 调度参数 ----------------------------

// 这些选项同样只控制 legacy 模板；"auto" 是形状启发式选择，不是实测 autotune。
static llvm::cl::opt<std::string> cudaMatmulSchedule(
    "cuda-matmul-schedule", llvm::cl::init("scalar"),
    llvm::cl::desc("CUDA MatMul template: scalar | shared-tiled | auto (shape heuristic, not autotuning)"));
static llvm::cl::opt<int64_t> cudaTileM("cuda-tile-m", llvm::cl::init(32));
static llvm::cl::opt<int64_t> cudaTileN("cuda-tile-n", llvm::cl::init(32));
static llvm::cl::opt<int64_t> cudaTileK("cuda-tile-k", llvm::cl::init(16));
static llvm::cl::opt<int64_t> cudaThreadTileM("cuda-thread-tile-m", llvm::cl::init(2));
static llvm::cl::opt<int64_t> cudaThreadTileN("cuda-thread-tile-n", llvm::cl::init(2));
static llvm::cl::opt<std::string> cudaReductionSchedule(
    "cuda-reduction-schedule", llvm::cl::init("scalar"),
    llvm::cl::desc("CUDA Softmax/LayerNorm template: scalar | cooperative"));
// Conv/DCNv2 中一个 CUDA 线程连续计算的输出通道数量。
static llvm::cl::opt<int64_t> cudaConvChannelTile(
    "cuda-conv-channel-tile", llvm::cl::init(1),
    llvm::cl::desc("CUDA Conv/DCNv2 output channels per thread: 1 | 2 | 4 | 8"));

int main(int argc, char **argv) {
  // LLVM CommandLine 负责解析上面声明的所有全局 cl::opt。
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "ONNX to native GAC MLIR compiler\n");

  // ------------------------------ 参数合法性 ----------------------------
  // 在构造 MLIRContext 之前拒绝互相冲突的选项，使错误不会被误报成 Lowering 失败。

  // --emit 决定流水线停止点，而不是选择一种不同的模型语义。
  if (emit != "import" && emit != "optimized" && emit != "linalg" && emit != "device" &&
      emit != "llvm") {
    llvm::errs() << "unknown --emit value\n";
    return 1;
  }
  if (target != "native" && target != "vortex" && target != "cuda") {
    llvm::errs() << "unknown --target value\n";
    return 1;
  }
  if (regularLowering != "linalg" && regularLowering != "legacy") {
    llvm::errs() << "--regular-lowering must be linalg or legacy\n";
    return 1;
  }
  if (emit == "linalg" && regularLowering != "linalg") {
    llvm::errs() << "--emit=linalg requires --regular-lowering=linalg\n";
    return 1;
  }
  // Triton 桥接依赖 Linalg 结构、Vortex 目标、完整 LLVM 阶段和 manifest 中的
  // dispatch 元数据，因此四个条件缺一不可。
  if (!vortexTritonDir.empty() &&
      (target != "vortex" || regularLowering != "linalg" || emit != "llvm" ||
       manifestPath.empty())) {
    llvm::errs() << "--vortex-triton-dir requires --target=vortex, "
                   "--regular-lowering=linalg, --emit=llvm and --manifest\n";
    return 1;
  }
  // 固定设备模板与 Linalg 路径表达的是两种显式策略。禁止把旧模板参数悄悄
  // 混入 Linalg 流水线，否则输出 IR 看似合法，却无法判断真正采用了哪种调度。
  if (regularLowering == "linalg" &&
      (vortexMatmulSchedule != "scalar" || cudaMatmulSchedule != "scalar" ||
       cudaReductionSchedule != "scalar" || cudaConvChannelTile != 1)) {
    llvm::errs() << "Existing CUDA/Vortex templates require --regular-lowering=legacy; "
                   "they are not Linalg schedules. No implicit bypass is allowed.\n";
    return 1;
  }
  // Vortex MatMul 模板、目标和寄存器分块之间的约束。
  if (vortexMatmulSchedule != "scalar" && vortexMatmulSchedule != "lmem-tiled" &&
      vortexMatmulSchedule != "dxa-double-buffer" && vortexMatmulSchedule != "tcu") {
    llvm::errs() << "unknown --vortex-matmul-schedule\n";
    return 1;
  }
  if (vortexMatmulSchedule != "scalar" && target != "vortex") {
    llvm::errs() << "Vortex MatMul schedules require --target=vortex\n";
    return 1;
  }
  if (vortexThreadTileM <= 0 || vortexThreadTileM > 4 ||
      vortexThreadTileN <= 0 || vortexThreadTileN > 4) {
    llvm::errs() << "Vortex register tile dimensions must be in 1..4\n";
    return 1;
  }
  // 当前 TCU fragment 布局固定，不能再叠加 LMEM/DXA 使用的线程寄存器块。
  if (vortexMatmulSchedule == "tcu" &&
      (vortexThreadTileM != 1 || vortexThreadTileN != 1)) {
    llvm::errs() << "TCU has a fixed fragment layout; register tiles apply to LMEM/DXA\n";
    return 1;
  }
  if (vortexTcuCombined && (target != "vortex" || vortexMatmulSchedule != "tcu")) {
    llvm::errs() << "--vortex-tcu-combined requires --target=vortex and the tcu schedule\n";
    return 1;
  }
  // CUDA 模板的枚举、目标绑定以及 Tile 几何约束。
  if ((cudaMatmulSchedule != "scalar" && cudaMatmulSchedule != "shared-tiled" && cudaMatmulSchedule != "auto") ||
      (cudaReductionSchedule != "scalar" && cudaReductionSchedule != "cooperative")) {
    llvm::errs() << "unknown CUDA MatMul or reduction schedule\n";
    return 1;
  }
  if (target != "cuda" && (cudaMatmulSchedule != "scalar" || cudaReductionSchedule != "scalar" || cudaConvChannelTile != 1)) {
    llvm::errs() << "CUDA schedules require --target=cuda\n";
    return 1;
  }
  if (cudaConvChannelTile != 1 && cudaConvChannelTile != 2 &&
      cudaConvChannelTile != 4 && cudaConvChannelTile != 8) {
    llvm::errs() << "CUDA channel tile must be 1, 2, 4 or 8\n";
    return 1;
  }
  if (cudaTileM <= 0 || cudaTileN <= 0 || cudaTileK <= 0 || cudaTileM > 128 ||
      cudaTileN > 128 || cudaTileK > 128 || cudaThreadTileM <= 0 ||
      cudaThreadTileN <= 0 || cudaThreadTileM > 8 || cudaThreadTileN > 8 ||
      cudaTileM % cudaThreadTileM || cudaTileN % cudaThreadTileN) {
    llvm::errs() << "invalid CUDA output/reduction or per-thread tile\n";
    return 1;
  }
  // 所有分块和硬件资源参数必须能被 RV32 地址/索引模型安全表示。
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
  // 把字符串配置转换成真正传给 ShapeInference/ConstantFolding Pass 的预算对象。
  gac::ConstantFoldBudget foldBudget =
      foldProfile == "bevformer-base" ? gac::ConstantFoldBudget::bevformerBase()
                                      : gac::ConstantFoldBudget();
  // 未知算子只能停留在导入检查产物中；后续 Pass 不允许把它当成可执行语义。
  if (allowUnsupported && emit != "import") {
    llvm::errs() << "--allow-unsupported is inspection-only (--emit=import)\n";
    return 1;
  }
  // ------------------------- Dialect 与 MLIRContext ----------------------
  // MLIR 只能解析、创建和验证已经注册的 Operation。这里把整条流水线可能出现的
  // 方言一次注册完整；这并不意味着每个输入模型一定会用到所有方言。
  DialectRegistry registry;
  // GAC 高层图方言：由 ONNX Importer 生成 gac.conv2d、gac.matmul 等 Operation。
  gac::registerGacDialects(registry);
  // 设备相关方言和接口：Vortex、CUDA、MemRef/Affine 等。
  gac::registerDeviceDialects(registry);
  // 规则算子的结构化中间层，例如 linalg.matmul / linalg.generic。
  gac::registerLinalgDialects(registry);
  // 后续 Pass 会产生的标准 MLIR 方言以及最终 LLVM Dialect。
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  scf::SCFDialect, cf::ControlFlowDialect, LLVM::LLVMDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // ---------------------------- 第 1 层：模型导入 ------------------------
  // importOnnx 直接创建 mlir::ModuleOp：ONNX 名称连接被替换成 SSA Value 连接，
  // 权重成为常量 Operation，算子成为 gac.* Operation。这里还没有执行模型。
  auto module = gac::importOnnx(input, context, allowUnsupported);
  // 每个主要阶段都开启 verifier；一旦类型、支配关系或 Operation 约束不成立，
  // 立即停止，避免错误 IR 继续进入后端。
  if (!module || failed(verify(*module)))
    return 1;

  // 把折叠策略写进 Module 属性，使后续 Pass 和保存下来的 IR 都携带同一配置。
  (*module)->setAttr("gac.constant_fold_profile",
                     StringAttr::get(&context, foldProfile));

  // 可选的阶段快照函数。它保存的是当前内存中同一个 ModuleOp 的文本表示，
  // enableDebugInfo() 会保留源位置，方便从某个 Operation 追溯回 ONNX 节点。
  // ToolOutputFile 只有调用 keep() 才会保留目标文件，失败时不会留下半写产物。
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

  // 01-import.mlir：未经图优化的导入结果，是检查 ONNX -> GAC 映射的边界。
  if (!snapshot("01-import.mlir"))
    return 1;

  // ------------------- 第 2～4 层：类型、优化与合法化 --------------------
  // --emit=import 在上面的边界停止；其他输出层都必须先获得可验证的静态语义。
  if (emit != "import") {
    // Preserve the shape/constant boundary even if later legalization rejects
    // an unresolved dimension. This is the same first pass and budget used by
    // the compilation pipeline, not a separate inspection approximation.
    PassManager shapes(&context);
    shapes.enableVerifier(true);
    shapes.addPass(gac::createShapeInferencePass(foldBudget));
    // 单独保存 shape 边界：即使稍后的合法化失败，仍可观察前端推导到了哪里。
    if (failed(shapes.run(*module)) || !snapshot("01-shaped.mlir"))
      return 1;

    // 这一组 Pass 仍在高层计算图上工作：先把后端不能直接接受的 Tensor 形式
    // 改写成规范形式，再根据改写后的图重新推导 shape。
    PassManager pm(&context);
    pm.enableVerifier(true);
    pm.addPass(gac::createLegalizeTensorOpsPass());
    pm.addPass(gac::createShapeInferencePass(foldBudget));
    if (!noOptimize) {
      // Provenance 规范化让来源属性稳定，便于 CSE 判断两个 Operation 是否等价。
      pm.addPass(gac::createNormalizeProvenancePass());
      // Canonicalizer 执行各 Operation 注册的局部重写；CSE 合并等价的纯计算。
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      // CSE exposes equal update tensors in statically indexed ScatterND
      // chains. Prove full overwrite/broadcast only after these SSA values
      // are shared, then remove the now-dead data buffers before fusion.
      pm.addPass(gac::createLegalizeTensorOpsPass());
      pm.addPass(gac::createShapeInferencePass(foldBudget));
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
      // 在规范化和公共子表达式消除之后匹配 Conv/Bias/Activation、Residual 等
      // 已注册融合模式；融合必须保持完整数学语义和使用关系。
      pm.addPass(gac::createGraphFusionPass());
      pm.addPass(gac::createNormalizeProvenancePass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }
    // 图结构发生变化后再做一次最终形状推导，并把布局信息向消费者传播。
    pm.addPass(gac::createShapeInferencePass(foldBudget));
    pm.addPass(gac::createLayoutPropagationPass());
    if (failed(pm.run(*module)))
      return 1;
    if (!snapshot("02-optimized.mlir"))
      return 1;
  }

  // plan 由设备 Lowering 填充，描述每个 dispatch、依赖、模板、Buffer 等 Runtime
  // 需要的信息；它和生成的 Kernel IR 是同一次 Lowering 的两个输出。
  llvm::json::Object plan;

  // 只有真正进入设备 Codegen 时才要求整张图都能被目标后端实现。
  // import/optimized/linalg 可以作为检查产物保留到更高层。
  if (emit == "device" || emit == "llvm")
    if (failed(gac::validateDeviceGraph(*module)))
      return 1;

  // ------------------------- 第 5～7 层：Linalg ---------------------------
  // 默认路线将规则算子转换成结构化 Linalg。Linalg 仍然表达相同数学运算，
  // 但显式暴露迭代空间和索引映射，供 Tiling、Fusion 和设备映射使用。
  if (regularLowering == "linalg" &&
      (emit == "linalg" || emit == "device" || emit == "llvm")) {
    PassManager structured(&context);
    structured.enableVerifier(true);
    structured.addPass(gac::createLowerRegularToLinalgPass());
    if (failed(structured.run(*module)) || !snapshot("02-linalg.mlir"))
      return 1;
  }

  // ---------------------- 第 8～9 层：设备 Codegen -----------------------
  if (emit == "device" || emit == "llvm") {
    // 将命令行配置复制到一个显式的 DeviceOptions。Lowering 只依赖这个对象，
    // 不直接读取全局 cl::opt，因此同一套 Pass 也能由测试或其他工具复用。
    gac::DeviceOptions options{
        target == "native" ? gac::DeviceTarget::Native
        : target == "cuda" ? gac::DeviceTarget::CUDA
                           : gac::DeviceTarget::Vortex,
        constantsDir};

    // Vortex 模板及硬件资源约束。
    options.vortexMatmulSchedule = vortexMatmulSchedule;
    options.tileM = tileM;
    options.tileN = tileN;
    options.tileK = tileK;
    options.vortexThreadTileM = vortexThreadTileM;
    options.vortexThreadTileN = vortexThreadTileN;
    options.vortexTcuCombined = vortexTcuCombined;
    options.vortexLanes = vortexLanes;
    options.lmemBudget = lmemBudget;

    // CUDA 模板、分块和每线程工作量。
    options.cudaMatmulSchedule = cudaMatmulSchedule;
    options.cudaTileM = cudaTileM;
    options.cudaTileN = cudaTileN;
    options.cudaTileK = cudaTileK;
    options.cudaThreadTileM = cudaThreadTileM;
    options.cudaThreadTileN = cudaThreadTileN;
    options.cudaReductionSchedule = cudaReductionSchedule;
    options.cudaConvChannelTile = cudaConvChannelTile;

    // 公共 Lowering 路线以及可选 Triton 导出设置。
    options.regularLowering = regularLowering;
    options.vortexTritonDirectory = vortexTritonDir;
    options.tritonBlock = tritonBlock;
    options.tritonNumWarps = tritonNumWarps;

    // createLowerToDevicePass 负责把每个合法子图变成目标 Kernel，并同时建立
    // dispatch/Buffer/ABI 计划。输出的 03-device.mlir 还不保证只剩 LLVM Dialect。
    PassManager lowering(&context);
    lowering.enableVerifier(true);
    lowering.addPass(gac::createLowerToDevicePass(options, &plan));
    if (failed(lowering.run(*module)))
      return 1;
    if (!snapshot("03-device.mlir"))
      return 1;
    if (emit == "llvm") {
      // ------------------------- LLVM Dialect 边界 ------------------------
      // Vortex 专用 Operation 先变成 RISC-V/Vortex intrinsic 或 inline asm；
      // CUDA 数学调用则映射到 libdevice。二者之后共用标准 MLIR 转换 Pass，
      // 将 Affine/SCF/Math/MemRef/Arith/CF 全部收敛到 LLVM Dialect。
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

  // 最终防线：输出之前再次验证整个 Module。这里成功只说明 MLIR 合法；
  // `--emit=llvm` 的结果仍需由外部 mlir-translate 转成 kernel.ll。
  if (failed(verify(*module)))
    return 1;

  // ------------------------------ 写出产物 ------------------------------
  // 主输出始终是当前停止层的 MLIR 文本。
  std::error_code error;
  llvm::ToolOutputFile file(output, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << error.message() << '\n';
    return 1;
  }
  module->print(file.os());
  file.os() << '\n';

  // manifest 是面向 Runtime/测试工具的 JSON 计划，不是 MLIR，也不包含权重数据。
  // std::move(plan) 表示写出后不再使用内存中的 plan 对象。
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
  // 只有所有输出都成功写完才提交主文件，保证失败不会伪装成完整编译结果。
  file.keep();
  return 0;
}
