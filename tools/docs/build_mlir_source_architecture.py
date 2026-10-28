#!/usr/bin/env python3
"""Generate an offline source map of the local LLVM/MLIR C++ framework.

Only reads the LLVM tree. It does not compile MLIR or change compiler sources.
Default excerpts were selected against the local LLVM 20.1.8 tree; each embeds
its actual line numbers and whole-file SHA-256. Recheck ranges on version changes.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[2]


def node(key, title, short, category, description, points, sources, related=()):
    return dict(id=key, title=title, short=short, category=category,
                description=description, points=points, sources=sources,
                related=list(related))


NODES = [
    node("opt", "mlir-opt", "tools/mlir-opt → lib/Tools", "工具入口",
         "注册方言和 Pass，读取 IR，按命令行配置运行变换，再输出结果。",
         ["main 调用通用 MlirOptMain 驱动；驱动组合 Parser、PassManager 与输出器。",
          "它不自动决定模型的最佳 Tiling，也不负责运行 GPU 推理。"],
         [("tools/mlir-opt/mlir-opt.cpp", 305, 28),
          ("lib/Tools/mlir-opt/MlirOptMain.cpp", 443, 37)], ["parser", "pass", "printer"]),
    node("parser", "Parser / Bytecode Reader", "include/mlir/Parser · lib/AsmParser", "读入 IR",
         "把文本或 MLIR bytecode 解析成统一的内存 IR，而不是执行这些操作。",
         ["公共 Parser 入口识别 bytecode；文本交给 AsmParser，二进制交给 Bytecode Reader。",
          "解析时借助 Context 中已注册的方言识别语法、类型和操作。"],
         [("lib/Parser/Parser.cpp", 20, 26),
          ("lib/AsmParser/Parser.cpp", 2840, 24),
          ("include/mlir/Bytecode/BytecodeReader.h", 75, 24)], ["operation", "context"]),
    node("bindings", "C API / Python 绑定", "lib/CAPI · lib/Bindings/Python", "API 入口",
         "其他语言通过绑定创建和操作同一套 C++ MLIR 对象。",
         ["Python 层可以调用真实的 MLIR；是否使用 MLIR 取决于底层对象与变换机制。",
          "C API 暴露 C 兼容句柄；Python 的 C++ 绑定位于 lib/Bindings/Python，包位于 python/mlir。"],
         [("include/mlir-c/IR.h", 45, 24),
          ("lib/CAPI/IR/IR.cpp", 1, 24),
          ("lib/Bindings/Python/IRCore.cpp", 1, 24)], ["builder", "operation"]),
    node("operation", "Operation · 公共节点", "include/mlir/IR/Operation.h", "所有方言共用的 IR 存储",
         "统一保存一个操作的名称、输入、结果、属性 / Properties、嵌套 Region 及后继等信息。",
         ["Operation 是 final 类。具体 AddFOp / ForOp 通过轻量 Op 包装访问它，不继承这份存储。",
          "实际操作实例位于 Block 中；Dialect 注册操作的描述与规则，不保存整个模型图。"],
         [("include/mlir/IR/Operation.h", 65, 42),
          ("lib/IR/Operation.cpp", 34, 26)], ["opwrapper", "value", "region", "types", "builder"]),
    node("context", "MLIRContext", "include/mlir/IR/MLIRContext.h", "共享服务",
         "维护已加载方言、类型与属性的唯一化存储，并提供诊断和线程等服务。",
         ["Type / Attribute 的句柄引用 Context 管理的存储，减少相同元数据的重复分配。",
          "Context 并不拥有所有 Module / Operation；IR 程序树的所有权需单独管理。"],
         [("include/mlir/IR/MLIRContext.h", 60, 30),
          ("lib/IR/MLIRContext.cpp", 289, 25)], ["dialect", "types", "operation"]),
    node("dialect", "Dialect / Registry", "include/mlir/IR/Dialect*.h", "扩展与注册",
         "按命名空间组织操作、类型、属性和接口。Registry 登记构造方法，Context 加载方言实例。",
         ["addOperations 注册具体操作的描述、验证等钩子；实际 Op 仍使用公共 IR 存储。",
          "arith、scf、linalg 等可在同一 Module 共存。方言不是硬件，也不是固定流水阶段。"],
         [("include/mlir/IR/Dialect.h", 38, 27),
          ("include/mlir/IR/Dialect.h", 269, 30),
          ("include/mlir/IR/DialectRegistry.h", 139, 28)], ["opwrapper", "interfaces", "tablegen", "transform"]),
    node("arith", "Arith 方言", "标量 / 向量 / 张量算术", "具体操作语义",
         "定义加法、乘法、比较和类型转换等操作；结果形状与类型由各操作规则约束。",
         ["例如 arith.addf 定义浮点加法；操作类仍使用通用 Operation。",
          "ArithToLLVM Pass 可以把适用的算术操作转换到 LLVM Dialect。"],
         [("include/mlir/Dialect/Arith/IR/ArithOps.td", 911, 31),
          ("lib/Conversion/ArithToLLVM/ArithToLLVM.cpp", 465, 30)], ["tablegen", "conversion", "llvm"]),
    node("scf", "SCF 方言", "结构化循环与条件", "具体操作语义",
         "用 scf.for、scf.if 等操作保留循环与条件结构，便于变换。",
         ["scf.for 本身是 Operation，循环体放在它的 Region / Block 中。",
          "循环迭代参数通过 BlockArgument 和 yield 传递；继续 Lowering 时可转换成控制流。"],
         [("include/mlir/Dialect/SCF/IR/SCFOps.td", 136, 35)], ["region", "block", "value", "conversion"]),
    node("linalg", "Linalg 方言", "结构化张量 / Buffer 运算", "具体操作语义",
         "保留迭代空间、索引映射和并行 / 归约维度，为分块、融合等变换提供结构。",
         ["与逐条标量循环相比，这一层保留更多矩阵与张量运算结构。",
          "接口名称：ODS 的 LinalgStructuredInterface 生成 C++ LinalgOp 接口；LinalgOpTilingInterface 外部模型再把具体 Linalg 操作接到公共 TilingInterface。",
          "输入：操作的 operand 维度和形状到迭代空间的映射；getIterationDomain 返回每个循环的 Range。",
          "输入：tile 的 offsets / sizes；getTiledImplementation 通过 makeTiledShapes 生成所需切片，克隆分块算子并调整索引，返回 TilingResult。",
          "接入动作：registerTilingInterfaceExternalModels 通过 registry extension 为 GenericOp 和命名 Linalg 操作 attachInterface；通用分块代码随后调用同一接口。"],
         [("include/mlir/Dialect/Linalg/IR/LinalgInterfaces.td", 225, 11),
          ("lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp", 95, 17),
          ("lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp", 113, 30),
          ("lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp", 568, 23)],
         ["transform", "interfaces", "conversion", "scf"]),
    node("llvm", "LLVM Dialect", "MLIR 内的 LLVM 风格操作", "低层 IR 方言",
         "以 MLIR Operation 表示接近 LLVM IR 的指针、运算、函数和控制流。",
         ["LLVM Dialect 仍然属于 MLIR。llvm.fadd 操作还不是 LLVM IR 的 llvm::Instruction。",
          "导出器再把这些操作转换为真正的 llvm::Module，交给 LLVM 工具链。"],
         [("include/mlir/Dialect/LLVMIR/LLVMOps.td", 217, 30)], ["translation", "conversion", "operation"]),
    node("pass", "Pass / PassManager", "include/mlir/Pass · lib/Pass", "组织变换",
         "管理配置好的 Pass 流水线，并在指定 Operation 上调用每个 Pass 的 runOnOperation。",
         ["Pass 可做分析、检查或修改；不是每个 Pass 都调用 Pattern，也不是 GPU 任务调度器。",
          "Pass 可选择 Greedy rewrite、Dialect Conversion 或 Transform interpreter 等不同机制。"],
         [("include/mlir/Pass/PassManager.h", 231, 30),
          ("lib/Pass/Pass.cpp", 505, 30),
          ("lib/Transforms/Canonicalizer.cpp", 55, 24)], ["analysis", "rewrite", "conversion", "transform"]),
    node("rewrite", "Pattern / Rewriter", "IR/PatternMatch · Transforms/Utils", "共享改写机制",
         "Pattern 描述匹配条件与改写方法；Rewriter 执行受管理的插入、替换和删除。",
         ["Greedy driver 选择并反复应用模式；不是每个方言各写一套 use-def 维护算法。",
          "共同框架负责安全更新 IR，具体 pattern 负责算子特有的数学与类型规则。"],
         [("include/mlir/IR/PatternMatch.h", 246, 31),
          ("lib/IR/PatternMatch.cpp", 133, 29),
          ("include/mlir/Transforms/GreedyPatternRewriteDriver.h", 118, 34)], ["value", "interfaces", "pass", "conversion"]),
    node("conversion", "Dialect Conversion", "Transforms/DialectConversion.h", "合法性驱动的转换框架",
         "按照目标合法性、类型转换与重写规则，将不被目标接受的操作转换成合法形式。",
         ["ConversionTarget 判定操作是否合法；TypeConverter 转换类型；ConversionPattern 描述具体改写。",
          "它与 Greedy rewrite 是不同的改写驱动。lib/Conversion/ 中放各条具体 Lowering，而非一个万能转换算法。"],
         [("include/mlir/Transforms/DialectConversion.h", 940, 30),
          ("include/mlir/Transforms/DialectConversion.h", 41, 25),
          ("lib/Transforms/Utils/DialectConversion.cpp", 3355, 32),
          ("lib/Conversion/ArithToLLVM/ArithToLLVM.cpp", 472, 25)], ["rewrite", "llvm", "translation"]),
    node("printer", "AsmPrinter / Bytecode", "lib/IR · lib/Bytecode", "序列化 IR",
         "把内存中的 MLIR 程序写成可读文本或紧凑二进制，便于检查、存储与交换。",
         ["Operation::print 输出 .mlir 文本；BytecodeWriter 输出 MLIR bytecode。",
          "这两种产物都保存 IR；它们不是 CPU/GPU 的机器码。"],
         [("lib/IR/AsmPrinter.cpp", 4009, 28),
          ("include/mlir/Bytecode/BytecodeWriter.h", 184, 26)], ["parser", "operation", "translation"]),
    node("translation", "LLVM IR Translation", "tools/mlir-translate · lib/Target/LLVMIR", "从 MLIR 导出 LLVM IR",
         "mlir-translate 运行所选翻译器；LLVM 导出器创建真正的 llvm::Module、函数和指令。",
         ["ModuleTranslation 通过方言翻译接口分派具体操作，再使用 LLVM IRBuilder 构建 LLVM IR。",
          "与 Dialect Conversion 区别：Conversion 在 MLIR 内改写；这里跨出 MLIR 表示。必须先满足导出器要求。"],
         [("tools/mlir-translate/mlir-translate.cpp", 39, 27),
          ("lib/Target/LLVMIR/ConvertToLLVMIR.cpp", 28, 26),
          ("lib/Target/LLVMIR/ModuleTranslation.cpp", 2029, 29),
          ("lib/Target/LLVMIR/ModuleTranslation.cpp", 931, 24)], ["llvm", "conversion", "engine"]),
    node("engine", "ExecutionEngine（可选）", "include/mlir/ExecutionEngine", "LLVM JIT 集成",
         "提供基于 LLVM JIT 的编译、链接与调用集成，供需要即时执行的应用使用。",
         ["这是 MLIR 项目中的可选库；并非每条编译流水都需要它。",
          "它不等于 Vortex 或 CUDA 的完整设备 Runtime。GAC 的设备加载、传参和启动由项目 Runtime 负责。"],
         [("include/mlir/ExecutionEngine/ExecutionEngine.h", 95, 39)], ["translation"]),
    node("module", "ModuleOp", "include/mlir/IR/BuiltinOps.td", "程序容器，也是一种 Op",
         "builtin.module 是顶层容器操作，通常容纳函数、符号和常量等。",
         ["ModuleOp 也通过类型化包装引用 Operation，没有另建一套图存储。",
          "Module 的 Region 中有 Block，Block 里可放 func.func；函数又可拥有自己的 Region。",
          "mlir-opt 默认可给没有显式 module 的输入补顶层 ModuleOp；工具 helper 调用 parseSourceFile<ModuleOp>，公共 parser 将读出的操作放入容器。"],
         [("include/mlir/IR/BuiltinOps.td", 33, 32),
          ("include/mlir/Dialect/Func/IR/FuncOps.td", 226, 28),
          ("include/mlir/Tools/ParseUtilities.h", 21, 14),
          ("include/mlir/Parser/Parser.h", 66, 10)], ["operation", "region", "opwrapper"]),
    node("region", "Region", "include/mlir/IR/Region.h", "包含关系",
         "Operation 内部的一组 Block，用于表达函数体、循环体或其他嵌套程序。",
         ["Region 的 Block 列表由 llvm::iplist<Block> 保存。",
          "Region 是控制流区域还是图区域，具体语义由所属操作及接口约定。"],
         [("include/mlir/IR/Region.h", 26, 35)], ["block", "operation", "scf"]),
    node("block", "Block", "include/mlir/IR/Block.h", "包含关系",
         "保存 Block 参数与有序 Operation 列表。它是编译器结构，不是 GPU 线程块。",
         ["BlockArgument 可表示函数参数，或控制流 / 循环中传递进来的值。",
          "Operation 列表可以继续嵌套 Region，从而表示函数、分支与循环。"],
         [("include/mlir/IR/Block.h", 32, 25),
          ("include/mlir/IR/Block.h", 129, 28)], ["region", "operation", "value"]),
    node("value", "Value / SSA", "include/mlir/IR/Value.h", "数据依赖关系",
         "带类型的 SSA 值，来自某个操作结果或 Block 参数，并记录它被哪些操作使用。",
         ["生产者的结果被消费者作为 operand 使用，形成叠加在嵌套结构上的数据依赖。",
          "Value 是程序表示中的句柄，不是装有推理结果的数组；replaceAllUsesWith 可更新其使用链。"],
         [("include/mlir/IR/Value.h", 96, 30),
          ("include/mlir/IR/Value.h", 161, 28)], ["types", "operation", "block", "rewrite"]),
    node("types", "Type / Attribute", "IR/Types.h · IR/Attributes.h", "共享元数据",
         "Type 描述值的数据类型；Attribute 保存编译期已知信息，如常量、形状列表和操作参数。",
         ["Type / Attribute 是轻量句柄，引用 Context 管理的唯一化存储。",
          "Operation 还支持自有的 C++ Properties；Properties 与 Attribute 并非同一份存储。"],
         [("include/mlir/IR/Types.h", 74, 29),
          ("include/mlir/IR/Attributes.h", 25, 28)], ["context", "value", "operation"]),
    node("opwrapper", "Op<T> / OpState", "include/mlir/IR/OpDefinition.h", "C++ 类型化包装",
         "具体 Op 类复用 Op 模板与 OpState，通过内部 Operation* 访问通用节点。",
         ["Op<T> 继承 OpState 并组合 Traits；Operation 自身是 final，因此不是传统的 Operation→MatMulOp 继承树。",
          "包装提供特定操作的访问方法与验证入口；创建新包装不会复制整个操作节点。",
          "OpState 与 OperationState 不同：前者保存已创建 Operation*；后者在创建前收集名称、operands、types、attributes 等参数。",
          "公共 OpBuilder::create(state) 返回 Operation*；模板 create<OpTy> 收到该指针后执行 dyn_cast<OpTy>(op)，返回类型化包装。"],
         [("include/mlir/IR/OpDefinition.h", 96, 30),
          ("include/mlir/IR/OpDefinition.h", 238, 23),
          ("include/mlir/IR/OpDefinition.h", 1658, 32),
          ("include/mlir/IR/OperationSupport.h", 947, 11)], ["operation", "interfaces", "tablegen", "op-create"]),
    node("interfaces", "Traits / Interfaces", "OpDefinition · Interfaces · Support", "公共算法怎样识别操作能力",
         "Traits 组合静态性质与约束；Interfaces 让通用算法查询不同操作的行为。",
         ["例如 OneResult 描述结果数量；内存副作用接口说明操作是否读写内存。",
          "通用算法可通过接口复用逻辑，但具体算子的语义和接口实现仍由其方言提供。",
          "可核查的调用：scf::tileUsingSCF 接受 TilingInterface 和用户分块选项，先调用 op.getIterationDomain(rewriter) 获取迭代域。",
          "普通 FullReduction 路径调用 op.getTiledImplementation(rewriter, offsets, sizes)；Linalg 的外部模型负责生成对应切片和分块算子。",
          "因此公共 SCF 分块算法负责循环组织，具体操作通过 TilingInterface 提供迭代域和 tile 实现；接口不自动选择最佳 tile 大小。"],
         [("include/mlir/IR/OpDefinition.h", 2078, 29),
          ("include/mlir/Interfaces/TilingInterface.td", 75, 38),
          ("lib/Dialect/SCF/Transforms/TileUsingInterface.cpp", 933, 19),
          ("lib/Dialect/SCF/Transforms/TileUsingInterface.cpp", 611, 8)],
         ["opwrapper", "rewrite", "dialect", "linalg", "scf"]),
    node("builder", "OpBuilder / OperationState", "IR/Builders.h · IR/OperationSupport.h", "创建 IR",
         "OperationState 暂存名称、操作数、类型等构造参数；OpBuilder 创建并插入真实 Operation。",
         ["Builder 的插入点决定新操作落在哪个 Block 的哪个位置。",
          "create<arith::MulFOp> 是创建一条乘法操作；编译器此时并不执行运行时张量的乘法。"],
         [("include/mlir/IR/Builders.h", 205, 29),
          ("lib/IR/Builders.cpp", 453, 17),
          ("include/mlir/IR/OperationSupport.h", 947, 30)], ["operation", "block", "value"]),
    node("analysis", "AnalysisManager", "include/mlir/Pass/AnalysisManager.h", "分析结果管理",
         "按需计算和缓存操作上的分析结果，并在变换后根据保留信息失效缓存。",
         ["Pass 可以查询支配关系、数据流等分析，而不必每次自行管理缓存。",
          "分析框架可复用，具体分析仍需相应的算法与语义信息。"],
         [("include/mlir/Pass/AnalysisManager.h", 295, 35)], ["pass", "interfaces"]),
    node("transform", "Transform Dialect", "lib/Dialect/Transform", "用 IR 表达编译变换",
         "Transform IR 描述变换步骤，handle 指向待改写的 payload IR；解释器在编译期间执行这些步骤。",
         ["Interpreter Pass 调用 TransformOpInterface::apply，进而改写 payload IR。",
          "它可以组织分块、融合等策略；并不等于自动调优搜索器，也不是在 GPU 上解释模型。"],
         [("lib/Dialect/Transform/Transforms/InterpreterPass.cpp", 129, 32),
          ("lib/Dialect/Transform/Transforms/TransformInterpreterUtils.cpp", 201, 38)], ["pass", "linalg", "rewrite"]),
    node("tablegen", "ODS / TableGen", "include/mlir/IR/OpBase.td · tools/mlir-tblgen", "生成操作类与辅助代码",
         "用 .td 声明操作的名字、参数、结果、Traits 与接口，再生成相应 C++ 代码。",
         ["例如 ArithOps.td 定义 Arith_AddFOp；生成的类使用相同的 Op<T> 基础设施。",
          "ODS 是可选的定义方式，也可手写 C++ Op。生成的 .inc 文件通常位于构建目录。"],
         [("include/mlir/IR/OpBase.td", 1, 32)], ["arith", "opwrapper", "interfaces"]),
    node("registration", "把 arith.addi 注册到共享框架", "ArithDialect → addOperations → Model<AddIOp>", "注册具体行为",
         "Arith 方言初始化时注册操作列表；公共注册模板为每种操作建立 Model，把名字和具体 C++ 钩子接起来。",
         ["输入：ArithDialect 实例，以及由 ArithOps.cpp.inc 提供的具体操作类型列表，其中包含 AddIOp。",
          "调用：ArithDialect::initialize → Dialect::addOperations<...> → RegisteredOperationName::insert<T>。",
          "输出 / 动作：insert 创建 Model<T>，其中保留具体操作的 parse、fold 等行为；这一步没有创建某条程序中的 addi 实例。"],
         [("lib/Dialect/Arith/IR/ArithDialect.cpp", 41, 5),
          ("include/mlir/IR/Dialect.h", 275, 10),
          ("include/mlir/IR/OperationSupport.h", 682, 10)],
         ["dialect", "context", "parse-hook", "fold-dispatch"]),
    node("parse-hook", "文本名字 → 解析钩子 → Operation", "Parser lookup → ConcreteOp::parse → create", "读入一条具体操作",
         "解析器用 arith.addi 这个名字查询注册信息，取得具体操作的解析函数，再把解析出的 OperationState 交给公共 Builder。",
         ["输入：文本中的操作名、操作数、结果类型和属性，以及当前 MLIRContext。",
          "调用：parseCustomOperationName 先 lookup；未命中则按命名空间 getOrLoadDialect，再取得 OperationName。",
          "调用：getParseAssemblyFn 返回 ConcreteOp::parse；CustomOpAsmParser 调它填充 OperationState，随后 opBuilder.create(opState)。",
          "输出 / 动作：创建内存中的公共 Operation；文本里的 %x 等名字被解析为 SSA Value 引用。"],
         [("lib/AsmParser/Parser.cpp", 1977, 32),
          ("lib/AsmParser/Parser.cpp", 2021, 13),
          ("include/mlir/IR/OperationSupport.h", 543, 3),
          ("lib/AsmParser/Parser.cpp", 2074, 22)],
         ["parser", "registration", "op-create", "value"]),
    node("op-create", "具体 Op 构造 → 公共创建与插入", "OpTy::build → Operation::create → Block.insert", "创建共享 IR 节点",
         "模板 Builder 先让具体 Op 填充构造参数，再走所有方言共用的 Operation 分配和 Block 插入逻辑。",
         ["输入：create<OpTy>(location, args...) 的位置、操作数、类型等参数，以及 Builder 当前插入点。",
          "调用：OpTy::build(*this, state, args...) 填 OperationState；create(state) 调 insert(Operation::create(state))。",
          "输出 / 动作：新 Operation 被插入当前 Block 的操作列表，并通知 listener；模板入口返回引用同一 Operation 的 OpTy 包装。",
          "有插入点时才插入 Block；没有插入点时返回尚未挂接的 Operation。创建 addi 表示记录一条运算，不在此时执行整数加法。"],
         [("include/mlir/IR/Builders.h", 502, 11),
          ("lib/IR/Builders.cpp", 452, 14),
          ("lib/IR/Builders.cpp", 416, 9)],
         ["builder", "operation", "block", "opwrapper"]),
    node("fold-driver", "Canonicalizer 逐个尝试 fold", "Canonicalizer → Greedy worklist → op->fold", "公共简化调度",
         "Canonicalizer 把根操作和模式集交给 Greedy driver；driver 处理工作队列时，先尝试操作自己的 fold 钩子，再尝试匹配重写模式。",
         ["输入：根 Operation、FrozenRewritePatternSet、GreedyRewriteConfig；工作队列中的每项仍是公共 Operation*。",
          "调用：Canonicalizer::runOnOperation → applyPatternsGreedily；driver 对非 ConstantLike 操作调用 op->fold(foldResults)。",
          "输出 / 动作：Operation::fold 从 operands 提取已知常量；对 addi %x, %c0，传给钩子的常量信息是 [未知, IntegerAttr(0)]。",
          "这条 x + 0 简化使用 fold 路径，不需要先经过 PatternApplicator；fold 成功后由公共 Rewriter 应用结果。"],
         [("lib/Transforms/Canonicalizer.cpp", 62, 7),
          ("lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp", 487, 8),
          ("lib/IR/Operation.cpp", 660, 8)],
         ["pass", "rewrite", "fold-dispatch", "replace"]),
    node("fold-dispatch", "公共 Operation → 具体 fold 实现", "OperationName hook → Model<T> → T::fold", "注册钩子的实际分派",
         "Operation::fold 不按 arith、scf 等名字写条件分支；它通过 OperationName 的实现对象，将请求交给已注册的具体 Op。",
         ["输入：当前 Operation*、每个 operand 对应的常量 Attribute（未知处为空）、待填充的 OpFoldResult 列表。",
          "调用：Operation::fold → name.foldHook → getImpl()->foldHook；注册的 Model<ConcreteOp> 再调用 ConcreteOp::getFoldHookFn()。",
          "调用：单结果操作进入 foldSingleResultHook，构造 FoldAdaptor，然后 cast<ConcreteOpT>(op).fold(adaptor)。",
          "输出 / 动作：具体规则返回的 Value 或 Attribute 被追加到公共结果列表；AddIOp 在 x + 0 情形返回已有的 lhs Value。"],
         [("lib/IR/Operation.cpp", 632, 10),
          ("include/mlir/IR/OperationSupport.h", 259, 6),
          ("include/mlir/IR/OperationSupport.h", 529, 8),
          ("include/mlir/IR/OpDefinition.h", 1902, 23)],
         ["registration", "operation", "addi-fold", "fold-driver"]),
    node("addi-fold", "AddIOp：x + 0 返回 x", "arith::AddIOp::fold(FoldAdaptor)", "具体算术语义",
         "Arith 提供 x + 0 = x 这条操作特有的规则；规则只返回左操作数，如何修改使用链与删除操作交给共享 Rewriter。",
         ["输入：arith.addi %x, %c0 : i32；FoldAdaptor 的右侧常量 Attribute 为零。",
          "调用 / 判定：matchPattern(adaptor.getRhs(), m_Zero()) 成功后执行 return getLhs()。",
          "输出：OpFoldResult 中保存现有 SSA Value %x；不创建新的加法，也不复制 %x 或计算运行时输入。",
          "ArithOps.td 的 hasFolder = 1 声明该操作具有 fold 钩子；数学规则的实际代码在 ArithOps.cpp。"],
         [("lib/Dialect/Arith/IR/ArithOps.cpp", 302, 9),
          ("include/mlir/Dialect/Arith/IR/ArithOps.td", 253, 7)],
         ["arith", "fold-dispatch", "replace", "value"]),
    node("replace", "把使用者改接到 x，并删除 addi", "Greedy replaceOp → replaceAllOpUsesWith → eraseOp", "公共 IR 修改",
         "Greedy driver 接收具体 fold 返回的 Value，再由 Rewriter 统一更新所有使用者并移除旧操作。",
         ["输入：待替换的 addi Operation*，以及 fold 得到的替代 SSA Value %x。",
          "调用：driver 把 Value 放入 replacements，再调用 rewriter.replaceOp(op, replacements)。",
          "输出 / 动作：RewriterBase::replaceOp 先 replaceAllOpUsesWith，再 eraseOp；原来读取 %a 的下一条加法改为读取 %x。",
          "原 addi 已经消失，因此它不会再被降成 llvm.add；只有仍留在 IR 中的另一条 addi 才进入后续 Lowering。"],
         [("lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp", 516, 8),
          ("lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp", 553, 8),
          ("lib/IR/PatternMatch.cpp", 130, 13)],
         ["addi-fold", "value", "rewrite", "arith-lowering"]),
    node("arith-lowering", "剩余 arith.addi → MLIR llvm.add", "AddIOpLowering → Dialect Conversion → 公共创建 / 替换", "仍在 MLIR 内转换",
         "ArithToLLVM 提供 AddIOp 到 LLVM::AddOp 的具体映射；公共 conversion driver 按目标合法性运行规则，公共 helper 创建并替换操作。",
         ["输入：仍存在的 arith.addi %x, %y、LLVMConversionTarget、LLVMTypeConverter 和转换 patterns。",
          "调用：Pass 填充 ArithToLLVM patterns 后调用 applyPartialConversion；公共入口建立 OperationConverter 并运行 convertOperations。",
          "规则：AddIOpLowering 采用 VectorConvertToLLVMPattern<arith::AddIOp, LLVM::AddOp>，向共享 helper 传目标名、已转换 operands、属性与类型转换器。",
          "输出 / 动作：标量路径的 oneToOneRewrite 创建 llvm.add，再用它的结果替换旧 addi；产物仍是 MLIR Operation，不是 llvm::Instruction。",
          "完整函数示例还需要 FuncToLLVM 转换 func.func 与 func.return；ArithToLLVM 本身只提供这里的算术转换。"],
         [("lib/Conversion/ArithToLLVM/ArithToLLVM.cpp", 64, 3),
          ("lib/Conversion/ArithToLLVM/ArithToLLVM.cpp", 472, 15),
          ("lib/Transforms/Utils/DialectConversion.cpp", 3355, 6),
          ("lib/Conversion/LLVMCommon/Pattern.cpp", 345, 27)],
         ["conversion", "llvm", "op-create", "func-lowering", "llvm-export"]),
    node("func-lowering", "函数与 return 也必须转换", "FuncToLLVM：func.func / func.return → LLVM dialect", "补全可导出的函数",
         "算术变为 llvm.add 后，函数容器和返回操作仍须转换；FuncToLLVM 为它们提供独立的 conversion patterns。",
         ["输入：func.func 的函数签名、函数体，以及 func.return 的返回 SSA Value；示例是两个 i32 参数和一个 i32 返回值。",
          "调用：Pass 填充 FuncToLLVM patterns，再以 LLVMConversionTarget 调用 applyPartialConversion。",
          "输出 / 动作：转换 helper 创建 LLVM::LLVMFuncOp，将原函数体移入新函数并转换入口参数；本例保留 i32 参数表示。",
          "输出 / 动作：单返回值路径以 LLVM::ReturnOp 替换 func.return。结果是包含 llvm.add 和 llvm.return 的 llvm.func，仍属于 MLIR。"],
         [("lib/Conversion/FuncToLLVM/FuncToLLVM.cpp", 785, 6),
          ("lib/Conversion/FuncToLLVM/FuncToLLVM.cpp", 345, 8),
          ("lib/Conversion/FuncToLLVM/FuncToLLVM.cpp", 433, 7),
          ("lib/Conversion/FuncToLLVM/FuncToLLVM.cpp", 702, 6)],
         ["arith-lowering", "conversion", "llvm", "llvm-export"]),
    node("llvm-export", "MLIR llvm.add → LLVM IR add", "ModuleTranslation → dialect interface → CreateAdd", "跨到 LLVM IR 表示",
         "LLVM 导出器按操作所属 dialect 查翻译接口；LLVM dialect 的实现通过生成的转换代码调用 LLVM IRBuilder，创建真正的 LLVM IR。",
         ["输入：已满足导出要求的 MLIR module；处理单条 llvm.add 时传 Operation*、llvm::IRBuilderBase& 和 ModuleTranslation 映射状态。",
          "调用：ModuleTranslation::convertOperation 用 iface.getInterfaceFor(&op) 找所属 dialect 的接口，再调用 opIface->convertOperation。",
          "调用：LLVM dialect 注册 LLVMDialectLLVMIRTranslationInterface，其 convertOperation 转调 convertOperationImpl；其中包含 LLVMConversions.inc。",
          "输出 / 动作：LLVMOps.td 的 Add 实例把 builder.Create + instName 生成为 CreateAdd；转换代码读取 operand 的 LLVM 值映射，并记录新 LLVM 结果。最终得到 llvm::Module 中的 add，而非另一个 MLIR Op。"],
         [("lib/Target/LLVMIR/ModuleTranslation.cpp", 931, 17),
          ("lib/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.cpp", 475, 13),
          ("include/mlir/Dialect/LLVMIR/LLVMOps.td", 74, 4),
          ("include/mlir/Dialect/LLVMIR/LLVMOps.td", 149, 6)],
         ["translation", "llvm", "arith-lowering", "interfaces", "llvm-values"]),
    node("llvm-values", "导出时怎样连接两套 Value", "lookupValue / mapValue：MLIR SSA → LLVM Value*", "LLVM 导出的数据流",
         "导出器创建 LLVM 对象时维护对应表；MLIR Value 仍是原 IR 的句柄，LLVM Value* 是新表示中的对象。",
         ["输入：某条 MLIR 操作及其 operands，已创建的 LLVM 值，以及 ModuleTranslation 的 valueMapping。",
          "分派依据：getInterfaceFor(Operation*) 实际调用 getInterfaceFor(op->getDialect())；选择的是所属方言的翻译接口。",
          "调用：TableGen 转换生成器把 operand 引用生成为 moduleTranslation.lookupValue(op.get...())，把 result 写入生成为 moduleTranslation.mapValue(op.get...())。",
          "输出 / 动作：生成代码用映射得到的 LLVM operands 调 IRBuilder，再把新 LLVM 结果记入表；后续操作查同一张表以连接 SSA 依赖。"],
         [("lib/IR/Dialect.cpp", 146, 6),
          ("include/mlir/Target/LLVMIR/ModuleTranslation.h", 77, 15),
          ("tools/mlir-tblgen/LLVMIRConversionGen.cpp", 138, 12),
          ("lib/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.cpp", 228, 9)],
         ["llvm-export", "translation", "value", "tablegen"]),
]


def build(root: Path, output: Path, snapshot_date: str) -> None:
    root = root.resolve()
    version_text = (root.parent / "cmake/Modules/LLVMVersion.cmake").read_text(encoding="utf-8")
    version = ".".join(re.search(rf"set\(LLVM_VERSION_{part}\s+(\d+)\)", version_text)[1]
                       for part in ("MAJOR", "MINOR", "PATCH"))
    if version != "20.1.8":
        raise ValueError(f"Source map ranges target 20.1.8, got {version}; review ranges first")
    cache = {}
    nodes = []
    for spec in NODES:
        item = dict(spec)
        item["sources"] = []
        for path, start, count in spec["sources"]:
            if path not in cache:
                raw = (root / path).read_bytes()
                cache[path] = (raw.decode("utf-8").splitlines(), hashlib.sha256(raw).hexdigest())
            lines, digest = cache[path]
            if not 1 <= start <= len(lines):
                raise ValueError(f"Invalid excerpt {path}:{start}")
            end = min(len(lines), start + count - 1)
            item["sources"].append(dict(path=path, start=start, end=end,
                                        sha256=digest, lines=lines[start - 1:end],
                                        uri=(root / path).as_uri()))
        nodes.append(item)
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes), "Duplicate node id"
    for n in nodes:
        assert set(n["related"]) <= ids, f"Missing related node in {n['id']}"
    topology = json.loads((PROJECT / "tools/docs/mlir_source_topology.json").read_text(encoding="utf-8"))
    views = topology["views"]
    if isinstance(views, dict):
        views = list(views.values())
    if not isinstance(views, list):
        raise ValueError("topology.views must be a list or an object of views")
    for index, view in enumerate(views):
        view_name = view.get("id", str(index))
        view_nodes = view["nodes"]
        edge_ids = [edge["id"] for edge in view["edges"]]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError(f"Duplicate topology edge id in view {view_name}")
        view_ids = {item["id"] for item in view_nodes}
        if len(view_ids) != len(view_nodes):
            raise ValueError(f"Duplicate topology node id in view {view_name}")
        for item in view_nodes:
            if item["ref"] not in ids:
                raise ValueError(f"Unknown source ref {item['ref']!r} in view {view_name}, node {item['id']}")
        for edge in view["edges"]:
            if edge["from"] not in view_ids or edge["to"] not in view_ids:
                raise ValueError(f"Unknown topology endpoint in view {view_name}: {edge['from']} -> {edge['to']}")
            refs = edge["refs"]
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"Edge refs must be a nonempty list in view {view_name}: {edge['from']} -> {edge['to']}")
            missing = set(refs) - ids
            if missing:
                raise ValueError(f"Unknown edge source refs in view {view_name}: {sorted(missing)}")
    data = dict(root=root.as_posix(), version=version, date=snapshot_date,
                nodes=nodes, topology=topology)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = (PROJECT / "tools/templates/mlir-source-architecture.html").read_text(encoding="utf-8")
    for source_ref in re.findall(r'data-source="([^"]+)"', template):
        if source_ref not in ids:
            raise ValueError(f"Unknown template source ref {source_ref}")
    for group in re.findall(r'data-nodes="([^"]+)"', template):
        assert set(group.split(",")) <= ids, f"Unknown node group {group}"
    assert template.count("__MLIR_DATA__") == 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__MLIR_DATA__", payload), encoding="utf-8")
    print(json.dumps(dict(output=str(output), version=version, nodes=len(nodes),
                          source_files=len(cache), bytes=output.stat().st_size), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir-root", type=Path, default=PROJECT.parent / "llvm-project/mlir")
    parser.add_argument("--output", type=Path, default=PROJECT / "docs/mlir-source-architecture.html")
    parser.add_argument("--date", default=date.today().isoformat(), help="Source snapshot date")
    args = parser.parse_args()
    build(args.mlir_root, args.output, args.date)
