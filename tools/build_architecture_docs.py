"""Generate the V8 code-linked architecture diagram from canonical sources.

Run from the repository: ``python tools/build_architecture_docs.py``.
``--check`` verifies the checked-in page without writing. Snippets are extracted
by Python symbol or an explicit unique anchor; line numbers are never copied
manually. The GitHub links pin the latest commit that changed compiler source.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gac.compiler.pipeline import COMPILER_STAGES


COMPILER = "src/gac/targets/vortex/compiler.py"
HOST = "src/gac/targets/vortex/host_runtime.py"
MEMORY = "src/gac/compiler/memory.py"
OUTPUT = ROOT / "docs/vortex-matmul-end-to-end-v8.html"
TEMPLATE = ROOT / "tools/templates/compiler-architecture-v8.html"
PLACEHOLDER = "__ARCHITECTURE_CATALOG_JSON__"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


class SourceCatalog:
    """Extract inspectable excerpts and verify their pinned source revision."""

    def __init__(self, revision: str):
        self.revision = revision
        self.sources: dict[str, dict] = {}

    def add(self, key: str, path: str, symbol: str, role: str, *,
            max_lines: int = 38, anchor: str | None = None, before: int = 0) -> str:
        text = (ROOT / path).read_text(encoding="utf-8")
        committed = subprocess.check_output(["git", "show", f"{self.revision}:{path}"], cwd=ROOT)
        if committed.decode("utf-8").replace("\r\n", "\n") != text:
            raise ValueError(f"{path} has uncommitted source changes; commit them before pinning the diagram")
        lines = text.splitlines()
        if anchor is None:
            owner = ast.parse(text)
            for name in symbol.split("."):
                owner = next(node for node in owner.body
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                             and node.name == name)
            start, end = owner.lineno, min(owner.end_lineno, owner.lineno + max_lines - 1)
        else:
            matches = [index for index, line in enumerate(lines) if anchor in line]
            if len(matches) != 1:
                raise ValueError(f"{path}: expected a unique anchor {anchor!r}, found {len(matches)}")
            start = max(1, matches[0] + 1 - before)
            end = min(len(lines), start + max_lines - 1)
        self.sources[key] = {
            "file": path, "symbol": symbol, "role": role, "start": start, "end": end,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "code": "\n".join(lines[start - 1:end]),
        }
        return key


def build_catalog() -> dict:
    revision = git("log", "-1", "--format=%H", "--", "src/gac")
    catalog = SourceCatalog(revision)
    add = catalog.add
    stage_symbols = (
        "stage_model_import", "stage_ir_conversion", "stage_graph_optimization", "stage_legalization",
        "stage_subgraph_partition", "stage_template_selection", "stage_tiling", "stage_schedule",
        "stage_codegen", "stage_runtime",
    )
    for stage, symbol in zip(COMPILER_STAGES, stage_symbols, strict=True):
        add(stage.name, COMPILER, symbol, f"第 {stage.number} 阶段的实际执行函数")
    add("pipeline", "src/gac/compiler/pipeline.py", "run_pipeline", "通用阶段顺序与失败阻断")
    add("entry", COMPILER, "compile_vortex_graph", "十阶段函数注册与编译入口")
    add("frontend", "src/gac/frontends/model_frontend.py", "load_model", "通用模型文件 → Canonical Graph", max_lines=50)
    add("model-api", "src/gac/models/api.py", "load_model_graph", "模型适配器 → 同一种 Graph")
    add("adapter-protocol", "src/gac/models/api.py", "ModelAdapter", "新增模型需要满足的 Graph 接口")
    add("adapter-register", "src/gac/models/api.py", "register_model_adapter", "注册实际模型适配器；不初始化设备")
    add("bevformer", "src/gac/models/bevformer/adapter.py", "BEVFormerStepAdapter.build", "完整 streaming-step 构图")
    add("uniad", "src/gac/models/uniad/adapter.py", "UniADFFNAdapter.build", "当前 UniAD 单层 FFN 构图")
    add("shared-ffn", "src/gac/models/common/ffn.py", "build_transformer_ffn_graph", "模型无关的共享 FFN")
    add("shape", "src/gac/compiler/passes.py", "ShapeInferencePass.run", "推导并校验 TensorSpec")
    add("constant-folding", "src/gac/compiler/passes.py", "ConstantFoldingPass.run", "图优化内部实现：常量折叠")
    add("dead-code", "src/gac/compiler/passes.py", "DeadCodeEliminationPass.run", "图优化内部实现：死代码消除")
    add("static", COMPILER, "_validate_static_specs", "静态形状、DType 与 RV32 范围")
    add("boolean-check", COMPILER, "_validate_boolean_operations", "合法化内部实现：布尔算子约束")
    add("templates", COMPILER, "_template", "按规范算子选择软件计算模板")
    add("target-matmul", "src/gac/targets/vortex/scalar_ops.py", "_matmul", "目标 GEMM 计算模板：load / K 轴累加")
    add("target-layernorm", "src/gac/targets/vortex/scalar_ops.py", "_layer_norm", "目标 LayerNorm 归约模板")
    add("target-softmax", "src/gac/targets/vortex/scalar_ops.py", "_softmax", "目标 Softmax 归约模板")
    add("scalar-arithmetic", "src/gac/targets/vortex/scalar_ir.py", "ScalarBuilder.binary", "实际生成 arith.addf / mulf 等 MLIR 运算")
    add("scalar-loop", "src/gac/targets/vortex/scalar_ir.py", "ScalarBuilder.loop", "Tiling 内部实现：生成 SCF 循环与归约累加值")
    add("memory-plan", MEMORY, "plan_memory", "Buffer 生命周期与布局规划合同", max_lines=24)
    add("memory-lifetimes", MEMORY, "plan_memory · definition / uses", "生产时刻、最后消费与输出存活期",
        anchor="definition: dict[str, int]", max_lines=22)
    add("memory-reuse", MEMORY, "_first_fit", "互不冲突的活跃区间复用 arena 地址")
    add("host-arena", HOST, "emit_graph_host · arena / offsets", "编译地址计划 → Host Buffer 表",
        anchor="buffers: list[tuple[str, int, str]]", max_lines=22)
    add("device-load", "src/gac/targets/vortex/scalar_ir.py", "ScalarBuilder.load", "生成普通设备内存读取")
    add("device-store", "src/gac/targets/vortex/scalar_ir.py", "ScalarBuilder.store", "生成普通设备内存写回")
    add("kernel-abi", COMPILER, "_entry", "uniform 参数解包、CTA CSR 和线程索引")
    add("kernel-body", COMPILER, "_kernel_body", "Codegen 内部实现：生成算子体与 store")
    add("host-generation", HOST, "emit_graph_host", "Runtime 阶段内部实现：生成 Host 调度程序", max_lines=38)
    add("host-allocate", HOST, "Device::initialize（生成的 Host C++）", "一个设备、一个队列与真实设备 Buffer",
        anchor="void initialize(const std::string& kernel)", max_lines=21)
    add("host-upload", HOST, "Device::upload（生成的 Host C++）", "Host 数据上传到设备全局 Buffer",
        anchor="CHECK(vx_enqueue_write(queue", before=4, max_lines=10)
    add("host-dispatch", HOST, "run · dispatch（生成的 Host C++）", "前驱事件、地址参数与 Kernel 启动",
        anchor="vx_event_h prior = nullptr", max_lines=20)
    add("host-readback", HOST, "run · readback（生成的 Host C++）", "等待设备完成，再读回最终输出",
        anchor="const auto computed = Clock::now();", max_lines=14)
    add("makefile", HOST, "emit_graph_makefile", "生成完整 MLIR → LLVM IR → VOLT 构建规则")
    add("translate", "src/gac/targets/vortex/ffn.py", "_direct_makefile · kernel.ll", "实际 mlir-translate 命令",
        anchor="kernel.opt.mlir: kernel.mlir |", max_lines=8)
    add("ffn-buffer-plan", "src/gac/targets/vortex/buffered_mlir.py", "VortexBufferPlan.to_dict",
        "单独的固定 FFN 优化路径：LMEM 两 Stage / DXA")
    add("ffn-double-buffer", "src/gac/targets/vortex/buffered_mlir.py", "_matmul_kernel",
        "单独 FFN 的 LMEM Tile 模板；不是当前全图默认实现", max_lines=32)

    # These descriptions explain contracts; actual order/artifact names always
    # come from the executable catalog used by the Vortex compiler itself.
    descriptions = (
        ("模型导入", "模型文件 / 已验证权重 + 模型合同",
         "通用前端或模型适配器先构图；阶段入口检查 Graph 并克隆，避免修改调用方对象。",
         "Graph：输入/输出、常量、算子与元数据",
         "读取文件在前端；本阶段接收已导入 Graph。导入尚未分配设备 Buffer。", ["frontend", "model-api", "bevformer", "uniad", "entry", "pipeline"]),
        ("IR 转换与类型", "导入得到的 Canonical Graph",
         "推导每个值的形状与 DType，检查算子集合、静态范围和布尔运算限制。",
         "含完整 TensorSpec 的 Graph",
         "这里的 IR 是 Python Graph；可执行 MLIR 在第 9 阶段生成。", ["shape"]),
        ("图优化 Pass", "已推导类型的 Graph",
         "执行常量折叠、死代码消除，然后重新推导类型；保存 Pass 报告。",
         "优化后的 Graph + Pass 报告",
         "当前全图路径不宣称实现任意融合或全自动硬件调优。", ["constant-folding", "dead-code", "shape"]),
        ("算子合法化", "优化后的图与 Vortex 目标约束",
         "复核图、存储/索引范围和输出数，拒绝无法由当前目标表达的节点。",
         "可继续 Lowering 的静态图",
         "Lowering 分阶段进行；真正生成目标指令语义在第 9 阶段。", ["static", "boolean-check"]),
        ("子图切分与 DAG", "合法算子与张量生产/消费关系",
         "当前基线每个节点形成一个可执行区域，计算区域间数据依赖。",
         "KernelPartition 列表和依赖 DAG",
         "这些区域全部分配到一个 Vortex；不是自动把模型分散到多个芯片。", []),
        ("计算模板选择", "每个区域的规范算子",
         "选择逐元素、布局、GEMM、卷积、归约或采样的软件模板。",
         "每个 Kernel 的模板 ID",
         "模型公共 FFN 是构图复用；这里的目标计算模板负责循环和访存。当前全图没有自动选入旧 FFN 的 LMEM 双缓冲模板。", ["templates", "target-matmul", "target-layernorm", "target-softmax"]),
        ("Tiling 与线程工作量", "选定模板和输出规模",
         "当前每个 SIMT 线程负责一个输出，归约轴由线程内串行循环完成。",
         "output_elements_per_thread=1、reduction=serial、LMEM=0",
         "全图标量基线没有 BM/BN/BK 调优。固定 FFN 的分块优化是另一条路径。", ["kernel-abi", "target-matmul", "scalar-loop"]),
        ("Schedule 与 Buffer", "区域依赖、Tile 计划和张量活跃区间",
         "规划全局 arena 地址复用，构造严格有序的 Dispatch；每次等待前一次完成。",
         "MemoryPlan + 带事件顺序的 Dispatch 列表",
         "Buffer 复用依赖这个顺序；增加并行执行或融合必须重新规划生命周期。", ["memory-lifetimes", "memory-reuse", "host-arena"]),
        ("Codegen 与 Lowering", "调度后的模板、地址 ABI 与线程索引",
         "生成 SCF/Arith/Math/LLVM Dialect；mlir-opt Lowering、mlir-translate、VOLT 编译与链接。",
         "kernel.mlir → LLVM IR → ELF → kernel.vxbin",
         "编译 API 生成源码/计划；最终机器码构建由生成的 Makefile 执行，C++ 只承担 Host 调度。", ["kernel-body", "scalar-arithmetic", "kernel-abi", "makefile", "translate"]),
        ("Runtime 与设备执行", "Dispatch、Buffer 计划与已构建 vxbin",
         "本阶段生成 Host 程序；运行该程序时分配真实设备内存、上传、按事件启动并读回。",
         "设备输出与执行状态",
         "编译阶段生成 Runtime 源码不等于已经运行设备；模拟器时间也不等于芯片延时。", ["host-generation", "host-allocate", "host-upload", "host-dispatch", "host-readback"]),
    )
    stages = []
    for stage, description in zip(COMPILER_STAGES, descriptions, strict=True):
        title, input_, action, output, boundary, extra_sources = description
        stages.append({"number": stage.number, "id": stage.name, "title": title,
                       "input": input_, "action": action, "output": output, "boundary": boundary,
                       "artifact": stage.artifact, "sources": [stage.name, *extra_sources]})
    return {
        "meta": {"commit": revision, "branch": "lg/bevformer-vortex-full-ops",
                 "date": git("show", "-s", "--format=%cs", revision)},
        "stages": stages, "sources": catalog.sources,
        "layers": [
            {"path": path, "title": title, "role": role, "depends": depends}
            for path, title, role, depends in (
                ("core/", "公共语义契约", "Graph、Node、TensorSpec、算子语义与异常", []),
                ("compiler/", "通用编译分析", "阶段顺序、Shape/Pass、分区、Buffer 生命周期", ["core/"]),
                ("frontends/", "通用模型导入", "JSON / ONNX / torch.export → Graph", ["core/", "compiler/"]),
                ("models/", "模型适配", "BEVFormer、UniAD 合同；common/ 共享构图", ["core/", "compiler/", "artifacts/"]),
                ("targets/", "目标实现", "Vortex / CUDA 模板、合法化、Codegen 与 Host 生成", ["core/", "compiler/", "toolchains/"]),
                ("execution/", "执行器", "Python 设备 Session、Dispatch 与性能采集", ["core/", "compiler/", "targets/"]),
                ("validation/", "独立数值验证", "NumPy 参考计算；不加载设备执行器", ["core/"]),
                ("tools/ · workflows/", "上层编排", "选择模型、目标、数据和执行方式", ["models/", "compiler/", "targets/", "execution/"]),
            )
        ],
        "models": [
            {"name": "bevformer-step", "scope": "完整冷/热帧 streaming-step，到原始检测输出；Top-K/框解码在 Host。", "sources": ["bevformer", "model-api", "shared-ffn"]},
            {"name": "uniad-ffn", "scope": "当前仅 Stage-2 BEV Encoder 的一层 FFN；不是完整 UniAD 网络。", "sources": ["uniad", "shared-ffn", "model-api"]},
            {"name": "后续 VLA / 其他模型（尚未接入）", "scope": "已有 ModelAdapter / Graph 扩展边界；当前未注册 VLA 适配器，也未验证其算子覆盖。", "sources": ["adapter-protocol", "adapter-register"]},
        ],
        "buffer_sources": ["memory-plan", "memory-lifetimes", "memory-reuse", "host-arena", "host-allocate",
                           "device-load", "device-store", "host-dispatch", "ffn-buffer-plan", "ffn-double-buffer"],
        "backend_sources": ["codegen", "kernel-abi", "makefile", "translate", "host-allocate", "host-upload",
                            "host-dispatch", "host-readback"],
    }


def render() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError("architecture template needs exactly one catalog placeholder")
    data = json.dumps(build_catalog(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return template.replace(PLACEHOLDER, data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated V8 page is stale")
    args = parser.parse_args()
    result = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != result:
            print("V8 architecture is stale; run python tools/build_architecture_docs.py", file=sys.stderr)
            return 1
    else:
        OUTPUT.write_text(result, encoding="utf-8", newline="\n")
    print(f"{'Verified' if args.check else 'Generated'} {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
