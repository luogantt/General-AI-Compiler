#!/usr/bin/env python3
"""Build the offline G0.5 compute/latency network viewer from recorded evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "docs/validation/ascend/galaxeavla-g05-full-910b2-2026-11-16"
SOURCE = "https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/"
SOURCES = {
    "vision": "src/g05/models/g05/qwen35/vision.py",
    "vlm": "src/g05/models/g05/qwen35/mixture_qwen35.py",
    "action_expert": "src/g05/models/g05/qwen35/modules.py",
    "proprio": "src/g05/models/g05/helpers/proprio_helper.py",
    "codec": "src/g05/tokenizer/models/actioncodec2_v2/modeling_actioncodec2v2.py",
    "other": "src/g05/models/g05/g05_policy.py",
}
LABELS = {"vision": "视觉编码器", "vlm": "Qwen3.5 VLM", "action_expert": "动作专家", "proprio": "本体状态 MLP", "codec": "ActionCodec", "other": "其他投影"}
STAGES = {
    "prefill": ("视觉与 VLM 前缀", "prefill_ms"),
    "cot": ("CoT 自回归", "cot_generate_text_ms"),
    "fm": ("连续动作 · FM", "fm_action_ms"),
    "action_ar": ("离散动作 · AR", "ar_decode_ms"),
    "codec": ("ActionCodec 解码", "decode_ar_total_ms"),
}


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")


def sum_flops(records: list[dict]) -> int:
    return sum(int(r["counted_flops"]) for r in records)


def make_data(result: dict, workload: dict, baseline: dict) -> dict:
    summary = result["measurement"]["summary"]
    for section, field in (("source", "revision"), ("checkpoint", "sha256"), ("input", "sha256")):
        if result[section][field] != workload[section][field]:
            raise ValueError(f"Timing/workload evidence mismatch: {section}.{field}")
    if result["configuration_contract"] != workload["configuration_contract"]:
        raise ValueError("Timing/workload configuration mismatch")
    if not result["output_contract"]["all_requests_actions_finite"]:
        raise ValueError("Benchmark outputs must have passed finite-value checks")
    if workload["counted_flops"] <= 0:
        raise ValueError("Missing counted workload")
    graphs: dict[str, dict] = {}

    def timing(key: str) -> dict:
        return {**summary[key], "kind": "measured", "label": "30 次实测"}

    def add_graph(key: str, title: str, desc: str, nodes: list, edges: list | None = None):
        graphs[key] = {"title": title, "desc": desc, "nodes": nodes, "edges": edges or []}

    def record_node(key: str, label: str, records: list[dict], col: int, row: int, sub: str | None = None, component: str = "other") -> dict:
        shapes = sorted({str(r.get("input_shape")) + " → " + str(r.get("output_shape")) for r in records})
        attrs = {
            "统计口径": "本次完整请求中实际执行的 Linear / 普通 Conv / AdaLN 条件投影；1 MAC = 2 FLOP。",
            "主算子调用次数": sum(r["calls"] for r in records),
            "形状记录": "\n".join(shapes[:8]),
            "独立计时": "未测量。该模块耗时包含在所属阶段，不能按 FLOP 比例分摊。",
        }
        if len(shapes) > 8:
            attrs["其余形状"] = f"另有 {len(shapes) - 8} 种，见下载的算力证据 JSON。"
        if len(records) <= 4:
            attrs["模块路径"] = "\n".join(r["module"] for r in records)
        return {"id": key, "label": label, "op": " + ".join(sorted({r["operator"] for r in records})), "shape": shapes[0] if len(shapes) == 1 else f"{len(shapes)} 种输入 / 输出形状", "col": col, "row": row, "flops": sum_flops(records), "timing": None, "sub": sub, "attrs": attrs, "source": SOURCE + SOURCES.get(component, SOURCES["other"])}

    def leaf_graph(key: str, title: str, records: list[dict], component: str):
        groups: dict[str, list] = {}
        for record in records:
            groups.setdefault(record["module"], []).append(record)
        nodes = []
        for index, (module, rows) in enumerate(sorted(groups.items())):
            label = ".".join(module.split(".")[-2:])
            item = record_node(f"op-{index}", label, rows, index % 3, index // 3, component=component)
            item["attrs"]["算式"] = "Linear：2 × output.numel × in_features；普通 Conv：2 × output.numel × (in_channels/groups) × kernel_volume；不计偏置加法。"
            nodes.append(item)
        add_graph(key, title, "按模块展示本次请求的累计主算子工作量；多次调用已累计。节点之间不虚构依赖边。", nodes)

    def component_graph(key: str, title: str, records: list[dict], component: str):
        layers: dict[int, list] = {}
        rest: list = []
        for record in records:
            found = re.search(r"\.(?:layers|blocks)\.(\d+)\.", record["module"])
            if found:
                layers.setdefault(int(found[1]), []).append(record)
            else:
                rest.append(record)
        if len(layers) <= 4:
            leaf_graph(key, title, records, component)
            return
        nodes = []
        for index, start in enumerate(range(0, max(layers) + 1, 4)):
            selected = {layer: rows for layer, rows in layers.items() if start <= layer < start + 4}
            rows = [r for part in selected.values() for r in part]
            sub = f"{key}-group-{start}"
            children = []
            for child_index, (layer, layer_rows) in enumerate(sorted(selected.items())):
                child_key = f"{sub}-layer-{layer}"
                layer_type = "GDN" if component == "vlm" and layer % 4 != 3 else "Attention"
                children.append(record_node(f"layer-{layer}", f"第 {layer} 层 · {layer_type}", layer_rows, child_index % 2, child_index // 2, child_key, component))
                leaf_graph(child_key, f"{title} / 第 {layer} 层", layer_rows, component)
            add_graph(sub, f"{title} / 层 {start}—{start + 3}", "层编号从 0 开始。计算量包含该请求中的全部调用，不代表逐层实测耗时。", children)
            nodes.append(record_node(f"group-{start}", f"第 {start}—{start + 3} 层", rows, index % 3, index // 3, sub, component))
        if rest:
            sub = key + "-projections"
            leaf_graph(sub, title + " / 入口、出口与投影", rest, component)
            index = len(nodes)
            nodes.append(record_node("projections", "入口、出口与投影", rest, index % 3, index // 3, sub, component))
        add_graph(key, title, "四层一组展开。主算子统计不包含 attention QK/PV、GDN 状态更新、激活等功能算子。", nodes)

    operators = workload["operators"]
    for stage, (title, _) in STAGES.items():
        records = [r for r in operators if r["stage"] == stage]
        components: dict[str, list] = {}
        for record in records:
            components.setdefault(record["component"], []).append(record)
        nodes = []
        for index, (component, rows) in enumerate(components.items()):
            key = f"{stage}-{slug(component)}"
            label = "动作占位编码 · RVQ" if stage == "prefill" and component == "codec" else LABELS.get(component, component)
            component_graph(key, f"{title} / {label}", rows, component)
            nodes.append(record_node(f"component-{component}", label, rows, index % 3, index // 3, key, component))
        add_graph(stage, title, "展开查看该阶段实际执行的主算子规模。总阶段有独立实测时间，内部模块仅记录工作量。", nodes)

    prefix = workload.get("prefill", {}).get("prefix_tokens")
    stage_flops = {stage: int(workload["stages"].get(stage, {}).get("counted_flops", 0)) for stage in STAGES}
    def phase(stage: str, col: int, row: int, shape: str, attrs: dict) -> dict:
        title, field = STAGES[stage]
        labels = {"prefill": "02 · ViT + Hybrid VLM", "cot": "03 · VLM decode", "fm": "04 · Action Expert ×10", "action_ar": "05 · LM head", "codec": "06 · RVQ + decoder"}
        return {"id": stage, "label": title, "op": labels[stage], "shape": shape, "col": col, "row": row, "sub": stage, "flops": stage_flops[stage], "timing": timing(field), "attrs": attrs}
    nodes = [
        {"id": "input", "label": "三相机 + 指令 + 状态", "op": "R1 Lite · 合成输入", "shape": "3 × 720 × 1280 · 状态 14D", "col": 0, "row": 0, "flops": None, "timing": None, "attrs": {"输入": "三路 uint8 图像；模型处理为 256×256。batch=1，固定任务指令。", "状态": "14 维原始状态按机器人布局合并并填充为 27 维。", "准确率范围": "合成输入吞吐测试；未进行真实机器人任务准确率或 CUDA 全模型数值对照。"}},
        {"id": "preprocess", "label": "预处理与数据传输", "op": "CPU → NPU", "shape": "3 × 256 × 256 · BF16 autocast", "col": 1, "row": 0, "flops": None, "timing": timing("cpu_preprocess_total_ms"), "attrs": {"卡片耗时范围": "CPU 原始观测整理与预处理；H2D 单独列出，不计入此卡片。", "H2D p50": f"{summary['h2d_ms']['p50']:.3f} ms", "线程": "OMP_NUM_THREADS=1，MKL_NUM_THREADS=1；PyTorch intra-op=1、inter-op=256。", "未统计": "CPU 图像处理、tokenization 的 FLOPs 未计入主算子总量。"}},
        phase("prefill", 2, 0, f"视觉 192 · 前缀 {prefix} token", {"执行": "原流程中的动作占位编码（Codec Encoder / RVQ）→ 三相机 Vision → 状态和文本融合 → 24 层 VLM。", "VLM": "18 层 GatedDeltaNet + 6 层全注意力。", "计时": "360.59 ms 是整个 prefill 的聚合时间；没有独立的 Vision / VLM NPU 子模块测量。"}),
        phase("cot", 3, 0, "20 raw token · 更新缓存", {"生成": "本次 CoT 调用返回 20 个原始 ID（含停止 token）；该序列可能已包含动作 token。", "预算": "AR max_new_tokens=300；这不是生成 300 token 的最坏情况测速。", "重复计算": "每次增量 VLM 和词表头调用均计入主算子计算量。"}),
        phase("fm", 3, 1, "10 次 Euler · [1,32,27]", {"动作专家": "24 层，32 个动作时刻，10 次迭代；每步复用 VLM 前缀缓存。", "每步摊算": f"{summary['fm_action_ms']['p50']/10:.3f} ms/步（按阶段 p50 除以 10，非逐步独立实测）。", "精度": "BF16 autocast，保留 FP32 精度区域；没有 torch.compile。"}),
        phase("action_ar", 2, 1, "1 raw token · 提前终止", {"停止规则": "本次只返回 1 个停止 ID（EOS 或 |，未保存具体 ID）。首 token 由已有 hidden 产生。", "不能混算": "18 个解析动作 token 来自完整序列，不能除以这一段的 1.83 ms 作为生成吞吐。"}),
        phase("codec", 1, 1, "18 个动作 token → 动作块", {"解析": "将输入、CoT 和动作序列合并解析，ActionCodec 解码为 32 步、27 维动作块。", "统计边界": "本图只统计被 hook 捕获的普通 Linear/Conv；ConvTranspose、DCT/IDCT、规则解码等未覆盖。"}),
        {"id": "output", "label": "双臂与夹爪动作", "op": "D2H + 后处理", "shape": "32 步 · 服务每轮使用 16 步", "col": 0, "row": 1, "flops": None, "timing": timing("postprocess_ms"), "attrs": {"卡片耗时": "仅 CPU 后处理；D2H 单独列出。", "D2H p50": f"{summary['d2h_ms']['p50']:.3f} ms", "输出形状": "左右臂各 [1,32,6]，左右夹爪各 [1,32,1]；连续与离散核心输出均 [1,32,27]。", "验证": "两轮共 72 次请求（冷启动、预热、正式测量）动作值有限且形状符合预期。", "完整请求 p50": f"{summary['request_total_ms']['p50']:.3f} ms，含预处理、传输与后处理。"}},
    ]
    order = ["input", "preprocess", "prefill", "cot", "fm", "action_ar", "codec", "output"]
    add_graph("main", "G0.5 · 910B2 整模计算与时延", "固定合成 R1 Lite 输入；完整双动作头路径。绿色时间为正式实测，计算量来自另一次不用于计时的算子诊断。", nodes, [{"from": a, "to": b} for a, b in zip(order, order[1:])])
    if sum(stage_flops.values()) != int(workload["counted_flops"]):
        raise ValueError("Unscoped operators must be reviewed before presenting the stage total")
    return {
        "title": "G0.5 · 910B2 算力与推理时间",
        "profile": {
            "request": summary["request_total_ms"], "model": summary["model_predict_action_ms"],
            "flops": workload["counted_flops"], "parameters": workload["parameters"]["total_numel"],
            "allocatedGiB": result["measurement"]["memory"]["max_memory_allocated_bytes"] / 2**30,
            "reservedGiB": result["measurement"]["memory"]["max_memory_reserved_bytes"] / 2**30,
            "repeats": result["measurement"]["iterations"], "warmup": result["warmup"]["iterations"],
            "prefixTokens": prefix, "samples": [s["request_total_ms"] for s in result["measurement"]["samples"]],
            "baselineP50": baseline["measurement"]["summary"]["request_total_ms"]["p50"],
            "summaryNotes": [
                "910B2 64 GiB；CANN 9.0.0；Torch 2.7.1 / TorchNPU 2.7.1.post4；eager + BF16 autocast，权重 dtype 见算力证据。",
                f"含独立 ActionCodec 后端，本次去重参数 {workload['parameters']['total_numel']:,} 个，参数张量约 {workload['parameters']['total_bytes']/2**30:.3f} GiB；诊断记录全部为 FP32 常驻，BF16 autocast 不等于把常驻权重转成 BF16。",
                "已统计主算子 = Linear + 普通 Conv + AdaLN 条件投影，1 MAC = 2 FLOP。每次真实调用均累计；不代表完整模型 FLOPs 或芯片指令数。",
                "未覆盖 attention QK/PV、GDN 状态运算和部分 functional conv、ConvTranspose、norm、激活、RoPE、DCT/IDCT、CPU 处理等。",
                "诊断运行仅统计形状和次数，不作为延迟来源；p50/p95 来自独立的 5 次预热 + 30 次正式测量。",
                "没有对各子模块独立计时；不能按 FLOPs 比例分摊阶段时间。各阶段 p50 也不能相加当作请求 p50。",
                "显存为 PyTorch allocator 统计，未包含所有驱动分配；不能把 15.93 GiB 当作保证可运行的最小卡容量。",
                "合成输入；无真实任务准确率、CUDA 整模数值对照或硬件利用率结论。"
            ],
        },
        "graphs": graphs,
        "evidence": {"benchmark": result, "workload": workload},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_DIR / "result.json")
    parser.add_argument("--baseline", type=Path, default=RESULT_DIR / "result-default-threads.json")
    parser.add_argument("--workload", type=Path, default=RESULT_DIR / "workload.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = [json.loads(p.read_text(encoding="utf-8")) for p in (args.result, args.workload, args.baseline)]
    data = make_data(*documents)
    data["evidence"]["file_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.result, args.workload, args.baseline)}
    template = Path(__file__).with_name("compute_view.template.html").read_text(encoding="utf-8")
    if template.count("__G05_VIEW_DATA__") != 1:
        raise ValueError("Expected exactly one data placeholder")
    html = template.replace("__G05_VIEW_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(f"Wrote {len(graphs := data['graphs'])} subgraphs to {args.output}")


if __name__ == "__main__":
    main()
