#!/usr/bin/env python3
"""Render profile_engine.py outputs as a Chinese, offline report (CPU/stdlib only).

Example:
  python tools/tensorrt/render_profile_report.py --root docs/validation/tensorrt/\
bevformer-base-2026-10-17/profiling

The root contains mixed-layers/ and/or fp32-layers/ (other *-layers directories
are also accepted). Source reports and raw profiler callbacks are never changed.
Only overview.json, layers-ranked.csv, operators.csv, index.html and README.md
are generated. Optional Nsight CSV files are read, never fabricated.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import sys
from urllib.parse import quote


SCHEMA_VERSION = 1
SCOPE = (
    "本次为无历史帧、合成投影条件下的静态引擎 profiling；"
    "不能作为完整真实时序性能或 nuScenes mAP 验收。"
)
DENOMINATOR = (
    "层占比 = 该回调 mean_ms / 全部 IProfiler 回调 mean_ms 之和；"
    "不是 baseline 全图耗时占比，也不是 ONNX 原始节点占比。"
)
FUSION_NOTE = (
    "每个 ordinal 回调仅计一次。分类依据唯一精确匹配的 inspector LayerType / PluginType；"
    "名称、Metadata 中的原始节点只供追溯，不用于推断或拆分耗时。"
    "TrainStation、kgen 和已融合执行层均保留原始回调整体；不将其内部节点额外加入统计。"
    "不据邻近层推断 TrainStation 的语义。"
    "回调统计不能证明底层 GPU 时间区间互斥，不能当作无重叠的 GPU 时间分解。"
)
PERTURBATION_NOTE = (
    "baseline、profiled 全图与逐层采样来自独立顺序执行。差值包含 profiling 扰动和运行间波动；"
    "加载、输入拷贝、warmup 不在计时内。CUDA event 也可能包含主机提交造成的 GPU 空闲。"
)
COMPARISON_NOTE = (
    "Mixed 与 FP32 是两套已构建引擎，执行层数、融合方式和 tactic 均可能不同；"
    "此对比不是仅改变 precision 的单变量实验。层/类别耗时差不能直接解释为某个精度选项的因果收益，"
    "也不能仅凭这些计时断言计算或显存带宽瓶颈。"
)
NSIGHT_NOTE = (
    "Nsight 为单独 trace pass。Total 为捕获范围内累计 kernel 时间；每帧累计 = Total / trace.frames，"
    "不等于单帧延迟，重叠 kernel 的时间也会累加。Instances 是所有捕获帧的调用总数。"
    "Kernel 占比分母为该 trace 全部 kernel 累计时间。NVTX push/pop 是 CPU range，不是 GPU 层耗时。"
)
PUBLICATION_NOTE = (
    "公开 SQLite 移除了全部环境元数据记录，并将 StringIds 中出现的凭据值替换为占位符；"
    "保留字符串 ID，其他采集数据（包括计时表）未改变。"
    "原始 .nsys-rep 二进制仅在本地保留，不纳入 Git 或下载 ZIP；"
    "profile-data.zip 是净化后的公开归档。"
)
PLUGIN_CATEGORIES = {
    "ModulatedDeformConv2d": "DCN",
    "MultiscaleDeformableAttnPlugin_TRT": "MSDA",
}
LAYER_CATEGORIES = {
    "Convolution": "Convolution",
    "CaskConvolution": "Convolution",
    "CaskGemmConvolution": "Convolution",
    "MatrixMultiply": "MatrixMultiply",
    "gemm": "MatrixMultiply",
    "Reformat": "Reformat",
    "TrainStation": "TrainStation（整体回调）",
    "kgen": "kgen（代码生成/融合层）",
}
STATS = ("count", "mean_ms", "median_ms", "min_ms", "max_ms", "p95_ms")


def read_json(path):
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source)


def json_text(value, indent=None):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent)


def write_json(path, value):
    path.write_text(json_text(value, indent=2) + "\n", encoding="utf-8")


def finite_number(value, context, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{context}: expected a finite {'positive' if positive else 'nonnegative'} number")
    return value


def classify(inspector):
    """Use explicit engine metadata only; never guess operators from layer names."""
    if inspector is None:
        return "unknown（未唯一匹配）", "unknown"
    layer_type = inspector.get("LayerType")
    plugin_type = inspector.get("PluginType")
    if layer_type in ("Plugin", "PluginV2", "PluginV3") and plugin_type:
        category = PLUGIN_CATEGORIES.get(plugin_type, f"其他插件 · {plugin_type}")
        return category, f"LayerType={layer_type}; PluginType={plugin_type}"
    if not layer_type:
        return "unknown（缺少 LayerType）", "unique exact match; missing LayerType"
    return (LAYER_CATEGORIES.get(layer_type, f"其他已知类型 · {layer_type}"),
            f"LayerType={layer_type}")


def source_link(root, path):
    return quote(path.relative_to(root).as_posix(), safe="/")


def read_nsight(root, variant):
    prefix = variant.removesuffix("-layers")
    csv_path = root / f"{prefix}-stats_cuda_gpu_kern_sum.csv"
    trace_path = root / f"{prefix}-trace" / "report.json"
    if not csv_path.is_file():
        return None
    frames = None
    trace_report = None
    if trace_path.is_file():
        trace_report = read_json(trace_path)
        frames = trace_report.get("trace", {}).get("frames")
        if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
            raise ValueError(f"{trace_path}: trace.frames must be a positive integer")
    kernels = []
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"Name", "Total Time (ns)", "Instances"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path}: unsupported Nsight kernel summary columns")
        for row in reader:
            total_ns = int(row["Total Time (ns)"].replace(",", ""))
            instances = int(row["Instances"].replace(",", ""))
            finite_number(total_ns, f"{csv_path}: Total Time (ns)")
            finite_number(instances, f"{csv_path}: Instances", positive=True)
            kernels.append({"name": row["Name"], "total_ms": total_ns / 1e6,
                            "instances": instances,
                            "cumulative_ms_per_frame": total_ns / 1e6 / frames if frames else None})
    total_ms = math.fsum(row["total_ms"] for row in kernels)
    for row in kernels:
        row["kernel_time_percent"] = row["total_ms"] / total_ms * 100 if total_ms else 0
    kernels.sort(key=lambda row: (-row["total_ms"], row["name"]))
    # Nsight report binaries can capture process environment secrets. They stay
    # local; publish a SQLite download only alongside the sanitization record.
    paths = [csv_path, root / f"{prefix}-stats_cuda_api_sum.csv",
             root / f"{prefix}-stats_nvtx_pushpop_sum.csv", trace_path, root / "nsys-summary.json",
             root / f"{prefix}-nsys-summary-kernel-ranked.csv",
             root / f"{prefix}-nsys-summary-kernel-calls.csv",
             root / f"{prefix}-nsys-summary-frames.csv"]
    if (root / "publication-sanitization.json").is_file():
        paths += [root / f"{prefix}-nsys.sqlite", root / "publication-sanitization.json"]
    return {"frames": frames, "kernel_names": len(kernels),
            "instances": sum(row["instances"] for row in kernels),
            "total_kernel_ms": total_ms,
            "cumulative_kernel_ms_per_frame": total_ms / frames if frames else None,
            "top10": kernels[:10], "note": NSIGHT_NOTE,
            "trace_engine_sha256": trace_report.get("engine_sha256") if trace_report else None,
            "downloads": [{"name": path.name, "href": source_link(root, path)}
                          for path in paths if path.is_file()]}


def read_variant(root, directory):
    report = read_json(directory / "report.json")
    if report.get("mode") != "layers":
        raise ValueError(f"{directory}: report mode must be layers")
    original = read_json(directory / "layers.json")
    inspection = read_json(directory / "engine.layers.json")
    inspected = inspection.get("Layers") if isinstance(inspection, dict) else inspection
    if not isinstance(original, list) or not original:
        raise ValueError(f"{directory}: layers.json must contain nonempty layer rows")
    if not isinstance(inspected, list):
        raise ValueError(f"{directory}: inspector must contain a Layers list")
    index = defaultdict(list)
    for position, item in enumerate(inspected):
        if not isinstance(item, dict):
            raise ValueError(f"{directory}: inspector layer must be an object")
        if isinstance(item.get("Name"), str):
            index[item["Name"]].append((position, item))
    callback_names = Counter(item.get("name") for item in original)
    layers = []
    for position, item in enumerate(original):
        ordinal = item.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal != position:
            raise ValueError(f"{directory}: expected contiguous callback ordinal {position}, got {ordinal}")
        if not isinstance(item.get("name"), str):
            raise ValueError(f"{directory}: callback {ordinal} has no string name")
        for key in STATS:
            finite_number(item.get(key), f"{directory}: callback {ordinal} {key}", positive=key == "count")
        matches = index.get(item["name"], [])
        exact = matches[0][1] if len(matches) == 1 and callback_names[item["name"]] == 1 else None
        status = ("exact_unique" if exact is not None else
                  "unknown_no_exact_match" if not matches else
                  "unknown_repeated_callback_name" if callback_names[item["name"]] > 1 else
                  "unknown_ambiguous_name")
        category, basis = classify(exact)
        # Preserve the callback statistics and ordinal; do not aggregate by name.
        row = {**item, "category": category, "classification_basis": basis,
               "inspector_match": status, "inspector_candidates": len(matches),
               "inspector_index": matches[0][0] if exact is not None else None,
               "inspector": exact}
        layers.append(row)
    total = math.fsum(row["mean_ms"] for row in layers)
    ranked = sorted(layers, key=lambda row: (-row["mean_ms"], row["ordinal"]))
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, layer_time_percent=row["mean_ms"] / total * 100 if total else 0)
    grouped = defaultdict(list)
    for row in layers:
        grouped[row["category"]].append(row)
    groups = []
    for category, rows in grouped.items():
        group_ms = math.fsum(row["mean_ms"] for row in rows)
        groups.append({"category": category, "callbacks": len(rows), "sum_mean_ms": group_ms,
                       "layer_time_percent": group_ms / total * 100 if total else 0,
                       "layer_types": sorted({(row["inspector"] or {}).get("LayerType", "unknown")
                                              for row in rows}),
                       "plugin_types": sorted({row["inspector"].get("PluginType") for row in rows
                                               if row["inspector"] and row["inspector"].get("PluginType")}),
                       "classification_basis": sorted({row["classification_basis"] for row in rows})})
    groups.sort(key=lambda group: (-group["sum_mean_ms"], group["category"]))
    baseline = report["baseline"]["cuda_event"]
    profiled = report["profiled_whole_engine"]["cuda_event"]
    for label, stats in (("baseline", baseline), ("profiled", profiled)):
        for key in STATS:
            finite_number(stats.get(key), f"{directory}: {label} {key}", positive=key == "count")
    def perturbation(key):
        return {"delta_ms": profiled[key] - baseline[key],
                "delta_percent": (profiled[key] / baseline[key] - 1) * 100 if baseline[key] else None}
    paths = [directory / name for name in ("report.json", "layers.json", "layers.raw.json",
                                           "layers.csv", "engine.layers.json")]
    warnings = []
    if report.get("status") != "completed":
        warnings.append(f"源 report 状态为 {report.get('status', 'unknown')}，不能视为成功测量。")
    expected = report.get("layer_summary", {}).get("sum_mean_ms")
    if expected is not None and not math.isclose(total, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{directory}: callback sum disagrees with report.layer_summary")
    expected_count = report.get("layer_summary", {}).get("callbacks_per_iteration")
    if expected_count is not None and expected_count != len(layers):
        raise ValueError(f"{directory}: callback count disagrees with report.layer_summary")
    nsight = read_nsight(root, directory.name)
    if nsight and nsight["trace_engine_sha256"] != report.get("engine_sha256"):
        warnings.append("Nsight trace 的 engine SHA256 缺失或不一致，请单独核对。")
    summary = {
        "id": directory.name, "label": {"mixed-layers": "Mixed", "fp32-layers": "FP32"}.get(directory.name, directory.name),
        "status": report.get("status"), "engine": report.get("engine"),
        "engine_sha256": report.get("engine_sha256"), "inputs_sha256": report.get("inputs_sha256"),
        "warmup": report.get("warmup"), "repeat": report.get("repeat"),
        "environment_before": report.get("environment_before"), "environment_after": report.get("environment_after"),
        "io": report.get("io"), "engine_aux_streams": report.get("engine_aux_streams"),
        "engine_profiling_verbosity": report.get("engine_profiling_verbosity"),
        "baseline": report["baseline"], "profiled_whole_engine": report["profiled_whole_engine"],
        "perturbation_mean": perturbation("mean_ms"), "perturbation_median": perturbation("median_ms"),
        "callback_count": len(layers), "inspector_count": len(inspected),
        "matching": dict(Counter(row["inspector_match"] for row in layers)),
        "repeated_callback_names": {name: count for name, count in Counter(row["name"] for row in layers).items() if count > 1},
        "sum_layer_mean_ms": total, "groups": groups,
        "top20": [{key: row[key] for key in ("rank", "ordinal", "name", "mean_ms", "layer_time_percent", "category")}
                  for row in ranked[:20]],
        "validation": report.get("validation"), "validation_scope": report.get("validation_scope"),
        "warnings": warnings, "nsight": nsight,
        "downloads": [{"name": path.name, "href": source_link(root, path)} for path in paths if path.is_file()],
    }
    return summary, layers


def write_csvs(root, variants):
    layer_fields = ["variant", "rank", "ordinal", "name", *STATS, "layer_time_percent", "category",
                    "classification_basis", "inspector_match", "inspector_candidates", "inspector_index",
                    "LayerType", "PluginType", "TacticName", "Metadata", "Inputs", "Outputs"]
    with (root / "layers-ranked.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=layer_fields)
        writer.writeheader()
        for summary, layers in variants:
            for layer in sorted(layers, key=lambda row: row["rank"]):
                row = {field: layer.get(field) for field in layer_fields}
                row["variant"] = summary["id"]
                inspector = layer["inspector"] or {}
                for field in ("LayerType", "PluginType", "TacticName", "Metadata"):
                    row[field] = inspector.get(field, "unknown")
                for field in ("Inputs", "Outputs"):
                    row[field] = json_text(inspector.get(field))
                writer.writerow(row)
    group_fields = ["variant", "category", "callbacks", "sum_mean_ms", "layer_time_percent",
                    "layer_types", "plugin_types", "classification_basis"]
    with (root / "operators.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=group_fields)
        writer.writeheader()
        for summary, _ in variants:
            for group in summary["groups"]:
                row = {"variant": summary["id"], **group}
                for field in ("layer_types", "plugin_types", "classification_basis"):
                    row[field] = json_text(row[field])
                writer.writerow(row)


def fmt(value, digits=3):
    return "unknown" if value is None else f"{value:.{digits}f}"


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def write_readme(root, overview):
    lines = ["# TensorRT 逐层性能报告", "", f"**{SCOPE}**", "",
             "打开 [index.html](index.html) 可离线搜索、排序、切换引擎并查看层详情。", ""]
    if (root / "publication-sanitization.json").is_file():
        lines += ["## 公开归档说明", "", PUBLICATION_NOTE, "",
                  "净化范围与计时表核对见 [publication-sanitization.json](publication-sanitization.json)。",
                  "[下载净化后的完整数据包](profile-data.zip)。", ""]
    lines += ["## 统计口径", "", DENOMINATOR, "", FUSION_NOTE, "", PERTURBATION_NOTE, "",
             COMPARISON_NOTE, "",
             "输出有限只检查各测量 pass 最后一次输出；本报告没有数值参考对比或 mAP 评估。", "",
             "## 全图测量", "", "| 引擎 | baseline mean / median / p95 (ms) | profiled mean / median / p95 (ms) | mean 扰动 | 回调数 | 层 mean 之和 (ms) | 唯一匹配 |",
             "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    for v in overview["variants"]:
        b, p = v["baseline"]["cuda_event"], v["profiled_whole_engine"]["cuda_event"]
        triple = lambda stats: " / ".join(fmt(stats[key]) for key in ("mean_ms", "median_ms", "p95_ms"))
        lines.append(f"| {md(v['label'])} | {triple(b)} | {triple(p)} | {fmt(v['perturbation_mean']['delta_ms'])} ms ({fmt(v['perturbation_mean']['delta_percent'], 2)}%) | {v['callback_count']} | {fmt(v['sum_layer_mean_ms'])} | {v['matching'].get('exact_unique', 0)} |")
    for v in overview["variants"]:
        lines += ["", f"## {md(v['label'])}", "", f"warmup={v['warmup']}，repeat={v['repeat']}；状态 `{v['status']}`。", "",
                  f"Engine SHA256: `{v['engine_sha256']}`", "", f"Input SHA256: `{v['inputs_sha256']}`", ""]
        for warning in v["warnings"]:
            lines += [f"**{warning}**", ""]
        lines += ["### 按可信执行层类别分组", "", "| 类别 | 回调数 | mean 之和 (ms) | 层占比 |", "| --- | ---: | ---: | ---: |"]
        for group in v["groups"]:
            lines.append(f"| {md(group['category'])} | {group['callbacks']} | {fmt(group['sum_mean_ms'])} | {fmt(group['layer_time_percent'], 2)}% |")
        lines += ["", "Convolution / MatrixMultiply 类别可能包含 bias、激活或其他融合内容；数字代表整条回调，不能解释为单一原始算子的独立耗时。", "",
                  "### 最慢 20 条回调", "", "| 排名 | ordinal（0 起） | mean (ms) | 层占比 | 类别 | 完整回调名 |", "| ---: | ---: | ---: | ---: | --- | --- |"]
        for row in v["top20"]:
            lines.append(f"| {row['rank']} | {row['ordinal']} | {fmt(row['mean_ms'])} | {fmt(row['layer_time_percent'], 2)}% | {md(row['category'])} | {md(row['name'])} |")
        if v["nsight"]:
            n = v["nsight"]
            lines += ["", "### Nsight kernel Top 10", "", NSIGHT_NOTE, "",
                      f"捕获帧数：{n['frames'] if n['frames'] else 'unknown'}；累计 kernel：{fmt(n['total_kernel_ms'])} ms；每帧累计：{fmt(n['cumulative_kernel_ms_per_frame'])} ms。", "",
                      "| Kernel | Total (ms) | 每帧累计 (ms) | Instances（全部帧） | Kernel 占比 |", "| --- | ---: | ---: | ---: | ---: |"]
            for row in n["top10"]:
                lines.append(f"| {md(row['name'])} | {fmt(row['total_ms'])} | {fmt(row['cumulative_ms_per_frame'])} | {row['instances']} | {fmt(row['kernel_time_percent'], 2)}% |")
        lines += ["", "### 采集数据（公开版）", ""]
        for entry in v["downloads"] + (v["nsight"]["downloads"] if v["nsight"] else []):
            lines.append(f"- [{entry['name']}]({entry['href']})")
    lines += ["", "## 完整导出与复现", "", "- [overview.json](overview.json)：全图统计、匹配覆盖、类别和前 20 层。",
              "- [layers-ranked.csv](layers-ranked.csv)：全部回调，按引擎和耗时排名；保留 ordinal、全部原始统计及匹配元数据（UTF-8 BOM）。",
              "- [operators.csv](operators.csv)：执行层类别汇总，非原始 ONNX 算子独立计时（UTF-8 BOM）。", "",
              "```powershell", f"python tools/tensorrt/render_profile_report.py --root \"{root.as_posix()}\"", "```", "",
              "脚本只需 Python 标准库，无需 TensorRT / CUDA；只读取 report、layers、inspector 和 trace 数据，不修改采集数据或执行净化。",
              "HTML 内嵌全部回调和匹配元数据，不使用 fetch、CDN 或外部 JavaScript。", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TensorRT 层性能报告</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#1d2c3d;background:#f2f5f8;font-size:14px}*{box-sizing:border-box}body{margin:0}main{max-width:1480px;margin:auto;padding:26px}h1{font-size:27px;margin:0 0 8px}h2{font-size:20px;margin:26px 0 12px}h3{margin:0 0 10px;font-size:16px}p{line-height:1.7;margin:8px 0}.muted{color:#596a7b}.warning{border:1px solid #d3a348;background:#fff6dd;padding:14px 17px;border-radius:7px;margin:16px 0;line-height:1.7}.card{background:white;border:1px solid #d8e1e9;border-radius:8px;padding:17px;min-width:0}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{font-size:26px;font-weight:650;margin-top:8px}.metric small{font-size:14px;font-weight:400}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}input,select,button{font:inherit;border:1px solid #aebfcc;border-radius:5px;padding:9px;background:white;color:inherit}input{flex:1;min-width:210px}button{cursor:pointer}button:hover{background:#eaf3fa}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #8ac4ef;outline-offset:2px}a{color:#1263a2}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #e0e6ec;vertical-align:top}th{background:#eaf0f5;font-weight:600;white-space:nowrap}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.table-wrap{overflow:auto;border:1px solid #d8e1e9;border-radius:7px}.layer-name{width:100%;font:inherit;color:#1263a2;background:none;border:0;padding:0;text-align:left;word-break:break-word;line-height:1.4}.name-cell{min-width:280px;max-width:700px;overflow-wrap:anywhere}.type-cell{max-width:220px;overflow-wrap:anywhere}.bar{width:120px;height:7px;background:#e3edf5;margin-top:6px}.bar>span{height:7px;display:block;background:#3b87be}.columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;font-size:12px;background:#f3f6f9;padding:13px;border-radius:5px}dialog{border:1px solid #9fb3c6;border-radius:9px;padding:24px;width:min(1100px,94vw);max-height:90vh;overflow:auto}dialog::backdrop{background:#182a3a99}.dialog-head{display:flex;align-items:flex-start;justify-content:space-between;gap:15px}#detail-title{overflow-wrap:anywhere}.downloads{display:flex;gap:14px;flex-wrap:wrap;line-height:1.7}.small{font-size:12px}.empty{padding:20px;color:#596a7b}@media(max-width:850px){main{padding:15px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.columns{grid-template-columns:1fr}.metric{font-size:23px}}
</style></head><body><main>
<h1>TensorRT 层性能报告</h1><p class="muted">BEVFormer · 回调级耗时、执行层类别与全图扰动 · 离线可交互</p>
<div class="warning"><strong>__SCOPE__</strong><br>最终输出有限性检查不能替代数值精度、真实时序或 mAP 验收。</div>
<div class="toolbar"><label for="variant">引擎</label><select id="variant"></select><span id="run-info" class="muted"></span></div>
<div id="warnings"></div><div id="cards" class="cards"></div>
<h2>全图测量与 profiling 扰动</h2><div class="table-wrap"><table><thead><tr><th>测量</th><th class="num">次数</th><th class="num">mean ms</th><th class="num">median ms</th><th class="num">p95 ms</th><th class="num">min / max ms</th></tr></thead><tbody id="whole"></tbody></table></div>
<p id="perturbation"></p><p class="muted small">__PERTURBATION__</p><p class="muted small">__COMPARISON__</p>
<h2>执行层类别汇总</h2><p>__DENOMINATOR__</p><p class="muted small">__FUSION__ Convolution / MatrixMultiply 类别可包含融合内容，其值代表整条回调。</p>
<div class="table-wrap"><table><thead><tr><th>可信类别</th><th class="num">回调数</th><th class="num">mean 之和 ms</th><th class="num">层占比</th><th>分类证据</th></tr></thead><tbody id="groups"></tbody></table></div>
<h2>全部回调层 <span class="muted small">默认显示最慢 20 层，点击名称查看完整详情</span></h2>
<div class="toolbar"><input id="search" type="search" placeholder="搜索完整名称、类别、LayerType、PluginType 或 ordinal" aria-label="搜索回调层"><label for="sort">排序</label><select id="sort"><option value="slow">mean 从高到低</option><option value="fast">mean 从低到高</option><option value="ordinal">原始 ordinal</option></select><label for="limit">显示</label><select id="limit"><option value="20">前 20</option><option value="100">前 100</option><option value="0">全部</option></select></div>
<p id="result-count" class="muted"></p><div class="table-wrap"><table><thead><tr><th class="num">耗时排名</th><th class="num">ordinal</th><th>回调名称</th><th>类别</th><th class="num">mean ms</th><th class="num">p95 ms</th><th class="num">层占比</th></tr></thead><tbody id="layers"></tbody></table></div>
<section id="nsight-section"><h2>Nsight kernel Top 10</h2><p class="muted small">__NSIGHT__</p><p id="nsight-info"></p><div class="table-wrap"><table><thead><tr><th>Kernel</th><th class="num">Total ms</th><th class="num">每帧累计 ms</th><th class="num">Instances<br>全部帧</th><th class="num">Kernel 占比</th></tr></thead><tbody id="kernels"></tbody></table></div><p id="nsight-empty" class="muted" hidden>未找到本引擎的 Nsight kernel 汇总；未推测任何 GPU kernel 数据。</p></section>
<h2>数据与下载</h2><p id="environment" class="muted"></p><div class="downloads"><a href="overview.json" download>overview.json</a><a href="layers-ranked.csv" download>全部层 CSV</a><a href="operators.csv" download>类别汇总 CSV</a><a href="README.md">中文说明</a></div>__PUBLICATION_SECTION__<h3 style="margin-top:18px">原始层数据</h3><div id="downloads" class="downloads"></div><h3 style="margin-top:18px">Nsight 公开数据</h3><div id="nsight-downloads" class="downloads"></div><p class="small muted">HTML 所有数据均已内嵌，无需网络或本地 HTTP 服务。原始 .nsys-rep 仅在本地保留，不提供公开下载。</p>
<details><summary>引擎、输入与环境追溯</summary><pre id="provenance"></pre></details>
</main><dialog id="detail"><div class="dialog-head"><h2 id="detail-title"></h2><button id="close" aria-label="关闭层详情">关闭</button></div><p id="detail-summary"></p><div class="columns"><section><h3>原始回调统计与匹配状态</h3><pre id="detail-callback"></pre></section><section><h3>Inspector 元数据（唯一精确匹配）</h3><pre id="detail-inspector"></pre></section></div><p class="muted small">元数据中的多个 ONNX 节点不等于多次独立计时；融合层不能按名称拆分耗时。ordinal 为一次执行内从 0 开始的回调序号。</p></dialog>
<script id="report-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("report-data").textContent);
const $=id=>document.getElementById(id);
const number=(x,d=3)=>x==null?"unknown":Number(x).toLocaleString("zh-CN",{minimumFractionDigits:d,maximumFractionDigits:d});
const cell=(row,value,cls="")=>{const e=document.createElement("td");e.textContent=value;e.className=cls;row.append(e);return e;};
const tr=()=>document.createElement("tr");
const current=()=>DATA.variants.find(v=>v.summary.id===$("variant").value);
function links(id,items){const box=$(id);box.replaceChildren();for(const item of items){const a=document.createElement("a");a.textContent=item.name;a.href=item.href;a.download="";box.append(a);}if(!items.length)box.textContent="无可用文件";}
function openDetail(layer){const {inspector,...callback}=layer;$("detail-title").textContent=layer.name;$("detail-summary").textContent=`ordinal ${layer.ordinal} · ${layer.category} · mean ${number(layer.mean_ms)} ms · ${number(layer.layer_time_percent,2)}% 层占比`;$("detail-callback").textContent=JSON.stringify(callback,null,2);$("detail-inspector").textContent=inspector?JSON.stringify(inspector,null,2):"unknown：没有唯一精确匹配，未附加任何推测的 inspector 信息。";$("detail").showModal();}
function renderLayers(){const v=current(),q=$("search").value.trim().toLocaleLowerCase();let rows=v.layers.filter(x=>!q||[x.name,x.category,x.ordinal,(x.inspector||{}).LayerType,(x.inspector||{}).PluginType].join(" ").toLocaleLowerCase().includes(q));const matched=rows.length;rows.sort($("sort").value==="ordinal"?(a,b)=>a.ordinal-b.ordinal:$("sort").value==="fast"?(a,b)=>a.mean_ms-b.mean_ms||a.ordinal-b.ordinal:(a,b)=>b.mean_ms-a.mean_ms||a.ordinal-b.ordinal);const limit=Number($("limit").value);if(limit)rows=rows.slice(0,limit);$("result-count").textContent=`显示 ${rows.length} / ${matched} 条匹配回调；总计 ${v.layers.length} 条。ordinal 始终保留原始执行顺序，不随排序改变。`;const fragment=document.createDocumentFragment();for(const x of rows){const row=tr();cell(row,x.rank,"num");cell(row,x.ordinal,"num");const name=cell(row,"","name-cell"),button=document.createElement("button");button.className="layer-name";button.textContent=x.name;button.addEventListener("click",()=>openDetail(x));name.append(button);cell(row,x.category,"type-cell");cell(row,number(x.mean_ms),"num");cell(row,number(x.p95_ms),"num");cell(row,number(x.layer_time_percent,2)+"%","num");fragment.append(row);}$("layers").replaceChildren(fragment);}
function render(){const v=current(),s=v.summary,b=s.baseline.cuda_event,p=s.profiled_whole_engine.cuda_event;$("run-info").textContent=`${s.callback_count} 条回调 · warmup ${s.warmup} · repeat ${s.repeat} · ${s.status}`;$("warnings").replaceChildren();for(const warning of s.warnings){const el=document.createElement("p");el.className="warning";el.textContent=warning;$("warnings").append(el);}$("cards").replaceChildren();for(const [label,value,unit]of [["baseline median",b.median_ms,"ms"],["profiled median",p.median_ms,"ms"],["全部层 mean 之和",s.sum_layer_mean_ms,"ms"],["唯一精确匹配",s.matching.exact_unique||0,`/ ${s.callback_count}`]]){const el=document.createElement("div");el.className="card";const title=document.createElement("div");title.className="muted";title.textContent=label;const metric=document.createElement("div");metric.className="metric";metric.textContent=number(value,unit==="ms"?3:0)+" ";const small=document.createElement("small");small.textContent=unit;metric.append(small);el.append(title,metric);$("cards").append(el);}$("whole").replaceChildren();const measures=[["baseline · CUDA event",b],["profiled · CUDA event",p],["baseline · host 提交 + 同步",s.baseline.host_submit_and_sync],["profiled · host 提交 + 同步",s.profiled_whole_engine.host_submit_and_sync],["profiled · host 提交 + 同步 + report",s.profiled_whole_engine.host_submit_sync_and_report]];for(const[label,stats]of measures){if(!stats)continue;const row=tr();cell(row,label);cell(row,stats.count,"num");for(const key of["mean_ms","median_ms","p95_ms"])cell(row,number(stats[key]),"num");cell(row,number(stats.min_ms)+" / "+number(stats.max_ms),"num");$("whole").append(row);}$("perturbation").textContent=`profiled − baseline：mean ${number(s.perturbation_mean.delta_ms)} ms（${number(s.perturbation_mean.delta_percent,2)}%）；median ${number(s.perturbation_median.delta_ms)} ms（${number(s.perturbation_median.delta_percent,2)}%）。`;$("groups").replaceChildren();for(const g of s.groups){const row=tr();cell(row,g.category);cell(row,g.callbacks,"num");cell(row,number(g.sum_mean_ms),"num");const percent=cell(row,number(g.layer_time_percent,2)+"%","num");const bar=document.createElement("div"),fill=document.createElement("span");bar.className="bar";fill.style.width=Math.min(100,Math.max(0,g.layer_time_percent))+"%";bar.append(fill);percent.append(bar);cell(row,g.classification_basis.join("；"),"small type-cell");$("groups").append(row);}const n=s.nsight;$("kernels").replaceChildren();$("nsight-empty").hidden=!!n;$("nsight-info").textContent=n?`捕获 ${n.frames??"unknown"} 帧 · ${n.instances.toLocaleString("zh-CN")} 次 kernel 调用 · 累计 ${number(n.total_kernel_ms)} ms · 每帧累计 ${number(n.cumulative_kernel_ms_per_frame)} ms`:"";if(n)for(const k of n.top10){const row=tr();cell(row,k.name,"name-cell");cell(row,number(k.total_ms),"num");cell(row,number(k.cumulative_ms_per_frame),"num");cell(row,k.instances.toLocaleString("zh-CN"),"num");cell(row,number(k.kernel_time_percent,2)+"%","num");$("kernels").append(row);}links("downloads",s.downloads);links("nsight-downloads",n?n.downloads:[]);const env=s.environment_before||{};$("environment").textContent=`${(env.gpu||{}).name||"GPU unknown"} · TensorRT ${env.tensorrt||"unknown"} · CUDA ${env.torch_cuda||"unknown"} · 采集 ${env.utc||"unknown"}`;$("provenance").textContent=JSON.stringify({engine:s.engine,engine_sha256:s.engine_sha256,inputs_sha256:s.inputs_sha256,io:s.io,engine_aux_streams:s.engine_aux_streams,engine_profiling_verbosity:s.engine_profiling_verbosity,environment_before:s.environment_before,environment_after:s.environment_after,matching:s.matching,validation:s.validation,validation_scope:s.validation_scope},null,2);renderLayers();}
for(const v of DATA.variants){const option=document.createElement("option");option.value=v.summary.id;option.textContent=v.summary.label;$("variant").append(option);}$("variant").addEventListener("change",render);$("search").addEventListener("input",renderLayers);$("sort").addEventListener("change",renderLayers);$("limit").addEventListener("change",renderLayers);$("close").addEventListener("click",()=>$("detail").close());render();
</script></body></html>'''


def write_html(root, overview, variants):
    # Escape HTML parser delimiters as well as JavaScript line separators. Values
    # are subsequently rendered with textContent, never interpreted as markup.
    payload = json_text({"schema_version": SCHEMA_VERSION,
                         "variants": [{"summary": summary, "layers": layers}
                                      for summary, layers in variants]})
    for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(char, escaped)
    document = HTML
    for token, value in (("__SCOPE__", SCOPE), ("__DENOMINATOR__", DENOMINATOR),
                         ("__PERTURBATION__", PERTURBATION_NOTE), ("__FUSION__", FUSION_NOTE),
                         ("__NSIGHT__", NSIGHT_NOTE), ("__COMPARISON__", COMPARISON_NOTE)):
        document = document.replace(token, html.escape(value))
    document = document.replace("__DATA__", payload)
    publication = ""
    if (root / "publication-sanitization.json").is_file():
        publication = (
            '<p class="small muted">' + html.escape(PUBLICATION_NOTE) + '</p>'
            '<div class="downloads"><a href="publication-sanitization.json">净化记录与计时表核对</a>'
            '<a href="profile-data.zip" download>下载净化后的完整数据包</a></div>'
        )
    document = document.replace("__PUBLICATION_SECTION__", publication)
    (root / "index.html").write_text(document, encoding="utf-8")


def render(root):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Missing profiling root: {root}")
    directories = sorted((path for path in root.glob("*-layers") if path.is_dir()),
                         key=lambda path: (path.name != "mixed-layers", path.name != "fp32-layers", path.name))
    variants = []
    for directory in directories:
        required = [directory / name for name in ("report.json", "layers.json", "engine.layers.json")]
        if not all(path.is_file() for path in required):
            raise ValueError(f"Incomplete layer data in {directory}; require report.json, layers.json, engine.layers.json")
        variants.append(read_variant(root, directory))
    if not variants:
        raise ValueError(f"No *-layers directories in {root}")
    overview = {"schema_version": SCHEMA_VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
                "scope": SCOPE, "denominator": DENOMINATOR, "fusion_note": FUSION_NOTE,
                "perturbation_note": PERTURBATION_NOTE, "comparison_note": COMPARISON_NOTE,
                "nsight_note": NSIGHT_NOTE,
                "variants": [summary for summary, _ in variants]}
    if (root / "publication-sanitization.json").is_file():
        overview["publication_note"] = PUBLICATION_NOTE
        overview["publication_record"] = "publication-sanitization.json"
    # Validate all inputs before replacing any generated output.
    write_json(root / "overview.json", overview)
    write_csvs(root, variants)
    write_readme(root, overview)
    write_html(root, overview, variants)
    return overview


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Profiling root containing *-layers/ directories")
    args = parser.parse_args()
    try:
        overview = render(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json_text({"status": "completed", "index": str((args.root / "index.html").resolve()),
                     "variants": [{"id": v["id"], "callbacks": v["callback_count"],
                                   "sum_layer_mean_ms": v["sum_layer_mean_ms"], "matching": v["matching"]}
                                  for v in overview["variants"]]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
