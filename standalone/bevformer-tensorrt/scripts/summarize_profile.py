#!/usr/bin/env python3
"""Summarize IProfiler callbacks using unique, exact inspector-name matches.

No CUDA/TensorRT dependency. Each callback ordinal contributes once; fused
layers stay intact. Names and ONNX metadata are never used to guess op types.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def nonnegative(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def validate_times(record, label):
    """Validate latency fields, including sample arrays, without interpreting deltas."""
    for key, value in record.items():
        if isinstance(value, dict):
            validate_times(value, f"{label}.{key}")
        elif key.endswith("_ms"):
            for item in value if isinstance(value, list) else [value]:
                nonnegative(item, f"{label}.{key}")


def summarize(report, callbacks, inspector):
    if not isinstance(callbacks, list) or not isinstance(inspector.get("Layers"), list):
        raise ValueError("Expected layers.json array and engine.layers.json Layers array")
    for key in ("baseline", "profiled_whole_engine"):
        validate_times(report.get(key, {}), key)
    by_name = defaultdict(list)
    for layer in inspector["Layers"]:
        if isinstance(layer.get("Name"), str):
            by_name[layer["Name"]].append(layer)
    seen = set()
    for row in callbacks:
        ordinal = row.get("ordinal")
        if type(ordinal) is not int or ordinal < 0 or ordinal in seen:
            raise ValueError(f"Invalid or duplicate callback ordinal: {ordinal!r}")
        seen.add(ordinal)
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"Missing callback name at ordinal {ordinal}")
        if type(row.get("count")) is not int or row["count"] <= 0:
            raise ValueError(f"Invalid sample count at ordinal {ordinal}")
        nonnegative(row.get("mean_ms"), f"callback {ordinal}.mean_ms")
        validate_times(row, f"callback {ordinal}")
    name_counts = Counter(row["name"] for row in callbacks)
    rows = []
    for row in callbacks:
        matches = by_name[row["name"]]
        status = ("duplicate_callback_name" if name_counts[row["name"]] != 1 else
                  "unmatched" if not matches else
                  "duplicate_inspector_name" if len(matches) != 1 else "matched")
        layer = matches[0] if status == "matched" else {}
        layer_type = layer.get("LayerType")
        plugin_type = layer.get("PluginType")
        if status == "matched" and (not isinstance(layer_type, str) or not layer_type):
            status = "missing_layer_type"
        if status == "matched" and layer_type.startswith("Plugin") and (not isinstance(plugin_type, str) or not plugin_type):
            status = "missing_plugin_type"
        category = (f"{layer_type}/{plugin_type}" if plugin_type else layer_type) if status == "matched" else "unknown"
        rows.append({**row, "category": category, "match_status": status,
                     "layer_type": layer_type, "plugin_type": plugin_type})
    total = math.fsum(row["mean_ms"] for row in rows)
    buckets = defaultdict(list)
    for row in rows:
        row["callback_share_percent"] = 100 * row["mean_ms"] / total if total else 0.0
        buckets[row["category"]].append(row)
    groups = []
    for category, members in buckets.items():
        ms = math.fsum(row["mean_ms"] for row in members)
        groups.append({"category": category, "callback_count": len(members), "sum_mean_ms": ms,
                       "callback_share_percent": 100 * ms / total if total else 0.0})
    ranked = sorted(rows, key=lambda row: (-row["mean_ms"], row["ordinal"]))
    return {
        "schema_version": 1,
        "scope": "IProfiler callback means; not a decomposition of unprofiled whole-engine latency",
        "matching": "Unique exact names in both callback and inspector lists; no metadata/name-based type inference",
        "engine_sha256": report.get("engine_sha256"), "inputs_sha256": report.get("inputs_sha256"),
        "baseline": report.get("baseline"), "profiled_whole_engine": report.get("profiled_whole_engine"),
        "timing_scope": report.get("timing_scope"), "validation_scope": report.get("validation_scope"),
        "callback_count": len(rows), "sum_mean_ms": total,
        "match_counts": dict(sorted(Counter(row["match_status"] for row in rows).items())),
        "groups": sorted(groups, key=lambda group: (-group["sum_mean_ms"], group["category"])),
        "top20": ranked[:20], "callbacks": sorted(rows, key=lambda row: row["ordinal"]),
    }


def markdown(summary):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    lines = ["# TensorRT 层性能摘要", "",
             "每个 ordinal 回调仅计一次。类别取自唯一精确匹配的 Inspector LayerType / PluginType；重名、未匹配或缺少类型的回调标为 unknown。融合层保持整体，不拆成内部 ONNX 算子。", "",
             f"回调数：{summary['callback_count']}；回调 mean 之和：{summary['sum_mean_ms']:.3f} ms。", "",
             "**下面的占比是回调 mean 之和的占比，不能当作无重叠的 GPU 时间分解，也不是 baseline 整网时间占比。** Profiling 可能扰动运行；全图计时排除加载、输入拷贝和预热。", ""]
    lines += ["| 测量 | CUDA Event mean (ms) |", "| --- | ---: |"]
    for title, key in (("未附加 profiler 的整网", "baseline"), ("附加 profiler 的整网", "profiled_whole_engine")):
        value = (summary.get(key) or {}).get("cuda_event", {}).get("mean_ms")
        lines.append(f"| {title} | {value:.3f} |" if value is not None else f"| {title} | 未提供 |")
    lines += ["", "## 按执行层类别", "", "| 类别 | 回调数 | mean 合计 (ms) | 回调占比 |", "| --- | ---: | ---: | ---: |"]
    for group in summary["groups"]:
        lines.append(f"| {cell(group['category'])} | {group['callback_count']} | {group['sum_mean_ms']:.3f} | {group['callback_share_percent']:.2f}% |")
    lines += ["", "## 最慢 20 条回调", "", "| ordinal | 类别 | mean (ms) | 名称 |", "| ---: | --- | ---: | --- |"]
    for row in summary["top20"]:
        lines.append(f"| {row['ordinal']} | {cell(row['category'])} | {row['mean_ms']:.3f} | {cell(row['name'])} |")
    lines += ["", "## 追溯", "", f"匹配状态：`{json.dumps(summary['match_counts'], ensure_ascii=False)}`。", "",
              "[完整摘要与全部 ordinal](summary.json) · [Top 20 CSV](top.csv)", "",
              f"Engine SHA-256：`{summary.get('engine_sha256')}`", "",
              f"输入 SHA-256：`{summary.get('inputs_sha256')}`", "",
              "本摘要只分析计时，不新增数值正确性、完整时序或 mAP/NDS 验证。", ""]
    return "\n".join(lines)


def load_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; never overwrite")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    paths = [args.profile_dir / name for name in ("report.json", "layers.json", "engine.layers.json")]
    summary = summarize(*(load_json(path) for path in paths))
    summary["sources"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "top.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["ordinal", "name", "category", "match_status", "count", "mean_ms", "callback_share_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["top20"])
    (args.output_dir / "README.md").write_text(markdown(summary), encoding="utf-8")
    print(f"Summarized {summary['callback_count']} callbacks -> {args.output_dir}")


if __name__ == "__main__":
    main()
