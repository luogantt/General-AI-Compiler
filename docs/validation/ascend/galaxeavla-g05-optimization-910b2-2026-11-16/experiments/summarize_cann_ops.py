#!/usr/bin/env python3
"""Summarize exported CANN operator CSV files without importing torch.

Each CSV is reported independently: details and statistic tables often describe
the same kernels and must never be added together. Summed kernel durations can
overlap and are not request wall time or proof of a stage/critical-path latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
import tempfile


PATTERNS = ("*operator_details*.csv", "*operator_statistic*.csv",
            "*op_summary*.csv", "*op_statistic*.csv", "*kernel_details*.csv")


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower().replace("µ", "u").replace("μ", "u"))


def column(headers, candidates):
    lookup = {normalized(value): value for value in headers}
    return next((lookup[key] for key in candidates if key in lookup), None)


def duration_column(headers, prefixes):
    for prefix in prefixes:
        for unit, factor in (("us", 1.0), ("ns", .001), ("ms", 1000.0), ("s", 1_000_000.0)):
            found = column(headers, (prefix + unit,))
            if found:
                return found, factor
    return None, None


def number(value):
    result = float(str(value).strip().replace(",", ""))
    if not math.isfinite(result) or result < 0:
        raise ValueError("Expected a finite, nonnegative number")
    return result


def family_summary(groups, total, top):
    """Optional name heuristic; retain the exact-name tables as primary evidence."""
    families = {}
    for item in groups:
        family = re.sub(r"_\d+$", "", item["op_type"])
        key = (family, item["task_type"])
        target = families.setdefault(key, {"name_family": family, "task_type": key[1],
            "exact_groups": 0, "invocations": 0, "summed_duration_us": 0.0,
            "example_exact_names": []})
        target["exact_groups"] += 1
        target["invocations"] += item["invocations"]
        target["summed_duration_us"] += item["summed_duration_us"]
        if len(target["example_exact_names"]) < 3:
            target["example_exact_names"].append(item["op_type"])
    for item in families.values():
        item["average_duration_us"] = item["summed_duration_us"] / item["invocations"]
        item["share_of_summed_duration_percent"] = item["summed_duration_us"] / total * 100 if total else 0.0
    return {"heuristic": "Strip only one trailing underscore+decimal integer, e.g. MatMul_123 -> MatMul; not a verified operator/stage mapping",
            "top_by_summed_duration": sorted(families.values(), key=lambda x: x["summed_duration_us"], reverse=True)[:top],
            "top_by_invocation_count": sorted(families.values(), key=lambda x: x["invocations"], reverse=True)[:top]}


def summarize(path, top):
    # Large shape or stack fields can exceed csv's small default field limit.
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        statistic = "statistic" in path.name.lower()
        prefixes = (("deviceselfduration", "totaltime", "totalduration", "taskduration", "duration") if statistic
                    else ("deviceselfduration", "taskduration", "duration"))
        duration, scale = duration_column(headers, prefixes)
        count = column(headers, ("count", "totalcount", "calls", "occurrences")) if statistic else None
        duration_is_average = False
        # PyTorch's operator_details has nested host/device Total columns.
        # Those include descendants and cannot be summed without double counting.
        # Its Device Self column excludes descendants; kernel_details instead
        # reports each kernel's elapsed Duration.
        is_operator_self = duration is not None and normalized(duration).startswith("deviceselfduration")
        if duration is None and statistic and count is not None:
            duration, scale = duration_column(headers, ("avgtime", "averagetime", "avgduration"))
            duration_is_average = duration is not None
        if duration is None:
            raise ValueError(f"No recognized duration column in {path}: {headers}")
        if statistic and count is None:
            raise ValueError(f"Statistic table has no recognized invocation count in {path}: {headers}")
        op = column(headers, ("optype", "operatortype", "type", "opname", "name", "kernelname"))
        task = column(headers, ("tasktype", "coretype", "acceleratorcore", "taskcategory"))
        name = column(headers, ("opname", "name", "kernelname", "operatorname"))
        shape = column(headers, ("inputshapes", "inputshape"))
        if op is None:
            raise ValueError(f"No recognized operator identity in {path}: {headers}")
        groups, excluded_examples = {}, []
        rows = excluded = name_fallback_rows = 0
        for index, row in enumerate(reader, start=2):
            rows += 1
            try:
                duration_us = number(row[duration]) * scale
                calls = number(row[count]) if count is not None else 1
                if calls != int(calls) or calls < 1:
                    raise ValueError("Invocation count must be a positive integer")
                calls = int(calls)
                if duration_is_average:
                    duration_us *= calls
            except (TypeError, KeyError, ValueError) as exc:
                excluded += 1
                if len(excluded_examples) < 5:
                    excluded_examples.append({"csv_line": index, "reason": str(exc),
                                              "duration": row.get(duration), "count": row.get(count)})
                continue
            op_value = row.get(op, "") or ""
            if op_value.strip().lower() in {"", "n/a", "na", "none", "nan", "unknown"}:
                op_value = row.get(name, "") if name else ""
                name_fallback_rows += 1
            key = (op_value or "<empty>", row.get(task, "") if task else "<unspecified>")
            item = groups.setdefault(key, {"op_type": key[0], "task_type": key[1],
                "invocations": 0, "source_rows": 0, "summed_duration_us": 0.0,
                "example_names": [], "example_input_shapes": []})
            item["invocations"] += calls
            item["source_rows"] += 1
            item["summed_duration_us"] += duration_us
            for source, destination in ((name, "example_names"), (shape, "example_input_shapes")):
                value = row.get(source, "") if source else ""
                if value and value not in item[destination] and len(item[destination]) < 3:
                    item[destination].append(value)
        if not groups:
            raise ValueError(f"No usable operator rows in {path}; excluded={excluded}")
    total = sum(item["summed_duration_us"] for item in groups.values())
    total_calls = sum(item["invocations"] for item in groups.values())
    for item in groups.values():
        item["average_duration_us"] = item["summed_duration_us"] / item["invocations"]
        item["share_of_summed_duration_percent"] = item["summed_duration_us"] / total * 100 if total else 0.0
    by_duration = sorted(groups.values(), key=lambda item: item["summed_duration_us"], reverse=True)
    by_count = sorted(groups.values(), key=lambda item: item["invocations"], reverse=True)
    return {"path": str(path.resolve()), "table_kind": "statistic" if statistic else "details",
        "duration_semantics": "operator attributed device self time" if is_operator_self else "kernel/operator reported duration",
        "nested_total_columns_ignored": [name for name in headers if normalized(name).startswith(("hosttotalduration", "devicetotalduration"))],
        "columns": {"duration": duration, "duration_to_us": scale,
                    "duration_multiplied_by_count": duration_is_average, "count": count,
                    "op_type": op, "task_type": task, "name": name, "input_shape": shape},
        "source_rows": rows, "excluded_rows": excluded, "excluded_examples": excluded_examples,
        "grouping": "nonmissing operator Type, falling back to Name per row",
        "name_fallback_rows": name_fallback_rows,
        "operator_groups": len(groups), "invocations": total_calls,
        "summed_duration_us": total,
        "top_by_summed_duration": by_duration[:top], "top_by_invocation_count": by_count[:top],
        "name_family_view": family_summary(groups.values(), total, top),
        "warning": "Durations may overlap. Nested Host/Device Total columns are never summed. Do not equate sums to request wall time, add different CSV tables, or infer model stages without trace ranges."}


def self_test():
    with tempfile.TemporaryDirectory(prefix="cann-csv-test-") as directory:
        root = Path(directory)
        detail = root / "operator_details.csv"
        detail.write_text("Op Name,OP Type,Task Type,Task Duration(us),Input Shapes\n"
                          "Add_1,Add,AI_VECTOR_CORE,2.5,32x128\n"
                          "Add_2,Add,AI_VECTOR_CORE,3.5,32x128\n"
                          "MatMul_1,MatMul,AI_CORE,10,128x128\n"
                          "bad,Add,AI_VECTOR_CORE,N/A,32x128\n", encoding="utf-8-sig")
        result = summarize(detail, 10)
        assert result["summed_duration_us"] == 16.0
        assert result["invocations"] == 3 and result["excluded_rows"] == 1
        assert result["top_by_invocation_count"][0]["op_type"] == "Add"
        assert result["top_by_summed_duration"][0]["op_type"] == "MatMul"
        statistic = root / "operator_statistic.csv"
        statistic.write_text("OP Type,Core Type,Count,Total Time(ms)\nAdd,AI_VECTOR_CORE,2,.006\n", encoding="utf-8")
        result = summarize(statistic, 10)
        assert result["summed_duration_us"] == 6 and result["invocations"] == 2
        statistic.write_text("OP Type,Count,Avg Time(ns)\nAdd,2,3000\n", encoding="utf-8")
        result = summarize(statistic, 10)
        assert result["summed_duration_us"] == 6 and result["columns"]["duration_multiplied_by_count"]
        statistic.write_text("OP Type,Count,Duration\nAdd,2,3\n", encoding="utf-8")
        try:
            summarize(statistic, 10)
        except ValueError as exc:
            assert "No recognized duration" in str(exc)
        else:
            raise AssertionError("Unit-less duration must not be silently interpreted")
        detail.write_text("Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us),Device Self Duration With AICore(us),Device Total Duration With AICore(us)\n"
                          "aten::root,32,,1,10,0,100,0,100\n"
                          "aten::mm,32,,9,9,100,100,100,100\n", encoding="utf-8")
        result = summarize(detail, 10)
        assert result["summed_duration_us"] == 100
        assert result["columns"]["duration"] == "Device Self Duration(us)"
        assert len(result["nested_total_columns_ignored"]) == 3
        kernel = root / "kernel_details.csv"
        kernel.write_text("Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num\n"
                          "0,MatMul_1,MatMul,AI_CORE,123,12,0,20\n", encoding="utf-8")
        result = summarize(kernel, 10)
        assert result["summed_duration_us"] == 12
        assert result["top_by_summed_duration"][0]["task_type"] == "AI_CORE"
        kernel.write_text("Device_id,Name,Type,Accelerator Core,Duration(us)\n"
                          "0,MatMul,N/A,N/A,12\n0,Softmax,N/A,N/A,4\n", encoding="utf-8")
        result = summarize(kernel, 10)
        assert result["operator_groups"] == 2 and result["name_fallback_rows"] == 2
        assert result["top_by_summed_duration"][0]["op_type"] == "MatMul"
        families = family_summary([
            {"op_type": "MatMul_1", "task_type": "N/A", "invocations": 10, "summed_duration_us": 100},
            {"op_type": "MatMul_22", "task_type": "N/A", "invocations": 10, "summed_duration_us": 90},
        ], 190, 10)
        assert len(families["top_by_summed_duration"]) == 1
        assert families["top_by_summed_duration"][0]["summed_duration_us"] == 190
        assert families["top_by_summed_duration"][0]["invocations"] == 20
    print("8 CANN CSV parser cases passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.top < 1 or not args.paths:
        parser.error("Supply a CSV/directory and a positive --top")
    files = set()
    for path in args.paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            for pattern in PATTERNS:
                files.update(item.resolve() for item in path.rglob(pattern))
        else:
            parser.error(f"Path does not exist: {path}")
    if not files:
        parser.error("No supported operator CSV files found")
    result = {"scope": "Each CSV is independent; no cross-table duration sum",
              "files": [summarize(path, args.top) for path in sorted(files)]}
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
