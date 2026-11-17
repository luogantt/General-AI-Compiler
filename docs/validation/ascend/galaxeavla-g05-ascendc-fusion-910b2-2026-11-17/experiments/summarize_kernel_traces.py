"""Summarize independent diagnostic traces; durations are not model wall time."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def summarize(directory):
    paths = list(Path(directory).rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one kernel_details.csv under {directory}: {paths}")
    path = paths[0]
    data = path.read_bytes()
    groups = defaultdict(lambda: [0, Decimal(0)])
    cores = defaultdict(int)
    rows = 0
    for row in csv.DictReader(data.decode("utf-8-sig").splitlines()):
        duration = Decimal(row["Duration(us)"].strip())
        group = groups[row["Type"]]
        group[0] += 1
        group[1] += duration
        cores[row["Accelerator Core"]] += 1
        rows += 1
    return {
        "path": str(path), "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(), "rows": rows,
        "duration_us_sum": float(sum(item[1] for item in groups.values())),
        "by_core": dict(sorted(cores.items())),
        "by_type": {key: {"count": value[0], "duration_us": float(value[1])}
                    for key, value in sorted(groups.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="label=directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "scope": "One post-timing Level1 diagnostic request per trace. Summed kernel durations may overlap and are not wall time.",
        "traces": {label: summarize(directory) for label, directory in
                   (item.split("=", 1) for item in args.trace)},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: value[k] for k in ("rows", "duration_us_sum", "by_core")}
                      for key, value in result["traces"].items()}))
