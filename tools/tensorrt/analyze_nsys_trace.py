#!/usr/bin/env python3
"""Export inference-frame kernel timings from an Nsight Systems SQLite trace.

The frame_N NVTX ranges must enclose a synchronized inference, as emitted by
profile_engine.py. Interval unions measure observed device activity, not SM
occupancy or a bandwidth/compute utilization counter. The source is read-only.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


def interval_union(intervals):
    total, finish = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('An activity ends before it starts')
        total += max(0, end - max(start, finish if finish is not None else start))
        finish = max(end, finish if finish is not None else end)
    return total


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(path, output_prefix):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as database:
        database.row_factory = sqlite3.Row
        names = dict(database.execute('SELECT id,value FROM StringIds'))
        ranges = database.execute('''SELECT n.start,n.end,coalesce(n.text,s.value) AS name
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
            WHERE n.end IS NOT NULL ORDER BY n.start''').fetchall()
        frames = [dict(row) for row in ranges if re.fullmatch(r'frame_\d+', row['name'] or '')]
        if not frames or len({x['name'] for x in frames}) != len(frames):
            raise ValueError('Expected uniquely named synchronized frame_N NVTX ranges')
        if any(a['end'] > b['start'] for a, b in zip(frames, frames[1:])):
            raise ValueError('Overlapping frame ranges cannot be assigned unambiguously')
        kernels = database.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        memory = []
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
            if table in tables:
                memory.extend(dict(row) for row in database.execute('SELECT start,end,bytes FROM ' + table))
        calls, groups, per_frame = [], defaultdict(list), []
        used = set()
        for frame in frames:
            selected = [(index, kernel) for index, kernel in enumerate(kernels)
                        if kernel['start'] >= frame['start'] and kernel['end'] <= frame['end']]
            gpu_intervals = [(row['start'], row['end']) for _, row in selected]
            mem = [row for row in memory if row['start'] >= frame['start'] and row['end'] <= frame['end']]
            all_intervals = gpu_intervals + [(row['start'], row['end']) for row in mem]
            for ordinal, (index, row) in enumerate(selected):
                if index in used:
                    raise ValueError('A kernel was assigned to more than one frame')
                used.add(index)
                duration = (row['end'] - row['start']) / 1e6
                name = names[row['demangledName']]
                groups[name].append(duration)
                calls.append({'frame': frame['name'], 'kernel_ordinal': ordinal,
                              'start_from_frame_ms': (row['start'] - frame['start']) / 1e6,
                              'duration_ms': duration, 'name': name,
                              'device_id': row['deviceId'], 'context_id': row['contextId'],
                              'stream_id': row['streamId'], 'correlation_id': row['correlationId'],
                              'grid': f"{row['gridX']}x{row['gridY']}x{row['gridZ']}",
                              'block': f"{row['blockX']}x{row['blockY']}x{row['blockZ']}",
                              'registers_per_thread': row['registersPerThread'],
                              'static_shared_bytes': row['staticSharedMemory'],
                              'dynamic_shared_bytes': row['dynamicSharedMemory']})
            wall = frame['end'] - frame['start']
            activity = interval_union(all_intervals)
            per_frame.append({'frame': frame['name'], 'host_range_ms': wall / 1e6,
                              'kernel_count': len(selected),
                              'kernel_sum_ms': sum(end-start for start, end in gpu_intervals) / 1e6,
                              'kernel_union_ms': interval_union(gpu_intervals) / 1e6,
                              'gpu_activity_union_ms': activity / 1e6,
                              'uncovered_range_ms': (wall-activity) / 1e6,
                              'memory_activity_count': len(mem),
                              'memory_activity_bytes': sum(row['bytes'] for row in mem)})
        if len(used) != len(kernels):
            raise ValueError(f'{len(kernels)-len(used)} kernels lie outside or cross the synchronized frame ranges')
    total = sum(row['duration_ms'] for row in calls)
    ranked = sorted([{'name': name, 'calls': len(values), 'total_ms': sum(values),
                      'per_frame_sum_ms': sum(values)/len(frames),
                      'mean_call_ms': statistics.mean(values),
                      'min_call_ms': min(values), 'max_call_ms': max(values),
                      'percent_of_kernel_duration_sum': 100*sum(values)/total}
                     for name, values in groups.items()], key=lambda row: -row['total_ms'])
    summary = {'sqlite': path.name, 'sqlite_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'frames': per_frame, 'frame_count': len(frames), 'kernel_count': len(calls),
               'distinct_kernel_names': len(ranked), 'top_kernels': ranked[:20],
               'mean_per_frame': {key: statistics.mean(row[key] for row in per_frame)
                                  for key in per_frame[0] if key != 'frame'},
               'notes': ['Nsight-instrumented inference; not the unprofiled latency baseline.',
                         'Kernel sum counts device execution durations; unions avoid counting overlapping intervals twice.',
                         'Host frame ranges include stream synchronization. Uncovered time is not necessarily recoverable overhead.',
                         'Memory activity bytes describe explicit CUDA copies/memsets, not kernel DRAM traffic.',
                         'No SM occupancy, DRAM bandwidth, or cache-miss counters were collected.']}
    write_csv(output_prefix.with_name(output_prefix.name + '-kernel-calls.csv'), calls)
    write_csv(output_prefix.with_name(output_prefix.name + '-kernel-ranked.csv'), ranked)
    write_csv(output_prefix.with_name(output_prefix.name + '-frames.csv'), per_frame)
    output_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    results = {mode: analyze(args.root / f'{mode}-nsys.sqlite', args.root / f'{mode}-nsys-summary')
               for mode in ('mixed', 'fp32')}
    (args.root / 'nsys-summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({mode: result['mean_per_frame'] for mode, result in results.items()}, indent=2))
