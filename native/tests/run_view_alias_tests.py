#!/usr/bin/env python3
"""Execute generated native code and check view-copy reduction and live storage.

ONNX Runtime is used only by the separate fixture generator. This runner calls
the compiler and its emitted machine code; it implements no tensor operators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from run_base_ops_tests import sha256
from run_full_model import load_plan
from run_tests import execute, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--llvm-bin', required=True, type=Path)
    parser.add_argument('--fixtures', required=True, type=Path)
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    spec = json.loads((args.fixtures / 'cases.json').read_text())
    report = {'scope': 'Independent view/lifetime fixtures executed as generated native CPU machine code',
              'compiler_sha256_at_start': sha256(args.compiler), 'reference': spec['reference'],
              'reference_versions': spec['versions'], 'python_operator_fallback': False, 'cases': []}
    copy_ops = {'gac.identity', 'gac.reshape', 'gac.squeeze', 'gac.unsqueeze'}
    for case in spec['cases']:
        folder = args.fixtures / case['name']
        entry = {'name': case['name'], 'passed': False, 'onnx_view_nodes': case['onnx_view_nodes'],
                 'model_sha256': sha256(folder / 'model.onnx')}
        try:
            manifest = folder / 'plan.json'
            constants = folder / 'constants'
            run([args.compiler, folder / 'model.onnx', '--target=native', '--emit=llvm',
                 '-o', folder / 'llvm.mlir', '--manifest', manifest, '--constants-dir', constants], timeout=240)
            plan = load_plan(manifest)
            entry['live_buffer_overlap_check'] = 'passed'
            entry['arena_bytes'] = plan['arena_bytes']
            if 'view_elision' in plan:
                entry['compiler_view_elision_metadata'] = plan['view_elision']
            counts = Counter(dispatch['op'] for dispatch in plan['dispatches'])
            entry['dispatch_operation_counts'] = dict(counts)
            copies = sum(counts[op] for op in copy_ops)
            entry['view_copy_dispatches'] = copies
            assert copies <= case['max_view_copy_dispatches'], f'Unnecessary view copies: {counts}'
            if case['maximum_dispatches'] is not None:
                assert len(plan['dispatches']) <= case['maximum_dispatches'], counts
            for op, count in case['required_ops'].items():
                assert counts[op] >= count, f'Required physical operation lost: {op}: {counts}'
            outputs = {tensor['name']: tensor['shape'] for tensor in plan['tensors']
                       if tensor['name'] in plan['outputs']}
            assert outputs == case['output_shapes'], f'Public output names/shapes changed: {outputs}'
            if case['constant_root_shape']:
                assert any(t['storage'] == 'constant' and t['shape'] == case['constant_root_shape']
                           for t in plan['tensors']), 'Expected original constant root was replaced'
            run([args.llvm_bin / 'mlir-translate', '--mlir-to-llvmir', folder / 'llvm.mlir',
                 '-o', folder / 'kernel.ll'], timeout=240)
            run([args.llvm_bin / 'opt', '-passes=verify', '-disable-output', folder / 'kernel.ll'])
            library = folder / 'kernels.so'
            run([args.llvm_bin / 'clang', '-shared', '-fPIC', '-O2', folder / 'kernel.ll',
                 '-lm', '-o', library], timeout=240)
            entry.update(execute(manifest, constants, library, folder))
            entry['library_sha256'] = sha256(library)
            entry['passed'] = True
        except (AssertionError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            entry['failure'] = str(error)
        report['cases'].append(entry)
        (args.fixtures / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {case['name']}", flush=True)
        if not entry['passed']:
            print(entry['failure'], flush=True)
    report['compiler_sha256_at_end'] = sha256(args.compiler)
    report['compiler_stable_during_run'] = report['compiler_sha256_at_start'] == report['compiler_sha256_at_end']
    report['passed'] = report['compiler_stable_during_run'] and all(case['passed'] for case in report['cases'])
    (args.fixtures / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{sum(case['passed'] for case in report['cases'])}/{len(report['cases'])} passed")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
