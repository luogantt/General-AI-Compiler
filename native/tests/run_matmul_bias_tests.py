#!/usr/bin/env python3
"""Validate emitted MatMul/bias machine code, fusion legality and memory plans."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
from run_base_ops_tests import sha256
from run_full_model import execute_host, load_plan, read_inputs
from run_tests import run


def check(actual, expected, exact):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(expected)
    if exact:
        np.testing.assert_array_equal(actual[finite], expected[finite])
    else:
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=2e-5, atol=2e-5)
    error = np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))
    return {'passed': True, 'max_finite_absolute_error': float(error.max(initial=0)),
            'nan_elements': int(np.isnan(actual).sum()), 'positive_infinities': int(np.isposinf(actual).sum()),
            'negative_infinities': int(np.isneginf(actual).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--llvm-bin', required=True, type=Path)
    parser.add_argument('--fixtures', required=True, type=Path)
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    spec = json.loads((args.fixtures / 'cases.json').read_text())
    report = {'scope': 'Independent MatMul+bias references versus generated native CPU machine code',
              'reference': spec['reference'], 'reference_versions': spec['versions'],
              'compiler_sha256_at_start': sha256(args.compiler), 'python_operator_fallback': False, 'cases': []}
    for case in spec['cases']:
        folder = args.fixtures / case['name']
        entry = {'name': case['name'], 'passed': False, 'reference': case['reference'],
                 'model_sha256': sha256(folder / 'model.onnx')}
        manifest, constants, library = folder / 'plan.json', folder / 'constants', folder / 'kernels.so'
        command = [args.compiler, folder / 'model.onnx', '--target=native', '--emit=llvm',
                   '-o', folder / 'llvm.mlir', '--manifest', manifest, '--constants-dir', constants]
        try:
            if not case['valid']:
                rejected = run(command, succeeds=False)
                assert re.search('broadcast|incompatible', rejected.stderr, re.IGNORECASE), rejected.stderr
                entry['expected_rejection'] = rejected.stderr.strip()
            else:
                run(command)
                plan = load_plan(manifest)
                entry['live_buffer_overlap_check'] = 'passed'
                entry['arena_bytes'] = plan['arena_bytes']
                counts = Counter(dispatch['op'] for dispatch in plan['dispatches'])
                entry['dispatch_operation_counts'] = dict(counts)
                fused = case['expected_fusion']
                if fused:
                    assert counts[fused] == 1 and len(plan['dispatches']) == 1, counts
                    assert sum(t['storage'] == 'arena' for t in plan['tensors']) == 1
                else:
                    assert not counts['gac.matmul_bias'] and not counts['gac.matmul_bias_relu'], counts
                    assert counts['gac.matmul'] == 1 and counts['gac.add'] == 1, counts
                outputs = {item['name']: item['shape'] for item in plan['tensors'] if item['name'] in plan['outputs']}
                assert outputs == case['output_shapes'], outputs
                run([args.llvm_bin / 'mlir-translate', '--mlir-to-llvmir', folder / 'llvm.mlir', '-o', folder / 'kernel.ll'])
                run([args.llvm_bin / 'opt', '-passes=verify', '-disable-output', folder / 'kernel.ll'])
                run([args.llvm_bin / 'clang', '-shared', '-fPIC', '-O2', folder / 'kernel.ll', '-lm', '-o', library])
                invocation = SimpleNamespace(manifest=manifest, constants_dir=constants, library=library,
                    inputs=folder / 'inputs.npz', output_dir=folder, resume=False, dump_intermediates=None,
                    intermediate_reference=None, progress_every=1000, checkpoint_every=0, stop_after=0,
                    rtol=2e-5, atol=2e-5)
                execution_report = {}
                actual = execute_host(invocation, plan, read_inputs(invocation.inputs, plan), execution_report)
                with np.load(folder / 'expected.npz', allow_pickle=False) as expected:
                    assert set(actual) == set(expected.files)
                    entry['outputs'] = {name: check(value, expected[name], case['exact_finite_comparison'])
                                        for name, value in actual.items()}
                entry['artifact_identity'] = execution_report['artifact_identity']
            entry['passed'] = True
        except (AssertionError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            entry['failure'] = str(error)
        report['cases'].append(entry)
        (args.fixtures / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(f"{'PASS' if entry['passed'] else 'FAIL'} {case['name']}", flush=True)
        if not entry['passed']:
            print(entry['failure'], flush=True)
    report['compiler_sha256_at_end'] = sha256(args.compiler)
    report['compiler_stable_during_run'] = report['compiler_sha256_at_start'] == report['compiler_sha256_at_end']
    report['passed'] = report['compiler_stable_during_run'] and all(case['passed'] for case in report['cases'])
    (args.fixtures / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{sum(case['passed'] for case in report['cases'])}/{len(report['cases'])} passed")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
