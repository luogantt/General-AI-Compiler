#!/usr/bin/env python3
"""Native numerical regressions plus proof that the unused fill buffer is gone.

Accepts run_tests.py arguments and records the compiler binary before and after
the validation window. Reference generation is separate from execution.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    args, _ = parser.parse_known_args()
    directory = args.fixtures.resolve()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'compiler_sha256_before': digest(args.compiler),
              'command': [sys.executable, *sys.argv], 'python_operator_fallback': False}
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_tests.py')),
                             *sys.argv[1:]], check=False)
    report['compiler_sha256_after'] = digest(args.compiler)
    report['compiler_unchanged'] = report['compiler_sha256_before'] == report['compiler_sha256_after']
    report['regression_exit_code'] = result.returncode
    (directory / 'compiler-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    assert report['compiler_unchanged'], 'Compiler changed during validation'
    result.check_returncode()
    extra = []
    cases = json.loads((directory / 'cases.json').read_text())['cases']
    for case in cases:
        if 'expect_no_constant_shape' not in case:
            continue
        plan = json.loads((directory / case['name'] / 'plan.json').read_text())
        assert all(not (item['storage'] == 'constant' and
                        item['shape'] == case['expect_no_constant_shape'])
                   for item in plan['tensors']), 'Dead zero data constant is still in the plan'
        assert all(item.get('op') != 'ConstantOfShape' and
                   item.get('operation') != 'gac.constant_of_shape'
                   for item in plan['dispatches'])
        # Exactly one real Expand remains; the input fill and all scatter
        # dispatches must disappear, rather than merely become unnamed copies.
        assert len(plan['dispatches']) == 1, plan['dispatches']
        extra.append({'name': case['name'], 'dead_zero_data_buffer_absent': True,
                      'remaining_dispatches': len(plan['dispatches'])})
    (directory / 'buffer-elimination-results.json').write_text(json.dumps({'cases': extra}, indent=2)+'\n')
    print('Verified dead fill buffer elimination:', extra)


if __name__ == '__main__':
    main()
