"""MSDA search configuration contracts, without CUDA/TensorRT imports."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("msda_config_benchmark", ROOT / "scripts" / "benchmark_msda.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class MSDABenchmarkConfigTests(unittest.TestCase):
    def parse(self, *args):
        return BENCH.make_parser().parse_args(['--library', 'plugin.so', '--report', 'report.json', *args])

    def test_default_search_remains_48_and_quick_18(self):
        configs = BENCH.candidate_options(self.parse())
        self.assertEqual(len(configs), 48)
        self.assertEqual(len({BENCH.config_sha256(config) for config in configs}), 48)
        self.assertTrue(all(config['specialize'] in (0, 1) and config['prefetch'] in (0, 1)
                            for config in configs))
        self.assertEqual(len(BENCH.candidate_options(self.parse('--quick'))), 18)
        self.assertIsNone(self.parse().baseline_config)

    def test_custom_list_and_independent_baseline_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            candidates = [BENCH.options(4, 128, 3, 2), BENCH.options(2, 512, 2, 1)]
            base = BENCH.options(4, 128, 1, 0)
            (folder / 'list.json').write_text(json.dumps(candidates), encoding='utf-8')
            (folder / 'base.json').write_text(json.dumps(base), encoding='utf-8')
            args = self.parse('--configs', str(folder / 'list.json'),
                              '--baseline-config', str(folder / 'base.json'))
            self.assertEqual(BENCH.candidate_options(args), candidates)
            self.assertEqual(BENCH.read_options(args.baseline_config), base)
            one = self.parse('--config', str(folder / 'base.json'))
            self.assertEqual(BENCH.candidate_options(one), [base])

    def test_config_and_configs_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.parse('--config', 'one.json', '--configs', 'list.json')
        self.assertEqual(caught.exception.code, 2)

    def test_exact_integer_ranges_and_fields(self):
        for field, values in BENCH.OPTION_VALUES.items():
            for value in values:
                config = BENCH.options()
                config[field] = value
                self.assertEqual(BENCH.validate_options(config)[field], value)
            for invalid in (True, False, float(values[0]), str(values[0]), None, -1, 999):
                config = BENCH.options()
                config[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    BENCH.validate_options(config)
        for invalid in ([], None, {}, {**BENCH.options(), 'extra': 1}, {'channels_per_thread': 1}):
            with self.assertRaises(ValueError):
                BENCH.validate_options(invalid)

    def test_json_list_must_be_nonempty_and_every_entry_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'list.json'
            for invalid in ([], {}, [None], [BENCH.options(), {'specialize': 3}]):
                path.write_text(json.dumps(invalid), encoding='utf-8')
                with self.assertRaises(ValueError):
                    BENCH.read_options(path, multiple=True)
            path.write_text('{"channels_per_thread":1,"channels_per_thread":4, '
                            '"block_threads":128,"specialize":3,"prefetch":2}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON field'):
                BENCH.read_options(path)

    def test_options_hash_is_order_independent_and_sensitive_to_value(self):
        config = BENCH.options(4, 128, 1, 0)
        reordered = dict(reversed(list(config.items())))
        self.assertEqual(BENCH.config_sha256(config), BENCH.config_sha256(reordered))
        self.assertNotEqual(BENCH.config_sha256(config), BENCH.config_sha256({**config, 'prefetch': 2}))


if __name__ == '__main__':
    unittest.main()
