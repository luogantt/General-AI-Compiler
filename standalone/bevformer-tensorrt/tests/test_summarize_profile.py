"""CPU checks for conservative attribution and lossless callback accounting."""

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("summarize_profile", Path(__file__).resolve().parents[1] / "scripts" / "summarize_profile.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def callback(ordinal, name, ms=1.0):
    return {"ordinal": ordinal, "name": name, "count": 10, "mean_ms": ms}


class SummarizeProfileTests(unittest.TestCase):
    def test_ambiguous_or_absent_names_are_unknown_even_if_name_looks_like_conv(self):
        rows = [callback(7, "Conv"), callback(2, "same"), callback(3, "same"), callback(4, "duplicate")]
        inspector = {"Layers": [{"Name": "same", "LayerType": "Convolution"},
                                {"Name": "duplicate", "LayerType": "Convolution"},
                                {"Name": "duplicate", "LayerType": "Convolution"}]}
        result = MODULE.summarize({}, rows, inspector)
        self.assertEqual(result["groups"], [{"category": "unknown", "callback_count": 4, "sum_mean_ms": 4.0, "callback_share_percent": 100.0}])
        self.assertEqual(result["match_counts"], {"duplicate_callback_name": 2, "duplicate_inspector_name": 1, "unmatched": 1})
        self.assertEqual([row["ordinal"] for row in result["callbacks"]], [2, 3, 4, 7])

    def test_fused_metadata_is_not_split_or_double_counted(self):
        inspector = {"Layers": [{"Name": "fused", "LayerType": "kgen", "Metadata": "Conv Add Relu"},
                                {"Name": "sample", "LayerType": "PluginV3", "PluginType": "GACMSDeformAttn"}]}
        result = MODULE.summarize({}, [callback(8, "fused", 3), callback(1, "sample", 2)], inspector)
        self.assertEqual(result["callback_count"], 2)
        self.assertEqual(result["sum_mean_ms"], 5)
        self.assertEqual([(g["category"], g["callback_share_percent"]) for g in result["groups"]], [("kgen", 60), ("PluginV3/GACMSDeformAttn", 40)])
        self.assertEqual([row["ordinal"] for row in result["top20"]], [8, 1])

    def test_invalid_latency_or_duplicate_ordinal_is_rejected(self):
        for value in (-1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.summarize({}, [callback(0, "x", value)], {"Layers": []})
        with self.assertRaises(ValueError):
            MODULE.summarize({}, [callback(0, "x"), callback(0, "y")], {"Layers": []})
        with self.assertRaises(ValueError):
            MODULE.summarize({"baseline": {"cuda_event": {"samples_ms": [1, -2]}}}, [], {"Layers": []})

    def test_cli_writes_traceable_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "profile"
            source.mkdir()
            payloads = {"report.json": {"engine_sha256": "fixture"},
                        "layers.json": [callback(17, "operator", 2.5)],
                        "engine.layers.json": {"Layers": [{"Name": "operator", "LayerType": "kgen"}]}}
            for name, payload in payloads.items():
                (source / name).write_text(json.dumps(payload), encoding="utf-8")
            destination = root / "summary"
            args = ["--profile-dir", str(source), "--output-dir", str(destination)]
            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main(args)
            original = (destination / "summary.json").read_bytes()
            result = json.loads(original)
            self.assertEqual(result["top20"][0]["ordinal"], 17)
            self.assertEqual(len(result["sources"]), 3)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in result["sources"]))
            self.assertEqual({path.name for path in destination.iterdir()}, {"summary.json", "top.csv", "README.md"})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                MODULE.main(args)
            self.assertEqual((destination / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
