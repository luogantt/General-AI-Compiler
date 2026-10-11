"""Security and path-boundary tests for the UniAD checkpoint exporter."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from gac.tools import uniad_export_ffn as exporter
from gac.uniad import encoder_ffn_checkpoint_keys


class _Tensor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class ExportPathTests(unittest.TestCase):
    def test_output_must_not_alias_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            with self.assertRaisesRegex(ValueError, "checkpoint and output"):
                exporter._resolve_export_paths(checkpoint, checkpoint)

    def test_provenance_must_not_alias_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "layer0.uniadffn"
            checkpoint = Path(str(output) + ".provenance.json")
            with self.assertRaisesRegex(ValueError, "checkpoint and provenance"):
                exporter._resolve_export_paths(checkpoint, output)


class CheckpointLoadingTests(unittest.TestCase):
    def test_legacy_pickle_loader_requires_explicit_trust(self):
        class LegacyTorch:
            def __init__(self):
                self.calls = []

            def load(self, path, map_location):
                self.calls.append((path, map_location))
                return {"state_dict": {}}

        torch = LegacyTorch()
        with self.assertRaisesRegex(RuntimeError, "--trusted-checkpoint"):
            exporter._torch_load(
                torch, Path("model.pth"), trusted_checkpoint=False
            )
        self.assertEqual(torch.calls, [])
        result = exporter._torch_load(
            torch, Path("model.pth"), trusted_checkpoint=True
        )
        self.assertEqual(result, {"state_dict": {}})
        self.assertEqual(torch.calls, [("model.pth", "cpu")])

    def test_supported_torch_always_uses_weights_only(self):
        class SafeTorch:
            def __init__(self):
                self.calls = []

            def load(self, path, map_location, weights_only=False):
                self.calls.append((path, map_location, weights_only))
                return {"state_dict": {}}

        torch = SafeTorch()
        exporter._torch_load(torch, Path("model.pth"), trusted_checkpoint=False)
        self.assertEqual(torch.calls, [("model.pth", "cpu", True)])

    def test_parser_exposes_trusted_checkpoint_flag(self):
        args = exporter.build_parser().parse_args(
            ["model.pth", "layer0.uniadffn", "--trusted-checkpoint"]
        )
        self.assertTrue(args.trusted_checkpoint)


class StateDictNormalizationTests(unittest.TestCase):
    def test_module_prefix_collision_is_rejected(self):
        key = encoder_ffn_checkpoint_keys(0)["fc1_bias"]
        tensor = _Tensor(np.zeros((512,), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "normalize to duplicate key"):
            exporter._selected_numpy_state_dict(
                {key: tensor, "module." + key: tensor}, layer=0
            )


if __name__ == "__main__":
    unittest.main()
