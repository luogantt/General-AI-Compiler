from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.artifacts.weights import write_weights
from gac.core.errors import ValidationError
from gac.models.galaxeavla.contract import GalaxeaQwen35MLPContract
from gac.tools import export_galaxeavla_mlp_onnx as exporter


_HAS_ONNX = importlib.util.find_spec("onnx") is not None


def _weights(contract: GalaxeaQwen35MLPContract) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(731)
    return {
        name: rng.normal(0.0, 0.15, shape).astype(np.float32)
        for name, shape in contract.source_weight_shapes.items()
    }


class GalaxeaONNXExportTests(unittest.TestCase):
    def test_tool_module_does_not_eagerly_import_onnx(self):
        environment = os.environ.copy()
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from gac.tools import export_galaxeavla_mlp_onnx as tool; "
                    "tool.build_parser(); "
                    "assert 'onnx' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        parser = exporter.build_parser()
        self.assertIn("expected-checkpoint-sha256", parser.format_help())

    def test_missing_optional_dependency_has_clear_error_and_publishes_nothing(self):
        contract = GalaxeaQwen35MLPContract(
            batch=1, tokens=2, hidden_size=3, intermediate_size=5
        )
        with tempfile.TemporaryDirectory() as directory:
            source = write_weights(
                Path(directory) / "slice.gacweights",
                _weights(contract),
                source_checkpoint_sha256="a" * 64,
            )
            destination = Path(directory) / "slice.onnx"
            original_import = exporter.importlib.import_module

            def reject_onnx(name, *args, **kwargs):
                if name == "onnx":
                    raise ModuleNotFoundError("test fixture: no onnx")
                return original_import(name, *args, **kwargs)

            with patch.object(exporter.importlib, "import_module", side_effect=reject_onnx):
                with self.assertRaisesRegex(ValidationError, "optional 'onnx'.*install"):
                    exporter.export_galaxeavla_mlp_onnx(
                        source,
                        destination,
                        expected_checkpoint_sha256="A" * 64,
                        batch=1,
                        tokens=2,
                        hidden_size=3,
                        intermediate_size=5,
                    )
            self.assertFalse(destination.exists())

    def test_checkpoint_attestation_is_required_before_optional_onnx_import(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing-is-rejected-before-source-read.gacweights"
            destination = Path(directory) / "slice.onnx"
            with patch.object(exporter, "_load_onnx") as load_onnx:
                with self.assertRaisesRegex(
                    ValidationError, "expected_checkpoint_sha256 is required"
                ):
                    exporter.export_galaxeavla_mlp_onnx(source, destination)
            load_onnx.assert_not_called()
            self.assertFalse(destination.exists())

    def test_checkpoint_attestation_mismatch_is_rejected_before_onnx_import(self):
        contract = GalaxeaQwen35MLPContract(
            batch=1, tokens=2, hidden_size=3, intermediate_size=5
        )
        with tempfile.TemporaryDirectory() as directory:
            source = write_weights(
                Path(directory) / "slice.gacweights",
                _weights(contract),
                source_checkpoint_sha256="b" * 64,
            )
            destination = Path(directory) / "slice.onnx"
            with patch.object(exporter, "_load_onnx") as load_onnx:
                with self.assertRaisesRegex(ValidationError, "expected digest"):
                    exporter.export_galaxeavla_mlp_onnx(
                        source,
                        destination,
                        expected_checkpoint_sha256="c" * 64,
                        batch=1,
                        tokens=2,
                        hidden_size=3,
                        intermediate_size=5,
                    )
            load_onnx.assert_not_called()
            self.assertFalse(destination.exists())

    @unittest.skipUnless(_HAS_ONNX, "optional onnx dependency is not installed")
    def test_onnx_is_static_canonical_atomic_and_matches_host_reference(self):
        import onnx
        from onnx import numpy_helper
        from onnx.reference import ReferenceEvaluator

        contract = GalaxeaQwen35MLPContract(
            batch=2, tokens=3, hidden_size=4, intermediate_size=7
        )
        weights = _weights(contract)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_weights(
                root / "slice.gacweights",
                weights,
                source_checkpoint_sha256="d" * 64,
            )
            destination = root / "nested" / "slice.onnx"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = exporter.main(
                    [
                        str(source),
                        "--output",
                        str(destination),
                        "--expected-checkpoint-sha256",
                        "D" * 64,
                        "--batch",
                        "2",
                        "--tokens",
                        "3",
                        "--hidden-size",
                        "4",
                        "--intermediate-size",
                        "7",
                    ]
                )
            self.assertEqual(result, 0)
            summary = json.loads(stdout.getvalue())
            payload = destination.read_bytes()
            self.assertEqual(summary["onnx_sha256"], sha256(payload).hexdigest())
            self.assertEqual(summary["input"]["shape"], [2, 3, 4])
            self.assertEqual(summary["output_tensor"]["shape"], [2, 3, 4])
            self.assertEqual(
                summary["source_weights_sha256"], sha256(source.read_bytes()).hexdigest()
            )
            self.assertEqual(
                summary["nodes"]["op_types"],
                ["MatMul", "Sigmoid", "Mul", "MatMul", "Mul", "MatMul"],
            )
            self.assertFalse(summary["full_model"])
            self.assertFalse(summary["official_g05_dimensions"])
            self.assertEqual(
                summary["upstream_source_revision"],
                "89f2322b4ad016e192437adc1a2c253b05bab246",
            )
            self.assertIn("LICENSE-G0.5", summary["upstream_license_url"])
            self.assertIn("non-commercial", summary["upstream_usage_notice"])
            self.assertEqual(
                summary["source_checkpoint_digest_kind"],
                "trusted-extractor-attestation-not-cryptographically-bound",
            )
            self.assertTrue(summary["expected_checkpoint_sha256_matched"])
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

            model = onnx.load(destination)
            onnx.checker.check_model(model)
            properties = {item.key: item.value for item in model.metadata_props}
            self.assertEqual(properties["gac.official_g05_dimensions"], "false")
            self.assertEqual(
                properties["gac.upstream_source_revision"],
                summary["upstream_source_revision"],
            )
            self.assertEqual(
                properties["gac.source_weights_sha256"],
                summary["source_weights_sha256"],
            )
            self.assertEqual(
                properties["gac.upstream_license_url"],
                summary["upstream_license_url"],
            )
            self.assertEqual(
                properties["gac.source_checkpoint_digest_kind"],
                summary["source_checkpoint_digest_kind"],
            )
            self.assertEqual(
                properties["gac.expected_checkpoint_sha256_matched"], "true"
            )
            self.assertEqual(
                [node.op_type for node in model.graph.node],
                ["MatMul", "Sigmoid", "Mul", "MatMul", "Mul", "MatMul"],
            )
            initializers = {
                value.name: numpy_helper.to_array(value)
                for value in model.graph.initializer
            }
            self.assertEqual(initializers["gate_proj_weight"].shape, (4, 7))
            self.assertEqual(initializers["up_proj_weight"].shape, (4, 7))
            self.assertEqual(initializers["down_proj_weight"].shape, (7, 4))
            np.testing.assert_array_equal(
                initializers["gate_proj_weight"],
                weights[next(iter(contract.source_weight_shapes))].T,
            )

            hidden = np.random.default_rng(912).normal(
                0.0, 0.2, contract.input_shape
            ).astype(np.float32)
            gate = hidden @ initializers["gate_proj_weight"]
            expected = (
                gate
                * (1.0 / (1.0 + np.exp(-gate)))
                * (hidden @ initializers["up_proj_weight"])
            ) @ initializers["down_proj_weight"]
            actual = ReferenceEvaluator(model).run(
                None, {"hidden_states": hidden}
            )[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
