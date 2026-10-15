from __future__ import annotations

from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.errors import ValidationError
from gac.frontends.bevformer_graph import load_bevformer_step_graph
from gac.weights import write_weights
from tests.test_bevformer_step_pipeline import small_config, make_weights


class BevFormerGraphFrontendTests(unittest.TestCase):
    def test_complete_graph_import_assembles_every_weight_group(self):
        config = small_config()
        encoder, decoder = make_weights(config)
        namespace = "gac.frontends.bevformer_graph."
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            source = write_weights(Path(directory) / "data.gacweights", {"fixture": np.zeros(1, np.float32)},
                                   source_checkpoint_sha256="a" * 64)
            mocks = {}
            for function, result in (
                ("validate_checkpoint_weights", encoder["backbone"]),
                ("extract_feature_prep_weights", encoder["feature_prep"]),
                ("extract_encoder_input_weights", encoder["encoder_inputs"]),
                ("extract_decoder_stack_weights", {key: value for key, value in decoder.items() if key != "detection"}),
                ("extract_detection_stack_weights", decoder["detection"]),
            ):
                mocks[function] = patches.enter_context(patch(namespace + function, return_value=result))
            layers = [dict(temporal=encoder["temporal"][i], spatial=encoder["spatial"][i], layer=encoder["layer"][i])
                      for i in range(config.encoder.encoder_stack.num_layers)]
            extraction = patches.enter_context(patch(namespace + "extract_encoder_layer_weights", side_effect=layers))
            graph = load_bevformer_step_graph(source, config, expected_checkpoint_sha256="A" * 64)
            graph.validate()
            self.assertGreater(len(graph.nodes), 1000)
            self.assertTrue({"Conv2D", "GridSample2D", "MSDeformableAttention", "LayerNorm", "Softmax", "MatMul"}
                            <= {node.op for node in graph.nodes})
            self.assertEqual(set(graph.inputs), {"images", "lidar2img", "can_bus_delta", "prev_bev", "has_prev_bev"})
            self.assertIn("next_bev", graph.outputs)
            self.assertTrue(graph.metadata["full_encoder_stack"])
            self.assertTrue(graph.metadata["full_decoder_stack"])
            self.assertTrue(graph.metadata["full_raw_detection_head"])
            self.assertEqual(graph.metadata["source_checkpoint_sha256"], "a" * 64)
            self.assertEqual(graph.metadata["source_weights_sha256"], sha256(source.read_bytes()).hexdigest())
            self.assertEqual(graph.metadata["source_weights_format"], "gacweights")
            self.assertEqual(graph.metadata["loaded_weight_count"], 1)
            self.assertEqual(graph.metadata["host_postprocess"], "decode_nms_free")
            self.assertEqual(extraction.call_count, config.encoder.encoder_stack.num_layers)
            self.assertEqual([call.args[1] for call in extraction.call_args_list], list(range(len(layers))))
            for mock in mocks.values():
                mock.assert_called_once()

    def test_provenance_mismatch_stops_before_weight_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_weights(Path(directory) / "data.gacweights", {"x": np.ones(1, np.float32)},
                                   source_checkpoint_sha256="a" * 64)
            with patch("gac.frontends.bevformer_graph._weight_groups") as groups:
                with self.assertRaisesRegex(ValidationError, "provenance mismatch"):
                    load_bevformer_step_graph(source, expected_checkpoint_sha256="b" * 64)
                groups.assert_not_called()

    def test_invalid_digest_and_config_are_rejected_before_io(self):
        with self.assertRaisesRegex(ValidationError, "64 hexadecimal"):
            load_bevformer_step_graph("absent.gacweights", expected_checkpoint_sha256="no")
        with self.assertRaisesRegex(ValidationError, "BevFormerStepPipelineConfig"):
            load_bevformer_step_graph("absent.gacweights", config={})

    def test_import_does_not_load_cuda_or_frameworks(self):
        source = ("import sys; import gac.frontends.bevformer_graph; "
                  "forbidden={'gac.backends.cuda','gac.compiler','gac.package','torch','mmcv'}; "
                  "assert not forbidden.intersection(sys.modules), forbidden.intersection(sys.modules)")
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
