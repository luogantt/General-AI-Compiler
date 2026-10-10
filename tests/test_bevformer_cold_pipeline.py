from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_cold_pipeline import (
    BevFormerColdPipelineConfig,
    build_bevformer_cold_pipeline_graph,
    cold_pipeline_weight_shapes,
    validate_cold_pipeline_inputs,
)
from gac.bevformer_decoder_pipeline import build_bevformer_decoder_pipeline_graph
from gac.bevformer_encoder_pipeline import build_bevformer_cold_encoder_pipeline_graph
from gac.errors import ValidationError
from gac.runtime import reference_run
from tests.test_bevformer_decoder_pipeline import (
    make_weights as make_decoder_weights,
    small_config as small_decoder_config,
)
from tests.test_bevformer_encoder_pipeline import (
    random_weights as make_encoder_weight_tuple,
    small_config as small_encoder_config,
)


def small_config() -> BevFormerColdPipelineConfig:
    decoder = small_decoder_config()
    matched_cross = replace(
        decoder.decoder.layer.cross_attention, bev_h=1, bev_w=2
    )
    matched_layer = replace(
        decoder.decoder.layer, cross_attention=matched_cross
    )
    decoder = replace(
        decoder, decoder=replace(decoder.decoder, layer=matched_layer)
    )
    return BevFormerColdPipelineConfig(
        encoder=small_encoder_config(), decoder=decoder
    )


def make_weights(config: BevFormerColdPipelineConfig):
    encoder_tuple = make_encoder_weight_tuple(config.encoder)
    encoder = dict(
        zip(
            ("backbone", "feature_prep", "encoder_inputs", "temporal", "spatial", "layer"),
            encoder_tuple,
            strict=True,
        )
    )
    return encoder, make_decoder_weights(config.decoder)


class BevFormerColdPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.encoder_weights, self.decoder_weights = make_weights(self.config)
        self.graph = build_bevformer_cold_pipeline_graph(
            self.encoder_weights, self.decoder_weights, self.config
        )
        self.inputs = {
            "images": np.ascontiguousarray(
                np.linspace(
                    -0.2,
                    0.3,
                    np.prod(self.config.encoder.backbone.input_shape),
                    dtype=np.float32,
                ).reshape(
                    self.config.encoder.feature_prep.batch,
                    self.config.encoder.feature_prep.cameras,
                    self.config.encoder.backbone.input_channels,
                    *self.config.encoder.backbone.input_hw,
                )
            ),
            "lidar2img": np.ascontiguousarray(
                np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
            ),
            "can_bus_delta": np.asarray([[0.1, -0.2, 0.3, 0.4]], dtype=np.float32),
        }

    def test_full_cold_graph_matches_encoder_then_decoder_oracle(self):
        actual = reference_run(self.graph, self.inputs)
        encoder = build_bevformer_cold_encoder_pipeline_graph(
            self.encoder_weights["backbone"],
            self.encoder_weights["feature_prep"],
            self.encoder_weights["encoder_inputs"],
            self.encoder_weights["temporal"],
            self.encoder_weights["spatial"],
            self.encoder_weights["layer"],
            self.config.encoder,
        )
        encoded = reference_run(encoder, self.inputs)
        decoder = build_bevformer_decoder_pipeline_graph(
            self.decoder_weights["query_init"],
            self.decoder_weights["self_attention"],
            self.decoder_weights["cross_attention"],
            self.decoder_weights["layer"],
            self.decoder_weights["refinement"],
            self.decoder_weights["detection"],
            self.config.decoder,
        )
        projected = reference_run(decoder, {"value": encoded["next_bev"]})
        np.testing.assert_allclose(actual["next_bev"], encoded["next_bev"], rtol=2e-5, atol=2e-5)
        for name in (
            "all_class_logits",
            "all_class_scores",
            "all_boxes",
            "final_class_logits",
            "final_class_scores",
            "final_boxes",
            "initial_reference_points",
            "intermediate_reference_points",
        ):
            np.testing.assert_allclose(actual[name], projected[name], rtol=3e-5, atol=3e-5)
        self.assertTrue(self.graph.metadata["cold_single_frame_core_connected"])
        self.assertTrue(self.graph.metadata["not_a_streaming_step"])

    def test_small_full_core_compiles_for_sm86(self):
        program = compile_cuda(self.graph)
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertGreater(len(program.launches), 200)
        self.assertEqual(
            program.reference_graph.value_specs["final_class_logits"].shape,
            (1, 3, 2),
        )

    def test_inventory_and_group_contracts(self):
        inventory = cold_pipeline_weight_shapes(self.config)
        self.assertEqual(set(inventory), {"encoder", "decoder"})
        bad_encoder = dict(self.encoder_weights)
        bad_encoder.pop("backbone")
        with self.assertRaisesRegex(ValidationError, "weight groups"):
            build_bevformer_cold_pipeline_graph(
                bad_encoder, self.decoder_weights, self.config
            )
        with self.assertRaises(ValidationError):
            cold_pipeline_weight_shapes(object())  # type: ignore[arg-type]

    def test_runtime_input_contract_accepts_exact_abi(self):
        validate_cold_pipeline_inputs(self.graph, self.inputs)

    def test_runtime_input_contract_rejects_name_mismatches(self):
        missing = dict(self.inputs)
        missing.pop("can_bus_delta")
        with self.assertRaisesRegex(ValidationError, "input names"):
            validate_cold_pipeline_inputs(self.graph, missing)

        extra = dict(self.inputs)
        extra["unexpected"] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "input names"):
            validate_cold_pipeline_inputs(self.graph, extra)

        invalid_name = dict(self.inputs)
        invalid_name[1] = invalid_name.pop("can_bus_delta")  # type: ignore[index]
        with self.assertRaisesRegex(ValidationError, "non-empty strings"):
            validate_cold_pipeline_inputs(self.graph, invalid_name)  # type: ignore[arg-type]

    def test_runtime_input_contract_rejects_shape_and_dtype(self):
        bad_shape = dict(self.inputs)
        bad_shape["can_bus_delta"] = np.zeros((4,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "requires"):
            validate_cold_pipeline_inputs(self.graph, bad_shape)

        bad_dtype = dict(self.inputs)
        bad_dtype["lidar2img"] = self.inputs["lidar2img"].astype(np.float64)
        with self.assertRaisesRegex(ValidationError, "float32"):
            validate_cold_pipeline_inputs(self.graph, bad_dtype)

    def test_runtime_input_contract_rejects_non_array_and_non_contiguous(self):
        non_array = dict(self.inputs)
        non_array["can_bus_delta"] = [[0.1, -0.2, 0.3, 0.4]]  # type: ignore[assignment]
        with self.assertRaisesRegex(ValidationError, "NumPy ndarray"):
            validate_cold_pipeline_inputs(self.graph, non_array)  # type: ignore[arg-type]

        non_contiguous = dict(self.inputs)
        non_contiguous["images"] = np.asfortranarray(self.inputs["images"])
        self.assertFalse(non_contiguous["images"].flags.c_contiguous)
        with self.assertRaisesRegex(ValidationError, "C-contiguous"):
            validate_cold_pipeline_inputs(self.graph, non_contiguous)

    def test_runtime_input_contract_rejects_non_finite_values(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                values = dict(self.inputs)
                values["lidar2img"] = self.inputs["lidar2img"].copy()
                values["lidar2img"][0, 0, 0, 0] = invalid
                with self.assertRaisesRegex(ValidationError, "finite"):
                    validate_cold_pipeline_inputs(self.graph, values)


if __name__ == "__main__":
    unittest.main()
