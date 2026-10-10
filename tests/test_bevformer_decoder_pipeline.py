"""Tests for the integrated decoder and raw detection pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder import DecoderCrossAttentionConfig
from gac.bevformer_decoder_layer import BevFormerDecoderLayerConfig
from gac.bevformer_decoder_pipeline import (
    BevFormerDecoderPipelineConfig,
    build_bevformer_decoder_pipeline_graph,
    decoder_pipeline_weight_shapes,
    validate_decoder_pipeline_inputs,
)
from gac.bevformer_decoder_self import DecoderSelfAttentionConfig
from gac.bevformer_decoder_stack import (
    BevFormerDecoderStackConfig,
    build_bevformer_decoder_stack_graph,
)
from gac.bevformer_detection_head import DetectionHeadConfig
from gac.bevformer_detection_stack import (
    DetectionStackConfig,
    build_detection_stack_graph,
)
from gac.bevformer_ffn import BEVFormerFFNConfig
from gac.bevformer_query_init import DecoderQueryInitConfig
from gac.bevformer_refinement import DecoderReferenceRefinementConfig
from gac.errors import ValidationError
from gac.runtime import reference_run


def small_config() -> BevFormerDecoderPipelineConfig:
    decoder = BevFormerDecoderStackConfig(
        query_init=DecoderQueryInitConfig(queries=3, embed_dim=4),
        layer=BevFormerDecoderLayerConfig(
            self_attention=DecoderSelfAttentionConfig(
                queries=3, embed_dim=4, heads=2
            ),
            cross_attention=DecoderCrossAttentionConfig(
                queries=3,
                bev_h=2,
                bev_w=2,
                embed_dim=4,
                heads=2,
                sampling_points=2,
            ),
            ffn=BEVFormerFFNConfig(bev_h=1, bev_w=3, embed_dim=4),
        ),
        refinement=DecoderReferenceRefinementConfig(
            queries=3, embed_dim=4, code_size=6
        ),
    )
    detection = DetectionStackConfig(
        head=DetectionHeadConfig(
            queries=3,
            embed_dim=4,
            classes=2,
            code_size=6,
            point_cloud_range=(-2.0, -3.0, -1.0, 2.0, 5.0, 3.0),
        )
    )
    return BevFormerDecoderPipelineConfig(decoder=decoder, detection=detection)


def _make_group(shapes, rng, *, zeros):
    result = {}
    for name, shape in shapes.items():
        if zeros:
            value = np.zeros(shape, dtype=np.float32)
            if name.endswith("norm_weight"):
                value.fill(1.0)
        elif name.endswith("norm_weight"):
            value = rng.uniform(0.8, 1.2, shape).astype(np.float32)
        elif name.endswith("norm_bias"):
            value = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
        else:
            value = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        result[name] = np.ascontiguousarray(value)
    return result


def make_weights(config, seed=8630, *, zeros=False):
    rng = np.random.default_rng(seed)
    inventory = decoder_pipeline_weight_shapes(config)
    decoder = inventory["decoder"]
    return {
        "query_init": _make_group(decoder["query_init"], rng, zeros=zeros),
        "self_attention": tuple(
            _make_group(shapes, rng, zeros=zeros)
            for shapes in decoder["self_attention"]
        ),
        "cross_attention": tuple(
            _make_group(shapes, rng, zeros=zeros)
            for shapes in decoder["cross_attention"]
        ),
        "layer": tuple(
            _make_group(shapes, rng, zeros=zeros)
            for shapes in decoder["layer"]
        ),
        "refinement": tuple(
            _make_group(shapes, rng, zeros=zeros)
            for shapes in decoder["refinement"]
        ),
        "detection": tuple(
            _make_group(shapes, rng, zeros=zeros)
            for shapes in inventory["detection"]
        ),
    }


def build_pipeline(config, weights):
    return build_bevformer_decoder_pipeline_graph(
        weights["query_init"],
        weights["self_attention"],
        weights["cross_attention"],
        weights["layer"],
        weights["refinement"],
        weights["detection"],
        config,
    )


class DecoderPipelineTests(unittest.TestCase):
    def test_small_pipeline_matches_two_stage_stack_oracle(self):
        config = small_config()
        weights = make_weights(config)
        graph = build_pipeline(config, weights)
        rng = np.random.default_rng(3070)
        inputs = {
            "value": np.ascontiguousarray(
                rng.standard_normal((1, 4, 4), dtype=np.float32)
            )
        }
        validate_decoder_pipeline_inputs(graph, inputs)
        actual = reference_run(graph, inputs)

        decoder_graph = build_bevformer_decoder_stack_graph(
            weights["query_init"],
            weights["self_attention"],
            weights["cross_attention"],
            weights["layer"],
            weights["refinement"],
            config.decoder,
        )
        decoded = reference_run(decoder_graph, inputs)
        detection_graph = build_detection_stack_graph(
            weights["detection"], config.detection
        )
        projected = reference_run(
            detection_graph,
            {
                "intermediate_states": decoded["intermediate_states"],
                "initial_reference_points": decoded[
                    "initial_reference_points"
                ],
                "intermediate_reference_points": decoded[
                    "intermediate_reference_points"
                ],
                "regressions": decoded["regressions"],
            },
        )

        for name in (
            "all_class_logits",
            "all_class_scores",
            "all_boxes",
            "final_class_logits",
            "final_class_scores",
            "final_boxes",
        ):
            np.testing.assert_allclose(
                actual[name], projected[name], rtol=1e-6, atol=1e-6
            )
        np.testing.assert_allclose(
            actual["initial_reference_points"],
            decoded["initial_reference_points"],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual["intermediate_reference_points"],
            decoded["intermediate_reference_points"],
            rtol=1e-6,
            atol=1e-6,
        )

        self.assertEqual(set(graph.inputs), {"value"})
        self.assertTrue(graph.metadata["full_decoder_stack"])
        self.assertTrue(graph.metadata["full_raw_detection_head"])
        self.assertTrue(graph.metadata["decoder_to_raw_head_connected"])
        self.assertEqual(graph.metadata["topk"], {"status": "outside-this-graph", "k": 300})
        self.assertEqual(graph.value_specs["all_boxes"].shape, (6, 1, 3, 6))
        self.assertEqual(
            graph.value_specs["intermediate_reference_points"].shape,
            (6, 1, 3, 3),
        )

    def test_default_profile_compiles_offline_for_sm86(self):
        config = BevFormerDecoderPipelineConfig()
        self.assertEqual(
            (
                config.batch,
                config.queries,
                config.embed_dim,
                config.classes,
                config.code_size,
            ),
            (1, 900, 256, 10, 10),
        )
        graph = build_pipeline(config, make_weights(config, zeros=True))
        program = compile_cuda(graph)
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertEqual(
            program.reference_graph.inputs["value"].shape, (1, 2500, 256)
        )
        self.assertEqual(
            program.reference_graph.value_specs["all_class_scores"].shape,
            (6, 1, 900, 10),
        )
        self.assertEqual(
            program.reference_graph.value_specs["final_boxes"].shape,
            (1, 900, 10),
        )
        self.assertEqual(
            program.reference_graph.value_specs[
                "intermediate_reference_points"
            ].shape,
            (6, 1, 900, 3),
        )

    def test_strict_profile_weights_and_runtime_input(self):
        config = small_config()
        weights = make_weights(config)
        graph = build_pipeline(config, weights)
        valid = {"value": np.zeros((1, 4, 4), dtype=np.float32)}
        validate_decoder_pipeline_inputs(graph, valid)

        with self.assertRaises(ValidationError):
            BevFormerDecoderPipelineConfig(decoder=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "queries"):
            BevFormerDecoderPipelineConfig(
                decoder=config.decoder,
                detection=DetectionStackConfig(
                    head=DetectionHeadConfig(
                        queries=4, embed_dim=4, classes=2, code_size=6
                    )
                ),
            )
        with self.assertRaises(ValidationError):
            decoder_pipeline_weight_shapes(object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "exactly 6"):
            build_bevformer_decoder_pipeline_graph(
                weights["query_init"],
                weights["self_attention"],
                weights["cross_attention"],
                weights["layer"],
                weights["refinement"],
                weights["detection"][:-1],
                config,
            )

        nonfinite = {"value": valid["value"].copy()}
        nonfinite["value"].flat[0] = np.nan
        with self.assertRaisesRegex(ValidationError, "contiguous and finite"):
            validate_decoder_pipeline_inputs(graph, nonfinite)
        with self.assertRaises(ValidationError):
            validate_decoder_pipeline_inputs(graph, {})


if __name__ == "__main__":
    unittest.main()
