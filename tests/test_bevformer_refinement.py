from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_refinement import (
    DecoderReferenceRefinementConfig,
    build_decoder_reference_refinement_graph,
    decoder_refinement_weight_shapes,
    validate_decoder_refinement_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


SMALL = DecoderReferenceRefinementConfig(
    batch=1, queries=3, embed_dim=4, code_size=6
)


def make_weights(config=SMALL, seed=3070):
    rng = np.random.default_rng(seed)
    return {
        name: np.ascontiguousarray(rng.normal(0, 0.1, shape).astype(np.float32))
        for name, shape in decoder_refinement_weight_shapes(config).items()
    }


def make_inputs(config=SMALL, seed=86):
    rng = np.random.default_rng(seed)
    return {
        "query": rng.normal(size=(config.batch, config.queries, config.embed_dim)).astype(np.float32),
        "reference_points": rng.uniform(
            0.05, 0.95, (config.batch, config.queries, 3)
        ).astype(np.float32),
    }


class DecoderReferenceRefinementTests(unittest.TestCase):
    def test_graph_matches_independent_numpy_oracle_and_compiles(self):
        weights = make_weights()
        inputs = make_inputs()
        graph = build_decoder_reference_refinement_graph(weights, SMALL)
        validate_decoder_refinement_inputs(graph, inputs)
        program = compile_cuda(graph)
        actual = reference_run(program.reference_graph, inputs)

        hidden0 = np.maximum(
            inputs["query"] @ weights["fc0_weight"] + weights["fc0_bias"], 0
        )
        hidden1 = np.maximum(
            hidden0 @ weights["fc1_weight"] + weights["fc1_bias"], 0
        )
        regression = hidden1 @ weights["fc2_weight"] + weights["fc2_bias"]
        reference = np.clip(inputs["reference_points"], 0, 1)
        logits = np.log(
            np.maximum(reference, SMALL.epsilon)
            / np.maximum(1 - reference, SMALL.epsilon)
        )
        expected_reference = np.concatenate(
            (
                1 / (1 + np.exp(-(regression[..., :2] + logits[..., :2]))),
                1 / (1 + np.exp(-(regression[..., 4:5] + logits[..., 2:3]))),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(actual["regression"], regression, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            actual["next_reference_points"], expected_reference, rtol=1e-6, atol=1e-6
        )
        self.assertEqual(program.graph.value_specs["next_reference_points"].shape, (1, 3, 3))
        self.assertIn("logf", program.source)

    def test_default_profile_and_boundary_metadata(self):
        config = DecoderReferenceRefinementConfig()
        self.assertEqual((config.queries, config.embed_dim, config.code_size), (900, 256, 10))
        graph = build_decoder_reference_refinement_graph(make_weights(config), config)
        self.assertFalse(graph.metadata["full_detection_head"])
        self.assertEqual(graph.metadata["reference_channels"], {"xy": [0, 1], "z": 4})

    def test_rejects_invalid_config_weights_and_inputs(self):
        for kwargs in (
            {"queries": 0},
            {"embed_dim": True},
            {"code_size": 4},
            {"epsilon": 0},
            {"epsilon": 0.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                DecoderReferenceRefinementConfig(**kwargs)

        weights = make_weights()
        broken = dict(weights)
        broken["fc2_weight"] = np.zeros((4, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "fc2_weight"):
            build_decoder_reference_refinement_graph(broken, SMALL)

        graph = build_decoder_reference_refinement_graph(weights, SMALL)
        values = make_inputs()
        values["reference_points"] = values["reference_points"].copy()
        values["reference_points"].flat[0] = 1.1
        with self.assertRaisesRegex(ValidationError, "normalized"):
            validate_decoder_refinement_inputs(graph, values)


if __name__ == "__main__":
    unittest.main()
