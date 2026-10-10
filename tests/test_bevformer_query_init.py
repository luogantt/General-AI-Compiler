from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_query_init import (
    DecoderQueryInitConfig,
    build_decoder_query_init_graph,
    decoder_query_init_weight_shapes,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


class DecoderQueryInitTests(unittest.TestCase):
    def test_small_graph_matches_split_linear_sigmoid_oracle(self):
        config = DecoderQueryInitConfig(queries=3, embed_dim=4)
        rng = np.random.default_rng(86)
        weights = {
            name: rng.normal(0, 0.2, shape).astype(np.float32)
            for name, shape in decoder_query_init_weight_shapes(config).items()
        }
        graph = build_decoder_query_init_graph(weights, config)
        program = compile_cuda(graph)
        actual = reference_run(program.reference_graph, {})
        embedding = weights["query_embedding"][None]
        query_pos = embedding[..., :4]
        query = embedding[..., 4:]
        logits = query_pos @ weights["reference_weight"] + weights["reference_bias"]
        reference = 1 / (1 + np.exp(-logits))
        np.testing.assert_array_equal(actual["query_pos"], query_pos)
        np.testing.assert_array_equal(actual["query"], query)
        np.testing.assert_allclose(actual["reference_points"], reference, rtol=1e-6, atol=1e-6)
        # Every input is a learned constant, so the default pass pipeline
        # materializes all three outputs at compile time.
        self.assertEqual(len(program.launches), 0)
        self.assertEqual(set(program.graph.outputs), {"query", "query_pos", "reference_points"})

    def test_default_profile_and_strict_contract(self):
        config = DecoderQueryInitConfig()
        self.assertEqual((config.queries, config.embed_dim), (900, 256))
        weights = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in decoder_query_init_weight_shapes(config).items()
        }
        graph = build_decoder_query_init_graph(weights, config)
        self.assertEqual(graph.value_specs["query"].shape, (1, 900, 256))
        self.assertEqual(graph.value_specs["reference_points"].shape, (1, 900, 3))
        self.assertEqual(graph.metadata["embedding_split_order"], ["query_pos", "query"])

        missing = dict(weights)
        del missing["reference_bias"]
        with self.assertRaises(ValidationError):
            build_decoder_query_init_graph(missing, config)
        with self.assertRaises(ValidationError):
            DecoderQueryInitConfig(batch=2)


if __name__ == "__main__":
    unittest.main()
