from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.errors import ValidationError
from gac.runtime import reference_run
from gac.transformer import MultiHeadAttentionConfig, build_multihead_attention_graph


def make_weights(channels: int, seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {}
    for prefix in ("q", "k", "v", "out"):
        result[f"{prefix}_weight"] = (
            rng.standard_normal((channels, channels), dtype=np.float32) / np.sqrt(channels)
        ).astype(np.float32)
        result[f"{prefix}_bias"] = rng.standard_normal((channels,), dtype=np.float32)
    return result


def attention_oracle(config, weights, query, key, value):
    batch, queries, channels = query.shape
    keys = key.shape[1]
    heads, head_dim = config.heads, config.head_dim
    q = (query @ weights["q_weight"] + weights["q_bias"]).reshape(batch, queries, heads, head_dim).transpose(0, 2, 1, 3)
    k = (key @ weights["k_weight"] + weights["k_bias"]).reshape(batch, keys, heads, head_dim).transpose(0, 2, 3, 1)
    v = (value @ weights["v_weight"] + weights["v_bias"]).reshape(batch, keys, heads, head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k) / np.sqrt(head_dim)
    exponential = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities = exponential / np.sum(exponential, axis=-1, keepdims=True)
    context = (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, queries, channels)
    output = context @ weights["out_weight"] + weights["out_bias"]
    return output + query if config.residual else output


class TransformerGraphTests(unittest.TestCase):
    def test_attention_lowering_matches_independent_oracle(self):
        config = MultiHeadAttentionConfig(
            batch=2,
            query_length=3,
            key_length=5,
            embed_dim=8,
            heads=2,
        )
        weights = make_weights(config.embed_dim)
        graph = build_multihead_attention_graph(config, weights)
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs[graph.outputs[0]].shape, (2, 3, 8))
        self.assertEqual(sum(node.op == "MatMul" for node in program.graph.nodes), 6)
        self.assertTrue(any("batch_index" in program.source for _ in [0]))

        rng = np.random.default_rng(33)
        inputs = {
            "query": rng.standard_normal((2, 3, 8), dtype=np.float32),
            "key": rng.standard_normal((2, 5, 8), dtype=np.float32),
            "value": rng.standard_normal((2, 5, 8), dtype=np.float32),
        }
        actual = reference_run(program.reference_graph, inputs)[graph.outputs[0]]
        expected = attention_oracle(config, weights, **inputs)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_cross_attention_without_residual(self):
        config = MultiHeadAttentionConfig(1, 2, 4, embed_dim=4, heads=1, residual=False)
        graph = build_multihead_attention_graph(config, make_weights(4))
        self.assertEqual(graph.outputs, ("projected",))
        program = compile_cuda(graph)
        self.assertNotIn("residual_add", [node.id for node in program.graph.nodes])

    def test_contract_rejects_bad_dimensions_and_weights(self):
        with self.assertRaisesRegex(ValidationError, "divisible"):
            MultiHeadAttentionConfig(1, 2, 2, embed_dim=7, heads=2)
        config = MultiHeadAttentionConfig(1, 2, 2, embed_dim=4, heads=2)
        weights = make_weights(4)
        weights["q_bias"] = np.zeros((3,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "q_bias"):
            build_multihead_attention_graph(config, weights)
        weights = make_weights(4)
        weights["q_weight"] = weights["q_weight"][:, ::-1]
        with self.assertRaisesRegex(ValidationError, "contiguous"):
            build_multihead_attention_graph(config, weights)


if __name__ == "__main__":
    unittest.main()
