"""Static BEVFormer decoder self-attention contract and oracle tests."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder_self import (
    DecoderSelfAttentionConfig,
    build_decoder_self_attention_graph,
    decoder_self_attention_weight_shapes,
    validate_decoder_self_attention_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def make_weights(
    config: DecoderSelfAttentionConfig, seed: int = 701
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, shape in decoder_self_attention_weight_shapes(config).items():
        if name == "norm_weight":
            result[name] = rng.uniform(0.8, 1.2, shape).astype(np.float32)
        elif name == "norm_bias":
            result[name] = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
        else:
            result[name] = (rng.standard_normal(shape) * 0.15).astype(np.float32)
    return result


def independent_oracle(config, weights, query, query_pos):
    """Direct NumPy translation of MMCV self-attention inference semantics."""

    batch, queries, channels = query.shape
    heads, head_dim = config.heads, config.head_dim
    positioned = query + query_pos
    q = positioned @ weights["q_weight"] + weights["q_bias"]
    k = positioned @ weights["k_weight"] + weights["k_bias"]
    v = query @ weights["v_weight"] + weights["v_bias"]
    q = q.reshape(batch, queries, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, queries, heads, head_dim).transpose(0, 2, 3, 1)
    v = v.reshape(batch, queries, heads, head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k) / np.sqrt(np.float32(head_dim))
    exponentials = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    context = (probabilities @ v).transpose(0, 2, 1, 3)
    context = context.reshape(batch, queries, channels)
    result = context @ weights["out_weight"] + weights["out_bias"] + query
    if config.post_layer_norm:
        mean = np.mean(result, axis=-1, keepdims=True)
        variance = np.mean((result - mean) ** 2, axis=-1, keepdims=True)
        result = (result - mean) / np.sqrt(
            variance + np.float32(config.norm_epsilon)
        )
        result = result * weights["norm_weight"] + weights["norm_bias"]
    return result


class DecoderSelfAttentionProfileTests(unittest.TestCase):
    def test_default_tiny_profile_and_exact_wiring(self):
        config = DecoderSelfAttentionConfig()
        graph = build_decoder_self_attention_graph(make_weights(config), config)

        self.assertEqual(graph.inputs["query"].shape, (1, 900, 256))
        self.assertEqual(graph.inputs["query_pos"].shape, (1, 900, 256))
        self.assertEqual(graph.value_specs["output"].shape, (1, 900, 256))
        self.assertTrue(all(spec.is_static for spec in graph.value_specs.values()))
        nodes = {node.id: node for node in graph.nodes}
        self.assertEqual(
            nodes["attention.q_linear"].inputs[0], "positioned_query"
        )
        self.assertEqual(
            nodes["attention.k_linear"].inputs[0], "positioned_query"
        )
        self.assertEqual(nodes["attention.v_linear"].inputs[0], "query")
        self.assertEqual(
            nodes["residual_add"].inputs, ("attention_projected", "query")
        )
        self.assertEqual(nodes["post_layer_norm"].inputs[0], "self_attention_output")
        self.assertEqual(
            nodes["post_layer_norm"].source_nodes,
            ("BEVFormerDecoderLayer.operation_order.norms.0",),
        )
        self.assertEqual(graph.metadata["dropout"], "inference-identity")
        self.assertEqual(graph.metadata["dropout_probability"], 0.1)
        self.assertFalse(graph.metadata["full_decoder"])
        self.assertEqual(
            graph.metadata["operation_order_fragment"], ["self_attn", "norms.0"]
        )
        self.assertIn("residual=False", graph.metadata["component_builder"])

    def test_optional_norm_is_removed_with_its_weights(self):
        config = DecoderSelfAttentionConfig(
            queries=2, embed_dim=4, heads=2, post_layer_norm=False
        )
        graph = build_decoder_self_attention_graph(make_weights(config), config)
        self.assertEqual(graph.nodes[-1].id, "residual_add")
        self.assertEqual(graph.nodes[-1].outputs, ("output",))
        self.assertNotIn("norm_weight", graph.constants)
        self.assertNotIn("norm_bias", graph.constants)
        self.assertEqual(graph.metadata["operation_order_fragment"], ["self_attn"])

    def test_small_profile_compiles_offline_deterministically(self):
        config = DecoderSelfAttentionConfig(queries=3, embed_dim=4, heads=2)
        graph = build_decoder_self_attention_graph(make_weights(config), config)
        first = compile_cuda(graph)
        second = compile_cuda(graph)

        self.assertEqual(first.source, second.source)
        self.assertEqual(len(first.launches), 19)
        self.assertEqual(len(first.describe()["zero_copy_views"]), 4)
        self.assertEqual(first.reference_graph.value_specs["output"].shape, (1, 3, 4))
        self.assertIn('extern "C" __global__', first.source)
        self.assertEqual(
            first.launches[-1].source_nodes,
            ("BEVFormerDecoderLayer.operation_order.norms.0",),
        )

    def test_config_rejects_invalid_static_profile(self):
        cases = [
            {"batch": 2},
            {"batch": True},
            {"queries": 0},
            {"embed_dim": 7, "heads": 2},
            {"heads": -1},
            {"post_layer_norm": 1},
            {"norm_epsilon": 0.0},
            {"norm_epsilon": float("nan")},
            {"norm_epsilon": 1e-50},
            {"dropout_probability": -0.1},
            {"dropout_probability": 1.0},
            {"dropout_probability": True},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                DecoderSelfAttentionConfig(**arguments)


class DecoderSelfAttentionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = DecoderSelfAttentionConfig(queries=3, embed_dim=4, heads=2)
        self.weights = make_weights(self.config)
        self.graph = build_decoder_self_attention_graph(self.weights, self.config)
        rng = np.random.default_rng(709)
        self.inputs = {
            "query": rng.standard_normal((1, 3, 4), dtype=np.float32),
            "query_pos": rng.standard_normal((1, 3, 4), dtype=np.float32),
        }

    def test_composed_graph_matches_independent_numpy_oracle(self):
        validate_decoder_self_attention_inputs(self.graph, self.inputs)
        actual = reference_run(self.graph, self.inputs)["output"]
        expected = independent_oracle(
            self.config, self.weights, self.inputs["query"], self.inputs["query_pos"]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_no_norm_oracle_preserves_original_query_residual(self):
        config = DecoderSelfAttentionConfig(
            queries=3, embed_dim=4, heads=2, post_layer_norm=False
        )
        weights = make_weights(config)
        graph = build_decoder_self_attention_graph(weights, config)
        actual = reference_run(graph, self.inputs)["output"]
        expected = independent_oracle(
            config, weights, self.inputs["query"], self.inputs["query_pos"]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_graph_owns_weight_copies(self):
        original = self.graph.constants["attention.q_weight"].copy()
        self.weights["q_weight"].fill(99.0)
        np.testing.assert_array_equal(
            self.graph.constants["attention.q_weight"], original
        )


class DecoderSelfAttentionContractTests(unittest.TestCase):
    def setUp(self):
        self.config = DecoderSelfAttentionConfig(queries=2, embed_dim=4, heads=2)
        self.weights = make_weights(self.config)

    def test_weight_contract_rejects_names_types_shapes_and_values(self):
        cases = []
        missing = dict(self.weights)
        missing.pop("q_bias")
        cases.append(missing)
        extra = dict(self.weights)
        extra["extra"] = np.zeros((1,), np.float32)
        cases.append(extra)
        non_string = dict(self.weights)
        non_string[1] = np.zeros((1,), np.float32)
        cases.append(non_string)
        not_array = dict(self.weights)
        not_array["q_bias"] = [0.0] * 4
        cases.append(not_array)
        wrong_shape = dict(self.weights)
        wrong_shape["k_weight"] = np.zeros((4, 5), np.float32)
        cases.append(wrong_shape)
        wrong_dtype = dict(self.weights)
        wrong_dtype["v_bias"] = wrong_dtype["v_bias"].astype(np.float64)
        cases.append(wrong_dtype)
        noncontiguous = dict(self.weights)
        noncontiguous["out_weight"] = np.zeros((4, 8), np.float32)[:, ::2]
        cases.append(noncontiguous)
        nonfinite = dict(self.weights)
        nonfinite["norm_weight"] = nonfinite["norm_weight"].copy()
        nonfinite["norm_weight"][0] = np.nan
        cases.append(nonfinite)

        for index, weights in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                build_decoder_self_attention_graph(weights, self.config)

    def test_norm_weight_names_match_mode_exactly(self):
        no_norm = DecoderSelfAttentionConfig(
            queries=2, embed_dim=4, heads=2, post_layer_norm=False
        )
        weights = make_weights(no_norm)
        weights["norm_weight"] = np.ones((4,), np.float32)
        weights["norm_bias"] = np.zeros((4,), np.float32)
        with self.assertRaises(ValidationError):
            build_decoder_self_attention_graph(weights, no_norm)

        missing_norm = dict(self.weights)
        missing_norm.pop("norm_bias")
        with self.assertRaises(ValidationError):
            build_decoder_self_attention_graph(missing_norm, self.config)

    def test_input_contract_rejects_names_types_shapes_and_values(self):
        graph = build_decoder_self_attention_graph(self.weights, self.config)
        valid = np.zeros((1, 2, 4), np.float32)
        bad_inputs = [
            {},
            {"query": valid, "other": valid},
            {"query": valid, "query_pos": [0.0]},
            {"query": np.zeros((1, 3, 4), np.float32), "query_pos": valid},
            {"query": valid.astype(np.float64), "query_pos": valid},
            {
                "query": valid,
                "query_pos": np.zeros((1, 4, 4), np.float32)[:, ::2, :],
            },
        ]
        nonfinite = valid.copy()
        nonfinite[0, 0, 0] = np.inf
        bad_inputs.append({"query": valid, "query_pos": nonfinite})
        for inputs in bad_inputs:
            with self.subTest(names=list(inputs)), self.assertRaises(ValidationError):
                validate_decoder_self_attention_inputs(graph, inputs)

    def test_builder_rejects_invalid_arguments(self):
        with self.assertRaises(ValidationError):
            decoder_self_attention_weight_shapes(object())  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_decoder_self_attention_graph([], self.config)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_decoder_self_attention_graph(
                self.weights, object()  # type: ignore[arg-type]
            )
        with self.assertRaises(ValidationError):
            build_decoder_self_attention_graph(self.weights, self.config, name="")


if __name__ == "__main__":
    unittest.main()
