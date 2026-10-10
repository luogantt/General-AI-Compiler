"""Contract, oracle, and CUDA lowering tests for BEVFormer FFN."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
    validate_ffn_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def make_weights(
    config: BEVFormerFFNConfig, seed: int = 101
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, shape in ffn_weight_shapes(config).items():
        if name == "norm_weight":
            weights[name] = rng.uniform(0.8, 1.2, shape).astype(np.float32)
        elif name == "norm_bias":
            weights[name] = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
        else:
            weights[name] = (rng.standard_normal(shape) * 0.12).astype(np.float32)
    return weights


def independent_oracle(
    config: BEVFormerFFNConfig,
    weights: dict[str, np.ndarray],
    query: np.ndarray,
) -> np.ndarray:
    """Direct NumPy translation, independent of GAC operator evaluation."""

    hidden = query @ weights["fc1_weight"] + weights["fc1_bias"]
    hidden = np.maximum(hidden, np.float32(0.0))
    projected = hidden @ weights["fc2_weight"] + weights["fc2_bias"]
    result = query + projected
    if config.post_layer_norm:
        mean = np.mean(result, axis=-1, keepdims=True)
        variance = np.mean((result - mean) ** 2, axis=-1, keepdims=True)
        result = (result - mean) / np.sqrt(
            variance + np.float32(config.layer_norm_epsilon)
        )
        result = result * weights["norm_weight"] + weights["norm_bias"]
    return result


class BEVFormerFFNProfileTests(unittest.TestCase):
    def test_default_profile_matches_tiny_config_and_source_mapping(self):
        config = BEVFormerFFNConfig()
        graph = build_bevformer_ffn_graph(make_weights(config), config)

        self.assertEqual(config.queries, 2500)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(graph.inputs["query"].shape, (1, 2500, 256))
        self.assertEqual(graph.value_specs["activated"].shape, (1, 2500, 512))
        self.assertEqual(graph.value_specs["output"].shape, (1, 2500, 256))
        self.assertTrue(all(spec.is_static for spec in graph.value_specs.values()))
        self.assertEqual(
            [node.op for node in graph.nodes],
            ["MatMul", "Add", "ReLU", "MatMul", "Add", "Add", "LayerNorm"],
        )
        self.assertTrue(all(node.source_nodes for node in graph.nodes))
        self.assertEqual(
            graph.nodes[-2].source_nodes,
            ("mmcv.cnn.bricks.transformer.FFN.forward.identity",),
        )
        self.assertEqual(
            graph.nodes[-1].source_nodes,
            ("BEVFormerLayer.operation_order.norm_after_ffn",),
        )
        self.assertEqual(graph.metadata["operation_order_fragment"], ["ffn", "norm"])
        self.assertEqual(graph.metadata["dropout"], "inference-identity")
        self.assertEqual(graph.metadata["ffn_dropout_probability"], 0.1)
        self.assertEqual(
            graph.metadata["weight_layout"], "[input_channel, output_channel]"
        )
        self.assertIn("checkpoint prefix", graph.metadata["checkpoint_mapping_boundary"])

    def test_ffn_without_following_norm_is_explicit(self):
        config = BEVFormerFFNConfig(
            bev_h=1, bev_w=2, embed_dim=4, post_layer_norm=False
        )
        graph = build_bevformer_ffn_graph(make_weights(config), config)
        self.assertEqual(len(graph.nodes), 6)
        self.assertEqual(graph.nodes[-1].id, "residual_add")
        self.assertEqual(graph.nodes[-1].outputs, ("output",))
        self.assertEqual(graph.metadata["operation_order_fragment"], ["ffn"])
        self.assertNotIn("norm_weight", graph.constants)

    def test_small_profile_compiles_offline_to_cuda(self):
        config = BEVFormerFFNConfig(bev_h=2, bev_w=3, embed_dim=4)
        graph = build_bevformer_ffn_graph(make_weights(config), config)
        first = compile_cuda(graph)
        second = compile_cuda(graph)

        self.assertEqual(first.source, second.source)
        self.assertEqual(len(first.launches), 6)
        self.assertEqual(sum(node.op == "AddReLU" for node in first.graph.nodes), 1)
        self.assertEqual(first.graph.outputs, ("output",))
        self.assertEqual(first.reference_graph.value_specs["output"].shape, (1, 6, 4))
        self.assertIn('extern "C" __global__', first.source)
        self.assertEqual(
            first.launches[-1].source_nodes,
            ("BEVFormerLayer.operation_order.norm_after_ffn",),
        )

    def test_config_rejects_invalid_static_contract(self):
        cases = [
            {"batch": 0},
            {"bev_h": True},
            {"bev_w": -1},
            {"embed_dim": 0},
            {"post_layer_norm": 1},
            {"layer_norm_epsilon": 0.0},
            {"layer_norm_epsilon": float("nan")},
            {"layer_norm_epsilon": 1e-50},
            {"ffn_dropout": -0.1},
            {"ffn_dropout": 1.0},
            {"ffn_dropout": True},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                BEVFormerFFNConfig(**arguments)


class BEVFormerFFNReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = BEVFormerFFNConfig(bev_h=2, bev_w=3, embed_dim=4)
        self.weights = make_weights(self.config)
        self.graph = build_bevformer_ffn_graph(self.weights, self.config)
        rng = np.random.default_rng(303)
        self.query = rng.standard_normal((1, 6, 4), dtype=np.float32)

    def test_lowered_graph_matches_independent_numpy_oracle(self):
        inputs = {"query": self.query}
        validate_ffn_inputs(self.graph, inputs)
        actual = reference_run(self.graph, inputs)["output"]
        expected = independent_oracle(self.config, self.weights, self.query)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_no_norm_oracle_keeps_input_residual(self):
        config = BEVFormerFFNConfig(
            bev_h=1, bev_w=2, embed_dim=4, post_layer_norm=False
        )
        weights = make_weights(config)
        graph = build_bevformer_ffn_graph(weights, config)
        query = self.query[:, :2].copy()
        actual = reference_run(graph, {"query": query})["output"]
        expected = independent_oracle(config, weights, query)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_builder_copies_weights(self):
        original = self.graph.constants["fc1_weight"].copy()
        self.weights["fc1_weight"].fill(99.0)
        np.testing.assert_array_equal(self.graph.constants["fc1_weight"], original)


class BEVFormerFFNContractTests(unittest.TestCase):
    def setUp(self):
        self.config = BEVFormerFFNConfig(bev_h=1, bev_w=2, embed_dim=4)
        self.weights = make_weights(self.config)

    def test_weight_contract_rejects_missing_extra_shape_dtype_layout_and_nan(self):
        cases: list[dict[str, np.ndarray]] = []

        missing = dict(self.weights)
        missing.pop("fc1_bias")
        cases.append(missing)

        extra = dict(self.weights)
        extra["unexpected"] = np.zeros((1,), np.float32)
        cases.append(extra)

        wrong_shape = dict(self.weights)
        wrong_shape["fc1_weight"] = np.zeros((4, 7), np.float32)
        cases.append(wrong_shape)

        wrong_dtype = dict(self.weights)
        wrong_dtype["fc2_bias"] = wrong_dtype["fc2_bias"].astype(np.float64)
        cases.append(wrong_dtype)

        noncontiguous = dict(self.weights)
        noncontiguous["fc1_weight"] = np.zeros((8, 4), np.float32).T
        self.assertEqual(noncontiguous["fc1_weight"].shape, (4, 8))
        self.assertFalse(noncontiguous["fc1_weight"].flags.c_contiguous)
        cases.append(noncontiguous)

        nonfinite = dict(self.weights)
        nonfinite["norm_bias"] = nonfinite["norm_bias"].copy()
        nonfinite["norm_bias"][0] = np.nan
        cases.append(nonfinite)

        for index, weights in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValidationError):
                build_bevformer_ffn_graph(weights, self.config)

    def test_norm_parameter_names_follow_post_norm_mode(self):
        no_norm = BEVFormerFFNConfig(
            bev_h=1, bev_w=2, embed_dim=4, post_layer_norm=False
        )
        weights = make_weights(no_norm)
        with_norm_keys = dict(weights)
        with_norm_keys["norm_weight"] = np.ones((4,), np.float32)
        with_norm_keys["norm_bias"] = np.zeros((4,), np.float32)
        with self.assertRaises(ValidationError):
            build_bevformer_ffn_graph(with_norm_keys, no_norm)

        missing_norm = dict(self.weights)
        missing_norm.pop("norm_weight")
        with self.assertRaises(ValidationError):
            build_bevformer_ffn_graph(missing_norm, self.config)

    def test_input_contract_rejects_names_shape_dtype_layout_and_nonfinite(self):
        graph = build_bevformer_ffn_graph(self.weights, self.config)
        valid = np.zeros((1, 2, 4), np.float32)
        invalid_inputs = [
            {},
            {"other": valid},
            {"query": np.zeros((1, 3, 4), np.float32)},
            {"query": valid.astype(np.float64)},
            {"query": np.zeros((1, 4, 4), np.float32)[:, ::2, :]},
        ]
        nonfinite = valid.copy()
        nonfinite[0, 0, 0] = np.inf
        invalid_inputs.append({"query": nonfinite})

        for inputs in invalid_inputs:
            with self.subTest(inputs=list(inputs)), self.assertRaises(ValidationError):
                validate_ffn_inputs(graph, inputs)

    def test_builder_rejects_invalid_config_name_and_mapping(self):
        with self.assertRaises(ValidationError):
            ffn_weight_shapes(object())  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_bevformer_ffn_graph([], self.config)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_bevformer_ffn_graph(self.weights, object())  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            build_bevformer_ffn_graph(self.weights, self.config, name="")


if __name__ == "__main__":
    unittest.main()
