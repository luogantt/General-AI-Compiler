"""Contract and independent-oracle tests for Temporal Self Attention."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_temporal import (
    TemporalSelfAttentionConfig,
    build_temporal_self_attention_graph,
    temporal_weight_shapes,
    validate_temporal_inputs,
)
from gac.errors import ValidationError
from gac.ops import evaluate_op
from gac.runtime import reference_run


def make_weights(config: TemporalSelfAttentionConfig, seed: int = 31) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {}
    for name, shape in temporal_weight_shapes(config).items():
        scale = 0.08 if name.endswith("weight") else 0.03
        result[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return result


def make_inputs(config: TemporalSelfAttentionConfig, seed: int = 47) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tensor_shape = (1, config.queries, config.embed_dim)
    reference_shape = (1, config.queries, 1, 2)
    previous = rng.standard_normal(tensor_shape, dtype=np.float32)
    current_value = rng.standard_normal(tensor_shape, dtype=np.float32)
    return {
        "query": rng.standard_normal(tensor_shape, dtype=np.float32),
        "temporal_value": np.ascontiguousarray(
            np.concatenate([previous, current_value], axis=0)
        ),
        "query_pos": rng.standard_normal(tensor_shape, dtype=np.float32),
        "prev_reference_points": rng.uniform(0.1, 0.9, reference_shape).astype(np.float32),
        "reference_points": rng.uniform(0.1, 0.9, reference_shape).astype(np.float32),
    }


def independent_oracle(config, weights, inputs):
    """Direct translation of TemporalSelfAttention.forward around MSDA."""

    b, queue = 1, 2
    q, c, h = config.queries, config.embed_dim, config.heads
    d, p, levels = config.head_dim, config.sampling_points, 1
    query = inputs["query"]
    positioned_query = query + inputs["query_pos"]
    queue_value = inputs["temporal_value"]
    temporal_query = np.concatenate(
        [queue_value[:b], positioned_query], axis=-1
    )

    value = queue_value @ weights["value_weight"] + weights["value_bias"]
    value = value.reshape(b * queue, q, h, d)

    offsets = temporal_query @ weights["offset_weight"] + weights["offset_bias"]
    offsets = offsets.reshape(b, q, h, queue, levels, p, 2)
    offsets = offsets.transpose(0, 3, 1, 2, 4, 5, 6).reshape(
        b * queue, q, h, levels, p, 2
    )
    normalizer = np.asarray([config.bev_w, config.bev_h], np.float32).reshape(
        1, 1, 1, 1, 1, 2
    )
    references = np.concatenate(
        [inputs["prev_reference_points"], inputs["reference_points"]], axis=0
    ).reshape(b * queue, q, 1, levels, 1, 2)
    locations = references + offsets / normalizer

    logits = temporal_query @ weights["attention_weight"] + weights["attention_bias"]
    logits = logits.reshape(b, q, h, queue, levels * p)
    exponential = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities = exponential / np.sum(exponential, axis=-1, keepdims=True)
    attention = probabilities.reshape(b, q, h, queue, levels, p)
    attention = attention.transpose(0, 3, 1, 2, 4, 5).reshape(
        b * queue, q, h, levels, p
    )

    (queue_context,) = evaluate_op(
        "MSDeformableAttention",
        (value, locations, attention),
        {"spatial_shapes": [[config.bev_h, config.bev_w]]},
    )
    context = queue_context.reshape(b, queue, q, c).mean(axis=1, dtype=np.float32)
    projected = context @ weights["output_weight"] + weights["output_bias"]
    return projected + query


class TemporalProfileTests(unittest.TestCase):
    def test_production_profile_and_source_mapping(self):
        config = TemporalSelfAttentionConfig()
        weights = make_weights(config)
        graph = build_temporal_self_attention_graph(weights, config)
        self.assertEqual(config.queries, 2500)
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(
            graph.inputs["query"].shape,
            (1, 2500, 256),
        )
        self.assertEqual(graph.inputs["temporal_value"].shape, (2, 2500, 256))
        self.assertEqual(graph.inputs["reference_points"].shape, (1, 2500, 1, 2))
        self.assertEqual(graph.value_specs["value_msda"].shape, (2, 2500, 8, 32))
        self.assertEqual(
            graph.value_specs["sampling_locations"].shape,
            (2, 2500, 8, 1, 4, 2),
        )
        self.assertEqual(graph.value_specs["attention_weights"].shape, (2, 2500, 8, 1, 4))
        self.assertEqual(graph.value_specs["output"].shape, (1, 2500, 256))
        self.assertTrue(all(spec.is_static for spec in graph.value_specs.values()))
        self.assertEqual(graph.metadata["queue_order"], ["previous", "current"])
        self.assertEqual(graph.metadata["dropout"], "inference-identity")

        nodes = {node.id: node for node in graph.nodes}
        self.assertEqual(nodes["previous_value_slice"].inputs, ("temporal_value",))
        self.assertEqual(
            nodes["previous_value_slice"].source_nodes,
            ("TemporalSelfAttention.forward.value_first_batch",),
        )
        self.assertEqual(
            nodes["temporal_msda"].source_nodes,
            ("TemporalSelfAttention.forward.ms_deformable_attention",),
        )
        self.assertEqual(
            nodes["context_queue_mean"].source_nodes,
            ("TemporalSelfAttention.forward.mean_queue",),
        )
        self.assertEqual(nodes["residual_add"].inputs, ("projected", "query"))
        self.assertEqual(
            nodes["residual_add"].source_nodes,
            ("TemporalSelfAttention.forward.residual",),
        )
        self.assertTrue(all(node.source_nodes for node in graph.nodes))

    def test_small_graph_compiles_to_canonical_nodes(self):
        config = TemporalSelfAttentionConfig(
            bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
        )
        graph = build_temporal_self_attention_graph(make_weights(config), config)
        program = compile_cuda(graph)
        self.assertEqual(len(program.graph.nodes), 28)
        self.assertEqual(len(program.launches), 20)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 8)
        self.assertEqual(sum(node.op == "MSDeformableAttention" for node in graph.nodes), 1)
        self.assertEqual(graph.outputs, ("output",))

    def test_config_rejects_nonfixed_or_invalid_dimensions(self):
        cases = [
            {"batch": 2},
            {"queue_length": 1},
            {"bev_h": 0},
            {"bev_w": True},
            {"embed_dim": 7, "heads": 2},
            {"sampling_points": -1},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                TemporalSelfAttentionConfig(**arguments)


class TemporalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = TemporalSelfAttentionConfig(
            bev_h=2,
            bev_w=2,
            embed_dim=4,
            heads=2,
            sampling_points=2,
        )
        self.weights = make_weights(self.config)
        self.graph = build_temporal_self_attention_graph(self.weights, self.config)
        self.inputs = make_inputs(self.config)

    def test_lowered_graph_matches_independent_numpy_oracle(self):
        validate_temporal_inputs(self.graph, self.inputs)
        actual = reference_run(self.graph, self.inputs)["output"]
        expected = independent_oracle(self.config, self.weights, self.inputs)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_queue_is_previous_then_current_and_residual_is_current_query(self):
        config = TemporalSelfAttentionConfig(
            bev_h=1,
            bev_w=2,
            embed_dim=2,
            heads=1,
            sampling_points=1,
        )
        weights = {
            name: np.zeros(shape, np.float32)
            for name, shape in temporal_weight_shapes(config).items()
        }
        weights["value_weight"][:] = np.eye(2, dtype=np.float32)
        weights["output_weight"][:] = np.eye(2, dtype=np.float32)
        graph = build_temporal_self_attention_graph(weights, config)
        query = np.asarray([[[2.0, 4.0], [6.0, 8.0]]], np.float32)
        previous = np.asarray([[[10.0, 20.0], [30.0, 40.0]]], np.float32)
        references = np.asarray([[[[0.25, 0.5]], [[0.75, 0.5]]]], np.float32)
        inputs = {
            "query": query,
            "temporal_value": np.concatenate([previous, query], axis=0),
            "query_pos": np.full_like(query, 100.0),
            "prev_reference_points": references.copy(),
            "reference_points": references.copy(),
        }
        output = reference_run(graph, inputs)["output"]
        expected = query + (previous + query) * np.float32(0.5)
        np.testing.assert_allclose(output, expected, rtol=0, atol=1e-6)


class TemporalContractTests(unittest.TestCase):
    def setUp(self):
        self.config = TemporalSelfAttentionConfig(
            bev_h=2, bev_w=2, embed_dim=4, heads=2, sampling_points=2
        )

    def test_weight_contract_rejects_names_shape_dtype_layout_and_nonfinite(self):
        good = make_weights(self.config)
        cases = []
        missing = dict(good)
        del missing["output_bias"]
        cases.append(missing)
        extra = dict(good, unexpected=np.zeros(1, np.float32))
        cases.append(extra)
        wrong_shape = dict(good)
        wrong_shape["value_bias"] = np.zeros((5,), np.float32)
        cases.append(wrong_shape)
        wrong_dtype = dict(good)
        wrong_dtype["attention_bias"] = wrong_dtype["attention_bias"].astype(np.float64)
        cases.append(wrong_dtype)
        noncontiguous = dict(good)
        noncontiguous["value_weight"] = np.zeros((4, 8), np.float32)[:, ::2]
        cases.append(noncontiguous)
        nonfinite = dict(good)
        nonfinite["output_bias"] = nonfinite["output_bias"].copy()
        nonfinite["output_bias"][0] = np.nan
        cases.append(nonfinite)
        for weights in cases:
            with self.subTest(keys=sorted(weights)), self.assertRaises(ValidationError):
                build_temporal_self_attention_graph(weights, self.config)

    def test_graph_owns_validated_weight_copies(self):
        weights = make_weights(self.config)
        graph = build_temporal_self_attention_graph(weights, self.config)
        before = graph.constants["value_weight"].copy()
        weights["value_weight"].fill(999)
        np.testing.assert_array_equal(graph.constants["value_weight"], before)

    def test_input_contract_rejects_names_shape_dtype_layout_nonfinite_and_range(self):
        graph = build_temporal_self_attention_graph(make_weights(self.config), self.config)
        good = make_inputs(self.config)
        cases = []
        missing = dict(good)
        del missing["query_pos"]
        cases.append(missing)
        wrong_shape = dict(good)
        wrong_shape["query"] = np.zeros((1, 3, 4), np.float32)
        cases.append(wrong_shape)
        wrong_dtype = dict(good)
        wrong_dtype["temporal_value"] = wrong_dtype["temporal_value"].astype(
            np.float64
        )
        cases.append(wrong_dtype)
        noncontiguous = dict(good)
        noncontiguous["query"] = np.zeros((1, 4, 8), np.float32)[:, :, ::2]
        cases.append(noncontiguous)
        nonfinite = dict(good)
        nonfinite["query_pos"] = nonfinite["query_pos"].copy()
        nonfinite["query_pos"][0, 0, 0] = np.inf
        cases.append(nonfinite)
        low_reference = dict(good)
        low_reference["prev_reference_points"] = low_reference["prev_reference_points"].copy()
        low_reference["prev_reference_points"][0, 0, 0, 0] = -0.01
        cases.append(low_reference)
        high_reference = dict(good)
        high_reference["reference_points"] = high_reference["reference_points"].copy()
        high_reference["reference_points"][0, 0, 0, 1] = 1.01
        cases.append(high_reference)
        for values in cases:
            with self.subTest(names=sorted(values)), self.assertRaises(ValidationError):
                validate_temporal_inputs(graph, values)


if __name__ == "__main__":
    unittest.main()
