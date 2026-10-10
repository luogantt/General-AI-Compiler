"""Contract and oracle tests for decoder deformable cross attention."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_decoder import (
    DecoderCrossAttentionConfig,
    build_decoder_cross_attention_graph,
    decoder_cross_attention_weight_shapes,
    validate_decoder_cross_attention_inputs,
)
from gac.errors import ValidationError
from gac.ops import evaluate_op
from gac.runtime import reference_run


SMALL_CONFIG = DecoderCrossAttentionConfig(
    queries=3,
    bev_h=2,
    bev_w=2,
    embed_dim=4,
    heads=2,
    sampling_points=2,
)


def make_weights(
    config: DecoderCrossAttentionConfig,
    *,
    seed: int = 83,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, shape in decoder_cross_attention_weight_shapes(config).items():
        scale = 0.16 if name.endswith("weight") else 0.05
        result[name] = np.ascontiguousarray(
            (rng.standard_normal(shape) * scale).astype(np.float32)
        )
    return result


def make_inputs(
    config: DecoderCrossAttentionConfig,
    *,
    seed: int = 97,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    query_shape = (config.batch, config.queries, config.embed_dim)
    return {
        "query": rng.standard_normal(query_shape, dtype=np.float32),
        "query_pos": rng.standard_normal(query_shape, dtype=np.float32),
        "value": rng.standard_normal(
            (config.batch, config.spatial_size, config.embed_dim),
            dtype=np.float32,
        ),
        "reference_points": rng.uniform(
            0.1,
            0.9,
            (config.batch, config.queries, config.levels, 2),
        ).astype(np.float32),
    }


def decoder_cross_attention_oracle(
    config: DecoderCrossAttentionConfig,
    weights: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Direct NumPy translation around the separately tested MSDA primitive."""

    batch = config.batch
    queries = config.queries
    channels = config.embed_dim
    heads = config.heads
    head_dim = config.head_dim
    levels = config.levels
    points = config.sampling_points

    positioned_query = inputs["query"] + inputs["query_pos"]
    value = (
        inputs["value"] @ weights["value_weight"] + weights["value_bias"]
    ).reshape(batch, config.spatial_size, heads, head_dim)

    offsets = (
        positioned_query @ weights["offset_weight"] + weights["offset_bias"]
    ).reshape(batch, queries, heads, levels, points, 2)
    normalizer = np.asarray(
        [config.bev_w, config.bev_h], dtype=np.float32
    ).reshape(1, 1, 1, levels, 1, 2)
    references = inputs["reference_points"][:, :, None, :, None, :]
    locations = references + offsets / normalizer

    logits = (
        positioned_query @ weights["attention_weight"]
        + weights["attention_bias"]
    ).reshape(batch, queries, heads, levels * points)
    exponential = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    attention = exponential / np.sum(exponential, axis=-1, keepdims=True)
    attention = attention.reshape(batch, queries, heads, levels, points)

    (context,) = evaluate_op(
        "MSDeformableAttention",
        (value, locations, attention),
        {"spatial_shapes": [list(shape) for shape in config.spatial_shapes]},
    )
    projected = context @ weights["output_weight"] + weights["output_bias"]
    return projected + inputs["query"]


class DecoderCrossAttentionGraphTests(unittest.TestCase):
    def test_tiny_profile_and_source_mapping(self):
        config = DecoderCrossAttentionConfig()
        graph = build_decoder_cross_attention_graph(make_weights(config), config)

        self.assertEqual(config.batch, 1)
        self.assertEqual(config.queries, 900)
        self.assertEqual(config.spatial_shapes, ((50, 50),))
        self.assertEqual(config.spatial_size, 2500)
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.heads, 8)
        self.assertEqual(config.head_dim, 32)
        self.assertEqual(config.levels, 1)
        self.assertEqual(config.sampling_points, 4)
        self.assertEqual(graph.inputs["query"].shape, (1, 900, 256))
        self.assertEqual(graph.inputs["value"].shape, (1, 2500, 256))
        self.assertEqual(
            graph.inputs["reference_points"].shape,
            (1, 900, 1, 2),
        )
        self.assertEqual(graph.value_specs["value_msda"].shape, (1, 2500, 8, 32))
        self.assertEqual(
            graph.value_specs["sampling_locations"].shape,
            (1, 900, 8, 1, 4, 2),
        )
        self.assertEqual(
            graph.value_specs["attention_weights"].shape,
            (1, 900, 8, 1, 4),
        )
        self.assertEqual(graph.value_specs["output"].shape, (1, 900, 256))
        self.assertTrue(all(spec.is_static for spec in graph.value_specs.values()))

        nodes = {node.id: node for node in graph.nodes}
        self.assertEqual(
            nodes["offset_projection"].inputs,
            ("positioned_query", "offset_weight"),
        )
        self.assertEqual(
            nodes["decoder_msda"].source_nodes,
            ("CustomMSDeformableAttention.forward.ms_deformable_attention",),
        )
        self.assertEqual(nodes["residual_add"].inputs, ("projected", "query"))
        self.assertEqual(
            nodes["residual_add"].source_nodes,
            ("CustomMSDeformableAttention.forward.residual",),
        )
        self.assertTrue(
            all(
                node.source_nodes[0].startswith("CustomMSDeformableAttention")
                for node in graph.nodes
            )
        )
        self.assertFalse(graph.metadata["full_decoder_stack"])
        self.assertEqual(
            graph.metadata["scope"],
            "single-CustomMSDeformableAttention-inference-call",
        )

    def test_small_graph_matches_independent_oracle_and_compiles(self):
        weights = make_weights(SMALL_CONFIG)
        inputs = make_inputs(SMALL_CONFIG)
        graph = build_decoder_cross_attention_graph(weights, SMALL_CONFIG)
        validate_decoder_cross_attention_inputs(graph, inputs)

        program = compile_cuda(graph)
        self.assertEqual(len(program.graph.nodes), 19)
        self.assertEqual(len(program.launches), 15)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 4)
        self.assertEqual(
            sum(node.op == "MSDeformableAttention" for node in program.graph.nodes),
            1,
        )
        self.assertTrue(
            any(launch.node_id == "decoder_msda" for launch in program.launches)
        )
        self.assertEqual(program.graph.value_specs["output"].shape, (1, 3, 4))

        actual = reference_run(program.reference_graph, inputs)["output"]
        expected = decoder_cross_attention_oracle(SMALL_CONFIG, weights, inputs)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_output_projection_is_followed_by_original_query_residual(self):
        weights = make_weights(SMALL_CONFIG)
        weights["output_weight"].fill(0.0)
        weights["output_bias"].fill(0.0)
        inputs = make_inputs(SMALL_CONFIG)
        inputs["query_pos"] *= np.float32(7.0)

        graph = build_decoder_cross_attention_graph(weights, SMALL_CONFIG)
        actual = reference_run(graph, inputs)["output"]
        np.testing.assert_array_equal(actual, inputs["query"])

    def test_tiny_profile_compiles_to_cuda_without_device_access(self):
        config = DecoderCrossAttentionConfig()
        program = compile_cuda(
            build_decoder_cross_attention_graph(make_weights(config), config)
        )
        self.assertEqual(program.graph.value_specs["output"].shape, (1, 900, 256))
        self.assertEqual(len(program.launches), 15)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 4)
        self.assertEqual(program.describe()["target"], "cuda-sm86")


class DecoderInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_decoder_cross_attention_graph(
            make_weights(SMALL_CONFIG), SMALL_CONFIG
        )
        self.inputs = make_inputs(SMALL_CONFIG)

    def test_valid_contract(self):
        validate_decoder_cross_attention_inputs(self.graph, self.inputs)

    def test_rejects_names_shape_dtype_and_non_array(self):
        missing = dict(self.inputs)
        del missing["query_pos"]

        wrong_shape = dict(self.inputs)
        wrong_shape["query"] = np.zeros((1, 2, 4), dtype=np.float32)

        wrong_dtype = dict(self.inputs)
        wrong_dtype["value"] = wrong_dtype["value"].astype(np.float16)

        not_array: dict[str, object] = dict(self.inputs)
        not_array["query_pos"] = self.inputs["query_pos"].tolist()

        for name, values, message in (
            ("names", missing, "input names"),
            ("shape", wrong_shape, "query"),
            ("dtype", wrong_dtype, "value"),
            ("array", not_array, "NumPy ndarray"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_decoder_cross_attention_inputs(  # type: ignore[arg-type]
                        self.graph, values
                    )

    def test_rejects_noncontiguous_nonfinite_and_out_of_range_references(self):
        noncontiguous = dict(self.inputs)
        backing = np.empty((1, 3, 8), dtype=np.float32)
        noncontiguous["query_pos"] = backing[..., ::2]
        self.assertFalse(noncontiguous["query_pos"].flags.c_contiguous)

        nonfinite = dict(self.inputs)
        nonfinite["value"] = nonfinite["value"].copy()
        nonfinite["value"].flat[0] = np.inf

        low_reference = dict(self.inputs)
        low_reference["reference_points"] = low_reference[
            "reference_points"
        ].copy()
        low_reference["reference_points"].flat[0] = np.float32(-0.01)

        high_reference = dict(self.inputs)
        high_reference["reference_points"] = high_reference[
            "reference_points"
        ].copy()
        high_reference["reference_points"].flat[0] = np.float32(1.01)

        for name, values, message in (
            ("contiguous", noncontiguous, "contiguous"),
            ("finite", nonfinite, "finite"),
            ("low-reference", low_reference, "normalized"),
            ("high-reference", high_reference, "normalized"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_decoder_cross_attention_inputs(self.graph, values)


class DecoderConfigurationValidationTests(unittest.TestCase):
    def test_rejects_illegal_configurations(self):
        cases = (
            (dict(batch=2), "batch=1"),
            (dict(queries=0), "queries"),
            (dict(bev_h=True), "bev_h"),
            (dict(embed_dim=5, heads=2), "divisible"),
            (dict(sampling_points=65), "at most 64"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValidationError, message):
                    DecoderCrossAttentionConfig(**overrides)

    def test_rejects_bad_weight_contracts(self):
        missing = make_weights(SMALL_CONFIG)
        del missing["value_bias"]

        wrong_shape = make_weights(SMALL_CONFIG)
        wrong_shape["offset_bias"] = np.zeros((7,), dtype=np.float32)

        wrong_dtype = make_weights(SMALL_CONFIG)
        wrong_dtype["attention_weight"] = wrong_dtype[
            "attention_weight"
        ].astype(np.float16)

        nonfinite = make_weights(SMALL_CONFIG)
        nonfinite["output_bias"][0] = np.nan

        noncontiguous = make_weights(SMALL_CONFIG)
        backing = np.empty((4, 8), dtype=np.float32)
        noncontiguous["output_weight"] = backing[:, ::2]
        self.assertFalse(noncontiguous["output_weight"].flags.c_contiguous)

        not_array: dict[str, object] = dict(make_weights(SMALL_CONFIG))
        not_array["value_bias"] = [0.0] * 4

        for name, weights, message in (
            ("missing", missing, "weight names"),
            ("shape", wrong_shape, "offset_bias"),
            ("dtype", wrong_dtype, "attention_weight"),
            ("finite", nonfinite, "output_bias"),
            ("contiguous", noncontiguous, "output_weight"),
            ("array", not_array, "NumPy ndarray"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    build_decoder_cross_attention_graph(  # type: ignore[arg-type]
                        weights, SMALL_CONFIG
                    )

    def test_rejects_non_mapping_mixed_names_and_bad_builder_arguments(self):
        with self.assertRaisesRegex(ValidationError, "string-to-NumPy-array"):
            build_decoder_cross_attention_graph(  # type: ignore[arg-type]
                None, SMALL_CONFIG
            )

        mixed: dict[object, np.ndarray] = dict(make_weights(SMALL_CONFIG))
        mixed[5] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "names must be strings"):
            build_decoder_cross_attention_graph(  # type: ignore[arg-type]
                mixed, SMALL_CONFIG
            )

        with self.assertRaisesRegex(ValidationError, "config must be"):
            build_decoder_cross_attention_graph(  # type: ignore[arg-type]
                make_weights(SMALL_CONFIG), object()
            )
        with self.assertRaisesRegex(ValidationError, "config must be"):
            build_decoder_cross_attention_graph(  # type: ignore[arg-type]
                make_weights(SMALL_CONFIG), False
            )
        with self.assertRaisesRegex(ValidationError, "name must be non-empty"):
            build_decoder_cross_attention_graph(
                make_weights(SMALL_CONFIG), SMALL_CONFIG, name=""
            )


if __name__ == "__main__":
    unittest.main()
