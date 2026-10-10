"""Tests for the fixed-upper-bound BEVFormer spatial attention lowering."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_spatial import (
    SpatialCrossAttentionConfig,
    build_spatial_cross_attention_graph,
    validate_spatial_inputs,
)
from gac.errors import ValidationError
from gac.ops import evaluate_op
from gac.runtime import reference_run


SMALL_CONFIG = SpatialCrossAttentionConfig(
    batch=1,
    cameras=2,
    queries=3,
    embed_dim=4,
    heads=2,
    spatial_shapes=((2, 2),),
    sampling_points=4,
    z_anchors=2,
)


def make_weights(
    config: SpatialCrossAttentionConfig = SMALL_CONFIG,
    *,
    seed: int = 41,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    channels = config.embed_dim
    offsets = config.heads * config.levels * config.sampling_points * 2
    attention = config.heads * config.levels * config.sampling_points
    shapes = {
        "value_weight": (channels, channels),
        "value_bias": (channels,),
        "offset_weight": (channels, offsets),
        "offset_bias": (offsets,),
        "attention_weight": (channels, attention),
        "attention_bias": (attention,),
        "output_weight": (channels, channels),
        "output_bias": (channels,),
    }
    return {
        name: np.ascontiguousarray(
            rng.standard_normal(shape, dtype=np.float32)
            / np.float32(np.sqrt(max(shape[0], 1)))
        )
        for name, shape in shapes.items()
    }


def make_inputs(
    config: SpatialCrossAttentionConfig = SMALL_CONFIG,
    *,
    seed: int = 73,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result = {
        "query": rng.standard_normal(
            (config.batch, config.queries, config.embed_dim), dtype=np.float32
        ),
        "query_pos": rng.standard_normal(
            (config.batch, config.queries, config.embed_dim), dtype=np.float32
        ),
        "value": rng.standard_normal(
            (
                config.batch,
                config.cameras,
                config.spatial_size,
                config.embed_dim,
            ),
            dtype=np.float32,
        ),
        "reference_points": rng.uniform(
            np.float32(0.1),
            np.float32(0.9),
            size=(
                config.batch,
                config.cameras,
                config.queries,
                config.z_anchors,
                2,
            ),
        ).astype(np.float32),
        "visibility": np.asarray(
            [[[[1.0], [0.0], [0.0]], [[1.0], [1.0], [0.0]]]],
            dtype=np.float32,
        ),
    }
    return {name: np.ascontiguousarray(value) for name, value in result.items()}


def spatial_attention_oracle(
    config: SpatialCrossAttentionConfig,
    weights: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    """Independent lowering oracle around the already-tested canonical MSDA op."""

    batch = config.batch
    cameras = config.cameras
    queries = config.queries
    channels = config.embed_dim
    heads = config.heads
    head_dim = config.head_dim
    levels = config.levels
    points = config.sampling_points
    anchors = config.z_anchors
    points_per_anchor = points // anchors
    effective_batch = batch * cameras

    positioned_query = inputs["query"] + inputs["query_pos"]
    camera_query = np.broadcast_to(
        positioned_query[:, None, :, :],
        (batch, cameras, queries, channels),
    ).reshape(effective_batch, queries, channels)
    camera_value = inputs["value"].reshape(
        effective_batch, config.spatial_size, channels
    )

    projected_value = (
        camera_value @ weights["value_weight"] + weights["value_bias"]
    ).reshape(effective_batch, config.spatial_size, heads, head_dim)

    offsets = (
        camera_query @ weights["offset_weight"] + weights["offset_bias"]
    ).reshape(
        effective_batch,
        queries,
        heads,
        levels,
        points_per_anchor,
        anchors,
        2,
    )
    normalizer = np.asarray(
        [[width, height] for height, width in config.spatial_shapes],
        dtype=np.float32,
    ).reshape(1, 1, 1, levels, 1, 1, 2)
    references = inputs["reference_points"].reshape(
        effective_batch, queries, anchors, 2
    )[:, :, None, None, None, :, :]
    locations = (references + offsets / normalizer).reshape(
        effective_batch, queries, heads, levels, points, 2
    )

    attention_logits = (
        camera_query @ weights["attention_weight"]
        + weights["attention_bias"]
    ).reshape(effective_batch, queries, heads, levels * points)
    attention_logits = attention_logits - np.max(
        attention_logits, axis=-1, keepdims=True
    )
    attention = np.exp(attention_logits)
    attention = attention / np.sum(attention, axis=-1, keepdims=True)
    attention = attention.reshape(
        effective_batch, queries, heads, levels, points
    )

    (context,) = evaluate_op(
        "MSDeformableAttention",
        (projected_value, locations, attention),
        {"spatial_shapes": [list(shape) for shape in config.spatial_shapes]},
    )
    context = context.reshape(batch, cameras, queries, channels)
    visibility = inputs["visibility"]
    summed = np.sum(context * visibility, axis=1)
    count = np.maximum(np.sum(visibility, axis=1), np.float32(1.0))
    averaged = summed / count
    return (
        averaged @ weights["output_weight"]
        + weights["output_bias"]
        + inputs["query"]
    )


class SpatialCrossAttentionGraphTests(unittest.TestCase):
    def test_small_graph_matches_independent_pre_and_postprocessing_oracle(self):
        weights = make_weights()
        inputs = make_inputs()
        graph = build_spatial_cross_attention_graph(weights, SMALL_CONFIG)
        validate_spatial_inputs(graph, inputs)

        program = compile_cuda(graph)
        self.assertEqual(
            program.graph.value_specs["output"].shape,
            (1, 3, 4),
        )
        self.assertEqual(
            sum(node.op == "MSDeformableAttention" for node in program.graph.nodes),
            1,
        )
        self.assertEqual(
            program.graph.value_specs["sampling_locations"].shape,
            (2, 3, 2, 1, 4, 2),
        )
        self.assertEqual(
            program.graph.value_specs["attention_weights"].shape,
            (2, 3, 2, 1, 4),
        )
        self.assertEqual(program.graph.metadata["query_count"], 3)
        self.assertEqual(program.graph.metadata["z_anchors"], 2)
        self.assertEqual(
            program.graph.metadata["visibility_semantics"],
            "one-iff-any-z-anchor-is-visible",
        )
        self.assertEqual(
            program.graph.metadata["query_position"],
            "explicit-input-residual-excludes-position",
        )
        residual = next(node for node in program.graph.nodes if node.id == "residual_add")
        self.assertEqual(residual.inputs, ("projected", "query"))
        self.assertIn("fmaxf(p0[i]", program.source)

        actual = reference_run(program.reference_graph, inputs)["output"]
        expected = spatial_attention_oracle(SMALL_CONFIG, weights, inputs)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

        # Query 2 is invisible in every camera.  The clamped divisor keeps the
        # context at zero, so the outer projection contributes only its bias.
        np.testing.assert_allclose(
            actual[:, 2],
            inputs["query"][:, 2] + weights["output_bias"],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_default_profile_carries_the_fixed_bevformer_upper_bound(self):
        config = SpatialCrossAttentionConfig()
        self.assertEqual(config.cameras, 6)
        self.assertEqual(config.queries, 2500)
        self.assertEqual(config.spatial_shapes, ((15, 25),))
        self.assertEqual(config.spatial_size, 375)
        self.assertEqual(config.sampling_points, 8)
        self.assertEqual(config.z_anchors, 4)
        self.assertEqual(config.head_dim, 32)


class SpatialInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.graph = build_spatial_cross_attention_graph(
            make_weights(), SMALL_CONFIG
        )
        self.inputs = make_inputs()

    def test_valid_contract(self):
        validate_spatial_inputs(self.graph, self.inputs)

    def test_rejects_non_mapping_inputs(self):
        with self.assertRaisesRegex(ValidationError, "must be a mapping"):
            validate_spatial_inputs(self.graph, None)  # type: ignore[arg-type]

    def test_rejects_wrong_shape_dtype_and_names(self):
        cases: list[tuple[str, dict[str, np.ndarray], str]] = []

        wrong_shape = dict(self.inputs)
        wrong_shape["query"] = np.zeros((1, 2, 4), dtype=np.float32)
        cases.append(("shape", wrong_shape, "query"))

        wrong_dtype = dict(self.inputs)
        wrong_dtype["reference_points"] = wrong_dtype["reference_points"].astype(
            np.float64
        )
        cases.append(("dtype", wrong_dtype, "reference_points"))

        wrong_names = dict(self.inputs)
        del wrong_names["value"]
        cases.append(("names", wrong_names, "names"))

        for name, values, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_spatial_inputs(self.graph, values)

    def test_rejects_nonbinary_mask_nonfinite_and_noncontiguous_inputs(self):
        nonbinary = dict(self.inputs)
        nonbinary["visibility"] = nonbinary["visibility"].copy()
        nonbinary["visibility"][0, 0, 0, 0] = np.float32(0.5)

        nonfinite = dict(self.inputs)
        nonfinite["value"] = nonfinite["value"].copy()
        nonfinite["value"].flat[0] = np.nan

        noncontiguous = dict(self.inputs)
        backing = np.empty((1, 3, 8), dtype=np.float32)
        noncontiguous["query"] = backing[..., ::2]
        self.assertEqual(noncontiguous["query"].shape, (1, 3, 4))
        self.assertFalse(noncontiguous["query"].flags.c_contiguous)

        for name, values, message in (
            ("visibility", nonbinary, "only 0 or 1"),
            ("finite", nonfinite, "finite"),
            ("contiguous", noncontiguous, "contiguous"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_spatial_inputs(self.graph, values)


class SpatialConfigurationValidationTests(unittest.TestCase):
    def test_rejects_illegal_configurations(self):
        cases = (
            (dict(embed_dim=5, heads=2), "divisible"),
            (dict(sampling_points=3, z_anchors=2), "z_anchors"),
            (dict(spatial_shapes=()), "non-empty"),
            (dict(spatial_shapes=((2, 0),)), "positive"),
            (dict(spatial_shapes=([2, 2],)), "pairs"),
            (dict(cameras=True), "cameras"),
            (dict(batch=2), "batch=1"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValidationError, message):
                    SpatialCrossAttentionConfig(**overrides)

    def test_rejects_bad_weight_contracts(self):
        cases: list[tuple[str, dict[str, np.ndarray], str]] = []

        missing = make_weights()
        del missing["value_bias"]
        cases.append(("missing", missing, "weights must be"))

        wrong_shape = make_weights()
        wrong_shape["offset_bias"] = np.zeros((15,), dtype=np.float32)
        cases.append(("shape", wrong_shape, "offset_bias"))

        wrong_dtype = make_weights()
        wrong_dtype["attention_weight"] = wrong_dtype[
            "attention_weight"
        ].astype(np.float16)
        cases.append(("dtype", wrong_dtype, "attention_weight"))

        nonfinite = make_weights()
        nonfinite["output_bias"][0] = np.inf
        cases.append(("finite", nonfinite, "output_bias"))

        noncontiguous = make_weights()
        backing = np.empty((4, 8), dtype=np.float32)
        noncontiguous["output_weight"] = backing[:, ::2]
        self.assertEqual(noncontiguous["output_weight"].shape, (4, 4))
        self.assertFalse(noncontiguous["output_weight"].flags.c_contiguous)
        cases.append(("contiguous", noncontiguous, "output_weight"))

        for name, weights, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    build_spatial_cross_attention_graph(weights, SMALL_CONFIG)

    def test_rejects_non_mapping_and_non_string_weight_names(self):
        with self.assertRaisesRegex(ValidationError, "must be a string-to-array mapping"):
            build_spatial_cross_attention_graph(None, SMALL_CONFIG)  # type: ignore[arg-type]

        mixed_keys: dict[object, np.ndarray] = dict(make_weights())
        mixed_keys[7] = np.zeros((1,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, "names must be strings"):
            build_spatial_cross_attention_graph(  # type: ignore[arg-type]
                mixed_keys, SMALL_CONFIG
            )


if __name__ == "__main__":
    unittest.main()
