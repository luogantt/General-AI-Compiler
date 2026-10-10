"""Composition and oracle tests for one BEVFormer encoder layer."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    build_bevformer_encoder_layer_graph,
    encoder_layer_weight_shapes,
    validate_encoder_layer_inputs,
)
from gac.bevformer_geometry import BevProjectionConfig, build_bev_projection_graph
from gac.bevformer_spatial import (
    SpatialCrossAttentionConfig,
    build_spatial_cross_attention_graph,
)
from gac.bevformer_temporal import (
    TemporalSelfAttentionConfig,
    build_temporal_self_attention_graph,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def small_config() -> BevFormerEncoderLayerConfig:
    return BevFormerEncoderLayerConfig(
        temporal=TemporalSelfAttentionConfig(
            bev_h=2,
            bev_w=2,
            embed_dim=4,
            heads=2,
            sampling_points=2,
        ),
        spatial=SpatialCrossAttentionConfig(
            batch=1,
            cameras=2,
            queries=4,
            embed_dim=4,
            heads=2,
            spatial_shapes=((2, 2),),
            sampling_points=4,
            z_anchors=2,
        ),
        projection=BevProjectionConfig(
            batch=1,
            cameras=2,
            bev_h=2,
            bev_w=2,
            z_anchors=2,
            image_h=1,
            image_w=1,
            point_cloud_range=(0.0, 0.0, 1.0, 1.0, 1.0, 2.0),
        ),
        ffn_hidden_dim=6,
    )


def make_weights(
    config: BevFormerEncoderLayerConfig,
    *,
    seed: int = 109,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    inventory = encoder_layer_weight_shapes(config)
    groups: dict[str, dict[str, np.ndarray]] = {}
    for scope, shapes in inventory.items():
        group: dict[str, np.ndarray] = {}
        for name, shape in shapes.items():
            if scope == "layer" and name.startswith("norm") and name.endswith(
                "weight"
            ):
                value = 1.0 + rng.standard_normal(shape) * 0.08
            else:
                scale = 0.11 if name.endswith("weight") else 0.035
                value = rng.standard_normal(shape) * scale
            group[name] = np.ascontiguousarray(value.astype(np.float32))
        groups[scope] = group
    return groups["temporal"], groups["spatial"], groups["layer"]


def make_inputs(
    config: BevFormerEncoderLayerConfig,
    *,
    seed: int = 127,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch, queries, channels = config.batch, config.queries, config.embed_dim
    query_shape = (batch, queries, channels)
    reference_shape = (batch, queries, config.temporal.levels, 2)
    matrices = np.broadcast_to(
        np.eye(4, dtype=np.float32),
        (batch, config.projection.cameras, 4, 4),
    ).copy()
    return {
        "query": rng.standard_normal(query_shape, dtype=np.float32),
        "temporal_value": rng.standard_normal(
            (
                batch * config.temporal.queue_length,
                queries,
                channels,
            ),
            dtype=np.float32,
        ),
        "query_pos": rng.standard_normal(query_shape, dtype=np.float32),
        # Official BEVFormerLayer passes query_pos=None to spatial attention.
        "spatial_query_pos": np.zeros(query_shape, dtype=np.float32),
        "prev_reference_points": rng.uniform(
            0.1, 0.9, reference_shape
        ).astype(np.float32),
        "reference_points": rng.uniform(0.1, 0.9, reference_shape).astype(
            np.float32
        ),
        "image_value": rng.standard_normal(
            (
                batch,
                config.spatial.cameras,
                config.spatial.spatial_size,
                channels,
            ),
            dtype=np.float32,
        ),
        "lidar2img": matrices,
    }


def layer_norm(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    mean = np.mean(value, axis=-1, keepdims=True)
    variance = np.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return (value - mean) / np.sqrt(variance + epsilon) * weight + bias


def stage_oracle(
    config: BevFormerEncoderLayerConfig,
    temporal_weights: dict[str, np.ndarray],
    spatial_weights: dict[str, np.ndarray],
    layer_weights: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Execute the already-tested components one stage at a time."""

    geometry_graph = build_bev_projection_graph(config.projection)
    geometry = reference_run(
        geometry_graph,
        {"lidar2img": inputs["lidar2img"]},
    )
    temporal_graph = build_temporal_self_attention_graph(
        temporal_weights, config.temporal
    )
    temporal = reference_run(
        temporal_graph,
        {
            "query": inputs["query"],
            "temporal_value": inputs["temporal_value"],
            "query_pos": inputs["query_pos"],
            "prev_reference_points": inputs["prev_reference_points"],
            "reference_points": inputs["reference_points"],
        },
    )["output"]
    norm0 = layer_norm(
        temporal,
        layer_weights["norm0_weight"],
        layer_weights["norm0_bias"],
        config.norm_epsilon,
    )
    spatial_graph = build_spatial_cross_attention_graph(
        spatial_weights, config.spatial
    )
    spatial = reference_run(
        spatial_graph,
        {
            "query": norm0,
            "query_pos": inputs["spatial_query_pos"],
            "value": inputs["image_value"],
            "reference_points": geometry["reference_points"],
            "visibility": geometry["visibility"],
        },
    )["output"]
    norm1 = layer_norm(
        spatial,
        layer_weights["norm1_weight"],
        layer_weights["norm1_bias"],
        config.norm_epsilon,
    )
    hidden = np.maximum(
        norm1 @ layer_weights["ffn_fc1_weight"]
        + layer_weights["ffn_fc1_bias"],
        np.float32(0.0),
    )
    projected = (
        hidden @ layer_weights["ffn_fc2_weight"]
        + layer_weights["ffn_fc2_bias"]
    )
    ffn_residual = projected + norm1
    output = layer_norm(
        ffn_residual,
        layer_weights["norm2_weight"],
        layer_weights["norm2_bias"],
        config.norm_epsilon,
    )
    return {
        "camera_reference_points": geometry["reference_points"],
        "camera_visibility": geometry["visibility"],
        "temporal_output": temporal,
        "norm0_output": norm0,
        "spatial_output": spatial,
        "norm1_output": norm1,
        "ffn_residual": ffn_residual,
        "output": output,
    }


class EncoderLayerCompositionTests(unittest.TestCase):
    def test_small_graph_matches_every_independently_executed_stage(self):
        config = small_config()
        temporal_weights, spatial_weights, layer_weights = make_weights(config)
        inputs = make_inputs(config)
        graph = build_bevformer_encoder_layer_graph(
            temporal_weights,
            spatial_weights,
            layer_weights,
            config,
        )
        validate_encoder_layer_inputs(graph, inputs)

        expected = stage_oracle(
            config,
            temporal_weights,
            spatial_weights,
            layer_weights,
            inputs,
        )
        debug_graph = graph.clone()
        debug_graph.outputs = tuple(expected)
        debug_graph.validate()
        actual = reference_run(debug_graph, inputs)
        for stage in expected:
            with self.subTest(stage=stage):
                np.testing.assert_allclose(
                    actual[stage], expected[stage], rtol=3e-5, atol=3e-5
                )

        self.assertTrue(np.all(actual["camera_visibility"] == 1.0))

    def test_small_graph_compiles_and_preserves_stage_order_and_bindings(self):
        config = small_config()
        graph = build_bevformer_encoder_layer_graph(*make_weights(config), config)
        program = compile_cuda(graph)

        self.assertEqual(program.graph.value_specs["output"].shape, (1, 4, 4))
        self.assertEqual(
            program.graph.metadata["operation_order"],
            [
                "temporal-self-attention",
                "norm0",
                "spatial-cross-attention",
                "norm1",
                "ffn-residual",
                "norm2",
            ],
        )
        nodes = {node.id: node for node in program.reference_graph.nodes}
        self.assertEqual(
            nodes["temporal.value_projection"].inputs[0],
            "temporal_value",
        )
        self.assertIn(
            "previous BEV, initial current BEV",
            program.graph.metadata["temporal_value_contract"],
        )
        self.assertEqual(
            nodes["spatial.query_position_add"].inputs,
            ("norm0_output", "spatial_query_pos"),
        )
        self.assertEqual(
            nodes["spatial.residual_add"].inputs,
            ("spatial.projected", "norm0_output"),
        )
        self.assertEqual(
            nodes["layer.ffn_residual"].inputs,
            ("ffn_projected", "norm1_output"),
        )
        self.assertEqual(program.graph.outputs, ("output",))
        self.assertEqual(
            sum(node.op == "MSDeformableAttention" for node in program.graph.nodes),
            2,
        )

    def test_default_profile_is_one_tiny_encoder_layer_and_compiles_offline(self):
        config = BevFormerEncoderLayerConfig()
        temporal_weights, spatial_weights, layer_weights = make_weights(config)
        graph = build_bevformer_encoder_layer_graph(
            temporal_weights,
            spatial_weights,
            layer_weights,
            config,
        )
        self.assertEqual(config.queries, 2500)
        self.assertEqual(config.embed_dim, 256)
        self.assertEqual(config.ffn_hidden_dim, 512)
        self.assertEqual(graph.inputs["temporal_value"].shape, (2, 2500, 256))
        self.assertEqual(graph.inputs["image_value"].shape, (1, 6, 375, 256))
        self.assertEqual(graph.inputs["lidar2img"].shape, (1, 6, 4, 4))
        self.assertFalse(graph.metadata["full_encoder_stack"])
        self.assertEqual(
            graph.metadata["scope"], "one-BEVFormerLayer-inference-call"
        )
        program = compile_cuda(graph)
        self.assertEqual(program.graph.value_specs["output"].shape, (1, 2500, 256))
        self.assertEqual(program.describe()["target"], "cuda-sm86")


class EncoderLayerContractTests(unittest.TestCase):
    def test_profile_consistency_is_strict(self):
        base = small_config()
        cases = (
            (
                dict(
                    spatial=SpatialCrossAttentionConfig(
                        batch=1,
                        cameras=2,
                        queries=5,
                        embed_dim=4,
                        heads=2,
                        spatial_shapes=((2, 2),),
                        sampling_points=4,
                        z_anchors=2,
                    )
                ),
                "queries",
            ),
            (
                dict(
                    projection=BevProjectionConfig(
                        batch=1,
                        cameras=3,
                        bev_h=2,
                        bev_w=2,
                        z_anchors=2,
                    )
                ),
                "cameras",
            ),
            (
                dict(
                    projection=BevProjectionConfig(
                        batch=1,
                        cameras=2,
                        bev_h=2,
                        bev_w=2,
                        z_anchors=3,
                    )
                ),
                "z_anchors",
            ),
            (dict(ffn_hidden_dim=0), "ffn_hidden_dim"),
            (dict(norm_epsilon=np.inf), "norm_epsilon"),
        )
        for overrides, message in cases:
            values = {
                "temporal": base.temporal,
                "spatial": base.spatial,
                "projection": base.projection,
                "ffn_hidden_dim": base.ffn_hidden_dim,
                "norm_epsilon": base.norm_epsilon,
            }
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValidationError, message):
                    BevFormerEncoderLayerConfig(**values)

    def test_weight_groups_are_strictly_validated(self):
        config = small_config()
        good_temporal, good_spatial, good_layer = make_weights(config)
        cases: list[
            tuple[
                str,
                object,
                object,
                object,
                str,
            ]
        ] = []

        missing_temporal = dict(good_temporal)
        del missing_temporal["value_bias"]
        cases.append(
            ("missing", missing_temporal, good_spatial, good_layer, "temporal")
        )

        wrong_spatial = dict(good_spatial)
        wrong_spatial["offset_bias"] = np.zeros((15,), dtype=np.float32)
        cases.append(
            ("shape", good_temporal, wrong_spatial, good_layer, "offset_bias")
        )

        wrong_layer = dict(good_layer)
        wrong_layer["ffn_fc1_weight"] = wrong_layer[
            "ffn_fc1_weight"
        ].astype(np.float16)
        cases.append(
            ("dtype", good_temporal, good_spatial, wrong_layer, "ffn_fc1_weight")
        )

        nonfinite_layer = {name: value.copy() for name, value in good_layer.items()}
        nonfinite_layer["norm2_bias"][0] = np.nan
        cases.append(
            ("finite", good_temporal, good_spatial, nonfinite_layer, "norm2_bias")
        )

        noncontiguous = dict(good_spatial)
        backing = np.empty((4, 8), dtype=np.float32)
        noncontiguous["value_weight"] = backing[:, ::2]
        cases.append(
            (
                "contiguous",
                good_temporal,
                noncontiguous,
                good_layer,
                "value_weight",
            )
        )

        not_array: dict[str, object] = dict(good_layer)
        not_array["norm0_bias"] = [0.0] * 4
        cases.append(
            ("array", good_temporal, good_spatial, not_array, "NumPy ndarray")
        )

        for name, temporal, spatial, layer, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    build_bevformer_encoder_layer_graph(  # type: ignore[arg-type]
                        temporal, spatial, layer, config
                    )

    def test_input_contract_rejects_names_shape_dtype_layout_finite_and_range(self):
        config = small_config()
        graph = build_bevformer_encoder_layer_graph(*make_weights(config), config)
        good = make_inputs(config)
        validate_encoder_layer_inputs(graph, good)

        missing = dict(good)
        del missing["spatial_query_pos"]

        wrong_shape = dict(good)
        wrong_shape["query"] = np.zeros((1, 3, 4), dtype=np.float32)

        wrong_dtype = dict(good)
        wrong_dtype["image_value"] = wrong_dtype["image_value"].astype(
            np.float16
        )

        noncontiguous = dict(good)
        backing = np.empty((1, 4, 8), dtype=np.float32)
        noncontiguous["query_pos"] = backing[..., ::2]

        nonfinite = dict(good)
        nonfinite["lidar2img"] = nonfinite["lidar2img"].copy()
        nonfinite["lidar2img"].flat[0] = np.inf

        out_of_range = dict(good)
        out_of_range["prev_reference_points"] = out_of_range[
            "prev_reference_points"
        ].copy()
        out_of_range["prev_reference_points"].flat[0] = np.float32(-0.01)

        not_array: dict[str, object] = dict(good)
        not_array["spatial_query_pos"] = good["spatial_query_pos"].tolist()

        for name, values, message in (
            ("names", missing, "input names"),
            ("shape", wrong_shape, "query"),
            ("dtype", wrong_dtype, "image_value"),
            ("layout", noncontiguous, "contiguous"),
            ("finite", nonfinite, "finite"),
            ("range", out_of_range, "normalized"),
            ("array", not_array, "NumPy ndarray"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_encoder_layer_inputs(  # type: ignore[arg-type]
                        graph, values
                    )


if __name__ == "__main__":
    unittest.main()
