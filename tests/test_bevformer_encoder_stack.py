"""Tests for the fixed three-layer BEVFormer-tiny encoder stack."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    build_bevformer_encoder_layer_graph,
)
from gac.bevformer_encoder_stack import (
    BevFormerEncoderStackConfig,
    build_bevformer_encoder_stack_graph,
    encoder_stack_weight_shapes,
    validate_encoder_stack_inputs,
)
from gac.bevformer_geometry import BevProjectionConfig, build_bev_projection_graph
from gac.bevformer_spatial import SpatialCrossAttentionConfig
from gac.bevformer_temporal import TemporalSelfAttentionConfig
from gac.errors import ValidationError
from gac.runtime import reference_run


def small_layer_config() -> BevFormerEncoderLayerConfig:
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


def small_stack_config(history_mode: bool) -> BevFormerEncoderStackConfig:
    return BevFormerEncoderStackConfig(
        layer=small_layer_config(),
        history_mode=history_mode,
    )


def make_weights(
    config: BevFormerEncoderStackConfig,
    *,
    seed: int = 211,
) -> tuple[
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
    list[dict[str, np.ndarray]],
]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, np.ndarray]]] = {
        "temporal": [],
        "spatial": [],
        "layer": [],
    }
    for inventory in encoder_stack_weight_shapes(config):
        for scope, shapes in inventory.items():
            values: dict[str, np.ndarray] = {}
            for name, shape in shapes.items():
                if (
                    scope == "layer"
                    and name.startswith("norm")
                    and name.endswith("weight")
                ):
                    value = 1.0 + rng.standard_normal(shape) * 0.07
                else:
                    scale = 0.10 if name.endswith("weight") else 0.025
                    value = rng.standard_normal(shape) * scale
                values[name] = np.ascontiguousarray(value.astype(np.float32))
            groups[scope].append(values)
    return groups["temporal"], groups["spatial"], groups["layer"]


def make_inputs(
    config: BevFormerEncoderStackConfig,
    *,
    seed: int = 223,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    layer = config.layer
    batch, queries, channels = config.batch, config.queries, config.embed_dim
    query_shape = (batch, queries, channels)
    reference_shape = (batch, queries, layer.temporal.levels, 2)
    query = rng.standard_normal(query_shape, dtype=np.float32)
    values = {
        "query": query,
        "query_pos": rng.standard_normal(query_shape, dtype=np.float32),
        "spatial_query_pos": np.zeros(query_shape, dtype=np.float32),
        "reference_points": rng.uniform(0.1, 0.9, reference_shape).astype(
            np.float32
        ),
        "image_value": rng.standard_normal(
            (
                batch,
                layer.spatial.cameras,
                layer.spatial.spatial_size,
                channels,
            ),
            dtype=np.float32,
        ),
        "lidar2img": np.broadcast_to(
            np.eye(4, dtype=np.float32),
            (batch, layer.projection.cameras, 4, 4),
        ).copy(),
    }
    if config.history_mode:
        previous = rng.standard_normal(query_shape, dtype=np.float32)
        values["temporal_value"] = np.ascontiguousarray(
            np.concatenate((previous, query), axis=0)
        )
        values["prev_reference_points"] = rng.uniform(
            0.1, 0.9, reference_shape
        ).astype(np.float32)
    return values


def run_three_single_layers(
    config: BevFormerEncoderStackConfig,
    temporal_weights: list[dict[str, np.ndarray]],
    spatial_weights: list[dict[str, np.ndarray]],
    layer_weights: list[dict[str, np.ndarray]],
    inputs: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Independent sequential oracle built from the tested one-layer graph."""

    current = inputs["query"]
    outputs: list[np.ndarray] = []
    for index in range(3):
        temporal_value = (
            inputs["temporal_value"]
            if config.history_mode
            else np.ascontiguousarray(np.concatenate((current, current), axis=0))
        )
        previous_reference = (
            inputs["prev_reference_points"]
            if config.history_mode
            else inputs["reference_points"]
        )
        graph = build_bevformer_encoder_layer_graph(
            temporal_weights[index],
            spatial_weights[index],
            layer_weights[index],
            config.layer,
            name=f"oracle_layer_{index}",
        )
        current = reference_run(
            graph,
            {
                "query": current,
                "temporal_value": temporal_value,
                "query_pos": inputs["query_pos"],
                "spatial_query_pos": inputs["spatial_query_pos"],
                "prev_reference_points": previous_reference,
                "reference_points": inputs["reference_points"],
                "image_value": inputs["image_value"],
                "lidar2img": inputs["lidar2img"],
            },
        )["output"]
        outputs.append(current)
    return outputs


class EncoderStackReferenceTests(unittest.TestCase):
    def test_history_and_cold_match_three_sequential_single_layers(self):
        for history_mode in (True, False):
            with self.subTest(history_mode=history_mode):
                config = small_stack_config(history_mode)
                temporal, spatial, layer = make_weights(config)
                inputs = make_inputs(config)
                graph = build_bevformer_encoder_stack_graph(
                    temporal, spatial, layer, config
                )
                validate_encoder_stack_inputs(graph, inputs)

                expected = run_three_single_layers(
                    config, temporal, spatial, layer, inputs
                )
                debug_graph = graph.clone()
                debug_graph.outputs = ("layer0.output", "layer1.output", "output")
                debug_graph.validate()
                actual = reference_run(debug_graph, inputs)
                for index, output_name in enumerate(debug_graph.outputs):
                    np.testing.assert_allclose(
                        actual[output_name],
                        expected[index],
                        rtol=5e-5,
                        atol=5e-5,
                    )

    def test_geometry_and_shared_inputs_are_not_duplicated(self):
        config = small_stack_config(True)
        graph = build_bevformer_encoder_stack_graph(*make_weights(config), config)
        geometry = build_bev_projection_graph(config.layer.projection)

        geometry_nodes = [
            node for node in graph.nodes if node.id.startswith("geometry.")
        ]
        self.assertEqual(len(geometry_nodes), len(geometry.nodes))
        self.assertEqual(
            sum(node.id == "geometry.project_reference_points" for node in graph.nodes),
            1,
        )
        nodes = {node.id: node for node in graph.nodes}
        for index in range(3):
            self.assertEqual(
                nodes[f"layer{index}.temporal.query_position_add"].inputs[1],
                "query_pos",
            )
            self.assertEqual(
                nodes[f"layer{index}.temporal.reference_queue_concat"].inputs,
                ("prev_reference_points", "reference_points"),
            )
            self.assertEqual(
                nodes[f"layer{index}.spatial.reference_flatten_cameras"].inputs,
                ("camera_reference_points",),
            )
            self.assertEqual(
                nodes[f"layer{index}.spatial.query_position_add"].inputs[1],
                "spatial_query_pos",
            )
            self.assertEqual(
                nodes[f"layer{index}.spatial.value_flatten_cameras"].inputs,
                ("image_value",),
            )
            self.assertEqual(
                nodes[f"layer{index}.spatial.visibility_apply"].inputs[1],
                "camera_visibility",
            )
            self.assertEqual(
                nodes[f"layer{index}.temporal.value_projection"].inputs[0],
                "temporal_value",
            )
        self.assertEqual(
            nodes["layer0.temporal.query_position_add"].inputs[0], "query"
        )
        self.assertEqual(
            nodes["layer1.temporal.query_position_add"].inputs[0],
            "layer0.output",
        )
        self.assertEqual(
            nodes["layer2.temporal.query_position_add"].inputs[0],
            "layer1.output",
        )
        self.assertEqual(
            sum(node.op == "MSDeformableAttention" for node in graph.nodes), 6
        )
        self.assertTrue(graph.metadata["full_encoder_stack"])
        self.assertEqual(graph.metadata["num_layers"], 3)
        self.assertTrue(graph.metadata["history_mode"])
        self.assertEqual(graph.metadata["geometry_evaluations"], 1)

    def test_cold_mode_builds_one_query_queue_per_layer(self):
        config = small_stack_config(False)
        graph = build_bevformer_encoder_stack_graph(*make_weights(config), config)
        nodes = {node.id: node for node in graph.nodes}

        self.assertNotIn("temporal_value", graph.inputs)
        self.assertNotIn("prev_reference_points", graph.inputs)
        expected_queries = ("query", "layer0.output", "layer1.output")
        for index, query in enumerate(expected_queries):
            concat = nodes[f"layer{index}.cold_temporal_value_concat"]
            self.assertEqual(concat.inputs, (query, query))
            self.assertEqual(concat.attrs, {"axis": 0})
            self.assertEqual(
                nodes[f"layer{index}.temporal.value_projection"].inputs[0],
                f"layer{index}.cold_temporal_value",
            )
            self.assertEqual(
                nodes[f"layer{index}.temporal.reference_queue_concat"].inputs,
                ("reference_points", "reference_points"),
            )
        self.assertFalse(graph.metadata["history_mode"])
        self.assertEqual(graph.metadata["temporal_mode"], "cold")

    def test_default_tiny_history_and_cold_profiles_compile_offline(self):
        for history_mode in (True, False):
            with self.subTest(history_mode=history_mode):
                config = BevFormerEncoderStackConfig(history_mode=history_mode)
                graph = build_bevformer_encoder_stack_graph(
                    *make_weights(config), config
                )
                program = compile_cuda(graph)
                self.assertEqual(
                    program.graph.value_specs["output"].shape, (1, 2500, 256)
                )
                self.assertEqual(program.describe()["target"], "cuda-sm86")
                self.assertTrue(program.graph.metadata["full_encoder_stack"])
                self.assertEqual(program.graph.metadata["num_layers"], 3)
                self.assertIs(
                    program.graph.metadata["history_mode"], history_mode
                )
                self.assertEqual(
                    sum(
                        node.id == "geometry.project_reference_points"
                        for node in program.reference_graph.nodes
                    ),
                    1,
                )


class EncoderStackContractTests(unittest.TestCase):
    def test_config_is_fixed_to_three_layers_and_strict_mode_type(self):
        layer = small_layer_config()
        for kwargs, message in (
            ({"num_layers": 2}, "exactly num_layers=3"),
            ({"num_layers": True}, "exactly num_layers=3"),
            ({"history_mode": 1}, "history_mode must be bool"),
            ({"layer": object()}, "BevFormerEncoderLayerConfig"),
        ):
            with self.subTest(kwargs=kwargs):
                values = {"layer": layer, "num_layers": 3, "history_mode": True}
                values.update(kwargs)
                with self.assertRaisesRegex(ValidationError, message):
                    BevFormerEncoderStackConfig(**values)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValidationError, "profiles disagree"):
            BevFormerEncoderStackConfig(
                layer=BevFormerEncoderLayerConfig(
                    temporal=layer.temporal,
                    spatial=SpatialCrossAttentionConfig(
                        batch=1,
                        cameras=2,
                        queries=5,
                        embed_dim=4,
                        heads=2,
                        spatial_shapes=((2, 2),),
                        sampling_points=4,
                        z_anchors=2,
                    ),
                    projection=layer.projection,
                    ffn_hidden_dim=6,
                )
            )

    def test_all_weight_sequences_require_exactly_three_strict_groups(self):
        config = small_stack_config(True)
        temporal, spatial, layer = make_weights(config)
        cases = (
            (temporal[:2], spatial, layer, "temporal.*exactly 3"),
            (temporal, spatial + [spatial[0]], layer, "spatial.*exactly 3"),
            (temporal, spatial, tuple(layer[:2]), "layer.*exactly 3"),
        )
        for temporal_case, spatial_case, layer_case, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    build_bevformer_encoder_stack_graph(
                        temporal_case, spatial_case, layer_case, config
                    )

        bad_temporal = list(temporal)
        bad_temporal[1] = dict(bad_temporal[1])
        del bad_temporal[1]["value_bias"]
        with self.assertRaisesRegex(ValidationError, r"temporal\[1\].*weight names"):
            build_bevformer_encoder_stack_graph(
                bad_temporal, spatial, layer, config
            )

        bad_spatial = list(spatial)
        bad_spatial[2] = dict(bad_spatial[2])
        bad_spatial[2]["offset_bias"] = np.zeros((15,), dtype=np.float32)
        with self.assertRaisesRegex(ValidationError, r"spatial\[2\].*offset_bias"):
            build_bevformer_encoder_stack_graph(
                temporal, bad_spatial, layer, config
            )

        bad_layer: list[dict[str, object]] = [dict(group) for group in layer]
        bad_layer[0]["norm0_bias"] = [0.0] * 4
        with self.assertRaisesRegex(ValidationError, "NumPy ndarray"):
            build_bevformer_encoder_stack_graph(  # type: ignore[arg-type]
                temporal, spatial, bad_layer, config
            )

    def test_history_and_cold_input_contracts_are_strict(self):
        history = small_stack_config(True)
        history_graph = build_bevformer_encoder_stack_graph(
            *make_weights(history), history
        )
        good = make_inputs(history)
        validate_encoder_stack_inputs(history_graph, good)

        shifted_history = dict(good)
        shifted_history["prev_reference_points"] = good[
            "prev_reference_points"
        ].copy()
        shifted_history["prev_reference_points"].flat[0] = np.float32(1.1)
        validate_encoder_stack_inputs(history_graph, shifted_history)

        missing = dict(good)
        del missing["temporal_value"]

        wrong_shape = dict(good)
        wrong_shape["query_pos"] = np.zeros((1, 3, 4), dtype=np.float32)

        wrong_dtype = dict(good)
        wrong_dtype["image_value"] = good["image_value"].astype(np.float16)

        backing = np.empty((1, 4, 8), dtype=np.float32)
        noncontiguous = dict(good)
        noncontiguous["spatial_query_pos"] = backing[..., ::2]

        nonfinite = dict(good)
        nonfinite["lidar2img"] = good["lidar2img"].copy()
        nonfinite["lidar2img"].flat[0] = np.inf

        bad_reference = dict(good)
        bad_reference["reference_points"] = good["reference_points"].copy()
        bad_reference["reference_points"].flat[0] = np.float32(1.01)

        bad_queue = dict(good)
        bad_queue["temporal_value"] = good["temporal_value"].copy()
        bad_queue["temporal_value"][-1, -1, -1] += np.float32(1.0)

        not_array: dict[str, object] = dict(good)
        not_array["query"] = good["query"].tolist()

        for name, values, message in (
            ("names", missing, "input names"),
            ("shape", wrong_shape, "query_pos"),
            ("dtype", wrong_dtype, "image_value"),
            ("layout", noncontiguous, "contiguous"),
            ("finite", nonfinite, "finite"),
            ("range", bad_reference, "normalized"),
            ("queue", bad_queue, "second queue entry"),
            ("array", not_array, "NumPy ndarray"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_encoder_stack_inputs(  # type: ignore[arg-type]
                        history_graph, values
                    )

        cold = small_stack_config(False)
        cold_graph = build_bevformer_encoder_stack_graph(*make_weights(cold), cold)
        cold_inputs = make_inputs(cold)
        validate_encoder_stack_inputs(cold_graph, cold_inputs)
        with_history_field = dict(cold_inputs)
        with_history_field["temporal_value"] = good["temporal_value"]
        with self.assertRaisesRegex(ValidationError, "input names"):
            validate_encoder_stack_inputs(cold_graph, with_history_field)


if __name__ == "__main__":
    unittest.main()
