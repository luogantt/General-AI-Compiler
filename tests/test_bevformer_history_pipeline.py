"""Tests for the complete warm-history BEVFormer encoder graph."""

from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_encoder_inputs import (
    BevFormerEncoderInputConfig,
    build_bevformer_encoder_input_graph,
    encoder_input_weight_shapes,
)
from gac.bevformer_encoder_layer import (
    BevFormerEncoderLayerConfig,
    encoder_layer_weight_shapes,
)
from gac.bevformer_encoder_stack import (
    BevFormerEncoderStackConfig,
    build_bevformer_encoder_stack_graph,
)
from gac.bevformer_feature_prep import (
    BevFormerFeaturePrepConfig,
    build_bevformer_feature_prep_graph,
    feature_prep_weight_shapes,
)
from gac.bevformer_geometry import BevProjectionConfig
from gac.bevformer_history_pipeline import (
    BevFormerHistoryPipelineConfig,
    build_bevformer_history_pipeline_graph,
    history_pipeline_weight_shapes,
    validate_history_pipeline_inputs,
)
from gac.bevformer_spatial import SpatialCrossAttentionConfig
from gac.bevformer_temporal import TemporalSelfAttentionConfig
from gac.errors import ValidationError
from gac.frontends.bevformer_backbone import (
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
    required_checkpoint_specs,
)
from gac.runtime import reference_run


def small_config() -> BevFormerHistoryPipelineConfig:
    backbone = ResNetFpnProfile(
        name="test-history-r50-shape",
        batch=2,
        input_channels=3,
        input_hw=(8, 8),
        stem_channels=2,
        stage_planes=(2, 2, 2, 2),
        blocks=(1, 1, 1, 1),
        expansion=1,
        fpn_channels=4,
        input_dtype="float32",
    )
    temporal = TemporalSelfAttentionConfig(
        bev_h=1, bev_w=2, embed_dim=4, heads=1, sampling_points=1
    )
    spatial = SpatialCrossAttentionConfig(
        cameras=2,
        queries=2,
        embed_dim=4,
        heads=1,
        spatial_shapes=((1, 1),),
        sampling_points=2,
        z_anchors=2,
    )
    projection = BevProjectionConfig(
        cameras=2,
        bev_h=1,
        bev_w=2,
        z_anchors=2,
        image_h=8,
        image_w=8,
    )
    layer = BevFormerEncoderLayerConfig(
        temporal=temporal,
        spatial=spatial,
        projection=projection,
        ffn_hidden_dim=6,
    )
    return BevFormerHistoryPipelineConfig(
        backbone=backbone,
        feature_prep=BevFormerFeaturePrepConfig(
            cameras=2, embed_dim=4, feature_h=1, feature_w=1
        ),
        encoder_inputs=BevFormerEncoderInputConfig(
            bev_h=1,
            bev_w=2,
            embed_dim=4,
            can_bus_dim=4,
            can_bus_hidden_dim=3,
        ),
        encoder_stack=BevFormerEncoderStackConfig(
            layer=layer, history_mode=True
        ),
    )


def random_weights(config: BevFormerHistoryPipelineConfig, seed: int = 421):
    rng = np.random.default_rng(seed)

    def arrays(shapes, scale=0.05):
        return {
            name: np.ascontiguousarray(
                rng.normal(0.0, scale, shape).astype(np.float32)
            )
            for name, shape in shapes.items()
        }

    backbone = {
        spec.key: np.ascontiguousarray(
            (
                np.ones(spec.shape, dtype=np.float32)
                if spec.role == "bn-running-variance"
                else rng.normal(0.0, 0.03, spec.shape).astype(np.float32)
            )
        )
        for spec in required_checkpoint_specs(config.backbone)
    }
    feature = arrays(feature_prep_weight_shapes(config.feature_prep))
    encoder_input = arrays(encoder_input_weight_shapes(config.encoder_inputs))
    inventory = encoder_layer_weight_shapes(config.encoder_stack.layer)
    temporal = [arrays(inventory["temporal"]) for _ in range(3)]
    spatial = [arrays(inventory["spatial"]) for _ in range(3)]
    layer = [arrays(inventory["layer"]) for _ in range(3)]
    return backbone, feature, encoder_input, temporal, spatial, layer


def runtime_inputs(config: BevFormerHistoryPipelineConfig):
    rng = np.random.default_rng(431)
    images_shape = (
        config.feature_prep.batch,
        config.feature_prep.cameras,
        config.backbone.input_channels,
        *config.backbone.input_hw,
    )
    return {
        "images": np.ascontiguousarray(
            rng.normal(0.0, 0.2, images_shape).astype(np.float32)
        ),
        "lidar2img": np.ascontiguousarray(
            np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
        ),
        "can_bus_delta": np.asarray(
            [[0.1, 0.2, -0.1, 0.3]], dtype=np.float32
        ),
        "aligned_prev_bev": np.ascontiguousarray(
            rng.normal(0.0, 0.1, (1, 2, 4)).astype(np.float32)
        ),
        "prev_reference_points": np.asarray(
            [[[[0.25, 0.5]], [[0.75, 0.5]]]], dtype=np.float32
        ),
    }


class BevFormerHistoryPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.weights = random_weights(self.config)
        self.graph = build_bevformer_history_pipeline_graph(
            *self.weights, self.config
        )
        self.inputs = runtime_inputs(self.config)

    def test_composition_matches_independent_staged_execution(self):
        backbone_weights, feature_weights, input_weights, temporal, spatial, layer = (
            self.weights
        )
        backbone = build_bevformer_backbone_graph(
            backbone_weights, self.config.backbone
        )
        fpn = reference_run(
            backbone,
            {
                "images": self.inputs["images"].reshape(
                    self.config.backbone.input_shape
                )
            },
        )["fpn_c5"]
        image_value = reference_run(
            build_bevformer_feature_prep_graph(
                feature_weights, self.config.feature_prep
            ),
            {"fpn_feature": fpn},
        )["image_value"]
        initialized = reference_run(
            build_bevformer_encoder_input_graph(
                input_weights, self.config.encoder_inputs
            ),
            {"can_bus_delta": self.inputs["can_bus_delta"]},
        )
        temporal_value = np.ascontiguousarray(
            np.concatenate(
                (self.inputs["aligned_prev_bev"], initialized["query"]),
                axis=0,
            )
        )
        expected = reference_run(
            build_bevformer_encoder_stack_graph(
                temporal, spatial, layer, self.config.encoder_stack
            ),
            {
                "query": initialized["query"],
                "query_pos": initialized["query_pos"],
                "spatial_query_pos": initialized["spatial_query_pos"],
                "reference_points": initialized["reference_points"],
                "image_value": image_value,
                "lidar2img": self.inputs["lidar2img"],
                "temporal_value": temporal_value,
                "prev_reference_points": self.inputs["prev_reference_points"],
            },
        )["output"]
        actual = reference_run(self.graph, self.inputs)["next_bev"]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)

    def test_one_official_queue_is_shared_by_all_three_layers(self):
        nodes = {node.id: node for node in self.graph.nodes}
        queue_nodes = [
            node
            for node in self.graph.nodes
            if node.id == "build_shared_temporal_queue"
        ]
        self.assertEqual(len(queue_nodes), 1)
        self.assertEqual(
            queue_nodes[0].inputs, ("aligned_prev_bev", "initial_bev_query")
        )
        self.assertEqual(queue_nodes[0].attrs, {"axis": 0})
        for index in range(3):
            self.assertEqual(
                nodes[f"encoder.layer{index}.temporal.value_projection"].inputs[0],
                "shared_temporal_value",
            )
            self.assertEqual(
                nodes[f"encoder.layer{index}.temporal.reference_queue_concat"].inputs,
                ("prev_reference_points", "encoder_inputs.reference_points"),
            )
        self.assertEqual(self.graph.inputs["images"].shape, (1, 2, 3, 8, 8))
        self.assertTrue(self.graph.metadata["history_mode"])
        self.assertIn("external", self.graph.metadata["history_alignment"])

    def test_grouped_weight_inventory_and_sm86_compile(self):
        inventory = history_pipeline_weight_shapes(self.config)
        self.assertEqual(
            set(inventory),
            {"backbone", "feature_prep", "encoder_inputs", "temporal", "spatial", "layer"},
        )
        self.assertEqual(len(inventory["temporal"]), 3)
        self.assertEqual(len(inventory["spatial"]), 3)
        self.assertEqual(len(inventory["layer"]), 3)
        program = compile_cuda(self.graph)
        self.assertGreater(len(program.launches), 50)
        self.assertEqual(program.describe()["target"], "cuda-sm86")
        self.assertEqual(
            program.reference_graph.value_specs["next_bev"].shape, (1, 2, 4)
        )

    def test_config_rejects_cold_mode_and_cross_stage_mismatches(self):
        with self.assertRaisesRegex(ValidationError, "history_mode=True"):
            BevFormerHistoryPipelineConfig(
                backbone=self.config.backbone,
                feature_prep=self.config.feature_prep,
                encoder_inputs=self.config.encoder_inputs,
                encoder_stack=BevFormerEncoderStackConfig(
                    layer=self.config.encoder_stack.layer, history_mode=False
                ),
            )
        with self.assertRaisesRegex(ValidationError, "component profiles disagree"):
            BevFormerHistoryPipelineConfig(
                backbone=self.config.backbone,
                feature_prep=BevFormerFeaturePrepConfig(
                    cameras=2, embed_dim=8, feature_h=1, feature_w=1
                ),
                encoder_inputs=self.config.encoder_inputs,
                encoder_stack=self.config.encoder_stack,
            )

    def test_runtime_abi_is_strict(self):
        validate_history_pipeline_inputs(self.graph, self.inputs)

        missing = dict(self.inputs)
        del missing["aligned_prev_bev"]
        wrong_shape = dict(self.inputs)
        wrong_shape["aligned_prev_bev"] = np.zeros((1, 1, 4), np.float32)
        wrong_dtype = dict(self.inputs)
        wrong_dtype["images"] = self.inputs["images"].astype(np.float16)
        backing = np.zeros((1, 2, 8), dtype=np.float32)
        noncontiguous = dict(self.inputs)
        noncontiguous["aligned_prev_bev"] = backing[..., ::2]
        nonfinite = dict(self.inputs)
        nonfinite["lidar2img"] = self.inputs["lidar2img"].copy()
        nonfinite["lidar2img"].flat[0] = np.nan
        not_array = dict(self.inputs)
        not_array["can_bus_delta"] = [[0.1, 0.2, -0.1, 0.3]]

        for case, message in (
            (missing, "input names"),
            (wrong_shape, "aligned_prev_bev"),
            (wrong_dtype, "images"),
            (noncontiguous, "C-contiguous"),
            (nonfinite, "finite"),
            (not_array, "NumPy ndarray"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidationError, message):
                    validate_history_pipeline_inputs(self.graph, case)  # type: ignore[arg-type]

        shifted_border = dict(self.inputs)
        shifted_border["prev_reference_points"] = self.inputs[
            "prev_reference_points"
        ].copy()
        shifted_border["prev_reference_points"].flat[0] = 1.1
        validate_history_pipeline_inputs(self.graph, shifted_border)


if __name__ == "__main__":
    unittest.main()
