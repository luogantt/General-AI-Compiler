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
from gac.bevformer_encoder_pipeline import (
    BevFormerColdEncoderPipelineConfig,
    build_bevformer_cold_encoder_pipeline_graph,
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
from gac.bevformer_spatial import SpatialCrossAttentionConfig
from gac.bevformer_temporal import TemporalSelfAttentionConfig
from gac.errors import ValidationError
from gac.frontends.bevformer_backbone import (
    ResNetFpnProfile,
    build_bevformer_backbone_graph,
    required_checkpoint_specs,
)
from gac.runtime import reference_run


def small_config() -> BevFormerColdEncoderPipelineConfig:
    backbone = ResNetFpnProfile(
        name="test-r50-shape",
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
    return BevFormerColdEncoderPipelineConfig(
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
            layer=layer, history_mode=False
        ),
    )


def random_weights(config: BevFormerColdEncoderPipelineConfig, seed: int = 17):
    rng = np.random.default_rng(seed)

    def arrays(shapes, scale=0.05):
        return {
            name: np.ascontiguousarray(rng.normal(0, scale, shape).astype(np.float32))
            for name, shape in shapes.items()
        }

    backbone = {
        spec.key: np.ascontiguousarray(
            (
                np.ones(spec.shape, dtype=np.float32)
                if spec.role == "bn-running-variance"
                else rng.normal(0, 0.03, spec.shape).astype(np.float32)
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


class BevFormerColdEncoderPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = small_config()
        self.weights = random_weights(self.config)
        self.graph = build_bevformer_cold_encoder_pipeline_graph(
            *self.weights, self.config
        )
        self.inputs = {
            "images": np.ascontiguousarray(
                np.linspace(
                    -0.5,
                    0.5,
                    np.prod(self.config.backbone.input_shape),
                    dtype=np.float32,
                ).reshape(
                    self.config.feature_prep.batch,
                    self.config.feature_prep.cameras,
                    self.config.backbone.input_channels,
                    *self.config.backbone.input_hw,
                )
            ),
            "lidar2img": np.ascontiguousarray(
                np.broadcast_to(np.eye(4, dtype=np.float32), (1, 2, 4, 4)).copy()
            ),
            "can_bus_delta": np.asarray([[0.1, 0.2, -0.1, 0.3]], dtype=np.float32),
        }

    def test_composition_matches_independent_stage_execution(self):
        backbone_weights, feature_weights, input_weights, temporal, spatial, layer = self.weights
        backbone = build_bevformer_backbone_graph(
            backbone_weights, self.config.backbone
        )
        fpn = reference_run(
            backbone,
            {"images": self.inputs["images"].reshape(self.config.backbone.input_shape)},
        )["fpn_c5"]
        feature = reference_run(
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
        stack_inputs = {
            "query": initialized["query"],
            "query_pos": initialized["query_pos"],
            "spatial_query_pos": initialized["spatial_query_pos"],
            "reference_points": initialized["reference_points"],
            "image_value": feature,
            "lidar2img": self.inputs["lidar2img"],
        }
        expected = reference_run(
            build_bevformer_encoder_stack_graph(
                temporal, spatial, layer, self.config.encoder_stack
            ),
            stack_inputs,
        )["output"]
        actual = reference_run(self.graph, self.inputs)["next_bev"]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        self.assertTrue(self.graph.metadata["full_encoder_stack"])
        self.assertTrue(self.graph.metadata["cold_start_only"])
        self.assertEqual(self.graph.inputs["images"].shape, (1, 2, 3, 8, 8))

    def test_small_pipeline_compiles_for_sm86(self):
        program = compile_cuda(self.graph)
        self.assertGreater(len(program.launches), 50)
        self.assertEqual(
            program.reference_graph.value_specs["next_bev"].shape, (1, 2, 4)
        )

    def test_profile_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "component profiles disagree"):
            BevFormerColdEncoderPipelineConfig(
                backbone=self.config.backbone,
                feature_prep=BevFormerFeaturePrepConfig(
                    cameras=2, embed_dim=8, feature_h=1, feature_w=1
                ),
                encoder_inputs=self.config.encoder_inputs,
                encoder_stack=self.config.encoder_stack,
            )


if __name__ == "__main__":
    unittest.main()
