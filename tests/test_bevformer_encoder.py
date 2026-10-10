from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_encoder import build_single_frame_spatial_stage_graph
from gac.bevformer_geometry import BevProjectionConfig, build_bev_projection_graph
from gac.bevformer_spatial import (
    SpatialCrossAttentionConfig,
    build_spatial_cross_attention_graph,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


PROJECTION = BevProjectionConfig(
    cameras=2,
    bev_h=2,
    bev_w=2,
    z_anchors=2,
    image_h=20,
    image_w=30,
    point_cloud_range=(-2, -2, 1, 2, 2, 5),
)
SPATIAL = SpatialCrossAttentionConfig(
    cameras=2,
    queries=4,
    embed_dim=4,
    heads=2,
    spatial_shapes=((2, 2),),
    sampling_points=4,
    z_anchors=2,
)


def make_weights(seed=73):
    rng = np.random.default_rng(seed)
    return {
        "value_weight": rng.normal(0, 0.1, (4, 4)).astype(np.float32),
        "value_bias": rng.normal(0, 0.1, (4,)).astype(np.float32),
        "offset_weight": rng.normal(0, 0.1, (4, 16)).astype(np.float32),
        "offset_bias": rng.normal(0, 0.1, (16,)).astype(np.float32),
        "attention_weight": rng.normal(0, 0.1, (4, 8)).astype(np.float32),
        "attention_bias": rng.normal(0, 0.1, (8,)).astype(np.float32),
        "output_weight": rng.normal(0, 0.1, (4, 4)).astype(np.float32),
        "output_bias": rng.normal(0, 0.1, (4,)).astype(np.float32),
    }


class BevFormerEncoderCompositionTests(unittest.TestCase):
    def test_composed_projection_and_spatial_stage_matches_sequential_graphs(self):
        weights = make_weights()
        graph = build_single_frame_spatial_stage_graph(
            weights,
            projection_config=PROJECTION,
            spatial_config=SPATIAL,
        )
        rng = np.random.default_rng(79)
        query = rng.normal(size=(1, 4, 4)).astype(np.float32)
        query_pos = rng.normal(size=(1, 4, 4)).astype(np.float32)
        value = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
        lidar2img = np.asarray(
            [
                [
                    [[7, 0, 0, 12], [0, 7, 0, 10], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[-6, 0, 0, 15], [0, 6, 0, 9], [0, 0, 1, 0], [0, 0, 0, 1]],
                ]
            ],
            dtype=np.float32,
        )
        geometry = reference_run(
            build_bev_projection_graph(PROJECTION), {"lidar2img": lidar2img}
        )
        expected = reference_run(
            build_spatial_cross_attention_graph(weights, SPATIAL),
            {
                "query": query,
                "query_pos": query_pos,
                "value": value,
                "reference_points": geometry["reference_points"],
                "visibility": geometry["visibility"],
            },
        )["output"]
        actual = reference_run(
            graph,
            {
                "query": query,
                "query_pos": query_pos,
                "value": value,
                "lidar2img": lidar2img,
            },
        )["output"]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertFalse(graph.metadata["full_encoder_layer"])
        self.assertTrue(all(node.id.startswith(("projection.", "spatial_attention.")) for node in graph.nodes))
        program = compile_cuda(graph)
        self.assertEqual(
            len(program.launches) + len(program.describe()["zero_copy_views"]),
            len(program.graph.nodes),
        )

    def test_mismatched_projection_and_attention_profiles_are_rejected(self):
        cases = [
            SpatialCrossAttentionConfig(cameras=3),
            SpatialCrossAttentionConfig(queries=5),
            SpatialCrossAttentionConfig(z_anchors=2, sampling_points=8),
        ]
        for spatial in cases:
            with self.subTest(spatial=spatial), self.assertRaises(ValidationError):
                build_single_frame_spatial_stage_graph(
                    make_weights(),
                    projection_config=PROJECTION,
                    spatial_config=spatial,
                )


if __name__ == "__main__":
    unittest.main()
