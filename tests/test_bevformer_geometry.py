from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_geometry import (
    BevProjectionConfig,
    build_bev_projection_graph,
    make_reference_grid,
    validate_projection_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


CONFIG = BevProjectionConfig(
    batch=1,
    cameras=2,
    bev_h=2,
    bev_w=3,
    z_anchors=2,
    image_h=20,
    image_w=30,
    point_cloud_range=(-3.0, -2.0, 1.0, 3.0, 2.0, 5.0),
)


# Independent oracle for CONFIG.  In the official implementation Z is the
# four-metre physical z extent, so two anchors are linspace(0.5, 3.5, 2), not
# the centers of two equal-height bins.
EXPECTED_GRID = np.asarray(
    [
        [
            [-2.0, -1.0, 1.5, 1.0],
            [0.0, -1.0, 1.5, 1.0],
            [2.0, -1.0, 1.5, 1.0],
            [-2.0, 1.0, 1.5, 1.0],
            [0.0, 1.0, 1.5, 1.0],
            [2.0, 1.0, 1.5, 1.0],
        ],
        [
            [-2.0, -1.0, 4.5, 1.0],
            [0.0, -1.0, 4.5, 1.0],
            [2.0, -1.0, 4.5, 1.0],
            [-2.0, 1.0, 4.5, 1.0],
            [0.0, 1.0, 4.5, 1.0],
            [2.0, 1.0, 4.5, 1.0],
        ],
    ],
    dtype=np.float32,
)


class BevFormerGeometryTests(unittest.TestCase):
    def test_reference_grid_matches_official_small_profile_oracle(self):
        grid = make_reference_grid(CONFIG)
        self.assertEqual(grid.shape, (2, 6, 4))
        np.testing.assert_array_equal(grid, EXPECTED_GRID)
        self.assertTrue(grid.flags.c_contiguous)

    def test_default_profile_uses_official_z_eight_normalization(self):
        config = BevProjectionConfig()
        self.assertEqual(config.reference_z_extent, 8.0)
        grid = make_reference_grid(config)
        np.testing.assert_allclose(
            grid[:, 0, 2],
            np.asarray([-4.5, -2.1666667, 0.1666667, 2.5], dtype=np.float32),
            rtol=0.0,
            atol=3e-7,
        )

        graph = build_bev_projection_graph(config)
        profile = graph.metadata["profile"]
        self.assertEqual(profile["reference_z_extent"], 8.0)
        self.assertEqual(
            profile["reference_z_normalization"],
            "linspace(0.5, Z - 0.5, z_anchors) / Z",
        )

    def test_projection_graph_matches_independent_numpy_oracle_and_compiles(self):
        graph = build_bev_projection_graph(CONFIG)
        self.assertEqual(graph.value_specs["reference_points"].shape, (1, 2, 6, 2, 2))
        self.assertEqual(graph.value_specs["visibility"].shape, (1, 2, 6, 1))
        lidar2img = np.asarray(
            [
                [
                    [[8, 0, 0, 12], [0, 8, 0, 10], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[-7, 0, 0, 15], [0, 6, 0, 9], [0, 0, 1, 0], [0, 0, 0, 1]],
                ]
            ],
            dtype=np.float32,
        )
        actual = reference_run(graph, {"lidar2img": lidar2img})

        homogeneous = EXPECTED_GRID
        projected = np.empty((1, 2, 2, 6, 4), dtype=np.float32)
        for batch in range(1):
            for camera in range(2):
                projected[batch, camera] = np.einsum(
                    "ij,dqj->dqi", lidar2img[batch, camera], homogeneous
                )
        depth = projected[..., 2:3]
        normalized = projected[..., :2] / np.maximum(depth, np.float32(CONFIG.epsilon))
        normalized /= np.asarray([CONFIG.image_w, CONFIG.image_h], dtype=np.float32)
        anchor_mask = (
            (depth > CONFIG.epsilon)
            & (normalized[..., 0:1] > 0)
            & (normalized[..., 0:1] < 1)
            & (normalized[..., 1:2] > 0)
            & (normalized[..., 1:2] < 1)
        )
        expected_reference = normalized.transpose(0, 1, 3, 2, 4)
        expected_visibility = (anchor_mask.sum(axis=2) > 0).astype(np.float32)
        np.testing.assert_allclose(actual["reference_points"], expected_reference, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(actual["visibility"], expected_visibility)

        program = compile_cuda(graph)
        self.assertEqual(len(program.launches), len(graph.nodes) - 1)
        self.assertEqual(len(program.describe()["zero_copy_views"]), 1)
        self.assertIn("? 1.0f : 0.0f", program.source)
        self.assertIn("int batch_index", program.source)

    def test_input_and_config_validation(self):
        graph = build_bev_projection_graph(CONFIG)
        good = np.zeros((1, 2, 4, 4), dtype=np.float32)
        validate_projection_inputs(graph, {"lidar2img": good})
        for values in (
            {},
            {"lidar2img": good.astype(np.float64)},
            {"lidar2img": np.full(good.shape, np.nan, dtype=np.float32)},
            {"lidar2img": np.zeros((1, 2, 4, 8), dtype=np.float32)[..., ::2]},
        ):
            with self.subTest(values=list(values)):
                with self.assertRaises(ValidationError):
                    validate_projection_inputs(graph, values)
        with self.assertRaises(ValidationError):
            BevProjectionConfig(bev_h=0)
        with self.assertRaises(ValidationError):
            BevProjectionConfig(point_cloud_range=(0, 0, 0, 0, 1, 1))


if __name__ == "__main__":
    unittest.main()
