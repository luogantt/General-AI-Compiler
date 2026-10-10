from __future__ import annotations

import math
import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_motion import (
    BevFormerMotionConfig,
    build_bevformer_motion_graph,
    validate_bevformer_motion_inputs,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def _torchvision_0101_rotate_nearest(
    prev_bev: np.ndarray,
    *,
    bev_h: int,
    bev_w: int,
    angle_degrees: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Independent NumPy transcription of torchvision 0.10.1 rotate.

    This follows ``functional.rotate -> _get_inverse_affine_matrix`` and
    ``functional_tensor.rotate -> _gen_affine_grid -> grid_sample``.  The
    official default interpolation is nearest and its nearest index uses
    ``nearbyint`` (round-to-nearest-even).
    """

    center_f = (center[0] - bev_w * 0.5, center[1] - bev_h * 0.5)
    # functional.rotate calls _get_inverse_affine_matrix with -angle.
    rotation = math.radians(-float(angle_degrees))
    a = math.cos(rotation)
    b = -math.sin(rotation)
    c = math.sin(rotation)
    d = math.cos(rotation)
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix[2] += matrix[0] * -center_f[0] + matrix[1] * -center_f[1]
    matrix[5] += matrix[3] * -center_f[0] + matrix[4] * -center_f[1]
    matrix[2] += center_f[0]
    matrix[5] += center_f[1]
    theta = np.asarray(matrix, dtype=np.float32).reshape(2, 3)

    x = np.linspace(
        -bev_w * 0.5 + 0.5,
        bev_w * 0.5 + 0.5 - 1.0,
        num=bev_w,
        dtype=np.float32,
    )
    y = np.linspace(
        -bev_h * 0.5 + 0.5,
        bev_h * 0.5 + 0.5 - 1.0,
        num=bev_h,
        dtype=np.float32,
    )
    base = np.ones((bev_h, bev_w, 3), dtype=np.float32)
    base[..., 0] = x
    base[..., 1] = y[:, None]
    grid = base.reshape(-1, 3) @ theta.T
    grid /= np.asarray([0.5 * bev_w, 0.5 * bev_h], dtype=np.float32)
    grid = grid.reshape(bev_h, bev_w, 2)

    image = prev_bev.reshape(bev_h, bev_w, prev_bev.shape[-1])
    output = np.zeros_like(image)
    for out_y in range(bev_h):
        for out_x in range(bev_w):
            grid_x, grid_y = grid[out_y, out_x]
            source_x = (grid_x + np.float32(1.0)) * bev_w * 0.5 - 0.5
            source_y = (grid_y + np.float32(1.0)) * bev_h * 0.5 - 0.5
            source_x = int(np.rint(source_x))
            source_y = int(np.rint(source_y))
            if 0 <= source_x < bev_w and 0 <= source_y < bev_h:
                output[out_y, out_x] = image[source_y, source_x]
    return output.reshape(prev_bev.shape)


def _official_shifted_reference(
    config: BevFormerMotionConfig, can_bus_delta: np.ndarray
) -> np.ndarray:
    y = (np.arange(config.bev_h, dtype=np.float32) + 0.5) / config.bev_h
    x = (np.arange(config.bev_w, dtype=np.float32) + 0.5) / config.bev_w
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    reference = np.stack((grid_x, grid_y), axis=-1).reshape(
        1, config.queries, 1, 2
    )

    delta_x = float(can_bus_delta[0, 0])
    delta_y = float(can_bus_delta[0, 1])
    ego_angle_degrees = float(can_bus_delta[0, -2]) / math.pi * 180.0
    translation_length = math.sqrt(delta_x**2 + delta_y**2)
    translation_angle = math.atan2(delta_y, delta_x) / math.pi * 180.0
    bev_angle = ego_angle_degrees - translation_angle
    shift_y = (
        translation_length
        * math.cos(bev_angle / 180.0 * math.pi)
        / config.grid_length_y
        / config.bev_h
    )
    shift_x = (
        translation_length
        * math.sin(bev_angle / 180.0 * math.pi)
        / config.grid_length_x
        / config.bev_w
    )
    return reference + np.asarray([shift_x, shift_y], dtype=np.float32).reshape(
        1, 1, 1, 2
    )


class BevFormerMotionTests(unittest.TestCase):
    def test_zero_motion_is_identity_and_returns_cell_centers(self):
        config = BevFormerMotionConfig(
            bev_h=2, bev_w=3, embed_dim=2, rotate_center=(100.0, 100.0)
        )
        graph = build_bevformer_motion_graph(config)
        prev_bev = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
        can_bus = np.zeros((1, 18), dtype=np.float32)
        actual = reference_run(
            graph, {"prev_bev": prev_bev, "can_bus_delta": can_bus}
        )
        np.testing.assert_array_equal(actual["rotated_prev_bev"], prev_bev)
        expected_reference = np.asarray(
            [
                [1 / 6, 1 / 4],
                [3 / 6, 1 / 4],
                [5 / 6, 1 / 4],
                [1 / 6, 3 / 4],
                [3 / 6, 3 / 4],
                [5 / 6, 3 / 4],
            ],
            dtype=np.float32,
        ).reshape(1, 6, 1, 2)
        np.testing.assert_allclose(
            actual["shifted_reference_points"], expected_reference, atol=1e-7
        )

    def test_motion_matches_official_shift_and_torchvision_rotate_oracles(self):
        # Keeping the official (100,100) center on a tiny grid makes a center
        # interpretation bug immediately visible while a small angle retains
        # enough in-bounds samples for a useful comparison.
        config = BevFormerMotionConfig(
            bev_h=4, bev_w=5, embed_dim=3, rotate_center=(100.0, 100.0)
        )
        graph = build_bevformer_motion_graph(config)
        prev_bev = np.arange(60, dtype=np.float32).reshape(1, 20, 3)
        can_bus = np.zeros((1, 18), dtype=np.float32)
        can_bus[0, 0] = np.float32(1.2)
        can_bus[0, 1] = np.float32(-0.3)
        can_bus[0, -2] = np.float32(0.4)
        can_bus[0, -1] = np.float32(0.5)
        actual = reference_run(
            graph, {"prev_bev": prev_bev, "can_bus_delta": can_bus}
        )

        expected_rotated = _torchvision_0101_rotate_nearest(
            prev_bev,
            bev_h=config.bev_h,
            bev_w=config.bev_w,
            angle_degrees=float(can_bus[0, -1]),
            center=config.rotate_center,
        )
        np.testing.assert_array_equal(actual["rotated_prev_bev"], expected_rotated)
        np.testing.assert_allclose(
            actual["shifted_reference_points"],
            _official_shifted_reference(config, can_bus),
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertEqual(
            graph.metadata["rotation"]["center_xy_upper_left_origin"],
            [100.0, 100.0],
        )
        self.assertIn("half-pixel", graph.metadata["numeric_boundary"])

    def test_default_profile_shape_inference_and_sm86_offline_compile(self):
        config = BevFormerMotionConfig()
        self.assertAlmostEqual(config.grid_length_y * config.bev_h, 102.4)
        self.assertAlmostEqual(config.grid_length_x * config.bev_w, 102.4)
        graph = build_bevformer_motion_graph(config)
        self.assertEqual(
            graph.value_specs["rotated_prev_bev"].shape, (1, 2500, 256)
        )
        self.assertEqual(
            graph.value_specs["shifted_reference_points"].shape,
            (1, 2500, 1, 2),
        )
        program = compile_cuda(graph)
        self.assertEqual(program.describe()["compute_capability"], [8, 6])
        self.assertIn("nearbyintf", program.source)
        self.assertIn("sinf", program.source)
        self.assertIn("cosf", program.source)
        rotate = next(
            node for node in program.reference_graph.nodes if node.id == "rotate_prev_bev"
        )
        self.assertEqual(rotate.attrs["mode"], "nearest")
        self.assertFalse(rotate.attrs["align_corners"])

    def test_strict_input_and_config_validation(self):
        config = BevFormerMotionConfig(bev_h=2, bev_w=3, embed_dim=2)
        graph = build_bevformer_motion_graph(config)
        values = {
            "prev_bev": np.zeros((1, 6, 2), dtype=np.float32),
            "can_bus_delta": np.zeros((1, 18), dtype=np.float32),
        }
        validate_bevformer_motion_inputs(graph, values)
        wrong_dtype = dict(values)
        wrong_dtype["prev_bev"] = values["prev_bev"].astype(np.float16)
        with self.assertRaisesRegex(ValidationError, "requires"):
            validate_bevformer_motion_inputs(graph, wrong_dtype)
        non_finite = dict(values)
        non_finite["can_bus_delta"] = values["can_bus_delta"].copy()
        non_finite["can_bus_delta"][0, 0] = np.nan
        with self.assertRaisesRegex(ValidationError, "contiguous and finite"):
            validate_bevformer_motion_inputs(graph, non_finite)
        with self.assertRaisesRegex(ValidationError, "exactly"):
            validate_bevformer_motion_inputs(graph, {"prev_bev": values["prev_bev"]})

        for kwargs, pattern in (
            ({"batch": 2}, "batch=1"),
            ({"can_bus_dim": 17}, "18-element"),
            ({"grid_length_x": 0.0}, "grid_length_x"),
            ({"grid_length_x": 1e-100}, "grid_length_x"),
            ({"grid_length_x": 1e100}, "grid_length_x"),
            ({"grid_length_x": np.finfo(np.float32).max}, "grid size"),
            ({"rotate_center": (100.0,)}, "rotate_center"),
            ({"rotate_center": (1e100, 100.0)}, "rotate_center"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValidationError, pattern
            ):
                BevFormerMotionConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
