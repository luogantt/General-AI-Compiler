"""Tests for the dependency-free UniAD encoder FFN checkpoint contract."""

from __future__ import annotations

import unittest

import numpy as np

from gac.uniad.checkpoint import (
    CANONICAL_WEIGHT_SHAPES,
    CheckpointContractError,
    encoder_ffn_checkpoint_keys,
    extract_encoder_ffn_weights,
)


def make_state_dict(layer: int = 0) -> dict[str, np.ndarray]:
    keys = encoder_ffn_checkpoint_keys(layer)
    return {
        keys["fc1_weight"]: np.arange(512 * 256, dtype=np.float32).reshape(512, 256),
        keys["fc1_bias"]: np.arange(512, dtype=np.float32),
        keys["fc2_weight"]: np.arange(256 * 512, dtype=np.float32).reshape(256, 512),
        keys["fc2_bias"]: np.arange(256, dtype=np.float32),
        keys["norm_weight"]: np.linspace(0.8, 1.2, 256, dtype=np.float32),
        keys["norm_bias"]: np.linspace(-0.1, 0.1, 256, dtype=np.float32),
    }


class UniADCheckpointContractTests(unittest.TestCase):
    def test_all_six_encoder_layers_resolve_exact_keys(self):
        for layer in range(6):
            with self.subTest(layer=layer):
                keys = encoder_ffn_checkpoint_keys(layer)
                prefix = f"pts_bbox_head.transformer.encoder.layers.{layer}."
                self.assertEqual(
                    keys["fc1_weight"], prefix + "ffns.0.layers.0.0.weight"
                )
                self.assertEqual(
                    keys["fc2_weight"], prefix + "ffns.0.layers.1.weight"
                )
                self.assertEqual(keys["norm_weight"], prefix + "norms.2.weight")

    def test_extract_transposes_linear_weights_and_copies_every_parameter(self):
        state = make_state_dict(layer=3)
        keys = encoder_ffn_checkpoint_keys(3)
        result = extract_encoder_ffn_weights(state, layer=3)

        self.assertEqual(set(result), set(CANONICAL_WEIGHT_SHAPES))
        for name, expected_shape in CANONICAL_WEIGHT_SHAPES.items():
            self.assertEqual(result[name].shape, expected_shape)
            self.assertEqual(result[name].dtype, np.float32)
            self.assertTrue(result[name].flags.c_contiguous)
            self.assertTrue(result[name].flags.owndata)

        np.testing.assert_array_equal(
            result["fc1_weight"], state[keys["fc1_weight"]].T
        )
        np.testing.assert_array_equal(
            result["fc2_weight"], state[keys["fc2_weight"]].T
        )
        np.testing.assert_array_equal(result["fc1_bias"], state[keys["fc1_bias"]])

        before = result["fc1_bias"].copy()
        state[keys["fc1_bias"]].fill(-1.0)
        np.testing.assert_array_equal(result["fc1_bias"], before)

    def test_extra_checkpoint_parameters_are_ignored(self):
        state = make_state_dict()
        state["unrelated.weight"] = np.ones((2, 2), dtype=np.float32)
        self.assertEqual(
            set(extract_encoder_ffn_weights(state, 0)), set(CANONICAL_WEIGHT_SHAPES)
        )

    def test_rejects_invalid_layer(self):
        for layer in (-1, 6, True, 1.5, "0"):
            with self.subTest(layer=layer):
                with self.assertRaises(CheckpointContractError):
                    encoder_ffn_checkpoint_keys(layer)  # type: ignore[arg-type]

    def test_rejects_missing_parameter(self):
        state = make_state_dict(layer=1)
        missing = encoder_ffn_checkpoint_keys(1)["norm_bias"]
        state.pop(missing)
        with self.assertRaisesRegex(CheckpointContractError, "missing.*norms.2.bias"):
            extract_encoder_ffn_weights(state, 1)

    def test_rejects_non_array_wrong_shape_and_wrong_dtype(self):
        keys = encoder_ffn_checkpoint_keys(0)
        cases = [
            ("fc1_bias", [0.0] * 512),
            ("fc2_bias", np.zeros((255,), dtype=np.float32)),
            ("norm_weight", np.ones((256,), dtype=np.float64)),
        ]
        for canonical_name, invalid in cases:
            with self.subTest(parameter=canonical_name):
                state = make_state_dict()
                state[keys[canonical_name]] = invalid  # type: ignore[assignment]
                with self.assertRaises(CheckpointContractError):
                    extract_encoder_ffn_weights(state, 0)

    def test_rejects_nonfinite_parameter(self):
        state = make_state_dict(layer=5)
        key = encoder_ffn_checkpoint_keys(5)["fc2_weight"]
        state[key][2, 3] = np.inf
        with self.assertRaisesRegex(CheckpointContractError, "finite"):
            extract_encoder_ffn_weights(state, 5)

    def test_rejects_non_mapping_state_dict(self):
        with self.assertRaisesRegex(CheckpointContractError, "must be a mapping"):
            extract_encoder_ffn_weights([], 0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
