from __future__ import annotations

import unittest

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.bevformer_encoder_inputs import (
    BevFormerEncoderInputConfig,
    build_bevformer_encoder_input_graph,
    encoder_input_weight_shapes,
    validate_encoder_input_values,
)
from gac.errors import ValidationError
from gac.runtime import reference_run


def make_weights(config: BevFormerEncoderInputConfig, seed: int = 5):
    rng = np.random.default_rng(seed)
    return {
        name: np.ascontiguousarray(rng.normal(0, 0.2, shape).astype(np.float32))
        for name, shape in encoder_input_weight_shapes(config).items()
    }


class BevFormerEncoderInputTests(unittest.TestCase):
    def setUp(self):
        self.config = BevFormerEncoderInputConfig(
            bev_h=2,
            bev_w=3,
            embed_dim=8,
            can_bus_dim=4,
            can_bus_hidden_dim=5,
        )
        self.weights = make_weights(self.config)
        self.graph = build_bevformer_encoder_input_graph(self.weights, self.config)

    def test_query_mlp_static_position_and_reference_grid(self):
        can_bus = np.asarray([[0.2, -0.1, 0.4, 0.3]], dtype=np.float32)
        actual = reference_run(self.graph, {"can_bus_delta": can_bus})
        hidden = np.maximum(
            can_bus @ self.weights["can_bus_fc0_weight"]
            + self.weights["can_bus_fc0_bias"],
            0,
        )
        embedded = np.maximum(
            hidden @ self.weights["can_bus_fc1_weight"]
            + self.weights["can_bus_fc1_bias"],
            0,
        )
        mean = embedded.mean(axis=-1, keepdims=True)
        variance = ((embedded - mean) ** 2).mean(axis=-1, keepdims=True)
        embedded = (embedded - mean) / np.sqrt(variance + self.config.norm_epsilon)
        embedded = (
            embedded * self.weights["can_bus_norm_weight"]
            + self.weights["can_bus_norm_bias"]
        )
        expected_query = self.weights["bev_embedding"][None] + embedded[:, None]
        np.testing.assert_allclose(actual["query"], expected_query, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(actual["spatial_query_pos"], 0.0)
        expected_position = np.concatenate(
            (
                np.broadcast_to(
                    self.weights["col_embedding"][None], (2, 3, 4)
                ),
                np.broadcast_to(
                    self.weights["row_embedding"][:, None], (2, 3, 4)
                ),
            ),
            axis=-1,
        ).reshape(1, 6, 8)
        np.testing.assert_array_equal(actual["query_pos"], expected_position)
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
        np.testing.assert_allclose(actual["reference_points"], expected_reference)
        self.assertEqual(actual["query_pos"].shape, (1, 6, 8))

    def test_default_profile_compiles_for_sm86(self):
        config = BevFormerEncoderInputConfig()
        zero = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in encoder_input_weight_shapes(config).items()
        }
        program = compile_cuda(build_bevformer_encoder_input_graph(zero, config))
        self.assertEqual(len(program.launches), 6)
        self.assertEqual(sum(node.op == "AddReLU" for node in program.graph.nodes), 2)
        self.assertEqual(program.reference_graph.outputs[0], "query")

    def test_validation_rejects_wrong_contract(self):
        value = np.zeros((1, 4), dtype=np.float32)
        validate_encoder_input_values(self.graph, {"can_bus_delta": value})
        with self.assertRaisesRegex(ValidationError, "requires"):
            validate_encoder_input_values(
                self.graph, {"can_bus_delta": value.astype(np.float16)}
            )
        bad = dict(self.weights)
        bad.pop("bev_embedding")
        with self.assertRaisesRegex(ValidationError, "weight names"):
            build_bevformer_encoder_input_graph(bad, self.config)


if __name__ == "__main__":
    unittest.main()
