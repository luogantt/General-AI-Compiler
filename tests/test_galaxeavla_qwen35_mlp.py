from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from gac.artifacts.weights import LoadedWeights, load_weights, write_weights
from gac.core.errors import ValidationError
from gac.core.ops import evaluate_node
from gac.models.api import get_model_adapter
from gac.models.galaxeavla.contract import (
    DOWN_PROJ_KEY,
    GATE_PROJ_KEY,
    SOURCE_WEIGHT_KEYS,
    UP_PROJ_KEY,
    ContractError,
    GalaxeaQwen35MLPContract,
)
from gac.models.galaxeavla.graph import build_qwen35_mlp_graph


def _source_weights(contract: GalaxeaQwen35MLPContract) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20261113)
    return {
        name: rng.normal(0.0, 0.2, shape).astype(np.float32)
        for name, shape in contract.source_weight_shapes.items()
    }


def _evaluate(graph, hidden_states):
    values = dict(graph.constants, hidden_states=hidden_states)
    for node in graph.nodes:
        values.update(evaluate_node(node, values))
    return values[graph.outputs[0]]


class GalaxeaQwen35MLPTests(unittest.TestCase):
    def test_default_contract_matches_action_expert_layer_zero(self):
        contract = GalaxeaQwen35MLPContract()
        self.assertEqual(contract.input_shape, (1, 32, 1024))
        self.assertEqual(contract.output_shape, (1, 32, 1024))
        self.assertEqual(contract.source_weight_shapes[GATE_PROJ_KEY], (4096, 1024))
        self.assertEqual(contract.source_weight_shapes[UP_PROJ_KEY], (4096, 1024))
        self.assertEqual(contract.source_weight_shapes[DOWN_PROJ_KEY], (1024, 4096))
        self.assertEqual(
            SOURCE_WEIGHT_KEYS,
            (
                "model.action_expert.layers.0.mlp.gate_proj.weight",
                "model.action_expert.layers.0.mlp.up_proj.weight",
                "model.action_expert.layers.0.mlp.down_proj.weight",
            ),
        )
        self.assertTrue(contract.official_g05_dimensions)
        self.assertEqual(
            contract.upstream_source_revision,
            "89f2322b4ad016e192437adc1a2c253b05bab246",
        )
        self.assertFalse(contract.to_dict()["full_model"])
        self.assertEqual(
            GalaxeaQwen35MLPContract.from_dict(contract.to_dict()), contract
        )

    def test_adapter_transposes_pytorch_weights_and_matches_swiglu_oracle(self):
        contract = GalaxeaQwen35MLPContract(
            batch=2, tokens=3, hidden_size=4, intermediate_size=7
        )
        weights = _source_weights(contract)
        snapshots = {name: value.copy() for name, value in weights.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = write_weights(
                Path(directory) / "layer0.gacweights",
                weights,
                source_checkpoint_sha256="a" * 64,
            )
            graph = get_model_adapter("galaxeavla-qwen35-mlp").load(
                path, contract=contract, expected_checkpoint_sha256="A" * 64
            )

        self.assertEqual(graph.constants["gate_proj_weight"].shape, (4, 7))
        self.assertEqual(graph.constants["up_proj_weight"].shape, (4, 7))
        self.assertEqual(graph.constants["down_proj_weight"].shape, (7, 4))
        np.testing.assert_array_equal(
            graph.constants["gate_proj_weight"], snapshots[GATE_PROJ_KEY].T
        )
        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(weights[name], snapshot)

        hidden = np.random.default_rng(97).normal(
            0.0, 0.3, contract.input_shape
        ).astype(np.float32)
        gate = hidden @ snapshots[GATE_PROJ_KEY].T
        activated = gate * (1.0 / (1.0 + np.exp(-gate)))
        up = hidden @ snapshots[UP_PROJ_KEY].T
        expected = (activated * up) @ snapshots[DOWN_PROJ_KEY].T
        np.testing.assert_allclose(
            _evaluate(graph, hidden), expected, rtol=2e-6, atol=2e-6
        )

        self.assertEqual(
            [node.op for node in graph.nodes],
            ["MatMul", "Sigmoid", "Mul", "MatMul", "Mul", "MatMul"],
        )
        self.assertEqual(graph.value_specs["output"].shape, contract.output_shape)
        self.assertEqual(graph.metadata["model_adapter"], "galaxeavla-qwen35-mlp")
        self.assertEqual(graph.metadata["source_checkpoint_sha256"], "a" * 64)
        self.assertFalse(graph.metadata["full_model"])
        self.assertFalse(graph.metadata["official_g05_dimensions"])
        self.assertTrue(
            all(
                source.startswith("model.action_expert.layers.0.mlp")
                for node in graph.nodes
                for source in node.source_nodes
            )
        )

    def test_build_accepts_verified_loaded_weights_and_ignores_unrelated_tensors(self):
        contract = GalaxeaQwen35MLPContract(
            batch=1, tokens=2, hidden_size=3, intermediate_size=5
        )
        weights = _source_weights(contract)
        weights["unrelated.weight"] = np.ones((2, 2), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = write_weights(
                Path(directory) / "slice.gacweights",
                weights,
                source_checkpoint_sha256="b" * 64,
            )
            loaded = load_weights(path)
            graph = get_model_adapter("galaxeavla-qwen35-mlp").build(
                loaded, contract=contract
            )
        self.assertEqual(graph.metadata["loaded_weight_count"], 4)
        self.assertEqual(graph.metadata["selected_weight_count"], 3)

    def test_invalid_contract_and_weight_inputs_are_rejected(self):
        for kwargs in (
            {"batch": 0},
            {"tokens": True},
            {"hidden_size": -1},
            {"intermediate_size": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ContractError):
                GalaxeaQwen35MLPContract(**kwargs)

        adapter = get_model_adapter("galaxeavla-qwen35-mlp")
        with self.assertRaisesRegex(ValidationError, "LoadedWeights"):
            adapter.build({})
        contract = GalaxeaQwen35MLPContract(
            batch=1, tokens=2, hidden_size=3, intermediate_size=5
        )
        with self.assertRaisesRegex(ValidationError, "contract"):
            adapter.build(
                LoadedWeights({}, "c" * 64, {}), contract=object()
            )
        with self.assertRaisesRegex(ValidationError, "expected digest"):
            adapter.build(
                LoadedWeights({}, "c" * 64, {}),
                expected_checkpoint_sha256="d" * 64,
            )
        with self.assertRaisesRegex(ValidationError, "64-character"):
            adapter.build(
                LoadedWeights({}, "c" * 64, {}),
                expected_checkpoint_sha256="not-a-digest",
            )

        cases = {
            "missing": {
                key: value
                for key, value in _source_weights(contract).items()
                if key != UP_PROJ_KEY
            },
            "wrong-shape": {
                **_source_weights(contract),
                GATE_PROJ_KEY: np.zeros((4, 3), dtype=np.float32),
            },
            "wrong-dtype": {
                **_source_weights(contract),
                DOWN_PROJ_KEY: np.zeros((3, 5), dtype=np.float64),
            },
        }
        for label, values in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = write_weights(
                    Path(directory) / f"{label}.gacweights",
                    values,
                    source_checkpoint_sha256="d" * 64,
                )
                with self.assertRaises(ValidationError):
                    adapter.build(load_weights(path), contract=contract)

    def test_graph_builder_rejects_noncanonical_weight_names(self):
        contract = GalaxeaQwen35MLPContract(
            batch=1, tokens=1, hidden_size=2, intermediate_size=3
        )
        canonical = {
            "gate_proj_weight": np.ones((2, 3), np.float32),
            "up_proj_weight": np.ones((2, 3), np.float32),
            "down_proj_weight": np.ones((3, 2), np.float32),
        }
        graph = build_qwen35_mlp_graph(canonical, contract)
        self.assertEqual(graph.value_specs["output"].shape, (1, 1, 2))
        canonical["extra"] = np.ones(1, np.float32)
        with self.assertRaisesRegex(ValidationError, "weight names"):
            build_qwen35_mlp_graph(canonical, contract)


if __name__ == "__main__":
    unittest.main()
