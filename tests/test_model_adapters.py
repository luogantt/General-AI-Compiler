from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.core.errors import ValidationError
from gac.core.ir import Graph, Node, TensorSpec
from gac.core.ops import evaluate_node
from gac.models.api import ModelAdapter, get_model_adapter, list_model_adapters, load_model_graph, register_model_adapter
from gac.models.common.ffn import TransformerFFNConfig, build_transformer_ffn_graph, ffn_weight_shapes


def _evaluate(graph, query):
    values = dict(graph.constants, query=query)
    for node in graph.nodes:
        values.update(evaluate_node(node, values))
    return values[graph.outputs[0]]


def _weights(shapes):
    rng = np.random.default_rng(94)
    return {name: (np.ones(shape, np.float32) if name == "norm_weight" else
                   rng.normal(0.0, 0.1, shape).astype(np.float32)) for name, shape in shapes.items()}


class ModelAdapterTests(unittest.TestCase):
    def test_registry_is_lazy_and_extension_uses_common_graph(self):
        calls = []

        class ToyAdapter:
            name = "unit-test-model"
            scope = "identity-test"

            def load(self, source, **options):
                return self.build(source, **options)

            def build(self, data, **options):
                return Graph("toy", {"x": TensorSpec((1,), "float32")}, ("y",),
                             [Node("copy", "Identity", ("x",), ("y",))])

        def factory():
            calls.append("created")
            return ToyAdapter()

        from gac.models import api
        with patch.dict(api._FACTORIES):
            register_model_adapter("unit-test-model", factory)
            self.assertEqual(calls, [])
            self.assertIn("unit-test-model", list_model_adapters())
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_model_adapter("unit-test-model", factory)
            adapter = get_model_adapter("unit-test-model")
            self.assertIsInstance(adapter, ModelAdapter)
            self.assertEqual(calls, ["created"])
            self.assertIsInstance(load_model_graph("unit-test-model", "unused"), Graph)
        with self.assertRaisesRegex(ValueError, "unknown model adapter"):
            get_model_adapter("vla")
        with self.assertRaises(TypeError):
            register_model_adapter("bad", None)

    def test_registry_validates_factory_result_and_explicit_replacement(self):
        from gac.models import api

        with patch.dict(api._FACTORIES):
            register_model_adapter("bad-result", lambda: object())
            with self.assertRaisesRegex(TypeError, "matching ModelAdapter"):
                get_model_adapter("bad-result")
            register_model_adapter("bad-result", lambda: get_model_adapter("uniad-ffn"), replace=True)
            with self.assertRaisesRegex(TypeError, "matching ModelAdapter"):
                get_model_adapter("bad-result")
        for name in ("", "Uppercase", "spaces bad", "../module"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                register_model_adapter(name, lambda: None)

    def test_builtin_adapters_reject_unknown_options_before_io(self):
        for name in ("bevformer-step", "uniad-ffn"):
            adapter = get_model_adapter(name)
            with self.subTest(name=name), self.assertRaises(TypeError):
                adapter.load("nonexistent", unknown_option=True)
            with self.subTest(name=name), self.assertRaises(TypeError):
                adapter.build(None, unknown_option=True)

    def test_common_loader_checks_graph_result_contract(self):
        from gac.models import api

        class InvalidGraphAdapter:
            name = "invalid-graph"
            scope = "unit-test"

            def load(self, source, **options):
                return None

            def build(self, data, **options):
                return None

        with patch.dict(api._FACTORIES):
            register_model_adapter("invalid-graph", InvalidGraphAdapter)
            with self.assertRaisesRegex(TypeError, "did not return a Graph"):
                load_model_graph("invalid-graph", "unused")

    def test_registry_and_selected_uniad_adapter_do_not_import_bevformer_or_targets(self):
        sources = [
            "from gac.models.api import list_model_adapters; assert list_model_adapters()==('bevformer-step','uniad-ffn'); "
            "blocked=('numpy','gac.targets','gac.execution','gac.models.bevformer','gac.models.uniad')",
            "from gac.models.api import get_model_adapter; get_model_adapter('uniad-ffn'); "
            "blocked=('torch','cupy','gac.targets','gac.execution','gac.models.bevformer')",
            "from gac.models.api import get_model_adapter; get_model_adapter('bevformer-step'); "
            "blocked=('numpy','torch','cupy','gac.targets','gac.execution','gac.models.uniad')",
        ]
        for code in sources:
            with self.subTest(code=code):
                code += "; import sys; assert not [m for m in sys.modules if any(m==p or m.startswith(p+'.') for p in blocked)]"
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_uniad_bundle_build_and_load_match_independent_ffn_oracle(self):
        from gac.models.uniad.artifact import write_bundle
        from gac.models.uniad.contract import UniADFFNContract

        contract = UniADFFNContract(layer=2, batch=1, bev_h=2, bev_w=3, embed_dim=4, hidden_dim=8)
        weights = _weights(contract.weight_shapes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uniad.bundle"
            data = write_bundle(path, contract, weights)
            adapter = get_model_adapter("uniad-ffn")
            built, loaded = adapter.build(data), adapter.load(path)
            query = np.random.default_rng(41).normal(0.0, 0.3, contract.input_shape).astype(np.float32)
            hidden = np.maximum(query @ weights["fc1_weight"] + weights["fc1_bias"], 0)
            residual = query + hidden @ weights["fc2_weight"] + weights["fc2_bias"]
            mean = residual.mean(axis=-1, keepdims=True)
            expected = (residual - mean) / np.sqrt(((residual - mean)**2).mean(axis=-1, keepdims=True) + contract.layer_norm_epsilon)
            expected = expected * weights["norm_weight"] + weights["norm_bias"]
            np.testing.assert_allclose(_evaluate(built, query), expected, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(_evaluate(loaded, query), expected, rtol=2e-5, atol=2e-6)
            self.assertEqual(loaded.metadata["source_bundle_sha256"], sha256(path.read_bytes()).hexdigest())
            self.assertEqual(built.metadata["model_scope"], "stage2-bev-encoder-single-layer-ffn")
            self.assertFalse(built.metadata["full_model"])
            self.assertTrue(all(source.startswith("UniAD.") for node in built.nodes for source in node.source_nodes))
            with self.assertRaisesRegex(ValidationError, "UniADFFNBundle"):
                adapter.build(weights)

    def test_bevformer_adapter_builds_complete_graph_from_loaded_weights(self):
        from gac.artifacts.weights import load_weights, write_weights
        from tests.test_bevformer_step_pipeline import small_config, make_weights

        config = small_config()
        groups = make_weights(config)
        adapter = get_model_adapter("bevformer-step")
        with tempfile.TemporaryDirectory() as directory:
            path = write_weights(Path(directory) / "step.gacweights", {"fixture": np.ones(1, np.float32)},
                                 source_checkpoint_sha256="a" * 64)
            loaded = load_weights(path)
            with patch("gac.models.bevformer.frontends.graph._weight_groups", return_value=groups):
                built = adapter.build(loaded, config=config)
                from_file = adapter.load(path, config=config, expected_checkpoint_sha256="A" * 64)
            self.assertGreater(len(built.nodes), 1000)
            self.assertEqual([node.to_dict() for node in built.nodes], [node.to_dict() for node in from_file.nodes])
            self.assertEqual(built.metadata["source_checkpoint_sha256"], "a" * 64)
            self.assertEqual(built.metadata["model_family"], "bevformer")
            self.assertEqual(built.metadata["model_adapter"], "bevformer-step")
            self.assertTrue(built.metadata["full_raw_detection_head"])
            self.assertIn("next_bev", built.outputs)
            with self.assertRaisesRegex(ValidationError, "LoadedWeights"):
                adapter.build({})


class SharedFFNTests(unittest.TestCase):
    def test_common_ffn_accepts_model_independent_hidden_width(self):
        config = TransformerFFNConfig(batch=1, tokens=3, embed_dim=4, hidden_dim=7, post_layer_norm=False)
        weights = _weights(ffn_weight_shapes(config))
        graph = build_transformer_ffn_graph(weights, config)
        query = np.random.default_rng(97).normal(size=(1, 3, 4)).astype(np.float32)
        expected = query + np.maximum(query @ weights["fc1_weight"] + weights["fc1_bias"], 0) @ weights["fc2_weight"] + weights["fc2_bias"]
        np.testing.assert_allclose(_evaluate(graph, query), expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(graph.value_specs["activated"].shape, (1, 3, 7))
        snapshot = graph.constants["fc1_weight"].copy()
        weights["fc1_weight"].fill(55)
        np.testing.assert_array_equal(graph.constants["fc1_weight"], snapshot)


if __name__ == "__main__":
    unittest.main()
