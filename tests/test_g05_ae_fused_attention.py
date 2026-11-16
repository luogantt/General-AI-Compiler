"""No torch/NPU required: strict masks, dispatch polarity and isolated restore."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools" / "galaxeavla" / "g05_ae_fused_attention.py"
SPEC = importlib.util.spec_from_file_location("g05_ae_fused_attention", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


class Tensor:
    item_reads = 0

    def __init__(self, data, dtype="float32", device="npu"):
        self.data = np.asarray(data)
        self.dtype, self.device = dtype, SimpleNamespace(type=device, index=0)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def result(self, value, dtype="bool"):
        return Tensor(value, dtype, self.device.type)

    def __eq__(self, value):
        return self.result(self.data == value)

    def __ne__(self, value):
        return self.result(self.data != value)

    def __and__(self, value):
        return self.result(self.data & value.data)

    def __or__(self, value):
        return self.result(self.data | value.data)

    def all(self):
        return self.result(self.data.all())

    def any(self, dim):
        return self.result(self.data.any(axis=dim))

    def item(self):
        Tensor.item_reads += 1
        return self.data.item()

    def __getitem__(self, key):
        return self.result(self.data[key], self.dtype)

    def contiguous(self):
        return self

    def transpose(self, dim1, dim2):
        return self.result(self.data.swapaxes(dim1, dim2), self.dtype)


def eager_attention_forward(*args, **kwargs):
    return "original", None


ALL_ATTENTION_FUNCTIONS = {}


class Decoder:
    def __init__(self):
        self.training = False
        self.self_attn = SimpleNamespace(num_heads=8, num_key_value_heads=2, head_dim=256)

    def forward(self, module, query, key, value, mask, scaling):
        # Mimic only the audited attention lookup to test private globals.
        return ALL_ATTENTION_FUNCTIONS.get("eager", eager_attention_forward)(
            module, query, key, value, mask, scaling=scaling)


class Mixture:
    def __init__(self):
        self.training = False
        self.layers = [Decoder() for _ in range(24)]
        self.layer_types = ["full_attention"] * 24


class Model:
    def __init__(self, mask):
        self.training = False
        self.action_expert = Mixture()
        self.fm_helper = SimpleNamespace(action_causal=False)
        self.attn_implementation = "eager"
        self.mask = mask

    def build_action_mask_and_position_ids(self):
        return self.mask, "original_position_ids"


class FusedAttentionTests(unittest.TestCase):
    def setUp(self):
        Tensor.item_reads = 0
        self.calls = []

        def fused(query, key, value, **kwargs):
            self.calls.append((query, key, value, kwargs))
            return query, None

        def sdpa(query, key, value, **kwargs):
            self.calls.append((query, key, value, kwargs))
            return query

        self.torch = SimpleNamespace(
            float32="float32", bfloat16="bfloat16", float16="float16",
            finfo=lambda dtype: SimpleNamespace(min=float(np.finfo(np.float32).min)),
            nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=sdpa)),
        )
        self.torch_npu = SimpleNamespace(npu_fusion_attention=fused)
        data = np.zeros((1, 1, 32, 191), dtype=np.float32)
        data[..., 0] = np.finfo(np.float32).min
        self.mask = Tensor(data)
        self.policy = SimpleNamespace(training=False, model=Model(self.mask))
        self.source = SimpleNamespace(MixtureQwen35=Mixture, MixtureDecoderLayerQwen35=Decoder)
        self.imports = {
            HELPER.SOURCE_MODULE: self.source,
            HELPER.MODEL_MODULE: SimpleNamespace(G05ModelQwen35=Model),
            "torch": self.torch, "torch_npu": self.torch_npu,
        }

    def inputs(self):
        query = Tensor(np.zeros((1, 8, 32, 256), np.float32), "bfloat16")
        key = Tensor(np.zeros((1, 8, 191, 256), np.float32), "bfloat16")
        proxy = SimpleNamespace(training=False, num_key_value_groups=1)
        return proxy, query, key, key, self.mask, .0625

    def test_mask_checks_one_read_and_rejects_bias_nan_full_mask_or_masked_action(self):
        HELPER.validate_action_mask(self.torch, self.mask)
        self.assertEqual(Tensor.item_reads, 1)
        for invalid in (.1, -1, np.nan, np.inf):
            data = self.mask.data.copy()
            data[0, 0, 0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                HELPER.validate_action_mask(self.torch, Tensor(data))
        data = self.mask.data.copy()
        data[0, 0, 0, :] = np.finfo(np.float32).min
        with self.assertRaises(ValueError):
            HELPER.validate_action_mask(self.torch, Tensor(data))
        data = self.mask.data.copy()
        data[..., -1] = -np.inf
        with self.assertRaises(ValueError):
            HELPER.validate_action_mask(self.torch, Tensor(data))
        data = self.mask.data.copy()
        data[..., 0] = -np.inf
        HELPER.validate_action_mask(self.torch, Tensor(data))

    def test_npu_and_sdpa_mask_polarity_scale_and_gqa_are_explicit(self):
        for backend in ("npu", "sdpa"):
            with self.subTest(backend=backend):
                attention = HELPER.make_attention(self.torch, self.torch_npu, backend)
                result, weights = attention(*self.inputs())
                self.assertEqual(result.shape, (1, 32, 8, 256))
                self.assertIsNone(weights)
                args = self.calls[-1][3]
                self.assertEqual(args["scale"], .0625)
                if backend == "npu":
                    np.testing.assert_array_equal(args["atten_mask"].data, self.mask.data != 0)
                    self.assertEqual(args["input_layout"], "BNSD")
                    self.assertEqual(args["head_num"], 8)
                    self.assertEqual(args["keep_prob"], 1.0)
                    self.assertEqual(args["inner_precise"], 0)
                else:
                    np.testing.assert_array_equal(args["attn_mask"].data, self.mask.data == 0)
                    self.assertEqual(args["dropout_p"], 0.0)
                    self.assertFalse(args["is_causal"])
                    self.assertFalse(args["enable_gqa"])
        self.assertEqual(Tensor.item_reads, 0, "No per-layer device synchronization")

    def test_installation_is_ae_local_keeps_mask_and_restores_exact_descriptor(self):
        original_globals = Decoder.forward.__globals__["ALL_ATTENTION_FUNCTIONS"]
        unrelated_vlm_layer = Decoder()
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            handle = HELPER.install_ae_fused_attention(self.policy)
        model = self.policy.model
        output = model.build_action_mask_and_position_ids()
        self.assertIs(output[0], self.mask)
        self.assertEqual(output[1], "original_position_ids")
        self.assertEqual(handle.metadata["mask_validations"], 1)
        self.assertEqual(Tensor.item_reads, 1)
        self.assertEqual(model.action_expert.layers[0].forward(*self.inputs())[0].shape, (1, 32, 8, 256))
        self.assertEqual(unrelated_vlm_layer.forward(*self.inputs())[0], "original")
        self.assertIs(Decoder.forward.__globals__["ALL_ATTENTION_FUNCTIONS"], original_globals)
        self.assertEqual(model.attn_implementation, "eager")
        self.assertFalse(handle.metadata["global_registry_modified"])
        self.assertTrue(handle.metadata["rounding_order_changed"])
        cloned = model.action_expert.layers[0].forward.__func__
        namespace_name = handle.metadata["private_module"]
        namespace = sys.modules[namespace_name]
        self.assertIs(cloned.__globals__, vars(namespace))
        self.assertEqual(cloned.__globals__["__name__"], namespace_name)
        self.assertEqual(cloned.__module__, namespace_name)
        self.assertIs(getattr(namespace, cloned.__qualname__), cloned)
        self.assertIsNot(namespace.ALL_ATTENTION_FUNCTIONS, original_globals)
        handle.restore()
        handle.restore()
        self.assertNotIn(namespace_name, sys.modules)
        self.assertNotIn("build_action_mask_and_position_ids", vars(model))
        for layer in model.action_expert.layers:
            self.assertNotIn("forward", vars(layer))
            self.assertIs(layer.forward.__func__, Decoder.forward)

    def test_refuses_duplicate_unsafe_runtime_or_nonmatching_layers(self):
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            handle = HELPER.install_ae_fused_attention(self.policy)
            with self.assertRaises(TypeError):
                HELPER.install_ae_fused_attention(self.policy)
        self.policy.model.fm_helper.action_causal = True
        with self.assertRaises(ValueError):
            self.policy.model.build_action_mask_and_position_ids()
        handle.restore()
        self.policy.model.fm_helper.action_causal = False
        self.policy.model.action_expert.layers[0].self_attn.num_key_value_heads = 1
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            with self.assertRaises(ValueError):
                HELPER.install_ae_fused_attention(self.policy)
        self.assertNotIn("build_action_mask_and_position_ids", vars(self.policy.model))

    def test_partial_install_failure_rolls_back_and_restore_preserves_later_changes(self):
        original_method_type = HELPER.MethodType
        counts = []

        def fail_second(*args):
            counts.append(1)
            if len(counts) == 2:
                raise RuntimeError("binding failed")
            return original_method_type(*args)

        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            namespaces_before = {name for name in sys.modules if name.startswith("_g05_ae_attention_")}
            with patch.object(HELPER, "MethodType", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "binding failed"):
                    HELPER.install_ae_fused_attention(self.policy)
            self.assertNotIn("build_action_mask_and_position_ids", vars(self.policy.model))
            self.assertNotIn("forward", vars(self.policy.model.action_expert.layers[0]))
            self.assertEqual(namespaces_before, {name for name in sys.modules if name.startswith("_g05_ae_attention_")})
            handle = HELPER.install_ae_fused_attention(self.policy)
        external = lambda: "external"
        self.policy.model.action_expert.layers[0].forward = external
        handle.restore()
        self.assertIs(self.policy.model.action_expert.layers[0].forward, external)
        self.assertTrue(handle.metadata["private_namespace_retained_for_external_replacement"])
        self.assertIn(handle.metadata["private_module"], sys.modules)
        # This test's external lambda does not retain the compiled function.
        sys.modules.pop(handle.metadata["private_module"])

    def test_separate_installations_get_independently_resolvable_namespaces(self):
        other_policy = SimpleNamespace(training=False, model=Model(self.mask))
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            first = HELPER.install_ae_fused_attention(self.policy, backend="npu")
            second = HELPER.install_ae_fused_attention(other_policy, backend="sdpa")
        try:
            self.assertNotEqual(first.metadata["private_module"], second.metadata["private_module"])
            self.policy.model.action_expert.layers[0].forward(*self.inputs())
            self.assertIn("atten_mask", self.calls[-1][3])
            other_policy.model.action_expert.layers[0].forward(*self.inputs())
            self.assertIn("attn_mask", self.calls[-1][3])
            first.restore()
            self.assertIn(second.metadata["private_module"], sys.modules)
            other_policy.model.action_expert.layers[0].forward(*self.inputs())
            self.assertIn("attn_mask", self.calls[-1][3])
        finally:
            first.restore()
            second.restore()

    def test_attention_rejects_training_dropout_softcap_and_unexpanded_gqa(self):
        attention = HELPER.make_attention(self.torch, self.torch_npu, "npu")
        for kwargs in ({"dropout": .1}, {"softcap": 3.0}, {"unknown": True}):
            with self.assertRaises(ValueError):
                attention(*self.inputs(), **kwargs)
        inputs = list(self.inputs())
        inputs[0].num_key_value_groups = 4
        with self.assertRaises(ValueError):
            attention(*inputs)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
