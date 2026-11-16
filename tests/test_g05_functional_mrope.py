"""CPU NumPy/mocks only: no Torch, NPU, model weights or remote execution."""
import ast
from collections.abc import Sequence
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_functional_mrope.py"
SPEC = importlib.util.spec_from_file_location("g05_functional_mrope", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)
DEVICE = SimpleNamespace(type="npu", index=0)


class Array:
    def __init__(self, data, device=DEVICE):
        self.data = np.asarray(data)
        self.shape, self.ndim = self.data.shape, self.data.ndim
        self.dtype, self.device = self.data.dtype, device

    def __getitem__(self, key):
        return Array(self.data[key], self.device)


class ConfigSequence(Sequence):
    """Container protocol used by OmegaConf without importing that dependency."""
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self.values[index]
    def __len__(self): return len(self.values)


def make_torch():
    return SimpleNamespace(float32=np.dtype("float32"), bool=np.dtype("bool"),
        is_grad_enabled=lambda: False,
        tensor=lambda values, dtype, device: Array(np.array(values, dtype=dtype), device),
        where=lambda mask, a, b: Array(np.where(mask.data, a.data, b.data), a.device))


class Rotary:
    training = False

    def __init__(self):
        self.mrope_section = [11, 11, 10]
        self.inv_freq = Array(np.zeros(32, dtype="float32"))

    def forward(self, x, position_ids):
        return "original forward"

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        selected = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            idx = slice(offset, mrope_section[dim] * 3, 3)
            selected.data[..., idx] = freqs.data[dim, ..., idx]
        return selected


def make_policy():
    def mixture(types):
        return SimpleNamespace(training=False, layer_types=types,
            layers=[SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=Rotary())) for _ in types])
    return SimpleNamespace(training=False, model=SimpleNamespace(training=False,
        vlm=mixture(["linear_attention"] * 3 + ["full_attention"] * 2),
        action_expert=mixture(["full_attention"] * 3)))


class FunctionalMRopeTests(unittest.TestCase):
    def setUp(self):
        self.torch = make_torch()
        self.policy = make_policy()
        self.source = SimpleNamespace(Qwen3_5TextRotaryEmbedding=Rotary)

    def installation(self):
        modules = {"torch": self.torch, HELPER.SOURCE_MODULE: self.source}
        hashes = {Rotary.forward: HELPER.AUDITED_FORWARD_SOURCE_SHA256,
                  Rotary.apply_interleaved_mrope: HELPER.AUDITED_INTERLEAVE_SOURCE_SHA256}
        with patch.object(HELPER.importlib, "import_module", side_effect=modules.__getitem__), \
             patch.object(HELPER, "_source_sha256", side_effect=hashes.__getitem__):
            return HELPER.install_functional_mrope(self.policy, verify=True)

    def test_all_32_indices_match_original_without_input_mutation(self):
        apply = HELPER.make_functional_interleave(self.torch, DEVICE)
        for batch, sequence in ((1, 32), (2, 159), (1, 1)):
            values = np.arange(3 * batch * sequence * 32, dtype="float32").reshape(3, batch, sequence, 32)
            source = Array(values.copy())
            expected = Rotary.apply_interleaved_mrope(Array(values.copy()), [11, 11, 10])
            result = apply(source, [11, 11, 10])
            np.testing.assert_array_equal(result.data, expected.data)
            np.testing.assert_array_equal(source.data, values)
            self.assertEqual(result.shape, (batch, sequence, 32))
            self.assertFalse(np.shares_memory(result.data, source.data))
            # Especially check the unequal section boundary: index 30 is T,
            # index 31 is H; there is no width index 32.
            np.testing.assert_array_equal(result.data[..., 30], values[0, ..., 30])
            np.testing.assert_array_equal(result.data[..., 31], values[1, ..., 31])

    def test_selection_preserves_fp32_bits_and_consumes_each_request(self):
        apply = HELPER.make_functional_interleave(self.torch, DEVICE)
        rng = np.random.default_rng(8)
        for _ in range(3):
            bits = rng.integers(0, 2**32, size=(3, 2, 7, 32), dtype=np.uint32)
            values = bits.view(np.float32)
            expected = Rotary.apply_interleaved_mrope(Array(values.copy()), [11, 11, 10]).data
            actual = apply(Array(values), (11, 11, 10)).data
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_rejects_shape_precision_section_device_and_gradient_changes(self):
        apply = HELPER.make_functional_interleave(self.torch, DEVICE)
        good = Array(np.zeros((3, 1, 32, 32), dtype="float32"))
        for section in ([11, 10, 11], [11, 11], [11.0, 11, 10], None):
            with self.assertRaises(ValueError): apply(good, section)
        for shape in ((1, 3, 32, 32), (3, 1, 32, 64), (3, 0, 32, 32), (3, 1, 0, 32), (3, 32, 32)):
            with self.assertRaises(ValueError): apply(Array(np.zeros(shape, dtype="float32")), [11, 11, 10])
        with self.assertRaises(TypeError): apply(Array(np.zeros(good.shape, dtype="float16")), [11, 11, 10])
        with self.assertRaises(ValueError): apply(Array(good.data, SimpleNamespace(type="cpu")), [11, 11, 10])
        self.torch.is_grad_enabled = lambda: True
        with self.assertRaises(ValueError): apply(good, [11, 11, 10])

    def test_patches_only_called_instances_and_restores(self):
        original = Rotary.apply_interleaved_mrope
        sections = {id(layer.self_attn.rotary_emb): layer.self_attn.rotary_emb.mrope_section
                    for mixture in (self.policy.model.vlm, self.policy.model.action_expert)
                    for layer in mixture.layers}
        handle = self.installation()
        self.assertEqual(handle.metadata["targets"], [
            {"mixture": "vlm", "first_full_attention_layer": 3},
            {"mixture": "action_expert", "first_full_attention_layer": 0}])
        self.assertIs(Rotary.apply_interleaved_mrope, original)
        for name, selected in (("vlm", 3), ("action_expert", 0)):
            for i, layer in enumerate(getattr(self.policy.model, name).layers):
                self.assertEqual("apply_interleaved_mrope" in vars(layer.self_attn.rotary_emb), i == selected)
                if i == selected: self.assertIs(layer.self_attn.rotary_emb.mrope_section, HELPER.SECTION)
        handle.restore()
        handle.restore()
        for mixture in (self.policy.model.vlm, self.policy.model.action_expert):
            self.assertTrue(all("apply_interleaved_mrope" not in vars(layer.self_attn.rotary_emb) for layer in mixture.layers))
            for layer in mixture.layers:
                rotary = layer.self_attn.rotary_emb
                self.assertIs(rotary.mrope_section, sections[id(rotary)])

    def test_config_sequence_is_normalized_without_mutating_config_and_restored(self):
        rotary = self.policy.model.vlm.layers[3].self_attn.rotary_emb
        config = ConfigSequence([11, 11, 10])
        rotary.mrope_section = config
        rotary.config = SimpleNamespace(rope_parameters={"mrope_section": config})
        handle = self.installation()
        self.assertIs(rotary.mrope_section, HELPER.SECTION)
        self.assertIs(rotary.config.rope_parameters["mrope_section"], config)
        self.assertEqual(list(config), [11, 11, 10])
        handle.restore()
        self.assertIs(rotary.mrope_section, config)
        for values in ([11, 11, True], [11, 11, 10.0], [11, 10, 11], [11, 11]):
            with self.assertRaises(ValueError): HELPER._validate_section(ConfigSequence(values))

    def test_restore_does_not_clobber_later_section_override(self):
        handle = self.installation()
        rotary = self.policy.model.vlm.layers[3].self_attn.rotary_emb
        later = [11, 11, 10]
        rotary.mrope_section = later
        handle.restore()
        self.assertIs(rotary.mrope_section, later)

    def test_restore_preserves_later_replacement_and_rejects_double_install(self):
        handle = self.installation()
        with self.assertRaises(RuntimeError): self.installation()
        rotary = self.policy.model.action_expert.layers[0].self_attn.rotary_emb
        external = lambda *args: "external"
        rotary.apply_interleaved_mrope = external
        handle.restore()
        self.assertIs(rotary.apply_interleaved_mrope, external)

    def test_requires_verification_and_audited_source(self):
        with self.assertRaises(ValueError): HELPER.install_functional_mrope(self.policy)
        modules = {"torch": self.torch, HELPER.SOURCE_MODULE: self.source}
        with patch.object(HELPER.importlib, "import_module", side_effect=modules.__getitem__), \
             patch.object(HELPER, "_source_sha256", return_value="different source"):
            with self.assertRaisesRegex(ValueError, "source differs"):
                HELPER.install_functional_mrope(self.policy, verify=True)

    def test_source_lock_ignores_ast_version_fields_but_rejects_body_changes(self):
        source = "    def forward(self, x):\n        return x + 1\n"
        modern = ast.parse("def forward(self, x):\n    return x + 1\n")
        function = modern.body[0]
        # Simulate the same source's 3.12+ versus 3.11 FunctionDef dump. This
        # test runs on either runtime and does not depend on its default fields.
        fields = tuple(name for name in function._fields if name != "type_params")
        function._fields = fields + ("type_params",)
        function.type_params = []
        modern_dump = ast.dump(modern, include_attributes=False)
        function._fields = fields
        legacy_dump = ast.dump(modern, include_attributes=False)
        self.assertNotEqual(modern_dump, legacy_dump)
        with patch.object(HELPER.inspect, "getsource", return_value=source), \
             patch.object(ast, "dump", side_effect=AssertionError("AST dump must not be used")):
            pinned = HELPER._source_sha256(Rotary.forward)
        with patch.object(HELPER.inspect, "getsource", return_value=source.replace("\n", "\r\n")):
            self.assertEqual(HELPER._source_sha256(Rotary.forward), pinned)
        with patch.object(HELPER.inspect, "getsource", return_value=source.replace("x + 1", "x + 2")):
            self.assertNotEqual(HELPER._source_sha256(Rotary.forward), pinned)

    def test_invalid_second_target_is_rejected_before_any_mutation(self):
        self.policy.model.action_expert.layers[0].self_attn.rotary_emb.mrope_section = [10, 11, 11]
        with self.assertRaises(ValueError): self.installation()
        rotary = self.policy.model.vlm.layers[3].self_attn.rotary_emb
        self.assertNotIn("apply_interleaved_mrope", vars(rotary))

    def test_rolls_back_first_patch_if_second_mask_allocation_fails(self):
        original = HELPER.make_functional_interleave
        rotary = self.policy.model.vlm.layers[3].self_attn.rotary_emb
        section = rotary.mrope_section
        calls = []
        def fail_second(*args):
            calls.append(1)
            if len(calls) == 2: raise RuntimeError("allocation failed")
            return original(*args)
        with patch.object(HELPER, "make_functional_interleave", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"): self.installation()
        rotary = self.policy.model.vlm.layers[3].self_attn.rotary_emb
        self.assertNotIn("apply_interleaved_mrope", vars(rotary))
        self.assertIs(rotary.mrope_section, section)


if __name__ == "__main__":
    unittest.main()
