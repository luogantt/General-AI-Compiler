"""CPU metadata/AST/lifecycle mocks; no torch, NPU or shared library execution."""
import ast
from contextlib import nullcontext
import importlib.util
import linecache
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_ae_qknorm_rope.py"
SPEC = importlib.util.spec_from_file_location("g05_ae_qknorm_rope", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)
DEVICE = SimpleNamespace(type="npu", index=0)


class Tensor:
    def __init__(self, shape, dtype="fp32", contiguous=True, requires_grad=False):
        self.shape, self.dtype, self.device = tuple(shape), dtype, DEVICE
        self.ndim, self.requires_grad, self.contiguous = len(shape), requires_grad, contiguous
    def is_contiguous(self): return self.contiguous


class Norm:
    training = False
    eps = 1e-6
    def __init__(self): self.weight = Tensor((256,), requires_grad=True)
    def forward(self, x): return x


class Decoder:
    training = False
    fail_install = False
    def __init__(self):
        self.self_attn = SimpleNamespace(num_heads=8, num_key_value_heads=2, head_dim=256,
                                        q_norm=Norm(), k_norm=Norm())
    def __setattr__(self, name, value):
        if name == "forward" and self.fail_install:
            raise RuntimeError("injected layer mutation failure")
        object.__setattr__(self, name, value)


class Mixture:
    training = False
    def __init__(self):
        self.layers = [Decoder() for _ in range(24)]
        self.layer_types = ["full_attention"] * 24


class QKNormRopeTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(float32="fp32", bfloat16="bf16", is_grad_enabled=lambda: False,
                                    autocast=lambda *args, **kwargs: nullcontext())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Path(self.temporary.name) / "forward.py"
        body = ("def forward(self, hidden_states, cos, sin):\n"
                "    preserved_before = 'before'\n"
                "    if self.layer_type == 'full_attention':\n" +
                textwrap.indent(HELPER._QK_SOURCE.strip(), "        ") +
                "\n        value_states = 'untouched value projection'\n" +
                textwrap.indent(HELPER._ROPE_SOURCE.strip(), "        ") +
                "\n        key_states_new = key_states\n"
                "    return ALL_ATTENTION_FUNCTIONS, preserved_before\n")
        self.fixture.write_text(body, encoding="utf-8")
        namespace = {"__name__": "qknorm_source_fixture", "ALL_ATTENTION_FUNCTIONS": object()}
        exec(compile(body, str(self.fixture), "exec"), namespace)
        Decoder.forward = namespace["forward"]
        self.original = Decoder.forward
        self.expert = Mixture()
        self.policy = SimpleNamespace(training=False, model=SimpleNamespace(training=False, action_expert=self.expert))
        self.source = SimpleNamespace(MixtureQwen35=Mixture, MixtureDecoderLayerQwen35=Decoder)

    def values(self, stride=512, length=32):
        return (Tensor((1, 32, 4096 if stride == 512 else 512), "bf16"),
                Tensor((256,), requires_grad=True), Tensor((1, 1, length, 64)), Tensor((1, 1, length, 64)))

    def install(self):
        modules = {"torch": self.torch, HELPER.SOURCE_MODULE: self.source,
                   HELPER.NORM_MODULE: SimpleNamespace(Qwen3_5RMSNorm=Norm)}
        with patch.object(HELPER.importlib, "import_module", side_effect=modules.__getitem__), \
             patch.object(HELPER, "register_qknorm_rope_op", return_value=(object(), {"operator": HELPER.OP_NAME})):
            return HELPER.install_ae_qknorm_rope(self.policy, verify=True)

    def test_metadata_contract_accepts_inference_parameters_and_long_dynamic_rope(self):
        for stride, heads in ((512, 8), (256, 2)):
            for length in (32, 159):
                self.assertEqual(HELPER.validate_inputs(self.torch, *self.values(stride, length), stride, 1e-6),
                                 (1, heads, 32, 256))
        seen = []
        call = HELPER.make_qknorm_rope(self.torch, lambda *args: seen.append(args) or "result")
        self.assertEqual(call(*self.values(), 512, 1e-6), "result")
        self.assertTrue(seen[0][1].requires_grad)
        self.torch.is_grad_enabled = lambda: True
        with self.assertRaisesRegex(ValueError, "gradients disabled"):
            call(*self.values(), 512, 1e-6)

    def test_contract_rejects_wrong_layout_dtype_geometry_or_epsilon(self):
        modifications = ((0, Tensor((1, 32, 4096), "fp32")), (0, Tensor((2, 32, 4096), "bf16")),
                         (0, Tensor((1, 32, 4096), "bf16", contiguous=False)),
                         (1, Tensor((256,), "bf16")), (2, Tensor((1, 1, 31, 64))))
        for index, value in modifications:
            values = list(self.values()); values[index] = value
            with self.subTest(index=index, shape=value.shape), self.assertRaises(ValueError):
                HELPER.validate_inputs(self.torch, *values, 512, 1e-6)
        for stride, epsilon in ((128, 1e-6), (True, 1e-6), (512, 0), (512, float("nan"))):
            with self.assertRaises(ValueError):
                HELPER.validate_inputs(self.torch, *self.values(), stride, epsilon)

    def test_ast_preserves_private_attention_gate_value_and_cache_statements(self):
        namespace, function = HELPER._make_private_forward(self.original, object())
        self.addCleanup(lambda: sys.modules.pop(namespace.__name__, None))
        self.addCleanup(lambda: linecache.cache.pop(namespace.__file__, None))
        self.assertIs(function.__globals__["ALL_ATTENTION_FUNCTIONS"], self.original.__globals__["ALL_ATTENTION_FUNCTIONS"])
        self.assertIs(sys.modules[function.__module__].ae_layer_forward, function)
        import inspect
        source = inspect.getsource(function)
        self.assertEqual(source.count(HELPER.CALLBACK + "("), 2)
        self.assertNotIn("attn.q_norm(query_states)", source)
        self.assertNotIn("rotate_half", source)
        for preserved in ("attn_gate.reshape", "untouched value projection", "key_states_new = key_states", "preserved_before"):
            self.assertIn(preserved, source)

    def test_ast_rejects_changed_qk_or_rope_without_creating_private_namespace(self):
        for old, new in (("attn.q_norm(query_states)", "attn.q_norm(query_states.float())"),
                         ("q_rot * _cos", "q_rot + _cos")):
            original_text = self.fixture.read_text()
            changed = original_text.replace(old, new)
            self.fixture.write_text(changed)
            namespace = {"__name__": "bad_fixture"}
            exec(compile(changed, str(self.fixture), "exec"), namespace)
            before = set(sys.modules)
            with self.assertRaisesRegex(ValueError, "audited"):
                HELPER._make_private_forward(namespace["forward"], object())
            self.assertEqual(set(sys.modules), before)
            self.fixture.write_text(original_text)

    def test_restore_is_instance_local_idempotent_and_keeps_norm_modules(self):
        original_norm = self.expert.layers[0].self_attn.q_norm.forward
        handle = self.install()
        self.assertEqual(len(handle._replacements), 24)
        self.assertEqual(self.expert.layers[0].self_attn.q_norm.forward, original_norm)
        self.assertIs(Decoder.forward, self.original)
        self.assertIn(handle._namespace.__name__, sys.modules)
        handle.restore(); handle.restore()
        self.assertNotIn(handle._namespace.__name__, sys.modules)
        self.assertTrue(all("forward" not in vars(layer) for layer in self.expert.layers))

    def test_install_rollback_and_external_replacement_lifetime(self):
        self.expert.layers[3].fail_install = True
        before = {name for name in sys.modules if name.startswith("_g05_ae_qknorm_rope_")}
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.install()
        self.assertTrue(all("forward" not in vars(layer) for layer in self.expert.layers))
        self.assertEqual(before, {name for name in sys.modules if name.startswith("_g05_ae_qknorm_rope_")})
        self.expert.layers[3].fail_install = False
        handle = self.install()
        replacement = lambda *args: None
        self.expert.layers[0].forward = replacement
        handle.restore()
        self.assertIs(self.expert.layers[0].forward, replacement)
        self.assertIn(handle._namespace.__name__, sys.modules)
        sys.modules.pop(handle._namespace.__name__)
        linecache.cache.pop(handle._namespace.__file__, None)

    def test_registration_fake_and_converter_preserve_schema_and_fresh_output(self):
        library = Path(self.temporary.name) / "bridge.so"
        library.write_bytes(b"mock only, never loaded")
        loaded, fake_functions, converters, ge_calls = [], [], [], []
        op = object()
        self.torch.ops = SimpleNamespace(load_library=loaded.append,
                                        gac_g05=SimpleNamespace(qk_norm_rope=SimpleNamespace(default=op)))
        self.torch.library = SimpleNamespace(register_fake=lambda name: lambda fn: fake_functions.append((name, fn)) or fn)
        self.torch.empty = lambda shape, **kwargs: Tensor(shape, kwargs["dtype"])
        ge = SimpleNamespace(custom_op=lambda *args, **kwargs: ge_calls.append((args, kwargs)) or "ge_output",
                             attr=SimpleNamespace(Int=lambda value: ("int", value), Float=lambda value: ("float", value)))
        modules = {"torch": self.torch, "torch_npu": object(), "torchair": SimpleNamespace(ge=ge),
                   "torchair.ge_concrete_graph.fx2ge_converter": SimpleNamespace(
                       register_fx_node_ge_converter=lambda target: lambda fn: converters.append((target, fn)) or fn)}
        with patch.object(HELPER, "_REGISTRATION", None), patch.dict(HELPER.os.environ, {"GAC_QKNORM_ROPE_LIBRARY": str(library)}), \
             patch.object(HELPER.importlib, "import_module", side_effect=modules.__getitem__):
            self.assertIs(HELPER.register_qknorm_rope_op()[0], op)
            self.assertIs(HELPER.register_qknorm_rope_op()[0], op)
            self.assertEqual(len(loaded), 1)
            values = self.values()
            output = fake_functions[0][1](*values, 512, 1e-6)
            self.assertEqual(output.shape, (1, 8, 32, 256))
            self.assertTrue(all(output is not value for value in values))
            self.assertEqual(converters[0][1](*values, 512, 1e-6), "ge_output")
            self.assertEqual(ge_calls[0][1]["outputs"], ["y"])
            self.assertEqual(ge_calls[0][1]["attrs"], {"head_stride": ("int", 512), "epsilon": ("float", 1e-6)})

    def test_probe_reference_keeps_intermediate_bf16_rounding_and_ignores_gate_suffix(self):
        probe_spec = importlib.util.spec_from_file_location("qknorm_probe", PATH.with_name("probe_ae_qknorm_rope.py"))
        probe = importlib.util.module_from_spec(probe_spec)
        probe_spec.loader.exec_module(probe)

        def bf16(value):
            value = np.asarray(value, dtype=np.float32)
            bits = value.view(np.uint32)
            rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
            return rounded.view(np.float32)

        class Array:
            def __init__(self, data): self.data = np.asarray(data, dtype=np.float32)
            def view(self, *shape): return Array(self.data.reshape(shape))
            def __getitem__(self, key): return Array(self.data[key])
            def float(self): return self
            def to(self, dtype): return Array(bf16(self.data))
            def transpose(self, a, b): return Array(self.data.swapaxes(a, b))
            def contiguous(self): return Array(np.ascontiguousarray(self.data))
            def pow(self, value): return Array(self.data ** value)
            def mean(self, axis, keepdim=False): return Array(self.data.mean(axis, keepdims=keepdim))
            def __mul__(self, other): return Array(self.data * (other.data if isinstance(other, Array) else other))
            def __add__(self, other): return Array(self.data + (other.data if isinstance(other, Array) else other))
            __radd__ = __add__
            def __neg__(self): return Array(-self.data)

        torch = SimpleNamespace(bfloat16="bf16", autocast=lambda *args, **kwargs: nullcontext(),
                                rsqrt=lambda value: Array(1 / np.sqrt(value.data)),
                                cat=lambda values, dim: Array(np.concatenate([value.data for value in values], axis=dim)))
        rng = np.random.default_rng(4)
        values = bf16(rng.normal(size=(1, 32, 4096)))
        weights = rng.uniform(-.9, .8, size=256).astype(np.float32)
        angle = np.repeat(rng.normal(size=(1, 1, 159, 32)), 2, axis=-1).astype(np.float32)
        cos, sin = np.cos(angle), np.sin(angle)
        formula = probe.make_formula(torch, None, head_stride=512, epsilon=1e-6)
        actual = formula(*map(Array, (values, weights, cos, sin))).data
        raw = values.reshape(1, 32, 8, 512)[..., :256]
        normalized = (raw * (1 / np.sqrt((raw ** 2).mean(-1, keepdims=True) + np.float32(1e-6)))) * (1 + weights)
        value = bf16(normalized).swapaxes(1, 2)
        partial = value[..., :64]
        rotated = np.concatenate((-partial[..., 32:], partial[..., :32]), axis=-1)
        expected = bf16(np.concatenate((partial * cos[:, :, :32] + rotated * sin[:, :, :32], value[..., 64:]), axis=-1))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[..., 64:], value[..., 64:])
        unrounded = normalized.swapaxes(1, 2)[..., :64]
        wrong = bf16(unrounded * cos[:, :, :32] + np.concatenate((-unrounded[..., 32:], unrounded[..., :32]), -1) * sin[:, :, :32])
        self.assertGreater(np.count_nonzero(wrong != actual[..., :64]), 100)
        changed = values.copy(); changed.reshape(1, 32, 8, 512)[..., 256:] = 90
        changed_cos, changed_sin = cos.copy(), sin.copy()
        changed_cos[:, :, 32:] = 8; changed_sin[:, :, 32:] = -9
        np.testing.assert_array_equal(formula(*map(Array, (changed, weights, changed_cos, changed_sin))).data, actual)


if __name__ == "__main__":
    unittest.main()
