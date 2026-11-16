"""CPU NumPy/mock tests for AST scope, composition and reversible installation."""
from contextlib import nullcontext
import importlib.util
import linecache
from pathlib import Path
import sys
import tempfile
import textwrap
from types import FunctionType, MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_ae_fused_rope.py"
SPEC = importlib.util.spec_from_file_location("g05_ae_fused_rope", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)
DEVICE = SimpleNamespace(type="npu")


class Tensor:
    def __init__(self, data, dtype="float32"):
        self.data = np.asarray(data, dtype=np.int64 if dtype == "int64" else np.float32)
        self.dtype, self.device = dtype, DEVICE
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def float(self): return Tensor(self.data)
    def to(self, dtype): return Tensor(self.data, dtype)
    def repeat(self, *sizes): return Tensor(np.tile(self.data, sizes), self.dtype)
    def __getitem__(self, key): return Tensor(self.data[key], self.dtype)
    def __setitem__(self, key, value):
        key = tuple(x.data if isinstance(x, Tensor) else x for x in key)
        self.data[key] = value
    def __neg__(self): return Tensor(-self.data, self.dtype)
    def __mul__(self, other): return Tensor(self.data * (other.data if isinstance(other, Tensor) else other), self.dtype)
    def __add__(self, other): return Tensor(self.data + (other.data if isinstance(other, Tensor) else other), self.dtype)


class Decoder:
    training = False
    fail_install = False
    def __init__(self):
        self.self_attn = SimpleNamespace(num_heads=8, num_key_value_heads=2, head_dim=256,
                                        q_proj=SimpleNamespace(weight=SimpleNamespace(device=DEVICE)))
    def __setattr__(self, name, value):
        if name == "forward" and self.fail_install:
            raise RuntimeError("injected install failure")
        object.__setattr__(self, name, value)
    def __call__(self, *args): return self.forward(*args)


class Mixture:
    training = False
    def __init__(self):
        self.layers = [Decoder() for _ in range(24)]
        self.layer_types = ["full_attention"] * 24


class FusedRopeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def cat(tensors, dim=-1): return Tensor(np.concatenate([x.data for x in tensors], axis=dim), tensors[0].dtype)
        self.torch = SimpleNamespace(float32="float32", bfloat16="bfloat16", float16="float16",
            is_grad_enabled=lambda: False, no_grad=nullcontext, autocast=lambda *a, **k: nullcontext(),
            cat=cat, ones_like=lambda x: Tensor(np.ones_like(x.data)), zeros_like=lambda x: Tensor(np.zeros_like(x.data)),
            zeros=lambda *shape, **kw: Tensor(np.zeros(shape), kw["dtype"]),
            arange=lambda n, **kw: Tensor(np.arange(n), "int64"))
        def rotary(x, c, s, *args):
            self.calls.append((x.shape, x.dtype, len(args)))
            if args:
                self.assertEqual(args[0], "half")
                rotated = x.data @ args[1].data
            else:
                first, second = np.split(x.data, 2, axis=-1)
                rotated = np.concatenate((-second, first), axis=-1)
            return Tensor(x.data * c.data + rotated * s.data)
        self.npu = SimpleNamespace(npu_rotary_mul=rotary)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        fixture = Path(self.temporary.name) / "fixture.py"
        fixture.write_text("def forward(self, query_states, key_states, cos, sin):\n" +
            textwrap.indent(textwrap.dedent(HELPER._AUDITED_BLOCK).strip(), "    ") +
            "\n    return query_states, key_states, ALL_ATTENTION_FUNCTIONS\n", encoding="utf-8")
        namespace = {"__name__": "rope_original_fixture", "torch": self.torch,
                     "ALL_ATTENTION_FUNCTIONS": {"registry": "original"},
                     "rotate_half": lambda x: cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)}
        exec(compile(fixture.read_text(), str(fixture), "exec"), namespace)
        Decoder.forward = namespace["forward"]
        self.original_globals = namespace
        self.expert = Mixture()
        self.policy = SimpleNamespace(training=False, model=SimpleNamespace(training=False, action_expert=self.expert))
        source = SimpleNamespace(MixtureQwen35=Mixture, MixtureDecoderLayerQwen35=Decoder)
        sources = {"torch": self.torch, "torch_npu": self.npu, HELPER.SOURCE_MODULE: source}
        imports = patch.object(HELPER.importlib, "import_module", side_effect=lambda name: sources[name])
        imports.start(); self.addCleanup(imports.stop)
        rng = np.random.default_rng(24)
        self.q = Tensor(rng.normal(size=(1,8,32,256)), "bfloat16")
        self.k = Tensor(rng.normal(size=(1,2,32,256)), "bfloat16")
        self.c = Tensor(rng.normal(size=(1,1,32,64)))
        self.s = Tensor(rng.normal(size=(1,1,32,64)))

    def test_both_modes_preserve_dynamic_inputs_fp32_tail_and_original_dispatch(self):
        original = self.expert.layers[0].forward
        for mode in ("sliced", "matrix"):
            handle = HELPER.install_ae_fused_rope(self.policy, mode, verify=True)
            for q, c in ((self.q, self.c), (-self.q, self.c * .8), (self.q, self.c)):
                reference = original(q, self.k, c, self.s)
                actual = self.expert.layers[0](q, self.k, c, self.s)
                for candidate, expected in zip(actual[:2], reference[:2]):
                    np.testing.assert_allclose(candidate.data, expected.data, atol=1e-6, rtol=1e-6)
                    self.assertEqual(candidate.dtype, "bfloat16")
                np.testing.assert_array_equal(actual[0].data[...,64:], q.data[...,64:])
                self.assertIs(actual[2], self.original_globals["ALL_ATTENTION_FUNCTIONS"])
            self.assertTrue(all(dtype == "float32" for _, dtype, _ in self.calls))
            self.assertFalse(handle.metadata["cached_cos_sin"])
            handle.restore(); self.calls.clear()

    def test_attention_private_globals_survive_ast_rewrite_and_restore(self):
        private = dict(self.original_globals)
        private_registry = object()
        private["ALL_ATTENTION_FUNCTIONS"] = private_registry
        clone = FunctionType(Decoder.forward.__code__, private, "forward")
        setattr(clone, HELPER.ATTENTION_MARKER, True)
        for layer in self.expert.layers: layer.forward = MethodType(clone, layer)
        handle = HELPER.install_ae_fused_rope(self.policy, verify=True)
        current = self.expert.layers[0].forward.__func__
        namespace = sys.modules[current.__module__]
        self.assertIs(getattr(namespace, current.__qualname__), current)
        self.assertIs(current.__globals__, vars(namespace))
        self.assertIs(self.expert.layers[0](self.q,self.k,self.c,self.s)[2], private_registry)
        handle.restore(); handle.restore()
        self.assertIs(self.expert.layers[0].forward.__func__, clone)
        self.assertNotIn(namespace.__name__, sys.modules)
        self.assertNotIn(namespace.__file__, linecache.cache)

    def test_unsupported_shapes_and_changed_ast_are_rejected_without_fallback(self):
        callback = HELPER.make_partial_rope(self.torch, self.npu, "sliced")
        with self.assertRaisesRegex(ValueError, "Expected AE Q"):
            callback(self.q[:,:4], self.k, self.c, self.s)
        with self.assertRaisesRegex(ValueError, "dynamic cos/sin"):
            callback(self.q, self.k, self.c[...,:32], self.s[...,:32])
        changed = inspect_source = HELPER.inspect.getsource(Decoder.forward).replace("q_rot * _cos", "q_rot * (_cos + 0)")
        before = {name for name in sys.modules if name.startswith("_g05_ae_rope_")}
        with patch.object(HELPER.inspect, "getsource", return_value=changed):
            with self.assertRaisesRegex(ValueError, "audited FP32 partial-RoPE"):
                HELPER.install_ae_fused_rope(self.policy, verify=True)
        self.assertEqual(before, {name for name in sys.modules if name.startswith("_g05_ae_rope_")})
        self.assertTrue(all("forward" not in vars(layer) for layer in self.expert.layers))

    def test_partial_install_failure_rolls_back_and_later_replacement_is_preserved(self):
        before = {name for name in sys.modules if name.startswith("_g05_ae_rope_")}
        self.expert.layers[3].fail_install = True
        with self.assertRaisesRegex(RuntimeError, "injected install failure"):
            HELPER.install_ae_fused_rope(self.policy, verify=True)
        self.assertTrue(all("forward" not in vars(layer) for layer in self.expert.layers))
        self.assertEqual(before, {name for name in sys.modules if name.startswith("_g05_ae_rope_")})
        self.expert.layers[3].fail_install = False
        handle = HELPER.install_ae_fused_rope(self.policy, verify=True)
        later = lambda *args: None
        self.expert.layers[0].forward = later
        handle.restore()
        self.assertIs(self.expert.layers[0].forward, later)
        self.assertTrue(handle.metadata["private_namespace_retained_for_external_replacement"])
        sys.modules.pop(handle._namespace.__name__, None)
        linecache.cache.pop(handle._namespace.__file__, None)

    def test_opt_in_and_installation_order_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "verify=True"):
            HELPER.install_ae_fused_rope(self.policy)
        handle = HELPER.install_ae_fused_rope(self.policy, verify=True)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            HELPER.install_ae_fused_rope(self.policy, verify=True)
        handle.restore()
        self.expert.forward = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "before AE.forward compilation"):
            HELPER.install_ae_fused_rope(self.policy, verify=True)


if __name__ == "__main__":
    unittest.main()
