"""No torch needed: validate FP32 boundaries, live gamma and reversible scope."""
from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools" / "galaxeavla" / "g05_ae_fused_norm.py"
SPEC = importlib.util.spec_from_file_location("g05_ae_fused_norm", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


class Tensor:
    def __init__(self, data, dtype="float32", device=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="npu", index=0)
        self.shape, self.ndim = self.data.shape, self.data.ndim

    def float(self):
        return Tensor(self.data, "float32", self.device)

    def to(self, *, dtype):
        return Tensor(self.data, dtype, self.device)

    def __radd__(self, scalar):
        return Tensor(scalar + self.data, self.dtype, self.device)


class Adaptive:
    def __init__(self):
        self.dim, self.eps, self.training = 1024, 1e-6, False
        self.dense = SimpleNamespace(weight=Tensor([1]))

    def _norm(self, x):
        return x

    def forward(self, x, cond):
        return self._norm(x), cond


class QK:
    def __init__(self):
        self.weight = Tensor(np.linspace(-.99, .8, 256), "bfloat16")
        self.eps, self.training = 1e-6, False

    def _norm(self, x):
        return x

    def forward(self, x):
        return x


class Decoder:
    def __init__(self):
        self.training = False
        self.input_layernorm, self.post_attention_layernorm = Adaptive(), Adaptive()
        self.self_attn = SimpleNamespace(q_norm=QK(), k_norm=QK())


class Mixture:
    def __init__(self):
        self.training = False
        self.layers = [Decoder() for _ in range(24)]
        self.layer_types = ["full_attention"] * 24
        self.norm = Adaptive()


class FusedNormTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.constants, self.autocast_stack = [], [], []
        self.grad_enabled = False

        @contextlib.contextmanager
        def autocast(*, device_type, enabled):
            self.assertEqual(device_type, "npu")
            self.autocast_stack.append(enabled)
            try:
                yield
            finally:
                self.autocast_stack.pop()

        def rms_norm(x, gamma, *, epsilon):
            self.assertEqual(self.autocast_stack[-1], False)
            self.calls.append((x, gamma, epsilon))
            value = x.data * (1 / np.sqrt(np.mean(np.square(x.data), axis=-1, keepdims=True) + epsilon))
            return Tensor(value * gamma.data, "float32", x.device), None

        def ones(size, *, device, dtype):
            result = Tensor(np.ones(size), dtype, device)
            self.constants.append(result)
            return result

        self.torch = SimpleNamespace(float32="float32", bfloat16="bfloat16", float16="float16",
            is_grad_enabled=lambda: self.grad_enabled, autocast=autocast, ones=ones)
        self.torch_npu = SimpleNamespace(npu_rms_norm=rms_norm)
        self.ae = Mixture()
        self.policy = SimpleNamespace(training=False, model=SimpleNamespace(training=False, action_expert=self.ae))
        self.imports = {"torch": self.torch, "torch_npu": self.torch_npu,
            HELPER.MIXTURE_MODULE: SimpleNamespace(MixtureQwen35=Mixture, MixtureDecoderLayerQwen35=Decoder),
            HELPER.ADAPTIVE_MODULE: SimpleNamespace(AdaptiveRMSNorm=Adaptive),
            HELPER.QK_MODULE: SimpleNamespace(Qwen3_5RMSNorm=QK)}

    def install(self):
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            return HELPER.install_ae_fused_norm(self.policy, verify=True)

    def test_adaptive_retains_fp32_and_qk_casts_only_after_live_fp32_gamma(self):
        handle = self.install()
        try:
            x = Tensor(np.arange(32 * 1024).reshape(1, 32, 1024) / 1000, "bfloat16")
            output = self.ae.norm._norm(x)
            self.assertEqual(output.dtype, "float32")
            self.assertEqual(self.calls[-1][0].dtype, "float32")
            self.assertEqual(self.calls[-1][1].dtype, "float32")
            np.testing.assert_array_equal(self.calls[-1][1].data, np.ones(1024))
            self.ae.layers[0].input_layernorm._norm(x)
            self.assertIs(self.calls[-1][1], self.calls[-2][1])
            q = Tensor(np.ones((1, 32, 8, 256)), "bfloat16")
            norm = self.ae.layers[0].self_attn.q_norm
            before = norm(q) if callable(norm) else norm.forward(q)
            self.assertEqual(before.dtype, "bfloat16")
            self.assertEqual(self.calls[-1][0].dtype, "float32")
            self.assertEqual(self.calls[-1][1].dtype, "float32")
            np.testing.assert_array_equal(self.calls[-1][1].data, 1.0 + norm.weight.data)
            norm.weight.data += .25
            after = norm.forward(q)
            self.assertFalse(np.array_equal(before.data, after.data), "Gamma must not be cached across weight changes")
            self.assertEqual(len(self.constants), 1)
        finally:
            handle.restore()

    def test_time_forward_and_vlm_untouched_restore_then_reinstall(self):
        norm = self.ae.norm
        time_forward = lambda x, cond: (norm._norm(x), cond)
        norm.forward = time_forward
        vlm_norm = QK()
        handle = self.install()
        self.assertIs(norm.forward, time_forward)
        self.assertIs(vlm_norm.forward.__func__, QK.forward)
        self.assertEqual(handle.metadata["adaptive_modules"], 49)
        self.assertEqual(handle.metadata["qk_modules"], 48)
        self.assertFalse(handle.metadata["class_methods_modified"])
        with patch.object(HELPER.importlib, "import_module", side_effect=self.imports.__getitem__):
            with self.assertRaisesRegex(TypeError, "already replaced"):
                HELPER.install_ae_fused_norm(self.policy, verify=True)
        handle.restore()
        handle.restore()
        self.assertIs(norm.forward, time_forward)
        self.assertNotIn("_norm", vars(norm))
        self.assertNotIn("forward", vars(self.ae.layers[0].self_attn.q_norm))
        second = self.install()
        second.restore()

    def test_verify_and_runtime_contracts_reject_before_kernel(self):
        for verify in (False, 1, None):
            with self.assertRaisesRegex(ValueError, "verify=True"):
                HELPER.install_ae_fused_norm(self.policy, verify=verify)
        self.ae.forward = lambda *args: None
        with self.assertRaisesRegex(TypeError, "before replacing or compiling"):
            self.install()
        del self.ae.forward
        handle = self.install()
        try:
            for x in (Tensor(np.zeros((1, 32, 512))),
                      Tensor(np.zeros((1, 32, 1024)), "float64"),
                      Tensor(np.zeros((1, 32, 1024)), device=SimpleNamespace(type="cpu", index=0))):
                with self.assertRaises((ValueError, TypeError)):
                    self.ae.norm._norm(x)
            with self.assertRaises(ValueError):
                self.ae.layers[0].self_attn.q_norm.forward(Tensor(np.zeros((1, 32, 2, 256))))
            self.grad_enabled = True
            with self.assertRaises(ValueError):
                self.ae.norm._norm(Tensor(np.zeros((1, 32, 1024))))
            self.assertFalse(self.calls)
        finally:
            handle.restore()

    def test_partial_install_rollback_and_later_replacement_preserved(self):
        with patch.object(HELPER, "make_qk_norm", side_effect=RuntimeError("factory failed")):
            with self.assertRaisesRegex(RuntimeError, "factory failed"):
                self.install()
        self.assertNotIn("_norm", vars(self.ae.norm))
        for layer in self.ae.layers:
            self.assertNotIn("_norm", vars(layer.input_layernorm))
            self.assertNotIn("_norm", vars(layer.post_attention_layernorm))
        handle = self.install()
        external = lambda x: x
        self.ae.layers[0].self_attn.q_norm.forward = external
        handle.restore()
        self.assertIs(self.ae.layers[0].self_attn.q_norm.forward, external)


if __name__ == "__main__":
    unittest.main()
