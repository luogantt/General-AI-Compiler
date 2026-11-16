"""NumPy-backed lifecycle/math tests; no torch or NPU import/execution."""
from contextlib import nullcontext
import importlib.util
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_time_conditioning.py"
SPEC = importlib.util.spec_from_file_location("g05_time_conditioning", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)


class Device:
    type = "cpu"
    def __str__(self):
        return "cpu"


class Tensor:
    def __init__(self, data, dtype=np.float32):
        self.a = np.asarray(data, dtype=dtype)
        self.device, self._version = Device(), 0
    @property
    def shape(self): return self.a.shape
    @property
    def ndim(self): return self.a.ndim
    @property
    def dtype(self): return self.a.dtype
    def data_ptr(self): return self.a.__array_interface__["data"][0]
    def detach(self): return self
    def clone(self): return Tensor(self.a.copy(), self.dtype)
    def cpu(self): return self
    def tolist(self): return self.a.tolist()
    def float(self): return self.to(np.float32)
    def to(self, dtype): return Tensor(self.a.astype(dtype))
    def unsqueeze(self, dim): return Tensor(np.expand_dims(self.a, dim))
    def numel(self): return self.a.size
    def element_size(self): return self.a.itemsize
    def __add__(self, other): return Tensor(self.a + (other.a if isinstance(other, Tensor) else other))
    __radd__ = __add__
    def __mul__(self, other): return Tensor(self.a * (other.a if isinstance(other, Tensor) else other))
    __rmul__ = __mul__
    def __isub__(self, value): self.a -= value; self._version += 1; return self
    def __iadd__(self, value): self.a += value; self._version += 1; return self


def linear(x, weight, bias):
    return Tensor(x.a @ weight.a.T + (bias.a if bias is not None else 0))


class Module:
    training = False
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def named_parameters(self):
        return [(name, value) for name, value in vars(self).items() if isinstance(value, Tensor)]


class Linear(Module):
    def __init__(self, nout, scale=.1):
        self.weight = Tensor(np.arange(nout * 2).reshape(nout, 2) * scale)
        self.bias = Tensor(np.arange(nout) * .01)
    def forward(self, x): return linear(x, self.weight, self.bias)


class Sinusoid(Module):
    dim, min_period, max_period = 2, .004, 4.
    def forward(self, t): return Tensor(np.stack((np.sin(t.a), np.cos(t.a)), axis=-1))


class AdaptiveRMSNorm(Module):
    def __init__(self, scale):
        self.dim, self.eps, self.dense = 2, 1e-6, Linear(6, scale)
    def _norm(self, x):
        return Tensor(x.a / np.sqrt((x.a.astype(np.float32) ** 2).mean(-1, keepdims=True) + self.eps))
    def forward(self, x, cond):
        result = self.dense(cond).unsqueeze(1)
        scale, shift, gate = [Tensor(a) for a in np.split(result.a, 3, axis=-1)]
        return self._norm(x) * (1 + scale) + shift, gate


class MixtureQwen35(Module):
    adaptive_mode = "adaLN"
    def __init__(self):
        self.layers = [SimpleNamespace(input_layernorm=AdaptiveRMSNorm((i + 1) * .01),
                                       post_attention_layernorm=AdaptiveRMSNorm((i + 1) * .02)) for i in range(24)]
        self.norm = AdaptiveRMSNorm(.99)
        self.time_embedding, self.time_mlp_in, self.time_mlp_out = Sinusoid(), Linear(2), Linear(2)
        self.encode_calls = 0
    def encode_time(self, t):
        self.encode_calls += 1
        return self.time_mlp_out(self.time_mlp_in(self.time_embedding(t)))
    def forward(self, x, time_cond):
        # Enough of the audited argument flow to test dynamic data, layer-local
        # lookup and final norm. Real transformer/graph parity remains NPU work.
        return self.norm(self.layers[0].input_layernorm(x, time_cond)[0], time_cond)[0]


class FMHelper:
    num_inference_steps = 10
    time_convention = "pi_convention"
    consume = 10
    fail_at = None
    callback = None

    def infer(self, model, *, batch=1, dtype=np.float32):
        # A deliberately controllable test fixture, admitted only by replacing
        # the audited source hash in this test process. Production pins upstream.
        t = Tensor(np.ones(batch) if self.time_convention == "pi_convention" else np.zeros(batch), dtype)
        outputs = []
        for index in range(self.consume):
            if self.callback is not None:
                self.callback(index)
            if index == self.fail_at:
                raise ValueError("inference failed deliberately")
            outputs.append(model.action_expert.encode_time(t))
            if self.time_convention == "pi_convention": t -= .1
            else: t += .1
        return outputs


class TimeConditioningTests(unittest.TestCase):
    def setUp(self):
        self.expert = MixtureQwen35()
        self.fm = FMHelper()
        self.policy = SimpleNamespace(training=False, model=SimpleNamespace(
            training=False, action_expert=self.expert, fm_helper=self.fm))
        self.grad = False
        self.matmul_precision = "highest"
        self.torch = SimpleNamespace(float32=np.dtype("float32"), is_grad_enabled=lambda: self.grad,
            get_float32_matmul_precision=lambda: self.matmul_precision,
            no_grad=nullcontext, autocast=lambda **kw: nullcontext(),
            ones=lambda n, **kw: Tensor(np.ones(n, dtype=kw["dtype"])),
            zeros=lambda n, **kw: Tensor(np.zeros(n, dtype=kw["dtype"])),
            chunk=lambda x, n, dim: tuple(Tensor(a) for a in np.split(x.a, n, axis=dim)),
            nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)))
        sources = {"torch": self.torch, HELPER.MIXTURE_MODULE: SimpleNamespace(MixtureQwen35=MixtureQwen35),
                   HELPER.NORM_MODULE: SimpleNamespace(AdaptiveRMSNorm=AdaptiveRMSNorm),
                   HELPER.FM_MODULE: SimpleNamespace(FMHelper=FMHelper)}
        imports = patch.object(HELPER.importlib, "import_module", side_effect=lambda name: sources[name])
        imports.start(); self.addCleanup(imports.stop)

    def request_installation(self):
        fixture_sha = hashlib.sha256(inspect.getsource(FMHelper.infer).encode()).hexdigest()
        with patch.object(HELPER, "AUDITED_FM_INFER_SHA256", fixture_sha):
            return HELPER.install_time_conditioning(self.policy, lookup="request")

    def schedule(self):
        t = Tensor([1. if self.fm.time_convention == "pi_convention" else 0.])
        result = []
        for _ in range(10):
            result.append(t.clone())
            if self.fm.time_convention == "pi_convention": t -= .1
            else: t += .1
        return result

    def test_exact_sequential_schedule_distinct_layer_weights_and_dynamic_hidden_inputs(self):
        times = self.schedule()
        x = Tensor([[[1., 2.], [-2., 3.]]])
        y = Tensor([[[-3., 2.], [1., -.5]]])
        old_encode, old_norm = self.expert.encode_time, self.expert.layers[0].input_layernorm.forward
        old_final = self.expert.norm.forward
        refs = [(old_norm(x, old_encode(t)), old_final(x, old_encode(t))) for t in times]
        original_calls = self.expert.encode_calls
        handle = HELPER.install_time_conditioning(self.policy)
        first = self.expert.encode_time(times[0])
        self.assertEqual(self.expert.encode_calls, original_calls + 10)
        self.assertEqual(len(first), 49)
        self.assertEqual(handle.metadata["table_bytes"], 10 * 49 * 3 * 2 * 4)
        for index in reversed(range(10)):  # Stateless lookup, not a Python cursor.
            cond = self.expert.encode_time(times[index])
            for norm, expected in ((self.expert.layers[0].input_layernorm, refs[index][0]),
                                   (self.expert.norm, refs[index][1])):
                actual = norm(x, cond)
                np.testing.assert_array_equal(actual[0].a, expected[0].a)
                np.testing.assert_array_equal(actual[1].a, expected[1].a)
        actual_y = self.expert.layers[0].input_layernorm(y, first)[0]
        np.testing.assert_array_equal(actual_y.a, old_norm(y, old_encode(times[0]))[0].a)
        self.assertFalse(np.array_equal(actual_y.a, refs[0][0][0].a))
        self.assertFalse(handle.metadata["cached_actions"])
        self.assertFalse(handle.metadata["cached_kv"])
        self.assertEqual(handle.metadata["table_builds"], 1)

    def test_rejects_linspace_schedule_and_changed_shape_dtype_or_step_contract(self):
        HELPER.install_time_conditioning(self.policy)
        self.expert.encode_time(self.schedule()[0])
        with self.assertRaisesRegex(ValueError, "exact member"):
            self.expert.encode_time(Tensor([.1]))  # Repeated FP32 subtraction gives .09999993.
        with self.assertRaisesRegex(RuntimeError, "batch/dtype/device"):
            self.expert.encode_time(Tensor([1., 1.]))
        with self.assertRaisesRegex(ValueError, "FP32"):
            self.expert.encode_time(Tensor([1.], np.float64))
        self.fm.num_inference_steps = 9
        with self.assertRaisesRegex(RuntimeError, "schedule changed"):
            self.expert.encode_time(Tensor([1.]))

    def test_changed_parameter_version_storage_and_module_identity_are_rejected(self):
        for mutation in ("version", "storage", "module"):
            with self.subTest(mutation=mutation):
                handle = HELPER.install_time_conditioning(self.policy)
                self.expert.encode_time(Tensor([1.]))
                if mutation == "version": self.expert.time_mlp_in.weight._version += 1
                elif mutation == "storage": self.expert.norm.dense.weight = self.expert.norm.dense.weight.clone()
                else: self.expert.time_mlp_out = Linear(2)
                with self.assertRaisesRegex(RuntimeError, "weights changed|module identity changed"):
                    self.expert.encode_time(Tensor([1.]))
                handle.restore()

    def test_float32_matmul_precision_change_is_not_silently_reused(self):
        handle = HELPER.install_time_conditioning(self.policy)
        self.expert.encode_time(Tensor([1.]))
        self.assertEqual(handle.metadata["matmul_precision_settings"], {"float32_matmul_precision": "highest"})
        self.matmul_precision = "medium"
        with self.assertRaisesRegex(RuntimeError, "precision policy changed"):
            self.expert.encode_time(Tensor([1.]))

    def test_training_or_grad_enabled_and_precompiled_install_are_rejected(self):
        handle = HELPER.install_time_conditioning(self.policy)
        self.grad = True
        with self.assertRaisesRegex(ValueError, "gradients"):
            self.expert.encode_time(Tensor([1.]))
        self.grad = False
        self.expert.training = True
        with self.assertRaisesRegex(ValueError, "evaluation"):
            self.expert.encode_time(Tensor([1.]))
        self.expert.training = False
        handle.restore()
        self.expert.forward = lambda *a, **k: None
        with self.assertRaisesRegex(ValueError, "before replacing/compiling"):
            HELPER.install_time_conditioning(self.policy)

    def test_restore_preserves_later_replacements_and_is_idempotent(self):
        handle = HELPER.install_time_conditioning(self.policy)
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            HELPER.install_time_conditioning(self.policy)
        later = lambda *a: None
        self.expert.norm.forward = later
        handle.restore(); handle.restore()
        self.assertIs(self.expert.norm.forward, later)
        self.assertNotIn("forward", vars(self.expert.layers[0].input_layernorm))
        self.assertNotIn("encode_time", vars(self.expert))

    def test_galaxea_forward_time_schedule_is_built_from_zero(self):
        self.fm.time_convention = "galaxea_convention"
        handle = HELPER.install_time_conditioning(self.policy)
        for t in self.schedule(): self.expert.encode_time(t)
        self.assertEqual(handle.metadata["time_values"][0], [0.])
        self.assertEqual(handle.metadata["calls"], 10)

    def test_request_lookup_requires_audited_original_method_and_explicit_mode(self):
        with self.assertRaisesRegex(ValueError, "audited sequential"):
            HELPER.install_time_conditioning(self.policy, lookup="request")
        with self.assertRaisesRegex(ValueError, "lookup must"):
            HELPER.install_time_conditioning(self.policy, lookup="automatic")
        self.fm.infer = lambda *a, **k: None
        with self.assertRaisesRegex(TypeError, "original exact"):
            self.request_installation()

    def test_request_cursor_matches_exact_entries_without_per_step_reads_or_contract_scans(self):
        exact = HELPER.install_time_conditioning(self.policy)
        expected = [self.expert.encode_time(t) for t in self.schedule()]
        exact.restore()
        handle = self.request_installation()
        # Build once in the cold request; all subsequent t-value reads forbidden.
        first = self.fm.infer(self.policy.model)
        signature = HELPER._parameter_signature
        with patch.object(Tensor, "cpu", side_effect=AssertionError("hot D2H")), \
             patch.object(HELPER, "_parameter_signature", wraps=signature) as signatures:
            second = self.fm.infer(self.policy.model)
            self.assertEqual(signatures.call_count, 1)
        for reference_steps in (first, second):
            for expected_norms, actual_norms in zip(expected, reference_steps):
                for expected_parts, actual_parts in zip(expected_norms, actual_norms):
                    for reference, actual in zip(expected_parts, actual_parts):
                        np.testing.assert_array_equal(reference.a, actual.a)
        self.assertEqual(handle.metadata["request_checks"], 2)
        self.assertEqual(handle.metadata["completed_requests"], 2)
        self.assertEqual(handle.metadata["calls"], 20)
        with self.assertRaisesRegex(RuntimeError, "only allowed inside"):
            self.expert.encode_time(Tensor([1.]))

    def test_request_finally_resets_after_failure_and_rejects_under_or_over_consumption(self):
        handle = self.request_installation()
        self.fm.fail_at = 3
        with self.assertRaisesRegex(ValueError, "failed deliberately"):
            self.fm.infer(self.policy.model)
        self.assertEqual(handle.metadata["last_request_consumed"], 3)
        self.fm.fail_at = None
        for count in (9, 11):
            self.fm.consume = count
            with self.subTest(count=count), self.assertRaisesRegex(RuntimeError, "consumed"):
                self.fm.infer(self.policy.model)
        self.fm.consume = 10
        self.fm.infer(self.policy.model)
        self.assertEqual(handle.metadata["completed_requests"], 1)
        self.assertEqual(handle.metadata["failed_requests"], 3)
        self.assertEqual(handle.metadata["last_request_consumed"], 10)

    def test_request_rejects_nested_concurrent_and_foreign_encode_calls(self):
        handle = self.request_installation()
        captured = []
        def callback(index):
            if index != 0: return
            with self.assertRaisesRegex(RuntimeError, "Concurrent or nested"):
                self.fm.infer(self.policy.model)
            def foreign():
                for operation in (lambda: self.fm.infer(self.policy.model), lambda: self.expert.encode_time(Tensor([1.]))):
                    try: operation()
                    except RuntimeError as error: captured.append(str(error))
            worker = threading.Thread(target=foreign)
            worker.start(); worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.fm.callback = callback
        self.fm.infer(self.policy.model)
        self.assertEqual(len(captured), 2)
        self.assertIn("Concurrent or nested", captured[0])
        self.assertIn("only allowed inside", captured[1])
        self.assertEqual(handle.metadata["completed_requests"], 1)

    def test_request_checks_shape_and_weights_between_requests_and_restores_method(self):
        handle = self.request_installation()
        self.fm.infer(self.policy.model)
        with self.assertRaisesRegex(RuntimeError, "batch/dtype/device"):
            self.fm.infer(self.policy.model, batch=2)
        self.expert.time_mlp_in.weight._version += 1
        with self.assertRaisesRegex(RuntimeError, "weights changed"):
            self.fm.infer(self.policy.model)
        handle.restore()
        self.assertNotIn("infer", vars(self.fm))
        self.assertNotIn("encode_time", vars(self.expert))


if __name__ == "__main__":
    unittest.main()
