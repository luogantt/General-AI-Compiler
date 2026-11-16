"""CPU-only operator contract and installation lifecycle tests; no Torch import."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/galaxeavla" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HELPER = load("g05_gdn_triton", "g05_gdn_triton.py")
SHARED = load("g05_gdn_triangular", "g05_gdn_triangular.py")


class Tensor:
    def __init__(self, shape=(80, 32, 32), dtype="float32", device="npu", contiguous=True, requires_grad=False):
        self.shape, self.dtype = tuple(shape), dtype
        self.device = SimpleNamespace(type=device)
        self.ndim, self.requires_grad = len(self.shape), requires_grad
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if -1 in shape:
            n = math.prod(self.shape) // math.prod(x for x in shape if x != -1)
            shape = tuple(n if x == -1 else x for x in shape)
        return Tensor(shape, self.dtype, self.device.type, self._contiguous, self.requires_grad)

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device.type, True, self.requires_grad)


class CustomOp:
    def __init__(self, implementation):
        self.implementation = implementation
        self.fake = None

    def register_fake(self, implementation):
        self.fake = implementation
        return implementation

    def __call__(self, value):
        return self.implementation(value)


class GdnTritonTests(unittest.TestCase):
    def setUp(self):
        self.registrations, self.launches = [], []

        def custom_op(name, **kwargs):
            self.registrations.append((name, kwargs))
            return CustomOp

        self.torch = SimpleNamespace(
            Tensor=Tensor, float32="float32", library=SimpleNamespace(custom_op=custom_op),
            empty_like=lambda value: Tensor(value.shape, value.dtype, value.device.type),
        )
        launches = self.launches

        class Kernel:
            def __getitem__(self, grid):
                def launch(value, output, **kwargs):
                    launches.append((grid, value, output, kwargs))
                return launch

        self.source = SimpleNamespace(torch_chunk_gated_delta_rule=lambda *args, **kwargs: "original")
        self.original = self.source.torch_chunk_gated_delta_rule
        self.imports = []

        def importer(name):
            self.imports.append(name)
            return {"torch": self.torch, "torch_npu": SimpleNamespace(),
                    HELPER.GDN_MODULE: self.source, "g05_gdn_triangular": SHARED}[name]

        self.patchers = [patch.object(HELPER, "_ROW_OP", None), patch.object(HELPER, "_KERNEL", None),
                         patch.object(HELPER, "_make_kernel", return_value=Kernel()),
                         patch.object(HELPER.importlib, "import_module", side_effect=importer)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_custom_op_schema_fake_and_fresh_output_contract(self):
        op = HELPER.register_row_inverse_op()
        self.assertIs(HELPER.register_row_inverse_op(), op)
        self.assertEqual(self.registrations, [(HELPER.OP_NAME, {"mutates_args": (), "device_types": "npu"})])
        value = Tensor()
        first, second = op(value), op(value)
        self.assertIsNot(first, second)
        self.assertIsNot(first, value)
        self.assertEqual(first.shape, value.shape)
        fake = op.fake(value)
        self.assertIsNot(fake, value)
        self.assertEqual((fake.shape, fake.dtype, fake.device.type), (first.shape, first.dtype, first.device.type))
        self.assertEqual(len(self.launches), 2, "fake execution must not launch a kernel")
        self.assertEqual(self.launches[0][0], (80,))
        self.assertEqual(self.launches[0][3], {"BLOCK": 32, "enable_fp_fusion": False})

    def test_unsupported_metadata_rejected_in_real_and_fake_without_launch(self):
        op = HELPER.register_row_inverse_op()
        bad = [Tensor(device="cpu"), Tensor(dtype="bfloat16"), Tensor(shape=(0, 32, 32)),
               Tensor(shape=(2, 64, 64)), Tensor(shape=(32, 32)),
               Tensor(contiguous=False), Tensor(requires_grad=True)]
        for value in bad:
            for call in (op, op.fake):
                with self.subTest(shape=value.shape, dtype=value.dtype, fn=call):
                    with self.assertRaises((ValueError, TypeError)):
                        call(value)
        self.assertFalse(self.launches)

    def test_verify_is_mandatory_and_failure_does_not_patch_source(self):
        for flag in (False, None, 1, "yes"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "verify=True"):
                HELPER.install_triton_chunk(verify=flag)
        self.assertFalse(self.imports)
        with patch.object(HELPER, "register_row_inverse_op", side_effect=RuntimeError("registration failure")):
            with self.assertRaisesRegex(RuntimeError, "registration failure"):
                HELPER.install_triton_chunk(verify=True)
        self.assertIs(self.source.torch_chunk_gated_delta_rule, self.original)

    def test_install_routes_multidimensional_inverse_and_restores(self):
        callback_calls = []

        def shared_chunk(*args, inverse_impl=None, **kwargs):
            callback_calls.append((args, kwargs, inverse_impl))
            return inverse_impl(Tensor(shape=(2, 16, 5, 32, 32)))

        with patch.object(SHARED, "torch_chunk_gated_delta_rule_solve", side_effect=shared_chunk):
            handle = HELPER.install_triton_chunk(verify=True)
            value = self.source.torch_chunk_gated_delta_rule(Tensor(), None, None, None, None, chunk_size=32)
        self.assertEqual(value.shape, (2, 16, 5, 32, 32))
        self.assertEqual(self.launches[0][0], (160,))
        self.assertEqual(len(callback_calls), 1)
        self.assertFalse(handle.metadata["fallback_allowed"])
        self.assertFalse(handle.metadata["bitwise_equivalence_claimed"])
        self.assertEqual(handle.metadata["qk_l2norm_epsilon"], 1e-6)
        self.assertFalse(handle.metadata["inverse_epsilon_added"])
        handle.restore()
        handle.restore()
        self.assertIs(self.source.torch_chunk_gated_delta_rule, self.original)

    def test_non_npu_or_other_chunk_rejected_without_original_fallback(self):
        handle = HELPER.install_triton_chunk(verify=True)
        with self.assertRaisesRegex(ValueError, "NPU"):
            handle.installed(Tensor(device="cpu"), None, None, None, None, chunk_size=32)
        for kwargs in ({}, {"chunk_size": 64}, {"chunk_size": 16}):
            with self.assertRaisesRegex(ValueError, "chunk_size=32"):
                handle.installed(Tensor(), None, None, None, None, **kwargs)
        with self.assertRaisesRegex(ValueError, "owns"):
            handle.installed(Tensor(), None, None, None, None, chunk_size=32, inverse_impl=lambda x: x)
        self.assertFalse(self.launches)

    def test_nested_algorithms_graphs_and_later_external_replacement(self):
        for marker in HELPER._OTHER_MARKERS:
            setattr(self.original, marker, True)
            with self.subTest(marker=marker), self.assertRaisesRegex(RuntimeError, "only once"):
                HELPER.install_triton_chunk(verify=True)
            delattr(self.original, marker)
        handle = HELPER.install_triton_chunk(verify=True)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            HELPER.install_triton_chunk(verify=True)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            SHARED.install_triangular_chunk()
        later = lambda *args, **kwargs: "external"
        self.source.torch_chunk_gated_delta_rule = later
        handle.restore()
        self.assertIs(self.source.torch_chunk_gated_delta_rule, later)

    def test_inverse_seam_is_optional_keyword_only_and_keeps_solve_branch(self):
        tree = ast.parse(Path(SHARED.__file__).read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "torch_chunk_gated_delta_rule_solve")
        self.assertEqual([arg.arg for arg in function.args.kwonlyargs], ["inverse_impl"])
        self.assertIsNone(function.args.kw_defaults[0].value)
        branch = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "inverse_impl is None")
        self.assertIn("torch.linalg.solve_triangular", ast.unparse(branch.body[1]))
        self.assertEqual(ast.unparse(branch.orelse[0]), "attn = inverse_impl(attn.float())")


if __name__ == "__main__":
    unittest.main()
