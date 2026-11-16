"""No Torch/NPU: fixed-geometry guards, live inputs and camera isolation."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "tools/galaxeavla/g05_static_vision.py"
SPEC = importlib.util.spec_from_file_location("g05_static_vision", PATH)
HELPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPER
SPEC.loader.exec_module(HELPER)
DEVICE = SimpleNamespace(type="npu")


class Array:
    def __init__(self, data, dtype=None, device=DEVICE):
        self.data = np.asarray(data, dtype=dtype)
        self.dtype, self.device = self.data.dtype.name, device
        self.shape, self._version = self.data.shape, 0

    def is_contiguous(self): return self.data.flags.c_contiguous
    def tolist(self): return self.data.tolist()
    def to(self, device): return Array(self.data, device=device)
    def int(self): return Array(self.data.astype("int32"), device=self.device)
    def clip(self, max): return Array(self.data.clip(max=max), device=self.device)
    def __getitem__(self, key): return Array(self.data[key], device=self.device)
    @property
    def T(self): return Array(self.data.T, device=self.device)

    def _binary(self, other, fn):
        data = other.data if isinstance(other, Array) else other
        result = fn(self.data, data)
        # Match the float32/int32 promotion relevant to interpolation geometry.
        if self.dtype == "float32" or isinstance(other, Array) and other.dtype == "float32":
            result = np.asarray(result, dtype="float32")
        return Array(result, device=self.device)

    def __add__(self, other): return self._binary(other, np.add)
    def __sub__(self, other): return self._binary(other, np.subtract)
    def __mul__(self, other): return self._binary(other, np.multiply)
    def __rsub__(self, other): return Array(np.asarray(other - self.data, dtype=self.dtype), device=self.device)
    def reshape(self, *shape): return Array(self.data.reshape(*shape), device=self.device)
    view = reshape
    def permute(self, *axes): return Array(self.data.transpose(*axes), device=self.device)
    def transpose(self, a, b): return Array(self.data.swapaxes(a, b), device=self.device)
    def unsqueeze(self, axis): return Array(np.expand_dims(self.data, axis), device=self.device)
    def squeeze(self, axis): return Array(np.squeeze(self.data, axis), device=self.device)
    def contiguous(self): return Array(np.ascontiguousarray(self.data), device=self.device)
    def repeat(self, *counts): return Array(np.tile(self.data, counts), device=self.device)
    def cos(self): return Array(np.cos(self.data), device=self.device)
    def sin(self): return Array(np.sin(self.data), device=self.device)
    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0: end_dim += self.data.ndim
        shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim + 1:]
        return self.reshape(shape)
    def split(self, lengths, dim=0):
        assert sum(lengths) == self.shape[dim]
        return tuple(Array(x, device=self.device) for x in np.split(self.data, np.cumsum(lengths)[:-1], axis=dim))


class Attention:
    training = False
    num_heads, head_dim, scaling = 16, 64, 0.125
    def _attend_spatial(self, *args): return "original"


class Block:
    training = False
    def __init__(self): self.attn = Attention()
    def __call__(self, value, **kwargs): return value + np.float32(0.01)


class Position:
    def __init__(self):
        self.weight = Array(np.zeros((2304, 1024), dtype="float32"))
        self.calls = 0
    def __call__(self, indices):
        self.calls += 1
        return Array(self.weight.data[indices.data], device=self.weight.device)


class PatchEmbed:
    def __init__(self): self.proj = SimpleNamespace(weight=Array(np.zeros(1, dtype="float32")))
    def __call__(self, patches): return Array(patches.data[:, :1024].copy())


class Vision:
    training, gradient_checkpointing = False, False
    spatial_merge_size, patch_size, num_grid_per_side = 2, 16, 48
    def __init__(self):
        self.config = SimpleNamespace(depth=24, hidden_size=1024, out_hidden_size=2048,
            num_heads=16, patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, num_position_embeddings=2304)
        self.blocks = [Block() for _ in range(24)]
        self.pos_embed, self.patch_embed = Position(), PatchEmbed()
        self.rotary_pos_emb = SimpleNamespace(inv_freq=Array(np.zeros(16, dtype="float32")))
    def forward(self, *args, **kwargs): return "original"
    def named_parameters(self):
        return [("pos_embed.weight", self.pos_embed.weight), ("patch.weight", self.patch_embed.proj.weight)]
    def modules(self): return [self, *self.blocks, *(b.attn for b in self.blocks)]
    def rot_pos_emb(self, grid): return Array(np.zeros((512, 32), dtype="float32"))
    def merger(self, hidden): return Array(hidden.data.reshape(128, 4096)[:, :2048].copy())


def make_torch():
    return SimpleNamespace(float32="float32", bfloat16="bfloat16", long="int64", int32="int32",
        no_grad=nullcontext, autocast=lambda *args, **kwargs: nullcontext(),
        is_grad_enabled=lambda: False, get_default_dtype=lambda: "float32",
        is_autocast_enabled=lambda device: True, get_autocast_dtype=lambda device: "bfloat16",
        tensor=lambda data, dtype, device: Array(data, dtype, device),
        linspace=lambda start, end, steps: Array(np.linspace(start, end, steps, dtype="float32")),
        cat=lambda values, dim=0: Array(np.concatenate([v.data for v in values], axis=dim)),
        split=lambda value, lengths, dim=0: value.split(lengths, dim))


class StaticVisionTests(unittest.TestCase):
    def setUp(self):
        self.torch, self.vision = make_torch(), Vision()
        self.policy = SimpleNamespace(training=False, model=SimpleNamespace(training=False, vision_tower=self.vision))
        self.source = SimpleNamespace(Qwen3_5VisionModel=Vision, Qwen3_5VisionBlock=Block,
            Qwen3_5VisionAttention=Attention, __file__=__file__)
        self.compile_calls = []
        def compile_fn(torch, fn, backend, scope, options):
            self.compile_calls.append((backend, options))
            return fn, {"backend": backend, "options": options}
        self.graphs = SimpleNamespace(_compile_callable=compile_fn)
        def importer(name):
            return {"torch": self.torch, HELPER._SOURCE: self.source, "g05_npu_graphs": self.graphs}[name]
        self.import_patch = patch.object(HELPER.importlib, "import_module", side_effect=importer)
        self.import_patch.start()
        self.addCleanup(self.import_patch.stop)

    def inputs(self, value=0):
        return Array(np.full((512, 1536), value, dtype="float32")), Array(HELPER._GRID, "int64")

    def test_geometry_and_learned_position_lookup_remain_separate(self):
        geometry = HELPER._prepare_geometry(self.torch, self.vision)
        self.assertEqual(set(geometry), {"indices", "weights", "cos", "sin", "cu_seqlens"})
        self.assertEqual(geometry["indices"].shape, (4, 512))
        np.testing.assert_allclose(geometry["weights"].data.sum(0), 1.0, atol=1e-7)
        self.assertEqual(geometry["cu_seqlens"].tolist(), [0, 256, 512])
        core = HELPER._make_tensor_core(self.torch, self.vision, geometry)
        first = core(self.inputs(0)[0])
        changed_image = core(self.inputs(1)[0])
        np.testing.assert_allclose(changed_image[0].data - first[0].data, 1.0, atol=2e-6)
        self.vision.pos_embed.weight.data.fill(0.5)
        changed_position = core(self.inputs(0)[0])
        np.testing.assert_allclose(changed_position[0].data - first[0].data, 0.5, atol=2e-6)
        self.assertEqual(self.vision.pos_embed.calls, 3)
        self.assertEqual((first[0].shape, first[1].shape), ((512, 1024), (128, 2048)))

    def test_attention_preserves_two_separate_camera_sdpa_calls(self):
        calls = []
        def sdpa(q, k, v, **kwargs):
            calls.append((q.shape, k.shape, v.shape, kwargs))
            # Zero q/k give uniform attention independently within each camera.
            return Array(np.broadcast_to(v.data.mean(axis=2, keepdims=True), v.shape).copy())
        self.torch.nn = SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=sdpa))
        q = Array(np.zeros((512, 1, 2), dtype="float32"))
        values = Array(np.concatenate([np.ones((256, 1, 2), dtype="float32"),
                                       np.full((256, 1, 2), 7, dtype="float32")]))
        result = HELPER._make_spatial_attention(self.torch)(Attention(), q, q, values, None)
        np.testing.assert_array_equal(result.data[:256], 1)
        np.testing.assert_array_equal(result.data[256:], 7)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[0] == (1, 1, 256, 2) and call[3] == {"scale": 0.125} for call in calls))

    def test_install_runs_fresh_inputs_and_restores_all_instance_methods(self):
        handle = HELPER.install_static_vision(self.policy, verify=True)
        a, grid = self.inputs(0)
        first = self.vision.forward(a, grid)
        second = self.vision.forward(self.inputs(1)[0], grid)
        self.assertFalse(np.array_equal(first[0].data, second[0].data))
        self.assertEqual(handle.metadata["calls"], 2)
        self.assertEqual(self.compile_calls, [("torchair_aclgraph", {"frozen_parameter": True})])
        handle.restore(); handle.restore()
        self.assertNotIn("forward", vars(self.vision))
        self.assertTrue(all("_attend_spatial" not in vars(block.attn) for block in self.vision.blocks))

    def test_geometry_content_is_checked_even_when_shape_matches(self):
        handle = HELPER.install_static_vision(self.policy, verify=True)
        patches, grid = self.inputs()
        grid.data[0] = [1, 8, 32]
        with self.assertRaisesRegex(ValueError, "exact two-camera"):
            self.vision.forward(patches, grid)
        for kwargs in ({"num_frames": 2}, {"bsz": 2}):
            with self.assertRaisesRegex(ValueError, "num_frames"):
                self.vision.forward(patches, self.inputs()[1], **kwargs)
        self.assertEqual(handle.metadata["calls"], 0)

    def test_weight_rope_and_training_changes_invalidate_before_graph_call(self):
        HELPER.install_static_vision(self.policy, verify=True)
        values = self.inputs()
        for tensor in (self.vision.pos_embed.weight, self.vision.rotary_pos_emb.inv_freq):
            tensor._version += 1
            with self.assertRaisesRegex(ValueError, "restore and reinstall"):
                self.vision.forward(*values)
            tensor._version -= 1
        self.vision.blocks[0].training = True
        with self.assertRaisesRegex(ValueError, "eval"):
            self.vision.forward(*values)
        self.vision.blocks[0].training = False
        self.torch.is_grad_enabled = lambda: True
        with self.assertRaisesRegex(ValueError, "no-grad"):
            self.vision.forward(*values)
        self.torch.is_grad_enabled = lambda: False
        self.torch.is_autocast_enabled = lambda device: False
        with self.assertRaisesRegex(ValueError, "BF16 autocast"):
            self.vision.forward(*values)

    def test_default_disabled_compile_failure_rollback_and_external_replacement(self):
        with self.assertRaisesRegex(ValueError, "verify=True"):
            HELPER.install_static_vision(self.policy)
        self.graphs._compile_callable = lambda *args: (_ for _ in ()).throw(RuntimeError("compile failure"))
        with self.assertRaisesRegex(RuntimeError, "compile failure"):
            HELPER.install_static_vision(self.policy, verify=True)
        self.assertNotIn("forward", vars(self.vision))
        self.assertTrue(all("_attend_spatial" not in vars(block.attn) for block in self.vision.blocks))
        self.graphs._compile_callable = lambda torch, fn, *args: (fn, {})
        handle = HELPER.install_static_vision(self.policy, verify=True)
        later = lambda *args: "external"
        self.vision.forward = later
        handle.restore()
        self.assertIs(self.vision.forward, later)


if __name__ == "__main__":
    unittest.main()
