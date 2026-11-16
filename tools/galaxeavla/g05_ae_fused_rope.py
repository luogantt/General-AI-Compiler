"""Opt-in AE partial RoPE: source-pinned AST replacement, FP32 arithmetic.

Install fused attention first (when selected), then this helper, then compile
AE.forward. Restore in reverse order. Current private attention globals are
inherited, never replaced by the upstream registry. Only the audited RoPE With
block is changed; projections, QK norms, attention, residuals and MLPs retain
their original source. Dynamic cos/sin are consumed on every call.

``sliced`` fuses the first 64 dimensions then concatenates the untouched tail.
``matrix`` passes full width 256 and a fixed nonlearned rotation matrix to the
NPU operator, with identity/zero padding of the current cos/sin. Both retain
FP32 RoPE and the upstream cast back to the incoming Q/K dtype. Device support,
graph conversion, full-model accuracy and latency require owner-run validation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.machinery
import inspect
import linecache
import sys
import textwrap
from types import MethodType, ModuleType
import uuid


SOURCE_MODULE = "g05.models.g05.qwen35.mixture_qwen35"
MARKER = "_g05_ae_fused_rope"
ATTENTION_MARKER = "_g05_ae_fused_attention"
_CALLBACK = "_gac_ae_partial_rope_fp32"
_AUDITED_BLOCK = """
with torch.autocast(query_states.device.type, enabled=False):
    orig_qk_dtype = query_states.dtype
    q_fp32 = query_states.float()
    k_fp32 = key_states.float()
    _cos = cos.float()
    _sin = sin.float()
    q_seq_len = q_fp32.shape[-2]
    if _cos.shape[-2] != q_seq_len:
        _cos = _cos[:, :, :q_seq_len, :]
        _sin = _sin[:, :, :q_seq_len, :]
    rotary_dim = _cos.shape[-1]
    q_rot, q_pass = q_fp32[..., :rotary_dim], q_fp32[..., rotary_dim:]
    k_rot, k_pass = k_fp32[..., :rotary_dim], k_fp32[..., rotary_dim:]
    q_rotated = q_rot * _cos + rotate_half(q_rot) * _sin
    k_rotated = k_rot * _cos + rotate_half(k_rot) * _sin
    query_states = torch.cat([q_rotated, q_pass], dim=-1).to(orig_qk_dtype)
    key_states = torch.cat([k_rotated, k_pass], dim=-1).to(orig_qk_dtype)
"""


def _ast_key(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


_AUDITED_KEY = _ast_key(ast.parse(textwrap.dedent(_AUDITED_BLOCK)).body[0])
AUDITED_ROPE_AST_SHA256 = hashlib.sha256(_AUDITED_KEY.encode()).hexdigest()


def make_partial_rope(torch, torch_npu, mode, rotation=None):
    if mode not in {"sliced", "matrix"}:
        raise ValueError("RoPE mode must be 'sliced' or 'matrix'")
    if mode == "matrix" and rotation is None:
        raise ValueError("Matrix RoPE requires its fixed rotation tensor")

    def partial_rope(query, key, cos, sin):
        if torch.is_grad_enabled():
            raise ValueError("Fused AE RoPE is inference-only")
        if query.device.type != "npu" or key.device != query.device or cos.device != query.device or sin.device != query.device:
            raise ValueError("Fused AE RoPE requires tensors on the same NPU")
        if (query.ndim != 4 or key.ndim != 4 or query.shape[1:] != (8, 32, 256)
                or key.shape != (query.shape[0], 2, 32, 256) or query.shape[0] < 1):
            raise ValueError("Expected AE Q[B,8,32,256] and K[B,2,32,256]")
        if (cos.ndim != 4 or sin.shape != cos.shape or cos.shape[0] not in (1, query.shape[0])
                or cos.shape[1] != 1 or cos.shape[-2] < 32 or cos.shape[-1] != 64):
            raise ValueError("Expected dynamic cos/sin [B or 1,1,S>=32,64]")
        if query.dtype not in (torch.float32, torch.bfloat16, torch.float16) or key.dtype != query.dtype:
            raise TypeError("Expected matching floating-point Q/K dtypes")
        original_dtype = query.dtype
        with torch.autocast(query.device.type, enabled=False):
            q, k = query.float(), key.float()
            c, s = cos.float(), sin.float()
            if c.shape[-2] != q.shape[-2]:
                c, s = c[:, :, :q.shape[-2], :], s[:, :, :q.shape[-2], :]
            if mode == "sliced":
                q_rot = torch_npu.npu_rotary_mul(q[..., :64], c, s)
                k_rot = torch_npu.npu_rotary_mul(k[..., :64], c, s)
                q_result = torch.cat((q_rot, q[..., 64:]), dim=-1)
                k_result = torch.cat((k_rot, k[..., 64:]), dim=-1)
            else:
                if rotation.device != query.device or rotation.dtype != torch.float32 or rotation.shape != (256, 256):
                    raise ValueError("Fixed FP32 rotation tensor does not match the current device")
                c = torch.cat((c, torch.ones_like(c).repeat(1, 1, 1, 3)), dim=-1)
                s = torch.cat((s, torch.zeros_like(s).repeat(1, 1, 1, 3)), dim=-1)
                q_result = torch_npu.npu_rotary_mul(q, c, s, "half", rotation)
                k_result = torch_npu.npu_rotary_mul(k, c, s, "half", rotation)
            return q_result.to(original_dtype), k_result.to(original_dtype)
    return partial_rope


def _make_private_forward(function, callback):
    """Rewrite exactly one audited AST subtree, preserving all other statements."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one plain upstream layer forward function")
    target = tree.body[0]
    count = 0
    class ReplaceRoPE(ast.NodeTransformer):
        def visit_With(self, node):
            nonlocal count
            if _ast_key(node) == _AUDITED_KEY:
                count += 1
                replacement = ast.parse(f"query_states, key_states = {_CALLBACK}(query_states, key_states, cos, sin)").body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)
    target = ReplaceRoPE().visit(target)
    if count != 1:
        raise ValueError(f"Expected exactly one audited FP32 partial-RoPE AST block, found {count}")
    if function.__closure__:
        raise ValueError("AE source forward must not close over nonlocal state")
    target.name, target.decorator_list = "ae_layer_forward", []
    rendered = ast.unparse(ast.fix_missing_locations(target)) + "\n"
    name = "_g05_ae_rope_" + uuid.uuid4().hex
    filename = "<" + name + ">"
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__)
    namespace.__name__, namespace.__package__, namespace.__file__ = name, "", filename
    namespace.__loader__ = None
    namespace.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    namespace.__dict__[_CALLBACK] = callback
    linecache.cache[filename] = (len(rendered), None, rendered.splitlines(keepends=True), filename)
    try:
        exec(compile(rendered, filename, "exec"), namespace.__dict__)
        cloned = namespace.ae_layer_forward
        cloned.__module__, cloned.__qualname__ = name, "ae_layer_forward"
        cloned.__defaults__, cloned.__kwdefaults__ = function.__defaults__, function.__kwdefaults__
        setattr(cloned, MARKER, True)
        sys.modules[name] = namespace
        return namespace, cloned
    except Exception:
        linecache.cache.pop(filename, None)
        raise


@dataclass
class FusedRopeInstallation:
    metadata: dict
    _namespace: object = field(repr=False)
    _replacements: list = field(default_factory=list, repr=False)
    _restored: bool = False

    def restore(self):
        if self._restored:
            return
        external = False
        for layer, installed, had_attribute, previous in reversed(self._replacements):
            if vars(layer).get("forward") is installed:
                if had_attribute:
                    layer.forward = previous
                else:
                    delattr(layer, "forward")
            else:
                external = True
        if not external:
            if sys.modules.get(self._namespace.__name__) is self._namespace:
                del sys.modules[self._namespace.__name__]
            linecache.cache.pop(self._namespace.__file__, None)
        self.metadata["private_namespace_retained_for_external_replacement"] = external
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_ae_fused_rope(policy, mode="sliced", *, verify=False):
    """Install after optional fused attention and before AE graph compilation."""
    if verify is not True:
        raise ValueError("Fused AE RoPE requires verify=True and independent full-model verification")
    if mode not in {"sliced", "matrix"}:
        raise ValueError("RoPE mode must be 'sliced' or 'matrix'")
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    source = importlib.import_module(SOURCE_MODULE)
    model, expert = policy.model, policy.model.action_expert
    if type(expert) is not source.MixtureQwen35 or len(expert.layers) != 24 or list(expert.layer_types) != ["full_attention"] * 24:
        raise TypeError("Fused RoPE requires the audited 24-full-attention AE")
    if any(module.training for module in (policy, model, expert, *expert.layers)):
        raise ValueError("Fused RoPE requires evaluation mode")
    if "forward" in vars(expert) or hasattr(expert, "_orig_mod"):
        raise RuntimeError("Install fused RoPE before AE.forward compilation")
    functions = []
    for layer in expert.layers:
        function = getattr(layer.forward, "__func__", None)
        if getattr(function, MARKER, False):
            raise RuntimeError("Fused AE RoPE is already installed")
        if type(layer) is not source.MixtureDecoderLayerQwen35 or function is None:
            raise TypeError("Unsupported AE layer class/forward")
        if function.__code__ is not source.MixtureDecoderLayerQwen35.forward.__code__:
            raise TypeError("Unsupported forward rewrite; only original or private attention clone is supported")
        if function is not source.MixtureDecoderLayerQwen35.forward and not getattr(function, ATTENTION_MARKER, False):
            raise TypeError("Unsupported private forward clone")
        if (layer.self_attn.num_heads, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim) != (8, 2, 256):
            raise ValueError("Expected AE attention dimensions 8/2 heads and width 256")
        functions.append(function)
    if any(function.__globals__ is not functions[0].__globals__ for function in functions):
        raise ValueError("AE layer forwards must share one audited attention namespace")
    rotation = None
    if mode == "matrix":
        device = expert.layers[0].self_attn.q_proj.weight.device
        if device.type != "npu":
            raise ValueError("Matrix RoPE requires the NPU model device")
        with torch.no_grad():
            rotation = torch.zeros(256, 256, device=device, dtype=torch.float32)
            indices = torch.arange(32, device=device)
            rotation[indices, indices + 32] = 1
            rotation[indices + 32, indices] = -1
    callback = make_partial_rope(torch, torch_npu, mode, rotation)
    namespace, cloned = _make_private_forward(functions[0], callback)
    metadata = {"enabled": True, "default_enabled": False, "experimental": True, "mode": mode,
                "scope": "Only AE layer partial RoPE block; 24 layers, head256/rotary64",
                "operator": "torch_npu.npu_rotary_mul", "math_dtype": "torch.float32",
                "output_dtype": "Original incoming Q/K dtype", "cached_cos_sin": False,
                "rotation": "constant FP32 256x256; R[i,i+32]=1, R[i+32,i]=-1 for i<32, all other entries zero" if mode == "matrix" else None,
                "current_attention_dispatch_inherited": True, "global_registry_modified": False,
                "audited_rope_ast_sha256": AUDITED_ROPE_AST_SHA256,
                "private_module": namespace.__name__, "source_namespace": functions[0].__globals__.get("__name__"),
                "fallback_allowed": False, "rounding_order_may_change": True,
                "runtime_validation": "Not performed by installer; full-model replay required"}
    handle = FusedRopeInstallation(metadata, namespace)
    try:
        for layer in expert.layers:
            installed = MethodType(cloned, layer)
            old = ("forward" in vars(layer), vars(layer).get("forward"))
            layer.forward = installed
            handle._replacements.append((layer, installed, *old))
    except Exception:
        handle.restore()
        raise
    return handle
