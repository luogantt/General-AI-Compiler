"""Default-off Ascend C fusion of AE Q/K RMSNorm, partial RoPE and layout.

Load the owner-built eager bridge from GAC_QKNORM_ROPE_LIBRARY. The companion
OPP package must already be visible to CANN; this module does not install it.
The operator preserves the BF16 norm-output round trip before FP32 RoPE.
Only 24 AE layer instances are rewritten. Install attention/time/native norm
first, this helper instead of the standalone RoPE helper, then compile AE.
Restore in reverse order. Operator/fake/converter registrations are process-wide
and deliberately persist after restoring the instance-local model changes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.machinery
import inspect
import linecache
import math
import os
from pathlib import Path
import sys
import textwrap
from types import MethodType, ModuleType
import uuid

SOURCE_MODULE = "g05.models.g05.qwen35.mixture_qwen35"
NORM_MODULE = "g05.models.g05.qwen35.modules"
OP_NAME = "gac_g05::qk_norm_rope"
GE_OP_NAME = "GacQkNormRope"
MARKER = "_g05_ae_qknorm_rope"
CALLBACK = "_gac_ae_qknorm_rope"
_REGISTRATION = None

_QK_SOURCE = """
qg = attn.q_proj(hidden_states)
query_states, attn_gate = torch.chunk(qg.view(bsz, seq_len, -1, head_dim * 2), 2, dim=-1)
attn_gate = attn_gate.reshape(bsz, seq_len, -1)
query_states = attn.q_norm(query_states).transpose(1, 2)
key_states = attn.k_proj(hidden_states)
key_states = key_states.view(bsz, seq_len, num_kv_heads, head_dim)
key_states = attn.k_norm(key_states).transpose(1, 2)
"""
_QK_REPLACEMENT = """
qg = attn.q_proj(hidden_states)
query_states, attn_gate = torch.chunk(qg.view(bsz, seq_len, -1, head_dim * 2), 2, dim=-1)
attn_gate = attn_gate.reshape(bsz, seq_len, -1)
query_states = _gac_ae_qknorm_rope(qg, attn.q_norm.weight, cos, sin, 512, attn.q_norm.eps)
key_states = attn.k_proj(hidden_states)
key_states = _gac_ae_qknorm_rope(key_states, attn.k_norm.weight, cos, sin, 256, attn.k_norm.eps)
"""
_ROPE_SOURCE = """
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


def _key(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


_QK_KEYS = tuple(_key(node) for node in ast.parse(_QK_SOURCE).body)
_ROPE_KEY = _key(ast.parse(_ROPE_SOURCE).body[0])
AUDITED_QK_AST_SHA256 = hashlib.sha256("\n".join(_QK_KEYS).encode()).hexdigest()
AUDITED_ROPE_AST_SHA256 = hashlib.sha256(_ROPE_KEY.encode()).hexdigest()


def validate_inputs(torch, x, weight, cos, sin, head_stride, epsilon):
    """Metadata-only contract, shared by eager dispatch and FakeTensor."""
    if type(head_stride) is not int or head_stride not in (256, 512):
        raise ValueError("head_stride must be 256 (K) or 512 (Q), in elements")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    heads, channels = (8, 4096) if head_stride == 512 else (2, 512)
    if tuple(x.shape) != (1, 32, channels) or x.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 x[1,32,4096] for Q or x[1,32,512] for K")
    if tuple(weight.shape) != (256,) or weight.dtype != torch.float32:
        raise ValueError("Expected original FP32 RMSNorm weight[256], before adding one")
    if (cos.ndim != 4 or tuple(cos.shape[:2]) != (1, 1) or cos.shape[2] < 32
            or cos.shape[3] != 64 or tuple(sin.shape) != tuple(cos.shape)
            or cos.dtype != torch.float32 or sin.dtype != torch.float32):
        raise ValueError("Expected dynamic FP32 cos/sin[1,1,S>=32,64]")
    if x.device.type != "npu" or any(value.device != x.device for value in (weight, cos, sin)):
        raise ValueError("QK fusion requires all inputs on the same NPU")
    if any(not value.is_contiguous() for value in (x, weight, cos, sin)):
        raise ValueError("QK fusion inputs must be contiguous; no hidden packing is allowed")
    # An eval Parameter can retain requires_grad=True while inference_mode is
    # active. The public model callback checks grad mode; fake inference must
    # not reject those unchanged model Parameters merely for their flag.
    return (1, heads, 32, 256)


def register_qknorm_rope_op():
    """Load one explicit .so and register fake/GE adapters once per process."""
    global _REGISTRATION
    configured = os.environ.get("GAC_QKNORM_ROPE_LIBRARY")
    if not configured:
        raise RuntimeError("Set GAC_QKNORM_ROPE_LIBRARY to the built eager-bridge shared library")
    path = Path(configured).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("GAC_QKNORM_ROPE_LIBRARY must name a file")
    if _REGISTRATION is not None:
        if _REGISTRATION[1]["library"] != str(path):
            raise RuntimeError("A different QK fusion library is already registered in this process")
        return _REGISTRATION
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    torch.ops.load_library(str(path))
    op = torch.ops.gac_g05.qk_norm_rope.default

    def fake(x, weight, cos, sin, head_stride: int, epsilon: float):
        shape = validate_inputs(torch, x, weight, cos, sin, head_stride, epsilon)
        return torch.empty(shape, device=x.device, dtype=torch.bfloat16)

    torch.library.register_fake(OP_NAME)(fake)
    torchair = importlib.import_module("torchair")

    converters = importlib.import_module("torchair.ge_concrete_graph.fx2ge_converter")

    @converters.register_fx_node_ge_converter(op)
    def converter(x, weight, cos, sin, head_stride: int, epsilon: float, meta_outputs=None):
        return torchair.ge.custom_op(
            GE_OP_NAME, inputs={"x": x, "weight": weight, "cos": cos, "sin": sin},
            attrs={"head_stride": torchair.ge.attr.Int(head_stride),
                   "epsilon": torchair.ge.attr.Float(epsilon)}, outputs=["y"])

    metadata = {"operator": OP_NAME, "ge_operator": GE_OP_NAME, "library": str(path),
                "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fake_registered": True, "ge_converter_registered": True,
                "registration_lifetime": "process; model restore does not unload the operator"}
    _REGISTRATION = (op, metadata)
    return _REGISTRATION


def make_qknorm_rope(torch, op):
    def call(x, weight, cos, sin, head_stride, epsilon):
        if torch.is_grad_enabled():
            raise ValueError("QK fusion requires gradients disabled")
        validate_inputs(torch, x, weight, cos, sin, head_stride, epsilon)
        with torch.autocast("npu", enabled=False):
            return op(x, weight, cos, sin, head_stride, float(epsilon))
    return call


def _make_private_forward(function, callback):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or function.__closure__:
        raise ValueError("Expected one plain, non-closure layer forward")
    target = tree.body[0]
    matches = []
    for node in ast.walk(target):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            body = node.body
            for i in range(len(body) - len(_QK_KEYS) + 1):
                if tuple(_key(item) for item in body[i:i + len(_QK_KEYS)]) == _QK_KEYS:
                    matches.append((body, i))
    if len(matches) != 1:
        raise ValueError(f"Expected one audited QK projection/norm sequence, found {len(matches)}")
    body, start = matches[0]
    rope_indices = [i for i, node in enumerate(body) if _key(node) == _ROPE_KEY]
    if len(rope_indices) != 1 or rope_indices[0] <= start + len(_QK_KEYS):
        raise ValueError("Expected one audited partial-RoPE block after QK/value projections")
    # Remove the original RoPE after validating both targets, then splice the
    # QK block. Everything else, including gate, V, cache and dispatch, stays.
    del body[rope_indices[0]]
    body[start:start + len(_QK_KEYS)] = ast.parse(_QK_REPLACEMENT).body
    target.name, target.decorator_list = "ae_layer_forward", []
    rendered = ast.unparse(ast.fix_missing_locations(target)) + "\n"
    name = "_g05_ae_qknorm_rope_" + uuid.uuid4().hex
    filename = "<" + name + ">"
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__)
    namespace.__name__, namespace.__package__, namespace.__file__ = name, "", filename
    namespace.__loader__ = None
    namespace.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    namespace.__dict__[CALLBACK] = callback
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
class QKNormRopeInstallation:
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


def install_ae_qknorm_rope(policy, *, verify=False):
    if verify is not True:
        raise ValueError("QK fusion requires verify=True and independent full-model verification")
    torch = importlib.import_module("torch")
    source = importlib.import_module(SOURCE_MODULE)
    norms = importlib.import_module(NORM_MODULE)
    model, expert = policy.model, policy.model.action_expert
    if type(expert) is not source.MixtureQwen35 or len(expert.layers) != 24 or list(expert.layer_types) != ["full_attention"] * 24:
        raise TypeError("QK fusion requires the exact 24-full-attention AE")
    if any(module.training for module in (policy, model, expert, *expert.layers)):
        raise ValueError("QK fusion requires evaluation mode")
    if "forward" in vars(expert) or hasattr(expert, "_orig_mod"):
        raise RuntimeError("Install QK fusion before AE compilation")
    functions = []
    for layer in expert.layers:
        function = getattr(layer.forward, "__func__", None)
        if type(layer) is not source.MixtureDecoderLayerQwen35 or function is None:
            raise TypeError("Unsupported AE layer")
        if function.__code__ is not source.MixtureDecoderLayerQwen35.forward.__code__:
            raise TypeError("QK fusion conflicts with existing layer rewrites, including standalone RoPE")
        if function is not source.MixtureDecoderLayerQwen35.forward and not getattr(function, "_g05_ae_fused_attention", False):
            raise TypeError("Only the original or private attention forward is supported")
        attn = layer.self_attn
        if (attn.num_heads, attn.num_key_value_heads, attn.head_dim) != (8, 2, 256):
            raise ValueError("Expected 8/2 heads and head width 256")
        for norm in (attn.q_norm, attn.k_norm):
            if type(norm) is not norms.Qwen3_5RMSNorm or norm.training:
                raise TypeError("Expected original eval Q/K RMSNorm modules")
            forward = norm.forward
            if (getattr(forward, "__func__", None) is not norms.Qwen3_5RMSNorm.forward
                    and not getattr(forward, "_g05_ae_fused_norm", False)):
                raise TypeError("Unsupported Q/K norm forward replacement")
            if norm.weight.dtype != torch.float32 or norm.weight.device.type != "npu" or tuple(norm.weight.shape) != (256,):
                raise ValueError("Expected FP32 NPU Q/K weight[256] islands")
            if isinstance(norm.eps, bool) or not isinstance(norm.eps, (int, float)) or not math.isfinite(norm.eps) or norm.eps <= 0:
                raise ValueError("Q/K RMSNorm epsilon must be finite and positive")
        functions.append(function)
    if any(fn.__globals__ is not functions[0].__globals__ for fn in functions):
        raise ValueError("AE layers must share one audited private attention namespace")
    op, registration = register_qknorm_rope_op()
    namespace, cloned = _make_private_forward(functions[0], make_qknorm_rope(torch, op))
    metadata = {**registration, "enabled": True, "default_enabled": False, "experimental": True,
                "scope": "24 AE layers; Q/K norm + partial RoPE64 + BSHD-to-BHSD layout",
                "input": "BF16 contiguous x[1,32,4096/512], FP32 weight[256], cos/sin[1,1,S>=32,64]",
                "output": "fresh contiguous BF16[1,8/2,32,256]",
                "precision": "FP32 RMSNorm with current (1+weight); RNE BF16 round trip; FP32 partial RoPE; RNE BF16 output",
                "existing_qk_norm_wrappers_bypassed": True, "adaptive_norm_unchanged": True,
                "native_attention_dispatch_inherited": True, "vlm_unchanged": True,
                "cached_model_inputs_or_outputs": False, "cached_positions_or_frequencies": False,
                "fallback_allowed": False, "verify_required": True,
                "qk_ast_sha256": AUDITED_QK_AST_SHA256, "rope_ast_sha256": AUDITED_ROPE_AST_SHA256,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "private_module": namespace.__name__, "runtime_validation": "not performed by installer"}
    handle = QKNormRopeInstallation(metadata, namespace)
    try:
        for layer in expert.layers:
            installed = MethodType(cloned, layer)
            had_attribute, previous = "forward" in vars(layer), vars(layer).get("forward")
            layer.forward = installed
            handle._replacements.append((layer, installed, had_attribute, previous))
    except Exception:
        handle.restore()
        raise
    return handle
