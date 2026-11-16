"""Opt-in AE-only fused attention experiment for pinned G0.5 LIBERO inference.

The real-number attention equation, input Q/K/V, scale, mask, existing GQA repeat,
FP32 QK normalization/RoPE and gated output are retained. Fused kernels change
intermediate rounding: this is NOT a bitwise-equivalence claim. Validate complete
FM10 actions against the unmodified eager path before accepting any timings.

Only AE layer forward functions receive a registered private Python module.
The VLM, Transformers registry, upstream source and site-packages are unchanged.
The original action-mask producer is wrapped to validate each newly built mask
once per request, before the ten-step FM loop. Its scalar device synchronization
remains inside the request's measured action time; no real inputs are cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.machinery
import sys
from types import FunctionType, MethodType, ModuleType
from typing import Any
import uuid


SOURCE_MODULE = "g05.models.g05.qwen35.mixture_qwen35"
MODEL_MODULE = "g05.models.g05.g05_model_qwen35"
_MARKER = "_g05_ae_fused_attention"


def _check_inference(policy, model, ae):
    if any(getattr(obj, "training", None) is not False for obj in (policy, model, ae, *ae.layers)):
        raise ValueError("AE fused attention requires evaluation mode")
    if getattr(model, "attn_implementation", None) != "eager":
        raise ValueError("AE fused attention requires the original eager attention contract")
    if getattr(model.fm_helper, "action_causal", None) is not False:
        raise ValueError("AE fused attention experiment requires noncausal FM actions")


def validate_action_mask(torch, mask):
    """One scalar read per request; caller must keep this outside compiled AE.

    Finite additive biases are deliberately unsupported: only exact zero,
    the mask dtype's minimum finite value, or negative infinity are accepted.
    Every row must retain an unmasked key, avoiding different all-masked-row
    semantics between finite-minimum eager softmax and fused attention.
    """
    if getattr(mask.device, "type", None) != "npu":
        raise ValueError("AE fused attention mask must be on NPU")
    if mask.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("AE action mask must be FP32/BF16/FP16 additive values")
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-2] != 32 or mask.shape[-1] < 32:
        raise ValueError("Expected AE action mask [B,1,32,K>=32]")
    allowed = (mask == 0) | (mask == torch.finfo(mask.dtype).min) | (mask == float("-inf"))
    # One .item(), not one per layer/FM step. Combining reductions preserves a
    # strict error for NaN, +inf, arbitrary additive bias or a fully masked row.
    valid = allowed.all() & (mask == 0).any(dim=-1).all() & (mask[..., -32:] == 0).all()
    if not bool(valid.item()):
        raise ValueError("AE mask must contain only zero/negative-extreme values and no fully masked row")
    # AE layers without VLM prefix read the trailing action-self mask; verify
    # its rows too, so a valid prefix cannot hide a fully masked action row.
    # All actions are bidirectional in this experiment: their full self block
    # must be zero, as included in the single read above.


def make_attention(torch, torch_npu, backend):
    """Construct the tensor-only replacement called after upstream GQA/RoPE."""
    if backend not in {"npu", "sdpa"}:
        raise ValueError(f"Unsupported AE fused attention backend: {backend}")

    def attention(module, query, key, value, attention_mask, scaling,
                  dropout=0.0, softcap=None, **kwargs):
        if module.training or module.num_key_value_groups != 1:
            raise ValueError("AE fused attention expects eval and already-expanded GQA")
        if dropout != 0.0 or softcap is not None or kwargs:
            raise ValueError("AE fused attention does not support dropout, softcap or additional arguments")
        if query.device.type != "npu" or key.device != query.device or value.device != query.device:
            raise ValueError("AE fused attention requires Q/K/V on the same NPU")
        if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
            raise TypeError("AE fused attention requires matching BF16/FP16 Q/K/V")
        if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape
                or query.shape[1:] != (8, 32, 256) or key.shape[0] != query.shape[0]
                or key.shape[1] != 8 or key.shape[-1] != 256 or key.shape[-2] < 32):
            raise ValueError("Expected expanded LIBERO AE Q[B,8,32,256], K/V[B,8,K>=32,256]")
        if scaling != 256 ** -0.5:
            raise ValueError("Expected unchanged AE attention scaling 1/sqrt(256)")
        if (attention_mask is None or attention_mask.device != query.device
                or attention_mask.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                or attention_mask.shape != (query.shape[0], 1, 32, key.shape[-2])):
            raise ValueError("Expected validated and upstream-sliced AE additive mask [B,1,32,K]")
        # Mask values were validated once by the original producer's wrapper.
        # NPU FA True means blocked; PyTorch SDPA True means allowed.
        if backend == "npu":
            output = torch_npu.npu_fusion_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                head_num=query.shape[1], input_layout="BNSD",
                atten_mask=(attention_mask != 0).contiguous(),
                scale=scaling, keep_prob=1.0, sparse_mode=0,
                pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0,
            )[0]
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=(attention_mask == 0).contiguous(), dropout_p=0.0,
                is_causal=False, scale=scaling, enable_gqa=False,
            )
        return output.transpose(1, 2).contiguous(), None

    return attention


class _PrivateAttentionDispatch:
    def __init__(self, attention):
        self.attention = attention

    def get(self, name, default=None):
        if name != "eager":
            raise ValueError("AE fused attention requires the eager entry-point name")
        return self.attention


def _make_private_forward(function, attention):
    """Give Dynamo a real, uniquely resolvable module for cloned globals.

    A copied globals dict retaining the original module name is insufficient:
    Dynamo may guard ALL_ATTENTION_FUNCTIONS through that original module and
    resolve Transformers' registry instead of our private dispatch object.
    Both the globals name and the function's module/qualname must resolve to
    the exact registered namespace and function used for execution.
    """
    name = "_g05_ae_attention_" + uuid.uuid4().hex
    namespace = ModuleType(name)
    namespace.__dict__.update(function.__globals__)
    namespace.__name__ = name
    namespace.__package__ = ""
    namespace.__loader__ = None
    namespace.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    namespace.ALL_ATTENTION_FUNCTIONS = _PrivateAttentionDispatch(attention)
    cloned = FunctionType(function.__code__, namespace.__dict__, "ae_layer_forward",
                          function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__module__, cloned.__qualname__ = name, "ae_layer_forward"
    cloned.__doc__ = function.__doc__
    setattr(cloned, _MARKER, True)
    namespace.ae_layer_forward = cloned
    sys.modules[name] = namespace
    return namespace, cloned


@dataclass
class FusedAttentionInstallation:
    metadata: dict[str, Any]
    _replacements: list = field(default_factory=list, repr=False)
    _private_module: Any = field(default=None, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self):
        """Restore backend forward wrappers first, then this handle; idempotent."""
        if self._restored:
            return
        external_replacement = False
        for obj, name, installed, had_attribute, prior_attribute in reversed(self._replacements):
            if getattr(obj, name, None) is installed:
                if had_attribute:
                    setattr(obj, name, prior_attribute)
                else:
                    delattr(obj, name)
            else:
                # A later wrapper could still call our cloned function. Keep
                # its guard namespace available rather than breaking it.
                external_replacement = True
        namespace = self._private_module
        if namespace is not None and not external_replacement:
            if sys.modules.get(namespace.__name__) is namespace:
                del sys.modules[namespace.__name__]
        self.metadata["private_namespace_retained_for_external_replacement"] = external_replacement
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def install_ae_fused_attention(policy, backend="npu"):
    """Install before AE compilation; requires same-device full-model verification.

    ``npu`` directly calls the fused operator. ``sdpa`` is a separate experiment:
    its selected implementation must be confirmed in a profiler and may use math.
    Neither path changes the model's serialized attention implementation field.
    """
    if backend not in {"npu", "sdpa"}:
        raise ValueError(f"Unsupported AE fused attention backend: {backend}")
    source = importlib.import_module(SOURCE_MODULE)
    model_source = importlib.import_module(MODEL_MODULE)
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu") if backend == "npu" else None
    model, ae = policy.model, policy.model.action_expert
    _check_inference(policy, model, ae)
    if type(model) is not model_source.G05ModelQwen35 or type(ae) is not source.MixtureQwen35:
        raise TypeError("AE fused attention requires exact upstream Qwen3.5 model and mixture classes")
    if len(ae.layers) != 24 or list(ae.layer_types) != ["full_attention"] * 24:
        raise ValueError("AE fused attention requires the 24 full-attention LIBERO layers")
    original_builder = model.build_action_mask_and_position_ids
    if getattr(original_builder, "__func__", None) is not model_source.G05ModelQwen35.build_action_mask_and_position_ids:
        raise TypeError("Unsupported or already replaced action-mask producer")
    originals = []
    for layer in ae.layers:
        if type(layer) is not source.MixtureDecoderLayerQwen35:
            raise TypeError("Unsupported AE decoder-layer subclass")
        original = layer.forward
        if getattr(original, "__func__", None) is not source.MixtureDecoderLayerQwen35.forward:
            raise TypeError("Install fused attention before any AE layer-forward replacement")
        if (layer.self_attn.num_heads, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim) != (8, 2, 256):
            raise ValueError("Expected released LIBERO AE attention heads 8/2 and head_dim 256")
        originals.append(original)
    metadata = {
        "enabled": True, "default_enabled": False, "experimental": True,
        "backend": backend, "scope": "24 AE full-attention kernels only; VLM unchanged",
        "operator": "torch_npu.npu_fusion_attention" if backend == "npu" else "torch.nn.functional.scaled_dot_product_attention",
        "input_layout": "BNSD", "query_shape": "[B,8,32,256]", "key_value_shape": "[B,8,K>=32,256]",
        "scale": 256 ** -0.5, "dropout": 0.0, "softcap": None, "causal": False,
        "gqa": "original upstream 2-to-8 KV repeat retained",
        "qk_norm_and_rope": "unchanged upstream FP32 paths",
        "mask": "original layer slice; only 0, dtype minimum or -inf; True means blocked for NPU and allowed for SDPA",
        "mask_validation": "original producer, once/request before FM10, device reduction and one scalar synchronization included in action timing",
        "mask_validations": 0, "rounding_order_changed": True,
        "npu_inner_precise": 0 if backend == "npu" else None,
        "global_registry_modified": False, "model_config_modified": False,
        "cached_model_inputs_or_outputs": False,
        "runtime_validation": "not performed by installer; full FM10 A/B/C/D verification required",
        "sdpa_dispatch_verified": False if backend == "sdpa" else None,
        "source_file": getattr(source, "__file__", None),
    }
    handle = FusedAttentionInstallation(metadata)
    attention = make_attention(torch, torch_npu, backend)
    namespace, cloned_forward = _make_private_forward(originals[0].__func__, attention)
    handle._private_module = namespace
    metadata["private_module"] = namespace.__name__
    metadata["private_module_registered_for_dynamo_guards"] = True

    def build_validated_mask(*args, **kwargs):
        _check_inference(policy, model, ae)
        result = original_builder(*args, **kwargs)
        validate_action_mask(torch, result[0])
        metadata["mask_validations"] += 1
        return result

    def replace(obj, name, installed):
        had_attribute = name in vars(obj)
        prior_attribute = vars(obj).get(name)
        setattr(obj, name, installed)
        handle._replacements.append((obj, name, installed, had_attribute, prior_attribute))

    try:
        replace(model, "build_action_mask_and_position_ids", build_validated_mask)
        for layer in ae.layers:
            # All exact-class layers share this forward implementation, with
            # distinct bound self values and their original module parameters.
            replace(layer, "forward", MethodType(cloned_forward, layer))
    except Exception:
        handle.restore()
        raise
    return handle
