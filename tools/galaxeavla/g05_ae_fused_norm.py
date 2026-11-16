"""Default-off FP32 NPU RMSNorm experiment for the released LIBERO AE.

Replace only 49 AdaptiveRMSNorm._norm methods and 48 Q/K RMSNorm forwards.
AdaLN's FP32 modulation/gate code is untouched; this composes with the optional
time-conditioning helper when installed AFTER it. Q/K gamma is formed from the
current weight in FP32 on every call, then output is cast to the input dtype.

No model parameters, class methods, VLM modules or operator registries change.
Fused reduction order may differ from the original FP32 square/mean/rsqrt path;
verify=True acknowledges the required same-device and whole-FM10 comparisons,
not a claim that the numerical comparisons have already passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Any


MIXTURE_MODULE = "g05.models.g05.qwen35.mixture_qwen35"
ADAPTIVE_MODULE = "g05.models.g05.model.modules"
QK_MODULE = "g05.models.g05.qwen35.modules"
_MARKER = "_g05_ae_fused_norm"


def _check_input(torch, module, x, *, width, ndim, heads=None):
    if module.training or torch.is_grad_enabled():
        raise ValueError("AE fused normalization requires evaluation with gradients disabled")
    if x.device.type != "npu":
        raise ValueError("AE fused normalization is NPU-only; no CPU/CUDA fallback")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("AE fused normalization accepts only FP32/BF16/FP16 inputs")
    if x.ndim != ndim or x.shape[-1] != width or x.shape[0] < 1 or x.shape[1] != 32:
        raise ValueError("Unexpected LIBERO AE normalization shape")
    if heads is not None and x.shape[-2] != heads:
        raise ValueError("Unexpected AE Q/K head count")


def make_adaptive_norm(torch, torch_npu, module, unit_gamma):
    """Tensor-only FP32 _norm replacement; unit gamma is input-independent."""
    def adaptive_norm(x):
        _check_input(torch, module, x, width=1024, ndim=3)
        if unit_gamma.device != x.device or unit_gamma.dtype != torch.float32:
            raise ValueError("AE unit gamma must remain FP32 on the input NPU")
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = torch_npu.npu_rms_norm(x.float(), unit_gamma, epsilon=module.eps)[0]
        if output.dtype != torch.float32:
            raise TypeError("NPU RMSNorm changed AdaptiveRMSNorm's FP32 output contract")
        # Original _norm promotes x * rsqrt(FP32 variance) to FP32. Casting here
        # would lose precision before the unchanged FP32 AdaLN scale/shift.
        return output
    setattr(adaptive_norm, _MARKER, True)
    return adaptive_norm


def make_qk_norm(torch, torch_npu, module, *, heads):
    """Tensor-only Q/K forward preserving FP32 (1 + weight) and final cast."""
    def qk_norm(x):
        _check_input(torch, module, x, width=256, ndim=4, heads=heads)
        if module.weight.device != x.device or tuple(module.weight.shape) != (256,):
            raise ValueError("AE Q/K RMSNorm weight device/shape changed")
        if module.weight.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Unsupported AE Q/K RMSNorm weight dtype")
        with torch.autocast(device_type=x.device.type, enabled=False):
            gamma = 1.0 + module.weight.float()
            output = torch_npu.npu_rms_norm(x.float(), gamma, epsilon=module.eps)[0]
            if output.dtype != torch.float32:
                raise TypeError("NPU RMSNorm Q/K intermediates must remain FP32")
            return output.to(dtype=x.dtype)
    setattr(qk_norm, _MARKER, True)
    return qk_norm


@dataclass
class FusedNormInstallation:
    metadata: dict[str, Any]
    _replacements: list = field(default_factory=list, repr=False)
    _constants: list = field(default_factory=list, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self):
        """Restore the compiled AE wrapper first; preserve later replacements."""
        if self._restored:
            return
        for owner, name, installed, had_attribute, previous in reversed(self._replacements):
            if getattr(owner, name, None) is installed:
                if had_attribute:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
        self._constants.clear()
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_ae_fused_norm(policy, *, verify=False):
    """Install on a loaded eval AE, AFTER time conditioning and BEFORE compile.

    This is an explicit experiment. verify=True is mandatory; the caller must
    perform and enforce actual eager-vs-candidate numerical validation.
    """
    if verify is not True:
        raise ValueError("AE fused normalization requires explicit verify=True")
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    mixture_source = importlib.import_module(MIXTURE_MODULE)
    adaptive_source = importlib.import_module(ADAPTIVE_MODULE)
    qk_source = importlib.import_module(QK_MODULE)
    if not callable(getattr(torch_npu, "npu_rms_norm", None)):
        raise RuntimeError("Installed TorchNPU does not expose npu_rms_norm")
    model, ae = policy.model, policy.model.action_expert
    if any(getattr(obj, "training", None) is not False for obj in (policy, model, ae)):
        raise ValueError("AE fused normalization requires an evaluation policy/model/AE")
    if type(ae) is not mixture_source.MixtureQwen35:
        raise TypeError("AE fused normalization requires exact upstream MixtureQwen35")
    if "forward" in vars(ae) or hasattr(ae, "_orig_mod"):
        raise TypeError("Install fused normalization before replacing or compiling AE.forward")
    if len(ae.layers) != 24 or list(ae.layer_types) != ["full_attention"] * 24:
        raise ValueError("Expected the released LIBERO AE's 24 full-attention layers")
    adaptive, qk = [], []
    for index, layer in enumerate(ae.layers):
        if type(layer) is not mixture_source.MixtureDecoderLayerQwen35 or layer.training:
            raise TypeError("Unsupported AE decoder-layer class or training mode")
        for name in ("input_layernorm", "post_attention_layernorm"):
            adaptive.append((f"layers.{index}.{name}", getattr(layer, name)))
        for name, heads in (("q_norm", 8), ("k_norm", 2)):
            qk.append((f"layers.{index}.self_attn.{name}", getattr(layer.self_attn, name), heads))
    adaptive.append(("norm", ae.norm))
    all_modules = [(name, module) for name, module in adaptive] + [(name, module) for name, module, _ in qk]
    if len({id(module) for _, module in all_modules}) != 97:
        raise ValueError("Expected 97 distinct AE normalization modules")
    devices = set()
    for name, module in adaptive:
        if type(module) is not adaptive_source.AdaptiveRMSNorm or module.training or module.dim != 1024:
            raise TypeError(f"Unsupported AdaptiveRMSNorm contract at {name}")
        if getattr(module._norm, "__func__", None) is not adaptive_source.AdaptiveRMSNorm._norm:
            raise TypeError(f"Adaptive _norm already replaced at {name}")
        devices.add(str(module.dense.weight.device))
        if module.dense.weight.device.type != "npu":
            raise ValueError("Load AE on NPU before installing fused normalization")
    for name, module, _ in qk:
        if type(module) is not qk_source.Qwen3_5RMSNorm or module.training or tuple(module.weight.shape) != (256,):
            raise TypeError(f"Unsupported Q/K RMSNorm contract at {name}")
        if (getattr(module.forward, "__func__", None) is not qk_source.Qwen3_5RMSNorm.forward
                or getattr(module._norm, "__func__", None) is not qk_source.Qwen3_5RMSNorm._norm):
            raise TypeError(f"Q/K normalization already replaced at {name}")
        devices.add(str(module.weight.device))
        if module.weight.device.type != "npu":
            raise ValueError("Load all AE Q/K weights on NPU before installation")
    for name, module in all_modules:
        if isinstance(module.eps, bool) or not isinstance(module.eps, (int, float)) or not math.isfinite(module.eps) or module.eps <= 0:
            raise ValueError(f"Expected a finite positive epsilon at {name}")
    if len(devices) != 1:
        raise ValueError("Fused normalization requires the entire AE on one NPU")
    metadata = {
        "enabled": True, "default_enabled": False, "experimental": True,
        "name": "npu_rms_norm_fp32", "operator": "torch_npu.npu_rms_norm",
        "scope": "AE only: 49 AdaptiveRMSNorm._norm + 48 Q/K Qwen3_5RMSNorm.forward",
        "adaptive_modules": 49, "qk_modules": 48,
        "math_dtype": "torch.float32", "adaptive_output_dtype": "torch.float32",
        "qk_output_dtype": "input dtype", "autocast": "disabled around all fused normalization math",
        "adaptive_gamma": "input-independent FP32 ones; no early cast before AdaLN modulation",
        "qk_gamma": "1.0 + current module.weight.float(), recomputed each call",
        "time_conditioning_compatibility": "install after time helper; its Adaptive forward remains untouched",
        "unit_gamma_numel": 1024, "unit_gamma_bytes": 4096,
        "device": next(iter(devices)), "epsilons": sorted({float(module.eps) for _, module in all_modules}),
        "rounding_order_changed": True, "verify_required": True, "fallback_allowed": False,
        "class_methods_modified": False, "global_registry_modified": False,
        "cached_model_inputs_or_outputs": False,
        "runtime_validation": "not performed by installer; eager and full-FM10 comparisons required",
    }
    unit_gamma = torch.ones(1024, device=adaptive[0][1].dense.weight.device, dtype=torch.float32)
    handle = FusedNormInstallation(metadata, _constants=[unit_gamma])

    def replace(owner, name, installed):
        had_attribute, previous = name in vars(owner), vars(owner).get(name)
        setattr(owner, name, installed)
        handle._replacements.append((owner, name, installed, had_attribute, previous))

    try:
        for _, module in adaptive:
            replace(module, "_norm", make_adaptive_norm(torch, torch_npu, module, unit_gamma))
        for _, module, heads in qk:
            replace(module, "forward", make_qk_norm(torch, torch_npu, module, heads=heads))
    except Exception:
        handle.restore()
        raise
    return handle
