"""Opt-in, instance-local MRoPE selection without in-place slice scatter.

Only the first full-attention rotary module in each of VLM and AE is patched.
The audited upstream forward owns a fresh local ``freqs`` and only consumes the
return value, so replacing its view mutation with an out-of-place selection has
no observable alias side effect. Current frequencies are always consumed; only
two fixed Boolean index masks are retained. Install before graph compilation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap
from typing import Any


SOURCE_MODULE = "g05.models.g05.qwen35.modules"
MARKER = "_g05_functional_mrope"
SECTION = (11, 11, 10)
AUDITED_FORWARD_SOURCE_SHA256 = "f9f027303e413ebb5ed673d334b47835f31ed1990eed61a9a452471cb188477e"
AUDITED_INTERLEAVE_SOURCE_SHA256 = "c7ba6085984cbc15273134a6aad378dea4ac5bdca9fe93479dab359720fa65d5"
_MISSING = object()


def _source_sha256(function):
    # Python 3.12 adds FunctionDef.type_params to ast.dump; other AST/default
    # fields can also change across versions. Pin the actual complete source,
    # including decorators, docstrings and comments, instead of its AST repr.
    # Normalize only transport newlines, outer class indentation and trailing
    # newlines. Internal code and whitespace remain part of the audited hash.
    source = inspect.getsource(function).replace("\r\n", "\n")
    canonical = textwrap.dedent(source).rstrip("\n") + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_section(section):
    if (not isinstance(section, Sequence) or isinstance(section, (str, bytes, bytearray))
            or len(section) != 3
            or any(type(value) is not int for value in section) or tuple(section) != SECTION):
        raise ValueError("Functional MRoPE requires mrope_section=[11,11,10]")


def make_functional_interleave(torch, device):
    """Create two static geometry masks; no model/request data is retained."""
    if device.type != "npu":
        raise ValueError("Functional MRoPE supports NPU only")
    height = torch.tensor([i in range(1, 33, 3) for i in range(32)],
                          dtype=torch.bool, device=device)
    width = torch.tensor([i in range(2, 30, 3) for i in range(32)],
                         dtype=torch.bool, device=device)

    def interleave(freqs, mrope_section):
        _validate_section(mrope_section)
        if torch.is_grad_enabled():
            raise ValueError("Functional MRoPE is inference-only")
        if (freqs.ndim != 4 or freqs.shape[0] != 3 or freqs.shape[-1] != 32
                or freqs.shape[1] < 1 or freqs.shape[2] < 1):
            raise ValueError("Functional MRoPE requires freqs[3,B>0,S>0,32]")
        if freqs.dtype != torch.float32:
            raise TypeError("Functional MRoPE requires original FP32 frequencies")
        if freqs.device != device:
            raise ValueError("Functional MRoPE frequency device changed; reinstall")
        selected = torch.where(height, freqs[1], freqs[0])
        return torch.where(width, freqs[2], selected)

    setattr(interleave, MARKER, True)
    return interleave


@dataclass
class FunctionalMRopeInstallation:
    metadata: dict[str, Any]
    _replacements: list = field(default_factory=list, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self):
        """Restore graph wrappers first; preserve later external replacements."""
        if self._restored:
            return
        for rotary, name, installed, previous in reversed(self._replacements):
            if vars(rotary).get(name, _MISSING) is installed:
                if previous is _MISSING:
                    delattr(rotary, name)
                else:
                    setattr(rotary, name, previous)
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_functional_mrope(policy, *, verify=False):
    """Patch only the two actually called rotary instances; default is off.

    Full-model independent verification is mandatory. No class/global registry
    is mutated. Restore this handle after restoring AE/VLM graph wrappers.
    """
    if verify is not True:
        raise ValueError("Functional MRoPE requires verify=True and whole-model verification")
    torch = importlib.import_module("torch")
    source = importlib.import_module(SOURCE_MODULE)
    cls = source.Qwen3_5TextRotaryEmbedding
    if (_source_sha256(cls.forward) != AUDITED_FORWARD_SOURCE_SHA256
            or _source_sha256(cls.apply_interleaved_mrope) != AUDITED_INTERLEAVE_SOURCE_SHA256):
        raise ValueError("MRoPE forward/interleave source differs from the audited local-frequency contract")
    if policy.training or policy.model.training:
        raise ValueError("Functional MRoPE requires evaluation mode")
    selected = []
    for name in ("vlm", "action_expert"):
        mixture = getattr(policy.model, name)
        if mixture.training or "forward" in vars(mixture) or hasattr(mixture, "_orig_mod"):
            raise ValueError("Install functional MRoPE on eager eval mixtures before compilation")
        if len(mixture.layers) != len(mixture.layer_types):
            raise ValueError("Mixture layer_types and layers do not align")
        index = next((i for i, kind in enumerate(mixture.layer_types) if kind == "full_attention"), None)
        if index is None:
            raise ValueError("Mixture has no full-attention rotary module")
        rotary = mixture.layers[index].self_attn.rotary_emb
        if type(rotary) is not cls or rotary.training:
            raise TypeError("Expected the original eval Qwen3_5TextRotaryEmbedding exact class")
        if (getattr(rotary.forward, "__func__", None) is not cls.forward
                or "forward" in vars(rotary)):
            raise TypeError("MRoPE forward instance replacement is unsupported")
        if ("apply_interleaved_mrope" in vars(rotary)
                or rotary.apply_interleaved_mrope is not cls.apply_interleaved_mrope):
            raise RuntimeError("MRoPE interleave already has a replacement")
        _validate_section(rotary.mrope_section)
        if rotary.inv_freq.device.type != "npu" or tuple(rotary.inv_freq.shape) != (32,):
            raise ValueError("Expected NPU rotary inv_freq[32]")
        selected.append((name, index, rotary))
    if selected[0][2] is selected[1][2]:
        raise ValueError("AE and VLM must own distinct rotary modules")

    metadata = {
        "enabled": True, "default_enabled": False, "experimental": True,
        "scope": "AE and VLM first full-attention rotary_emb instances only",
        "operator": "two torch.where selections; no in-place slice writes",
        "freqs_shape": "[3,B>0,S>0,32]", "math_dtype": "torch.float32",
        "mrope_section": list(SECTION), "cached": "two static Boolean index masks per selected rotary instance",
        "section_normalization": "selected instance attribute only: validated Sequence to tuple; original config object untouched and restored by identity",
        "height_indices": list(range(1, 33, 3)), "width_indices": list(range(2, 30, 3)),
        "cached_model_inputs_or_outputs": False, "cached_positions_or_frequencies": False,
        "selection_has_no_floating_point_arithmetic": True,
        "alias_contract": "audited forward creates local freqs and consumes only the returned tensor",
        "class_or_global_registry_modified": False, "fallback_allowed": False,
        "source_hash_canonicalization": "inspect.getsource; CRLF to LF; dedent outer class; one trailing LF; full body/decorators/comments retained",
        "forward_source_sha256": AUDITED_FORWARD_SOURCE_SHA256,
        "interleave_source_sha256": AUDITED_INTERLEAVE_SOURCE_SHA256,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "targets": [{"mixture": name, "first_full_attention_layer": index} for name, index, _ in selected],
        "runtime_validation": "not performed by installer; full-model A/B/C/D/E/F replay required",
    }
    handle = FunctionalMRopeInstallation(metadata)
    try:
        for _, _, rotary in selected:
            installed = make_functional_interleave(torch, rotary.inv_freq.device)
            previous_section = vars(rotary).get("mrope_section", _MISSING)
            handle._replacements.append((rotary, "mrope_section", SECTION, previous_section))
            # OmegaConf ListConfig implements MutableSequence. The canonical
            # instance tuple avoids bringing its dynamic container machinery
            # into Dynamo. Do not mutate the original config/container object.
            rotary.mrope_section = SECTION
            previous = vars(rotary).get("apply_interleaved_mrope", _MISSING)
            handle._replacements.append((rotary, "apply_interleaved_mrope", installed, previous))
            # Instance function assignment deliberately retains the staticmethod
            # call signature (freqs, mrope_section), with no implicit self.
            rotary.apply_interleaved_mrope = installed
    except Exception:
        handle.restore()
        raise
    return handle
