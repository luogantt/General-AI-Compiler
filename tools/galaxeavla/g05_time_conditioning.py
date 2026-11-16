"""Opt-in fixed FM10 time conditioning; no action, observation or KV caching.

Install on the audited eager AE before compiling AE.forward. encode_time remains
outside that graph and returns a nested tuple of per-norm FP32 constants instead
of a single Tensor. MixtureQwen35 only forwards this argument to its AdaLN norms.
Those norms still normalize the current dynamic hidden states on every call.

The first encode_time call builds all ten entries with the original device,
batch, dtype, sequential Euler time updates, original encode_time and distinct
norm weights. It is cold/setup work, never silently removed from a request.
Default lookup='exact' checks time values via one small D2H transfer per call;
that synchronization remains part of measured inference. Explicit lookup='request'
uses a cursor only inside the source-pinned original FMHelper.infer. It checks the
immutable model once/request, rejects concurrent/nested/external calls, consumes
exactly ten entries and clears its state even when inference raises.

The model must remain immutable while installed. Parameter versions, identity,
storage, dtype and shape are checked, but PyTorch .data writes can evade version
tracking: restore/reinstall after any weight edit or reload, including .data.
This module does not establish GE/NPU graph support or numerical equivalence;
independent per-step and full-model replay verification is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import importlib
import inspect
import threading
import time
from typing import Any


MIXTURE_MODULE = "g05.models.g05.qwen35.mixture_qwen35"
NORM_MODULE = "g05.models.g05.model.modules"
FM_MODULE = "g05.models.g05.helpers.fm_helper"
# inspect.getsource(FMHelper.infer), normalized LF, audited upstream 89f2322.
# Its t = ones/zeros, encode_time before update, and sequential +/- 1/N are the
# reason request mode may avoid reading device t values. Unknown code is refused.
AUDITED_FM_INFER_SHA256 = "ec047027ba31dbc7b5efd70147240e9bf5a6a89d36c9b74382b2290d032571e7"
MARKER = "_g05_fixed_time_conditioning"


@dataclass
class TimeConditioningInstallation:
    metadata: dict[str, Any]
    _replacements: list = field(repr=False)
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self):
        """Idempotent; preserve any later experiment's replacement."""
        if self._restored:
            return
        for owner, name, installed, had_attribute, saved in reversed(self._replacements):
            if getattr(owner, name, None) is installed:
                if had_attribute:
                    setattr(owner, name, saved)
                else:
                    delattr(owner, name)
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def _parameter_signature(modules, torch):
    signature = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if parameter.dtype != torch.float32:
                raise ValueError(f"Time conditioning requires FP32 island parameter {prefix}.{name}")
            signature.append((prefix, name, id(parameter), parameter._version,
                              parameter.data_ptr(), tuple(parameter.shape),
                              parameter.dtype, str(parameter.device)))
    return tuple(signature)


def _precision_signature(torch):
    """Record available public matmul precision switches without setting them."""
    result = []
    getter = getattr(torch, "get_float32_matmul_precision", None)
    if getter is not None:
        result.append(("float32_matmul_precision", getter()))
    for path in ("backends.cuda.matmul.allow_tf32", "npu.matmul.allow_hf32"):
        owner = torch
        for part in path.split("."):
            owner = getattr(owner, part, None)
            if owner is None:
                break
        if owner is not None:
            result.append((path, owner))
    return tuple(result)


def install_time_conditioning(policy: Any, *, lookup: str = "exact") -> TimeConditioningInstallation:
    """Install reversible fixed-time FP32 tables; no backend is enabled by default.

    Scope is the audited 24-layer AdaLN AE with its final norm and FM10. Changes
    to batch/device/time dtype, schedule, relevant weights or training mode fail
    explicitly. Restore and reinstall to choose another contract. At inference
    use torch.no_grad()/inference_mode(), as in the official inferencer.
    """
    if lookup not in {"exact", "request"}:
        raise ValueError("lookup must be 'exact' or 'request'")
    torch = importlib.import_module("torch")
    mixture_source = importlib.import_module(MIXTURE_MODULE)
    norm_source = importlib.import_module(NORM_MODULE)
    model = policy.model
    expert, fm = model.action_expert, model.fm_helper
    original_encode = expert.encode_time
    if getattr(original_encode, MARKER, False):
        raise RuntimeError("Time conditioning is already installed")
    if type(expert) is not mixture_source.MixtureQwen35:
        raise TypeError("Expected the exact audited MixtureQwen35 class")
    if getattr(original_encode, "__func__", None) is not mixture_source.MixtureQwen35.encode_time:
        raise TypeError("encode_time was already replaced")
    if "forward" in vars(expert) or hasattr(expert, "_orig_mod"):
        raise ValueError("Install time conditioning before replacing/compiling AE.forward")
    if len(expert.layers) != 24 or getattr(expert, "adaptive_mode", None) != "adaLN":
        raise ValueError("Expected the fixed 24-layer AdaLN Action Expert")
    original_infer = fm_source = infer_function = infer_code = infer_sha = None
    if lookup == "request":
        fm_source = importlib.import_module(FM_MODULE)
        original_infer = fm.infer
        infer_function = fm_source.FMHelper.infer
        if type(fm) is not fm_source.FMHelper or getattr(original_infer, "__func__", None) is not infer_function:
            raise TypeError("Request lookup requires the original exact FMHelper.infer method")
        infer_code = infer_function.__code__
        infer_sha = hashlib.sha256(inspect.getsource(infer_function).replace("\r\n", "\n").encode("utf-8")).hexdigest()
        if infer_sha != AUDITED_FM_INFER_SHA256:
            raise ValueError("FMHelper.infer source is not the audited sequential time schedule")
    schedule = (fm.num_inference_steps, fm.time_convention)
    if type(schedule[0]) is not int or schedule[0] != 10 or schedule[1] not in {"pi_convention", "galaxea_convention"}:
        raise ValueError("Time conditioning requires ten original Euler steps")
    norms = []
    for index, layer in enumerate(expert.layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            norms.append((f"layers.{index}.{name}", getattr(layer, name)))
    norms.append(("norm", expert.norm))
    for name, norm in norms:
        if type(norm) is not norm_source.AdaptiveRMSNorm:
            raise TypeError(f"Unsupported norm class at {name}")
        if getattr(norm.forward, "__func__", None) is not norm_source.AdaptiveRMSNorm.forward:
            raise TypeError(f"Norm forward was already replaced at {name}")
        if getattr(norm._norm, "__func__", None) is not norm_source.AdaptiveRMSNorm._norm:
            raise TypeError(f"Norm math was already replaced at {name}")
    parameter_modules = [("time_embedding", expert.time_embedding),
                         ("time_mlp_in", expert.time_mlp_in), ("time_mlp_out", expert.time_mlp_out),
                         *[(name + ".dense", norm.dense) for name, norm in norms]]
    signature = _parameter_signature(parameter_modules, torch)
    precision = _precision_signature(torch)
    frequency = (expert.time_embedding.dim, expert.time_embedding.min_period, expert.time_embedding.max_period)
    epsilons = tuple(norm.eps for _, norm in norms)
    module_owners = [(expert, name, getattr(expert, name)) for name in ("time_embedding", "time_mlp_in", "time_mlp_out", "norm")]
    for layer in expert.layers:
        module_owners.extend((layer, name, getattr(layer, name)) for name in ("input_layernorm", "post_attention_layernorm"))
    module_owners.extend((norm, "dense", norm.dense) for _, norm in norms)
    layer_objects = tuple(expert.layers)
    forward_sources = [(module, getattr(module.forward, "__func__", module.forward))
                       for _, module in parameter_modules]

    def check_contract(require_no_grad=True):
        if any(getattr(module, "training", None) is not False
               for module in (policy, model, expert, *(norm for _, norm in norms),
                              *(module for _, module in parameter_modules))):
            raise ValueError("Time conditioning requires evaluation mode throughout its modules")
        if require_no_grad and torch.is_grad_enabled():
            raise ValueError("Time conditioning is inference-only; gradients must be disabled")
        if policy.model is not model or model.action_expert is not expert or model.fm_helper is not fm:
            raise RuntimeError("Policy model/Action Expert/FM helper identity changed")
        if tuple(expert.layers) != layer_objects or any(getattr(owner, name) is not module for owner, name, module in module_owners):
            raise RuntimeError("Time-conditioning module identity changed")
        if any(getattr(module.forward, "__func__", module.forward) is not source for module, source in forward_sources):
            raise RuntimeError("Time-conditioning module forward changed")
        if (fm.num_inference_steps, fm.time_convention) != schedule:
            raise RuntimeError("FM time schedule changed; restore and reinstall")
        if (expert.time_embedding.dim, expert.time_embedding.min_period, expert.time_embedding.max_period) != frequency:
            raise RuntimeError("Time embedding frequency configuration changed")
        if tuple(norm.eps for _, norm in norms) != epsilons:
            raise RuntimeError("AdaLN epsilon changed")
        if _parameter_signature(parameter_modules, torch) != signature:
            raise RuntimeError("Time-conditioning weights changed; restore and reinstall")
        if _precision_signature(torch) != precision:
            raise RuntimeError("FP32 matmul precision policy changed; restore and reinstall")

    check_contract(require_no_grad=False)
    metadata = {"enabled": True, "default_enabled": False, "experimental": True,
                "scope": "Fixed FM10 encode_time and 49 independent AdaLN conditioning projections",
                "num_steps": schedule[0], "time_convention": schedule[1], "norms": len(norms),
                "precision": "Original FP32 islands; dynamic RMS reduction, affine order and output casts retained",
                "matmul_precision_settings": dict(precision),
                "cached_actions": False, "cached_kv": False, "cached_observations": False,
                "cached_model_outputs": False, "training_supported": False, "fallback_allowed": False,
                "lookup_mode": lookup,
                "lookup": ("Exact original t values; t.detach().cpu().tolist() once each call, synchronization included in inference"
                           if lookup == "exact" else "Cursor only inside source-pinned original FMHelper.infer; one contract check/request; no per-step D2H"),
                "fm_infer_source_sha256": infer_sha,
                "request_checks": 0, "request_calls": 0, "completed_requests": 0,
                "failed_requests": 0, "last_request_consumed": None,
                "weight_mutation_contract": "Immutable model; versions/storage checked, .data writes prohibited; restore/reinstall after edits",
                "table_builds": 0, "calls": 0, "table_build_wall_ms": None, "table_bytes": 0,
                "graph_validation": "Not validated by installer; full-model A/B/A/B and C/D replay required"}
    table = {}
    ordered_table = ()
    bound_input = None
    request_lock = threading.Lock()
    request_state = {"owner": None, "cursor": 0}

    def build_table(t):
        start = time.perf_counter()
        current = (torch.ones if schedule[1] == "pi_convention" else torch.zeros)(
            t.shape[0], device=t.device, dtype=t.dtype)
        built = {}
        byte_count = 0
        times = []
        with torch.no_grad(), torch.autocast(device_type=t.device.type, enabled=False):
            for _ in range(schedule[0]):
                key = tuple(current.detach().cpu().tolist())
                if key in built:
                    raise RuntimeError("The time dtype cannot represent distinct Euler steps")
                # Sequential original shape, not one larger 10*B GEMM.
                condition = original_encode(current).detach().clone()
                if condition.dtype != torch.float32:
                    raise ValueError("Original encode_time did not produce an FP32 island")
                projected = []
                for _, norm in norms:
                    modulation = torch.nn.functional.linear(condition.float(), norm.dense.weight.float(),
                        norm.dense.bias.float() if norm.dense.bias is not None else None)
                    if modulation.ndim == 2:
                        modulation = modulation.unsqueeze(1)
                    scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
                    # Same FP32 add as upstream, evaluated once for a fixed t.
                    entry = tuple(value.detach().clone() for value in (1 + scale.float(), shift.float(), gate))
                    byte_count += sum(value.numel() * value.element_size() for value in entry)
                    projected.append(entry)
                built[key] = tuple(projected)
                times.append(list(key))
                if schedule[1] == "pi_convention":
                    current -= 1.0 / schedule[0]
                else:
                    current += 1.0 / schedule[0]
        # Account for all device work in the build's separately reported wall time.
        current.detach().cpu().tolist()
        metadata.update(table_builds=1, table_bytes=byte_count,
                        table_build_wall_ms=(time.perf_counter() - start) * 1000,
                        time_values=times, input_shape=list(t.shape), input_dtype=str(t.dtype),
                        device=str(t.device))
        return built

    @functools.wraps(original_encode)
    def encode_time(t):
        nonlocal table, ordered_table, bound_input
        if lookup == "exact":
            check_contract()
        elif request_state["owner"] != threading.get_ident():
            raise RuntimeError("Request time lookup is only allowed inside this thread's original FMHelper.infer")
        if t.ndim != 1 or t.shape[0] < 1 or t.dtype != torch.float32:
            raise ValueError("Expected original FP32 t with shape [positive batch]")
        current_input = (tuple(t.shape), t.dtype, str(t.device))
        if bound_input is not None and current_input != bound_input:
            raise RuntimeError("Time batch/dtype/device changed; restore and reinstall")
        if not table:
            table = build_table(t)
            ordered_table = tuple(table.values())
            bound_input = current_input
        if lookup == "exact":
            key = tuple(t.detach().cpu().tolist())
            if key not in table:
                raise ValueError("t is not an exact member of the original sequential FM10 schedule")
            selected = table[key]
        else:
            index = request_state["cursor"]
            if index >= schedule[0]:
                raise RuntimeError("Original FM request consumed more than ten time conditions")
            selected = ordered_table[index]
            request_state["cursor"] = index + 1
        metadata["calls"] += 1
        return selected

    replacements = []
    def replace(owner, name, function):
        replacements.append((owner, name, function, name in vars(owner), vars(owner).get(name)))
        setattr(owner, name, function)

    for index, (_, norm) in enumerate(norms):
        def make_forward(module, norm_index):
            @functools.wraps(module.forward)
            def forward(x, cond):
                if module.training:
                    raise ValueError("Time conditioning requires evaluation mode")
                if not isinstance(cond, tuple) or len(cond) != len(norms):
                    raise TypeError("Expected the installed per-norm time tuple from encode_time")
                scale_plus_one, shift, gate = cond[norm_index]
                with torch.autocast(device_type=x.device.type, enabled=False):
                    dynamic_norm = module._norm(x)
                    result = dynamic_norm * scale_plus_one + shift
                    return result.to(x.dtype), gate.to(x.dtype)
            return forward
        replace(norm, "forward", make_forward(norm, index))
    setattr(encode_time, MARKER, True)
    replace(expert, "encode_time", encode_time)
    if lookup == "request":
        @functools.wraps(original_infer)
        def infer_request(*args, **kwargs):
            if not request_lock.acquire(blocking=False):
                raise RuntimeError("Concurrent or nested FM time-conditioning requests are prohibited")
            completed = False
            try:
                check_contract()
                if fm_source.FMHelper.infer is not infer_function or infer_function.__code__ is not infer_code:
                    raise RuntimeError("Audited FMHelper.infer implementation changed")
                if (args[0] if args else kwargs.get("model")) is not model:
                    raise RuntimeError("Request lookup received a different model")
                metadata["request_checks"] += 1
                metadata["request_calls"] += 1
                request_state.update(owner=threading.get_ident(), cursor=0)
                result = original_infer(*args, **kwargs)
                completed = True
                return result
            finally:
                consumed = request_state["cursor"]
                request_state.update(owner=None, cursor=0)
                request_lock.release()
                metadata["last_request_consumed"] = consumed
                valid = completed and consumed == schedule[0]
                metadata["completed_requests" if valid else "failed_requests"] += 1
                if completed and not valid:
                    raise RuntimeError(f"Original FM request consumed {consumed} time conditions; expected ten")
        replace(fm, "infer", infer_request)
    return TimeConditioningInstallation(metadata, replacements)
