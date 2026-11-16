#!/usr/bin/env python3
"""Observe selected arithmetic in one official G0.5 request, without timing it.

This is a shape-based count of called Linear/Conv modules plus the pinned
AdaptiveRMSNorm's functional conditioning projection. It is NOT a complete
model FLOP count or a hardware instruction/counter measurement. The inference
inputs, weights, return values and execution switches are left unchanged.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import math
from pathlib import Path
import sys
from typing import Any
import json

import benchmark_g05_full as benchmark


def shape(value: Any) -> list[int] | None:
    return [int(n) for n in value.shape] if hasattr(value, "shape") else None


def linear_flops(input_shape: list[int], in_features: int, out_features: int) -> int:
    if not input_shape or input_shape[-1] != in_features:
        raise ValueError("Linear input shape does not match in_features")
    return 2 * math.prod(input_shape[:-1]) * in_features * out_features


def convolution_flops(output_shape: list[int], weight_shape: list[int]) -> int:
    # weight [Cout, Cin/groups, spatial kernel...]; bias is deliberately excluded.
    return 2 * math.prod(output_shape) * math.prod(weight_shape[1:])


def component_for(name: str) -> str:
    if "vision_tower" in name:
        return "vision"
    if "action_expert" in name:
        return "action_expert"
    if "proprio_embedder" in name:
        return "proprio"
    if name.startswith("codec") or "action_tokenizer" in name:
        return "codec"
    if ".vlm" in name:
        return "vlm"
    return "other"


class WorkloadCounter:
    def __init__(self, torch_module: Any):
        self.torch = torch_module
        self.stage = "unscoped"
        self.handles: list[Any] = []
        self.restore: list[tuple[Any, str, Any, bool]] = []
        self.records: dict[tuple, dict] = {}
        self.model_calls: list[dict] = []
        self.prefill: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {}
        self.roots: list[dict[str, str]] = []

    @contextlib.contextmanager
    def phase(self, name: str):
        previous = self.stage
        self.stage = name
        try:
            yield
        finally:
            self.stage = previous

    def wrap(self, obj: Any, attr: str, stage: str, after=None):
        original = getattr(obj, attr)
        had_override = attr in vars(obj)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            current = self.stage if stage == "action_ar" and self.stage == "cot" else stage
            with self.phase(current):
                result = original(*args, **kwargs)
                if after is not None:
                    after(result)
                return result

        setattr(obj, attr, wrapped)
        self.restore.append((obj, attr, original, had_override))

    def add(self, name: str, operator: str, inputs: Any, output: Any, flops: int,
            output_shape_override=None, dtype_override=None):
        input_shape = shape(inputs)
        output_shape = output_shape_override if output_shape_override is not None else shape(output)
        key = (self.stage, name, operator, tuple(input_shape or []), tuple(output_shape or []))
        if key not in self.records:
            self.records[key] = {
                "stage": self.stage,
                "component": component_for(name),
                "module": name,
                "operator": operator,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "input_dtype": dtype_override or str(getattr(inputs, "dtype", "unknown")),
                "output_dtype": dtype_override or str(getattr(output, "dtype", "unknown")),
                "calls": 0,
                "flops_per_call": flops,
                "counted_flops": 0,
            }
        record = self.records[key]
        if record["flops_per_call"] != flops:
            raise RuntimeError("Same recorded operator shape produced inconsistent arithmetic")
        record["calls"] += 1
        record["counted_flops"] += flops

    def arithmetic_hook(self, name: str):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs.get("input", kwargs.get("x"))
            if isinstance(module, self.torch.nn.Linear):
                flops = linear_flops(shape(x), module.in_features, module.out_features)
                self.add(name, "Linear", x, output, flops)
            elif isinstance(module, (self.torch.nn.Conv1d, self.torch.nn.Conv2d, self.torch.nn.Conv3d)):
                flops = convolution_flops(shape(output), shape(module.weight))
                self.add(name, type(module).__name__, x, output, flops)
            elif type(module).__name__ == "AdaptiveRMSNorm":
                # Pinned upstream uses F.linear(cond, dense.weight, dense.bias):
                # dense.forward is bypassed, so its Linear hook does not fire.
                cond = args[1] if len(args) > 1 else kwargs["cond"]
                dense = module.dense
                flops = linear_flops(shape(cond), dense.in_features, dense.out_features)
                # Derive the functional projection shape without allocating a
                # tensor. The projection precedes broadcast across the horizon.
                self.add(name + ".dense[functional]", "AdaLN conditioning Linear", cond, output[0], flops,
                         output_shape_override=[*shape(cond)[:-1], dense.out_features],
                         dtype_override="torch.float32")
        return hook

    def model_hook(self, component: str):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("inputs_embeds", kwargs.get("hidden_states"))
            mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            self.model_calls.append({
                "stage": self.stage,
                "component": component,
                "input_shape": shape(x),
                "attention_mask_shape": shape(mask),
            })
        return hook

    def attach(self, policy: Any):
        nn = self.torch.nn
        roots = [("policy", policy)]
        # VQActionTokenizer is not nn.Module, so its codec is absent from
        # policy.named_modules()/named_parameters(). Include the module root.
        codec_holder = getattr(policy, "action_tokenizer", None)
        codec = getattr(codec_holder, "action_tokenizer", None)
        if isinstance(codec, nn.Module):
            roots.append(("codec", codec))
        else:
            raise RuntimeError("Expected the pinned policy ActionCodec module root")

        seen_modules, seen_parameters = set(), set()
        groups: dict[str, Any] = {}
        for prefix, root in roots:
            self.roots.append({"name": prefix, "type": type(root).__name__})
            for relative_name, parameter in root.named_parameters():
                if id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                group = component_for(prefix + "." + relative_name)
                item = groups.setdefault(group, {"numel": 0, "bytes": 0, "dtypes": {}, "devices": {}})
                numel = int(parameter.numel())
                nbytes = numel * int(parameter.element_size())
                item["numel"] += numel
                item["bytes"] += nbytes
                dtype, device = str(parameter.dtype), str(parameter.device)
                item["dtypes"][dtype] = item["dtypes"].get(dtype, 0) + numel
                item["devices"][device] = item["devices"].get(device, 0) + numel
            for relative_name, module in root.named_modules():
                if id(module) in seen_modules:
                    continue
                seen_modules.add(id(module))
                name = prefix + ("." + relative_name if relative_name else "")
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) or type(module).__name__ == "AdaptiveRMSNorm":
                    self.handles.append(module.register_forward_hook(self.arithmetic_hook(name), with_kwargs=True))

        self.parameters = {
            "total_numel": sum(group["numel"] for group in groups.values()),
            "total_bytes": sum(group["bytes"] for group in groups.values()),
            "groups": groups,
            "deduplication": "Parameter object identity; shared embedding/LM-head weight counted once",
            "scope": "Parameters only; excludes buffers, caches, activations and allocator/runtime overhead",
        }

        def record_prefill(state):
            self.prefill = {
                "input_ids_shape": shape(state.input_ids),
                "prefix_tokens": int(state.input_ids.shape[-1]),
                "attention_mask_shape": shape(state.attention_mask),
                "last_hidden_shape": shape(state.last_hidden),
            }

        self.wrap(policy, "prefill", "prefill", after=record_prefill)
        self.wrap(policy, "generate_text", "cot")
        self.wrap(policy.model, "inference_fm", "fm")
        self.wrap(policy.model, "inference_ar", "action_ar")
        self.wrap(policy.processor, "decode_ar", "codec")
        for attr, label in (("vlm", "vlm"), ("action_expert", "action_expert")):
            module = getattr(policy.model, attr)
            self.handles.append(module.register_forward_pre_hook(self.model_hook(label), with_kwargs=True))

    def close(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        for obj, attr, original, had_override in reversed(self.restore):
            if had_override:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)
        self.restore.clear()

    def summary(self):
        records = list(self.records.values())
        stages: dict[str, Any] = {}
        for record in records:
            stage = stages.setdefault(record["stage"], {"counted_flops": 0, "calls": 0, "components": {}})
            component = stage["components"].setdefault(record["component"], {"counted_flops": 0, "calls": 0})
            for target in (stage, component):
                target["counted_flops"] += record["counted_flops"]
                target["calls"] += record["calls"]
        return {"counted_flops": sum(record["counted_flops"] for record in records), "stages": stages, "operators": records}


def main(argv=None) -> int:
    parser = benchmark.build_parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    args.smoke_only = True
    args.warmup = 0
    args.iterations = 0
    if args.action_steps != 16:
        parser.error("the pinned R1 Lite service contract uses --action-steps 16")
    source = args.source_dir.expanduser().resolve()
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source))
    if args.device.split(":", 1)[0] == "npu":
        importlib.import_module("torch_npu")
    import torch
    serve = importlib.import_module("scripts.serve_policy")
    original_setup = serve.setup
    counter = WorkloadCounter(torch)

    @functools.wraps(original_setup)
    def setup_and_instrument(*setup_args, **setup_kwargs):
        policy, processor = original_setup(*setup_args, **setup_kwargs)
        counter.attach(policy)
        return policy, processor

    serve.setup = setup_and_instrument
    try:
        measured = benchmark.run_benchmark(args)
    finally:
        serve.setup = original_setup
        counter.close()

    sample = measured["cold_request"]["timing_ms"]
    report = {
        "schema_version": 1,
        "result_scope": "instrumentation-only counted operator workload",
        "arithmetic_convention": "One multiplication and one accumulation count as 2 FLOPs; dense Linear/Conv and AdaLN conditioning projections only",
        "not_counted": [
            "attention QK/AV matmuls and SDPA internals",
            "GatedDeltaNet chunk/recurrent rule functional tensor operations",
            "functional convolutions bypassing Conv module forward (including GDN decode)",
            "ConvTranspose operations, including ActionCodec decoder upsampling",
            "bias adds, activations, normalization, elementwise operations, reductions, RoPE and Euler updates",
            "embedding/codebook lookups, data movement, preprocessing and backend padding/recomputation",
        ],
        "timing_scope": "Diagnostic hooks perturb execution; no diagnostic latency is published. Use the separate uninstrumented benchmark results.",
        "source": measured["source"],
        "checkpoint": measured["checkpoint"],
        "input": measured["input"],
        "configuration_contract": measured["configuration_contract"],
        "configuration_compatibility": measured["configuration_compatibility"],
        "environment": measured["environment"],
        "instrumentation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest(),
        "roots": counter.roots,
        "parameters": counter.parameters,
        "prefill": counter.prefill,
        "model_calls": counter.model_calls,
        "generated_tokens": {key: sample[key] for key in ("cot_raw_generated_tokens", "action_ar_raw_generated_tokens", "ar_decoded_action_tokens")},
        "output_validation": sample["output_validation"],
        **counter.summary(),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote counted operator workload: {output}")
    print(f"Selected arithmetic: {report['counted_flops'] / 1e12:.6f} TFLOP/request (incomplete count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
