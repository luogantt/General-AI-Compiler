"""Opt-in removal of unconsumed ActionCodec work in FM-only context encoding.

This does not cache inputs or inference outputs. It removes only a private copy's
action field after checking the retained template has no consumer for that field.
The original encoder still processes every request's task, image metadata and
proprio. Actual vision and proprio model execution are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import importlib
from typing import Any, Callable


SOURCE_MODULE = "g05.models.g05.io.input_preprocessor"
_MARKER = "_g05_context_only_optimization"
# Exact classes, not subclasses: custom processors can read sample_dict.action.
# MLPDropout is intentionally excluded: upstream draws randomness even in eval.
_SAFE_PROCESSOR_NAMES = (
    "BuiltinTextProcessor", "BuiltinImageProcessor", "BuiltinQwen35ImageProcessor",
    "BuiltinProprioMLPProcessor", "BuiltinProprioNullProcessor",
)


def _check_policy(policy: Any, processor: Any) -> None:
    required = {"training": False, "continuous_action": True,
                "discrete_action": False, "predict_cot": False}
    for name, expected in required.items():
        if getattr(policy, name, None) is not expected:
            raise ValueError(f"Context-only optimization requires policy.{name}={expected}")
    if getattr(policy, "processor", None) is not processor:
        raise RuntimeError("Policy processor changed after context-only optimization installation")


@dataclass
class ContextOnlyInstallation:
    metadata: dict[str, Any]
    _processor: Any = field(repr=False)
    _original: Callable = field(repr=False)
    _installed: Callable = field(repr=False)
    _had_instance_attribute: bool = field(repr=False)
    _instance_attribute: Any = field(repr=False)
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self) -> None:
        """Idempotent; do not overwrite another later experiment's replacement."""
        if self._restored:
            return
        if getattr(self._processor, "encode_inference", None) is self._installed:
            if self._had_instance_attribute:
                self._processor.encode_inference = self._instance_attribute
            else:
                delattr(self._processor, "encode_inference")
        self._restored = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


def install_context_only_optimization(policy: Any) -> ContextOnlyInstallation:
    """Wrap policy.processor.encode_inference; default callers remain unchanged.

    Only mode='ar' (the upstream prefill's context_only selection), evaluation,
    continuous-action-only inference without CoT is accepted. Unsafe templates or
    processors raise instead of silently bypassing work. Install before warmup;
    restore in finally. This does not alter source, site-packages, or model weights.
    """
    processor = getattr(policy, "processor", None)
    _check_policy(policy, processor)
    source = importlib.import_module(SOURCE_MODULE)
    if type(processor) is not source.InputPreprocessor:
        raise TypeError("Context-only optimization requires the upstream InputPreprocessor exact class")
    original = processor.encode_inference
    if getattr(original, _MARKER, False):
        raise RuntimeError("Context-only optimization is already installed")
    if getattr(original, "__func__", None) is not source.InputPreprocessor.encode_inference:
        raise TypeError("Unsupported replacement of InputPreprocessor.encode_inference")
    # The safety decision must use the same audited parser as original preprocess.
    parser_methods = {name: getattr(source.InputPreprocessor, name) for name in (
        "_prepare_segments", "_parse_template", "_parse_control_tokens", "_get_processor",
        "preprocess", "_preprocess_single_sample",
    )}
    safe_classes = {getattr(source, name) for name in _SAFE_PROCESSOR_NAMES}
    safe_methods = {cls: cls.process for cls in safe_classes}
    metadata = {
        "enabled": True, "default_enabled": False,
        "scope": "FM-only InputPreprocessor.encode_inference(mode='ar') unused action field",
        "source_module": SOURCE_MODULE, "source_file": getattr(source, "__file__", None),
        "optimization": "omit ActionCodec encoding only when retained context has no action consumer",
        "cached_inputs": False, "cached_model_outputs": False,
        "images_state_task_recomputed": True,
        "safe_processor_classes": list(_SAFE_PROCESSOR_NAMES),
        "unknown_processors_rejected": True, "fallback_allowed": False,
        "calls": 0, "samples_with_action_removed": 0, "samples_without_action": 0,
        "omitted_side_effect": "unused ActionCodec _last_encode_state update; AR/CoT paths prohibited",
    }

    @functools.wraps(original)
    def encode_context(samples, device, mode="fm", training=False, **kwargs):
        _check_policy(policy, processor)
        if mode != "ar" or training is not False:
            raise ValueError("Context-only optimization requires mode='ar' and training=False")
        if "control_flag" in kwargs:
            raise ValueError("encode_inference controls context_only internally")
        if not isinstance(samples, list) or not samples or not all(isinstance(s, dict) for s in samples):
            raise TypeError("Expected a nonempty list of sample dictionaries")
        for name, expected in parser_methods.items():
            if getattr(getattr(processor, name), "__func__", None) is not expected:
                raise RuntimeError(f"Unsupported replacement of InputPreprocessor.{name}")
        private_samples = [dict(sample) for sample in samples]
        # _prepare_segments performs token resolution and EOC slicing, including
        # per-sample template selection. Re-evaluate on every request.
        segments = processor._prepare_segments(
            private_samples, kwargs.get("template"), control_flag="context_only",
        )
        if len(segments) != len(private_samples):
            raise RuntimeError("Template parser returned a different number of samples")
        for sample_segments in segments:
            for segment in sample_segments:
                if getattr(segment, "sample_key", None) == "action" or getattr(segment, "processor_key", None) == "action":
                    raise ValueError("Retained context template references action; ActionCodec cannot be omitted")
                if segment.type == "static":
                    continue
                if segment.type != "dynamic":
                    raise ValueError(f"Unexpected retained template segment type: {segment.type}")
                modality = processor._get_processor(segment.processor_key)
                cls = type(modality)
                if cls not in safe_classes:
                    raise TypeError(f"Unsafe context modality processor: {cls.__module__}.{cls.__name__}")
                if getattr(getattr(modality, "process", None), "__func__", None) is not safe_methods[cls]:
                    raise TypeError(f"Replaced process method on {cls.__name__} is not supported")
        removed = 0
        for sample in private_samples:
            if "action" in sample:
                del sample["action"]
                removed += 1
        # The original method owns deepcopy/tokenization/padding/H2D exactly as
        # before. It receives current dynamic fields, never a cached token batch.
        result = original(private_samples, device=device, mode=mode, training=training, **kwargs)
        metadata["calls"] += 1
        metadata["samples_with_action_removed"] += removed
        metadata["samples_without_action"] += len(private_samples) - removed
        return result

    setattr(encode_context, _MARKER, True)
    had_attribute = "encode_inference" in vars(processor)
    instance_attribute = vars(processor).get("encode_inference")
    installation = ContextOnlyInstallation(
        metadata, processor, original, encode_context, had_attribute, instance_attribute,
    )
    processor.encode_inference = encode_context
    return installation
