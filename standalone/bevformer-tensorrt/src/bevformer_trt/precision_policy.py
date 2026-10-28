"""Auditable TensorRT constraints for INT8 backbone convolutions and FP16 heads.

Apply once, after ONNX parsing, to a weakly typed TensorRT 10 network. Calibration
and its data provenance belong to the caller. This module supplies no invented
dynamic ranges and performs no inference. Requested constraints are not evidence
of an engine's actual precision, numerical accuracy, or performance.
"""

from collections import Counter
import hashlib
import json
import re


POLICY_VERSION = "gac.trt.backbone-conv-int8.v1"
COMPUTE_LAYER_TYPES = (
    "CONVOLUTION", "DECONVOLUTION", "MATRIX_MULTIPLY", "FULLY_CONNECTED",
    "ACTIVATION", "POOLING", "ELEMENTWISE", "REDUCE", "SOFTMAX",
    "NORMALIZATION", "SCALE", "UNARY", "PARAMETRIC_RELU", "GRID_SAMPLE",
)
BACKBONE = re.compile(r"(?:^|[/.:])img_backbone(?:_\d+)?[/.](.*)")
BODY = re.compile(r"layer[1-4](?:_\d+)?(?:[/.]|$)")
STEM = re.compile(r"(?:conv1|bn1|relu|maxpool|stem)(?:_\d+)?(?:[/.]|$)")
OFFSET_PREDICTOR = re.compile(
    r"(?:^|[/.])(?:conv_offset(?:_mask)?|offset_conv|offset_mask_conv|conv_mask|mask_conv|offset)"
    r"(?:_\d+)?(?:[/.]|$)"
)


def policy_definition(*, plugins_fp32=False):
    """Return stable rules suitable for including in a calibration cache key."""
    return {
        "version": POLICY_VERSION,
        "network_typing": "weak",
        "precision_constraints": "OBEY_PRECISION_CONSTRAINTS",
        "backbone_scopes": "img_backbone or img_backbone_<camera>/layer1..layer4",
        "backbone_convolution_compute": "int8",
        "backbone_convolution_floating_output": "fp16",
        "offset_mask_predictor_compute_and_output": "fp16",
        "stem_and_other_floating_compute_and_output": "fp16",
        "compute_layer_types": list(COMPUTE_LAYER_TYPES),
        "plugin_floating_edges": "fp32" if plugins_fp32 else "fp16",
        "plugin_method": "Identity-constrained floating inputs and output types; no layer.precision",
        "shape_integer_edges": "unchanged",
        "public_input_output_dtypes": "unchanged",
        "tensorRT_TF32_flag": False,
        "calibration": "caller must supply representative real-data calibration; no synthetic ranges",
    }


def policy_fingerprint(*, plugins_fp32=False):
    serialized = json.dumps(policy_definition(plugins_fp32=plugins_fp32), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def classify_scope(name):
    match = BACKBONE.search(name)
    if match is None:
        return "head_or_other"
    suffix = match.group(1)
    if BODY.match(suffix):
        return "backbone_body"
    if STEM.match(suffix):
        return "backbone_stem"
    return "backbone_other"


def _types(enum, names):
    return {getattr(enum, name) for name in names if hasattr(enum, name)}


def _float_types(trt):
    return {trt.float16, trt.float32}


def constrain_plugin_float_edges(network, layer, trt, *, precision="fp32"):
    """Constrain DCN/MSDA floating boundaries without changing INT32 shape edges.

    This helper also serves the plain FP16 + FP32-plugin baseline. The caller
    must enable OBEY_PRECISION_CONSTRAINTS. Both checked plugin ABIs require all
    floating input/output types to match; MSDA's spatial shapes/offsets stay INT32.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError("Plugin floating precision must be fp16 or fp32")
    target = trt.float16 if precision == "fp16" else trt.float32
    inputs, outputs = [], []
    for index in range(layer.num_inputs):
        tensor = layer.get_input(index)
        if tensor is not None and tensor.dtype in _float_types(trt):
            identity = network.add_identity(tensor)
            if identity is None:
                raise RuntimeError(f"Could not constrain {layer.name} input {index}")
            identity.name = f"{layer.name}/{precision}_input_{index}"
            identity.set_output_type(0, target)
            layer.set_input(index, identity.get_output(0))
            inputs.append(index)
    for index in range(layer.num_outputs):
        if layer.get_output(index).dtype in _float_types(trt):
            layer.set_output_type(index, target)
            outputs.append(index)
    return {
        "name": layer.name, "floating_precision": precision,
        "float_input_indices": inputs, "float_output_indices": outputs,
        "method": "Identity-constrained floating edges; integer edges unchanged; no layer.precision",
    }


def _plugin_kind(name):
    lowered = name.lower()
    if "modulateddeformconv" in lowered:
        return "dcn"
    if "msdeformattn" in lowered or "multiscaledeformableattn" in lowered:
        return "msda"
    return None


def apply_precision_policy(network, config, trt, *, plugins_fp32=False):
    """Apply the policy and return a JSON-serializable constraint audit.

    INT8 applies to ordinary residual-block Conv computation (including
    downsample), not offset/mask prediction, DCNv2, or every backbone operator.
    Conv outputs are constrained to FP16, so inter-op quantize/dequantize/reformat
    costs remain possible. Public input/output tensor dtypes are not rewritten.
    """
    for flag in ("INT8", "FP16", "OBEY_PRECISION_CONSTRAINTS"):
        config.set_flag(getattr(trt.BuilderFlag, flag))
    config.clear_flag(trt.BuilderFlag.TF32)
    compute_types = _types(trt.LayerType, COMPUTE_LAYER_TYPES)
    plugin_types = _types(trt.LayerType, ("PLUGIN_V2", "PLUGIN_V3"))
    # Snapshot first: input-boundary identities append layers to the network.
    layers = [network.get_layer(i) for i in range(network.num_layers)]
    records, exceptions, skipped = [], [], Counter()
    for layer in layers:
        scope = classify_scope(layer.name)
        floats = [i for i in range(layer.num_outputs)
                  if layer.get_output(i).dtype in _float_types(trt)
                  and not getattr(layer.get_output(i), "is_shape_tensor", False)]
        if not floats:
            skipped["integer_or_nonfloating_output"] += 1
            continue
        if layer.type in plugin_types:
            kind = _plugin_kind(layer.name)
            if kind is None:
                raise ValueError(f"Unrecognized floating plugin needs an explicit precision rule: {layer.name}")
            precision = "fp32" if plugins_fp32 else "fp16"
            record = constrain_plugin_float_edges(network, layer, trt, precision=precision)
            record.update(scope=scope, plugin_kind=kind, compute_precision="plugin ABI selects matching floating types")
            records.append(record)
            if kind == "dcn":
                exceptions.append({"name": layer.name, "precision": precision,
                                   "reason": "Built-in DCNv2 plugin has no INT8 format"})
            if plugins_fp32:
                exceptions.append({"name": layer.name, "precision": "fp32",
                                   "reason": "Explicit plugins_fp32 accuracy-baseline override"})
            continue
        if layer.type not in compute_types:
            skipped["layout_shape_or_unlisted_layer_type"] += 1
            continue
        offset = scope == "backbone_body" and OFFSET_PREDICTOR.search(layer.name) is not None
        int8 = scope == "backbone_body" and layer.type == trt.LayerType.CONVOLUTION and not offset
        layer.precision = trt.int8 if int8 else trt.float16
        for output in floats:
            layer.set_output_type(output, trt.float16)
        records.append({"name": layer.name, "scope": scope,
                        "compute_precision": "int8" if int8 else "fp16",
                        "floating_output_precision": "fp16", "float_output_indices": floats})
        if offset and layer.type == trt.LayerType.CONVOLUTION:
            exceptions.append({"name": layer.name, "precision": "fp16",
                               "reason": "Keep DCN offset/mask predictor floating to protect sampling coordinates"})
    return {
        "status": "constraints_applied_actual_engine_unverified",
        "rules": policy_definition(plugins_fp32=plugins_fp32),
        "fingerprint_sha256": policy_fingerprint(plugins_fp32=plugins_fp32),
        "original_layer_count": len(layers), "result_layer_count": network.num_layers,
        "constraint_counts": dict(Counter(r.get("floating_precision", r["compute_precision"]) for r in records)),
        "layers": records, "exceptions": exceptions, "skipped": dict(skipped),
        "validation": {"calibration_quality": "not_checked", "actual_engine_precision": "not_checked",
                       "numerical_accuracy": "not_checked", "performance": "not_measured",
                       "internal_accumulation_precision": "backend dependent; not established by these constraints"},
    }
