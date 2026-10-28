"""Transfer measured backbone ranges through TensorRT's dynamic-range API.

This is a version-locked TensorRT 10.13.3.9 cache-format adapter, not a full-model
calibration cache. NVIDIA Model Optimizer decodes cache scales as big-endian
IEEE754 float32; the cache format itself is not a stable TensorRT public ABI.
The destination network keeps its independently enforced mixed-precision policy.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from .precision_policy import policy_fingerprint
from .quantization import CACHE_MAGIC, DEFAULT_INPUT_SHAPE


TRT_VERSION = "10.13.3.9"
CACHE_HEADER = "TRT-101303-EntropyCalibration2"
PRODUCTION_KIND = "bevformer_backbone_calibration_subgraph"
TEST_KIND = "identity_calibration_interface_test"
FORMAT_KIND = "gac_tensorrt_calibrated_dynamic_ranges"
SOURCE = "https://github.com/NVIDIA/Model-Optimizer/blob/d7cbff3486a3ffb1a357e32d9810da0bfc649709/modelopt/onnx/quantization/calib_utils.py#L143-L165"
FLOAT32_MAX = 3.4028234663852886e38


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA-256: {label}")
    return value


def _names(values, label):
    if (not isinstance(values, list) or not values
            or any(not isinstance(name, str) or not name or "\n" in name for name in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must contain unique nonempty names")
    return set(values)


def _validate_manifest(manifest, *, allow_synthetic_for_testing=False):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported subgraph manifest schema")
    kind = manifest.get("kind")
    testing = kind == TEST_KIND
    if kind != PRODUCTION_KIND and not (testing and allow_synthetic_for_testing):
        raise ValueError("Expected production backbone manifest; identity tests require explicit opt-in")
    full = _sha(manifest.get("source_full_onnx_sha256"), "full ONNX")
    subgraph = _sha(manifest.get("subgraph_onnx_sha256"), "subgraph ONNX")
    if testing and full != subgraph:
        raise ValueError("Identity test manifest must have identical full/subgraph hashes")
    names = _names(manifest.get("int8_convolution_names"), "INT8 convolution coverage")
    coverage = manifest.get("coverage", {})
    if (coverage.get("expected_int8_convolutions") != len(names)
            or coverage.get("retained_int8_convolutions") != len(names)
            or coverage.get("missing") != [] or coverage.get("extra") != []):
        raise ValueError("Incomplete INT8 convolution coverage")
    if not testing and len(names) != 462:
        raise ValueError("BEVFormer Base production manifest must cover exactly 462 INT8 convolutions")
    preservation = manifest.get("preservation", {})
    for name in ("node_protobufs_identical", "initializer_protobufs_identical",
                 "input_value_info_identical", "tensor_names_identical"):
        if preservation.get(name) is not True:
            raise ValueError(f"Subgraph preservation was not established: {name}")
    rule = _sha(manifest.get("policy_rule_fingerprint_sha256"), "policy rule")
    if rule not in (policy_fingerprint(), policy_fingerprint(plugins_fp32=True)):
        raise ValueError("Manifest precision rules do not match this implementation")
    _sha(manifest.get("source_full_policy_fingerprint"), "source full policy")
    _sha(manifest.get("precision_plan_sha256"), "precision plan")
    inputs = manifest.get("inputs", [])
    if len(inputs) != 1 or inputs[0].get("dtype") != "float32":
        raise ValueError("Manifest must preserve one FP32 input")
    if not testing and inputs[0].get("shape") != list(DEFAULT_INPUT_SHAPE):
        raise ValueError("Production manifest must preserve the complete six-camera input")
    mapping = manifest.get("tensor_mapping", [])
    if not mapping:
        raise ValueError("Manifest requires an explicit tensor mapping")
    mapped = {}
    for item in mapping:
        source, target = item.get("source_name"), item.get("subgraph_name")
        if not isinstance(source, str) or not source or source != target or target in mapped:
            raise ValueError("Tensor mapping must be unique and preserve exact original names")
        mapped[target] = item
    return names, mapped, testing


def _read_bound_cache(path):
    raw = Path(path).read_bytes()
    if not raw.startswith(CACHE_MAGIC):
        raise ValueError("Only a GAC provenance-bound calibration cache is accepted")
    header_bytes, separator, payload = raw[len(CACHE_MAGIC):].partition(b"\n")
    if not separator or not payload:
        raise ValueError("Truncated bound calibration cache")
    header = json.loads(header_bytes)
    binding = header.get("binding")
    if (header.get("schema_version") != 1 or not isinstance(binding, dict)
            or header.get("fingerprint") != _digest(_canonical(binding))
            or header.get("payload_sha256") != _digest(payload)):
        raise ValueError("Bound cache checksum/fingerprint mismatch")
    if binding.get("tensorrt") != TRT_VERSION or binding.get("algorithm") != "IInt8EntropyCalibrator2":
        raise ValueError(f"Only TensorRT {TRT_VERSION} EntropyCalibrator2 caches are accepted")
    _sha(binding.get("policy_fingerprint"), "cache policy")
    return raw, header, payload


def _required_edges(manifest, names, mapping):
    """New manifests prove required edges before conversion; older ones defer to TRT."""
    rows = manifest.get("int8_activation_edges")
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ValueError("INT8 activation edges must be a list")
    edges = {}
    for row in rows:
        key, tensor = (row.get("layer"), row.get("edge")), row.get("tensor")
        if (key in edges or key[0] not in names or key[1] not in ("input0", "output0")
                or tensor not in mapping or mapping[tensor].get("dtype") not in ("float16", "float32")):
            raise ValueError(f"Invalid/duplicate required INT8 activation edge: {row}")
        edges[key] = tensor
    if set(edges) != {(name, edge) for name in names for edge in ("input0", "output0")}:
        raise ValueError("Incomplete required INT8 activation edge coverage")
    return edges


def _parse_scales(payload, *, invalid_records=None):
    """Decode observed 1..8 hex digits; optionally audit unusable scale records.

    TRT 10.13.3.9 writes zero constants as `0`, not `00000000`. Zero/Inf records
    may describe unused constants. They are never converted to dynamic ranges.
    The default remains strict for callers that have not supplied a scope audit.
    """
    lines = payload.decode("utf-8").splitlines()
    if not lines or lines[0] != CACHE_HEADER:
        raise ValueError(f"Expected exact cache header {CACHE_HEADER}")
    scales, seen = {}, set()
    for line in lines[1:]:
        if not line:
            continue
        name, separator, encoded = line.rpartition(": ")
        if not separator or not name or name in seen or not re.fullmatch(r"[0-9a-fA-F]{1,8}", encoded):
            raise ValueError(f"Invalid/duplicate tensor scale record: {line!r}")
        seen.add(name)
        scale = struct.unpack("!f", bytes.fromhex(encoded.zfill(8)))[0]
        if not math.isfinite(scale) or scale <= 0 or scale * 127 > FLOAT32_MAX:
            reason = ("nonfinite" if not math.isfinite(scale) else "zero" if scale == 0
                      else "negative" if scale < 0 else "dynamic_range_overflow")
            if invalid_records is None:
                raise ValueError(f"Nonpositive, nonfinite or overflowing scale for {name}")
            invalid_records.append({"tensor": name, "raw_hex": encoded, "reason": reason})
            continue
        scales[name] = scale
    if not seen:
        raise ValueError("Calibration cache contains no scales")
    return scales


def convert_bound_cache(*, cache, subgraph_manifest, output=None, allow_synthetic_for_testing=False):
    """Return/save an independently bound range artifact; never emit a TRT cache."""
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Range artifact already exists: {output}")
    manifest_raw = Path(subgraph_manifest).read_bytes()
    manifest = json.loads(manifest_raw)
    names, mapping, testing = _validate_manifest(manifest, allow_synthetic_for_testing=allow_synthetic_for_testing)
    required_edges = _required_edges(manifest, names, mapping)
    raw, header, payload = _read_bound_cache(cache)
    binding = header["binding"]
    if binding.get("model_sha256") != manifest["subgraph_onnx_sha256"]:
        raise ValueError("Calibration cache was not measured on the declared subgraph")
    dataset = binding.get("calibration", {})
    provenance = dataset.get("provenance", {})
    expected_kind = "synthetic_interface_test" if testing else "real_preprocessed"
    if provenance.get("source_kind") != expected_kind:
        raise ValueError(f"Expected calibration source_kind={expected_kind}")
    samples = dataset.get("ordered_samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Bound cache must identify measured calibration samples")
    input_spec = manifest["inputs"][0]
    if (dataset.get("input_name") != input_spec["name"] or dataset.get("input_dtype") != "float32"
            or dataset.get("input_shape") != input_spec["shape"]):
        raise ValueError("Calibration input does not match the preserved source input")
    excluded = []
    scales = _parse_scales(payload, invalid_records=excluded)
    if required_edges is not None:
        invalid_by_name = {row["tensor"]: row for row in excluded}
        for (layer, edge), tensor in required_edges.items():
            if tensor in invalid_by_name:
                raise ValueError(f"Invalid required dynamic range: {layer}/{edge} -> {invalid_by_name[tensor]}")
            if tensor not in scales:
                raise ValueError(f"Missing required dynamic range: {layer}/{edge} -> {tensor}")
    for row in excluded:
        row["mapped_to_source"] = row["tensor"] in mapping
        row["required_edge_check"] = ("excluded_nonrequired" if required_edges is not None
            else "deferred_to_destination_network" if row["mapped_to_source"] else "not_in_source_mapping")
    ranges, unmapped = {}, []
    for name, scale in scales.items():
        if name not in mapping:
            unmapped.append(name)
            continue
        if mapping[name].get("dtype") not in ("float16", "float32"):
            raise ValueError(f"Calibration scale maps to a nonfloating/unknown tensor: {name}")
        ranges[name] = {"scale": scale, "max_abs": 127.0 * scale}
    if not ranges:
        raise ValueError("No calibrated tensors match the declared source graph")
    artifact = {
        "schema_version": 1, "kind": FORMAT_KIND, "tensorrt": TRT_VERSION,
        "status": "measured_subgraph_ranges_not_full_model_calibration",
        "source_full_onnx_sha256": manifest["source_full_onnx_sha256"],
        "subgraph_onnx_sha256": manifest["subgraph_onnx_sha256"],
        "policy_rule_fingerprint_sha256": manifest["policy_rule_fingerprint_sha256"],
        "source_manifest": {"sha256": _digest(manifest_raw), "document": manifest},
        "source_cache": {"sha256": _digest(raw), "payload_sha256": header["payload_sha256"],
                         "binding_fingerprint": header["fingerprint"], "binding": binding},
        "conversion": {"cache_header": CACHE_HEADER, "encoding": "1..8 hex digits of big-endian IEEE754 float32 bits",
                       "dynamic_range": "(-127 * scale, +127 * scale)", "source": SOURCE,
                       "compatibility": "Version-locked adapter; not a stable TensorRT cache ABI"},
        "ranges": ranges, "unmapped_cache_tensors": unmapped,
        "excluded_invalid_scales": excluded,
        "required_edges_validation": "manifest_checked" if required_edges is not None else "deferred_to_destination_network",
        "calibration_sample_count": len(samples), "synthetic_interface_test": testing,
        "validation": {"destination_engine_precision": "not_checked", "numerical_accuracy": "not_checked"},
    }
    artifact["artifact_sha256"] = _digest(_canonical(artifact))
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return artifact


def apply_calibrated_ranges(network, trt, *, path, model_sha256, precision_policy,
                            allow_synthetic_for_testing=False):
    """Set measured backbone activations, including residual-fusion boundaries.

    Conv activation edges are mandatory. Other mapped, measured floating
    activations enable fused Conv/Add/ReLU formats without changing precision.
    Initializer/constant weights and their layout aliases are excluded.
    """
    if trt.__version__ != TRT_VERSION:
        raise ValueError(f"Range adapter requires exactly TensorRT {TRT_VERSION}")
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_digest = artifact.pop("artifact_sha256", None)
    if expected_digest != _digest(_canonical(artifact)):
        raise ValueError("Range artifact checksum mismatch")
    if (artifact.get("schema_version") != 1 or artifact.get("kind") != FORMAT_KIND
            or artifact.get("tensorrt") != TRT_VERSION
            or artifact.get("source_full_onnx_sha256") != _sha(model_sha256, "destination model")):
        raise ValueError("Range artifact does not match destination model/version/schema")
    manifest = artifact["source_manifest"]["document"]
    names, mapping, testing = _validate_manifest(manifest, allow_synthetic_for_testing=allow_synthetic_for_testing)
    required_edges = _required_edges(manifest, names, mapping)
    if (manifest["source_full_onnx_sha256"] != model_sha256
            or manifest["subgraph_onnx_sha256"] != artifact["subgraph_onnx_sha256"]
            or artifact["policy_rule_fingerprint_sha256"] != manifest["policy_rule_fingerprint_sha256"]
            or precision_policy.get("fingerprint_sha256") != manifest["policy_rule_fingerprint_sha256"]):
        raise ValueError("Source graph or precision-policy binding mismatch")
    records = [item for item in precision_policy.get("layers", []) if item.get("compute_precision") == "int8"]
    actual_names = _names([item["name"] for item in records], "destination INT8 layers")
    if actual_names != names:
        raise ValueError("Destination INT8 convolution set differs from proven subgraph coverage")
    ranges = artifact.get("ranges", {})
    excluded = {row["tensor"]: row for row in artifact.get("excluded_invalid_scales", [])}
    target_layers, targets, layer_edges = {}, {}, []
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    for layer in layers:
        if layer.name in names:
            if layer.name in target_layers:
                raise ValueError(f"Duplicate destination layer name: {layer.name}")
            target_layers[layer.name] = layer
    if set(target_layers) != names:
        raise ValueError("Covered INT8 convolution is missing in destination network")

    def checked_maximum(tensor_name):
        entry = ranges[tensor_name]
        scale, maximum = entry.get("scale"), entry.get("max_abs")
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
                or not math.isfinite(scale) or scale <= 0 or maximum != 127.0 * scale
                or maximum > FLOAT32_MAX):
            raise ValueError(f"Invalid measured dynamic range for {tensor_name}")
        return maximum

    for name in sorted(names):
        layer = target_layers[name]
        if layer.type != trt.LayerType.CONVOLUTION or layer.precision != trt.int8:
            raise ValueError(f"Destination layer is not an explicitly INT8 convolution: {name}")
        edges = [("input0", layer.get_input(0))]
        edges.extend((f"output{index}", layer.get_output(index)) for index in range(layer.num_outputs))
        for edge, tensor in edges:
            if (tensor is None or tensor.dtype not in (trt.float16, trt.float32)
                    or getattr(tensor, "is_shape_tensor", False)):
                raise ValueError(f"Unexpected nonfloating Conv activation edge: {name}/{edge}")
            if required_edges is not None and required_edges.get((name, edge)) != tensor.name:
                raise ValueError(f"Required edge differs from destination network: {name}/{edge} -> {tensor.name}")
            if tensor.name in excluded:
                raise ValueError(f"Invalid required dynamic range: {name}/{edge} -> {excluded[tensor.name]}")
            entry = ranges.get(tensor.name)
            if entry is None or tensor.name not in mapping:
                raise ValueError(f"Missing measured dynamic range: {name}/{edge} -> {tensor.name}")
            maximum = checked_maximum(tensor.name)
            # TRT may return different Python wrappers for the same ITensor.
            # Shared Conv edges intentionally collapse by the verified graph name.
            targets[tensor.name] = (tensor, maximum)
            layer_edges.append({"layer": name, "edge": edge, "tensor": tensor.name, "max_abs": maximum})
    required_tensor_count = len(targets)
    # A constant's Identity/Shuffle output may carry an activation-style name.
    # Keep these weights out too; TensorRT derives constant-weight scales itself.
    constants = {name for name, row in mapping.items() if row.get("producer_name") in ("weight", "initializer")}
    constant_type = getattr(trt.LayerType, "CONSTANT", None)
    alias_types = {getattr(trt.LayerType, name) for name in ("IDENTITY", "SHUFFLE", "CAST")
                   if hasattr(trt.LayerType, name)}
    aliases, candidates = [], []
    for layer in layers:
        outputs = [layer.get_output(index) for index in range(layer.num_outputs)]
        if constant_type is not None and layer.type == constant_type:
            constants.update(tensor.name for tensor in outputs if tensor is not None)
        elif layer.type in alias_types and getattr(layer, "num_inputs", 0):
            aliases.append((layer.get_input(0), outputs))
        candidates.extend((layer.name, tensor) for tensor in outputs)
    while True:
        before = len(constants)
        for source, outputs in aliases:
            if source is not None and source.name in constants:
                constants.update(tensor.name for tensor in outputs if tensor is not None)
        if len(constants) == before:
            break
    candidates.extend(("network_input", network.get_input(index))
                      for index in range(getattr(network, "num_inputs", 0)))
    additional = []
    for producer, tensor in candidates:
        if tensor is None or tensor.name in targets or tensor.name in constants:
            continue
        name = tensor.name
        if (name not in mapping or name not in ranges or name in excluded
                or mapping[name].get("dtype") not in ("float16", "float32")
                or tensor.dtype not in (trt.float16, trt.float32)
                or getattr(tensor, "is_shape_tensor", False)):
            continue
        maximum = checked_maximum(name)
        targets[name] = (tensor, maximum)
        additional.append({"tensor": name, "producer": producer, "max_abs": maximum})
    if constants & set(targets):
        raise ValueError("A required Conv activation unexpectedly resolves to constant weights")
    # Validate ALL candidates before writing any ranges; no fabricated defaults.
    for tensor, maximum in targets.values():
        tensor.dynamic_range = (-maximum, maximum)
        actual = tensor.dynamic_range
        if (actual is None or len(actual) != 2
                or not math.isclose(actual[0], -maximum, rel_tol=1e-6, abs_tol=0.0)
                or not math.isclose(actual[1], maximum, rel_tol=1e-6, abs_tol=0.0)):
            raise RuntimeError(f"TensorRT rejected measured dynamic range for {tensor.name}")
    return {
        "status": "measured_ranges_applied_actual_engine_unverified",
        "scope": "Required INT8 Conv edges and measured mapped backbone activations; no full-model calibrator/cache",
        "artifact_sha256": expected_digest, "source_full_onnx_sha256": model_sha256,
        "subgraph_onnx_sha256": manifest["subgraph_onnx_sha256"],
        "source_cache_sha256": artifact["source_cache"]["sha256"],
        "source_manifest_sha256": artifact["source_manifest"]["sha256"],
        "tensorrt": TRT_VERSION, "synthetic_interface_test": testing,
        "calibration_sample_count": artifact["calibration_sample_count"],
        "covered_int8_convolutions": len(names), "applied_tensor_count": len(targets),
        "required_conv_edge_count": len(layer_edges), "required_conv_tensor_count": required_tensor_count,
        "additional_activation_count": len(additional), "additional_activations": additional,
        "edges": layer_edges, "unused_measured_tensor_count": len(set(ranges) - set(targets)),
        "excluded_invalid_scale_count": len(excluded),
        "weights": "Original constant Conv weights; TensorRT derives their quantization scales",
        "validation": {"actual_engine_precision": "not_checked", "numerical_accuracy": "not_checked"},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="Bound subgraph cache -> separate range JSON")
    convert.add_argument("--cache", type=Path, required=True)
    convert.add_argument("--subgraph-manifest", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--synthetic-interface-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        result = convert_bound_cache(cache=args.cache, subgraph_manifest=args.subgraph_manifest,
                                     output=args.output, allow_synthetic_for_testing=args.synthetic_interface_test)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "measured_tensors": len(result["ranges"]),
                      "source_full_onnx_sha256": result["source_full_onnx_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
