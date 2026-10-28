"""CPU-only, structurally audited MSDA plugin replacement and range rebinding.

No weights, edges, shape metadata or precision rules are changed. Structural
preservation does not establish custom-plugin numerical accuracy/performance.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


PLUGIN_DOMAIN = "trt.plugins"
NVIDIA_PLUGIN = "MultiscaleDeformableAttnPlugin_TRT"
GAC_PLUGIN = "GACMSDeformAttn"
DEFAULT_FIELDS = {"channels_per_thread": 2, "block_threads": 256, "specialize": 1, "prefetch": 0}
FIELD_CHOICES = {"channels_per_thread": (1, 2, 4), "block_threads": (128, 256, 512, 768),
                 "specialize": (0, 1, 2, 3), "prefetch": (0, 1, 2)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fields(overrides=None):
    result = dict(DEFAULT_FIELDS)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(result):
            raise ValueError("MSDA fields must contain only channels_per_thread, block_threads, specialize and prefetch")
        result.update(overrides)
    for name, value in result.items():
        if type(value) is not int or value not in FIELD_CHOICES[name]:
            raise ValueError(f"Invalid INT32 MSDA field {name}: {value!r}; expected {FIELD_CHOICES[name]}")
    return result


def _attributes(node):
    from onnx import helper
    result = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    if len(result) != len(node.attribute):
        raise ValueError(f"Duplicate MSDA attributes: {node.name}")
    return result


def _check_node(node, *, custom=False):
    from onnx import AttributeProto
    plugin = GAC_PLUGIN if custom else NVIDIA_PLUGIN
    if node.domain != PLUGIN_DOMAIN or node.op_type != plugin:
        raise ValueError(f"Expected {PLUGIN_DOMAIN}::{plugin}, found {node.domain}::{node.op_type}: {node.name}")
    if not node.name or len(node.input) != 5 or len(node.output) != 1 or not all((*node.input, *node.output)):
        raise ValueError(f"MSDA requires a named node with the adapted five-input/one-output ABI: {node.name}")
    attributes = _attributes(node)
    expected = {"plugin_version", "plugin_namespace"} | (set(DEFAULT_FIELDS) if custom else set())
    if set(attributes) - expected or attributes.get("plugin_version") != (b"1" if custom else b"2"):
        raise ValueError(f"Unsupported MSDA plugin version/fields: {node.name}")
    if attributes.get("plugin_namespace", b"") != b"":
        raise ValueError(f"MSDA requires an empty plugin namespace: {node.name}")
    if custom:
        if not set(DEFAULT_FIELDS) <= set(attributes):
            raise ValueError(f"Custom MSDA node is missing explicit fields: {node.name}")
        for attribute in node.attribute:
            if attribute.name in DEFAULT_FIELDS and attribute.type != AttributeProto.INT:
                raise ValueError(f"MSDA field must be an integer: {node.name}/{attribute.name}")
        return _fields({name: attributes[name] for name in DEFAULT_FIELDS})
    return None


def _check_embedded(model):
    from onnx import TensorProto
    if model.functions or model.training_info:
        raise ValueError("MSDA rewrite requires an inference graph without local functions/training metadata")
    tensors, pending = [], [model.graph]
    while pending:
        graph = pending.pop()
        if graph.sparse_initializer:
            raise ValueError("MSDA rewrite does not support sparse weights")
        tensors.extend(graph.initializer)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
                if attribute.HasField("g"):
                    pending.append(attribute.g)
                pending.extend(attribute.graphs)
    if any(tensor.data_location == TensorProto.EXTERNAL or tensor.external_data for tensor in tensors):
        raise ValueError("MSDA rewrite requires embedded weights; run bevformer-trt adapt first")


def prove_msda_only(source, destination):
    """Verify every protobuf field, allowing only checked NVIDIA -> GAC nodes."""
    _check_embedded(source)
    _check_embedded(destination)
    if len(source.graph.node) != len(destination.graph.node):
        raise ValueError("MSDA rewrite changed the node count")
    restored = deepcopy(destination)
    records = []
    for index, (before, after) in enumerate(zip(source.graph.node, destination.graph.node)):
        if before.SerializeToString() == after.SerializeToString():
            continue
        _check_node(before)
        fields = _check_node(after, custom=True)
        probe = deepcopy(after)
        probe.op_type = before.op_type
        probe.ClearField("attribute")
        probe.attribute.extend(before.attribute)
        if probe.SerializeToString() != before.SerializeToString():
            raise ValueError(f"MSDA rewrite changed node metadata or tensor edges: {before.name}")
        restored.graph.node[index].CopyFrom(before)
        records.append({"index": index, "name": before.name, "inputs": list(before.input),
                        "outputs": list(before.output), "source_plugin": NVIDIA_PLUGIN,
                        "source_version": "2", "plugin": GAC_PLUGIN, "plugin_version": "1",
                        "plugin_namespace": "", "fields": fields})
    if not records:
        raise ValueError("No NVIDIA-to-GAC MSDA replacements were found")
    if restored.SerializeToString() != source.SerializeToString():
        raise ValueError("Graph/weights/shapes/model metadata changed outside MSDA plugin identity and fields")
    return records


def rewrite_model(model, *, node_names=None, fields=None, node_configs=None, expected_msda_count=18):
    """Return a copied model and proof; an explicit name list permits subsets."""
    from onnx import helper
    _check_embedded(model)
    defaults = _fields(fields)
    nodes = list(model.graph.node)
    candidates = [node for node in nodes if node.domain == PLUGIN_DOMAIN and node.op_type == NVIDIA_PLUGIN]
    if expected_msda_count is not None and len(candidates) != expected_msda_count:
        raise ValueError(f"Expected {expected_msda_count} adapted NVIDIA MSDA nodes, found {len(candidates)}")
    for node in candidates:
        _check_node(node)
    names = [node.name for node in nodes]
    requested = [node.name for node in candidates] if node_names is None else list(node_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("MSDA selection must contain unique nonempty exact node names")
    for name in requested:
        if names.count(name) != 1:
            raise ValueError(f"MSDA node name is missing or ambiguous: {name}")
        _check_node(nodes[names.index(name)])
    configs = {} if node_configs is None else node_configs
    if not isinstance(configs, dict) or set(configs) - set(requested):
        raise ValueError("Per-node configuration contains unselected/unknown exact node names")
    selected = set(requested)
    result = deepcopy(model)
    for node in result.graph.node:
        if node.name not in selected:
            continue
        overrides = configs.get(node.name, {})
        _fields(overrides)
        config = _fields({**defaults, **overrides})
        node.op_type = GAC_PLUGIN
        node.ClearField("attribute")
        node.attribute.extend(helper.make_attribute(name, value) for name, value in {
            "plugin_version": "1", "plugin_namespace": "", **config}.items())
    records = prove_msda_only(model, result)
    return result, {"schema_version": 1, "kind": "gac_msda_plugin_rewrite",
                    "source_msda_count": len(candidates), "replaced_node_count": len(records), "nodes": records,
                    "preservation": {"all_other_model_protobuf_fields_identical": True,
                                     "weights_shapes_tensor_edges_identical": True},
                    "validation": {"structural_preservation": "passed", "numerical_accuracy": "not_checked",
                                   "performance": "not_measured"}}


def _output_paths(inputs, outputs):
    paths = [Path(path).resolve() for path in (*inputs, *outputs)]
    if len(paths) != len(set(paths)):
        raise ValueError("Input and output artifact paths must all differ")
    for path in outputs:
        if Path(path).exists():
            raise FileExistsError(f"Output already exists: {path}")


def _write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rewrite_file(source, output, report, **options):
    import onnx
    source, output, report = map(Path, (source, output, report))
    _output_paths((source,), (output, report))
    source_hash = _sha_file(source)
    model = onnx.load(source, load_external_data=False)
    result, audit = rewrite_model(model, **options)
    payload = result.SerializeToString()
    if _sha_file(source) != source_hash:
        raise ValueError("Source ONNX changed while rewriting")
    audit.update(source_onnx=str(source.resolve()), source_onnx_sha256=source_hash,
                 output_onnx=str(output.resolve()), output_onnx_sha256=_digest(payload))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    _write_json_new(report, audit)
    return audit


def _prove_calibrated_ancestors(model, manifest, mapping):
    """Require the exact INT8 Conv set and show no mapped tensor depends on MSDA."""
    from .backbone_calibration import _description, _is_int8_conv
    from onnx import AttributeProto
    names = [node.name for node in model.graph.node if _is_int8_conv(node)]
    if len(set(names)) != len(names) or set(names) != set(manifest["int8_convolution_names"]):
        raise ValueError("Source model INT8 convolution set differs from the calibration manifest")
    if [_description(value) for value in model.graph.input] != manifest["inputs"]:
        raise ValueError("Source model inputs differ from the calibration manifest")
    terminals = {value.name for value in model.graph.input} | {weight.name for weight in model.graph.initializer}
    producers = {}
    by_name = {}
    for node in model.graph.node:
        by_name.setdefault(node.name, []).append(node)
        for name in node.output:
            if name in producers or name in terminals:
                raise ValueError(f"Ambiguous tensor producer: {name}")
            producers[name] = node
    pending, visited = list(mapping), set()
    while pending:
        name = pending.pop()
        if not name or name in visited:
            continue
        visited.add(name)
        if name in terminals:
            continue
        if name not in producers:
            raise ValueError(f"Calibrated tensor ancestor is absent from source graph: {name}")
        node = producers[name]
        if node.op_type in (NVIDIA_PLUGIN, GAC_PLUGIN, "MSDeformAttn"):
            raise ValueError(f"Calibrated tensor depends on MSDA: {name}")
        if any(attribute.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for attribute in node.attribute):
            raise ValueError(f"Calibrated tensor has unsupported control-flow ancestors: {name}")
        pending.extend(node.input)
    edges = manifest.get("int8_activation_edges")
    if edges is None:
        raise ValueError("Range rebinding requires explicit INT8 activation edges in the source manifest")
    for row in edges:
        nodes = by_name.get(row["layer"], [])
        if len(nodes) != 1:
            raise ValueError(f"Missing/ambiguous calibrated Conv: {row['layer']}")
        node = nodes[0]
        actual = node.input[0] if row["edge"] == "input0" else node.output[0]
        if actual != row["tensor"]:
            raise ValueError(f"Calibrated Conv edge differs from source model: {row['layer']}/{row['edge']}")


def rebind_ranges(*, source_onnx, destination_onnx, ranges, output):
    """Derive a new explicit target binding, preserving the measured cache intact."""
    import onnx
    from .calibration_ranges import FORMAT_KIND, TRT_VERSION, _required_edges, _validate_manifest
    _output_paths((source_onnx, destination_onnx, ranges), (output,))
    source_hash, destination_hash = _sha_file(source_onnx), _sha_file(destination_onnx)
    original_raw = Path(ranges).read_bytes()
    original = json.loads(original_raw)
    unsigned = {key: value for key, value in original.items() if key != "artifact_sha256"}
    if original.get("artifact_sha256") != _digest(_canonical(unsigned)):
        raise ValueError("Original range artifact checksum mismatch")
    if (original.get("schema_version") != 1 or original.get("kind") != FORMAT_KIND
            or original.get("tensorrt") != TRT_VERSION
            or original.get("source_full_onnx_sha256") != source_hash):
        raise ValueError("Original range artifact is not bound to the source ONNX/version/schema")
    if "msda_rebinding" in original:
        raise ValueError("Rebind directly from original measured ranges, not a previously rebound artifact")
    manifest = original["source_manifest"]["document"]
    names, mapping, testing = _validate_manifest(manifest)
    edges = _required_edges(manifest, names, mapping)
    cache = original["source_cache"]
    binding = cache["binding"]
    if (manifest["source_full_onnx_sha256"] != source_hash
            or original["subgraph_onnx_sha256"] != manifest["subgraph_onnx_sha256"]
            or binding.get("model_sha256") != manifest["subgraph_onnx_sha256"]
            or original["policy_rule_fingerprint_sha256"] != manifest["policy_rule_fingerprint_sha256"]
            or cache.get("binding_fingerprint") != _digest(_canonical(binding))
            or binding.get("tensorrt") != TRT_VERSION or binding.get("algorithm") != "IInt8EntropyCalibrator2"
            or binding.get("calibration", {}).get("provenance", {}).get("source_kind") != "real_preprocessed"):
        raise ValueError("Original model/manifest/cache/policy binding mismatch")
    if testing or original.get("synthetic_interface_test") is not False:
        raise ValueError("MSDA range rebinding requires measured production backbone ranges")
    if edges is None or any(tensor not in original.get("ranges", {}) for tensor in edges.values()):
        raise ValueError("Original range artifact is missing required calibrated Conv edges")
    source = onnx.load(source_onnx, load_external_data=False)
    destination = onnx.load(destination_onnx, load_external_data=False)
    records = prove_msda_only(source, destination)
    _prove_calibrated_ancestors(source, manifest, mapping)
    if (_sha_file(source_onnx), _sha_file(destination_onnx)) != (source_hash, destination_hash):
        raise ValueError("ONNX files changed during range rebinding")
    result = deepcopy(unsigned)
    # This is a newly derived manifest, not the original extraction manifest.
    # Preserve the complete original artifact alongside the explicit proof.
    derived_manifest = deepcopy(manifest)
    derived_manifest["source_full_onnx_sha256"] = destination_hash
    derived_manifest["source_full_onnx"] = str(Path(destination_onnx).resolve())
    derived_manifest["derivation"] = {"kind": "msda_only_target_rebinding",
        "original_source_full_onnx_sha256": source_hash,
        "original_source_manifest_sha256": original["source_manifest"]["sha256"],
        "original_precision_plan_binding": "precision_plan_sha256/source_full_policy_fingerprint describe the original source graph"}
    result["source_full_onnx_sha256"] = destination_hash
    result["source_manifest"] = {"sha256": _digest(_canonical(derived_manifest)), "document": derived_manifest,
                                 "encoding": "canonical JSON, sorted keys, compact separators, UTF-8"}
    result["msda_rebinding"] = {"schema_version": 1, "kind": "explicit_msda_only_target_rebinding",
        "source_onnx": str(Path(source_onnx).resolve()), "source_onnx_sha256": source_hash,
        "destination_onnx": str(Path(destination_onnx).resolve()), "destination_onnx_sha256": destination_hash,
        "original_ranges_file_sha256": _digest(original_raw), "original_artifact": original,
        "changed_nodes": records, "all_other_model_protobuf_fields_identical": True,
        "mapped_calibration_tensors_have_no_msda_ancestors": True,
        "measured_cache_binding_unchanged": True,
        "scope": "Exact upstream backbone graph/ranges preserved; custom MSDA accuracy and performance remain unverified"}
    result["artifact_sha256"] = _digest(_canonical(result))
    _write_json_new(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    rewrite = commands.add_parser("msda-rewrite", help="Replace adapted NVIDIA MSDA nodes")
    rewrite.add_argument("--onnx", type=Path, required=True)
    rewrite.add_argument("--output", type=Path, required=True)
    rewrite.add_argument("--report", type=Path, required=True)
    rewrite.add_argument("--node", action="append", help="Exact node name to replace; repeatable; default all")
    rewrite.add_argument("--node-config", type=Path, help="JSON mapping exact selected node names to field overrides")
    rewrite.add_argument("--expected-msda-count", type=int, default=18,
                         help="Assert original NVIDIA MSDA count (default 18 for BEVFormer Base)")
    for name, choices in FIELD_CHOICES.items():
        rewrite.add_argument("--" + name.replace("_", "-"), type=int, choices=choices, default=DEFAULT_FIELDS[name])
    rebind = commands.add_parser("msda-rebind-ranges", help="Derive ranges bound to a proved MSDA-only rewritten graph")
    rebind.add_argument("--source-onnx", type=Path, required=True)
    rebind.add_argument("--onnx", type=Path, required=True, help="Rewritten destination ONNX")
    rebind.add_argument("--ranges", type=Path, required=True, help="Original production range artifact")
    rebind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "msda-rewrite":
            configs = json.loads(args.node_config.read_text(encoding="utf-8")) if args.node_config else None
            result = rewrite_file(args.onnx, args.output, args.report, node_names=args.node,
                                  fields={name: getattr(args, name) for name in DEFAULT_FIELDS},
                                  node_configs=configs, expected_msda_count=args.expected_msda_count)
            print(json.dumps(result, indent=2))
        else:
            result = rebind_ranges(source_onnx=args.source_onnx, destination_onnx=args.onnx,
                                   ranges=args.ranges, output=args.output)
            print(json.dumps({"output": str(args.output), "source_full_onnx_sha256": result["source_full_onnx_sha256"],
                              "original_source_full_onnx_sha256": result["msda_rebinding"]["source_onnx_sha256"],
                              "measured_cache_binding_unchanged": True}, indent=2))
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
