"""Explicit, structurally audited DCNv2 substitution at the parser boundary.

The source ONNX and its calibration artifact stay untouched. The builder must
validate calibration provenance against that source before parsing these bytes.
This module proves graph preservation, not numerical accuracy or a speedup.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


PLUGIN_DOMAIN = "trt.plugins"
NVIDIA_PLUGIN = "ModulatedDeformConv2d"
GAC_PLUGIN = "GACModulatedDeformConv2d"
CONV_FIELDS = {"stride", "padding", "dilation", "group", "deformable_group"}
OPTION_CHOICES = {"block_threads": (128, 256, 512, 1024), "specialize": (1, 3, 4)}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_config(config):
    """Require an explicit subset and exact integer options; never select all."""
    if not isinstance(config, dict) or set(config) != {"node_names", *OPTION_CHOICES}:
        raise ValueError("DCN config requires exactly node_names, block_threads and specialize")
    names = config["node_names"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(names) != len(set(names))):
        raise ValueError("DCN node_names must be a nonempty list of unique exact names")
    for name, choices in OPTION_CHOICES.items():
        if type(config[name]) is not int or config[name] not in choices:
            raise ValueError(f"DCN {name} must be an integer in {choices}")
    return {"node_names": list(names), **{name: config[name] for name in OPTION_CHOICES}}


def _decode_config(payload):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Duplicate DCN config key: {name}")
            result[name] = value
        return result
    return validate_config(json.loads(payload.decode("utf-8-sig"), object_pairs_hook=unique))


def read_config(path):
    return _decode_config(Path(path).read_bytes())


def _check_embedded(model):
    from onnx import TensorProto
    if model.functions or model.training_info:
        raise ValueError("DCN substitution requires an inference graph without local functions")
    pending = [model.graph]
    while pending:
        graph = pending.pop()
        if graph.sparse_initializer:
            raise ValueError("DCN substitution does not support sparse weights")
        tensors = list(graph.initializer)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensors.append(attribute.t)
                tensors.extend(attribute.tensors)
                if attribute.HasField("g"):
                    pending.append(attribute.g)
                pending.extend(attribute.graphs)
        if any(t.data_location == TensorProto.EXTERNAL or t.external_data for t in tensors):
            raise ValueError("DCN substitution requires embedded weights")


def _check_node(node, *, custom=False):
    from onnx import AttributeProto, helper
    plugin = GAC_PLUGIN if custom else NVIDIA_PLUGIN
    if node.domain != PLUGIN_DOMAIN or node.op_type != plugin:
        raise ValueError(f"Expected {PLUGIN_DOMAIN}::{plugin}: {node.name}")
    if not node.name or len(node.input) != 4 or len(node.output) != 1 or not all((*node.input, *node.output)):
        raise ValueError(f"DCN replacement requires four nonempty inputs and one output: {node.name}")
    attributes = {attribute.name: attribute for attribute in node.attribute}
    if len(attributes) != len(node.attribute):
        raise ValueError(f"Duplicate DCN node attributes: {node.name}")
    allowed = CONV_FIELDS | {"plugin_version", "plugin_namespace"} | (set(OPTION_CHOICES) if custom else set())
    required = CONV_FIELDS | {"plugin_version"} | (set(OPTION_CHOICES) if custom else set())
    if not required <= set(attributes) <= allowed:
        raise ValueError(f"Unexpected or missing DCN attributes: {node.name}")
    version = attributes["plugin_version"]
    namespace = attributes.get("plugin_namespace")
    if (version.type != AttributeProto.STRING or version.s != (b"1" if custom else b"2")
            or namespace is not None and (namespace.type != AttributeProto.STRING or namespace.s != b"")):
        raise ValueError(f"Unsupported DCN plugin version/namespace: {node.name}")
    for name in ("stride", "padding", "dilation"):
        attribute = attributes[name]
        minimum = 0 if name == "padding" else 1
        if (attribute.type != AttributeProto.INTS or len(attribute.ints) != 2
                or any(value < minimum or value > 2147483647 for value in attribute.ints)):
            raise ValueError(f"Invalid DCN {name} [width,height] field: {node.name}")
    for name in ("group", "deformable_group"):
        if attributes[name].type != AttributeProto.INT or attributes[name].i != 1:
            raise ValueError(f"DCN replacement currently requires {name}=1: {node.name}")
    options = {}
    if custom:
        for name, choices in OPTION_CHOICES.items():
            attribute = attributes[name]
            if attribute.type != AttributeProto.INT or attribute.i not in choices:
                raise ValueError(f"Invalid custom DCN {name}: {node.name}")
            options[name] = attribute.i
    return {name: helper.get_attribute_value(attributes[name]) for name in CONV_FIELDS}, options


def _known_kernel_shape(model, weight_name):
    """Reject a known non-3x3 weight; the plugin must check unknown shapes."""
    weights = [tensor for tensor in model.graph.initializer if tensor.name == weight_name]
    if len(weights) > 1:
        raise ValueError(f"Ambiguous DCN initializer: {weight_name}")
    if weights:
        shape = list(weights[0].dims)
    else:
        values = [value for value in (*model.graph.input, *model.graph.value_info)
                  if value.name == weight_name]
        if len(values) > 1:
            raise ValueError(f"Ambiguous DCN weight shape: {weight_name}")
        if not values or not values[0].type.HasField("tensor_type"):
            return None
        shape = [dim.dim_value if dim.HasField("dim_value") else None
                 for dim in values[0].type.tensor_type.shape.dim]
    if len(shape) != 4 or any(value is not None and value <= 0 for value in shape):
        raise ValueError(f"DCN weight must have positive rank-four shape: {weight_name}")
    if any(value is not None and value != 3 for value in shape[-2:]):
        raise ValueError(f"DCN replacement requires a 3x3 kernel: {weight_name}")
    return shape


def prove_dcn_only(source, destination):
    """Restore allowed identity/options changes and compare the entire protobuf."""
    _check_embedded(source)
    _check_embedded(destination)
    if len(source.graph.node) != len(destination.graph.node):
        raise ValueError("DCN rewrite changed the node count")
    restored = deepcopy(destination)
    records = []
    for index, (before, after) in enumerate(zip(source.graph.node, destination.graph.node)):
        if before.SerializeToString() == after.SerializeToString():
            continue
        attributes, _ = _check_node(before)
        after_attributes, options = _check_node(after, custom=True)
        if attributes != after_attributes:
            raise ValueError(f"DCN rewrite changed convolution semantics: {before.name}")
        # Replacing all attributes during restoration could otherwise conceal
        # a change to a convolution attribute's metadata or protobuf encoding.
        before_fields = {a.name: a.SerializeToString() for a in before.attribute if a.name in CONV_FIELDS}
        after_fields = {a.name: a.SerializeToString() for a in after.attribute if a.name in CONV_FIELDS}
        if before_fields != after_fields:
            raise ValueError(f"DCN rewrite changed convolution attribute metadata: {before.name}")
        probe = deepcopy(after)
        probe.op_type = before.op_type
        probe.ClearField("attribute")
        probe.attribute.extend(before.attribute)
        if probe.SerializeToString() != before.SerializeToString():
            raise ValueError(f"DCN rewrite changed node metadata or tensor edges: {before.name}")
        restored.graph.node[index].CopyFrom(before)
        records.append({"index": index, "name": before.name, "inputs": list(before.input),
                        "outputs": list(before.output), "source_plugin": NVIDIA_PLUGIN,
                        "source_version": "2", "plugin": GAC_PLUGIN, "plugin_version": "1",
                        "plugin_namespace": "", "attributes": attributes, "fields": options,
                        "known_weight_shape": _known_kernel_shape(source, before.input[3])})
    if not records:
        raise ValueError("No explicit NVIDIA-to-GAC DCN replacements found")
    if len({row["name"] for row in records}) != len(records):
        raise ValueError("DCN replacement names are ambiguous")
    if restored.SerializeToString() != source.SerializeToString():
        raise ValueError("Graph/weights/shapes/model metadata changed outside DCN plugin identity/options")
    return records


def rewrite_model(model, config):
    """Copy only explicitly selected top-level DCN nodes; preserve every edge."""
    from onnx import helper
    config = validate_config(config)
    _check_embedded(model)
    names = [node.name for node in model.graph.node]
    for name in config["node_names"]:
        if names.count(name) != 1:
            raise ValueError(f"DCN exact node name is missing or ambiguous: {name}")
        node = model.graph.node[names.index(name)]
        _check_node(node)
        _known_kernel_shape(model, node.input[3])
    selected = set(config["node_names"])
    result = deepcopy(model)
    for node in result.graph.node:
        if node.name not in selected:
            continue
        node.op_type = GAC_PLUGIN
        for attribute in node.attribute:
            if attribute.name == "plugin_version":
                attribute.s = b"1"
        node.attribute.extend(helper.make_attribute(name, config[name]) for name in OPTION_CHOICES)
    records = prove_dcn_only(model, result)
    return result, {"schema_version": 1, "kind": "gac_dcn_parser_substitution", "config": config,
                    "replaced_node_count": len(records), "records": records,
                    "preservation": {"all_other_model_protobuf_fields_identical": True,
                                     "weights_shapes_tensor_edges_identical": True},
                    "validation": {"structural_preservation": "passed", "numerical_accuracy": "not_checked",
                                   "performance": "not_measured"}}


def rewrite_for_parser(source_bytes, config):
    """In-memory API for builders that already own source/config provenance."""
    if not isinstance(source_bytes, bytes):
        raise TypeError("DCN parser input must be immutable source bytes")
    if config is None:
        return source_bytes, {"schema_version": 1, "kind": "gac_dcn_parser_substitution", "enabled": False,
                              "replaced_node_count": 0, "records": []}
    import onnx
    rewritten, audit = rewrite_model(onnx.load_model_from_string(source_bytes), config)
    audit["enabled"] = True
    return rewritten.SerializeToString(), audit


def prepare_parser_input(source, config=None):
    """Return parser bytes plus provenance; no files or calibration data are changed."""
    source = Path(source)
    original = source.read_bytes()
    source_hash = _sha(original)
    if config is None:
        return original, {"schema_version": 1, "kind": "gac_dcn_parser_substitution", "enabled": False,
                          "source_onnx_sha256": source_hash, "transformed_onnx_sha256": source_hash,
                          "config_sha256": None, "replaced_node_count": 0, "records": []}
    config_path = Path(config)
    config_bytes = config_path.read_bytes()
    selected = _decode_config(config_bytes)
    payload, audit = rewrite_for_parser(original, selected)
    audit.update(enabled=True, source_onnx=str(source.resolve()), source_onnx_sha256=source_hash,
                 transformed_onnx_sha256=_sha(payload), config_path=str(config_path.resolve()),
                 config_sha256=_sha(config_bytes),
                 frozen_calibration={"source_provenance_must_be_validated_by_builder": True,
                     "range_values_and_tensor_bindings_unchanged": True, "recalibrated": False,
                     "note": "Reuse original source-bound ranges for this controlled experiment; "
                             "this is not a new calibration or a numerical/precision acceptance."})
    return payload, audit
