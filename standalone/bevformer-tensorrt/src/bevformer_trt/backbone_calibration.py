#!/usr/bin/env python3
"""Extract an exact six-camera backbone for a separate, lower-memory calibration.

The source must be the adapted Base ONNX. Only the six final layer4 activations
become outputs; internal Conv activations are NOT all marked as outputs. Node,
weight, and tensor identities remain unchanged. This is graph extraction, not
inference, calibration, or evidence that either engine fits the target GPU.

The resulting entropy cache belongs to this subgraph. Never present it to the
full model's calibrator or change its model hash to bypass cache validation.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import onnx
from onnx import AttributeProto, TensorProto, helper

from .adapt_onnx import DCN_PLUGIN, PLUGIN_DOMAIN
from .precision_policy import BACKBONE, OFFSET_PREDICTOR, classify_scope, policy_fingerprint


BASE_INPUT_SHAPE = (1, 6, 3, 928, 1600)
LAST_STAGE = re.compile(r"layer4(?:_\d+)?(?:[/.]|$)")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _description(value):
    tensor = value.type.tensor_type
    return {
        "name": value.name,
        "dtype": str(helper.tensor_dtype_to_np_dtype(tensor.elem_type)),
        "shape": [int(d.dim_value) if d.HasField("dim_value") else d.dim_param or None
                  for d in tensor.shape.dim],
    }


def _is_int8_conv(node):
    # Exactly the ordinary-Conv name predicate used by precision_policy.py.
    return (node.domain in ("", "ai.onnx") and node.op_type == "Conv"
            and classify_scope(node.name) == "backbone_body"
            and OFFSET_PREDICTOR.search(node.name) is None)


def _index_graph(graph):
    sources = {value.name: ("input", i) for i, value in enumerate(graph.input)}
    if len(sources) != len(graph.input):
        raise ValueError("Duplicate graph input names")
    for i, weight in enumerate(graph.initializer):
        if weight.name in sources:
            raise ValueError(f"Duplicate tensor producer: {weight.name}")
        if weight.data_location == TensorProto.EXTERNAL:
            raise ValueError("Extraction requires embedded weights, not external data")
        sources[weight.name] = ("weight", i)
    consumers = defaultdict(list)
    for i, node in enumerate(graph.node):
        for name in node.input:
            if name:
                consumers[name].append(i)
        for name in node.output:
            if name:
                if name in sources:
                    raise ValueError(f"Duplicate tensor producer: {name}")
                sources[name] = ("node", i)
    return sources, consumers


def _ancestors(output_names, graph, sources):
    pending, tensors, nodes, weights, inputs = list(output_names), set(), set(), set(), set()
    while pending:
        name = pending.pop()
        if not name or name in tensors:
            continue
        tensors.add(name)
        if name not in sources:
            raise ValueError(f"Unresolved ancestor tensor: {name}")
        kind, index = sources[name]
        if kind == "input":
            inputs.add(index)
        elif kind == "weight":
            weights.add(index)
        elif index not in nodes:
            node = graph.node[index]
            if any(a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS) for a in node.attribute):
                raise ValueError(f"Nested control flow is unsupported in backbone extraction: {node.name}")
            nodes.add(index)
            tensors.update(n for n in node.output if n)
            pending.extend(node.input)
    return tensors, nodes, weights, inputs


def _infer_backbone_shapes(model):
    """Infer metadata only; keep the actual saved DCNv2 nodes byte-identical.

    DCNv2 has the Conv output shape for the same X/W/stride/padding/dilation.
    Standard ONNX cannot infer the plugin schema, so a separate throwaway model
    uses this shape equivalence. No Conv replacement is saved or executed.
    """
    shape_model = deepcopy(model)
    for node in shape_model.graph.node:
        if node.domain in ("", "ai.onnx"):
            continue
        if (node.domain, node.op_type) != (PLUGIN_DOMAIN, DCN_PLUGIN):
            raise ValueError(f"Backbone requires an adapted DCNv2 plugin, found {node.domain}::{node.op_type}")
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if len(node.input) not in (4, 5) or len(node.output) != 1:
            raise ValueError(f"Unsupported DCNv2 ABI: {node.name}")
        for key in ("stride", "padding", "dilation"):
            value = attrs.get(key)
            if not isinstance(value, list) or len(value) != 2 or any(not isinstance(x, int) for x in value):
                raise ValueError(f"Invalid DCNv2 {key}: {node.name}")
        if attrs.get("plugin_version") != b"2" or attrs.get("group", 0) <= 0:
            raise ValueError(f"Unverified DCNv2 plugin version/group: {node.name}")
        conv = helper.make_node(
            "Conv", [node.input[0], node.input[3], *node.input[4:]], list(node.output),
            name=node.name, strides=attrs["stride"], pads=attrs["padding"] * 2,
            dilations=attrs["dilation"], group=attrs["group"],
        )
        node.CopyFrom(conv)
    inferred = onnx.shape_inference.infer_shapes(shape_model, strict_mode=True, data_prop=True)
    values = {v.name: v for v in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)}
    for weight in inferred.graph.initializer:
        values[weight.name] = helper.make_tensor_value_info(weight.name, weight.data_type, weight.dims)
    return values


def extract_model(model, *, expected_input_shape=BASE_INPUT_SHAPE, expected_output_count=6,
                  expected_int8_count=462, expected_int8_names=None):
    """Return submodel/audit; non-Base expected values exist only for CPU tests."""
    graph = model.graph
    if graph.sparse_initializer:
        raise ValueError("Sparse initializers are unsupported in this extraction path")
    if len(graph.input) != 1 or graph.input[0].name != "images":
        raise ValueError("Expected exactly one graph input named images")
    description = _description(graph.input[0])
    if description["dtype"] != "float32" or description["shape"] != list(expected_input_shape):
        raise ValueError(f"Unexpected images input contract: {description}")
    sources, consumers = _index_graph(graph)
    outputs = []
    for node in graph.node:
        scope = BACKBONE.search(node.name)
        if (scope is None or not LAST_STAGE.match(scope.group(1))
                or node.domain not in ("", "ai.onnx") or node.op_type != "Relu"):
            continue
        for name in node.output:
            users = [graph.node[i] for i in consumers[name]]
            if users and all(BACKBONE.search(user.name) is None for user in users):
                outputs.append(name)
    if len(outputs) != expected_output_count:
        raise ValueError(f"Expected {expected_output_count} final layer4 outputs, found {len(outputs)}: {outputs}")

    full_int8 = [n.name for n in graph.node if _is_int8_conv(n)]
    if len(full_int8) != len(set(full_int8)):
        raise ValueError("INT8 Conv node names must be unique")
    if len(full_int8) != expected_int8_count:
        raise ValueError(f"Expected {expected_int8_count} source INT8 Conv nodes, found {len(full_int8)}")
    if expected_int8_names is not None and set(full_int8) != set(expected_int8_names):
        raise ValueError("Source ONNX INT8 Conv names disagree with the full TensorRT precision plan")
    tensors, nodes, weights, inputs = _ancestors(outputs, graph, sources)
    if inputs != {0}:
        raise ValueError("Every calibration subgraph must retain the images input")
    retained_int8 = [n.name for i, n in enumerate(graph.node) if i in nodes and _is_int8_conv(n)]
    if set(retained_int8) != set(full_int8):
        raise ValueError(f"Extraction loses INT8 Conv nodes: {sorted(set(full_int8) - set(retained_int8))}")
    branches = []
    for output in outputs:
        branch_nodes = _ancestors([output], graph, sources)[1]
        names = [n.name for i, n in enumerate(graph.node) if i in branch_nodes and _is_int8_conv(n)]
        branches.append({"output_tensor": output, "int8_convolution_names": names})
    counts = Counter(name for branch in branches for name in branch["int8_convolution_names"])
    if any(count != 1 for count in counts.values()) or any(
            len(branch["int8_convolution_names"]) * expected_output_count != expected_int8_count for branch in branches):
        raise ValueError("Expected disjoint, equally covered camera backbone branches")

    result = deepcopy(model)
    retained_nodes = [node for i, node in enumerate(graph.node) if i in nodes]
    retained_weights = [weight for i, weight in enumerate(graph.initializer) if i in weights]
    result.graph.ClearField("node")
    result.graph.node.extend(retained_nodes)
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(retained_weights)
    result.graph.ClearField("output")
    result.graph.output.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs)
    result.graph.ClearField("value_info")
    result.graph.value_info.extend(v for v in graph.value_info if v.name in tensors and v.name not in outputs)
    result.graph.ClearField("quantization_annotation")
    result.graph.quantization_annotation.extend(a for a in graph.quantization_annotation if a.tensor_name in tensors)
    inferred = _infer_backbone_shapes(result)
    for value in result.graph.output:
        if value.name not in inferred:
            raise ValueError(f"Could not infer output shape: {value.name}")
        info = _description(inferred[value.name])
        if (info["dtype"] != "float32" or not info["shape"]
                or any(not isinstance(d, int) or d <= 0 for d in info["shape"])):
            raise ValueError(f"Final backbone shape must be static FP32: {info}")
        value.CopyFrom(inferred[value.name])
    onnx.checker.check_model(result)
    # Exact copying is the structural equivalence proof, not an accuracy claim.
    assert all(a.SerializeToString() == b.SerializeToString() for a, b in zip(retained_nodes, result.graph.node))
    assert all(a.SerializeToString() == b.SerializeToString() for a, b in zip(retained_weights, result.graph.initializer))
    tensor_map = []
    for name in sorted(tensors):
        info = _description(inferred[name]) if name in inferred else {"dtype": None, "shape": None}
        kind, index = sources[name]
        tensor_map.append({"source_name": name, "subgraph_name": name,
                           "dtype": info["dtype"], "shape": info["shape"],
                           "producer_name": graph.node[index].name if kind == "node" else kind})
    audit = {
        "schema_version": 1, "kind": "bevformer_backbone_calibration_subgraph",
        "scope": "Backbone calibration only; not a full-model cache or full-model inference result",
        "policy_rule_fingerprint_sha256": policy_fingerprint(),
        "inputs": [_description(v) for v in result.graph.input],
        "outputs": [_description(v) for v in result.graph.output],
        "source_node_count": len(graph.node), "subgraph_node_count": len(result.graph.node),
        "source_initializer_count": len(graph.initializer), "subgraph_initializer_count": len(result.graph.initializer),
        "int8_convolution_names": retained_int8, "tensor_mapping": tensor_map,
        "int8_activation_edges": [
            {"layer": node.name, "edge": edge, "tensor": tensor}
            for node in retained_nodes if _is_int8_conv(node)
            for edge, tensor in (("input0", node.input[0]), ("output0", node.output[0]))
        ],
        "coverage": {"expected_int8_convolutions": expected_int8_count,
                     "retained_int8_convolutions": len(retained_int8), "missing": [], "extra": [], "per_output": branches},
        "preservation": {"node_protobufs_identical": True, "initializer_protobufs_identical": True,
                         "input_value_info_identical": result.graph.input == graph.input, "tensor_names_identical": True},
        "shape_evidence": "ONNX inference on a throwaway shape model where DCNv2 uses equivalent Conv shape rules; saved plugins are unchanged",
        "validation": {"structural_extraction": "passed", "numerical_accuracy": "not_checked",
                       "calibration": "not_run", "full_engine": "not_built", "memory_savings": "not_measured"},
        "cache_contract": "Bind calibration cache to subgraph_onnx_sha256 and its own precision plan/data; never reuse it as a full-model cache",
    }
    return result, audit


def write_subgraph(source, output, manifest_path, precision_plan):
    source, output, manifest_path, precision_plan = map(Path, (source, output, manifest_path, precision_plan))
    paths = [p.resolve() for p in (source, output, manifest_path, precision_plan)]
    if len(set(paths)) != 4:
        raise ValueError("Model, output, manifest and precision-plan paths must differ")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Output/manifest already exists; select new paths")
    source_hash = sha256_file(source)
    plan = json.loads(precision_plan.read_text(encoding="utf-8"))
    policy = plan["precision_policy"]["mixed_layer_policy"]
    if plan["onnx_file_sha256"] != source_hash:
        raise ValueError("Precision plan is not bound to the source full ONNX")
    if policy["fingerprint_sha256"] != policy_fingerprint():
        raise ValueError("Precision plan does not match the current default mixed-INT8 policy")
    names = [row["name"] for row in policy["layers"] if row.get("compute_precision") == "int8"]
    if len(names) != len(set(names)):
        raise ValueError("Precision plan contains duplicate INT8 layer names")
    model = onnx.load(source, load_external_data=False)
    result, audit = extract_model(model, expected_int8_names=names)
    payload = result.SerializeToString()
    audit.update(source_full_onnx_sha256=source_hash,
                 subgraph_onnx_sha256=hashlib.sha256(payload).hexdigest(),
                 precision_plan_sha256=sha256_file(precision_plan),
                 source_full_policy_fingerprint=plan["policy_fingerprint"],
                 source_full_onnx=str(source.resolve()), subgraph_onnx=str(output.resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="Adapted full Base ONNX with embedded weights")
    parser.add_argument("--precision-plan", type=Path, required=True, help="Full model's mixed-int8 --plan-only report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = write_subgraph(args.onnx, args.output, args.manifest, args.precision_plan)
    print(json.dumps({"subgraph": str(args.output), "manifest": str(args.manifest),
                      "outputs": audit["outputs"], "int8_convolutions": len(audit["int8_convolution_names"]),
                      "scope": audit["scope"]}, indent=2))


if __name__ == "__main__":
    main()
