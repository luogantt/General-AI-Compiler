"""Adapt GAC's exported BEVFormer custom ops to TensorRT 10 plugin ABIs.

This is a graph-format adapter, not an inference implementation. Standard ONNX
nodes, model inputs/outputs and weight values are retained. Conv weight/bias
Identity aliases of immutable initializers are replaced by direct references;
activation edges and the Identity nodes themselves are retained. This avoids
TensorRT parsing shared constant weights as dynamic convolution inputs.
The two plugin contracts
are NVIDIA TensorRT's built-in version-2 creators, registered by
``trt.init_libnvinfer_plugins(logger, "")`` before parsing the result.

MSDA needs five inputs (including INT32 level offsets) and returns a rank-four
tensor. GAC exports four inputs and a rank-three output. Consequently changing
only the operation name is insufficient: we add constants and an output reshape.
DCN uses plugin field names rather than MMCV's attribute names. Non-square
stride/padding/dilation are deliberately rejected until both the plugin's shape
inference and CUDA implementation have been validated for those configurations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


PLUGIN_DOMAIN = "trt.plugins"
PLUGIN_VERSION = "2"
MSDA_PLUGIN = "MultiscaleDeformableAttnPlugin_TRT"
DCN_PLUGIN = "ModulatedDeformConv2d"
INT32_MAX = np.iinfo(np.int32).max


class AdaptationError(ValueError):
    """The exported operation cannot be represented by the checked plugin ABI."""


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    result = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if len(result) != len(node.attribute):
        raise AdaptationError(f"{node.name or node.op_type}: duplicate attributes")
    return result


def _integer(value: Any, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AdaptationError(f"{description} must be an integer")
    value = int(value)
    if not minimum <= value <= INT32_MAX:
        raise AdaptationError(f"{description} must be in [{minimum}, {INT32_MAX}]")
    return value


def _constant_tensors(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Constant":
            continue
        attrs = _attributes(node)
        if len(node.output) != 1:
            continue
        if "value" in attrs:
            constants[node.output[0]] = numpy_helper.to_array(attrs["value"])
        elif "value_ints" in attrs:
            constants[node.output[0]] = np.asarray(attrs["value_ints"], dtype=np.int64)
    return constants


def _tensor_shapes(graph: onnx.GraphProto) -> dict[str, tuple[int | str | None, ...]]:
    shapes = {}
    for value in (*graph.input, *graph.output, *graph.value_info):
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            shapes[value.name] = tuple(
                int(d.dim_value) if d.HasField("dim_value") else d.dim_param or None
                for d in tensor.shape.dim
            )
    for tensor in graph.initializer:
        shapes[tensor.name] = tuple(tensor.dims)
    return shapes


def _canonicalize_conv_weight_aliases(graph: onnx.GraphProto) -> dict[str, Any]:
    """Bypass only proven constant Identity chains on Conv weight/bias edges.

    An initializer also listed as a graph input is overridable, so it is not a
    build-time constant. No tensor values are evaluated or copied, and unused
    Identity nodes remain available to their other consumers.
    """
    initializers = {tensor.name for tensor in graph.initializer}
    overridable = {value.name for value in graph.input}
    producers = {output: node for node in graph.node for output in node.output}
    aliases = []
    rewritten_conv_count = 0
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Conv":
            continue
        rewritten = False
        for index, role in ((1, "weight"), (2, "bias")):
            if index >= len(node.input) or not node.input[index]:
                continue
            original = current = node.input[index]
            chain = []
            visited = set()
            while current not in initializers and current not in visited:
                visited.add(current)
                producer = producers.get(current)
                if (producer is None or producer.domain not in ("", "ai.onnx")
                        or producer.op_type != "Identity" or len(producer.input) != 1
                        or len(producer.output) != 1 or producer.attribute):
                    break
                chain.append(producer.name or producer.output[0])
                current = producer.input[0]
            if chain and current in initializers and current not in overridable:
                node.input[index] = current
                rewritten = True
                aliases.append({
                    "node": node.name, "input_index": index, "role": role,
                    "previous_tensor": original, "initializer": current,
                    "identity_chain": chain,
                })
        rewritten_conv_count += int(rewritten)
    return {
        "rewritten_input_count": len(aliases),
        "rewritten_conv_count": rewritten_conv_count,
        "aliases": aliases,
    }


class _Adapter:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.graph = model.graph
        self.constants = _constant_tensors(self.graph)
        self.shapes = _tensor_shapes(self.graph)
        self.used_names = {
            name for node in self.graph.node for name in (*node.input, *node.output)
        } | {t.name for t in self.graph.initializer}
        self.nodes: list[onnx.NodeProto] = []
        self.records: list[dict[str, Any]] = []

    def name(self, base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in self.used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self.used_names.add(candidate)
        return candidate

    def constant(self, base: str, array: np.ndarray) -> str:
        name = self.name(base)
        self.graph.initializer.append(numpy_helper.from_array(array, name))
        return name

    def emit(self, op: str, inputs: list[str], output: str, **attrs: Any) -> None:
        self.nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))

    def check_shape(self, name: str, rank: int) -> tuple[int | str | None, ...] | None:
        shape = self.shapes.get(name)
        if shape is not None and len(shape) != rank:
            raise AdaptationError(f"{name}: expected rank {rank}, found {shape}")
        return shape

    @staticmethod
    def check_equal(left: Any, right: Any, description: str) -> None:
        # Different symbolic names do not prove unequal runtime dimensions.
        if isinstance(left, int) and isinstance(right, int) and left != right:
            raise AdaptationError(f"{description}: {left} != {right}")

    def msda(self, node: onnx.NodeProto) -> None:
        if len(node.input) != 4 or len(node.output) != 1 or any(not x for x in node.input):
            raise AdaptationError(f"{node.name}: MSDeformAttn requires four inputs and one output")
        attrs = _attributes(node)
        expected = {"align_corners": 0, "mode": b"bilinear", "padding_mode": b"zeros", "version": 1}
        unknown = attrs.keys() - expected.keys()
        if unknown:
            raise AdaptationError(f"{node.name}: unsupported MSDA attributes {sorted(unknown)}")
        for key, value in attrs.items():
            if value != expected[key]:
                raise AdaptationError(f"{node.name}: MSDA {key}={value!r} is unsupported")

        value, spatial_name, sampling, attention = node.input
        spatial = self.constants.get(spatial_name)
        if spatial is None:
            raise AdaptationError(f"{node.name}: spatial_shapes must be an initializer or Constant")
        if spatial.ndim != 2 or spatial.shape[1] != 2 or spatial.shape[0] == 0:
            raise AdaptationError(f"{node.name}: spatial_shapes must be nonempty [levels, 2]")
        if spatial.dtype.kind not in "iu":
            raise AdaptationError(f"{node.name}: spatial_shapes must have integer elements")
        levels = [[_integer(x, "spatial_shapes dimension", 1) for x in pair] for pair in spatial]
        starts: list[int] = []
        total = 0
        for height, width in levels:
            starts.append(total)
            total += height * width
            if total > INT32_MAX:
                raise AdaptationError(f"{node.name}: spatial_shapes total exceeds INT32 indexing")

        value_shape = self.check_shape(value, 4)
        location_shape = self.check_shape(sampling, 6)
        weight_shape = self.check_shape(attention, 5)
        output_shape = self.check_shape(node.output[0], 3)
        if value_shape:
            self.check_equal(value_shape[1], total, "MSDA value sequence/spatial_shapes sum")
        if location_shape:
            self.check_equal(location_shape[3], len(levels), "MSDA sampling level count")
            self.check_equal(location_shape[5], 2, "MSDA sampling coordinate count")
        if weight_shape:
            self.check_equal(weight_shape[3], len(levels), "MSDA weight level count")
        if value_shape and location_shape:
            for vi, si, description in [(0, 0, "batch"), (2, 2, "head count")]:
                self.check_equal(value_shape[vi], location_shape[si], f"MSDA {description}")
        if location_shape and weight_shape:
            for axis in range(5):
                self.check_equal(location_shape[axis], weight_shape[axis], f"MSDA attention axis {axis}")
        if output_shape and value_shape and location_shape:
            self.check_equal(output_shape[0], value_shape[0], "MSDA output batch")
            self.check_equal(output_shape[1], location_shape[1], "MSDA output queries")
            if all(isinstance(d, int) for d in value_shape[2:]):
                self.check_equal(output_shape[2], value_shape[2] * value_shape[3], "MSDA output channels")

        prefix = (node.name or node.output[0]) + "__trt"
        shapes_i32 = self.constant(prefix + "_spatial_shapes", np.asarray(levels, dtype=np.int32))
        starts_i32 = self.constant(prefix + "_level_start_index", np.asarray(starts, dtype=np.int32))
        plugin_output = self.name(prefix + "_rank4")
        self.nodes.append(helper.make_node(
            MSDA_PLUGIN, [value, shapes_i32, starts_i32, sampling, attention], [plugin_output],
            name=node.name, domain=PLUGIN_DOMAIN,
            plugin_version=PLUGIN_VERSION, plugin_namespace="",
        ))

        if (value_shape and location_shape
                and all(isinstance(d, int) for d in (value_shape[0], value_shape[2], value_shape[3], location_shape[1]))):
            shape = [value_shape[0], location_shape[1], value_shape[2] * value_shape[3]]
            reshape_shape = self.constant(prefix + "_output_shape", np.asarray(shape, dtype=np.int64))
            reshape_kind = "static"
        else:
            # Build [B, Q, H*D] from the plugin's [B,Q,H,D] output. Unlike
            # [0,0,-1], this also has unambiguous semantics for zero dimensions.
            rank4_shape = self.name(prefix + "_shape4")
            self.emit("Shape", [plugin_output], rank4_shape)
            dimensions = []
            for label, axes in [("bq", [0, 1]), ("heads", [2]), ("channels", [3])]:
                indices = self.constant(prefix + "_" + label + "_indices", np.asarray(axes, dtype=np.int64))
                dim = self.name(prefix + "_" + label)
                self.emit("Gather", [rank4_shape, indices], dim, axis=0)
                dimensions.append(dim)
            flattened = self.name(prefix + "_flattened_channels")
            self.emit("Mul", dimensions[1:], flattened)
            reshape_shape = self.name(prefix + "_output_shape")
            self.emit("Concat", [dimensions[0], flattened], reshape_shape, axis=0)
            reshape_kind = "dynamic"
        # allowzero=1 uses the explicitly calculated dimensions literally.
        self.emit("Reshape", [plugin_output, reshape_shape], node.output[0], allowzero=1)
        self.records.append({
            "node": node.name, "source": "gac::MSDeformAttn", "plugin": MSDA_PLUGIN,
            "plugin_version": PLUGIN_VERSION, "spatial_shapes": levels,
            "level_start_index": starts, "spatial_elements": total,
            "output_reshape": reshape_kind, "preserved_output": node.output[0],
        })

    def dcn(self, node: onnx.NodeProto) -> None:
        inputs = list(node.input)
        if len(inputs) == 5 and inputs[-1] == "":
            inputs.pop()
        if len(inputs) not in (4, 5) or len(node.output) != 1 or any(not x for x in inputs):
            raise AdaptationError(f"{node.name}: DCN requires four/five inputs and one output")
        attrs = _attributes(node)
        required = {"groups", "deform_groups", "stride", "padding", "dilation"}
        if attrs.keys() != required:
            raise AdaptationError(
                f"{node.name}: DCN requires exactly {sorted(required)}; "
                f"missing={sorted(required - attrs.keys())}, unsupported={sorted(attrs.keys() - required)}"
            )
        fields = {
            "group": _integer(attrs["groups"], "DCN groups", 1),
            "deformable_group": _integer(attrs["deform_groups"], "DCN deform_groups", 1),
        }
        for key in ("stride", "padding", "dilation"):
            pair = attrs[key]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise AdaptationError(f"{node.name}: DCN {key} must be [height, width]")
            height, width = [_integer(x, f"DCN {key}", 0 if key == "padding" else 1) for x in pair]
            if height != width:
                raise AdaptationError(
                    f"{node.name}: asymmetric DCN {key}={pair} is not yet validated against "
                    "both TensorRT plugin shape inference and CUDA enqueue; refusing conversion"
                )
            # MMCV stores [H,W]; plugin enqueue passes the fields as [W,H].
            fields[key] = [width, height]
        self.nodes.append(helper.make_node(
            DCN_PLUGIN, inputs, list(node.output), name=node.name, domain=PLUGIN_DOMAIN,
            plugin_version=PLUGIN_VERSION, plugin_namespace="", **fields,
        ))
        self.records.append({
            "node": node.name, "source": "mmcv::MMCVModulatedDeformConv2d", "plugin": DCN_PLUGIN,
            "plugin_version": PLUGIN_VERSION, "fields": fields, "has_bias": len(inputs) == 5,
            "preserved_output": node.output[0],
        })


def adapt_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, dict[str, Any]]:
    """Return an independent adapted model and a JSON-serializable audit report.

    Only static MSDA level sizes and symmetric DCN spatial attributes are
    accepted. Dynamic batch/query/channel shapes are retained for MSDA. Standard
    Conv weight/bias aliases of immutable initializers are canonicalized.
    """
    opsets = {o.domain: o.version for o in model.opset_import}
    if opsets.get("", opsets.get("ai.onnx", 0)) < 14:
        raise AdaptationError("ONNX opset >=14 is required for Reshape allowzero semantics")
    result = deepcopy(model)
    conv_weight_aliases = _canonicalize_conv_weight_aliases(result.graph)
    adapter = _Adapter(result)
    for node in list(result.graph.node):
        if (node.domain, node.op_type) == ("gac", "MSDeformAttn"):
            adapter.msda(node)
        elif (node.domain, node.op_type) == ("mmcv", "MMCVModulatedDeformConv2d"):
            adapter.dcn(node)
        elif node.domain in ("gac", "mmcv"):
            raise AdaptationError(f"{node.name}: unsupported custom operation {node.domain}::{node.op_type}")
        else:
            adapter.nodes.append(node)
    del result.graph.node[:]
    result.graph.node.extend(adapter.nodes)
    if adapter.records and PLUGIN_DOMAIN not in opsets:
        result.opset_import.append(helper.make_opsetid(PLUGIN_DOMAIN, 1))
    onnx.checker.check_model(result)
    report = {
        "plugin_domain": PLUGIN_DOMAIN, "plugin_version": PLUGIN_VERSION,
        "converted": dict(Counter(record["source"] for record in adapter.records)),
        "source_node_count": len(model.graph.node), "adapted_node_count": len(result.graph.node),
        "inputs": [v.name for v in model.graph.input], "outputs": [v.name for v in model.graph.output],
        "nodes": adapter.records,
        "static_conv_weight_aliases": conv_weight_aliases,
        "limitations": ["MSDA spatial_shapes must be constant", "asymmetric DCN attributes are rejected"],
    }
    return result, report


def adapt_file(source: str | Path, destination: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    source, destination = Path(source), Path(destination)
    report_path = Path(report_path) if report_path else destination.with_suffix(".report.json")
    if source.resolve() == destination.resolve():
        raise AdaptationError("Use a separate output path to preserve the exported source model")
    if report_path.resolve() in (source.resolve(), destination.resolve()):
        raise AdaptationError("Report must have a distinct path from both ONNX files")
    if destination.exists() or report_path.exists():
        raise AdaptationError("Output model/report already exists; choose new paths")
    # Load external tensor data before relocation so the output never contains
    # stale references relative to the old model directory.
    model, report = adapt_model(onnx.load(source, load_external_data=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, destination)
    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                result.update(chunk)
        return result.hexdigest()

    report.update({"source": str(source), "destination": str(destination),
                   "source_onnx_sha256": digest(source), "destination_onnx_sha256": digest(destination),
                   "hash_scope": "ONNX files only; source external weight files are not included"})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = adapt_file(args.source, args.destination, args.report)
    except (AdaptationError, onnx.checker.ValidationError) as error:
        parser.exit(2, f"adaptation failed: {error}\n")
    summary = {k: v for k, v in report.items() if k != "nodes"}
    summary["static_conv_weight_aliases"] = {
        k: v for k, v in report["static_conv_weight_aliases"].items() if k != "aliases"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
