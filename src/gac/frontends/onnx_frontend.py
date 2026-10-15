"""ONNX to GAC Canonical Graph import.

The frontend keeps ONNX as an optional dependency and translates framework
semantics into the small, device-independent IR in :mod:`gac.ir`.  Operators
whose semantics do not yet have a faithful GAC representation are reported
together instead of being silently approximated.
"""

from __future__ import annotations

from collections import Counter
import importlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass


class OnnxFrontendError(ValueError):
    """Raised when an ONNX model cannot be represented faithfully."""


_DIRECT_OPS = {
    "MatMul": "MatMul",
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Relu": "ReLU",
    "Identity": "Identity",
    "Sigmoid": "Sigmoid",
    "Log": "Log",
    "Sin": "Sin",
    "Cos": "Cos",
    "Transpose": "Transpose",
    "Concat": "Concat",
    "Softmax": "Softmax",
}


def _onnx_module() -> Any:
    try:
        onnx = importlib.import_module("onnx")
    except (ImportError, OSError) as exc:
        raise OnnxFrontendError(
            "ONNX import requires the optional dependency; install "
            "gac-heterogeneous[onnx]"
        ) from exc
    match = re.match(r"^(\d+)\.(\d+)", str(getattr(onnx, "__version__", "")))
    if match is None or tuple(int(item) for item in match.groups()) < (1, 21):
        raise OnnxFrontendError(
            "ONNX import requires onnx>=1.21 because older releases have known "
            "external-data parser vulnerabilities"
        )
    return onnx


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_decode(item) for item in value]
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _unique(base: str, occupied: set[str]) -> str:
    candidate = base or "value"
    if candidate not in occupied:
        occupied.add(candidate)
        return candidate
    index = 1
    while f"{candidate}_{index}" in occupied:
        index += 1
    result = f"{candidate}_{index}"
    occupied.add(result)
    return result


def _tensor_spec(value_info: Any, onnx: Any) -> TensorSpec:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.elem_type:
        raise OnnxFrontendError(f"tensor {value_info.name!r} has no element type")
    try:
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).name
    except (KeyError, TypeError, ValueError) as exc:
        raise OnnxFrontendError(
            f"tensor {value_info.name!r} uses an unsupported ONNX dtype "
            f"{tensor_type.elem_type}"
        ) from exc
    if dtype not in {
        "float16",
        "float32",
        "float64",
        "int8",
        "int32",
        "int64",
        "bool",
    }:
        raise OnnxFrontendError(
            f"tensor {value_info.name!r} uses dtype {dtype!r}, which the current "
            "GAC Canonical Graph import contract does not support"
        )
    if not tensor_type.HasField("shape"):
        raise OnnxFrontendError(
            f"tensor {value_info.name!r} has unknown rank; GAC requires a known rank"
        )
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            shape.append(str(dimension.dim_param))
        else:
            shape.append(None)
    return TensorSpec(tuple(shape), dtype)


def _attributes(proto: Any, onnx: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in proto.attribute:
        value = onnx.helper.get_attribute_value(attribute)
        if hasattr(value, "node") and hasattr(value, "initializer"):
            raise OnnxFrontendError(
                f"node {proto.name or proto.op_type!r} contains a nested graph; "
                "ONNX control flow is not supported yet"
            )
        if hasattr(value, "dims") and hasattr(value, "data_type"):
            result[str(attribute.name)] = np.ascontiguousarray(
                onnx.numpy_helper.to_array(value)
            )
        else:
            result[str(attribute.name)] = _decode(value)
    return result


def _constant_vector(
    name: str,
    constants: Mapping[str, np.ndarray[Any, Any]],
    *,
    node: str,
) -> list[int]:
    if name not in constants:
        raise OnnxFrontendError(
            f"node {node!r} requires compile-time constant input {name!r}"
        )
    value = np.asarray(constants[name])
    if value.ndim > 1 or not np.issubdtype(value.dtype, np.integer):
        raise OnnxFrontendError(
            f"node {node!r} constant input {name!r} must be an integer scalar or vector"
        )
    return [int(item) for item in value.reshape(-1)]


def _constant_scalar(
    name: str,
    constants: Mapping[str, np.ndarray[Any, Any]],
    *,
    node: str,
) -> int | float | None:
    if not name:
        return None
    if name not in constants:
        raise OnnxFrontendError(
            f"node {node!r} requires compile-time constant input {name!r}"
        )
    value = np.asarray(constants[name])
    if value.size != 1 or not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise OnnxFrontendError(
            f"node {node!r} constant input {name!r} must be one numeric scalar"
        )
    return value.reshape(()).item()


def _constant_bool(
    name: str,
    constants: Mapping[str, np.ndarray[Any, Any]],
    *,
    node: str,
) -> bool:
    if name not in constants:
        raise OnnxFrontendError(
            f"node {node!r} requires compile-time constant input {name!r}"
        )
    value = np.asarray(constants[name])
    if value.size != 1 or value.dtype != np.dtype("bool"):
        raise OnnxFrontendError(
            f"node {node!r} constant input {name!r} must be one boolean scalar"
        )
    return bool(value.reshape(()).item())


def _pairs(value: Any, default: Sequence[int]) -> list[int]:
    if value is None:
        return [int(item) for item in default]
    return [int(item) for item in value]


def _require_default_auto_pad(attrs: Mapping[str, Any], node_id: str) -> None:
    auto_pad = str(attrs.get("auto_pad", "NOTSET"))
    if auto_pad not in {"NOTSET", ""}:
        raise OnnxFrontendError(
            f"node {node_id!r} auto_pad={auto_pad!r} must be legalized to explicit pads"
        )


def _unsupported_name(proto: Any) -> str:
    domain = str(proto.domain or "ai.onnx")
    return f"{domain}::{proto.op_type}"


def _translate_node(
    proto: Any,
    *,
    index: int,
    constants: dict[str, np.ndarray[Any, Any]],
    value_specs: Mapping[str, TensorSpec],
    occupied_values: set[str],
    occupied_ids: set[str],
    onnx_opset: int,
    onnx: Any,
) -> tuple[list[Node], str | None]:
    source_id = str(proto.name or f"{proto.op_type}_{index}")
    node_id = _unique(source_id, occupied_ids)
    # ONNX uses an empty string to preserve the position of an omitted optional
    # input.  Keep those slots for operator-specific decoding; filtering first
    # would turn Clip(data, "", max) into Clip(data, min), for example.
    raw_inputs = tuple(str(name) for name in proto.input)
    inputs = tuple(name for name in raw_inputs if name)
    outputs = tuple(str(name) for name in proto.output if name)
    attrs = _attributes(proto, onnx)
    domain = str(proto.domain or "")

    if domain not in {"", "ai.onnx"}:
        return (
            [Node(node_id, _unsupported_name(proto), inputs, outputs, attrs, (source_id,))],
            _unsupported_name(proto),
        )
    if not outputs:
        raise OnnxFrontendError(f"ONNX node {source_id!r} has no named output")

    if proto.op_type == "Constant":
        if len(outputs) != 1:
            raise OnnxFrontendError(f"Constant node {source_id!r} must have one output")
        if "value" in attrs:
            value = np.asarray(attrs["value"])
        elif "value_float" in attrs:
            value = np.asarray(attrs["value_float"], dtype=np.float32)
        elif "value_int" in attrs:
            value = np.asarray(attrs["value_int"], dtype=np.int64)
        elif "value_floats" in attrs:
            value = np.asarray(attrs["value_floats"], dtype=np.float32)
        elif "value_ints" in attrs:
            value = np.asarray(attrs["value_ints"], dtype=np.int64)
        else:
            raise OnnxFrontendError(
                f"Constant node {source_id!r} uses an unsupported value encoding"
            )
        constants[outputs[0]] = np.ascontiguousarray(value)
        occupied_values.add(outputs[0])
        return [], None

    direct = _DIRECT_OPS.get(str(proto.op_type))
    if direct is not None:
        if len(outputs) != 1:
            raise OnnxFrontendError(
                f"node {source_id!r} has {len(outputs)} outputs; canonical {direct} "
                "currently requires one"
            )
        canonical_attrs: dict[str, Any] = {}
        if proto.op_type in {"Add", "Sub", "Mul", "Div"} and onnx_opset <= 6:
            if int(attrs.get("broadcast", 0)) != 0 or "axis" in attrs:
                raise OnnxFrontendError(
                    f"node {source_id!r} uses legacy ONNX axis-aligned broadcasting; "
                    "convert the model to opset 7 or newer before import"
                )
            operand_specs = [value_specs.get(name) for name in inputs]
            if (
                len(operand_specs) != 2
                or any(spec is None for spec in operand_specs)
                or operand_specs[0].shape != operand_specs[1].shape
            ):
                raise OnnxFrontendError(
                    f"node {source_id!r} in ONNX opset {onnx_opset} requires equal "
                    "operand shapes when broadcast=0"
                )
        if direct == "Transpose":
            if "perm" in attrs:
                canonical_attrs["perm"] = [int(axis) for axis in attrs["perm"]]
            else:
                input_spec = value_specs.get(inputs[0]) if inputs else None
                if input_spec is None:
                    raise OnnxFrontendError(
                        f"Transpose node {source_id!r} needs an inferred input rank "
                        "when perm is omitted"
                    )
                canonical_attrs["perm"] = list(reversed(range(input_spec.rank)))
        elif direct == "Concat":
            if "axis" not in attrs:
                raise OnnxFrontendError(f"Concat node {source_id!r} has no axis")
            canonical_attrs["axis"] = int(attrs["axis"])
        elif direct == "Softmax":
            axis = int(attrs.get("axis", -1 if onnx_opset >= 13 else 1))
            if onnx_opset < 13:
                input_spec = value_specs.get(inputs[0]) if inputs else None
                if input_spec is None:
                    raise OnnxFrontendError(
                        f"Softmax node {source_id!r} needs an inferred input rank for "
                        f"ONNX opset {onnx_opset} semantics"
                    )
                normalized_axis = axis + input_spec.rank if axis < 0 else axis
                if normalized_axis != input_spec.rank - 1:
                    raise OnnxFrontendError(
                        f"Softmax node {source_id!r} uses pre-opset-13 flatten-from-axis "
                        f"semantics at axis {axis}; canonical Softmax only reduces one axis"
                    )
            canonical_attrs["axis"] = axis
        return [Node(node_id, direct, inputs, outputs, canonical_attrs, (source_id,))], None

    if proto.op_type == "Gelu":
        approximation = str(attrs.get("approximate", "none")).lower()
        if approximation not in {"tanh"}:
            raise OnnxFrontendError(
                f"Gelu node {source_id!r} uses approximation={approximation!r}; "
                "the current GAC GELU contract is tanh"
            )
        return [Node(node_id, "GELU", inputs, outputs, {"approximation": "tanh"}, (source_id,))], None

    if proto.op_type == "LayerNormalization":
        if len(outputs) != 1:
            raise OnnxFrontendError(
                f"LayerNormalization node {source_id!r} exposes training/statistic outputs"
            )
        input_spec = value_specs.get(inputs[0]) if inputs else None
        if input_spec is None or input_spec.dtype != "float32":
            raise OnnxFrontendError(
                f"LayerNormalization node {source_id!r} requires float32 input until "
                "the Canonical LayerNorm records ONNX stash/accumulator precision"
            )
        if int(attrs.get("stash_type", 1)) != 1:
            raise OnnxFrontendError(
                f"LayerNormalization node {source_id!r} uses unsupported "
                f"stash_type={attrs['stash_type']!r}"
            )
        canonical = {
            "axis": int(attrs.get("axis", -1)),
            "epsilon": float(attrs.get("epsilon", 1e-5)),
        }
        return [Node(node_id, "LayerNorm", inputs, outputs, canonical, (source_id,))], None

    if proto.op_type == "Cast":
        if "to" not in attrs:
            raise OnnxFrontendError(f"Cast node {source_id!r} has no target dtype")
        try:
            dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(int(attrs["to"]))).name
        except (KeyError, TypeError, ValueError) as exc:
            raise OnnxFrontendError(
                f"Cast node {source_id!r} has unsupported dtype {attrs['to']!r}"
            ) from exc
        return [Node(node_id, "Cast", inputs, outputs, {"to": dtype}, (source_id,))], None

    if proto.op_type == "Reshape":
        if len(inputs) != 2:
            raise OnnxFrontendError(f"Reshape node {source_id!r} needs data and shape")
        if int(attrs.get("allowzero", 0)) != 0:
            raise OnnxFrontendError(
                f"Reshape node {source_id!r} uses allowzero=1, which GAC does not support"
            )
        shape = _constant_vector(inputs[1], constants, node=source_id)
        if 0 in shape:
            input_spec = value_specs.get(inputs[0])
            if input_spec is None:
                raise OnnxFrontendError(
                    f"Reshape node {source_id!r} needs input shape information to resolve zero dimensions"
                )
            resolved_shape: list[int | str | None] = []
            for axis, dim in enumerate(shape):
                if dim != 0:
                    resolved_shape.append(dim)
                    continue
                if axis >= input_spec.rank:
                    raise OnnxFrontendError(
                        f"Reshape node {source_id!r} zero dimension at position {axis} "
                        "has no corresponding input dimension"
                    )
                resolved_shape.append(input_spec.shape[axis])
            shape = resolved_shape
            if not all(isinstance(dim, int) for dim in shape):
                raise OnnxFrontendError(
                    f"Reshape node {source_id!r} has a zero dimension referring to a dynamic input"
                )
        return [Node(node_id, "Reshape", (inputs[0],), outputs, {"shape": shape}, (source_id,))], None

    if proto.op_type == "Expand":
        if len(inputs) != 2:
            raise OnnxFrontendError(f"Expand node {source_id!r} needs data and shape")
        shape = _constant_vector(inputs[1], constants, node=source_id)
        return [Node(node_id, "Expand", (inputs[0],), outputs, {"shape": shape}, (source_id,))], None

    if proto.op_type == "Slice":
        if len(raw_inputs) < 3 or not raw_inputs[0] or not raw_inputs[1] or not raw_inputs[2]:
            raise OnnxFrontendError(f"Slice node {source_id!r} needs starts and ends")
        canonical = {
            "starts": _constant_vector(raw_inputs[1], constants, node=source_id),
            "ends": _constant_vector(raw_inputs[2], constants, node=source_id),
        }
        if len(raw_inputs) >= 4 and raw_inputs[3]:
            canonical["axes"] = _constant_vector(raw_inputs[3], constants, node=source_id)
        if len(raw_inputs) >= 5 and raw_inputs[4]:
            canonical["steps"] = _constant_vector(raw_inputs[4], constants, node=source_id)
        return [Node(node_id, "Slice", (raw_inputs[0],), outputs, canonical, (source_id,))], None

    if proto.op_type in {"ReduceSum", "ReduceMean"}:
        canonical = {"keepdims": bool(int(attrs.get("keepdims", 1)))}
        if len(raw_inputs) >= 2 and raw_inputs[1]:
            axes = _constant_vector(raw_inputs[1], constants, node=source_id)
        else:
            axes = [int(axis) for axis in attrs.get("axes", [])]
        if axes:
            canonical["axes"] = axes
        if int(attrs.get("noop_with_empty_axes", 0)) != 0:
            raise OnnxFrontendError(
                f"node {source_id!r} uses noop_with_empty_axes=1"
            )
        op = "ReduceSum" if proto.op_type == "ReduceSum" else "ReduceMean"
        return [Node(node_id, op, (inputs[0],), outputs, canonical, (source_id,))], None

    if proto.op_type == "Clip":
        minimum = (
            _constant_scalar(raw_inputs[1], constants, node=source_id)
            if len(raw_inputs) >= 2 and raw_inputs[1]
            else attrs.get("min")
        )
        maximum = (
            _constant_scalar(raw_inputs[2], constants, node=source_id)
            if len(raw_inputs) >= 3 and raw_inputs[2]
            else attrs.get("max")
        )
        canonical = {name: value for name, value in (("min", minimum), ("max", maximum)) if value is not None}
        return [Node(node_id, "Clamp", (inputs[0],), outputs, canonical, (source_id,))], None

    if proto.op_type == "Conv":
        _require_default_auto_pad(attrs, source_id)
        kernel = attrs.get("kernel_shape")
        if kernel is not None and len(kernel) != 2:
            raise OnnxFrontendError(f"Conv node {source_id!r} is not two-dimensional")
        canonical = {
            "strides": _pairs(attrs.get("strides"), (1, 1)),
            "dilations": _pairs(attrs.get("dilations"), (1, 1)),
            "pads": _pairs(attrs.get("pads"), (0, 0, 0, 0)),
            "groups": int(attrs.get("group", 1)),
        }
        return [Node(node_id, "Conv2D", inputs, outputs, canonical, (source_id,))], None

    if proto.op_type == "BatchNormalization":
        legacy_training = onnx_opset <= 6 and int(attrs.get("is_test", 0)) == 0
        modern_training = int(attrs.get("training_mode", 0)) != 0
        legacy_nonspatial = onnx_opset <= 7 and int(attrs.get("spatial", 1)) != 1
        if len(outputs) != 1 or legacy_training or modern_training or legacy_nonspatial:
            raise OnnxFrontendError(
                f"BatchNormalization node {source_id!r} must be inference-only with one output"
            )
        return [
            Node(
                node_id,
                "BatchNorm",
                inputs,
                outputs,
                {"epsilon": float(attrs.get("epsilon", 1e-5))},
                (source_id,),
            )
        ], None

    if proto.op_type == "MaxPool":
        _require_default_auto_pad(attrs, source_id)
        if len(outputs) != 1:
            raise OnnxFrontendError(
                f"MaxPool node {source_id!r} index output is not supported"
            )
        kernel = _pairs(attrs.get("kernel_shape"), ())
        if len(kernel) != 2:
            raise OnnxFrontendError(f"MaxPool node {source_id!r} is not two-dimensional")
        canonical = {
            "kernel_shape": kernel,
            "strides": _pairs(attrs.get("strides"), (1, 1)),
            "dilations": _pairs(attrs.get("dilations"), (1, 1)),
            "pads": _pairs(attrs.get("pads"), (0, 0, 0, 0)),
            "ceil_mode": bool(int(attrs.get("ceil_mode", 0))),
        }
        if int(attrs.get("storage_order", 0)) != 0:
            raise OnnxFrontendError(f"MaxPool node {source_id!r} uses storage_order=1")
        return [Node(node_id, "MaxPool2D", inputs, outputs, canonical, (source_id,))], None

    if proto.op_type == "GridSample":
        source_mode = str(attrs.get("mode", "linear"))
        canonical = {
            "mode": (
                "bilinear" if source_mode in {"linear", "bilinear"} else source_mode
            ),
            "padding_mode": str(attrs.get("padding_mode", "zeros")),
            "align_corners": bool(int(attrs.get("align_corners", 0))),
        }
        return [Node(node_id, "GridSample2D", inputs, outputs, canonical, (source_id,))], None

    if proto.op_type == "Dropout":
        if len(outputs) != 1:
            raise OnnxFrontendError(
                f"Dropout node {source_id!r} mask output is not supported"
            )
        if onnx_opset <= 6 and int(attrs.get("is_test", 0)) == 0:
            raise OnnxFrontendError(
                f"Dropout node {source_id!r} uses training semantics in ONNX "
                f"opset {onnx_opset}"
            )
        if onnx_opset >= 12 and len(raw_inputs) >= 3 and raw_inputs[2]:
            training = _constant_bool(raw_inputs[2], constants, node=source_id)
            if training:
                raise OnnxFrontendError(
                    f"Dropout node {source_id!r} is in training mode"
                )
        return [Node(node_id, "Identity", (inputs[0],), outputs, {}, (source_id,))], None

    if proto.op_type == "Flatten":
        input_spec = value_specs.get(inputs[0])
        if input_spec is None or not input_spec.is_static:
            raise OnnxFrontendError(
                f"Flatten node {source_id!r} requires a static input shape"
            )
        axis = int(attrs.get("axis", 1))
        axis = axis + input_spec.rank if axis < 0 else axis
        if axis < 0 or axis > input_spec.rank:
            raise OnnxFrontendError(f"Flatten node {source_id!r} has invalid axis {axis}")
        left = int(np.prod(input_spec.shape[:axis], dtype=np.int64))
        right = int(np.prod(input_spec.shape[axis:], dtype=np.int64))
        return [Node(node_id, "Reshape", (inputs[0],), outputs, {"shape": [left, right]}, (source_id,))], None

    if proto.op_type == "GlobalAveragePool":
        input_spec = value_specs.get(inputs[0])
        if input_spec is None or input_spec.rank < 3:
            raise OnnxFrontendError(
                f"GlobalAveragePool node {source_id!r} requires rank >= 3"
            )
        axes = list(range(2, input_spec.rank))
        return [
            Node(
                node_id,
                "ReduceMean",
                (inputs[0],),
                outputs,
                {"axes": axes, "keepdims": True},
                (source_id,),
            )
        ], None

    if proto.op_type in {"Squeeze", "Unsqueeze"}:
        output_spec = value_specs.get(outputs[0])
        if output_spec is None or not output_spec.is_static:
            raise OnnxFrontendError(
                f"{proto.op_type} node {source_id!r} requires a static inferred output shape"
            )
        return [
            Node(
                node_id,
                "Reshape",
                (inputs[0],),
                outputs,
                {"shape": list(output_spec.shape)},
                (source_id,),
            )
        ], None

    if proto.op_type == "Gemm":
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        if alpha != 1.0 or beta != 1.0 or int(attrs.get("transA", 0)) != 0:
            raise OnnxFrontendError(
                f"Gemm node {source_id!r} requires alpha=beta=1 and transA=0"
            )
        if len(inputs) not in {2, 3}:
            raise OnnxFrontendError(f"Gemm node {source_id!r} has invalid arity")
        left, right = inputs[:2]
        lowered: list[Node] = []
        if int(attrs.get("transB", 0)):
            if right in constants:
                transposed_name = _unique(f"{right}__{node_id}_transposed", occupied_values)
                constants[transposed_name] = np.ascontiguousarray(constants[right].T)
                right = transposed_name
            else:
                transposed_name = _unique(f"{node_id}__transposed_b", occupied_values)
                lowered.append(
                    Node(
                        _unique(f"{node_id}__transpose", occupied_ids),
                        "Transpose",
                        (right,),
                        (transposed_name,),
                        {"perm": [1, 0]},
                        (source_id,),
                    )
                )
                right = transposed_name
        matmul_output = outputs[0] if len(inputs) == 2 else _unique(f"{node_id}__matmul", occupied_values)
        lowered.append(
            Node(
                _unique(f"{node_id}__matmul", occupied_ids),
                "MatMul",
                (left, right),
                (matmul_output,),
                {},
                (source_id,),
            )
        )
        if len(inputs) == 3:
            lowered.append(
                Node(node_id, "Add", (matmul_output, inputs[2]), outputs, {}, (source_id,))
            )
        return lowered, None

    unsupported = _unsupported_name(proto)
    return [Node(node_id, unsupported, inputs, outputs, attrs, (source_id,))], unsupported


def import_onnx_model(
    model: Any,
    *,
    source_path: str | Path | None = None,
    check_model: bool = True,
    infer_shapes: bool = True,
    strict: bool = True,
) -> Graph:
    """Translate an in-memory ONNX ``ModelProto`` to a GAC graph.

    ``strict=False`` retains unsupported nodes with qualified ONNX operation
    names.  Such a graph is useful for support analysis but is not compilable
    until those nodes are legalized.
    """

    onnx = _onnx_module()
    try:
        if check_model:
            onnx.checker.check_model(model)
        if infer_shapes:
            model = onnx.shape_inference.infer_shapes(
                model, check_type=True, strict_mode=False, data_prop=True
            )
    except Exception as exc:
        raise OnnxFrontendError(f"invalid ONNX model: {exc}") from exc

    graph_proto = model.graph
    if graph_proto.sparse_initializer:
        raise OnnxFrontendError("ONNX sparse initializers are not supported yet")
    base_dir = str(Path(source_path).resolve().parent) if source_path is not None else ""
    constants: dict[str, np.ndarray[Any, Any]] = {}
    for initializer in graph_proto.initializer:
        try:
            value = onnx.numpy_helper.to_array(initializer, base_dir=base_dir)
        except Exception as exc:
            raise OnnxFrontendError(
                f"cannot decode initializer {initializer.name!r}: {exc}"
            ) from exc
        constants[str(initializer.name)] = np.ascontiguousarray(value)

    declared_input_names = {str(value.name) for value in graph_proto.input if value.name}
    overrideable_initializers = sorted(declared_input_names & set(constants))
    # Before IR v4 initializers were required to appear in graph.input and were
    # not overrideable runtime inputs.  Starting at IR v4 the same spelling
    # denotes a default value that callers may replace.
    overrideable_initializers = (
        overrideable_initializers if int(model.ir_version) >= 4 else []
    )
    if overrideable_initializers:
        raise OnnxFrontendError(
            "ONNX initializers also declared as graph inputs have override/default-input "
            "semantics that the Canonical Graph cannot represent faithfully: "
            + ", ".join(repr(name) for name in overrideable_initializers[:12])
        )

    value_specs: dict[str, TensorSpec] = {}
    required_spec_names = {
        str(value.name) for value in (*graph_proto.input, *graph_proto.output) if value.name
    }
    for value_info in (*graph_proto.input, *graph_proto.value_info, *graph_proto.output):
        if not value_info.name:
            continue
        try:
            value_specs[str(value_info.name)] = _tensor_spec(value_info, onnx)
        except OnnxFrontendError:
            if str(value_info.name) in required_spec_names:
                raise
    for name, value in constants.items():
        value_specs[name] = TensorSpec(tuple(value.shape), value.dtype)

    inputs = {
        str(value.name): value_specs[str(value.name)]
        for value in graph_proto.input
        if value.name and value.name not in constants
    }
    occupied_values = set(inputs) | set(constants)
    occupied_values.update(str(output) for node in graph_proto.node for output in node.output if output)
    occupied_ids: set[str] = set()
    nodes: list[Node] = []
    unsupported_nodes: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    opsets = {
        str(item.domain or "ai.onnx"): int(item.version)
        for item in model.opset_import
    }
    onnx_opset = opsets.get("ai.onnx")
    if onnx_opset is None:
        raise OnnxFrontendError("ONNX model has no ai.onnx opset import")
    for index, proto in enumerate(graph_proto.node):
        source_counts[_unsupported_name(proto)] += 1
        lowered, unsupported = _translate_node(
            proto,
            index=index,
            constants=constants,
            value_specs=value_specs,
            occupied_values=occupied_values,
            occupied_ids=occupied_ids,
            onnx_opset=onnx_opset,
            onnx=onnx,
        )
        nodes.extend(lowered)
        canonical_counts.update(node.op for node in lowered)
        if unsupported is not None:
            unsupported_nodes.append(
                {
                    "node": str(proto.name or f"{proto.op_type}_{index}"),
                    "op": unsupported,
                }
            )

    if strict and unsupported_nodes:
        descriptions = ", ".join(
            f"{item['node']} ({item['op']})" for item in unsupported_nodes[:12]
        )
        if len(unsupported_nodes) > 12:
            descriptions += f", ... and {len(unsupported_nodes) - 12} more"
        raise OnnxFrontendError(
            "ONNX model contains operators without a faithful GAC lowering: "
            + descriptions
        )

    outputs = tuple(str(value.name) for value in graph_proto.output if value.name)
    if not outputs:
        raise OnnxFrontendError("ONNX graph has no named tensor outputs")
    metadata = {
        "import": {
            "format": "onnx",
            "producer_name": str(model.producer_name),
            "producer_version": str(model.producer_version),
            "ir_version": int(model.ir_version),
            "opsets": opsets,
            "source_operator_counts": dict(sorted(source_counts.items())),
            "canonical_operator_counts": dict(sorted(canonical_counts.items())),
            "unsupported_nodes": unsupported_nodes,
            "strict": bool(strict),
        }
    }
    graph = Graph(
        name=str(graph_proto.name or (Path(source_path).stem if source_path else "onnx_model")),
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        constants=constants,
        value_specs=value_specs,
        metadata=metadata,
    )
    try:
        graph.validate()
        if strict:
            ShapeInferencePass().run(graph)
            graph.validate()
    except Exception as exc:
        raise OnnxFrontendError(f"imported ONNX graph is invalid: {exc}") from exc
    return graph


def load_onnx(
    path: str | Path,
    *,
    check_model: bool = True,
    infer_shapes: bool = True,
    strict: bool = True,
) -> Graph:
    """Load an ONNX file, including external tensor data, and import it."""

    onnx = _onnx_module()
    model_path = Path(path)
    try:
        model = onnx.load(str(model_path), load_external_data=True)
    except Exception as exc:
        raise OnnxFrontendError(f"cannot read ONNX model {model_path}: {exc}") from exc
    return import_onnx_model(
        model,
        source_path=model_path,
        check_model=check_model,
        infer_shapes=infer_shapes,
        strict=strict,
    )
