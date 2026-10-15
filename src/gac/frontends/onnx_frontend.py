"""ONNX to GAC Canonical Graph import.

The frontend keeps ONNX as an optional dependency and translates framework
semantics into the small, device-independent IR in :mod:`gac.ir`.  Operators
whose semantics do not yet have a faithful GAC representation are reported
together instead of being silently approximated.

阅读这份前端时，可以把数据流理解为：

``ModelProto -> initializer/NodeProto -> NumPy/Node -> Graph``

* initializer 中的真实权重转为 ``numpy.ndarray``，最终保存在
  ``Graph.constants[张量名]`` 中；
* ``Node.inputs`` 只保存张量名，通过名字引用输入、权重和中间结果；
* ``TensorSpec`` 只描述 shape、dtype 和 layout，不保存张量数据；
* 本文件不分配 Device DDR、LMEM 或双 Buffer。设备内存规划由后端和
  Runtime 在 Graph 之后完成。

MLIR 后端接收这里返回的 ``Graph``，再把常量和节点分别变成 MLIR SSA
值与算子。因此 ONNX 前端不直接调用 ``mlir-opt``。
"""

from __future__ import annotations

from collections import Counter
import importlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


class OnnxFrontendError(ValueError):
    """Raised when an ONNX model cannot be represented faithfully."""


class _UnsupportedOnnxSemantics(OnnxFrontendError):
    """A valid source operation needs a lowering the canonical IR lacks.

    Unlike malformed ONNX, this can be retained verbatim for inspection when
    ``strict=False``. It never makes the operation executable.
    """


# 可以保持语义不变、直接改名进入 GAC Canonical Graph 的算子。
# 其余算子需要在 _translate_node() 中显式处理属性、opset 差异，或者
# 拆成多个规范算子，避免“名字相同但语义不同”的静默错误。
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
    """延迟导入可选 ONNX 依赖，并在解析任何模型前检查安全版本下限。"""

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
    """把 ONNX 属性值转换为 Graph 元数据可以保存的 Python 值。"""

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


def _contiguous_tensor(value: Any) -> np.ndarray:
    """Keep the ONNX tensor rank while ensuring contiguous storage.

    ``np.ascontiguousarray`` promotes scalar tensors from shape ``()`` to
    ``(1,)``. Scalars are already contiguous, so return them without changing
    the rank used by broadcasting, shape programs, and serialized TensorSpecs.
    """

    array = np.asarray(value)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _unique(base: str, occupied: set[str]) -> str:
    """生成不重复的 Node id 或张量名，维持 Graph 的 SSA 式命名规则。"""

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
    """将 ONNX ValueInfoProto 转为只含类型信息的 TensorSpec。"""

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
    """读取一个 NodeProto 的属性；Tensor 属性在编译器侧保存为数组。"""

    result: dict[str, Any] = {}
    for attribute in proto.attribute:
        value = onnx.helper.get_attribute_value(attribute)
        if hasattr(value, "node") and hasattr(value, "initializer"):
            raise OnnxFrontendError(
                f"node {proto.name or proto.op_type!r} contains a nested graph; "
                "ONNX control flow is not supported yet"
            )
        if hasattr(value, "dims") and hasattr(value, "data_type"):
            result[str(attribute.name)] = _contiguous_tensor(
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
    """读取必须在编译期已知的整数 shape/axes 等一维参数。"""

    if name not in constants:
        raise _UnsupportedOnnxSemantics(
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
    """读取 Clip 等算子要求在编译期已知的标量输入。"""

    if not name:
        return None
    if name not in constants:
        raise _UnsupportedOnnxSemantics(
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
    """读取 Dropout training_mode 等编译期布尔开关。"""

    if name not in constants:
        raise _UnsupportedOnnxSemantics(
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
        raise _UnsupportedOnnxSemantics(
            f"node {node_id!r} auto_pad={auto_pad!r} must be legalized to explicit pads"
        )


def _unsupported_name(proto: Any) -> str:
    """保留 domain，避免不同 ONNX 扩展域中的同名算子发生冲突。"""

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
    """将一个 ONNX 节点合法化为零个、一个或多个 GAC 节点。

    返回值中的节点列表可能为空（例如 Constant 被吸收到 constants），
    也可能包含多个节点（例如 Gemm 被拆成 MatMul + Add）。第二个返回值
    记录无法忠实转换的源算子；strict 模式会在扫描完整张图后统一报错。
    """

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
        # Constant 是图中的常量生产者。这里直接把真实数据放进
        # constants[输出名]，因此 Canonical Graph 不再需要一个运行时节点。
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
        constants[outputs[0]] = _contiguous_tensor(value)
        occupied_values.add(outputs[0])
        return [], None

    direct = _DIRECT_OPS.get(str(proto.op_type))
    if direct is not None:
        # 即使是“直接映射”，也必须先修正 opset 默认值和历史语义差异。
        if len(outputs) != 1:
            raise OnnxFrontendError(
                f"node {source_id!r} has {len(outputs)} outputs; canonical {direct} "
                "currently requires one"
            )
        canonical_attrs: dict[str, Any] = {}
        if proto.op_type in {"Add", "Sub", "Mul", "Div"} and onnx_opset <= 6:
            if int(attrs.get("broadcast", 0)) != 0 or "axis" in attrs:
                raise _UnsupportedOnnxSemantics(
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
                    raise _UnsupportedOnnxSemantics(
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
                    raise _UnsupportedOnnxSemantics(
                        f"Softmax node {source_id!r} needs an inferred input rank for "
                        f"ONNX opset {onnx_opset} semantics"
                    )
                normalized_axis = axis + input_spec.rank if axis < 0 else axis
                if normalized_axis != input_spec.rank - 1:
                    raise _UnsupportedOnnxSemantics(
                        f"Softmax node {source_id!r} uses pre-opset-13 flatten-from-axis "
                        f"semantics at axis {axis}; canonical Softmax only reduces one axis"
                    )
            canonical_attrs["axis"] = axis
        return [Node(node_id, direct, inputs, outputs, canonical_attrs, (source_id,))], None

    if proto.op_type == "Gelu":
        approximation = str(attrs.get("approximate", "none")).lower()
        if approximation not in {"tanh"}:
            raise _UnsupportedOnnxSemantics(
                f"Gelu node {source_id!r} uses approximation={approximation!r}; "
                "the current GAC GELU contract is tanh"
            )
        return [Node(node_id, "GELU", inputs, outputs, {"approximation": "tanh"}, (source_id,))], None

    if proto.op_type == "LayerNormalization":
        if len(outputs) != 1:
            raise _UnsupportedOnnxSemantics(
                f"LayerNormalization node {source_id!r} exposes training/statistic outputs"
            )
        input_spec = value_specs.get(inputs[0]) if inputs else None
        if input_spec is None or input_spec.dtype != "float32":
            raise _UnsupportedOnnxSemantics(
                f"LayerNormalization node {source_id!r} requires float32 input until "
                "the Canonical LayerNorm records ONNX stash/accumulator precision"
            )
        if int(attrs.get("stash_type", 1)) != 1:
            raise _UnsupportedOnnxSemantics(
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
            raise _UnsupportedOnnxSemantics(
                f"Reshape node {source_id!r} uses allowzero=1, which GAC does not support"
            )
        # ONNX 把目标 shape 表示成第二个输入；GAC 将其固化为节点属性，
        # 所以后端只看到数据输入，不需要为 shape 张量分配运行时 Buffer。
        shape = _constant_vector(inputs[1], constants, node=source_id)
        if 0 in shape:
            input_spec = value_specs.get(inputs[0])
            if input_spec is None:
                raise _UnsupportedOnnxSemantics(
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
                raise _UnsupportedOnnxSemantics(
                    f"Reshape node {source_id!r} has a zero dimension referring to a dynamic input"
                )
        return [Node(node_id, "Reshape", (inputs[0],), outputs, {"shape": shape}, (source_id,))], None

    if proto.op_type == "Expand":
        if len(inputs) != 2:
            raise OnnxFrontendError(f"Expand node {source_id!r} needs data and shape")
        shape = _constant_vector(inputs[1], constants, node=source_id)
        return [Node(node_id, "Expand", (inputs[0],), outputs, {"shape": shape}, (source_id,))], None

    if proto.op_type == "Slice":
        # starts/ends/axes/steps 同样在前端读取并写入 attrs；保留 raw_inputs
        # 是为了不丢失 ONNX 用空字符串表达的可选输入位置。
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
            raise _UnsupportedOnnxSemantics(
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
            raise _UnsupportedOnnxSemantics(f"Conv node {source_id!r} is not two-dimensional")
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
            raise _UnsupportedOnnxSemantics(
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
            raise _UnsupportedOnnxSemantics(
                f"MaxPool node {source_id!r} index output is not supported"
            )
        kernel = _pairs(attrs.get("kernel_shape"), ())
        if len(kernel) != 2:
            raise _UnsupportedOnnxSemantics(f"MaxPool node {source_id!r} is not two-dimensional")
        canonical = {
            "kernel_shape": kernel,
            "strides": _pairs(attrs.get("strides"), (1, 1)),
            "dilations": _pairs(attrs.get("dilations"), (1, 1)),
            "pads": _pairs(attrs.get("pads"), (0, 0, 0, 0)),
            "ceil_mode": bool(int(attrs.get("ceil_mode", 0))),
        }
        if int(attrs.get("storage_order", 0)) != 0:
            raise _UnsupportedOnnxSemantics(f"MaxPool node {source_id!r} uses storage_order=1")
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
            raise _UnsupportedOnnxSemantics(
                f"Dropout node {source_id!r} mask output is not supported"
            )
        if onnx_opset <= 6 and int(attrs.get("is_test", 0)) == 0:
            raise _UnsupportedOnnxSemantics(
                f"Dropout node {source_id!r} uses training semantics in ONNX "
                f"opset {onnx_opset}"
            )
        if onnx_opset >= 12 and len(raw_inputs) >= 3 and raw_inputs[2]:
            training = _constant_bool(raw_inputs[2], constants, node=source_id)
            if training:
                raise _UnsupportedOnnxSemantics(
                    f"Dropout node {source_id!r} is in training mode"
                )
        return [Node(node_id, "Identity", (inputs[0],), outputs, {}, (source_id,))], None

    if proto.op_type == "Flatten":
        input_spec = value_specs.get(inputs[0])
        if input_spec is None or not input_spec.is_static:
            raise _UnsupportedOnnxSemantics(
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
        if input_spec is None:
            raise _UnsupportedOnnxSemantics(
                f"GlobalAveragePool node {source_id!r} needs an inferred input rank"
            )
        if input_spec.rank < 3:
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
            raise _UnsupportedOnnxSemantics(
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
        # GAC 没有独立 Gemm：A*B 转为 MatMul，有 C 时再追加 Add。
        # transB 若作用于常量权重，就在编译期生成一份连续的转置数组；
        # 若 B 是运行时值，则显式插入 Transpose 节点。
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        if alpha != 1.0 or beta != 1.0 or int(attrs.get("transA", 0)) != 0:
            raise _UnsupportedOnnxSemantics(
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
    # 非 strict 模式保留源算子，供支持率统计和子图切分使用；它还不能
    # 直接进入要求全部算子合法化的设备后端。
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
    names, including recognized operations whose parameters cannot yet be
    lowered. Their source edges and attributes remain intact; the import
    report records the lowering constraint. Such a graph is useful for support
    analysis but is not compilable until those nodes are legalized. Invalid
    models and unsupported nested control flow still raise an error.

    该函数完成五个阶段：

    1. 校验 ONNX 并执行 ONNX shape inference；
    2. 将 initializer 解码到 ``constants``；遍历节点时再合并 Constant；
    3. 建立每个值的 ``TensorSpec``；
    4. 按拓扑顺序把 NodeProto 合法化为 GAC ``Node``；
    5. 组装并验证一个与设备无关的 ``Graph``。
    """

    onnx = _onnx_module()
    # 阶段 1：先让 ONNX 官方 checker 和 shape inference 处理格式与类型。
    # 后面的合法化依赖这里补出的 value_info；失败时不应猜测形状。
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

    # 阶段 2：权重、bias、shape、axes 等常量张量保存在这个字典中。
    #
    # key   = ONNX 张量名，例如 "encoder.ffn.weight"
    # value = C-contiguous NumPy ndarray，例如 shape=(256, 1024), dtype=float32
    #
    # Node 不复制矩阵，只在 inputs 中保存 key；Graph 构造时该字典成为
    # graph.constants，MLIR 后端再通过同一个 key 查找权重。
    constants: dict[str, np.ndarray[Any, Any]] = {}
    for initializer in graph_proto.initializer:
        try:
            value = onnx.numpy_helper.to_array(initializer, base_dir=base_dir)
        except Exception as exc:
            raise OnnxFrontendError(
                f"cannot decode initializer {initializer.name!r}: {exc}"
            ) from exc
        constants[str(initializer.name)] = _contiguous_tensor(value)

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

    # 阶段 3：value_specs 是“类型表”，只含 shape/dtype/layout，不含数据。
    # 输入、权重、中间结果和输出都使用同一名字空间，便于后端建立 SSA 映射。
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
        # initializer 可能不在 value_info 中，因此用真实 ndarray 补齐类型。
        value_specs[name] = TensorSpec(tuple(value.shape), value.dtype)

    # initializer 不是模型调用者提供的输入；这里只留下真正的运行时输入。
    inputs = {
        str(value.name): value_specs[str(value.name)]
        for value in graph_proto.input
        if value.name and value.name not in constants
    }
    occupied_values = set(inputs) | set(constants)
    occupied_values.update(str(output) for node in graph_proto.node for output in node.output if output)
    occupied_ids: set[str] = set()
    # 阶段 4：nodes 保存拓扑顺序和张量名引用，不保存任何张量 payload。
    nodes: list[Node] = []
    unsupported_nodes: list[dict[str, Any]] = []
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
        # Inspection must not lose a valid node merely because its shape/axes
        # input is a runtime value. Snapshot names, not weight payloads, so a
        # rejected decomposition cannot leak temporary constants or SSA names.
        snapshot = (
            (set(constants), occupied_values.copy(), occupied_ids.copy())
            if not strict else None
        )
        reason: str | None = None
        try:
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
        except _UnsupportedOnnxSemantics as exc:
            if strict:
                raise
            assert snapshot is not None
            previous_constants, previous_values, previous_ids = snapshot
            for name in set(constants) - previous_constants:
                del constants[name]
            occupied_values.intersection_update(previous_values)
            occupied_ids.intersection_update(previous_ids)
            source_id = str(proto.name or f"{proto.op_type}_{index}")
            unsupported = _unsupported_name(proto)
            lowered = [
                Node(
                    _unique(source_id, occupied_ids),
                    unsupported,
                    tuple(str(name) for name in proto.input if name),
                    tuple(str(name) for name in proto.output if name),
                    _attributes(proto, onnx),
                    (source_id,),
                )
            ]
            reason = str(exc)
        # 一个 ONNX 节点可能产生多个规范节点；Constant 则产生零个节点。
        nodes.extend(lowered)
        canonical_counts.update(node.op for node in lowered)
        if unsupported is not None:
            diagnostic: dict[str, Any] = {
                "node": str(proto.name or f"{proto.op_type}_{index}"),
                "op": unsupported,
            }
            if reason is not None:
                diagnostic["reason"] = reason
            # Node.inputs only contains actual SSA edges. Preserve omitted
            # optional slots separately so Clip(x, "", max) remains readable
            # as the original ONNX operation rather than Clip(x, min).
            if any(not name for name in proto.input):
                diagnostic["input_slots"] = [str(name) for name in proto.input]
            if any(not name for name in proto.output):
                diagnostic["output_slots"] = [str(name) for name in proto.output]
            unsupported_nodes.append(diagnostic)

    if strict and unsupported_nodes:
        # 完整扫描后一次列出所有不支持算子，比遇到第一个就停止更利于
        # 判断子图覆盖率和安排后续 Lowering。
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
    # 阶段 5：这是 ONNX 前端交给优化 Pass、子图切分和 MLIR 后端的唯一对象。
    # 其中 graph.constants 持有权重 ndarray，graph.nodes 通过字符串名字引用它们。
    # 本文件到此只生成 Graph；compile_mlir() 负责生成 MLIR 文本，
    # mlir_toolchain.py 再负责调用外部 mlir-opt。
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
            # GAC 自己再做一次 shape inference，确认规范算子的类型关系成立，
            # 而不是只相信源 ONNX 中可能不完整的 value_info。
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
    """Load an ONNX file, including external tensor data, and import it.

    ``onnx.load`` 负责读取 protobuf 和外置权重文件；此函数随后把内存中的
    ModelProto 交给 ``import_onnx_model``，最终返回一个 ``Graph``。
    """

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
