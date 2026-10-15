"""Import a functional ``torch.export.ExportedProgram`` into GAC IR.

PyTorch is deliberately imported only by :func:`load_exported_program`.
In-memory importing is implemented against the public ExportedProgram/FX
protocol so the core compiler remains usable without a Torch installation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gac.ir import Graph, Node, TensorSpec
from gac.passes import ShapeInferencePass


class TorchExportFrontendError(ValueError):
    """Raised when ExportedProgram semantics cannot be represented faithfully."""


_DTYPES = {
    "float16": "float16",
    "half": "float16",
    "float32": "float32",
    "float": "float32",
    "float64": "float64",
    "double": "float64",
    "int8": "int8",
    "int32": "int32",
    "int": "int32",
    "int64": "int64",
    "long": "int64",
    "bool": "bool",
}
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchExportFrontendError(
            "torch.export import requires PyTorch; install a compatible PyTorch 2.x "
            "build or gac-heterogeneous[torch-export]"
        ) from exc


def _kind_name(kind: Any) -> str:
    return str(getattr(kind, "name", kind)).split(".")[-1].lower()


def _argument_name(argument: Any) -> str | None:
    name = getattr(argument, "name", None)
    return str(name) if isinstance(name, str) and name else None


def _is_fx_node(value: Any) -> bool:
    return all(hasattr(value, field) for field in ("op", "name", "args", "kwargs"))


def _dtype_name(dtype: Any, *, where: str) -> str:
    raw = str(dtype).replace("torch.", "").lower()
    try:
        return _DTYPES[raw]
    except KeyError as exc:
        raise TorchExportFrontendError(
            f"{where} uses unsupported dtype {dtype!r}; bfloat16, quantized, complex, "
            "and float8 values require a later canonical type extension"
        ) from exc


def _dimension(value: Any) -> int | str:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        result = int(value)
        if result < 0:
            raise TorchExportFrontendError(f"tensor dimension must be non-negative, got {result}")
        return result
    text = str(value)
    if not text:
        raise TorchExportFrontendError("symbolic tensor dimension has no printable name")
    return text


def _tensor_spec(value: Any, *, where: str) -> TensorSpec:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TorchExportFrontendError(
            f"{where} is not a tensor value or has no torch.export tensor metadata"
        )
    return TensorSpec(tuple(_dimension(dim) for dim in shape), _dtype_name(dtype, where=where))


def _node_spec(node: Any) -> TensorSpec:
    metadata = getattr(node, "meta", None)
    if not isinstance(metadata, Mapping) or "val" not in metadata:
        raise TorchExportFrontendError(
            f"FX node {node.name!r} has no meta['val'] tensor information"
        )
    value = metadata["val"]
    if isinstance(value, (tuple, list, dict)):
        raise TorchExportFrontendError(
            f"FX node {node.name!r} returns multiple or structured values; "
            "tuple-valued ATen operations are not legalized yet"
        )
    return _tensor_spec(value, where=f"FX node {node.name!r}")


def _numpy_tensor(value: Any, *, where: str) -> np.ndarray[Any, Any]:
    try:
        tensor = value.detach() if hasattr(value, "detach") else value
        tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
        tensor = tensor.contiguous() if hasattr(tensor, "contiguous") else tensor
        array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    except Exception as exc:
        raise TorchExportFrontendError(f"cannot materialize {where}: {exc}") from exc
    try:
        dtype = _dtype_name(getattr(value, "dtype", array.dtype), where=where)
        result = np.asarray(array, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise TorchExportFrontendError(f"cannot convert {where} to NumPy: {exc}") from exc
    return np.ascontiguousarray(result.copy())


def _target_name(target: Any) -> str:
    text = str(target)
    if text and not text.startswith("<"):
        return text.replace("::", ".")
    schema = getattr(target, "_schema", None)
    if schema is not None:
        base = str(getattr(schema, "name", "")).replace("::", ".")
        overload = str(getattr(schema, "overload_name", ""))
        if base:
            return f"{base}.{overload or 'default'}"
    module = str(getattr(target, "__module__", ""))
    name = str(getattr(target, "__name__", ""))
    return f"{module}.{name}".strip(".") or repr(target)


def _normalize_target(target: Any) -> str:
    return _target_name(target).replace("torch.ops.", "").replace("::", ".").lower()


def _unique(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        occupied.add(base)
        return base
    index = 1
    while f"{base}_{index}" in occupied:
        index += 1
    result = f"{base}_{index}"
    occupied.add(result)
    return result


def _literal_ints(value: Any, *, where: str) -> list[int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]
    if not isinstance(value, (tuple, list)) or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in value
    ):
        raise TorchExportFrontendError(f"{where} must be a compile-time integer list")
    return [int(item) for item in value]


def _pair(value: Any, *, where: str) -> list[int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value), int(value)]
    values = _literal_ints(value, where=where)
    if len(values) != 2:
        raise TorchExportFrontendError(f"{where} must contain two integers")
    return values


def _optional_tensor_name(value: Any, *, where: str) -> str | None:
    if value is None:
        return None
    if not _is_fx_node(value):
        raise TorchExportFrontendError(f"{where} must be a tensor or None")
    return str(value.name)


def _json_value(value: Any) -> Any:
    if _is_fx_node(value):
        return {"value": str(value.name)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Importer:
    def __init__(self, program: Any, *, strict: bool) -> None:
        self.program = program
        self.strict = strict
        self.constants: dict[str, np.ndarray[Any, Any]] = {}
        self.inputs: dict[str, TensorSpec] = {}
        self.value_specs: dict[str, TensorSpec] = {}
        self.source_value_specs: dict[str, TensorSpec] = {}
        self.nodes: list[Node] = []
        self.occupied_values: set[str] = set()
        self.occupied_ids: set[str] = set()
        self.unsupported: list[dict[str, str]] = []
        self.source_counts: Counter[str] = Counter()
        self.canonical_counts: Counter[str] = Counter()

    @property
    def graph_nodes(self) -> list[Any]:
        graph_module = getattr(self.program, "graph_module", None)
        graph = getattr(graph_module, "graph", None)
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            raise TorchExportFrontendError("object is not a torch.export.ExportedProgram")
        return list(nodes)

    def _add_constant(self, base: str, value: Any, *, where: str) -> str:
        name = _unique(base, self.occupied_values)
        array = _numpy_tensor(value, where=where)
        self.constants[name] = array
        self.value_specs[name] = TensorSpec(tuple(array.shape), array.dtype)
        return name

    def _scalar(self, value: Any, *, node: Any) -> str:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isrealobj(value)
        ):
            raise TorchExportFrontendError(
                f"FX node {node.name!r} expected a tensor or numeric scalar, got {value!r}"
            )
        spec = _node_spec(node)
        name = _unique(f"{node.name}__scalar", self.occupied_values)
        self.constants[name] = np.asarray(value, dtype=spec.dtype)
        self.value_specs[name] = TensorSpec((), spec.dtype)
        return name

    def _value(self, value: Any, *, node: Any) -> str:
        return str(value.name) if _is_fx_node(value) else self._scalar(value, node=node)

    def _emit(
        self,
        node: Any,
        op: str,
        inputs: Iterable[str],
        *,
        attrs: Mapping[str, Any] | None = None,
        output: str | None = None,
        suffix: str | None = None,
    ) -> str:
        output_name = output or str(node.name)
        node_id = _unique(
            str(node.name) if suffix is None else f"{node.name}__{suffix}",
            self.occupied_ids,
        )
        self.nodes.append(
            Node(
                node_id,
                op,
                tuple(inputs),
                (output_name,),
                dict(attrs or {}),
                (f"torch.export:{node.name}",),
            )
        )
        self.canonical_counts[op] += 1
        self.occupied_values.add(output_name)
        return output_name

    def _placeholder_inputs(self, placeholders: Mapping[str, Any]) -> None:
        signature = getattr(self.program, "graph_signature", None)
        input_specs = list(getattr(signature, "input_specs", ()))
        state = dict(getattr(self.program, "state_dict", {}) or {})
        constants = dict(getattr(self.program, "constants", {}) or {})
        if not input_specs:
            for name, node in placeholders.items():
                spec = _node_spec(node)
                self.inputs[name] = spec
                self.value_specs[name] = spec
                self.occupied_values.add(name)
            return

        declared: set[str] = set()
        for input_spec in input_specs:
            name = _argument_name(getattr(input_spec, "arg", None))
            if name is None or name not in placeholders:
                raise TorchExportFrontendError(
                    "torch.export graph signature does not match FX placeholders"
                )
            declared.add(name)
            kind = _kind_name(getattr(input_spec, "kind", ""))
            target = getattr(input_spec, "target", None)
            target_name = str(target) if target is not None else name
            if kind == "user_input":
                spec = _node_spec(placeholders[name])
                self.inputs[name] = spec
                self.value_specs[name] = spec
                self.occupied_values.add(name)
                continue
            if kind in {"parameter", "buffer", "constant_tensor"}:
                source = state.get(target_name)
                if source is None:
                    source = constants.get(target_name)
                if source is None:
                    source = state.get(name, constants.get(name))
                if source is None:
                    raise TorchExportFrontendError(
                        f"lifted {kind} {target_name!r} for placeholder {name!r} is missing"
                    )
                array = _numpy_tensor(source, where=f"lifted {kind} {target_name!r}")
                self.constants[name] = array
                self.value_specs[name] = TensorSpec(tuple(array.shape), array.dtype)
                self.occupied_values.add(name)
                continue
            raise TorchExportFrontendError(
                f"placeholder {name!r} has unsupported graph-signature kind {kind!r}; "
                "tokens, custom objects, and mutation inputs are not inference tensors"
            )
        missing = set(placeholders) - declared
        if missing:
            raise TorchExportFrontendError(
                f"FX placeholders are absent from graph signature: {sorted(missing)}"
            )

    def _linear(self, node: Any) -> None:
        if len(node.args) < 2:
            raise TorchExportFrontendError(f"linear node {node.name!r} has invalid arity")
        value = self._value(node.args[0], node=node)
        weight = self._value(node.args[1], node=node)
        bias = node.args[2] if len(node.args) >= 3 else None
        if weight in self.constants:
            transposed = self._add_constant(
                f"{weight}__{node.name}_t",
                self.constants[weight].T,
                where=f"linear weight {weight!r}",
            )
        else:
            spec = self.value_specs.get(weight)
            if spec is None or spec.rank != 2:
                raise TorchExportFrontendError(
                    f"linear node {node.name!r} requires a rank-2 weight"
                )
            transposed = _unique(f"{node.name}__weight_t", self.occupied_values)
            self._emit(
                node,
                "Transpose",
                (weight,),
                attrs={"perm": [1, 0]},
                output=transposed,
                suffix="transpose",
            )
        if bias is None:
            self._emit(node, "MatMul", (value, transposed))
        else:
            product = _unique(f"{node.name}__matmul", self.occupied_values)
            self._emit(
                node,
                "MatMul",
                (value, transposed),
                output=product,
                suffix="matmul",
            )
            self._emit(node, "Add", (product, self._value(bias, node=node)))

    def _binary(self, node: Any, op: str) -> None:
        if len(node.args) < 2:
            raise TorchExportFrontendError(f"{op} node {node.name!r} has invalid arity")
        if op in {"Add", "Sub"}:
            alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
            if alpha != 1:
                raise TorchExportFrontendError(
                    f"{op} node {node.name!r} uses alpha={alpha!r}; legalize scaling first"
                )
        if op == "Div" and node.kwargs.get("rounding_mode") is not None:
            raise TorchExportFrontendError(
                f"Div node {node.name!r} uses integer rounding semantics"
            )
        self._emit(
            node,
            op,
            (self._value(node.args[0], node=node), self._value(node.args[1], node=node)),
        )

    def _conv2d(self, node: Any, *, convolution: bool) -> None:
        args = list(node.args)
        if len(args) < 2:
            raise TorchExportFrontendError(f"Conv node {node.name!r} has invalid arity")
        value = self._value(args[0], node=node)
        weight = self._value(args[1], node=node)
        bias = args[2] if len(args) > 2 else None
        if convolution:
            stride = args[3] if len(args) > 3 else [1, 1]
            padding = args[4] if len(args) > 4 else [0, 0]
            dilation = args[5] if len(args) > 5 else [1, 1]
            transposed = bool(args[6]) if len(args) > 6 else False
            output_padding = args[7] if len(args) > 7 else [0, 0]
            groups = int(args[8]) if len(args) > 8 else 1
            if transposed or any(_literal_ints(output_padding, where="output_padding")):
                raise TorchExportFrontendError(
                    f"Conv node {node.name!r} uses transposed/output-padding semantics"
                )
        else:
            stride = args[3] if len(args) > 3 else [1, 1]
            padding = args[4] if len(args) > 4 else [0, 0]
            dilation = args[5] if len(args) > 5 else [1, 1]
            groups = int(args[6]) if len(args) > 6 else 1
        pad = _pair(padding, where=f"Conv node {node.name!r} padding")
        inputs = [value, weight]
        bias_name = _optional_tensor_name(bias, where=f"Conv node {node.name!r} bias")
        if bias_name is not None:
            inputs.append(bias_name)
        self._emit(
            node,
            "Conv2D",
            inputs,
            attrs={
                "strides": _pair(stride, where=f"Conv node {node.name!r} stride"),
                "dilations": _pair(dilation, where=f"Conv node {node.name!r} dilation"),
                "pads": [pad[0], pad[1], pad[0], pad[1]],
                "groups": groups,
            },
        )

    def _translate(self, node: Any) -> None:
        target = _normalize_target(node.target)
        self.source_counts[target] += 1
        args = list(node.args)
        if target in {"aten.mm.default", "aten.bmm.default", "aten.matmul.default"}:
            self._emit(node, "MatMul", (self._value(args[0], node=node), self._value(args[1], node=node)))
            return
        binary = {
            "aten.add.tensor": "Add",
            "aten.add.scalar": "Add",
            "aten.sub.tensor": "Sub",
            "aten.sub.scalar": "Sub",
            "aten.mul.tensor": "Mul",
            "aten.mul.scalar": "Mul",
            "aten.div.tensor": "Div",
            "aten.div.scalar": "Div",
        }.get(target)
        if binary:
            self._binary(node, binary)
            return
        unary = {
            "aten.relu.default": "ReLU",
            "aten.sigmoid.default": "Sigmoid",
            "aten.log.default": "Log",
            "aten.sin.default": "Sin",
            "aten.cos.default": "Cos",
        }.get(target)
        if unary:
            self._emit(node, unary, (self._value(args[0], node=node),))
            return
        if target == "aten.gelu.default":
            approximation = node.kwargs.get("approximate", args[1] if len(args) > 1 else "none")
            if str(approximation) != "tanh":
                raise TorchExportFrontendError(
                    f"GELU node {node.name!r} uses approximate={approximation!r}; "
                    "the current GAC GELU contract is tanh"
                )
            self._emit(node, "GELU", (self._value(args[0], node=node),), attrs={"approximation": "tanh"})
            return
        if target == "aten.linear.default":
            self._linear(node)
            return
        if target in {"aten.view.default", "aten.reshape.default", "aten._unsafe_view.default"}:
            shape = _literal_ints(args[1], where=f"Reshape node {node.name!r} shape")
            self._emit(node, "Reshape", (self._value(args[0], node=node),), attrs={"shape": shape})
            return
        if target in {
            "aten.flatten.using_ints",
            "aten.squeeze.dim",
            "aten.squeeze.dims",
            "aten.unsqueeze.default",
        }:
            output_spec = _node_spec(node)
            if not output_spec.is_static or any(
                not isinstance(dim, int) or dim <= 0 for dim in output_spec.shape
            ):
                raise TorchExportFrontendError(
                    f"shape-only node {node.name!r} requires a positive static output shape"
                )
            self._emit(
                node,
                "Reshape",
                (self._value(args[0], node=node),),
                attrs={"shape": list(output_spec.shape)},
            )
            return
        if target == "aten.permute.default":
            perm = _literal_ints(args[1], where=f"Transpose node {node.name!r} permutation")
            self._emit(node, "Transpose", (self._value(args[0], node=node),), attrs={"perm": perm})
            return
        if target in {"aten.transpose.int", "aten.transpose.dimname"}:
            value_name = self._value(args[0], node=node)
            spec = self.value_specs.get(value_name)
            if spec is None:
                raise TorchExportFrontendError(f"Transpose node {node.name!r} has no input spec")
            first, second = int(args[1]), int(args[2])
            first = first + spec.rank if first < 0 else first
            second = second + spec.rank if second < 0 else second
            perm = list(range(spec.rank))
            try:
                perm[first], perm[second] = perm[second], perm[first]
            except IndexError as exc:
                raise TorchExportFrontendError(
                    f"Transpose node {node.name!r} has invalid dimensions"
                ) from exc
            self._emit(node, "Transpose", (value_name,), attrs={"perm": perm})
            return
        if target == "aten.t.default":
            self._emit(node, "Transpose", (self._value(args[0], node=node),), attrs={"perm": [1, 0]})
            return
        if target == "aten.cat.default":
            tensors = args[0]
            if not isinstance(tensors, (tuple, list)) or not tensors:
                raise TorchExportFrontendError(f"Concat node {node.name!r} has no tensor list")
            axis = int(args[1] if len(args) > 1 else node.kwargs.get("dim", 0))
            self._emit(node, "Concat", (self._value(item, node=node) for item in tensors), attrs={"axis": axis})
            return
        if target == "aten.expand.default":
            value_name = self._value(args[0], node=node)
            shape = _literal_ints(args[1], where=f"Expand node {node.name!r} shape")
            spec = self.value_specs.get(value_name)
            if spec is None or len(shape) != spec.rank:
                raise TorchExportFrontendError(
                    f"Expand node {node.name!r} requires shape with the input rank"
                )
            resolved = [
                spec.shape[index] if dim == -1 else dim for index, dim in enumerate(shape)
            ]
            if not all(isinstance(dim, int) and dim > 0 for dim in resolved):
                raise TorchExportFrontendError(
                    f"Expand node {node.name!r} has a dynamic or invalid target shape"
                )
            self._emit(node, "Expand", (value_name,), attrs={"shape": resolved})
            return
        if target == "aten.slice.tensor":
            axis = int(args[1] if len(args) > 1 else 0)
            start = int(args[2] if len(args) > 2 and args[2] is not None else 0)
            end = int(args[3] if len(args) > 3 and args[3] is not None else 2**63 - 1)
            step = int(args[4] if len(args) > 4 else 1)
            self._emit(
                node,
                "Slice",
                (self._value(args[0], node=node),),
                attrs={"starts": [start], "ends": [end], "axes": [axis], "steps": [step]},
            )
            return
        if target in {"aten.sum.dim_intlist", "aten.mean.dim"}:
            axes = _literal_ints(args[1], where=f"Reduction node {node.name!r} axes")
            keepdims = bool(args[2] if len(args) > 2 else node.kwargs.get("keepdim", False))
            dtype = node.kwargs.get("dtype", args[3] if len(args) > 3 else None)
            if dtype is not None:
                raise TorchExportFrontendError(
                    f"Reduction node {node.name!r} specifies dtype={dtype!r}; "
                    "legalize the accumulation cast explicitly first"
                )
            op = "ReduceSum" if target.startswith("aten.sum") else "ReduceMean"
            self._emit(node, op, (self._value(args[0], node=node),), attrs={"axes": axes, "keepdims": keepdims})
            return
        if target in {"aten.softmax.int", "aten._softmax.default"}:
            if target == "aten._softmax.default" and len(args) > 2 and bool(args[2]):
                raise TorchExportFrontendError(
                    f"Softmax node {node.name!r} requests half_to_float"
                )
            if target == "aten.softmax.int":
                dtype = node.kwargs.get("dtype", args[2] if len(args) > 2 else None)
                if dtype is not None:
                    raise TorchExportFrontendError(
                        f"Softmax node {node.name!r} specifies dtype={dtype!r}; "
                        "legalize the input cast explicitly first"
                    )
            self._emit(node, "Softmax", (self._value(args[0], node=node),), attrs={"axis": int(args[1])})
            return
        if target == "aten.layer_norm.default":
            value_name = self._value(args[0], node=node)
            normalized = _literal_ints(args[1], where=f"LayerNorm node {node.name!r} normalized_shape")
            spec = self.value_specs.get(value_name)
            if spec is None or not normalized or len(normalized) > spec.rank:
                raise TorchExportFrontendError(f"LayerNorm node {node.name!r} has invalid normalized_shape")
            if spec.dtype != "float32":
                raise TorchExportFrontendError(
                    f"LayerNorm node {node.name!r} requires float32 input until the "
                    "Canonical LayerNorm records ATen accumulator precision"
                )
            inputs = [value_name]
            for position, label in ((2, "weight"), (3, "bias")):
                if len(args) > position and args[position] is not None:
                    inputs.append(self._value(args[position], node=node))
                elif position == 2 and len(args) > 3 and args[3] is not None:
                    raise TorchExportFrontendError(
                        f"LayerNorm node {node.name!r} has bias without weight"
                    )
            epsilon = float(args[4] if len(args) > 4 else node.kwargs.get("eps", 1e-5))
            self._emit(node, "LayerNorm", inputs, attrs={"axis": spec.rank - len(normalized), "epsilon": epsilon})
            return
        if target in {"aten._to_copy.default", "aten.to.dtype"}:
            dtype = node.kwargs.get("dtype", args[1] if len(args) > 1 else None)
            if dtype is None:
                raise TorchExportFrontendError(f"Cast node {node.name!r} has no dtype")
            self._emit(
                node,
                "Cast",
                (self._value(args[0], node=node),),
                attrs={"to": _dtype_name(dtype, where=f"Cast node {node.name!r}")},
            )
            return
        if target == "aten.conv2d.default":
            self._conv2d(node, convolution=False)
            return
        if target == "aten.convolution.default":
            self._conv2d(node, convolution=True)
            return
        if target == "aten.max_pool2d.default":
            kernel = _pair(args[1], where=f"MaxPool node {node.name!r} kernel")
            stride_arg = args[2] if len(args) > 2 and args[2] else kernel
            padding = _pair(args[3] if len(args) > 3 else 0, where=f"MaxPool node {node.name!r} padding")
            dilation = _pair(args[4] if len(args) > 4 else 1, where=f"MaxPool node {node.name!r} dilation")
            ceil_mode = bool(args[5] if len(args) > 5 else False)
            self._emit(
                node,
                "MaxPool2D",
                (self._value(args[0], node=node),),
                attrs={
                    "kernel_shape": kernel,
                    "strides": _pair(stride_arg, where=f"MaxPool node {node.name!r} stride"),
                    "pads": [padding[0], padding[1], padding[0], padding[1]],
                    "dilations": dilation,
                    "ceil_mode": ceil_mode,
                },
            )
            return
        if target == "aten.clamp.default":
            minimum = args[1] if len(args) > 1 else node.kwargs.get("min")
            maximum = args[2] if len(args) > 2 else node.kwargs.get("max")
            attrs = {
                name: value
                for name, value in (("min", minimum), ("max", maximum))
                if value is not None
            }
            if not attrs or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in attrs.values()
            ):
                raise TorchExportFrontendError(f"Clamp node {node.name!r} needs literal numeric bounds")
            self._emit(node, "Clamp", (self._value(args[0], node=node),), attrs=attrs)
            return
        if target in {
            "aten.clone.default",
            "aten.alias.default",
            "aten.detach.default",
            "aten.contiguous.default",
        }:
            self._emit(node, "Identity", (self._value(args[0], node=node),))
            return

        tensor_inputs: list[str] = []
        stack = list(args) + list(node.kwargs.values())
        while stack:
            item = stack.pop(0)
            if _is_fx_node(item):
                tensor_inputs.append(str(item.name))
            elif isinstance(item, (tuple, list)):
                stack[0:0] = list(item)
        qualified = f"torch::{_target_name(node.target)}"
        self.unsupported.append({"node": str(node.name), "op": qualified})
        if self.strict:
            return
        self._emit(
            node,
            qualified,
            tensor_inputs,
            attrs={"args": _json_value(node.args), "kwargs": _json_value(node.kwargs)},
        )

    def run(self, *, name: str | None) -> Graph:
        graph_nodes = self.graph_nodes
        placeholders = {str(node.name): node for node in graph_nodes if node.op == "placeholder"}
        self._placeholder_inputs(placeholders)
        output_node: Any | None = None
        for node in graph_nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                source_spec = _node_spec(node)
                self.source_value_specs[str(node.name)] = source_spec
                self.value_specs[str(node.name)] = source_spec
                self._translate(node)
            elif node.op == "output":
                output_node = node
            else:
                raise TorchExportFrontendError(
                    f"FX node {node.name!r} uses {node.op!r}; exported inference IR must "
                    "contain only placeholders, call_function nodes, and output"
                )

        if self.strict and self.unsupported:
            detail = ", ".join(
                f"{item['node']} ({item['op']})" for item in self.unsupported[:12]
            )
            if len(self.unsupported) > 12:
                detail += f", ... and {len(self.unsupported) - 12} more"
            raise TorchExportFrontendError(
                "ExportedProgram contains ATen/custom operations without a faithful "
                f"GAC lowering: {detail}"
            )
        if output_node is None:
            raise TorchExportFrontendError("ExportedProgram FX graph has no output node")

        signature = getattr(self.program, "graph_signature", None)
        output_specs = list(getattr(signature, "output_specs", ()))
        outputs: list[str] = []
        if output_specs:
            for output_spec in output_specs:
                kind = _kind_name(getattr(output_spec, "kind", ""))
                output_name = _argument_name(getattr(output_spec, "arg", None))
                if kind != "user_output":
                    raise TorchExportFrontendError(
                        f"graph output {output_name!r} has unsupported signature kind {kind!r}; "
                        "mutations, gradients, losses, and tokens are not inference outputs"
                    )
                if output_name is None:
                    raise TorchExportFrontendError("user output is not a tensor argument")
                outputs.append(output_name)
        else:
            def flatten(value: Any) -> None:
                if _is_fx_node(value):
                    outputs.append(str(value.name))
                elif isinstance(value, Mapping):
                    for item in value.values():
                        flatten(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        flatten(item)
                else:
                    raise TorchExportFrontendError(
                        f"ExportedProgram output contains non-tensor value {value!r}"
                    )

            flatten(output_node.args[0])
        if not outputs:
            raise TorchExportFrontendError("ExportedProgram has no user tensor outputs")

        range_constraints = {
            str(symbol): str(constraint)
            for symbol, constraint in dict(
                getattr(self.program, "range_constraints", {}) or {}
            ).items()
        }
        metadata = {
            "import": {
                "format": "torch.export",
                "source_operator_counts": dict(sorted(self.source_counts.items())),
                "canonical_operator_counts": dict(sorted(self.canonical_counts.items())),
                "unsupported_nodes": self.unsupported,
                "range_constraints": range_constraints,
                "strict": self.strict,
            }
        }
        graph_module = getattr(self.program, "graph_module", None)
        graph = Graph(
            name=name or type(graph_module).__name__ or "exported_program",
            inputs=self.inputs,
            outputs=tuple(outputs),
            nodes=self.nodes,
            constants=self.constants,
            value_specs=self.value_specs,
            metadata=metadata,
        )
        try:
            graph.validate()
            if self.strict:
                ShapeInferencePass().run(graph)
                mismatches: list[str] = []
                for value_name, source_spec in self.source_value_specs.items():
                    inferred_spec = graph.value_specs.get(value_name)
                    if inferred_spec is None:
                        mismatches.append(f"{value_name}: canonical value is missing")
                    elif (
                        inferred_spec.shape != source_spec.shape
                        or inferred_spec.dtype != source_spec.dtype
                    ):
                        mismatches.append(
                            f"{value_name}: torch.export={source_spec.shape}/{source_spec.dtype}, "
                            f"canonical={inferred_spec.shape}/{inferred_spec.dtype}"
                        )
                if mismatches:
                    detail = "; ".join(mismatches[:12])
                    if len(mismatches) > 12:
                        detail += f"; ... and {len(mismatches) - 12} more"
                    raise TorchExportFrontendError(
                        "ATen-to-Canonical legalization changed output shape or dtype: "
                        + detail
                    )
                graph.validate()
        except Exception as exc:
            raise TorchExportFrontendError(
                f"imported torch.export graph is invalid: {exc}"
            ) from exc
        return graph


def import_exported_program(
    program: Any,
    *,
    name: str | None = None,
    strict: bool = True,
) -> Graph:
    """Import an in-memory, inference-only ExportedProgram."""

    return _Importer(program, strict=strict).run(name=name)


def load_exported_program(
    path: str | Path,
    *,
    trusted: bool = False,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> Graph:
    """Load a ``.pt2`` archive and import it.

    Loading is opt-in because PyTorch archive compatibility and deserialization
    must be treated as a trusted-model boundary.  A supplied SHA-256 is checked
    before PyTorch is imported or the archive is opened.
    """

    model_path = Path(path)
    verified_archive: Any | None = None
    if expected_sha256 is not None:
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise TorchExportFrontendError(
                "expected_sha256 must contain exactly 64 hexadecimal characters"
            )
        digest_builder = hashlib.sha256()
        verified_archive = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
        try:
            with model_path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest_builder.update(block)
                    verified_archive.write(block)
        except OSError as exc:
            verified_archive.close()
            raise TorchExportFrontendError(
                f"cannot read torch.export archive {model_path}: {exc}"
            ) from exc
        digest = digest_builder.hexdigest()
        expected_digest = expected_sha256.lower()
        if digest != expected_digest:
            verified_archive.close()
            raise TorchExportFrontendError(
                f"torch.export archive SHA-256 mismatch: expected {expected_digest}, "
                f"got {digest}"
            )
        verified_archive.seek(0)
    if not trusted:
        if verified_archive is not None:
            verified_archive.close()
        raise TorchExportFrontendError(
            "loading a torch.export .pt2 archive requires trusted=True; verify the "
            "archive source or provide --trusted-model at the CLI boundary"
        )
    try:
        torch = _torch_module()
    except Exception:
        if verified_archive is not None:
            verified_archive.close()
        raise
    try:
        program = torch.export.load(
            verified_archive if verified_archive is not None else str(model_path)
        )
    except Exception as exc:
        raise TorchExportFrontendError(
            f"cannot load torch.export archive {model_path}: {exc}"
        ) from exc
    finally:
        if verified_archive is not None:
            verified_archive.close()
    return import_exported_program(program, name=model_path.stem, strict=strict)
