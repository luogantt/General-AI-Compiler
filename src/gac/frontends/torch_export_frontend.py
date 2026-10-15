"""Import a functional ``torch.export.ExportedProgram`` into GAC IR.

PyTorch is deliberately imported only by :func:`load_exported_program`.
In-memory importing is implemented against the public ExportedProgram/FX
protocol so the core compiler remains usable without a Torch installation.
"""

# 阅读主线：
#   导出的 .pt2 文件 → load_exported_program() → ExportedProgram
#   → _Importer.run() 遍历 FX 图 → GAC 的 Graph / Node / TensorSpec。
# ExportedProgram 包含计算图、输入/输出签名及模型状态；仅有 state_dict 的
# .pth 权重文件并不包含这里所需的完整计算图，应先在 PyTorch 侧完成 export。
#
# 区分两类信息：
#   1. node.meta['val']：提供 shape/dtype，常为 FakeTensor，不作为真实矩阵读取。
#   2. state_dict / constants：包含真实权重，复制成 NumPy 后存入 graph.constants。
# 本文件建立模型语义与数据依赖；后续编译阶段规划 Buffer、目标后端生成 MLIR，
# Runtime 再分配真实设备内存并执行。
# 建议按 load_exported_program → import_exported_program → _Importer.run
# → _placeholder_inputs / _translate 的顺序阅读。

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gac.core.ir import Graph, Node, TensorSpec
from gac.compiler.passes import ShapeInferencePass


class TorchExportFrontendError(ValueError):
    """Raised when ExportedProgram semantics cannot be represented faithfully."""


# PyTorch 的类型别名 → GAC 规范类型名。前端可表达某种类型，
# 不代表每个目标后端都支持该类型；目标合法化还会继续检查。
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
    # 延迟导入：只有读取 PyTorch 导出归档时才需要安装 Torch。
    # 单独 import 本模块、或按 ExportedProgram/FX 协议导入内存对象时不触发它。
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchExportFrontendError(
            "torch.export import requires PyTorch; install a compatible PyTorch 2.x "
            "build or gac-heterogeneous[torch-export]"
        ) from exc


def _kind_name(kind: Any) -> str:
    # 兼容枚举对象和字符串表示，如 InputKind.PARAMETER → parameter。
    return str(getattr(kind, "name", kind)).split(".")[-1].lower()


def _argument_name(argument: Any) -> str | None:
    name = getattr(argument, "name", None)
    return str(name) if isinstance(name, str) and name else None


def _is_fx_node(value: Any) -> bool:
    # 按 FX 节点协议识别，不直接依赖 torch.fx.Node 的具体 Python 类型。
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
    # 静态维度保留整数；符号维度保留名字。保留符号不等于后端已支持动态形状。
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
    # TensorSpec 只记录规格，不保存数值、设备指针或 SRAM 分配结果。
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TorchExportFrontendError(
            f"{where} is not a tensor value or has no torch.export tensor metadata"
        )
    return TensorSpec(tuple(_dimension(dim) for dim in shape), _dtype_name(dtype, where=where))


def _node_spec(node: Any) -> TensorSpec:
    # 从 export 已推导的元数据取得输出规格，不运行这个算子。
    # 当前一个 FX 算子只接受单 Tensor 输出；多输出算子需要另外合法化。
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
    # 这里才读取真实权重：脱离 autograd，必要时从设备复制到 CPU，整理连续布局。
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
    # 独立复制，避免导入后原始模型修改权重时连带改变 GAC 的常量。
    return np.ascontiguousarray(result.copy())


def _target_name(target: Any) -> str:
    # 同时处理 ATen OpOverload 和普通 callable；保留 overload，便于区分语义。
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
    # 一个源算子可能拆成多个节点/中间值，新增名字不能覆盖已有值。
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
    # 形状、轴和步长等属性要求编译期可知；不把运行时 Tensor 当作整数列表。
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
    # 为诊断模式保存原始参数；FX 节点转成“值名字引用”，而不是序列化节点对象。
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
    # 一次导入的上下文。所有字典以张量/值的名字连接，尚未分配设备内存。
    def __init__(self, program: Any, *, strict: bool) -> None:
        self.program = program
        self.strict = strict
        self.constants: dict[str, np.ndarray[Any, Any]] = {}  # 真实权重及字面量数值。
        self.inputs: dict[str, TensorSpec] = {}  # 调用者每次推理需要提供的输入规格。
        # value_specs 会被规范图的 Shape 推导更新；source_value_specs 保存
        # FX 原始输出规格，用于在最后检查转换前后是否一致。
        self.value_specs: dict[str, TensorSpec] = {}
        self.source_value_specs: dict[str, TensorSpec] = {}
        self.nodes: list[Node] = []  # 规范算子序列；输入/输出通过字符串名字相连。
        self.occupied_values: set[str] = set()
        self.occupied_ids: set[str] = set()
        self.unsupported: list[dict[str, str]] = []
        self.source_counts: Counter[str] = Counter()
        self.canonical_counts: Counter[str] = Counter()

    @property
    def graph_nodes(self) -> list[Any]:
        # 按 FX 图自身的节点顺序读取；导出的数据依赖在此顺序中展开。
        graph_module = getattr(self.program, "graph_module", None)
        graph = getattr(graph_module, "graph", None)
        nodes = getattr(graph, "nodes", None)
        if nodes is None:
            raise TorchExportFrontendError("object is not a torch.export.ExportedProgram")
        return list(nodes)

    def _add_constant(self, base: str, value: Any, *, where: str) -> str:
        # 数值进入 constants，规格进入 value_specs，算子只持有返回的名字。
        name = _unique(base, self.occupied_values)
        array = _numpy_tensor(value, where=where)
        self.constants[name] = array
        self.value_specs[name] = TensorSpec(tuple(array.shape), array.dtype)
        return name

    def _scalar(self, value: Any, *, node: Any) -> str:
        # 把 x + 1 之类的字面量提升为零维常量 Tensor，使二元算子统一按值名连边。
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isrealobj(value)
        ):
            raise TorchExportFrontendError(
                f"FX node {node.name!r} expected a tensor or numeric scalar, got {value!r}"
            )
        # 使用源输出 dtype 构造标量；这并非完整的 ATen 类型提升实现，
        # 末尾还会对比规范图推导结果，拒绝已发现的 shape/dtype 语义差异。
        spec = _node_spec(node)
        name = _unique(f"{node.name}__scalar", self.occupied_values)
        self.constants[name] = np.asarray(value, dtype=spec.dtype)
        self.value_specs[name] = TensorSpec((), spec.dtype)
        return name

    def _value(self, value: Any, *, node: Any) -> str:
        # Tensor 参数引用其 FX 名字；数字参数则先转成可被图引用的常量。
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
        # 创建 GAC Node，不计算输出。attrs 记录轴、shape、epsilon 等语义属性；
        # source_nodes 保留来源，供错误定位、图检查和后续性能分析使用。
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
        # export 会把参数、注册 buffer 和用户输入都提升为 FX placeholder。
        # graph_signature 才说明各 placeholder 是运行时输入还是模型常量。
        signature = getattr(self.program, "graph_signature", None)
        input_specs = list(getattr(signature, "input_specs", ()))
        state = dict(getattr(self.program, "state_dict", {}) or {})
        constants = dict(getattr(self.program, "constants", {}) or {})
        if not input_specs:
            # 没有签名时，只能按协议把全部 placeholder 视为用户输入；不猜测权重。
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
                # 输入只登记 shape/dtype，真实输入数值由 Runtime 的调用者提供。
                spec = _node_spec(placeholders[name])
                self.inputs[name] = spec
                self.value_specs[name] = spec
                self.occupied_values.add(name)
                continue
            if kind in {"parameter", "buffer", "constant_tensor"}:
                # 这里的 buffer 指 PyTorch 注册状态，如 BatchNorm 的 running_mean；
                # 它不是后端 LMEM/SRAM 的 Buffer。推理导入时与参数一起作为常量保存。
                # target_name 通常是模块状态名；图内部仍按 placeholder 的 name 引用。
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
                self.constants[name] = array  # 最终原样交给 Graph.constants。
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
        # ATen Linear：X @ W.T + bias，W 按 [out_features, in_features] 保存。
        # 规范 MatMul 的右输入要求 [in_features, out_features]。
        if len(node.args) < 2:
            raise TorchExportFrontendError(f"linear node {node.name!r} has invalid arity")
        value = self._value(node.args[0], node=node)
        weight = self._value(node.args[1], node=node)
        bias = node.args[2] if len(node.args) >= 3 else None
        if weight in self.constants:
            # 常量权重在导入时完成转置并复制，图中无需再执行该 Transpose。
            transposed = self._add_constant(
                f"{weight}__{node.name}_t",
                self.constants[weight].T,
                where=f"linear weight {weight!r}",
            )
        else:
            # 非常量权重的数值到运行时才确定，必须显式插入 Transpose 节点。
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
            # 有 bias 时将一个 Linear 拆为 MatMul + Add，后续 Pass 可再决定融合。
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
        # Add/Sub 的 alpha 和 Div 的 rounding_mode 会改变数值语义，不能忽略。
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
        # 两种 ATen 卷积签名的参数位置不同；统一为规范 Conv2D 属性。
        # 当前表达普通 2D 卷积，转置卷积和非零 output_padding 明确拒绝。
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
        # ATen 的两个对称 padding 值展开为 [top, left, bottom, right]。
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
        # 算子合法化分发表：依据完整 ATen target/overload，而不只看 Python 函数名。
        # 一个分支可发出一个规范算子，也可像 Linear 一样拆成多个规范算子。
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
            # 当前规范 GELU 使用 tanh 近似；不能把精确 erf 版本悄悄换成近似版本。
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
            # 记录形状变换语义；是否可零拷贝由后续布局/内存分析决定。
            shape = _literal_ints(args[1], where=f"Reshape node {node.name!r} shape")
            self._emit(node, "Reshape", (self._value(args[0], node=node),), attrs={"shape": shape})
            return
        if target in {
            "aten.flatten.using_ints",
            "aten.squeeze.dim",
            "aten.squeeze.dims",
            "aten.unsqueeze.default",
        }:
            # 这些操作只改变形状，直接使用 FX 输出元数据构造静态 Reshape。
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
            # 保存轴排列，前端不在这里搬运输入矩阵的数据。
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
            # Expand 中的 -1 表示保留对应输入维度，先解析成明确的目标 shape。
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
            # starts/ends/axes/steps 显式保留，不能把 Slice 误当成任意连续视图。
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
            # 归约轴和 keepdims 影响输出形状；dtype override 还影响累加精度。
            # 当前没有替这种 override 插入 Cast，所以遇到它就报错。
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
            # 保留 Softmax 的轴，拒绝当前规范算子未表达的输入/计算精度转换。
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
            # normalized_shape 的长度确定尾部归约从哪一轴开始。
            # 当前仅接收 FP32，避免丢失 ATen 的低精度存储/累加精度约定。
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
            # dtype 转换成为显式 Cast 节点，后端才能看到它的精度和存储影响。
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
            # 当前 Clamp 只接收字面量数值边界，不把 Tensor 边界当作静态属性。
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
            # 函数式推理图中保留数值依赖；不在规范图里模拟 autograd 或存储别名。
            self._emit(node, "Identity", (self._value(args[0], node=node),))
            return

        # 未覆盖的算子仍记录其原始限定名。strict=True 稍后汇总报错；
        # strict=False 保存 torch::... 节点用于诊断，不代表后端可以执行或自动回退。
        # 非严格模式也不会跳过前面已识别算子的属性、精度等语义检查。
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
        # 第一步：先处理所有 placeholder，区分真实输入与提升后的模型权重。
        graph_nodes = self.graph_nodes
        placeholders = {str(node.name): node for node in graph_nodes if node.op == "placeholder"}
        self._placeholder_inputs(placeholders)
        output_node: Any | None = None
        # 第二步：按 FX 顺序转换算子。这里只生成图，不执行一次 PyTorch forward。
        for node in graph_nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                # 先留存源输出规格，再生成规范节点，最后进行独立 Shape 推导对齐。
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

        # 收齐未知算子后一次报告，便于定位模型还有哪些语义缺口。
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

        # 第三步：确定公开输出。签名里的状态修改、梯度等不能混作推理 Tensor 输出。
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
            # 协议对象缺少输出签名时，按容器顺序展平 Tensor 叶子；不保留原 Python 容器。
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

        # 符号维度的范围仅写入来源元数据；这里没有生成动态分配或运行时守卫。
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
        # 第四步：把导入上下文交付为统一 Graph。
        # 数值在 constants，规格在 inputs/value_specs，依赖在 nodes 的输入/输出名字中。
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
                # 按 GAC 算子语义重新推导，并逐值比较 FX 原始 shape/dtype。
                # 此检查验证规格一致性，不等同于逐元素数值正确性验证。
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

    # 内存入口：已持有 ExportedProgram 的调用者直接使用，无需再读写 .pt2 文件。
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
    # 文件入口：按当前顺序先做可选 SHA-256 校验，再检查 trusted，最后才加载 Torch。
    if expected_sha256 is not None:
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise TorchExportFrontendError(
                "expected_sha256 must contain exactly 64 hexadecimal characters"
            )
        digest_builder = hashlib.sha256()
        # 同一份字节既用于哈希也交给 torch.export.load，避免校验后重读路径时内容改变。
        # 小归档暂存在内存，超过 64 MiB 后由 SpooledTemporaryFile 转存临时文件。
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
        # 摘要匹配确认内容一致，不能替代调用方显式声明来源可信。
        if verified_archive is not None:
            verified_archive.close()
        raise TorchExportFrontendError(
            "loading a torch.export .pt2 archive requires trusted=True; verify the "
            "archive source or provide --trusted-model at the CLI boundary"
        )
    try:
        # 只有文件反序列化需要真正导入 PyTorch；它不参与后面的设备 Codegen。
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
    # 归档加载完成后，与内存入口共用同一条 FX → Graph 转换流程。
    return import_exported_program(program, name=model_path.stem, strict=strict)
