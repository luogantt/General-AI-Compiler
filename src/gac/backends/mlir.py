"""Deterministic lowering from the canonical GAC graph to standard MLIR.

This backend deliberately emits only registered upstream dialects.  The first
vertical slice targets the tensor/linalg level so that ``mlir-opt`` can parse,
verify, and transform the result without a GAC dialect plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, Sequence

import numpy as np

from ..errors import BackendError, GACError
from ..ir import Graph, Node, TensorSpec
from ..passes import (
    AddReLUFusionPass,
    MatMulBiasGELUFusionPass,
    PassManager,
    ShapeInferencePass,
    default_pass_manager,
)


TARGET = "mlir-linalg"
DEVICE_TARGET_INTENT = "cuda-sm86"
SUPPORTED_OPS = frozenset(
    {
        "MatMul",
        "Add",
        "AddReLU",
        "ReLU",
        "Identity",
        "LayerNorm",
        "GELU",
        "MatMulBiasGELU",
    }
)


@dataclass(frozen=True, slots=True)
class MlirProgram:
    """A generated textual MLIR compilation artifact.

    ``source`` is intentionally kept as text: callers can pass it directly to
    the locally built ``mlir-opt`` or save it as a reviewable ``.mlir`` file.
    No MLIR shared library is loaded by this Python package, so external
    ``mlir-opt`` verification remains a distinct toolchain step.
    """

    graph: Graph
    reference_graph: Graph
    source: str
    pass_reports: tuple[dict[str, Any], ...]

    @property
    def text(self) -> str:
        """Alias used by integrations that call a textual module ``text``."""

        return self.source

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "compiler_backend": "mlir-linalg",
            "target_intent": DEVICE_TARGET_INTENT,
            "planned_codegen_backend": "mlir-nvvm",
            "lowering_stage": "tensor",
            "executable": False,
            "artifact_kind": "textual-mlir-module",
            "dialects": ["builtin", "func", "tensor", "linalg", "arith", "math"],
            "name": self.graph.name,
            "inputs": {
                name: spec.to_dict() for name, spec in self.graph.inputs.items()
            },
            "outputs": {
                name: self.graph.value_specs[name].to_dict()
                for name in self.graph.outputs
            },
            "pass_reports": list(self.pass_reports),
        }


def _reject(node: Node, reason: str) -> None:
    raise BackendError(f"{TARGET}: node {node.id!r} ({node.op}): {reason}")


def _check_specs(graph: Graph) -> None:
    for name, spec in graph.value_specs.items():
        if spec.dtype != "float32":
            raise BackendError(
                f"{TARGET}: value {name!r}: requires float32, got {spec.dtype}"
            )
        if spec.layout != "contiguous":
            raise BackendError(
                f"{TARGET}: value {name!r}: requires contiguous layout, "
                f"got {spec.layout!r}"
            )
        if not spec.is_static or any(dim <= 0 for dim in spec.shape):
            raise BackendError(
                f"{TARGET}: value {name!r}: requires static positive dimensions"
            )
    for name, value in graph.constants.items():
        if not value.flags.c_contiguous:
            raise BackendError(
                f"{TARGET}: constant {name!r}: requires C-contiguous storage"
            )
        if not np.all(np.isfinite(value)):
            raise BackendError(
                f"{TARGET}: constant {name!r}: requires finite float32 values"
            )


def _check_node(node: Node, specs: dict[str, TensorSpec]) -> None:
    if node.op not in SUPPORTED_OPS:
        _reject(node, "operation is not supported by the standard-dialect slice")
    if len(node.outputs) != 1:
        _reject(node, "exactly one output is required")
    allowed_attrs = {
        "LayerNorm": {"axis", "epsilon"},
        "GELU": {"approximation"},
        "MatMulBiasGELU": {"approximation"},
    }.get(node.op, set())
    unexpected = set(node.attrs) - allowed_attrs
    if unexpected:
        _reject(node, f"unsupported attributes: {sorted(unexpected)}")

    expected_arity = {
        "MatMul": 2,
        "Add": 2,
        "AddReLU": 2,
        "ReLU": 1,
        "Identity": 1,
        "GELU": 1,
        "MatMulBiasGELU": 3,
    }.get(node.op)
    if node.op == "LayerNorm":
        if len(node.inputs) not in {1, 2, 3}:
            _reject(node, f"requires one to three inputs, got {len(node.inputs)}")
    elif len(node.inputs) != expected_arity:
        _reject(node, f"requires {expected_arity} inputs, got {len(node.inputs)}")

    inputs = [specs[name] for name in node.inputs]
    output = specs[node.outputs[0]]
    if node.op in {"MatMul", "MatMulBiasGELU"}:
        lhs, rhs = inputs[:2]
        if lhs.rank < 2:
            _reject(node, "left MatMul input must have rank >= 2")
        if rhs.rank != 2:
            _reject(node, "right MatMul input must have rank 2")
        if lhs.shape[-1] != rhs.shape[0]:
            _reject(node, "MatMul contracting dimensions do not match")
        expected = (*lhs.shape[:-1], rhs.shape[1])
        if output.shape != expected:
            _reject(node, f"MatMul output must have shape {expected}")
    if node.op in {"Add", "AddReLU"}:
        try:
            broadcast = np.broadcast_shapes(inputs[0].shape, inputs[1].shape)
        except ValueError:
            _reject(node, "inputs are not NumPy-broadcast compatible")
        if output.shape != tuple(broadcast):
            _reject(node, f"broadcast output must have shape {tuple(broadcast)}")
    if node.op in {"ReLU", "Identity", "GELU"} and output.shape != inputs[0].shape:
        _reject(node, "output shape must match the input")
    if node.op in {"GELU", "MatMulBiasGELU"}:
        if node.attrs.get("approximation", "tanh") != "tanh":
            _reject(node, "only tanh GELU is supported")
    if node.op == "MatMulBiasGELU":
        bias = inputs[2]
        try:
            broadcast = np.broadcast_shapes(output.shape, bias.shape)
        except ValueError:
            _reject(node, "bias is not broadcast compatible with MatMul output")
        if tuple(broadcast) != output.shape:
            _reject(node, "bias broadcast would change the MatMul output shape")
    if node.op == "LayerNorm":
        value = inputs[0]
        axis = node.attrs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            _reject(node, "axis must be an integer selecting the last dimension")
        if axis not in {-1, value.rank - 1}:
            _reject(node, "only the last normalization axis is supported")
        if value.rank < 1:
            _reject(node, "normalization input must have rank >= 1")
        for parameter in inputs[1:]:
            try:
                parameter_output = np.broadcast_shapes(
                    (value.shape[-1],), parameter.shape
                )
            except ValueError:
                _reject(
                    node,
                    "scale and bias must broadcast to shape [last_dimension]",
                )
            if tuple(parameter_output) != (value.shape[-1],):
                _reject(
                    node,
                    "scale and bias must broadcast to shape [last_dimension]",
                )
        if output.shape != value.shape:
            _reject(node, "output shape must match the normalized input")
        epsilon = node.attrs.get("epsilon", 1e-5)
        with np.errstate(over="ignore", invalid="ignore"):
            epsilon_f32 = (
                np.float32(epsilon)
                if isinstance(epsilon, (int, float)) and not isinstance(epsilon, bool)
                else np.float32(np.nan)
            )
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon))
            or not np.isfinite(epsilon_f32)
            or epsilon_f32 <= 0
        ):
            _reject(node, "epsilon must be finite and positive in float32")


def _float_literal(value: float | np.floating[Any]) -> str:
    number = float(np.float32(value))
    if not math.isfinite(number):
        raise BackendError(f"{TARGET}: cannot encode non-finite float constant")
    text = format(number, ".9g")
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _dense_literal(value: np.ndarray) -> str:
    flat = value.reshape(-1)
    # Numeric equality would collapse a mixture of +0.0 and -0.0 into one
    # splat and silently change its IEEE-754 bit pattern.
    bits = flat.view(np.uint32)
    if flat.size and np.all(bits == bits[0]):
        return _float_literal(flat[0])

    def nested(array: np.ndarray) -> str:
        if array.ndim == 0:
            return _float_literal(array.item())
        return "[" + ", ".join(nested(item) for item in array) + "]"

    return nested(value)


def _tensor_type(spec_or_shape: TensorSpec | Sequence[int]) -> str:
    shape = spec_or_shape.shape if isinstance(spec_or_shape, TensorSpec) else spec_or_shape
    if not shape:
        return "tensor<f32>"
    return "tensor<" + "x".join(str(int(dim)) for dim in shape) + "xf32>"


def _symbol_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_$.]", "_", name)
    if not clean or not re.match(r"[A-Za-z_$]", clean[0]):
        clean = "gac_" + clean
    return clean


def _location(node: Node, suffix: str = "") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]", "_", node.id)
    return f'loc("{safe}{suffix}")'


def _affine_map(loop_rank: int, results: Sequence[str]) -> str:
    dims = ", ".join(f"d{index}" for index in range(loop_rank))
    result = ", ".join(results)
    return f"affine_map<({dims}) -> ({result})>"


def _identity_map(rank: int) -> str:
    return _affine_map(rank, [f"d{index}" for index in range(rank)])


def _broadcast_map(input_shape: Sequence[int], output_shape: Sequence[int]) -> str:
    """Return a static NumPy-broadcast indexing map for ``linalg.generic``."""

    if len(input_shape) > len(output_shape):
        raise ValueError("input rank exceeds output rank")
    offset = len(output_shape) - len(input_shape)
    results: list[str] = []
    for index, size in enumerate(input_shape):
        output_index = offset + index
        output_size = output_shape[output_index]
        if size == output_size:
            results.append(f"d{output_index}")
        elif size == 1:
            results.append("0")
        else:
            raise ValueError("incompatible broadcast dimension")
    return _affine_map(len(output_shape), results)


class _Emitter:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self.lines: list[str] = []
        self.values: dict[str, str] = {}
        self.next_value = 0

    def new_value(self) -> str:
        result = f"%v{self.next_value}"
        self.next_value += 1
        return result

    def line(self, text: str, indent: int = 2) -> None:
        self.lines.append(" " * indent + text)

    def empty(self, shape: Sequence[int], *, indent: int = 4) -> str:
        value = self.new_value()
        self.line(f"{value} = tensor.empty() : {_tensor_type(shape)}", indent)
        return value

    def scalar(self, value: float, *, indent: int = 4) -> str:
        result = self.new_value()
        self.line(
            f"{result} = arith.constant {_float_literal(value)} : f32", indent
        )
        return result

    def fill(
        self, scalar: str, empty: str, shape: Sequence[int], *, indent: int = 4
    ) -> str:
        result = self.new_value()
        tensor_type = _tensor_type(shape)
        self.line(
            f"{result} = linalg.fill ins({scalar} : f32) "
            f"outs({empty} : {tensor_type}) -> {tensor_type}",
            indent,
        )
        return result

    def generic(
        self,
        node: Node,
        inputs: Sequence[tuple[str, TensorSpec]],
        output_shape: Sequence[int],
        scalar_body: Callable[[list[str], str], list[str]],
        *,
        location_suffix: str = "",
        input_maps: Sequence[str] | None = None,
    ) -> str:
        output_type = _tensor_type(output_shape)
        init = self.empty(output_shape)
        result = self.new_value()
        if input_maps is None:
            maps = [
                _broadcast_map(spec.shape, output_shape) for _, spec in inputs
            ]
        else:
            if len(input_maps) != len(inputs):
                raise AssertionError("one indexing map is required per input")
            maps = list(input_maps)
        maps.append(_identity_map(len(output_shape)))
        iterators = ", ".join('"parallel"' for _ in output_shape)
        self.line(f"{result} = linalg.generic {{", 4)
        self.line(f"indexing_maps = [{', '.join(maps)}],", 6)
        self.line(f"iterator_types = [{iterators}]", 6)
        input_values = ", ".join(value for value, _ in inputs)
        input_types = ", ".join(_tensor_type(spec) for _, spec in inputs)
        self.line(
            f"}} ins({input_values} : {input_types}) "
            f"outs({init} : {output_type}) {{",
            4,
        )
        block_inputs = [f"%in{index}" for index in range(len(inputs))]
        arguments = ", ".join(
            [*(f"{value}: f32" for value in block_inputs), "%out: f32"]
        )
        self.line(f"^bb0({arguments}):", 4)
        for body_line in scalar_body(block_inputs, "%out"):
            self.line(body_line, 6)
        self.line(f"}} -> {output_type} {_location(node, location_suffix)}", 4)
        return result

    def emit_matmul_values(
        self, node: Node, lhs_name: str, rhs_name: str, *, suffix: str = ""
    ) -> str:
        lhs_spec = self.graph.value_specs[lhs_name]
        rhs_spec = self.graph.value_specs[rhs_name]
        lhs = self.values[lhs_name]
        rhs = self.values[rhs_name]
        matrix_lhs_shape: tuple[int, int]
        if lhs_spec.rank == 2:
            matrix_lhs = lhs
            matrix_lhs_shape = (int(lhs_spec.shape[0]), int(lhs_spec.shape[1]))
        else:
            rows = math.prod(int(dim) for dim in lhs_spec.shape[:-1])
            matrix_lhs_shape = (rows, int(lhs_spec.shape[-1]))
            matrix_lhs = self.new_value()
            leading = ", ".join(str(index) for index in range(lhs_spec.rank - 1))
            reassociation = f"[[{leading}], [{lhs_spec.rank - 1}]]"
            self.line(
                f"{matrix_lhs} = tensor.collapse_shape {lhs} {reassociation} : "
                f"{_tensor_type(lhs_spec)} into {_tensor_type(matrix_lhs_shape)} "
                f"{_location(node, suffix + '.collapse')}",
                4,
            )

        matrix_output_shape = (matrix_lhs_shape[0], int(rhs_spec.shape[1]))
        empty = self.empty(matrix_output_shape)
        zero = self.scalar(0.0)
        init = self.fill(zero, empty, matrix_output_shape)
        matrix_result = self.new_value()
        self.line(
            f"{matrix_result} = linalg.matmul "
            f"ins({matrix_lhs}, {rhs} : {_tensor_type(matrix_lhs_shape)}, "
            f"{_tensor_type(rhs_spec)}) "
            f"outs({init} : {_tensor_type(matrix_output_shape)}) "
            f"-> {_tensor_type(matrix_output_shape)} {_location(node, suffix + '.matmul')}",
            4,
        )
        if lhs_spec.rank == 2:
            return matrix_result

        output_shape = (*lhs_spec.shape[:-1], rhs_spec.shape[1])
        expanded = self.new_value()
        leading = ", ".join(str(index) for index in range(lhs_spec.rank - 1))
        reassociation = f"[[{leading}], [{lhs_spec.rank - 1}]]"
        shape_values = ", ".join(str(int(dim)) for dim in output_shape)
        self.line(
            f"{expanded} = tensor.expand_shape {matrix_result} {reassociation} "
            f"output_shape [{shape_values}] : {_tensor_type(matrix_output_shape)} "
            f"into {_tensor_type(output_shape)} {_location(node, suffix + '.expand')}",
            4,
        )
        return expanded

    @staticmethod
    def add_body(inputs: list[str], _out: str) -> list[str]:
        return [
            f"%scalar0 = arith.addf {inputs[0]}, {inputs[1]} : f32",
            "linalg.yield %scalar0 : f32",
        ]

    @staticmethod
    def add_relu_body(inputs: list[str], _out: str) -> list[str]:
        return [
            "%zero = arith.constant 0.0 : f32",
            f"%sum = arith.addf {inputs[0]}, {inputs[1]} : f32",
            # maximumf propagates NaN and orders -0.0 below +0.0, matching
            # the canonical NumPy evaluator's maximum(value, +0.0).
            "%relu = arith.maximumf %sum, %zero : f32",
            "linalg.yield %relu : f32",
        ]

    @staticmethod
    def relu_body(inputs: list[str], _out: str) -> list[str]:
        return [
            "%zero = arith.constant 0.0 : f32",
            f"%relu = arith.maximumf {inputs[0]}, %zero : f32",
            "linalg.yield %relu : f32",
        ]

    @staticmethod
    def gelu_body(inputs: list[str], _out: str) -> list[str]:
        # tanh GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        return [
            "%half = arith.constant 0.5 : f32",
            "%one = arith.constant 1.0 : f32",
            "%alpha = arith.constant 0.797884583 : f32",
            "%beta = arith.constant 0.044715 : f32",
            f"%x2 = arith.mulf {inputs[0]}, {inputs[0]} : f32",
            f"%x3 = arith.mulf %x2, {inputs[0]} : f32",
            "%scaled_x3 = arith.mulf %beta, %x3 : f32",
            f"%inner = arith.addf {inputs[0]}, %scaled_x3 : f32",
            "%tanh_input = arith.mulf %alpha, %inner : f32",
            "%tanh = math.tanh %tanh_input : f32",
            "%shifted = arith.addf %one, %tanh : f32",
            f"%half_x = arith.mulf %half, {inputs[0]} : f32",
            "%gelu = arith.mulf %half_x, %shifted : f32",
            "linalg.yield %gelu : f32",
        ]

    def emit_layer_norm(self, node: Node) -> str:
        value_name = node.inputs[0]
        value_spec = self.graph.value_specs[value_name]
        value = self.values[value_name]
        input_shape = tuple(int(dim) for dim in value_spec.shape)
        prefix_shape = input_shape[:-1]
        input_type = _tensor_type(input_shape)
        prefix_type = _tensor_type(prefix_shape)
        axis = len(input_shape) - 1

        zero = self.scalar(0.0)
        mean_empty = self.empty(prefix_shape)
        mean_init = self.fill(zero, mean_empty, prefix_shape)
        total = self.new_value()
        self.line(
            f"{total} = linalg.reduce ins({value} : {input_type}) "
            f"outs({mean_init} : {prefix_type}) dimensions = [{axis}]",
            4,
        )
        self.line("(%in: f32, %acc: f32) {", 6)
        self.line("%sum = arith.addf %acc, %in : f32", 8)
        self.line("linalg.yield %sum : f32", 8)
        self.line(f"}} {_location(node, '.mean_reduce')}", 6)

        inv_width = np.float32(1.0 / input_shape[-1])
        centered = self.generic(
            node,
            [(value, value_spec), (total, TensorSpec(prefix_shape, "float32"))],
            input_shape,
            lambda inputs, _out: [
                f"%inv_width = arith.constant {_float_literal(inv_width)} : f32",
                f"%mean = arith.mulf {inputs[1]}, %inv_width : f32",
                f"%centered = arith.subf {inputs[0]}, %mean : f32",
                "linalg.yield %centered : f32",
            ],
            location_suffix=".center",
            input_maps=[
                _identity_map(len(input_shape)),
                _affine_map(
                    len(input_shape),
                    [f"d{index}" for index in range(len(prefix_shape))],
                ),
            ],
        )

        squared = self.generic(
            node,
            [(centered, value_spec)],
            input_shape,
            lambda inputs, _out: [
                f"%square = arith.mulf {inputs[0]}, {inputs[0]} : f32",
                "linalg.yield %square : f32",
            ],
            location_suffix=".square",
        )

        variance_empty = self.empty(prefix_shape)
        variance_init = self.fill(zero, variance_empty, prefix_shape)
        square_total = self.new_value()
        self.line(
            f"{square_total} = linalg.reduce ins({squared} : {input_type}) "
            f"outs({variance_init} : {prefix_type}) dimensions = [{axis}]",
            4,
        )
        self.line("(%in: f32, %acc: f32) {", 6)
        self.line("%sum = arith.addf %acc, %in : f32", 8)
        self.line("linalg.yield %sum : f32", 8)
        self.line(f"}} {_location(node, '.variance_reduce')}", 6)

        epsilon = np.float32(node.attrs.get("epsilon", 1e-5))
        prefix_spec = TensorSpec(prefix_shape, "float32")
        final_inputs: list[tuple[str, TensorSpec]] = [
            (centered, value_spec),
            (square_total, prefix_spec),
        ]
        final_maps = [
            _identity_map(len(input_shape)),
            _affine_map(
                len(input_shape),
                [f"d{index}" for index in range(len(prefix_shape))],
            ),
        ]
        for parameter_name in node.inputs[1:]:
            parameter_spec = self.graph.value_specs[parameter_name]
            final_inputs.append((self.values[parameter_name], parameter_spec))
            final_maps.append(_broadcast_map(parameter_spec.shape, input_shape))

        def normalize_body(inputs: list[str], _out: str) -> list[str]:
            lines = [
                f"%inv_width = arith.constant {_float_literal(inv_width)} : f32",
                f"%epsilon = arith.constant {_float_literal(epsilon)} : f32",
                f"%variance = arith.mulf {inputs[1]}, %inv_width : f32",
                "%adjusted = arith.addf %variance, %epsilon : f32",
                "%inv_stddev = math.rsqrt %adjusted : f32",
                f"%normalized = arith.mulf {inputs[0]}, %inv_stddev : f32",
            ]
            result = "%normalized"
            if len(inputs) >= 3:
                lines.append(f"%scaled = arith.mulf {result}, {inputs[2]} : f32")
                result = "%scaled"
            if len(inputs) >= 4:
                lines.append(f"%shifted = arith.addf {result}, {inputs[3]} : f32")
                result = "%shifted"
            lines.append(f"linalg.yield {result} : f32")
            return lines

        return self.generic(
            node,
            final_inputs,
            input_shape,
            normalize_body,
            location_suffix=".normalize",
            input_maps=final_maps,
        )

    def emit_node(self, node: Node) -> None:
        output_name = node.outputs[0]
        output_spec = self.graph.value_specs[output_name]
        output_shape = tuple(int(dim) for dim in output_spec.shape)
        if node.op == "Identity":
            self.values[output_name] = self.values[node.inputs[0]]
            return
        if node.op == "MatMul":
            self.values[output_name] = self.emit_matmul_values(
                node, node.inputs[0], node.inputs[1]
            )
            return
        if node.op in {"Add", "AddReLU"}:
            body = self.add_body if node.op == "Add" else self.add_relu_body
            self.values[output_name] = self.generic(
                node,
                [
                    (self.values[name], self.graph.value_specs[name])
                    for name in node.inputs
                ],
                output_shape,
                body,
            )
            return
        if node.op in {"ReLU", "GELU"}:
            body = self.relu_body if node.op == "ReLU" else self.gelu_body
            input_name = node.inputs[0]
            self.values[output_name] = self.generic(
                node,
                [(self.values[input_name], self.graph.value_specs[input_name])],
                output_shape,
                body,
            )
            return
        if node.op == "LayerNorm":
            self.values[output_name] = self.emit_layer_norm(node)
            return
        if node.op == "MatMulBiasGELU":
            matmul = self.emit_matmul_values(
                node, node.inputs[0], node.inputs[1], suffix=".fused"
            )
            bias_name = node.inputs[2]
            biased = self.generic(
                node,
                [
                    (matmul, output_spec),
                    (self.values[bias_name], self.graph.value_specs[bias_name]),
                ],
                output_shape,
                self.add_body,
                location_suffix=".bias",
            )
            self.values[output_name] = self.generic(
                node,
                [(biased, output_spec)],
                output_shape,
                self.gelu_body,
                location_suffix=".gelu",
            )
            return
        _reject(node, "internal lowering dispatch is missing")

    def emit(self) -> str:
        arguments: list[str] = []
        for index, (name, spec) in enumerate(self.graph.inputs.items()):
            value = f"%arg{index}"
            self.values[name] = value
            arguments.append(f"{value}: {_tensor_type(spec)}")

        output_types = [
            _tensor_type(self.graph.value_specs[name]) for name in self.graph.outputs
        ]
        if len(output_types) == 1:
            result_signature = output_types[0]
        else:
            result_signature = "(" + ", ".join(output_types) + ")"

        self.lines.append("module {")
        self.line(
            f"func.func @{_symbol_name(self.graph.name)}({', '.join(arguments)}) "
            f"-> {result_signature} {{",
            2,
        )
        for name in sorted(self.graph.constants):
            value = self.graph.constants[name]
            result = self.new_value()
            self.values[name] = result
            tensor_type = _tensor_type(self.graph.value_specs[name])
            self.line(
                f"{result} = arith.constant dense<{_dense_literal(value)}> : "
                f"{tensor_type}",
                4,
            )
        for node in self.graph.nodes:
            self.emit_node(node)

        results = ", ".join(self.values[name] for name in self.graph.outputs)
        self.line(
            f"func.return {results} : {', '.join(output_types)}",
            4,
        )
        self.line("}", 2)
        self.lines.append("}")
        return "\n".join(self.lines) + "\n"


def compile_mlir(graph: Graph, *, fuse: bool = True) -> MlirProgram:
    """Optimize ``graph`` and lower it to registered standard MLIR dialects.

    The input graph is never mutated.  This phase performs no device access and
    does not require Python MLIR bindings; external verification is done by
    piping ``source`` through the local ``mlir-opt`` executable.
    """

    try:
        reference = PassManager([ShapeInferencePass()]).run(graph).graph
    except GACError as exc:
        raise BackendError(f"{TARGET}: canonical graph preparation failed: {exc}") from exc
    _check_specs(reference)
    for node in reference.nodes:
        _check_node(node, reference.value_specs)

    passes = default_pass_manager().passes
    if not fuse:
        passes = tuple(
            graph_pass
            for graph_pass in passes
            if not isinstance(
                graph_pass, (MatMulBiasGELUFusionPass, AddReLUFusionPass)
            )
        )
    try:
        result = PassManager(passes).run(reference)
    except GACError as exc:
        raise BackendError(f"{TARGET}: optimization failed: {exc}") from exc
    _check_specs(result.graph)
    for node in result.graph.nodes:
        _check_node(node, result.graph.value_specs)
    source = _Emitter(result.graph).emit()
    return MlirProgram(
        graph=result.graph,
        reference_graph=reference,
        source=source,
        pass_reports=tuple(report.to_dict() for report in result.reports),
    )


def export_mlir(graph: Graph, *, fuse: bool = True) -> str:
    """Convenience API returning only the textual MLIR module."""

    return compile_mlir(graph, fuse=fuse).source
