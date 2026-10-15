"""Small typed MLIR builder for portable scalar Vortex software kernels.

Arithmetic, math and SCF are lowered by upstream MLIR; only pointer access and
the Vortex launch ABI use LLVM dialect. All tensor indexing is explicit i32.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence


@dataclass(frozen=True)
class Value:
    name: str
    type: str

    def __str__(self) -> str:
        return self.name


TYPES = {"float32": "f32", "float16": "f16", "bool": "i8", "int32": "i32", "int64": "i64"}


class ScalarBuilder:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.serial = 0
        self.indent = 2

    def emit(self, line: str) -> None:
        self.lines.append("  " * self.indent + line)

    def fresh(self, type: str) -> Value:
        self.serial += 1
        return Value(f"%s{self.serial}", type)

    def result(self, expression: str, type: str) -> Value:
        value = self.fresh(type)
        self.emit(f"{value} = {expression}")
        return value

    def const(self, value: int | float, type: str = "i32") -> Value:
        type = TYPES.get(type, type)
        if type.startswith("f"):
            if math.isinf(float(value)):
                # MLIR hex floating literals denote the exact IEEE payload.
                literal = "0xFF800000" if value < 0 else "0x7F800000"
            else:
                literal = f"{float(value):.9e}"
        else:
            literal = str(int(value))
        return self.result(f"arith.constant {literal} : {type}", type)

    def binary(self, op: str, left: Value, right: Value | int | float) -> Value:
        if not isinstance(right, Value):
            right = self.const(right, left.type)
        if left.type != right.type:
            raise ValueError(f"MLIR operand types differ: {left.type}, {right.type}")
        floating = left.type.startswith("f")
        opcode = {"add": "addf" if floating else "addi", "sub": "subf" if floating else "subi",
                  "mul": "mulf" if floating else "muli", "div": "divf" if floating else "divsi",
                  "rem": "remf" if floating else "remsi", "and": "andi", "or": "ori"}[op]
        return self.result(f"arith.{opcode} {left}, {right} : {left.type}", left.type)

    def add(self, a: Value, b: Value | int | float) -> Value:
        return self.binary("add", a, b)

    def sub(self, a: Value, b: Value | int | float) -> Value:
        return self.binary("sub", a, b)

    def mul(self, a: Value, b: Value | int | float) -> Value:
        return self.binary("mul", a, b)

    def div(self, a: Value, b: Value | int | float) -> Value:
        return self.binary("div", a, b)

    def rem(self, a: Value, b: Value | int) -> Value:
        return self.binary("rem", a, b)

    def cmp(self, predicate: str, a: Value, b: Value | int | float) -> Value:
        if not isinstance(b, Value):
            b = self.const(b, a.type)
        floating = a.type.startswith("f")
        pred = {"lt": "olt" if floating else "slt", "le": "ole" if floating else "sle",
                "gt": "ogt" if floating else "sgt", "ge": "oge" if floating else "sge",
                "eq": "oeq" if floating else "eq", "ne": "une" if floating else "ne"}[predicate]
        return self.result(f"arith.cmp{'f' if floating else 'i'} {pred}, {a}, {b} : {a.type}", "i1")

    def select(self, cond: Value, a: Value, b: Value) -> Value:
        return self.result(f"arith.select {cond}, {a}, {b} : {a.type}", a.type)

    def cast(self, value: Value, type: str) -> Value:
        type = TYPES.get(type, type)
        if value.type == type:
            return value
        src_float, dst_float = value.type.startswith("f"), type.startswith("f")
        if src_float and dst_float:
            op = "extf" if int(type[1:]) > int(value.type[1:]) else "truncf"
        elif src_float:
            op = "fptosi"
        elif dst_float:
            op = "uitofp" if value.type in {"i1", "i8"} else "sitofp"
        else:
            op = ("extui" if value.type in {"i1", "i8"} else "extsi") if int(type[1:]) > int(value.type[1:]) else "trunci"
        return self.result(f"arith.{op} {value} : {value.type} to {type}", type)

    def math(self, name: str, value: Value) -> Value:
        return self.result(f"math.{name} {value} : {value.type}", value.type)

    def load(self, ptr: str, index: Value, dtype: str = "float32") -> Value:
        type = TYPES.get(dtype, dtype)
        address = self.result(f"llvm.getelementptr {ptr}[{index}] : (!llvm.ptr, i32) -> !llvm.ptr, {type}", "!llvm.ptr")
        value = self.result(f"llvm.load {address} : !llvm.ptr -> {type}", type)
        return self.cmp("ne", value, 0) if dtype == "bool" else value

    def f32(self, ptr: str, index: Value, spec) -> Value:
        return self.cast(self.load(ptr, index, spec.dtype), "f32")

    def store(self, value: Value, ptr: str, index: Value, dtype: str = "float32") -> None:
        value = self.cast(value, dtype)
        address = self.result(f"llvm.getelementptr {ptr}[{index}] : (!llvm.ptr, i32) -> !llvm.ptr, {value.type}", "!llvm.ptr")
        self.emit(f"llvm.store {value}, {address} : {value.type}, !llvm.ptr")

    def loop(self, count: int | Value, initial: Sequence[Value], body: Callable) -> tuple[Value, ...]:
        zero, one = self.const(0), self.const(1)
        end = count if isinstance(count, Value) else self.const(count)
        index = self.fresh("i32")
        args = [self.fresh(value.type) for value in initial]
        results = [self.fresh(value.type) for value in initial]
        assignment = ", ".join(str(x) for x in results) + " = " if results else ""
        iters = " iter_args(" + ", ".join(f"{a} = {v}" for a, v in zip(args, initial, strict=True)) + ")" if args else ""
        types = " -> (" + ", ".join(v.type for v in initial) + ")" if args else ""
        self.emit(f"{assignment}scf.for {index} = {zero} to {end} step {one}{iters}{types} : i32 {{")
        self.indent += 1
        yielded = tuple(body(index, tuple(args)))
        if yielded:
            self.emit("scf.yield " + ", ".join(str(v) for v in yielded) + " : " + ", ".join(v.type for v in yielded))
        self.indent -= 1
        self.emit("}")
        return tuple(results)

    def if_value(self, condition: Value, then: Callable[[], Value], otherwise: Callable[[], Value]) -> Value:
        # Capture the bodies first to infer their common result type.
        original = self.lines
        self.lines = []
        self.indent += 1
        true_value = then()
        true_lines = self.lines
        self.lines = []
        false_value = otherwise()
        false_lines = self.lines
        self.lines = original
        self.indent -= 1
        if true_value.type != false_value.type:
            raise ValueError("scf.if branches have different result types")
        result = self.fresh(true_value.type)
        self.emit(f"{result} = scf.if {condition} -> ({result.type}) {{")
        self.lines.extend(true_lines)
        self.emit(f"  scf.yield {true_value} : {result.type}")
        self.emit("} else {")
        self.lines.extend(false_lines)
        self.emit(f"  scf.yield {false_value} : {result.type}")
        self.emit("}")
        return result

    def coords(self, index: Value, shape: Sequence[int]) -> tuple[Value, ...]:
        stride = math.prod(shape)
        result = []
        for dimension in shape:
            stride //= dimension
            result.append(self.rem(self.div(index, stride), dimension))
        return tuple(result)

    def linear(self, coords: Sequence[Value], shape: Sequence[int]) -> Value:
        result = self.const(0)
        for coord, dimension in zip(coords, shape, strict=True):
            result = self.add(self.mul(result, dimension), coord)
        return result

    def broadcast_index(self, index: Value, output_shape: Sequence[int], input_shape: Sequence[int]) -> Value:
        coords = self.coords(index, output_shape)
        aligned = coords[len(output_shape) - len(input_shape):] if input_shape else ()
        return self.linear(tuple(self.const(0) if dim == 1 else coord for coord, dim in zip(aligned, input_shape, strict=True)), input_shape)
