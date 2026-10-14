"""BEVFormer FFN lowering from verified MLIR to Vortex 3.0.

This deliberately narrow vertical slice accepts the canonical BEVFormer FFN
graph, verifies its tensor/linalg form with an external ``mlir-opt``, audits
the operation and buffer contracts, and emits a ``vortex2`` host runner.

Two device boundaries are kept explicit.  ``cpp-template`` is the validated
compatibility bridge that lets VOLT lower generated Vortex C++.  The newer
``direct-mlir`` mode materializes LLVM-dialect device kernels, including the
Vortex launch ABI, and uses one pinned Vortex LLVM 20.1.8 tree for
``mlir-opt -> mlir-translate -> VOLT -> ELF``.  ``vxbin.py`` packages the linked
ELF in both modes.  The direct mode removes device C++ without claiming that
upstream Vortex ships a general Vortex MLIR dialect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from ..errors import BackendError
from ..ir import Graph
from ..mlir_toolchain import (
    DEFAULT_TIMEOUT_SECONDS,
    probe_mlir_opt_source_revision,
    probe_mlir_opt_version,
    verify_mlir,
)
from ..runtime import reference_run
from .mlir import MlirProgram, compile_mlir
from .vortex_buffered_mlir import (
    VortexBufferPlan,
    audit_bevformer_ffn_buffered_device_mlir,
    emit_bevformer_ffn_buffered_device_mlir,
)


TARGET = "vortex-v3"
ARTIFACT_SCHEMA_VERSION = 1
DIRECT_MLIR_LOWERING = "direct-mlir"
LEGACY_CPP_LOWERING = "cpp-template"
VOLT_LLVM_VERSION = "20.1.8"
VOLT_LLVM_REVISION = "b2ecfe086df2d00233a87f3d3ebe6a74aba129e2"
_REQUIRED_CONSTANTS = (
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
    "norm_weight",
    "norm_bias",
)


def _direct_argument_abi() -> dict[str, Any]:
    def field(
        name: str,
        offset_bytes: int,
        llvm_type: str,
        size_bytes: int,
        alignment_bytes: int,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "offset_bytes": offset_bytes,
            "llvm_type": llvm_type,
            "size_bytes": size_bytes,
            "alignment_bytes": alignment_bytes,
        }

    pointer_conversion = {
        "storage_type": "i64",
        "device_pointer_bits": 32,
        "operations": ["llvm.trunc i64 to i32", "llvm.inttoptr i32 to !llvm.ptr"],
    }
    return {
        "data_layout": "e-m:e-p:32:32-i64:64-n32-S128",
        "byte_order": "little-endian",
        "device_pointer_conversion": pointer_conversion,
        "buffers": {
            "bevformer_fc1_bias_relu": {
                "alignment_bytes": 8,
                "size_bytes": 48,
                "fields": [
                    field("rows", 0, "i32", 4, 4),
                    field("channels", 4, "i32", 4, 4),
                    field("hidden", 8, "i32", 4, 4),
                    field("_padding0", 12, "i32", 4, 4),
                    field("query_addr", 16, "i64", 8, 8),
                    field("weight_addr", 24, "i64", 8, 8),
                    field("bias_addr", 32, "i64", 8, 8),
                    field("output_addr", 40, "i64", 8, 8),
                ],
            },
            "bevformer_fc2_bias_residual": {
                "alignment_bytes": 8,
                "size_bytes": 56,
                "fields": [
                    field("rows", 0, "i32", 4, 4),
                    field("channels", 4, "i32", 4, 4),
                    field("hidden", 8, "i32", 4, 4),
                    field("_padding0", 12, "i32", 4, 4),
                    field("query_addr", 16, "i64", 8, 8),
                    field("activation_addr", 24, "i64", 8, 8),
                    field("weight_addr", 32, "i64", 8, 8),
                    field("bias_addr", 40, "i64", 8, 8),
                    field("output_addr", 48, "i64", 8, 8),
                ],
            },
            "bevformer_layer_norm": {
                "alignment_bytes": 8,
                "size_bytes": 48,
                "fields": [
                    field("rows", 0, "i32", 4, 4),
                    field("channels", 4, "i32", 4, 4),
                    field("epsilon", 8, "f32", 4, 4),
                    field("_padding0", 12, "i32", 4, 4),
                    field("input_addr", 16, "i64", 8, 8),
                    field("scale_addr", 24, "i64", 8, 8),
                    field("bias_addr", 32, "i64", 8, 8),
                    field("output_addr", 40, "i64", 8, 8),
                ],
            },
        },
    }


@dataclass(frozen=True, slots=True)
class VortexFFNPlan:
    rows: int
    channels: int
    hidden: int
    epsilon: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    kernels: tuple[str, ...] = (
        "bevformer_fc1_bias_relu",
        "bevformer_fc2_bias_residual",
        "bevformer_layer_norm",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "channels": self.channels,
            "hidden": self.hidden,
            "epsilon": self.epsilon,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "kernels": list(self.kernels),
        }


@dataclass(frozen=True, slots=True)
class VortexProgram:
    """One reviewable Vortex program derived from externally verified MLIR."""

    graph: Graph
    reference_graph: Graph
    mlir_program: MlirProgram
    optimized_mlir: str
    vortex_mlir: str
    mlir_opt_version: str
    mlir_opt_source_revision: str | None
    plan: VortexFFNPlan
    kernel_source: str
    common_header: str
    host_source: str
    model_config_header: str
    makefile: str
    verifier_source: str
    device_lowering: str = LEGACY_CPP_LOWERING
    device_mlir: str | None = None
    buffer_plan: VortexBufferPlan | None = None

    @property
    def source(self) -> str:
        if self.device_lowering == DIRECT_MLIR_LOWERING:
            if self.device_mlir is None:
                raise BackendError(f"{TARGET}: direct MLIR program has no device module")
            return self.device_mlir
        return self.kernel_source

    def describe(self) -> dict[str, Any]:
        pre_hash = _sha256_text(self.mlir_program.source)
        post_hash = _sha256_text(self.optimized_mlir)
        vortex_hash = _sha256_text(self.vortex_mlir)
        direct = self.device_lowering == DIRECT_MLIR_LOWERING
        lowering_backend = (
            "audited BEVFormer FFN -> LLVM dialect device MLIR -> "
            "mlir-translate LLVM IR -> VOLT RISC-V/Vortex machine code"
            if direct
            else (
                "audited BEVFormer FFN pattern -> verified standard func.call "
                "Vortex runtime plan -> Vortex kernel C++ -> VOLT"
            )
        )
        lowering: dict[str, Any] = {
            "frontend": "GAC canonical BEVFormer FFN graph",
            "middle_end": "upstream tensor/linalg/arith/math MLIR",
            "mlir_opt_passes": ["verify-each", "canonicalize", "cse"],
            "device_lowering": self.device_lowering,
            "backend": lowering_backend,
            "pre_opt_mlir_sha256": pre_hash,
            "post_opt_mlir_sha256": post_hash,
            "vortex_call_plan_mlir_sha256": vortex_hash,
            "external_weight_sha256": _weight_sha256(self.graph),
            "mlir_opt_version": self.mlir_opt_version,
            "mlir_opt_source_revision": self.mlir_opt_source_revision,
            "llvm_version_boundary": (
                "one VOLT LLVM 20.1.8 tree provides mlir-opt, mlir-translate, "
                "clang, lld, and the Vortex RISC-V backend"
                if direct
                else (
                    "MLIR middle-end and VOLT are separate textual/source stages; "
                    "LLVM bitcode is not exchanged across versions"
                )
            ),
        }
        if direct:
            if self.device_mlir is None:
                raise BackendError(f"{TARGET}: direct MLIR program has no device module")
            lowering["device_kernel_mlir_sha256"] = _sha256_text(self.device_mlir)
            lowering["volt_llvm_revision_expected"] = VOLT_LLVM_REVISION
            lowering["device_argument_abi"] = _direct_argument_abi()
            if self.buffer_plan is not None:
                lowering["buffer_plan"] = self.buffer_plan.to_dict()
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "compiler_backend": TARGET,
            "target": {
                "platform": "Vortex 3.0",
                "xlen": 32,
                "isa": "RV32IMAF+xvortex+zicond",
                "binary_format": ".vxbin",
                "runtime": "vortex2",
                "drivers": ["simx", "rtlsim", "xrt", "opae", "aved", "firesim"],
            },
            "lowering": lowering,
            "graph": self.graph.name,
            "model_provenance": self.graph.metadata.get(
                "weights_provenance", "provided canonical graph constants"
            ),
            "profile": self.plan.to_dict(),
            "dispatch_count": len(self.plan.kernels),
            "oracle": {
                "kind": "GAC NumPy reference_run",
                "query": "deterministic query.bin",
                "output": "golden.bin",
                "runtime_recomputes_oracle": False,
            },
            "fallbacks": {
                "cuda": False,
                "ptx": False,
                "cublas": False,
                "cupy": False,
            },
            "status": (
                "direct MLIR Vortex executable project; run make to translate "
                "kernel.mlir and produce kernel.vxbin"
                if direct
                else "Vortex executable project; run make to produce kernel.vxbin"
            ),
        }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _weight_sha256(graph: Graph) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            np.ascontiguousarray(graph.constants[name], dtype="<f4").tobytes(order="C")
        ).hexdigest()
        for name in _REQUIRED_CONSTANTS
    }


def _match_bevformer_ffn(program: MlirProgram) -> VortexFFNPlan:
    graph = program.graph
    if graph.metadata.get("kind") != "bevformer-transformer-ffn":
        raise BackendError(
            f"{TARGET}: requires a canonical BEVFormer FFN graph, got "
            f"kind={graph.metadata.get('kind')!r}"
        )
    if tuple(graph.inputs) != ("query",) or graph.outputs != ("output",):
        raise BackendError(f"{TARGET}: requires query -> output graph ABI")
    if set(graph.constants) != set(_REQUIRED_CONSTANTS):
        raise BackendError(
            f"{TARGET}: requires constants {list(_REQUIRED_CONSTANTS)}, got "
            f"{sorted(graph.constants)}"
        )
    expected_nodes = (
        ("MatMul", ("query", "fc1_weight"), "fc1_mm"),
        ("AddReLU", ("fc1_mm", "fc1_bias"), "activated"),
        ("MatMul", ("activated", "fc2_weight"), "fc2_mm"),
        ("Add", ("fc2_mm", "fc2_bias"), "projected"),
        ("Add", ("query", "projected"), "ffn_output"),
        (
            "LayerNorm",
            ("ffn_output", "norm_weight", "norm_bias"),
            "output",
        ),
    )
    actual_nodes = tuple(
        (node.op, node.inputs, node.outputs[0]) for node in graph.nodes
    )
    if actual_nodes != expected_nodes:
        raise BackendError(
            f"{TARGET}: graph does not match the fused BEVFormer FFN lowering pattern; "
            f"got {actual_nodes!r}"
        )

    query = graph.value_specs["query"]
    output = graph.value_specs["output"]
    if query.shape != output.shape or query.rank != 3 or query.shape[0] != 1:
        raise BackendError(
            f"{TARGET}: query and output require the BEVFormer [1, rows, channels] ABI"
        )
    rows = math.prod(query.shape[:-1])
    channels = query.shape[-1]
    hidden = graph.constants["fc1_weight"].shape[1]
    from .vortex_device_mlir import validate_bevformer_ffn_rv32_extents

    rows, channels, hidden = validate_bevformer_ffn_rv32_extents(
        rows, channels, hidden
    )
    expected_shapes = {
        "fc1_weight": (channels, hidden),
        "fc1_bias": (hidden,),
        "fc2_weight": (hidden, channels),
        "fc2_bias": (channels,),
        "norm_weight": (channels,),
        "norm_bias": (channels,),
    }
    for name, shape in expected_shapes.items():
        value = graph.constants[name]
        if value.shape != shape or value.dtype != np.float32:
            raise BackendError(
                f"{TARGET}: {name} requires {shape}/float32, got "
                f"{value.shape}/{value.dtype}"
            )
    epsilon = graph.nodes[-1].attrs.get("epsilon", 1e-5)
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise BackendError(f"{TARGET}: LayerNorm epsilon must be numeric")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise BackendError(f"{TARGET}: LayerNorm epsilon must be finite and positive")
    return VortexFFNPlan(
        rows=rows,
        channels=channels,
        hidden=hidden,
        epsilon=epsilon_value,
        input_shape=query.shape,
        output_shape=output.shape,
    )


_SSA = r"%[A-Za-z0-9_.$-]+"
_FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)"
_RUNTIME_CALLEES = (
    "gac_vortex_fc1_bias_relu",
    "gac_vortex_fc2_bias_residual",
    "gac_vortex_layer_norm",
)
_RUNTIME_KERNELS = (
    "bevformer_fc1_bias_relu",
    "bevformer_fc2_bias_residual",
    "bevformer_layer_norm",
)


def _parse_static_f32_type(value: str, *, kind: str) -> tuple[int, ...]:
    text = value.strip()
    if not text.endswith("xf32"):
        raise BackendError(f"{TARGET}: {kind} must be a static f32 shaped type")
    dimensions = text[:-4].split("x")
    if not dimensions or any(re.fullmatch(r"[1-9][0-9]*", dim) is None for dim in dimensions):
        raise BackendError(f"{TARGET}: {kind} must have positive static dimensions")
    return tuple(int(dim) for dim in dimensions)


def _same_f32(left: float, right: float) -> bool:
    return np.float32(left).tobytes() == np.float32(right).tobytes()


def _require_same_plan(actual: VortexFFNPlan, expected: VortexFFNPlan, *, stage: str) -> None:
    fields = ("rows", "channels", "hidden", "input_shape", "output_shape", "kernels")
    mismatches = [
        name for name in fields if getattr(actual, name) != getattr(expected, name)
    ]
    if not _same_f32(actual.epsilon, expected.epsilon):
        mismatches.append("epsilon")
    if mismatches:
        raise BackendError(
            f"{TARGET}: {stage} changed the supported FFN contract: "
            + ", ".join(mismatches)
        )


def _parse_optimized_mlir(source: str, expected: VortexFFNPlan) -> VortexFFNPlan:
    """Derive the supported FFN contract from post-``mlir-opt`` tensor MLIR."""

    checks = {
        "one module": source.count("module {") == 1,
        "one function": len(re.findall(r"\bfunc\.func\s+@", source)) == 1,
        "two matrix multiplications": source.count("linalg.matmul") == 2,
        "one fused ReLU": source.count("arith.maximumf") == 1,
        "one LayerNorm reciprocal square root": source.count("math.rsqrt") == 1,
        "one function return": len(
            re.findall(r"(?m)^\s*(?:func\.)?return\b", source)
        )
        == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise BackendError(
            f"{TARGET}: optimized MLIR is outside the supported FFN pattern: "
            + ", ".join(failed)
        )

    function = re.search(
        rf"func\.func\s+@[A-Za-z_$][A-Za-z0-9_.$-]*\s*\(\s*{_SSA}\s*:\s*"
        r"tensor<([^>]+)>\s*\)\s*->\s*tensor<([^>]+)>\s*\{",
        source,
    )
    if function is None:
        raise BackendError(f"{TARGET}: optimized MLIR has an unsupported function ABI")
    input_shape = _parse_static_f32_type(function.group(1), kind="function input")
    output_shape = _parse_static_f32_type(function.group(2), kind="function output")
    if input_shape != output_shape or len(input_shape) != 3 or input_shape[0] != 1:
        raise BackendError(
            f"{TARGET}: optimized MLIR requires the BEVFormer [1, rows, channels] ABI"
        )
    rows = math.prod(input_shape[:-1])
    channels = input_shape[-1]

    matmul_pattern = re.compile(
        rf"linalg\.matmul\s+ins\(\s*{_SSA}\s*,\s*{_SSA}\s*:\s*"
        r"tensor<([^>]+)>\s*,\s*tensor<([^>]+)>\s*\)\s*"
        rf"outs\(\s*{_SSA}\s*:\s*tensor<([^>]+)>\s*\)\s*"
        r"->\s*tensor<([^>]+)>",
    )
    matmuls = [
        tuple(
            _parse_static_f32_type(value, kind=f"MatMul {index} type")
            for value in match.groups()
        )
        for index, match in enumerate(matmul_pattern.finditer(source), start=1)
    ]
    if len(matmuls) != 2:
        raise BackendError(
            f"{TARGET}: optimized MLIR matrix multiplication signatures are unsupported"
        )
    first_lhs, first_rhs, first_init, first_result = matmuls[0]
    second_lhs, second_rhs, second_init, second_result = matmuls[1]
    if len(first_rhs) != 2:
        raise BackendError(f"{TARGET}: optimized MLIR FC1 weight must be rank two")
    hidden = first_rhs[1]
    expected_matmuls = (
        ((rows, channels), (channels, hidden), (rows, hidden), (rows, hidden)),
        ((rows, hidden), (hidden, channels), (rows, channels), (rows, channels)),
    )
    if matmuls != list(expected_matmuls):
        raise BackendError(f"{TARGET}: optimized MLIR MatMul shapes changed the FFN")

    epsilon_value: float | None = None
    add_pattern = re.compile(
        rf"(?P<result>{_SSA})\s*=\s*arith\.addf\s+{_SSA}\s*,\s*"
        rf"(?P<epsilon>{_SSA})\s*:\s*f32"
    )
    for add in add_pattern.finditer(source):
        result = re.escape(add.group("result"))
        if re.search(rf"math\.rsqrt\s+{result}\s*:\s*f32", source) is None:
            continue
        epsilon_ssa = re.escape(add.group("epsilon"))
        constant = re.search(
            rf"{epsilon_ssa}\s*=\s*arith\.constant\s+({_FLOAT})\s*:\s*f32",
            source,
        )
        if constant is None or epsilon_value is not None:
            raise BackendError(f"{TARGET}: optimized MLIR LayerNorm epsilon is ambiguous")
        epsilon_value = float(constant.group(1))
    if epsilon_value is None or not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise BackendError(f"{TARGET}: optimized MLIR LayerNorm epsilon is unsupported")

    actual = VortexFFNPlan(
        rows=rows,
        channels=channels,
        hidden=hidden,
        epsilon=float(np.float32(epsilon_value)),
        input_shape=input_shape,
        output_shape=output_shape,
    )
    _require_same_plan(actual, expected, stage="optimized tensor MLIR")
    return actual


def _audit_optimized_weights(source: str, graph: Graph) -> None:
    """Require every externally stored weight to occur bit-for-bit in MLIR.

    ``canonicalize``/CSE may merge identical constants, so the comparison is
    over unique ``(shape, payload)`` fingerprints rather than SSA definition
    count.  The standard MLIR float printer emits enough digits to round-trip
    finite f32 values exactly.
    """

    actual_by_ssa: dict[str, tuple[tuple[int, ...], str]] = {}
    dense_pattern = re.compile(
        rf"(?P<ssa>{_SSA})\s*=\s*arith\.constant\s+"
        r"dense<(?P<payload>.*?)>\s*:\s*"
        r"tensor<(?P<type>[^>]+)>",
        flags=re.DOTALL,
    )
    for index, match in enumerate(dense_pattern.finditer(source), start=1):
        shape = _parse_static_f32_type(
            match.group("type"), kind=f"dense constant {index}"
        )
        payload = match.group("payload").strip()
        element_count = math.prod(shape)
        raw_hex = re.fullmatch(r'"0x(?P<hex>[0-9A-Fa-f]+)"', payload)
        if raw_hex is not None:
            raw_text = raw_hex.group("hex")
            if len(raw_text) % 2:
                raise BackendError(
                    f"{TARGET}: optimized MLIR dense constant {index} has "
                    "a malformed raw-byte literal"
                )
            raw_bytes = bytes.fromhex(raw_text)
            if len(raw_bytes) != element_count * np.dtype("<f4").itemsize:
                raise BackendError(
                    f"{TARGET}: optimized MLIR dense constant {index} has "
                    f"{len(raw_bytes)} raw bytes, expected {element_count * 4}"
                )
            payload_bytes = raw_bytes
        elif re.fullmatch(_FLOAT, payload):
            values = np.full(element_count, float(payload), dtype="<f4")
            payload_bytes = values.tobytes(order="C")
        else:
            remainder = re.sub(_FLOAT, "", payload)
            if re.fullmatch(r"[\[\],\s]*", remainder) is None:
                raise BackendError(
                    f"{TARGET}: optimized MLIR dense constant {index} is unsupported"
                )
            tokens = re.findall(_FLOAT, payload)
            if len(tokens) != element_count:
                raise BackendError(
                    f"{TARGET}: optimized MLIR dense constant {index} has "
                    f"{len(tokens)} values, expected {element_count}"
                )
            values = np.asarray(tokens, dtype="<f4")
            payload_bytes = values.tobytes(order="C")
        actual_by_ssa[match.group("ssa")] = (
            shape,
            hashlib.sha256(payload_bytes).hexdigest(),
        )

    expected_by_role = {
        name: (
            graph.constants[name].shape,
            hashlib.sha256(
                np.ascontiguousarray(graph.constants[name], dtype="<f4").tobytes(
                    order="C"
                )
            ).hexdigest(),
        )
        for name in _REQUIRED_CONSTANTS
    }
    actual = set(actual_by_ssa.values())
    expected = set(expected_by_role.values())
    if actual != expected:
        missing = len(expected - actual)
        unexpected = len(actual - expected)
        raise BackendError(
            f"{TARGET}: optimized MLIR dense weights do not match the canonical "
            f"graph bit-for-bit (missing={missing}, unexpected={unexpected})"
        )

    matmul_uses = list(
        re.finditer(
            rf"linalg\.matmul\s+ins\(\s*(?P<lhs>{_SSA})\s*,\s*"
            rf"(?P<rhs>{_SSA})\s*:",
            source,
        )
    )
    generic_uses: list[tuple[tuple[str, ...], str]] = []
    for generic in re.finditer(
        rf"{_SSA}\s*=\s*linalg\.generic\b.*?\bins\((?P<inputs>[^)]*)\)\s*"
        r"outs\([^)]*\)\s*\{(?P<body>.*?)^\s*\}\s*->",
        source,
        flags=re.DOTALL | re.MULTILINE,
    ):
        input_values = generic.group("inputs").split(":", 1)[0]
        generic_uses.append(
            (tuple(re.findall(_SSA, input_values)), generic.group("body"))
        )

    relu_uses = [item for item in generic_uses if "arith.maximumf" in item[1]]
    fc2_bias_uses = [
        item
        for item in generic_uses
        if "arith.addf" in item[1]
        and "arith.maximumf" not in item[1]
        and "math.rsqrt" not in item[1]
        and len(item[0]) == 2
        and item[0][1] in actual_by_ssa
    ]
    layer_norm_uses = [item for item in generic_uses if "math.rsqrt" in item[1]]
    if (
        len(matmul_uses) != 2
        or len(relu_uses) != 1
        or len(fc2_bias_uses) != 1
        or len(layer_norm_uses) != 1
        or len(relu_uses[0][0]) != 2
        or len(layer_norm_uses[0][0]) != 4
    ):
        raise BackendError(
            f"{TARGET}: optimized MLIR weight dataflow is outside the supported FFN pattern"
        )
    role_ssas = {
        "fc1_weight": matmul_uses[0].group("rhs"),
        "fc1_bias": relu_uses[0][0][1],
        "fc2_weight": matmul_uses[1].group("rhs"),
        "fc2_bias": fc2_bias_uses[0][0][1],
        "norm_weight": layer_norm_uses[0][0][2],
        "norm_bias": layer_norm_uses[0][0][3],
    }
    for role, ssa in role_ssas.items():
        if actual_by_ssa.get(ssa) != expected_by_role[role]:
            raise BackendError(
                f"{TARGET}: optimized MLIR dense weight role {role!r} changed"
            )


def _memref_type(shape: tuple[int, ...]) -> str:
    return "memref<" + "x".join(str(dim) for dim in shape) + "xf32>"


def _mlir_float_literal(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "." not in text and "e" not in text and "E" not in text:
        text += ".0"
    return text


def _runtime_call_mlir(
    plan: VortexFFNPlan,
    optimized_sha256: str,
    weight_sha256: Mapping[str, str],
) -> str:
    """Lower an optimized FFN contract to the standard Vortex call ABI."""

    input_type = _memref_type(plan.input_shape)
    hidden_type = _memref_type((*plan.input_shape[:-1], plan.hidden))
    w1_type = _memref_type((plan.channels, plan.hidden))
    b1_type = _memref_type((plan.hidden,))
    w2_type = _memref_type((plan.hidden, plan.channels))
    channel_type = _memref_type((plan.channels,))
    fc1_launch = plan.rows * plan.hidden
    fc2_launch = plan.rows * plan.channels
    if set(weight_sha256) != set(_REQUIRED_CONSTANTS) or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in weight_sha256.values()
    ):
        raise BackendError(f"{TARGET}: external weight hashes are incomplete")
    attributes = ", ".join(
        f'gac.{name}_sha256 = "{weight_sha256[name]}"'
        for name in _REQUIRED_CONSTANTS
    )
    return f'''module {{
  func.func private @gac_vortex_fc1_bias_relu(index, {input_type}, {w1_type}, {b1_type}, {hidden_type})
  func.func private @gac_vortex_fc2_bias_residual(index, {input_type}, {hidden_type}, {w2_type}, {channel_type}, {input_type})
  func.func private @gac_vortex_layer_norm(index, {input_type}, {channel_type}, {channel_type}, {input_type}, f32)

  func.func @bevformer_vortex_plan_{optimized_sha256}(
      %query: {input_type},
      %fc1_weight: {w1_type},
      %fc1_bias: {b1_type},
      %fc2_weight: {w2_type},
      %fc2_bias: {channel_type},
      %norm_weight: {channel_type},
      %norm_bias: {channel_type},
      %activation: {hidden_type},
      %residual: {input_type},
      %output: {input_type}) attributes {{{attributes}}} {{
    %fc1_launch = arith.constant {fc1_launch} : index
    %fc2_launch = arith.constant {fc2_launch} : index
    %norm_launch = arith.constant {plan.rows} : index
    %epsilon = arith.constant {_mlir_float_literal(plan.epsilon)} : f32
    func.call @gac_vortex_fc1_bias_relu(%fc1_launch, %query, %fc1_weight, %fc1_bias, %activation) : (index, {input_type}, {w1_type}, {b1_type}, {hidden_type}) -> ()
    func.call @gac_vortex_fc2_bias_residual(%fc2_launch, %query, %activation, %fc2_weight, %fc2_bias, %residual) : (index, {input_type}, {hidden_type}, {w2_type}, {channel_type}, {input_type}) -> ()
    func.call @gac_vortex_layer_norm(%norm_launch, %residual, %norm_weight, %norm_bias, %output, %epsilon) : (index, {input_type}, {channel_type}, {channel_type}, {input_type}, f32) -> ()
    return
  }}
}}
'''


def _split_mlir_types(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _shape_from_memref(value: str, *, kind: str) -> tuple[int, ...]:
    match = re.fullmatch(r"memref<([^>]+)>", value.strip())
    if match is None:
        raise BackendError(f"{TARGET}: {kind} must use a static f32 memref")
    return _parse_static_f32_type(match.group(1), kind=kind)


def _parse_vortex_mlir(
    source: str,
    *,
    optimized_sha256: str,
    expected: VortexFFNPlan,
    expected_weight_sha256: Mapping[str, str],
) -> VortexFFNPlan:
    """Parse verified runtime-call MLIR; this result drives source generation."""

    symbol = f"bevformer_vortex_plan_{optimized_sha256}"
    function = re.search(
        rf"func\.func\s+@{re.escape(symbol)}\s*\((?P<args>.*?)\)\s*"
        rf"(?P<attrs>attributes\s*\{{[^}}]*\}})\s*\{{"
        rf"(?P<body>.*?)^\s*\}}",
        source,
        flags=re.DOTALL | re.MULTILINE,
    )
    if function is None:
        raise BackendError(
            f"{TARGET}: verified Vortex MLIR is missing the source-bound plan function"
        )
    if len(re.findall(r"func\.func\s+@bevformer_vortex_plan_", source)) != 1:
        raise BackendError(f"{TARGET}: verified Vortex MLIR has an ambiguous plan")
    attributes = {
        match.group("name"): match.group("digest")
        for match in re.finditer(
            r"gac\.(?P<name>[A-Za-z0-9_]+)_sha256\s*=\s*"
            r'"(?P<digest>[0-9a-f]{64})"',
            function.group("attrs"),
        )
    }
    if attributes != dict(expected_weight_sha256):
        raise BackendError(f"{TARGET}: verified Vortex MLIR external weight binding changed")

    arguments = tuple(
        (match.group("ssa"), match.group("type").strip())
        for match in re.finditer(
            rf"(?P<ssa>{_SSA})\s*:\s*(?P<type>[^,\n]+)", function.group("args")
        )
    )
    argument_ssas = tuple(ssa for ssa, _type in arguments)
    argument_types = tuple(_type for _ssa, _type in arguments)
    if len(argument_types) != 10:
        raise BackendError(f"{TARGET}: verified Vortex MLIR plan ABI requires ten buffers")
    shapes = tuple(
        _shape_from_memref(value, kind=f"runtime argument {index}")
        for index, value in enumerate(argument_types)
    )
    input_shape = shapes[0]
    if len(input_shape) != 3 or input_shape[0] != 1:
        raise BackendError(f"{TARGET}: Vortex runtime input must have shape [1, rows, channels]")
    rows = math.prod(input_shape[:-1])
    channels = input_shape[-1]
    if len(shapes[1]) != 2:
        raise BackendError(f"{TARGET}: Vortex runtime FC1 weight must be rank two")
    hidden = shapes[1][1]
    expected_shapes = (
        input_shape,
        (channels, hidden),
        (hidden,),
        (hidden, channels),
        (channels,),
        (channels,),
        (channels,),
        (*input_shape[:-1], hidden),
        input_shape,
        input_shape,
    )
    if shapes != expected_shapes:
        raise BackendError(f"{TARGET}: verified Vortex MLIR buffer roles or shapes changed")

    body = function.group("body")
    calls = list(
        re.finditer(
            r"(?:func\.)?call\s+@(?P<callee>[A-Za-z_$][A-Za-z0-9_.$-]*)\s*"
            r"\((?P<args>[^)]*)\)\s*:\s*\((?P<types>[^)]*)\)\s*->\s*\(\)",
            body,
        )
    )
    callees = tuple(match.group("callee") for match in calls)
    if callees != _RUNTIME_CALLEES:
        raise BackendError(f"{TARGET}: verified Vortex MLIR kernel order changed")

    expected_call_types = (
        ("index", *argument_types[0:3], argument_types[7]),
        ("index", argument_types[0], argument_types[7], *argument_types[3:5], argument_types[8]),
        ("index", argument_types[8], *argument_types[5:7], argument_types[9], "f32"),
    )
    expected_buffer_arguments = (
        (argument_ssas[0], argument_ssas[1], argument_ssas[2], argument_ssas[7]),
        (
            argument_ssas[0],
            argument_ssas[7],
            argument_ssas[3],
            argument_ssas[4],
            argument_ssas[8],
        ),
        (
            argument_ssas[8],
            argument_ssas[5],
            argument_ssas[6],
            argument_ssas[9],
        ),
    )
    parsed_call_arguments: list[tuple[str, ...]] = []
    for index, (call, callee, expected_types) in enumerate(
        zip(calls, _RUNTIME_CALLEES, expected_call_types)
    ):
        if _split_mlir_types(call.group("types")) != tuple(expected_types):
            raise BackendError(f"{TARGET}: verified Vortex MLIR call {index} ABI changed")
        declaration = re.search(
            rf"func\.func\s+private\s+@{re.escape(callee)}\s*\(([^)]*)\)",
            source,
        )
        if declaration is None or _split_mlir_types(declaration.group(1)) != tuple(
            expected_types
        ):
            raise BackendError(
                f"{TARGET}: verified Vortex MLIR declaration {index} ABI changed"
            )
        call_arguments = tuple(item.strip() for item in call.group("args").split(","))
        parsed_call_arguments.append(call_arguments)
        buffer_arguments = (
            call_arguments[1:-1] if index == 2 else call_arguments[1:]
        )
        if buffer_arguments != expected_buffer_arguments[index]:
            raise BackendError(
                f"{TARGET}: verified Vortex MLIR call {index} buffer dataflow changed"
            )

    constants: dict[str, tuple[str, str]] = {}
    for constant in re.finditer(
        rf"(?P<ssa>{_SSA})\s*=\s*arith\.constant\s+"
        rf"(?P<value>{_FLOAT}|[0-9]+)\s*:\s*(?P<type>index|f32)",
        body,
    ):
        constants[constant.group("ssa")] = (
            constant.group("value"),
            constant.group("type"),
        )
    expected_launches = (rows * hidden, rows * channels, rows)
    for index, (call_arguments, expected_launch) in enumerate(
        zip(parsed_call_arguments, expected_launches)
    ):
        if not call_arguments or call_arguments[0] not in constants:
            raise BackendError(f"{TARGET}: verified Vortex MLIR call {index} lacks launch size")
        value, value_type = constants[call_arguments[0]]
        if value_type != "index" or int(value) != expected_launch:
            raise BackendError(f"{TARGET}: verified Vortex MLIR call {index} launch size changed")
    norm_arguments = parsed_call_arguments[2]
    if len(norm_arguments) != 6 or norm_arguments[-1] not in constants:
        raise BackendError(f"{TARGET}: verified Vortex MLIR LayerNorm epsilon is missing")
    epsilon_text, epsilon_type = constants[norm_arguments[-1]]
    epsilon = float(epsilon_text)
    if epsilon_type != "f32" or not math.isfinite(epsilon) or epsilon <= 0:
        raise BackendError(f"{TARGET}: verified Vortex MLIR LayerNorm epsilon is invalid")

    actual = VortexFFNPlan(
        rows=rows,
        channels=channels,
        hidden=hidden,
        epsilon=float(np.float32(epsilon)),
        input_shape=input_shape,
        output_shape=shapes[9],
        kernels=_RUNTIME_KERNELS,
    )
    _require_same_plan(actual, expected, stage="verified Vortex runtime-call MLIR")
    return actual


def _common_header() -> str:
    return r'''#ifndef GAC_BEVFORMER_VORTEX_COMMON_H
#define GAC_BEVFORMER_VORTEX_COMMON_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint32_t rows;
  uint32_t channels;
  uint32_t hidden;
  uint32_t _padding0;
  uint64_t query_addr;
  uint64_t weight_addr;
  uint64_t bias_addr;
  uint64_t output_addr;
} fc1_args_t;

typedef struct {
  uint32_t rows;
  uint32_t channels;
  uint32_t hidden;
  uint32_t _padding0;
  uint64_t query_addr;
  uint64_t activation_addr;
  uint64_t weight_addr;
  uint64_t bias_addr;
  uint64_t output_addr;
} fc2_args_t;

typedef struct {
  uint32_t rows;
  uint32_t channels;
  float epsilon;
  uint32_t _padding0;
  uint64_t input_addr;
  uint64_t scale_addr;
  uint64_t bias_addr;
  uint64_t output_addr;
} layer_norm_args_t;

#if defined(__cplusplus)
static_assert(sizeof(fc1_args_t) == 48, "fc1_args_t size must match device ABI");
static_assert(alignof(fc1_args_t) == 8, "fc1_args_t alignment must match device ABI");
static_assert(offsetof(fc1_args_t, rows) == 0, "fc1_args_t.rows ABI offset");
static_assert(offsetof(fc1_args_t, channels) == 4, "fc1_args_t.channels ABI offset");
static_assert(offsetof(fc1_args_t, hidden) == 8, "fc1_args_t.hidden ABI offset");
static_assert(offsetof(fc1_args_t, _padding0) == 12, "fc1_args_t._padding0 ABI offset");
static_assert(offsetof(fc1_args_t, query_addr) == 16, "fc1_args_t.query_addr ABI offset");
static_assert(offsetof(fc1_args_t, weight_addr) == 24, "fc1_args_t.weight_addr ABI offset");
static_assert(offsetof(fc1_args_t, bias_addr) == 32, "fc1_args_t.bias_addr ABI offset");
static_assert(offsetof(fc1_args_t, output_addr) == 40, "fc1_args_t.output_addr ABI offset");

static_assert(sizeof(fc2_args_t) == 56, "fc2_args_t size must match device ABI");
static_assert(alignof(fc2_args_t) == 8, "fc2_args_t alignment must match device ABI");
static_assert(offsetof(fc2_args_t, rows) == 0, "fc2_args_t.rows ABI offset");
static_assert(offsetof(fc2_args_t, channels) == 4, "fc2_args_t.channels ABI offset");
static_assert(offsetof(fc2_args_t, hidden) == 8, "fc2_args_t.hidden ABI offset");
static_assert(offsetof(fc2_args_t, _padding0) == 12, "fc2_args_t._padding0 ABI offset");
static_assert(offsetof(fc2_args_t, query_addr) == 16, "fc2_args_t.query_addr ABI offset");
static_assert(offsetof(fc2_args_t, activation_addr) == 24, "fc2_args_t.activation_addr ABI offset");
static_assert(offsetof(fc2_args_t, weight_addr) == 32, "fc2_args_t.weight_addr ABI offset");
static_assert(offsetof(fc2_args_t, bias_addr) == 40, "fc2_args_t.bias_addr ABI offset");
static_assert(offsetof(fc2_args_t, output_addr) == 48, "fc2_args_t.output_addr ABI offset");

static_assert(sizeof(layer_norm_args_t) == 48, "layer_norm_args_t size must match device ABI");
static_assert(alignof(layer_norm_args_t) == 8, "layer_norm_args_t alignment must match device ABI");
static_assert(offsetof(layer_norm_args_t, rows) == 0, "layer_norm_args_t.rows ABI offset");
static_assert(offsetof(layer_norm_args_t, channels) == 4, "layer_norm_args_t.channels ABI offset");
static_assert(offsetof(layer_norm_args_t, epsilon) == 8, "layer_norm_args_t.epsilon ABI offset");
static_assert(offsetof(layer_norm_args_t, _padding0) == 12, "layer_norm_args_t._padding0 ABI offset");
static_assert(offsetof(layer_norm_args_t, input_addr) == 16, "layer_norm_args_t.input_addr ABI offset");
static_assert(offsetof(layer_norm_args_t, scale_addr) == 24, "layer_norm_args_t.scale_addr ABI offset");
static_assert(offsetof(layer_norm_args_t, bias_addr) == 32, "layer_norm_args_t.bias_addr ABI offset");
static_assert(offsetof(layer_norm_args_t, output_addr) == 40, "layer_norm_args_t.output_addr ABI offset");
#endif

#endif
'''


def _kernel_source(post_opt_sha256: str, vortex_mlir_sha256: str) -> str:
    return f'''// Generated by GAC from optimized MLIR SHA-256 {post_opt_sha256}
// Vortex runtime-call MLIR SHA-256 {vortex_mlir_sha256}
#include <math.h>
#include <stdint.h>
#include <vx_spawn2.h>

#include "common.h"

static inline uint32_t global_id() {{
  return blockIdx.x * blockDim.x + threadIdx.x;
}}

__kernel void bevformer_fc1_bias_relu(fc1_args_t* __UNIFORM__ arg) {{
  const uint32_t index = global_id();
  const uint32_t count = arg->rows * arg->hidden;
  if (index >= count)
    return;
  const uint32_t row = index / arg->hidden;
  const uint32_t column = index - row * arg->hidden;
  const float* query = reinterpret_cast<const float*>(arg->query_addr);
  const float* weight = reinterpret_cast<const float*>(arg->weight_addr);
  const float* bias = reinterpret_cast<const float*>(arg->bias_addr);
  float* output = reinterpret_cast<float*>(arg->output_addr);

  float value = bias[column];
  for (uint32_t k = 0; k < arg->channels; ++k)
    value += query[row * arg->channels + k] * weight[k * arg->hidden + column];
  output[index] = value > 0.0f ? value : 0.0f;
}}

__kernel void bevformer_fc2_bias_residual(fc2_args_t* __UNIFORM__ arg) {{
  const uint32_t index = global_id();
  const uint32_t count = arg->rows * arg->channels;
  if (index >= count)
    return;
  const uint32_t row = index / arg->channels;
  const uint32_t column = index - row * arg->channels;
  const float* query = reinterpret_cast<const float*>(arg->query_addr);
  const float* activation = reinterpret_cast<const float*>(arg->activation_addr);
  const float* weight = reinterpret_cast<const float*>(arg->weight_addr);
  const float* bias = reinterpret_cast<const float*>(arg->bias_addr);
  float* output = reinterpret_cast<float*>(arg->output_addr);

  float value = bias[column];
  for (uint32_t k = 0; k < arg->hidden; ++k)
    value += activation[row * arg->hidden + k] * weight[k * arg->channels + column];
  output[index] = query[index] + value;
}}

__kernel void bevformer_layer_norm(layer_norm_args_t* __UNIFORM__ arg) {{
  const uint32_t row = global_id();
  if (row >= arg->rows)
    return;
  const float* input = reinterpret_cast<const float*>(arg->input_addr);
  const float* scale = reinterpret_cast<const float*>(arg->scale_addr);
  const float* bias = reinterpret_cast<const float*>(arg->bias_addr);
  float* output = reinterpret_cast<float*>(arg->output_addr);
  const uint32_t base = row * arg->channels;

  float mean = 0.0f;
  for (uint32_t i = 0; i < arg->channels; ++i)
    mean += input[base + i];
  mean /= static_cast<float>(arg->channels);
  float variance = 0.0f;
  for (uint32_t i = 0; i < arg->channels; ++i) {{
    const float centered = input[base + i] - mean;
    variance += centered * centered;
  }}
  variance /= static_cast<float>(arg->channels);
  const float inverse_stddev = 1.0f / sqrtf(variance + arg->epsilon);
  for (uint32_t i = 0; i < arg->channels; ++i)
    output[base + i] = (input[base + i] - mean) * inverse_stddev * scale[i] + bias[i];
}}
'''


def _model_config_header(
    plan: VortexFFNPlan,
    post_opt_sha256: str,
    vortex_mlir_sha256: str,
) -> str:
    return f'''#ifndef GAC_BEVFORMER_VORTEX_MODEL_CONFIG_H
#define GAC_BEVFORMER_VORTEX_MODEL_CONFIG_H

#include <stdint.h>

static constexpr uint32_t kRows = {plan.rows}u;
static constexpr uint32_t kChannels = {plan.channels}u;
static constexpr uint32_t kHidden = {plan.hidden}u;
static constexpr float kLayerNormEpsilon = {plan.epsilon:.9g}f;
static constexpr const char* kOptimizedMlirSha256 = "{post_opt_sha256}";
static constexpr const char* kVortexMlirSha256 = "{vortex_mlir_sha256}";

#endif
'''


def _host_source(buffer_plan: VortexBufferPlan | None = None) -> str:
    source = r'''#include <vortex2.h>

#include "common.h"
#include "model_config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(expr) do {                                                     \
  vx_result_t result_ = (expr);                                              \
  if (result_ != VX_SUCCESS) {                                               \
    std::fprintf(stderr, "Vortex failure %s:%d: %s -> %s\n",               \
                 __FILE__, __LINE__, #expr, vx_result_string(result_));       \
    std::exit(2);                                                            \
  }                                                                          \
} while (0)

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> read_f32(const std::string& path, size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open " + path);
  const std::streamsize expected = static_cast<std::streamsize>(count * sizeof(float));
  if (stream.tellg() != expected)
    throw std::runtime_error("unexpected byte size for " + path);
  stream.seekg(0);
  std::vector<float> result(count);
  if (!stream.read(reinterpret_cast<char*>(result.data()), expected))
    throw std::runtime_error("cannot read " + path);
  return result;
}

std::string join_path(const std::string& root, const char* name) {
  if (root.empty() || root == ".")
    return std::string(name);
  const char last = root.back();
  return root + ((last == '/' || last == '\\') ? "" : "/") + name;
}

uint32_t parse_u32_option(const char* text, const char* option) {
  const std::string value = text ? text : "";
  if (value.empty())
    throw std::runtime_error(std::string(option) + " requires an unsigned integer");
  uint64_t parsed = 0;
  for (const char character : value) {
    if (character < '0' || character > '9')
      throw std::runtime_error(std::string(option) + " requires decimal digits only");
    const uint32_t digit = static_cast<uint32_t>(character - '0');
    if (parsed > (std::numeric_limits<uint32_t>::max() - digit) / 10u)
      throw std::runtime_error(std::string(option) + " exceeds uint32 range");
    parsed = parsed * 10u + digit;
  }
  return static_cast<uint32_t>(parsed);
}

struct DeviceBuffers {
  vx_buffer_h query = nullptr;
  vx_buffer_h w1 = nullptr;
  vx_buffer_h b1 = nullptr;
  vx_buffer_h activation = nullptr;
  vx_buffer_h w2 = nullptr;
  vx_buffer_h b2 = nullptr;
  vx_buffer_h residual = nullptr;
  vx_buffer_h scale = nullptr;
  vx_buffer_h norm_bias = nullptr;
  vx_buffer_h output = nullptr;
};

uint64_t address(vx_buffer_h buffer) {
  uint64_t result = 0;
  CHECK(vx_buffer_address(buffer, &result));
  if (result > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("device address exceeds the RV32 pointer ABI");
  return result;
}

vx_launch_info_t launch_info(vx_device_h device, vx_kernel_h kernel,
                             const void* args, size_t args_size,
                             uint32_t global_size) {
  uint32_t grid[1] = {1};
  uint32_t block[1] = {1};
  const uint32_t global[1] = {global_size};
  CHECK(vx_device_max_occupancy_grid(device, 1, global, grid, block));
  vx_launch_info_t info = {};
  info.struct_size = sizeof(info);
  info.kernel = kernel;
  info.args_host = args;
  info.args_size = args_size;
  info.ndim = 1;
  info.grid_dim[0] = grid[0];
  info.block_dim[0] = block[0];
  return info;
}

double profile_ms(vx_event_h event) {
  vx_profile_info_t profile = {};
  if (vx_event_get_profiling(event, &profile) != VX_SUCCESS ||
      profile.end_ns < profile.start_ns)
    return -1.0;
  return static_cast<double>(profile.end_ns - profile.start_ns) / 1.0e6;
}

struct IterationResult {
  double h2d_ms = 0.0;
  double host_kernel_wall_ms = 0.0;
  double device_kernel_ms = -1.0;
  double d2h_ms = 0.0;
  double end_to_end_ms = 0.0;
};

IterationResult run_once(
    vx_queue_h queue,
    vx_device_h device,
    vx_kernel_h fc1_kernel,
    vx_kernel_h fc2_kernel,
    vx_kernel_h norm_kernel,
    const DeviceBuffers& buffers,
    const std::vector<float>& query,
    std::vector<float>* output) {
  const uint64_t query_bytes = query.size() * sizeof(float);
  const uint64_t output_bytes = output->size() * sizeof(float);
  IterationResult timings;
  const auto e2e_start = Clock::now();

  vx_event_h write_event = nullptr;
  const auto h2d_start = Clock::now();
  CHECK(vx_enqueue_write(queue, buffers.query, 0, query.data(), query_bytes,
                         0, nullptr, &write_event));
  CHECK(vx_event_wait_value(write_event, 1, VX_TIMEOUT_INFINITE));
  timings.h2d_ms = elapsed_ms(h2d_start, Clock::now());
  CHECK(vx_event_release(write_event));

  fc1_args_t fc1 = {kRows, kChannels, kHidden, 0u,
                    address(buffers.query), address(buffers.w1),
                    address(buffers.b1), address(buffers.activation)};
  fc2_args_t fc2 = {kRows, kChannels, kHidden, 0u,
                    address(buffers.query), address(buffers.activation),
                    address(buffers.w2), address(buffers.b2),
                    address(buffers.residual)};
  layer_norm_args_t norm = {kRows, kChannels, kLayerNormEpsilon, 0u,
                            address(buffers.residual), address(buffers.scale),
                            address(buffers.norm_bias), address(buffers.output)};
  const vx_launch_info_t fc1_launch = launch_info(
      device, fc1_kernel, &fc1, sizeof(fc1), kRows * kHidden);
  const vx_launch_info_t fc2_launch = launch_info(
      device, fc2_kernel, &fc2, sizeof(fc2), kRows * kChannels);
  const vx_launch_info_t norm_launch = launch_info(
      device, norm_kernel, &norm, sizeof(norm), kRows);

  vx_event_h fc1_event = nullptr;
  vx_event_h fc2_event = nullptr;
  vx_event_h norm_event = nullptr;
  const auto kernel_start = Clock::now();
  CHECK(vx_enqueue_launch(queue, &fc1_launch, 0, nullptr, &fc1_event));
  CHECK(vx_enqueue_launch(queue, &fc2_launch, 1, &fc1_event, &fc2_event));
  CHECK(vx_enqueue_launch(queue, &norm_launch, 1, &fc2_event, &norm_event));
  CHECK(vx_event_wait_value(norm_event, 1, VX_TIMEOUT_INFINITE));
  timings.host_kernel_wall_ms = elapsed_ms(kernel_start, Clock::now());
  const double fc1_ms = profile_ms(fc1_event);
  const double fc2_ms = profile_ms(fc2_event);
  const double norm_ms = profile_ms(norm_event);
  if (fc1_ms >= 0.0 && fc2_ms >= 0.0 && norm_ms >= 0.0)
    timings.device_kernel_ms = fc1_ms + fc2_ms + norm_ms;

  vx_event_h read_event = nullptr;
  const auto d2h_start = Clock::now();
  CHECK(vx_enqueue_read(queue, output->data(), buffers.output, 0, output_bytes,
                        1, &norm_event, &read_event));
  CHECK(vx_event_wait_value(read_event, 1, VX_TIMEOUT_INFINITE));
  timings.d2h_ms = elapsed_ms(d2h_start, Clock::now());
  timings.end_to_end_ms = elapsed_ms(e2e_start, Clock::now());
  CHECK(vx_event_release(read_event));
  CHECK(vx_event_release(norm_event));
  CHECK(vx_event_release(fc2_event));
  CHECK(vx_event_release(fc1_event));
  return timings;
}

double mean(const std::vector<double>& values) {
  double total = 0.0;
  for (double value : values)
    total += value;
  return total / static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string data_dir = ".";
    std::string kernel_file = "kernel.vxbin";
    uint32_t warmup = 1;
    uint32_t repeat = 3;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--data-dir" && i + 1 < argc)
        data_dir = argv[++i];
      else if (arg == "--kernel" && i + 1 < argc)
        kernel_file = argv[++i];
      else if (arg == "--warmup" && i + 1 < argc)
        warmup = parse_u32_option(argv[++i], "--warmup");
      else if (arg == "--repeat" && i + 1 < argc)
        repeat = parse_u32_option(argv[++i], "--repeat");
      else {
        std::fprintf(stderr, "usage: %s [--data-dir DIR] [--kernel FILE] "
                             "[--warmup N] [--repeat N]\n", argv[0]);
        return 2;
      }
    }
    if (repeat == 0)
      throw std::runtime_error("repeat must be positive");

    const auto w1 = read_f32(join_path(data_dir, "fc1_weight.bin"),
                             static_cast<size_t>(kChannels) * kHidden);
    const auto b1 = read_f32(join_path(data_dir, "fc1_bias.bin"), kHidden);
    const auto w2 = read_f32(join_path(data_dir, "fc2_weight.bin"),
                             static_cast<size_t>(kHidden) * kChannels);
    const auto b2 = read_f32(join_path(data_dir, "fc2_bias.bin"), kChannels);
    const auto scale = read_f32(join_path(data_dir, "norm_weight.bin"), kChannels);
    const auto norm_bias = read_f32(join_path(data_dir, "norm_bias.bin"), kChannels);
    const auto query = read_f32(
        join_path(data_dir, "query.bin"), static_cast<size_t>(kRows) * kChannels);
    const auto expected = read_f32(
        join_path(data_dir, "golden.bin"), static_cast<size_t>(kRows) * kChannels);
    std::vector<float> output(expected.size());

    vx_device_h device = nullptr;
    CHECK(vx_device_open(0, &device));
    vx_queue_info_t queue_info = {};
    queue_info.struct_size = sizeof(queue_info);
    queue_info.priority = VX_QUEUE_PRIORITY_NORMAL;
    queue_info.flags = VX_QUEUE_PROFILING_ENABLE;
    vx_queue_h queue = nullptr;
    CHECK(vx_queue_create(device, &queue_info, &queue));

    DeviceBuffers buffers;
    const uint64_t query_bytes = query.size() * sizeof(float);
    const uint64_t activation_bytes = static_cast<uint64_t>(kRows) * kHidden * sizeof(float);
    const uint64_t output_bytes = expected.size() * sizeof(float);
    CHECK(vx_buffer_create(device, query_bytes, VX_MEM_READ, &buffers.query));
    CHECK(vx_buffer_create(device, w1.size() * sizeof(float), VX_MEM_READ, &buffers.w1));
    CHECK(vx_buffer_create(device, b1.size() * sizeof(float), VX_MEM_READ, &buffers.b1));
    CHECK(vx_buffer_create(device, activation_bytes, VX_MEM_READ_WRITE, &buffers.activation));
    CHECK(vx_buffer_create(device, w2.size() * sizeof(float), VX_MEM_READ, &buffers.w2));
    CHECK(vx_buffer_create(device, b2.size() * sizeof(float), VX_MEM_READ, &buffers.b2));
    CHECK(vx_buffer_create(device, output_bytes, VX_MEM_READ_WRITE, &buffers.residual));
    CHECK(vx_buffer_create(device, scale.size() * sizeof(float), VX_MEM_READ, &buffers.scale));
    CHECK(vx_buffer_create(device, norm_bias.size() * sizeof(float), VX_MEM_READ, &buffers.norm_bias));
    CHECK(vx_buffer_create(device, output_bytes, VX_MEM_WRITE, &buffers.output));

    auto upload = [&](vx_buffer_h buffer, const std::vector<float>& values) {
      vx_event_h event = nullptr;
      CHECK(vx_enqueue_write(queue, buffer, 0, values.data(),
                             values.size() * sizeof(float), 0, nullptr, &event));
      CHECK(vx_event_wait_value(event, 1, VX_TIMEOUT_INFINITE));
      CHECK(vx_event_release(event));
    };
    upload(buffers.w1, w1);
    upload(buffers.b1, b1);
    upload(buffers.w2, w2);
    upload(buffers.b2, b2);
    upload(buffers.scale, scale);
    upload(buffers.norm_bias, norm_bias);

    vx_module_h module = nullptr;
    CHECK(vx_module_load_file(device, kernel_file.c_str(), &module));
    vx_kernel_h fc1_kernel = nullptr;
    vx_kernel_h fc2_kernel = nullptr;
    vx_kernel_h norm_kernel = nullptr;
    CHECK(vx_module_get_kernel(module, "bevformer_fc1_bias_relu", &fc1_kernel));
    CHECK(vx_module_get_kernel(module, "bevformer_fc2_bias_residual", &fc2_kernel));
    CHECK(vx_module_get_kernel(module, "bevformer_layer_norm", &norm_kernel));

    for (uint32_t i = 0; i < warmup; ++i)
      run_once(queue, device, fc1_kernel, fc2_kernel, norm_kernel,
               buffers, query, &output);

    std::vector<double> h2d, kernel_wall, kernel_device, d2h, e2e;
    double max_abs_error = 0.0;
    bool finite = true;
    for (uint32_t i = 0; i < repeat; ++i) {
      const IterationResult result = run_once(
          queue, device, fc1_kernel, fc2_kernel, norm_kernel,
          buffers, query, &output);
      h2d.push_back(result.h2d_ms);
      kernel_wall.push_back(result.host_kernel_wall_ms);
      if (result.device_kernel_ms >= 0.0)
        kernel_device.push_back(result.device_kernel_ms);
      d2h.push_back(result.d2h_ms);
      e2e.push_back(result.end_to_end_ms);
      for (size_t j = 0; j < output.size(); ++j) {
        finite = finite && std::isfinite(output[j]);
        max_abs_error = std::max(
            max_abs_error,
            static_cast<double>(std::fabs(output[j] - expected[j])));
      }
    }
    const double tolerance = 1.0e-3;
    const bool passed = finite && max_abs_error <= tolerance;
    const char* driver = std::getenv("VORTEX_DRIVER");
    std::printf(
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"backend\": \"vortex-v3\",\n"
        "  \"driver\": \"%s\",\n"
        "  \"shape\": [%u, %u, %u],\n"
        "  \"hidden\": %u,\n"
        "  \"warmup\": %u,\n"
        "  \"repeat\": %u,\n"
        "  \"h2d_mean_ms\": %.6f,\n"
        "  \"kernel_host_wall_mean_ms\": %.6f,\n"
        "  \"kernel_device_profile_mean_ms\": %.6f,\n"
        "  \"d2h_mean_ms\": %.6f,\n"
        "  \"end_to_end_mean_ms\": %.6f,\n"
        "  \"max_abs_error\": %.9g,\n"
        "  \"tolerance\": %.9g,\n"
        "  \"finite\": %s,\n"
        "  \"verified\": %s,\n"
        "  \"optimized_mlir_sha256\": \"%s\",\n"
        "  \"vortex_call_plan_mlir_sha256\": \"%s\"\n"
        "}\n",
        driver ? driver : "unset", 1u, kRows, kChannels, kHidden,
        warmup, repeat, mean(h2d), mean(kernel_wall),
        kernel_device.empty() ? -1.0 : mean(kernel_device),
        mean(d2h), mean(e2e), max_abs_error, tolerance,
        finite ? "true" : "false", passed ? "true" : "false",
        kOptimizedMlirSha256, kVortexMlirSha256);
    vx_device_dump_perf(device, stdout);

    CHECK(vx_kernel_release(norm_kernel));
    CHECK(vx_kernel_release(fc2_kernel));
    CHECK(vx_kernel_release(fc1_kernel));
    CHECK(vx_module_release(module));
    CHECK(vx_buffer_release(buffers.output));
    CHECK(vx_buffer_release(buffers.norm_bias));
    CHECK(vx_buffer_release(buffers.scale));
    CHECK(vx_buffer_release(buffers.residual));
    CHECK(vx_buffer_release(buffers.b2));
    CHECK(vx_buffer_release(buffers.w2));
    CHECK(vx_buffer_release(buffers.activation));
    CHECK(vx_buffer_release(buffers.b1));
    CHECK(vx_buffer_release(buffers.w1));
    CHECK(vx_buffer_release(buffers.query));
    CHECK(vx_queue_release(queue));
    CHECK(vx_device_release(device));
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "gac-vortex: %s\n", error.what());
    return 2;
  }
}
'''
    if buffer_plan is None:
        return source
    return _buffered_host_source(source, buffer_plan)


def _buffered_host_source(source: str, plan: VortexBufferPlan) -> str:
    source = source.replace("#include <vortex2.h>", '''#include <vortex2.h>
#include <dxa.h>
#include <VX_config.h>
static_assert(VX_CFG_NUM_BARRIERS >= 2,
              "buffered MLIR requires two independent transaction barrier slots");''')
    constants = f'''static constexpr uint32_t kTileM = {plan.tile_m}u;
static constexpr uint32_t kTileN = {plan.tile_n}u;
static constexpr uint32_t kTileK = {plan.tile_k}u;
static constexpr uint32_t kTileThreads = {plan.threads}u;
static constexpr uint32_t kTileLmemBytes = {plan.lmem_bytes}u;
'''
    source = source.replace("namespace {\n", "namespace {\n\n" + constants, 1)
    launch = '''vx_launch_info_t tiled_launch_info(vx_device_h device, vx_kernel_h kernel,
                                   const void* args, size_t args_size,
                                   uint32_t rows, uint32_t columns) {
  if (vx_check_occupancy(device, kTileThreads, kTileLmemBytes) != VX_SUCCESS)
    throw std::runtime_error("tile threads or LMEM exceed device occupancy limits");
  vx_launch_info_t info = {};
  info.struct_size = sizeof(info);
  info.kernel = kernel;
  info.args_host = args;
  info.args_size = args_size;
  info.ndim = 1;
  info.grid_dim[0] = ((rows + kTileM - 1) / kTileM)
                   * ((columns + kTileN - 1) / kTileN);
  info.block_dim[0] = kTileThreads;
  info.lmem_size = kTileLmemBytes;
  return info;
}

void program_tiles(vx_device_h device, const DeviceBuffers& buffers) {
  uint64_t isa = 0, threads = 0;
  CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa));
  CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads));
  if ((isa & VX_ISA_EXT_DXA) == 0 || (isa & VX_ISA_EXT_LMEM) == 0 || threads < 4)
    throw std::runtime_error("buffered MLIR requires DXA, LMEM and at least four lanes");
  auto descriptor = [&](uint32_t slot, uint64_t base, uint32_t columns,
                        uint32_t rows, uint32_t tile_columns, uint32_t tile_rows) {
    if (vortex::dxa::program_2d(device, slot, base, columns, rows,
                              columns * sizeof(float), tile_columns, tile_rows,
                              sizeof(float)) != VX_SUCCESS)
      throw std::runtime_error("cannot program physical DXA tile descriptor");
  };
  descriptor(0, address(buffers.query), kChannels, kRows, kTileK, kTileM);
  descriptor(1, address(buffers.w1), kHidden, kChannels, kTileN, kTileK);
  descriptor(2, address(buffers.activation), kHidden, kRows, kTileK, kTileM);
  descriptor(3, address(buffers.w2), kChannels, kHidden, kTileN, kTileK);
}

'''
    source = source.replace("double profile_ms(vx_event_h event)", launch + "double profile_ms(vx_event_h event)", 1)
    for kernel, dimension in (("fc1", "kHidden"), ("fc2", "kChannels")):
        old = f"launch_info(\n      device, {kernel}_kernel, &{kernel}, sizeof({kernel}), kRows * {dimension})"
        new = f"tiled_launch_info(\n      device, {kernel}_kernel, &{kernel}, sizeof({kernel}), kRows, {dimension})"
        source = source.replace(old, new, 1)
    # DXA bypasses per-core address translation, so its source addresses must be physical.
    for flags in ("VX_MEM_READ_WRITE", "VX_MEM_READ", "VX_MEM_WRITE"):
        source = source.replace(f"{flags}, &buffers.", f"{flags} | VX_MEM_PHYS, &buffers.")
    return source.replace("    upload(buffers.norm_bias, norm_bias);", "    upload(buffers.norm_bias, norm_bias);\n    program_tiles(device, buffers);", 1)


def _makefile() -> str:
    return r'''ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

ifneq ($(XLEN),32)
$(error this GAC Vortex artifact targets the validated RV32 ABI; use an XLEN=32 build)
endif

PROJECT := gac_bevformer_vortex
SRC_DIR := $(CURDIR)
SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/model_config.h
HDRS := $(VX_HDRS)
KERNEL_LIB := vortex2
OPTS ?= --data-dir . --kernel kernel.vxbin --warmup 1 --repeat 3

include ../common.mk

.PHONY: verify-artifact
verify-artifact:
	python3 verify_artifact.py

# GNU make merges these prerequisites with the rules in common.mk.  Order-only
# edges prevent both parallel builds and direct run targets from consuming an
# artifact before its complete manifest has been checked.
.depend vx_start.o $(APP) kernel.elf kernel.vxbin kernel.dump: | verify-artifact
run-simx run-rtlsim run-firesim run-opae run-xrt run-aved: | verify-artifact

all: verify-artifact
'''


def _direct_makefile() -> str:
    return r'''ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

ifneq ($(XLEN),32)
$(error this GAC Vortex artifact targets the validated RV32 ABI; use an XLEN=32 build)
endif

PROJECT := gac_bevformer_vortex
SRC_DIR := $(CURDIR)
SRCS := $(SRC_DIR)/main.cpp
VOLT_LLVM_REVISION_EXPECTED := b2ecfe086df2d00233a87f3d3ebe6a74aba129e2
override MLIR_OPT := $(LLVM_PATH)/bin/mlir-opt
override MLIR_TRANSLATE := $(LLVM_PATH)/bin/mlir-translate
VX_SRCS := kernel.ll
VX_HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/model_config.h
HDRS := $(VX_HDRS)
KERNEL_LIB := vortex2
OPTS ?= --data-dir . --kernel kernel.vxbin --warmup 1 --repeat 3

include ../common.mk

# Keep the final ELF link in the same pinned LLVM tree as MLIR translation and
# Vortex instruction selection.  The Vortex toolchain documents ld.lld as its
# device linker; spelling it explicitly avoids accidentally selecting a GNU ld
# from RISCV_TOOLCHAIN_PATH.
VX_LDFLAGS += -fuse-ld=lld --ld-path=$(LLVM_PATH)/bin/ld.lld

.PHONY: verify-artifact
verify-artifact:
	python3 verify_artifact.py

.PHONY: verify-toolchain
verify-toolchain:
	test -x "$(MLIR_OPT)"
	test -x "$(MLIR_TRANSLATE)"
	test -x "$(LLVM_PATH)/bin/clang"
	test -x "$(LLVM_PATH)/bin/clang++"
	test -x "$(LLVM_PATH)/bin/ld.lld"
	test -x "$(LLVM_PATH)/bin/llvm-objdump"
	test -x "$(LLVM_PATH)/bin/llvm-objcopy"
	test -r "$(LLVM_PATH)/.gac-build-revision"
	test "$$(cat "$(LLVM_PATH)/.gac-build-revision")" = "$(VOLT_LLVM_REVISION_EXPECTED)"
	$(MLIR_OPT) --version | grep -F "20.1.8" >/dev/null
	$(MLIR_TRANSLATE) --version | grep -F "20.1.8" >/dev/null
	$(LLVM_PATH)/bin/clang --version | grep -F "20.1.8" >/dev/null
	$(LLVM_PATH)/bin/clang++ --version | grep -F "20.1.8" >/dev/null
	$(LLVM_PATH)/bin/ld.lld --version | grep -F "20.1.8" >/dev/null

# Both MLIR tools and VOLT clang come from LLVM_PATH.  This deliberately keeps
# LLVM dialect parsing, LLVM IR translation, and Vortex code generation on the
# same LLVM 20.1.8 source revision.
.PHONY: kernel.opt.mlir kernel.ll vx_start.o $(APP) kernel.elf kernel.vxbin kernel.dump

# Build outputs are intentionally outside the source-artifact manifest.  Force
# the complete derived chain to rebuild so a pre-existing or future-dated
# kernel.ll/object/ELF/vxbin can never bypass the manifest-bound kernel.mlir.
kernel.opt.mlir: kernel.mlir | verify-artifact verify-toolchain
	$(MLIR_OPT) --verify-each --canonicalize --cse $< -o $@

kernel.ll: kernel.opt.mlir
	$(MLIR_TRANSLATE) --mlir-to-llvmir $< -o $@

.PHONY: clean-direct-mlir
clean-direct-mlir:
	rm -f kernel.opt.mlir kernel.ll

clean-kernel: clean-direct-mlir

# GNU make merges these prerequisites with the rules in common.mk.  Order-only
# edges prevent both parallel builds and direct run targets from consuming an
# artifact before its complete manifest has been checked.
kernel.opt.mlir kernel.ll vx_start.o $(APP) kernel.elf kernel.vxbin kernel.dump: | verify-artifact verify-toolchain
run-simx run-rtlsim run-firesim run-opae run-xrt run-aved: | verify-artifact

all: verify-artifact
'''


def _artifact_verifier_source(
    *,
    device_file: str = "kernel.cpp",
    bind_device_mlir: bool = False,
    buffer_plan: VortexBufferPlan | None = None,
) -> str:
    source = r'''#!/usr/bin/env python3
"""Verify the generated source artifact before VOLT consumes it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


EXPECTED_FILES = {
    "Makefile",
    "common.h",
    "fc1_bias.bin",
    "fc1_weight.bin",
    "fc2_bias.bin",
    "fc2_weight.bin",
    "frontend.mlir",
    "golden.bin",
    "__GAC_DEVICE_FILE__",
    "main.cpp",
    "model_config.h",
    "norm_bias.bin",
    "norm_weight.bin",
    "optimized.mlir",
    "query.bin",
    "verify_artifact.py",
    "vortex.mlir",
}


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token):
    raise ValueError(f"non-finite JSON number {token!r}")


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported manifest schema")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("manifest files must be a non-empty object")
        if set(files) != EXPECTED_FILES:
            missing = sorted(EXPECTED_FILES - set(files))
            unexpected = sorted(set(files) - EXPECTED_FILES)
            raise ValueError(
                f"manifest file set mismatch: missing={missing}, unexpected={unexpected}"
            )
        for name, record in files.items():
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or name in {".", "..", "manifest.json"}
                or not isinstance(record, dict)
            ):
                raise ValueError(f"invalid file record {name!r}")
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or unsafe artifact file {name!r}")
            payload = path.read_bytes()
            if record.get("bytes") != len(payload):
                raise ValueError(f"size mismatch for {name!r}")
            digest = hashlib.sha256(payload).hexdigest()
            if record.get("sha256") != digest:
                raise ValueError(f"SHA-256 mismatch for {name!r}")
        lowering = manifest.get("lowering", {})
        if files["frontend.mlir"]["sha256"] != lowering.get(
            "pre_opt_mlir_sha256"
        ):
            raise ValueError("frontend MLIR hash is not bound to lowering metadata")
        if files["optimized.mlir"]["sha256"] != lowering.get(
            "post_opt_mlir_sha256"
        ):
            raise ValueError("optimized MLIR hash is not bound to lowering metadata")
        if files["vortex.mlir"]["sha256"] != lowering.get(
            "vortex_call_plan_mlir_sha256"
        ):
            raise ValueError("Vortex MLIR hash is not bound to lowering metadata")
__GAC_DEVICE_HASH_CHECK__
        vortex_text = (root / "vortex.mlir").read_text(encoding="utf-8")
        weight_bindings = {
            match.group("name"): match.group("digest")
            for match in re.finditer(
                r"gac\.(?P<name>[A-Za-z0-9_]+)_sha256\s*=\s*"
                r'"(?P<digest>[0-9a-f]{64})"',
                vortex_text,
            )
        }
        expected_weight_names = {
            "fc1_weight",
            "fc1_bias",
            "fc2_weight",
            "fc2_bias",
            "norm_weight",
            "norm_bias",
        }
        if set(weight_bindings) != expected_weight_names:
            raise ValueError("Vortex MLIR external weight binding set is incomplete")
        if weight_bindings != lowering.get("external_weight_sha256"):
            raise ValueError("Vortex MLIR weight hashes are not bound to metadata")
        for name, digest in weight_bindings.items():
            if files[f"{name}.bin"]["sha256"] != digest:
                raise ValueError(f"external weight {name!r} does not match Vortex MLIR")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        print(f"gac-vortex artifact verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"artifact_verified": True, "file_count": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    hash_check = ""
    if bind_device_mlir:
        hash_check = '''        if files["kernel.mlir"]["sha256"] != lowering.get(
            "device_kernel_mlir_sha256"
        ):
            raise ValueError("device kernel MLIR hash is not bound to lowering metadata")'''
    if buffer_plan is not None:
        hash_check += f'''\n        if lowering.get("buffer_plan") != {buffer_plan.to_dict()!r}:
            raise ValueError("buffer plan does not match the generated host and device")'''
    return source.replace("__GAC_DEVICE_FILE__", device_file).replace(
        "__GAC_DEVICE_HASH_CHECK__", hash_check
    )


def compile_vortex(
    graph: Graph,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> VortexProgram:
    """Compile one BEVFormer FFN graph to a Vortex source-level program.

    ``mlir-opt`` is mandatory.  The function never silently skips the MLIR
    stage and never substitutes CUDA/PTX execution.
    """

    mlir_program = compile_mlir(graph, fuse=True)
    graph_plan = _match_bevformer_ffn(mlir_program)
    optimized = verify_mlir(
        mlir_program.source,
        mlir_opt=mlir_opt,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
    )
    optimized_plan = _parse_optimized_mlir(optimized, graph_plan)
    _audit_optimized_weights(optimized, mlir_program.graph)
    optimized_hash = _sha256_text(optimized)
    weight_sha256 = _weight_sha256(mlir_program.graph)
    vortex_mlir_input = _runtime_call_mlir(
        optimized_plan,
        optimized_hash,
        weight_sha256,
    )
    vortex_mlir = verify_mlir(
        vortex_mlir_input,
        mlir_opt=mlir_opt,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
    )
    plan = _parse_vortex_mlir(
        vortex_mlir,
        optimized_sha256=optimized_hash,
        expected=optimized_plan,
        expected_weight_sha256=weight_sha256,
    )
    version = probe_mlir_opt_version(
        mlir_opt=mlir_opt,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
    )
    revision = None
    if mlir_opt is not None:
        revision = probe_mlir_opt_source_revision(
            mlir_opt,
            use_wsl=use_wsl,
            wsl_distribution=wsl_distribution,
            timeout=timeout,
        )
    vortex_mlir_hash = _sha256_text(vortex_mlir)
    return VortexProgram(
        graph=mlir_program.graph,
        reference_graph=mlir_program.reference_graph,
        mlir_program=mlir_program,
        optimized_mlir=optimized,
        vortex_mlir=vortex_mlir,
        mlir_opt_version=version,
        mlir_opt_source_revision=revision,
        plan=plan,
        kernel_source=_kernel_source(optimized_hash, vortex_mlir_hash),
        common_header=_common_header(),
        host_source=_host_source(),
        model_config_header=_model_config_header(
            plan,
            optimized_hash,
            vortex_mlir_hash,
        ),
        makefile=_makefile(),
        verifier_source=_artifact_verifier_source(),
    )


def _require_volt_llvm20_toolchain(program: VortexProgram) -> None:
    if re.search(r"\b(?:LLVM\s+)?version\s+20\.1\.8\b", program.mlir_opt_version) is None:
        raise BackendError(
            f"{TARGET}: direct MLIR lowering requires VOLT LLVM {VOLT_LLVM_VERSION}; "
            f"selected mlir-opt reported {program.mlir_opt_version!r}"
        )
    revision = program.mlir_opt_source_revision
    if revision != VOLT_LLVM_REVISION:
        raise BackendError(
            f"{TARGET}: direct MLIR lowering requires VOLT revision "
            f"{VOLT_LLVM_REVISION}, selected mlir-opt records {revision!r}; "
            "build it with tools/build_volt_mlir_wsl.sh so the revision marker "
            "is installed next to the toolchain"
        )


def compile_vortex_direct(
    graph: Graph,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    buffer_plan: VortexBufferPlan | None = None,
) -> VortexProgram:
    """Compile the supported BEVFormer FFN to device LLVM-dialect MLIR.

    This mode removes the generated device C++ kernel.  The returned artifact
    translates ``kernel.mlir`` with ``mlir-translate`` and sends the resulting
    LLVM IR directly through the VOLT RISC-V/Vortex backend.  All LLVM tools
    must come from the pinned LLVM 20.1.8 Vortex fork.
    """

    program = compile_vortex(
        graph,
        mlir_opt=mlir_opt,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
    )
    _require_volt_llvm20_toolchain(program)

    # Keep this import local so the legacy C++ bridge remains independently
    # usable and so the device emitter has no import cycle with VortexProgram.
    from .vortex_device_mlir import (
        audit_bevformer_ffn_device_mlir,
        emit_bevformer_ffn_device_mlir,
    )

    optimized_hash = _sha256_text(program.optimized_mlir)
    call_plan_hash = _sha256_text(program.vortex_mlir)
    emitter = emit_bevformer_ffn_device_mlir
    auditor = audit_bevformer_ffn_device_mlir
    schedule_args: dict[str, Any] = {}
    if buffer_plan is not None:
        if not isinstance(buffer_plan, VortexBufferPlan):
            raise BackendError(f"{TARGET}: buffer_plan must be a VortexBufferPlan")
        emitter = emit_bevformer_ffn_buffered_device_mlir
        auditor = audit_bevformer_ffn_buffered_device_mlir
        schedule_args["buffer_plan"] = buffer_plan
    device_input = emitter(
        rows=program.plan.rows,
        channels=program.plan.channels,
        hidden=program.plan.hidden,
        epsilon=program.plan.epsilon,
        optimized_mlir_sha256=optimized_hash,
        call_plan_mlir_sha256=call_plan_hash,
        **schedule_args,
    )
    device_mlir = verify_mlir(
        device_input,
        mlir_opt=mlir_opt,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
    )
    auditor(
        device_mlir,
        kernels=program.plan.kernels,
        rows=program.plan.rows,
        channels=program.plan.channels,
        hidden=program.plan.hidden,
        epsilon=program.plan.epsilon,
        optimized_mlir_sha256=optimized_hash,
        call_plan_mlir_sha256=call_plan_hash,
        **schedule_args,
    )
    return replace(
        program,
        kernel_source="",
        makefile=(
            _direct_makefile() + "\nVX_CFLAGS += -ffixed-x31\n"
            if buffer_plan is not None else _direct_makefile()
        ),
        verifier_source=_artifact_verifier_source(
            device_file="kernel.mlir",
            bind_device_mlir=True,
            buffer_plan=buffer_plan,
        ),
        device_lowering=DIRECT_MLIR_LOWERING,
        device_mlir=device_mlir,
        host_source=_host_source(buffer_plan),
        buffer_plan=buffer_plan,
    )


def _write_f32(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f4")
    path.write_bytes(array.tobytes(order="C"))


def _deterministic_query(plan: VortexFFNPlan) -> np.ndarray:
    count = plan.rows * plan.channels
    indices = np.arange(count, dtype=np.uint64)
    values = ((indices * 17 + 13) % 101).astype(np.int32) - 50
    return np.ascontiguousarray(
        (values.astype(np.float32) / np.float32(32.0)).reshape(plan.input_shape)
    )


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_vortex_artifact(
    destination: str | os.PathLike[str],
    program: VortexProgram,
) -> dict[str, Any]:
    """Atomically publish a self-contained Vortex regression project."""

    if not isinstance(program, VortexProgram):
        raise TypeError("program must be a VortexProgram")
    target = Path(destination)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Vortex artifact destination must not exist or must be empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        text_files: dict[str, str] = {
            "frontend.mlir": program.mlir_program.source,
            "optimized.mlir": program.optimized_mlir,
            "vortex.mlir": program.vortex_mlir,
            "common.h": program.common_header,
            "main.cpp": program.host_source,
            "model_config.h": program.model_config_header,
            "Makefile": program.makefile,
            "verify_artifact.py": program.verifier_source,
        }
        if program.device_lowering == DIRECT_MLIR_LOWERING:
            if program.device_mlir is None or program.kernel_source:
                raise BackendError(f"{TARGET}: invalid direct MLIR artifact state")
            text_files["kernel.mlir"] = program.device_mlir
        else:
            if program.device_mlir is not None:
                raise BackendError(f"{TARGET}: invalid legacy C++ artifact state")
            text_files["kernel.cpp"] = program.kernel_source
        for name, contents in text_files.items():
            (staging / name).write_text(contents, encoding="utf-8", newline="\n")
        for name in _REQUIRED_CONSTANTS:
            _write_f32(staging / f"{name}.bin", program.graph.constants[name])
        query = _deterministic_query(program.plan)
        golden_outputs = reference_run(
            program.reference_graph,
            {"query": query},
        )
        golden = np.asarray(golden_outputs["output"])
        if (
            golden.shape != program.plan.output_shape
            or golden.dtype != np.float32
            or not np.isfinite(golden).all()
        ):
            raise BackendError(
                f"{TARGET}: NumPy oracle produced an invalid output contract"
            )
        _write_f32(staging / "query.bin", query)
        _write_f32(staging / "golden.bin", golden)

        manifest = program.describe()
        manifest["files"] = {
            path.name: _file_record(path)
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
        return {
            **manifest,
            "artifact_directory": str(target.resolve(strict=False)),
            "manifest_sha256": hashlib.sha256(
                (target / "manifest.json").read_bytes()
            ).hexdigest(),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DIRECT_MLIR_LOWERING",
    "LEGACY_CPP_LOWERING",
    "TARGET",
    "VOLT_LLVM_REVISION",
    "VOLT_LLVM_VERSION",
    "VortexFFNPlan",
    "VortexProgram",
    "compile_vortex",
    "compile_vortex_direct",
    "write_vortex_artifact",
]
