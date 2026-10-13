"""Direct LLVM-dialect MLIR device kernels for the Vortex/VOLT backend.

This module is intentionally independent from :mod:`gac.backends.vortex`.
It emits the first direct device lowering boundary used by that backend:

``LLVM dialect MLIR -> LLVM IR -> VOLT RISC-V backend -> ELF -> vxbin``.

The generated entry points use the Vortex CTA CSR interface and the RV32 host
argument blobs already consumed by the legacy C++ kernels.  Kernel discovery
is represented in the LLVM IR itself through ``llvm.global.annotations``;
``llvm.used``, ``llvm.compiler.used``, and a ``.vx_entry.*`` section keep each
address-dispatched body alive through optimization and linker garbage
collection.
"""

from __future__ import annotations

import math
import re
from textwrap import dedent, indent

from ..errors import BackendError


TARGET_TRIPLE = "riscv32-unknown-elf"
DATA_LAYOUT = "e-m:e-p:32:32-i64:64-n32-S128"

CTA_THREAD_ID_X_CSR = 0xCD3
CTA_BLOCK_ID_X_CSR = 0xCD6
CTA_BLOCK_DIM_X_CSR = 0xCD9

VECTOR_ADD_KERNEL = "vector_add"
BEVFORMER_FFN_KERNELS = (
    "bevformer_fc1_bias_relu",
    "bevformer_fc2_bias_residual",
    "bevformer_layer_norm",
)

_UINT32_MAX = (1 << 32) - 1
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_SYMBOL_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$]*\Z")


def _require_positive_u32(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BackendError(f"vortex-v3: {name} must be an integer")
    if value <= 0 or value > _UINT32_MAX:
        raise BackendError(
            f"vortex-v3: {name} must be in the inclusive range 1..{_UINT32_MAX}"
        )
    return value


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BackendError(
            f"vortex-v3: {name} must be a 64-character hexadecimal SHA-256"
        )
    return value.lower()


def _require_kernel_tuple(kernels: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(kernels, tuple) or not kernels:
        raise BackendError("vortex-v3: kernels must be a non-empty tuple")
    if len(set(kernels)) != len(kernels):
        raise BackendError("vortex-v3: kernels must not contain duplicates")
    for kernel in kernels:
        if not isinstance(kernel, str) or _SYMBOL_RE.fullmatch(kernel) is None:
            raise BackendError(f"vortex-v3: invalid kernel symbol {kernel!r}")
    return kernels


def _annotation_globals(kernels: tuple[str, ...]) -> str:
    """Emit Clang-compatible annotation and retention globals."""

    annotation_type = "!llvm.struct<(ptr, ptr, ptr, i32, ptr)>"
    annotations_type = f"!llvm.array<{len(kernels)} x struct<(ptr, ptr, ptr, i32, ptr)>>"
    used_type = f"!llvm.array<{len(kernels)} x ptr>"

    annotation_lines = [
        f"%annotations0 = llvm.mlir.undef : {annotations_type}",
        f"%annotation_record0 = llvm.mlir.undef : {annotation_type}",
        "%annotation_text = llvm.mlir.addressof @__gac_vortex_kernel_annotation : !llvm.ptr",
        "%source_file = llvm.mlir.addressof @__gac_vortex_source_file : !llvm.ptr",
        "%source_line = llvm.mlir.constant(1 : i32) : i32",
        "%null = llvm.mlir.zero : !llvm.ptr",
    ]
    previous_array = "%annotations0"
    for index, kernel in enumerate(kernels):
        record = f"%annotation_record{index}"
        if index:
            annotation_lines.append(
                f"{record} = llvm.mlir.undef : {annotation_type}"
            )
        annotation_lines.extend(
            (
                f"%kernel{index} = llvm.mlir.addressof @{kernel} : !llvm.ptr",
                f"%record{index}_1 = llvm.insertvalue %kernel{index}, {record}[0] : {annotation_type}",
                f"%record{index}_2 = llvm.insertvalue %annotation_text, %record{index}_1[1] : {annotation_type}",
                f"%record{index}_3 = llvm.insertvalue %source_file, %record{index}_2[2] : {annotation_type}",
                f"%record{index}_4 = llvm.insertvalue %source_line, %record{index}_3[3] : {annotation_type}",
                f"%record{index}_5 = llvm.insertvalue %null, %record{index}_4[4] : {annotation_type}",
                f"%annotations{index + 1} = llvm.insertvalue %record{index}_5, {previous_array}[{index}] : {annotations_type}",
            )
        )
        previous_array = f"%annotations{index + 1}"
    annotation_lines.append(f"llvm.return {previous_array} : {annotations_type}")

    used_lines = [f"%used0 = llvm.mlir.undef : {used_type}"]
    previous_used = "%used0"
    for index, kernel in enumerate(kernels):
        used_lines.extend(
            (
                f"%used_kernel{index} = llvm.mlir.addressof @{kernel} : !llvm.ptr",
                f"%used{index + 1} = llvm.insertvalue %used_kernel{index}, {previous_used}[{index}] : {used_type}",
            )
        )
        previous_used = f"%used{index + 1}"
    used_lines.append(f"llvm.return {previous_used} : {used_type}")

    compiler_used_lines = [
        f"%compiler_used0 = llvm.mlir.undef : {used_type}"
    ]
    previous_compiler_used = "%compiler_used0"
    for index, kernel in enumerate(kernels):
        compiler_used_lines.extend(
            (
                f"%compiler_used_kernel{index} = llvm.mlir.addressof @{kernel} : !llvm.ptr",
                f"%compiler_used{index + 1} = llvm.insertvalue %compiler_used_kernel{index}, {previous_compiler_used}[{index}] : {used_type}",
            )
        )
        previous_compiler_used = f"%compiler_used{index + 1}"
    compiler_used_lines.append(
        f"llvm.return {previous_compiler_used} : {used_type}"
    )

    return "\n".join(
        (
            'llvm.mlir.global private unnamed_addr constant @__gac_vortex_kernel_annotation("vortex.kernel\\00") {section = "llvm.metadata"} : !llvm.array<14 x i8>',
            'llvm.mlir.global private unnamed_addr constant @__gac_vortex_source_file("gac.mlir\\00") {section = "llvm.metadata"} : !llvm.array<9 x i8>',
            "",
            f'llvm.mlir.global appending @llvm.global.annotations() {{section = "llvm.metadata"}} : {annotations_type} {{',
            indent("\n".join(annotation_lines), "  "),
            "}",
            "",
            f'llvm.mlir.global appending @llvm.used() {{section = "llvm.metadata"}} : {used_type} {{',
            indent("\n".join(used_lines), "  "),
            "}",
            "",
            f'llvm.mlir.global appending @llvm.compiler.used() {{section = "llvm.metadata"}} : {used_type} {{',
            indent("\n".join(compiler_used_lines), "  "),
            "}",
        )
    )


def _module(kernels: tuple[str, ...], body: str, metadata: str = "") -> str:
    pieces = [_annotation_globals(kernels)]
    if metadata:
        pieces.append(metadata.strip())
    pieces.append(body.strip())
    contents = indent("\n\n".join(pieces), "  ")
    return (
        f'module attributes {{llvm.data_layout = "{DATA_LAYOUT}", '
        f'llvm.target_triple = "{TARGET_TRIPLE}"}} {{\n'
        f"{contents}\n"
        "}\n"
    )


def _vector_add_kernel() -> str:
    return dedent(
        f'''\
        llvm.func @{VECTOR_ADD_KERNEL}(%arg_blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel"], section = ".vx_entry.{VECTOR_ADD_KERNEL}"}} {{
          %args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg_blob) : (!llvm.ptr) -> !llvm.ptr
          %n_ptr = llvm.getelementptr %args[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64, i64, i64)>
          %lhs_addr_ptr = llvm.getelementptr %args[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64, i64, i64)>
          %rhs_addr_ptr = llvm.getelementptr %args[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64, i64, i64)>
          %out_addr_ptr = llvm.getelementptr %args[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i64, i64, i64)>
          %n = llvm.load %n_ptr : !llvm.ptr -> i32
          %lhs_addr = llvm.load %lhs_addr_ptr : !llvm.ptr -> i64
          %rhs_addr = llvm.load %rhs_addr_ptr : !llvm.ptr -> i64
          %out_addr = llvm.load %out_addr_ptr : !llvm.ptr -> i64

          %thread_csr = llvm.mlir.constant({CTA_THREAD_ID_X_CSR} : i32) : i32
          %block_csr = llvm.mlir.constant({CTA_BLOCK_ID_X_CSR} : i32) : i32
          %block_dim_csr = llvm.mlir.constant({CTA_BLOCK_DIM_X_CSR} : i32) : i32
          %thread_id = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %index = llvm.add %block_base, %thread_id : i32
          %in_bounds = llvm.icmp "ult" %index, %n : i32
          llvm.cond_br %in_bounds, ^compute, ^exit

        ^compute:
          %lhs = llvm.inttoptr %lhs_addr : i64 to !llvm.ptr
          %rhs = llvm.inttoptr %rhs_addr : i64 to !llvm.ptr
          %out = llvm.inttoptr %out_addr : i64 to !llvm.ptr
          %lhs_element = llvm.getelementptr %lhs[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %rhs_element = llvm.getelementptr %rhs[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %out_element = llvm.getelementptr %out[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %lhs_value = llvm.load %lhs_element : !llvm.ptr -> f32
          %rhs_value = llvm.load %rhs_element : !llvm.ptr -> f32
          %sum = llvm.fadd %lhs_value, %rhs_value : f32
          llvm.store %sum, %out_element : f32, !llvm.ptr
          llvm.br ^exit

        ^exit:
          llvm.return
        }}
        '''
    )


def emit_vector_add_device_mlir() -> str:
    """Return the minimal direct Vortex VectorAdd device module."""

    return _module((VECTOR_ADD_KERNEL,), _vector_add_kernel())


def _fc1_kernel() -> str:
    name = BEVFORMER_FFN_KERNELS[0]
    return dedent(
        f'''\
        llvm.func @{name}(%arg_blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel"], section = ".vx_entry.{name}"}} {{
          %args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg_blob) : (!llvm.ptr) -> !llvm.ptr
          %rows_ptr = llvm.getelementptr %args[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %channels_ptr = llvm.getelementptr %args[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %hidden_ptr = llvm.getelementptr %args[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %query_addr_ptr = llvm.getelementptr %args[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %weight_addr_ptr = llvm.getelementptr %args[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %bias_addr_ptr = llvm.getelementptr %args[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %output_addr_ptr = llvm.getelementptr %args[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>
          %rows = llvm.load %rows_ptr : !llvm.ptr -> i32
          %channels = llvm.load %channels_ptr : !llvm.ptr -> i32
          %hidden = llvm.load %hidden_ptr : !llvm.ptr -> i32
          %query_addr = llvm.load %query_addr_ptr : !llvm.ptr -> i64
          %weight_addr = llvm.load %weight_addr_ptr : !llvm.ptr -> i64
          %bias_addr = llvm.load %bias_addr_ptr : !llvm.ptr -> i64
          %output_addr = llvm.load %output_addr_ptr : !llvm.ptr -> i64

          %thread_csr = llvm.mlir.constant({CTA_THREAD_ID_X_CSR} : i32) : i32
          %block_csr = llvm.mlir.constant({CTA_BLOCK_ID_X_CSR} : i32) : i32
          %block_dim_csr = llvm.mlir.constant({CTA_BLOCK_DIM_X_CSR} : i32) : i32
          %thread_id = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %index = llvm.add %block_base, %thread_id : i32
          %count = llvm.mul %rows, %hidden : i32
          %in_bounds = llvm.icmp "ult" %index, %count : i32
          llvm.cond_br %in_bounds, ^setup, ^exit

        ^setup:
          %row = llvm.udiv %index, %hidden : i32
          %column = llvm.urem %index, %hidden : i32
          %query = llvm.inttoptr %query_addr : i64 to !llvm.ptr
          %weight = llvm.inttoptr %weight_addr : i64 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr : i64 to !llvm.ptr
          %output = llvm.inttoptr %output_addr : i64 to !llvm.ptr
          %bias_element = llvm.getelementptr %bias[%column] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %bias_value = llvm.load %bias_element : !llvm.ptr -> f32
          %zero_i = llvm.mlir.constant(0 : i32) : i32
          llvm.br ^dot_loop(%zero_i, %bias_value : i32, f32)

        ^dot_loop(%k: i32, %accumulator: f32):
          %continue = llvm.icmp "ult" %k, %channels : i32
          llvm.cond_br %continue, ^dot_body, ^dot_done

        ^dot_body:
          %query_row = llvm.mul %row, %channels : i32
          %query_index = llvm.add %query_row, %k : i32
          %weight_row = llvm.mul %k, %hidden : i32
          %weight_index = llvm.add %weight_row, %column : i32
          %query_element = llvm.getelementptr %query[%query_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %weight_element = llvm.getelementptr %weight[%weight_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %query_value = llvm.load %query_element : !llvm.ptr -> f32
          %weight_value = llvm.load %weight_element : !llvm.ptr -> f32
          %product = llvm.fmul %query_value, %weight_value : f32
          %next_accumulator = llvm.fadd %accumulator, %product : f32
          %one_i = llvm.mlir.constant(1 : i32) : i32
          %next_k = llvm.add %k, %one_i : i32
          llvm.br ^dot_loop(%next_k, %next_accumulator : i32, f32)

        ^dot_done:
          %zero_f = llvm.mlir.constant(0.000000e+00 : f32) : f32
          %positive = llvm.fcmp "ogt" %accumulator, %zero_f : f32
          %relu = llvm.select %positive, %accumulator, %zero_f : i1, f32
          %output_element = llvm.getelementptr %output[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          llvm.store %relu, %output_element : f32, !llvm.ptr
          llvm.br ^exit

        ^exit:
          llvm.return
        }}
        '''
    )


def _fc2_kernel() -> str:
    name = BEVFORMER_FFN_KERNELS[1]
    arg_type = "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>"
    return dedent(
        f'''\
        llvm.func @{name}(%arg_blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel"], section = ".vx_entry.{name}"}} {{
          %args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg_blob) : (!llvm.ptr) -> !llvm.ptr
          %rows_ptr = llvm.getelementptr %args[0, 0] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %channels_ptr = llvm.getelementptr %args[0, 1] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %hidden_ptr = llvm.getelementptr %args[0, 2] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %query_addr_ptr = llvm.getelementptr %args[0, 4] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %activation_addr_ptr = llvm.getelementptr %args[0, 5] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %weight_addr_ptr = llvm.getelementptr %args[0, 6] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %bias_addr_ptr = llvm.getelementptr %args[0, 7] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %output_addr_ptr = llvm.getelementptr %args[0, 8] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %rows = llvm.load %rows_ptr : !llvm.ptr -> i32
          %channels = llvm.load %channels_ptr : !llvm.ptr -> i32
          %hidden = llvm.load %hidden_ptr : !llvm.ptr -> i32
          %query_addr = llvm.load %query_addr_ptr : !llvm.ptr -> i64
          %activation_addr = llvm.load %activation_addr_ptr : !llvm.ptr -> i64
          %weight_addr = llvm.load %weight_addr_ptr : !llvm.ptr -> i64
          %bias_addr = llvm.load %bias_addr_ptr : !llvm.ptr -> i64
          %output_addr = llvm.load %output_addr_ptr : !llvm.ptr -> i64

          %thread_csr = llvm.mlir.constant({CTA_THREAD_ID_X_CSR} : i32) : i32
          %block_csr = llvm.mlir.constant({CTA_BLOCK_ID_X_CSR} : i32) : i32
          %block_dim_csr = llvm.mlir.constant({CTA_BLOCK_DIM_X_CSR} : i32) : i32
          %thread_id = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %index = llvm.add %block_base, %thread_id : i32
          %count = llvm.mul %rows, %channels : i32
          %in_bounds = llvm.icmp "ult" %index, %count : i32
          llvm.cond_br %in_bounds, ^setup, ^exit

        ^setup:
          %row = llvm.udiv %index, %channels : i32
          %column = llvm.urem %index, %channels : i32
          %query = llvm.inttoptr %query_addr : i64 to !llvm.ptr
          %activation = llvm.inttoptr %activation_addr : i64 to !llvm.ptr
          %weight = llvm.inttoptr %weight_addr : i64 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr : i64 to !llvm.ptr
          %output = llvm.inttoptr %output_addr : i64 to !llvm.ptr
          %bias_element = llvm.getelementptr %bias[%column] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %bias_value = llvm.load %bias_element : !llvm.ptr -> f32
          %zero_i = llvm.mlir.constant(0 : i32) : i32
          llvm.br ^dot_loop(%zero_i, %bias_value : i32, f32)

        ^dot_loop(%k: i32, %accumulator: f32):
          %continue = llvm.icmp "ult" %k, %hidden : i32
          llvm.cond_br %continue, ^dot_body, ^dot_done

        ^dot_body:
          %activation_row = llvm.mul %row, %hidden : i32
          %activation_index = llvm.add %activation_row, %k : i32
          %weight_row = llvm.mul %k, %channels : i32
          %weight_index = llvm.add %weight_row, %column : i32
          %activation_element = llvm.getelementptr %activation[%activation_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %weight_element = llvm.getelementptr %weight[%weight_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %activation_value = llvm.load %activation_element : !llvm.ptr -> f32
          %weight_value = llvm.load %weight_element : !llvm.ptr -> f32
          %product = llvm.fmul %activation_value, %weight_value : f32
          %next_accumulator = llvm.fadd %accumulator, %product : f32
          %one_i = llvm.mlir.constant(1 : i32) : i32
          %next_k = llvm.add %k, %one_i : i32
          llvm.br ^dot_loop(%next_k, %next_accumulator : i32, f32)

        ^dot_done:
          %query_element = llvm.getelementptr %query[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %query_value = llvm.load %query_element : !llvm.ptr -> f32
          %residual = llvm.fadd %query_value, %accumulator : f32
          %output_element = llvm.getelementptr %output[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          llvm.store %residual, %output_element : f32, !llvm.ptr
          llvm.br ^exit

        ^exit:
          llvm.return
        }}
        '''
    )


def _layer_norm_kernel() -> str:
    name = BEVFORMER_FFN_KERNELS[2]
    arg_type = "!llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>"
    return dedent(
        f'''\
        llvm.func @{name}(%arg_blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel"], section = ".vx_entry.{name}"}} {{
          %args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg_blob) : (!llvm.ptr) -> !llvm.ptr
          %rows_ptr = llvm.getelementptr %args[0, 0] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %channels_ptr = llvm.getelementptr %args[0, 1] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %epsilon_ptr = llvm.getelementptr %args[0, 2] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %input_addr_ptr = llvm.getelementptr %args[0, 4] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %scale_addr_ptr = llvm.getelementptr %args[0, 5] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %bias_addr_ptr = llvm.getelementptr %args[0, 6] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %output_addr_ptr = llvm.getelementptr %args[0, 7] : (!llvm.ptr) -> !llvm.ptr, {arg_type}
          %rows = llvm.load %rows_ptr : !llvm.ptr -> i32
          %channels = llvm.load %channels_ptr : !llvm.ptr -> i32
          %epsilon = llvm.load %epsilon_ptr : !llvm.ptr -> f32
          %input_addr = llvm.load %input_addr_ptr : !llvm.ptr -> i64
          %scale_addr = llvm.load %scale_addr_ptr : !llvm.ptr -> i64
          %bias_addr = llvm.load %bias_addr_ptr : !llvm.ptr -> i64
          %output_addr = llvm.load %output_addr_ptr : !llvm.ptr -> i64

          %thread_csr = llvm.mlir.constant({CTA_THREAD_ID_X_CSR} : i32) : i32
          %block_csr = llvm.mlir.constant({CTA_BLOCK_ID_X_CSR} : i32) : i32
          %block_dim_csr = llvm.mlir.constant({CTA_BLOCK_DIM_X_CSR} : i32) : i32
          %thread_id = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %row = llvm.add %block_base, %thread_id : i32
          %in_bounds = llvm.icmp "ult" %row, %rows : i32
          llvm.cond_br %in_bounds, ^setup, ^exit

        ^setup:
          %input = llvm.inttoptr %input_addr : i64 to !llvm.ptr
          %scale = llvm.inttoptr %scale_addr : i64 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr : i64 to !llvm.ptr
          %output = llvm.inttoptr %output_addr : i64 to !llvm.ptr
          %base = llvm.mul %row, %channels : i32
          %zero_i = llvm.mlir.constant(0 : i32) : i32
          %zero_f = llvm.mlir.constant(0.000000e+00 : f32) : f32
          llvm.br ^mean_loop(%zero_i, %zero_f : i32, f32)

        ^mean_loop(%mean_i: i32, %mean_sum: f32):
          %mean_continue = llvm.icmp "ult" %mean_i, %channels : i32
          llvm.cond_br %mean_continue, ^mean_body, ^mean_done

        ^mean_body:
          %mean_input_index = llvm.add %base, %mean_i : i32
          %mean_input_element = llvm.getelementptr %input[%mean_input_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %mean_input_value = llvm.load %mean_input_element : !llvm.ptr -> f32
          %mean_next_sum = llvm.fadd %mean_sum, %mean_input_value : f32
          %mean_one_i = llvm.mlir.constant(1 : i32) : i32
          %mean_next_i = llvm.add %mean_i, %mean_one_i : i32
          llvm.br ^mean_loop(%mean_next_i, %mean_next_sum : i32, f32)

        ^mean_done:
          %channels_f = llvm.uitofp %channels : i32 to f32
          %mean = llvm.fdiv %mean_sum, %channels_f : f32
          llvm.br ^variance_loop(%zero_i, %zero_f : i32, f32)

        ^variance_loop(%variance_i: i32, %variance_sum: f32):
          %variance_continue = llvm.icmp "ult" %variance_i, %channels : i32
          llvm.cond_br %variance_continue, ^variance_body, ^variance_done

        ^variance_body:
          %variance_input_index = llvm.add %base, %variance_i : i32
          %variance_input_element = llvm.getelementptr %input[%variance_input_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %variance_input_value = llvm.load %variance_input_element : !llvm.ptr -> f32
          %variance_centered = llvm.fsub %variance_input_value, %mean : f32
          %variance_square = llvm.fmul %variance_centered, %variance_centered : f32
          %variance_next_sum = llvm.fadd %variance_sum, %variance_square : f32
          %variance_one_i = llvm.mlir.constant(1 : i32) : i32
          %variance_next_i = llvm.add %variance_i, %variance_one_i : i32
          llvm.br ^variance_loop(%variance_next_i, %variance_next_sum : i32, f32)

        ^variance_done:
          %variance = llvm.fdiv %variance_sum, %channels_f : f32
          %variance_epsilon = llvm.fadd %variance, %epsilon : f32
          %stddev = llvm.call_intrinsic "llvm.sqrt"(%variance_epsilon) : (f32) -> f32
          %one_f = llvm.mlir.constant(1.000000e+00 : f32) : f32
          %inverse_stddev = llvm.fdiv %one_f, %stddev : f32
          llvm.br ^output_loop(%zero_i : i32)

        ^output_loop(%output_i: i32):
          %output_continue = llvm.icmp "ult" %output_i, %channels : i32
          llvm.cond_br %output_continue, ^output_body, ^exit

        ^output_body:
          %output_input_index = llvm.add %base, %output_i : i32
          %output_input_element = llvm.getelementptr %input[%output_input_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %output_scale_element = llvm.getelementptr %scale[%output_i] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %output_bias_element = llvm.getelementptr %bias[%output_i] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %output_element = llvm.getelementptr %output[%output_input_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %output_input_value = llvm.load %output_input_element : !llvm.ptr -> f32
          %output_scale_value = llvm.load %output_scale_element : !llvm.ptr -> f32
          %output_bias_value = llvm.load %output_bias_element : !llvm.ptr -> f32
          %output_centered = llvm.fsub %output_input_value, %mean : f32
          %normalized = llvm.fmul %output_centered, %inverse_stddev : f32
          %scaled = llvm.fmul %normalized, %output_scale_value : f32
          %affine = llvm.fadd %scaled, %output_bias_value : f32
          llvm.store %affine, %output_element : f32, !llvm.ptr
          %output_one_i = llvm.mlir.constant(1 : i32) : i32
          %output_next_i = llvm.add %output_i, %output_one_i : i32
          llvm.br ^output_loop(%output_next_i : i32)

        ^exit:
          llvm.return
        }}
        '''
    )


def emit_bevformer_ffn_device_mlir(
    *,
    rows: int,
    channels: int,
    hidden: int,
    epsilon: float,
    optimized_mlir_sha256: str,
    call_plan_mlir_sha256: str,
) -> str:
    """Emit the three scalar BEVFormer FFN kernels in LLVM dialect MLIR.

    Shapes and epsilon remain part of the launch blob so the device ABI stays
    compatible with the host runner.  The supplied compile-time values and
    input hashes are also recorded as immutable globals, binding the generated
    device module to the audited middle-end program that selected it.
    """

    rows = _require_positive_u32("rows", rows)
    channels = _require_positive_u32("channels", channels)
    hidden = _require_positive_u32("hidden", hidden)
    if rows * channels > _UINT32_MAX or rows * hidden > _UINT32_MAX:
        raise BackendError("vortex-v3: FFN launch element count exceeds RV32")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise BackendError("vortex-v3: epsilon must be a finite positive float")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise BackendError("vortex-v3: epsilon must be a finite positive float")
    optimized_hash = _require_sha256(
        "optimized_mlir_sha256", optimized_mlir_sha256
    )
    call_plan_hash = _require_sha256(
        "call_plan_mlir_sha256", call_plan_mlir_sha256
    )

    epsilon_literal = f"{epsilon_value:.9e}"
    metadata = dedent(
        f'''\
        llvm.mlir.global private constant @__gac_plan_rows({rows} : i32) {{section = "llvm.metadata"}} : i32
        llvm.mlir.global private constant @__gac_plan_channels({channels} : i32) {{section = "llvm.metadata"}} : i32
        llvm.mlir.global private constant @__gac_plan_hidden({hidden} : i32) {{section = "llvm.metadata"}} : i32
        llvm.mlir.global private constant @__gac_plan_epsilon({epsilon_literal} : f32) {{section = "llvm.metadata"}} : f32
        llvm.mlir.global private unnamed_addr constant @__gac_optimized_mlir_sha256("{optimized_hash}\\00") {{section = "llvm.metadata"}} : !llvm.array<65 x i8>
        llvm.mlir.global private unnamed_addr constant @__gac_call_plan_mlir_sha256("{call_plan_hash}\\00") {{section = "llvm.metadata"}} : !llvm.array<65 x i8>
        '''
    )
    body = "\n\n".join((_fc1_kernel(), _fc2_kernel(), _layer_norm_kernel()))
    return _module(BEVFORMER_FFN_KERNELS, body, metadata)


def _normalized(source: str) -> str:
    return re.sub(r"\s+", " ", source).strip()


def audit_bevformer_ffn_device_mlir(
    source: str,
    *,
    kernels: tuple[str, ...],
) -> None:
    """Reject a direct FFN module that no longer satisfies the Vortex ABI."""

    expected_kernels = _require_kernel_tuple(kernels)
    if expected_kernels != BEVFORMER_FFN_KERNELS:
        raise BackendError(
            "vortex-v3: direct device kernels do not match the BEVFormer FFN plan"
        )
    if not isinstance(source, str) or not source.strip():
        raise BackendError("vortex-v3: device MLIR must be a non-empty string")

    normalized = _normalized(source)
    failures: list[str] = []

    if f'llvm.target_triple = "{TARGET_TRIPLE}"' not in normalized:
        failures.append("RV32 target triple")
    if f'llvm.data_layout = "{DATA_LAYOUT}"' not in normalized:
        failures.append("RV32 data layout")
    if not re.search(r'@(?:"llvm\.global\.annotations"|llvm\.global\.annotations)\b', source):
        failures.append("llvm.global.annotations")
    if not re.search(r'@(?:"llvm\.used"|llvm\.used)\b', source):
        failures.append("llvm.used")
    if not re.search(r'@(?:"llvm\.compiler\.used"|llvm\.compiler\.used)\b', source):
        failures.append("llvm.compiler.used")
    if "vortex.kernel\\00" not in source:
        failures.append("vortex.kernel annotation text")
    if normalized.count('llvm.inline_asm has_side_effects "csrr $0, $1", "=r,i"') != 9:
        failures.append("three CTA CSR reads per kernel")
    for csr in (CTA_THREAD_ID_X_CSR, CTA_BLOCK_ID_X_CSR, CTA_BLOCK_DIM_X_CSR):
        if len(re.findall(rf"constant\({csr}\s*:\s*i32\)", source)) != 3:
            failures.append(f"CTA CSR 0x{csr:x}")

    function_names = tuple(
        match.group(1)
        for match in re.finditer(r"\bllvm\.func\s+@([A-Za-z_$][A-Za-z0-9_.$]*)\s*\(", source)
    )
    if function_names != expected_kernels:
        failures.append("exact ordered kernel entry set")
    for kernel in expected_kernels:
        if source.count(f"llvm.func @{kernel}") != 1:
            failures.append(f"single definition of {kernel}")
        if f'section = ".vx_entry.{kernel}"' not in source:
            failures.append(f"retained section for {kernel}")
        # One function pointer appears in llvm.global.annotations, llvm.used,
        # and llvm.compiler.used.  This remains stable after mlir-opt.
        if len(re.findall(rf"llvm\.mlir\.addressof\s+@{re.escape(kernel)}\b", source)) != 3:
            failures.append(f"annotation/retention references for {kernel}")

    required_layouts = (
        "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>",
        "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>",
        "!llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>",
    )
    for layout in required_layouts:
        if layout not in normalized:
            failures.append(f"RV32 argument layout {layout}")
    for provenance in (
        "@__gac_plan_rows",
        "@__gac_plan_channels",
        "@__gac_plan_hidden",
        "@__gac_plan_epsilon",
        "@__gac_optimized_mlir_sha256",
        "@__gac_call_plan_mlir_sha256",
    ):
        if provenance not in source:
            failures.append(f"provenance global {provenance}")

    if normalized.count('passthrough = ["vortex-kernel"]') != 3:
        failures.append("VOLT kernel calling-convention attributes")
    if normalized.count('llvm.call_intrinsic "llvm.riscv.vx.uniform"') < 9:
        failures.append("Vortex uniformity markers")
    if 'llvm.call_intrinsic "llvm.sqrt"' not in normalized:
        failures.append("LayerNorm square root")
    forbidden = ("func.func", "gpu.", "nvvm.", "#include", "kernel.cpp", "cuda", "ptx")
    for token in forbidden:
        if token.lower() in source.lower():
            failures.append(f"forbidden non-device-LLVM token {token!r}")

    if failures:
        unique_failures = ", ".join(dict.fromkeys(failures))
        raise BackendError(
            f"vortex-v3: invalid direct BEVFormer device MLIR: {unique_failures}"
        )


__all__ = [
    "BEVFORMER_FFN_KERNELS",
    "CTA_BLOCK_DIM_X_CSR",
    "CTA_BLOCK_ID_X_CSR",
    "CTA_THREAD_ID_X_CSR",
    "DATA_LAYOUT",
    "TARGET_TRIPLE",
    "VECTOR_ADD_KERNEL",
    "audit_bevformer_ffn_device_mlir",
    "emit_bevformer_ffn_device_mlir",
    "emit_vector_add_device_mlir",
]
