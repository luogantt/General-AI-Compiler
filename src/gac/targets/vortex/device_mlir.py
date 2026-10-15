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
import struct
from textwrap import dedent, indent

from gac.core.errors import BackendError


# Match the canonical triple produced by Clang's
# ``--target=riscv32-unknown-elf`` driver spelling.  Vortex builds use
# ``-Werror``; a shorter module triple would otherwise trigger
# ``-Woverride-module`` when Clang compiles the translated LLVM IR.
TARGET_TRIPLE = "riscv32-unknown-unknown-elf"
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


def validate_bevformer_ffn_rv32_extents(
    rows: int,
    channels: int,
    hidden: int,
) -> tuple[int, int, int]:
    """Validate every indexed element count and FP32 buffer against RV32."""

    rows = _require_positive_u32("rows", rows)
    channels = _require_positive_u32("channels", channels)
    hidden = _require_positive_u32("hidden", hidden)
    element_counts = {
        "rows*channels": rows * channels,
        "rows*hidden": rows * hidden,
        "channels*hidden": channels * hidden,
    }
    oversized_indices = [
        name for name, count in element_counts.items() if count > _UINT32_MAX
    ]
    if oversized_indices:
        raise BackendError(
            "vortex-v3: FFN element count exceeds RV32 for "
            + ", ".join(oversized_indices)
        )

    fp32_buffers = {
        "query/residual/output": element_counts["rows*channels"],
        "activation": element_counts["rows*hidden"],
        "fc1_weight/fc2_weight": element_counts["channels*hidden"],
        "fc1_bias": hidden,
        "fc2_bias/norm_weight/norm_bias": channels,
    }
    oversized_buffers = [
        name
        for name, count in fp32_buffers.items()
        if count > _UINT32_MAX // 4
    ]
    if oversized_buffers:
        raise BackendError(
            "vortex-v3: FP32 buffer byte size exceeds the RV32 address space for "
            + ", ".join(oversized_buffers)
        )
    return rows, channels, hidden


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
          %thread_id = llvm.inline_asm "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %index = llvm.add %block_base, %thread_id : i32
          %in_bounds = llvm.icmp "ult" %index, %n : i32
          llvm.cond_br %in_bounds, ^compute, ^exit

        ^compute:
          %lhs_addr_i32 = llvm.trunc %lhs_addr : i64 to i32
          %rhs_addr_i32 = llvm.trunc %rhs_addr : i64 to i32
          %out_addr_i32 = llvm.trunc %out_addr : i64 to i32
          %lhs = llvm.inttoptr %lhs_addr_i32 : i32 to !llvm.ptr
          %rhs = llvm.inttoptr %rhs_addr_i32 : i32 to !llvm.ptr
          %out = llvm.inttoptr %out_addr_i32 : i32 to !llvm.ptr
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
          %thread_id = llvm.inline_asm "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
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
          %query_addr_i32 = llvm.trunc %query_addr : i64 to i32
          %weight_addr_i32 = llvm.trunc %weight_addr : i64 to i32
          %bias_addr_i32 = llvm.trunc %bias_addr : i64 to i32
          %output_addr_i32 = llvm.trunc %output_addr : i64 to i32
          %query = llvm.inttoptr %query_addr_i32 : i32 to !llvm.ptr
          %weight = llvm.inttoptr %weight_addr_i32 : i32 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr_i32 : i32 to !llvm.ptr
          %output = llvm.inttoptr %output_addr_i32 : i32 to !llvm.ptr
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
          %thread_id = llvm.inline_asm "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
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
          %query_addr_i32 = llvm.trunc %query_addr : i64 to i32
          %activation_addr_i32 = llvm.trunc %activation_addr : i64 to i32
          %weight_addr_i32 = llvm.trunc %weight_addr : i64 to i32
          %bias_addr_i32 = llvm.trunc %bias_addr : i64 to i32
          %output_addr_i32 = llvm.trunc %output_addr : i64 to i32
          %query = llvm.inttoptr %query_addr_i32 : i32 to !llvm.ptr
          %activation = llvm.inttoptr %activation_addr_i32 : i32 to !llvm.ptr
          %weight = llvm.inttoptr %weight_addr_i32 : i32 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr_i32 : i32 to !llvm.ptr
          %output = llvm.inttoptr %output_addr_i32 : i32 to !llvm.ptr
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
          %thread_id = llvm.inline_asm "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          %block_id_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_csr : (i32) -> i32
          %block_dim_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %block_dim_csr : (i32) -> i32
          %block_id = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_id_raw) : (i32) -> i32
          %block_dim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%block_dim_raw) : (i32) -> i32
          %block_base = llvm.mul %block_id, %block_dim : i32
          %row = llvm.add %block_base, %thread_id : i32
          %in_bounds = llvm.icmp "ult" %row, %rows : i32
          llvm.cond_br %in_bounds, ^setup, ^exit

        ^setup:
          %input_addr_i32 = llvm.trunc %input_addr : i64 to i32
          %scale_addr_i32 = llvm.trunc %scale_addr : i64 to i32
          %bias_addr_i32 = llvm.trunc %bias_addr : i64 to i32
          %output_addr_i32 = llvm.trunc %output_addr : i64 to i32
          %input = llvm.inttoptr %input_addr_i32 : i32 to !llvm.ptr
          %scale = llvm.inttoptr %scale_addr_i32 : i32 to !llvm.ptr
          %bias = llvm.inttoptr %bias_addr_i32 : i32 to !llvm.ptr
          %output = llvm.inttoptr %output_addr_i32 : i32 to !llvm.ptr
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

    rows, channels, hidden = validate_bevformer_ffn_rv32_extents(
        rows, channels, hidden
    )
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


_SSA_VALUE = r"%[A-Za-z0-9_.$-]+"
_BLOCK_LABEL = r"\^[A-Za-z0-9_.$-]+"


def _function_body(source: str, name: str) -> str | None:
    """Return one emitted LLVM function body without depending on SSA names."""

    match = re.search(rf"\bllvm\.func\s+@{re.escape(name)}\s*\(", source)
    if match is None:
        return None
    line_end = source.find("\n", match.start())
    if line_end < 0:
        line_end = len(source)
    signature = source[match.start() : line_end]
    relative_open = signature.rfind("{")
    if relative_open < 0:
        return None
    open_brace = match.start() + relative_open
    depth = 0
    for offset in range(open_brace, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : offset]
    return None


def _global_initializer_body(source: str, name: str) -> str | None:
    rendered_name = rf'(?:"{re.escape(name)}"|{re.escape(name)})'
    match = re.search(
        rf"\bllvm\.mlir\.global\b[^\n]*@{rendered_name}\(\)", source
    )
    if match is None:
        return None
    line_end = source.find("\n", match.start())
    if line_end < 0:
        line_end = len(source)
    signature = source[match.start() : line_end]
    relative_open = signature.rfind("{")
    if relative_open < 0:
        return None
    open_brace = match.start() + relative_open
    depth = 0
    for offset in range(open_brace, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : offset]
    return None


def _ssa_definitions(body: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line in body.splitlines():
        match = re.match(rf"\s*({_SSA_VALUE})\s*=\s*(.*?)\s*$", line)
        if match is not None:
            definitions[match.group(1)] = _normalized(match.group(2))
    return definitions


def _unique_definition(
    definitions: dict[str, str],
    pattern: str,
    description: str,
    failures: list[str],
) -> str | None:
    matches = [
        result
        for result, expression in definitions.items()
        if re.fullmatch(pattern, expression) is not None
    ]
    if len(matches) != 1:
        failures.append(description)
        return None
    return matches[0]


def _insertvalue_definition(
    definitions: dict[str, str],
    value: str,
    container: str,
    index: int,
    value_type: str,
    description: str,
    failures: list[str],
) -> str | None:
    return _unique_definition(
        definitions,
        rf"llvm\.insertvalue\s+{re.escape(value)},\s*{re.escape(container)}"
        rf"\[{index}\]\s*:\s*{re.escape(value_type)}",
        description,
        failures,
    )


def _audit_retention_global(
    source: str,
    *,
    name: str,
    kernels: tuple[str, ...],
    failures: list[str],
) -> None:
    rendered_name = rf'(?:"{re.escape(name)}"|{re.escape(name)})'
    array_type = f"!llvm.array<{len(kernels)} x ptr>"
    header = re.search(
        rf"llvm\.mlir\.global\s+appending\s+@{rendered_name}\(\)"
        rf"[^\n]*section\s*=\s*\"llvm\.metadata\"[^\n]*:\s*"
        rf"{re.escape(array_type)}\s*\{{",
        source,
    )
    body = _global_initializer_body(source, name)
    if header is None or body is None:
        failures.append(f"canonical {name} initializer")
        return
    definitions = _ssa_definitions(body)
    current = _unique_definition(
        definitions,
        rf"llvm\.mlir\.undef\s*:\s*{re.escape(array_type)}",
        f"{name} array seed",
        failures,
    )
    if current is None:
        return
    for index, kernel in enumerate(kernels):
        pointer = _unique_definition(
            definitions,
            rf"llvm\.mlir\.addressof\s+@{re.escape(kernel)}\s*:\s*!llvm\.ptr",
            f"{name} member {kernel}",
            failures,
        )
        if pointer is None:
            return
        current = _insertvalue_definition(
            definitions,
            pointer,
            current,
            index,
            array_type,
            f"{name} ordered member {kernel}",
            failures,
        )
        if current is None:
            return
    if re.search(
        rf"\bllvm\.return\s+{re.escape(current)}\s*:\s*{re.escape(array_type)}",
        body,
    ) is None:
        failures.append(f"{name} returned retention array")


def _audit_annotation_global(
    source: str,
    *,
    kernels: tuple[str, ...],
    failures: list[str],
) -> None:
    name = "llvm.global.annotations"
    rendered_name = rf'(?:"{re.escape(name)}"|{re.escape(name)})'
    record_type = "!llvm.struct<(ptr, ptr, ptr, i32, ptr)>"
    array_type = (
        f"!llvm.array<{len(kernels)} x struct<(ptr, ptr, ptr, i32, ptr)>>"
    )
    header = re.search(
        rf"llvm\.mlir\.global\s+appending\s+@{rendered_name}\(\)"
        rf"[^\n]*section\s*=\s*\"llvm\.metadata\"[^\n]*:\s*"
        rf"{re.escape(array_type)}\s*\{{",
        source,
    )
    body = _global_initializer_body(source, name)
    annotation_string = re.search(
        r"llvm\.mlir\.global\s+private\s+unnamed_addr\s+constant\s+"
        r'@__gac_vortex_kernel_annotation\("vortex\.kernel\\00"\)'
        r'[^\n]*section\s*=\s*"llvm\.metadata"',
        source,
    )
    source_string = re.search(
        r"llvm\.mlir\.global\s+private\s+unnamed_addr\s+constant\s+"
        r'@__gac_vortex_source_file\("gac\.mlir\\00"\)'
        r'[^\n]*section\s*=\s*"llvm\.metadata"',
        source,
    )
    if (
        header is None
        or body is None
        or annotation_string is None
        or source_string is None
    ):
        failures.append("canonical llvm.global.annotations initializer")
        return
    definitions = _ssa_definitions(body)
    annotation_pointer = _unique_definition(
        definitions,
        r"llvm\.mlir\.addressof\s+@__gac_vortex_kernel_annotation\s*:\s*!llvm\.ptr",
        "annotation text pointer",
        failures,
    )
    source_pointer = _unique_definition(
        definitions,
        r"llvm\.mlir\.addressof\s+@__gac_vortex_source_file\s*:\s*!llvm\.ptr",
        "annotation source pointer",
        failures,
    )
    source_line = _unique_definition(
        definitions,
        r"llvm\.mlir\.constant\(1\s*:\s*i32\)\s*:\s*i32",
        "annotation source line",
        failures,
    )
    null_pointer = _unique_definition(
        definitions,
        r"llvm\.mlir\.zero\s*:\s*!llvm\.ptr",
        "annotation null argument",
        failures,
    )
    current_array = _unique_definition(
        definitions,
        rf"llvm\.mlir\.undef\s*:\s*{re.escape(array_type)}",
        "annotation array seed",
        failures,
    )
    if None in (
        annotation_pointer,
        source_pointer,
        source_line,
        null_pointer,
        current_array,
    ):
        return
    record_seeds = [
        result
        for result, expression in definitions.items()
        if re.fullmatch(
            rf"llvm\.mlir\.undef\s*:\s*{re.escape(record_type)}", expression
        )
    ]
    if not record_seeds:
        failures.append("annotation record seed")
        return

    for index, kernel in enumerate(kernels):
        kernel_pointer = _unique_definition(
            definitions,
            rf"llvm\.mlir\.addressof\s+@{re.escape(kernel)}\s*:\s*!llvm\.ptr",
            f"annotation member {kernel}",
            failures,
        )
        if kernel_pointer is None:
            return
        field0_candidates: list[str] = []
        for seed in record_seeds:
            result = _insertvalue_definition(
                definitions,
                kernel_pointer,
                seed,
                0,
                record_type,
                "",
                [],
            )
            if result is not None:
                field0_candidates.append(result)
        if len(field0_candidates) != 1:
            failures.append(f"annotation function field for {kernel}")
            return
        record = field0_candidates[0]
        for field_index, value, description in (
            (1, annotation_pointer, "annotation text"),
            (2, source_pointer, "annotation source"),
            (3, source_line, "annotation line"),
            (4, null_pointer, "annotation argument"),
        ):
            record = _insertvalue_definition(
                definitions,
                value,
                record,
                field_index,
                record_type,
                f"{description} field for {kernel}",
                failures,
            )
            if record is None:
                return
        current_array = _insertvalue_definition(
            definitions,
            record,
            current_array,
            index,
            array_type,
            f"ordered annotation record for {kernel}",
            failures,
        )
        if current_array is None:
            return
    if re.search(
        rf"\bllvm\.return\s+{re.escape(current_array)}\s*:\s*"
        rf"{re.escape(array_type)}",
        body,
    ) is None:
        failures.append("returned annotation array")


def _binary_definition(
    definitions: dict[str, str],
    operation: str,
    lhs: str,
    rhs: str,
    value_type: str,
    description: str,
    failures: list[str],
    *,
    commutative: bool = False,
) -> str | None:
    operands = [(lhs, rhs)]
    if commutative and lhs != rhs:
        operands.append((rhs, lhs))
    alternatives = "|".join(
        rf"{re.escape(left)},\s*{re.escape(right)}" for left, right in operands
    )
    return _unique_definition(
        definitions,
        rf"llvm\.{operation}\s+(?:{alternatives})\s*:\s*{re.escape(value_type)}",
        description,
        failures,
    )


def _gep_definition(
    definitions: dict[str, str],
    base: str,
    index: str,
    description: str,
    failures: list[str],
) -> str | None:
    return _unique_definition(
        definitions,
        rf"llvm\.getelementptr\s+{re.escape(base)}\[{re.escape(index)}\]\s*:\s*"
        rf"\(!llvm\.ptr,\s*i32\)\s*->\s*!llvm\.ptr,\s*f32",
        description,
        failures,
    )


def _load_definition(
    definitions: dict[str, str],
    pointer: str,
    value_type: str,
    description: str,
    failures: list[str],
) -> str | None:
    return _unique_definition(
        definitions,
        rf"llvm\.load\s+{re.escape(pointer)}(?:\s+\{{[^{{}}]*\}})?\s*:\s*"
        rf"!llvm\.ptr\s*->\s*{re.escape(value_type)}",
        description,
        failures,
    )


def _audit_argument_unpack(
    body: str,
    *,
    kernel: str,
    layout: str,
    fields: tuple[tuple[str, int, str], ...],
    failures: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Audit one fixed RV32 argument blob and return scalar/pointer SSA values."""

    definitions = _ssa_definitions(body)
    layout_pattern = re.escape(layout)
    geps: list[tuple[str, int, str]] = []
    for result, expression in definitions.items():
        match = re.fullmatch(
            rf"llvm\.getelementptr\s+({_SSA_VALUE})\[0,\s*(\d+)\]\s*:\s*"
            rf"\(!llvm\.ptr\)\s*->\s*!llvm\.ptr,\s*{layout_pattern}",
            expression,
        )
        if match is not None:
            geps.append((result, int(match.group(2)), match.group(1)))

    expected_indices = [field for _name, field, _value_type in fields]
    actual_indices = [field for _result, field, _base in geps]
    bases = {base for _result, _field, base in geps}
    if actual_indices != expected_indices or len(bases) != 1:
        failures.append(f"{kernel} exact argument field GEPs")
        return definitions, {}

    argument_base = next(iter(bases))
    uniform_expression = definitions.get(argument_base, "")
    if re.fullmatch(
        rf'llvm\.call_intrinsic\s+"llvm\.riscv\.vx\.uniform"\({_SSA_VALUE}\)\s*:\s*'
        rf"\(!llvm\.ptr\)\s*->\s*!llvm\.ptr",
        uniform_expression,
    ) is None:
        failures.append(f"{kernel} uniform argument buffer")

    values: dict[str, str] = {}
    for (result, actual_field, _base), (name, field, value_type) in zip(geps, fields):
        if actual_field != field:
            failures.append(f"{kernel} argument field {name}")
            continue
        loaded = _load_definition(
            definitions,
            result,
            value_type,
            f"{kernel} {name} field load",
            failures,
        )
        if loaded is None:
            continue
        if value_type != "i64":
            values[name] = loaded
            continue
        truncated = _unique_definition(
            definitions,
            rf"llvm\.trunc\s+{re.escape(loaded)}\s*:\s*i64\s+to\s+i32",
            f"{kernel} {name} RV32 address truncation",
            failures,
        )
        if truncated is None:
            continue
        pointer = _unique_definition(
            definitions,
            rf"llvm\.inttoptr\s+{re.escape(truncated)}\s*:\s*i32\s+to\s+!llvm\.ptr",
            f"{kernel} {name} RV32 pointer conversion",
            failures,
        )
        if pointer is not None:
            values[name] = pointer
    return definitions, values


def _icmp_lhs(
    definitions: dict[str, str],
    predicate: str,
    rhs: str,
    description: str,
    failures: list[str],
) -> str | None:
    matches: list[str] = []
    for expression in definitions.values():
        match = re.fullmatch(
            rf'llvm\.icmp\s+"{re.escape(predicate)}"\s+({_SSA_VALUE}),\s*'
            rf"{re.escape(rhs)}\s*:\s*i32",
            expression,
        )
        if match is not None:
            matches.append(match.group(1))
    if len(matches) != 1:
        failures.append(description)
        return None
    return matches[0]


def _audit_fc1_structure(body: str, failures: list[str]) -> None:
    kernel = BEVFORMER_FFN_KERNELS[0]
    layout = "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64)>"
    fields = (
        ("rows", 0, "i32"),
        ("channels", 1, "i32"),
        ("hidden", 2, "i32"),
        ("query", 4, "i64"),
        ("weight", 5, "i64"),
        ("bias", 6, "i64"),
        ("output", 7, "i64"),
    )
    definitions, values = _audit_argument_unpack(
        body, kernel=kernel, layout=layout, fields=fields, failures=failures
    )
    if set(values) != {name for name, _field, _type in fields}:
        return
    count = _binary_definition(
        definitions, "mul", values["rows"], values["hidden"], "i32",
        f"{kernel} rows*hidden launch count", failures, commutative=True,
    )
    if count is None:
        return
    index = _icmp_lhs(definitions, "ult", count, f"{kernel} launch bounds", failures)
    if index is None:
        return
    row = _binary_definition(
        definitions, "udiv", index, values["hidden"], "i32",
        f"{kernel} output row", failures,
    )
    column = _binary_definition(
        definitions, "urem", index, values["hidden"], "i32",
        f"{kernel} output column", failures,
    )
    k = _icmp_lhs(
        definitions, "ult", values["channels"], f"{kernel} dot-product bound", failures
    )
    if row is None or column is None or k is None:
        return
    query_row = _binary_definition(
        definitions, "mul", row, values["channels"], "i32",
        f"{kernel} query row stride", failures, commutative=True,
    )
    weight_row = _binary_definition(
        definitions, "mul", k, values["hidden"], "i32",
        f"{kernel} weight row stride", failures, commutative=True,
    )
    if query_row is None or weight_row is None:
        return
    query_index = _binary_definition(
        definitions, "add", query_row, k, "i32",
        f"{kernel} query index", failures, commutative=True,
    )
    weight_index = _binary_definition(
        definitions, "add", weight_row, column, "i32",
        f"{kernel} weight index", failures, commutative=True,
    )
    bias_element = _gep_definition(
        definitions, values["bias"], column, f"{kernel} bias indexing", failures
    )
    output_element = _gep_definition(
        definitions, values["output"], index, f"{kernel} output indexing", failures
    )
    if query_index is None or weight_index is None:
        return
    query_element = _gep_definition(
        definitions, values["query"], query_index, f"{kernel} query indexing", failures
    )
    weight_element = _gep_definition(
        definitions, values["weight"], weight_index, f"{kernel} weight indexing", failures
    )
    if None in (bias_element, output_element, query_element, weight_element):
        return
    bias_value = _load_definition(
        definitions, bias_element, "f32", f"{kernel} bias load", failures
    )
    query_value = _load_definition(
        definitions, query_element, "f32", f"{kernel} query load", failures
    )
    weight_value = _load_definition(
        definitions, weight_element, "f32", f"{kernel} weight load", failures
    )
    if None in (bias_value, query_value, weight_value):
        return
    product = _binary_definition(
        definitions, "fmul", query_value, weight_value, "f32",
        f"{kernel} multiply", failures, commutative=True,
    )
    if product is None:
        return
    if not any(
        re.fullmatch(
            rf"llvm\.fadd\s+(?:{_SSA_VALUE},\s*{re.escape(product)}|"
            rf"{re.escape(product)},\s*{_SSA_VALUE})\s*:\s*f32",
            expression,
        )
        for expression in definitions.values()
    ):
        failures.append(f"{kernel} dot-product accumulation")
    zero = _unique_definition(
        definitions,
        r"llvm\.mlir\.constant\(0(?:\.0+)?(?:[eE][+-]?0+)?\s*:\s*f32\)\s*:\s*f32",
        f"{kernel} ReLU zero",
        failures,
    )
    relu: str | None = None
    if zero is not None:
        comparisons: list[tuple[str, str]] = []
        for result, expression in definitions.items():
            match = re.fullmatch(
                rf'llvm\.fcmp\s+"ogt"\s+({_SSA_VALUE}),\s*{re.escape(zero)}\s*:\s*f32',
                expression,
            )
            if match is not None:
                comparisons.append((result, match.group(1)))
        if len(comparisons) != 1:
            failures.append(f"{kernel} ReLU comparison")
        else:
            comparison, accumulator = comparisons[0]
            relu = _unique_definition(
                definitions,
                rf"llvm\.select\s+{re.escape(comparison)},\s*{re.escape(accumulator)},\s*"
                rf"{re.escape(zero)}\s*:\s*i1,\s*f32",
                f"{kernel} ReLU select",
                failures,
            )
    normalized = _normalized(body)
    if relu is not None and re.search(
        rf"llvm\.store\s+{re.escape(relu)},\s*{re.escape(output_element)}\s*:\s*"
        rf"f32,\s*!llvm\.ptr",
        normalized,
    ) is None:
        failures.append(f"{kernel} ReLU output store")
    if re.search(
        rf"llvm\.br\s+{_BLOCK_LABEL}\("
        rf"{_SSA_VALUE},\s*{re.escape(bias_value)}\s*:\s*i32,\s*f32\)",
        normalized,
    ) is None:
        failures.append(f"{kernel} bias accumulator seed")


def _audit_fc2_structure(body: str, failures: list[str]) -> None:
    kernel = BEVFORMER_FFN_KERNELS[1]
    layout = "!llvm.struct<(i32, i32, i32, i32, i64, i64, i64, i64, i64)>"
    fields = (
        ("rows", 0, "i32"),
        ("channels", 1, "i32"),
        ("hidden", 2, "i32"),
        ("query", 4, "i64"),
        ("activation", 5, "i64"),
        ("weight", 6, "i64"),
        ("bias", 7, "i64"),
        ("output", 8, "i64"),
    )
    definitions, values = _audit_argument_unpack(
        body, kernel=kernel, layout=layout, fields=fields, failures=failures
    )
    if set(values) != {name for name, _field, _type in fields}:
        return
    count = _binary_definition(
        definitions, "mul", values["rows"], values["channels"], "i32",
        f"{kernel} rows*channels launch count", failures, commutative=True,
    )
    if count is None:
        return
    index = _icmp_lhs(definitions, "ult", count, f"{kernel} launch bounds", failures)
    if index is None:
        return
    row = _binary_definition(
        definitions, "udiv", index, values["channels"], "i32",
        f"{kernel} output row", failures,
    )
    column = _binary_definition(
        definitions, "urem", index, values["channels"], "i32",
        f"{kernel} output column", failures,
    )
    k = _icmp_lhs(
        definitions, "ult", values["hidden"], f"{kernel} dot-product bound", failures
    )
    if row is None or column is None or k is None:
        return
    activation_row = _binary_definition(
        definitions, "mul", row, values["hidden"], "i32",
        f"{kernel} activation row stride", failures, commutative=True,
    )
    weight_row = _binary_definition(
        definitions, "mul", k, values["channels"], "i32",
        f"{kernel} weight row stride", failures, commutative=True,
    )
    if activation_row is None or weight_row is None:
        return
    activation_index = _binary_definition(
        definitions, "add", activation_row, k, "i32",
        f"{kernel} activation index", failures, commutative=True,
    )
    weight_index = _binary_definition(
        definitions, "add", weight_row, column, "i32",
        f"{kernel} weight index", failures, commutative=True,
    )
    bias_element = _gep_definition(
        definitions, values["bias"], column, f"{kernel} bias indexing", failures
    )
    query_element = _gep_definition(
        definitions, values["query"], index, f"{kernel} residual indexing", failures
    )
    output_element = _gep_definition(
        definitions, values["output"], index, f"{kernel} output indexing", failures
    )
    if activation_index is None or weight_index is None:
        return
    activation_element = _gep_definition(
        definitions, values["activation"], activation_index,
        f"{kernel} activation indexing", failures,
    )
    weight_element = _gep_definition(
        definitions, values["weight"], weight_index, f"{kernel} weight indexing", failures
    )
    if None in (
        bias_element, query_element, output_element, activation_element, weight_element
    ):
        return
    bias_value = _load_definition(
        definitions, bias_element, "f32", f"{kernel} bias load", failures
    )
    query_value = _load_definition(
        definitions, query_element, "f32", f"{kernel} residual load", failures
    )
    activation_value = _load_definition(
        definitions, activation_element, "f32", f"{kernel} activation load", failures
    )
    weight_value = _load_definition(
        definitions, weight_element, "f32", f"{kernel} weight load", failures
    )
    if None in (bias_value, query_value, activation_value, weight_value):
        return
    product = _binary_definition(
        definitions, "fmul", activation_value, weight_value, "f32",
        f"{kernel} multiply", failures, commutative=True,
    )
    if product is None:
        return
    dot_adds = [
        result
        for result, expression in definitions.items()
        if re.fullmatch(
            rf"llvm\.fadd\s+(?:{_SSA_VALUE},\s*{re.escape(product)}|"
            rf"{re.escape(product)},\s*{_SSA_VALUE})\s*:\s*f32",
            expression,
        )
    ]
    if len(dot_adds) != 1:
        failures.append(f"{kernel} dot-product accumulation")
    residuals = [
        result
        for result, expression in definitions.items()
        if re.fullmatch(
            rf"llvm\.fadd\s+(?:{re.escape(query_value)},\s*{_SSA_VALUE}|"
            rf"{_SSA_VALUE},\s*{re.escape(query_value)})\s*:\s*f32",
            expression,
        )
    ]
    if len(residuals) != 1:
        failures.append(f"{kernel} residual add")
    normalized = _normalized(body)
    if len(residuals) == 1 and re.search(
        rf"llvm\.store\s+{re.escape(residuals[0])},\s*{re.escape(output_element)}\s*:\s*"
        rf"f32,\s*!llvm\.ptr",
        normalized,
    ) is None:
        failures.append(f"{kernel} residual output store")
    if re.search(
        rf"llvm\.br\s+{_BLOCK_LABEL}\("
        rf"{_SSA_VALUE},\s*{re.escape(bias_value)}\s*:\s*i32,\s*f32\)",
        normalized,
    ) is None:
        failures.append(f"{kernel} bias accumulator seed")


def _audit_layer_norm_structure(body: str, failures: list[str]) -> None:
    kernel = BEVFORMER_FFN_KERNELS[2]
    layout = "!llvm.struct<(i32, i32, f32, i32, i64, i64, i64, i64)>"
    fields = (
        ("rows", 0, "i32"),
        ("channels", 1, "i32"),
        ("epsilon", 2, "f32"),
        ("input", 4, "i64"),
        ("scale", 5, "i64"),
        ("bias", 6, "i64"),
        ("output", 7, "i64"),
    )
    definitions, values = _audit_argument_unpack(
        body, kernel=kernel, layout=layout, fields=fields, failures=failures
    )
    if set(values) != {name for name, _field, _type in fields}:
        return
    row = _icmp_lhs(
        definitions, "ult", values["rows"], f"{kernel} row launch bounds", failures
    )
    if row is None:
        return
    base = _binary_definition(
        definitions, "mul", row, values["channels"], "i32",
        f"{kernel} row stride", failures, commutative=True,
    )
    channels_float = _unique_definition(
        definitions,
        rf"llvm\.uitofp\s+{re.escape(values['channels'])}\s*:\s*i32\s+to\s+f32",
        f"{kernel} channel-count conversion",
        failures,
    )
    if base is None or channels_float is None:
        return

    loop_indices: list[str] = []
    for expression in definitions.values():
        match = re.fullmatch(
            rf'llvm\.icmp\s+"ult"\s+({_SSA_VALUE}),\s*'
            rf"{re.escape(values['channels'])}\s*:\s*i32",
            expression,
        )
        if match is not None:
            loop_indices.append(match.group(1))
    if len(loop_indices) != 3 or len(set(loop_indices)) != 3:
        failures.append(f"{kernel} mean/variance/output loop bounds")
        return

    indexed_inputs: dict[str, tuple[str, str]] = {}
    for loop_index in loop_indices:
        element_index = _binary_definition(
            definitions, "add", base, loop_index, "i32",
            f"{kernel} row element index", failures, commutative=True,
        )
        if element_index is None:
            continue
        element = _gep_definition(
            definitions, values["input"], element_index,
            f"{kernel} input row indexing", failures,
        )
        if element is None:
            continue
        loaded = _load_definition(
            definitions, element, "f32", f"{kernel} input row load", failures
        )
        if loaded is not None:
            indexed_inputs[loop_index] = (element_index, loaded)
    if len(indexed_inputs) != 3:
        return

    fdivs = [
        (result, expression)
        for result, expression in definitions.items()
        if expression.startswith("llvm.fdiv ") and expression.endswith(": f32")
    ]
    means = [
        result
        for result, expression in fdivs
        if re.fullmatch(
            rf"llvm\.fdiv\s+{_SSA_VALUE},\s*{re.escape(channels_float)}\s*:\s*f32",
            expression,
        )
    ]
    if len(means) != 2:
        failures.append(f"{kernel} mean and variance division")
        return

    centered: str | None = None
    square: str | None = None
    mean_value: str | None = None
    for possible_mean in means:
        for _loop_index, (_element_index, input_value) in indexed_inputs.items():
            candidate = _binary_definition(
                definitions, "fsub", input_value, possible_mean, "f32",
                f"{kernel} centered input", [],
            )
            if candidate is None:
                continue
            possible_square = _binary_definition(
                definitions, "fmul", candidate, candidate, "f32",
                f"{kernel} variance square", [],
            )
            if possible_square is not None:
                centered = candidate
                square = possible_square
                mean_value = possible_mean
                break
        if square is not None:
            break
    if centered is None or square is None or mean_value is None:
        failures.append(f"{kernel} centered variance accumulation")
        return
    if not any(
        re.fullmatch(
            rf"llvm\.fadd\s+(?:{_SSA_VALUE},\s*{re.escape(square)}|"
            rf"{re.escape(square)},\s*{_SSA_VALUE})\s*:\s*f32",
            expression,
        )
        for expression in definitions.values()
    ):
        failures.append(f"{kernel} variance accumulation")
        return

    variance_candidates = [value for value in means if value != mean_value]
    if len(variance_candidates) != 1:
        failures.append(f"{kernel} variance value")
        return
    variance = variance_candidates[0]
    variance_epsilon = _binary_definition(
        definitions, "fadd", variance, values["epsilon"], "f32",
        f"{kernel} epsilon addition", failures, commutative=True,
    )
    if variance_epsilon is None:
        return
    stddev = _unique_definition(
        definitions,
        rf'llvm\.call_intrinsic\s+"llvm\.sqrt"\({re.escape(variance_epsilon)}\)\s*'
        rf":\s*\(f32\)\s*->\s*f32",
        f"{kernel} square root",
        failures,
    )
    if stddev is None:
        return
    one = _unique_definition(
        definitions,
        r"llvm\.mlir\.constant\(1(?:\.0+)?(?:[eE][+-]?0+)?\s*:\s*f32\)\s*:\s*f32",
        f"{kernel} reciprocal numerator",
        failures,
    )
    if one is None:
        return
    inverse_stddev = _binary_definition(
        definitions,
        "fdiv",
        one,
        stddev,
        "f32",
        f"{kernel} reciprocal standard deviation",
        failures,
    )
    if inverse_stddev is None:
        return

    output_matches = 0
    for output_index, (element_index, input_value) in indexed_inputs.items():
        scale_element = _gep_definition(
            definitions, values["scale"], output_index, f"{kernel} scale indexing", []
        )
        bias_element = _gep_definition(
            definitions, values["bias"], output_index, f"{kernel} bias indexing", []
        )
        output_element = _gep_definition(
            definitions, values["output"], element_index, f"{kernel} output indexing", []
        )
        if None in (scale_element, bias_element, output_element):
            continue
        scale_value = _load_definition(definitions, scale_element, "f32", "", [])
        bias_value = _load_definition(definitions, bias_element, "f32", "", [])
        if scale_value is None or bias_value is None:
            continue
        output_centered = _binary_definition(
            definitions, "fsub", input_value, mean_value, "f32", "", []
        )
        if output_centered is None:
            continue
        normalized_value = _binary_definition(
            definitions,
            "fmul",
            output_centered,
            inverse_stddev,
            "f32",
            "",
            [],
            commutative=True,
        )
        if normalized_value is None:
            continue
        scaled = _binary_definition(
            definitions, "fmul", normalized_value, scale_value, "f32", "", [],
            commutative=True,
        )
        if scaled is None:
            continue
        affine = _binary_definition(
            definitions, "fadd", scaled, bias_value, "f32", "", [],
            commutative=True,
        )
        if affine is None:
            continue
        if re.search(
            rf"llvm\.store\s+{re.escape(affine)},\s*{re.escape(output_element)}\s*:\s*"
            rf"f32,\s*!llvm\.ptr",
            _normalized(body),
        ) is not None:
            output_matches += 1
    if output_matches != 1:
        failures.append(f"{kernel} affine output structure")


def audit_bevformer_ffn_device_mlir(
    source: str,
    *,
    kernels: tuple[str, ...],
    rows: int,
    channels: int,
    hidden: int,
    epsilon: float,
    optimized_mlir_sha256: str,
    call_plan_mlir_sha256: str,
) -> None:
    """Reject a direct FFN module that no longer satisfies the Vortex ABI."""

    expected_kernels = _require_kernel_tuple(kernels)
    expected_rows = _require_positive_u32("rows", rows)
    expected_channels = _require_positive_u32("channels", channels)
    expected_hidden = _require_positive_u32("hidden", hidden)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise BackendError("vortex-v3: epsilon must be a finite positive float")
    expected_epsilon = float(epsilon)
    if not math.isfinite(expected_epsilon) or expected_epsilon <= 0.0:
        raise BackendError("vortex-v3: epsilon must be a finite positive float")
    expected_optimized_hash = _require_sha256(
        "optimized_mlir_sha256", optimized_mlir_sha256
    )
    expected_call_plan_hash = _require_sha256(
        "call_plan_mlir_sha256", call_plan_mlir_sha256
    )
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
    _audit_annotation_global(
        source,
        kernels=expected_kernels,
        failures=failures,
    )
    _audit_retention_global(
        source,
        name="llvm.used",
        kernels=expected_kernels,
        failures=failures,
    )
    _audit_retention_global(
        source,
        name="llvm.compiler.used",
        kernels=expected_kernels,
        failures=failures,
    )
    if normalized.count('llvm.inline_asm "csrr $0, $1", "=r,i"') != 9:
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
    integer_globals = {
        "rows": expected_rows,
        "channels": expected_channels,
        "hidden": expected_hidden,
    }
    for name, expected_value in integer_globals.items():
        matches = re.findall(
            rf"@__gac_plan_{name}\(\s*(\d+)\s*:\s*i32\s*\)", source
        )
        if matches != [str(expected_value)]:
            failures.append(f"exact {name} provenance")
    epsilon_matches = re.findall(
        r"@__gac_plan_epsilon\(\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*f32\s*\)",
        source,
    )
    try:
        epsilon_matches_expected = (
            len(epsilon_matches) == 1
            and struct.pack("<f", float(epsilon_matches[0]))
            == struct.pack("<f", expected_epsilon)
        )
    except (OverflowError, ValueError):
        epsilon_matches_expected = False
    if not epsilon_matches_expected:
        failures.append("exact epsilon provenance")
    hash_globals = {
        "optimized_mlir": expected_optimized_hash,
        "call_plan_mlir": expected_call_plan_hash,
    }
    for name, expected_hash in hash_globals.items():
        matches = re.findall(
            rf'@__gac_{name}_sha256\("([0-9a-fA-F]{{64}})\\00"\)', source
        )
        if [value.lower() for value in matches] != [expected_hash]:
            failures.append(f"exact {name} SHA-256 provenance")

    if normalized.count('passthrough = ["vortex-kernel"]') != 3:
        failures.append("VOLT kernel calling-convention attributes")
    if normalized.count('llvm.call_intrinsic "llvm.riscv.vx.uniform"') < 9:
        failures.append("Vortex uniformity markers")
    if normalized.count("llvm.trunc") != 13 or normalized.count("llvm.inttoptr") != 13:
        failures.append("explicit RV32 device-address conversion")
    if 'llvm.call_intrinsic "llvm.sqrt"' not in normalized:
        failures.append("LayerNorm square root")
    forbidden = ("func.func", "gpu.", "nvvm.", "#include", "kernel.cpp", "cuda", "ptx")
    for token in forbidden:
        if token.lower() in source.lower():
            failures.append(f"forbidden non-device-LLVM token {token!r}")

    bodies = {kernel: _function_body(source, kernel) for kernel in expected_kernels}
    for kernel, body in bodies.items():
        if body is None:
            failures.append(f"parseable body for {kernel}")
    if all(body is not None for body in bodies.values()):
        _audit_fc1_structure(bodies[BEVFORMER_FFN_KERNELS[0]] or "", failures)
        _audit_fc2_structure(bodies[BEVFORMER_FFN_KERNELS[1]] or "", failures)
        _audit_layer_norm_structure(bodies[BEVFORMER_FFN_KERNELS[2]] or "", failures)

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
    "validate_bevformer_ffn_rv32_extents",
]
