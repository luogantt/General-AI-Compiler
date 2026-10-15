"""FP32 FFN lowering to Vortex DXA and two LMEM tile stages.

Each CTA computes one M/N tile. DXA copies the next K tile while the CTA
multiplies the current tile; every thread reaches the completion and reuse
barriers, including threads outside a ragged output boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from textwrap import dedent

from gac.core.errors import BackendError
import gac.targets.vortex.device_mlir as scalar


@dataclass(frozen=True, slots=True)
class VortexBufferPlan:
    tile_m: int = 2
    tile_n: int = 2
    tile_k: int = 16

    def __post_init__(self) -> None:
        for name in ("tile_m", "tile_n", "tile_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise BackendError(f"vortex-v3: {name} must be an integer in 1..65535")
        if self.threads < 4:
            raise BackendError("vortex-v3: DXA WGATHER requires at least four CTA threads")
        if self.lmem_bytes > (1 << 32) - 1:
            raise BackendError("vortex-v3: LMEM tile storage exceeds RV32")

    @property
    def threads(self) -> int:
        return self.tile_m * self.tile_n

    @property
    def stage_elements(self) -> int:
        return self.tile_k * (self.tile_m + self.tile_n)

    @property
    def lmem_bytes(self) -> int:
        return 2 * self.stage_elements * 4

    def grid_size(self, rows: int, columns: int) -> int:
        return ((rows + self.tile_m - 1) // self.tile_m) * ((columns + self.tile_n - 1) // self.tile_n)

    def to_dict(self) -> dict:
        return {
            "implementation": "dxa-double-buffer",
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "stages": 2,
            "threads_per_cta": self.threads,
            "lmem_bytes_per_cta": self.lmem_bytes,
            "stage_stride_bytes": self.stage_elements * 4,
            "a_buffer_bytes": self.tile_m * self.tile_k * 4,
            "b_buffer_bytes": self.tile_k * self.tile_n * 4,
            "thread_mapping": "local_row = thread_x / tile_n; local_col = thread_x % tile_n",
            "descriptor_slots": {"fc1": [0, 1], "fc2": [2, 3]},
            "boundary": "DXA zero-fill; guarded output; valid K accumulation",
            "accumulator": "one FP32 SSA value per output thread, retained across K tiles",
        }


def _csr(name: str, number: int) -> str:
    return f'''%{name}_csr = llvm.mlir.constant({number} : i32) : i32
          %{name}_raw = llvm.inline_asm "csrr $0, $1", "=r,i" %{name}_csr : (i32) -> i32
          %{name} = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%{name}_raw) : (i32) -> i32'''


def _barrier(bar: str) -> str:
    return f'''llvm.call @__gac_cta_barrier(%{bar}, %warps) {{convergent}} : (i32, i32) -> ()'''


def _collective_helpers() -> str:
    # LLVM 20 InlineAsmOp has no convergent call attribute. Small optnone
    # wrappers retain that semantic under optimization; WGATHER and DXA must
    # stay together so no active-lane-only copy of x31 can lose packed lanes.
    return dedent('''\
        llvm.func internal @__gac_dxa_expect_two(%bar: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
          %opcode = llvm.mlir.constant(11 : i32) : i32
          %count = llvm.mlir.constant(-2147483646 : i32) : i32
          %discard = llvm.inline_asm has_side_effects ".insn r $1, 6, 0, $0, $2, $3", "=r,i,r,r,~{memory}" %opcode, %bar, %count : (i32, i32, i32) -> i32
          llvm.return
        }
        llvm.func internal @__gac_dxa_issue_2d(%dst: i32, %bar: i32, %slot: i32, %coord0: i32, %coord1: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
          %custom0 = llvm.mlir.constant(11 : i32) : i32
          %custom1 = llvm.mlir.constant(43 : i32) : i32
          %four = llvm.mlir.constant(4 : i32) : i32
          %shift = llvm.shl %bar, %four : i32
          %meta = llvm.or %shift, %slot : i32
          %packed = llvm.inline_asm has_side_effects ".insn r4 $1, 0, 0, $0, $3, $4, $5\\0A\\09.insn r $6, 0, 3, x0, $0, x0", "={x31},i,0,r,r,r,i,~{memory}" %custom1, %dst, %meta, %coord0, %coord1, %custom0 : (i32, i32, i32, i32, i32, i32) -> i32
          llvm.return
        }
        llvm.func internal @__gac_cta_barrier(%bar: i32, %warps: i32) attributes {convergent, passthrough = ["noinline", "optnone"]} {
          %opcode = llvm.mlir.constant(11 : i32) : i32
          llvm.inline_asm has_side_effects ".insn r $0, 4, 0, x0, $1, $2", "i,r,r,~{memory}" %opcode, %bar, %warps : (i32, i32, i32) -> ()
          llvm.return
        }
        ''')


def _copy(prefix: str, bar: str, stage: str, k: str, slot: int) -> str:
    """Keep the WGATHER result in x31 until DXA consumes all four lanes."""
    return f'''llvm.call @__gac_dxa_expect_two(%{bar}) {{convergent}} : (i32) -> ()
          %{prefix}_stage_offset = llvm.mul %{stage}, %stage_elems : i32
          %{prefix}_a = llvm.getelementptr %lmem[%{prefix}_stage_offset] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %{prefix}_b_offset = llvm.add %{prefix}_stage_offset, %a_elems : i32
          %{prefix}_b = llvm.getelementptr %lmem[%{prefix}_b_offset] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %{prefix}_a_addr = llvm.ptrtoint %{prefix}_a : !llvm.ptr to i32
          %{prefix}_b_addr = llvm.ptrtoint %{prefix}_b : !llvm.ptr to i32
          %{prefix}_slot_a = llvm.mlir.constant({slot} : i32) : i32
          %{prefix}_slot_b = llvm.mlir.constant({slot + 1} : i32) : i32
          llvm.call @__gac_dxa_issue_2d(%{prefix}_a_addr, %{bar}, %{prefix}_slot_a, %{k}, %row_base) {{convergent}} : (i32, i32, i32, i32, i32) -> ()
          llvm.call @__gac_dxa_issue_2d(%{prefix}_b_addr, %{bar}, %{prefix}_slot_b, %col_base, %{k}) {{convergent}} : (i32, i32, i32, i32, i32) -> ()'''


def _matmul_kernel(plan: VortexBufferPlan, second: bool) -> str:
    name = scalar.BEVFORMER_FFN_KERNELS[int(second)]
    pointer_fields = ("query", "activation", "weight", "bias", "output") if second else ("query", "weight", "bias", "output")
    used_pointers = ("query", "bias", "output") if second else ("bias", "output")
    layout = "!llvm.struct<(i32, i32, i32, i32, " + ", ".join("i64" for _ in pointer_fields) + ")>"
    unpack = []
    for field, index, typ in [("rows", 0, "i32"), ("channels", 1, "i32"), ("hidden", 2, "i32")] + [(field + "_addr", index + 4, "i64") for index, field in enumerate(pointer_fields) if field in used_pointers]:
        unpack.extend((f"%{field}_ptr = llvm.getelementptr %args[0, {index}] : (!llvm.ptr) -> !llvm.ptr, {layout}", f"%{field} = llvm.load %{field}_ptr : !llvm.ptr -> {typ}"))
    for field in used_pointers:
        unpack.extend((f"%{field}_i32 = llvm.trunc %{field}_addr : i64 to i32", f"%{field} = llvm.inttoptr %{field}_i32 : i32 to !llvm.ptr"))
    unpack_text = "\n          ".join(unpack)
    n, k = ("channels", "hidden") if second else ("hidden", "channels")
    finish = '''%residual_element = llvm.getelementptr %query[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %residual = llvm.load %residual_element : !llvm.ptr -> f32
          %result = llvm.fadd %final_acc, %residual : f32''' if second else '''%positive = llvm.fcmp "ogt" %final_acc, %zero_f : f32
          %result = llvm.select %positive, %final_acc, %zero_f : i1, f32'''
    return dedent(f'''\
        llvm.func @{name}(%arg_blob: !llvm.ptr) attributes {{passthrough = ["vortex-kernel", "convergent"], section = ".vx_entry.{name}"}} {{
          %args = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%arg_blob) : (!llvm.ptr) -> !llvm.ptr
          {unpack_text}
          %thread_csr = llvm.mlir.constant(3283 : i32) : i32
          %thread = llvm.inline_asm "csrr $0, $1", "=r,i" %thread_csr : (i32) -> i32
          {_csr("block", 0xCD6)}
          {_csr("lmem_addr", 0xCDF)}
          {_csr("cta", 0xCD0)}
          {_csr("warps", 0xCD2)}
          {_csr("rank", 0xCD1)}
          %lmem = llvm.inttoptr %lmem_addr : i32 to !llvm.ptr
          %zero = llvm.mlir.constant(0 : i32) : i32
          %one = llvm.mlir.constant(1 : i32) : i32
          %eight = llvm.mlir.constant(8 : i32) : i32
          %zero_f = llvm.mlir.constant(0.000000e+00 : f32) : f32
          %bm = llvm.mlir.constant({plan.tile_m} : i32) : i32
          %bn = llvm.mlir.constant({plan.tile_n} : i32) : i32
          %bk = llvm.mlir.constant({plan.tile_k} : i32) : i32
          %a_elems = llvm.mlir.constant({plan.tile_m * plan.tile_k} : i32) : i32
          %stage_elems = llvm.mlir.constant({plan.stage_elements} : i32) : i32
          %n_minus_one = llvm.sub %{n}, %one : i32
          %n_tile_floor = llvm.udiv %n_minus_one, %bn : i32
          %n_tiles = llvm.add %n_tile_floor, %one : i32
          %tile_row = llvm.udiv %block, %n_tiles : i32
          %tile_col = llvm.urem %block, %n_tiles : i32
          %row_base = llvm.mul %tile_row, %bm : i32
          %col_base = llvm.mul %tile_col, %bn : i32
          %local_row = llvm.udiv %thread, %bn : i32
          %local_col = llvm.urem %thread, %bn : i32
          %row = llvm.add %row_base, %local_row : i32
          %column = llvm.add %col_base, %local_col : i32
          %row_valid = llvm.icmp "ult" %row, %rows : i32
          %col_valid = llvm.icmp "ult" %column, %{n} : i32
          %valid = llvm.and %row_valid, %col_valid : i1
          %loader = llvm.icmp "eq" %rank, %zero : i32
          llvm.cond_br %valid, ^bias_load, ^bias_ready(%zero_f : f32)
        ^bias_load:
          %bias_element = llvm.getelementptr %bias[%column] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %bias_value = llvm.load %bias_element : !llvm.ptr -> f32
          llvm.br ^bias_ready(%bias_value : f32)
        ^bias_ready(%initial_acc: f32):
          llvm.cond_br %loader, ^prime, ^tile_loop(%zero, %zero, %initial_acc : i32, i32, f32)
        ^prime:
          {_copy("prime", "cta", "zero", "zero", 2 if second else 0)}
          llvm.br ^tile_loop(%zero, %zero, %initial_acc : i32, i32, f32)
        ^tile_loop(%k_base: i32, %cur: i32, %acc: f32):
          %next_k = llvm.add %k_base, %bk : i32
          %has_next = llvm.icmp "ult" %next_k, %{k} : i32
          %nxt = llvm.xor %cur, %one : i32
          %cur_bar_shift = llvm.shl %cur, %eight : i32
          %cur_bar = llvm.or %cur_bar_shift, %cta : i32
          %next_bar_shift = llvm.shl %nxt, %eight : i32
          %next_bar = llvm.or %next_bar_shift, %cta : i32
          %issue_next = llvm.and %has_next, %loader : i1
          llvm.cond_br %issue_next, ^prefetch, ^ready
        ^prefetch:
          {_copy("next", "next_bar", "nxt", "next_k", 2 if second else 0)}
          llvm.br ^ready
        ^ready:
          {_barrier("cur_bar")}
          %stage_offset = llvm.mul %cur, %stage_elems : i32
          %a = llvm.getelementptr %lmem[%stage_offset] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %b_offset = llvm.add %stage_offset, %a_elems : i32
          %b = llvm.getelementptr %lmem[%b_offset] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %remaining = llvm.sub %{k}, %k_base : i32
          %tail = llvm.icmp "ult" %remaining, %bk : i32
          %valid_k = llvm.select %tail, %remaining, %bk : i1, i32
          llvm.br ^dot_loop(%zero, %acc : i32, f32)
        ^dot_loop(%j: i32, %partial: f32):
          %more = llvm.icmp "ult" %j, %valid_k : i32
          llvm.cond_br %more, ^dot_body, ^tile_done(%partial : f32)
        ^dot_body:
          %a_row = llvm.mul %local_row, %bk : i32
          %a_index = llvm.add %a_row, %j : i32
          %b_row = llvm.mul %j, %bn : i32
          %b_index = llvm.add %b_row, %local_col : i32
          %a_element = llvm.getelementptr %a[%a_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %b_element = llvm.getelementptr %b[%b_index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          %a_value = llvm.load %a_element : !llvm.ptr -> f32
          %b_value = llvm.load %b_element : !llvm.ptr -> f32
          %product = llvm.fmul %a_value, %b_value : f32
          %next_partial = llvm.fadd %partial, %product : f32
          %next_j = llvm.add %j, %one : i32
          llvm.br ^dot_loop(%next_j, %next_partial : i32, f32)
        ^tile_done(%tile_acc: f32):
          {_barrier("cur_bar")}
          llvm.cond_br %has_next, ^tile_loop(%next_k, %nxt, %tile_acc : i32, i32, f32), ^finished(%tile_acc : f32)
        ^finished(%final_acc: f32):
          llvm.cond_br %valid, ^store, ^exit
        ^store:
          %row_index = llvm.mul %row, %{n} : i32
          %index = llvm.add %row_index, %column : i32
          {finish}
          %out_element = llvm.getelementptr %output[%index] : (!llvm.ptr, i32) -> !llvm.ptr, f32
          llvm.store %result, %out_element : f32, !llvm.ptr
          llvm.br ^exit
        ^exit:
          llvm.return
        }}
        ''')


def emit_bevformer_ffn_buffered_device_mlir(
    *, rows: int, channels: int, hidden: int, epsilon: float,
    optimized_mlir_sha256: str, call_plan_mlir_sha256: str,
    buffer_plan: VortexBufferPlan = VortexBufferPlan(),
) -> str:
    """Emit directly launchable DXA FC1/FC2 plus the scalar LayerNorm."""
    # Reuse scalar validation and immutable shape/weight provenance metadata.
    base = scalar.emit_bevformer_ffn_device_mlir(
        rows=rows, channels=channels, hidden=hidden, epsilon=epsilon,
        optimized_mlir_sha256=optimized_mlir_sha256,
        call_plan_mlir_sha256=call_plan_mlir_sha256,
    )
    if not isinstance(buffer_plan, VortexBufferPlan):
        raise BackendError("vortex-v3: buffer_plan must be a VortexBufferPlan")
    metadata = "\n".join(line.strip() for line in base.splitlines() if "llvm.mlir.global" in line and any(name in line for name in ("@__gac_plan_", "@__gac_optimized_mlir_", "@__gac_call_plan_mlir_")))
    metadata += "\n" + "\n".join(f'llvm.mlir.global private constant @__gac_buffer_{name}({value} : i32) {{section = "llvm.metadata"}} : i32' for name, value in (("tile_m", buffer_plan.tile_m), ("tile_n", buffer_plan.tile_n), ("tile_k", buffer_plan.tile_k), ("stages", 2), ("lmem_bytes", buffer_plan.lmem_bytes)))
    return scalar._module(scalar.BEVFORMER_FFN_KERNELS, "\n\n".join((_collective_helpers(), _matmul_kernel(buffer_plan, False), _matmul_kernel(buffer_plan, True), scalar._layer_norm_kernel())), metadata)


def audit_bevformer_ffn_buffered_device_mlir(
    source: str, *, kernels: tuple[str, ...], rows: int, channels: int,
    hidden: int, epsilon: float, optimized_mlir_sha256: str,
    call_plan_mlir_sha256: str, buffer_plan: VortexBufferPlan = VortexBufferPlan(),
) -> None:
    """Check provenance/ABI and the essential LMEM and DXA boundary.

    Native MLIR/LLVM verification and execution remain the numerical gates;
    this audit prevents artifact/host buffer contracts from drifting.
    """
    emit_bevformer_ffn_buffered_device_mlir(
        rows=rows, channels=channels, hidden=hidden, epsilon=epsilon,
        optimized_mlir_sha256=optimized_mlir_sha256,
        call_plan_mlir_sha256=call_plan_mlir_sha256, buffer_plan=buffer_plan,
    )
    if kernels != scalar.BEVFORMER_FFN_KERNELS or not isinstance(source, str):
        raise BackendError("vortex-v3: invalid buffered FFN kernel contract")
    failures: list[str] = []
    for token in (f'llvm.target_triple = "{scalar.TARGET_TRIPLE}"', f'llvm.data_layout = "{scalar.DATA_LAYOUT}"'):
        if token not in source:
            failures.append(token)
    for name, value in (("plan_rows", rows), ("plan_channels", channels), ("plan_hidden", hidden), ("buffer_tile_m", buffer_plan.tile_m), ("buffer_tile_n", buffer_plan.tile_n), ("buffer_tile_k", buffer_plan.tile_k), ("buffer_stages", 2), ("buffer_lmem_bytes", buffer_plan.lmem_bytes)):
        if re.findall(rf"@__gac_{name}\(\s*(\d+)\s*:\s*i32\)", source) != [str(value)]:
            failures.append(f"{name} provenance")
    match = re.search(r"@__gac_plan_epsilon\(([^:]+):\s*f32\)", source)
    if match is None or struct.pack("<f", float(match[1])) != struct.pack("<f", epsilon):
        failures.append("epsilon provenance")
    for name, value in (("optimized_mlir", optimized_mlir_sha256), ("call_plan_mlir", call_plan_mlir_sha256)):
        if re.findall(rf'@__gac_{name}_sha256\("([0-9a-fA-F]{{64}})\\00"\)', source) != [value.lower()]:
            failures.append(f"{name} provenance")
    scalar._audit_annotation_global(source, kernels=kernels, failures=failures)
    for name in ("llvm.used", "llvm.compiler.used"):
        scalar._audit_retention_global(source, name=name, kernels=kernels, failures=failures)
    for helper in ("__gac_dxa_expect_two", "__gac_dxa_issue_2d", "__gac_cta_barrier"):
        match = re.search(rf"llvm\.func internal @{helper}\([^\n]+", source)
        if match is None or any(token not in match[0] for token in ("convergent", '"noinline"', '"optnone"')):
            failures.append(f"{helper} convergence contract")
    for instruction in (".insn r4 $1, 0, 0, $0, $3, $4, $5\\0A\\09.insn r $6, 0, 3, x0, $0, x0", "={x31},i,0,r,r,r,i,~{memory}", ".insn r $1, 6, 0, $0, $2, $3", ".insn r $0, 4, 0, x0, $1, $2"):
        if instruction not in source:
            failures.append("collective hardware instruction contract")
    for second, kernel in enumerate(kernels[:2]):
        body = scalar._function_body(source, kernel)
        if body is None:
            failures.append(f"{kernel} body")
            continue
        pointers = ("query", "activation", "weight", "bias", "output") if second else ("query", "weight", "bias", "output")
        layout = "!llvm.struct<(i32, i32, i32, i32, " + ", ".join("i64" for _ in pointers) + ")>"
        used_pointers = ("query", "bias", "output") if second else ("bias", "output")
        scalar._audit_argument_unpack(body, kernel=kernel, layout=layout,
            fields=(("rows", 0, "i32"), ("channels", 1, "i32"), ("hidden", 2, "i32")) + tuple((name, i + 4, "i64") for i, name in enumerate(pointers) if name in used_pointers), failures=failures)
        for number in (0xCD3, 0xCD6, 0xCDF, 0xCD0, 0xCD2, 0xCD1):
            if not re.search(rf"constant\({number}\s*:\s*i32\)", body):
                failures.append(f"{kernel} CTA CSR {number}")
        for instruction, count in (("llvm.call @__gac_dxa_issue_2d", 4), ("llvm.call @__gac_dxa_expect_two", 2), ("llvm.call @__gac_cta_barrier", 2)):
            if body.count(instruction) != count:
                failures.append(f"{kernel} DXA/barrier instruction contract")
        if body.count("llvm.fmul") != 1 or body.count("llvm.store") != 1:
            failures.append(f"{kernel} multiply/writeback contract")
    norm = scalar._function_body(source, kernels[2])
    if norm is None:
        failures.append("LayerNorm body")
    else:
        scalar._audit_layer_norm_structure(norm, failures)
    if failures:
        raise BackendError("vortex-v3: invalid buffered device MLIR: " + ", ".join(dict.fromkeys(failures)))


__all__ = ["VortexBufferPlan", "emit_bevformer_ffn_buffered_device_mlir", "audit_bevformer_ffn_buffered_device_mlir"]
