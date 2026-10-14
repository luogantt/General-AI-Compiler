# Vortex Direct-MLIR ABI and Phase-1 Bring-up

## Status and scope

This document fixes the ABI contract for removing the generated device-side
`kernel.cpp` from the GAC Vortex path.  The implemented and validated phase-1
pipeline is:

```text
GAC graph
  -> Tensor/Linalg/SCF MLIR
  -> LLVM-dialect MLIR
  -> mlir-translate --mlir-to-llvmir
  -> Vortex LLVM 20.1.8 (VOLT) instruction selection
  -> RISC-V object
  -> Vortex startup + libvortex2.a + link32.ld
  -> ELF
  -> vxbin.py
  -> .vxbin
  -> vortex2 host
  -> SimX
```

The full run and its reproducible evidence are recorded in the
[direct-MLIR validation report](../validation/vortex/direct-mlir-2026-10-13/README.md).

The device kernel is generated from MLIR/LLVM IR.  VOLT remains the machine
code backend: it still performs instruction selection, register allocation,
SIMT divergence lowering, scheduling, relocation emission, and ELF emission.
The host launcher and the assembly/device runtime are not removed by this
change.

The source and binary tools must come from the same Vortex LLVM 20 tree.  The
currently selected source revision is:

```text
branch: volt-llvm20.1.8
commit: b2ecfe086df2d00233a87f3d3ebe6a74aba129e2
projects: clang;lld;mlir
targets: RISCV;X86
```

Do not feed LLVM 24 bitcode to this LLVM 20 backend.  Textual IR is also not a
supported long-term version boundary.  Build `mlir-opt`, `mlir-translate`,
`opt`, `llc`, `clang`, `ld.lld`, `llvm-nm`, `llvm-readelf`, `llvm-objdump`, and
`llvm-objcopy` from this one tree.

The implementation sources that define this contract are:

| Contract | Source of truth |
|---|---|
| annotation to entry alias | `llvm/lib/Target/RISCV/RISCVVortexKEntry.cpp` |
| kernel callee-save convention | `llvm/lib/Target/RISCV/RISCVRegisterInfo.cpp` |
| SIMT divergence/uniform rules | `llvm/lib/Target/RISCV/VortexBranchDivergence.cpp` |
| C/C++ spelling of kernel and uniform attributes | `$VORTEX_HOME/sw/kernel/include/vx_spawn2.h`, `vx_intrinsics.h` |
| startup and `a0` delivery | `$VORTEX_HOME/sw/kernel/src/vx_start.S` |
| CSR numbers | `$VORTEX_HOME/VX_types.toml` |
| ELF-to-vxbin symbol table | `$VORTEX_HOME/sw/kernel/scripts/vxbin.py` |
| target/link flags | `$VORTEX_HOME/tests/regression/common.mk` |

## RV32 target contract

The first target is the current RV32 FP32 Vortex configuration:

```text
target triple:      riscv32-unknown-unknown-elf
driver spelling:    --target=riscv32-unknown-elf
data layout:        e-m:e-p:32:32-i64:64-n32-S128
ISA/ABI:            rv32imaf / ilp32f
Vortex features:    +xvortex,+zicond
code model:         medany (LLVM code-model=medium)
startup address:    0x80000000
byte order:         little-endian
```

The checked-in hardware configuration used for bring-up has one core, four
warps per core, four threads per warp, LMEM enabled, and TCU disabled.  These
values are validation context rather than constants to bake into generated
code; launch dimensions still come from the CTA CSRs.

The data layout is the value returned by the checked-out RISC-V target for
RV32.  In particular, pointers are 32 bits and `i64` has 8-byte ABI alignment.
Phase 1 uses generic address space zero for the argument buffer and all global
device pointers.

## Kernel entry ABI

Every launchable function has one argument:

```llvm
define void @vector_add(ptr %argument_buffer)
```

The KMU startup code reads the argument-buffer address from
`VX_CSR_MSCRATCH` into `a0` and calls the selected entry.  The ordinary RV32 C
ABI therefore supplies `%argument_buffer` in `a0`.  MLIR memref descriptors
are not passed across this boundary.  The wrapper loads scalar fields and raw
device addresses from the packed argument buffer and then calls or contains
the lowered kernel body.

Four pieces must be emitted for a canonical launchable function:

1. Add the string function attribute `"vortex-kernel"`.  The VOLT register
   information code recognizes it and uses the kernel callee-save set (only
   `ra` when needed).
2. Add the function to `llvm.global.annotations` with the exact string
   `"vortex.kernel"`.  The register-information code also recognizes this
   form, and `RISCVVortexKEntry` uses this annotation to generate the launch
   symbol.
3. Add the function to `llvm.used`.  This is the IR representation of
   Clang's `retain` attribute and causes ELF `SHF_GNU_RETAIN`, protecting the
   section from `--gc-sections`.
4. Add the function to `llvm.compiler.used`.  This is the IR representation
   needed for Clang's `used` behavior and protects the unreferenced body from
   compiler elimination.

The distinction between the first two items matters.  In the current VOLT
source, `"vortex-kernel"` alone changes the callee-save convention but does
**not** create a `__vx_kentry_*` symbol.  The alias-emission pass scans only
`llvm.global.annotations` for `"vortex.kernel"`.

For a function named `vector_add`, VOLT emits:

```llvm
@__vx_kentry_vector_add = alias void (ptr), ptr @vector_add
```

`vxbin.py` scans the linked ELF for `__vx_kentry_*`.  It publishes this entry
as `vector_add`.  The conventional name `kernel_main` is the one special case:
it is published as `main`.

The current backend emits an ELF alias at the function address and places the
function body in the retained executable section `.vx_entry.<kernel>`.  It
does not emit a separate entry trampoline: the alias and implementation have
the same address.  Direct MLIR follows the current `RISCVVortexKEntry.cpp`
implementation.  The shared startup code reads
`VX_CSR_CTA_ENTRY`, reads `VX_CSR_MSCRATCH`, executes `jalr`, drains the warp,
and terminates it.

## Argument-buffer layout

The host copies one byte-for-byte argument structure into the launch request.
The MLIR wrapper must use fixed byte offsets derived from the RV32 data layout,
not a host-language struct reconstructed opportunistically.

The minimal vector-add buffer is:

| Offset | LLVM type | Meaning |
|---:|---|---|
| 0 | `i32` | element count |
| 4 | 4 bytes padding | align the first `i64` |
| 8 | `i64` | source A device address |
| 16 | `i64` | source B device address |
| 24 | `i64` | destination device address |
| 32 | end | total size 32, alignment 8 |

On RV32, load each address as `i64`, truncate it to `i32`, and apply
`inttoptr`.  This preserves the public argument-buffer format used by the
existing host while making the 32-bit device pointer conversion explicit.
The host rejects any runtime buffer address above `UINT32_MAX` before launch,
so the truncation cannot silently discard a live address bit.

The phase-1 BEVFormer FFN buffers are fixed as follows:

| Kernel | Scalar offsets | Address offsets | Size |
|---|---|---|---:|
| FC1 | rows 0, channels 4, hidden 8 | query 16, weight 24, bias 32, output 40 | 48 |
| FC2 | rows 0, channels 4, hidden 8 | query 16, activation 24, weight 32, bias 40, output 48 | 56 |
| LayerNorm | rows 0, channels 4, epsilon/f32 8 | input 16, scale 24, bias 32, output 40 | 48 |

Changing a field or address width requires changing both the host packer and
the MLIR ABI wrapper.  The artifact manifest should record every field offset
and total size.

## CTA identifiers and synchronization

The initial `gpu.block_id`, `gpu.thread_id`, and `gpu.block_dim` lowering is a
non-volatile CSR read.  The CSR number must remain an immediate operand of the
LLVM inline-assembly call.  VOLT's divergence analysis examines that operand;
hard-coding the CSR only inside the assembly string bypasses that analysis and
can access a nonexistent argument in its current implementation.

| MLIR/GPU value | CSR | Value |
|---|---|---:|
| local CTA id / barrier id | `VX_CSR_CTA_ID` | `0xCD0` |
| warp rank in CTA | `VX_CSR_CTA_RANK` | `0xCD1` |
| warp count in CTA | `VX_CSR_CTA_SIZE` | `0xCD2` |
| thread id x/y/z | `VX_CSR_CTA_THREAD_ID_X/Y/Z` | `0xCD3`/`0xCD4`/`0xCD5` |
| block id x/y/z | `VX_CSR_CTA_BLOCK_ID_X/Y/Z` | `0xCD6`/`0xCD7`/`0xCD8` |
| block dim x/y/z | `VX_CSR_CTA_BLOCK_DIM_X/Y/Z` | `0xCD9`/`0xCDA`/`0xCDB` |
| grid dim x/y/z | `VX_CSR_CTA_GRID_DIM_X/Y/Z` | `0xCDC`/`0xCDD`/`0xCDE` |
| CTA local-memory base | `VX_CSR_CTA_LMEM_ADDR` | `0xCDF` |

Canonical LLVM IR for a CSR read is:

```llvm
%thread.x = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
```

`block_id`, `block_dim`, the argument-buffer pointer, and launch dimensions are
uniform.  Mark their defining instructions with `!vortex.uniform` or insert
`llvm.riscv.vx.uniform.*` through the lowering pass.  Do not mark
`thread_id` uniform.

The CTA barrier is Vortex custom opcode 0 (`0x0B`), funct3 `4`, funct7 `0`:

```llvm
call void asm sideeffect
  ".insn r $0, 4, 0, x0, $1, $2",
  "i,r,r,~{memory}"(i32 11, i32 %cta_id, i32 %cta_warps)
```

The `sideeffect` flag and memory clobber are mandatory.  The operands match
`vx_barrier(get_local_group_id(), get_num_sub_groups())`; substituting thread
or global grid identifiers changes the rendezvous domain and is incorrect.

## Golden LLVM IR: vector add without a C++ device kernel

The following module is the phase-1 ABI golden.  It deliberately includes a
bounds check so the VOLT SIMT divergence pass is exercised.

```llvm
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

@.vortex.kernel = private unnamed_addr constant [14 x i8] c"vortex.kernel\00", section "llvm.metadata"
@.vortex.file = private unnamed_addr constant [4 x i8] c"gac\00", section "llvm.metadata"

@llvm.global.annotations = appending global [1 x { ptr, ptr, ptr, i32, ptr }]
  [{ ptr, ptr, ptr, i32, ptr }
   { ptr @vector_add, ptr @.vortex.kernel, ptr @.vortex.file, i32 1, ptr null }],
  section "llvm.metadata"

@llvm.used = appending global [1 x ptr] [ptr @vector_add], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @vector_add], section "llvm.metadata"

define void @vector_add(ptr %args) #0 section ".vx_entry.vector_add" {
entry:
  %args.uniform = getelementptr i8, ptr %args, i32 0, !vortex.uniform !0

  %n.ptr = getelementptr i8, ptr %args.uniform, i32 0
  %n = load i32, ptr %n.ptr, align 8, !vortex.uniform !0

  %a.slot = getelementptr i8, ptr %args.uniform, i32 8
  %b.slot = getelementptr i8, ptr %args.uniform, i32 16
  %c.slot = getelementptr i8, ptr %args.uniform, i32 24
  %a.i64 = load i64, ptr %a.slot, align 8, !vortex.uniform !0
  %b.i64 = load i64, ptr %b.slot, align 8, !vortex.uniform !0
  %c.i64 = load i64, ptr %c.slot, align 8, !vortex.uniform !0
  %a.i32 = trunc i64 %a.i64 to i32
  %b.i32 = trunc i64 %b.i64 to i32
  %c.i32 = trunc i64 %c.i64 to i32
  %a = inttoptr i32 %a.i32 to ptr
  %b = inttoptr i32 %b.i32 to ptr
  %c = inttoptr i32 %c.i32 to ptr

  %tid = call i32 asm "csrr $0, $1", "=r,i"(i32 3283)
  %bid = call i32 asm "csrr $0, $1", "=r,i"(i32 3286), !vortex.uniform !0
  %bdim = call i32 asm "csrr $0, $1", "=r,i"(i32 3289), !vortex.uniform !0
  %base = mul i32 %bid, %bdim
  %index = add i32 %base, %tid
  %inside = icmp ult i32 %index, %n
  br i1 %inside, label %body, label %exit

body:
  %ap = getelementptr float, ptr %a, i32 %index
  %bp = getelementptr float, ptr %b, i32 %index
  %cp = getelementptr float, ptr %c, i32 %index
  %av = load float, ptr %ap, align 4
  %bv = load float, ptr %bp, align 4
  %sum = fadd float %av, %bv
  store float %sum, ptr %cp, align 4
  br label %exit

exit:
  ret void
}

attributes #0 = { nounwind "vortex-kernel" }
!0 = !{}
```

After VOLT code generation, the object must contain both `vector_add` and
`__vx_kentry_vector_add`.  Absence of the latter is an ABI failure, even if
the object otherwise disassembles correctly.

## LLVM-dialect MLIR body

The corresponding minimal kernel body is below.  The implemented emitter also
materializes the three LLVM globals shown above (`llvm.global.annotations`,
`llvm.used`, and `llvm.compiler.used`) in the same MLIR module before
translation.  A string pass-through attribute alone is insufficient.

```mlir
module attributes {
  llvm.data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
  llvm.target_triple = "riscv32-unknown-unknown-elf"
} {
  llvm.func @vector_add(%args: !llvm.ptr)
      attributes {passthrough = ["vortex-kernel"]} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c16 = llvm.mlir.constant(16 : i32) : i32
    %c24 = llvm.mlir.constant(24 : i32) : i32
    %tid.csr = llvm.mlir.constant(3283 : i32) : i32
    %bid.csr = llvm.mlir.constant(3286 : i32) : i32
    %bdim.csr = llvm.mlir.constant(3289 : i32) : i32

    %args.u = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%args)
        : (!llvm.ptr) -> !llvm.ptr
    %n.slot = llvm.getelementptr %args.u[%c0]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %a.slot = llvm.getelementptr %args.u[%c8]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %b.slot = llvm.getelementptr %args.u[%c16]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %c.slot = llvm.getelementptr %args.u[%c24]
        : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %n = llvm.load %n.slot {alignment = 8 : i64} : !llvm.ptr -> i32
    %a64 = llvm.load %a.slot {alignment = 8 : i64} : !llvm.ptr -> i64
    %b64 = llvm.load %b.slot {alignment = 8 : i64} : !llvm.ptr -> i64
    %c64 = llvm.load %c.slot {alignment = 8 : i64} : !llvm.ptr -> i64
    %a32 = llvm.trunc %a64 : i64 to i32
    %b32 = llvm.trunc %b64 : i64 to i32
    %c32 = llvm.trunc %c64 : i64 to i32
    %a = llvm.inttoptr %a32 : i32 to !llvm.ptr
    %b = llvm.inttoptr %b32 : i32 to !llvm.ptr
    %c = llvm.inttoptr %c32 : i32 to !llvm.ptr

    %tid = llvm.inline_asm "csrr $0, $1", "=r,i" %tid.csr
        : (i32) -> i32
    %bid.raw = llvm.inline_asm "csrr $0, $1", "=r,i" %bid.csr
        : (i32) -> i32
    %bdim.raw = llvm.inline_asm "csrr $0, $1", "=r,i" %bdim.csr
        : (i32) -> i32
    %bid = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%bid.raw)
        : (i32) -> i32
    %bdim = llvm.call_intrinsic "llvm.riscv.vx.uniform"(%bdim.raw)
        : (i32) -> i32
    %base = llvm.mul %bid, %bdim : i32
    %index = llvm.add %base, %tid : i32
    %inside = llvm.icmp "ult" %index, %n : i32
    llvm.cond_br %inside, ^body, ^exit

  ^body:
    %ap = llvm.getelementptr %a[%index]
        : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %bp = llvm.getelementptr %b[%index]
        : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %cp = llvm.getelementptr %c[%index]
        : (!llvm.ptr, i32) -> !llvm.ptr, f32
    %av = llvm.load %ap {alignment = 4 : i64} : !llvm.ptr -> f32
    %bv = llvm.load %bp {alignment = 4 : i64} : !llvm.ptr -> f32
    %sum = llvm.fadd %av, %bv : f32
    llvm.store %sum, %cp {alignment = 4 : i64} : f32, !llvm.ptr
    llvm.br ^exit

  ^exit:
    llvm.return
  }
}
```

The `llvm.riscv.vx.uniform` calls are compiler markers.  VOLT's divergence
pass consumes and erases them before instruction selection.  Do not compile a
module containing these markers with `-vortex-branch-divergence=0`, because
they have no standalone machine instruction.

## Build and validation commands

The repository provides a pinned WSL/Linux build script.  It rejects any LLVM
checkout other than revision `b2ecfe086df2d00233a87f3d3ebe6a74aba129e2`,
copies source blobs into a Linux filesystem with LF line endings, builds all
required tools, and writes the revision marker checked by the compiler:

```bash
bash tools/build_volt_mlir_wsl.sh
```

The equivalent manual configuration on a Linux Vortex machine is:

```bash
VOLT_REV=b2ecfe086df2d00233a87f3d3ebe6a74aba129e2
test "$(git -C "$LLVM_SRC" rev-parse HEAD)" = "$VOLT_REV"
cmake -G Ninja -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_PREFIX" \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF
cmake --build "$LLVM_BUILD" --target \
  mlir-opt mlir-translate opt llc clang lld llvm-nm llvm-readelf \
  llvm-objdump llvm-objcopy
cmake --install "$LLVM_BUILD"
"$LLVM_PREFIX/bin/mlir-opt" --version | grep -F '20.1.8'
"$LLVM_PREFIX/bin/mlir-translate" --version | grep -F '20.1.8'
"$LLVM_PREFIX/bin/clang" --version | grep -F '20.1.8'
for tool in clang++ ld.lld llvm-nm llvm-readelf llvm-objdump llvm-objcopy; do
  test -x "$LLVM_PREFIX/bin/$tool"
done
"$LLVM_PREFIX/bin/ld.lld" --version | grep -F '20.1.8'
printf '%s\n' "$VOLT_REV" > "$LLVM_PREFIX/.gac-build-revision"
```

Lower and verify the IR:

```bash
"$LLVM_PREFIX/bin/mlir-opt" kernel.mlir \
  --canonicalize --cse -o kernel.opt.mlir
"$LLVM_PREFIX/bin/mlir-translate" --mlir-to-llvmir \
  kernel.opt.mlir -o kernel.ll

"$LLVM_PREFIX/bin/opt" -passes=verify -disable-output kernel.ll
"$LLVM_PREFIX/bin/llc" kernel.ll -o kernel.check.o \
  -mtriple=riscv32-unknown-unknown-elf \
  -mattr=+m,+a,+f,+xvortex,+zicond \
  -target-abi=ilp32f \
  -relocation-model=static -code-model=medium \
  -O3 -filetype=obj
"$LLVM_PREFIX/bin/llvm-nm" kernel.check.o | grep -E \
  'vector_add|__vx_kentry_vector_add'
```

No post-translation ABI attachment tool is required: `kernel.mlir` already
contains the canonical annotation and both retention lists.  Treat absence of
any of these globals in `kernel.ll` as a compiler error.

The existing Vortex regression Makefile can compile and link the `.ll` device
kernel without a C++ device source while retaining its known-good startup,
linker script, libraries, host launcher, and `.vxbin` packaging:

```bash
cp kernel.ll "$VORTEX_HOME/tests/regression/vecadd/kernel.ll"
make -C "$VORTEX_HOME/tests/regression/vecadd" clean
make -C "$VORTEX_HOME/tests/regression/vecadd" \
  LLVM_PATH="$LLVM_PREFIX" \
  VX_SRCS="$VORTEX_HOME/tests/regression/vecadd/kernel.ll" \
  KERNEL_LIB=vortex2 XLEN=32

readelf -s -W "$VORTEX_HOME/tests/regression/vecadd/kernel.elf" \
  | grep __vx_kentry_vector_add
"$LLVM_PREFIX/bin/llvm-objdump" -d \
  "$VORTEX_HOME/tests/regression/vecadd/kernel.elf" \
  > "$VORTEX_HOME/tests/regression/vecadd/kernel.direct-mlir.dump"

make -C "$VORTEX_HOME/tests/regression/vecadd" run-simx \
  LLVM_PATH="$LLVM_PREFIX" \
  VX_SRCS="$VORTEX_HOME/tests/regression/vecadd/kernel.ll" \
  KERNEL_LIB=vortex2 XLEN=32 OPTS=-n64
```

The acceptance gates are:

1. MLIR and LLVM verification pass.
2. `kernel.check.o` and the final ELF contain `__vx_kentry_vector_add`.
3. Disassembly contains CSR reads for `0xCD3`, `0xCD6`, and `0xCD9`, Vortex
   split/join around the divergent bounds check, and an FP32 add/store.
4. `vxbin.py` writes a `VXSYMTAB` entry named `vector_add`.
5. SimX vector-add output matches the CPU golden for all tested lengths,
   including a length that is not a multiple of the block size.
6. The same `.vxbin` passes on RTL/FPGA before performance claims are made.

## Phase-1 limits

The first milestone intentionally supports only:

- RV32/ILP32F and raw-pointer argument-buffer wrappers;
- static-shape FP32 elementwise operations and the scalar BEVFormer FFN
  kernels;
- one-dimensional grid/block mapping;
- VOLT's existing scalar SIMT divergence lowering;
- existing Vortex startup assembly, `libvortex2.a`, linker script, runtime,
  and host launcher.

It does not yet provide generic MemRef ABI interoperability, dynamic shapes,
shared-memory tiling, vector lowering, double buffering, WMMA/WGMMA, TCU
support, autotuning, or the complete BEVFormer operator set.  Removing
`kernel.cpp` proves the compiler path and does not by itself improve latency;
performance work starts after scalar numerical parity is established.

The optional `dxa-double-buffer` schedule added on 2026-10-14 preserves this
argument ABI and supplies a tiled DXA/LMEM implementation for FC1/FC2.
See [buffered lowering](vortex_buffered_mlir.md) for the additional LMEM,
physical-address descriptor, collective and cache-visibility contracts.
