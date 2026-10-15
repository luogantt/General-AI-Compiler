"""Executable SM86 MatMul probe with an NVGPU async-copy pipeline.

The kernel in this module is deliberately small and fixed-shape.  It computes
``C[16, 16] = A[16, 64] @ B[64, 16]`` from caller-owned FP32 buffers.  Four
``K=16`` tiles move through two physical workgroup-memory stages using
``nvgpu.device_async_copy``.  The schedule keeps one future copy group in
flight while the previous tile is consumed, then reuses the stages in ping-pong
order.

This is a functional compiler and CUDA-device probe.  It proves real global
input loads, shared-memory tiling, double buffering, arithmetic, and global
output stores in one kernel.  The fixed shape keeps its ABI and numerical
oracle unambiguous; it is not wired into the production BEVFormer FFN runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import re

from gac.core.errors import BackendError
from gac.toolchains.mlir import MlirPtxArtifact, lower_gpu_mlir_to_ptx


ASYNC_MATMUL_SYMBOL = "gac_async_tiled_matmul_f32_16x16x64"
ASYNC_MATMUL_M = 16
ASYNC_MATMUL_N = 16
ASYNC_MATMUL_K = 64
ASYNC_MATMUL_TILE_K = 16
ASYNC_MATMUL_STAGES = 2
ASYNC_MATMUL_BLOCK = (64, 1, 1)
ASYNC_MATMUL_GRID = (1, 1, 1)
ASYNC_MATMUL_SHARED_BYTES = (
    2 * ASYNC_MATMUL_STAGES * ASYNC_MATMUL_M * ASYNC_MATMUL_TILE_K * 4
)


@dataclass(frozen=True, slots=True)
class AsyncTiledMatmulPtxEvidence:
    """Fail-closed textual PTX evidence for the functional kernel."""

    shared_declarations: int
    static_shared_bytes: int
    async_copy_instructions: int
    async_commit_groups: int
    async_wait_group_depths: tuple[int, ...]
    barriers: int
    shared_loads: int
    fp32_multiplies: int
    fp32_adds: int
    fma_instructions: int
    global_stores: int

    @property
    def has_ping_pong_schedule(self) -> bool:
        return (
            self.async_copy_instructions == 8
            and self.async_commit_groups == 4
            and self.async_wait_group_depths == (1, 1, 1, 0)
        )

    @property
    def computes_real_matmul(self) -> bool:
        has_arithmetic = self.fma_instructions >= 4 or (
            self.fp32_multiplies >= 4 and self.fp32_adds >= 4
        )
        return self.shared_loads >= 8 and has_arithmetic and self.global_stores >= 4


@dataclass(frozen=True, slots=True)
class AsyncTiledMatmulPtxProgram:
    """Lowered executable fixed-shape kernel and exact launch contract."""

    symbol: str
    m: int
    n: int
    k: int
    tile_k: int
    stages: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    gpu_source: str
    artifact: MlirPtxArtifact
    evidence: AsyncTiledMatmulPtxEvidence

    @property
    def ptx(self) -> str:
        return self.artifact.ptx


def _consume_stage(
    *,
    suffix: str,
    stage: str,
    input_accumulators: tuple[str, str, str, str],
) -> str:
    """Emit one K=16 shared-memory reduction with four outputs per thread."""

    result_names = ", ".join(f"%acc{i}_{suffix}" for i in range(4))
    iter_args = ", ".join(
        f"%inner{i}_{suffix} = {value}"
        for i, value in enumerate(input_accumulators)
    )
    result_types = ", ".join("f32" for _ in range(4))
    yields = ", ".join(f"%next{i}_{suffix}" for i in range(4))
    body = [
        f"      {result_names} = scf.for %kk_{suffix} = %c0 to %c16 step %c1",
        f"          iter_args({iter_args}) -> ({result_types}) {{",
        f"        %lhs_{suffix} = memref.load %shared_a[{stage}, %row, %kk_{suffix}]",
        "          : memref<2x16x16xf32, #gpu.address_space<workgroup>>",
    ]
    for index in range(4):
        body.extend(
            (
                f"        %rhs{index}_{suffix} = memref.load %shared_b[{stage}, %kk_{suffix}, %col{index}]",
                "          : memref<2x16x16xf32, #gpu.address_space<workgroup>>",
                f"        %product{index}_{suffix} = arith.mulf %lhs_{suffix}, %rhs{index}_{suffix} : f32",
                f"        %next{index}_{suffix} = arith.addf %inner{index}_{suffix}, %product{index}_{suffix} : f32",
            )
        )
    body.extend((f"        scf.yield {yields} : {result_types}", "      }"))
    return "\n".join(body)


def emit_async_tiled_matmul_gpu_mlir() -> str:
    """Emit real-input FP32 MatMul with a two-stage ``cp.async`` pipeline."""

    consume0 = _consume_stage(
        suffix="s0_first",
        stage="%c0",
        input_accumulators=("%zero", "%zero", "%zero", "%zero"),
    )
    consume1 = _consume_stage(
        suffix="s1_first",
        stage="%c1",
        input_accumulators=tuple(f"%acc{i}_s0_first" for i in range(4)),
    )
    consume2 = _consume_stage(
        suffix="s0_second",
        stage="%c0",
        input_accumulators=tuple(f"%acc{i}_s1_first" for i in range(4)),
    )
    consume3 = _consume_stage(
        suffix="s1_second",
        stage="%c1",
        input_accumulators=tuple(f"%acc{i}_s0_second" for i in range(4)),
    )

    return f'''// Functional SM86 async-copy MatMul candidate.
// ABI: row-major A[16,64], B[64,16], C[16,16] as three bare pointers.
// Schedule: prefetch K0/K1; wait1+consume K0; refill stage0 with K2;
//           wait1+consume K1; refill stage1 with K3; consume K2/K3.
module attributes {{gpu.container_module}} {{
  gpu.module @gac_async_tiled_matmul_sm86 {{
    gpu.func @{ASYNC_MATMUL_SYMBOL}(
        %a: memref<16x64xf32, #gpu.address_space<global>>,
        %b: memref<64x16xf32, #gpu.address_space<global>>,
        %c: memref<16x16xf32, #gpu.address_space<global>>)
        workgroup(
          %shared_a: memref<2x16x16xf32, #gpu.address_space<workgroup>>,
          %shared_b: memref<2x16x16xf32, #gpu.address_space<workgroup>>)
        kernel attributes {{known_block_size = array<i32: 64, 1, 1>}} {{
      %tid = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %zero = arith.constant 0.0 : f32
      %row = arith.divui %tid, %c4 : index
      %vector_id = arith.remui %tid, %c4 : index
      %col0 = arith.muli %vector_id, %c4 : index
      %col1 = arith.addi %col0, %c1 : index
      %col2 = arith.addi %col0, %c2 : index
      %col3 = arith.addi %col0, %c3 : index
      %a_col1 = arith.addi %col0, %c16 : index
      %a_col2 = arith.addi %col0, %c32 : index
      %a_col3 = arith.addi %col0, %c48 : index

      // Fill both physical stages.  Each thread copies one aligned float4
      // from A and one aligned float4 from B for every K tile.
      %a_copy0 = nvgpu.device_async_copy
        %a[%row, %col0], %shared_a[%c0, %row, %col0], 4
        : memref<16x64xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %b_copy0 = nvgpu.device_async_copy
        %b[%row, %col0], %shared_b[%c0, %row, %col0], 4
        : memref<64x16xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %group0 = nvgpu.device_async_create_group %a_copy0, %b_copy0

      %a_copy1 = nvgpu.device_async_copy
        %a[%row, %a_col1], %shared_a[%c1, %row, %col0], 4
        : memref<16x64xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %b_row1 = arith.addi %row, %c16 : index
      %b_copy1 = nvgpu.device_async_copy
        %b[%b_row1, %col0], %shared_b[%c1, %row, %col0], 4
        : memref<64x16xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %group1 = nvgpu.device_async_create_group %a_copy1, %b_copy1

      // At most one newer copy group may remain in flight while stage 0 is read.
      nvgpu.device_async_wait %group0 numGroups = 1
      gpu.barrier
{consume0}
      gpu.barrier

      // Refill stage 0 with K=[32,48) while stage 1 remains resident.
      %a_copy2 = nvgpu.device_async_copy
        %a[%row, %a_col2], %shared_a[%c0, %row, %col0], 4
        : memref<16x64xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %b_row2 = arith.addi %row, %c32 : index
      %b_copy2 = nvgpu.device_async_copy
        %b[%b_row2, %col0], %shared_b[%c0, %row, %col0], 4
        : memref<64x16xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %group2 = nvgpu.device_async_create_group %a_copy2, %b_copy2

      nvgpu.device_async_wait %group1 numGroups = 1
      gpu.barrier
{consume1}
      gpu.barrier

      // Refill stage 1 with K=[48,64) while the stage-0 refill is in flight.
      %a_copy3 = nvgpu.device_async_copy
        %a[%row, %a_col3], %shared_a[%c1, %row, %col0], 4
        : memref<16x64xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %b_row3 = arith.addi %row, %c48 : index
      %b_copy3 = nvgpu.device_async_copy
        %b[%b_row3, %col0], %shared_b[%c1, %row, %col0], 4
        : memref<64x16xf32, #gpu.address_space<global>> to memref<2x16x16xf32, #gpu.address_space<workgroup>>
      %group3 = nvgpu.device_async_create_group %a_copy3, %b_copy3

      nvgpu.device_async_wait %group2 numGroups = 1
      gpu.barrier
{consume2}
      gpu.barrier

      nvgpu.device_async_wait %group3 numGroups = 0
      gpu.barrier
{consume3}

      memref.store %acc0_s1_second, %c[%row, %col0]
        : memref<16x16xf32, #gpu.address_space<global>>
      memref.store %acc1_s1_second, %c[%row, %col1]
        : memref<16x16xf32, #gpu.address_space<global>>
      memref.store %acc2_s1_second, %c[%row, %col2]
        : memref<16x16xf32, #gpu.address_space<global>>
      memref.store %acc3_s1_second, %c[%row, %col3]
        : memref<16x16xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def _static_shared_memory_bytes(ptx: str) -> int:
    total = 0
    declaration = re.compile(
        r"(?m)^[ \t]*\.shared\b[^\r\n;]*\.([busf])(8|16|32|64)\b"
        r"[^\r\n;]*\[([0-9]+)\][ \t]*;"
    )
    for match in declaration.finditer(ptx):
        total += (int(match.group(2)) // 8) * int(match.group(3))
    return total


def inspect_async_tiled_matmul_ptx(ptx: str) -> AsyncTiledMatmulPtxEvidence:
    """Extract concrete async-pipeline and computation evidence from PTX."""

    if not isinstance(ptx, str):
        raise TypeError("ptx must be a string")
    if not ptx.strip():
        raise ValueError("ptx must be non-empty")
    return AsyncTiledMatmulPtxEvidence(
        shared_declarations=len(re.findall(r"(?m)^[ \t]*\.shared\b", ptx)),
        static_shared_bytes=_static_shared_memory_bytes(ptx),
        async_copy_instructions=len(
            re.findall(r"\bcp\.async\.(?:ca|cg)\.shared\.global\b", ptx)
        ),
        async_commit_groups=len(re.findall(r"\bcp\.async\.commit_group\b", ptx)),
        async_wait_group_depths=tuple(
            int(value)
            for value in re.findall(r"\bcp\.async\.wait_group[ \t]+([0-9]+)\b", ptx)
        ),
        barriers=len(re.findall(r"\bbar(?:\.sync|\.warp)?\b", ptx)),
        shared_loads=len(re.findall(r"\bld(?:\.volatile)?\.shared(?:\.|\b)", ptx)),
        fp32_multiplies=len(re.findall(r"\bmul(?:\.[A-Za-z0-9_]+)*\.f32\b", ptx)),
        fp32_adds=len(re.findall(r"\badd(?:\.[A-Za-z0-9_]+)*\.f32\b", ptx)),
        fma_instructions=len(re.findall(r"\bfma(?:\.[A-Za-z0-9_]+)+\b", ptx)),
        global_stores=len(re.findall(r"\bst(?:\.volatile)?\.global(?:\.|\b)", ptx)),
    )


def _validate_ptx_abi(ptx: str) -> None:
    pattern = re.compile(
        rf"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+{ASYNC_MATMUL_SYMBOL}"
        rf"[ \t]*\((.*?)\)[ \t]*\r?\n[ \t]*\.maxntid[ \t]+([^\r\n]+)"
    )
    matches = pattern.findall(ptx)
    if len(matches) != 1:
        raise BackendError("async tiled MatMul PTX has no unambiguous entry ABI")
    header, maxntid = matches[0]
    parameters = [line.strip() for line in header.splitlines() if line.strip()]
    pointer = re.compile(
        r"^\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?"
        r"\s+[A-Za-z_$][A-Za-z0-9_.$]*,?$"
    )
    if len(parameters) != 3 or any(pointer.fullmatch(value) is None for value in parameters):
        raise BackendError("async tiled MatMul PTX requires exactly three bare global pointers")
    dimensions = tuple(int(value) for value in re.findall(r"[0-9]+", maxntid))
    if dimensions != ASYNC_MATMUL_BLOCK:
        raise BackendError("async tiled MatMul PTX requires .maxntid 64, 1, 1")


def _validate_pipeline_order(ptx: str) -> None:
    """Require PTX commit/wait ordering that proves overlap and stage reuse."""

    events = tuple(
        (match.start(), "commit" if match.group(1) is not None else f"wait{match.group(2)}")
        for match in re.finditer(
            r"\b(?:(cp\.async\.commit_group)|cp\.async\.wait_group[ \t]+([0-9]+))\b",
            ptx,
        )
    )
    labels = tuple(label for _, label in events)
    expected = (
        "commit",
        "commit",
        "wait1",
        "commit",
        "wait1",
        "commit",
        "wait1",
        "wait0",
    )
    if labels != expected:
        raise BackendError(f"async tiled MatMul PTX has an unexpected commit/wait order: {labels}")


def validate_async_tiled_matmul_ptx(ptx: str) -> AsyncTiledMatmulPtxEvidence:
    """Validate the fixed ABI, double-buffer schedule, and real computation."""

    _validate_ptx_abi(ptx)
    _validate_pipeline_order(ptx)
    evidence = inspect_async_tiled_matmul_ptx(ptx)
    if evidence.shared_declarations != 2:
        raise BackendError("async tiled MatMul PTX must contain two shared-memory arrays")
    if evidence.static_shared_bytes != ASYNC_MATMUL_SHARED_BYTES:
        raise BackendError(
            "async tiled MatMul PTX shared allocation must be exactly "
            f"{ASYNC_MATMUL_SHARED_BYTES} bytes, got {evidence.static_shared_bytes}"
        )
    if not evidence.has_ping_pong_schedule:
        raise BackendError(
            "async tiled MatMul PTX lost the required 8-copy, 4-group, "
            "wait[1,1,1,0] ping-pong schedule"
        )
    if evidence.barriers < 7:
        raise BackendError("async tiled MatMul PTX lost required block barriers")
    if not evidence.computes_real_matmul:
        raise BackendError("async tiled MatMul PTX lost shared arithmetic or global stores")
    if re.search(r"\bmalloc\b|\blocal_depot", ptx):
        raise BackendError("async tiled MatMul shared stages became local allocations")
    return evidence


def lower_async_tiled_matmul_to_ptx(
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> AsyncTiledMatmulPtxProgram:
    """Lower the functional async MatMul and fail closed on missing evidence."""

    source = emit_async_tiled_matmul_gpu_mlir()
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(ASYNC_MATMUL_SYMBOL,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    evidence = validate_async_tiled_matmul_ptx(artifact.ptx)
    return AsyncTiledMatmulPtxProgram(
        symbol=ASYNC_MATMUL_SYMBOL,
        m=ASYNC_MATMUL_M,
        n=ASYNC_MATMUL_N,
        k=ASYNC_MATMUL_K,
        tile_k=ASYNC_MATMUL_TILE_K,
        stages=ASYNC_MATMUL_STAGES,
        grid=ASYNC_MATMUL_GRID,
        block=ASYNC_MATMUL_BLOCK,
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )
