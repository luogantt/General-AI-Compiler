"""Explicit shared-memory MLIR GPU kernels for BEVFormer FFN MatMuls.

This module is intentionally independent from the production ``mlir_ptx``
backend.  It is a small, reviewable optimization candidate: Python emits an
already mapped GPU kernel, and the stock LLVM/MLIR GPU-to-NVVM pipeline turns
that kernel into PTX.  There is no CUDA C, NVRTC, cuBLAS, or implicit fallback
in this path.

Each block computes one ``tile_m x tile_n`` output tile.  One thread owns one
output element while every thread cooperatively fills the two workgroup-memory
tiles.  Both loads use strided loops, so ``tile_k`` remains configurable even
when the shared tiles contain more or fewer elements than the block has
threads.  All threads reach both barriers, including threads in a partial edge
tile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
import re

from gac.core.errors import BackendError
from gac.toolchains.mlir import MlirPtxArtifact, lower_gpu_mlir_to_ptx


BEVFORMER_ROWS = 2_500
UNIAD_ROWS = 40_000
SUPPORTED_FFN_ROWS = (BEVFORMER_ROWS, UNIAD_ROWS)
BEVFORMER_CHANNELS = 256
BEVFORMER_HIDDEN_CHANNELS = 512
DEFAULT_TILE = 16
MAX_CUDA_THREADS_PER_BLOCK = 1_024
DEFAULT_SHARED_MEMORY_LIMIT_BYTES = 48 * 1_024
ASYNC_DOUBLE_BUFFER_SYMBOL = "gac_nvgpu_async_double_buffer_smoke"
ASYNC_DOUBLE_BUFFER_STAGES = 2
ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS = 16
ASYNC_DOUBLE_BUFFER_SHARED_BYTES = (
    ASYNC_DOUBLE_BUFFER_STAGES * ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS * 4
)
FUNCTIONAL_MMA_TF32_SYMBOL = "gac_nvgpu_mma_tf32_matmul_16x8x16"
FUNCTIONAL_MMA_FP16_SYMBOL = "gac_nvgpu_mma_fp16_matmul_16x8x32"
FUNCTIONAL_MMA_M = 16
FUNCTIONAL_MMA_N = 8
FUNCTIONAL_MMA_K = 16
FUNCTIONAL_MMA_K_STEP = 4
FUNCTIONAL_MMA_K_SLICES = FUNCTIONAL_MMA_K // FUNCTIONAL_MMA_K_STEP
FUNCTIONAL_MMA_FP16_K = 32
FUNCTIONAL_MMA_FP16_K_STEP = 16
FUNCTIONAL_MMA_FP16_K_SLICES = FUNCTIONAL_MMA_FP16_K // FUNCTIONAL_MMA_FP16_K_STEP
FUNCTIONAL_MMA_BLOCK = (32, 1, 1)
FUNCTIONAL_MMA_GRID = (1, 1, 1)


_SYMBOL_PATTERN = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$]*\Z")


@dataclass(frozen=True, slots=True)
class TiledMatmulConfig:
    """Static matrix and launch contract for one tiled ``A @ B`` kernel.

    ``block_x`` must equal ``tile_n`` and ``block_y`` must equal ``tile_m`` in
    the current one-output-per-thread implementation.  They are explicit so a
    future register-tiled implementation can relax this invariant without
    changing the serialized tuning record.
    """

    symbol: str
    m: int
    n: int
    k: int
    tile_m: int = DEFAULT_TILE
    tile_n: int = DEFAULT_TILE
    tile_k: int = DEFAULT_TILE
    block_x: int = DEFAULT_TILE
    block_y: int = DEFAULT_TILE
    fuse_bias_relu: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or _SYMBOL_PATTERN.fullmatch(self.symbol) is None:
            raise ValueError("symbol must be a valid MLIR/PTX identifier")
        for name in ("m", "n", "k", "tile_m", "tile_n", "tile_k", "block_x", "block_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.fuse_bias_relu, bool):
            raise TypeError("fuse_bias_relu must be bool")
        if self.block_x != self.tile_n or self.block_y != self.tile_m:
            raise ValueError(
                "the scalar-output tiled kernel requires block_x == tile_n and "
                "block_y == tile_m"
            )
        if self.threads_per_block > MAX_CUDA_THREADS_PER_BLOCK:
            raise ValueError("the tiled kernel exceeds CUDA's 1024-thread block limit")
        if self.shared_memory_bytes > DEFAULT_SHARED_MEMORY_LIMIT_BYTES:
            raise ValueError("the tiled kernel exceeds the conservative 48 KiB shared-memory limit")

    @property
    def threads_per_block(self) -> int:
        return self.block_x * self.block_y

    @property
    def shared_memory_bytes(self) -> int:
        return 4 * (self.tile_m * self.tile_k + self.tile_k * self.tile_n)

    @property
    def grid(self) -> tuple[int, int, int]:
        return (
            (self.n + self.tile_n - 1) // self.tile_n,
            (self.m + self.tile_m - 1) // self.tile_m,
            1,
        )

    @property
    def block(self) -> tuple[int, int, int]:
        return (self.block_x, self.block_y, 1)


@dataclass(frozen=True, slots=True)
class TiledPtxEvidence:
    """Instruction evidence extracted from textual PTX.

    ``mma_instructions`` is deliberately separate from ordinary FP32 FMA.  A
    kernel is never described as a Tensor Core kernel unless an ``mma.sync`` or
    ``wmma`` instruction is present in the emitted PTX.
    """

    shared_declarations: int
    static_shared_bytes: int
    shared_loads: int
    shared_stores: int
    global_stores: int
    vector_shared_loads: int
    barriers: int
    fp32_multiplies: int
    fp32_adds: int
    fma_instructions: int
    mma_instructions: int
    async_copy_instructions: int
    async_commit_groups: int
    async_wait_groups: int
    async_wait_group_depths: tuple[int, ...]

    @property
    def uses_tensor_cores(self) -> bool:
        return self.mma_instructions > 0

    @property
    def uses_async_global_to_shared_copy(self) -> bool:
        return self.async_copy_instructions > 0

    @property
    def computes_real_matmul(self) -> bool:
        """Whether PTX retains floating-point accumulation and an output store."""

        has_arithmetic = self.fma_instructions > 0 or (
            self.fp32_multiplies > 0 and self.fp32_adds > 0
        )
        return has_arithmetic and self.global_stores > 0

    @property
    def has_two_stage_async_pipeline(self) -> bool:
        """Whether PTX proves the two-group wait-one/consume/wait-zero schedule."""

        return (
            self.async_copy_instructions == 2
            and self.async_commit_groups == 2
            and self.async_wait_groups == 2
            and self.async_wait_group_depths == (1, 0)
        )


@dataclass(frozen=True, slots=True)
class TiledMatmulPtxProgram:
    """One lowered tiled MatMul plus its exact launch and PTX evidence."""

    config: TiledMatmulConfig
    gpu_source: str
    artifact: MlirPtxArtifact
    evidence: TiledPtxEvidence

    @property
    def ptx(self) -> str:
        return self.artifact.ptx

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.config.grid

    @property
    def block(self) -> tuple[int, int, int]:
        return self.config.block


@dataclass(frozen=True, slots=True)
class MmaPtxProbe:
    """A compiler-capability probe, not a complete production MatMul kernel."""

    mode: str
    gpu_source: str
    artifact: MlirPtxArtifact
    evidence: TiledPtxEvidence


@dataclass(frozen=True, slots=True)
class FunctionalMmaPtxEvidence:
    """PTX evidence for the real-input warp-level MMA MatMul probe."""

    global_loads: int
    global_stores: int
    mma_instructions: int
    tf32_mma_instructions: int
    fp16_mma_instructions: int
    loads_precede_first_mma: bool
    stores_follow_last_mma: bool

    @property
    def uses_real_global_inputs(self) -> bool:
        return self.global_loads >= 3 and self.loads_precede_first_mma

    @property
    def writes_global_output(self) -> bool:
        return self.global_stores >= 4 and self.stores_follow_last_mma


@dataclass(frozen=True, slots=True)
class FunctionalMmaPtxProbe:
    """Executable fixed-shape TF32/FP16 MMA MatMul and its launch contract.

    This is a real-data capability candidate: the kernel consumes caller-owned
    A and B buffers and writes every element of C.  Its small fixed shape makes
    correctness easy to validate on a CUDA device.  It is deliberately not a
    BEVFormer production FFN kernel.
    """

    symbol: str
    mode: str
    m: int
    n: int
    k: int
    k_step: int
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    input_dtype: str
    output_dtype: str
    gpu_source: str
    artifact: MlirPtxArtifact
    evidence: FunctionalMmaPtxEvidence

    @property
    def ptx(self) -> str:
        return self.artifact.ptx


@dataclass(frozen=True, slots=True)
class AsyncDoubleBufferPtxProbe:
    """SM86 async-copy double-buffer compiler probe.

    The probe contains a real two-stage workgroup allocation and overlaps the
    second global-to-shared copy group with consumption of the first stage.  It
    proves compiler/code-generation support only; it is not wired into the
    production BEVFormer FFN MatMuls.
    """

    stage_count: int
    stage_elements: int
    shared_memory_bytes: int
    gpu_source: str
    artifact: MlirPtxArtifact
    evidence: TiledPtxEvidence

    @property
    def ptx(self) -> str:
        return self.artifact.ptx


def _require_supported_ffn_rows(rows: int) -> int:
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be an integer")
    if rows not in SUPPORTED_FFN_ROWS:
        raise ValueError(
            f"rows must be one of the validated production profiles {SUPPORTED_FFN_ROWS}"
        )
    return rows


def bevformer_fc1_tiled_config(
    *,
    rows: int = BEVFORMER_ROWS,
    tile_m: int = DEFAULT_TILE,
    tile_n: int = DEFAULT_TILE,
    tile_k: int = DEFAULT_TILE,
) -> TiledMatmulConfig:
    """Return the FC1 tiled config for a validated static FFN row profile.

    The default retains the BEVFormer ``2500``-row ABI.  ``40000`` selects the
    otherwise identical UniAD stage-2 encoder profile.
    """

    return TiledMatmulConfig(
        symbol="gac_bevformer_ffn_fc1_tiled_f32",
        m=_require_supported_ffn_rows(rows),
        n=BEVFORMER_HIDDEN_CHANNELS,
        k=BEVFORMER_CHANNELS,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_x=tile_n,
        block_y=tile_m,
        fuse_bias_relu=True,
    )


def bevformer_fc2_tiled_config(
    *,
    rows: int = BEVFORMER_ROWS,
    tile_m: int = DEFAULT_TILE,
    tile_n: int = DEFAULT_TILE,
    tile_k: int = DEFAULT_TILE,
) -> TiledMatmulConfig:
    """Return the FC2 tiled config for a validated static FFN row profile."""

    return TiledMatmulConfig(
        symbol="gac_bevformer_ffn_fc2_tiled_f32",
        m=_require_supported_ffn_rows(rows),
        n=BEVFORMER_CHANNELS,
        k=BEVFORMER_HIDDEN_CHANNELS,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_x=tile_n,
        block_y=tile_m,
        fuse_bias_relu=False,
    )


def bevformer_tiled_candidates() -> tuple[tuple[int, int, int], ...]:
    """Conservative candidate tiles suitable for later on-device tuning."""

    return (
        (16, 16, 8),
        (16, 16, 16),
        (16, 16, 32),
        (8, 32, 16),
        (32, 8, 16),
    )


def emit_tiled_matmul_gpu_mlir(config: TiledMatmulConfig) -> str:
    """Emit explicit GPU Dialect for one shared-memory FP32 MatMul.

    The returned module is ready for ``gpu-lower-to-nvvm-pipeline``.  The
    matrix ABI is row-major contiguous bare pointers.  When
    ``fuse_bias_relu`` is true, the ABI is ``A, B, bias, C``; otherwise it is
    ``A, B, C``.
    """

    if not isinstance(config, TiledMatmulConfig):
        raise TypeError("config must be a TiledMatmulConfig")

    bias_parameter = ""
    bias_epilogue = "%result = arith.addf %outer_acc, %zero : f32"
    argument_note = "A, B, C"
    if config.fuse_bias_relu:
        bias_parameter = (
            f"        %bias: memref<{config.n}xf32, #gpu.address_space<global>>,\n"
        )
        bias_epilogue = f"""%bias_value = memref.load %bias[%global_col]
          : memref<{config.n}xf32, #gpu.address_space<global>>
        %biased = arith.addf %outer_acc, %bias_value : f32
        %result = arith.maximumf %biased, %zero : f32"""
        argument_note = "A, B, bias, C"

    return f'''// GAC explicit tiled FP32 MatMul; bare-pointer ABI: {argument_note}.
module attributes {{gpu.container_module}} {{
  gpu.module @gac_tiled_matmul_sm86 {{
    gpu.func @{config.symbol}(
        %a: memref<{config.m}x{config.k}xf32, #gpu.address_space<global>>,
        %b: memref<{config.k}x{config.n}xf32, #gpu.address_space<global>>,
{bias_parameter}        %c: memref<{config.m}x{config.n}xf32, #gpu.address_space<global>>)
        workgroup(
          %shared_a: memref<{config.tile_m}x{config.tile_k}xf32, #gpu.address_space<workgroup>>,
          %shared_b: memref<{config.tile_k}x{config.tile_n}xf32, #gpu.address_space<workgroup>>)
        kernel attributes {{known_block_size = array<i32: {config.block_x}, {config.block_y}, 1>}} {{
      %block_x = gpu.block_id x
      %block_y = gpu.block_id y
      %thread_x = gpu.thread_id x
      %thread_y = gpu.thread_id y
      %tile_m = arith.constant {config.tile_m} : index
      %tile_n = arith.constant {config.tile_n} : index
      %tile_k = arith.constant {config.tile_k} : index
      %block_x_size = arith.constant {config.block_x} : index
      %thread_count = arith.constant {config.threads_per_block} : index
      %a_tile_count = arith.constant {config.tile_m * config.tile_k} : index
      %b_tile_count = arith.constant {config.tile_k * config.tile_n} : index
      %m = arith.constant {config.m} : index
      %n = arith.constant {config.n} : index
      %k = arith.constant {config.k} : index
      %zero_index = arith.constant 0 : index
      %one = arith.constant 1 : index
      %zero = arith.constant 0.0 : f32
      %block_row = arith.muli %block_y, %tile_m : index
      %block_col = arith.muli %block_x, %tile_n : index
      %thread_y_offset = arith.muli %thread_y, %block_x_size : index
      %thread_linear = arith.addi %thread_y_offset, %thread_x : index
      %global_row = arith.addi %block_row, %thread_y : index
      %global_col = arith.addi %block_col, %thread_x : index
      %outer_acc = scf.for %k_base = %zero_index to %k step %tile_k
          iter_args(%tile_acc = %zero) -> (f32) {{
        scf.for %a_linear = %thread_linear to %a_tile_count step %thread_count {{
          %a_local_row = arith.divui %a_linear, %tile_k : index
          %a_local_k = arith.remui %a_linear, %tile_k : index
          %a_global_row = arith.addi %block_row, %a_local_row : index
          %a_global_k = arith.addi %k_base, %a_local_k : index
          %a_row_ok = arith.cmpi ult, %a_global_row, %m : index
          %a_k_ok = arith.cmpi ult, %a_global_k, %k : index
          %a_ok = arith.andi %a_row_ok, %a_k_ok : i1
          %a_value = scf.if %a_ok -> (f32) {{
            %loaded_a = memref.load %a[%a_global_row, %a_global_k]
              : memref<{config.m}x{config.k}xf32, #gpu.address_space<global>>
            scf.yield %loaded_a : f32
          }} else {{
            scf.yield %zero : f32
          }}
          memref.store %a_value, %shared_a[%a_local_row, %a_local_k]
            : memref<{config.tile_m}x{config.tile_k}xf32, #gpu.address_space<workgroup>>
        }}
        scf.for %b_linear = %thread_linear to %b_tile_count step %thread_count {{
          %b_local_k = arith.divui %b_linear, %tile_n : index
          %b_local_col = arith.remui %b_linear, %tile_n : index
          %b_global_k = arith.addi %k_base, %b_local_k : index
          %b_global_col = arith.addi %block_col, %b_local_col : index
          %b_k_ok = arith.cmpi ult, %b_global_k, %k : index
          %b_col_ok = arith.cmpi ult, %b_global_col, %n : index
          %b_ok = arith.andi %b_k_ok, %b_col_ok : i1
          %b_value = scf.if %b_ok -> (f32) {{
            %loaded_b = memref.load %b[%b_global_k, %b_global_col]
              : memref<{config.k}x{config.n}xf32, #gpu.address_space<global>>
            scf.yield %loaded_b : f32
          }} else {{
            scf.yield %zero : f32
          }}
          memref.store %b_value, %shared_b[%b_local_k, %b_local_col]
            : memref<{config.tile_k}x{config.tile_n}xf32, #gpu.address_space<workgroup>>
        }}
        gpu.barrier
        %next_acc = scf.for %kk = %zero_index to %tile_k step %one
            iter_args(%inner_acc = %tile_acc) -> (f32) {{
          %lhs = memref.load %shared_a[%thread_y, %kk]
            : memref<{config.tile_m}x{config.tile_k}xf32, #gpu.address_space<workgroup>>
          %rhs = memref.load %shared_b[%kk, %thread_x]
            : memref<{config.tile_k}x{config.tile_n}xf32, #gpu.address_space<workgroup>>
          %product = arith.mulf %lhs, %rhs : f32
          %next = arith.addf %inner_acc, %product : f32
          scf.yield %next : f32
        }}
        gpu.barrier
        scf.yield %next_acc : f32
      }}
      %row_ok = arith.cmpi ult, %global_row, %m : index
      %col_ok = arith.cmpi ult, %global_col, %n : index
      %output_ok = arith.andi %row_ok, %col_ok : i1
      scf.if %output_ok {{
        {bias_epilogue}
        memref.store %result, %c[%global_row, %global_col]
          : memref<{config.m}x{config.n}xf32, #gpu.address_space<global>>
      }}
      gpu.return
    }}
  }}
}}
'''


def _static_shared_memory_bytes(ptx: str) -> int:
    """Return bytes reserved by statically sized textual PTX shared arrays."""

    total = 0
    declaration = re.compile(
        r"(?m)^[ \t]*\.shared\b[^\r\n;]*"
        r"\.([busf])(8|16|32|64)\b[^\r\n;]*"
        r"\[([0-9]+)\][ \t]*;"
    )
    for match in declaration.finditer(ptx):
        element_bits = int(match.group(2))
        element_count = int(match.group(3))
        total += (element_bits // 8) * element_count
    return total


def inspect_tiled_matmul_ptx(ptx: str) -> TiledPtxEvidence:
    """Count code-generation evidence in one textual PTX module."""

    if not isinstance(ptx, str):
        raise TypeError("ptx must be a string")
    if not ptx.strip():
        raise ValueError("ptx must be non-empty")
    wait_group_depths = tuple(
        int(value)
        for value in re.findall(r"\bcp\.async\.wait_group[ \t]+([0-9]+)\b", ptx)
    )
    return TiledPtxEvidence(
        shared_declarations=len(re.findall(r"(?m)^[ \t]*\.shared\b", ptx)),
        static_shared_bytes=_static_shared_memory_bytes(ptx),
        shared_loads=len(re.findall(r"\bld(?:\.volatile)?\.shared(?:\.|\b)", ptx)),
        shared_stores=len(re.findall(r"\bst(?:\.volatile)?\.shared(?:\.|\b)", ptx)),
        global_stores=len(re.findall(r"\bst(?:\.volatile)?\.global(?:\.|\b)", ptx)),
        vector_shared_loads=len(
            re.findall(r"\bld(?:\.volatile)?\.shared\.v(?:2|4|8)(?:\.|\b)", ptx)
        ),
        barriers=len(re.findall(r"\bbar(?:\.sync|\.warp)?\b", ptx)),
        fp32_multiplies=len(
            re.findall(r"\bmul(?:\.[A-Za-z0-9_]+)*\.f32\b", ptx)
        ),
        fp32_adds=len(re.findall(r"\badd(?:\.[A-Za-z0-9_]+)*\.f32\b", ptx)),
        fma_instructions=len(re.findall(r"\bfma(?:\.[A-Za-z0-9_]+)+\b", ptx)),
        mma_instructions=len(re.findall(r"\b(?:mma\.sync|wmma\.)", ptx)),
        async_copy_instructions=len(
            re.findall(r"\bcp\.async\.(?:ca|cg)\.shared\.global\b", ptx)
        ),
        async_commit_groups=len(re.findall(r"\bcp\.async\.commit_group\b", ptx)),
        async_wait_groups=len(re.findall(r"\bcp\.async\.wait_group\b", ptx)),
        async_wait_group_depths=wait_group_depths,
    )


def _validate_tiled_matmul_ptx_abi(
    ptx: str, config: TiledMatmulConfig
) -> None:
    """Require the bare-pointer entry ABI and launch shape emitted by MLIR."""

    pattern = re.compile(
        rf"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+{re.escape(config.symbol)}"
        rf"[ \t]*\((.*?)\)[ \t]*\r?\n[ \t]*\.maxntid[ \t]+([^\r\n]+)"
    )
    matches = pattern.findall(ptx)
    if len(matches) != 1:
        raise BackendError(
            f"tiled MatMul PTX entry {config.symbol!r} has no unambiguous ABI header"
        )
    header, maxntid = matches[0]
    parameters = [line.strip() for line in header.splitlines() if line.strip()]
    expected_count = 4 if config.fuse_bias_relu else 3
    bare_global_pointer = re.compile(
        r"^\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?"
        r"\s+[A-Za-z_$][A-Za-z0-9_.$]*,?$"
    )
    if len(parameters) != expected_count or any(
        bare_global_pointer.fullmatch(parameter) is None
        for parameter in parameters
    ):
        raise BackendError(
            f"tiled MatMul PTX entry {config.symbol!r} requires exactly "
            f"{expected_count} bare global pointers"
        )
    dimensions = tuple(int(value) for value in re.findall(r"[0-9]+", maxntid))
    if dimensions != config.block:
        raise BackendError(
            f"tiled MatMul PTX entry {config.symbol!r} requires .maxntid "
            f"{config.block[0]}, {config.block[1]}, {config.block[2]}"
        )


def lower_tiled_matmul_to_ptx(
    config: TiledMatmulConfig,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> TiledMatmulPtxProgram:
    """Lower and prove one tiled MatMul with the stock SM86 MLIR pipeline."""

    source = emit_tiled_matmul_gpu_mlir(config)
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(config.symbol,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    _validate_tiled_matmul_ptx_abi(artifact.ptx, config)
    evidence = inspect_tiled_matmul_ptx(artifact.ptx)
    if evidence.shared_declarations != 2:
        raise BackendError("tiled MatMul PTX must contain exactly two shared-memory arrays")
    if evidence.shared_loads == 0 or evidence.shared_stores == 0:
        raise BackendError("tiled MatMul PTX lost the expected shared-memory loads/stores")
    if not evidence.computes_real_matmul:
        raise BackendError(
            "tiled MatMul PTX lost floating-point accumulation or its global output store"
        )
    if evidence.barriers < 2:
        raise BackendError("tiled MatMul PTX lost the two required block barriers")
    if re.search(r"\bmalloc\b|\blocal_depot", artifact.ptx):
        raise BackendError("tiled MatMul shared tiles were incorrectly lowered to local allocation")
    return TiledMatmulPtxProgram(
        config=config,
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )


def emit_nvgpu_async_double_buffer_smoke_gpu_mlir() -> str:
    """Emit a two-stage NVGPU async-copy overlap probe for SM86.

    Four threads each issue one aligned 16-byte global-to-shared copy for stage
    0, commit it, then do the same for stage 1.  ``wait numGroups = 1`` makes
    stage 0 available while stage 1 may remain in flight.  The stage-0 shared
    load and global store therefore occupy the overlap window before the final
    ``wait numGroups = 0``.  The leading dimension of ``%stages`` is the two
    physical workgroup-memory stages.

    This intentionally remains a compiler capability/optimization probe.  It
    is not a complete GEMM and is not called by the BEVFormer FFN runtime.
    """

    return f'''// SM86 async-copy double-buffer capability probe only.
// Schedule: commit(stage 0), commit(stage 1), wait 1, consume stage 0,
//           wait 0, consume stage 1.
module attributes {{gpu.container_module}} {{
  gpu.module @gac_nvgpu_async_double_buffer_sm86 {{
    gpu.func @{ASYNC_DOUBLE_BUFFER_SYMBOL}(
        %input: memref<32xf32, #gpu.address_space<global>>,
        %output: memref<32xf32, #gpu.address_space<global>>)
        workgroup(
          %stages: memref<{ASYNC_DOUBLE_BUFFER_STAGES}x{ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS}xf32, #gpu.address_space<workgroup>>)
        kernel attributes {{known_block_size = array<i32: 4, 1, 1>}} {{
      %tid = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %lane_base = arith.muli %tid, %c4 : index
      %next_base = arith.addi %lane_base, %c16 : index

      %copy0 = nvgpu.device_async_copy
        %input[%lane_base], %stages[%c0, %lane_base], 4
        : memref<32xf32, #gpu.address_space<global>> to memref<{ASYNC_DOUBLE_BUFFER_STAGES}x{ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS}xf32, #gpu.address_space<workgroup>>
      %group0 = nvgpu.device_async_create_group %copy0

      %copy1 = nvgpu.device_async_copy
        %input[%next_base], %stages[%c1, %lane_base], 4
        : memref<32xf32, #gpu.address_space<global>> to memref<{ASYNC_DOUBLE_BUFFER_STAGES}x{ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS}xf32, #gpu.address_space<workgroup>>
      %group1 = nvgpu.device_async_create_group %copy1

      nvgpu.device_async_wait %group0 numGroups = 1
      gpu.barrier
      %stage0 = memref.load %stages[%c0, %lane_base]
        : memref<{ASYNC_DOUBLE_BUFFER_STAGES}x{ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS}xf32, #gpu.address_space<workgroup>>
      memref.store %stage0, %output[%lane_base]
        : memref<32xf32, #gpu.address_space<global>>

      nvgpu.device_async_wait %group1 numGroups = 0
      gpu.barrier
      %stage1 = memref.load %stages[%c1, %lane_base]
        : memref<{ASYNC_DOUBLE_BUFFER_STAGES}x{ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS}xf32, #gpu.address_space<workgroup>>
      memref.store %stage1, %output[%next_base]
        : memref<32xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def _validate_async_double_buffer_ptx_abi(ptx: str) -> None:
    """Require the two-pointer, four-thread ABI of the async-copy probe."""

    pattern = re.compile(
        rf"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+"
        rf"{re.escape(ASYNC_DOUBLE_BUFFER_SYMBOL)}[ \t]*\((.*?)\)"
        rf"[ \t]*\r?\n[ \t]*\.maxntid[ \t]+([^\r\n]+)"
    )
    matches = pattern.findall(ptx)
    if len(matches) != 1:
        raise BackendError("async double-buffer PTX has no unambiguous entry ABI")
    header, maxntid = matches[0]
    parameters = [line.strip() for line in header.splitlines() if line.strip()]
    bare_global_pointer = re.compile(
        r"^\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?"
        r"\s+[A-Za-z_$][A-Za-z0-9_.$]*,?$"
    )
    if len(parameters) != 2 or any(
        bare_global_pointer.fullmatch(parameter) is None
        for parameter in parameters
    ):
        raise BackendError(
            "async double-buffer PTX entry requires exactly two bare global pointers"
        )
    dimensions = tuple(int(value) for value in re.findall(r"[0-9]+", maxntid))
    if dimensions != (4, 1, 1):
        raise BackendError("async double-buffer PTX entry requires .maxntid 4, 1, 1")


def probe_nvgpu_async_double_buffer_to_ptx(
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> AsyncDoubleBufferPtxProbe:
    """Lower the two-stage NVGPU probe and require concrete PTX evidence."""

    source = emit_nvgpu_async_double_buffer_smoke_gpu_mlir()
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(ASYNC_DOUBLE_BUFFER_SYMBOL,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    _validate_async_double_buffer_ptx_abi(artifact.ptx)
    evidence = inspect_tiled_matmul_ptx(artifact.ptx)
    if evidence.shared_declarations != 1:
        raise BackendError(
            "async double-buffer PTX must contain exactly one two-stage shared array"
        )
    if evidence.static_shared_bytes != ASYNC_DOUBLE_BUFFER_SHARED_BYTES:
        raise BackendError(
            "async double-buffer PTX shared allocation must be exactly "
            f"{ASYNC_DOUBLE_BUFFER_SHARED_BYTES} bytes, got "
            f"{evidence.static_shared_bytes}"
        )
    if evidence.async_copy_instructions != 2:
        raise BackendError("async double-buffer PTX must contain exactly two cp.async copies")
    if evidence.async_commit_groups != 2:
        raise BackendError(
            "async double-buffer PTX must contain exactly two cp.async commit groups"
        )
    if evidence.async_wait_group_depths != (1, 0):
        raise BackendError(
            "async double-buffer PTX must wait_group 1 then wait_group 0, got "
            f"{evidence.async_wait_group_depths}"
        )
    if evidence.barriers < 2 or evidence.shared_loads < 2:
        raise BackendError(
            "async double-buffer PTX lost the barriers or shared loads that consume both stages"
        )

    wait_one = re.search(r"\bcp\.async\.wait_group[ \t]+1\b", artifact.ptx)
    wait_zero = re.search(r"\bcp\.async\.wait_group[ \t]+0\b", artifact.ptx)
    first_overlap_load = None
    if wait_one is not None:
        first_overlap_load = re.search(
            r"\bld(?:\.volatile)?\.shared(?:\.|\b)",
            artifact.ptx[wait_one.end() :],
        )
    if (
        wait_one is None
        or wait_zero is None
        or first_overlap_load is None
        or wait_one.end() + first_overlap_load.start() >= wait_zero.start()
    ):
        raise BackendError(
            "async double-buffer PTX must consume stage 0 between wait_group 1 and 0"
        )
    if re.search(r"\bmalloc\b|\blocal_depot", artifact.ptx):
        raise BackendError(
            "async double-buffer stages were incorrectly lowered to local allocation"
        )
    return AsyncDoubleBufferPtxProbe(
        stage_count=ASYNC_DOUBLE_BUFFER_STAGES,
        stage_elements=ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS,
        shared_memory_bytes=ASYNC_DOUBLE_BUFFER_SHARED_BYTES,
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )


def emit_nvgpu_mma_smoke_gpu_mlir(mode: str) -> str:
    """Emit a warp-wide NVGPU MMA lowering probe for ``fp16`` or ``tf32``.

    This deliberately proves only the LLVM 24 compiler route from
    ``nvgpu.mma.sync`` to a PTX ``mma.sync`` instruction.  Production fragment
    loading, lane ownership, edge handling, and epilogue stores remain separate
    work and are not represented by this smoke module.
    """

    if mode == "fp16":
        symbol = "gac_nvgpu_mma_fp16_smoke"
        fragments = """%a = arith.constant dense<1.0> : vector<4x2xf16>
      %b = arith.constant dense<1.0> : vector<2x2xf16>
      %c = arith.constant dense<0.0> : vector<2x2xf32>
      %d = nvgpu.mma.sync (%a, %b, %c) mmaShape = [16, 8, 16] : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>"""
    elif mode == "tf32":
        symbol = "gac_nvgpu_mma_tf32_smoke"
        fragments = """%a = arith.constant dense<1.0> : vector<2x1xf32>
      %b = arith.constant dense<1.0> : vector<1x1xf32>
      %c = arith.constant dense<0.0> : vector<2x2xf32>
      %d = nvgpu.mma.sync (%a, %b, %c) mmaShape = [16, 8, 4] tf32Enabled : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>"""
    else:
        raise ValueError("mode must be 'fp16' or 'tf32'")

    return f'''// Compiler capability probe only; not a production GEMM.
module attributes {{gpu.container_module}} {{
  gpu.module @gac_nvgpu_mma_smoke {{
    gpu.func @{symbol}(
        %output: memref<32xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: 32, 1, 1>}} {{
      %lane = gpu.thread_id x
      {fragments}
      %owned = vector.extract %d[0, 0] : f32 from vector<2x2xf32>
      memref.store %owned, %output[%lane]
        : memref<32xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def probe_nvgpu_mma_to_ptx(
    mode: str,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> MmaPtxProbe:
    """Lower an NVGPU MMA smoke module and require PTX Tensor Core evidence."""

    source = emit_nvgpu_mma_smoke_gpu_mlir(mode)
    symbol = f"gac_nvgpu_mma_{mode}_smoke"
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(symbol,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    evidence = inspect_tiled_matmul_ptx(artifact.ptx)
    if not evidence.uses_tensor_cores:
        raise BackendError(
            f"the {mode} NVGPU capability probe produced no PTX mma.sync instruction"
        )
    return MmaPtxProbe(
        mode=mode,
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )


def emit_nvgpu_tf32_matmul_gpu_mlir() -> str:
    """Emit an executable real-input ``16x8x16`` TF32 MMA MatMul.

    One 32-thread warp owns the complete output tile.  For each of the four
    K-slices, every lane loads its assigned A and B values from caller-provided
    global buffers and feeds them to ``nvgpu.mma.sync``.  The accumulator is
    carried by the ``scf.for`` loop and each lane finally stores four distinct
    output elements.  The ABI is three row-major contiguous f32 bare pointers:
    ``A[16,16], B[16,8], C[16,8]``.
    """

    return f'''// Executable real-input TF32 MMA MatMul capability probe.
// Bare-pointer ABI: A[16,16] f32, B[16,8] f32, C[16,8] f32.
// Launch exactly grid=(1,1,1), block=(32,1,1).
module attributes {{gpu.container_module}} {{
  gpu.module @gac_nvgpu_mma_tf32_matmul_sm86 {{
    gpu.func @{FUNCTIONAL_MMA_TF32_SYMBOL}(
        %a: memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_K}xf32, #gpu.address_space<global>>,
        %b: memref<{FUNCTIONAL_MMA_K}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>,
        %c: memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: 32, 1, 1>}} {{
      %lane = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant {FUNCTIONAL_MMA_K_STEP} : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant {FUNCTIONAL_MMA_K} : index
      %row0 = arith.divui %lane, %c4 : index
      %lane_mod4 = arith.remui %lane, %c4 : index
      %row1 = arith.addi %row0, %c8 : index
      %col0 = arith.muli %lane_mod4, %c2 : index
      %col1 = arith.addi %col0, %c1 : index
      %zero_acc = arith.constant dense<0.0> : vector<2x2xf32>
      %result = scf.for %k_base = %c0 to %c16 step %c4
          iter_args(%acc = %zero_acc) -> (vector<2x2xf32>) {{
        %k_index = arith.addi %k_base, %lane_mod4 : index
        %a_row0 = memref.load %a[%row0, %k_index]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_K}xf32, #gpu.address_space<global>>
        %a_row1 = memref.load %a[%row1, %k_index]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_K}xf32, #gpu.address_space<global>>
        %b_value = memref.load %b[%k_index, %row0]
          : memref<{FUNCTIONAL_MMA_K}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
        %a_empty = arith.constant dense<0.0> : vector<2x1xf32>
        %a_first = vector.insert %a_row0, %a_empty [0, 0]
          : f32 into vector<2x1xf32>
        %a_fragment = vector.insert %a_row1, %a_first [1, 0]
          : f32 into vector<2x1xf32>
        %b_empty = arith.constant dense<0.0> : vector<1x1xf32>
        %b_fragment = vector.insert %b_value, %b_empty [0, 0]
          : f32 into vector<1x1xf32>
        %next = nvgpu.mma.sync (%a_fragment, %b_fragment, %acc)
          mmaShape = [16, 8, 4] tf32Enabled
          : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
        scf.yield %next : vector<2x2xf32>
      }}
      %c00 = vector.extract %result[0, 0] : f32 from vector<2x2xf32>
      %c01 = vector.extract %result[0, 1] : f32 from vector<2x2xf32>
      %c10 = vector.extract %result[1, 0] : f32 from vector<2x2xf32>
      %c11 = vector.extract %result[1, 1] : f32 from vector<2x2xf32>
      memref.store %c00, %c[%row0, %col0]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c01, %c[%row0, %col1]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c10, %c[%row1, %col0]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c11, %c[%row1, %col1]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def emit_nvgpu_fp16_matmul_gpu_mlir() -> str:
    """Emit an executable real-input ``16x8x32`` FP16 MMA MatMul.

    A single warp evaluates two K=16 Tensor Core slices.  Each lane loads the
    eight A and four B values assigned by the documented ``m16n8k16`` row/col
    fragment layout.  Inputs use f16 storage, the loop-carried accumulator and
    complete C output use f32.
    """

    return f'''// Executable real-input FP16 MMA MatMul capability probe.
// Bare-pointer ABI: A[16,32] f16, B[32,8] f16, C[16,8] f32.
// Launch exactly grid=(1,1,1), block=(32,1,1).
module attributes {{gpu.container_module}} {{
  gpu.module @gac_nvgpu_mma_fp16_matmul_sm86 {{
    gpu.func @{FUNCTIONAL_MMA_FP16_SYMBOL}(
        %a: memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>,
        %b: memref<{FUNCTIONAL_MMA_FP16_K}x{FUNCTIONAL_MMA_N}xf16, #gpu.address_space<global>>,
        %c: memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>)
        kernel attributes {{known_block_size = array<i32: 32, 1, 1>}} {{
      %lane = gpu.thread_id x
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant {FUNCTIONAL_MMA_FP16_K_STEP} : index
      %c32 = arith.constant {FUNCTIONAL_MMA_FP16_K} : index
      %row0 = arith.divui %lane, %c4 : index
      %lane_mod4 = arith.remui %lane, %c4 : index
      %row1 = arith.addi %row0, %c8 : index
      %pair_base = arith.muli %lane_mod4, %c2 : index
      %pair_next = arith.addi %pair_base, %c1 : index
      %pair_base8 = arith.addi %pair_base, %c8 : index
      %pair_next8 = arith.addi %pair_next, %c8 : index
      %col0 = arith.muli %lane_mod4, %c2 : index
      %col1 = arith.addi %col0, %c1 : index
      %zero_acc = arith.constant dense<0.0> : vector<2x2xf32>
      %result = scf.for %k_base = %c0 to %c32 step %c16
          iter_args(%acc = %zero_acc) -> (vector<2x2xf32>) {{
        %k0 = arith.addi %k_base, %pair_base : index
        %k1 = arith.addi %k_base, %pair_next : index
        %k8 = arith.addi %k_base, %pair_base8 : index
        %k9 = arith.addi %k_base, %pair_next8 : index

        %a00 = memref.load %a[%row0, %k0]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a01 = memref.load %a[%row0, %k1]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a10 = memref.load %a[%row1, %k0]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a11 = memref.load %a[%row1, %k1]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a08 = memref.load %a[%row0, %k8]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a09 = memref.load %a[%row0, %k9]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a18 = memref.load %a[%row1, %k8]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a19 = memref.load %a[%row1, %k9]
          : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_FP16_K}xf16, #gpu.address_space<global>>
        %a_empty = arith.constant dense<0.0> : vector<4x2xf16>
        %af0 = vector.insert %a00, %a_empty [0, 0] : f16 into vector<4x2xf16>
        %af1 = vector.insert %a01, %af0 [0, 1] : f16 into vector<4x2xf16>
        %af2 = vector.insert %a10, %af1 [1, 0] : f16 into vector<4x2xf16>
        %af3 = vector.insert %a11, %af2 [1, 1] : f16 into vector<4x2xf16>
        %af4 = vector.insert %a08, %af3 [2, 0] : f16 into vector<4x2xf16>
        %af5 = vector.insert %a09, %af4 [2, 1] : f16 into vector<4x2xf16>
        %af6 = vector.insert %a18, %af5 [3, 0] : f16 into vector<4x2xf16>
        %a_fragment = vector.insert %a19, %af6 [3, 1] : f16 into vector<4x2xf16>

        %b0 = memref.load %b[%k0, %row0]
          : memref<{FUNCTIONAL_MMA_FP16_K}x{FUNCTIONAL_MMA_N}xf16, #gpu.address_space<global>>
        %b1 = memref.load %b[%k1, %row0]
          : memref<{FUNCTIONAL_MMA_FP16_K}x{FUNCTIONAL_MMA_N}xf16, #gpu.address_space<global>>
        %b8 = memref.load %b[%k8, %row0]
          : memref<{FUNCTIONAL_MMA_FP16_K}x{FUNCTIONAL_MMA_N}xf16, #gpu.address_space<global>>
        %b9 = memref.load %b[%k9, %row0]
          : memref<{FUNCTIONAL_MMA_FP16_K}x{FUNCTIONAL_MMA_N}xf16, #gpu.address_space<global>>
        %b_empty = arith.constant dense<0.0> : vector<2x2xf16>
        %bf0 = vector.insert %b0, %b_empty [0, 0] : f16 into vector<2x2xf16>
        %bf1 = vector.insert %b1, %bf0 [0, 1] : f16 into vector<2x2xf16>
        %bf2 = vector.insert %b8, %bf1 [1, 0] : f16 into vector<2x2xf16>
        %b_fragment = vector.insert %b9, %bf2 [1, 1] : f16 into vector<2x2xf16>

        %next = nvgpu.mma.sync (%a_fragment, %b_fragment, %acc)
          mmaShape = [16, 8, 16]
          : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
        scf.yield %next : vector<2x2xf32>
      }}
      %c00 = vector.extract %result[0, 0] : f32 from vector<2x2xf32>
      %c01 = vector.extract %result[0, 1] : f32 from vector<2x2xf32>
      %c10 = vector.extract %result[1, 0] : f32 from vector<2x2xf32>
      %c11 = vector.extract %result[1, 1] : f32 from vector<2x2xf32>
      memref.store %c00, %c[%row0, %col0]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c01, %c[%row0, %col1]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c10, %c[%row1, %col0]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      memref.store %c11, %c[%row1, %col1]
        : memref<{FUNCTIONAL_MMA_M}x{FUNCTIONAL_MMA_N}xf32, #gpu.address_space<global>>
      gpu.return
    }}
  }}
}}
'''


def inspect_functional_mma_ptx(ptx: str) -> FunctionalMmaPtxEvidence:
    """Extract real-input and Tensor Core evidence from textual PTX."""

    if not isinstance(ptx, str):
        raise TypeError("ptx must be a string")
    if not ptx.strip():
        raise ValueError("ptx must be non-empty")
    global_loads = tuple(
        re.finditer(r"\bld(?:\.[A-Za-z0-9_]+)*\.global(?:\.[A-Za-z0-9_]+)*\b", ptx)
    )
    global_stores = tuple(
        re.finditer(r"\bst(?:\.[A-Za-z0-9_]+)*\.global(?:\.[A-Za-z0-9_]+)*\b", ptx)
    )
    mma = tuple(re.finditer(r"\bmma\.sync(?:\.[A-Za-z0-9_]+)*\b", ptx))
    tf32_mma = tuple(
        re.finditer(
            r"\bmma\.sync\.aligned\.m16n8k4\.row\.col\.f32\.tf32\.tf32\.f32\b",
            ptx,
        )
    )
    fp16_mma = tuple(
        re.finditer(
            r"\bmma\.sync\.aligned\.m16n8k16\.row\.col\.f32\.f16\.f16\.f32\b",
            ptx,
        )
    )
    first_mma = mma[0].start() if mma else -1
    last_mma = mma[-1].end() if mma else -1
    return FunctionalMmaPtxEvidence(
        global_loads=len(global_loads),
        global_stores=len(global_stores),
        mma_instructions=len(mma),
        tf32_mma_instructions=len(tf32_mma),
        fp16_mma_instructions=len(fp16_mma),
        loads_precede_first_mma=bool(
            global_loads and first_mma >= 0 and global_loads[0].start() < first_mma
        ),
        stores_follow_last_mma=bool(
            global_stores and last_mma >= 0 and global_stores[-1].start() > last_mma
        ),
    )


def _functional_mma_contract(
    mode: str,
) -> tuple[str, int, int, int, str, int, str]:
    """Return symbol, K, step, slices, dtype, loads/slice and PTX signature."""

    if mode == "tf32":
        return (
            FUNCTIONAL_MMA_TF32_SYMBOL,
            FUNCTIONAL_MMA_K,
            FUNCTIONAL_MMA_K_STEP,
            FUNCTIONAL_MMA_K_SLICES,
            "float32",
            3,
            "m16n8k4 f32.tf32.tf32.f32",
        )
    if mode == "fp16":
        return (
            FUNCTIONAL_MMA_FP16_SYMBOL,
            FUNCTIONAL_MMA_FP16_K,
            FUNCTIONAL_MMA_FP16_K_STEP,
            FUNCTIONAL_MMA_FP16_K_SLICES,
            "float16",
            12,
            "m16n8k16 f32.f16.f16.f32",
        )
    raise ValueError("mode must be 'tf32' or 'fp16'")


def _validate_functional_mma_ptx_abi(ptx: str, mode: str = "tf32") -> None:
    """Require the exact three-pointer, one-warp launch contract."""

    symbol, *_ = _functional_mma_contract(mode)
    pattern = re.compile(
        rf"(?ms)^[ \t]*\.visible[ \t]+\.entry[ \t]+"
        rf"{re.escape(symbol)}[ \t]*\((.*?)\)"
        rf"[ \t]*\r?\n[ \t]*\.maxntid[ \t]+([^\r\n]+)"
    )
    matches = pattern.findall(ptx)
    if len(matches) != 1:
        raise BackendError("functional MMA PTX has no unambiguous entry ABI")
    header, maxntid = matches[0]
    parameters = [line.strip() for line in header.splitlines() if line.strip()]
    bare_global_pointer = re.compile(
        r"^\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?"
        r"\s+[A-Za-z_$][A-Za-z0-9_.$]*,?$"
    )
    if len(parameters) != 3 or any(
        bare_global_pointer.fullmatch(parameter) is None for parameter in parameters
    ):
        raise BackendError(
            "functional MMA PTX entry requires exactly three bare global pointers"
        )
    dimensions = tuple(int(value) for value in re.findall(r"[0-9]+", maxntid))
    if dimensions != FUNCTIONAL_MMA_BLOCK:
        raise BackendError(
            "functional MMA PTX entry requires .maxntid 32, 1, 1"
        )


def validate_functional_mma_ptx(
    ptx: str, mode: str = "tf32"
) -> FunctionalMmaPtxEvidence:
    """Validate the functional probe ABI and return its codegen evidence."""

    (
        _symbol,
        _k,
        _k_step,
        k_slices,
        _input_dtype,
        fragment_loads_per_slice,
        mma_signature,
    ) = _functional_mma_contract(mode)
    _validate_functional_mma_ptx_abi(ptx, mode)
    evidence = inspect_functional_mma_ptx(ptx)
    expected_fragment_loads = fragment_loads_per_slice * k_slices
    if (
        not evidence.uses_real_global_inputs
        or evidence.global_loads < expected_fragment_loads
    ):
        raise BackendError(
            "functional MMA PTX must load real A/B values from global memory "
            f"for all {k_slices} {mode} K slices before mma.sync"
        )
    if not evidence.writes_global_output:
        raise BackendError(
            "functional MMA PTX must store all four lane-owned outputs after mma.sync"
        )
    typed_mma_instructions = (
        evidence.tf32_mma_instructions
        if mode == "tf32"
        else evidence.fp16_mma_instructions
    )
    if evidence.mma_instructions != k_slices or typed_mma_instructions != k_slices:
        raise BackendError(
            "functional MMA PTX must contain exactly "
            f"{k_slices} {mma_signature} mma.sync instructions, one per K slice"
        )
    return evidence


def probe_nvgpu_tf32_matmul_to_ptx(
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> FunctionalMmaPtxProbe:
    """Lower and prove the executable real-input TF32 MMA MatMul probe."""

    source = emit_nvgpu_tf32_matmul_gpu_mlir()
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(FUNCTIONAL_MMA_TF32_SYMBOL,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    evidence = validate_functional_mma_ptx(artifact.ptx, "tf32")
    return FunctionalMmaPtxProbe(
        symbol=FUNCTIONAL_MMA_TF32_SYMBOL,
        mode="tf32",
        m=FUNCTIONAL_MMA_M,
        n=FUNCTIONAL_MMA_N,
        k=FUNCTIONAL_MMA_K,
        k_step=FUNCTIONAL_MMA_K_STEP,
        grid=FUNCTIONAL_MMA_GRID,
        block=FUNCTIONAL_MMA_BLOCK,
        input_dtype="float32",
        output_dtype="float32",
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )


def probe_nvgpu_fp16_matmul_to_ptx(
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = 30.0,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> FunctionalMmaPtxProbe:
    """Lower and prove the executable real-input FP16 MMA MatMul probe."""

    source = emit_nvgpu_fp16_matmul_gpu_mlir()
    artifact = lower_gpu_mlir_to_ptx(
        source,
        expected_entries=(FUNCTIONAL_MMA_FP16_SYMBOL,),
        mlir_opt=mlir_opt,
        command=command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
        timeout=timeout,
        env=env,
        llvm_root=llvm_root,
    )
    evidence = validate_functional_mma_ptx(artifact.ptx, "fp16")
    return FunctionalMmaPtxProbe(
        symbol=FUNCTIONAL_MMA_FP16_SYMBOL,
        mode="fp16",
        m=FUNCTIONAL_MMA_M,
        n=FUNCTIONAL_MMA_N,
        k=FUNCTIONAL_MMA_FP16_K,
        k_step=FUNCTIONAL_MMA_FP16_K_STEP,
        grid=FUNCTIONAL_MMA_GRID,
        block=FUNCTIONAL_MMA_BLOCK,
        input_dtype="float16",
        output_dtype="float32",
        gpu_source=source,
        artifact=artifact,
        evidence=evidence,
    )


__all__ = [
    "ASYNC_DOUBLE_BUFFER_SHARED_BYTES",
    "ASYNC_DOUBLE_BUFFER_STAGE_ELEMENTS",
    "ASYNC_DOUBLE_BUFFER_STAGES",
    "ASYNC_DOUBLE_BUFFER_SYMBOL",
    "BEVFORMER_CHANNELS",
    "BEVFORMER_HIDDEN_CHANNELS",
    "BEVFORMER_ROWS",
    "SUPPORTED_FFN_ROWS",
    "UNIAD_ROWS",
    "AsyncDoubleBufferPtxProbe",
    "FUNCTIONAL_MMA_BLOCK",
    "FUNCTIONAL_MMA_FP16_K",
    "FUNCTIONAL_MMA_FP16_K_SLICES",
    "FUNCTIONAL_MMA_FP16_K_STEP",
    "FUNCTIONAL_MMA_FP16_SYMBOL",
    "FUNCTIONAL_MMA_GRID",
    "FUNCTIONAL_MMA_K",
    "FUNCTIONAL_MMA_K_SLICES",
    "FUNCTIONAL_MMA_K_STEP",
    "FUNCTIONAL_MMA_M",
    "FUNCTIONAL_MMA_N",
    "FUNCTIONAL_MMA_TF32_SYMBOL",
    "FunctionalMmaPtxEvidence",
    "FunctionalMmaPtxProbe",
    "MmaPtxProbe",
    "TiledMatmulConfig",
    "TiledMatmulPtxProgram",
    "TiledPtxEvidence",
    "bevformer_fc1_tiled_config",
    "bevformer_fc2_tiled_config",
    "bevformer_tiled_candidates",
    "emit_nvgpu_async_double_buffer_smoke_gpu_mlir",
    "emit_nvgpu_fp16_matmul_gpu_mlir",
    "emit_nvgpu_tf32_matmul_gpu_mlir",
    "emit_tiled_matmul_gpu_mlir",
    "emit_nvgpu_mma_smoke_gpu_mlir",
    "inspect_tiled_matmul_ptx",
    "inspect_functional_mma_ptx",
    "lower_tiled_matmul_to_ptx",
    "probe_nvgpu_async_double_buffer_to_ptx",
    "probe_nvgpu_mma_to_ptx",
    "probe_nvgpu_fp16_matmul_to_ptx",
    "probe_nvgpu_tf32_matmul_to_ptx",
    "validate_functional_mma_ptx",
]
